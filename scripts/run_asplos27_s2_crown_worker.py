#!/usr/bin/env python3
"""Run one fresh six-order S2 native/direct/canonical timing worker."""

# pylint: disable=wrong-import-position,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring,protected-access,duplicate-code

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.asplos27_s2_crown_pipeline import PreparedS2CrownProgramV1
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_d2b_staged_backward import (
    PreparedR3D2BStagedBackwardCandidateV1,
)
from boundflow.runtime.r3_structured_owner_custom_backward import (
    _evaluate_full_region,
    bind_r31_runtime_inputs_v1,
    compile_r31_full_region_plan_v1,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
    production_tensor_sha256,
)
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

WORKER_SCHEMA = "boundflow.asplos27-s2-crown-worker/v1"
ORDERS = ("NDP", "NPD", "DNP", "DPN", "PND", "PDN")
WARMUP_GROUPS = 5
SAMPLE_GROUPS = 30


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _values(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _prepare(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    raw = torch.load(args.source_capture, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
    device = torch.device("cuda:0")

    def bind():  # type: ignore[no-untyped-def]
        return bind_r31_runtime_inputs_v1(plan, module, snapshot, device=device)

    native_tensors = bind()
    direct_tensors = bind()
    canonical_tensors = bind()
    direct_started = time.perf_counter_ns()
    direct = PreparedR3D2BStagedBackwardCandidateV1(plan, trace, direct_tensors)
    direct_prepare_ns = time.perf_counter_ns() - direct_started
    canonical_started = time.perf_counter_ns()
    canonical = PreparedS2CrownProgramV1(plan, trace, canonical_tensors)
    canonical_prepare_ns = time.perf_counter_ns() - canonical_started
    return (
        plan,
        trace,
        native_tensors,
        direct,
        canonical,
        direct_prepare_ns,
        canonical_prepare_ns,
    )


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("S2 timing worker requires CUDA")
    if args.order not in ORDERS or args.run_index != ORDERS.index(args.order):
        raise ValueError("S2 timing worker order identity differs")
    (
        plan,
        trace,
        native_tensors,
        direct,
        canonical,
        direct_prepare_ns,
        canonical_prepare_ns,
    ) = _prepare(args)
    native_alpha = native_tensors[plan.p_alpha_input_ordinal]
    stream = torch.cuda.Stream(device="cuda:0")

    def native():  # type: ignore[no-untyped-def]
        lower = _evaluate_full_region(plan, native_tensors)
        gradient = torch.autograd.grad(-lower.sum(), native_alpha)[0]
        return lower, gradient

    def direct_run():  # type: ignore[no-untyped-def]
        direct.begin_sample()
        direct.begin_evaluation(0)
        lower = direct.forward()
        gradient = direct.backward(direct.upstream_gradient)
        return lower, gradient

    def canonical_run():  # type: ignore[no-untyped-def]
        return canonical.run_vjp(
            canonical.tensors[canonical.plan.p_alpha_input_ordinal],
            canonical.upstream_gradient,
        )

    functions = {"N": native, "D": direct_run, "P": canonical_run}
    for _ in range(WARMUP_GROUPS):
        for mode in args.order:
            with torch.cuda.stream(stream):
                functions[mode]()
            stream.synchronize()

    samples: dict[str, list[int]] = {"N": [], "D": [], "P": []}
    for _ in range(SAMPLE_GROUPS):
        for mode in args.order:
            stream.synchronize()
            started = time.perf_counter_ns()
            with torch.cuda.stream(stream):
                functions[mode]()
            stream.synchronize()
            samples[mode].append(time.perf_counter_ns() - started)

    semantic: dict[str, dict[str, object]] = {}
    for mode in "NDP":
        with torch.cuda.stream(stream):
            lower, gradient = functions[mode]()
        stream.synchronize()
        lower_cpu = lower.detach().cpu().contiguous().clone()
        gradient_cpu = gradient.detach().cpu().contiguous().clone()
        semantic[mode] = {
            "lower": _values(lower_cpu),
            "gradient": _values(gradient_cpu),
            "lower_sha256": production_tensor_sha256(lower_cpu),
            "gradient_sha256": production_tensor_sha256(gradient_cpu),
        }

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    with torch.cuda.stream(stream):
        canonical_run()
    stream.synchronize()
    canonical_dynamic_allocated = max(
        0, torch.cuda.max_memory_allocated() - allocated_before
    )
    canonical_dynamic_reserved = max(
        0, torch.cuda.max_memory_reserved() - reserved_before
    )
    properties = torch.cuda.get_device_properties(0)
    receipt = canonical.execution_receipt().to_dict()
    direct_receipt = asdict(direct.d2b_receipt())
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "run_index": args.run_index,
        "order": args.order,
        "source_capture_sha256": _file_hash(args.source_capture),
        "model_sha256": _file_hash(args.model),
        "plan_hash": plan.stable_hash(),
        "trace_hash": trace.stable_hash(),
        "warmup_groups": WARMUP_GROUPS,
        "sample_groups": SAMPLE_GROUPS,
        "latency_ns": samples,
        "median_latency_ns": {
            mode: statistics.median(values) for mode, values in samples.items()
        },
        "semantic": semantic,
        "direct_receipt": direct_receipt,
        "canonical_receipt": receipt,
        "prepare_ns": {
            "D": direct_prepare_ns,
            "P": canonical_prepare_ns,
            "P_relax_compile": int(
                canonical.selected_value.compiled.compile_ms * 1_000_000
            ),
        },
        "memory": {
            "allocated_before": allocated_before,
            "reserved_before": reserved_before,
            "canonical_peak_dynamic_allocated": canonical_dynamic_allocated,
            "canonical_peak_dynamic_reserved": canonical_dynamic_reserved,
        },
        "environment": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "device_index": 0,
        },
        "clock": "host-perf-counter-ns-with-device-boundary-sync",
        "same_solver_claimed": False,
        "complete_query_claimed": False,
        "tenx_claimed": False,
        "performance_claimed": False,
    }
    args.result.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--order", choices=ORDERS, required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    payload = _run(args)
    medians = payload["median_latency_ns"]
    print(
        "S2 worker "
        f"order={args.order} N={medians['N']} D={medians['D']} P={medians['P']} "
        "performance_claimed=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
