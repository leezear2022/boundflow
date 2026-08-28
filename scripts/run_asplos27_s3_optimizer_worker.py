#!/usr/bin/env python3
"""Run one fresh six-order S3 native/direct/canonical optimizer worker."""

# pylint: disable=wrong-import-position,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring,protected-access,duplicate-code
# pylint: disable=too-many-branches

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
from typing import cast

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.asplos27_s2_crown_pipeline import PreparedS2CrownProgramV1
from boundflow.runtime.asplos27_s3_optimizer_pipeline import (
    execute_asplos27_s3_optimizer_v1,
)
from boundflow.runtime.fsg4_b3_terminal_optimizer_schedule import (
    compile_terminal_optimizer_schedule_v1,
)
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_d2b_staged_backward import (
    PreparedR3D2BStagedBackwardCandidateV1,
)
from boundflow.runtime.r3_optimizer_trajectory_timing import (
    _candidate_evaluate,
    execute_r32b_wrapper_v1,
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

WORKER_SCHEMA = "boundflow.asplos27-s3-optimizer-worker/v1"
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


def _tensor_payload(tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().contiguous()
    return {
        "values": _values(value),
        "sha256": production_tensor_sha256(value),
        "shape": list(value.shape),
    }


def _optimizer_state(
    optimizer: torch.optim.Optimizer, alpha: torch.Tensor
) -> tuple[float, torch.Tensor, torch.Tensor]:
    state = optimizer.state.get(alpha)
    if not state:
        raise RuntimeError("S3 worker optimizer state is absent")
    raw_step = state["step"]
    return (
        float(raw_step.item() if torch.is_tensor(raw_step) else raw_step),
        state["exp_avg"],
        state["exp_avg_sq"],
    )


def _prepare(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    raw = torch.load(args.source_capture, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
    schedule = compile_terminal_optimizer_schedule_v1()
    device = torch.device("cuda:0")

    def bind():  # type: ignore[no-untyped-def]
        return bind_r31_runtime_inputs_v1(plan, module, snapshot, device=device)

    native_tensors = bind()
    direct_tensors = bind()
    candidate_tensors = bind()
    direct_started = time.perf_counter_ns()
    direct = PreparedR3D2BStagedBackwardCandidateV1(plan, trace, direct_tensors)
    direct_prepare_ns = time.perf_counter_ns() - direct_started
    candidate_started = time.perf_counter_ns()
    candidate = PreparedS2CrownProgramV1(plan, trace, candidate_tensors)
    candidate_prepare_ns = time.perf_counter_ns() - candidate_started
    return (
        plan,
        trace,
        schedule,
        native_tensors,
        direct,
        candidate,
        direct_prepare_ns,
        candidate_prepare_ns,
    )


def _capture_legacy(plan, tensors, schedule, candidate):  # type: ignore[no-untyped-def]
    alpha = tensors[plan.p_alpha_input_ordinal]
    optimizer = torch.optim.Adam([alpha], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    rows = []
    if candidate is not None:
        candidate.begin_sample()
    for action in schedule.actions:
        alpha_before = alpha.detach().clone()
        if candidate is None:
            lower = _evaluate_full_region(plan, tensors)
            gradient = torch.autograd.grad(-lower.sum(), alpha)[0]
        else:
            lower, gradient = _candidate_evaluate(candidate, action.evaluation_ordinal)
        if action.update_after:
            optimizer.zero_grad(set_to_none=True)
            alpha.grad = gradient
            optimizer.step()
            with torch.no_grad():
                alpha.clamp_(0.0, 1.0)
            scheduler.step()
        step, exp_avg, exp_avg_sq = _optimizer_state(optimizer, alpha)
        rows.append(
            {
                "evaluation_ordinal": action.evaluation_ordinal,
                "update_after": action.update_after,
                "alpha_learning_rate": action.alpha_learning_rate,
                "optimizer_step": step,
                "alpha_before": _tensor_payload(alpha_before),
                "lower": _tensor_payload(lower),
                "gradient": _tensor_payload(gradient),
                "alpha_after": _tensor_payload(alpha),
                "optimizer_exp_avg": _tensor_payload(exp_avg),
                "optimizer_exp_avg_sq": _tensor_payload(exp_avg_sq),
            }
        )
    return rows


def _capture_candidate(result):  # type: ignore[no-untyped-def]
    rows = []
    for step in result.steps:
        rows.append(
            {
                "evaluation_ordinal": step.evaluation_ordinal,
                "update_after": step.update_after,
                "alpha_learning_rate": step.alpha_learning_rate,
                "optimizer_step": step.optimizer_step,
                "alpha_before": _tensor_payload(step.alpha_before),
                "lower": _tensor_payload(step.lower),
                "gradient": _tensor_payload(step.gradient),
                "alpha_after": _tensor_payload(step.alpha_after),
                "optimizer_exp_avg": _tensor_payload(step.optimizer_exp_avg),
                "optimizer_exp_avg_sq": _tensor_payload(step.optimizer_exp_avg_sq),
            }
        )
    return rows


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("S3 optimizer worker requires CUDA")
    if args.order not in ORDERS or args.run_index != ORDERS.index(args.order):
        raise ValueError("S3 optimizer worker order identity differs")
    if args.replicate_index not in range(3):
        raise ValueError("S3 optimizer worker replicate identity differs")
    (
        plan,
        trace,
        schedule,
        native_tensors,
        direct,
        candidate,
        direct_prepare_ns,
        candidate_prepare_ns,
    ) = _prepare(args)
    modes = {
        "N": (native_tensors, None),
        "D": (direct.tensors, direct),
        "P": (candidate.tensors, candidate),
    }
    initial_alpha = {
        mode: tensors[plan.p_alpha_input_ordinal].detach().clone()
        for mode, (tensors, _) in modes.items()
    }
    stream = torch.cuda.Stream(device="cuda:0")

    def reset(mode: str) -> None:
        tensors, _ = modes[mode]
        alpha = tensors[plan.p_alpha_input_ordinal]
        with torch.no_grad():
            alpha.copy_(initial_alpha[mode])
        alpha.grad = None

    def execute(mode: str):  # type: ignore[no-untyped-def]
        tensors, prepared = modes[mode]
        if mode == "P":
            return execute_asplos27_s3_optimizer_v1(
                plan, tensors, schedule, prepared, capture=False
            )
        return execute_r32b_wrapper_v1(plan, tensors, schedule, candidate=prepared)

    for _ in range(WARMUP_GROUPS):
        for mode in args.order:
            reset(mode)
            with torch.cuda.stream(stream):
                execute(mode)
            stream.synchronize()

    samples: dict[str, list[int]] = {"N": [], "D": [], "P": []}
    for _ in range(SAMPLE_GROUPS):
        for mode in args.order:
            reset(mode)
            stream.synchronize()
            started = time.perf_counter_ns()
            with torch.cuda.stream(stream):
                execute(mode)
            stream.synchronize()
            samples[mode].append(time.perf_counter_ns() - started)

    semantic: dict[str, object] = {}
    for mode in "NDP":
        reset(mode)
        tensors, prepared = modes[mode]
        with torch.cuda.stream(stream):
            if mode == "P":
                result = execute_asplos27_s3_optimizer_v1(
                    plan, tensors, schedule, prepared, capture=True
                )
                rows = _capture_candidate(result)
            else:
                rows = _capture_legacy(plan, tensors, schedule, prepared)
        stream.synchronize()
        semantic[mode] = rows

    reset("P")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    with torch.cuda.stream(stream):
        candidate_result = execute("P")
    stream.synchronize()
    properties = torch.cuda.get_device_properties(0)
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "run_index": args.run_index,
        "replicate_index": args.replicate_index,
        "order": args.order,
        "source_capture_sha256": _file_hash(args.source_capture),
        "model_sha256": _file_hash(args.model),
        "plan_hash": plan.stable_hash(),
        "trace_hash": trace.stable_hash(),
        "schedule_hash": schedule.stable_hash(),
        "warmup_groups": WARMUP_GROUPS,
        "sample_groups": SAMPLE_GROUPS,
        "latency_ns": samples,
        "median_latency_ns": {
            mode: statistics.median(values) for mode, values in samples.items()
        },
        "semantic": semantic,
        "direct_receipt": asdict(direct.d2b_receipt()),
        "candidate_receipt": candidate_result.receipt.to_dict(),
        "prepare_ns": {"D": direct_prepare_ns, "P": candidate_prepare_ns},
        "memory": {
            "allocated_before": allocated_before,
            "reserved_before": reserved_before,
            "candidate_peak_dynamic_allocated": max(
                0, torch.cuda.max_memory_allocated() - allocated_before
            ),
            "candidate_peak_dynamic_reserved": max(
                0, torch.cuda.max_memory_reserved() - reserved_before
            ),
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
    parser.add_argument("--replicate-index", type=int, default=0)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    payload = _run(args)
    medians = cast(dict[str, object], payload["median_latency_ns"])
    print(
        "S3 optimizer worker "
        f"replicate={args.replicate_index} order={args.order} "
        f"N={medians['N']} D={medians['D']} P={medians['P']} "
        "performance_claimed=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
