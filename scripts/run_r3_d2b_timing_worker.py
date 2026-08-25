#!/usr/bin/env python3
"""Run one fresh native, D1-C, or D2-B wrapper timing worker."""

# pylint: disable=wrong-import-position,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring,duplicate-code,protected-access

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
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
from boundflow.runtime.fsg4_b3_terminal_optimizer_schedule import (
    compile_terminal_optimizer_schedule_v1,
)
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_d1c_cumulative_wrapper import (
    PreparedR3D1CCumulativeCandidateV1,
)
from boundflow.runtime.r3_d2b_staged_backward import (
    PreparedR3D2BStagedBackwardCandidateV1,
)
from boundflow.runtime.r3_optimizer_trajectory_timing import (
    execute_r32b_wrapper_v1,
    PreparedR32BTimingCandidateV1,
)
from boundflow.runtime.r3_structured_owner_custom_backward import (
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

WORKER_SCHEMA = "boundflow.r3-d2b-wrapper-timing-worker/v1"
WARMUP_COUNT = 3
SAMPLE_COUNT = 30


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("R3-D2B timing worker requires CUDA")
    device = torch.device("cuda:0")
    raw = torch.load(args.source_capture, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
    tensors = bind_r31_runtime_inputs_v1(plan, module, snapshot, device=device)
    alpha = tensors[plan.p_alpha_input_ordinal]
    initial_alpha = alpha.detach().clone()
    schedule = compile_terminal_optimizer_schedule_v1()
    candidate: PreparedR32BTimingCandidateV1 | None
    if args.mode == "native":
        candidate = None
    elif args.mode == "d1c":
        candidate = PreparedR3D1CCumulativeCandidateV1(plan, trace, tensors)
    else:
        candidate = PreparedR3D2BStagedBackwardCandidateV1(plan, trace, tensors)
    stream = torch.cuda.Stream(device=device)

    def reset() -> None:
        with torch.no_grad():
            alpha.copy_(initial_alpha)
        alpha.grad = None
        torch.cuda.synchronize(device)

    def execute():  # type: ignore[no-untyped-def]
        with torch.cuda.stream(stream):
            result = execute_r32b_wrapper_v1(
                plan, tensors, schedule, candidate=candidate
            )
        stream.synchronize()
        return result

    for _ in range(WARMUP_COUNT):
        reset()
        execute()
    samples_ns = []
    for _ in range(SAMPLE_COUNT):
        reset()
        started = time.perf_counter_ns()
        execute()
        samples_ns.append(time.perf_counter_ns() - started)
    reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    result = execute()
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    terminal_lower = result.terminal_lower.detach().cpu().contiguous().clone()
    terminal_alpha = result.terminal_alpha.detach().cpu().contiguous().clone()
    region_ms = None
    if candidate is not None:
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        original = candidate._coefficient_sign_pass

        def measured(*values):  # type: ignore[no-untyped-def]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = original(*values)
            end.record()
            events.append((start, end))
            return output

        candidate._coefficient_sign_pass = measured  # type: ignore[method-assign]
        try:
            reset()
            execute()
        finally:
            candidate._coefficient_sign_pass = original  # type: ignore[method-assign]
        if len(events) != 10:
            raise RuntimeError("R3-D2B region event count differs")
        region_ms = sum(start.elapsed_time(end) for start, end in events)
    d1c_receipt = None
    d2b_receipt = None
    arena_pointers = None
    if isinstance(candidate, PreparedR3D1CCumulativeCandidateV1):
        d1c_receipt = asdict(candidate.d1c_receipt())
        arena_pointers = (
            candidate.forward_executor.scratch_0.data_ptr(),
            candidate.forward_executor.scratch_1.data_ptr(),
        )
    if isinstance(candidate, PreparedR3D2BStagedBackwardCandidateV1):
        d2b_receipt = asdict(candidate.d2b_receipt())
    properties = torch.cuda.get_device_properties(device)
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "run_index": args.run_index,
        "mode": args.mode,
        "source_capture_sha256": _file_hash(args.source_capture),
        "model_sha256": _file_hash(args.model),
        "plan_hash": plan.stable_hash(),
        "trace_hash": trace.stable_hash(),
        "warmup_count": WARMUP_COUNT,
        "sample_count": SAMPLE_COUNT,
        "latency_ns": samples_ns,
        "median_latency_ns": statistics.median(samples_ns),
        "coefficient_sign_region_ms": region_ms,
        "region_event_count": 0 if region_ms is None else 10,
        "region_headline_forbidden": True,
        "terminal_lower": terminal_lower,
        "terminal_alpha": terminal_alpha,
        "terminal_lower_sha256": production_tensor_sha256(terminal_lower),
        "terminal_alpha_sha256": production_tensor_sha256(terminal_alpha),
        "execution": {
            "evaluation_count": result.evaluation_count,
            "optimizer_mutation_count": result.optimizer_mutation_count,
            "scheduler_mutation_count": result.scheduler_mutation_count,
            "custom_forward_count": result.custom_forward_count,
            "custom_backward_count": result.custom_backward_count,
            "fallback_count": result.fallback_count,
            "eager_candidate_count": result.eager_candidate_count,
            "native_shadow_count": result.native_shadow_count,
            "timing_capture_count": result.timing_capture_count,
        },
        "d1c_receipt": d1c_receipt,
        "d2b_receipt": d2b_receipt,
        "arena_pointers": arena_pointers,
        "memory": {
            "allocated_before": allocated_before,
            "reserved_before": reserved_before,
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
        },
        "environment": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "device_index": device.index,
        },
        "clock": "host-perf-counter-ns-with-device-boundary-sync",
        "performance_claimed": False,
    }
    torch.save(payload, args.result)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "d1c", "d2b"), required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.run_index not in range(5):
        raise ValueError("R3-D2B timing run index differs")
    payload = _run(args)
    print(
        f"R3-D2B timing run={args.run_index} mode={args.mode} "
        f"median_ns={payload['median_latency_ns']} performance_claimed=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
