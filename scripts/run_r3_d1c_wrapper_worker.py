#!/usr/bin/env python3
"""Run one fresh native, frozen-B3, or D1-C cumulative wrapper worker."""

# pylint: disable=wrong-import-position,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring,duplicate-code

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

WORKER_SCHEMA = "boundflow.r3-d1c-wrapper-worker/v1"
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
        raise RuntimeError("R3-D1C worker requires CUDA")
    device = torch.device("cuda:0")
    raw = torch.load(args.source_capture, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
    tensors = bind_r31_runtime_inputs_v1(
        plan, module, snapshot, device=device, dtype=torch.float32
    )
    alpha = tensors[plan.p_alpha_input_ordinal]
    initial_alpha = alpha.detach().clone()
    schedule = compile_terminal_optimizer_schedule_v1()
    candidate: PreparedR32BTimingCandidateV1 | None
    if args.mode == "native":
        candidate = None
    elif args.mode == "b3":
        candidate = PreparedR32BTimingCandidateV1(plan, trace, tensors)
    else:
        candidate = PreparedR3D1CCumulativeCandidateV1(plan, trace, tensors)
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
    properties = torch.cuda.get_device_properties(device)
    d1c_receipt = None
    forward_launch_count = 0
    arena_pointers = None
    if candidate is not None:
        forward_launch_count = candidate.forward_executor.launch_count
    if isinstance(candidate, PreparedR3D1CCumulativeCandidateV1):
        d1c_receipt = asdict(candidate.d1c_receipt())
        arena_pointers = (
            candidate.forward_executor.scratch_0.data_ptr(),
            candidate.forward_executor.scratch_1.data_ptr(),
        )
        receipt_pointers = d1c_receipt["scratch_region_pointers"]
        if receipt_pointers != (
            arena_pointers[1] + 6144 * 4,
            arena_pointers[0] + 12288 * 4,
        ):
            raise RuntimeError("R3-D1C arena offset receipt differs")
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
            "forward_launch_count_last_evaluation": forward_launch_count,
            "fallback_count": result.fallback_count,
            "eager_candidate_count": result.eager_candidate_count,
            "native_shadow_count": result.native_shadow_count,
            "timing_capture_count": result.timing_capture_count,
        },
        "d1c_receipt": d1c_receipt,
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
        "formal_performance_claimed": False,
    }
    torch.save(payload, args.result)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "b3", "d1c"), required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.run_index < 0:
        raise ValueError("R3-D1C worker run index differs")
    payload = _run(args)
    print(
        f"R3-D1C run={payload['run_index']} mode={payload['mode']} "
        f"median_ns={payload['median_latency_ns']} formal_performance_claimed=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
