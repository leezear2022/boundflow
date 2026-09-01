#!/usr/bin/env python3
"""Run one isolated native or compiled R3-2A optimizer trajectory."""

# pylint: disable=wrong-import-position,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import sys

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.fsg4_b3_terminal_optimizer_schedule import (
    compile_terminal_optimizer_schedule_v1,
)
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_compiled_p_alpha_vjp import (
    PreparedR31B2CompiledCustomBackwardV1,
)
from boundflow.runtime.r3_optimizer_trajectory import (
    execute_r32a_optimizer_trajectory_v1,
    R32AExecutionMode,
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
)
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

WORKER_SCHEMA = "boundflow.r3-2a-optimizer-trajectory-worker/v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_step(step):  # type: ignore[no-untyped-def]
    return {
        "metadata": step.metadata(),
        "alpha_before": step.alpha_before,
        "lower": step.lower,
        "gradient": step.gradient,
        "alpha_after": step.alpha_after,
        "optimizer_exp_avg": step.optimizer_after.exp_avg,
        "optimizer_exp_avg_sq": step.optimizer_after.exp_avg_sq,
    }


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("R3-2A worker requires CUDA")
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
    mode = R32AExecutionMode(args.mode)
    if mode == R32AExecutionMode.CANDIDATE:
        warm = PreparedR31B2CompiledCustomBackwardV1(plan, trace, tensors)
        del warm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    result = execute_r32a_optimizer_trajectory_v1(
        plan,
        trace,
        tensors,
        schedule=compile_terminal_optimizer_schedule_v1(),
        mode=mode,
    )
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "run_index": args.run_index,
        "mode": mode.value,
        "source_capture_sha256": _file_sha256(args.source_capture),
        "model_sha256": _file_sha256(args.model),
        "plan_hash": plan.stable_hash(),
        "trace_hash": trace.stable_hash(),
        "trajectory_metadata": result.metadata(),
        "trajectory_raw": {
            "initial_alpha": result.initial_alpha,
            "terminal_alpha": result.terminal_alpha,
            "steps": [_raw_step(step) for step in result.steps],
        },
        "memory": {
            "allocated_before": allocated_before,
            "reserved_before": reserved_before,
            "peak_allocated": result.peak_allocated_bytes,
            "peak_reserved": result.peak_reserved_bytes,
        },
        "environment": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "device_index": device.index,
        },
        "timing_recorded": False,
        "performance_claimed": False,
    }
    torch.save(payload, args.result)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "candidate"), required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.run_index < 0:
        raise ValueError("R3-2A worker run index differs")
    payload = _run(args)
    print(
        f"R3-2A run={payload['run_index']} mode={payload['mode']} "
        f"plan={str(payload['plan_hash'])[:12]} performance_claimed=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
