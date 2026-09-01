#!/usr/bin/env python3
"""Diagnostic-only phase attribution for one D1-C 10/9 wrapper."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=missing-function-docstring,import-outside-toplevel

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

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
from boundflow.runtime.r3_optimizer_trajectory_timing import execute_r32b_wrapper_v1
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


def _events() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _instrument(
    candidate: PreparedR3D1CCumulativeCandidateV1,
) -> dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]:
    rows: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        "forward": [],
        "backward": [],
        "residual11": [],
        "residual6": [],
    }

    def wrap(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def measured(*args: Any, **kwargs: Any) -> Any:
            start, end = _events()
            start.record()
            result = original(*args, **kwargs)
            end.record()
            rows[name].append((start, end))
            return result

        return measured

    candidate._run_forward_fast = wrap(  # type: ignore[method-assign]
        "forward", candidate._run_forward_fast
    )
    candidate.backward = wrap("backward", candidate.backward)  # type: ignore[method-assign]
    candidate._dispatch_residual11 = wrap(  # type: ignore[method-assign]
        "residual11", candidate._dispatch_residual11
    )
    candidate._dispatch_residual6 = wrap(  # type: ignore[method-assign]
        "residual6", candidate._dispatch_residual6
    )
    return rows


def run(capture: Path, model: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("R3-D1C attribution requires CUDA")
    device = torch.device("cuda:0")
    raw = torch.load(capture, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
    tensors = bind_r31_runtime_inputs_v1(plan, module, snapshot, device=device)
    candidate = PreparedR3D1CCumulativeCandidateV1(plan, trace, tensors)
    stream = torch.cuda.Stream(device=device)
    alpha = tensors[plan.p_alpha_input_ordinal]
    initial_alpha = alpha.detach().clone()
    schedule = compile_terminal_optimizer_schedule_v1()
    for _ in range(3):
        with torch.no_grad():
            alpha.copy_(initial_alpha)
        alpha.grad = None
        with torch.cuda.stream(stream):
            execute_r32b_wrapper_v1(plan, tensors, schedule, candidate=candidate)
        stream.synchronize()
    with torch.no_grad():
        alpha.copy_(initial_alpha)
    alpha.grad = None
    rows = _instrument(candidate)
    with torch.cuda.stream(stream):
        started = time.perf_counter_ns()
        result = execute_r32b_wrapper_v1(
            plan,
            tensors,
            schedule,
            candidate=candidate,
        )
        stream.synchronize()
        host_ns = time.perf_counter_ns() - started
    elapsed_ms = {
        name: sum(start.elapsed_time(end) for start, end in pairs)
        for name, pairs in rows.items()
    }
    forward_ms = elapsed_ms["forward"]
    backward_ms = elapsed_ms["backward"]
    result_payload: dict[str, object] = {
        "schema_version": "boundflow.r3-d1c-attribution/v1",
        "host_wrapper_ms": host_ns / 1_000_000.0,
        "cuda_phase_ms": elapsed_ms,
        "residual_combined_ms": elapsed_ms["residual11"] + elapsed_ms["residual6"],
        "non_residual_forward_ms": forward_ms
        - elapsed_ms["residual11"]
        - elapsed_ms["residual6"],
        "host_uncovered_ms": host_ns / 1_000_000.0 - forward_ms - backward_ms,
        "evaluation_count": result.evaluation_count,
        "optimizer_mutation_count": result.optimizer_mutation_count,
        "scheduler_mutation_count": result.scheduler_mutation_count,
        "diagnostic_only": True,
        "formal_performance_claimed": False,
    }
    return result_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.source_capture, args.model)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
