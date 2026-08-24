#!/usr/bin/env python3
"""Run one fresh diagnostic-only R3-D0 profiler worker."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=too-many-statements,missing-function-docstring,duplicate-code
# pylint: disable=too-few-public-methods,import-outside-toplevel

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import gc
import hashlib
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterator, Mapping

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
from boundflow.runtime.r3_d0_microphysics_attribution import (
    canonical_hash,
    derive_worker_ledger,
    extract_torch_profiler_events,
)
from boundflow.runtime.r3_optimizer_trajectory_timing import (
    execute_r32b_wrapper_v1,
    PreparedR32BTimingCandidateV1,
    R32BWrapperResultV1,
)
from boundflow.runtime import r3_optimizer_trajectory_timing as timing_runtime
from boundflow.runtime.r3_structured_owner_custom_backward import (
    bind_r31_runtime_inputs_v1,
    compile_r31_full_region_plan_v1,
)
from boundflow.runtime import r3_structured_owner_custom_backward as owner_runtime
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
    production_tensor_sha256,
)
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

WORKER_SCHEMA = "boundflow.r3-d0-microphysics-worker/v1"
WARMUP_COUNT = 3
SAMPLE_COUNT = 30


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _MarkerState:
    evaluation_ordinal = -1
    phase = "setup"


def _phase(state: _MarkerState, root: str) -> str:
    return f"{root}.evaluation.{state.evaluation_ordinal:02d}"


def _b2_phase(state: _MarkerState, symbol: str) -> str:
    if "pack_" in symbol:
        root = "backward.sign"
    elif "effective_" in symbol:
        root = "backward.effective"
    elif "compressed_gradient" in symbol:
        root = "backward.compressed-gradient"
    elif "clear_arenas" in symbol:
        root = "backward.clear"
    else:
        root = "backward.recompute"
    return _phase(state, root)


@contextmanager
def _candidate_launch_markers(
    candidate: PreparedR32BTimingCandidateV1, state: _MarkerState
) -> Iterator[None]:
    from torch.profiler import record_function

    forward_owner = candidate.forward_executor
    original_forward = forward_owner._launch
    original_b1 = candidate._launch_b1
    original_b2 = candidate._launch_b2

    def forward(symbol: str, *tensors: torch.Tensor) -> None:
        with record_function(f"boundflow::r3d0::{_phase(state, 'forward')}::{symbol}"):
            original_forward(symbol, *tensors)

    def backward_b1(symbol: str, *tensors: torch.Tensor) -> None:
        with record_function(
            f"boundflow::r3d0::{_phase(state, 'backward.recompute')}::{symbol}"
        ):
            original_b1(symbol, *tensors)

    def backward_b2(symbol: str, *tensors: torch.Tensor) -> None:
        with record_function(f"boundflow::r3d0::{_b2_phase(state, symbol)}::{symbol}"):
            original_b2(symbol, *tensors)

    forward_owner._launch = forward  # type: ignore[method-assign]
    candidate._launch_b1 = backward_b1  # type: ignore[method-assign]
    candidate._launch_b2 = backward_b2  # type: ignore[method-assign]
    try:
        yield
    finally:
        forward_owner._launch = original_forward  # type: ignore[method-assign]
        candidate._launch_b1 = original_b1  # type: ignore[method-assign]
        candidate._launch_b2 = original_b2  # type: ignore[method-assign]


def _diagnostic_execute(
    plan: Any,
    tensors: tuple[torch.Tensor, ...],
    schedule: Any,
    candidate: PreparedR32BTimingCandidateV1 | None,
    state: _MarkerState,
) -> R32BWrapperResultV1:
    from torch.profiler import record_function

    alpha = tensors[plan.p_alpha_input_ordinal]
    if candidate is not None:
        candidate.begin_sample()
    optimizer = torch.optim.Adam([alpha], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    terminal = None
    for action in schedule.actions:
        state.evaluation_ordinal = action.evaluation_ordinal
        if candidate is None:
            with record_function(
                f"boundflow::r3d0::{_phase(state, 'forward')}::native-forward"
            ):
                lower = owner_runtime._evaluate_full_region(plan, tensors)
            with record_function(
                f"boundflow::r3d0::{_phase(state, 'backward')}::native-autograd"
            ):
                gradient = torch.autograd.grad(-lower.sum(), alpha)[0]
        else:
            candidate.begin_evaluation(action.evaluation_ordinal)
            key = hashlib.sha256(
                f"{id(candidate)}:{candidate._sample_ordinal}:{action.evaluation_ordinal}".encode()
            ).hexdigest()
            if key in timing_runtime._EXECUTOR_REGISTRY:
                raise RuntimeError("R3-D0 execution key repeats")
            timing_runtime._EXECUTOR_REGISTRY[key] = candidate
            try:
                state.phase = "forward"
                with record_function(
                    f"boundflow::r3d0::{_phase(state, 'forward')}::compiled-forward"
                ):
                    lower = timing_runtime._R31B2CompiledFunction.apply(
                        key, *candidate.tensors
                    )
                state.phase = "backward"
                with record_function(
                    f"boundflow::r3d0::{_phase(state, 'backward')}::compiled-backward"
                ):
                    gradient = torch.autograd.grad(
                        lower,
                        candidate.tensors[candidate.plan.p_alpha_input_ordinal],
                        grad_outputs=candidate.upstream_gradient,
                    )[0]
            finally:
                timing_runtime._EXECUTOR_REGISTRY.pop(key, None)
        if action.update_after:
            optimizer.zero_grad(set_to_none=True)
            alpha.grad = gradient
            with record_function(
                f"boundflow::r3d0::{_phase(state, 'optimizer.adam')}::adam-step"
            ):
                optimizer.step()
            with record_function(
                f"boundflow::r3d0::{_phase(state, 'optimizer.clamp')}::alpha-clamp"
            ):
                with torch.no_grad():
                    alpha.clamp_(0.0, 1.0)
            with record_function(
                f"boundflow::r3d0::{_phase(state, 'optimizer.scheduler')}::lr-step"
            ):
                scheduler.step()
        else:
            terminal = lower
    if terminal is None:
        raise RuntimeError("R3-D0 terminal lower is absent")
    result = R32BWrapperResultV1(
        terminal_lower=terminal.detach(),
        terminal_alpha=alpha.detach(),
        evaluation_count=10,
        optimizer_mutation_count=9,
        scheduler_mutation_count=9,
        custom_forward_count=10 if candidate is not None else 0,
        custom_backward_count=10 if candidate is not None else 0,
    )
    result.validate(candidate=candidate is not None)
    return result


def _run(args: argparse.Namespace) -> dict[str, object]:
    from torch.profiler import profile, ProfilerActivity, record_function

    if not torch.cuda.is_available():
        raise RuntimeError("R3-D0 worker requires CUDA")
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
    candidate = (
        PreparedR32BTimingCandidateV1(plan, trace, tensors)
        if args.mode == "candidate"
        else None
    )
    stream = torch.cuda.Stream(device=device)

    def reset() -> None:
        with torch.no_grad():
            alpha.copy_(initial_alpha)
        alpha.grad = None
        torch.cuda.synchronize(device)

    def execute_frozen() -> R32BWrapperResultV1:
        with torch.cuda.stream(stream):
            result = execute_r32b_wrapper_v1(
                plan, tensors, schedule, candidate=candidate
            )
        stream.synchronize()
        return result

    for _ in range(WARMUP_COUNT):
        reset()
        execute_frozen()
    samples_ns = []
    for _ in range(SAMPLE_COUNT):
        reset()
        started = time.perf_counter_ns()
        execute_frozen()
        samples_ns.append(time.perf_counter_ns() - started)
    reset()
    state = _MarkerState()
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    marker_context = (
        _candidate_launch_markers(candidate, state)
        if candidate is not None
        else nullcontext()
    )
    with marker_context:
        with profile(
            activities=activities, record_shapes=False, with_stack=False
        ) as profiler:
            with record_function("boundflow::r3d0::wrapper::wrapper"):
                cuda_start.record(stream)
                profiled_started = time.perf_counter_ns()
                with torch.cuda.stream(stream):
                    profiled_result = _diagnostic_execute(
                        plan, tensors, schedule, candidate, state
                    )
                stream.synchronize()
                profiled_host_wall_ns = time.perf_counter_ns() - profiled_started
                cuda_end.record(stream)
                cuda_end.synchronize()
    cuda_event_elapsed_ns = round(cuda_start.elapsed_time(cuda_end) * 1_000_000.0)
    events = extract_torch_profiler_events(profiler.events(), mode=args.mode)
    median_ns = round(statistics.median(samples_ns))
    ledger = derive_worker_ledger(
        events,
        mode=args.mode,
        unprofiled_median_ns=median_ns,
        profiled_host_wall_ns=profiled_host_wall_ns,
        cuda_event_elapsed_ns=cuda_event_elapsed_ns,
    )
    gc.collect()
    properties = torch.cuda.get_device_properties(device)
    terminal_lower = profiled_result.terminal_lower.cpu().contiguous().clone()
    terminal_alpha = profiled_result.terminal_alpha.cpu().contiguous().clone()
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
        "median_latency_ns": median_ns,
        "profiled_host_wall_ns": profiled_host_wall_ns,
        "cuda_event_elapsed_ns": cuda_event_elapsed_ns,
        "terminal_lower": terminal_lower,
        "terminal_alpha": terminal_alpha,
        "terminal_lower_sha256": production_tensor_sha256(terminal_lower),
        "terminal_alpha_sha256": production_tensor_sha256(terminal_alpha),
        "events": [event.to_dict() for event in events],
        "event_hash": canonical_hash([event.to_dict() for event in events]),
        "ledger": ledger,
        "environment": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "device_index": device.index,
            "profiler": "torch-profiler-cupti",
        },
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
        raise ValueError("R3-D0 worker run index differs")
    payload = _run(args)
    ledger = payload["ledger"]
    if not isinstance(ledger, Mapping):
        raise TypeError("R3-D0 worker ledger differs")
    print(
        f"R3-D0 run={payload['run_index']} mode={payload['mode']} "
        f"median_ns={payload['median_latency_ns']} "
        f"calibration={ledger['calibration_admitted']} "
        "performance_claimed=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
