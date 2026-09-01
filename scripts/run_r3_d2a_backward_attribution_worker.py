#!/usr/bin/env python3
"""Run one fresh read-only D2-A backward microphysics attribution worker."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=too-many-statements,missing-function-docstring,duplicate-code
# pylint: disable=line-too-long

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
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
    production_tensor_sha256,
)
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

D1C_ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d1c-wrapper-formal-v1"
WORKER_SCHEMA = "boundflow.r3-d2a-backward-attribution-worker/v1"
WARMUP_COUNT = 3
MAX_WARMUP_COUNT = 10
READINESS_FORMAL_TOLERANCE = 0.10
READINESS_SPREAD_MAX = 1.05
ANCHOR_PHASE_TOLERANCE = 0.10


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _elapsed(
    rows: list[tuple[torch.cuda.Event, torch.cuda.Event]],
) -> list[float]:
    return [float(start.elapsed_time(end)) for start, end in rows]


def _formal_reference(run_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads((D1C_ARTIFACT / "summary.json").read_text(encoding="utf-8"))
    matches = sorted((D1C_ARTIFACT / "raw").glob(f"run-{run_index:02d}-*-d1c.pt"))
    if len(matches) != 1:
        raise ValueError("R3-D2A D1-C reference inventory differs")
    raw = torch.load(matches[0], map_location="cpu", weights_only=True)
    return summary["triplet_metrics"][run_index], raw


def _instrument_phases(candidate: PreparedR3D1CCumulativeCandidateV1):  # type: ignore[no-untyped-def]
    phase_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        "forward": [],
        "backward": [],
        "coefficient_sign": [],
        "effective_value": [],
        "recompute_a26": [],
    }

    def wrap_phase(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def measured(*args: Any, **kwargs: Any) -> Any:
            start, end = _event_pair()
            start.record()
            result = original(*args, **kwargs)
            end.record()
            phase_events[name].append((start, end))
            return result

        return measured

    candidate._run_forward_fast = wrap_phase(  # type: ignore[method-assign]
        "forward", candidate._run_forward_fast
    )
    candidate.backward = wrap_phase("backward", candidate.backward)  # type: ignore[method-assign]
    candidate._coefficient_sign_pass = wrap_phase(  # type: ignore[method-assign]
        "coefficient_sign", candidate._coefficient_sign_pass
    )
    candidate._effective_value_pass = wrap_phase(  # type: ignore[method-assign]
        "effective_value", candidate._effective_value_pass
    )
    candidate._recompute_a26 = wrap_phase(  # type: ignore[method-assign]
        "recompute_a26", candidate._recompute_a26
    )
    return phase_events


def _instrument_symbols(candidate: PreparedR3D1CCumulativeCandidateV1):  # type: ignore[no-untyped-def]
    symbol_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
        defaultdict(list)
    )

    def wrap_launch(kind: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def measured(symbol: str, *args: Any) -> Any:
            start, end = _event_pair()
            start.record()
            result = original(symbol, *args)
            end.record()
            symbol_events[f"{kind}:{symbol}"].append((start, end))
            return result

        return measured

    candidate._launch_b1 = wrap_launch("b1", candidate._launch_b1)  # type: ignore[method-assign]
    candidate._launch_b2 = wrap_launch("b2", candidate._launch_b2)  # type: ignore[method-assign]
    return symbol_events


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("R3-D2A worker requires CUDA")
    reference_metric, reference_raw = _formal_reference(args.run_index)
    device = torch.device("cuda:0")
    source = torch.load(args.source_capture, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(source["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
    tensors = bind_r31_runtime_inputs_v1(plan, module, snapshot, device=device)
    alpha = tensors[plan.p_alpha_input_ordinal]
    initial_alpha = alpha.detach().clone()
    schedule = compile_terminal_optimizer_schedule_v1()
    stream = torch.cuda.Stream(device=device)

    def reset() -> None:
        with torch.no_grad():
            alpha.copy_(initial_alpha)
        alpha.grad = None

    def execute_once(candidate: PreparedR3D1CCumulativeCandidateV1):  # type: ignore[no-untyped-def]
        with torch.cuda.stream(stream):
            started = time.perf_counter_ns()
            result = execute_r32b_wrapper_v1(
                plan, tensors, schedule, candidate=candidate
            )
            stream.synchronize()
            elapsed = time.perf_counter_ns() - started
        return result, elapsed

    def fixed_warmup(candidate: PreparedR3D1CCumulativeCandidateV1) -> None:
        for _ in range(WARMUP_COUNT):
            reset()
            execute_once(candidate)

    phase_candidate = PreparedR3D1CCumulativeCandidateV1(plan, trace, tensors)
    warmup_host_ns = []
    formal_d1c_ns = float(reference_metric["d1c_median_ns"])
    readiness_pass = False
    for _ in range(MAX_WARMUP_COUNT):
        reset()
        _warmup_result, elapsed_ns = execute_once(phase_candidate)
        warmup_host_ns.append(elapsed_ns)
        if len(warmup_host_ns) >= WARMUP_COUNT:
            recent = warmup_host_ns[-WARMUP_COUNT:]
            ratios = [value / formal_d1c_ns for value in recent]
            readiness_pass = (
                all(
                    1.0 - READINESS_FORMAL_TOLERANCE
                    <= ratio
                    <= 1.0 + READINESS_FORMAL_TOLERANCE
                    for ratio in ratios
                )
                and max(recent) / min(recent) <= READINESS_SPREAD_MAX
            )
            if readiness_pass:
                break
    if not readiness_pass:
        raise RuntimeError("R3-D2A phase readiness gate differs")
    reset()
    _anchor_result, anchor_host_ns = execute_once(phase_candidate)
    if not (
        1.0 - READINESS_FORMAL_TOLERANCE
        <= anchor_host_ns / formal_d1c_ns
        <= 1.0 + READINESS_FORMAL_TOLERANCE
    ):
        raise RuntimeError("R3-D2A phase anchor differs")
    reset()
    phase_events = _instrument_phases(phase_candidate)
    with torch.cuda.stream(stream):
        started = time.perf_counter_ns()
        phase_result = execute_r32b_wrapper_v1(
            plan, tensors, schedule, candidate=phase_candidate
        )
        stream.synchronize()
        host_wrapper_ns = time.perf_counter_ns() - started
    if not (
        1.0 - ANCHOR_PHASE_TOLERANCE
        <= host_wrapper_ns / anchor_host_ns
        <= 1.0 + ANCHOR_PHASE_TOLERANCE
    ):
        raise RuntimeError("R3-D2A phase/anchor calibration differs")
    phase_ms = {name: _elapsed(rows) for name, rows in phase_events.items()}
    phase_totals = {name: sum(values) for name, values in phase_ms.items()}
    backward_children = sum(
        phase_totals[name]
        for name in ("coefficient_sign", "effective_value", "recompute_a26")
    )
    terminal_residual = phase_totals["backward"] - backward_children
    if terminal_residual <= 0.0:
        raise RuntimeError("R3-D2A terminal backward residual differs")
    terminal_lower = phase_result.terminal_lower.detach().cpu().contiguous().clone()
    terminal_alpha = phase_result.terminal_alpha.detach().cpu().contiguous().clone()
    lower_diff = float(
        (terminal_lower - reference_raw["terminal_lower"]).abs().max().item()
    )
    alpha_diff = float(
        (terminal_alpha - reference_raw["terminal_alpha"]).abs().max().item()
    )
    sign_exact = torch.equal(
        torch.sign(terminal_lower), torch.sign(reference_raw["terminal_lower"])
    )
    if lower_diff > 2e-4 or alpha_diff > 2e-5 or not sign_exact:
        raise ValueError("R3-D2A instrumentation semantics differ")

    reset()
    symbol_candidate = PreparedR3D1CCumulativeCandidateV1(plan, trace, tensors)
    fixed_warmup(symbol_candidate)
    reset()
    symbol_events = _instrument_symbols(symbol_candidate)
    with torch.cuda.stream(stream):
        symbol_started = time.perf_counter_ns()
        symbol_result = execute_r32b_wrapper_v1(
            plan, tensors, schedule, candidate=symbol_candidate
        )
        stream.synchronize()
        symbol_profile_host_ns = time.perf_counter_ns() - symbol_started
    symbol_ms = {name: _elapsed(rows) for name, rows in sorted(symbol_events.items())}
    symbol_lower = symbol_result.terminal_lower.detach().cpu().contiguous().clone()
    symbol_alpha = symbol_result.terminal_alpha.detach().cpu().contiguous().clone()
    symbol_lower_diff = float((symbol_lower - terminal_lower).abs().max().item())
    symbol_alpha_diff = float((symbol_alpha - terminal_alpha).abs().max().item())
    if symbol_lower_diff > 2e-4 or symbol_alpha_diff > 2e-5:
        raise ValueError("R3-D2A symbol profile semantics differ")
    properties = torch.cuda.get_device_properties(device)
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "run_index": args.run_index,
        "source_capture_sha256": _file_hash(args.source_capture),
        "model_sha256": _file_hash(args.model),
        "d1c_manifest_sha256": _file_hash(D1C_ARTIFACT / "manifest.json"),
        "plan_hash": plan.stable_hash(),
        "trace_hash": trace.stable_hash(),
        "minimum_warmup_count": WARMUP_COUNT,
        "maximum_warmup_count": MAX_WARMUP_COUNT,
        "actual_warmup_count": len(warmup_host_ns),
        "warmup_host_ns": warmup_host_ns,
        "readiness_formal_tolerance": READINESS_FORMAL_TOLERANCE,
        "readiness_spread_max": READINESS_SPREAD_MAX,
        "anchor_phase_tolerance": ANCHOR_PHASE_TOLERANCE,
        "readiness_pass": readiness_pass,
        "anchor_host_ns": anchor_host_ns,
        "host_wrapper_ns": host_wrapper_ns,
        "formal_reference_native_ns": reference_metric["native_median_ns"],
        "formal_reference_d1c_ns": reference_metric["d1c_median_ns"],
        "phase_ms": phase_ms,
        "phase_totals_ms": phase_totals,
        "terminal_backward_residual_ms": terminal_residual,
        "symbol_profile_host_ns": symbol_profile_host_ns,
        "symbol_ms": symbol_ms,
        "terminal_lower": terminal_lower,
        "terminal_alpha": terminal_alpha,
        "terminal_lower_sha256": production_tensor_sha256(terminal_lower),
        "terminal_alpha_sha256": production_tensor_sha256(terminal_alpha),
        "reference_lower_diff": lower_diff,
        "reference_alpha_diff": alpha_diff,
        "reference_sign_exact": sign_exact,
        "symbol_phase_lower_diff": symbol_lower_diff,
        "symbol_phase_alpha_diff": symbol_alpha_diff,
        "execution": {
            "evaluation_count": phase_result.evaluation_count,
            "optimizer_mutation_count": phase_result.optimizer_mutation_count,
            "scheduler_mutation_count": phase_result.scheduler_mutation_count,
            "custom_forward_count": phase_result.custom_forward_count,
            "custom_backward_count": phase_result.custom_backward_count,
            "fallback_count": phase_result.fallback_count,
            "eager_candidate_count": phase_result.eager_candidate_count,
            "native_shadow_count": phase_result.native_shadow_count,
        },
        "environment": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "device_index": device.index,
        },
        "single_stream_no_overlap": True,
        "symbol_profile_headline_forbidden": True,
        "diagnostic_only": True,
        "performance_claimed": False,
    }
    torch.save(payload, args.result)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.run_index not in range(5):
        raise ValueError("R3-D2A run index differs")
    payload = _run(args)
    totals = payload["phase_totals_ms"]
    assert isinstance(totals, dict)
    print(
        f"R3-D2A run={args.run_index} host_ms={payload['host_wrapper_ns']/1e6:.3f} "  # type: ignore[operator]
        f"backward_ms={totals['backward']:.3f} performance_claimed=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
