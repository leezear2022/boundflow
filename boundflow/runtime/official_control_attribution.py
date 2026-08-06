"""Contracts and reconstruction for the official αβ-CROWN B0 control trace."""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Any, Mapping, Sequence, Tuple

from .gpu_attribution import (
    CacheState,
    CriticalPathSegment,
    FeatureActivationLedger,
    FullStackAttributionRun,
    FullStackSpan,
    ReplacementMode,
    ResourceKind,
    SolverPhase,
    StackLayer,
    summarize_run,
)

OFFICIAL_CONTROL_WORKER_SCHEMA_VERSION = "boundflow.fsg1-official-control-worker/v1"
MAXIMUM_PROFILE_PERTURBATION_RATIO = 1.05

_WORKER_FIELDS = {
    "schema_version",
    "run_id",
    "configuration_id",
    "workload_id",
    "mode",
    "repeat_index",
    "pair_order",
    "source",
    "protocol",
    "environment",
    "result",
    "scope_ns",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "calls",
    "performance_claimed",
}
_CALL_FIELDS = {
    "call_id",
    "parent_call_id",
    "depth",
    "method",
    "phase",
    "external_phase",
    "host_start_ns",
    "host_end_ns",
    "cuda_start_ns",
    "cuda_end_ns",
    "stream_id",
    "memory_allocated_before_bytes",
    "memory_allocated_after_bytes",
    "memory_reserved_before_bytes",
    "memory_reserved_after_bytes",
    "bound_lower",
    "bound_upper",
    "kwargs_keys",
}
_SOURCE_FIELDS = {
    "abcrown_commit",
    "auto_lirpa_commit",
    "vnncomp_commit",
    "model_relative_path",
    "property_relative_path",
    "model_sha256",
    "property_sha256",
}
_RESULT_FIELDS = {"status", "success", "visited_domains"}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a sequence")
    return value


def _strict_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    return value


def _strict_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _strict_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _exact_fields(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{label} fields differ")


def validate_official_control_worker(  # pylint: disable=too-many-branches
    record: Mapping[str, Any],
) -> None:
    """Fail closed on malformed control/profile worker output."""

    _exact_fields(record, _WORKER_FIELDS, "official control worker")
    if record["schema_version"] != OFFICIAL_CONTROL_WORKER_SCHEMA_VERSION:
        raise ValueError("official control worker schema differs")
    for field in ("run_id", "configuration_id", "workload_id", "pair_order"):
        _strict_str(record[field], f"worker {field}")
    if record["configuration_id"] != "B0":
        raise ValueError("official control worker configuration differs")
    mode = _strict_str(record["mode"], "worker mode")
    if mode not in {"control", "profile"}:
        raise ValueError("official control worker mode differs")
    if _strict_int(record["repeat_index"], "worker repeat") < 0:
        raise ValueError("official control worker repeat is negative")
    if _strict_int(record["scope_ns"], "worker scope") <= 0:
        raise ValueError("official control worker scope must be positive")
    for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
        if _strict_int(record[field], f"worker {field}") < 0:
            raise ValueError(f"official control worker {field} is negative")
    if record["performance_claimed"] is not False:
        raise ValueError("official control worker cannot claim performance")

    source = _mapping(record["source"], "worker source")
    _exact_fields(source, _SOURCE_FIELDS, "worker source")
    for field in _SOURCE_FIELDS:
        _strict_str(source[field], f"worker source {field}")
    protocol = _mapping(record["protocol"], "worker protocol")
    if not protocol:
        raise ValueError("official control worker protocol is empty")
    environment = _mapping(record["environment"], "worker environment")
    for field in ("python", "torch", "torch_cuda", "gpu_name", "gpu_total_memory"):
        if field not in environment:
            raise ValueError("official control worker environment differs")
    result = _mapping(record["result"], "worker result")
    _exact_fields(result, _RESULT_FIELDS, "worker result")
    _strict_str(result["status"], "worker result status")
    _strict_bool(result["success"], "worker result success")
    visited = _sequence(result["visited_domains"], "worker visited domains")
    if any(_strict_int(item, "visited domain") < 0 for item in visited):
        raise ValueError("official control visited domain is negative")

    calls = _sequence(record["calls"], "worker calls")
    if mode == "control" and calls:
        raise ValueError("unprofiled control worker cannot contain call spans")
    if mode == "profile" and not calls:
        raise ValueError("profile worker requires call spans")
    _validate_calls(calls, scope_ns=_strict_int(record["scope_ns"], "worker scope"))


# pylint: disable-next=too-many-locals
def _validate_calls(calls: Sequence[Any], *, scope_ns: int) -> None:
    by_id: dict[int, Mapping[str, Any]] = {}
    for raw in calls:
        call = _mapping(raw, "official control call")
        _exact_fields(call, _CALL_FIELDS, "official control call")
        call_id = _strict_int(call["call_id"], "call ID")
        if call_id < 0 or call_id in by_id:
            raise ValueError("official control call ID differs")
        parent = call["parent_call_id"]
        if parent is not None:
            _strict_int(parent, "call parent")
        depth = _strict_int(call["depth"], "call depth")
        host_start = _strict_int(call["host_start_ns"], "call host start")
        host_end = _strict_int(call["host_end_ns"], "call host end")
        cuda_start = _strict_int(call["cuda_start_ns"], "call CUDA start")
        cuda_end = _strict_int(call["cuda_end_ns"], "call CUDA end")
        interval_checks = (
            depth >= 0,
            host_start >= 0,
            host_end > host_start,
            cuda_start >= 0,
            cuda_end >= cuda_start,
            host_end <= scope_ns,
            cuda_end <= scope_ns,
        )
        if not all(interval_checks):
            raise ValueError("official control call interval differs")
        SolverPhase(_strict_str(call["phase"], "call phase"))
        _strict_str(call["method"], "call method")
        _strict_str(call["external_phase"], "call external phase")
        _strict_str(call["stream_id"], "call stream")
        _strict_bool(call["bound_lower"], "call bound_lower")
        _strict_bool(call["bound_upper"], "call bound_upper")
        kwargs_keys = _sequence(call["kwargs_keys"], "call kwargs keys")
        if list(kwargs_keys) != sorted(set(kwargs_keys)) or not all(
            isinstance(item, str) for item in kwargs_keys
        ):
            raise ValueError("official control call kwargs keys differ")
        for field in (
            "memory_allocated_before_bytes",
            "memory_allocated_after_bytes",
            "memory_reserved_before_bytes",
            "memory_reserved_after_bytes",
        ):
            if _strict_int(call[field], field) < 0:
                raise ValueError("official control call memory counter is negative")
        by_id[call_id] = call
    for call_id, call in by_id.items():
        parent_id = call["parent_call_id"]
        if parent_id is None:
            if call["depth"] != 0:
                raise ValueError("root official call depth differs")
            continue
        if parent_id not in by_id or parent_id == call_id:
            raise ValueError("official control call parent differs")
        parent = by_id[parent_id]
        if (
            call["depth"] != parent["depth"] + 1
            or parent["host_start_ns"] > call["host_start_ns"]
            or parent["host_end_ns"] < call["host_end_ns"]
        ):
            raise ValueError("official control call nesting differs")


def _critical_segments(
    calls: Sequence[Mapping[str, Any]], *, scope_ns: int
) -> Tuple[CriticalPathSegment, ...]:
    boundaries = sorted(
        {0, scope_ns}
        | {int(call["host_start_ns"]) for call in calls}
        | {int(call["host_end_ns"]) for call in calls}
    )
    first_call_start = min(int(call["host_start_ns"]) for call in calls)
    last_call_end = max(int(call["host_end_ns"]) for call in calls)
    raw_segments: list[tuple[StackLayer, SolverPhase, int, int, str]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        active = [
            call
            for call in calls
            if int(call["host_start_ns"]) <= start and int(call["host_end_ns"]) >= end
        ]
        if active:
            owner = max(
                active, key=lambda call: (int(call["depth"]), int(call["call_id"]))
            )
            layer = StackLayer.OPERATOR_EXECUTION
            phase = SolverPhase(str(owner["phase"]))
            source = f"host-call-{owner['call_id']}"
        else:
            layer = StackLayer.SOLVER_CONTROL
            phase = (
                SolverPhase.SETUP
                if end <= first_call_start
                else (
                    SolverPhase.TERMINATION
                    if start >= last_call_end
                    else SolverPhase.UNCLASSIFIED
                )
            )
            source = "host-root"
        if (
            raw_segments
            and raw_segments[-1][:2] == (layer, phase)
            and raw_segments[-1][4] == source
        ):
            previous = raw_segments[-1]
            raw_segments[-1] = (layer, phase, previous[2], end, source)
        else:
            raw_segments.append((layer, phase, start, end, source))
    return tuple(
        CriticalPathSegment(
            segment_id=f"critical-{ordinal}",
            layer=layer,
            phase=phase,
            start_ns=start,
            end_ns=end,
            source_span_ids=(source,),
        )
        for ordinal, (layer, phase, start, end, source) in enumerate(raw_segments)
    )


def build_official_control_run(  # pylint: disable=too-many-locals
    record: Mapping[str, Any],
) -> FullStackAttributionRun:
    """Reconstruct one B0 profile run from raw host/CUDA event records."""

    validate_official_control_worker(record)
    if record["mode"] != "profile":
        raise ValueError("full-stack reconstruction requires a profile worker")
    calls = tuple(
        _mapping(call, "official control call")
        for call in _sequence(record["calls"], "worker calls")
    )
    scope_ns = int(record["scope_ns"])
    spans: list[FullStackSpan] = [
        FullStackSpan(
            span_id="host-root",
            parent_span_id=None,
            layer=StackLayer.SOLVER_CONTROL,
            phase=SolverPhase.SETUP,
            resource=ResourceKind.HOST_THREAD,
            cache_state=CacheState.COLD_COMPILE,
            start_ns=0,
            end_ns=scope_ns,
        )
    ]
    for call in calls:
        call_id = int(call["call_id"])
        parent_id = call["parent_call_id"]
        spans.append(
            FullStackSpan(
                span_id=f"host-call-{call_id}",
                parent_span_id=(
                    "host-root" if parent_id is None else f"host-call-{parent_id}"
                ),
                layer=StackLayer.OPERATOR_EXECUTION,
                phase=SolverPhase(str(call["phase"])),
                resource=ResourceKind.HOST_THREAD,
                cache_state=CacheState.COLD_COMPILE,
                start_ns=int(call["host_start_ns"]),
                end_ns=int(call["host_end_ns"]),
            )
        )
        spans.append(
            FullStackSpan(
                span_id=f"cuda-call-{call_id}",
                parent_span_id="host-root",
                layer=StackLayer.OPERATOR_EXECUTION,
                phase=SolverPhase(str(call["phase"])),
                resource=ResourceKind.CUDA_STREAM,
                cache_state=CacheState.COLD_COMPILE,
                start_ns=int(call["cuda_start_ns"]),
                end_ns=int(call["cuda_end_ns"]),
                stream_id=str(call["stream_id"]),
            )
        )
    streams = {str(call["stream_id"]) for call in calls}
    run = FullStackAttributionRun(
        run_id=str(record["run_id"]),
        configuration_id="B0",
        scope_start_ns=0,
        scope_end_ns=scope_ns,
        spans=tuple(spans),
        critical_path=_critical_segments(calls, scope_ns=scope_ns),
        features=FeatureActivationLedger(
            bound_graph_ir_drives_execution=False,
            plan_compiled_before_execution=False,
            task_schedule_drives_execution=False,
            backend_kind="official_alpha_beta_crown_torch_cuda",
            physical_backend_dispatches=len(calls),
            fallback_dispatches=0,
            jit_cache_state=CacheState.NOT_APPLICABLE,
            stream_count=len(streams),
            event_count=2 * len(calls) + 2,
            wait_count=1,
            storage_plan_enforced=False,
            replacement_mode=ReplacementMode.ORIGINAL_PROVIDER,
        ),
    )
    run.validate()
    return run


def validate_control_profile_pair(
    control: Mapping[str, Any], profile: Mapping[str, Any]
) -> float:
    """Require same official semantics and return instrumentation perturbation."""

    validate_official_control_worker(control)
    validate_official_control_worker(profile)
    if control["mode"] != "control" or profile["mode"] != "profile":
        raise ValueError("official pair modes differ")
    same_fields = (
        "workload_id",
        "repeat_index",
        "pair_order",
        "source",
        "protocol",
        "result",
    )
    if any(control[field] != profile[field] for field in same_fields):
        raise ValueError("official control/profile semantics differ")
    control_ns = int(control["scope_ns"])
    profile_ns = int(profile["scope_ns"])
    ratio = profile_ns / control_ns
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("official profile perturbation differs")
    return ratio


# pylint: disable-next=too-many-locals
def derive_official_control_evidence(
    worker_records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Pair five fresh B0 controls/profiles and reconstruct every profile trace."""

    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in worker_records:
        validate_official_control_worker(record)
        key = (str(record["workload_id"]), int(record["repeat_index"]))
        mode = str(record["mode"])
        if mode in grouped[key]:
            raise ValueError("official control worker mode duplicates")
        grouped[key][mode] = record
    if not grouped:
        raise ValueError("official control evidence is empty")

    pairs: list[dict[str, object]] = []
    runs: list[FullStackAttributionRun] = []
    summaries: list[dict[str, object]] = []
    ratios_by_workload: dict[str, list[float]] = defaultdict(list)
    repeats_by_workload: dict[str, set[int]] = defaultdict(set)
    for (workload_id, repeat_index), records in sorted(grouped.items()):
        if set(records) != {"control", "profile"}:
            raise ValueError("official control/profile pair is incomplete")
        control = records["control"]
        profile = records["profile"]
        ratio = validate_control_profile_pair(control, profile)
        run = build_official_control_run(profile)
        summary = summarize_run(run)
        if summary["attribution_passed"] is not True:
            raise ValueError("official control attribution gate failed")
        ratios_by_workload[workload_id].append(ratio)
        repeats_by_workload[workload_id].add(repeat_index)
        runs.append(run)
        summaries.append(summary)
        pairs.append(
            {
                "workload_id": workload_id,
                "repeat_index": repeat_index,
                "pair_order": control["pair_order"],
                "control_run_id": control["run_id"],
                "profile_run_id": profile["run_id"],
                "control_scope_ns": control["scope_ns"],
                "profile_scope_ns": profile["scope_ns"],
                "perturbation_ratio": ratio,
                "semantic_match": True,
                "profile_run_hash": run.stable_hash(),
                "profile_summary_hash": summary["summary_hash"],
                "closure_passed": summary["closure_passed"],
                "residual_passed": summary["residual_passed"],
                "peak_allocated_bytes": profile["peak_allocated_bytes"],
                "peak_reserved_bytes": profile["peak_reserved_bytes"],
                "performance_claimed": False,
            }
        )

    workload_summary: dict[str, object] = {}
    all_gates_passed = True
    for workload_id, ratios in sorted(ratios_by_workload.items()):
        repeats = sorted(repeats_by_workload[workload_id])
        if repeats != list(range(len(repeats))) or len(repeats) < 1:
            raise ValueError("official control repeat coverage differs")
        median_ratio = statistics.median(ratios)
        gate = median_ratio <= MAXIMUM_PROFILE_PERTURBATION_RATIO
        all_gates_passed = all_gates_passed and gate
        workload_summary[workload_id] = {
            "repeat_count": len(ratios),
            "perturbation_ratios": ratios,
            "median_perturbation_ratio": median_ratio,
            "perturbation_gate": MAXIMUM_PROFILE_PERTURBATION_RATIO,
            "perturbation_passed": gate,
        }
    return {
        "status": "validated_b0_control" if all_gates_passed else "not_auditable",
        "configuration_id": "B0",
        "pair_count": len(pairs),
        "workloads": workload_summary,
        "pairs": pairs,
        "runs": [run.to_dict() for run in runs],
        "run_summaries": summaries,
        "performance_claimed": False,
    }
