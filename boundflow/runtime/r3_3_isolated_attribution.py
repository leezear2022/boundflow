"""Replayable R3-3 isolated profiler events, conservation, and route gates."""

# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import statistics
from typing import Any, Mapping, Sequence

EVENT_SCHEMA = "boundflow.r3-3-isolated-profiler-event/v1"
LEDGER_SCHEMA = "boundflow.r3-3-isolated-attribution-ledger/v1"
ROUTE_SCHEMA = "boundflow.r3-3-isolated-route/v1"
MARKER_PREFIX = "boundflow::r33attr::"
CURRENT_SPEEDUP = 0.6682752922794841
TARGET_SPEEDUP = 1.05
MAX_REQUIRED_BUCKET_SPEEDUP = 10.0
MAX_PROFILE_PERTURBATION = 1.20
MAX_UNEXPLAINED_SHARE = 0.05


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class R33ProfilerEventV1:
    """One path-free explicit marker or correlated CUDA kernel."""

    ordinal: int
    kind: str
    name: str
    phase: str
    start_ns: int
    end_ns: int
    correlation_id: int
    stream_id: int
    attribution_method: str
    marker_ordinal: int | None
    schema_version: str = EVENT_SCHEMA

    def validate(self) -> None:
        if (
            self.schema_version != EVENT_SCHEMA
            or self.ordinal < 0
            or self.kind not in {"marker", "cuda_kernel"}
            or not self.name
            or not self.phase
            or self.start_ns < 0
            or self.end_ns <= self.start_ns
            or self.correlation_id < 0
            or self.stream_id < -1
            or self.attribution_method
            not in {"explicit_marker", "correlation_parent", "marker_containment"}
            or (self.kind == "marker" and self.marker_ordinal is None)
            or (self.marker_ordinal is not None and self.marker_ordinal < 0)
        ):
            raise ValueError("R3-3 attribution event differs")
        if any(token in self.name for token in ("/home/", "file://", "\\Users\\")):
            raise ValueError("R3-3 attribution event leaks a local path")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "ordinal": self.ordinal,
            "kind": self.kind,
            "name": self.name,
            "phase": self.phase,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.end_ns - self.start_ns,
            "correlation_id": self.correlation_id,
            "stream_id": self.stream_id,
            "attribution_method": self.attribution_method,
            "marker_ordinal": self.marker_ordinal,
        }


def event_from_dict(value: Mapping[str, Any]) -> R33ProfilerEventV1:
    expected = {
        "schema_version",
        "ordinal",
        "kind",
        "name",
        "phase",
        "start_ns",
        "end_ns",
        "duration_ns",
        "correlation_id",
        "stream_id",
        "attribution_method",
        "marker_ordinal",
    }
    if set(value) != expected:
        raise ValueError("R3-3 attribution event fields differ")
    event = R33ProfilerEventV1(
        schema_version=str(value["schema_version"]),
        ordinal=int(value["ordinal"]),
        kind=str(value["kind"]),
        name=str(value["name"]),
        phase=str(value["phase"]),
        start_ns=int(value["start_ns"]),
        end_ns=int(value["end_ns"]),
        correlation_id=int(value["correlation_id"]),
        stream_id=int(value["stream_id"]),
        attribution_method=str(value["attribution_method"]),
        marker_ordinal=(
            None if value["marker_ordinal"] is None else int(value["marker_ordinal"])
        ),
    )
    event.validate()
    if event.to_dict() != dict(value):
        raise ValueError("R3-3 attribution event derivation differs")
    return event


def _device_type(value: object) -> str:
    text = str(value).lower()
    if "cuda" in text:
        return "cuda"
    if "cpu" in text:
        return "cpu"
    return text


def _range_ns(event: Any) -> tuple[int, int]:
    time_range = getattr(event, "time_range", None)
    start_us = float(getattr(time_range, "start", 0.0))
    end_us = float(getattr(time_range, "end", start_us))
    return round(start_us * 1000.0), round(end_us * 1000.0)


def _marker_phase(name: str) -> str | None:
    return name[len(MARKER_PREFIX) :] if name.startswith(MARKER_PREFIX) else None


def _ancestor_marker(event: Any) -> Any | None:
    current = event
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if _marker_phase(str(getattr(current, "name", ""))) is not None:
            return current
        current = getattr(current, "cpu_parent", None)
    return None


def extract_profiler_events(events: Sequence[Any]) -> tuple[R33ProfilerEventV1, ...]:
    cpu_events = [
        event
        for event in events
        if _device_type(getattr(event, "device_type", "")) == "cpu"
    ]
    markers = [
        event
        for event in cpu_events
        if _marker_phase(str(getattr(event, "name", ""))) is not None
    ]
    markers.sort(key=lambda event: (*_range_ns(event), str(getattr(event, "name", ""))))
    marker_ordinals = {id(marker): ordinal for ordinal, marker in enumerate(markers)}
    cpu_by_id: dict[int, list[Any]] = defaultdict(list)
    for event in cpu_events:
        cpu_by_id[int(getattr(event, "id", 0))].append(event)
    rows: list[R33ProfilerEventV1] = []
    for marker in markers:
        start_ns, end_ns = _range_ns(marker)
        phase = _marker_phase(str(getattr(marker, "name", "")))
        if phase is None or end_ns <= start_ns:
            continue
        rows.append(
            R33ProfilerEventV1(
                ordinal=len(rows),
                kind="marker",
                name=str(getattr(marker, "name", "")),
                phase=phase,
                start_ns=start_ns,
                end_ns=end_ns,
                correlation_id=max(int(getattr(marker, "id", 0)), 0),
                stream_id=-1,
                attribution_method="explicit_marker",
                marker_ordinal=marker_ordinals[id(marker)],
            )
        )
    for event in events:
        if _device_type(getattr(event, "device_type", "")) != "cuda" or bool(
            getattr(event, "is_user_annotation", False)
        ):
            continue
        start_ns, end_ns = _range_ns(event)
        if end_ns <= start_ns:
            continue
        linked = int(getattr(event, "linked_correlation_id", getattr(event, "id", 0)))
        correlated = []
        for key in (linked, int(getattr(event, "id", 0))):
            for candidate in cpu_by_id.get(key, ()):
                marker = _ancestor_marker(candidate)
                if marker is not None:
                    correlated.append(marker)
        marker = (
            min(correlated, key=lambda item: _range_ns(item)[1] - _range_ns(item)[0])
            if correlated
            else None
        )
        method = "correlation_parent"
        if marker is None:
            containing = [
                item
                for item in markers
                if _range_ns(item)[0] <= start_ns and end_ns <= _range_ns(item)[1]
            ]
            if containing:
                marker = min(
                    containing, key=lambda item: _range_ns(item)[1] - _range_ns(item)[0]
                )
                method = "marker_containment"
        phase = (
            "unattributed"
            if marker is None
            else str(_marker_phase(str(getattr(marker, "name", ""))))
        )
        rows.append(
            R33ProfilerEventV1(
                ordinal=len(rows),
                kind="cuda_kernel",
                name=str(getattr(event, "name", "")),
                phase=phase,
                start_ns=start_ns,
                end_ns=end_ns,
                correlation_id=max(linked, 0),
                stream_id=int(getattr(event, "device_resource_id", -1)),
                attribution_method=method,
                marker_ordinal=(
                    None if marker is None else marker_ordinals[id(marker)]
                ),
            )
        )
    if not rows or not any(row.kind == "cuda_kernel" for row in rows):
        raise ValueError("R3-3 attribution captured no CUDA activity")
    return tuple(rows)


Interval = tuple[int, int]


def _merge(intervals: Sequence[Interval]) -> list[Interval]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    result = []
    left, right = ordered[0]
    for start, end in ordered[1:]:
        if start <= right:
            right = max(right, end)
        else:
            result.append((left, right))
            left, right = start, end
    result.append((left, right))
    return result


def _subtract(
    intervals: Sequence[Interval], occupied: Sequence[Interval]
) -> list[Interval]:
    remaining = _merge(intervals)
    for block_left, block_right in _merge(occupied):
        next_rows = []
        for left, right in remaining:
            if block_right <= left or right <= block_left:
                next_rows.append((left, right))
                continue
            if left < block_left:
                next_rows.append((left, block_left))
            if block_right < right:
                next_rows.append((block_right, right))
        remaining = next_rows
    return remaining


def _duration(intervals: Sequence[Interval]) -> int:
    return sum(right - left for left, right in _merge(intervals))


def _clip(intervals: Sequence[Interval], boundary: Interval) -> list[Interval]:
    left, right = boundary
    return [
        (max(start, left), min(end, right))
        for start, end in intervals
        if max(start, left) < min(end, right)
    ]


def derive_ledger(
    events: Sequence[R33ProfilerEventV1],
    *,
    unprofiled_median_ns: int,
    profiled_cuda_event_ns: int,
    calibration_cuda_event_ns: int,
) -> dict[str, object]:
    if (
        unprofiled_median_ns <= 0
        or profiled_cuda_event_ns <= 0
        or calibration_cuda_event_ns <= 0
        or tuple(event.ordinal for event in events) != tuple(range(len(events)))
    ):
        raise ValueError("R3-3 attribution ledger input differs")
    for event in events:
        event.validate()
    markers = [event for event in events if event.kind == "marker"]
    kernels = [event for event in events if event.kind == "cuda_kernel"]
    wrapper = [event for event in markers if event.phase == "wrapper"]
    calibration = [event for event in kernels if event.phase == "calibration"]
    if len(wrapper) != 1 or not calibration:
        phases = sorted((event.kind, event.phase) for event in events)
        raise ValueError(
            "R3-3 attribution marker inventory differs: "
            f"wrapper={len(wrapper)} calibration={len(calibration)} phases={phases}"
        )
    boundary = (wrapper[0].start_ns, wrapper[0].end_ns)
    wrapper_ns = boundary[1] - boundary[0]
    forward_kernel = _clip(
        [(row.start_ns, row.end_ns) for row in kernels if row.phase == "forward-ffi"],
        boundary,
    )
    backward_kernel = _clip(
        [(row.start_ns, row.end_ns) for row in kernels if row.phase == "backward-ffi"],
        boundary,
    )
    bridge = _clip(
        [
            (row.start_ns, row.end_ns)
            for row in markers
            if row.phase in {"forward-ffi", "backward-ffi"}
        ],
        boundary,
    )
    autograd = _clip(
        [
            (row.start_ns, row.end_ns)
            for row in markers
            if row.phase
            in {
                "autograd-apply",
                "autograd-grad",
                "output-allocation-forward",
                "output-allocation-backward",
            }
        ],
        boundary,
    )
    other = _clip(
        [
            (row.start_ns, row.end_ns)
            for row in markers
            if row.phase == "prepare-executor"
        ],
        boundary,
    )
    assigned: list[Interval] = []
    buckets: dict[str, int] = {}
    for name, intervals in (
        ("forward_kernel_union", forward_kernel),
        ("backward_kernel_union", backward_kernel),
        ("bridge_launch_idle", bridge),
        ("autograd_allocation", autograd),
        ("other_explained", other),
    ):
        exclusive = _subtract(intervals, assigned)
        buckets[name] = _duration(exclusive)
        assigned.extend(exclusive)
    wrapper_interval = [boundary]
    buckets["unexplained"] = _duration(_subtract(wrapper_interval, assigned))
    total = sum(buckets.values())
    conservation_error_ns = abs(total - wrapper_ns)
    calibration_kernel_ns = _duration(
        [(row.start_ns, row.end_ns) for row in calibration]
    )
    calibration_residual_ns = abs(calibration_cuda_event_ns - calibration_kernel_ns)
    calibration_threshold_ns = max(5_000, round(calibration_cuda_event_ns * 0.02))
    perturbation_ratio = profiled_cuda_event_ns / unprofiled_median_ns
    fallback_count = sum(
        event.kind == "cuda_kernel" and event.attribution_method == "marker_containment"
        for event in events
    )
    unattributed_count = sum(
        event.kind == "cuda_kernel" and event.phase == "unattributed"
        for event in events
    )
    shares = {name: value / wrapper_ns for name, value in buckets.items()}
    conservation_threshold_ns = max(10_000, round(wrapper_ns * 0.05))
    admitted = (
        calibration_residual_ns <= calibration_threshold_ns
        and perturbation_ratio <= MAX_PROFILE_PERTURBATION
        and conservation_error_ns <= conservation_threshold_ns
        and shares["unexplained"] <= MAX_UNEXPLAINED_SHARE
        and fallback_count <= math.floor(len(kernels) * 0.05)
        and unattributed_count == 0
        and len({event.stream_id for event in kernels if event.phase != "calibration"})
        == 1
    )
    admission_failures = []
    if calibration_residual_ns > calibration_threshold_ns:
        admission_failures.append("calibration-residual")
    if perturbation_ratio > MAX_PROFILE_PERTURBATION:
        admission_failures.append("profiler-perturbation")
    if conservation_error_ns > conservation_threshold_ns:
        admission_failures.append("conservation")
    if shares["unexplained"] > MAX_UNEXPLAINED_SHARE:
        admission_failures.append("unexplained-share")
    if fallback_count > math.floor(len(kernels) * 0.05):
        admission_failures.append("containment-fallback")
    if unattributed_count != 0:
        admission_failures.append("unattributed-cuda")
    if len({event.stream_id for event in kernels if event.phase != "calibration"}) != 1:
        admission_failures.append("multi-stream")
    if admitted != (not admission_failures):
        raise AssertionError("R3-3 attribution admission derivation differs")
    ledger: dict[str, object] = {
        "schema_version": LEDGER_SCHEMA,
        "event_count": len(events),
        "event_hash": canonical_hash([event.to_dict() for event in events]),
        "unprofiled_median_ns": unprofiled_median_ns,
        "profiled_cuda_event_ns": profiled_cuda_event_ns,
        "profiled_wrapper_marker_ns": wrapper_ns,
        "profile_perturbation_ratio": perturbation_ratio,
        "calibration_cuda_event_ns": calibration_cuda_event_ns,
        "calibration_kernel_ns": calibration_kernel_ns,
        "calibration_residual_ns": calibration_residual_ns,
        "calibration_threshold_ns": calibration_threshold_ns,
        "bucket_ns": buckets,
        "bucket_share": shares,
        "conservation_error_ns": conservation_error_ns,
        "conservation_threshold_ns": conservation_threshold_ns,
        "fallback_count": fallback_count,
        "unattributed_count": unattributed_count,
        "stream_ids": sorted(
            {event.stream_id for event in kernels if event.phase != "calibration"}
        ),
        "admission_failures": admission_failures,
        "attribution_admitted": admitted,
        "performance_claimed": False,
    }
    ledger["ledger_hash"] = canonical_hash(ledger)
    return ledger


def _required_bucket_speedup(share: float, total_required: float) -> float | None:
    denominator = 1.0 / total_required - (1.0 - share)
    return None if denominator <= 0.0 else share / denominator


def derive_route(ledgers: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    if len(ledgers) != 5 or any(
        ledger.get("schema_version") != LEDGER_SCHEMA
        or ledger.get("attribution_admitted") is not True
        for ledger in ledgers
    ):
        raise ValueError("R3-3 route ledger boundary differs")
    total_required = TARGET_SPEEDUP / CURRENT_SPEEDUP
    names = (
        "forward_kernel_union",
        "backward_kernel_union",
        "bridge_launch_idle",
        "autograd_allocation",
        "other_explained",
        "unexplained",
    )
    shares: dict[str, list[float]] = {
        name: [float(ledger["bucket_share"][name]) for ledger in ledgers]
        for name in names
    }
    groups = {
        "kernel": [
            shares["forward_kernel_union"][index]
            + shares["backward_kernel_union"][index]
            for index in range(5)
        ],
        "bridge": shares["bridge_launch_idle"],
        "autograd": shares["autograd_allocation"],
        "cumulative": [
            shares["bridge_launch_idle"][index] + shares["autograd_allocation"][index]
            for index in range(5)
        ],
    }
    rows = {}
    for name, values in groups.items():
        minimum = min(values)
        required = _required_bucket_speedup(minimum, total_required)
        rows[name] = {
            "shares": values,
            "minimum_share": minimum,
            "median_share": statistics.median(values),
            "maximum_share": max(values),
            "required_speedup_from_minimum": required,
            "within_10x": required is not None
            and required <= MAX_REQUIRED_BUCKET_SPEEDUP,
        }
    if rows["kernel"]["within_10x"]:
        route = "KERNEL"
    elif rows["bridge"]["within_10x"]:
        route = "BRIDGE"
    elif rows["autograd"]["within_10x"]:
        route = "AUTOGRAD"
    elif rows["cumulative"]["within_10x"]:
        route = "CUMULATIVE"
    else:
        route = "STOP"
    result: dict[str, object] = {
        "schema_version": ROUTE_SCHEMA,
        "current_speedup": CURRENT_SPEEDUP,
        "target_speedup": TARGET_SPEEDUP,
        "total_required_speedup": total_required,
        "minimum_single_bucket_share": 1.0 - 1.0 / total_required,
        "bucket_share_statistics": {
            name: {
                "values": values,
                "minimum": min(values),
                "median": statistics.median(values),
                "maximum": max(values),
            }
            for name, values in shares.items()
        },
        "route_rows": rows,
        "route": route,
        "r3_4_open": False,
        "same_solver_open": False,
        "performance_claimed": False,
    }
    result["route_hash"] = canonical_hash(result)
    return result


def derive_route_or_stop(ledgers: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Derive the frozen route, or fail closed when attribution is inadmissible."""

    if len(ledgers) != 5:
        raise ValueError("R3-3 route ledger count differs")
    if all(ledger.get("attribution_admitted") is True for ledger in ledgers):
        return derive_route(ledgers)
    names = (
        "forward_kernel_union",
        "backward_kernel_union",
        "bridge_launch_idle",
        "autograd_allocation",
        "other_explained",
        "unexplained",
    )
    diagnostics: dict[str, dict[str, object]] = {}
    for name in names:
        values = [float(ledger["bucket_share"][name]) for ledger in ledgers]
        diagnostics[name] = {
            "values": values,
            "minimum": min(values),
            "median": statistics.median(values),
            "maximum": max(values),
            "admissible_for_route": False,
        }
    failed = [
        index
        for index, ledger in enumerate(ledgers)
        if ledger.get("attribution_admitted") is not True
    ]
    failure_counts: dict[str, int] = defaultdict(int)
    for index in failed:
        failures = ledgers[index].get("admission_failures")
        if not isinstance(failures, list) or not failures:
            raise ValueError("R3-3 failed ledger reasons differ")
        for failure in failures:
            failure_counts[str(failure)] += 1
    result: dict[str, object] = {
        "schema_version": ROUTE_SCHEMA,
        "current_speedup": CURRENT_SPEEDUP,
        "target_speedup": TARGET_SPEEDUP,
        "total_required_speedup": TARGET_SPEEDUP / CURRENT_SPEEDUP,
        "minimum_single_bucket_share": 1.0 - CURRENT_SPEEDUP / TARGET_SPEEDUP,
        "route": "STOP",
        "route_reason": "attribution-quality",
        "failed_run_ordinals": failed,
        "failure_counts": dict(sorted(failure_counts.items())),
        "diagnostic_bucket_share_statistics": diagnostics,
        "diagnostic_shares_admitted": False,
        "r3_4_open": False,
        "same_solver_open": False,
        "performance_claimed": False,
    }
    result["route_hash"] = canonical_hash(result)
    return result


__all__ = [
    "R33ProfilerEventV1",
    "canonical_hash",
    "derive_ledger",
    "derive_route",
    "derive_route_or_stop",
    "event_from_dict",
    "extract_profiler_events",
]
