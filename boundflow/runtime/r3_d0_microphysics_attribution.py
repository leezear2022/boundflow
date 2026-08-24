"""Replayable R3-D0 profiler events and Amdahl route gates."""

# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=too-many-boolean-expressions,too-many-locals,too-many-branches

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import statistics
from typing import Any, Mapping, Sequence

EVENT_SCHEMA = "boundflow.r3-d0-profiler-event/v1"
LEDGER_SCHEMA = "boundflow.r3-d0-microphysics-ledger/v1"
ROUTE_SCHEMA = "boundflow.r3-d0-amdahl-route/v1"
TARGET_SPEEDUP = 1.20
MAX_REQUIRED_REGION_SPEEDUP = 10.0
MARKER_PREFIX = "boundflow::r3d0::"


def canonical_hash(value: object) -> str:
    """Return the stable digest used by D0 artifacts and replay."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class R3D0ProfilerEventV1:
    """One path-free marker or CUDA activity row."""

    ordinal: int
    kind: str
    name: str
    phase: str
    family: str
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
            or not self.family
            or self.start_ns < 0
            or self.end_ns <= self.start_ns
            or self.correlation_id < 0
            or self.stream_id < -1
            or self.attribution_method
            not in {"explicit_marker", "correlation_parent", "marker_containment"}
            or (self.kind == "marker" and self.marker_ordinal is None)
            or (self.marker_ordinal is not None and self.marker_ordinal < 0)
        ):
            raise ValueError("R3-D0 profiler event differs")
        if any(
            token in self.name or token in self.family
            for token in ("/home/", "file://", "\\Users\\")
        ):
            raise ValueError("R3-D0 profiler event leaks a local path")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "ordinal": self.ordinal,
            "kind": self.kind,
            "name": self.name,
            "phase": self.phase,
            "family": self.family,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.end_ns - self.start_ns,
            "correlation_id": self.correlation_id,
            "stream_id": self.stream_id,
            "attribution_method": self.attribution_method,
            "marker_ordinal": self.marker_ordinal,
        }


def event_from_dict(value: Mapping[str, Any]) -> R3D0ProfilerEventV1:
    """Parse a canonical event, including its derived duration."""

    expected = {
        "schema_version",
        "ordinal",
        "kind",
        "name",
        "phase",
        "family",
        "start_ns",
        "end_ns",
        "duration_ns",
        "correlation_id",
        "stream_id",
        "attribution_method",
        "marker_ordinal",
    }
    if set(value) != expected:
        raise ValueError("R3-D0 profiler event fields differ")
    event = R3D0ProfilerEventV1(
        schema_version=str(value["schema_version"]),
        ordinal=int(value["ordinal"]),
        kind=str(value["kind"]),
        name=str(value["name"]),
        phase=str(value["phase"]),
        family=str(value["family"]),
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
        raise ValueError("R3-D0 profiler event derivation differs")
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


def _marker_parts(name: str) -> tuple[str, str] | None:
    if not name.startswith(MARKER_PREFIX):
        return None
    parts = name[len(MARKER_PREFIX) :].split("::", maxsplit=1)
    return parts[0], parts[0] if len(parts) == 1 else parts[1]


def _ancestor_marker(event: Any) -> Any | None:
    current = event
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if _marker_parts(str(getattr(current, "name", ""))) is not None:
            return current
        current = getattr(current, "cpu_parent", None)
    return None


def _native_kernel_family(name: str) -> str:
    lowered = name.lower()
    if "cudnn" in lowered and "conv" in lowered:
        return "cudnn-convolution"
    if "gemm" in lowered or "matmul" in lowered:
        return "gemm-matmul"
    if "elementwise" in lowered or "vectorized" in lowered:
        return "elementwise"
    if "reduce" in lowered:
        return "reduction"
    if "index" in lowered or "gather" in lowered or "scatter" in lowered:
        return "indexing"
    # Kernel names are retained only as a stable family digest, not a local path.
    return f"kernel-{hashlib.sha256(name.encode()).hexdigest()[:16]}"


def _compiled_kernel_family(name: str) -> str | None:
    if not name.startswith("boundflow_"):
        return None
    stem = name.rsplit("_kernel", maxsplit=1)[0]
    return stem if stem else None


def extract_torch_profiler_events(
    events: Sequence[Any], *, mode: str
) -> tuple[R3D0ProfilerEventV1, ...]:
    """Extract explicit markers and attributed CUDA kernels from one session."""

    if mode not in {"native", "candidate"}:
        raise ValueError("R3-D0 profiler mode differs")
    cpu_events = [
        event
        for event in events
        if _device_type(getattr(event, "device_type", "")) == "cpu"
    ]
    markers = [
        event
        for event in cpu_events
        if _marker_parts(str(getattr(event, "name", ""))) is not None
    ]
    markers.sort(key=lambda event: (*_range_ns(event), str(getattr(event, "name", ""))))
    marker_ordinals = {id(marker): ordinal for ordinal, marker in enumerate(markers)}
    cpu_by_id: dict[int, list[Any]] = defaultdict(list)
    for event in cpu_events:
        cpu_by_id[int(getattr(event, "id", 0))].append(event)
    rows: list[R3D0ProfilerEventV1] = []
    for marker in markers:
        parts = _marker_parts(str(getattr(marker, "name", "")))
        if parts is None:
            continue
        start_ns, end_ns = _range_ns(marker)
        rows.append(
            R3D0ProfilerEventV1(
                ordinal=len(rows),
                kind="marker",
                name=str(getattr(marker, "name", "")),
                phase=parts[0],
                family=parts[1],
                start_ns=start_ns,
                end_ns=end_ns,
                correlation_id=max(int(getattr(marker, "id", 0)), 0),
                stream_id=-1,
                attribution_method="explicit_marker",
                marker_ordinal=marker_ordinals[id(marker)],
            )
        )
    for event in events:
        if _device_type(getattr(event, "device_type", "")) != "cuda":
            continue
        name = str(getattr(event, "name", ""))
        if bool(getattr(event, "is_user_annotation", False)):
            continue
        start_ns, end_ns = _range_ns(event)
        if end_ns <= start_ns:
            continue
        marker = None
        linked = int(getattr(event, "linked_correlation_id", getattr(event, "id", 0)))
        correlation_keys = (linked, int(getattr(event, "id", 0)))
        correlated_markers = []
        for correlation_key in correlation_keys:
            for candidate in cpu_by_id.get(correlation_key, ()):
                candidate_marker = _ancestor_marker(candidate)
                if candidate_marker is not None:
                    correlated_markers.append(candidate_marker)
        if correlated_markers:
            marker = min(
                correlated_markers,
                key=lambda item: _range_ns(item)[1] - _range_ns(item)[0],
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
                    containing,
                    key=lambda item: _range_ns(item)[1] - _range_ns(item)[0],
                )
                method = "marker_containment"
        parts = (
            None if marker is None else _marker_parts(str(getattr(marker, "name", "")))
        )
        phase = "unattributed" if parts is None else parts[0]
        compiled_family = _compiled_kernel_family(name)
        family = (
            compiled_family
            if mode == "candidate" and compiled_family is not None
            else (
                _native_kernel_family(name)
                if mode == "native" or parts is None
                else parts[1]
            )
        )
        rows.append(
            R3D0ProfilerEventV1(
                ordinal=len(rows),
                kind="cuda_kernel",
                name=name,
                phase=phase,
                family=family,
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
        raise ValueError("R3-D0 profiler captured no CUDA activity")
    return tuple(rows)


def _union_ns(events: Sequence[R3D0ProfilerEventV1]) -> int:
    intervals = sorted((event.start_ns, event.end_ns) for event in events)
    if not intervals:
        return 0
    total = 0
    left, right = intervals[0]
    for start, end in intervals[1:]:
        if start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + right - left


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1)]


def derive_worker_ledger(
    events: Sequence[R3D0ProfilerEventV1],
    *,
    mode: str,
    unprofiled_median_ns: int,
    profiled_host_wall_ns: int,
    cuda_event_elapsed_ns: int,
) -> dict[str, object]:
    """Rebuild one worker's physical timing ledger from raw activity rows."""

    if (
        mode not in {"native", "candidate"}
        or unprofiled_median_ns <= 0
        or profiled_host_wall_ns <= 0
        or cuda_event_elapsed_ns <= 0
        or not events
        or tuple(event.ordinal for event in events) != tuple(range(len(events)))
    ):
        raise ValueError("R3-D0 worker ledger input differs")
    for event in events:
        event.validate()
    markers = [event for event in events if event.kind == "marker"]
    kernels = [event for event in events if event.kind == "cuda_kernel"]
    if not markers or not kernels:
        raise ValueError("R3-D0 worker event inventory differs")
    wrapper_markers = [event for event in markers if event.phase == "wrapper"]
    if len(wrapper_markers) != 1:
        raise ValueError("R3-D0 wrapper marker inventory differs")
    marker_host_ns = wrapper_markers[0].end_ns - wrapper_markers[0].start_ns
    kernel_union_ns = _union_ns(kernels)
    kernel_sum_ns = sum(event.end_ns - event.start_ns for event in kernels)
    kernel_envelope_ns = max(event.end_ns for event in kernels) - min(
        event.start_ns for event in kernels
    )
    host_marker_residual_ns = abs(profiled_host_wall_ns - marker_host_ns)
    cuda_envelope_residual_ns = abs(cuda_event_elapsed_ns - kernel_envelope_ns)
    host_threshold_ns = max(5_000_000, round(profiled_host_wall_ns * 0.05))
    cuda_threshold_ns = max(2_000_000, round(cuda_event_elapsed_ns * 0.05))
    fallback_count = sum(
        event.attribution_method == "marker_containment" for event in kernels
    )
    unattributed_count = sum(event.phase == "unattributed" for event in kernels)
    calibration_admitted = (
        host_marker_residual_ns <= host_threshold_ns
        and cuda_envelope_residual_ns <= cuda_threshold_ns
        and fallback_count <= math.floor(len(kernels) * 0.05)
        and unattributed_count == 0
    )
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    phase_grouped: dict[str, list[R3D0ProfilerEventV1]] = defaultdict(list)
    for event in kernels:
        duration = event.end_ns - event.start_ns
        phase_bucket = event.phase.split(".evaluation.", maxsplit=1)[0]
        grouped[(phase_bucket, event.family)].append(duration)
        root = event.phase.split(".", maxsplit=1)[0]
        phase_grouped[root].append(event)
    families = [
        {
            "phase": phase,
            "family": family,
            "kernel_count": len(durations),
            "kernel_sum_ns": sum(durations),
            "kernel_share": sum(durations) / kernel_sum_ns,
            "p50_ns": round(statistics.median(durations)),
            "p95_ns": _percentile(durations, 0.95),
        }
        for (phase, family), durations in sorted(grouped.items())
    ]
    phase_rows = {
        phase: {
            "kernel_count": len(rows),
            "kernel_sum_ns": sum(row.end_ns - row.start_ns for row in rows),
            "kernel_union_ns": _union_ns(rows),
        }
        for phase, rows in sorted(phase_grouped.items())
    }
    host_residual_ns = max(unprofiled_median_ns - kernel_union_ns, 0)
    compiled_kernels = [
        event
        for event in kernels
        if mode == "candidate" and _compiled_kernel_family(event.name) is not None
    ]
    compiled_kernel_sum_ns = sum(
        event.end_ns - event.start_ns for event in compiled_kernels
    )
    ledger: dict[str, object] = {
        "schema_version": LEDGER_SCHEMA,
        "mode": mode,
        "unprofiled_median_ns": unprofiled_median_ns,
        "profiled_host_wall_ns": profiled_host_wall_ns,
        "marker_host_ns": marker_host_ns,
        "cuda_event_elapsed_ns": cuda_event_elapsed_ns,
        "kernel_envelope_ns": kernel_envelope_ns,
        "kernel_union_ns": kernel_union_ns,
        "kernel_sum_ns": kernel_sum_ns,
        "kernel_overlap_ns": kernel_sum_ns - kernel_union_ns,
        "host_residual_ns": host_residual_ns,
        "kernel_count": len(kernels),
        "compiled_launch_marker_count": sum(
            event.family.startswith("boundflow_") for event in markers
        ),
        "compiled_region": {
            "kernel_count": len(compiled_kernels),
            "kernel_sum_ns": compiled_kernel_sum_ns,
            "kernel_union_ns": _union_ns(compiled_kernels),
            "wrapper_share": compiled_kernel_sum_ns / unprofiled_median_ns,
            "persistent_dense_a": False if mode == "candidate" else None,
            "scratch_buffer_count": 2 if mode == "candidate" else None,
            "semantic_closure_source": (
                "R3-2A-validated-full-region-10/9" if mode == "candidate" else None
            ),
        },
        "stream_ids": sorted({event.stream_id for event in kernels}),
        "fallback_count": fallback_count,
        "unattributed_count": unattributed_count,
        "host_marker_residual_ns": host_marker_residual_ns,
        "host_marker_threshold_ns": host_threshold_ns,
        "cuda_envelope_residual_ns": cuda_envelope_residual_ns,
        "cuda_envelope_threshold_ns": cuda_threshold_ns,
        "calibration_admitted": calibration_admitted,
        "families": families,
        "phases": phase_rows,
        "bytes_recovered": False,
        "bytes_reason": "torch-profiler-kernel-rows-have-no-stable-byte-contract",
        "performance_claimed": False,
    }
    ledger["ledger_hash"] = canonical_hash(ledger)
    return ledger


def derive_pair_route(
    native_ledger: Mapping[str, Any], candidate_ledger: Mapping[str, Any]
) -> dict[str, object]:
    """Apply the preregistered graph and closed-family Amdahl route equations."""

    if (
        native_ledger.get("schema_version") != LEDGER_SCHEMA
        or candidate_ledger.get("schema_version") != LEDGER_SCHEMA
        or native_ledger.get("mode") != "native"
        or candidate_ledger.get("mode") != "candidate"
        or native_ledger.get("calibration_admitted") is not True
        or candidate_ledger.get("calibration_admitted") is not True
    ):
        raise ValueError("R3-D0 route ledger boundary differs")
    native_ns = int(native_ledger["unprofiled_median_ns"])
    candidate_ns = int(candidate_ledger["unprofiled_median_ns"])
    target_ns = native_ns / TARGET_SPEEDUP
    required_saving_ns = candidate_ns - target_ns
    host_residual_ns = int(candidate_ledger["host_residual_ns"])
    graph_optimistic_ns = candidate_ns - host_residual_ns
    graph_physical = graph_optimistic_ns <= target_ns
    rows = candidate_ledger.get("families")
    if not isinstance(rows, list):
        raise TypeError("R3-D0 family ledger differs")
    family_routes: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("R3-D0 family row differs")
        share = float(row["kernel_sum_ns"]) / candidate_ns
        denominator = target_ns / candidate_ns - (1.0 - share)
        required = None if denominator <= 0.0 else share / denominator
        family_routes.append(
            {
                "phase": str(row["phase"]),
                "family": str(row["family"]),
                "wrapper_share": share,
                "denominator": denominator,
                "required_speedup": required,
                "within_10x": required is not None
                and required <= MAX_REQUIRED_REGION_SPEEDUP,
                "semantic_closure_admitted": False,
            }
        )
    compiled = candidate_ledger.get("compiled_region")
    if not isinstance(compiled, Mapping):
        raise TypeError("R3-D0 compiled region differs")
    compiled_share = float(compiled["kernel_sum_ns"]) / candidate_ns
    compiled_denominator = target_ns / candidate_ns - (1.0 - compiled_share)
    compiled_required = (
        None if compiled_denominator <= 0.0 else compiled_share / compiled_denominator
    )
    compiled_semantic = (
        compiled.get("persistent_dense_a") is False
        and int(compiled.get("scratch_buffer_count", -1)) <= 2
        and compiled.get("semantic_closure_source")
        == "R3-2A-validated-full-region-10/9"
    )
    compiled_route_open = (
        compiled_semantic
        and compiled_required is not None
        and compiled_required <= MAX_REQUIRED_REGION_SPEEDUP
    )
    route = (
        "compiled-region-schedule"
        if compiled_route_open
        else "graph-opportunity" if graph_physical else "diagnostic-no-route"
    )
    result: dict[str, object] = {
        "schema_version": ROUTE_SCHEMA,
        "native_ns": native_ns,
        "candidate_ns": candidate_ns,
        "target_speedup": TARGET_SPEEDUP,
        "target_candidate_ns": target_ns,
        "required_saving_ns": required_saving_ns,
        "candidate_host_residual_ns": host_residual_ns,
        "graph_optimistic_ns": graph_optimistic_ns,
        "graph_physical": graph_physical,
        "graph_capture_admitted": False,
        "family_routes": family_routes,
        "compiled_region_route": {
            "wrapper_share": compiled_share,
            "denominator": compiled_denominator,
            "required_speedup": compiled_required,
            "semantic_closure_admitted": compiled_semantic,
            "within_10x": compiled_route_open,
        },
        "route": route,
        "performance_claimed": False,
    }
    result["route_hash"] = canonical_hash(result)
    return result


__all__ = [
    "R3D0ProfilerEventV1",
    "canonical_hash",
    "derive_pair_route",
    "derive_worker_ledger",
    "event_from_dict",
    "extract_torch_profiler_events",
]
