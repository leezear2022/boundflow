"""Typed raw-event and Amdahl contracts for FSG4/B4-0 attribution."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=too-many-locals,too-many-arguments,missing-function-docstring
# pylint: disable=too-many-branches,too-many-statements

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

B4_ATTRIBUTION_SCHEMA = "boundflow.fsg4-b4-kernel-attribution/v1"
B4_MARKER_PREFIX = "boundflow::b4::"
B3_QUERY_RATIO_TO_B0 = 0.9100012637918488
B3_CORE_QUERY_SHARE = 0.17735758999613638
B3_OPTIMIZER_QUERY_SHARE = 0.07933101562082898
B3_CROWN14_QUERY_SHARE = 0.12010163988903595
MATERIALIZATION_OPS = frozenset(
    {
        "aten::_to_copy",
        "aten::cat",
        "aten::clone",
        "aten::contiguous",
        "aten::copy_",
        "aten::empty",
        "aten::empty_like",
        "aten::empty_strided",
        "aten::new_empty",
        "aten::resize_",
        "aten::stack",
        "aten::to",
        "aten::zeros",
        "aten::zeros_like",
    }
)


def canonical_hash(value: object) -> str:
    """Return the stable JSON SHA256 used by the B4-0 artifact."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def query_speedup(*, share: float, region_speedup: float) -> float:
    """Apply Amdahl's law to one baseline-side query share."""

    if (
        not math.isfinite(share)
        or not math.isfinite(region_speedup)
        or not 0.0 <= share < 1.0
        or region_speedup <= 0.0
    ):
        raise ValueError("FSG4/B4 Amdahl input differs")
    return 1.0 / ((1.0 - share) + share / region_speedup)


def infinite_query_speedup(*, share: float) -> float:
    """Return the deletion-only upper bound for one baseline-side share."""

    if not math.isfinite(share) or not 0.0 <= share < 1.0:
        raise ValueError("FSG4/B4 Amdahl share differs")
    return 1.0 / (1.0 - share)


def required_region_speedup(*, share: float, target: float) -> float | None:
    """Return the finite region speedup needed for a target, or ``None``."""

    if (
        not math.isfinite(share)
        or not math.isfinite(target)
        or not 0.0 <= share < 1.0
        or target <= 0.0
    ):
        raise ValueError("FSG4/B4 required-speedup input differs")
    denominator = 1.0 / target - (1.0 - share)
    if denominator <= 0.0:
        return None
    result = share / denominator
    return result if math.isfinite(result) and result > 0.0 else None


def _device_type_name(value: object) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.lower()
    text = str(value).rsplit(".", maxsplit=1)[-1].lower()
    if text not in {"cpu", "cuda"}:
        raise ValueError(f"FSG4/B4 profiler device type differs: {value}")
    return text


def _phase_from_name(name: str) -> str | None:
    if not name.startswith(B4_MARKER_PREFIX):
        return None
    phase = name[len(B4_MARKER_PREFIX) :]
    if not phase or any(character.isspace() for character in phase):
        raise ValueError("FSG4/B4 phase marker differs")
    return phase


def _parent_chain(event: Any) -> Iterable[Any]:
    current = event
    seen: set[int] = set()
    while current is not None:
        identity = id(current)
        if identity in seen:
            raise ValueError("FSG4/B4 profiler parent cycle")
        seen.add(identity)
        yield current
        current = getattr(current, "cpu_parent", None)


def _phase_from_cpu_event(event: Any) -> str | None:
    chain = tuple(_parent_chain(event))
    names = tuple(str(getattr(current, "name", "")) for current in chain)
    for name in names:
        phase = _phase_from_name(name)
        if phase is not None:
            if phase == "optimizer":
                if any("autograd::engine::evaluate_function" in item for item in names):
                    return "optimizer.autograd_backward"
                if any("Optimizer.step#Adam.step" in item for item in names):
                    return "optimizer.adam"
                if any(item.startswith("aten::clamp") for item in names):
                    return "optimizer.clamp"
                return "optimizer.overhead"
            return phase
    return None


def _shape_payload(value: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    shapes: list[tuple[int, ...]] = []
    for raw in value:
        if not isinstance(raw, (list, tuple)) or any(
            not isinstance(item, int) for item in raw
        ):
            continue
        shapes.append(tuple(int(item) for item in raw))
    return tuple(shapes)


def _time_range_ns(event: Any) -> tuple[int, int]:
    time_range = getattr(event, "time_range", None)
    start_us = float(getattr(time_range, "start", 0.0))
    end_us = float(getattr(time_range, "end", start_us))
    return (
        max(int(round(start_us * 1000.0)), 0),
        max(int(round(end_us * 1000.0)), 0),
    )


def _temporal_marker(event: Any, markers: Sequence[Any]) -> Any | None:
    start_ns, end_ns = _time_range_ns(event)
    candidates = []
    for marker in markers:
        marker_start_ns, marker_end_ns = _time_range_ns(marker)
        if marker_start_ns <= start_ns and end_ns <= marker_end_ns:
            candidates.append((marker_end_ns - marker_start_ns, marker))
    return None if not candidates else min(candidates, key=lambda item: item[0])[1]


@dataclass(frozen=True)
class B4ProfilerEvent:
    """One path-sanitized raw profiler event assigned to a B4 phase."""

    event_ordinal: int
    correlation_id: int
    event_kind: str
    phase: str
    name: str
    parent_name: str | None
    duration_ns: int
    start_ns: int
    end_ns: int
    device_index: int
    stream_id: int
    thread_id: int
    input_shapes: tuple[tuple[int, ...], ...]
    cpu_memory_delta_bytes: int
    device_memory_delta_bytes: int
    attribution_method: str = "cpu_parent"
    schema_version: str = B4_ATTRIBUTION_SCHEMA

    def validate(self) -> None:
        if (
            self.schema_version != B4_ATTRIBUTION_SCHEMA
            or self.event_ordinal < 0
            or self.correlation_id < 0
            or self.event_kind not in {"cpu_op", "cuda_kernel", "phase_device_total"}
            or self.attribution_method
            not in {"cpu_parent", "device_marker", "temporal_marker", "unattributed"}
            or not self.phase
            or not self.name
            or self.duration_ns < 0
            or self.start_ns < 0
            or self.end_ns < self.start_ns
            or self.stream_id < -1
            or self.thread_id < 0
            or any(
                any(dimension < 0 for dimension in shape) for shape in self.input_shapes
            )
        ):
            raise ValueError("FSG4/B4 profiler event differs")
        local_tokens = ("/home/", "\\Users\\", "file://")
        if any(token in self.name for token in local_tokens) or (
            self.parent_name is not None
            and any(token in self.parent_name for token in local_tokens)
        ):
            raise ValueError("FSG4/B4 profiler event leaks a local path")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "event_ordinal": self.event_ordinal,
            "correlation_id": self.correlation_id,
            "event_kind": self.event_kind,
            "phase": self.phase,
            "name": self.name,
            "parent_name": self.parent_name,
            "duration_ns": self.duration_ns,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "device_index": self.device_index,
            "stream_id": self.stream_id,
            "thread_id": self.thread_id,
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "cpu_memory_delta_bytes": self.cpu_memory_delta_bytes,
            "device_memory_delta_bytes": self.device_memory_delta_bytes,
            "attribution_method": self.attribution_method,
        }


def b4_profiler_event_from_dict(value: Mapping[str, Any]) -> B4ProfilerEvent:
    """Parse one canonical raw profiler row and reject derived-field drift."""

    expected = {
        "schema_version",
        "event_ordinal",
        "correlation_id",
        "event_kind",
        "phase",
        "name",
        "parent_name",
        "duration_ns",
        "start_ns",
        "end_ns",
        "device_index",
        "stream_id",
        "thread_id",
        "input_shapes",
        "cpu_memory_delta_bytes",
        "device_memory_delta_bytes",
        "attribution_method",
    }
    if set(value) != expected or not isinstance(value.get("input_shapes"), list):
        raise ValueError("FSG4/B4 profiler event payload differs")
    event = B4ProfilerEvent(
        event_ordinal=int(value["event_ordinal"]),
        correlation_id=int(value["correlation_id"]),
        event_kind=str(value["event_kind"]),
        phase=str(value["phase"]),
        name=str(value["name"]),
        parent_name=(
            None if value["parent_name"] is None else str(value["parent_name"])
        ),
        duration_ns=int(value["duration_ns"]),
        start_ns=int(value["start_ns"]),
        end_ns=int(value["end_ns"]),
        device_index=int(value["device_index"]),
        stream_id=int(value["stream_id"]),
        thread_id=int(value["thread_id"]),
        input_shapes=tuple(
            tuple(int(dimension) for dimension in shape)
            for shape in value["input_shapes"]
            if isinstance(shape, list)
        ),
        cpu_memory_delta_bytes=int(value["cpu_memory_delta_bytes"]),
        device_memory_delta_bytes=int(value["device_memory_delta_bytes"]),
        attribution_method=str(value["attribution_method"]),
        schema_version=str(value["schema_version"]),
    )
    event.validate()
    if event.to_dict() != dict(value):
        raise ValueError("FSG4/B4 profiler event is not canonical")
    return event


def extract_profiler_events(events: Sequence[Any]) -> tuple[B4ProfilerEvent, ...]:
    """Assign CPU ops and CUDA kernels to explicit B4 record-function markers."""

    cpu_by_correlation: dict[int, list[Any]] = {}
    markers: list[Any] = []
    for event in events:
        if _device_type_name(getattr(event, "device_type", "")) == "cpu":
            cpu_by_correlation.setdefault(int(getattr(event, "id", 0)), []).append(
                event
            )
            if _phase_from_name(str(getattr(event, "name", ""))) is not None:
                markers.append(event)

    rows: list[B4ProfilerEvent] = []
    for event in events:
        device_type = _device_type_name(getattr(event, "device_type", ""))
        name = str(getattr(event, "name", ""))
        phase: str | None
        event_kind: str
        attribution_method: str
        parent: Any = None
        if device_type == "cpu":
            phase = _phase_from_cpu_event(event)
            event_kind = "cpu_op"
            attribution_method = "cpu_parent"
            parent = getattr(event, "cpu_parent", None)
        else:
            marker_phase = _phase_from_name(name)
            candidates = cpu_by_correlation.get(int(getattr(event, "id", 0)), [])
            parent = next(
                (
                    candidate
                    for candidate in candidates
                    if _phase_from_cpu_event(candidate)
                ),
                None,
            )
            if parent is None and candidates:
                parent = candidates[0]
            if marker_phase is not None or bool(
                getattr(event, "is_user_annotation", False)
            ):
                phase = (
                    marker_phase
                    if marker_phase is not None
                    else None if parent is None else _phase_from_cpu_event(parent)
                )
                if phase is None:
                    temporal_parent = _temporal_marker(event, markers)
                    if temporal_parent is not None:
                        parent = temporal_parent
                        phase = _phase_from_name(str(getattr(parent, "name", "")))
                        attribution_method = "temporal_marker"
                    else:
                        phase = "unattributed"
                        attribution_method = "unattributed"
                else:
                    attribution_method = (
                        "device_marker" if marker_phase is not None else "cpu_parent"
                    )
                event_kind = "phase_device_total"
            else:
                event_kind = "cuda_kernel"
                phase = None if parent is None else _phase_from_cpu_event(parent)
                if phase is None:
                    temporal_parent = _temporal_marker(event, markers)
                    if temporal_parent is not None:
                        parent = temporal_parent
                        phase = _phase_from_name(str(getattr(parent, "name", "")))
                        attribution_method = "temporal_marker"
                    else:
                        phase = "unattributed"
                        attribution_method = "unattributed"
                else:
                    attribution_method = "cpu_parent"
        if phase is None:
            continue
        start_ns, end_ns = _time_range_ns(event)
        duration_us = (
            float(getattr(event, "device_time_total", 0.0))
            if device_type == "cuda"
            else float(getattr(event, "cpu_time_total", 0.0))
        )
        row = B4ProfilerEvent(
            event_ordinal=len(rows),
            correlation_id=max(int(getattr(event, "id", 0)), 0),
            event_kind=event_kind,
            phase=phase,
            name=name,
            parent_name=(None if parent is None else str(getattr(parent, "name", ""))),
            duration_ns=max(int(round(duration_us * 1000.0)), 0),
            start_ns=start_ns,
            end_ns=end_ns,
            device_index=int(getattr(event, "device_index", -1)),
            stream_id=(
                int(getattr(event, "device_resource_id", -1))
                if device_type == "cuda"
                else -1
            ),
            thread_id=max(int(getattr(event, "thread", 0)), 0),
            input_shapes=_shape_payload(getattr(event, "input_shapes", ())),
            cpu_memory_delta_bytes=int(getattr(event, "cpu_memory_usage", 0)),
            device_memory_delta_bytes=int(getattr(event, "device_memory_usage", 0)),
            attribution_method=attribution_method,
        )
        row.validate()
        rows.append(row)
    if not rows or not any(row.event_kind == "cuda_kernel" for row in rows):
        raise ValueError("FSG4/B4 profiler captured no attributed CUDA kernel")
    return tuple(rows)


def _root_phase(phase: str) -> str:
    return phase.split(".", maxsplit=1)[0]


def derive_b4_attribution(
    events: Sequence[B4ProfilerEvent],
    *,
    run_id: str,
    source_identity: str,
    protocol_identity: str,
    query_wall_ns: int,
    core_wall_ns: int,
) -> dict[str, object]:
    """Aggregate only raw attributed kernels and freeze opportunity equations."""

    if (
        not run_id
        or len(source_identity) != 64
        or len(protocol_identity) != 64
        or query_wall_ns <= 0
        or core_wall_ns <= 0
        or core_wall_ns > query_wall_ns
    ):
        raise ValueError("FSG4/B4 attribution run identity differs")
    payloads = [event.to_dict() for event in events]
    if len({event.event_ordinal for event in events}) != len(events):
        raise ValueError("FSG4/B4 profiler event ordinal repeats")
    phase_rows: dict[str, dict[str, int]] = {}
    root_phase_rows: dict[str, dict[str, int]] = {}
    kernel_rows: dict[tuple[str, str], dict[str, int]] = {}
    operator_rows: dict[tuple[str, str], dict[str, int]] = {}
    materialization_rows: dict[tuple[str, str], dict[str, int]] = {}
    for event in events:
        phase = event.phase
        if event.event_kind == "cpu_op":
            operator_row = operator_rows.setdefault(
                (phase, event.name),
                {
                    "operator_count": 0,
                    "cpu_operator_sum_ns": 0,
                    "cpu_memory_delta_bytes": 0,
                    "device_memory_delta_bytes": 0,
                },
            )
            operator_row["operator_count"] += 1
            operator_row["cpu_operator_sum_ns"] += event.duration_ns
            operator_row["cpu_memory_delta_bytes"] += event.cpu_memory_delta_bytes
            operator_row["device_memory_delta_bytes"] += event.device_memory_delta_bytes
            if event.name in MATERIALIZATION_OPS:
                materialization_rows[(phase, event.name)] = dict(operator_row)
            continue
        if event.event_kind != "cuda_kernel":
            continue
        phase_row = phase_rows.setdefault(
            phase, {"kernel_count": 0, "cuda_kernel_sum_ns": 0}
        )
        phase_row["kernel_count"] = int(phase_row["kernel_count"]) + 1
        phase_row["cuda_kernel_sum_ns"] = (
            int(phase_row["cuda_kernel_sum_ns"]) + event.duration_ns
        )
        root_phase = _root_phase(phase)
        root_phase_row = root_phase_rows.setdefault(
            root_phase, {"kernel_count": 0, "cuda_kernel_sum_ns": 0}
        )
        root_phase_row["kernel_count"] += 1
        root_phase_row["cuda_kernel_sum_ns"] += event.duration_ns
        kernel_row = kernel_rows.setdefault(
            (phase, event.name), {"kernel_count": 0, "cuda_kernel_sum_ns": 0}
        )
        kernel_row["kernel_count"] = int(kernel_row["kernel_count"]) + 1
        kernel_row["cuda_kernel_sum_ns"] = (
            int(kernel_row["cuda_kernel_sum_ns"]) + event.duration_ns
        )
    target = 1.0 / B3_QUERY_RATIO_TO_B0
    opportunity = {
        "b3_to_b0_parity_target": target,
        "optimizer_only": {
            "query_share": B3_OPTIMIZER_QUERY_SHARE,
            "infinite_query_speedup": infinite_query_speedup(
                share=B3_OPTIMIZER_QUERY_SHARE
            ),
            "required_region_speedup": required_region_speedup(
                share=B3_OPTIMIZER_QUERY_SHARE, target=target
            ),
        },
        "crown14": {
            "query_share": B3_CROWN14_QUERY_SHARE,
            "infinite_query_speedup": infinite_query_speedup(
                share=B3_CROWN14_QUERY_SHARE
            ),
            "required_region_speedup": required_region_speedup(
                share=B3_CROWN14_QUERY_SHARE, target=target
            ),
        },
        "whole_core": {
            "query_share": B3_CORE_QUERY_SHARE,
            "infinite_query_speedup": infinite_query_speedup(share=B3_CORE_QUERY_SHARE),
            "required_region_speedup": required_region_speedup(
                share=B3_CORE_QUERY_SHARE, target=target
            ),
        },
    }
    summary: dict[str, object] = {
        "schema_version": B4_ATTRIBUTION_SCHEMA,
        "run_id": run_id,
        "source_identity": source_identity,
        "protocol_identity": protocol_identity,
        "query_wall_ns": query_wall_ns,
        "core_wall_ns": core_wall_ns,
        "event_count": len(events),
        "cuda_kernel_count": sum(event.event_kind == "cuda_kernel" for event in events),
        "phase_closure": {
            "accounted_cuda_kernel_count": sum(
                event.event_kind == "cuda_kernel" for event in events
            ),
            "attributed_cuda_kernel_count": sum(
                event.event_kind == "cuda_kernel" and event.phase != "unattributed"
                for event in events
            ),
            "unattributed_cuda_kernel_count": sum(
                event.event_kind == "cuda_kernel" and event.phase == "unattributed"
                for event in events
            ),
            "attribution_method_counts": {
                method: sum(
                    event.event_kind == "cuda_kernel"
                    and event.attribution_method == method
                    for event in events
                )
                for method in (
                    "cpu_parent",
                    "device_marker",
                    "temporal_marker",
                    "unattributed",
                )
            },
        },
        "raw_event_hash": canonical_hash(payloads),
        "phase_attribution": dict(sorted(phase_rows.items())),
        "root_phase_attribution": dict(sorted(root_phase_rows.items())),
        "kernel_attribution": [
            {"phase": phase, "kernel_name": name, **values}
            for (phase, name), values in sorted(
                kernel_rows.items(),
                key=lambda item: (-int(item[1]["cuda_kernel_sum_ns"]), item[0]),
            )
        ],
        "operator_attribution": [
            {"phase": phase, "operator_name": name, **values}
            for (phase, name), values in sorted(
                operator_rows.items(),
                key=lambda item: (-int(item[1]["cpu_operator_sum_ns"]), item[0]),
            )
        ],
        "materialization_attribution": [
            {"phase": phase, "operator_name": name, **values}
            for (phase, name), values in sorted(
                materialization_rows.items(),
                key=lambda item: (
                    -abs(int(item[1]["device_memory_delta_bytes"])),
                    -int(item[1]["cpu_operator_sum_ns"]),
                    item[0],
                ),
            )
        ],
        "opportunity": opportunity,
        "b4_0_attribution_only": True,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


__all__ = [
    "B3_CORE_QUERY_SHARE",
    "B3_CROWN14_QUERY_SHARE",
    "B3_OPTIMIZER_QUERY_SHARE",
    "B3_QUERY_RATIO_TO_B0",
    "B4_ATTRIBUTION_SCHEMA",
    "B4_MARKER_PREFIX",
    "B4ProfilerEvent",
    "MATERIALIZATION_OPS",
    "b4_profiler_event_from_dict",
    "canonical_hash",
    "derive_b4_attribution",
    "extract_profiler_events",
    "infinite_query_speedup",
    "query_speedup",
    "required_region_speedup",
]
