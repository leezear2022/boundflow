"""Typed host/device ledgers and routing gates for MR7 attribution."""

# pylint: disable=too-many-locals,too-many-branches,too-many-statements
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=too-many-instance-attributes,protected-access

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import statistics
import time
from typing import Any, Iterator, Mapping, Sequence, cast

from boundflow.runtime.mr3_provider_hook_feasibility import canonical_hash
from boundflow.runtime import mr3_production_bridge_timing as timing_math
from boundflow.runtime.mr5_multi_conv_timing import validate_worker as validate_base
from boundflow.runtime.mr6_guard_attribution import validate_guard_receipt

WORKER_SCHEMA = "boundflow.mr7-launch-materialization-worker/v1"
RAW_SCHEMA = "boundflow.mr7-launch-materialization-formal/v1"
MARKER_PREFIX = "boundflow::mr7::"
HOST_CATEGORIES = (
    "admission_handoff",
    "layout_materialization",
    "ffi_dlpack_stream",
    "post_output_guard",
    "optimizer_and_residual",
)
DEVICE_CATEGORIES = ("forward_device_kernel", "backward_device_kernel")
SITE_ORDER = ("C2", "C1", "C0")
PARITY_TARGET = 1.107412
RESEARCH_TARGET = 1.273523


class MR7HostLedger:
    """Accumulate mutually exclusive nested CPU spans on the worker thread."""

    def __init__(self) -> None:
        self._stack: list[list[object]] = []
        self.category_ns = {category: 0 for category in HOST_CATEGORIES}
        self.category_calls = {category: 0 for category in HOST_CATEGORIES}

    @contextmanager
    def span(self, category: str) -> Iterator[None]:
        if category not in self.category_ns:
            raise ValueError(f"MR7 host category differs: {category}")
        frame: list[object] = [category, time.perf_counter_ns(), 0]
        self._stack.append(frame)
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            if not self._stack or self._stack.pop() is not frame:
                raise RuntimeError("MR7 host span stack differs")
            elapsed = end - cast(int, frame[1])
            child = cast(int, frame[2])
            exclusive = elapsed - child
            if exclusive < 0:
                raise RuntimeError("MR7 host exclusive duration differs")
            self.category_ns[category] += exclusive
            self.category_calls[category] += 1
            if self._stack:
                self._stack[-1][2] = cast(int, self._stack[-1][2]) + elapsed

    def receipt(self, *, outer_ns: int) -> dict[str, object]:
        if self._stack or outer_ns <= 0:
            raise ValueError("MR7 host ledger lifecycle differs")
        known = sum(self.category_ns.values())
        if known > outer_ns:
            raise ValueError("MR7 host categories exceed outer")
        values = dict(self.category_ns)
        values["optimizer_and_residual"] += outer_ns - known
        closure = sum(values.values())
        receipt: dict[str, object] = {
            "outer_host_ns": outer_ns,
            "category_ns": values,
            "category_calls": dict(self.category_calls),
            "closure_error_ns": abs(outer_ns - closure),
            "closure_error_ratio": abs(outer_ns - closure) / outer_ns,
            "performance_claimed": False,
        }
        receipt["receipt_hash"] = canonical_hash(receipt)
        return receipt


@dataclass(frozen=True)
class MR7DeviceEvent:
    """One CUDA kernel attributed through an explicit MR7 launch marker."""

    direction: str
    site: str
    ordinal: int
    kernel_name: str
    duration_ns: int
    correlation_id: int
    stream_id: int
    attribution_method: str

    def validate(self) -> None:
        if (
            self.direction not in {"forward", "backward"}
            or self.site not in SITE_ORDER
            or self.ordinal < 0
            or not self.kernel_name
            or self.duration_ns <= 0
            or self.correlation_id < 0
            or self.stream_id < -1
            or self.attribution_method != "cpu_parent"
            or any(
                token in self.kernel_name
                for token in ("/home/", "\\Users\\", "file://")
            )
        ):
            raise ValueError("MR7 device event differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "direction": self.direction,
            "site": self.site,
            "ordinal": self.ordinal,
            "kernel_name": self.kernel_name,
            "duration_ns": self.duration_ns,
            "correlation_id": self.correlation_id,
            "stream_id": self.stream_id,
            "attribution_method": self.attribution_method,
        }


def device_event_from_dict(value: Mapping[str, Any]) -> MR7DeviceEvent:
    expected = {
        "direction",
        "site",
        "ordinal",
        "kernel_name",
        "duration_ns",
        "correlation_id",
        "stream_id",
        "attribution_method",
    }
    if set(value) != expected:
        raise ValueError("MR7 device event payload differs")
    event = MR7DeviceEvent(
        direction=str(value["direction"]),
        site=str(value["site"]),
        ordinal=int(value["ordinal"]),
        kernel_name=str(value["kernel_name"]),
        duration_ns=int(value["duration_ns"]),
        correlation_id=int(value["correlation_id"]),
        stream_id=int(value["stream_id"]),
        attribution_method=str(value["attribution_method"]),
    )
    event.validate()
    if event.to_dict() != dict(value):
        raise ValueError("MR7 device event is not canonical")
    return event


def _device_name(value: object) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.lower()
    return str(value).rsplit(".", maxsplit=1)[-1].lower()


def _marker_from_parent(
    event: Any, *, include_self: bool = False
) -> tuple[str, str, int] | None:
    current = event if include_self else getattr(event, "cpu_parent", None)
    seen: set[int] = set()
    while current is not None:
        identity = id(current)
        if identity in seen:
            raise ValueError("MR7 profiler parent cycle")
        seen.add(identity)
        name = str(getattr(current, "name", ""))
        if name.startswith(MARKER_PREFIX):
            parts = name[len(MARKER_PREFIX) :].split(".")
            if len(parts) == 3 and parts[0] in {"forward", "backward"}:
                return parts[0], parts[1], int(parts[2])
        current = getattr(current, "cpu_parent", None)
    return None


def extract_device_events(events: Sequence[Any]) -> tuple[MR7DeviceEvent, ...]:
    cpu_by_correlation: dict[int, list[Any]] = {}
    for event in events:
        if _device_name(getattr(event, "device_type", "")) == "cpu":
            cpu_by_correlation.setdefault(int(getattr(event, "id", 0)), []).append(
                event
            )
    rows: list[MR7DeviceEvent] = []
    for event in events:
        if _device_name(getattr(event, "device_type", "")) != "cuda":
            continue
        if bool(getattr(event, "is_user_annotation", False)):
            continue
        marker = _marker_from_parent(event)
        if marker is None:
            marker = next(
                (
                    candidate_marker
                    for candidate in cpu_by_correlation.get(
                        int(getattr(event, "id", 0)), []
                    )
                    if (
                        candidate_marker := _marker_from_parent(
                            candidate, include_self=True
                        )
                    )
                    is not None
                ),
                None,
            )
        if marker is None:
            continue
        direction, site, ordinal = marker
        duration_ns = max(
            int(round(float(getattr(event, "device_time_total", 0.0)) * 1000.0)),
            0,
        )
        if duration_ns == 0:
            continue
        row = MR7DeviceEvent(
            direction=direction,
            site=site,
            ordinal=ordinal,
            kernel_name=str(getattr(event, "name", "")),
            duration_ns=duration_ns,
            correlation_id=max(int(getattr(event, "id", 0)), 0),
            stream_id=int(getattr(event, "device_resource_id", -1)),
            attribution_method="cpu_parent",
        )
        row.validate()
        rows.append(row)
    if not rows:
        raise ValueError("MR7 profiler captured no device event")
    return tuple(rows)


def extract_device_marker_totals(events: Sequence[Any]) -> dict[str, int]:
    """Extract CUPTI device totals emitted for explicit record-function markers."""

    totals: dict[str, int] = {}
    for event in events:
        if _device_name(getattr(event, "device_type", "")) != "cuda" or not bool(
            getattr(event, "is_user_annotation", False)
        ):
            continue
        name = str(getattr(event, "name", ""))
        if not name.startswith(MARKER_PREFIX):
            continue
        phase = name[len(MARKER_PREFIX) :]
        parts = phase.split(".")
        if len(parts) != 3 or parts[0] not in {"forward", "backward"}:
            continue
        totals[phase] = totals.get(phase, 0) + max(
            int(round(float(getattr(event, "device_time_total", 0.0)) * 1000.0)),
            0,
        )
    if not totals:
        raise ValueError("MR7 profiler captured no device marker total")
    return dict(sorted(totals.items()))


def validate_host_receipt(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("MR7 host receipt absent")
    unsigned = dict(value)
    receipt_hash = unsigned.pop("receipt_hash", None)
    categories = value.get("category_ns")
    calls = value.get("category_calls")
    outer = value.get("outer_host_ns")
    if (
        receipt_hash != canonical_hash(unsigned)
        or not isinstance(categories, Mapping)
        or not isinstance(calls, Mapping)
        or set(categories) != set(HOST_CATEGORIES)
        or set(calls) != set(HOST_CATEGORIES)
        or not isinstance(outer, int)
        or outer <= 0
        or any(not isinstance(item, int) or item < 0 for item in categories.values())
        or sum(cast(Mapping[str, int], categories).values()) != outer
        or value.get("closure_error_ns") != 0
        or value.get("closure_error_ratio") != 0.0
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("MR7 host receipt differs")
    return cast(Mapping[str, Any], value)


def required_region_speedup(*, share: float, target: float) -> float | None:
    if not 0.0 <= share < 1.0 or target <= 0.0:
        raise ValueError("MR7 Amdahl input differs")
    denominator = (1.0 / target) - (1.0 - share)
    if denominator <= 0.0:
        return None
    result = share / denominator
    return result if math.isfinite(result) and result > 0.0 else None


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("MR7 median input absent")
    return float(statistics.median(values))


def derive_summary(raw: Mapping[str, Any]) -> dict[str, object]:
    unsigned = dict(raw)
    raw_hash = unsigned.pop("raw_hash", None)
    runs_value = raw.get("runs")
    source_commit = raw.get("source_commit")
    expected_order = [
        [0, 0, "control"],
        [0, 1, "profile"],
        [1, 0, "profile"],
        [1, 1, "control"],
        [2, 0, "control"],
        [2, 1, "profile"],
    ]
    if (
        raw.get("schema_version") != RAW_SCHEMA
        or not isinstance(source_commit, str)
        or len(source_commit) != 40
        or raw.get("run_order") != expected_order
        or raw_hash != canonical_hash(unsigned)
        or not isinstance(runs_value, list)
        or len(runs_value) != 6
    ):
        raise ValueError("MR7 raw provenance differs")

    controls: dict[int, Mapping[str, Any]] = {}
    profiles: dict[int, Mapping[str, Any]] = {}
    module_hashes: set[str] = set()
    for expected, wrapper in zip(expected_order, runs_value):
        if (
            not isinstance(wrapper, Mapping)
            or [
                wrapper.get("pair_index"),
                wrapper.get("position"),
                wrapper.get("kind"),
            ]
            != expected
        ):
            raise ValueError("MR7 run order differs")
        worker = wrapper.get("worker")
        if not isinstance(worker, Mapping):
            raise ValueError("MR7 worker absent")
        worker_unsigned = dict(worker)
        worker_hash = worker_unsigned.pop("worker_hash", None)
        host = validate_host_receipt(worker.get("host_receipt"))
        base = worker.get("base_worker")
        if (
            worker.get("schema_version") != WORKER_SCHEMA
            or worker.get("kind") != wrapper.get("kind")
            or worker_hash != canonical_hash(worker_unsigned)
            or not isinstance(base, Mapping)
            or base.get("mode") != "bridge"
            or worker.get("performance_claimed") is not False
            or host["outer_host_ns"] != base.get("measurement", {}).get("host_ns")
        ):
            raise ValueError("MR7 worker envelope differs")
        validate_base(base, mode="bridge")
        validate_guard_receipt(worker.get("guard_receipt"), mode="diagnostic")
        bridge = base.get("bridge_receipt")
        module = base.get("candidate_module_receipt")
        if not isinstance(bridge, Mapping) or not isinstance(module, Mapping):
            raise ValueError("MR7 inherited receipt absent")
        if (
            bridge.get("forward_launches") != {"C0": 10, "C1": 10, "C2": 10}
            or bridge.get("backward_launches") != {"C0": 9, "C1": 9, "C2": 9}
            or bridge.get("fallback_count") != 0
            or bridge.get("eager_count") != 0
        ):
            raise ValueError("MR7 launch lifecycle differs")
        module_hashes.add(canonical_hash(module))
        target = controls if wrapper["kind"] == "control" else profiles
        target[cast(int, wrapper["pair_index"])] = worker
    if len(controls) != 3 or len(profiles) != 3 or len(module_hashes) != 1:
        raise ValueError("MR7 pair/module identity differs")

    pair_metrics: list[dict[str, object]] = []
    host_shares: dict[str, list[float]] = {name: [] for name in HOST_CATEGORIES}
    device_shares: dict[str, list[float]] = {name: [] for name in DEVICE_CATEGORIES}
    site_device_ns: dict[str, list[int]] = {site: [] for site in SITE_ORDER}
    all_profile_events: list[MR7DeviceEvent] = []
    for pair in range(3):
        control = controls[pair]
        profile = profiles[pair]
        control_host = validate_host_receipt(control["host_receipt"])
        profile_host = validate_host_receipt(profile["host_receipt"])
        control_outer = cast(int, control_host["outer_host_ns"])
        profile_outer = cast(int, profile_host["outer_host_ns"])
        semantic_metric = timing_math._pair_metric(
            pair,
            cast(Mapping[str, Any], control["base_worker"]),
            cast(Mapping[str, Any], profile["base_worker"]),
        )
        event_ratio = cast(
            float, profile["base_worker"]["measurement"]["cuda_event_ms"]
        ) / cast(float, control["base_worker"]["measurement"]["cuda_event_ms"])
        for name in HOST_CATEGORIES:
            host_shares[name].append(
                cast(int, control_host["category_ns"][name]) / control_outer
            )
        events = tuple(
            device_event_from_dict(cast(Mapping[str, Any], item))
            for item in cast(Sequence[object], profile.get("device_events", []))
        )
        all_profile_events.extend(events)
        marker_totals = profile.get("device_marker_totals")
        if not isinstance(marker_totals, Mapping):
            raise ValueError("MR7 device marker totals absent")
        direction_counts = {
            direction: len(
                {
                    (event.site, event.ordinal)
                    for event in events
                    if event.direction == direction
                }
            )
            for direction in ("forward", "backward")
        }
        if direction_counts != {"forward": 30, "backward": 27}:
            raise ValueError("MR7 attributed launch count differs")
        by_marker: dict[str, int] = {}
        for event in events:
            phase = f"{event.direction}.{event.site}.{event.ordinal:02d}"
            by_marker[phase] = by_marker.get(phase, 0) + event.duration_ns
        if set(by_marker) != set(marker_totals):
            raise ValueError("MR7 device envelope marker set differs")
        kernel_envelope_ns = sum(by_marker.values())
        marker_envelope_ns = sum(int(value) for value in marker_totals.values())
        envelope_error_ratio = abs(kernel_envelope_ns - marker_envelope_ns) / max(
            marker_envelope_ns, 1
        )
        outer_device_ns = int(
            round(
                cast(float, profile["base_worker"]["measurement"]["cuda_event_ms"])
                * 1_000_000.0
            )
        )
        direction_ns = {
            direction: sum(
                event.duration_ns for event in events if event.direction == direction
            )
            for direction in ("forward", "backward")
        }
        device_shares["forward_device_kernel"].append(
            direction_ns["forward"] / outer_device_ns
        )
        device_shares["backward_device_kernel"].append(
            direction_ns["backward"] / outer_device_ns
        )
        site_totals = {
            site: sum(event.duration_ns for event in events if event.site == site)
            for site in SITE_ORDER
        }
        for site, value in site_totals.items():
            site_device_ns[site].append(value)
        pair_metrics.append(
            {
                "pair_index": pair,
                "profile_control_cuda_event_ratio": event_ratio,
                "control_outer_host_ns": control_outer,
                "profile_outer_host_ns": profile_outer,
                "forward_device_ns": direction_ns["forward"],
                "backward_device_ns": direction_ns["backward"],
                "site_device_ns": site_totals,
                "device_envelope_error_ratio": envelope_error_ratio,
                "semantic_allclose": semantic_metric["allclose"],
                "semantic_sign_exact": semantic_metric["sign_exact"],
                "semantic_maximum_absolute_difference": semantic_metric[
                    "semantic_maximum_absolute_difference"
                ],
            }
        )

    median_host_share = {name: _median(values) for name, values in host_shares.items()}
    median_device_share = {
        name: _median(values) for name, values in device_shares.items()
    }
    boundary_share = sum(
        median_host_share[name]
        for name in ("ffi_dlpack_stream", "layout_materialization", "post_output_guard")
    )
    median_outer_ns = _median(
        [
            float(validate_host_receipt(worker["host_receipt"])["outer_host_ns"])
            for worker in controls.values()
        ]
    )
    boundary_ns = boundary_share * median_outer_ns
    kernel_share = sum(median_device_share.values())
    launch_share = median_host_share["ffi_dlpack_stream"]

    def slowest_site(run_index: int) -> str:
        return max(((site_device_ns[site][run_index], site) for site in SITE_ORDER))[1]

    slowest = [slowest_site(run) for run in range(3)]
    same_site_slowest = len(set(slowest)) == 1
    event_ratio_gate = all(
        cast(float, item["profile_control_cuda_event_ratio"]) <= 1.10
        for item in pair_metrics
    )
    route = "NO_GO_CURRENT_CONV_REPLACEMENT"
    selected_share = 0.0
    if boundary_share >= 0.15 and boundary_ns >= 15_000_000:
        route = "MR7_A_COMPILED_REGION_ARENA_FFI"
        selected_share = boundary_share
    elif kernel_share >= 0.50 and same_site_slowest:
        route = "MR7_B_PER_SITE_SCHEDULE"
        selected_share = kernel_share
    elif launch_share >= 0.15:
        route = "MR7_C_CROSS_SITE_STEP_EXECUTION_GRAPH"
        selected_share = launch_share
    required = (
        required_region_speedup(share=selected_share, target=PARITY_TARGET)
        if selected_share
        else None
    )
    if required is None or required > 10.0:
        route = "NO_GO_CURRENT_CONV_REPLACEMENT"
    gates = {
        "pair_count": len(pair_metrics) == 3,
        "semantic_exact": all(
            item["semantic_allclose"] and item["semantic_sign_exact"]
            for item in pair_metrics
        ),
        "module_stability": len(module_hashes) == 1,
        "host_closure": True,
        "device_envelope_closure": all(
            cast(float, item["device_envelope_error_ratio"]) <= 0.02
            for item in pair_metrics
        ),
        "launch_counts": True,
        "profile_control_event_ratio": event_ratio_gate,
        "amdahl_reachable": required is not None and required <= 10.0,
    }
    if not all(gates.values()):
        route = "INVALID_MR7_ATTRIBUTION"
    summary: dict[str, object] = {
        "schema_version": RAW_SCHEMA,
        "source_commit": source_commit,
        "status": route,
        "run_count": 6,
        "pair_count": 3,
        "pair_metrics": pair_metrics,
        "median_host_share": median_host_share,
        "median_device_share": median_device_share,
        "boundary_share": boundary_share,
        "boundary_median_ns": boundary_ns,
        "device_kernel_share": kernel_share,
        "launch_host_share": launch_share,
        "site_device_ns": site_device_ns,
        "slowest_site_by_run": slowest,
        "same_site_slowest": same_site_slowest,
        "selected_share": selected_share,
        "required_parity_region_speedup": required,
        "parity_target": PARITY_TARGET,
        "research_target": RESEARCH_TARGET,
        "gates": gates,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


__all__ = [
    "DEVICE_CATEGORIES",
    "HOST_CATEGORIES",
    "MARKER_PREFIX",
    "MR7DeviceEvent",
    "MR7HostLedger",
    "RAW_SCHEMA",
    "WORKER_SCHEMA",
    "derive_summary",
    "device_event_from_dict",
    "extract_device_events",
    "extract_device_marker_totals",
    "required_region_speedup",
    "validate_host_receipt",
]
