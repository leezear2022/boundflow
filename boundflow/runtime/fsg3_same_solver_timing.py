"""Typed raw contracts and replay aggregation for FSG3 B0/B1/B2 timing."""

# pylint: disable=too-many-branches,too-many-locals,too-many-lines,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import statistics
from typing import Any, cast, Mapping, Optional, Sequence, Tuple

FSG3_TIMING_SCHEMA_VERSION = "boundflow.fsg3-same-solver-timing/v4"
FSG3_REPEAT_COUNT = 6
FSG3_PROFILE_PERTURBATION_LIMIT = 1.05
FSG3_CLOSURE_ERROR_LIMIT = 0.01
FSG3_RESIDUAL_LIMIT = 0.03
FSG3_FLOAT_ATOL = 2.0e-4
FSG3_FLOAT_RTOL = 2.0e-4


class FSG3Configuration(str, Enum):
    """Frozen cumulative configurations in the FSG3 baseline."""

    B0 = "B0"
    B1 = "B1"
    B2 = "B2"


class FSG3Mode(str, Enum):
    """Unprofiled headline control or attribution-only profile."""

    CONTROL = "control"
    PROFILE = "profile"


FSG3_PROFILE_SPAN_LAYOUT: Mapping[FSG3Configuration, Tuple[Tuple[str, str], ...]] = {
    FSG3Configuration.B0: (
        ("core", "provider_core"),
        ("post", "official_post_queue"),
    ),
    FSG3Configuration.B1: (
        ("core", "typed_pre_state"),
        ("core", "provider_core"),
        ("post", "official_post_queue"),
    ),
    FSG3Configuration.B2: (
        ("compile", "compile"),
        ("core", "typed_pre_state"),
        ("core", "optimizer"),
        ("core", "backward"),
        ("core", "kfsb"),
        ("core", "atomic_commit"),
        ("post", "official_post_queue"),
    ),
}


FSG3_CONFIG_ORDERS: Tuple[Tuple[FSG3Configuration, ...], ...] = (
    (FSG3Configuration.B0, FSG3Configuration.B1, FSG3Configuration.B2),
    (FSG3Configuration.B0, FSG3Configuration.B2, FSG3Configuration.B1),
    (FSG3Configuration.B1, FSG3Configuration.B0, FSG3Configuration.B2),
    (FSG3Configuration.B1, FSG3Configuration.B2, FSG3Configuration.B0),
    (FSG3Configuration.B2, FSG3Configuration.B0, FSG3Configuration.B1),
    (FSG3Configuration.B2, FSG3Configuration.B1, FSG3Configuration.B0),
)


def expected_fsg3_sequence() -> (
    Tuple[Tuple[int, int, FSG3Configuration, FSG3Mode], ...]
):
    """Return the immutable 36-worker preregistered sequence."""

    rows: list[tuple[int, int, FSG3Configuration, FSG3Mode]] = []
    for block_index, configurations in enumerate(FSG3_CONFIG_ORDERS):
        modes = (
            (FSG3Mode.CONTROL, FSG3Mode.PROFILE)
            if block_index % 2 == 0
            else (FSG3Mode.PROFILE, FSG3Mode.CONTROL)
        )
        position = 0
        for configuration in configurations:
            for mode in modes:
                rows.append((block_index, position, configuration, mode))
                position += 1
    return tuple(rows)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: object) -> str:
    """Return a deterministic SHA256 over canonical JSON."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"FSG3 {label} fields differ")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"FSG3 {label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"FSG3 {label} must be a sequence")
    return value


@dataclass(frozen=True)
class FSG3TimingMetrics:  # pylint: disable=too-many-instance-attributes
    """Non-overloaded wall, CUDA-event, compile, validation, and peak metrics."""

    cold_total_ns: int
    boundflow_compile_ns: int
    query_wall_ns: int
    query_gpu_ns: int
    core_wall_ns: int
    core_gpu_ns: int
    post_validation_ns: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def validate(self, configuration: FSG3Configuration) -> None:
        """Reject negative, inverted, or scope-overlapping measurements."""

        for name in (
            "cold_total_ns",
            "boundflow_compile_ns",
            "query_wall_ns",
            "query_gpu_ns",
            "core_wall_ns",
            "core_gpu_ns",
            "post_validation_ns",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"FSG3 timing {name} must be non-negative")
        for name in ("cold_total_ns", "query_wall_ns", "query_gpu_ns", "core_wall_ns"):
            if getattr(self, name) == 0:
                raise ValueError(f"FSG3 timing {name} must be positive")
        if self.core_gpu_ns == 0:
            raise ValueError("FSG3 timing core_gpu_ns must be positive")
        if self.core_wall_ns > self.query_wall_ns:
            raise ValueError("FSG3 core wall escapes query scope")
        if self.core_gpu_ns > self.query_gpu_ns:
            raise ValueError("FSG3 core GPU time escapes query scope")
        if self.peak_allocated_bytes > self.peak_reserved_bytes:
            raise ValueError("FSG3 allocated memory exceeds reserved memory")
        if configuration in {FSG3Configuration.B0, FSG3Configuration.B1}:
            if self.boundflow_compile_ns != 0:
                raise ValueError("FSG3 B0/B1 cannot report BoundFlow compile")
            if self.cold_total_ns != self.query_wall_ns:
                raise ValueError("FSG3 B0/B1 cold and query scope must match")
        else:
            if self.boundflow_compile_ns == 0:
                raise ValueError("FSG3 B2 requires a measured cold compile")
            if self.cold_total_ns < self.boundflow_compile_ns + self.query_wall_ns:
                raise ValueError("FSG3 B2 cold scope omits compile or query")

    def to_dict(self) -> dict[str, int]:
        """Return the stable metric payload."""

        return {
            "cold_total_ns": self.cold_total_ns,
            "boundflow_compile_ns": self.boundflow_compile_ns,
            "query_wall_ns": self.query_wall_ns,
            "query_gpu_ns": self.query_gpu_ns,
            "core_wall_ns": self.core_wall_ns,
            "core_gpu_ns": self.core_gpu_ns,
            "post_validation_ns": self.post_validation_ns,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
        }


@dataclass(frozen=True)
class FSG3ProfileSpan:  # pylint: disable=too-many-instance-attributes
    """One non-overlapping host/CUDA attribution interval."""

    scope: str
    name: str
    stack_layer: str
    solver_phase: str
    resource: str
    cache_state: str
    start_offset_ns: int
    end_offset_ns: int
    wall_ns: int
    gpu_ns: int

    def validate(self) -> None:
        """Reject incomplete, inverted, or non-canonical spans."""

        if self.scope not in {"compile", "core", "post"}:
            raise ValueError("FSG3 profile span scope differs")
        if not all(
            (
                self.name,
                self.stack_layer,
                self.solver_phase,
                self.resource,
                self.cache_state,
            )
        ):
            raise ValueError("FSG3 profile span metadata is empty")
        if self.start_offset_ns < 0 or self.end_offset_ns <= self.start_offset_ns:
            raise ValueError("FSG3 profile span interval differs")
        if self.wall_ns != self.end_offset_ns - self.start_offset_ns:
            raise ValueError("FSG3 profile span wall projection differs")
        if self.gpu_ns < 0:
            raise ValueError("FSG3 profile span GPU time is negative")
        if self.resource == "host" and self.gpu_ns != 0:
            raise ValueError("FSG3 host-only span cannot report GPU time")
        if self.cache_state not in {"cold", "process-hit"}:
            raise ValueError("FSG3 profile span cache state differs")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical span payload."""

        self.validate()
        return {
            "scope": self.scope,
            "name": self.name,
            "stack_layer": self.stack_layer,
            "solver_phase": self.solver_phase,
            "resource": self.resource,
            "cache_state": self.cache_state,
            "start_offset_ns": self.start_offset_ns,
            "end_offset_ns": self.end_offset_ns,
            "wall_ns": self.wall_ns,
            "gpu_ns": self.gpu_ns,
        }


@dataclass(frozen=True)
class FSG3SemanticResult:  # pylint: disable=too-many-instance-attributes
    """Post-measurement semantic and queue projection for one worker."""

    status: str
    success: bool
    visited_domains: Tuple[int, ...]
    queue_before: int
    queue_input: int
    queue_accepted: int
    queue_pruned: int
    queue_after: int
    depths: Tuple[int, ...]
    history_count: int
    lower_shape: Tuple[int, ...]
    lower_values: Tuple[float, ...]
    upper_shape: Tuple[int, ...]
    upper_values: Tuple[float, ...]
    upper_positive_infinity_mask: Tuple[bool, ...]
    final_decision: Tuple[Tuple[int, int], ...]
    split_depth: int
    batch_size: int
    n_verified: int
    n_splits: int

    def validate(self) -> None:
        """Reject incomplete, non-finite, or internally inconsistent semantics."""

        if not self.status or not self.visited_domains:
            raise ValueError("FSG3 solver result identity is empty")
        counts = (
            self.queue_before,
            self.queue_input,
            self.queue_accepted,
            self.queue_pruned,
            self.queue_after,
            self.history_count,
            self.split_depth,
            self.batch_size,
            self.n_verified,
            self.n_splits,
        )
        if any(value < 0 for value in counts):
            raise ValueError("FSG3 semantic count is negative")
        if self.queue_input != self.queue_accepted + self.queue_pruned:
            raise ValueError("FSG3 queue input accounting differs")
        if self.queue_after != self.queue_before + self.queue_accepted:
            raise ValueError("FSG3 queue output accounting differs")
        if (
            len(self.depths) != self.queue_input
            or self.history_count != self.queue_input
        ):
            raise ValueError("FSG3 queue depth/history accounting differs")
        if len(self.final_decision) != self.batch_size:
            raise ValueError("FSG3 decision batch accounting differs")
        if self.n_verified + self.n_splits != self.batch_size:
            raise ValueError("FSG3 verified/split accounting differs")
        expected_lower = math.prod(self.lower_shape)
        expected_upper = math.prod(self.upper_shape)
        if expected_lower != len(self.lower_values) or expected_upper != len(
            self.upper_values
        ):
            raise ValueError("FSG3 semantic tensor shape differs")
        if not self.lower_values or not self.upper_values:
            raise ValueError("FSG3 semantic tensor is empty")
        if not all(math.isfinite(value) for value in self.lower_values):
            raise ValueError("FSG3 semantic tensor is non-finite")
        if len(self.upper_positive_infinity_mask) != len(self.upper_values):
            raise ValueError("FSG3 upper infinity mask shape differs")
        if not all(math.isfinite(value) for value in self.upper_values):
            raise ValueError("FSG3 encoded upper tensor is non-finite")
        if any(
            is_infinite and value != 0.0
            for value, is_infinite in zip(
                self.upper_values, self.upper_positive_infinity_mask
            )
        ):
            raise ValueError("FSG3 upper infinity placeholder differs")

    def to_dict(self) -> dict[str, object]:
        """Return the stable semantic payload."""

        self.validate()
        return {
            "status": self.status,
            "success": self.success,
            "visited_domains": list(self.visited_domains),
            "queue_before": self.queue_before,
            "queue_input": self.queue_input,
            "queue_accepted": self.queue_accepted,
            "queue_pruned": self.queue_pruned,
            "queue_after": self.queue_after,
            "depths": list(self.depths),
            "history_count": self.history_count,
            "lower_shape": list(self.lower_shape),
            "lower_values": list(self.lower_values),
            "upper_shape": list(self.upper_shape),
            "upper_values": list(self.upper_values),
            "upper_positive_infinity_mask": list(self.upper_positive_infinity_mask),
            "final_decision": [list(value) for value in self.final_decision],
            "split_depth": self.split_depth,
            "batch_size": self.batch_size,
            "n_verified": self.n_verified,
            "n_splits": self.n_splits,
        }


@dataclass(frozen=True)
class FSG3ExecutionCounters:
    """Direct evidence for passthrough and no-provider replacement behavior."""

    typed_validation_count: int
    provider_core_call_count: int
    provider_compute_bounds_call_count: int
    provider_update_bounds_call_count: int
    fallback_dispatch_count: int
    backend_kind: str
    replacement_mode: str

    def validate(self, configuration: FSG3Configuration) -> None:
        """Enforce the preregistered physical behavior of B0, B1, and B2."""

        values = (
            self.typed_validation_count,
            self.provider_core_call_count,
            self.provider_compute_bounds_call_count,
            self.provider_update_bounds_call_count,
            self.fallback_dispatch_count,
        )
        if any(value < 0 for value in values) or not self.backend_kind:
            raise ValueError("FSG3 execution counters differ")
        expected = {
            FSG3Configuration.B0: (0, 1, "original_provider"),
            FSG3Configuration.B1: (1, 1, "rvir_passthrough"),
            FSG3Configuration.B2: (1, 0, "whole_call"),
        }[configuration]
        if (
            self.typed_validation_count,
            self.provider_core_call_count,
            self.replacement_mode,
        ) != expected:
            raise ValueError("FSG3 replacement behavior differs")
        if self.fallback_dispatch_count != 0:
            raise ValueError("FSG3 fallback dispatch is forbidden")
        if configuration == FSG3Configuration.B2 and (
            self.provider_compute_bounds_call_count != 0
            or self.provider_update_bounds_call_count != 0
            or self.backend_kind != "torch-eager-reference"
        ):
            raise ValueError("FSG3 B2 provider-free reference backend differs")
        if configuration in {FSG3Configuration.B0, FSG3Configuration.B1} and (
            self.provider_compute_bounds_call_count <= 0
            or self.provider_update_bounds_call_count <= 0
            or self.backend_kind != "auto-lirpa"
        ):
            raise ValueError("FSG3 B0/B1 provider execution differs")

    def to_dict(self) -> dict[str, object]:
        """Return the stable execution payload."""

        return {
            "typed_validation_count": self.typed_validation_count,
            "provider_core_call_count": self.provider_core_call_count,
            "provider_compute_bounds_call_count": self.provider_compute_bounds_call_count,
            "provider_update_bounds_call_count": self.provider_update_bounds_call_count,
            "fallback_dispatch_count": self.fallback_dispatch_count,
            "backend_kind": self.backend_kind,
            "replacement_mode": self.replacement_mode,
        }


@dataclass(frozen=True)
class FSG3EnvironmentGate:  # pylint: disable=too-many-instance-attributes
    """Raw worker-adjacent exclusion and thermal evidence."""

    gpu_uuid: str
    gpu_name: str
    runtime_identity: str
    external_compute_processes: Tuple[str, ...]
    software_thermal_signal: bool
    software_power_cap_signal: bool
    software_thermal_power_counters_coupled: bool
    hardware_thermal_slowdown: bool
    worker_overlap: bool
    device_identity_stable: bool
    ac_powered: bool

    def validate(self) -> None:
        """Reject missing device identity without precomputing the gate result."""

        if not self.gpu_uuid or not self.gpu_name or not self.runtime_identity:
            raise ValueError("FSG3 GPU identity is empty")
        if self.software_thermal_power_counters_coupled and not (
            self.software_thermal_signal and self.software_power_cap_signal
        ):
            raise ValueError("FSG3 coupled software counter projection differs")

    @property
    def independent_thermal_slowdown(self) -> bool:
        """Exclude exact driver-coupled power/thermal telemetry aliases."""

        return self.hardware_thermal_slowdown or (
            self.software_thermal_signal
            and not self.software_thermal_power_counters_coupled
        )

    @property
    def admitted(self) -> bool:
        """Return the deterministic environment admission decision."""

        self.validate()
        return (
            not self.external_compute_processes
            and not self.independent_thermal_slowdown
            and not self.worker_overlap
            and self.device_identity_stable
            and self.ac_powered
        )

    def to_dict(self) -> dict[str, object]:
        """Return raw inputs plus the recomputed admission projection."""

        return {
            "gpu_uuid": self.gpu_uuid,
            "gpu_name": self.gpu_name,
            "runtime_identity": self.runtime_identity,
            "external_compute_processes": list(self.external_compute_processes),
            "software_thermal_signal": self.software_thermal_signal,
            "software_power_cap_signal": self.software_power_cap_signal,
            "software_thermal_power_counters_coupled": (
                self.software_thermal_power_counters_coupled
            ),
            "hardware_thermal_slowdown": self.hardware_thermal_slowdown,
            "independent_thermal_slowdown": self.independent_thermal_slowdown,
            "worker_overlap": self.worker_overlap,
            "device_identity_stable": self.device_identity_stable,
            "ac_powered": self.ac_powered,
            "admitted": self.admitted,
        }


@dataclass(frozen=True)
class FSG3TimingRun:  # pylint: disable=too-many-instance-attributes
    """One canonical fresh-process control or profile result."""

    run_id: str
    block_index: int
    sequence_position: int
    configuration: FSG3Configuration
    mode: FSG3Mode
    source_identity: str
    protocol_identity: str
    metrics: FSG3TimingMetrics
    semantics: FSG3SemanticResult
    execution: FSG3ExecutionCounters
    environment: FSG3EnvironmentGate
    profile_spans: Tuple[FSG3ProfileSpan, ...]
    profile_closure_error: Optional[float]
    profile_residual_share: Optional[float]
    performance_claimed: bool = False
    schema_version: str = FSG3_TIMING_SCHEMA_VERSION

    def validate(self) -> None:
        """Validate identity, physical behavior, metrics, and profile projections."""

        if self.schema_version != FSG3_TIMING_SCHEMA_VERSION:
            raise ValueError("FSG3 timing schema version differs")
        if not self.run_id or not self.source_identity or not self.protocol_identity:
            raise ValueError("FSG3 run identity is empty")
        if self.block_index not in range(
            FSG3_REPEAT_COUNT
        ) or self.sequence_position not in range(6):
            raise ValueError("FSG3 run position differs")
        if self.performance_claimed:
            raise ValueError("FSG3 raw run cannot claim performance")
        self.metrics.validate(self.configuration)
        self.semantics.validate()
        self.execution.validate(self.configuration)
        self.environment.validate()
        if self.mode == FSG3Mode.CONTROL:
            if (
                self.profile_spans
                or self.profile_closure_error is not None
                or self.profile_residual_share is not None
            ):
                raise ValueError("FSG3 control cannot contain profile closure")
        else:
            expected_layout = FSG3_PROFILE_SPAN_LAYOUT[self.configuration]
            observed_layout = tuple(
                (span.scope, span.name) for span in self.profile_spans
            )
            if observed_layout != expected_layout:
                raise ValueError("FSG3 profile span layout differs")
            previous_end = -1
            for span in self.profile_spans:
                span.validate()
                if span.start_offset_ns < previous_end:
                    raise ValueError("FSG3 profile spans overlap")
                previous_end = span.end_offset_ns
            values = (self.profile_closure_error, self.profile_residual_share)
            if any(
                value is None or not math.isfinite(value) or value < 0
                for value in values
            ):
                raise ValueError("FSG3 profile closure is incomplete")
            covered_core_ns = sum(
                span.wall_ns for span in self.profile_spans if span.scope == "core"
            )
            closure = abs(self.metrics.core_wall_ns - covered_core_ns) / float(
                self.metrics.core_wall_ns
            )
            residual = max(self.metrics.core_wall_ns - covered_core_ns, 0) / float(
                self.metrics.core_wall_ns
            )
            if not math.isclose(
                cast(float, self.profile_closure_error),
                closure,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ) or not math.isclose(
                cast(float, self.profile_residual_share),
                residual,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("FSG3 profile closure projection differs")

    def to_dict(self) -> dict[str, object]:
        """Return the complete canonical raw record."""

        self.validate()
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "block_index": self.block_index,
            "sequence_position": self.sequence_position,
            "configuration": self.configuration.value,
            "mode": self.mode.value,
            "source_identity": self.source_identity,
            "protocol_identity": self.protocol_identity,
            "metrics": self.metrics.to_dict(),
            "semantics": self.semantics.to_dict(),
            "execution": self.execution.to_dict(),
            "environment": self.environment.to_dict(),
            "profile_spans": [span.to_dict() for span in self.profile_spans],
            "profile_closure_error": self.profile_closure_error,
            "profile_residual_share": self.profile_residual_share,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        """Bind every raw timing, semantic, counter, and environment input."""

        return canonical_hash(self.to_dict())


def _timing_from_dict(value: Mapping[str, Any]) -> FSG3TimingMetrics:
    names = {
        "cold_total_ns",
        "boundflow_compile_ns",
        "query_wall_ns",
        "query_gpu_ns",
        "core_wall_ns",
        "core_gpu_ns",
        "post_validation_ns",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }
    _exact_keys(value, names, "timing metrics")
    return FSG3TimingMetrics(**{name: int(value[name]) for name in names})


def _semantics_from_dict(value: Mapping[str, Any]) -> FSG3SemanticResult:
    names = {
        "status",
        "success",
        "visited_domains",
        "queue_before",
        "queue_input",
        "queue_accepted",
        "queue_pruned",
        "queue_after",
        "depths",
        "history_count",
        "lower_shape",
        "lower_values",
        "upper_shape",
        "upper_values",
        "upper_positive_infinity_mask",
        "final_decision",
        "split_depth",
        "batch_size",
        "n_verified",
        "n_splits",
    }
    _exact_keys(value, names, "semantic result")
    decision = tuple(
        (int(item[0]), int(item[1]))
        for item in _sequence(value["final_decision"], "final decision")
    )
    return FSG3SemanticResult(
        status=str(value["status"]),
        success=bool(value["success"]),
        visited_domains=tuple(
            int(item) for item in _sequence(value["visited_domains"], "visited domains")
        ),
        queue_before=int(value["queue_before"]),
        queue_input=int(value["queue_input"]),
        queue_accepted=int(value["queue_accepted"]),
        queue_pruned=int(value["queue_pruned"]),
        queue_after=int(value["queue_after"]),
        depths=tuple(int(item) for item in _sequence(value["depths"], "depths")),
        history_count=int(value["history_count"]),
        lower_shape=tuple(
            int(item) for item in _sequence(value["lower_shape"], "lower shape")
        ),
        lower_values=tuple(
            float(item) for item in _sequence(value["lower_values"], "lower values")
        ),
        upper_shape=tuple(
            int(item) for item in _sequence(value["upper_shape"], "upper shape")
        ),
        upper_values=tuple(
            float(item) for item in _sequence(value["upper_values"], "upper values")
        ),
        upper_positive_infinity_mask=tuple(
            bool(item)
            for item in _sequence(
                value["upper_positive_infinity_mask"], "upper infinity mask"
            )
        ),
        final_decision=decision,
        split_depth=int(value["split_depth"]),
        batch_size=int(value["batch_size"]),
        n_verified=int(value["n_verified"]),
        n_splits=int(value["n_splits"]),
    )


def _profile_span_from_dict(value: Mapping[str, Any]) -> FSG3ProfileSpan:
    names = {
        "scope",
        "name",
        "stack_layer",
        "solver_phase",
        "resource",
        "cache_state",
        "start_offset_ns",
        "end_offset_ns",
        "wall_ns",
        "gpu_ns",
    }
    _exact_keys(value, names, "profile span")
    span = FSG3ProfileSpan(
        scope=str(value["scope"]),
        name=str(value["name"]),
        stack_layer=str(value["stack_layer"]),
        solver_phase=str(value["solver_phase"]),
        resource=str(value["resource"]),
        cache_state=str(value["cache_state"]),
        start_offset_ns=int(value["start_offset_ns"]),
        end_offset_ns=int(value["end_offset_ns"]),
        wall_ns=int(value["wall_ns"]),
        gpu_ns=int(value["gpu_ns"]),
    )
    span.validate()
    return span


def fsg3_timing_run_from_dict(value: Mapping[str, Any]) -> FSG3TimingRun:
    """Parse a canonical raw run and reject derived-field tampering."""

    _exact_keys(
        value,
        {
            "schema_version",
            "run_id",
            "block_index",
            "sequence_position",
            "configuration",
            "mode",
            "source_identity",
            "protocol_identity",
            "metrics",
            "semantics",
            "execution",
            "environment",
            "profile_spans",
            "profile_closure_error",
            "profile_residual_share",
            "performance_claimed",
        },
        "timing run",
    )
    execution_value = _mapping(value["execution"], "execution counters")
    _exact_keys(
        execution_value,
        {
            "typed_validation_count",
            "provider_core_call_count",
            "provider_compute_bounds_call_count",
            "provider_update_bounds_call_count",
            "fallback_dispatch_count",
            "backend_kind",
            "replacement_mode",
        },
        "execution counters",
    )
    environment_value = _mapping(value["environment"], "environment gate")
    _exact_keys(
        environment_value,
        {
            "gpu_uuid",
            "gpu_name",
            "runtime_identity",
            "external_compute_processes",
            "software_thermal_signal",
            "software_power_cap_signal",
            "software_thermal_power_counters_coupled",
            "hardware_thermal_slowdown",
            "independent_thermal_slowdown",
            "worker_overlap",
            "device_identity_stable",
            "ac_powered",
            "admitted",
        },
        "environment gate",
    )
    environment = FSG3EnvironmentGate(
        gpu_uuid=str(environment_value["gpu_uuid"]),
        gpu_name=str(environment_value["gpu_name"]),
        runtime_identity=str(environment_value["runtime_identity"]),
        external_compute_processes=tuple(
            str(item)
            for item in _sequence(
                environment_value["external_compute_processes"], "external processes"
            )
        ),
        software_thermal_signal=bool(environment_value["software_thermal_signal"]),
        software_power_cap_signal=bool(environment_value["software_power_cap_signal"]),
        software_thermal_power_counters_coupled=bool(
            environment_value["software_thermal_power_counters_coupled"]
        ),
        hardware_thermal_slowdown=bool(environment_value["hardware_thermal_slowdown"]),
        worker_overlap=bool(environment_value["worker_overlap"]),
        device_identity_stable=bool(environment_value["device_identity_stable"]),
        ac_powered=bool(environment_value["ac_powered"]),
    )
    if environment.admitted is not bool(environment_value["admitted"]):
        raise ValueError("FSG3 environment admission projection differs")
    if environment.independent_thermal_slowdown is not bool(
        environment_value["independent_thermal_slowdown"]
    ):
        raise ValueError("FSG3 independent thermal projection differs")
    run = FSG3TimingRun(
        schema_version=str(value["schema_version"]),
        run_id=str(value["run_id"]),
        block_index=int(value["block_index"]),
        sequence_position=int(value["sequence_position"]),
        configuration=FSG3Configuration(str(value["configuration"])),
        mode=FSG3Mode(str(value["mode"])),
        source_identity=str(value["source_identity"]),
        protocol_identity=str(value["protocol_identity"]),
        metrics=_timing_from_dict(_mapping(value["metrics"], "timing metrics")),
        semantics=_semantics_from_dict(_mapping(value["semantics"], "semantic result")),
        execution=FSG3ExecutionCounters(
            typed_validation_count=int(execution_value["typed_validation_count"]),
            provider_core_call_count=int(execution_value["provider_core_call_count"]),
            provider_compute_bounds_call_count=int(
                execution_value["provider_compute_bounds_call_count"]
            ),
            provider_update_bounds_call_count=int(
                execution_value["provider_update_bounds_call_count"]
            ),
            fallback_dispatch_count=int(execution_value["fallback_dispatch_count"]),
            backend_kind=str(execution_value["backend_kind"]),
            replacement_mode=str(execution_value["replacement_mode"]),
        ),
        environment=environment,
        profile_spans=tuple(
            _profile_span_from_dict(_mapping(item, "profile span"))
            for item in _sequence(value["profile_spans"], "profile spans")
        ),
        profile_closure_error=(
            None
            if value["profile_closure_error"] is None
            else float(value["profile_closure_error"])
        ),
        profile_residual_share=(
            None
            if value["profile_residual_share"] is None
            else float(value["profile_residual_share"])
        ),
        performance_claimed=bool(value["performance_claimed"]),
    )
    run.validate()
    if run.to_dict() != dict(value):
        raise ValueError("FSG3 timing run canonical payload differs")
    return run


def _float_tolerance(reference: float) -> float:
    return FSG3_FLOAT_ATOL + FSG3_FLOAT_RTOL * abs(reference)


def _semantic_pair_failures(
    reference: FSG3SemanticResult,
    candidate: FSG3SemanticResult,
    *,
    label: str,
) -> list[str]:
    failures: list[str] = []
    discrete_names = (
        "status",
        "success",
        "visited_domains",
        "queue_before",
        "queue_input",
        "queue_accepted",
        "queue_pruned",
        "queue_after",
        "depths",
        "history_count",
        "lower_shape",
        "upper_shape",
        "upper_positive_infinity_mask",
        "final_decision",
        "split_depth",
        "batch_size",
        "n_verified",
        "n_splits",
    )
    for name in discrete_names:
        if getattr(reference, name) != getattr(candidate, name):
            failures.append(f"{label}:{name}:differs")
    for polarity, baseline, observed in (
        ("lower", reference.lower_values, candidate.lower_values),
        ("upper", reference.upper_values, candidate.upper_values),
    ):
        if len(baseline) != len(observed):
            failures.append(f"{label}:{polarity}:length-differs")
            continue
        for index, (expected, actual) in enumerate(zip(baseline, observed)):
            if polarity == "upper" and reference.upper_positive_infinity_mask[index]:
                continue
            tolerance = _float_tolerance(expected)
            if abs(actual - expected) > tolerance:
                failures.append(f"{label}:{polarity}[{index}]:allclose-failed")
            if polarity == "lower" and actual > expected + tolerance:
                failures.append(f"{label}:{polarity}[{index}]:optimistic")
            if polarity == "upper" and actual < expected - tolerance:
                failures.append(f"{label}:{polarity}[{index}]:optimistic")
    return failures


def _metric_summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != FSG3_REPEAT_COUNT or any(
        not math.isfinite(value) or value <= 0.0 for value in values
    ):
        raise ValueError("FSG3 paired metric coverage differs")
    median = statistics.median(values)
    return {
        "raw": list(values),
        "median": median,
        "minimum": min(values),
        "maximum": max(values),
        "mad": statistics.median(abs(value - median) for value in values),
        "geometric_mean": math.exp(
            sum(math.log(value) for value in values) / len(values)
        ),
    }


def _speedup_summary(
    controls: Mapping[tuple[int, FSG3Configuration], FSG3TimingRun],
    candidate: FSG3Configuration,
    metric: str,
) -> dict[str, object]:
    values = [
        getattr(controls[(block, FSG3Configuration.B0)].metrics, metric)
        / getattr(controls[(block, candidate)].metrics, metric)
        for block in range(FSG3_REPEAT_COUNT)
    ]
    return _metric_summary(values)


def _profile_attribution(
    indexed: Mapping[tuple[int, FSG3Configuration, FSG3Mode], FSG3TimingRun],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for configuration in FSG3Configuration:
        by_name: dict[str, object] = {}
        for scope, name in FSG3_PROFILE_SPAN_LAYOUT[configuration]:
            wall_values: list[float] = []
            share_values: list[float] = []
            gpu_values: list[float] = []
            for block in range(FSG3_REPEAT_COUNT):
                run = indexed[(block, configuration, FSG3Mode.PROFILE)]
                span = next(item for item in run.profile_spans if item.name == name)
                denominator = {
                    "compile": run.metrics.cold_total_ns,
                    "core": run.metrics.core_wall_ns,
                    "post": run.metrics.query_wall_ns,
                }[scope]
                wall_values.append(float(span.wall_ns))
                share_values.append(span.wall_ns / float(denominator))
                gpu_values.append(float(span.gpu_ns))
            by_name[name] = {
                "scope": scope,
                "wall_ns": _metric_summary(wall_values),
                "scope_share": _metric_summary(share_values),
                "gpu_ns": (
                    _metric_summary(gpu_values)
                    if all(value > 0 for value in gpu_values)
                    else {"raw": gpu_values, "not_applicable": True}
                ),
            }
        result[configuration.value] = by_name
    return result


def derive_fsg3_timing_evidence(  # pylint: disable=too-many-statements
    runs: Sequence[FSG3TimingRun],
) -> dict[str, object]:
    """Rebuild sequence, semantics, perturbation, statistics, and FSG3 decision."""

    if len(runs) != len(expected_fsg3_sequence()):
        raise ValueError("FSG3 run count differs")
    expected = expected_fsg3_sequence()
    for run, (block, position, configuration, mode) in zip(runs, expected):
        run.validate()
        if (run.block_index, run.sequence_position, run.configuration, run.mode) != (
            block,
            position,
            configuration,
            mode,
        ):
            raise ValueError("FSG3 preregistered sequence differs")
    if len({run.run_id for run in runs}) != len(runs):
        raise ValueError("FSG3 run identity duplicates")
    if (
        len({run.source_identity for run in runs}) != 1
        or len({run.protocol_identity for run in runs}) != 1
    ):
        raise ValueError("FSG3 source or protocol identity differs")

    indexed = {(run.block_index, run.configuration, run.mode): run for run in runs}
    controls = {
        (block, configuration): indexed[(block, configuration, FSG3Mode.CONTROL)]
        for block in range(FSG3_REPEAT_COUNT)
        for configuration in FSG3Configuration
    }
    failures: list[str] = []
    environment_identities = {
        (
            run.environment.gpu_uuid,
            run.environment.gpu_name,
            run.environment.runtime_identity,
        )
        for run in runs
    }
    if len(environment_identities) != 1:
        failures.append("environment-device-or-runtime-identity-differs")
    for run in runs:
        if not run.environment.admitted:
            failures.append(f"{run.run_id}:environment-not-admitted")
        if run.mode == FSG3Mode.PROFILE:
            if (
                run.profile_closure_error is not None
                and run.profile_closure_error > FSG3_CLOSURE_ERROR_LIMIT
            ):
                failures.append(f"{run.run_id}:closure-failed")
            if (
                run.profile_residual_share is not None
                and run.profile_residual_share > FSG3_RESIDUAL_LIMIT
            ):
                failures.append(f"{run.run_id}:residual-failed")

    perturbation: dict[str, object] = {}
    for configuration in FSG3Configuration:
        ratios = [
            indexed[(block, configuration, FSG3Mode.PROFILE)].metrics.query_wall_ns
            / indexed[(block, configuration, FSG3Mode.CONTROL)].metrics.query_wall_ns
            for block in range(FSG3_REPEAT_COUNT)
        ]
        summary = _metric_summary(ratios)
        passed = float(summary["median"]) <= FSG3_PROFILE_PERTURBATION_LIMIT
        summary["gate"] = FSG3_PROFILE_PERTURBATION_LIMIT
        summary["passed"] = passed
        perturbation[configuration.value] = summary
        if not passed:
            failures.append(f"{configuration.value}:profile-perturbation-failed")

    for block in range(FSG3_REPEAT_COUNT):
        for mode in FSG3Mode:
            reference = indexed[(block, FSG3Configuration.B0, mode)].semantics
            passthrough = indexed[(block, FSG3Configuration.B1, mode)]
            original = indexed[(block, FSG3Configuration.B0, mode)]
            if (
                passthrough.execution.provider_compute_bounds_call_count
                != original.execution.provider_compute_bounds_call_count
                or passthrough.execution.provider_update_bounds_call_count
                != original.execution.provider_update_bounds_call_count
            ):
                failures.append(f"block-{block}:{mode.value}:B1:provider-count-differs")
            for configuration in (FSG3Configuration.B1, FSG3Configuration.B2):
                candidate = indexed[(block, configuration, mode)].semantics
                failures.extend(
                    _semantic_pair_failures(
                        reference,
                        candidate,
                        label=f"block-{block}:{mode.value}:{configuration.value}",
                    )
                )
        for configuration in FSG3Configuration:
            control = indexed[(block, configuration, FSG3Mode.CONTROL)].semantics
            profile = indexed[(block, configuration, FSG3Mode.PROFILE)].semantics
            failures.extend(
                _semantic_pair_failures(
                    control,
                    profile,
                    label=f"block-{block}:{configuration.value}:profile-control",
                )
            )

    metric_names = (
        "cold_total_ns",
        "query_wall_ns",
        "core_wall_ns",
        "query_gpu_ns",
        "core_gpu_ns",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    speedups = {
        configuration.value: {
            metric: _speedup_summary(controls, configuration, metric)
            for metric in metric_names
        }
        for configuration in (FSG3Configuration.B1, FSG3Configuration.B2)
    }
    b0_query = statistics.median(
        controls[(block, FSG3Configuration.B0)].metrics.query_wall_ns
        for block in range(FSG3_REPEAT_COUNT)
    )
    b2_query = statistics.median(
        controls[(block, FSG3Configuration.B2)].metrics.query_wall_ns
        for block in range(FSG3_REPEAT_COUNT)
    )
    b2_compile = statistics.median(
        controls[(block, FSG3Configuration.B2)].metrics.boundflow_compile_ns
        for block in range(FSG3_REPEAT_COUNT)
    )
    saving = b0_query - b2_query
    break_even: int | str = (
        math.ceil(b2_compile / saving) if saving > 0 else "not_reachable"
    )
    status = "validated-fsg3-b0-b1-b2-baseline" if not failures else "not-auditable"
    result: dict[str, object] = {
        "schema_version": FSG3_TIMING_SCHEMA_VERSION,
        "status": status,
        "run_count": len(runs),
        "block_count": FSG3_REPEAT_COUNT,
        "control_count_by_configuration": {
            item.value: FSG3_REPEAT_COUNT for item in FSG3Configuration
        },
        "profile_count_by_configuration": {
            item.value: FSG3_REPEAT_COUNT for item in FSG3Configuration
        },
        "sequence": [run.run_id for run in runs],
        "run_hashes": [run.stable_hash() for run in runs],
        "perturbation": perturbation,
        "speedups_b0_over_candidate": speedups,
        "profile_attribution": _profile_attribution(indexed),
        "b2_compile_break_even_queries": break_even,
        "failure_rows": failures,
        "correctness_passed": not any("block-" in item for item in failures),
        "environment_passed": not any("environment" in item for item in failures),
        "measurement_auditable": not failures,
        "performance_claimed": False,
    }
    result["summary_hash"] = canonical_hash(result)
    return result
