"""Typed B0/B2/B3 contracts and replay aggregation for FSG4 timing."""

# pylint: disable=too-many-branches,too-many-locals,too-many-lines,duplicate-code
# pylint: disable=too-many-instance-attributes,too-many-statements
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import statistics
from typing import Any, Mapping, Optional, Sequence, Tuple, cast

from .fsg3_same_solver_timing import (
    _profile_span_from_dict,
    _semantic_pair_failures,
    _semantics_from_dict,
    _timing_from_dict,
    canonical_hash,
    FSG3Configuration,
    FSG3EnvironmentGate,
    FSG3Mode,
    FSG3ProfileSpan,
    FSG3SemanticResult,
    FSG3TimingMetrics,
)
from .fsg4_b3_explicit_counters import (
    COUNTER_NAMES,
    EXPECTED_B2_FIXED_COUNTERS,
    EXPECTED_B3C_FIXED_COUNTERS,
)

FSG4_B3_TIMING_SCHEMA_VERSION = "boundflow.fsg4-b3-same-solver-timing/v1"
FSG4_B3_ACTIVATION_SCHEMA_VERSION = "boundflow.fsg4-b3-timing-activation/v1"
FSG4_B3_REPEAT_COUNT = 6
FSG4_B3_PROFILE_PERTURBATION_LIMIT = 1.05
FSG4_B3_CLOSURE_ERROR_LIMIT = 0.01
FSG4_B3_RESIDUAL_LIMIT = 0.03


class FSG4B3TimingConfiguration(str, Enum):
    """Frozen cumulative configurations for the B3 experiment."""

    B0 = "B0"
    B2 = "B2"
    B3 = "B3"


FSG4_B3_PROFILE_SPAN_LAYOUT: Mapping[
    FSG4B3TimingConfiguration, Tuple[Tuple[str, str], ...]
] = {
    FSG4B3TimingConfiguration.B0: (
        ("core", "provider_core"),
        ("post", "official_post_queue"),
    ),
    FSG4B3TimingConfiguration.B2: (
        ("compile", "compile"),
        ("core", "typed_pre_state"),
        ("core", "optimizer"),
        ("core", "backward"),
        ("core", "kfsb"),
        ("core", "atomic_commit"),
        ("post", "official_post_queue"),
    ),
    FSG4B3TimingConfiguration.B3: (
        ("compile", "compile"),
        ("core", "typed_pre_state"),
        ("core", "optimizer"),
        ("core", "backward"),
        ("core", "kfsb"),
        ("core", "atomic_commit"),
        ("post", "official_post_queue"),
    ),
}

FSG4_B3_CONFIG_ORDERS: Tuple[Tuple[FSG4B3TimingConfiguration, ...], ...] = (
    (
        FSG4B3TimingConfiguration.B0,
        FSG4B3TimingConfiguration.B2,
        FSG4B3TimingConfiguration.B3,
    ),
    (
        FSG4B3TimingConfiguration.B0,
        FSG4B3TimingConfiguration.B3,
        FSG4B3TimingConfiguration.B2,
    ),
    (
        FSG4B3TimingConfiguration.B2,
        FSG4B3TimingConfiguration.B0,
        FSG4B3TimingConfiguration.B3,
    ),
    (
        FSG4B3TimingConfiguration.B2,
        FSG4B3TimingConfiguration.B3,
        FSG4B3TimingConfiguration.B0,
    ),
    (
        FSG4B3TimingConfiguration.B3,
        FSG4B3TimingConfiguration.B0,
        FSG4B3TimingConfiguration.B2,
    ),
    (
        FSG4B3TimingConfiguration.B3,
        FSG4B3TimingConfiguration.B2,
        FSG4B3TimingConfiguration.B0,
    ),
)


def expected_fsg4_b3_sequence() -> (
    Tuple[Tuple[int, int, FSG4B3TimingConfiguration, FSG3Mode], ...]
):
    """Return the immutable 36-worker B0/B2/B3 sequence."""

    rows: list[tuple[int, int, FSG4B3TimingConfiguration, FSG3Mode]] = []
    for block_index, configurations in enumerate(FSG4_B3_CONFIG_ORDERS):
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


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"FSG4/B3 {label} fields differ")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"FSG4/B3 {label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"FSG4/B3 {label} must be a sequence")
    return value


def _base_configuration(
    configuration: FSG4B3TimingConfiguration,
) -> FSG3Configuration:
    return (
        FSG3Configuration.B0
        if configuration == FSG4B3TimingConfiguration.B0
        else FSG3Configuration.B2
    )


@dataclass(frozen=True)
class FSG4B3ExecutionCounters:
    """Provider/fallback evidence plus an exact physical replacement label."""

    typed_validation_count: int
    provider_core_call_count: int
    provider_compute_bounds_call_count: int
    provider_update_bounds_call_count: int
    fallback_dispatch_count: int
    backend_kind: str
    replacement_mode: str

    def validate(self, configuration: FSG4B3TimingConfiguration) -> None:
        values = (
            self.typed_validation_count,
            self.provider_core_call_count,
            self.provider_compute_bounds_call_count,
            self.provider_update_bounds_call_count,
            self.fallback_dispatch_count,
        )
        if any(value < 0 for value in values) or not self.backend_kind:
            raise ValueError("FSG4/B3 execution counters differ")
        expected = {
            FSG4B3TimingConfiguration.B0: (0, 1, "original_provider"),
            FSG4B3TimingConfiguration.B2: (1, 0, "whole_call_reference"),
            FSG4B3TimingConfiguration.B3: (1, 0, "b3_ir_graph_plan_schedule"),
        }[configuration]
        if (
            self.typed_validation_count,
            self.provider_core_call_count,
            self.replacement_mode,
        ) != expected:
            raise ValueError("FSG4/B3 replacement behavior differs")
        if self.fallback_dispatch_count != 0:
            raise ValueError("FSG4/B3 fallback dispatch is forbidden")
        if configuration == FSG4B3TimingConfiguration.B0:
            if (
                self.provider_compute_bounds_call_count <= 0
                or self.provider_update_bounds_call_count <= 0
                or self.backend_kind != "auto-lirpa"
            ):
                raise ValueError("FSG4/B3 B0 provider execution differs")
        elif (
            self.provider_compute_bounds_call_count != 0
            or self.provider_update_bounds_call_count != 0
            or self.backend_kind != "torch-eager-reference"
        ):
            raise ValueError("FSG4/B3 replacement must be provider-free")

    def to_dict(self) -> dict[str, object]:
        return {
            "typed_validation_count": self.typed_validation_count,
            "provider_core_call_count": self.provider_core_call_count,
            "provider_compute_bounds_call_count": (
                self.provider_compute_bounds_call_count
            ),
            "provider_update_bounds_call_count": (
                self.provider_update_bounds_call_count
            ),
            "fallback_dispatch_count": self.fallback_dispatch_count,
            "backend_kind": self.backend_kind,
            "replacement_mode": self.replacement_mode,
        }


@dataclass(frozen=True)
class FSG4B3ActivationReceipt:  # pylint: disable=too-many-instance-attributes
    """Direct B2/B3 executor receipts and profile-only physical counters."""

    prepared_core_template_count: int
    prepared_core_instance_count: int
    terminal_optimizer_schedule_count: int
    assembly_count: int
    commit_receipt_count: int
    device_commit_audit_count: int
    post_query_audit_ns: int
    post_query_audit_excluded_from_timing: bool
    headline_content_digest_count: Optional[int]
    candidate_d2h_copy_count: Optional[int]
    detailed_counts_by_name: Optional[Tuple[Tuple[str, int], ...]]
    performance_claimed: bool = False
    schema_version: str = FSG4_B3_ACTIVATION_SCHEMA_VERSION

    @property
    def detailed_counts(self) -> Optional[dict[str, int]]:
        return (
            None
            if self.detailed_counts_by_name is None
            else dict(self.detailed_counts_by_name)
        )

    def validate(
        self,
        configuration: FSG4B3TimingConfiguration,
        mode: FSG3Mode,
    ) -> None:
        if self.schema_version != FSG4_B3_ACTIVATION_SCHEMA_VERSION:
            raise ValueError("FSG4/B3 activation schema differs")
        values = (
            self.prepared_core_template_count,
            self.prepared_core_instance_count,
            self.terminal_optimizer_schedule_count,
            self.assembly_count,
            self.commit_receipt_count,
            self.device_commit_audit_count,
            self.post_query_audit_ns,
        )
        if any(value < 0 for value in values) or self.performance_claimed:
            raise ValueError("FSG4/B3 activation count differs")
        expected_receipts = {
            FSG4B3TimingConfiguration.B0: (0, 0, 0, 0, 0, 0),
            FSG4B3TimingConfiguration.B2: (0, 0, 0, 1, 1, 0),
            FSG4B3TimingConfiguration.B3: (1, 1, 1, 1, 1, 1),
        }[configuration]
        if values[:6] != expected_receipts:
            raise ValueError("FSG4/B3 direct activation receipt differs")
        if not self.post_query_audit_excluded_from_timing:
            raise ValueError("FSG4/B3 post-query audit scope differs")
        if configuration == FSG4B3TimingConfiguration.B3:
            if (
                self.post_query_audit_ns <= 0
                or self.headline_content_digest_count != 0
                or self.candidate_d2h_copy_count != 0
            ):
                raise ValueError("FSG4/B3 B3 device activation differs")
        elif (
            self.post_query_audit_ns != 0
            or self.headline_content_digest_count is not None
            or self.candidate_d2h_copy_count is not None
        ):
            raise ValueError("FSG4/B3 non-B3 audit projection differs")
        counts = self.detailed_counts
        detailed_required = mode == FSG3Mode.PROFILE and configuration in {
            FSG4B3TimingConfiguration.B2,
            FSG4B3TimingConfiguration.B3,
        }
        if detailed_required != (counts is not None):
            raise ValueError("FSG4/B3 profile counter admission differs")
        if counts is not None:
            if set(counts) != set(COUNTER_NAMES) or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in counts.values()
            ):
                raise ValueError("FSG4/B3 profile counter inventory differs")
            expected = (
                EXPECTED_B2_FIXED_COUNTERS
                if configuration == FSG4B3TimingConfiguration.B2
                else EXPECTED_B3C_FIXED_COUNTERS
            )
            for name, expected_value in expected.items():
                if counts[name] != expected_value:
                    raise ValueError(
                        f"FSG4/B3 profile counter differs: {name}:"
                        f"expected={expected_value}:observed={counts[name]}"
                    )
            for name in (
                "tensor_content_hash_count",
                "gpu_tensor_content_hash_count",
                "typed_validate_call_count",
                "stable_hash_call_count",
            ):
                if counts[name] <= 0:
                    raise ValueError(f"FSG4/B3 profile counter is empty: {name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "prepared_core_template_count": self.prepared_core_template_count,
            "prepared_core_instance_count": self.prepared_core_instance_count,
            "terminal_optimizer_schedule_count": (
                self.terminal_optimizer_schedule_count
            ),
            "assembly_count": self.assembly_count,
            "commit_receipt_count": self.commit_receipt_count,
            "device_commit_audit_count": self.device_commit_audit_count,
            "post_query_audit_ns": self.post_query_audit_ns,
            "post_query_audit_excluded_from_timing": (
                self.post_query_audit_excluded_from_timing
            ),
            "headline_content_digest_count": self.headline_content_digest_count,
            "candidate_d2h_copy_count": self.candidate_d2h_copy_count,
            "detailed_counts": self.detailed_counts,
            "performance_claimed": False,
        }


@dataclass(frozen=True)
class FSG4B3TimingRun:  # pylint: disable=too-many-instance-attributes
    """One canonical fresh-process B0, B2, or B3 result."""

    run_id: str
    block_index: int
    sequence_position: int
    configuration: FSG4B3TimingConfiguration
    mode: FSG3Mode
    source_identity: str
    protocol_identity: str
    metrics: FSG3TimingMetrics
    semantics: FSG3SemanticResult
    execution: FSG4B3ExecutionCounters
    environment: FSG3EnvironmentGate
    activation: FSG4B3ActivationReceipt
    profile_spans: Tuple[FSG3ProfileSpan, ...]
    profile_closure_error: Optional[float]
    profile_residual_share: Optional[float]
    performance_claimed: bool = False
    schema_version: str = FSG4_B3_TIMING_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != FSG4_B3_TIMING_SCHEMA_VERSION:
            raise ValueError("FSG4/B3 timing schema differs")
        if not self.run_id or not self.source_identity or not self.protocol_identity:
            raise ValueError("FSG4/B3 run identity is empty")
        if self.block_index not in range(
            FSG4_B3_REPEAT_COUNT
        ) or self.sequence_position not in range(6):
            raise ValueError("FSG4/B3 run position differs")
        if self.performance_claimed:
            raise ValueError("FSG4/B3 raw run cannot claim performance")
        self.metrics.validate(_base_configuration(self.configuration))
        self.semantics.validate()
        self.execution.validate(self.configuration)
        self.environment.validate()
        self.activation.validate(self.configuration, self.mode)
        if self.mode == FSG3Mode.CONTROL:
            if (
                self.profile_spans
                or self.profile_closure_error is not None
                or self.profile_residual_share is not None
            ):
                raise ValueError("FSG4/B3 control cannot contain profile closure")
            return
        observed_layout = tuple((span.scope, span.name) for span in self.profile_spans)
        if observed_layout != FSG4_B3_PROFILE_SPAN_LAYOUT[self.configuration]:
            raise ValueError("FSG4/B3 profile span layout differs")
        previous_end = -1
        for span in self.profile_spans:
            span.validate()
            if span.start_offset_ns < previous_end:
                raise ValueError("FSG4/B3 profile spans overlap")
            previous_end = span.end_offset_ns
        values = (self.profile_closure_error, self.profile_residual_share)
        if any(
            value is None or not math.isfinite(value) or value < 0 for value in values
        ):
            raise ValueError("FSG4/B3 profile closure is incomplete")
        covered = sum(
            span.wall_ns for span in self.profile_spans if span.scope == "core"
        )
        closure = abs(self.metrics.core_wall_ns - covered) / float(
            self.metrics.core_wall_ns
        )
        residual = max(self.metrics.core_wall_ns - covered, 0) / float(
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
            raise ValueError("FSG4/B3 profile closure projection differs")

    def to_dict(self) -> dict[str, object]:
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
            "activation": self.activation.to_dict(),
            "profile_spans": [span.to_dict() for span in self.profile_spans],
            "profile_closure_error": self.profile_closure_error,
            "profile_residual_share": self.profile_residual_share,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return canonical_hash(self.to_dict())


def _environment_from_dict(value: Mapping[str, Any]) -> FSG3EnvironmentGate:
    _exact_keys(
        value,
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
        gpu_uuid=str(value["gpu_uuid"]),
        gpu_name=str(value["gpu_name"]),
        runtime_identity=str(value["runtime_identity"]),
        external_compute_processes=tuple(
            str(item)
            for item in _sequence(value["external_compute_processes"], "processes")
        ),
        software_thermal_signal=bool(value["software_thermal_signal"]),
        software_power_cap_signal=bool(value["software_power_cap_signal"]),
        software_thermal_power_counters_coupled=bool(
            value["software_thermal_power_counters_coupled"]
        ),
        hardware_thermal_slowdown=bool(value["hardware_thermal_slowdown"]),
        worker_overlap=bool(value["worker_overlap"]),
        device_identity_stable=bool(value["device_identity_stable"]),
        ac_powered=bool(value["ac_powered"]),
    )
    if environment.admitted is not bool(value["admitted"]):
        raise ValueError("FSG4/B3 environment admission projection differs")
    if environment.independent_thermal_slowdown is not bool(
        value["independent_thermal_slowdown"]
    ):
        raise ValueError("FSG4/B3 thermal projection differs")
    return environment


def _execution_from_dict(value: Mapping[str, Any]) -> FSG4B3ExecutionCounters:
    _exact_keys(
        value,
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
    return FSG4B3ExecutionCounters(
        typed_validation_count=int(value["typed_validation_count"]),
        provider_core_call_count=int(value["provider_core_call_count"]),
        provider_compute_bounds_call_count=int(
            value["provider_compute_bounds_call_count"]
        ),
        provider_update_bounds_call_count=int(
            value["provider_update_bounds_call_count"]
        ),
        fallback_dispatch_count=int(value["fallback_dispatch_count"]),
        backend_kind=str(value["backend_kind"]),
        replacement_mode=str(value["replacement_mode"]),
    )


def _activation_from_dict(value: Mapping[str, Any]) -> FSG4B3ActivationReceipt:
    _exact_keys(
        value,
        {
            "schema_version",
            "prepared_core_template_count",
            "prepared_core_instance_count",
            "terminal_optimizer_schedule_count",
            "assembly_count",
            "commit_receipt_count",
            "device_commit_audit_count",
            "post_query_audit_ns",
            "post_query_audit_excluded_from_timing",
            "headline_content_digest_count",
            "candidate_d2h_copy_count",
            "detailed_counts",
            "performance_claimed",
        },
        "activation receipt",
    )
    raw_counts = value["detailed_counts"]
    if raw_counts is not None and not isinstance(raw_counts, Mapping):
        raise TypeError("FSG4/B3 activation counters differ")
    return FSG4B3ActivationReceipt(
        prepared_core_template_count=int(value["prepared_core_template_count"]),
        prepared_core_instance_count=int(value["prepared_core_instance_count"]),
        terminal_optimizer_schedule_count=int(
            value["terminal_optimizer_schedule_count"]
        ),
        assembly_count=int(value["assembly_count"]),
        commit_receipt_count=int(value["commit_receipt_count"]),
        device_commit_audit_count=int(value["device_commit_audit_count"]),
        post_query_audit_ns=int(value["post_query_audit_ns"]),
        post_query_audit_excluded_from_timing=bool(
            value["post_query_audit_excluded_from_timing"]
        ),
        headline_content_digest_count=(
            None
            if value["headline_content_digest_count"] is None
            else int(value["headline_content_digest_count"])
        ),
        candidate_d2h_copy_count=(
            None
            if value["candidate_d2h_copy_count"] is None
            else int(value["candidate_d2h_copy_count"])
        ),
        detailed_counts_by_name=(
            None
            if raw_counts is None
            else tuple(
                sorted((str(name), int(count)) for name, count in raw_counts.items())
            )
        ),
        performance_claimed=bool(value["performance_claimed"]),
        schema_version=str(value["schema_version"]),
    )


def fsg4_b3_timing_run_from_dict(value: Mapping[str, Any]) -> FSG4B3TimingRun:
    """Parse one canonical FSG4 timing row and recompute all projections."""

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
            "activation",
            "profile_spans",
            "profile_closure_error",
            "profile_residual_share",
            "performance_claimed",
        },
        "timing run",
    )
    run = FSG4B3TimingRun(
        run_id=str(value["run_id"]),
        block_index=int(value["block_index"]),
        sequence_position=int(value["sequence_position"]),
        configuration=FSG4B3TimingConfiguration(str(value["configuration"])),
        mode=FSG3Mode(str(value["mode"])),
        source_identity=str(value["source_identity"]),
        protocol_identity=str(value["protocol_identity"]),
        metrics=_timing_from_dict(_mapping(value["metrics"], "metrics")),
        semantics=_semantics_from_dict(_mapping(value["semantics"], "semantics")),
        execution=_execution_from_dict(
            _mapping(value["execution"], "execution counters")
        ),
        environment=_environment_from_dict(
            _mapping(value["environment"], "environment gate")
        ),
        activation=_activation_from_dict(
            _mapping(value["activation"], "activation receipt")
        ),
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
        schema_version=str(value["schema_version"]),
    )
    run.validate()
    if run.to_dict() != dict(value):
        raise ValueError("FSG4/B3 timing run canonical payload differs")
    return run


def _metric_summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != FSG4_B3_REPEAT_COUNT or any(
        not math.isfinite(value) or value <= 0.0 for value in values
    ):
        raise ValueError("FSG4/B3 paired metric coverage differs")
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


def _ratio_summary(
    controls: Mapping[tuple[int, FSG4B3TimingConfiguration], FSG4B3TimingRun],
    numerator: FSG4B3TimingConfiguration,
    denominator: FSG4B3TimingConfiguration,
    metric: str,
) -> dict[str, object]:
    return _metric_summary(
        [
            getattr(controls[(block, numerator)].metrics, metric)
            / getattr(controls[(block, denominator)].metrics, metric)
            for block in range(FSG4_B3_REPEAT_COUNT)
        ]
    )


def _profile_attribution(
    indexed: Mapping[tuple[int, FSG4B3TimingConfiguration, FSG3Mode], FSG4B3TimingRun],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for configuration in FSG4B3TimingConfiguration:
        by_name: dict[str, object] = {}
        for scope, name in FSG4_B3_PROFILE_SPAN_LAYOUT[configuration]:
            wall_values: list[float] = []
            share_values: list[float] = []
            gpu_values: list[float] = []
            for block in range(FSG4_B3_REPEAT_COUNT):
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


def derive_fsg4_b3_timing_evidence(  # pylint: disable=too-many-statements
    runs: Sequence[FSG4B3TimingRun],
) -> dict[str, object]:
    """Rebuild correctness, measurements, ratios, and the preregistered decision."""

    expected = expected_fsg4_b3_sequence()
    if len(runs) != len(expected):
        raise ValueError("FSG4/B3 run count differs")
    for run, (block, position, configuration, mode) in zip(runs, expected):
        run.validate()
        if (run.block_index, run.sequence_position, run.configuration, run.mode) != (
            block,
            position,
            configuration,
            mode,
        ):
            raise ValueError("FSG4/B3 preregistered sequence differs")
    if len({run.run_id for run in runs}) != len(runs):
        raise ValueError("FSG4/B3 run identity duplicates")
    if (
        len({run.source_identity for run in runs}) != 1
        or len({run.protocol_identity for run in runs}) != 1
    ):
        raise ValueError("FSG4/B3 source or protocol identity differs")
    indexed = {(run.block_index, run.configuration, run.mode): run for run in runs}
    controls = {
        (block, configuration): indexed[(block, configuration, FSG3Mode.CONTROL)]
        for block in range(FSG4_B3_REPEAT_COUNT)
        for configuration in FSG4B3TimingConfiguration
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
            if cast(float, run.profile_closure_error) > FSG4_B3_CLOSURE_ERROR_LIMIT:
                failures.append(f"{run.run_id}:closure-failed")
            if cast(float, run.profile_residual_share) > FSG4_B3_RESIDUAL_LIMIT:
                failures.append(f"{run.run_id}:residual-failed")
    perturbation: dict[str, object] = {}
    for configuration in FSG4B3TimingConfiguration:
        summary = _metric_summary(
            [
                indexed[(block, configuration, FSG3Mode.PROFILE)].metrics.query_wall_ns
                / indexed[
                    (block, configuration, FSG3Mode.CONTROL)
                ].metrics.query_wall_ns
                for block in range(FSG4_B3_REPEAT_COUNT)
            ]
        )
        passed = float(summary["median"]) <= FSG4_B3_PROFILE_PERTURBATION_LIMIT
        summary["gate"] = FSG4_B3_PROFILE_PERTURBATION_LIMIT
        summary["passed"] = passed
        perturbation[configuration.value] = summary
        if not passed:
            failures.append(f"{configuration.value}:profile-perturbation-failed")
    for block in range(FSG4_B3_REPEAT_COUNT):
        for mode in FSG3Mode:
            reference = indexed[(block, FSG4B3TimingConfiguration.B0, mode)].semantics
            for configuration in (
                FSG4B3TimingConfiguration.B2,
                FSG4B3TimingConfiguration.B3,
            ):
                failures.extend(
                    _semantic_pair_failures(
                        reference,
                        indexed[(block, configuration, mode)].semantics,
                        label=f"block-{block}:{mode.value}:{configuration.value}",
                    )
                )
        for configuration in FSG4B3TimingConfiguration:
            failures.extend(
                _semantic_pair_failures(
                    indexed[(block, configuration, FSG3Mode.CONTROL)].semantics,
                    indexed[(block, configuration, FSG3Mode.PROFILE)].semantics,
                    label=f"block-{block}:{configuration.value}:profile-control",
                )
            )
    metrics = (
        "cold_total_ns",
        "query_wall_ns",
        "core_wall_ns",
        "query_gpu_ns",
        "core_gpu_ns",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    b0_over = {
        configuration.value: {
            metric: _ratio_summary(
                controls,
                FSG4B3TimingConfiguration.B0,
                configuration,
                metric,
            )
            for metric in metrics
        }
        for configuration in (
            FSG4B3TimingConfiguration.B2,
            FSG4B3TimingConfiguration.B3,
        )
    }
    b2_over_b3 = {
        metric: _ratio_summary(
            controls,
            FSG4B3TimingConfiguration.B2,
            FSG4B3TimingConfiguration.B3,
            metric,
        )
        for metric in metrics
    }
    core_speedup = float(cast(Any, b2_over_b3["core_wall_ns"]["geometric_mean"]))
    query_incremental = float(cast(Any, b2_over_b3["query_wall_ns"]["geometric_mean"]))
    query_cumulative = float(
        cast(Mapping[str, Any], b0_over["B3"])["query_wall_ns"]["geometric_mean"]
    )
    worst_pair_core = min(cast(Sequence[float], b2_over_b3["core_wall_ns"]["raw"]))
    if failures:
        status = "not-auditable"
    elif (
        core_speedup >= 1.15
        and query_cumulative >= 1.0
        and worst_pair_core >= 1.0 / 1.05
    ):
        status = "validated-b3"
    elif core_speedup >= 1.05 and query_incremental >= 1.0:
        status = "validated-reduced-b3"
    else:
        status = "validated-no-go-b3"
    result: dict[str, object] = {
        "schema_version": FSG4_B3_TIMING_SCHEMA_VERSION,
        "status": status,
        "run_count": len(runs),
        "block_count": FSG4_B3_REPEAT_COUNT,
        "control_count_by_configuration": {
            item.value: FSG4_B3_REPEAT_COUNT for item in FSG4B3TimingConfiguration
        },
        "profile_count_by_configuration": {
            item.value: FSG4_B3_REPEAT_COUNT for item in FSG4B3TimingConfiguration
        },
        "sequence": [run.run_id for run in runs],
        "run_hashes": [run.stable_hash() for run in runs],
        "perturbation": perturbation,
        "speedups_b0_over_candidate": b0_over,
        "speedups_b2_over_b3": b2_over_b3,
        "profile_attribution": _profile_attribution(indexed),
        "decision_inputs": {
            "b2_over_b3_core_geomean": core_speedup,
            "b2_over_b3_query_geomean": query_incremental,
            "b0_over_b3_query_geomean": query_cumulative,
            "worst_pair_b2_over_b3_core": worst_pair_core,
            "validated_core_gate": 1.15,
            "reduced_core_gate": 1.05,
            "cumulative_query_gate": 1.0,
            "pair_degradation_floor": 1.0 / 1.05,
        },
        "failure_rows": failures,
        "correctness_passed": not any("block-" in item for item in failures),
        "environment_passed": not any("environment" in item for item in failures),
        "measurement_auditable": not failures,
        "performance_claimed": False,
    }
    result["summary_hash"] = canonical_hash(result)
    return result


__all__ = [
    "derive_fsg4_b3_timing_evidence",
    "expected_fsg4_b3_sequence",
    "FSG4_B3_ACTIVATION_SCHEMA_VERSION",
    "FSG4_B3_CONFIG_ORDERS",
    "FSG4_B3_PROFILE_SPAN_LAYOUT",
    "FSG4_B3_REPEAT_COUNT",
    "FSG4_B3_TIMING_SCHEMA_VERSION",
    "FSG4B3ActivationReceipt",
    "FSG4B3ExecutionCounters",
    "FSG4B3TimingConfiguration",
    "FSG4B3TimingRun",
    "fsg4_b3_timing_run_from_dict",
]
