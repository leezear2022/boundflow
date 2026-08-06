"""Typed contracts for full-stack GPU attribution and cumulative ablation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Optional, Sequence, Tuple

FULL_STACK_GPU_ATTRIBUTION_SCHEMA_VERSION = "boundflow.full-stack-gpu-attribution/v1"


class StackLayer(str, Enum):
    """Exclusive primary owner of one critical-path segment."""

    SOLVER_CONTROL = "solver_control"
    ADAPTER_TRANSPORT = "adapter_transport"
    IR_GRAPH = "ir_graph"
    PLAN_SCHEDULE = "plan_schedule"
    BACKEND_COMPILE_JIT = "backend_compile_jit"
    OPERATOR_EXECUTION = "operator_execution"
    GRAPH_BOUNDARY = "graph_boundary"
    RUNTIME_SCHEDULE = "runtime_schedule"
    MEMORY_ALLOCATOR = "memory_allocator"
    UNCLASSIFIED_RESIDUAL = "unclassified_residual"


class SolverPhase(str, Enum):
    """Solver phase orthogonal to stack ownership."""

    SETUP = "setup"
    INITIAL_CROWN = "initial_crown"
    SELECTED_CROWN = "selected_crown"
    ALPHA_OPTIMIZE = "alpha_optimize"
    BETA_SPLIT = "beta_split"
    INTERSECT = "intersect"
    FORWARD_PROPAGATE = "forward_propagate"
    BRANCH_SCORE = "branch_score"
    QUEUE_COMMIT = "queue_commit"
    TERMINATION = "termination"
    UNCLASSIFIED = "unclassified"


class ResourceKind(str, Enum):
    """Physical resource on which a raw span was observed."""

    HOST_THREAD = "host_thread"
    CUDA_STREAM = "cuda_stream"
    CUDA_RUNTIME_API = "cuda_runtime_api"
    MEMORY = "memory"
    IPC = "ipc"


class CacheState(str, Enum):
    """Compile/cache state carried by every observed span."""

    COLD_COMPILE = "cold_compile"
    PROCESS_HIT = "process_hit"
    DISK_HIT = "disk_hit"
    WARM_EXECUTE = "warm_execute"
    NOT_APPLICABLE = "not_applicable"


class ReplacementMode(str, Enum):
    """How much of the external provider is actually replaced."""

    ORIGINAL_PROVIDER = "original_provider"
    RVIR_PASSTHROUGH = "rvir_passthrough"
    SHADOW_ONLY = "shadow_only"
    NESTED_REGION = "nested_region"
    WHOLE_CALL = "whole_call"


class FeatureKind(str, Enum):
    """Features whose physical activation must be evidenced, not inferred."""

    BOUND_GRAPH_IR = "bound_graph_ir"
    PLAN_SCHEDULE_IR = "plan_schedule_ir"
    PHYSICAL_BACKEND = "physical_backend"
    JIT_CACHE = "jit_cache"
    MULTI_STREAM = "multi_stream"
    STORAGE_PLAN = "storage_plan"
    BOUNDFLOW_REPLACEMENT = "boundflow_replacement"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: object) -> str:
    """Return one deterministic SHA256 over canonical JSON."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{label} must be a sequence")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields differ")


@dataclass(frozen=True)
class FullStackSpan:  # pylint: disable=too-many-instance-attributes
    """One raw host, CUDA, memory, or IPC activity interval."""

    span_id: str
    parent_span_id: Optional[str]
    layer: StackLayer
    phase: SolverPhase
    resource: ResourceKind
    cache_state: CacheState
    start_ns: int
    end_ns: int
    stream_id: Optional[str] = None
    dependency_span_ids: Tuple[str, ...] = ()

    def validate(self) -> None:
        """Reject malformed identity, timing, or stream ownership."""

        if not self.span_id:
            raise ValueError("full-stack span ID must be non-empty")
        if self.parent_span_id == self.span_id:
            raise ValueError("full-stack span cannot parent itself")
        if self.start_ns < 0 or self.end_ns < self.start_ns:
            raise ValueError("full-stack span interval is invalid")
        if len(set(self.dependency_span_ids)) != len(self.dependency_span_ids):
            raise ValueError("full-stack span dependencies duplicate")
        if self.span_id in self.dependency_span_ids:
            raise ValueError("full-stack span cannot depend on itself")
        if self.resource == ResourceKind.CUDA_STREAM and not self.stream_id:
            raise ValueError("CUDA span requires a stream identity")

    @property
    def duration_ns(self) -> int:
        """Return the raw interval duration."""

        return self.end_ns - self.start_ns

    def to_dict(self) -> dict[str, object]:
        """Return the stable raw-event representation."""

        self.validate()
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "layer": self.layer.value,
            "phase": self.phase.value,
            "resource": self.resource.value,
            "cache_state": self.cache_state.value,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "stream_id": self.stream_id,
            "dependency_span_ids": list(self.dependency_span_ids),
        }


@dataclass(frozen=True)
class CriticalPathSegment:
    """One exclusive segment after host/CUDA dependency reconstruction."""

    segment_id: str
    layer: StackLayer
    phase: SolverPhase
    start_ns: int
    end_ns: int
    source_span_ids: Tuple[str, ...]

    def validate(self) -> None:
        """Reject malformed segment identity, timing, or sources."""

        if not self.segment_id:
            raise ValueError("critical-path segment ID must be non-empty")
        if self.start_ns < 0 or self.end_ns <= self.start_ns:
            raise ValueError("critical-path segment interval is invalid")
        if len(set(self.source_span_ids)) != len(self.source_span_ids):
            raise ValueError("critical-path source spans duplicate")
        if not self.source_span_ids:
            raise ValueError("critical-path segment requires a source span")

    @property
    def duration_ns(self) -> int:
        """Return the exclusive critical-path duration."""

        return self.end_ns - self.start_ns

    def to_dict(self) -> dict[str, object]:
        """Return the stable exclusive-segment representation."""

        self.validate()
        return {
            "segment_id": self.segment_id,
            "layer": self.layer.value,
            "phase": self.phase.value,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "source_span_ids": list(self.source_span_ids),
        }


@dataclass(frozen=True)
class FeatureActivationLedger:  # pylint: disable=too-many-instance-attributes
    """Evidence that compiler/runtime features physically drive execution."""

    bound_graph_ir_drives_execution: bool
    plan_compiled_before_execution: bool
    task_schedule_drives_execution: bool
    backend_kind: str
    physical_backend_dispatches: int
    fallback_dispatches: int
    jit_cache_state: CacheState
    stream_count: int
    event_count: int
    wait_count: int
    storage_plan_enforced: bool
    replacement_mode: ReplacementMode

    def validate(self) -> None:
        """Reject impossible counters or absent backend identity."""

        if not self.backend_kind:
            raise ValueError("feature ledger backend kind must be non-empty")
        for name in (
            "physical_backend_dispatches",
            "fallback_dispatches",
            "stream_count",
            "event_count",
            "wait_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"feature ledger {name} must be non-negative")
        if self.stream_count == 0 and (self.event_count or self.wait_count):
            raise ValueError("stream events require at least one physical stream")

    def activated_features(self) -> Tuple[FeatureKind, ...]:
        """Return only features with direct activation evidence."""

        self.validate()
        found: list[FeatureKind] = []
        if self.bound_graph_ir_drives_execution:
            found.append(FeatureKind.BOUND_GRAPH_IR)
        if self.plan_compiled_before_execution and self.task_schedule_drives_execution:
            found.append(FeatureKind.PLAN_SCHEDULE_IR)
        if self.physical_backend_dispatches > 0:
            found.append(FeatureKind.PHYSICAL_BACKEND)
        if self.jit_cache_state != CacheState.NOT_APPLICABLE:
            found.append(FeatureKind.JIT_CACHE)
        if self.stream_count > 1 and self.event_count > 0:
            found.append(FeatureKind.MULTI_STREAM)
        if self.storage_plan_enforced:
            found.append(FeatureKind.STORAGE_PLAN)
        if self.replacement_mode in {
            ReplacementMode.NESTED_REGION,
            ReplacementMode.WHOLE_CALL,
        }:
            found.append(FeatureKind.BOUNDFLOW_REPLACEMENT)
        return tuple(found)

    def missing(self, required: Sequence[FeatureKind]) -> Tuple[FeatureKind, ...]:
        """Return claim-required features lacking physical evidence."""

        active = set(self.activated_features())
        return tuple(feature for feature in required if feature not in active)

    def to_dict(self) -> dict[str, object]:
        """Return evidence without inferring inactive features."""

        self.validate()
        return {
            "bound_graph_ir_drives_execution": self.bound_graph_ir_drives_execution,
            "plan_compiled_before_execution": self.plan_compiled_before_execution,
            "task_schedule_drives_execution": self.task_schedule_drives_execution,
            "backend_kind": self.backend_kind,
            "physical_backend_dispatches": self.physical_backend_dispatches,
            "fallback_dispatches": self.fallback_dispatches,
            "jit_cache_state": self.jit_cache_state.value,
            "stream_count": self.stream_count,
            "event_count": self.event_count,
            "wait_count": self.wait_count,
            "storage_plan_enforced": self.storage_plan_enforced,
            "replacement_mode": self.replacement_mode.value,
            "activated_features": [item.value for item in self.activated_features()],
        }


@dataclass(frozen=True)
class FullStackAttributionRun:  # pylint: disable=too-many-instance-attributes
    """One raw run with an independently reconstructed critical path."""

    run_id: str
    configuration_id: str
    scope_start_ns: int
    scope_end_ns: int
    spans: Tuple[FullStackSpan, ...]
    critical_path: Tuple[CriticalPathSegment, ...]
    features: FeatureActivationLedger
    performance_claimed: bool = False
    schema_version: str = FULL_STACK_GPU_ATTRIBUTION_SCHEMA_VERSION

    def validate(self) -> None:
        """Validate the raw forest, critical path, and feature evidence."""

        if self.schema_version != FULL_STACK_GPU_ATTRIBUTION_SCHEMA_VERSION:
            raise ValueError("full-stack attribution schema version differs")
        if not self.run_id or not self.configuration_id:
            raise ValueError("full-stack run identity must be non-empty")
        if self.scope_start_ns < 0 or self.scope_end_ns <= self.scope_start_ns:
            raise ValueError("full-stack run scope is invalid")
        if self.performance_claimed:
            raise ValueError("raw attribution run cannot claim performance")
        _validate_span_forest(
            self.spans,
            scope_start_ns=self.scope_start_ns,
            scope_end_ns=self.scope_end_ns,
        )
        _validate_critical_path(
            self.critical_path,
            span_ids={span.span_id for span in self.spans},
            scope_start_ns=self.scope_start_ns,
            scope_end_ns=self.scope_end_ns,
        )
        self.features.validate()

    def to_dict(self) -> dict[str, object]:
        """Return the complete stable raw-run representation."""

        self.validate()
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "configuration_id": self.configuration_id,
            "scope_start_ns": self.scope_start_ns,
            "scope_end_ns": self.scope_end_ns,
            "spans": [span.to_dict() for span in self.spans],
            "critical_path": [segment.to_dict() for segment in self.critical_path],
            "features": self.features.to_dict(),
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        """Bind raw timing, dependencies, and feature activation."""

        return canonical_hash(self.to_dict())


def full_stack_run_from_dict(value: Mapping[str, Any]) -> FullStackAttributionRun:
    """Parse one fail-closed raw-run payload into typed contracts."""

    _exact_keys(
        value,
        {
            "schema_version",
            "run_id",
            "configuration_id",
            "scope_start_ns",
            "scope_end_ns",
            "spans",
            "critical_path",
            "features",
            "performance_claimed",
        },
        "full-stack run",
    )
    spans = tuple(
        _span_from_dict(_mapping(item, "full-stack span"))
        for item in _sequence(value["spans"], "full-stack spans")
    )
    critical_path = tuple(
        _critical_segment_from_dict(_mapping(item, "critical-path segment"))
        for item in _sequence(value["critical_path"], "critical path")
    )
    run = FullStackAttributionRun(
        schema_version=str(value["schema_version"]),
        run_id=str(value["run_id"]),
        configuration_id=str(value["configuration_id"]),
        scope_start_ns=int(value["scope_start_ns"]),
        scope_end_ns=int(value["scope_end_ns"]),
        spans=spans,
        critical_path=critical_path,
        features=_features_from_dict(_mapping(value["features"], "feature ledger")),
        performance_claimed=bool(value["performance_claimed"]),
    )
    run.validate()
    if run.to_dict() != dict(value):
        raise ValueError("full-stack run canonical payload differs")
    return run


def _span_from_dict(value: Mapping[str, Any]) -> FullStackSpan:
    _exact_keys(
        value,
        {
            "span_id",
            "parent_span_id",
            "layer",
            "phase",
            "resource",
            "cache_state",
            "start_ns",
            "end_ns",
            "stream_id",
            "dependency_span_ids",
        },
        "full-stack span",
    )
    parent = value["parent_span_id"]
    stream = value["stream_id"]
    return FullStackSpan(
        span_id=str(value["span_id"]),
        parent_span_id=None if parent is None else str(parent),
        layer=StackLayer(str(value["layer"])),
        phase=SolverPhase(str(value["phase"])),
        resource=ResourceKind(str(value["resource"])),
        cache_state=CacheState(str(value["cache_state"])),
        start_ns=int(value["start_ns"]),
        end_ns=int(value["end_ns"]),
        stream_id=None if stream is None else str(stream),
        dependency_span_ids=tuple(
            str(item)
            for item in _sequence(value["dependency_span_ids"], "span dependencies")
        ),
    )


def _critical_segment_from_dict(value: Mapping[str, Any]) -> CriticalPathSegment:
    _exact_keys(
        value,
        {
            "segment_id",
            "layer",
            "phase",
            "start_ns",
            "end_ns",
            "source_span_ids",
        },
        "critical-path segment",
    )
    return CriticalPathSegment(
        segment_id=str(value["segment_id"]),
        layer=StackLayer(str(value["layer"])),
        phase=SolverPhase(str(value["phase"])),
        start_ns=int(value["start_ns"]),
        end_ns=int(value["end_ns"]),
        source_span_ids=tuple(
            str(item)
            for item in _sequence(value["source_span_ids"], "critical sources")
        ),
    )


def _features_from_dict(value: Mapping[str, Any]) -> FeatureActivationLedger:
    _exact_keys(
        value,
        {
            "bound_graph_ir_drives_execution",
            "plan_compiled_before_execution",
            "task_schedule_drives_execution",
            "backend_kind",
            "physical_backend_dispatches",
            "fallback_dispatches",
            "jit_cache_state",
            "stream_count",
            "event_count",
            "wait_count",
            "storage_plan_enforced",
            "replacement_mode",
            "activated_features",
        },
        "feature ledger",
    )
    ledger = FeatureActivationLedger(
        bound_graph_ir_drives_execution=bool(value["bound_graph_ir_drives_execution"]),
        plan_compiled_before_execution=bool(value["plan_compiled_before_execution"]),
        task_schedule_drives_execution=bool(value["task_schedule_drives_execution"]),
        backend_kind=str(value["backend_kind"]),
        physical_backend_dispatches=int(value["physical_backend_dispatches"]),
        fallback_dispatches=int(value["fallback_dispatches"]),
        jit_cache_state=CacheState(str(value["jit_cache_state"])),
        stream_count=int(value["stream_count"]),
        event_count=int(value["event_count"]),
        wait_count=int(value["wait_count"]),
        storage_plan_enforced=bool(value["storage_plan_enforced"]),
        replacement_mode=ReplacementMode(str(value["replacement_mode"])),
    )
    if [item.value for item in ledger.activated_features()] != list(
        _sequence(value["activated_features"], "activated features")
    ):
        raise ValueError("feature activation projection differs")
    return ledger


def _validate_span_forest(
    spans: Sequence[FullStackSpan], *, scope_start_ns: int, scope_end_ns: int
) -> None:
    by_id: dict[str, FullStackSpan] = {}
    for span in spans:
        span.validate()
        if span.span_id in by_id:
            raise ValueError("full-stack span ID duplicates")
        if span.start_ns < scope_start_ns or span.end_ns > scope_end_ns:
            raise ValueError("full-stack span escapes run scope")
        by_id[span.span_id] = span
    for span in spans:
        if span.parent_span_id is not None:
            parent = by_id.get(span.parent_span_id)
            if parent is None:
                raise ValueError("full-stack span parent is missing")
            if parent.start_ns > span.start_ns or parent.end_ns < span.end_ns:
                raise ValueError("full-stack parent does not contain child")
        for dependency_id in span.dependency_span_ids:
            dependency = by_id.get(dependency_id)
            if dependency is None:
                raise ValueError("full-stack span dependency is missing")
            if dependency.end_ns > span.start_ns:
                raise ValueError("full-stack dependency ends after dependent starts")
    _reject_cycles(
        {
            span.span_id: (
                () if span.parent_span_id is None else (span.parent_span_id,)
            )
            for span in spans
        },
        "full-stack parent",
    )
    _reject_cycles(
        {span.span_id: span.dependency_span_ids for span in spans},
        "full-stack dependency",
    )


def _reject_cycles(edges: Mapping[str, Sequence[str]], label: str) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"{label} graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for adjacent in edges[node]:
            visit(adjacent)
        visiting.remove(node)
        visited.add(node)

    for node in edges:
        visit(node)


def _validate_critical_path(
    segments: Sequence[CriticalPathSegment],
    *,
    span_ids: set[str],
    scope_start_ns: int,
    scope_end_ns: int,
) -> None:
    segment_ids: set[str] = set()
    previous_end = scope_start_ns
    for segment in sorted(segments, key=lambda item: (item.start_ns, item.end_ns)):
        segment.validate()
        if segment.segment_id in segment_ids:
            raise ValueError("critical-path segment ID duplicates")
        if segment.start_ns < scope_start_ns or segment.end_ns > scope_end_ns:
            raise ValueError("critical-path segment escapes run scope")
        if segment.start_ns < previous_end:
            raise ValueError("critical-path segments overlap")
        if any(source not in span_ids for source in segment.source_span_ids):
            raise ValueError("critical-path source span is missing")
        segment_ids.add(segment.segment_id)
        previous_end = segment.end_ns


def interval_union_ns(intervals: Sequence[tuple[int, int]]) -> int:
    """Return interval union duration without double-counting overlap."""

    total = 0
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    for start, end in sorted(intervals):
        if start < 0 or end < start:
            raise ValueError("attribution interval is invalid")
        if current_start is None or current_end is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    if current_start is not None and current_end is not None:
        total += current_end - current_start
    return total


def summarize_run(
    run: FullStackAttributionRun,
    *,
    maximum_closure_error: float = 0.01,
    maximum_residual_share: float = 0.03,
) -> dict[str, object]:
    """Build mutually exclusive critical-path and asynchronous GPU summaries."""

    run.validate()
    if not 0.0 <= maximum_closure_error < 1.0:
        raise ValueError("maximum closure error must be in [0, 1)")
    if not 0.0 <= maximum_residual_share <= 1.0:
        raise ValueError("maximum residual share must be in [0, 1]")
    scope_ns = run.scope_end_ns - run.scope_start_ns
    critical_ns = sum(segment.duration_ns for segment in run.critical_path)
    closure_error = abs(scope_ns - critical_ns) / scope_ns
    layer_ns = {layer.value: 0 for layer in StackLayer}
    phase_ns = {phase.value: 0 for phase in SolverPhase}
    for segment in run.critical_path:
        layer_ns[segment.layer.value] += segment.duration_ns
        phase_ns[segment.phase.value] += segment.duration_ns
    residual_share = layer_ns[StackLayer.UNCLASSIFIED_RESIDUAL.value] / scope_ns
    residual_passed = residual_share <= maximum_residual_share
    cuda_spans = [
        span for span in run.spans if span.resource == ResourceKind.CUDA_STREAM
    ]
    gpu_sum_ns = sum(span.duration_ns for span in cuda_spans)
    gpu_union_ns = interval_union_ns(
        [(span.start_ns, span.end_ns) for span in cuda_spans]
    )
    summary: dict[str, object] = {
        "schema_version": FULL_STACK_GPU_ATTRIBUTION_SCHEMA_VERSION,
        "run_id": run.run_id,
        "configuration_id": run.configuration_id,
        "scope_ns": scope_ns,
        "critical_path_ns": critical_ns,
        "closure_error": closure_error,
        "closure_passed": closure_error <= maximum_closure_error,
        "residual_share": residual_share,
        "residual_passed": residual_passed,
        "attribution_passed": closure_error <= maximum_closure_error
        and residual_passed,
        "layer_ns": layer_ns,
        "layer_share": {key: value / scope_ns for key, value in layer_ns.items()},
        "phase_ns": phase_ns,
        "gpu_union_ns": gpu_union_ns,
        "gpu_sum_ns": gpu_sum_ns,
        "gpu_overlap_ns": gpu_sum_ns - gpu_union_ns,
        "features": run.features.to_dict(),
        "run_hash": run.stable_hash(),
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


def deletion_only_ceiling(share: float) -> Optional[float]:
    """Return the Amdahl ceiling if exactly one baseline region becomes free."""

    if not 0.0 <= share <= 1.0:
        raise ValueError("Amdahl share must be in [0, 1]")
    denominator = 1.0 - share
    return None if denominator == 0.0 else 1.0 / denominator


def joint_amdahl_speedup(
    baseline_shares: Mapping[str, float],
    region_speedups: Mapping[str, Optional[float]],
) -> Optional[float]:
    """Project joint speedup from baseline shares; None means an infinite region."""

    if set(region_speedups) - set(baseline_shares):
        raise ValueError("Amdahl speedup refers to an unknown baseline region")
    total_share = sum(baseline_shares.values())
    if any(not 0.0 <= share <= 1.0 for share in baseline_shares.values()):
        raise ValueError("Amdahl baseline shares must be in [0, 1]")
    if total_share > 1.0 + 1e-12:
        raise ValueError("Amdahl baseline shares exceed one")
    denominator = max(0.0, 1.0 - total_share)
    for region, share in baseline_shares.items():
        speedup = region_speedups.get(region, 1.0)
        if speedup is None:
            continue
        if speedup <= 0.0:
            raise ValueError("Amdahl region speedup must be positive")
        denominator += share / speedup
    return None if denominator == 0.0 else 1.0 / denominator


def summarize_cumulative_ablation(
    wall_ns_by_configuration: Mapping[str, int],
    ordered_configurations: Sequence[str],
    *,
    leave_one_out_wall_ns: Mapping[str, int],
) -> dict[str, object]:
    """Report cumulative, incremental, leave-one-out, and interaction effects."""

    if len(ordered_configurations) < 2 or len(set(ordered_configurations)) != len(
        ordered_configurations
    ):
        raise ValueError("ablation order must contain distinct baseline and candidate")
    if set(ordered_configurations) != set(wall_ns_by_configuration):
        raise ValueError("ablation timing coverage differs from order")
    if any(value <= 0 for value in wall_ns_by_configuration.values()):
        raise ValueError("ablation wall time must be positive")
    baseline_id = ordered_configurations[0]
    full_id = ordered_configurations[-1]
    baseline_ns = wall_ns_by_configuration[baseline_id]
    full_ns = wall_ns_by_configuration[full_id]
    cumulative = {
        config: baseline_ns / wall_ns_by_configuration[config]
        for config in ordered_configurations
    }
    incremental = {
        current: wall_ns_by_configuration[previous] / wall_ns_by_configuration[current]
        for previous, current in zip(ordered_configurations, ordered_configurations[1:])
    }
    if any(value <= 0 for value in leave_one_out_wall_ns.values()):
        raise ValueError("leave-one-out wall time must be positive")
    penalties = {
        feature: value - full_ns for feature, value in leave_one_out_wall_ns.items()
    }
    total_savings = baseline_ns - full_ns
    interaction_residual = total_savings - sum(penalties.values())
    result: dict[str, object] = {
        "baseline_configuration": baseline_id,
        "full_configuration": full_id,
        "ordered_configurations": list(ordered_configurations),
        "cumulative_speedup": cumulative,
        "incremental_speedup": incremental,
        "leave_one_out_penalty_ns": penalties,
        "total_savings_ns": total_savings,
        "interaction_residual_ns": interaction_residual,
        "performance_claimed": False,
    }
    result["summary_hash"] = canonical_hash(result)
    return result
