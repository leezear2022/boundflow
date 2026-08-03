"""First-class Schedule IR v1 schema, verifier, and reference lowering."""

# Cross-layer schedule verification intentionally keeps the execution ledger together.
# pylint: disable=too-many-branches,too-many-instance-attributes,too-many-lines,too-many-locals,too-many-statements,missing-class-docstring,missing-function-docstring,duplicate-code

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Optional, Tuple

from .bound import BFBoundModule, BoundRepresentation
from .plan import (
    BackendCandidate,
    MaterializationCandidate,
    PlanInstance,
    PlanTemplate,
    RegionCandidate,
    StateAction,
    StateCandidate,
    StorageBinding,
)

SCHEDULE_IR_SCHEMA_VERSION = "boundflow.schedule_ir/v1.0"


class ScheduleActionKind(Enum):
    """Closed action set implemented by the IR-3A synchronous reference path."""

    CHECK_BUDGET = "check_budget"
    ALLOCATE = "allocate"
    MATERIALIZE = "materialize"
    TRANSFER = "transfer"
    LAUNCH = "launch"
    BATCH_LOOP = "batch_loop"
    RECORD_EVENT = "record_event"
    WAIT_EVENT = "wait_event"
    STATE_LOAD = "state_load"
    STATE_STORE = "state_store"
    STATE_INVALIDATE = "state_invalidate"
    FALLBACK = "fallback"
    RETRY = "retry"
    REQUEST_REPLAN = "request_replan"
    EMIT_RESULT = "emit_result"
    FREE = "free"


@dataclass(frozen=True)
class ScheduleBuffer:
    """One Plan IR storage binding carried into the execution schedule."""

    value_id: str
    arena_id: str
    offset_bytes: int
    logical_size_bytes: int
    size_bytes: int
    representation: BoundRepresentation
    live_from_op_id: str
    live_to_op_id: str

    @classmethod
    def from_storage_binding(cls, binding: StorageBinding) -> "ScheduleBuffer":
        return cls(**asdict(binding))

    def validate(self) -> None:
        StorageBinding(**asdict(self)).validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["representation"] = self.representation.value
        return payload


@dataclass(frozen=True)
class CheckBudgetAction:
    action_id: str
    required_peak_bytes: int
    kind: ScheduleActionKind = ScheduleActionKind.CHECK_BUDGET

    def validate(self) -> None:
        if not self.action_id or self.required_peak_bytes <= 0:
            raise ValueError("schedule budget action is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "required_peak_bytes": self.required_peak_bytes,
        }


@dataclass(frozen=True)
class AllocateAction:
    action_id: str
    arena_id: str
    size_bytes: int
    kind: ScheduleActionKind = ScheduleActionKind.ALLOCATE

    def validate(self) -> None:
        if not self.action_id or not self.arena_id or self.size_bytes <= 0:
            raise ValueError("schedule allocate action is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "arena_id": self.arena_id,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class MaterializeAction:
    action_id: str
    transition_candidate_id: str
    source_value_id: str
    before_op_id: str
    source_representation: BoundRepresentation
    target_representation: BoundRepresentation
    kind: ScheduleActionKind = ScheduleActionKind.MATERIALIZE

    def validate(self) -> None:
        if any(
            not value
            for value in (
                self.action_id,
                self.transition_candidate_id,
                self.source_value_id,
                self.before_op_id,
            )
        ):
            raise ValueError("schedule materialize action IDs are invalid")
        if self.source_representation == self.target_representation:
            raise ValueError(
                "schedule materialize action does not change representation"
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "transition_candidate_id": self.transition_candidate_id,
            "source_value_id": self.source_value_id,
            "before_op_id": self.before_op_id,
            "source_representation": self.source_representation.value,
            "target_representation": self.target_representation.value,
        }


class TransferDirection(Enum):
    """Explicit host/device transfer direction."""

    HOST_TO_DEVICE = "host_to_device"
    DEVICE_TO_HOST = "device_to_host"
    DEVICE_TO_DEVICE = "device_to_device"


@dataclass(frozen=True)
class TransferAction:
    """Move one logical value without changing its Bound semantics."""

    action_id: str
    value_id: str
    source_device: str
    target_device: str
    direction: TransferDirection
    stream_id: str
    kind: ScheduleActionKind = ScheduleActionKind.TRANSFER

    def validate(self) -> None:
        if any(
            not value
            for value in (
                self.action_id,
                self.value_id,
                self.source_device,
                self.target_device,
                self.stream_id,
            )
        ):
            raise ValueError("schedule transfer action is incomplete")
        if self.source_device == self.target_device:
            raise ValueError("schedule transfer must change physical device")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "value_id": self.value_id,
            "source_device": self.source_device,
            "target_device": self.target_device,
            "direction": self.direction.value,
            "stream_id": self.stream_id,
        }


@dataclass(frozen=True)
class LaunchAction:
    action_id: str
    task_id: str
    region_id: str
    backend_candidate_id: str
    backend_artifact_key: Optional[str]
    input_value_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]
    stream_id: str
    kind: ScheduleActionKind = ScheduleActionKind.LAUNCH

    def validate(self) -> None:
        for name in (
            "action_id",
            "task_id",
            "region_id",
            "backend_candidate_id",
            "stream_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"schedule launch {name} must be non-empty")
        if not self.input_value_ids or not self.output_value_ids:
            raise ValueError("schedule launch requires input/output values")
        if len(self.input_value_ids) != len(set(self.input_value_ids)):
            raise ValueError("schedule launch contains duplicate inputs")
        if len(self.output_value_ids) != len(set(self.output_value_ids)):
            raise ValueError("schedule launch contains duplicate outputs")
        if self.backend_artifact_key is not None and not self.backend_artifact_key:
            raise ValueError("schedule launch artifact key is empty")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "region_id": self.region_id,
            "backend_candidate_id": self.backend_candidate_id,
            "backend_artifact_key": self.backend_artifact_key,
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
            "stream_id": self.stream_id,
        }


@dataclass(frozen=True)
class EmitResultAction:
    action_id: str
    query_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]
    kind: ScheduleActionKind = ScheduleActionKind.EMIT_RESULT

    def validate(self) -> None:
        if not self.action_id or not self.query_ids or not self.output_value_ids:
            raise ValueError("schedule emit-result action is incomplete")
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("schedule emit-result duplicates query IDs")
        if len(self.output_value_ids) != len(set(self.output_value_ids)):
            raise ValueError("schedule emit-result duplicates output values")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "query_ids": list(self.query_ids),
            "output_value_ids": list(self.output_value_ids),
        }


@dataclass(frozen=True)
class FreeAction:
    action_id: str
    arena_id: str
    kind: ScheduleActionKind = ScheduleActionKind.FREE

    def validate(self) -> None:
        if not self.action_id or not self.arena_id:
            raise ValueError("schedule free action is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "arena_id": self.arena_id,
        }


@dataclass(frozen=True)
class QueryBatchSlice:
    """One ordered, non-overlapping query slice in a BatchLoop."""

    slice_id: str
    query_ids: Tuple[str, ...]
    start_index: Optional[int] = None
    stop_index: Optional[int] = None

    def validate(self) -> None:
        if not self.slice_id or not self.query_ids:
            raise ValueError("schedule batch slice is incomplete")
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("schedule batch slice duplicates query IDs")
        if (self.start_index is None) != (self.stop_index is None):
            raise ValueError("schedule batch slice range is partially specified")
        if self.start_index is not None:
            if self.stop_index is None:
                raise AssertionError("validated slice range unexpectedly lacks stop")
            if self.start_index < 0 or self.stop_index <= self.start_index:
                raise ValueError("schedule batch slice range is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "slice_id": self.slice_id,
            "query_ids": list(self.query_ids),
        }
        if self.start_index is not None:
            payload["start_index"] = self.start_index
            payload["stop_index"] = self.stop_index
        return payload


@dataclass(frozen=True)
class BatchLoopAction:
    """Explicit ordered query partition for one batching axis."""

    action_id: str
    axis: str
    slices: Tuple[QueryBatchSlice, ...]
    kind: ScheduleActionKind = ScheduleActionKind.BATCH_LOOP

    def validate(self) -> None:
        if not self.action_id or self.axis not in {"domain", "spec", "sample"}:
            raise ValueError("schedule batch-loop identity/axis is invalid")
        if not self.slices:
            raise ValueError("schedule batch-loop requires slices")
        for item in self.slices:
            item.validate()
        if len({item.slice_id for item in self.slices}) != len(self.slices):
            raise ValueError("schedule batch-loop slice IDs must be unique")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "axis": self.axis,
            "slices": [item.to_dict() for item in self.slices],
        }


@dataclass(frozen=True)
class RecordEventAction:
    """Record one stream event."""

    action_id: str
    event_id: str
    stream_id: str
    kind: ScheduleActionKind = ScheduleActionKind.RECORD_EVENT

    def validate(self) -> None:
        if not self.action_id or not self.event_id or not self.stream_id:
            raise ValueError("schedule record-event action is incomplete")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "event_id": self.event_id,
            "stream_id": self.stream_id,
        }


@dataclass(frozen=True)
class WaitEventAction:
    """Wait for one previously recorded stream event."""

    action_id: str
    event_id: str
    stream_id: str
    kind: ScheduleActionKind = ScheduleActionKind.WAIT_EVENT

    def validate(self) -> None:
        if not self.action_id or not self.event_id or not self.stream_id:
            raise ValueError("schedule wait-event action is incomplete")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "event_id": self.event_id,
            "stream_id": self.stream_id,
        }


@dataclass(frozen=True)
class StateLoadAction:
    """Load one exact-valid cached state."""

    action_id: str
    state_id: str
    source_value_id: str
    state_version: str
    kind: ScheduleActionKind = ScheduleActionKind.STATE_LOAD

    def validate(self) -> None:
        if any(
            not value
            for value in (
                self.action_id,
                self.state_id,
                self.source_value_id,
                self.state_version,
            )
        ):
            raise ValueError("schedule state-load action is incomplete")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "state_id": self.state_id,
            "source_value_id": self.source_value_id,
            "state_version": self.state_version,
        }


@dataclass(frozen=True)
class StateStoreAction:
    """Store one versioned state after execution."""

    action_id: str
    state_id: str
    source_value_id: str
    state_version: str
    kind: ScheduleActionKind = ScheduleActionKind.STATE_STORE

    def validate(self) -> None:
        if any(
            not value
            for value in (
                self.action_id,
                self.state_id,
                self.source_value_id,
                self.state_version,
            )
        ):
            raise ValueError("schedule state-store action is incomplete")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "state_id": self.state_id,
            "source_value_id": self.source_value_id,
            "state_version": self.state_version,
        }


@dataclass(frozen=True)
class StateInvalidateAction:
    """Invalidate one state with an explicit reason."""

    action_id: str
    state_id: str
    reason: str
    kind: ScheduleActionKind = ScheduleActionKind.STATE_INVALIDATE

    def validate(self) -> None:
        if not self.action_id or not self.state_id or not self.reason:
            raise ValueError("schedule state-invalidate action is incomplete")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "state_id": self.state_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FallbackAction:
    """One declared backend fallback for a retry policy."""

    action_id: str
    retry_action_id: str
    backend_candidate_id: str
    reason: str
    kind: ScheduleActionKind = ScheduleActionKind.FALLBACK

    def validate(self) -> None:
        if any(
            not value
            for value in (
                self.action_id,
                self.retry_action_id,
                self.backend_candidate_id,
                self.reason,
            )
        ):
            raise ValueError("schedule fallback action is incomplete")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "retry_action_id": self.retry_action_id,
            "backend_candidate_id": self.backend_candidate_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RetryAction:
    """Bounded retry policy attached to one launch."""

    action_id: str
    launch_action_id: str
    fallback_action_ids: Tuple[str, ...]
    max_attempts: int
    retry_on: Tuple[str, ...]
    kind: ScheduleActionKind = ScheduleActionKind.RETRY

    def validate(self) -> None:
        if not self.action_id or not self.launch_action_id:
            raise ValueError("schedule retry identity is incomplete")
        if self.max_attempts <= 0:
            raise ValueError("schedule retry max_attempts must be positive")
        if self.max_attempts != 1 + len(self.fallback_action_ids):
            raise ValueError("schedule retry attempts/fallback ladder mismatch")
        if not self.retry_on or any(reason != "oom" for reason in self.retry_on):
            raise ValueError("Schedule IR v1 retry only supports explicit OOM")
        if len(self.fallback_action_ids) != len(set(self.fallback_action_ids)):
            raise ValueError("schedule retry duplicates fallback actions")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "launch_action_id": self.launch_action_id,
            "fallback_action_ids": list(self.fallback_action_ids),
            "max_attempts": self.max_attempts,
            "retry_on": list(self.retry_on),
        }


@dataclass(frozen=True)
class RequestReplanAction:
    """Constrained runtime re-plan request that cannot alter Bound semantics."""

    action_id: str
    reason: str
    preserved_bound_module_hash: str
    allowed_decision_axes: Tuple[str, ...]
    kind: ScheduleActionKind = ScheduleActionKind.REQUEST_REPLAN

    def validate(self) -> None:
        if (
            not self.action_id
            or not self.reason
            or len(self.preserved_bound_module_hash) != 64
        ):
            raise ValueError("schedule replan request is incomplete")
        allowed = {"backend", "representation", "batch", "storage", "state"}
        if not self.allowed_decision_axes or not set(
            self.allowed_decision_axes
        ).issubset(allowed):
            raise ValueError("schedule replan request changes a forbidden axis")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind.value,
            "action_id": self.action_id,
            "reason": self.reason,
            "preserved_bound_module_hash": self.preserved_bound_module_hash,
            "allowed_decision_axes": list(self.allowed_decision_axes),
        }


ScheduleAction = (
    CheckBudgetAction
    | AllocateAction
    | MaterializeAction
    | TransferAction
    | LaunchAction
    | BatchLoopAction
    | RecordEventAction
    | WaitEventAction
    | StateLoadAction
    | StateStoreAction
    | StateInvalidateAction
    | FallbackAction
    | RetryAction
    | RequestReplanAction
    | EmitResultAction
    | FreeAction
)


@dataclass(frozen=True)
class ScheduleModule:  # pylint: disable=too-many-instance-attributes
    """One concrete synchronous execution schedule for a PlanInstance."""

    schedule_id: str
    bound_module_hash: str
    plan_template_hash: str
    plan_instance_hash: str
    query_ids: Tuple[str, ...]
    buffers: Tuple[ScheduleBuffer, ...]
    actions: Tuple[ScheduleAction, ...]
    schema_version: str = SCHEDULE_IR_SCHEMA_VERSION

    def validate(
        self,
        *,
        bound_module: BFBoundModule,
        template: PlanTemplate,
        instance: PlanInstance,
    ) -> None:
        if self.schema_version != SCHEDULE_IR_SCHEMA_VERSION:
            raise ValueError(f"unsupported Schedule IR schema: {self.schema_version}")
        if not self.schedule_id or not self.query_ids:
            raise ValueError("Schedule IR identity/query IDs are incomplete")
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("Schedule IR query IDs must be unique")
        instance.validate(template=template, bound_module=bound_module)
        expected_hashes = (
            bound_module.stable_hash(),
            template.stable_hash(bound_module=bound_module),
            instance.stable_hash(template=template, bound_module=bound_module),
        )
        if (
            self.bound_module_hash,
            self.plan_template_hash,
            self.plan_instance_hash,
        ) != expected_hashes:
            raise ValueError("Schedule IR input hashes do not match typed inputs")
        if not self.buffers or not self.actions:
            raise ValueError("Schedule IR requires buffers and actions")
        for buffer in self.buffers:
            buffer.validate()
        for action in self.actions:
            action.validate()
        if len({buffer.value_id for buffer in self.buffers}) != len(self.buffers):
            raise ValueError("Schedule IR contains duplicate value buffers")
        action_ids = tuple(action.action_id for action in self.actions)
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Schedule IR action IDs must be unique")

        storage = template.candidate_map()[instance.storage_decision.candidate_id]
        expected_buffers = tuple(
            ScheduleBuffer.from_storage_binding(binding)
            for binding in storage.bindings  # type: ignore[union-attr]
        )
        if self.buffers != expected_buffers:
            raise ValueError("Schedule IR buffers differ from PlanInstance storage")
        arena_sizes: dict[str, int] = {}
        for buffer in self.buffers:
            arena_sizes[buffer.arena_id] = max(
                arena_sizes.get(buffer.arena_id, 0),
                buffer.offset_bytes + buffer.size_bytes,
            )
        expected_peak = sum(arena_sizes.values())
        if expected_peak != instance.cost_summary.predicted_peak_bytes:
            raise ValueError("Schedule IR arena ledger differs from PlanInstance peak")
        selected_batch = next(
            candidate
            for candidate in template.batch_candidates
            if candidate.candidate_id == instance.batch_decision.candidate_id
        )

        selected_regions = _selected_regions(template, instance)
        backend_by_region = _selected_backends(template, instance)
        selected_transitions = _selected_transitions(template, instance)
        selected_states = _selected_states(template, instance)
        reused_value_ids = {
            candidate.source_value_id
            for candidate in selected_states.values()
            if candidate.action == StateAction.REUSE
        }
        state_reused_region_ids = {
            region.region_id
            for region in selected_regions
            if set(region.output_value_ids).issubset(reused_value_ids)
        }
        required_transition_ids = {
            candidate_id
            for representation in template.representation_candidates
            if representation.region_id not in state_reused_region_ids
            and representation.candidate_id
            in {decision.candidate_id for decision in instance.representation_decisions}
            for candidate_id in representation.required_transition_candidate_ids
        }
        region_by_op = {
            op_id: region for region in selected_regions for op_id in region.op_ids
        }
        action_by_id = {action.action_id: action for action in self.actions}
        fallback_actions = {
            action.action_id: action
            for action in self.actions
            if isinstance(action, FallbackAction)
        }
        retry_actions = tuple(
            action for action in self.actions if isinstance(action, RetryAction)
        )
        backend_candidates = {
            candidate.candidate_id: candidate
            for candidate in template.backend_candidates
        }
        selected_representation_by_region = {
            decision.region_id: decision.candidate_id
            for decision in instance.representation_decisions
        }
        for retry in retry_actions:
            launch = action_by_id.get(retry.launch_action_id)
            if not isinstance(launch, LaunchAction):
                raise ValueError("Schedule IR retry references a non-launch action")
            if self.actions.index(retry) > self.actions.index(launch):
                raise ValueError("Schedule IR retry policy must precede its launch")
            fallback_backend_ids: list[str] = []
            for fallback_id in retry.fallback_action_ids:
                fallback = fallback_actions.get(fallback_id)
                if fallback is None or fallback.retry_action_id != retry.action_id:
                    raise ValueError("Schedule IR retry/fallback references mismatch")
                backend = backend_candidates.get(fallback.backend_candidate_id)
                if (
                    backend is None
                    or not backend.static_legal
                    or backend.region_id != launch.region_id
                    or selected_representation_by_region[launch.region_id]
                    not in backend.compatible_representation_candidate_ids
                ):
                    raise ValueError(
                        "Schedule IR fallback backend is illegal or wrong-region"
                    )
                fallback_backend_ids.append(backend.candidate_id)
            if (
                len({launch.backend_candidate_id, *fallback_backend_ids})
                != retry.max_attempts
            ):
                raise ValueError("Schedule IR retry ladder repeats a backend")
        referenced_fallbacks = {
            fallback_id
            for retry in retry_actions
            for fallback_id in retry.fallback_action_ids
        }
        if referenced_fallbacks != set(fallback_actions):
            raise ValueError("Schedule IR contains orphan fallback actions")

        allocated: set[str] = set()
        available_values = set(bound_module.graph.inputs)
        value_stream = {value_id: "external" for value_id in available_values}
        launched_regions: set[str] = set()
        performed_transitions: set[str] = set()
        recorded_events: dict[str, str] = {}
        waited_streams: dict[str, set[str]] = {"sync": {"external", "sync"}}
        batch_accounted = False
        state_actions: dict[str, ScheduleActionKind] = {}
        emitted = False
        checked_budget = False
        live_bytes = 0
        peak_bytes = 0
        for action in self.actions:
            if isinstance(action, CheckBudgetAction):
                if checked_budget or allocated or launched_regions:
                    raise ValueError(
                        "Schedule IR budget check must occur once before work"
                    )
                if action.required_peak_bytes != expected_peak:
                    raise ValueError("Schedule IR budget check uses the wrong peak")
                if action.required_peak_bytes > min(
                    instance.available_memory_bytes,
                    instance.memory_budget_bytes,
                    template.hardware.total_memory_bytes,
                ):
                    raise ValueError("Schedule IR budget check exceeds plan budget")
                checked_budget = True
            elif isinstance(action, AllocateAction):
                if not checked_budget or action.arena_id in allocated:
                    raise ValueError("Schedule IR arena allocation order is invalid")
                if arena_sizes.get(action.arena_id) != action.size_bytes:
                    raise ValueError("Schedule IR allocates the wrong arena size")
                allocated.add(action.arena_id)
                live_bytes += action.size_bytes
                peak_bytes = max(peak_bytes, live_bytes)
            elif isinstance(action, BatchLoopAction):
                if batch_accounted or launched_regions:
                    raise ValueError(
                        "Schedule IR batch-loop must occur once before launches"
                    )
                if action.axis == "spec":
                    if (
                        selected_batch.spec_batch_size
                        >= template.workload.spec_batch_size
                        or selected_batch.domain_batch_size
                        != template.workload.domain_batch_size
                        or selected_batch.sample_batch_size
                        != template.workload.sample_batch_size
                    ):
                        raise ValueError(
                            "Schedule IR spec-loop differs from selected batch axes"
                        )
                    expected_start = 0
                    for item in action.slices:
                        if (
                            item.query_ids != self.query_ids
                            or item.start_index != expected_start
                            or item.stop_index is None
                            or item.stop_index - expected_start
                            > selected_batch.spec_batch_size
                        ):
                            raise ValueError(
                                "Schedule IR spec-loop loses or overlaps objective slices"
                            )
                        expected_start = item.stop_index
                    if expected_start != template.workload.spec_batch_size:
                        raise ValueError(
                            "Schedule IR spec-loop does not cover every objective"
                        )
                else:
                    flattened = tuple(
                        query_id
                        for item in action.slices
                        for query_id in item.query_ids
                    )
                    if flattened != self.query_ids:
                        raise ValueError(
                            "Schedule IR batch-loop loses, duplicates, or reorders queries"
                        )
                    if any(
                        item.start_index is not None or item.stop_index is not None
                        for item in action.slices
                    ):
                        raise ValueError(
                            "Schedule IR query-loop unexpectedly carries index slices"
                        )
                batch_accounted = True
            elif isinstance(action, RecordEventAction):
                if action.event_id in recorded_events:
                    raise ValueError("Schedule IR records one event more than once")
                recorded_events[action.event_id] = action.stream_id
            elif isinstance(action, WaitEventAction):
                source_stream = recorded_events.get(action.event_id)
                if source_stream is None:
                    raise ValueError("Schedule IR waits before event record")
                waited_streams.setdefault(
                    action.stream_id, {"external", action.stream_id}
                ).add(source_stream)
            elif isinstance(action, StateLoadAction):
                candidate = selected_states.get(action.state_id)
                if (
                    candidate is None
                    or candidate.action != StateAction.REUSE
                    or action.source_value_id != candidate.source_value_id
                    or action.state_version != candidate.state_version
                    or launched_regions
                ):
                    raise ValueError("Schedule IR state load/Plan decision mismatch")
                if action.state_id in state_actions:
                    raise ValueError("Schedule IR duplicates a state action")
                state_actions[action.state_id] = action.kind
                available_values.add(action.source_value_id)
                value_stream[action.source_value_id] = "external"
            elif isinstance(action, StateStoreAction):
                candidate = selected_states.get(action.state_id)
                if (
                    candidate is None
                    or candidate.action != StateAction.CACHE
                    or action.source_value_id != candidate.source_value_id
                    or action.state_version != candidate.state_version
                    or action.source_value_id not in available_values
                ):
                    raise ValueError("Schedule IR state store/Plan decision mismatch")
                if action.state_id in state_actions:
                    raise ValueError("Schedule IR duplicates a state action")
                state_actions[action.state_id] = action.kind
            elif isinstance(action, StateInvalidateAction):
                candidate = selected_states.get(action.state_id)
                if candidate is None or candidate.action != StateAction.EVICT:
                    raise ValueError(
                        "Schedule IR state invalidation/Plan decision mismatch"
                    )
                if action.state_id in state_actions:
                    raise ValueError("Schedule IR duplicates a state action")
                state_actions[action.state_id] = action.kind
            elif isinstance(action, (FallbackAction, RetryAction)):
                pass
            elif isinstance(action, RequestReplanAction):
                if action.preserved_bound_module_hash != self.bound_module_hash:
                    raise ValueError(
                        "Schedule IR replan may not change Bound semantics"
                    )
                if emitted:
                    raise ValueError("Schedule IR replan request occurs after result")
            elif isinstance(action, MaterializeAction):
                transition = selected_transitions.get(action.transition_candidate_id)
                if transition is None or not _transition_matches(action, transition):
                    raise ValueError(
                        "Schedule IR materialization/Plan transition mismatch"
                    )
                before_region = region_by_op[action.before_op_id]
                if before_region.region_id in launched_regions:
                    raise ValueError(
                        "Schedule IR materializes after its consumer launch"
                    )
                if action.source_value_id not in available_values:
                    raise ValueError(
                        "Schedule IR materializes a value before definition"
                    )
                performed_transitions.add(action.transition_candidate_id)
            elif isinstance(action, TransferAction):
                if action.value_id not in available_values:
                    raise ValueError("Schedule IR transfers a value before definition")
                if action.direction == TransferDirection.HOST_TO_DEVICE:
                    if (
                        action.source_device != "host"
                        or action.target_device != template.workload.device
                    ):
                        raise ValueError(
                            "Schedule IR host-to-device transfer targets wrong device"
                        )
                elif action.direction == TransferDirection.DEVICE_TO_HOST:
                    if (
                        action.source_device != template.workload.device
                        or action.target_device != "host"
                    ):
                        raise ValueError(
                            "Schedule IR device-to-host transfer uses wrong device"
                        )
                elif (
                    action.source_device != template.workload.device
                    or action.target_device != template.workload.device
                ):
                    raise ValueError(
                        "Schedule IR device-to-device transfer uses unknown device"
                    )
                value_stream[action.value_id] = action.stream_id
            elif isinstance(action, LaunchAction):
                region = next(
                    (
                        candidate
                        for candidate in selected_regions
                        if candidate.region_id == action.region_id
                    ),
                    None,
                )
                if region is None or region.region_id in launched_regions:
                    raise ValueError(
                        "Schedule IR launch region is unknown or duplicated"
                    )
                backend = backend_by_region[region.region_id]
                if not _launch_matches(action, region=region, backend=backend):
                    raise ValueError("Schedule IR launch differs from PlanInstance")
                if not set(action.input_value_ids).issubset(available_values):
                    raise ValueError("Schedule IR launch has use-before-def")
                synchronized = waited_streams.setdefault(
                    action.stream_id, {"external", action.stream_id}
                )
                for value_id in action.input_value_ids:
                    producer_stream = value_stream[value_id]
                    if producer_stream not in synchronized:
                        raise ValueError(
                            "Schedule IR cross-stream dependency lacks event wait"
                        )
                required = {
                    transition_id
                    for representation in template.representation_candidates
                    if representation.region_id == region.region_id
                    and representation.candidate_id
                    in {
                        decision.candidate_id
                        for decision in instance.representation_decisions
                    }
                    for transition_id in representation.required_transition_candidate_ids
                }
                if not required.issubset(performed_transitions):
                    raise ValueError(
                        "Schedule IR launch omits required materialization"
                    )
                available_values.update(action.output_value_ids)
                for value_id in action.output_value_ids:
                    value_stream[value_id] = action.stream_id
                launched_regions.add(region.region_id)
            elif isinstance(action, EmitResultAction):
                if emitted or set(action.output_value_ids) != set(
                    bound_module.graph.outputs
                ):
                    raise ValueError("Schedule IR emit-result outputs are invalid")
                if action.query_ids != self.query_ids:
                    raise ValueError(
                        "Schedule IR emit-result query accounting mismatch"
                    )
                if not set(action.output_value_ids).issubset(available_values):
                    raise ValueError("Schedule IR emits unavailable outputs")
                emitted = True
            elif isinstance(action, FreeAction):
                if not emitted or action.arena_id not in allocated:
                    raise ValueError("Schedule IR arena free order is invalid")
                live_bytes -= arena_sizes[action.arena_id]
                allocated.remove(action.arena_id)
        if not checked_budget or not emitted:
            raise ValueError("Schedule IR omits budget check or result emission")
        if not batch_accounted:
            raise ValueError("Schedule IR omits explicit batch query accounting")
        if allocated or live_bytes != 0:
            raise ValueError("Schedule IR leaks allocated arenas")
        if peak_bytes != expected_peak:
            raise ValueError("Schedule IR runtime ledger peak mismatch")
        expected_launched_regions = {
            region.region_id
            for region in selected_regions
            if region.region_id not in state_reused_region_ids
        }
        if launched_regions != expected_launched_regions:
            raise ValueError(
                "Schedule IR launch/state-reuse region coverage is incomplete"
            )
        if performed_transitions != required_transition_ids:
            raise ValueError("Schedule IR does not execute every selected transition")
        expected_state_actions = {
            state_id: {
                StateAction.REUSE: ScheduleActionKind.STATE_LOAD,
                StateAction.CACHE: ScheduleActionKind.STATE_STORE,
                StateAction.EVICT: ScheduleActionKind.STATE_INVALIDATE,
            }[candidate.action]
            for state_id, candidate in selected_states.items()
            if candidate.action != StateAction.RECOMPUTE
        }
        if state_actions != expected_state_actions:
            raise ValueError(
                "Schedule IR does not implement every selected state action"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "bound_module_hash": self.bound_module_hash,
            "plan_template_hash": self.plan_template_hash,
            "plan_instance_hash": self.plan_instance_hash,
            "query_ids": list(self.query_ids),
            "buffers": [buffer.to_dict() for buffer in self.buffers],
            "actions": [action.to_dict() for action in self.actions],
        }

    def canonical_json(
        self,
        *,
        bound_module: BFBoundModule,
        template: PlanTemplate,
        instance: PlanInstance,
    ) -> str:
        self.validate(bound_module=bound_module, template=template, instance=instance)
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(
        self,
        *,
        bound_module: BFBoundModule,
        template: PlanTemplate,
        instance: PlanInstance,
    ) -> str:
        return hashlib.sha256(
            self.canonical_json(
                bound_module=bound_module,
                template=template,
                instance=instance,
            ).encode("utf-8")
        ).hexdigest()


def lower_plan_instance_to_reference_schedule(
    bound_module: BFBoundModule,
    *,
    template: PlanTemplate,
    instance: PlanInstance,
    query_ids: Tuple[str, ...],
) -> ScheduleModule:
    """Lower one verified PlanInstance into a synchronous Schedule IR."""

    instance.validate(template=template, bound_module=bound_module)
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError("schedule lowering requires unique query IDs")
    storage = next(
        candidate
        for candidate in template.storage_candidates
        if candidate.candidate_id == instance.storage_decision.candidate_id
    )
    buffers = tuple(
        ScheduleBuffer.from_storage_binding(binding) for binding in storage.bindings
    )
    arena_sizes: dict[str, int] = {}
    for buffer in buffers:
        arena_sizes[buffer.arena_id] = max(
            arena_sizes.get(buffer.arena_id, 0),
            buffer.offset_bytes + buffer.size_bytes,
        )
    actions: list[ScheduleAction] = [
        CheckBudgetAction("0000.check-budget", sum(arena_sizes.values()))
    ]
    for index, (arena_id, size_bytes) in enumerate(
        sorted(arena_sizes.items()), start=1
    ):
        actions.append(
            AllocateAction(f"{index:04d}.allocate:{arena_id}", arena_id, size_bytes)
        )
    action_index = len(actions)
    batch = next(
        candidate
        for candidate in template.batch_candidates
        if candidate.candidate_id == instance.batch_decision.candidate_id
    )
    if batch.spec_batch_size < template.workload.spec_batch_size:
        if (
            batch.domain_batch_size != template.workload.domain_batch_size
            or batch.sample_batch_size != template.workload.sample_batch_size
        ):
            raise NotImplementedError(
                "Schedule IR v1 spec slicing cannot also reduce domain/sample axes"
            )
        slices = tuple(
            QueryBatchSlice(
                slice_id=f"spec-slice:{start:04d}:{stop:04d}",
                query_ids=query_ids,
                start_index=start,
                stop_index=stop,
            )
            for start in range(
                0, template.workload.spec_batch_size, batch.spec_batch_size
            )
            for stop in (
                min(
                    start + batch.spec_batch_size,
                    template.workload.spec_batch_size,
                ),
            )
        )
        batch_axis = "spec"
    else:
        query_capacity = (
            batch.domain_batch_size * batch.spec_batch_size * batch.sample_batch_size
        )
        slices = tuple(
            QueryBatchSlice(
                slice_id=f"slice:{index // query_capacity:04d}",
                query_ids=query_ids[index : index + query_capacity],
            )
            for index in range(0, len(query_ids), query_capacity)
        )
        batch_axis = "domain"
    actions.append(
        BatchLoopAction(
            action_id=f"{action_index:04d}.batch-loop",
            axis=batch_axis,
            slices=slices,
        )
    )
    action_index += 1
    selected_states = _selected_states(template, instance)
    for state_id, state in sorted(selected_states.items()):
        if state.action == StateAction.REUSE:
            actions.append(
                StateLoadAction(
                    action_id=f"{action_index:04d}.state-load",
                    state_id=state_id,
                    source_value_id=state.source_value_id,
                    state_version=state.state_version,
                )
            )
            action_index += 1
        elif state.action == StateAction.EVICT:
            actions.append(
                StateInvalidateAction(
                    action_id=f"{action_index:04d}.state-invalidate",
                    state_id=state_id,
                    reason="selected_plan_evict",
                )
            )
            action_index += 1
    regions = _selected_regions(template, instance)
    backends = _selected_backends(template, instance)
    transitions = _selected_transitions(template, instance)
    op_index = {op.op_id: index for index, op in enumerate(bound_module.graph.ops)}
    regions = tuple(
        sorted(regions, key=lambda region: min(op_index[op] for op in region.op_ids))
    )
    reused_value_ids = {
        state.source_value_id
        for state in selected_states.values()
        if state.action == StateAction.REUSE
    }
    for region in regions:
        if set(region.output_value_ids).issubset(reused_value_ids):
            continue
        for transition in sorted(
            (
                candidate
                for candidate in transitions.values()
                if candidate.before_op_id in region.op_ids
            ),
            key=lambda candidate: candidate.candidate_id,
        ):
            actions.append(
                MaterializeAction(
                    action_id=f"{action_index:04d}.materialize",
                    transition_candidate_id=transition.candidate_id,
                    source_value_id=transition.source_value_id,
                    before_op_id=transition.before_op_id,
                    source_representation=transition.source_representation,
                    target_representation=transition.target_representation,
                )
            )
            action_index += 1
        backend = backends[region.region_id]
        actions.append(
            LaunchAction(
                action_id=f"{action_index:04d}.launch",
                task_id=f"task:{region.region_id}",
                region_id=region.region_id,
                backend_candidate_id=backend.candidate_id,
                backend_artifact_key=backend.compiled_artifact_key,
                input_value_ids=region.input_value_ids,
                output_value_ids=region.output_value_ids,
                stream_id="sync",
            )
        )
        action_index += 1
    for state_id, state in sorted(selected_states.items()):
        if state.action == StateAction.CACHE:
            actions.append(
                StateStoreAction(
                    action_id=f"{action_index:04d}.state-store",
                    state_id=state_id,
                    source_value_id=state.source_value_id,
                    state_version=state.state_version,
                )
            )
            action_index += 1
    actions.append(
        EmitResultAction(
            action_id=f"{action_index:04d}.emit-result",
            query_ids=query_ids,
            output_value_ids=bound_module.graph.outputs,
        )
    )
    action_index += 1
    for arena_id in sorted(arena_sizes, reverse=True):
        actions.append(FreeAction(f"{action_index:04d}.free:{arena_id}", arena_id))
        action_index += 1
    identity = "|".join(
        (
            instance.stable_hash(template=template, bound_module=bound_module),
            *query_ids,
            *(action.action_id for action in actions),
        )
    )
    schedule = ScheduleModule(
        schedule_id="schedule:"
        + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
        bound_module_hash=bound_module.stable_hash(),
        plan_template_hash=template.stable_hash(bound_module=bound_module),
        plan_instance_hash=instance.stable_hash(
            template=template, bound_module=bound_module
        ),
        query_ids=query_ids,
        buffers=buffers,
        actions=tuple(actions),
    )
    schedule.validate(bound_module=bound_module, template=template, instance=instance)
    return schedule


def _selected_regions(
    template: PlanTemplate, instance: PlanInstance
) -> Tuple[RegionCandidate, ...]:
    candidates = {
        candidate.candidate_id: candidate for candidate in template.region_candidates
    }
    return tuple(candidates[item.candidate_id] for item in instance.region_decisions)


def _selected_backends(
    template: PlanTemplate, instance: PlanInstance
) -> dict[str, BackendCandidate]:
    candidates = {
        candidate.candidate_id: candidate for candidate in template.backend_candidates
    }
    return {
        item.region_id: candidates[item.candidate_id]
        for item in instance.backend_decisions
    }


def _selected_transitions(
    template: PlanTemplate, instance: PlanInstance
) -> dict[str, MaterializationCandidate]:
    candidates = {
        candidate.candidate_id: candidate
        for candidate in template.materialization_candidates
    }
    return {
        item.candidate_id: candidates[item.candidate_id]
        for item in instance.materialization_decisions
    }


def _selected_states(
    template: PlanTemplate, instance: PlanInstance
) -> dict[str, StateCandidate]:
    candidates = {
        candidate.candidate_id: candidate for candidate in template.state_candidates
    }
    return {
        item.state_id: candidates[item.candidate_id]
        for item in instance.state_decisions
    }


def _transition_matches(
    action: MaterializeAction, candidate: MaterializationCandidate
) -> bool:
    return (
        action.source_value_id == candidate.source_value_id
        and action.before_op_id == candidate.before_op_id
        and action.source_representation == candidate.source_representation
        and action.target_representation == candidate.target_representation
    )


def _launch_matches(
    action: LaunchAction,
    *,
    region: RegionCandidate,
    backend: BackendCandidate,
) -> bool:
    return (
        action.task_id == f"task:{region.region_id}"
        and action.backend_candidate_id == backend.candidate_id
        and action.backend_artifact_key == backend.compiled_artifact_key
        and action.input_value_ids == region.input_value_ids
        and action.output_value_ids == region.output_value_ids
    )
