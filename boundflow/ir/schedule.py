"""First-class Schedule IR v1 schema, verifier, and reference lowering."""

# Cross-layer schedule verification intentionally keeps the execution ledger together.
# pylint: disable=too-many-branches,too-many-instance-attributes,too-many-locals,too-many-statements,missing-class-docstring,missing-function-docstring

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
    StorageBinding,
)

SCHEDULE_IR_SCHEMA_VERSION = "boundflow.schedule_ir/v1.0"


class ScheduleActionKind(Enum):
    """Closed action set implemented by the IR-3A synchronous reference path."""

    CHECK_BUDGET = "check_budget"
    ALLOCATE = "allocate"
    MATERIALIZE = "materialize"
    LAUNCH = "launch"
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


ScheduleAction = (
    CheckBudgetAction
    | AllocateAction
    | MaterializeAction
    | LaunchAction
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

        selected_regions = _selected_regions(template, instance)
        backend_by_region = _selected_backends(template, instance)
        selected_transitions = _selected_transitions(template, instance)
        region_by_op = {
            op_id: region for region in selected_regions for op_id in region.op_ids
        }
        allocated: set[str] = set()
        available_values = set(bound_module.graph.inputs)
        launched_regions: set[str] = set()
        performed_transitions: set[str] = set()
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
                if action.stream_id != "sync":
                    raise ValueError(
                        "Schedule IR v1 reference path requires sync stream"
                    )
                if not set(action.input_value_ids).issubset(available_values):
                    raise ValueError("Schedule IR launch has use-before-def")
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
        if allocated or live_bytes != 0:
            raise ValueError("Schedule IR leaks allocated arenas")
        if peak_bytes != expected_peak:
            raise ValueError("Schedule IR runtime ledger peak mismatch")
        if launched_regions != {region.region_id for region in selected_regions}:
            raise ValueError("Schedule IR does not launch every selected region")
        if performed_transitions != set(selected_transitions):
            raise ValueError("Schedule IR does not execute every selected transition")

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
    regions = _selected_regions(template, instance)
    backends = _selected_backends(template, instance)
    transitions = _selected_transitions(template, instance)
    op_index = {op.op_id: index for index, op in enumerate(bound_module.graph.ops)}
    regions = tuple(
        sorted(regions, key=lambda region: min(op_index[op] for op in region.op_ids))
    )
    for region in regions:
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
