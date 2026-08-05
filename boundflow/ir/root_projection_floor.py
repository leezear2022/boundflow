"""Typed consumer/liveness IR for ranking-floor root projection."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,missing-class-docstring,duplicate-code

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from typing import Tuple


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class NativeRootProjectionFloorTaskKind(str, Enum):
    ADMIT_SOURCE = "admit_source"
    ANALYZE_CONSUMERS = "analyze_consumers"
    EXECUTE_BASELINE = "execute_baseline"
    REFINE_OBJECTIVES = "refine_objectives"
    EXECUTE_ROOT_PROJECTIONS = "execute_root_projections"
    RANK_ROOTS = "rank_roots"
    EMIT_FLOOR = "emit_floor"


@dataclass(frozen=True)
class NativeRootProjectionClauseOwnerIR:
    original_clause_index: int
    objective_hash: str
    threshold_hash: str

    def validate(self, *, clause_count: int) -> None:
        if (
            not 0 <= self.original_clause_index < clause_count
            or not _is_sha256(self.objective_hash)
            or not _is_sha256(self.threshold_hash)
        ):
            raise ValueError("root-projection clause owner is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "original_clause_index": self.original_clause_index,
            "objective_hash": self.objective_hash,
            "threshold_hash": self.threshold_hash,
        }


@dataclass(frozen=True)
class NativeRootProjectionFloorPlanIR:
    plan_id: str
    source_plan_hash: str
    source_task_ir_hash: str
    source_schedule_hash: str
    clause_count: int
    consumed_result_fields: Tuple[str, ...]
    full_max_nodes: int = 31
    full_max_depth: int = 4
    projected_max_nodes: int = 1
    projected_max_depth: int = 0
    soundness_mode: str = "ranking_only_sound_less_complete_v1"
    semantics_owner: str = "boundflow_root_projection_floor"
    performance_claimed: bool = False
    schema_version: str = "boundflow.root-projection-floor-plan/v1"

    def validate(self) -> None:
        if (
            self.schema_version != "boundflow.root-projection-floor-plan/v1"
            or not self.plan_id
            or any(
                not _is_sha256(value)
                for value in (
                    self.source_plan_hash,
                    self.source_task_ir_hash,
                    self.source_schedule_hash,
                )
            )
            or self.clause_count != 9
            or self.consumed_result_fields
            != (
                "root.lower",
                "root.upper",
                "root.branch_candidate",
                "query.status",
                "counterexample.evidence",
            )
            or (self.full_max_nodes, self.full_max_depth) != (31, 4)
            or (self.projected_max_nodes, self.projected_max_depth) != (1, 0)
            or self.soundness_mode != "ranking_only_sound_less_complete_v1"
            or self.semantics_owner != "boundflow_root_projection_floor"
            or self.performance_claimed is not False
        ):
            raise ValueError("root-projection floor Plan IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "source_plan_hash": self.source_plan_hash,
            "source_task_ir_hash": self.source_task_ir_hash,
            "source_schedule_hash": self.source_schedule_hash,
            "clause_count": self.clause_count,
            "consumed_result_fields": list(self.consumed_result_fields),
            "full_max_nodes": self.full_max_nodes,
            "full_max_depth": self.full_max_depth,
            "projected_max_nodes": self.projected_max_nodes,
            "projected_max_depth": self.projected_max_depth,
            "soundness_mode": self.soundness_mode,
            "semantics_owner": self.semantics_owner,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeRootProjectionFloorInstanceIR:
    instance_id: str
    plan_hash: str
    objective_matrix_hash: str
    thresholds_hash: str
    clause_owners: Tuple[NativeRootProjectionClauseOwnerIR, ...]
    semantic_token: str
    schema_version: str = "boundflow.root-projection-floor-instance/v1"

    def semantic_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "instance_id": self.instance_id,
            "plan_hash": self.plan_hash,
            "objective_matrix_hash": self.objective_matrix_hash,
            "thresholds_hash": self.thresholds_hash,
            "clause_owners": [item.to_dict() for item in self.clause_owners],
        }

    def validate(self, *, plan: NativeRootProjectionFloorPlanIR) -> None:
        plan.validate()
        if (
            self.schema_version != "boundflow.root-projection-floor-instance/v1"
            or not self.instance_id
            or self.plan_hash != plan.stable_hash()
            or not _is_sha256(self.objective_matrix_hash)
            or not _is_sha256(self.thresholds_hash)
            or len(self.clause_owners) != plan.clause_count
            or tuple(item.original_clause_index for item in self.clause_owners)
            != tuple(range(plan.clause_count))
            or not _is_sha256(self.semantic_token)
            or self.semantic_token != _canonical_hash(self.semantic_dict())
        ):
            raise ValueError("root-projection floor Instance IR differs")
        for item in self.clause_owners:
            item.validate(clause_count=plan.clause_count)

    @classmethod
    def create(
        cls,
        *,
        plan: NativeRootProjectionFloorPlanIR,
        objective_matrix_hash: str,
        thresholds_hash: str,
        clause_owners: Tuple[NativeRootProjectionClauseOwnerIR, ...],
    ) -> "NativeRootProjectionFloorInstanceIR":
        value = cls(
            instance_id=f"{plan.plan_id}:instance",
            plan_hash=plan.stable_hash(),
            objective_matrix_hash=objective_matrix_hash,
            thresholds_hash=thresholds_hash,
            clause_owners=clause_owners,
            semantic_token="0" * 64,
        )
        value = replace(value, semantic_token=_canonical_hash(value.semantic_dict()))
        value.validate(plan=plan)
        return value

    def to_dict(self, *, plan: NativeRootProjectionFloorPlanIR) -> dict[str, object]:
        self.validate(plan=plan)
        return {**self.semantic_dict(), "semantic_token": self.semantic_token}

    def stable_hash(self, *, plan: NativeRootProjectionFloorPlanIR) -> str:
        return _canonical_hash(self.to_dict(plan=plan))


@dataclass(frozen=True)
class NativeRootProjectionFloorTaskIRUnit:
    task_id: str
    kind: NativeRootProjectionFloorTaskKind
    dependency_task_ids: Tuple[str, ...]
    input_value_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]

    def validate(self) -> None:
        if (
            not self.task_id
            or not self.input_value_ids
            or not self.output_value_ids
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
        ):
            raise ValueError("root-projection floor Task unit is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "dependency_task_ids": list(self.dependency_task_ids),
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
        }


@dataclass(frozen=True)
class NativeRootProjectionFloorTaskIRModule:
    module_id: str
    plan_hash: str
    instance_hash: str
    tasks: Tuple[NativeRootProjectionFloorTaskIRUnit, ...]
    output_task_id: str
    schema_version: str = "boundflow.root-projection-floor-task-ir/v1"

    def validate(
        self,
        *,
        plan: NativeRootProjectionFloorPlanIR,
        instance: NativeRootProjectionFloorInstanceIR,
    ) -> None:
        instance.validate(plan=plan)
        if (
            self.schema_version != "boundflow.root-projection-floor-task-ir/v1"
            or not self.module_id
            or self.plan_hash != plan.stable_hash()
            or self.instance_hash != instance.stable_hash(plan=plan)
            or tuple(item.kind for item in self.tasks)
            != tuple(NativeRootProjectionFloorTaskKind)
            or self.output_task_id != self.tasks[-1].task_id
        ):
            raise ValueError("root-projection floor Task module differs")
        completed: set[str] = set()
        available = {"projection.plan", "projection.instance", "source.program"}
        for task in self.tasks:
            task.validate()
            if any(item not in completed for item in task.dependency_task_ids):
                raise ValueError("root-projection floor Task dependency is late")
            if any(item not in available for item in task.input_value_ids):
                raise ValueError("root-projection floor Task input is late")
            if any(item in available for item in task.output_value_ids):
                raise ValueError("root-projection floor Task output is redefined")
            completed.add(task.task_id)
            available.update(task.output_value_ids)

    def to_dict(
        self,
        *,
        plan: NativeRootProjectionFloorPlanIR,
        instance: NativeRootProjectionFloorInstanceIR,
    ) -> dict[str, object]:
        self.validate(plan=plan, instance=instance)
        return {
            "schema_version": self.schema_version,
            "module_id": self.module_id,
            "plan_hash": self.plan_hash,
            "instance_hash": self.instance_hash,
            "tasks": [item.to_dict() for item in self.tasks],
            "output_task_id": self.output_task_id,
        }

    def stable_hash(
        self,
        *,
        plan: NativeRootProjectionFloorPlanIR,
        instance: NativeRootProjectionFloorInstanceIR,
    ) -> str:
        return _canonical_hash(self.to_dict(plan=plan, instance=instance))


@dataclass(frozen=True)
class NativeRootProjectionFloorScheduleAction:
    action_id: str
    sequence: int
    task_id: str

    def validate(self) -> None:
        if not self.action_id or self.sequence < 0 or not self.task_id:
            raise ValueError("root-projection floor Schedule action is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "action_id": self.action_id,
            "sequence": self.sequence,
            "task_id": self.task_id,
        }


@dataclass(frozen=True)
class NativeRootProjectionFloorScheduleIR:
    schedule_id: str
    plan_hash: str
    instance_hash: str
    task_module_hash: str
    actions: Tuple[NativeRootProjectionFloorScheduleAction, ...]
    full_evaluation_budget: int
    projected_evaluation_budget: int
    schema_version: str = "boundflow.root-projection-floor-schedule-ir/v1"

    def validate(
        self,
        *,
        plan: NativeRootProjectionFloorPlanIR,
        instance: NativeRootProjectionFloorInstanceIR,
        task_module: NativeRootProjectionFloorTaskIRModule,
    ) -> None:
        task_module.validate(plan=plan, instance=instance)
        if (
            self.schema_version != "boundflow.root-projection-floor-schedule-ir/v1"
            or not self.schedule_id
            or self.plan_hash != plan.stable_hash()
            or self.instance_hash != instance.stable_hash(plan=plan)
            or self.task_module_hash
            != task_module.stable_hash(plan=plan, instance=instance)
            or len(self.actions) != len(task_module.tasks)
            or self.full_evaluation_budget != plan.clause_count * plan.full_max_nodes
            or self.projected_evaluation_budget
            != plan.clause_count * plan.projected_max_nodes
        ):
            raise ValueError("root-projection floor Schedule differs")
        for index, (action, task) in enumerate(zip(self.actions, task_module.tasks)):
            action.validate()
            if action.sequence != index or action.task_id != task.task_id:
                raise ValueError("root-projection floor Schedule/Task differs")

    def to_dict(
        self,
        *,
        plan: NativeRootProjectionFloorPlanIR,
        instance: NativeRootProjectionFloorInstanceIR,
        task_module: NativeRootProjectionFloorTaskIRModule,
    ) -> dict[str, object]:
        self.validate(plan=plan, instance=instance, task_module=task_module)
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "plan_hash": self.plan_hash,
            "instance_hash": self.instance_hash,
            "task_module_hash": self.task_module_hash,
            "actions": [item.to_dict() for item in self.actions],
            "full_evaluation_budget": self.full_evaluation_budget,
            "projected_evaluation_budget": self.projected_evaluation_budget,
        }

    def stable_hash(
        self,
        *,
        plan: NativeRootProjectionFloorPlanIR,
        instance: NativeRootProjectionFloorInstanceIR,
        task_module: NativeRootProjectionFloorTaskIRModule,
    ) -> str:
        return _canonical_hash(
            self.to_dict(plan=plan, instance=instance, task_module=task_module)
        )


@dataclass(frozen=True)
class NativeRootProjectionClauseTraceIR:
    original_clause_index: int
    objective_hash: str
    refinement_plan_hash: str
    refinement_trace_hash: str
    query_trace_hash: str
    root_evaluation_hash: str
    evaluation_count: int
    decision_count: int
    child_node_count: int
    status: str

    def validate(self, *, plan: NativeRootProjectionFloorPlanIR) -> None:
        hashes = (
            self.objective_hash,
            self.refinement_plan_hash,
            self.refinement_trace_hash,
            self.query_trace_hash,
            self.root_evaluation_hash,
        )
        if (
            not 0 <= self.original_clause_index < plan.clause_count
            or any(not _is_sha256(value) for value in hashes)
            or self.evaluation_count != 1
            or self.decision_count != 1
            or self.child_node_count != 0
            or self.status not in {"verified", "unsafe", "unknown"}
        ):
            raise ValueError("root-projection clause Trace differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "original_clause_index": self.original_clause_index,
            "objective_hash": self.objective_hash,
            "refinement_plan_hash": self.refinement_plan_hash,
            "refinement_trace_hash": self.refinement_trace_hash,
            "query_trace_hash": self.query_trace_hash,
            "root_evaluation_hash": self.root_evaluation_hash,
            "evaluation_count": self.evaluation_count,
            "decision_count": self.decision_count,
            "child_node_count": self.child_node_count,
            "status": self.status,
        }


@dataclass(frozen=True)
class NativeRootProjectionFloorTraceIR:
    plan_hash: str
    instance_hash: str
    task_module_hash: str
    schedule_hash: str
    source_floor_trace_hash: str
    baseline_trace_hash: str
    shared_refinement_trace_hash: str
    clause_traces: Tuple[NativeRootProjectionClauseTraceIR, ...]
    completed_original_clause_indices: Tuple[int, ...]
    final_status: str
    performance_claimed: bool = False
    schema_version: str = "boundflow.root-projection-floor-trace/v1"

    def validate(
        self,
        *,
        plan: NativeRootProjectionFloorPlanIR,
        instance: NativeRootProjectionFloorInstanceIR,
        task_module: NativeRootProjectionFloorTaskIRModule,
        schedule: NativeRootProjectionFloorScheduleIR,
    ) -> None:
        schedule.validate(plan=plan, instance=instance, task_module=task_module)
        hashes = (
            self.plan_hash,
            self.instance_hash,
            self.task_module_hash,
            self.schedule_hash,
            self.source_floor_trace_hash,
            self.baseline_trace_hash,
            self.shared_refinement_trace_hash,
        )
        if (
            self.schema_version != "boundflow.root-projection-floor-trace/v1"
            or any(not _is_sha256(value) for value in hashes)
            or self.plan_hash != plan.stable_hash()
            or self.instance_hash != instance.stable_hash(plan=plan)
            or self.task_module_hash
            != task_module.stable_hash(plan=plan, instance=instance)
            or self.schedule_hash
            != schedule.stable_hash(
                plan=plan, instance=instance, task_module=task_module
            )
            or len(self.clause_traces) != plan.clause_count
            or tuple(item.original_clause_index for item in self.clause_traces)
            != tuple(range(plan.clause_count))
            or self.completed_original_clause_indices != tuple(range(plan.clause_count))
            or self.final_status not in {"verified", "unsafe", "unknown"}
            or self.performance_claimed is not False
        ):
            raise ValueError("root-projection floor Trace differs")
        for item in self.clause_traces:
            item.validate(plan=plan)

    def to_dict(
        self,
        *,
        plan: NativeRootProjectionFloorPlanIR,
        instance: NativeRootProjectionFloorInstanceIR,
        task_module: NativeRootProjectionFloorTaskIRModule,
        schedule: NativeRootProjectionFloorScheduleIR,
    ) -> dict[str, object]:
        self.validate(
            plan=plan,
            instance=instance,
            task_module=task_module,
            schedule=schedule,
        )
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "instance_hash": self.instance_hash,
            "task_module_hash": self.task_module_hash,
            "schedule_hash": self.schedule_hash,
            "source_floor_trace_hash": self.source_floor_trace_hash,
            "baseline_trace_hash": self.baseline_trace_hash,
            "shared_refinement_trace_hash": self.shared_refinement_trace_hash,
            "clause_traces": [item.to_dict() for item in self.clause_traces],
            "completed_original_clause_indices": list(
                self.completed_original_clause_indices
            ),
            "final_status": self.final_status,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(
        self,
        *,
        plan: NativeRootProjectionFloorPlanIR,
        instance: NativeRootProjectionFloorInstanceIR,
        task_module: NativeRootProjectionFloorTaskIRModule,
        schedule: NativeRootProjectionFloorScheduleIR,
    ) -> str:
        return _canonical_hash(
            self.to_dict(
                plan=plan,
                instance=instance,
                task_module=task_module,
                schedule=schedule,
            )
        )


def lower_native_root_projection_floor_ir(
    plan: NativeRootProjectionFloorPlanIR,
    instance: NativeRootProjectionFloorInstanceIR,
) -> tuple[NativeRootProjectionFloorTaskIRModule, NativeRootProjectionFloorScheduleIR]:
    instance.validate(plan=plan)
    definitions = (
        (
            NativeRootProjectionFloorTaskKind.ADMIT_SOURCE,
            ("projection.plan", "projection.instance", "source.program"),
            ("projection.source",),
        ),
        (
            NativeRootProjectionFloorTaskKind.ANALYZE_CONSUMERS,
            ("projection.source",),
            ("projection.consumer_contract",),
        ),
        (
            NativeRootProjectionFloorTaskKind.EXECUTE_BASELINE,
            ("projection.source",),
            ("projection.baseline",),
        ),
        (
            NativeRootProjectionFloorTaskKind.REFINE_OBJECTIVES,
            ("projection.baseline",),
            ("projection.refinements",),
        ),
        (
            NativeRootProjectionFloorTaskKind.EXECUTE_ROOT_PROJECTIONS,
            ("projection.consumer_contract", "projection.refinements"),
            ("projection.roots",),
        ),
        (
            NativeRootProjectionFloorTaskKind.RANK_ROOTS,
            ("projection.roots",),
            ("projection.ranking",),
        ),
        (
            NativeRootProjectionFloorTaskKind.EMIT_FLOOR,
            ("projection.ranking",),
            ("projection.floor",),
        ),
    )
    tasks = []
    dependencies: tuple[str, ...] = ()
    for kind, inputs, outputs in definitions:
        task_id = f"{plan.plan_id}:{kind.value}"
        tasks.append(
            NativeRootProjectionFloorTaskIRUnit(
                task_id=task_id,
                kind=kind,
                dependency_task_ids=dependencies,
                input_value_ids=inputs,
                output_value_ids=outputs,
            )
        )
        dependencies = (task_id,)
    task_module = NativeRootProjectionFloorTaskIRModule(
        module_id=f"{plan.plan_id}:tasks",
        plan_hash=plan.stable_hash(),
        instance_hash=instance.stable_hash(plan=plan),
        tasks=tuple(tasks),
        output_task_id=tasks[-1].task_id,
    )
    schedule = NativeRootProjectionFloorScheduleIR(
        schedule_id=f"{plan.plan_id}:schedule",
        plan_hash=plan.stable_hash(),
        instance_hash=instance.stable_hash(plan=plan),
        task_module_hash=task_module.stable_hash(plan=plan, instance=instance),
        actions=tuple(
            NativeRootProjectionFloorScheduleAction(
                action_id=f"{plan.plan_id}:launch:{index:04d}:{task.kind.value}",
                sequence=index,
                task_id=task.task_id,
            )
            for index, task in enumerate(tasks)
        ),
        full_evaluation_budget=plan.clause_count * plan.full_max_nodes,
        projected_evaluation_budget=plan.clause_count * plan.projected_max_nodes,
    )
    schedule.validate(plan=plan, instance=instance, task_module=task_module)
    return task_module, schedule


__all__ = [
    "NativeRootProjectionClauseOwnerIR",
    "NativeRootProjectionClauseTraceIR",
    "NativeRootProjectionFloorInstanceIR",
    "NativeRootProjectionFloorPlanIR",
    "NativeRootProjectionFloorScheduleIR",
    "NativeRootProjectionFloorTaskIRModule",
    "NativeRootProjectionFloorTaskKind",
    "NativeRootProjectionFloorTraceIR",
    "lower_native_root_projection_floor_ir",
]
