"""Prepared validation ownership IR for intermediate refinement."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,missing-class-docstring,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Optional, Tuple


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class NativePreparedRefinementTaskKind(str, Enum):
    ADMIT_EXACT = "admit_exact"
    CONSUME_PLAN_TARGETS = "consume_plan_targets"
    EXECUTE_SELECTED_CROWN = "execute_selected_crown"
    COMMIT_RESULT = "commit_result"
    EMIT_RECEIPT = "emit_receipt"


@dataclass(frozen=True)
class NativePreparedIntermediateRefinementCapsuleIR:
    capsule_id: str
    source_plan_hash: str
    source_task_module_hash: str
    source_schedule_hash: str
    primal_graph_hash: str
    input_bounds_hash: str
    split_state_hash: str
    initial_intermediate_bounds_hash: str
    objective_hash: Optional[str]
    source_refinement_plan_hash: Optional[str]
    source_refinement_semantic_trace_hash: Optional[str]
    target_table_hash: str
    target_count: int
    full_validation_receipt: str
    full_validation_count: int = 1
    semantics_owner: str = "boundflow_prepared_intermediate_refinement"
    performance_claimed: bool = False
    schema_version: str = "boundflow.prepared-intermediate-refinement-capsule/v1"

    def receipt_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "capsule_id": self.capsule_id,
            "source_plan_hash": self.source_plan_hash,
            "source_task_module_hash": self.source_task_module_hash,
            "source_schedule_hash": self.source_schedule_hash,
            "primal_graph_hash": self.primal_graph_hash,
            "input_bounds_hash": self.input_bounds_hash,
            "split_state_hash": self.split_state_hash,
            "initial_intermediate_bounds_hash": (self.initial_intermediate_bounds_hash),
            "objective_hash": self.objective_hash,
            "source_refinement_plan_hash": self.source_refinement_plan_hash,
            "source_refinement_semantic_trace_hash": (
                self.source_refinement_semantic_trace_hash
            ),
            "target_table_hash": self.target_table_hash,
            "target_count": self.target_count,
            "full_validation_count": self.full_validation_count,
            "semantics_owner": self.semantics_owner,
            "performance_claimed": self.performance_claimed,
        }

    def validate(self) -> None:
        required = (
            self.source_plan_hash,
            self.source_task_module_hash,
            self.source_schedule_hash,
            self.primal_graph_hash,
            self.input_bounds_hash,
            self.split_state_hash,
            self.initial_intermediate_bounds_hash,
            self.target_table_hash,
            self.full_validation_receipt,
        )
        optional = (
            self.objective_hash,
            self.source_refinement_plan_hash,
            self.source_refinement_semantic_trace_hash,
        )
        if (
            self.schema_version
            != "boundflow.prepared-intermediate-refinement-capsule/v1"
            or not self.capsule_id
            or any(not _is_sha256(value) for value in required)
            or any(value is not None and not _is_sha256(value) for value in optional)
            or self.target_count < 1
            or self.full_validation_count != 1
            or self.semantics_owner != "boundflow_prepared_intermediate_refinement"
            or self.performance_claimed is not False
            or self.full_validation_receipt != _canonical_hash(self.receipt_dict())
        ):
            raise ValueError("prepared refinement capsule is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            **self.receipt_dict(),
            "full_validation_receipt": self.full_validation_receipt,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativePreparedRefinementTaskIRUnit:
    task_id: str
    kind: NativePreparedRefinementTaskKind
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
            raise ValueError("prepared refinement Task is invalid")

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
class NativePreparedRefinementTaskIRModule:
    module_id: str
    capsule_hash: str
    tasks: Tuple[NativePreparedRefinementTaskIRUnit, ...]
    output_task_id: str
    schema_version: str = "boundflow.prepared-intermediate-refinement-task-ir/v1"

    def validate(
        self, *, capsule: NativePreparedIntermediateRefinementCapsuleIR
    ) -> None:
        capsule.validate()
        if (
            self.schema_version
            != "boundflow.prepared-intermediate-refinement-task-ir/v1"
            or not self.module_id
            or self.capsule_hash != capsule.stable_hash()
            or tuple(task.kind for task in self.tasks)
            != tuple(NativePreparedRefinementTaskKind)
            or self.output_task_id != self.tasks[-1].task_id
        ):
            raise ValueError("prepared refinement Task module differs")
        completed: set[str] = set()
        available = {"prepared.capsule", "source.program"}
        for task in self.tasks:
            task.validate()
            if any(value not in completed for value in task.dependency_task_ids):
                raise ValueError("prepared refinement Task dependency is late")
            if any(value not in available for value in task.input_value_ids):
                raise ValueError("prepared refinement Task input is late")
            if any(value in available for value in task.output_value_ids):
                raise ValueError("prepared refinement Task output is redefined")
            completed.add(task.task_id)
            available.update(task.output_value_ids)

    def to_dict(
        self, *, capsule: NativePreparedIntermediateRefinementCapsuleIR
    ) -> dict[str, object]:
        self.validate(capsule=capsule)
        return {
            "schema_version": self.schema_version,
            "module_id": self.module_id,
            "capsule_hash": self.capsule_hash,
            "tasks": [task.to_dict() for task in self.tasks],
            "output_task_id": self.output_task_id,
        }

    def stable_hash(
        self, *, capsule: NativePreparedIntermediateRefinementCapsuleIR
    ) -> str:
        return _canonical_hash(self.to_dict(capsule=capsule))


@dataclass(frozen=True)
class NativePreparedRefinementScheduleAction:
    action_id: str
    sequence: int
    task_id: str

    def validate(self) -> None:
        if not self.action_id or self.sequence < 0 or not self.task_id:
            raise ValueError("prepared refinement Schedule action is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "action_id": self.action_id,
            "sequence": self.sequence,
            "task_id": self.task_id,
        }


@dataclass(frozen=True)
class NativePreparedRefinementScheduleIR:
    schedule_id: str
    capsule_hash: str
    task_module_hash: str
    actions: Tuple[NativePreparedRefinementScheduleAction, ...]
    full_validation_launches: int = 1
    runtime_target_selection_launches: int = 1
    schema_version: str = "boundflow.prepared-intermediate-refinement-schedule/v1"

    def validate(
        self,
        *,
        capsule: NativePreparedIntermediateRefinementCapsuleIR,
        task_module: NativePreparedRefinementTaskIRModule,
    ) -> None:
        task_module.validate(capsule=capsule)
        if (
            self.schema_version
            != "boundflow.prepared-intermediate-refinement-schedule/v1"
            or not self.schedule_id
            or self.capsule_hash != capsule.stable_hash()
            or self.task_module_hash != task_module.stable_hash(capsule=capsule)
            or len(self.actions) != len(task_module.tasks)
            or self.full_validation_launches != 1
            or self.runtime_target_selection_launches != 1
        ):
            raise ValueError("prepared refinement Schedule differs")
        for index, (action, task) in enumerate(zip(self.actions, task_module.tasks)):
            action.validate()
            if action.sequence != index or action.task_id != task.task_id:
                raise ValueError("prepared refinement Schedule/Task differs")

    def to_dict(
        self,
        *,
        capsule: NativePreparedIntermediateRefinementCapsuleIR,
        task_module: NativePreparedRefinementTaskIRModule,
    ) -> dict[str, object]:
        self.validate(capsule=capsule, task_module=task_module)
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "capsule_hash": self.capsule_hash,
            "task_module_hash": self.task_module_hash,
            "actions": [action.to_dict() for action in self.actions],
            "full_validation_launches": self.full_validation_launches,
            "runtime_target_selection_launches": (
                self.runtime_target_selection_launches
            ),
        }

    def stable_hash(
        self,
        *,
        capsule: NativePreparedIntermediateRefinementCapsuleIR,
        task_module: NativePreparedRefinementTaskIRModule,
    ) -> str:
        return _canonical_hash(self.to_dict(capsule=capsule, task_module=task_module))


@dataclass(frozen=True)
class NativePreparedRefinementExecutionTraceIR:
    capsule_hash: str
    task_module_hash: str
    schedule_hash: str
    source_execution_semantic_trace_hash: str
    final_intermediate_bounds_hash: str
    full_validation_count: int
    runtime_target_selection_count: int
    performance_claimed: bool = False
    schema_version: str = "boundflow.prepared-intermediate-refinement-trace/v1"

    def validate(
        self,
        *,
        capsule: NativePreparedIntermediateRefinementCapsuleIR,
        task_module: NativePreparedRefinementTaskIRModule,
        schedule: NativePreparedRefinementScheduleIR,
    ) -> None:
        schedule.validate(capsule=capsule, task_module=task_module)
        if (
            self.schema_version != "boundflow.prepared-intermediate-refinement-trace/v1"
            or self.capsule_hash != capsule.stable_hash()
            or self.task_module_hash != task_module.stable_hash(capsule=capsule)
            or self.schedule_hash
            != schedule.stable_hash(capsule=capsule, task_module=task_module)
            or not _is_sha256(self.source_execution_semantic_trace_hash)
            or not _is_sha256(self.final_intermediate_bounds_hash)
            or self.full_validation_count != 1
            or self.runtime_target_selection_count != 1
            or self.performance_claimed is not False
        ):
            raise ValueError("prepared refinement execution Trace differs")

    def to_dict(
        self,
        *,
        capsule: NativePreparedIntermediateRefinementCapsuleIR,
        task_module: NativePreparedRefinementTaskIRModule,
        schedule: NativePreparedRefinementScheduleIR,
    ) -> dict[str, object]:
        self.validate(capsule=capsule, task_module=task_module, schedule=schedule)
        return {
            "schema_version": self.schema_version,
            "capsule_hash": self.capsule_hash,
            "task_module_hash": self.task_module_hash,
            "schedule_hash": self.schedule_hash,
            "source_execution_semantic_trace_hash": (
                self.source_execution_semantic_trace_hash
            ),
            "final_intermediate_bounds_hash": self.final_intermediate_bounds_hash,
            "full_validation_count": self.full_validation_count,
            "runtime_target_selection_count": self.runtime_target_selection_count,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(
        self,
        *,
        capsule: NativePreparedIntermediateRefinementCapsuleIR,
        task_module: NativePreparedRefinementTaskIRModule,
        schedule: NativePreparedRefinementScheduleIR,
    ) -> str:
        return _canonical_hash(
            self.to_dict(capsule=capsule, task_module=task_module, schedule=schedule)
        )


def lower_native_prepared_refinement_ir(
    capsule: NativePreparedIntermediateRefinementCapsuleIR,
) -> tuple[NativePreparedRefinementTaskIRModule, NativePreparedRefinementScheduleIR]:
    capsule.validate()
    definitions = (
        (
            NativePreparedRefinementTaskKind.ADMIT_EXACT,
            ("prepared.capsule", "source.program"),
            ("prepared.source",),
        ),
        (
            NativePreparedRefinementTaskKind.CONSUME_PLAN_TARGETS,
            ("prepared.source",),
            ("prepared.targets",),
        ),
        (
            NativePreparedRefinementTaskKind.EXECUTE_SELECTED_CROWN,
            ("prepared.source", "prepared.targets"),
            ("prepared.bounds",),
        ),
        (
            NativePreparedRefinementTaskKind.COMMIT_RESULT,
            ("prepared.bounds",),
            ("prepared.execution",),
        ),
        (
            NativePreparedRefinementTaskKind.EMIT_RECEIPT,
            ("prepared.execution",),
            ("prepared.receipt",),
        ),
    )
    tasks = []
    dependencies: tuple[str, ...] = ()
    for kind, inputs, outputs in definitions:
        task_id = f"{capsule.capsule_id}:{kind.value}"
        tasks.append(
            NativePreparedRefinementTaskIRUnit(
                task_id=task_id,
                kind=kind,
                dependency_task_ids=dependencies,
                input_value_ids=inputs,
                output_value_ids=outputs,
            )
        )
        dependencies = (task_id,)
    task_module = NativePreparedRefinementTaskIRModule(
        module_id=f"{capsule.capsule_id}:tasks",
        capsule_hash=capsule.stable_hash(),
        tasks=tuple(tasks),
        output_task_id=tasks[-1].task_id,
    )
    schedule = NativePreparedRefinementScheduleIR(
        schedule_id=f"{capsule.capsule_id}:schedule",
        capsule_hash=capsule.stable_hash(),
        task_module_hash=task_module.stable_hash(capsule=capsule),
        actions=tuple(
            NativePreparedRefinementScheduleAction(
                action_id=f"{capsule.capsule_id}:launch:{index:04d}:{task.kind.value}",
                sequence=index,
                task_id=task.task_id,
            )
            for index, task in enumerate(tasks)
        ),
    )
    schedule.validate(capsule=capsule, task_module=task_module)
    return task_module, schedule


__all__ = [
    "NativePreparedIntermediateRefinementCapsuleIR",
    "NativePreparedRefinementExecutionTraceIR",
    "NativePreparedRefinementScheduleIR",
    "NativePreparedRefinementTaskIRModule",
    "NativePreparedRefinementTaskKind",
    "lower_native_prepared_refinement_ir",
]
