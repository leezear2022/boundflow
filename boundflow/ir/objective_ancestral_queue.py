"""Typed dynamic control IR for objective-root ancestral BaB search."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Optional, Tuple

from .refinement import NativeIntermediateRefinementPolicyIR
from .search_scaling import NativeBabSearchBudgetIR

OBJECTIVE_ANCESTRAL_QUEUE_PLAN_IR_SCHEMA_VERSION = (
    "boundflow.objective-ancestral-queue-plan-ir/v1"
)
OBJECTIVE_ANCESTRAL_QUEUE_TASK_IR_SCHEMA_VERSION = (
    "boundflow.objective-ancestral-queue-task-ir/v1"
)
OBJECTIVE_ANCESTRAL_QUEUE_SCHEDULE_IR_SCHEMA_VERSION = (
    "boundflow.objective-ancestral-queue-schedule-ir/v1"
)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class ObjectiveAncestralQueueTaskKind(Enum):
    """Closed task vocabulary for one dynamic queue instance."""

    ADMIT_ROOT_SOURCE = "admit_root_source"
    EVALUATE_ROOT = "evaluate_root"
    COMPILE_CHILD_REFINEMENT = "compile_child_refinement"
    EXECUTE_CHILD_REFINEMENT = "execute_child_refinement"
    EVALUATE_CHILD = "evaluate_child"
    TRANSITION_QUEUE = "transition_queue"
    EMIT_RESULT = "emit_result"


@dataclass(frozen=True)
class NativeObjectiveAncestralQueuePlanIR:
    """Static query/root-source policy and dynamic task template contract."""

    plan_id: str
    primal_graph_hash: str
    input_bounds_hash: str
    objective_hash: str
    threshold_hash: str
    root_refinement_plan_hash: str
    root_refinement_semantic_trace_hash: str
    root_intermediate_bounds_hash: str
    optimizer_policy_hash: str
    search_budget: NativeBabSearchBudgetIR
    child_refinement_policy: NativeIntermediateRefinementPolicyIR
    whole_query_timeout_ns: int = 60 * 1_000_000_000
    semantics_owner: str = "boundflow_native_objective_ancestral_queue"
    schema_version: str = OBJECTIVE_ANCESTRAL_QUEUE_PLAN_IR_SCHEMA_VERSION

    def validate(self) -> None:
        self.search_budget.validate()
        self.child_refinement_policy.validate()
        if (
            self.schema_version != OBJECTIVE_ANCESTRAL_QUEUE_PLAN_IR_SCHEMA_VERSION
            or not self.plan_id
            or any(
                not _is_sha256(value)
                for value in (
                    self.primal_graph_hash,
                    self.input_bounds_hash,
                    self.objective_hash,
                    self.threshold_hash,
                    self.root_refinement_plan_hash,
                    self.root_refinement_semantic_trace_hash,
                    self.root_intermediate_bounds_hash,
                    self.optimizer_policy_hash,
                )
            )
            or (self.search_budget.max_nodes, self.search_budget.max_depth) != (31, 4)
            or self.child_refinement_policy.passes != 1
            or self.child_refinement_policy.max_neurons_per_relu != 128
            or self.child_refinement_policy.backward_chunk_size != 32
            or self.child_refinement_policy.candidate_policy_id
            != "objective_influence_width_per_relu_v1"
            or self.child_refinement_policy.refinement_method
            != "selected_plain_crown_v1"
            or self.whole_query_timeout_ns != 60 * 1_000_000_000
            or self.semantics_owner != "boundflow_native_objective_ancestral_queue"
        ):
            raise ValueError("objective ancestral queue Plan IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "primal_graph_hash": self.primal_graph_hash,
            "input_bounds_hash": self.input_bounds_hash,
            "objective_hash": self.objective_hash,
            "threshold_hash": self.threshold_hash,
            "root_refinement_plan_hash": self.root_refinement_plan_hash,
            "root_refinement_semantic_trace_hash": (
                self.root_refinement_semantic_trace_hash
            ),
            "root_intermediate_bounds_hash": self.root_intermediate_bounds_hash,
            "optimizer_policy_hash": self.optimizer_policy_hash,
            "search_budget": self.search_budget.to_dict(),
            "child_refinement_policy": self.child_refinement_policy.to_dict(),
            "whole_query_timeout_ns": self.whole_query_timeout_ns,
            "task_template": [kind.value for kind in ObjectiveAncestralQueueTaskKind],
            "node_dispatch": "serial_dynamic_parent_before_child",
            "source_consumption": "sound_constraint_only",
            "deadline_enforcement": "whole_query_cooperative_stage_boundaries",
            "semantics_owner": self.semantics_owner,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveAncestralQueueTaskIRUnit:
    """One committed dynamic root, child, transition, or emit task."""

    sequence: int
    task_id: str
    kind: ObjectiveAncestralQueueTaskKind
    dependency_task_ids: Tuple[str, ...]
    node_id: Optional[str]
    parent_node_id: Optional[str]
    node_split_state_hash: Optional[str]
    input_hashes: Tuple[Tuple[str, str], ...]
    output_hash: str

    def validate(self) -> None:
        node_owned = self.kind != ObjectiveAncestralQueueTaskKind.EMIT_RESULT
        child_owned = self.kind in {
            ObjectiveAncestralQueueTaskKind.COMPILE_CHILD_REFINEMENT,
            ObjectiveAncestralQueueTaskKind.EXECUTE_CHILD_REFINEMENT,
            ObjectiveAncestralQueueTaskKind.EVALUATE_CHILD,
        }
        if (
            self.sequence < 0
            or not self.task_id
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or node_owned != (self.node_id is not None)
            or (child_owned and self.parent_node_id is None)
            or (
                self.kind
                in {
                    ObjectiveAncestralQueueTaskKind.ADMIT_ROOT_SOURCE,
                    ObjectiveAncestralQueueTaskKind.EVALUATE_ROOT,
                }
                and self.parent_node_id is not None
            )
            or (not node_owned and self.parent_node_id is not None)
            or node_owned != _is_sha256(self.node_split_state_hash)
            or len(self.input_hashes)
            != len({name for name, _value in self.input_hashes})
            or any(
                not name or not _is_sha256(value) for name, value in self.input_hashes
            )
            or not _is_sha256(self.output_hash)
        ):
            raise ValueError("objective ancestral queue Task unit is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "dependency_task_ids": list(self.dependency_task_ids),
            "node_id": self.node_id,
            "parent_node_id": self.parent_node_id,
            "node_split_state_hash": self.node_split_state_hash,
            "input_hashes": dict(self.input_hashes),
            "output_hash": self.output_hash,
        }


@dataclass(frozen=True)
class NativeObjectiveAncestralQueueTaskIRModule:
    """Committed dynamic task instance graph."""

    plan_hash: str
    tasks: Tuple[NativeObjectiveAncestralQueueTaskIRUnit, ...]
    schema_version: str = OBJECTIVE_ANCESTRAL_QUEUE_TASK_IR_SCHEMA_VERSION

    def validate(self) -> None:
        task_ids = tuple(task.task_id for task in self.tasks)
        if (
            self.schema_version != OBJECTIVE_ANCESTRAL_QUEUE_TASK_IR_SCHEMA_VERSION
            or not _is_sha256(self.plan_hash)
            or len(self.tasks) < 4
            or len(task_ids) != len(set(task_ids))
            or tuple(task.sequence for task in self.tasks)
            != tuple(range(len(self.tasks)))
            or self.tasks[0].kind != ObjectiveAncestralQueueTaskKind.ADMIT_ROOT_SOURCE
            or self.tasks[1].kind != ObjectiveAncestralQueueTaskKind.EVALUATE_ROOT
            or self.tasks[-1].kind != ObjectiveAncestralQueueTaskKind.EMIT_RESULT
        ):
            raise ValueError("objective ancestral queue Task IR is invalid")
        available: set[str] = set()
        root_id = self.tasks[0].node_id
        node_kinds: dict[str, list[ObjectiveAncestralQueueTaskKind]] = {}
        parent_by_node: dict[str, Optional[str]] = {}
        for task in self.tasks:
            task.validate()
            if any(
                dependency not in available for dependency in task.dependency_task_ids
            ):
                raise ValueError("objective ancestral queue dependency order differs")
            available.add(task.task_id)
            if task.node_id is not None:
                node_kinds.setdefault(task.node_id, []).append(task.kind)
                parent_by_node.setdefault(task.node_id, task.parent_node_id)
                if parent_by_node[task.node_id] != task.parent_node_id:
                    raise ValueError("objective ancestral queue parent binding differs")
        if root_id is None or parent_by_node.get(root_id) is not None:
            raise ValueError("objective ancestral queue root binding differs")
        for node_id, kinds in node_kinds.items():
            if node_id == root_id:
                if kinds[0:2] != [
                    ObjectiveAncestralQueueTaskKind.ADMIT_ROOT_SOURCE,
                    ObjectiveAncestralQueueTaskKind.EVALUATE_ROOT,
                ]:
                    raise ValueError(
                        "objective ancestral queue root task order differs"
                    )
            elif kinds[0:3] != [
                ObjectiveAncestralQueueTaskKind.COMPILE_CHILD_REFINEMENT,
                ObjectiveAncestralQueueTaskKind.EXECUTE_CHILD_REFINEMENT,
                ObjectiveAncestralQueueTaskKind.EVALUATE_CHILD,
            ]:
                raise ValueError("objective ancestral queue child task order differs")
        expected_emit_dependencies = tuple(
            task.task_id
            for task in self.tasks[:-1]
            if task.kind
            in {
                ObjectiveAncestralQueueTaskKind.EVALUATE_ROOT,
                ObjectiveAncestralQueueTaskKind.EVALUATE_CHILD,
                ObjectiveAncestralQueueTaskKind.TRANSITION_QUEUE,
            }
        )
        if self.tasks[-1].dependency_task_ids != expected_emit_dependencies:
            raise ValueError(
                "objective ancestral queue emit dependencies differ from committed proof"
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveAncestralQueueScheduleActionIR:
    """One sequential dispatch bound exactly to a dynamic task."""

    sequence: int
    action_id: str
    task_id: str
    kind: ObjectiveAncestralQueueTaskKind

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
        }


@dataclass(frozen=True)
class NativeObjectiveAncestralQueueScheduleIR:
    """Sequential committed-proof schedule."""

    plan_hash: str
    task_ir_hash: str
    actions: Tuple[NativeObjectiveAncestralQueueScheduleActionIR, ...]
    schema_version: str = OBJECTIVE_ANCESTRAL_QUEUE_SCHEDULE_IR_SCHEMA_VERSION

    def validate_against(
        self, task_ir: NativeObjectiveAncestralQueueTaskIRModule
    ) -> None:
        task_ir.validate()
        if (
            self.schema_version != OBJECTIVE_ANCESTRAL_QUEUE_SCHEDULE_IR_SCHEMA_VERSION
            or self.plan_hash != task_ir.plan_hash
            or self.task_ir_hash != task_ir.stable_hash()
            or len(self.actions) != len(task_ir.tasks)
        ):
            raise ValueError("objective ancestral queue Schedule IR differs")
        for index, (action, task) in enumerate(zip(self.actions, task_ir.tasks)):
            if (
                action.sequence != index
                or action.action_id != f"objective-ancestral.launch.{index:04d}"
                or action.task_id != task.task_id
                or action.kind != task.kind
            ):
                raise ValueError(
                    "objective ancestral queue Schedule/Task binding differs"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "actions": [action.to_dict() for action in self.actions],
            "dispatch": "sequential_committed_dynamic_tasks",
        }

    def stable_hash(self, task_ir: NativeObjectiveAncestralQueueTaskIRModule) -> str:
        self.validate_against(task_ir)
        return _canonical_hash(self.to_dict())


def lower_native_objective_ancestral_queue_schedule(
    plan: NativeObjectiveAncestralQueuePlanIR,
    tasks: Tuple[NativeObjectiveAncestralQueueTaskIRUnit, ...],
) -> tuple[
    NativeObjectiveAncestralQueueTaskIRModule,
    NativeObjectiveAncestralQueueScheduleIR,
]:
    plan.validate()
    task_ir = NativeObjectiveAncestralQueueTaskIRModule(
        plan_hash=plan.stable_hash(), tasks=tasks
    )
    task_ir.validate()
    schedule = NativeObjectiveAncestralQueueScheduleIR(
        plan_hash=plan.stable_hash(),
        task_ir_hash=task_ir.stable_hash(),
        actions=tuple(
            NativeObjectiveAncestralQueueScheduleActionIR(
                sequence=index,
                action_id=f"objective-ancestral.launch.{index:04d}",
                task_id=task.task_id,
                kind=task.kind,
            )
            for index, task in enumerate(tasks)
        ),
    )
    schedule.validate_against(task_ir)
    return task_ir, schedule


__all__ = [
    "NativeObjectiveAncestralQueuePlanIR",
    "NativeObjectiveAncestralQueueScheduleIR",
    "NativeObjectiveAncestralQueueTaskIRModule",
    "NativeObjectiveAncestralQueueTaskIRUnit",
    "ObjectiveAncestralQueueTaskKind",
    "lower_native_objective_ancestral_queue_schedule",
]
