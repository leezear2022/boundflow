"""Typed sibling-packed objective-ancestral queue IR."""

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

SIBLING_PACK_PLAN_IR_SCHEMA_VERSION = "boundflow.sibling-pack-plan-ir/v1"
SIBLING_PACK_TASK_IR_SCHEMA_VERSION = "boundflow.sibling-pack-task-ir/v1"
SIBLING_PACK_SCHEDULE_IR_SCHEMA_VERSION = "boundflow.sibling-pack-schedule-ir/v1"
SIBLING_GROUP_EXECUTION_IR_SCHEMA_VERSION = "boundflow.sibling-group-execution-ir/v1"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class NativeObjectiveAncestralSiblingPackTaskKind(Enum):
    """Closed action vocabulary for admitted roots and atomic sibling groups."""

    ADMIT_ROOT_SOURCE = "admit_root_source"
    PROJECT_OBJECTIVE = "project_objective"
    EVALUATE_ROOT = "evaluate_root"
    TRANSITION_QUEUE = "transition_queue"
    COMPILE_CHILD_REFINEMENT = "compile_child_refinement"
    EXECUTE_CHILD_REFINEMENT = "execute_child_refinement"
    COMPILE_PACKED_EVALUATOR = "compile_packed_evaluator"
    EXECUTE_PACKED_EVALUATOR = "execute_packed_evaluator"
    EMIT_RESULT = "emit_result"


@dataclass(frozen=True)
class NativeObjectiveAncestralSiblingPackPlanIR:
    """Static typed-root, objective-projection, and sibling-group contract."""

    plan_id: str
    primal_graph_hash: str
    input_bounds_hash: str
    source_objective_hash: str
    evaluator_objective_hash: str
    threshold_hash: str
    root_refinement_plan_hash: str
    root_refinement_semantic_trace_hash: str
    root_intermediate_bounds_hash: str
    optimizer_policy_hash: str
    search_budget: NativeBabSearchBudgetIR
    child_refinement_policy: NativeIntermediateRefinementPolicyIR
    objective_projection: str = "drop_singleton_domain_axis_v1"
    sibling_group_size: int = 2
    expansion_batch_size: int = 1
    max_eval_batch_size: int = 2
    whole_query_timeout_ns: int = 60 * 1_000_000_000
    semantics_owner: str = "boundflow_native_objective_ancestral_sibling_pack"
    schema_version: str = SIBLING_PACK_PLAN_IR_SCHEMA_VERSION

    def validate(self) -> None:
        self.search_budget.validate()
        self.child_refinement_policy.validate()
        hashes = (
            self.primal_graph_hash,
            self.input_bounds_hash,
            self.source_objective_hash,
            self.evaluator_objective_hash,
            self.threshold_hash,
            self.root_refinement_plan_hash,
            self.root_refinement_semantic_trace_hash,
            self.root_intermediate_bounds_hash,
            self.optimizer_policy_hash,
        )
        if (
            self.schema_version != SIBLING_PACK_PLAN_IR_SCHEMA_VERSION
            or not self.plan_id
            or any(not _is_sha256(value) for value in hashes)
            or self.source_objective_hash == self.evaluator_objective_hash
            or (self.search_budget.max_nodes, self.search_budget.max_depth) != (31, 4)
            or self.child_refinement_policy.passes != 1
            or self.child_refinement_policy.max_neurons_per_relu != 128
            or self.child_refinement_policy.backward_chunk_size != 32
            or self.child_refinement_policy.candidate_policy_id
            != "objective_influence_width_per_relu_v1"
            or self.child_refinement_policy.refinement_method
            != "selected_plain_crown_v1"
            or self.objective_projection != "drop_singleton_domain_axis_v1"
            or self.sibling_group_size != 2
            or self.expansion_batch_size != 1
            or self.max_eval_batch_size != 2
            or self.whole_query_timeout_ns != 60 * 1_000_000_000
            or self.semantics_owner
            != "boundflow_native_objective_ancestral_sibling_pack"
        ):
            raise ValueError("objective ancestral sibling-pack Plan IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "primal_graph_hash": self.primal_graph_hash,
            "input_bounds_hash": self.input_bounds_hash,
            "source_objective_hash": self.source_objective_hash,
            "evaluator_objective_hash": self.evaluator_objective_hash,
            "threshold_hash": self.threshold_hash,
            "root_refinement_plan_hash": self.root_refinement_plan_hash,
            "root_refinement_semantic_trace_hash": (
                self.root_refinement_semantic_trace_hash
            ),
            "root_intermediate_bounds_hash": self.root_intermediate_bounds_hash,
            "optimizer_policy_hash": self.optimizer_policy_hash,
            "search_budget": self.search_budget.to_dict(),
            "child_refinement_policy": self.child_refinement_policy.to_dict(),
            "objective_projection": self.objective_projection,
            "sibling_group_size": self.sibling_group_size,
            "expansion_batch_size": self.expansion_batch_size,
            "max_eval_batch_size": self.max_eval_batch_size,
            "whole_query_timeout_ns": self.whole_query_timeout_ns,
            "source_consumption": "sound_constraint_only",
            "group_commit": "atomic_complete_sibling_pair_only",
            "semantics_owner": self.semantics_owner,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveAncestralSiblingGroupExecutionIR:
    """One complete same-parent (-1,+1) group and its exact proof identity."""

    group_id: str
    group_index: int
    parent_node_id: str
    parent_evaluation_hash: str
    child_node_ids: Tuple[str, str]
    child_branch_values: Tuple[int, int]
    child_split_state_hashes: Tuple[str, str]
    parent_refinement_plan_hash: str
    parent_refinement_semantic_trace_hash: str
    parent_final_intermediate_bounds_hash: str
    child_refinement_plan_hashes: Tuple[str, str]
    child_refinement_semantic_trace_hashes: Tuple[str, str]
    child_final_intermediate_bounds_hashes: Tuple[str, str]
    optimizer_ir_hash: str
    optimizer_execution_trace_hash: str
    native_ir_hash: str
    child_evaluation_hashes: Tuple[str, str]
    atomic_commit_hash: str
    schema_version: str = SIBLING_GROUP_EXECUTION_IR_SCHEMA_VERSION

    def semantic_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "group_id": self.group_id,
            "group_index": self.group_index,
            "parent_node_id": self.parent_node_id,
            "parent_evaluation_hash": self.parent_evaluation_hash,
            "child_node_ids": list(self.child_node_ids),
            "child_branch_values": list(self.child_branch_values),
            "child_split_state_hashes": list(self.child_split_state_hashes),
            "parent_refinement_plan_hash": self.parent_refinement_plan_hash,
            "parent_refinement_semantic_trace_hash": (
                self.parent_refinement_semantic_trace_hash
            ),
            "parent_final_intermediate_bounds_hash": (
                self.parent_final_intermediate_bounds_hash
            ),
            "child_refinement_plan_hashes": list(self.child_refinement_plan_hashes),
            "child_refinement_semantic_trace_hashes": list(
                self.child_refinement_semantic_trace_hashes
            ),
            "child_final_intermediate_bounds_hashes": list(
                self.child_final_intermediate_bounds_hashes
            ),
            "optimizer_ir_hash": self.optimizer_ir_hash,
            "optimizer_execution_trace_hash": self.optimizer_execution_trace_hash,
            "native_ir_hash": self.native_ir_hash,
            "child_evaluation_hashes": list(self.child_evaluation_hashes),
            "commit": "atomic_complete_sibling_pair_only",
        }

    def validate(self) -> None:
        hashes: tuple[object, ...] = (
            self.parent_evaluation_hash,
            *self.child_split_state_hashes,
            self.parent_refinement_plan_hash,
            self.parent_refinement_semantic_trace_hash,
            self.parent_final_intermediate_bounds_hash,
            *self.child_refinement_plan_hashes,
            *self.child_refinement_semantic_trace_hashes,
            *self.child_final_intermediate_bounds_hashes,
            self.optimizer_ir_hash,
            self.optimizer_execution_trace_hash,
            self.native_ir_hash,
            *self.child_evaluation_hashes,
        )
        if (
            self.schema_version != SIBLING_GROUP_EXECUTION_IR_SCHEMA_VERSION
            or not self.group_id
            or self.group_index < 0
            or not self.parent_node_id
            or len(set(self.child_node_ids)) != 2
            or not all(self.child_node_ids)
            or self.child_branch_values != (-1, 1)
            or len(set(self.child_split_state_hashes)) != 2
            or any(not _is_sha256(value) for value in hashes)
            or self.atomic_commit_hash
            != _canonical_hash({**self.semantic_dict(), "atomic_commit_hash": None})
        ):
            raise ValueError("objective ancestral sibling-group execution is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self.semantic_dict(), "atomic_commit_hash": self.atomic_commit_hash}

    @classmethod
    def committed(
        cls, **kwargs: object
    ) -> "NativeObjectiveAncestralSiblingGroupExecutionIR":
        provisional = cls(**kwargs, atomic_commit_hash="0" * 64)  # type: ignore[arg-type]
        commit_hash = _canonical_hash(
            {**provisional.semantic_dict(), "atomic_commit_hash": None}
        )
        value = cls(**kwargs, atomic_commit_hash=commit_hash)  # type: ignore[arg-type]
        value.validate()
        return value


@dataclass(frozen=True)
class NativeObjectiveAncestralSiblingPackTaskIRUnit:
    """One root, queue, child, packed evaluator, or emit action."""

    sequence: int
    task_id: str
    kind: NativeObjectiveAncestralSiblingPackTaskKind
    dependency_task_ids: Tuple[str, ...]
    group_id: Optional[str]
    node_ids: Tuple[str, ...]
    input_hashes: Tuple[Tuple[str, str], ...]
    output_hash: str

    def validate(self) -> None:
        group_owned = self.kind in {
            NativeObjectiveAncestralSiblingPackTaskKind.COMPILE_CHILD_REFINEMENT,
            NativeObjectiveAncestralSiblingPackTaskKind.EXECUTE_CHILD_REFINEMENT,
            NativeObjectiveAncestralSiblingPackTaskKind.COMPILE_PACKED_EVALUATOR,
            NativeObjectiveAncestralSiblingPackTaskKind.EXECUTE_PACKED_EVALUATOR,
        }
        if (
            self.sequence < 0
            or not self.task_id
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or group_owned != (self.group_id is not None)
            or not self.node_ids
            or len(self.node_ids) != len(set(self.node_ids))
            or len(self.input_hashes)
            != len({name for name, _value in self.input_hashes})
            or any(
                not name or not _is_sha256(value) for name, value in self.input_hashes
            )
            or not _is_sha256(self.output_hash)
        ):
            raise ValueError("objective ancestral sibling-pack Task unit is invalid")
        if group_owned:
            expected_nodes = (
                1
                if self.kind
                in {
                    NativeObjectiveAncestralSiblingPackTaskKind.COMPILE_CHILD_REFINEMENT,
                    NativeObjectiveAncestralSiblingPackTaskKind.EXECUTE_CHILD_REFINEMENT,
                }
                else 2
            )
            if len(self.node_ids) != expected_nodes:
                raise ValueError("sibling-pack Task group node coverage differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "dependency_task_ids": list(self.dependency_task_ids),
            "group_id": self.group_id,
            "node_ids": list(self.node_ids),
            "input_hashes": dict(self.input_hashes),
            "output_hash": self.output_hash,
        }


@dataclass(frozen=True)
class NativeObjectiveAncestralSiblingPackTaskIRModule:
    """Committed root plus atomic sibling-group task graph."""

    plan_hash: str
    tasks: Tuple[NativeObjectiveAncestralSiblingPackTaskIRUnit, ...]
    schema_version: str = SIBLING_PACK_TASK_IR_SCHEMA_VERSION

    def validate(self) -> None:
        task_ids = tuple(task.task_id for task in self.tasks)
        if (
            self.schema_version != SIBLING_PACK_TASK_IR_SCHEMA_VERSION
            or not _is_sha256(self.plan_hash)
            or len(self.tasks) < 4
            or len(task_ids) != len(set(task_ids))
            or tuple(task.sequence for task in self.tasks)
            != tuple(range(len(self.tasks)))
            or tuple(task.kind for task in self.tasks[:3])
            != (
                NativeObjectiveAncestralSiblingPackTaskKind.ADMIT_ROOT_SOURCE,
                NativeObjectiveAncestralSiblingPackTaskKind.PROJECT_OBJECTIVE,
                NativeObjectiveAncestralSiblingPackTaskKind.EVALUATE_ROOT,
            )
            or self.tasks[-1].kind
            != NativeObjectiveAncestralSiblingPackTaskKind.EMIT_RESULT
        ):
            raise ValueError("objective ancestral sibling-pack Task IR is invalid")
        available: set[str] = set()
        for task in self.tasks:
            task.validate()
            if any(item not in available for item in task.dependency_task_ids):
                raise ValueError("sibling-pack Task dependency order differs")
            available.add(task.task_id)
        expected_emit = tuple(
            task.task_id
            for task in self.tasks[:-1]
            if task.kind
            in {
                NativeObjectiveAncestralSiblingPackTaskKind.EVALUATE_ROOT,
                NativeObjectiveAncestralSiblingPackTaskKind.TRANSITION_QUEUE,
                NativeObjectiveAncestralSiblingPackTaskKind.EXECUTE_PACKED_EVALUATOR,
            }
        )
        if self.tasks[-1].dependency_task_ids != expected_emit:
            raise ValueError("sibling-pack emit dependencies differ")

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
class NativeObjectiveAncestralSiblingPackScheduleActionIR:
    """One sequential launch bound exactly to a sibling-pack Task."""

    sequence: int
    action_id: str
    task_id: str
    kind: NativeObjectiveAncestralSiblingPackTaskKind

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
        }


@dataclass(frozen=True)
class NativeObjectiveAncestralSiblingPackScheduleIR:
    """Sequential committed schedule for root and sibling groups."""

    plan_hash: str
    task_ir_hash: str
    actions: Tuple[NativeObjectiveAncestralSiblingPackScheduleActionIR, ...]
    schema_version: str = SIBLING_PACK_SCHEDULE_IR_SCHEMA_VERSION

    def validate_against(
        self, task_ir: NativeObjectiveAncestralSiblingPackTaskIRModule
    ) -> None:
        task_ir.validate()
        if (
            self.schema_version != SIBLING_PACK_SCHEDULE_IR_SCHEMA_VERSION
            or self.plan_hash != task_ir.plan_hash
            or self.task_ir_hash != task_ir.stable_hash()
            or len(self.actions) != len(task_ir.tasks)
        ):
            raise ValueError("objective ancestral sibling-pack Schedule IR differs")
        for index, (action, task) in enumerate(zip(self.actions, task_ir.tasks)):
            if (
                action.sequence != index
                or action.action_id != f"sibling-pack.launch.{index:04d}"
                or action.task_id != task.task_id
                or action.kind != task.kind
            ):
                raise ValueError("sibling-pack Schedule/Task binding differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "actions": [action.to_dict() for action in self.actions],
            "dispatch": "sequential_root_atomic_sibling_groups",
        }

    def stable_hash(
        self, task_ir: NativeObjectiveAncestralSiblingPackTaskIRModule
    ) -> str:
        self.validate_against(task_ir)
        return _canonical_hash(self.to_dict())


def lower_native_objective_ancestral_sibling_pack_schedule(
    plan: NativeObjectiveAncestralSiblingPackPlanIR,
    tasks: Tuple[NativeObjectiveAncestralSiblingPackTaskIRUnit, ...],
) -> tuple[
    NativeObjectiveAncestralSiblingPackTaskIRModule,
    NativeObjectiveAncestralSiblingPackScheduleIR,
]:
    plan.validate()
    task_ir = NativeObjectiveAncestralSiblingPackTaskIRModule(
        plan_hash=plan.stable_hash(), tasks=tasks
    )
    task_ir.validate()
    schedule = NativeObjectiveAncestralSiblingPackScheduleIR(
        plan_hash=plan.stable_hash(),
        task_ir_hash=task_ir.stable_hash(),
        actions=tuple(
            NativeObjectiveAncestralSiblingPackScheduleActionIR(
                sequence=index,
                action_id=f"sibling-pack.launch.{index:04d}",
                task_id=task.task_id,
                kind=task.kind,
            )
            for index, task in enumerate(tasks)
        ),
    )
    schedule.validate_against(task_ir)
    return task_ir, schedule


__all__ = [
    "NativeObjectiveAncestralSiblingGroupExecutionIR",
    "NativeObjectiveAncestralSiblingPackPlanIR",
    "NativeObjectiveAncestralSiblingPackScheduleIR",
    "NativeObjectiveAncestralSiblingPackTaskIRModule",
    "NativeObjectiveAncestralSiblingPackTaskIRUnit",
    "NativeObjectiveAncestralSiblingPackTaskKind",
    "lower_native_objective_ancestral_sibling_pack_schedule",
]
