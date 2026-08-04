"""Shared-parametric objective-ancestral queue IR."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Optional, Tuple

from .objective_ancestral_sibling_pack import (
    NativeObjectiveAncestralSiblingPackPlanIR,
)

SHARED_PARAMETRIC_ANCESTRAL_PLAN_IR_SCHEMA_VERSION = (
    "boundflow.shared-parametric-ancestral-plan-ir/v1"
)
SHARED_PARAMETRIC_ANCESTRAL_BATCH_IR_SCHEMA_VERSION = (
    "boundflow.shared-parametric-ancestral-batch-ir/v1"
)
SHARED_PARAMETRIC_ANCESTRAL_TASK_IR_SCHEMA_VERSION = (
    "boundflow.shared-parametric-ancestral-task-ir/v1"
)
SHARED_PARAMETRIC_ANCESTRAL_SCHEDULE_IR_SCHEMA_VERSION = (
    "boundflow.shared-parametric-ancestral-schedule-ir/v1"
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


@dataclass(frozen=True)
class NativeSharedParametricAncestralPlanIR:
    """Query contract adding shared-template ownership to a frozen sibling plan."""

    plan_id: str
    sibling_pack_plan: NativeObjectiveAncestralSiblingPackPlanIR
    primal_graph_hash: str
    input_bounds_hash: str
    source_objective_hash: str
    evaluator_objective_hash: str
    threshold_hash: str
    root_refinement_semantic_trace_hash: str
    optimizer_policy_hash: str
    max_nodes: int
    max_depth: int
    child_refinement_cap: int
    whole_query_timeout_ns: int
    template_sharing_scope: str = "one_query_cross_batch_cross_clause_v1"
    template_contract_excludes: Tuple[str, ...] = (
        "objective_content",
        "split_state",
        "intermediate_bounds",
        "warm_state",
        "refinement_lineage",
        "batch_size",
    )
    evaluator_mode: str = "parametric_template_instance_no_native_reexecution"
    sibling_commit: str = "atomic_complete_pair_only"
    semantics_owner: str = "boundflow_native_shared_parametric_ancestral"
    schema_version: str = SHARED_PARAMETRIC_ANCESTRAL_PLAN_IR_SCHEMA_VERSION

    def validate(self) -> None:
        self.sibling_pack_plan.validate()
        hashes = (
            self.primal_graph_hash,
            self.input_bounds_hash,
            self.source_objective_hash,
            self.evaluator_objective_hash,
            self.threshold_hash,
            self.root_refinement_semantic_trace_hash,
            self.optimizer_policy_hash,
        )
        if (
            self.schema_version != SHARED_PARAMETRIC_ANCESTRAL_PLAN_IR_SCHEMA_VERSION
            or not self.plan_id
            or self.plan_id != self.sibling_pack_plan.plan_id
            or self.primal_graph_hash != self.sibling_pack_plan.primal_graph_hash
            or self.input_bounds_hash != self.sibling_pack_plan.input_bounds_hash
            or self.source_objective_hash
            != self.sibling_pack_plan.source_objective_hash
            or self.evaluator_objective_hash
            != self.sibling_pack_plan.evaluator_objective_hash
            or self.threshold_hash != self.sibling_pack_plan.threshold_hash
            or self.root_refinement_semantic_trace_hash
            != self.sibling_pack_plan.root_refinement_semantic_trace_hash
            or self.optimizer_policy_hash
            != self.sibling_pack_plan.optimizer_policy_hash
            or self.max_nodes != self.sibling_pack_plan.search_budget.max_nodes
            or self.max_depth != self.sibling_pack_plan.search_budget.max_depth
            or self.child_refinement_cap
            != self.sibling_pack_plan.child_refinement_policy.max_neurons_per_relu
            or self.whole_query_timeout_ns
            != self.sibling_pack_plan.whole_query_timeout_ns
            or any(not _is_sha256(value) for value in hashes)
            or self.source_objective_hash == self.evaluator_objective_hash
            or (self.max_nodes, self.max_depth) != (31, 4)
            or self.child_refinement_cap != 128
            or self.whole_query_timeout_ns != 60 * 1_000_000_000
            or self.template_sharing_scope != "one_query_cross_batch_cross_clause_v1"
            or self.template_contract_excludes
            != (
                "objective_content",
                "split_state",
                "intermediate_bounds",
                "warm_state",
                "refinement_lineage",
                "batch_size",
            )
            or self.evaluator_mode
            != "parametric_template_instance_no_native_reexecution"
            or self.sibling_commit != "atomic_complete_pair_only"
            or self.semantics_owner != "boundflow_native_shared_parametric_ancestral"
        ):
            raise ValueError("shared-parametric ancestral Plan IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "sibling_pack_plan": self.sibling_pack_plan.to_dict(),
            "sibling_pack_plan_hash": self.sibling_pack_plan.stable_hash(),
            "primal_graph_hash": self.primal_graph_hash,
            "input_bounds_hash": self.input_bounds_hash,
            "source_objective_hash": self.source_objective_hash,
            "evaluator_objective_hash": self.evaluator_objective_hash,
            "threshold_hash": self.threshold_hash,
            "root_refinement_semantic_trace_hash": (
                self.root_refinement_semantic_trace_hash
            ),
            "optimizer_policy_hash": self.optimizer_policy_hash,
            "max_nodes": self.max_nodes,
            "max_depth": self.max_depth,
            "child_refinement_cap": self.child_refinement_cap,
            "whole_query_timeout_ns": self.whole_query_timeout_ns,
            "template_sharing_scope": self.template_sharing_scope,
            "template_contract_excludes": list(self.template_contract_excludes),
            "evaluator_mode": self.evaluator_mode,
            "sibling_commit": self.sibling_commit,
            "semantics_owner": self.semantics_owner,
            "audit_hash_chain_constructed": False,
            "selected_native_reexecution": False,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeSharedParametricAncestralBatchIR:
    """Exact root or complete sibling-pair commit bound to compiler evidence."""

    batch_index: int
    batch_id: str
    commit_kind: str
    node_ids: Tuple[str, ...]
    parent_node_id: Optional[str]
    node_split_state_hashes: Tuple[str, ...]
    refinement_semantic_trace_hashes: Tuple[str, ...]
    production_batch_trace_hash: str
    compiler_batch_trace_hash: str
    cache_event_hash: str
    instance_hash: str
    template_hash: str
    optimizer_task_hash: str
    optimizer_schedule_hash: str
    evaluation_hashes: Tuple[str, ...]
    atomic_commit_hash: str
    selected_native_reexecution: bool = False
    schema_version: str = SHARED_PARAMETRIC_ANCESTRAL_BATCH_IR_SCHEMA_VERSION

    def semantic_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "batch_index": self.batch_index,
            "batch_id": self.batch_id,
            "commit_kind": self.commit_kind,
            "node_ids": list(self.node_ids),
            "parent_node_id": self.parent_node_id,
            "node_split_state_hashes": list(self.node_split_state_hashes),
            "refinement_semantic_trace_hashes": list(
                self.refinement_semantic_trace_hashes
            ),
            "production_batch_trace_hash": self.production_batch_trace_hash,
            "compiler_batch_trace_hash": self.compiler_batch_trace_hash,
            "cache_event_hash": self.cache_event_hash,
            "instance_hash": self.instance_hash,
            "template_hash": self.template_hash,
            "optimizer_task_hash": self.optimizer_task_hash,
            "optimizer_schedule_hash": self.optimizer_schedule_hash,
            "evaluation_hashes": list(self.evaluation_hashes),
            "selected_native_reexecution": self.selected_native_reexecution,
            "commit": "exact_batch_after_complete_execution",
        }

    def validate(self) -> None:
        hashes: tuple[object, ...] = (
            *self.node_split_state_hashes,
            *self.refinement_semantic_trace_hashes,
            self.production_batch_trace_hash,
            self.compiler_batch_trace_hash,
            self.cache_event_hash,
            self.instance_hash,
            self.template_hash,
            self.optimizer_task_hash,
            self.optimizer_schedule_hash,
            *self.evaluation_hashes,
        )
        root = self.commit_kind == "root"
        sibling = self.commit_kind == "atomic_sibling_pair"
        if (
            self.schema_version != SHARED_PARAMETRIC_ANCESTRAL_BATCH_IR_SCHEMA_VERSION
            or self.batch_index < 0
            or not self.batch_id
            or not (root or sibling)
            or not self.node_ids
            or len(self.node_ids) != len(set(self.node_ids))
            or len(self.node_ids) != len(self.node_split_state_hashes)
            or len(self.node_ids) != len(self.refinement_semantic_trace_hashes)
            or len(self.node_ids) != len(self.evaluation_hashes)
            or any(not _is_sha256(value) for value in hashes)
            or self.selected_native_reexecution is not False
            or (
                root
                and (
                    self.batch_index != 0
                    or len(self.node_ids) != 1
                    or self.parent_node_id is not None
                )
            )
            or (
                sibling
                and (
                    self.batch_index < 1
                    or len(self.node_ids) != 2
                    or not self.parent_node_id
                )
            )
            or self.atomic_commit_hash
            != _canonical_hash({**self.semantic_dict(), "atomic_commit_hash": None})
        ):
            raise ValueError("shared-parametric ancestral Batch IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {**self.semantic_dict(), "atomic_commit_hash": self.atomic_commit_hash}

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())

    @classmethod
    def committed(cls, **kwargs: object) -> "NativeSharedParametricAncestralBatchIR":
        provisional = cls(**kwargs, atomic_commit_hash="0" * 64)  # type: ignore[arg-type]
        commit_hash = _canonical_hash(
            {**provisional.semantic_dict(), "atomic_commit_hash": None}
        )
        value = cls(**kwargs, atomic_commit_hash=commit_hash)  # type: ignore[arg-type]
        value.validate()
        return value


class NativeSharedParametricAncestralTaskKind(Enum):
    """Closed vocabulary for shared-template batch ownership."""

    ADMIT_QUERY = "admit_query"
    ACQUIRE_TEMPLATE = "acquire_template"
    INSTANTIATE_BATCH = "instantiate_batch"
    EXECUTE_BATCH = "execute_batch"
    COMMIT_ROOT = "commit_root"
    COMMIT_SIBLING_PAIR = "commit_sibling_pair"
    TRANSITION_QUEUE = "transition_queue"
    EMIT_RESULT = "emit_result"


@dataclass(frozen=True)
class NativeSharedParametricAncestralTaskIRUnit:
    """One query, compiler-batch, queue transition, or result task."""

    sequence: int
    task_id: str
    kind: NativeSharedParametricAncestralTaskKind
    dependency_task_ids: Tuple[str, ...]
    batch_id: Optional[str]
    input_hashes: Tuple[Tuple[str, str], ...]
    output_hash: str

    def validate(self) -> None:
        batch_owned = self.kind not in {
            NativeSharedParametricAncestralTaskKind.ADMIT_QUERY,
            NativeSharedParametricAncestralTaskKind.EMIT_RESULT,
        }
        if (
            self.sequence < 0
            or not self.task_id
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or batch_owned != (self.batch_id is not None)
            or len(self.input_hashes)
            != len({name for name, _value in self.input_hashes})
            or any(
                not name or not _is_sha256(value) for name, value in self.input_hashes
            )
            or not _is_sha256(self.output_hash)
        ):
            raise ValueError("shared-parametric ancestral Task unit is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "dependency_task_ids": list(self.dependency_task_ids),
            "batch_id": self.batch_id,
            "input_hashes": dict(self.input_hashes),
            "output_hash": self.output_hash,
        }


@dataclass(frozen=True)
class NativeSharedParametricAncestralTaskIRModule:
    """Ordered task graph covering every committed dynamic batch."""

    plan_hash: str
    tasks: Tuple[NativeSharedParametricAncestralTaskIRUnit, ...]
    batch_hashes: Tuple[str, ...]
    schema_version: str = SHARED_PARAMETRIC_ANCESTRAL_TASK_IR_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != SHARED_PARAMETRIC_ANCESTRAL_TASK_IR_SCHEMA_VERSION
            or not _is_sha256(self.plan_hash)
            or not self.tasks
            or not self.batch_hashes
            or any(not _is_sha256(value) for value in self.batch_hashes)
            or self.tasks[0].kind != NativeSharedParametricAncestralTaskKind.ADMIT_QUERY
            or self.tasks[-1].kind
            != NativeSharedParametricAncestralTaskKind.EMIT_RESULT
            or tuple(task.sequence for task in self.tasks)
            != tuple(range(len(self.tasks)))
            or len({task.task_id for task in self.tasks}) != len(self.tasks)
        ):
            raise ValueError("shared-parametric ancestral Task IR is invalid")
        available: set[str] = set()
        batch_commit_count = 0
        for task in self.tasks:
            task.validate()
            if any(item not in available for item in task.dependency_task_ids):
                raise ValueError("shared-parametric Task dependency order differs")
            if task.kind in {
                NativeSharedParametricAncestralTaskKind.COMMIT_ROOT,
                NativeSharedParametricAncestralTaskKind.COMMIT_SIBLING_PAIR,
            }:
                batch_commit_count += 1
            available.add(task.task_id)
        if batch_commit_count != len(self.batch_hashes):
            raise ValueError("shared-parametric Task/Batch coverage differs")
        expected_emit = tuple(
            task.task_id
            for task in self.tasks[:-1]
            if task.kind
            in {
                NativeSharedParametricAncestralTaskKind.COMMIT_ROOT,
                NativeSharedParametricAncestralTaskKind.COMMIT_SIBLING_PAIR,
                NativeSharedParametricAncestralTaskKind.TRANSITION_QUEUE,
            }
        )
        if self.tasks[-1].dependency_task_ids != expected_emit:
            raise ValueError("shared-parametric emit dependencies differ")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "tasks": [task.to_dict() for task in self.tasks],
            "batch_hashes": list(self.batch_hashes),
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeSharedParametricAncestralScheduleActionIR:
    """One sequential launch bound exactly to a shared-parametric task."""

    sequence: int
    action_id: str
    task_id: str
    kind: NativeSharedParametricAncestralTaskKind

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
        }


@dataclass(frozen=True)
class NativeSharedParametricAncestralScheduleIR:
    """Sequential query schedule preserving atomic sibling commits."""

    plan_hash: str
    task_ir_hash: str
    actions: Tuple[NativeSharedParametricAncestralScheduleActionIR, ...]
    schema_version: str = SHARED_PARAMETRIC_ANCESTRAL_SCHEDULE_IR_SCHEMA_VERSION

    def validate_against(
        self, task_ir: NativeSharedParametricAncestralTaskIRModule
    ) -> None:
        task_ir.validate()
        if (
            self.schema_version
            != SHARED_PARAMETRIC_ANCESTRAL_SCHEDULE_IR_SCHEMA_VERSION
            or self.plan_hash != task_ir.plan_hash
            or self.task_ir_hash != task_ir.stable_hash()
            or len(self.actions) != len(task_ir.tasks)
        ):
            raise ValueError("shared-parametric ancestral Schedule IR differs")
        for index, (action, task) in enumerate(zip(self.actions, task_ir.tasks)):
            if (
                action.sequence != index
                or action.action_id != f"shared-parametric.launch.{index:04d}"
                or action.task_id != task.task_id
                or action.kind != task.kind
            ):
                raise ValueError("shared-parametric Schedule/Task binding differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "actions": [item.to_dict() for item in self.actions],
            "dispatch": "sequential_shared_template_atomic_sibling_batches",
        }

    def stable_hash(self, task_ir: NativeSharedParametricAncestralTaskIRModule) -> str:
        self.validate_against(task_ir)
        return _canonical_hash(self.to_dict())


def lower_native_shared_parametric_ancestral_schedule(
    plan: NativeSharedParametricAncestralPlanIR,
    tasks: Tuple[NativeSharedParametricAncestralTaskIRUnit, ...],
    batches: Tuple[NativeSharedParametricAncestralBatchIR, ...],
) -> tuple[
    NativeSharedParametricAncestralTaskIRModule,
    NativeSharedParametricAncestralScheduleIR,
]:
    plan.validate()
    for index, batch in enumerate(batches):
        batch.validate()
        if batch.batch_index != index:
            raise ValueError("shared-parametric Batch order differs")
    task_ir = NativeSharedParametricAncestralTaskIRModule(
        plan_hash=plan.stable_hash(),
        tasks=tasks,
        batch_hashes=tuple(batch.stable_hash() for batch in batches),
    )
    task_ir.validate()
    schedule = NativeSharedParametricAncestralScheduleIR(
        plan_hash=plan.stable_hash(),
        task_ir_hash=task_ir.stable_hash(),
        actions=tuple(
            NativeSharedParametricAncestralScheduleActionIR(
                sequence=index,
                action_id=f"shared-parametric.launch.{index:04d}",
                task_id=task.task_id,
                kind=task.kind,
            )
            for index, task in enumerate(tasks)
        ),
    )
    schedule.validate_against(task_ir)
    return task_ir, schedule


__all__ = [
    "NativeSharedParametricAncestralBatchIR",
    "NativeSharedParametricAncestralPlanIR",
    "NativeSharedParametricAncestralScheduleIR",
    "NativeSharedParametricAncestralTaskIRModule",
    "NativeSharedParametricAncestralTaskIRUnit",
    "NativeSharedParametricAncestralTaskKind",
    "lower_native_shared_parametric_ancestral_schedule",
]
