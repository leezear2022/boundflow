"""Typed Phase-A evidence for objective-branch scorer ownership."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=too-many-arguments,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Tuple

SCORER_OWNERSHIP_PLAN_SCHEMA_VERSION = (
    "boundflow.objective-branch-scorer-ownership-plan/v1"
)
SCORER_OWNERSHIP_ROW_SCHEMA_VERSION = (
    "boundflow.objective-branch-scorer-ownership-row/v1"
)
SCORER_OWNERSHIP_PARITY_SCHEMA_VERSION = (
    "boundflow.objective-branch-scorer-ownership-parity/v1"
)
SCORER_OWNERSHIP_DECISION_SCHEMA_VERSION = (
    "boundflow.objective-branch-scorer-ownership-decision/v1"
)
SCORER_OWNERSHIP_TASK_SCHEMA_VERSION = (
    "boundflow.objective-branch-scorer-ownership-task/v1"
)
SCORER_OWNERSHIP_SCHEDULE_SCHEMA_VERSION = (
    "boundflow.objective-branch-scorer-ownership-schedule/v1"
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
class NativeObjectiveBranchScorerOwnershipPlanIR:
    """Preregistered fixed-31 paired timing and ownership contract."""

    plan_id: str
    source_cost_formal_hash: str
    clause_ordinals: Tuple[int, ...] = (2, 3)
    paired_orders: Tuple[Tuple[str, ...], ...] = (
        ("historical", "prevalidated"),
        ("prevalidated", "historical"),
        ("historical", "prevalidated"),
    )
    required_nodes: int = 31
    required_sibling_groups: int = 15
    historical_enumerations_per_clause: int = 341
    prevalidated_compile_enumerations_per_clause: int = 31
    prevalidated_execute_enumerations_per_clause: int = 0
    maximum_queue_median_ratio: float = 0.75
    torch_threads: int = 8
    semantics_owner: str = "boundflow_native_objective_branch_scorer_ownership"
    performance_claimed: bool = False
    schema_version: str = SCORER_OWNERSHIP_PLAN_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != SCORER_OWNERSHIP_PLAN_SCHEMA_VERSION
            or not self.plan_id
            or not _is_sha256(self.source_cost_formal_hash)
            or self.clause_ordinals != (2, 3)
            or self.paired_orders
            != (
                ("historical", "prevalidated"),
                ("prevalidated", "historical"),
                ("historical", "prevalidated"),
            )
            or self.required_nodes != 31
            or self.required_sibling_groups != 15
            or self.historical_enumerations_per_clause != 341
            or self.prevalidated_compile_enumerations_per_clause != 31
            or self.prevalidated_execute_enumerations_per_clause != 0
            or self.maximum_queue_median_ratio != 0.75
            or self.torch_threads != 8
            or self.semantics_owner
            != "boundflow_native_objective_branch_scorer_ownership"
            or self.performance_claimed is not False
        ):
            raise ValueError("objective branch scorer ownership Plan differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "source_cost_formal_hash": self.source_cost_formal_hash,
            "clause_ordinals": list(self.clause_ordinals),
            "paired_orders": [list(value) for value in self.paired_orders],
            "required_nodes": self.required_nodes,
            "required_sibling_groups": self.required_sibling_groups,
            "historical_enumerations_per_clause": (
                self.historical_enumerations_per_clause
            ),
            "prevalidated_compile_enumerations_per_clause": (
                self.prevalidated_compile_enumerations_per_clause
            ),
            "prevalidated_execute_enumerations_per_clause": (
                self.prevalidated_execute_enumerations_per_clause
            ),
            "maximum_queue_median_ratio": self.maximum_queue_median_ratio,
            "torch_threads": self.torch_threads,
            "semantics_owner": self.semantics_owner,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveBranchScorerOwnershipRowIR:
    """One raw fixed-31 production queue measurement."""

    plan_hash: str
    repeat_index: int
    original_clause_index: int
    mode: str
    order_position: int
    queue_elapsed_ns: int
    whole_elapsed_ns: int
    accepted_nodes: int
    sibling_group_count: int
    branch_execution_count: int
    enumeration_call_count: int
    compile_enumeration_count: int
    execute_enumeration_count: int
    queue_semantic_hash: str
    branch_semantic_hash: str
    capsule_table_hash: str
    performance_claimed: bool = False
    schema_version: str = SCORER_OWNERSHIP_ROW_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != SCORER_OWNERSHIP_ROW_SCHEMA_VERSION
            or not _is_sha256(self.plan_hash)
            or self.repeat_index not in {0, 1, 2}
            or self.original_clause_index not in {2, 3}
            or self.mode not in {"historical", "prevalidated"}
            or self.order_position not in {0, 1}
            or min(self.queue_elapsed_ns, self.whole_elapsed_ns) <= 0
            or self.whole_elapsed_ns < self.queue_elapsed_ns
            or self.accepted_nodes != 31
            or self.sibling_group_count != 15
            or self.branch_execution_count != 31
            or any(
                not _is_sha256(value)
                for value in (
                    self.queue_semantic_hash,
                    self.branch_semantic_hash,
                    self.capsule_table_hash,
                )
            )
            or self.performance_claimed is not False
        ):
            raise ValueError("objective branch scorer ownership row differs")
        expected = (341, 0, 0) if self.mode == "historical" else (31, 31, 0)
        if (
            self.enumeration_call_count,
            self.compile_enumeration_count,
            self.execute_enumeration_count,
        ) != expected:
            raise ValueError("objective branch scorer enumeration ownership differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "repeat_index": self.repeat_index,
            "original_clause_index": self.original_clause_index,
            "mode": self.mode,
            "order_position": self.order_position,
            "queue_elapsed_ns": self.queue_elapsed_ns,
            "whole_elapsed_ns": self.whole_elapsed_ns,
            "accepted_nodes": self.accepted_nodes,
            "sibling_group_count": self.sibling_group_count,
            "branch_execution_count": self.branch_execution_count,
            "enumeration_call_count": self.enumeration_call_count,
            "compile_enumeration_count": self.compile_enumeration_count,
            "execute_enumeration_count": self.execute_enumeration_count,
            "queue_semantic_hash": self.queue_semantic_hash,
            "branch_semantic_hash": self.branch_semantic_hash,
            "capsule_table_hash": self.capsule_table_hash,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveBranchScorerParityIR:
    """Exact old/new queue, branch, state, and refinement comparison."""

    plan_hash: str
    repeat_index: int
    original_clause_index: int
    historical_row_hash: str
    prevalidated_row_hash: str
    queue_semantic_hash: str
    branch_semantic_hash: str
    selected_state_hash: str
    refinement_semantic_hash: str
    exact: bool
    performance_claimed: bool = False
    schema_version: str = SCORER_OWNERSHIP_PARITY_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != SCORER_OWNERSHIP_PARITY_SCHEMA_VERSION
            or self.repeat_index not in {0, 1, 2}
            or self.original_clause_index not in {2, 3}
            or any(
                not _is_sha256(value)
                for value in (
                    self.plan_hash,
                    self.historical_row_hash,
                    self.prevalidated_row_hash,
                    self.queue_semantic_hash,
                    self.branch_semantic_hash,
                    self.selected_state_hash,
                    self.refinement_semantic_hash,
                )
            )
            or self.exact is not True
            or self.performance_claimed is not False
        ):
            raise ValueError("objective branch scorer exact parity differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "repeat_index": self.repeat_index,
            "original_clause_index": self.original_clause_index,
            "historical_row_hash": self.historical_row_hash,
            "prevalidated_row_hash": self.prevalidated_row_hash,
            "queue_semantic_hash": self.queue_semantic_hash,
            "branch_semantic_hash": self.branch_semantic_hash,
            "selected_state_hash": self.selected_state_hash,
            "refinement_semantic_hash": self.refinement_semantic_hash,
            "exact": self.exact,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveBranchScorerClauseMetricIR:
    """Three-repeat median/MAD reduction for one clause."""

    original_clause_index: int
    historical_queue_median_ns: int
    prevalidated_queue_median_ns: int
    historical_queue_mad_ns: int
    prevalidated_queue_mad_ns: int
    median_ratio: float
    median_improvement_ns: int
    ratio_passed: bool
    mad_passed: bool

    def validate(self) -> None:
        expected_ratio = (
            self.prevalidated_queue_median_ns / self.historical_queue_median_ns
        )
        if (
            self.original_clause_index not in {2, 3}
            or min(
                self.historical_queue_median_ns,
                self.prevalidated_queue_median_ns,
            )
            <= 0
            or min(self.historical_queue_mad_ns, self.prevalidated_queue_mad_ns) < 0
            or not math.isclose(self.median_ratio, expected_ratio, abs_tol=1e-12)
            or self.median_improvement_ns
            != self.historical_queue_median_ns - self.prevalidated_queue_median_ns
            or self.ratio_passed != (self.median_ratio <= 0.75)
            or self.mad_passed
            != (
                self.median_improvement_ns
                > max(
                    self.historical_queue_mad_ns,
                    self.prevalidated_queue_mad_ns,
                )
            )
        ):
            raise ValueError("objective branch scorer clause metric differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "original_clause_index": self.original_clause_index,
            "historical_queue_median_ns": self.historical_queue_median_ns,
            "prevalidated_queue_median_ns": self.prevalidated_queue_median_ns,
            "historical_queue_mad_ns": self.historical_queue_mad_ns,
            "prevalidated_queue_mad_ns": self.prevalidated_queue_mad_ns,
            "median_ratio": self.median_ratio,
            "median_improvement_ns": self.median_improvement_ns,
            "ratio_passed": self.ratio_passed,
            "mad_passed": self.mad_passed,
        }


@dataclass(frozen=True)
class NativeObjectiveBranchScorerOwnershipDecisionIR:
    """Automatic Phase-A gate and conditional Phase-B route."""

    plan_hash: str
    clause_metrics: Tuple[NativeObjectiveBranchScorerClauseMetricIR, ...]
    parity_passed: bool
    enumeration_ownership_passed: bool
    internal_cost_passed: bool
    phase_a_go: bool
    next_route: str
    reason: str
    performance_claimed: bool = False
    schema_version: str = SCORER_OWNERSHIP_DECISION_SCHEMA_VERSION

    def validate(self) -> None:
        for metric in self.clause_metrics:
            metric.validate()
        expected_cost = all(
            metric.ratio_passed and metric.mad_passed for metric in self.clause_metrics
        )
        expected_go = (
            self.parity_passed and self.enumeration_ownership_passed and expected_cost
        )
        if (
            self.schema_version != SCORER_OWNERSHIP_DECISION_SCHEMA_VERSION
            or not _is_sha256(self.plan_hash)
            or tuple(metric.original_clause_index for metric in self.clause_metrics)
            != (2, 3)
            or self.parity_passed is not True
            or self.enumeration_ownership_passed is not True
            or self.internal_cost_passed != expected_cost
            or self.phase_a_go != expected_go
            or self.next_route
            != ("run_phase_b_global_60s" if expected_go else "close_nrir42_no_go")
            or not self.reason
            or self.performance_claimed is not False
        ):
            raise ValueError("objective branch scorer ownership Decision differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "clause_metrics": [metric.to_dict() for metric in self.clause_metrics],
            "parity_passed": self.parity_passed,
            "enumeration_ownership_passed": self.enumeration_ownership_passed,
            "internal_cost_passed": self.internal_cost_passed,
            "phase_a_go": self.phase_a_go,
            "next_route": self.next_route,
            "reason": self.reason,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


class NativeObjectiveBranchScorerOwnershipTaskKind(Enum):
    COMPILE_CAPSULES = "compile_capsules"
    EXECUTE_PAIRED_QUEUES = "execute_paired_queues"
    VALIDATE_EXACT_PARITY = "validate_exact_parity"
    REDUCE_COST = "reduce_cost"
    EMIT_DECISION = "emit_decision"


@dataclass(frozen=True)
class NativeObjectiveBranchScorerOwnershipTaskIR:
    task_id: str
    kind: NativeObjectiveBranchScorerOwnershipTaskKind
    dependency_task_ids: Tuple[str, ...]
    output_hash: str
    semantics_owner: str = "boundflow_native_objective_branch_scorer_ownership"

    def validate(self) -> None:
        if (
            not self.task_id
            or not _is_sha256(self.output_hash)
            or self.semantics_owner
            != "boundflow_native_objective_branch_scorer_ownership"
        ):
            raise ValueError("objective branch scorer ownership Task differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "dependency_task_ids": list(self.dependency_task_ids),
            "output_hash": self.output_hash,
            "semantics_owner": self.semantics_owner,
        }


@dataclass(frozen=True)
class NativeObjectiveBranchScorerOwnershipTaskIRModule:
    plan_hash: str
    tasks: Tuple[NativeObjectiveBranchScorerOwnershipTaskIR, ...]
    schema_version: str = SCORER_OWNERSHIP_TASK_SCHEMA_VERSION

    def validate(self) -> None:
        completed: set[str] = set()
        for task in self.tasks:
            task.validate()
            if any(item not in completed for item in task.dependency_task_ids):
                raise ValueError("objective branch scorer ownership dependency differs")
            completed.add(task.task_id)
        if (
            self.schema_version != SCORER_OWNERSHIP_TASK_SCHEMA_VERSION
            or not _is_sha256(self.plan_hash)
            or tuple(task.kind for task in self.tasks)
            != tuple(NativeObjectiveBranchScorerOwnershipTaskKind)
        ):
            raise ValueError("objective branch scorer ownership Task module differs")

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
class NativeObjectiveBranchScorerOwnershipScheduleIR:
    task_module_hash: str
    launch_task_ids: Tuple[str, ...]
    schema_version: str = SCORER_OWNERSHIP_SCHEDULE_SCHEMA_VERSION

    def validate(
        self, task_module: NativeObjectiveBranchScorerOwnershipTaskIRModule
    ) -> None:
        task_module.validate()
        if (
            self.schema_version != SCORER_OWNERSHIP_SCHEDULE_SCHEMA_VERSION
            or self.task_module_hash != task_module.stable_hash()
            or self.launch_task_ids != tuple(task.task_id for task in task_module.tasks)
        ):
            raise ValueError("objective branch scorer ownership Schedule differs")

    def to_dict(
        self, task_module: NativeObjectiveBranchScorerOwnershipTaskIRModule
    ) -> dict[str, object]:
        self.validate(task_module)
        return {
            "schema_version": self.schema_version,
            "task_module_hash": self.task_module_hash,
            "launch_task_ids": list(self.launch_task_ids),
        }

    def stable_hash(
        self, task_module: NativeObjectiveBranchScorerOwnershipTaskIRModule
    ) -> str:
        return _canonical_hash(self.to_dict(task_module))


def lower_native_objective_branch_scorer_ownership_schedule(
    plan: NativeObjectiveBranchScorerOwnershipPlanIR,
    *,
    capsule_table_hash: str,
    paired_rows_hash: str,
    parity_rows_hash: str,
    metrics_hash: str,
    decision_hash: str,
) -> tuple[
    NativeObjectiveBranchScorerOwnershipTaskIRModule,
    NativeObjectiveBranchScorerOwnershipScheduleIR,
]:
    plan.validate()
    outputs = (
        capsule_table_hash,
        paired_rows_hash,
        parity_rows_hash,
        metrics_hash,
        decision_hash,
    )
    tasks: list[NativeObjectiveBranchScorerOwnershipTaskIR] = []
    dependencies: tuple[str, ...] = ()
    for kind, output_hash in zip(NativeObjectiveBranchScorerOwnershipTaskKind, outputs):
        task = NativeObjectiveBranchScorerOwnershipTaskIR(
            task_id=f"objective_branch_scorer_ownership.{kind.value}",
            kind=kind,
            dependency_task_ids=dependencies,
            output_hash=output_hash,
        )
        tasks.append(task)
        dependencies = (task.task_id,)
    task_module = NativeObjectiveBranchScorerOwnershipTaskIRModule(
        plan_hash=plan.stable_hash(), tasks=tuple(tasks)
    )
    task_module.validate()
    schedule = NativeObjectiveBranchScorerOwnershipScheduleIR(
        task_module_hash=task_module.stable_hash(),
        launch_task_ids=tuple(task.task_id for task in tasks),
    )
    schedule.validate(task_module)
    return task_module, schedule


__all__ = [
    "NativeObjectiveBranchScorerClauseMetricIR",
    "NativeObjectiveBranchScorerOwnershipDecisionIR",
    "NativeObjectiveBranchScorerOwnershipPlanIR",
    "NativeObjectiveBranchScorerOwnershipRowIR",
    "NativeObjectiveBranchScorerParityIR",
    "lower_native_objective_branch_scorer_ownership_schedule",
]
