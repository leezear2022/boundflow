"""Typed staged-control IR for unresolved-clause verifier escalation."""

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

HARD_CLAUSE_ESCALATION_PLAN_IR_SCHEMA_VERSION = (
    "boundflow.hard-clause-escalation-plan-ir/v1"
)
HARD_CLAUSE_ESCALATION_DECISION_IR_SCHEMA_VERSION = (
    "boundflow.hard-clause-escalation-decision-ir/v1"
)
HARD_CLAUSE_ESCALATION_TASK_IR_SCHEMA_VERSION = (
    "boundflow.hard-clause-escalation-task-ir/v1"
)
HARD_CLAUSE_ESCALATION_SCHEDULE_IR_SCHEMA_VERSION = (
    "boundflow.hard-clause-escalation-schedule-ir/v1"
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


class HardClauseEscalationTaskKind(Enum):
    """Closed baseline-to-aggregate stage graph."""

    EXECUTE_BASELINE = "execute_baseline"
    ADMIT_HARD_CLAUSES = "admit_hard_clauses"
    COMPILE_REFINEMENT = "compile_refinement"
    EXECUTE_REFINEMENT = "execute_refinement"
    PROJECT_HARD_CLAUSES = "project_hard_clauses"
    EXECUTE_ESCALATION = "execute_escalation"
    AGGREGATE_VERDICTS = "aggregate_verdicts"
    EMIT_RESULT = "emit_result"


@dataclass(frozen=True)
class NativeHardClauseEscalationPlanIR:
    """Static staged-verifier contract for one exact query."""

    plan_id: str
    primal_graph_hash: str
    input_bounds_hash: str
    objective_matrix_hash: str
    thresholds_hash: str
    clause_count: int
    whole_query_timeout_ns: int
    baseline_budget: NativeBabSearchBudgetIR
    escalation_budget: NativeBabSearchBudgetIR
    refinement_policy: NativeIntermediateRefinementPolicyIR
    search_policy_hash: str
    optimizer_policy_hash: str
    semantics_owner: str = "boundflow_native_hard_clause_escalation"
    schema_version: str = HARD_CLAUSE_ESCALATION_PLAN_IR_SCHEMA_VERSION

    def validate(self) -> None:
        self.baseline_budget.validate()
        self.escalation_budget.validate()
        self.refinement_policy.validate()
        if (
            self.schema_version != HARD_CLAUSE_ESCALATION_PLAN_IR_SCHEMA_VERSION
            or not self.plan_id
            or any(
                not _is_sha256(value)
                for value in (
                    self.primal_graph_hash,
                    self.input_bounds_hash,
                    self.objective_matrix_hash,
                    self.thresholds_hash,
                    self.search_policy_hash,
                    self.optimizer_policy_hash,
                )
            )
            or self.clause_count < 1
            or self.whole_query_timeout_ns != 60 * 1_000_000_000
            or (self.baseline_budget.max_nodes, self.baseline_budget.max_depth)
            != (7, 2)
            or (self.escalation_budget.max_nodes, self.escalation_budget.max_depth)
            != (31, 4)
            or self.refinement_policy.passes != 1
            or self.refinement_policy.max_neurons_per_relu != 128
            or self.refinement_policy.backward_chunk_size != 32
            or self.refinement_policy.candidate_policy_id
            != "top_ambiguous_width_per_relu_v1"
            or self.refinement_policy.refinement_method != "selected_plain_crown_v1"
            or self.semantics_owner != "boundflow_native_hard_clause_escalation"
        ):
            raise ValueError("hard-clause escalation Plan IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "primal_graph_hash": self.primal_graph_hash,
            "input_bounds_hash": self.input_bounds_hash,
            "objective_matrix_hash": self.objective_matrix_hash,
            "thresholds_hash": self.thresholds_hash,
            "clause_count": self.clause_count,
            "whole_query_timeout_ns": self.whole_query_timeout_ns,
            "baseline_budget": self.baseline_budget.to_dict(),
            "escalation_budget": self.escalation_budget.to_dict(),
            "refinement_policy": self.refinement_policy.to_dict(),
            "search_policy_hash": self.search_policy_hash,
            "optimizer_policy_hash": self.optimizer_policy_hash,
            "baseline_intermediate_bound_source": "local_forward",
            "escalation_intermediate_bound_source": "native_refined",
            "admission": "exact_baseline_unresolved_ordinals",
            "fallback": "preserve_baseline_verdicts",
            "semantics_owner": self.semantics_owner,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeHardClauseEscalationDecisionIR:
    """Dynamic admission result derived only from a validated baseline query."""

    decision_id: str
    plan_hash: str
    baseline_query_trace_hash: str
    clause_count: int
    baseline_completed_clause_indices: Tuple[int, ...]
    baseline_verified_clause_indices: Tuple[int, ...]
    baseline_unresolved_clause_indices: Tuple[int, ...]
    baseline_pending_clause_indices: Tuple[int, ...]
    baseline_unsafe_clause_index: Optional[int]
    escalated_clause_indices: Tuple[int, ...]
    reason: str
    schema_version: str = HARD_CLAUSE_ESCALATION_DECISION_IR_SCHEMA_VERSION

    def validate(self) -> None:
        all_indices = set(range(self.clause_count))
        completed = set(self.baseline_completed_clause_indices)
        verified = set(self.baseline_verified_clause_indices)
        unresolved = set(self.baseline_unresolved_clause_indices)
        pending = set(self.baseline_pending_clause_indices)
        escalated = set(self.escalated_clause_indices)
        sequences = (
            self.baseline_completed_clause_indices,
            self.baseline_verified_clause_indices,
            self.baseline_unresolved_clause_indices,
            self.baseline_pending_clause_indices,
            self.escalated_clause_indices,
        )
        if (
            self.schema_version != HARD_CLAUSE_ESCALATION_DECISION_IR_SCHEMA_VERSION
            or not self.decision_id
            or not _is_sha256(self.plan_hash)
            or not _is_sha256(self.baseline_query_trace_hash)
            or self.clause_count < 1
            or any(tuple(sorted(set(values))) != values for values in sequences)
            or not completed <= all_indices
            or not verified <= completed
            or not unresolved <= completed
            or verified & unresolved
            or not pending <= all_indices
            or pending & completed
            or not escalated <= all_indices
        ):
            raise ValueError("hard-clause escalation Decision IR is invalid")
        if self.baseline_unsafe_clause_index is not None:
            if (
                self.baseline_unsafe_clause_index not in completed
                or self.escalated_clause_indices
                or self.reason != "baseline_unsafe_short_circuit"
            ):
                raise ValueError("unsafe escalation admission differs")
        elif pending:
            if (
                self.escalated_clause_indices
                or self.reason != "baseline_deadline_pending"
            ):
                raise ValueError("pending escalation admission differs")
        elif unresolved:
            if (
                self.escalated_clause_indices != self.baseline_unresolved_clause_indices
                or self.reason != "escalate_exact_unresolved"
            ):
                raise ValueError("unresolved escalation admission differs")
        elif self.escalated_clause_indices or self.reason != "baseline_verified_all":
            raise ValueError("closed escalation admission differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "plan_hash": self.plan_hash,
            "baseline_query_trace_hash": self.baseline_query_trace_hash,
            "clause_count": self.clause_count,
            "baseline_completed_clause_indices": list(
                self.baseline_completed_clause_indices
            ),
            "baseline_verified_clause_indices": list(
                self.baseline_verified_clause_indices
            ),
            "baseline_unresolved_clause_indices": list(
                self.baseline_unresolved_clause_indices
            ),
            "baseline_pending_clause_indices": list(
                self.baseline_pending_clause_indices
            ),
            "baseline_unsafe_clause_index": self.baseline_unsafe_clause_index,
            "escalated_clause_indices": list(self.escalated_clause_indices),
            "reason": self.reason,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeHardClauseEscalationTaskIRUnit:
    """One guarded task in the staged escalation graph."""

    task_id: str
    kind: HardClauseEscalationTaskKind
    dependency_task_ids: Tuple[str, ...]
    input_value_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]
    guard: str

    def validate(self) -> None:
        if (
            not self.task_id
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or not self.output_value_ids
            or len(self.output_value_ids) != len(set(self.output_value_ids))
            or self.guard not in {"always", "escalated_clauses_nonempty"}
        ):
            raise ValueError("hard-clause escalation Task IR unit is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "dependency_task_ids": list(self.dependency_task_ids),
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
            "guard": self.guard,
        }


@dataclass(frozen=True)
class NativeHardClauseEscalationTaskIRModule:
    """Exact static task graph compiled from an escalation Plan."""

    plan_hash: str
    tasks: Tuple[NativeHardClauseEscalationTaskIRUnit, ...]
    schema_version: str = HARD_CLAUSE_ESCALATION_TASK_IR_SCHEMA_VERSION

    def validate(self) -> None:
        task_ids = tuple(item.task_id for item in self.tasks)
        expected_kinds = tuple(HardClauseEscalationTaskKind)
        if (
            self.schema_version != HARD_CLAUSE_ESCALATION_TASK_IR_SCHEMA_VERSION
            or not _is_sha256(self.plan_hash)
            or tuple(item.kind for item in self.tasks) != expected_kinds
            or len(task_ids) != len(set(task_ids))
        ):
            raise ValueError("hard-clause escalation Task IR module is invalid")
        available: set[str] = set()
        for task in self.tasks:
            task.validate()
            if any(item not in available for item in task.dependency_task_ids):
                raise ValueError("hard-clause escalation Task dependency order differs")
            expected_guard = (
                "escalated_clauses_nonempty"
                if task.kind
                in {
                    HardClauseEscalationTaskKind.COMPILE_REFINEMENT,
                    HardClauseEscalationTaskKind.EXECUTE_REFINEMENT,
                    HardClauseEscalationTaskKind.PROJECT_HARD_CLAUSES,
                    HardClauseEscalationTaskKind.EXECUTE_ESCALATION,
                }
                else "always"
            )
            if task.guard != expected_guard:
                raise ValueError("hard-clause escalation Task guard differs")
            available.add(task.task_id)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "tasks": [item.to_dict() for item in self.tasks],
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeHardClauseEscalationScheduleActionIR:
    """One sequential guarded dispatch bound to a Task IR unit."""

    sequence: int
    action_id: str
    task_id: str
    kind: HardClauseEscalationTaskKind
    guard: str

    def validate(self) -> None:
        if (
            self.sequence < 0
            or not self.action_id
            or not self.task_id
            or self.guard not in {"always", "escalated_clauses_nonempty"}
        ):
            raise ValueError("hard-clause escalation Schedule action is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "guard": self.guard,
        }


@dataclass(frozen=True)
class NativeHardClauseEscalationScheduleIR:
    """Sequential guarded execution order for the staged verifier."""

    plan_hash: str
    task_ir_hash: str
    actions: Tuple[NativeHardClauseEscalationScheduleActionIR, ...]
    schema_version: str = HARD_CLAUSE_ESCALATION_SCHEDULE_IR_SCHEMA_VERSION

    def validate_against(self, task_ir: NativeHardClauseEscalationTaskIRModule) -> None:
        task_ir.validate()
        if (
            self.schema_version != HARD_CLAUSE_ESCALATION_SCHEDULE_IR_SCHEMA_VERSION
            or self.plan_hash != task_ir.plan_hash
            or self.task_ir_hash != task_ir.stable_hash()
            or len(self.actions) != len(task_ir.tasks)
        ):
            raise ValueError("hard-clause escalation Schedule IR differs")
        for sequence, (action, task) in enumerate(zip(self.actions, task_ir.tasks)):
            action.validate()
            if (
                action.sequence != sequence
                or action.action_id != f"escalation.launch.{sequence:04d}"
                or action.task_id != task.task_id
                or action.kind != task.kind
                or action.guard != task.guard
            ):
                raise ValueError("hard-clause escalation Schedule/Task binding differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "actions": [item.to_dict() for item in self.actions],
            "dispatch": "sequential_guarded_unresolved_clause_escalation",
            "deadline_owner": "whole_query_monotonic_clock",
        }

    def stable_hash(self, task_ir: NativeHardClauseEscalationTaskIRModule) -> str:
        self.validate_against(task_ir)
        return _canonical_hash(self.to_dict())


def lower_native_hard_clause_escalation_ir(
    plan: NativeHardClauseEscalationPlanIR,
) -> tuple[
    NativeHardClauseEscalationTaskIRModule,
    NativeHardClauseEscalationScheduleIR,
]:
    plan.validate()
    prefix = plan.plan_id
    definitions = (
        (
            "baseline",
            HardClauseEscalationTaskKind.EXECUTE_BASELINE,
            (),
            ("query.objectives", "query.thresholds"),
            ("escalation.baseline",),
            "always",
        ),
        (
            "admit",
            HardClauseEscalationTaskKind.ADMIT_HARD_CLAUSES,
            (f"{prefix}:baseline",),
            ("escalation.baseline",),
            ("escalation.decision",),
            "always",
        ),
        (
            "compile-refinement",
            HardClauseEscalationTaskKind.COMPILE_REFINEMENT,
            (f"{prefix}:admit",),
            ("escalation.decision", "query.module", "query.input"),
            ("escalation.refinement-program",),
            "escalated_clauses_nonempty",
        ),
        (
            "execute-refinement",
            HardClauseEscalationTaskKind.EXECUTE_REFINEMENT,
            (f"{prefix}:compile-refinement",),
            ("escalation.refinement-program",),
            ("escalation.refined-bounds", "escalation.refinement-trace"),
            "escalated_clauses_nonempty",
        ),
        (
            "project-hard-clauses",
            HardClauseEscalationTaskKind.PROJECT_HARD_CLAUSES,
            (f"{prefix}:admit",),
            (
                "escalation.decision",
                "query.objectives",
                "query.thresholds",
            ),
            (
                "escalation.projected-objectives",
                "escalation.projected-thresholds",
                "escalation.ordinal-map",
            ),
            "escalated_clauses_nonempty",
        ),
        (
            "execute-escalation",
            HardClauseEscalationTaskKind.EXECUTE_ESCALATION,
            (
                f"{prefix}:execute-refinement",
                f"{prefix}:project-hard-clauses",
            ),
            (
                "escalation.refined-bounds",
                "escalation.projected-objectives",
                "escalation.projected-thresholds",
            ),
            ("escalation.hard-query",),
            "escalated_clauses_nonempty",
        ),
        (
            "aggregate",
            HardClauseEscalationTaskKind.AGGREGATE_VERDICTS,
            (
                f"{prefix}:baseline",
                f"{prefix}:admit",
                f"{prefix}:execute-escalation",
            ),
            (
                "escalation.baseline",
                "escalation.decision",
                "escalation.hard-query",
                "escalation.ordinal-map",
            ),
            ("escalation.aggregate",),
            "always",
        ),
        (
            "emit",
            HardClauseEscalationTaskKind.EMIT_RESULT,
            (f"{prefix}:aggregate",),
            ("escalation.aggregate",),
            ("escalation.result",),
            "always",
        ),
    )
    tasks = tuple(
        NativeHardClauseEscalationTaskIRUnit(
            task_id=f"{prefix}:{suffix}",
            kind=kind,
            dependency_task_ids=dependencies,
            input_value_ids=inputs,
            output_value_ids=outputs,
            guard=guard,
        )
        for suffix, kind, dependencies, inputs, outputs, guard in definitions
    )
    task_ir = NativeHardClauseEscalationTaskIRModule(
        plan_hash=plan.stable_hash(), tasks=tasks
    )
    task_ir.validate()
    schedule = NativeHardClauseEscalationScheduleIR(
        plan_hash=plan.stable_hash(),
        task_ir_hash=task_ir.stable_hash(),
        actions=tuple(
            NativeHardClauseEscalationScheduleActionIR(
                sequence=index,
                action_id=f"escalation.launch.{index:04d}",
                task_id=task.task_id,
                kind=task.kind,
                guard=task.guard,
            )
            for index, task in enumerate(tasks)
        ),
    )
    schedule.validate_against(task_ir)
    return task_ir, schedule


__all__ = [
    "HARD_CLAUSE_ESCALATION_DECISION_IR_SCHEMA_VERSION",
    "HARD_CLAUSE_ESCALATION_PLAN_IR_SCHEMA_VERSION",
    "HARD_CLAUSE_ESCALATION_SCHEDULE_IR_SCHEMA_VERSION",
    "HARD_CLAUSE_ESCALATION_TASK_IR_SCHEMA_VERSION",
    "HardClauseEscalationTaskKind",
    "NativeHardClauseEscalationDecisionIR",
    "NativeHardClauseEscalationPlanIR",
    "NativeHardClauseEscalationScheduleActionIR",
    "NativeHardClauseEscalationScheduleIR",
    "NativeHardClauseEscalationTaskIRModule",
    "NativeHardClauseEscalationTaskIRUnit",
    "lower_native_hard_clause_escalation_ir",
]
