"""First-class per-clause objective-directed escalation control IR."""

# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Optional, Tuple

from .refinement import NativeIntermediateRefinementPolicyIR

OBJECTIVE_HARD_CLAUSE_ESCALATION_PLAN_IR_SCHEMA_VERSION = (
    "boundflow.objective-hard-clause-escalation-plan-ir/v1"
)
OBJECTIVE_HARD_CLAUSE_ESCALATION_TASK_IR_SCHEMA_VERSION = (
    "boundflow.objective-hard-clause-escalation-task-ir/v1"
)
OBJECTIVE_HARD_CLAUSE_ESCALATION_SCHEDULE_IR_SCHEMA_VERSION = (
    "boundflow.objective-hard-clause-escalation-schedule-ir/v1"
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


class ObjectiveHardClauseEscalationTaskKind(Enum):
    """Closed static stage vocabulary for objective-directed escalation."""

    EXECUTE_BASELINE = "execute_baseline"
    ADMIT_HARD_CLAUSES = "admit_hard_clauses"
    COMPILE_SHARED_REFINEMENT = "compile_shared_refinement"
    EXECUTE_SHARED_REFINEMENT = "execute_shared_refinement"
    COMPILE_OBJECTIVE_REFINEMENT = "compile_objective_refinement"
    EXECUTE_OBJECTIVE_REFINEMENT = "execute_objective_refinement"
    EXECUTE_OBJECTIVE_QUERY = "execute_objective_query"
    AGGREGATE_VERDICTS = "aggregate_verdicts"
    EMIT_RESULT = "emit_result"


@dataclass(frozen=True)
class NativeObjectiveHardClauseEscalationPlanIR:
    """Static extension bound to one frozen NRIR-30 base program."""

    plan_id: str
    base_plan_hash: str
    base_task_ir_hash: str
    base_schedule_hash: str
    clause_count: int
    objective_refinement_policy: NativeIntermediateRefinementPolicyIR
    semantics_owner: str = "boundflow_native_objective_hard_clause_escalation"
    schema_version: str = OBJECTIVE_HARD_CLAUSE_ESCALATION_PLAN_IR_SCHEMA_VERSION

    def validate(self) -> None:
        self.objective_refinement_policy.validate()
        if (
            self.schema_version
            != OBJECTIVE_HARD_CLAUSE_ESCALATION_PLAN_IR_SCHEMA_VERSION
            or not self.plan_id
            or any(
                not _is_sha256(value)
                for value in (
                    self.base_plan_hash,
                    self.base_task_ir_hash,
                    self.base_schedule_hash,
                )
            )
            or self.clause_count < 1
            or self.objective_refinement_policy.passes != 1
            or self.objective_refinement_policy.max_neurons_per_relu != 128
            or self.objective_refinement_policy.backward_chunk_size != 32
            or self.objective_refinement_policy.candidate_policy_id
            != "objective_influence_width_per_relu_v1"
            or self.objective_refinement_policy.refinement_method
            != "selected_plain_crown_v1"
            or self.semantics_owner
            != "boundflow_native_objective_hard_clause_escalation"
        ):
            raise ValueError("objective hard-clause escalation Plan IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "base_plan_hash": self.base_plan_hash,
            "base_task_ir_hash": self.base_task_ir_hash,
            "base_schedule_hash": self.base_schedule_hash,
            "clause_count": self.clause_count,
            "objective_refinement_policy": (self.objective_refinement_policy.to_dict()),
            "source_refinement": "validated_shared_nrir30_execution",
            "admission": "exact_baseline_unresolved_original_ordinals",
            "fallback": "preserve_baseline_or_completed_child_verdicts",
            "semantics_owner": self.semantics_owner,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveHardClauseEscalationTaskIRUnit:
    """One global or original-clause-owned task."""

    task_id: str
    kind: ObjectiveHardClauseEscalationTaskKind
    dependency_task_ids: Tuple[str, ...]
    input_value_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]
    guard: str
    original_clause_index: Optional[int] = None

    def validate(self, *, clause_count: int) -> None:
        per_clause = self.kind in {
            ObjectiveHardClauseEscalationTaskKind.COMPILE_OBJECTIVE_REFINEMENT,
            ObjectiveHardClauseEscalationTaskKind.EXECUTE_OBJECTIVE_REFINEMENT,
            ObjectiveHardClauseEscalationTaskKind.EXECUTE_OBJECTIVE_QUERY,
        }
        expected_guard = (
            "clause_is_admitted"
            if per_clause
            else (
                "admitted_clauses_nonempty"
                if self.kind
                in {
                    ObjectiveHardClauseEscalationTaskKind.COMPILE_SHARED_REFINEMENT,
                    ObjectiveHardClauseEscalationTaskKind.EXECUTE_SHARED_REFINEMENT,
                }
                else "always"
            )
        )
        if (
            not self.task_id
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or not self.output_value_ids
            or len(self.output_value_ids) != len(set(self.output_value_ids))
            or self.guard != expected_guard
            or per_clause != (self.original_clause_index is not None)
            or (
                self.original_clause_index is not None
                and not 0 <= self.original_clause_index < clause_count
            )
        ):
            raise ValueError("objective hard-clause escalation Task unit is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "dependency_task_ids": list(self.dependency_task_ids),
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
            "guard": self.guard,
            "original_clause_index": self.original_clause_index,
        }


@dataclass(frozen=True)
class NativeObjectiveHardClauseEscalationTaskIRModule:
    """Static unrolled task graph for all original clause ordinals."""

    plan_hash: str
    clause_count: int
    tasks: Tuple[NativeObjectiveHardClauseEscalationTaskIRUnit, ...]
    schema_version: str = OBJECTIVE_HARD_CLAUSE_ESCALATION_TASK_IR_SCHEMA_VERSION

    def validate(self) -> None:
        expected_count = 4 + 3 * self.clause_count + 2
        task_ids = tuple(task.task_id for task in self.tasks)
        if (
            self.schema_version
            != OBJECTIVE_HARD_CLAUSE_ESCALATION_TASK_IR_SCHEMA_VERSION
            or not _is_sha256(self.plan_hash)
            or self.clause_count < 1
            or len(self.tasks) != expected_count
            or len(task_ids) != len(set(task_ids))
        ):
            raise ValueError("objective hard-clause escalation Task IR is invalid")
        expected_prefix = (
            ObjectiveHardClauseEscalationTaskKind.EXECUTE_BASELINE,
            ObjectiveHardClauseEscalationTaskKind.ADMIT_HARD_CLAUSES,
            ObjectiveHardClauseEscalationTaskKind.COMPILE_SHARED_REFINEMENT,
            ObjectiveHardClauseEscalationTaskKind.EXECUTE_SHARED_REFINEMENT,
        )
        if tuple(task.kind for task in self.tasks[:4]) != expected_prefix:
            raise ValueError("objective hard-clause escalation Task prefix differs")
        available: set[str] = set()
        for position, task in enumerate(self.tasks):
            task.validate(clause_count=self.clause_count)
            if any(item not in available for item in task.dependency_task_ids):
                raise ValueError("objective hard-clause Task dependency order differs")
            if 4 <= position < 4 + 3 * self.clause_count:
                ordinal = (position - 4) // 3
                kinds = (
                    ObjectiveHardClauseEscalationTaskKind.COMPILE_OBJECTIVE_REFINEMENT,
                    ObjectiveHardClauseEscalationTaskKind.EXECUTE_OBJECTIVE_REFINEMENT,
                    ObjectiveHardClauseEscalationTaskKind.EXECUTE_OBJECTIVE_QUERY,
                )
                if (
                    task.original_clause_index != ordinal
                    or task.kind != kinds[(position - 4) % 3]
                ):
                    raise ValueError("objective hard-clause Task ordinal order differs")
            available.add(task.task_id)
        if tuple(task.kind for task in self.tasks[-2:]) != (
            ObjectiveHardClauseEscalationTaskKind.AGGREGATE_VERDICTS,
            ObjectiveHardClauseEscalationTaskKind.EMIT_RESULT,
        ):
            raise ValueError("objective hard-clause escalation Task suffix differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "clause_count": self.clause_count,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveHardClauseEscalationScheduleActionIR:
    """One sequential action bound to an exact Task unit."""

    sequence: int
    action_id: str
    task_id: str
    kind: ObjectiveHardClauseEscalationTaskKind
    guard: str
    original_clause_index: Optional[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "guard": self.guard,
            "original_clause_index": self.original_clause_index,
        }


@dataclass(frozen=True)
class NativeObjectiveHardClauseEscalationScheduleIR:
    """Whole-deadline sequential schedule with statically unrolled clauses."""

    plan_hash: str
    task_ir_hash: str
    actions: Tuple[NativeObjectiveHardClauseEscalationScheduleActionIR, ...]
    schema_version: str = OBJECTIVE_HARD_CLAUSE_ESCALATION_SCHEDULE_IR_SCHEMA_VERSION

    def validate_against(
        self, task_ir: NativeObjectiveHardClauseEscalationTaskIRModule
    ) -> None:
        task_ir.validate()
        if (
            self.schema_version
            != OBJECTIVE_HARD_CLAUSE_ESCALATION_SCHEDULE_IR_SCHEMA_VERSION
            or self.plan_hash != task_ir.plan_hash
            or self.task_ir_hash != task_ir.stable_hash()
            or len(self.actions) != len(task_ir.tasks)
        ):
            raise ValueError("objective hard-clause escalation Schedule IR differs")
        for sequence, (action, task) in enumerate(zip(self.actions, task_ir.tasks)):
            if (
                action.sequence != sequence
                or action.action_id != f"objective-escalation.launch.{sequence:04d}"
                or action.task_id != task.task_id
                or action.kind != task.kind
                or action.guard != task.guard
                or action.original_clause_index != task.original_clause_index
            ):
                raise ValueError("objective hard-clause Schedule/Task binding differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "actions": [action.to_dict() for action in self.actions],
            "dispatch": "sequential_guarded_original_clause_escalation",
            "deadline_owner": "whole_query_monotonic_clock",
        }

    def stable_hash(
        self, task_ir: NativeObjectiveHardClauseEscalationTaskIRModule
    ) -> str:
        self.validate_against(task_ir)
        return _canonical_hash(self.to_dict())


def lower_native_objective_hard_clause_escalation_ir(
    plan: NativeObjectiveHardClauseEscalationPlanIR,
) -> tuple[
    NativeObjectiveHardClauseEscalationTaskIRModule,
    NativeObjectiveHardClauseEscalationScheduleIR,
]:
    plan.validate()
    prefix = plan.plan_id
    TaskDefinition = tuple[
        str,
        ObjectiveHardClauseEscalationTaskKind,
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
        str,
        Optional[int],
    ]
    definitions: list[TaskDefinition] = [
        (
            "baseline",
            ObjectiveHardClauseEscalationTaskKind.EXECUTE_BASELINE,
            (),
            ("query.objectives", "query.thresholds"),
            ("objective-escalation.baseline",),
            "always",
            None,
        ),
        (
            "admit",
            ObjectiveHardClauseEscalationTaskKind.ADMIT_HARD_CLAUSES,
            (f"{prefix}:baseline",),
            ("objective-escalation.baseline",),
            ("objective-escalation.decision",),
            "always",
            None,
        ),
        (
            "compile-shared",
            ObjectiveHardClauseEscalationTaskKind.COMPILE_SHARED_REFINEMENT,
            (f"{prefix}:admit",),
            ("query.module", "query.input", "objective-escalation.decision"),
            ("objective-escalation.shared-program",),
            "admitted_clauses_nonempty",
            None,
        ),
        (
            "execute-shared",
            ObjectiveHardClauseEscalationTaskKind.EXECUTE_SHARED_REFINEMENT,
            (f"{prefix}:compile-shared",),
            ("objective-escalation.shared-program",),
            ("objective-escalation.shared-execution",),
            "admitted_clauses_nonempty",
            None,
        ),
    ]
    query_task_ids: list[str] = []
    for ordinal in range(plan.clause_count):
        suffix = f"clause-{ordinal:04d}"
        compile_id = f"{prefix}:{suffix}:compile-objective"
        execute_id = f"{prefix}:{suffix}:execute-objective"
        query_id = f"{prefix}:{suffix}:execute-query"
        definitions.extend(
            [
                (
                    f"{suffix}:compile-objective",
                    ObjectiveHardClauseEscalationTaskKind.COMPILE_OBJECTIVE_REFINEMENT,
                    (f"{prefix}:admit", f"{prefix}:execute-shared"),
                    (
                        "objective-escalation.decision",
                        "objective-escalation.shared-execution",
                        f"query.objective.{ordinal:04d}",
                    ),
                    (f"objective-escalation.objective-program.{ordinal:04d}",),
                    "clause_is_admitted",
                    ordinal,
                ),
                (
                    f"{suffix}:execute-objective",
                    ObjectiveHardClauseEscalationTaskKind.EXECUTE_OBJECTIVE_REFINEMENT,
                    (compile_id,),
                    (f"objective-escalation.objective-program.{ordinal:04d}",),
                    (f"objective-escalation.objective-execution.{ordinal:04d}",),
                    "clause_is_admitted",
                    ordinal,
                ),
                (
                    f"{suffix}:execute-query",
                    ObjectiveHardClauseEscalationTaskKind.EXECUTE_OBJECTIVE_QUERY,
                    (execute_id,),
                    (
                        f"objective-escalation.objective-execution.{ordinal:04d}",
                        f"query.objective.{ordinal:04d}",
                        f"query.threshold.{ordinal:04d}",
                    ),
                    (f"objective-escalation.query.{ordinal:04d}",),
                    "clause_is_admitted",
                    ordinal,
                ),
            ]
        )
        query_task_ids.append(query_id)
    definitions.extend(
        [
            (
                "aggregate",
                ObjectiveHardClauseEscalationTaskKind.AGGREGATE_VERDICTS,
                (f"{prefix}:baseline", f"{prefix}:admit", *query_task_ids),
                ("objective-escalation.baseline", "objective-escalation.children"),
                ("objective-escalation.aggregate",),
                "always",
                None,
            ),
            (
                "emit",
                ObjectiveHardClauseEscalationTaskKind.EMIT_RESULT,
                (f"{prefix}:aggregate",),
                ("objective-escalation.aggregate",),
                ("objective-escalation.result",),
                "always",
                None,
            ),
        ]
    )
    tasks = tuple(
        NativeObjectiveHardClauseEscalationTaskIRUnit(
            task_id=f"{prefix}:{suffix}",
            kind=kind,
            dependency_task_ids=dependencies,
            input_value_ids=inputs,
            output_value_ids=outputs,
            guard=guard,
            original_clause_index=ordinal,
        )
        for suffix, kind, dependencies, inputs, outputs, guard, ordinal in definitions
    )
    task_ir = NativeObjectiveHardClauseEscalationTaskIRModule(
        plan_hash=plan.stable_hash(), clause_count=plan.clause_count, tasks=tasks
    )
    task_ir.validate()
    schedule = NativeObjectiveHardClauseEscalationScheduleIR(
        plan_hash=plan.stable_hash(),
        task_ir_hash=task_ir.stable_hash(),
        actions=tuple(
            NativeObjectiveHardClauseEscalationScheduleActionIR(
                sequence=index,
                action_id=f"objective-escalation.launch.{index:04d}",
                task_id=task.task_id,
                kind=task.kind,
                guard=task.guard,
                original_clause_index=task.original_clause_index,
            )
            for index, task in enumerate(tasks)
        ),
    )
    schedule.validate_against(task_ir)
    return task_ir, schedule


__all__ = [
    "NativeObjectiveHardClauseEscalationPlanIR",
    "NativeObjectiveHardClauseEscalationScheduleIR",
    "NativeObjectiveHardClauseEscalationTaskIRModule",
    "NativeObjectiveHardClauseEscalationTaskIRUnit",
    "ObjectiveHardClauseEscalationTaskKind",
    "lower_native_objective_hard_clause_escalation_ir",
]
