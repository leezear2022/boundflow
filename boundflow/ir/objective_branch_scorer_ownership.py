"""Typed IR for compile-owned objective-branch candidate tables."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Tuple

from .branch import NativeObjectiveBranchPlanIR, ObjectiveBranchTaskKind

OBJECTIVE_BRANCH_SCORER_TASK_IR_SCHEMA_VERSION = (
    "boundflow.objective-branch-scorer-task-ir/v1"
)
OBJECTIVE_BRANCH_SCORER_SCHEDULE_IR_SCHEMA_VERSION = (
    "boundflow.objective-branch-scorer-schedule-ir/v1"
)
VALIDATED_BRANCH_PROGRAM_CAPSULE_IR_SCHEMA_VERSION = (
    "boundflow.validated-branch-program-capsule-ir/v1"
)


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


@dataclass(frozen=True)
class NativeObjectiveBranchScorerTaskIRUnit:
    """One scorer task with an explicit compile-owned candidate source."""

    task_id: str
    kind: ObjectiveBranchTaskKind
    input_value_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]
    dependency_task_ids: Tuple[str, ...]
    semantics_owner: str = "boundflow_native_objective_branch_scorer_ownership"

    def validate(self) -> None:
        if (
            not self.task_id
            or not self.input_value_ids
            or not self.output_value_ids
            or len(self.input_value_ids) != len(set(self.input_value_ids))
            or len(self.output_value_ids) != len(set(self.output_value_ids))
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or self.semantics_owner
            != "boundflow_native_objective_branch_scorer_ownership"
        ):
            raise ValueError("objective branch scorer Task IR unit is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
            "dependency_task_ids": list(self.dependency_task_ids),
            "semantics_owner": self.semantics_owner,
        }


@dataclass(frozen=True)
class NativeObjectiveBranchScorerTaskIRModule:
    """Five-stage scorer whose first task reads the Plan candidate table."""

    module_id: str
    branch_plan_hash: str
    tasks: Tuple[NativeObjectiveBranchScorerTaskIRUnit, ...]
    output_task_id: str
    schema_version: str = OBJECTIVE_BRANCH_SCORER_TASK_IR_SCHEMA_VERSION

    def validate(self, *, plan: NativeObjectiveBranchPlanIR) -> None:
        plan.validate()
        if (
            self.schema_version != OBJECTIVE_BRANCH_SCORER_TASK_IR_SCHEMA_VERSION
            or not self.module_id
            or self.branch_plan_hash != plan.stable_hash()
            or tuple(task.kind for task in self.tasks) != tuple(ObjectiveBranchTaskKind)
            or not self.output_task_id
            or self.tasks[-1].task_id != self.output_task_id
        ):
            raise ValueError("objective branch scorer Task IR module is invalid")
        completed: set[str] = set()
        available = {
            "branch.plan.candidates",
            "branch.relu_pre",
            "branch.split_state",
            "branch.selected_state",
            "branch.objective",
        }
        for task in self.tasks:
            task.validate()
            if any(item not in completed for item in task.dependency_task_ids):
                raise ValueError("objective branch scorer dependency is absent or late")
            if any(item not in available for item in task.input_value_ids):
                raise ValueError("objective branch scorer input is absent or late")
            if any(item in available for item in task.output_value_ids):
                raise ValueError("objective branch scorer output redefines a value")
            completed.add(task.task_id)
            available.update(task.output_value_ids)
        first = self.tasks[0]
        if first.input_value_ids != ("branch.plan.candidates",):
            raise ValueError("objective branch scorer candidate ownership differs")

    def to_dict(self, *, plan: NativeObjectiveBranchPlanIR) -> dict[str, object]:
        self.validate(plan=plan)
        return {
            "schema_version": self.schema_version,
            "module_id": self.module_id,
            "branch_plan_hash": self.branch_plan_hash,
            "tasks": [task.to_dict() for task in self.tasks],
            "output_task_id": self.output_task_id,
        }

    def stable_hash(self, *, plan: NativeObjectiveBranchPlanIR) -> str:
        return _canonical_hash(self.to_dict(plan=plan))


@dataclass(frozen=True)
class NativeObjectiveBranchScorerScheduleAction:
    """One synchronous launch in the prevalidated scorer schedule."""

    action_id: str
    sequence: int
    task_id: str
    input_value_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]

    def validate(self) -> None:
        if (
            not self.action_id
            or self.sequence < 0
            or not self.task_id
            or not self.input_value_ids
            or not self.output_value_ids
        ):
            raise ValueError("objective branch scorer Schedule action is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "action_id": self.action_id,
            "sequence": self.sequence,
            "task_id": self.task_id,
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
        }


@dataclass(frozen=True)
class NativeObjectiveBranchScorerScheduleIR:
    """Exact five-stage action order for the prevalidated scorer."""

    schedule_id: str
    branch_plan_hash: str
    branch_task_module_hash: str
    actions: Tuple[NativeObjectiveBranchScorerScheduleAction, ...]
    selected_candidate_value_id: str
    schema_version: str = OBJECTIVE_BRANCH_SCORER_SCHEDULE_IR_SCHEMA_VERSION

    def validate(
        self,
        *,
        plan: NativeObjectiveBranchPlanIR,
        task_module: NativeObjectiveBranchScorerTaskIRModule,
    ) -> None:
        task_module.validate(plan=plan)
        if (
            self.schema_version != OBJECTIVE_BRANCH_SCORER_SCHEDULE_IR_SCHEMA_VERSION
            or not self.schedule_id
            or self.branch_plan_hash != plan.stable_hash()
            or self.branch_task_module_hash != task_module.stable_hash(plan=plan)
            or len(self.actions) != len(task_module.tasks)
            or not self.selected_candidate_value_id
        ):
            raise ValueError("objective branch scorer Schedule IR is invalid")
        for sequence, (action, task) in enumerate(zip(self.actions, task_module.tasks)):
            action.validate()
            if (
                action.sequence != sequence
                or action.task_id != task.task_id
                or action.input_value_ids != task.input_value_ids
                or action.output_value_ids != task.output_value_ids
            ):
                raise ValueError("objective branch scorer Schedule/Task differs")
        if self.selected_candidate_value_id not in self.actions[-1].output_value_ids:
            raise ValueError("objective branch scorer Schedule lacks selection")

    def to_dict(
        self,
        *,
        plan: NativeObjectiveBranchPlanIR,
        task_module: NativeObjectiveBranchScorerTaskIRModule,
    ) -> dict[str, object]:
        self.validate(plan=plan, task_module=task_module)
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "branch_plan_hash": self.branch_plan_hash,
            "branch_task_module_hash": self.branch_task_module_hash,
            "actions": [action.to_dict() for action in self.actions],
            "selected_candidate_value_id": self.selected_candidate_value_id,
        }

    def stable_hash(
        self,
        *,
        plan: NativeObjectiveBranchPlanIR,
        task_module: NativeObjectiveBranchScorerTaskIRModule,
    ) -> str:
        return _canonical_hash(self.to_dict(plan=plan, task_module=task_module))


@dataclass(frozen=True)
class NativeValidatedBranchProgramCapsuleIR:
    """Immutable admission token binding compile-time candidates to execution."""

    capsule_id: str
    branch_plan_hash: str
    branch_task_module_hash: str
    branch_schedule_hash: str
    objective_hash: str
    relu_pre_hash: str
    split_state_hash: str
    selected_state_hash: str
    state_scope_hash: str
    optimizer_policy_hash: str
    branch_policy_hash: str
    candidate_table_hash: str
    candidate_count: int
    intermediate_bound_source: str
    refine_external_constraints: bool
    compile_enumeration_count: int
    execute_enumeration_count: int
    semantic_token: str
    candidate_source: str = "plan_owned_immutable"
    performance_claimed: bool = False
    schema_version: str = VALIDATED_BRANCH_PROGRAM_CAPSULE_IR_SCHEMA_VERSION

    def semantic_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "capsule_id": self.capsule_id,
            "branch_plan_hash": self.branch_plan_hash,
            "branch_task_module_hash": self.branch_task_module_hash,
            "branch_schedule_hash": self.branch_schedule_hash,
            "objective_hash": self.objective_hash,
            "relu_pre_hash": self.relu_pre_hash,
            "split_state_hash": self.split_state_hash,
            "selected_state_hash": self.selected_state_hash,
            "state_scope_hash": self.state_scope_hash,
            "optimizer_policy_hash": self.optimizer_policy_hash,
            "branch_policy_hash": self.branch_policy_hash,
            "candidate_table_hash": self.candidate_table_hash,
            "candidate_count": self.candidate_count,
            "intermediate_bound_source": self.intermediate_bound_source,
            "refine_external_constraints": self.refine_external_constraints,
            "compile_enumeration_count": self.compile_enumeration_count,
            "execute_enumeration_count": self.execute_enumeration_count,
            "candidate_source": self.candidate_source,
            "performance_claimed": self.performance_claimed,
        }

    def validate(
        self,
        *,
        plan: NativeObjectiveBranchPlanIR,
        task_module: NativeObjectiveBranchScorerTaskIRModule,
        schedule: NativeObjectiveBranchScorerScheduleIR,
    ) -> None:
        plan.validate()
        task_module.validate(plan=plan)
        schedule.validate(plan=plan, task_module=task_module)
        hashes = (
            self.branch_plan_hash,
            self.branch_task_module_hash,
            self.branch_schedule_hash,
            self.objective_hash,
            self.relu_pre_hash,
            self.split_state_hash,
            self.selected_state_hash,
            self.state_scope_hash,
            self.optimizer_policy_hash,
            self.branch_policy_hash,
            self.candidate_table_hash,
            self.semantic_token,
        )
        if (
            self.schema_version != VALIDATED_BRANCH_PROGRAM_CAPSULE_IR_SCHEMA_VERSION
            or not self.capsule_id
            or any(not _is_sha256(value) for value in hashes)
            or self.branch_plan_hash != plan.stable_hash()
            or self.branch_task_module_hash != task_module.stable_hash(plan=plan)
            or self.branch_schedule_hash
            != schedule.stable_hash(plan=plan, task_module=task_module)
            or self.split_state_hash != plan.split_state_hash
            or self.selected_state_hash != plan.selected_state_hash
            or self.state_scope_hash != plan.state_scope_hash
            or self.branch_policy_hash != plan.policy_hash
            or self.candidate_table_hash
            != _canonical_hash([candidate.to_dict() for candidate in plan.candidates])
            or self.candidate_count != len(plan.candidates)
            or self.intermediate_bound_source != plan.intermediate_bound_source
            or not isinstance(self.refine_external_constraints, bool)
            or self.compile_enumeration_count != 1
            or self.execute_enumeration_count != 0
            or self.candidate_source != "plan_owned_immutable"
            or self.performance_claimed is not False
            or self.semantic_token != _canonical_hash(self.semantic_dict())
        ):
            raise ValueError("validated branch program capsule differs")

    def to_dict(
        self,
        *,
        plan: NativeObjectiveBranchPlanIR,
        task_module: NativeObjectiveBranchScorerTaskIRModule,
        schedule: NativeObjectiveBranchScorerScheduleIR,
    ) -> dict[str, object]:
        self.validate(plan=plan, task_module=task_module, schedule=schedule)
        return {**self.semantic_dict(), "semantic_token": self.semantic_token}

    def stable_hash(
        self,
        *,
        plan: NativeObjectiveBranchPlanIR,
        task_module: NativeObjectiveBranchScorerTaskIRModule,
        schedule: NativeObjectiveBranchScorerScheduleIR,
    ) -> str:
        return _canonical_hash(
            self.to_dict(plan=plan, task_module=task_module, schedule=schedule)
        )


def lower_native_objective_branch_scorer_ir(
    plan: NativeObjectiveBranchPlanIR,
) -> tuple[
    NativeObjectiveBranchScorerTaskIRModule,
    NativeObjectiveBranchScorerScheduleIR,
]:
    """Lower a Plan-owned candidate table to the exact five score stages."""

    plan.validate()
    definitions = (
        (
            ObjectiveBranchTaskKind.ENUMERATE_CANDIDATES,
            ("branch.plan.candidates",),
            ("branch.candidates",),
        ),
        (
            ObjectiveBranchTaskKind.MATERIALIZE_CHILDREN,
            ("branch.candidates", "branch.split_state"),
            ("branch.child_splits",),
        ),
        (
            ObjectiveBranchTaskKind.EVALUATE_CHILD_BOUNDS,
            (
                "branch.child_splits",
                "branch.selected_state",
                "branch.objective",
                "branch.relu_pre",
            ),
            ("branch.child_lowers",),
        ),
        (
            ObjectiveBranchTaskKind.REDUCE_WORST_CHILD,
            ("branch.candidates", "branch.child_lowers"),
            ("branch.scores",),
        ),
        (
            ObjectiveBranchTaskKind.SELECT_CANDIDATE,
            ("branch.scores",),
            ("branch.selected_candidate",),
        ),
    )
    tasks: list[NativeObjectiveBranchScorerTaskIRUnit] = []
    previous: tuple[str, ...] = ()
    for kind, inputs, outputs in definitions:
        task_id = f"objective_branch_scorer.{kind.value}"
        tasks.append(
            NativeObjectiveBranchScorerTaskIRUnit(
                task_id=task_id,
                kind=kind,
                input_value_ids=inputs,
                output_value_ids=outputs,
                dependency_task_ids=previous,
            )
        )
        previous = (task_id,)
    task_module = NativeObjectiveBranchScorerTaskIRModule(
        module_id=f"{plan.plan_id}.scorer.tasks",
        branch_plan_hash=plan.stable_hash(),
        tasks=tuple(tasks),
        output_task_id=tasks[-1].task_id,
    )
    actions = tuple(
        NativeObjectiveBranchScorerScheduleAction(
            action_id=f"launch.{index:04d}.{task.task_id}",
            sequence=index,
            task_id=task.task_id,
            input_value_ids=task.input_value_ids,
            output_value_ids=task.output_value_ids,
        )
        for index, task in enumerate(tasks)
    )
    schedule = NativeObjectiveBranchScorerScheduleIR(
        schedule_id=f"{plan.plan_id}.scorer.schedule",
        branch_plan_hash=plan.stable_hash(),
        branch_task_module_hash=task_module.stable_hash(plan=plan),
        actions=actions,
        selected_candidate_value_id="branch.selected_candidate",
    )
    schedule.validate(plan=plan, task_module=task_module)
    return task_module, schedule


__all__ = [
    "NativeObjectiveBranchScorerScheduleIR",
    "NativeObjectiveBranchScorerTaskIRModule",
    "NativeValidatedBranchProgramCapsuleIR",
    "lower_native_objective_branch_scorer_ir",
]
