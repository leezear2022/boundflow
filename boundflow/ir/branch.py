"""Typed Plan, Task, and Schedule IR for objective-aware ReLU branching."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Tuple

OBJECTIVE_BRANCH_PLAN_IR_SCHEMA_VERSION = "boundflow.objective_branch_plan_ir/v1"
OBJECTIVE_BRANCH_TASK_IR_SCHEMA_VERSION = "boundflow.objective_branch_task_ir/v1"
OBJECTIVE_BRANCH_SCHEDULE_IR_SCHEMA_VERSION = (
    "boundflow.objective_branch_schedule_ir/v1"
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ObjectiveBranchTaskKind(Enum):
    """Closed objective-branch score pipeline."""

    ENUMERATE_CANDIDATES = "enumerate_candidates"
    MATERIALIZE_CHILDREN = "materialize_children"
    EVALUATE_CHILD_BOUNDS = "evaluate_child_bounds"
    REDUCE_WORST_CHILD = "reduce_worst_child"
    SELECT_CANDIDATE = "select_candidate"


@dataclass(frozen=True)
class NativeObjectiveBranchCandidateIR:
    """One stable unsplit ambiguous ReLU candidate."""

    ordinal: int
    relu_input: str
    neuron_index: int
    lower: float
    upper: float
    width: float

    def validate(self) -> None:
        if (
            self.ordinal < 0
            or not self.relu_input
            or self.neuron_index < 0
            or not all(math.isfinite(value) for value in (self.lower, self.upper))
            or not self.lower < 0.0 < self.upper
            or self.width <= 0.0
            or abs(self.width - (self.upper - self.lower)) > 1e-6
        ):
            raise ValueError("objective branch candidate IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "ordinal": self.ordinal,
            "relu_input": self.relu_input,
            "neuron_index": self.neuron_index,
            "lower": self.lower,
            "upper": self.upper,
            "width": self.width,
        }


@dataclass(frozen=True)
class NativeObjectiveBranchPlanIR:
    """Objective-bound-impact decision over one exact optimizer state."""

    plan_id: str
    objective_hash: str
    split_state_hash: str
    selected_state_hash: str
    state_scope_hash: str
    policy_hash: str
    candidate_policy_id: str
    candidates_per_relu: int
    candidate_batch_size: int
    max_candidates: int
    intermediate_bound_source: str
    candidates: Tuple[NativeObjectiveBranchCandidateIR, ...]
    reduce_policy: str = "maximize_worst_child_then_mean"
    schema_version: str = OBJECTIVE_BRANCH_PLAN_IR_SCHEMA_VERSION

    def validate(self) -> None:
        identities = tuple(
            (candidate.relu_input, candidate.neuron_index)
            for candidate in self.candidates
        )
        if (
            self.schema_version != OBJECTIVE_BRANCH_PLAN_IR_SCHEMA_VERSION
            or not self.plan_id
            or any(
                not _is_sha256(value)
                for value in (
                    self.objective_hash,
                    self.split_state_hash,
                    self.selected_state_hash,
                    self.state_scope_hash,
                    self.policy_hash,
                )
            )
            or self.candidate_policy_id != "top_width_per_relu_v1"
            or self.candidates_per_relu < 1
            or self.candidate_batch_size < 1
            or self.max_candidates < 1
            or len(self.candidates) > self.max_candidates
            or self.intermediate_bound_source
            not in {
                "local_forward",
                "native_refined",
                "external_verifier",
                "external_verifier_refined",
            }
            or not self.candidates
            or len(identities) != len(set(identities))
            or self.reduce_policy != "maximize_worst_child_then_mean"
        ):
            raise ValueError("objective branch Plan IR is invalid")
        for ordinal, candidate in enumerate(self.candidates):
            candidate.validate()
            if candidate.ordinal != ordinal:
                raise ValueError("objective branch candidate ordinal differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "objective_hash": self.objective_hash,
            "split_state_hash": self.split_state_hash,
            "selected_state_hash": self.selected_state_hash,
            "state_scope_hash": self.state_scope_hash,
            "policy_hash": self.policy_hash,
            "candidate_policy_id": self.candidate_policy_id,
            "candidates_per_relu": self.candidates_per_relu,
            "candidate_batch_size": self.candidate_batch_size,
            "max_candidates": self.max_candidates,
            "intermediate_bound_source": self.intermediate_bound_source,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reduce_policy": self.reduce_policy,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveBranchTaskIRUnit:
    """One score-pipeline task and its data dependencies."""

    task_id: str
    kind: ObjectiveBranchTaskKind
    input_value_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]
    dependency_task_ids: Tuple[str, ...]
    semantics_owner: str = "boundflow_native_objective_branching"

    def validate(self) -> None:
        if (
            not self.task_id
            or not self.input_value_ids
            or not self.output_value_ids
            or len(self.input_value_ids) != len(set(self.input_value_ids))
            or len(self.output_value_ids) != len(set(self.output_value_ids))
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or self.semantics_owner != "boundflow_native_objective_branching"
        ):
            raise ValueError("objective branch Task IR unit is invalid")

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
class NativeObjectiveBranchTaskIRModule:
    """Five-stage Task IR lowered from one objective branch plan."""

    module_id: str
    branch_plan_hash: str
    tasks: Tuple[NativeObjectiveBranchTaskIRUnit, ...]
    output_task_id: str
    schema_version: str = OBJECTIVE_BRANCH_TASK_IR_SCHEMA_VERSION

    def validate(self, *, plan: NativeObjectiveBranchPlanIR) -> None:
        plan.validate()
        expected = tuple(ObjectiveBranchTaskKind)
        if (
            self.schema_version != OBJECTIVE_BRANCH_TASK_IR_SCHEMA_VERSION
            or not self.module_id
            or self.branch_plan_hash != plan.stable_hash()
            or tuple(task.kind for task in self.tasks) != expected
            or not self.output_task_id
            or self.tasks[-1].task_id != self.output_task_id
        ):
            raise ValueError("objective branch Task IR module is invalid")
        completed: set[str] = set()
        available = {
            "branch.relu_pre",
            "branch.split_state",
            "branch.selected_state",
            "branch.objective",
        }
        for task in self.tasks:
            task.validate()
            if any(item not in completed for item in task.dependency_task_ids):
                raise ValueError("objective branch task dependency is absent or late")
            if any(item not in available for item in task.input_value_ids):
                raise ValueError("objective branch task input is absent or late")
            if any(item in available for item in task.output_value_ids):
                raise ValueError("objective branch task output redefines a value")
            completed.add(task.task_id)
            available.update(task.output_value_ids)

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
class NativeObjectiveBranchScheduleAction:
    """One synchronous launch of an objective branch Task IR unit."""

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
            raise ValueError("objective branch Schedule action is invalid")

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
class NativeObjectiveBranchScheduleIR:
    """Exact synchronous action order for objective branch scoring."""

    schedule_id: str
    branch_plan_hash: str
    branch_task_module_hash: str
    actions: Tuple[NativeObjectiveBranchScheduleAction, ...]
    selected_candidate_value_id: str
    schema_version: str = OBJECTIVE_BRANCH_SCHEDULE_IR_SCHEMA_VERSION

    def validate(
        self,
        *,
        plan: NativeObjectiveBranchPlanIR,
        task_module: NativeObjectiveBranchTaskIRModule,
    ) -> None:
        task_module.validate(plan=plan)
        if (
            self.schema_version != OBJECTIVE_BRANCH_SCHEDULE_IR_SCHEMA_VERSION
            or not self.schedule_id
            or self.branch_plan_hash != plan.stable_hash()
            or self.branch_task_module_hash != task_module.stable_hash(plan=plan)
            or len(self.actions) != len(task_module.tasks)
            or not self.selected_candidate_value_id
        ):
            raise ValueError("objective branch Schedule IR is invalid")
        for sequence, (action, task) in enumerate(zip(self.actions, task_module.tasks)):
            action.validate()
            if (
                action.sequence != sequence
                or action.task_id != task.task_id
                or action.input_value_ids != task.input_value_ids
                or action.output_value_ids != task.output_value_ids
            ):
                raise ValueError("objective branch Schedule/Task linkage differs")
        if self.selected_candidate_value_id not in self.actions[-1].output_value_ids:
            raise ValueError("objective branch Schedule does not emit selection")

    def to_dict(
        self,
        *,
        plan: NativeObjectiveBranchPlanIR,
        task_module: NativeObjectiveBranchTaskIRModule,
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
        task_module: NativeObjectiveBranchTaskIRModule,
    ) -> str:
        return _canonical_hash(self.to_dict(plan=plan, task_module=task_module))


def lower_native_objective_branch_ir(
    plan: NativeObjectiveBranchPlanIR,
) -> tuple[NativeObjectiveBranchTaskIRModule, NativeObjectiveBranchScheduleIR]:
    """Lower the fixed five-stage score pipeline to Task and Schedule IR."""

    plan.validate()
    definitions = (
        (
            ObjectiveBranchTaskKind.ENUMERATE_CANDIDATES,
            ("branch.relu_pre", "branch.split_state"),
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
    tasks: list[NativeObjectiveBranchTaskIRUnit] = []
    previous: tuple[str, ...] = ()
    for kind, inputs, outputs in definitions:
        task_id = f"objective_branch.{kind.value}"
        tasks.append(
            NativeObjectiveBranchTaskIRUnit(
                task_id=task_id,
                kind=kind,
                input_value_ids=inputs,
                output_value_ids=outputs,
                dependency_task_ids=previous,
            )
        )
        previous = (task_id,)
    task_module = NativeObjectiveBranchTaskIRModule(
        module_id=f"{plan.plan_id}.tasks",
        branch_plan_hash=plan.stable_hash(),
        tasks=tuple(tasks),
        output_task_id=tasks[-1].task_id,
    )
    actions = tuple(
        NativeObjectiveBranchScheduleAction(
            action_id=f"launch.{index:04d}.{task.task_id}",
            sequence=index,
            task_id=task.task_id,
            input_value_ids=task.input_value_ids,
            output_value_ids=task.output_value_ids,
        )
        for index, task in enumerate(tasks)
    )
    schedule = NativeObjectiveBranchScheduleIR(
        schedule_id=f"{plan.plan_id}.schedule",
        branch_plan_hash=plan.stable_hash(),
        branch_task_module_hash=task_module.stable_hash(plan=plan),
        actions=actions,
        selected_candidate_value_id="branch.selected_candidate",
    )
    schedule.validate(plan=plan, task_module=task_module)
    return task_module, schedule
