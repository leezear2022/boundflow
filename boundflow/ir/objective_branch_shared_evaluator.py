"""Typed composite IR for objective branching over the shared evaluator."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,missing-class-docstring

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Tuple

from .shared_parametric_ancestral import NativeSharedParametricAncestralPlanIR

OBJECTIVE_BRANCH_SHARED_PLAN_SCHEMA_VERSION = (
    "boundflow.objective-branch-shared-evaluator-plan/v1"
)
OBJECTIVE_BRANCH_SHARED_TASK_SCHEMA_VERSION = (
    "boundflow.objective-branch-shared-evaluator-task/v1"
)
OBJECTIVE_BRANCH_SHARED_SCHEDULE_SCHEMA_VERSION = (
    "boundflow.objective-branch-shared-evaluator-schedule/v1"
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
class NativeObjectiveBranchSharedPlanIR:
    """Frozen shared-evaluator semantics plus one branch-policy variable."""

    plan_id: str
    shared_plan: NativeSharedParametricAncestralPlanIR
    branch_policy_hash: str
    candidates_per_relu: int
    candidate_batch_size: int
    max_candidates: int
    minimum_worst_active_lower_improvement: float = 1.0
    median_lower_tolerance: float = 1e-5
    candidate_policy_id: str = "top_width_per_relu_v1"
    reduce_policy: str = "maximize_worst_child_then_mean"
    control_branch_mode: str = "widest_unsplit_ambiguous_relu"
    candidate_branch_mode: str = "objective_bound_impact"
    frozen_variables: Tuple[str, ...] = (
        "source_objective",
        "threshold",
        "optimizer_policy",
        "ancestral_refinement",
        "parent_warm_state",
        "template_cache_ownership",
        "best_first_queue",
        "node_and_depth_budget",
        "atomic_sibling_commit",
        "dtype",
        "device",
    )
    semantics_owner: str = "boundflow_objective_branch_shared_evaluator"
    performance_claimed: bool = False
    schema_version: str = OBJECTIVE_BRANCH_SHARED_PLAN_SCHEMA_VERSION

    def validate(self) -> None:
        self.shared_plan.validate()
        if (
            self.schema_version != OBJECTIVE_BRANCH_SHARED_PLAN_SCHEMA_VERSION
            or not self.plan_id
            or self.plan_id != self.shared_plan.plan_id
            or not _is_sha256(self.branch_policy_hash)
            or self.candidates_per_relu != 8
            or self.candidate_batch_size != 64
            or self.max_candidates != 256
            or not math.isfinite(self.minimum_worst_active_lower_improvement)
            or self.minimum_worst_active_lower_improvement != 1.0
            or not math.isfinite(self.median_lower_tolerance)
            or self.median_lower_tolerance != 1e-5
            or self.candidate_policy_id != "top_width_per_relu_v1"
            or self.reduce_policy != "maximize_worst_child_then_mean"
            or self.control_branch_mode != "widest_unsplit_ambiguous_relu"
            or self.candidate_branch_mode != "objective_bound_impact"
            or len(self.frozen_variables) != len(set(self.frozen_variables))
            or set(self.frozen_variables)
            != {
                "source_objective",
                "threshold",
                "optimizer_policy",
                "ancestral_refinement",
                "parent_warm_state",
                "template_cache_ownership",
                "best_first_queue",
                "node_and_depth_budget",
                "atomic_sibling_commit",
                "dtype",
                "device",
            }
            or self.shared_plan.max_nodes != 31
            or self.shared_plan.max_depth != 4
            or self.shared_plan.child_refinement_cap != 128
            or self.semantics_owner != "boundflow_objective_branch_shared_evaluator"
            or self.performance_claimed is not False
        ):
            raise ValueError("objective-branch shared Plan IR differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "shared_plan": self.shared_plan.to_dict(),
            "shared_plan_hash": self.shared_plan.stable_hash(),
            "branch_policy_hash": self.branch_policy_hash,
            "candidates_per_relu": self.candidates_per_relu,
            "candidate_batch_size": self.candidate_batch_size,
            "max_candidates": self.max_candidates,
            "candidate_policy_id": self.candidate_policy_id,
            "reduce_policy": self.reduce_policy,
            "control_branch_mode": self.control_branch_mode,
            "candidate_branch_mode": self.candidate_branch_mode,
            "minimum_worst_active_lower_improvement": (
                self.minimum_worst_active_lower_improvement
            ),
            "median_lower_tolerance": self.median_lower_tolerance,
            "frozen_variables": list(self.frozen_variables),
            "semantics_owner": self.semantics_owner,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveBranchBindingIR:
    """One queue evaluation bound to one exact objective-branch execution."""

    node_id: str
    evaluation_hash: str
    split_state_hash: str
    selected_state_hash: str
    branch_plan_hash: str
    branch_task_hash: str
    branch_schedule_hash: str
    branch_trace_hash: str
    selected_relu_input: str
    selected_neuron_index: int
    selected_candidate_ordinal: int
    candidate_count: int

    def validate(self) -> None:
        if (
            not self.node_id
            or any(
                not _is_sha256(value)
                for value in (
                    self.evaluation_hash,
                    self.split_state_hash,
                    self.selected_state_hash,
                    self.branch_plan_hash,
                    self.branch_task_hash,
                    self.branch_schedule_hash,
                    self.branch_trace_hash,
                )
            )
            or not self.selected_relu_input
            or self.selected_neuron_index < 0
            or self.selected_candidate_ordinal < 0
            or self.candidate_count < 1
            or self.selected_candidate_ordinal >= self.candidate_count
        ):
            raise ValueError("objective-branch execution binding differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "node_id": self.node_id,
            "evaluation_hash": self.evaluation_hash,
            "split_state_hash": self.split_state_hash,
            "selected_state_hash": self.selected_state_hash,
            "branch_plan_hash": self.branch_plan_hash,
            "branch_task_hash": self.branch_task_hash,
            "branch_schedule_hash": self.branch_schedule_hash,
            "branch_trace_hash": self.branch_trace_hash,
            "selected_relu_input": self.selected_relu_input,
            "selected_neuron_index": self.selected_neuron_index,
            "selected_candidate_ordinal": self.selected_candidate_ordinal,
            "candidate_count": self.candidate_count,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveBranchSharedDecisionIR:
    """Preregistered fixed-budget branch-selection decision."""

    plan_hash: str
    control_execution_hash: str
    candidate_execution_hash: str
    control_active_count: int
    candidate_active_count: int
    branch_execution_count: int
    control_root_lower: float
    candidate_root_lower: float
    root_lower_abs_diff: float
    control_worst_active_lower: float
    candidate_worst_active_lower: float
    worst_active_lower_improvement: float
    control_median_active_lower: float
    candidate_median_active_lower: float
    median_active_lower_delta: float
    structure_passed: bool
    root_parity_passed: bool
    median_not_weaker: bool
    go: bool
    reason: str

    def validate(self) -> None:
        numeric = (
            self.control_root_lower,
            self.candidate_root_lower,
            self.root_lower_abs_diff,
            self.control_worst_active_lower,
            self.candidate_worst_active_lower,
            self.worst_active_lower_improvement,
            self.control_median_active_lower,
            self.candidate_median_active_lower,
            self.median_active_lower_delta,
        )
        expected_go = (
            self.structure_passed
            and self.root_parity_passed
            and self.median_not_weaker
            and self.worst_active_lower_improvement >= 1.0
        )
        expected_reason = (
            "candidate_passed_preregistered_gate"
            if expected_go
            else (
                "candidate_structure_failed"
                if not self.structure_passed
                else (
                    "candidate_root_parity_failed"
                    if not self.root_parity_passed
                    else (
                        "candidate_median_active_lower_weaker"
                        if not self.median_not_weaker
                        else "candidate_worst_improvement_below_gate"
                    )
                )
            )
        )
        if (
            any(
                not _is_sha256(value)
                for value in (
                    self.plan_hash,
                    self.control_execution_hash,
                    self.candidate_execution_hash,
                )
            )
            or min(
                self.control_active_count,
                self.candidate_active_count,
                self.branch_execution_count,
            )
            < 1
            or not all(math.isfinite(value) for value in numeric)
            or abs(
                self.root_lower_abs_diff
                - abs(self.candidate_root_lower - self.control_root_lower)
            )
            > 1e-9
            or abs(
                self.worst_active_lower_improvement
                - (self.candidate_worst_active_lower - self.control_worst_active_lower)
            )
            > 1e-9
            or abs(
                self.median_active_lower_delta
                - (
                    self.candidate_median_active_lower
                    - self.control_median_active_lower
                )
            )
            > 1e-9
            or self.go != expected_go
            or self.reason != expected_reason
        ):
            raise ValueError("objective-branch shared decision differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "plan_hash": self.plan_hash,
            "control_execution_hash": self.control_execution_hash,
            "candidate_execution_hash": self.candidate_execution_hash,
            "control_active_count": self.control_active_count,
            "candidate_active_count": self.candidate_active_count,
            "branch_execution_count": self.branch_execution_count,
            "control_root_lower": self.control_root_lower,
            "candidate_root_lower": self.candidate_root_lower,
            "root_lower_abs_diff": self.root_lower_abs_diff,
            "control_worst_active_lower": self.control_worst_active_lower,
            "candidate_worst_active_lower": self.candidate_worst_active_lower,
            "worst_active_lower_improvement": self.worst_active_lower_improvement,
            "control_median_active_lower": self.control_median_active_lower,
            "candidate_median_active_lower": self.candidate_median_active_lower,
            "median_active_lower_delta": self.median_active_lower_delta,
            "structure_passed": self.structure_passed,
            "root_parity_passed": self.root_parity_passed,
            "median_not_weaker": self.median_not_weaker,
            "go": self.go,
            "reason": self.reason,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


class NativeObjectiveBranchSharedTaskKind(Enum):
    ADMIT_SHARED_PLAN = "admit_shared_plan"
    EXECUTE_SHARED_QUEUE = "execute_shared_queue"
    SCORE_OBJECTIVE_BRANCHES = "score_objective_branches"
    VERIFY_BINDINGS = "verify_bindings"
    DECIDE = "decide"
    EMIT = "emit"


@dataclass(frozen=True)
class NativeObjectiveBranchSharedTaskIRUnit:
    sequence: int
    task_id: str
    kind: NativeObjectiveBranchSharedTaskKind
    dependency_task_ids: Tuple[str, ...]
    input_hashes: Tuple[Tuple[str, str], ...]
    output_hash: str

    def validate(self) -> None:
        if (
            self.sequence < 0
            or not self.task_id
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or len(self.input_hashes)
            != len({name for name, _value in self.input_hashes})
            or any(
                not name or not _is_sha256(value) for name, value in self.input_hashes
            )
            or not _is_sha256(self.output_hash)
        ):
            raise ValueError("objective-branch shared Task unit differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "dependency_task_ids": list(self.dependency_task_ids),
            "input_hashes": dict(self.input_hashes),
            "output_hash": self.output_hash,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveBranchSharedTaskIRModule:
    plan_hash: str
    tasks: Tuple[NativeObjectiveBranchSharedTaskIRUnit, ...]
    branch_binding_hashes: Tuple[str, ...]
    schema_version: str = OBJECTIVE_BRANCH_SHARED_TASK_SCHEMA_VERSION

    def validate(self) -> None:
        expected = tuple(NativeObjectiveBranchSharedTaskKind)
        if (
            self.schema_version != OBJECTIVE_BRANCH_SHARED_TASK_SCHEMA_VERSION
            or not _is_sha256(self.plan_hash)
            or tuple(task.kind for task in self.tasks) != expected
            or tuple(task.sequence for task in self.tasks)
            != tuple(range(len(self.tasks)))
            or len({task.task_id for task in self.tasks}) != len(self.tasks)
            or not self.branch_binding_hashes
            or any(not _is_sha256(value) for value in self.branch_binding_hashes)
        ):
            raise ValueError("objective-branch shared Task module differs")
        available: set[str] = set()
        for task in self.tasks:
            task.validate()
            if any(item not in available for item in task.dependency_task_ids):
                raise ValueError("objective-branch shared Task dependency differs")
            available.add(task.task_id)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "tasks": [task.to_dict() for task in self.tasks],
            "branch_binding_hashes": list(self.branch_binding_hashes),
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveBranchSharedScheduleActionIR:
    sequence: int
    action_id: str
    task_id: str
    kind: NativeObjectiveBranchSharedTaskKind
    task_hash: str

    def validate(self) -> None:
        if (
            self.sequence < 0
            or not self.action_id
            or not self.task_id
            or not _is_sha256(self.task_hash)
        ):
            raise ValueError("objective-branch shared Schedule action differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "task_hash": self.task_hash,
        }


@dataclass(frozen=True)
class NativeObjectiveBranchSharedScheduleIR:
    plan_hash: str
    task_ir_hash: str
    actions: Tuple[NativeObjectiveBranchSharedScheduleActionIR, ...]
    schema_version: str = OBJECTIVE_BRANCH_SHARED_SCHEDULE_SCHEMA_VERSION

    def validate_against(
        self, task_ir: NativeObjectiveBranchSharedTaskIRModule
    ) -> None:
        task_ir.validate()
        if (
            self.schema_version != OBJECTIVE_BRANCH_SHARED_SCHEDULE_SCHEMA_VERSION
            or self.plan_hash != task_ir.plan_hash
            or self.task_ir_hash != task_ir.stable_hash()
            or len(self.actions) != len(task_ir.tasks)
            or tuple(action.sequence for action in self.actions)
            != tuple(range(len(self.actions)))
        ):
            raise ValueError("objective-branch shared Schedule differs")
        for action, task in zip(self.actions, task_ir.tasks):
            action.validate()
            if (
                action.task_id != task.task_id
                or action.kind != task.kind
                or action.task_hash != task.stable_hash()
            ):
                raise ValueError(
                    "objective-branch shared Schedule/Task binding differs"
                )

    def to_dict(
        self, *, task_ir: NativeObjectiveBranchSharedTaskIRModule
    ) -> dict[str, object]:
        self.validate_against(task_ir)
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "actions": [action.to_dict() for action in self.actions],
        }

    def stable_hash(self, task_ir: NativeObjectiveBranchSharedTaskIRModule) -> str:
        return _canonical_hash(self.to_dict(task_ir=task_ir))


def lower_native_objective_branch_shared_schedule(
    plan: NativeObjectiveBranchSharedPlanIR,
    *,
    shared_execution_hash: str,
    bindings: Tuple[NativeObjectiveBranchBindingIR, ...],
    decision: NativeObjectiveBranchSharedDecisionIR,
) -> tuple[
    NativeObjectiveBranchSharedTaskIRModule,
    NativeObjectiveBranchSharedScheduleIR,
]:
    plan.validate()
    decision.validate()
    for binding in bindings:
        binding.validate()
    if (
        not _is_sha256(shared_execution_hash)
        or not bindings
        or decision.plan_hash != plan.stable_hash()
        or decision.candidate_execution_hash != shared_execution_hash
        or decision.branch_execution_count != len(bindings)
    ):
        raise ValueError("objective-branch shared lowering inputs differ")
    plan_hash = plan.stable_hash()
    binding_hashes = tuple(binding.stable_hash() for binding in bindings)
    tasks: list[NativeObjectiveBranchSharedTaskIRUnit] = []

    def add(
        kind: NativeObjectiveBranchSharedTaskKind,
        inputs: dict[str, str],
        output: object,
    ) -> None:
        dependencies = () if not tasks else (tasks[-1].task_id,)
        tasks.append(
            NativeObjectiveBranchSharedTaskIRUnit(
                sequence=len(tasks),
                task_id=f"{plan.plan_id}:{kind.value}",
                kind=kind,
                dependency_task_ids=dependencies,
                input_hashes=tuple(sorted(inputs.items())),
                output_hash=_canonical_hash(output),
            )
        )

    add(
        NativeObjectiveBranchSharedTaskKind.ADMIT_SHARED_PLAN,
        {"plan": plan_hash},
        plan.to_dict(),
    )
    add(
        NativeObjectiveBranchSharedTaskKind.EXECUTE_SHARED_QUEUE,
        {"shared_plan": plan.shared_plan.stable_hash()},
        {"candidate_execution_hash": shared_execution_hash},
    )
    add(
        NativeObjectiveBranchSharedTaskKind.SCORE_OBJECTIVE_BRANCHES,
        {"branch_policy": plan.branch_policy_hash},
        {"binding_hashes": list(binding_hashes)},
    )
    add(
        NativeObjectiveBranchSharedTaskKind.VERIFY_BINDINGS,
        {"candidate_execution": shared_execution_hash},
        [binding.to_dict() for binding in bindings],
    )
    add(
        NativeObjectiveBranchSharedTaskKind.DECIDE,
        {"control_execution": decision.control_execution_hash},
        decision.to_dict(),
    )
    add(
        NativeObjectiveBranchSharedTaskKind.EMIT,
        {"decision": decision.stable_hash()},
        {
            "plan_hash": plan_hash,
            "candidate_execution_hash": shared_execution_hash,
            "decision_hash": decision.stable_hash(),
        },
    )
    task_ir = NativeObjectiveBranchSharedTaskIRModule(
        plan_hash=plan_hash,
        tasks=tuple(tasks),
        branch_binding_hashes=binding_hashes,
    )
    task_ir.validate()
    schedule = NativeObjectiveBranchSharedScheduleIR(
        plan_hash=plan_hash,
        task_ir_hash=task_ir.stable_hash(),
        actions=tuple(
            NativeObjectiveBranchSharedScheduleActionIR(
                sequence=index,
                action_id=f"{plan.plan_id}:launch:{index:02d}",
                task_id=task.task_id,
                kind=task.kind,
                task_hash=task.stable_hash(),
            )
            for index, task in enumerate(task_ir.tasks)
        ),
    )
    schedule.validate_against(task_ir)
    return task_ir, schedule


__all__ = [
    "NativeObjectiveBranchBindingIR",
    "NativeObjectiveBranchSharedDecisionIR",
    "NativeObjectiveBranchSharedPlanIR",
    "NativeObjectiveBranchSharedScheduleIR",
    "NativeObjectiveBranchSharedTaskIRModule",
    "NativeObjectiveBranchSharedTaskKind",
    "lower_native_objective_branch_shared_schedule",
]
