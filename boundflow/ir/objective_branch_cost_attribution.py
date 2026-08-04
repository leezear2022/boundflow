"""Typed IR for NRIR-41 objective-branch production cost attribution."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=too-many-locals,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Tuple

OBJECTIVE_BRANCH_COST_PLAN_SCHEMA_VERSION = (
    "boundflow.objective-branch-cost-attribution-plan/v1"
)
OBJECTIVE_BRANCH_COST_TASK_SCHEMA_VERSION = (
    "boundflow.objective-branch-cost-attribution-task/v1"
)
OBJECTIVE_BRANCH_COST_SCHEDULE_SCHEMA_VERSION = (
    "boundflow.objective-branch-cost-attribution-schedule/v1"
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


class NativeObjectiveBranchCostTaskKind(Enum):
    ADMIT_SOURCE = "admit_source"
    RECONSTRUCT_PREFIX = "reconstruct_prefix"
    EXECUTE_PAIRED_WALL = "execute_paired_wall"
    PROFILE_OBJECTIVE = "profile_objective"
    DECIDE_CAUSE = "decide_cause"
    EMIT = "emit"


@dataclass(frozen=True)
class NativeObjectiveBranchCostAttributionPlanIR:
    plan_id: str
    source_pilot_hash: str
    source_formal_hash: str
    clause_ordinals: Tuple[int, ...] = (2, 3)
    prefix_node_counts: Tuple[int, ...] = (21, 23, 29, 31)
    paired_orders: Tuple[Tuple[str, ...], ...] = (
        ("widest", "objective"),
        ("objective", "widest"),
        ("widest", "objective"),
    )
    required_nodes: int = 31
    required_sibling_groups: int = 15
    minimum_frontier_improvement: float = 1.0
    minimum_queue_ratio: float = 1.2
    minimum_branch_program_share: float = 0.2
    torch_threads: int = 8
    candidate_policy_id: str = "objective_bound_impact"
    control_policy_id: str = "widest_unsplit_ambiguous_relu"
    semantics_owner: str = "boundflow_objective_branch_cost_attribution"
    performance_claimed: bool = False
    schema_version: str = OBJECTIVE_BRANCH_COST_PLAN_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != OBJECTIVE_BRANCH_COST_PLAN_SCHEMA_VERSION
            or not self.plan_id
            or not _is_sha256(self.source_pilot_hash)
            or not _is_sha256(self.source_formal_hash)
            or self.clause_ordinals != (2, 3)
            or self.prefix_node_counts != (21, 23, 29, 31)
            or self.paired_orders
            != (
                ("widest", "objective"),
                ("objective", "widest"),
                ("widest", "objective"),
            )
            or self.required_nodes != 31
            or self.required_sibling_groups != 15
            or self.required_sibling_groups * 2 + 1 != self.required_nodes
            or self.minimum_frontier_improvement != 1.0
            or self.minimum_queue_ratio != 1.2
            or self.minimum_branch_program_share != 0.2
            or self.torch_threads != 8
            or self.candidate_policy_id != "objective_bound_impact"
            or self.control_policy_id != "widest_unsplit_ambiguous_relu"
            or self.semantics_owner != "boundflow_objective_branch_cost_attribution"
            or self.performance_claimed is not False
        ):
            raise ValueError("objective-branch cost attribution Plan IR differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "source_pilot_hash": self.source_pilot_hash,
            "source_formal_hash": self.source_formal_hash,
            "clause_ordinals": list(self.clause_ordinals),
            "prefix_node_counts": list(self.prefix_node_counts),
            "paired_orders": [list(value) for value in self.paired_orders],
            "required_nodes": self.required_nodes,
            "required_sibling_groups": self.required_sibling_groups,
            "minimum_frontier_improvement": self.minimum_frontier_improvement,
            "minimum_queue_ratio": self.minimum_queue_ratio,
            "minimum_branch_program_share": self.minimum_branch_program_share,
            "torch_threads": self.torch_threads,
            "candidate_policy_id": self.candidate_policy_id,
            "control_policy_id": self.control_policy_id,
            "semantics_owner": self.semantics_owner,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveBranchPrefixAttributionIR:
    plan_hash: str
    original_clause_index: int
    policy_id: str
    accepted_nodes: int
    active_node_ids: Tuple[str, ...]
    active_evaluation_hashes: Tuple[str, ...]
    active_count: int
    worst_active_lower: float
    median_active_lower: float
    source_rows_hash: str

    def validate(self) -> None:
        if (
            not _is_sha256(self.plan_hash)
            or self.original_clause_index not in (2, 3)
            or self.policy_id
            not in {"widest_unsplit_ambiguous_relu", "objective_bound_impact"}
            or self.accepted_nodes not in (21, 23, 29, 31)
            or self.accepted_nodes % 2 != 1
            or self.active_count != (self.accepted_nodes + 1) // 2
            or len(self.active_node_ids) != self.active_count
            or len(self.active_node_ids) != len(set(self.active_node_ids))
            or len(self.active_evaluation_hashes) != self.active_count
            or any(not node_id for node_id in self.active_node_ids)
            or any(not _is_sha256(value) for value in self.active_evaluation_hashes)
            or not all(
                math.isfinite(value)
                for value in (self.worst_active_lower, self.median_active_lower)
            )
            or not _is_sha256(self.source_rows_hash)
        ):
            raise ValueError("objective-branch prefix attribution IR differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "plan_hash": self.plan_hash,
            "original_clause_index": self.original_clause_index,
            "policy_id": self.policy_id,
            "accepted_nodes": self.accepted_nodes,
            "active_node_ids": list(self.active_node_ids),
            "active_evaluation_hashes": list(self.active_evaluation_hashes),
            "active_count": self.active_count,
            "worst_active_lower": self.worst_active_lower,
            "median_active_lower": self.median_active_lower,
            "source_rows_hash": self.source_rows_hash,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveBranchWallAttributionIR:
    plan_hash: str
    repeat_index: int
    original_clause_index: int
    policy_id: str
    order_position: int
    execution_hash: str
    queue_trace_hash: str
    root_lower: float
    worst_active_lower: float
    median_active_lower: float
    accepted_nodes: int
    sibling_group_count: int
    source_elapsed_ns: int
    queue_elapsed_ns: int
    whole_elapsed_ns: int
    cache_miss_count: int
    cache_hit_count: int
    branch_execution_count: int

    def validate(self) -> None:
        if (
            not _is_sha256(self.plan_hash)
            or self.repeat_index not in range(3)
            or self.original_clause_index not in (2, 3)
            or self.policy_id
            not in {"widest_unsplit_ambiguous_relu", "objective_bound_impact"}
            or self.order_position not in (0, 1)
            or not _is_sha256(self.execution_hash)
            or not _is_sha256(self.queue_trace_hash)
            or not all(
                math.isfinite(value)
                for value in (
                    self.root_lower,
                    self.worst_active_lower,
                    self.median_active_lower,
                )
            )
            or self.accepted_nodes != 31
            or self.sibling_group_count != 15
            or min(
                self.source_elapsed_ns,
                self.queue_elapsed_ns,
                self.whole_elapsed_ns,
            )
            < 0
            or self.queue_elapsed_ns <= 0
            or self.whole_elapsed_ns < self.queue_elapsed_ns
            or self.cache_miss_count != 1
            or self.cache_hit_count != 15
            or self.branch_execution_count
            != (31 if self.policy_id == "objective_bound_impact" else 0)
        ):
            raise ValueError("objective-branch wall attribution IR differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "plan_hash": self.plan_hash,
            "repeat_index": self.repeat_index,
            "original_clause_index": self.original_clause_index,
            "policy_id": self.policy_id,
            "order_position": self.order_position,
            "execution_hash": self.execution_hash,
            "queue_trace_hash": self.queue_trace_hash,
            "root_lower": self.root_lower,
            "worst_active_lower": self.worst_active_lower,
            "median_active_lower": self.median_active_lower,
            "accepted_nodes": self.accepted_nodes,
            "sibling_group_count": self.sibling_group_count,
            "source_elapsed_ns": self.source_elapsed_ns,
            "queue_elapsed_ns": self.queue_elapsed_ns,
            "whole_elapsed_ns": self.whole_elapsed_ns,
            "cache_miss_count": self.cache_miss_count,
            "cache_hit_count": self.cache_hit_count,
            "branch_execution_count": self.branch_execution_count,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveBranchProfilePhaseIR:
    plan_hash: str
    original_clause_index: int
    phase_id: str
    primitive_calls: int
    total_ns: int
    cumulative_ns: int
    profile_queue_elapsed_ns: int

    def validate(self) -> None:
        if (
            not _is_sha256(self.plan_hash)
            or self.original_clause_index not in (2, 3)
            or self.phase_id
            not in {
                "branch_program",
                "enumerate_candidates",
                "materialize_children",
                "evaluate_child_bounds",
            }
            or self.primitive_calls < 1
            or min(self.total_ns, self.cumulative_ns) < 0
            or self.cumulative_ns < self.total_ns
            or self.profile_queue_elapsed_ns <= 0
            or self.cumulative_ns > self.profile_queue_elapsed_ns
        ):
            raise ValueError("objective-branch profile phase IR differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "plan_hash": self.plan_hash,
            "original_clause_index": self.original_clause_index,
            "phase_id": self.phase_id,
            "primitive_calls": self.primitive_calls,
            "total_ns": self.total_ns,
            "cumulative_ns": self.cumulative_ns,
            "profile_queue_elapsed_ns": self.profile_queue_elapsed_ns,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveBranchCostDecisionIR:
    plan_hash: str
    frontier_improvements: Tuple[Tuple[str, float], ...]
    queue_ratios: Tuple[Tuple[str, float], ...]
    branch_program_shares: Tuple[Tuple[str, float], ...]
    frontier_order_retained: bool
    scoring_cost_dominant: bool
    next_route: str
    reason: str

    def validate(self) -> None:
        frontier = dict(self.frontier_improvements)
        ratios = dict(self.queue_ratios)
        shares = dict(self.branch_program_shares)
        if (
            not _is_sha256(self.plan_hash)
            or len(frontier) != 8
            or set(ratios) != {"clause_2", "clause_3"}
            or set(shares) != {"clause_2", "clause_3"}
            or any(not math.isfinite(value) for value in frontier.values())
            or any(
                not math.isfinite(value) or value <= 0.0 for value in ratios.values()
            )
            or any(not math.isfinite(value) or value < 0.0 for value in shares.values())
            or self.next_route
            not in {
                "optimize_scorer_ownership",
                "attribute_atomic_tail_scheduling",
                "freeze_objective_branch_production",
            }
            or not self.reason
        ):
            raise ValueError("objective-branch cost Decision IR differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "plan_hash": self.plan_hash,
            "frontier_improvements": dict(self.frontier_improvements),
            "queue_ratios": dict(self.queue_ratios),
            "branch_program_shares": dict(self.branch_program_shares),
            "frontier_order_retained": self.frontier_order_retained,
            "scoring_cost_dominant": self.scoring_cost_dominant,
            "next_route": self.next_route,
            "reason": self.reason,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveBranchCostTaskIRUnit:
    sequence: int
    task_id: str
    kind: NativeObjectiveBranchCostTaskKind
    dependency_task_ids: Tuple[str, ...]
    input_hashes: Tuple[Tuple[str, str], ...]
    output_hash: str

    def validate(self) -> None:
        if (
            self.sequence < 0
            or not self.task_id
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or not self.input_hashes
            or len(self.input_hashes) != len(dict(self.input_hashes))
            or any(not key or not _is_sha256(value) for key, value in self.input_hashes)
            or not _is_sha256(self.output_hash)
        ):
            raise ValueError("objective-branch cost Task IR unit differs")

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
class NativeObjectiveBranchCostTaskIRModule:
    plan_hash: str
    tasks: Tuple[NativeObjectiveBranchCostTaskIRUnit, ...]
    prefix_hashes: Tuple[str, ...]
    wall_hashes: Tuple[str, ...]
    profile_hashes: Tuple[str, ...]
    decision_hash: str
    schema_version: str = OBJECTIVE_BRANCH_COST_TASK_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != OBJECTIVE_BRANCH_COST_TASK_SCHEMA_VERSION
            or not _is_sha256(self.plan_hash)
            or tuple(task.sequence for task in self.tasks)
            != tuple(range(len(self.tasks)))
            or tuple(task.kind for task in self.tasks)
            != tuple(NativeObjectiveBranchCostTaskKind)
            or len({task.task_id for task in self.tasks}) != len(self.tasks)
            or len(self.prefix_hashes) != 16
            or len(self.wall_hashes) != 12
            or len(self.profile_hashes) != 8
            or any(
                not _is_sha256(value)
                for value in (
                    *self.prefix_hashes,
                    *self.wall_hashes,
                    *self.profile_hashes,
                    self.decision_hash,
                )
            )
        ):
            raise ValueError("objective-branch cost Task IR module differs")
        known: set[str] = set()
        for task in self.tasks:
            task.validate()
            if any(value not in known for value in task.dependency_task_ids):
                raise ValueError("objective-branch cost Task dependency differs")
            known.add(task.task_id)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "tasks": [task.to_dict() for task in self.tasks],
            "prefix_hashes": list(self.prefix_hashes),
            "wall_hashes": list(self.wall_hashes),
            "profile_hashes": list(self.profile_hashes),
            "decision_hash": self.decision_hash,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveBranchCostScheduleActionIR:
    sequence: int
    action_id: str
    task_id: str
    kind: NativeObjectiveBranchCostTaskKind
    task_hash: str

    def validate(self) -> None:
        if (
            self.sequence < 0
            or not self.action_id
            or not self.task_id
            or not _is_sha256(self.task_hash)
        ):
            raise ValueError("objective-branch cost Schedule action differs")

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
class NativeObjectiveBranchCostScheduleIR:
    plan_hash: str
    task_module_hash: str
    actions: Tuple[NativeObjectiveBranchCostScheduleActionIR, ...]
    schema_version: str = OBJECTIVE_BRANCH_COST_SCHEDULE_SCHEMA_VERSION

    def validate_against(
        self, task_module: NativeObjectiveBranchCostTaskIRModule
    ) -> None:
        task_module.validate()
        if (
            self.schema_version != OBJECTIVE_BRANCH_COST_SCHEDULE_SCHEMA_VERSION
            or self.plan_hash != task_module.plan_hash
            or self.task_module_hash != task_module.stable_hash()
            or len(self.actions) != len(task_module.tasks)
        ):
            raise ValueError("objective-branch cost Schedule IR differs")
        for action, task in zip(self.actions, task_module.tasks):
            action.validate()
            if (
                action.sequence != task.sequence
                or action.task_id != task.task_id
                or action.kind != task.kind
                or action.task_hash != task.stable_hash()
            ):
                raise ValueError("objective-branch cost Schedule/Task binding differs")

    def to_dict(
        self, task_module: NativeObjectiveBranchCostTaskIRModule
    ) -> dict[str, object]:
        self.validate_against(task_module)
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "task_module_hash": self.task_module_hash,
            "actions": [action.to_dict() for action in self.actions],
        }

    def stable_hash(self, task_module: NativeObjectiveBranchCostTaskIRModule) -> str:
        return _canonical_hash(self.to_dict(task_module))


def lower_native_objective_branch_cost_schedule(
    plan: NativeObjectiveBranchCostAttributionPlanIR,
    prefixes: Tuple[NativeObjectiveBranchPrefixAttributionIR, ...],
    walls: Tuple[NativeObjectiveBranchWallAttributionIR, ...],
    profiles: Tuple[NativeObjectiveBranchProfilePhaseIR, ...],
    decision: NativeObjectiveBranchCostDecisionIR,
) -> tuple[
    NativeObjectiveBranchCostTaskIRModule, NativeObjectiveBranchCostScheduleIR
]:  # pylint: disable=too-many-locals
    plan.validate()
    for collection in (prefixes, walls, profiles):
        for item in collection:
            item.validate()
            if item.plan_hash != plan.stable_hash():
                raise ValueError("objective-branch cost evidence Plan differs")
    decision.validate()
    if decision.plan_hash != plan.stable_hash():
        raise ValueError("objective-branch cost Decision Plan differs")
    outputs = (
        _canonical_hash(
            {
                "source_pilot_hash": plan.source_pilot_hash,
                "source_formal_hash": plan.source_formal_hash,
            }
        ),
        _canonical_hash([item.to_dict() for item in prefixes]),
        _canonical_hash([item.to_dict() for item in walls]),
        _canonical_hash([item.to_dict() for item in profiles]),
        decision.stable_hash(),
        _canonical_hash(
            {
                "decision_hash": decision.stable_hash(),
                "performance_claimed": False,
            }
        ),
    )
    tasks: list[NativeObjectiveBranchCostTaskIRUnit] = []
    for sequence, (kind, output_hash) in enumerate(
        zip(NativeObjectiveBranchCostTaskKind, outputs)
    ):
        task_id = f"{plan.plan_id}:{kind.value}:{sequence:02d}"
        dependencies = () if sequence == 0 else (tasks[-1].task_id,)
        input_hashes = (
            ("plan_hash", plan.stable_hash()),
            (
                "dependency_hash",
                plan.stable_hash() if sequence == 0 else tasks[-1].output_hash,
            ),
        )
        task = NativeObjectiveBranchCostTaskIRUnit(
            sequence=sequence,
            task_id=task_id,
            kind=kind,
            dependency_task_ids=dependencies,
            input_hashes=input_hashes,
            output_hash=output_hash,
        )
        task.validate()
        tasks.append(task)
    task_module = NativeObjectiveBranchCostTaskIRModule(
        plan_hash=plan.stable_hash(),
        tasks=tuple(tasks),
        prefix_hashes=tuple(item.stable_hash() for item in prefixes),
        wall_hashes=tuple(item.stable_hash() for item in walls),
        profile_hashes=tuple(item.stable_hash() for item in profiles),
        decision_hash=decision.stable_hash(),
    )
    task_module.validate()
    schedule = NativeObjectiveBranchCostScheduleIR(
        plan_hash=plan.stable_hash(),
        task_module_hash=task_module.stable_hash(),
        actions=tuple(
            NativeObjectiveBranchCostScheduleActionIR(
                sequence=task.sequence,
                action_id=f"objective-branch-cost.launch.{task.sequence:04d}",
                task_id=task.task_id,
                kind=task.kind,
                task_hash=task.stable_hash(),
            )
            for task in task_module.tasks
        ),
    )
    schedule.validate_against(task_module)
    return task_module, schedule


__all__ = [
    "NativeObjectiveBranchCostAttributionPlanIR",
    "NativeObjectiveBranchCostDecisionIR",
    "NativeObjectiveBranchCostScheduleIR",
    "NativeObjectiveBranchCostTaskIRModule",
    "NativeObjectiveBranchCostTaskKind",
    "NativeObjectiveBranchPrefixAttributionIR",
    "NativeObjectiveBranchProfilePhaseIR",
    "NativeObjectiveBranchWallAttributionIR",
    "lower_native_objective_branch_cost_schedule",
]
