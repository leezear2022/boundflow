"""Typed IR for exact-frontier tightness attribution and one-variable replay."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,missing-class-docstring,too-many-arguments

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Optional, Tuple

FRONTIER_TIGHTNESS_PLAN_SCHEMA_VERSION = (
    "boundflow.frontier-tightness-attribution-plan/v1"
)
FRONTIER_TIGHTNESS_TASK_SCHEMA_VERSION = (
    "boundflow.frontier-tightness-attribution-task/v1"
)
FRONTIER_TIGHTNESS_SCHEDULE_SCHEMA_VERSION = (
    "boundflow.frontier-tightness-attribution-schedule/v1"
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


class NativeFrontierTightnessTaskKind(Enum):
    """Closed attribution pipeline; no search-control mutation is permitted."""

    ADMIT_SOURCE = "admit_source"
    ENUMERATE_FRONTIER = "enumerate_frontier"
    SUMMARIZE_SOURCE = "summarize_source"
    REPLAY_BASELINE = "replay_baseline"
    EVALUATE_CANDIDATE = "evaluate_candidate"
    DECIDE = "decide"
    EMIT = "emit"


@dataclass(frozen=True)
class NativeFrontierTightnessAttributionPlanIR:
    """Exact source identity and preregistered single-variable acceptance gate."""

    plan_id: str
    source_execution_hash: str
    source_plan_hash: str
    source_queue_trace_hash: str
    objective_hash: str
    threshold_hash: str
    original_clause_index: int
    active_node_split_hashes: Tuple[Tuple[str, str], ...]
    baseline_optimizer_policy_hash: str
    candidate_optimizer_policy_hash: str
    baseline_optimizer_steps: int
    candidate_optimizer_steps: int
    required_active_depth: int
    required_active_nodes: int
    lower_delta_tolerance: float = 1e-5
    minimum_worst_lower_improvement: float = 1.0
    minimum_improved_nodes: int = 12
    candidate_id: str = "optimizer_steps_15"
    frozen_variables: Tuple[str, ...] = (
        "objective",
        "threshold",
        "split_state",
        "ancestral_refinement",
        "parent_warm_state",
        "sibling_grouping",
        "dtype",
        "device",
    )
    semantics_owner: str = "boundflow_frontier_tightness_attribution"
    performance_claimed: bool = False
    schema_version: str = FRONTIER_TIGHTNESS_PLAN_SCHEMA_VERSION

    def validate(self) -> None:
        node_ids = tuple(
            node_id for node_id, _split_hash in self.active_node_split_hashes
        )
        split_hashes = tuple(
            split_hash for _node_id, split_hash in self.active_node_split_hashes
        )
        if (
            self.schema_version != FRONTIER_TIGHTNESS_PLAN_SCHEMA_VERSION
            or not self.plan_id
            or self.original_clause_index < 0
            or any(
                not _is_sha256(value)
                for value in (
                    self.source_execution_hash,
                    self.source_plan_hash,
                    self.source_queue_trace_hash,
                    self.objective_hash,
                    self.threshold_hash,
                    self.baseline_optimizer_policy_hash,
                    self.candidate_optimizer_policy_hash,
                )
            )
            or self.baseline_optimizer_policy_hash
            == self.candidate_optimizer_policy_hash
            or self.baseline_optimizer_steps != 5
            or self.candidate_optimizer_steps != 15
            or self.required_active_depth < 1
            or self.required_active_nodes < 1
            or len(node_ids) != self.required_active_nodes
            or len(node_ids) != len(set(node_ids))
            or any(not node_id for node_id in node_ids)
            or any(not _is_sha256(value) for value in split_hashes)
            or not math.isfinite(self.lower_delta_tolerance)
            or self.lower_delta_tolerance <= 0.0
            or not math.isfinite(self.minimum_worst_lower_improvement)
            or self.minimum_worst_lower_improvement <= 0.0
            or self.minimum_improved_nodes < 1
            or self.minimum_improved_nodes > self.required_active_nodes
            or self.candidate_id != "optimizer_steps_15"
            or len(self.frozen_variables) != len(set(self.frozen_variables))
            or set(self.frozen_variables)
            != {
                "objective",
                "threshold",
                "split_state",
                "ancestral_refinement",
                "parent_warm_state",
                "sibling_grouping",
                "dtype",
                "device",
            }
            or self.semantics_owner != "boundflow_frontier_tightness_attribution"
            or self.performance_claimed is not False
        ):
            raise ValueError("frontier tightness attribution Plan IR differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "source_execution_hash": self.source_execution_hash,
            "source_plan_hash": self.source_plan_hash,
            "source_queue_trace_hash": self.source_queue_trace_hash,
            "objective_hash": self.objective_hash,
            "threshold_hash": self.threshold_hash,
            "original_clause_index": self.original_clause_index,
            "active_node_split_hashes": dict(self.active_node_split_hashes),
            "baseline_optimizer_policy_hash": self.baseline_optimizer_policy_hash,
            "candidate_optimizer_policy_hash": self.candidate_optimizer_policy_hash,
            "baseline_optimizer_steps": self.baseline_optimizer_steps,
            "candidate_optimizer_steps": self.candidate_optimizer_steps,
            "required_active_depth": self.required_active_depth,
            "required_active_nodes": self.required_active_nodes,
            "lower_delta_tolerance": self.lower_delta_tolerance,
            "minimum_worst_lower_improvement": (self.minimum_worst_lower_improvement),
            "minimum_improved_nodes": self.minimum_improved_nodes,
            "candidate_id": self.candidate_id,
            "frozen_variables": list(self.frozen_variables),
            "semantics_owner": self.semantics_owner,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeFrontierNodeAttributionIR:
    """Source-node bound, path, refinement, and optimizer-state attribution."""

    node_id: str
    parent_node_id: Optional[str]
    split_state_hash: str
    evaluation_hash: str
    depth: int
    active: bool
    lower: float
    upper: float
    proof_deficit: float
    parent_lower_gain: Optional[float]
    refinement_plan_hash: str
    refinement_semantic_trace_hash: str
    final_intermediate_bounds_hash: str
    selected_target_count: int
    tightened_neuron_count: int
    width_reduction_sum: float
    initial_ambiguous_count: int
    final_ambiguous_count: int
    alpha_count: int
    alpha_boundary_count: int
    alpha_interior_count: int
    beta_count: int
    beta_positive_count: int

    def validate(self) -> None:
        if (
            not self.node_id
            or any(
                not _is_sha256(value)
                for value in (
                    self.split_state_hash,
                    self.evaluation_hash,
                    self.refinement_plan_hash,
                    self.refinement_semantic_trace_hash,
                    self.final_intermediate_bounds_hash,
                )
            )
            or self.depth < 0
            or (self.depth == 0) != (self.parent_node_id is None)
            or not all(
                math.isfinite(value)
                for value in (self.lower, self.upper, self.proof_deficit)
            )
            or self.lower > self.upper
            or (
                self.parent_lower_gain is not None
                and not math.isfinite(self.parent_lower_gain)
            )
            or (self.depth == 0) != (self.parent_lower_gain is None)
            or min(
                self.selected_target_count,
                self.tightened_neuron_count,
                self.initial_ambiguous_count,
                self.final_ambiguous_count,
                self.alpha_count,
                self.alpha_boundary_count,
                self.alpha_interior_count,
                self.beta_count,
                self.beta_positive_count,
            )
            < 0
            or self.final_ambiguous_count > self.initial_ambiguous_count
            or self.alpha_count != self.alpha_boundary_count + self.alpha_interior_count
            or self.beta_positive_count > self.beta_count
            or not math.isfinite(self.width_reduction_sum)
            or self.width_reduction_sum < 0.0
        ):
            raise ValueError("frontier node attribution differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "node_id": self.node_id,
            "split_state_hash": self.split_state_hash,
            "evaluation_hash": self.evaluation_hash,
            "depth": self.depth,
            "active": self.active,
            "lower": self.lower,
            "upper": self.upper,
            "proof_deficit": self.proof_deficit,
            "refinement_plan_hash": self.refinement_plan_hash,
            "refinement_semantic_trace_hash": self.refinement_semantic_trace_hash,
            "final_intermediate_bounds_hash": self.final_intermediate_bounds_hash,
            "selected_target_count": self.selected_target_count,
            "tightened_neuron_count": self.tightened_neuron_count,
            "width_reduction_sum": self.width_reduction_sum,
            "initial_ambiguous_count": self.initial_ambiguous_count,
            "final_ambiguous_count": self.final_ambiguous_count,
            "alpha_count": self.alpha_count,
            "alpha_boundary_count": self.alpha_boundary_count,
            "alpha_interior_count": self.alpha_interior_count,
            "beta_count": self.beta_count,
            "beta_positive_count": self.beta_positive_count,
        }
        if self.parent_node_id is not None:
            payload["parent_node_id"] = self.parent_node_id
            payload["parent_lower_gain"] = self.parent_lower_gain
        return payload

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeFrontierCandidateNodeIR:
    """Exact sibling-batch baseline replay and steps15 counterfactual."""

    node_id: str
    sibling_batch_index: int
    split_state_hash: str
    source_evaluation_hash: str
    source_refinement_hash: str
    baseline_refinement_hash: str
    candidate_refinement_hash: str
    baseline_selected_state_hash: str
    candidate_selected_state_hash: str
    source_lower: float
    source_upper: float
    replay_lower: float
    replay_upper: float
    candidate_lower: float
    candidate_upper: float
    replay_lower_diff: float
    replay_upper_diff: float
    candidate_lower_delta: float

    def validate(self) -> None:
        if (
            not self.node_id
            or self.sibling_batch_index < 0
            or any(
                not _is_sha256(value)
                for value in (
                    self.split_state_hash,
                    self.source_evaluation_hash,
                    self.source_refinement_hash,
                    self.baseline_refinement_hash,
                    self.candidate_refinement_hash,
                    self.baseline_selected_state_hash,
                    self.candidate_selected_state_hash,
                )
            )
            or self.source_refinement_hash != self.baseline_refinement_hash
            or self.source_refinement_hash != self.candidate_refinement_hash
            or not all(
                math.isfinite(value)
                for value in (
                    self.source_lower,
                    self.source_upper,
                    self.replay_lower,
                    self.replay_upper,
                    self.candidate_lower,
                    self.candidate_upper,
                    self.replay_lower_diff,
                    self.replay_upper_diff,
                    self.candidate_lower_delta,
                )
            )
            or self.source_lower > self.source_upper
            or self.replay_lower > self.replay_upper
            or self.candidate_lower > self.candidate_upper
            or abs(self.replay_lower_diff - (self.replay_lower - self.source_lower))
            > 1e-9
            or abs(self.replay_upper_diff - (self.replay_upper - self.source_upper))
            > 1e-9
            or abs(
                self.candidate_lower_delta - (self.candidate_lower - self.replay_lower)
            )
            > 1e-9
        ):
            raise ValueError("frontier candidate node evidence differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "node_id": self.node_id,
            "sibling_batch_index": self.sibling_batch_index,
            "split_state_hash": self.split_state_hash,
            "source_evaluation_hash": self.source_evaluation_hash,
            "source_refinement_hash": self.source_refinement_hash,
            "baseline_refinement_hash": self.baseline_refinement_hash,
            "candidate_refinement_hash": self.candidate_refinement_hash,
            "baseline_selected_state_hash": self.baseline_selected_state_hash,
            "candidate_selected_state_hash": self.candidate_selected_state_hash,
            "source_lower": self.source_lower,
            "source_upper": self.source_upper,
            "replay_lower": self.replay_lower,
            "replay_upper": self.replay_upper,
            "candidate_lower": self.candidate_lower,
            "candidate_upper": self.candidate_upper,
            "replay_lower_diff": self.replay_lower_diff,
            "replay_upper_diff": self.replay_upper_diff,
            "candidate_lower_delta": self.candidate_lower_delta,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeFrontierTightnessDecisionIR:
    """Recomputed preregistered GO/NO-GO decision for one original clause."""

    plan_hash: str
    source_coverage_passed: bool
    baseline_replay_passed: bool
    candidate_bounds_valid: bool
    active_node_count: int
    improved_node_count: int
    regressed_node_count: int
    replay_lower_max_abs_diff: float
    replay_upper_max_abs_diff: float
    minimum_candidate_lower_delta: float
    median_candidate_lower_delta: float
    source_worst_active_lower: float
    replay_worst_active_lower: float
    candidate_worst_active_lower: float
    worst_active_lower_improvement: float
    go: bool
    reason: str
    candidate_id: str = "optimizer_steps_15"

    def validate(self) -> None:
        if (
            not _is_sha256(self.plan_hash)
            or self.active_node_count < 1
            or min(self.improved_node_count, self.regressed_node_count) < 0
            or max(self.improved_node_count, self.regressed_node_count)
            > self.active_node_count
            or not all(
                math.isfinite(value)
                for value in (
                    self.replay_lower_max_abs_diff,
                    self.replay_upper_max_abs_diff,
                    self.minimum_candidate_lower_delta,
                    self.median_candidate_lower_delta,
                    self.source_worst_active_lower,
                    self.replay_worst_active_lower,
                    self.candidate_worst_active_lower,
                    self.worst_active_lower_improvement,
                )
            )
            or min(
                self.replay_lower_max_abs_diff,
                self.replay_upper_max_abs_diff,
            )
            < 0.0
            or abs(
                self.worst_active_lower_improvement
                - (self.candidate_worst_active_lower - self.replay_worst_active_lower)
            )
            > 1e-9
            or not self.reason
            or self.candidate_id != "optimizer_steps_15"
        ):
            raise ValueError("frontier tightness decision differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "plan_hash": self.plan_hash,
            "source_coverage_passed": self.source_coverage_passed,
            "baseline_replay_passed": self.baseline_replay_passed,
            "candidate_bounds_valid": self.candidate_bounds_valid,
            "active_node_count": self.active_node_count,
            "improved_node_count": self.improved_node_count,
            "regressed_node_count": self.regressed_node_count,
            "replay_lower_max_abs_diff": self.replay_lower_max_abs_diff,
            "replay_upper_max_abs_diff": self.replay_upper_max_abs_diff,
            "minimum_candidate_lower_delta": self.minimum_candidate_lower_delta,
            "median_candidate_lower_delta": self.median_candidate_lower_delta,
            "source_worst_active_lower": self.source_worst_active_lower,
            "replay_worst_active_lower": self.replay_worst_active_lower,
            "candidate_worst_active_lower": self.candidate_worst_active_lower,
            "worst_active_lower_improvement": self.worst_active_lower_improvement,
            "go": self.go,
            "reason": self.reason,
            "candidate_id": self.candidate_id,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeFrontierTightnessTaskIRUnit:
    sequence: int
    task_id: str
    kind: NativeFrontierTightnessTaskKind
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
            or any(
                not name or not _is_sha256(value) for name, value in self.input_hashes
            )
            or not _is_sha256(self.output_hash)
        ):
            raise ValueError("frontier tightness Task IR unit differs")

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
class NativeFrontierTightnessTaskIRModule:
    plan_hash: str
    tasks: Tuple[NativeFrontierTightnessTaskIRUnit, ...]
    node_attribution_hashes: Tuple[str, ...]
    candidate_node_hashes: Tuple[str, ...]
    decision_hash: str
    schema_version: str = FRONTIER_TIGHTNESS_TASK_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != FRONTIER_TIGHTNESS_TASK_SCHEMA_VERSION
            or not _is_sha256(self.plan_hash)
            or not self.tasks
            or tuple(task.sequence for task in self.tasks)
            != tuple(range(len(self.tasks)))
            or len({task.task_id for task in self.tasks}) != len(self.tasks)
            or not self.node_attribution_hashes
            or not self.candidate_node_hashes
            or any(
                not _is_sha256(value)
                for value in (
                    *self.node_attribution_hashes,
                    *self.candidate_node_hashes,
                    self.decision_hash,
                )
            )
        ):
            raise ValueError("frontier tightness Task IR module differs")
        known: set[str] = set()
        for task in self.tasks:
            task.validate()
            if any(dependency not in known for dependency in task.dependency_task_ids):
                raise ValueError("frontier tightness Task dependency differs")
            known.add(task.task_id)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "tasks": [task.to_dict() for task in self.tasks],
            "node_attribution_hashes": list(self.node_attribution_hashes),
            "candidate_node_hashes": list(self.candidate_node_hashes),
            "decision_hash": self.decision_hash,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeFrontierTightnessScheduleActionIR:
    sequence: int
    action_id: str
    task_id: str
    kind: NativeFrontierTightnessTaskKind
    task_hash: str

    def validate(self) -> None:
        if (
            self.sequence < 0
            or not self.action_id
            or not self.task_id
            or not _is_sha256(self.task_hash)
        ):
            raise ValueError("frontier tightness Schedule action differs")

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
class NativeFrontierTightnessScheduleIR:
    plan_hash: str
    task_ir_hash: str
    actions: Tuple[NativeFrontierTightnessScheduleActionIR, ...]
    schema_version: str = FRONTIER_TIGHTNESS_SCHEDULE_SCHEMA_VERSION

    def validate_against(self, task_ir: NativeFrontierTightnessTaskIRModule) -> None:
        task_ir.validate()
        if (
            self.schema_version != FRONTIER_TIGHTNESS_SCHEDULE_SCHEMA_VERSION
            or self.plan_hash != task_ir.plan_hash
            or self.task_ir_hash != task_ir.stable_hash()
            or len(self.actions) != len(task_ir.tasks)
        ):
            raise ValueError("frontier tightness Schedule IR differs")
        for action, task in zip(self.actions, task_ir.tasks):
            action.validate()
            if (
                action.sequence != task.sequence
                or action.task_id != task.task_id
                or action.kind != task.kind
                or action.task_hash != task.stable_hash()
            ):
                raise ValueError("frontier tightness Schedule/Task binding differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "actions": [action.to_dict() for action in self.actions],
        }

    def stable_hash(self, task_ir: NativeFrontierTightnessTaskIRModule) -> str:
        self.validate_against(task_ir)
        return _canonical_hash(self.to_dict())


def _task(
    tasks: list[NativeFrontierTightnessTaskIRUnit],
    plan: NativeFrontierTightnessAttributionPlanIR,
    *,
    kind: NativeFrontierTightnessTaskKind,
    dependencies: Tuple[str, ...],
    inputs: Tuple[Tuple[str, str], ...],
    output: object,
) -> str:
    task_id = f"{plan.plan_id}:{kind.value}"
    unit = NativeFrontierTightnessTaskIRUnit(
        sequence=len(tasks),
        task_id=task_id,
        kind=kind,
        dependency_task_ids=dependencies,
        input_hashes=tuple(sorted(inputs)),
        output_hash=_canonical_hash(output),
    )
    unit.validate()
    tasks.append(unit)
    return task_id


def lower_native_frontier_tightness_attribution_schedule(
    plan: NativeFrontierTightnessAttributionPlanIR,
    node_rows: Tuple[NativeFrontierNodeAttributionIR, ...],
    candidate_rows: Tuple[NativeFrontierCandidateNodeIR, ...],
    decision: NativeFrontierTightnessDecisionIR,
) -> tuple[NativeFrontierTightnessTaskIRModule, NativeFrontierTightnessScheduleIR]:
    """Lower attribution evidence into a closed seven-stage Task/Schedule."""

    plan.validate()
    for node_row in node_rows:
        node_row.validate()
    for candidate_row in candidate_rows:
        candidate_row.validate()
    decision.validate()
    if decision.plan_hash != plan.stable_hash():
        raise ValueError("frontier tightness Decision/Plan binding differs")
    tasks: list[NativeFrontierTightnessTaskIRUnit] = []
    admit = _task(
        tasks,
        plan,
        kind=NativeFrontierTightnessTaskKind.ADMIT_SOURCE,
        dependencies=(),
        inputs=(("source_execution", plan.source_execution_hash),),
        output=plan.to_dict(),
    )
    enumerate_frontier = _task(
        tasks,
        plan,
        kind=NativeFrontierTightnessTaskKind.ENUMERATE_FRONTIER,
        dependencies=(admit,),
        inputs=(("source_queue", plan.source_queue_trace_hash),),
        output=dict(plan.active_node_split_hashes),
    )
    summarize = _task(
        tasks,
        plan,
        kind=NativeFrontierTightnessTaskKind.SUMMARIZE_SOURCE,
        dependencies=(enumerate_frontier,),
        inputs=(("plan", plan.stable_hash()),),
        output=[row.stable_hash() for row in node_rows],
    )
    replay = _task(
        tasks,
        plan,
        kind=NativeFrontierTightnessTaskKind.REPLAY_BASELINE,
        dependencies=(summarize,),
        inputs=(("baseline_policy", plan.baseline_optimizer_policy_hash),),
        output=[
            _canonical_hash(
                {
                    "node_id": row.node_id,
                    "replay_lower": row.replay_lower,
                    "replay_upper": row.replay_upper,
                    "refinement": row.baseline_refinement_hash,
                }
            )
            for row in candidate_rows
        ],
    )
    candidate = _task(
        tasks,
        plan,
        kind=NativeFrontierTightnessTaskKind.EVALUATE_CANDIDATE,
        dependencies=(replay,),
        inputs=(("candidate_policy", plan.candidate_optimizer_policy_hash),),
        output=[row.stable_hash() for row in candidate_rows],
    )
    decide = _task(
        tasks,
        plan,
        kind=NativeFrontierTightnessTaskKind.DECIDE,
        dependencies=(candidate,),
        inputs=(
            (
                "candidate_rows",
                _canonical_hash([row.stable_hash() for row in candidate_rows]),
            ),
        ),
        output=decision.to_dict(),
    )
    _task(
        tasks,
        plan,
        kind=NativeFrontierTightnessTaskKind.EMIT,
        dependencies=(decide,),
        inputs=(("decision", decision.stable_hash()),),
        output={"go": decision.go, "reason": decision.reason},
    )
    task_ir = NativeFrontierTightnessTaskIRModule(
        plan_hash=plan.stable_hash(),
        tasks=tuple(tasks),
        node_attribution_hashes=tuple(row.stable_hash() for row in node_rows),
        candidate_node_hashes=tuple(row.stable_hash() for row in candidate_rows),
        decision_hash=decision.stable_hash(),
    )
    task_ir.validate()
    schedule = NativeFrontierTightnessScheduleIR(
        plan_hash=plan.stable_hash(),
        task_ir_hash=task_ir.stable_hash(),
        actions=tuple(
            NativeFrontierTightnessScheduleActionIR(
                sequence=task.sequence,
                action_id=f"{task.task_id}:action",
                task_id=task.task_id,
                kind=task.kind,
                task_hash=task.stable_hash(),
            )
            for task in task_ir.tasks
        ),
    )
    schedule.validate_against(task_ir)
    return task_ir, schedule


__all__ = [
    "NativeFrontierCandidateNodeIR",
    "NativeFrontierNodeAttributionIR",
    "NativeFrontierTightnessAttributionPlanIR",
    "NativeFrontierTightnessDecisionIR",
    "NativeFrontierTightnessScheduleIR",
    "NativeFrontierTightnessTaskIRModule",
    "NativeFrontierTightnessTaskKind",
    "lower_native_frontier_tightness_attribution_schedule",
]
