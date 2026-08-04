"""Typed multi-clause priority and cooperative time-slice IR."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,too-many-lines,too-many-locals

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Literal, Optional, Tuple

MULTI_CLAUSE_ANYTIME_POLICY_SCHEMA_VERSION = "boundflow.multi-clause-anytime-policy/v1"
MULTI_CLAUSE_ANYTIME_PLAN_SCHEMA_VERSION = "boundflow.multi-clause-anytime-plan/v1"
MULTI_CLAUSE_ANYTIME_CANDIDATE_SCHEMA_VERSION = (
    "boundflow.multi-clause-anytime-candidate/v1"
)
MULTI_CLAUSE_ANYTIME_DECISION_SCHEMA_VERSION = (
    "boundflow.multi-clause-anytime-decision/v1"
)
MULTI_CLAUSE_ANYTIME_TASK_SCHEMA_VERSION = "boundflow.multi-clause-anytime-task/v1"
MULTI_CLAUSE_ANYTIME_SCHEDULE_SCHEMA_VERSION = (
    "boundflow.multi-clause-anytime-schedule/v1"
)
MULTI_CLAUSE_ANYTIME_SLICE_SCHEMA_VERSION = "boundflow.multi-clause-anytime-slice/v1"
MULTI_CLAUSE_ANYTIME_OUTCOME_SCHEMA_VERSION = (
    "boundflow.multi-clause-anytime-outcome/v1"
)
MULTI_CLAUSE_ANYTIME_AGGREGATE_SCHEMA_VERSION = (
    "boundflow.multi-clause-anytime-aggregate/v1"
)
FinalStatus = Literal["verified", "unsafe", "unknown"]


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class NativeMultiClauseAnytimeTaskKind(Enum):
    """Closed static scheduler vocabulary."""

    EXECUTE_FLOOR = "execute_floor"
    RANK_CANDIDATES = "rank_candidates"
    COMPILE_PACKED_PLAN = "compile_packed_plan"
    EXECUTE_PACKED_SLICE = "execute_packed_slice"
    AGGREGATE_ORIGINAL_ORDINALS = "aggregate_original_ordinals"
    EMIT_RESULT = "emit_result"


@dataclass(frozen=True)
class NativeMultiClauseAnytimePolicyIR:
    """Frozen priority and dynamic equal-remaining slice policy."""

    max_selected_clauses: int = 2
    priority_metric: str = "root_lower_margin_desc_ordinal_asc"
    slice_policy: str = "dynamic_equal_remaining_selected_v1"
    cutoff_policy: str = "one_shot_global_expiry_signal_v1"
    schema_version: str = MULTI_CLAUSE_ANYTIME_POLICY_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != MULTI_CLAUSE_ANYTIME_POLICY_SCHEMA_VERSION
            or self.max_selected_clauses != 2
            or self.priority_metric != "root_lower_margin_desc_ordinal_asc"
            or self.slice_policy != "dynamic_equal_remaining_selected_v1"
            or self.cutoff_policy != "one_shot_global_expiry_signal_v1"
        ):
            raise ValueError("multi-clause anytime Policy IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "max_selected_clauses": self.max_selected_clauses,
            "priority_metric": self.priority_metric,
            "slice_policy": self.slice_policy,
            "cutoff_policy": self.cutoff_policy,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeMultiClauseAnytimePlanIR:
    """Static NRIR-31 floor plus two guarded packed-slice slots."""

    plan_id: str
    floor_plan_hash: str
    floor_task_ir_hash: str
    floor_schedule_hash: str
    objective_matrix_hash: str
    thresholds_hash: str
    search_policy_hash: str
    optimizer_policy_hash: str
    allocation_policy: NativeMultiClauseAnytimePolicyIR
    clause_count: int = 9
    packed_max_nodes: int = 31
    packed_max_depth: int = 4
    sibling_group_size: int = 2
    child_refinement_cap: int = 128
    whole_query_timeout_ns: int = 60 * 1_000_000_000
    semantics_owner: str = "boundflow_multi_clause_anytime_priority"
    schema_version: str = MULTI_CLAUSE_ANYTIME_PLAN_SCHEMA_VERSION

    def validate(self) -> None:
        self.allocation_policy.validate()
        hashes = (
            self.floor_plan_hash,
            self.floor_task_ir_hash,
            self.floor_schedule_hash,
            self.objective_matrix_hash,
            self.thresholds_hash,
            self.search_policy_hash,
            self.optimizer_policy_hash,
        )
        if (
            self.schema_version != MULTI_CLAUSE_ANYTIME_PLAN_SCHEMA_VERSION
            or not self.plan_id
            or any(not _is_sha256(value) for value in hashes)
            or self.clause_count != 9
            or (self.packed_max_nodes, self.packed_max_depth) != (31, 4)
            or self.sibling_group_size != 2
            or self.child_refinement_cap != 128
            or self.whole_query_timeout_ns != 60 * 1_000_000_000
            or self.semantics_owner != "boundflow_multi_clause_anytime_priority"
        ):
            raise ValueError("multi-clause anytime Plan IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "floor_plan_hash": self.floor_plan_hash,
            "floor_task_ir_hash": self.floor_task_ir_hash,
            "floor_schedule_hash": self.floor_schedule_hash,
            "objective_matrix_hash": self.objective_matrix_hash,
            "thresholds_hash": self.thresholds_hash,
            "search_policy_hash": self.search_policy_hash,
            "optimizer_policy_hash": self.optimizer_policy_hash,
            "allocation_policy": self.allocation_policy.to_dict(),
            "clause_count": self.clause_count,
            "packed_max_nodes": self.packed_max_nodes,
            "packed_max_depth": self.packed_max_depth,
            "sibling_group_size": self.sibling_group_size,
            "child_refinement_cap": self.child_refinement_cap,
            "whole_query_timeout_ns": self.whole_query_timeout_ns,
            "semantics_owner": self.semantics_owner,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeMultiClauseAnytimeCandidateIR:
    """One accepted unresolved floor clause and its exact source."""

    original_clause_index: int
    threshold: float
    root_lower: float
    root_upper: float
    root_lower_margin: float
    root_refinement_plan_hash: str
    root_refinement_semantic_trace_hash: str
    root_final_intermediate_bounds_hash: str
    schema_version: str = MULTI_CLAUSE_ANYTIME_CANDIDATE_SCHEMA_VERSION

    def validate(self, *, clause_count: int) -> None:
        if (
            self.schema_version != MULTI_CLAUSE_ANYTIME_CANDIDATE_SCHEMA_VERSION
            or not 0 <= self.original_clause_index < clause_count
            or not all(
                math.isfinite(value)
                for value in (
                    self.threshold,
                    self.root_lower,
                    self.root_upper,
                    self.root_lower_margin,
                )
            )
            or self.root_lower > self.root_upper
            or self.root_lower_margin != self.root_lower - self.threshold
            or any(
                not _is_sha256(value)
                for value in (
                    self.root_refinement_plan_hash,
                    self.root_refinement_semantic_trace_hash,
                    self.root_final_intermediate_bounds_hash,
                )
            )
        ):
            raise ValueError("multi-clause anytime candidate is invalid")

    def to_dict(self, *, clause_count: int) -> dict[str, object]:
        self.validate(clause_count=clause_count)
        return {
            "schema_version": self.schema_version,
            "original_clause_index": self.original_clause_index,
            "threshold": self.threshold,
            "root_lower": self.root_lower,
            "root_upper": self.root_upper,
            "root_lower_margin": self.root_lower_margin,
            "root_refinement_plan_hash": self.root_refinement_plan_hash,
            "root_refinement_semantic_trace_hash": (
                self.root_refinement_semantic_trace_hash
            ),
            "root_final_intermediate_bounds_hash": (
                self.root_final_intermediate_bounds_hash
            ),
        }

    def stable_hash(self, *, clause_count: int) -> str:
        return _canonical_hash(self.to_dict(clause_count=clause_count))


@dataclass(frozen=True)
class NativeMultiClauseAnytimeDecisionIR:
    """Ranked dynamic candidates and selected priority prefix."""

    plan_hash: str
    floor_trace_hash: str
    floor_completed_original_clause_indices: Tuple[int, ...]
    floor_status: FinalStatus
    floor_verified_clause_indices: Tuple[int, ...]
    floor_unresolved_clause_indices: Tuple[int, ...]
    floor_unsafe_clause_index: Optional[int]
    candidates: Tuple[NativeMultiClauseAnytimeCandidateIR, ...]
    ranked_original_clause_indices: Tuple[int, ...]
    selected_original_clause_indices: Tuple[int, ...]
    reason: str
    schema_version: str = MULTI_CLAUSE_ANYTIME_DECISION_SCHEMA_VERSION

    def validate_against(self, plan: NativeMultiClauseAnytimePlanIR) -> None:
        plan.validate()
        sequences = (
            self.floor_completed_original_clause_indices,
            self.floor_verified_clause_indices,
            self.floor_unresolved_clause_indices,
        )
        ordinals = tuple(
            candidate.original_clause_index for candidate in self.candidates
        )
        for candidate in self.candidates:
            candidate.validate(clause_count=plan.clause_count)
        expected_rank = tuple(
            candidate.original_clause_index
            for candidate in sorted(
                self.candidates,
                key=lambda item: (-item.root_lower_margin, item.original_clause_index),
            )
        )
        if (
            self.schema_version != MULTI_CLAUSE_ANYTIME_DECISION_SCHEMA_VERSION
            or self.plan_hash != plan.stable_hash()
            or not _is_sha256(self.floor_trace_hash)
            or self.floor_status not in {"verified", "unsafe", "unknown"}
            or any(tuple(sorted(set(values))) != values for values in sequences)
            or any(
                not set(values) <= set(range(plan.clause_count)) for values in sequences
            )
            or len(ordinals) != len(set(ordinals))
            or not set(ordinals) <= set(self.floor_unresolved_clause_indices)
            or self.ranked_original_clause_indices != expected_rank
            or not self.reason
        ):
            raise ValueError("multi-clause anytime Decision IR is invalid")
        admitted = (
            self.floor_status == "unknown"
            and self.floor_completed_original_clause_indices
            == tuple(range(plan.clause_count))
            and self.floor_unsafe_clause_index is None
            and bool(self.candidates)
        )
        expected_selected = (
            expected_rank[: plan.allocation_policy.max_selected_clauses]
            if admitted
            else ()
        )
        expected_reason = (
            "ranked_unresolved_candidates_selected"
            if admitted
            else "floor_not_eligible_for_multi_clause_anytime"
        )
        if (
            self.selected_original_clause_indices != expected_selected
            or self.reason != expected_reason
        ):
            raise ValueError("multi-clause anytime selection differs from policy")

    def candidate(
        self, original_clause_index: int
    ) -> NativeMultiClauseAnytimeCandidateIR:
        return next(
            item
            for item in self.candidates
            if item.original_clause_index == original_clause_index
        )

    def to_dict(self, plan: NativeMultiClauseAnytimePlanIR) -> dict[str, object]:
        self.validate_against(plan)
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "floor_trace_hash": self.floor_trace_hash,
            "floor_completed_original_clause_indices": list(
                self.floor_completed_original_clause_indices
            ),
            "floor_status": self.floor_status,
            "floor_verified_clause_indices": list(self.floor_verified_clause_indices),
            "floor_unresolved_clause_indices": list(
                self.floor_unresolved_clause_indices
            ),
            "floor_unsafe_clause_index": self.floor_unsafe_clause_index,
            "candidates": [
                item.to_dict(clause_count=plan.clause_count) for item in self.candidates
            ],
            "ranked_original_clause_indices": list(self.ranked_original_clause_indices),
            "selected_original_clause_indices": list(
                self.selected_original_clause_indices
            ),
            "reason": self.reason,
        }

    def stable_hash(self, plan: NativeMultiClauseAnytimePlanIR) -> str:
        return _canonical_hash(self.to_dict(plan))


@dataclass(frozen=True)
class NativeMultiClauseAnytimeTaskIRUnit:
    """One static floor, rank, slice-slot, aggregate, or emit task."""

    sequence: int
    task_id: str
    kind: NativeMultiClauseAnytimeTaskKind
    dependency_task_ids: Tuple[str, ...]
    guard: str
    priority_position: Optional[int]
    input_contract_hash: str
    output_contract: str

    def validate(self) -> None:
        slot = self.kind in {
            NativeMultiClauseAnytimeTaskKind.COMPILE_PACKED_PLAN,
            NativeMultiClauseAnytimeTaskKind.EXECUTE_PACKED_SLICE,
        }
        if (
            self.sequence < 0
            or not self.task_id
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or self.guard not in {"always", "selected_slot_available_before_deadline"}
            or slot != (self.guard == "selected_slot_available_before_deadline")
            or slot != (self.priority_position is not None)
            or self.priority_position not in {None, 0, 1}
            or not _is_sha256(self.input_contract_hash)
            or not self.output_contract
        ):
            raise ValueError("multi-clause anytime Task unit is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "dependency_task_ids": list(self.dependency_task_ids),
            "guard": self.guard,
            "priority_position": self.priority_position,
            "input_contract_hash": self.input_contract_hash,
            "output_contract": self.output_contract,
        }


@dataclass(frozen=True)
class NativeMultiClauseAnytimeTaskIRModule:
    """Eight-stage static two-slot Task IR."""

    plan_hash: str
    tasks: Tuple[NativeMultiClauseAnytimeTaskIRUnit, ...]
    schema_version: str = MULTI_CLAUSE_ANYTIME_TASK_SCHEMA_VERSION

    def validate(self) -> None:
        expected_kinds = (
            NativeMultiClauseAnytimeTaskKind.EXECUTE_FLOOR,
            NativeMultiClauseAnytimeTaskKind.RANK_CANDIDATES,
            NativeMultiClauseAnytimeTaskKind.COMPILE_PACKED_PLAN,
            NativeMultiClauseAnytimeTaskKind.EXECUTE_PACKED_SLICE,
            NativeMultiClauseAnytimeTaskKind.COMPILE_PACKED_PLAN,
            NativeMultiClauseAnytimeTaskKind.EXECUTE_PACKED_SLICE,
            NativeMultiClauseAnytimeTaskKind.AGGREGATE_ORIGINAL_ORDINALS,
            NativeMultiClauseAnytimeTaskKind.EMIT_RESULT,
        )
        if (
            self.schema_version != MULTI_CLAUSE_ANYTIME_TASK_SCHEMA_VERSION
            or not _is_sha256(self.plan_hash)
            or len(self.tasks) != len(expected_kinds)
            or tuple(task.sequence for task in self.tasks) != tuple(range(8))
            or tuple(task.kind for task in self.tasks) != expected_kinds
            or len({task.task_id for task in self.tasks}) != len(self.tasks)
        ):
            raise ValueError("multi-clause anytime Task IR is invalid")
        available: set[str] = set()
        for task in self.tasks:
            task.validate()
            if any(item not in available for item in task.dependency_task_ids):
                raise ValueError("multi-clause anytime Task dependency order differs")
            available.add(task.task_id)

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
class NativeMultiClauseAnytimeScheduleActionIR:
    """One sequential action bound to a static Task unit."""

    sequence: int
    action_id: str
    task_id: str
    kind: NativeMultiClauseAnytimeTaskKind
    guard: str
    priority_position: Optional[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "guard": self.guard,
            "priority_position": self.priority_position,
        }


@dataclass(frozen=True)
class NativeMultiClauseAnytimeScheduleIR:
    """Sequential floor-first two-slice Schedule IR."""

    plan_hash: str
    task_ir_hash: str
    actions: Tuple[NativeMultiClauseAnytimeScheduleActionIR, ...]
    schema_version: str = MULTI_CLAUSE_ANYTIME_SCHEDULE_SCHEMA_VERSION

    def validate_against(self, task_ir: NativeMultiClauseAnytimeTaskIRModule) -> None:
        task_ir.validate()
        if (
            self.schema_version != MULTI_CLAUSE_ANYTIME_SCHEDULE_SCHEMA_VERSION
            or self.plan_hash != task_ir.plan_hash
            or self.task_ir_hash != task_ir.stable_hash()
            or len(self.actions) != len(task_ir.tasks)
        ):
            raise ValueError("multi-clause anytime Schedule IR differs")
        for index, (action, task) in enumerate(zip(self.actions, task_ir.tasks)):
            if (
                action.sequence != index
                or action.action_id != f"multi-clause-anytime.launch.{index:04d}"
                or action.task_id != task.task_id
                or action.kind != task.kind
                or action.guard != task.guard
                or action.priority_position != task.priority_position
            ):
                raise ValueError("multi-clause anytime Schedule/Task binding differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "actions": [action.to_dict() for action in self.actions],
            "dispatch": "sequential_floor_rank_dynamic_equal_remaining_slices",
        }

    def stable_hash(self, task_ir: NativeMultiClauseAnytimeTaskIRModule) -> str:
        self.validate_against(task_ir)
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeMultiClauseAnytimeSliceIR:
    """One actual dynamic allocation and optional committed packed result."""

    plan_hash: str
    decision_hash: str
    priority_position: int
    original_clause_index: int
    dispatch_started_elapsed_ns: int
    remaining_before_ns: int
    remaining_selected_count: int
    allocated_slice_ns: int
    slice_cutoff_elapsed_ns: int
    finished_elapsed_ns: int
    source_refinement_plan_hash: str
    source_refinement_semantic_trace_hash: str
    source_final_intermediate_bounds_hash: str
    packed_plan_hash: Optional[str]
    packed_queue_trace_hash: Optional[str]
    packed_verdict_trace_hash: Optional[str]
    packed_status: Optional[FinalStatus]
    accepted_nodes: int
    sibling_group_count: int
    cutoff_signaled: bool
    reason: str
    schema_version: str = MULTI_CLAUSE_ANYTIME_SLICE_SCHEMA_VERSION

    def validate_against(
        self,
        plan: NativeMultiClauseAnytimePlanIR,
        decision: NativeMultiClauseAnytimeDecisionIR,
    ) -> None:
        decision.validate_against(plan)
        present = self.packed_queue_trace_hash is not None
        candidate = decision.candidate(self.original_clause_index)
        if (
            self.schema_version != MULTI_CLAUSE_ANYTIME_SLICE_SCHEMA_VERSION
            or self.plan_hash != plan.stable_hash()
            or self.decision_hash != decision.stable_hash(plan)
            or not 0
            <= self.priority_position
            < len(decision.selected_original_clause_indices)
            or decision.selected_original_clause_indices[self.priority_position]
            != self.original_clause_index
            or min(
                self.dispatch_started_elapsed_ns,
                self.remaining_before_ns,
                self.allocated_slice_ns,
                self.slice_cutoff_elapsed_ns,
                self.finished_elapsed_ns,
                self.accepted_nodes,
                self.sibling_group_count,
            )
            < 0
            or self.remaining_selected_count
            != len(decision.selected_original_clause_indices) - self.priority_position
            or self.remaining_before_ns
            != max(0, plan.whole_query_timeout_ns - self.dispatch_started_elapsed_ns)
            or self.allocated_slice_ns
            != self.remaining_before_ns // self.remaining_selected_count
            or self.slice_cutoff_elapsed_ns
            != min(
                plan.whole_query_timeout_ns,
                self.dispatch_started_elapsed_ns + self.allocated_slice_ns,
            )
            or self.finished_elapsed_ns < self.dispatch_started_elapsed_ns
            or self.source_refinement_plan_hash != candidate.root_refinement_plan_hash
            or self.source_refinement_semantic_trace_hash
            != candidate.root_refinement_semantic_trace_hash
            or self.source_final_intermediate_bounds_hash
            != candidate.root_final_intermediate_bounds_hash
            or present != (self.packed_plan_hash is not None)
            or present != (self.packed_verdict_trace_hash is not None)
            or present != (self.packed_status is not None)
            or any(
                value is not None and not _is_sha256(value)
                for value in (
                    self.packed_plan_hash,
                    self.packed_queue_trace_hash,
                    self.packed_verdict_trace_hash,
                )
            )
            or self.packed_status not in {None, "verified", "unsafe", "unknown"}
            or not self.reason
        ):
            raise ValueError("multi-clause anytime slice is invalid")
        if present:
            if (
                self.accepted_nodes != 1 + 2 * self.sibling_group_count
                or self.reason
                not in {
                    "packed_slice_verified",
                    "packed_slice_unsafe",
                    "packed_slice_unknown",
                }
            ):
                raise ValueError("multi-clause anytime packed slice differs")
        elif (
            self.accepted_nodes != 0
            or self.sibling_group_count != 0
            or self.reason
            not in {
                "prior_slice_unsafe",
                "slice_deadline_before_compile",
                "slice_deadline_after_compile",
                "slice_deadline_during_packed_root",
            }
        ):
            raise ValueError("multi-clause anytime skipped slice differs")

    def to_dict(
        self,
        plan: NativeMultiClauseAnytimePlanIR,
        decision: NativeMultiClauseAnytimeDecisionIR,
    ) -> dict[str, object]:
        self.validate_against(plan, decision)
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "decision_hash": self.decision_hash,
            "priority_position": self.priority_position,
            "original_clause_index": self.original_clause_index,
            "dispatch_started_elapsed_ns": self.dispatch_started_elapsed_ns,
            "remaining_before_ns": self.remaining_before_ns,
            "remaining_selected_count": self.remaining_selected_count,
            "allocated_slice_ns": self.allocated_slice_ns,
            "slice_cutoff_elapsed_ns": self.slice_cutoff_elapsed_ns,
            "finished_elapsed_ns": self.finished_elapsed_ns,
            "source_refinement_plan_hash": self.source_refinement_plan_hash,
            "source_refinement_semantic_trace_hash": (
                self.source_refinement_semantic_trace_hash
            ),
            "source_final_intermediate_bounds_hash": (
                self.source_final_intermediate_bounds_hash
            ),
            "packed_plan_hash": self.packed_plan_hash,
            "packed_queue_trace_hash": self.packed_queue_trace_hash,
            "packed_verdict_trace_hash": self.packed_verdict_trace_hash,
            "packed_status": self.packed_status,
            "accepted_nodes": self.accepted_nodes,
            "sibling_group_count": self.sibling_group_count,
            "cutoff_signaled": self.cutoff_signaled,
            "reason": self.reason,
        }

    def stable_hash(
        self,
        plan: NativeMultiClauseAnytimePlanIR,
        decision: NativeMultiClauseAnytimeDecisionIR,
    ) -> str:
        return _canonical_hash(self.to_dict(plan, decision))


@dataclass(frozen=True)
class NativeMultiClauseAnytimeOutcomeIR:
    """Sound packed verdict projected to one original ordinal."""

    original_clause_index: int
    packed_queue_trace_hash: str
    packed_verdict_trace_hash: str
    status: FinalStatus
    schema_version: str = MULTI_CLAUSE_ANYTIME_OUTCOME_SCHEMA_VERSION

    def validate(self, *, clause_count: int) -> None:
        if (
            self.schema_version != MULTI_CLAUSE_ANYTIME_OUTCOME_SCHEMA_VERSION
            or not 0 <= self.original_clause_index < clause_count
            or not _is_sha256(self.packed_queue_trace_hash)
            or not _is_sha256(self.packed_verdict_trace_hash)
            or self.status not in {"verified", "unsafe", "unknown"}
        ):
            raise ValueError("multi-clause anytime outcome is invalid")

    def to_dict(self, *, clause_count: int) -> dict[str, object]:
        self.validate(clause_count=clause_count)
        return {
            "schema_version": self.schema_version,
            "original_clause_index": self.original_clause_index,
            "packed_queue_trace_hash": self.packed_queue_trace_hash,
            "packed_verdict_trace_hash": self.packed_verdict_trace_hash,
            "status": self.status,
        }


@dataclass(frozen=True)
class NativeMultiClauseAnytimeAggregateIR:
    """Monotone multi-outcome aggregate over all original ordinals."""

    plan_hash: str
    decision_hash: str
    floor_trace_hash: str
    slice_hashes: Tuple[str, ...]
    outcomes: Tuple[NativeMultiClauseAnytimeOutcomeIR, ...]
    floor_status: FinalStatus
    floor_verified_clause_indices: Tuple[int, ...]
    floor_unresolved_clause_indices: Tuple[int, ...]
    floor_unsafe_clause_index: Optional[int]
    final_status: FinalStatus
    final_verified_clause_indices: Tuple[int, ...]
    final_unresolved_clause_indices: Tuple[int, ...]
    final_unsafe_clause_index: Optional[int]
    original_clause_indices: Tuple[int, ...] = tuple(range(9))
    schema_version: str = MULTI_CLAUSE_ANYTIME_AGGREGATE_SCHEMA_VERSION

    def validate_against(
        self,
        plan: NativeMultiClauseAnytimePlanIR,
        decision: NativeMultiClauseAnytimeDecisionIR,
        slices: Tuple[NativeMultiClauseAnytimeSliceIR, ...],
    ) -> None:
        decision.validate_against(plan)
        for item in slices:
            item.validate_against(plan, decision)
        for outcome in self.outcomes:
            outcome.validate(clause_count=plan.clause_count)
        outcome_ordinals = tuple(item.original_clause_index for item in self.outcomes)
        expected_outcomes = tuple(
            NativeMultiClauseAnytimeOutcomeIR(
                original_clause_index=item.original_clause_index,
                packed_queue_trace_hash=item.packed_queue_trace_hash or "",
                packed_verdict_trace_hash=item.packed_verdict_trace_hash or "",
                status=item.packed_status or "unknown",
            )
            for item in slices
            if item.packed_queue_trace_hash is not None
        )
        unsafe_positions = tuple(
            index for index, item in enumerate(slices) if item.packed_status == "unsafe"
        )
        if (
            self.schema_version != MULTI_CLAUSE_ANYTIME_AGGREGATE_SCHEMA_VERSION
            or self.plan_hash != plan.stable_hash()
            or self.decision_hash != decision.stable_hash(plan)
            or self.floor_trace_hash != decision.floor_trace_hash
            or self.slice_hashes
            != tuple(item.stable_hash(plan, decision) for item in slices)
            or self.outcomes != expected_outcomes
            or len(outcome_ordinals) != len(set(outcome_ordinals))
            or not set(outcome_ordinals)
            <= set(decision.selected_original_clause_indices)
            or self.floor_status != decision.floor_status
            or self.floor_verified_clause_indices
            != decision.floor_verified_clause_indices
            or self.floor_unresolved_clause_indices
            != decision.floor_unresolved_clause_indices
            or self.floor_unsafe_clause_index != decision.floor_unsafe_clause_index
            or self.original_clause_indices != tuple(range(plan.clause_count))
            or unsafe_positions
            and any(
                item.packed_queue_trace_hash is not None
                for item in slices[unsafe_positions[0] + 1 :]
            )
        ):
            raise ValueError("multi-clause anytime aggregate is invalid")
        verified = set(self.floor_verified_clause_indices)
        unresolved = list(self.floor_unresolved_clause_indices)
        unsafe = self.floor_unsafe_clause_index
        status = self.floor_status
        for outcome in self.outcomes:
            ordinal = outcome.original_clause_index
            if outcome.status == "verified":
                verified.add(ordinal)
                unresolved = [item for item in unresolved if item != ordinal]
            elif outcome.status == "unsafe":
                status = "unsafe"
                unsafe = ordinal
                break
        if status != "unsafe":
            status = "verified" if len(verified) == plan.clause_count else "unknown"
            unsafe = None
        expected = (
            status,
            tuple(sorted(verified)),
            tuple(unresolved),
            unsafe,
        )
        actual = (
            self.final_status,
            self.final_verified_clause_indices,
            self.final_unresolved_clause_indices,
            self.final_unsafe_clause_index,
        )
        if actual != expected:
            raise ValueError("multi-clause anytime aggregate is non-monotone")

    def to_dict(
        self,
        plan: NativeMultiClauseAnytimePlanIR,
        decision: NativeMultiClauseAnytimeDecisionIR,
        slices: Tuple[NativeMultiClauseAnytimeSliceIR, ...],
    ) -> dict[str, object]:
        self.validate_against(plan, decision, slices)
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "decision_hash": self.decision_hash,
            "floor_trace_hash": self.floor_trace_hash,
            "slice_hashes": list(self.slice_hashes),
            "outcomes": [
                item.to_dict(clause_count=plan.clause_count) for item in self.outcomes
            ],
            "floor_status": self.floor_status,
            "floor_verified_clause_indices": list(self.floor_verified_clause_indices),
            "floor_unresolved_clause_indices": list(
                self.floor_unresolved_clause_indices
            ),
            "floor_unsafe_clause_index": self.floor_unsafe_clause_index,
            "final_status": self.final_status,
            "final_verified_clause_indices": list(self.final_verified_clause_indices),
            "final_unresolved_clause_indices": list(
                self.final_unresolved_clause_indices
            ),
            "final_unsafe_clause_index": self.final_unsafe_clause_index,
            "original_clause_indices": list(self.original_clause_indices),
            "aggregation": "floor_monotone_multi_clause_packed_outcomes",
        }

    def stable_hash(
        self,
        plan: NativeMultiClauseAnytimePlanIR,
        decision: NativeMultiClauseAnytimeDecisionIR,
        slices: Tuple[NativeMultiClauseAnytimeSliceIR, ...],
    ) -> str:
        return _canonical_hash(self.to_dict(plan, decision, slices))


def lower_native_multi_clause_anytime_ir(
    plan: NativeMultiClauseAnytimePlanIR,
) -> tuple[NativeMultiClauseAnytimeTaskIRModule, NativeMultiClauseAnytimeScheduleIR]:
    plan.validate()
    kinds = (
        NativeMultiClauseAnytimeTaskKind.EXECUTE_FLOOR,
        NativeMultiClauseAnytimeTaskKind.RANK_CANDIDATES,
        NativeMultiClauseAnytimeTaskKind.COMPILE_PACKED_PLAN,
        NativeMultiClauseAnytimeTaskKind.EXECUTE_PACKED_SLICE,
        NativeMultiClauseAnytimeTaskKind.COMPILE_PACKED_PLAN,
        NativeMultiClauseAnytimeTaskKind.EXECUTE_PACKED_SLICE,
        NativeMultiClauseAnytimeTaskKind.AGGREGATE_ORIGINAL_ORDINALS,
        NativeMultiClauseAnytimeTaskKind.EMIT_RESULT,
    )
    dependencies = ((), (0,), (1,), (2,), (3,), (4,), (1, 3, 5), (6,))
    positions: tuple[Optional[int], ...] = (None, None, 0, 0, 1, 1, None, None)
    guards = tuple(
        "selected_slot_available_before_deadline" if position is not None else "always"
        for position in positions
    )
    outputs = (
        "native_objective_hard_clause_escalation_execution",
        "multi_clause_anytime_decision_ir",
        "priority_slot_0_packed_plan",
        "priority_slot_0_packed_slice",
        "priority_slot_1_packed_plan",
        "priority_slot_1_packed_slice",
        "multi_clause_anytime_aggregate_ir",
        "multi_clause_anytime_execution",
    )
    task_ids = tuple(
        f"{plan.plan_id}:{kind.value}:{index:02d}" for index, kind in enumerate(kinds)
    )
    tasks = tuple(
        NativeMultiClauseAnytimeTaskIRUnit(
            sequence=index,
            task_id=task_ids[index],
            kind=kind,
            dependency_task_ids=tuple(task_ids[item] for item in dependencies[index]),
            guard=guards[index],
            priority_position=positions[index],
            input_contract_hash=_canonical_hash(
                {
                    "plan_hash": plan.stable_hash(),
                    "kind": kind.value,
                    "dependencies": list(dependencies[index]),
                    "priority_position": positions[index],
                }
            ),
            output_contract=outputs[index],
        )
        for index, kind in enumerate(kinds)
    )
    task_ir = NativeMultiClauseAnytimeTaskIRModule(
        plan_hash=plan.stable_hash(), tasks=tasks
    )
    task_ir.validate()
    schedule = NativeMultiClauseAnytimeScheduleIR(
        plan_hash=plan.stable_hash(),
        task_ir_hash=task_ir.stable_hash(),
        actions=tuple(
            NativeMultiClauseAnytimeScheduleActionIR(
                sequence=index,
                action_id=f"multi-clause-anytime.launch.{index:04d}",
                task_id=task.task_id,
                kind=task.kind,
                guard=task.guard,
                priority_position=task.priority_position,
            )
            for index, task in enumerate(tasks)
        ),
    )
    schedule.validate_against(task_ir)
    return task_ir, schedule


__all__ = [
    "FinalStatus",
    "NativeMultiClauseAnytimeAggregateIR",
    "NativeMultiClauseAnytimeCandidateIR",
    "NativeMultiClauseAnytimeDecisionIR",
    "NativeMultiClauseAnytimeOutcomeIR",
    "NativeMultiClauseAnytimePlanIR",
    "NativeMultiClauseAnytimePolicyIR",
    "NativeMultiClauseAnytimeScheduleIR",
    "NativeMultiClauseAnytimeSliceIR",
    "NativeMultiClauseAnytimeTaskIRModule",
    "NativeMultiClauseAnytimeTaskKind",
    "lower_native_multi_clause_anytime_ir",
]
