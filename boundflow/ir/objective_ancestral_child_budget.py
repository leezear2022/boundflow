"""Typed child-refinement budget calibration and selected Plan IR."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Optional, Tuple

from .objective_ancestral_queue import ObjectiveAncestralQueueTaskKind
from .refinement import NativeIntermediateRefinementPolicyIR
from .search_scaling import NativeBabSearchBudgetIR

OBJECTIVE_ANCESTRAL_CHILD_BUDGET_PLAN_IR_SCHEMA_VERSION = (
    "boundflow.objective-ancestral-child-budget-plan-ir/v1"
)
OBJECTIVE_ANCESTRAL_CHILD_BUDGET_POLICY_IR_SCHEMA_VERSION = (
    "boundflow.objective-ancestral-child-budget-policy-ir/v1"
)
OBJECTIVE_ANCESTRAL_CHILD_BUDGET_DECISION_IR_SCHEMA_VERSION = (
    "boundflow.objective-ancestral-child-budget-decision-ir/v1"
)
CHILD_BUDGET_CANDIDATE_CAPS = (8, 16, 32, 64, 128)
CHILD_BUDGET_PILOT_ORDER = (32, 8, 128, 16, 64)


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
class NativeObjectiveAncestralChildBudgetPolicyIR:
    """Pre-registered cap candidates and deterministic winner rule."""

    candidate_caps: Tuple[int, ...] = CHILD_BUDGET_CANDIDATE_CAPS
    pilot_order: Tuple[int, ...] = CHILD_BUDGET_PILOT_ORDER
    reference_cap: int = 128
    minimum_gain_retention: float = 0.90
    root_parity_tolerance: float = 1e-5
    selection_rule: str = "minimum_cap_meeting_reference_gain_retention_v1"
    schema_version: str = OBJECTIVE_ANCESTRAL_CHILD_BUDGET_POLICY_IR_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version
            != OBJECTIVE_ANCESTRAL_CHILD_BUDGET_POLICY_IR_SCHEMA_VERSION
            or self.candidate_caps != CHILD_BUDGET_CANDIDATE_CAPS
            or self.pilot_order != CHILD_BUDGET_PILOT_ORDER
            or tuple(sorted(self.pilot_order)) != self.candidate_caps
            or self.reference_cap != 128
            or self.minimum_gain_retention != 0.90
            or self.root_parity_tolerance != 1e-5
            or self.selection_rule != "minimum_cap_meeting_reference_gain_retention_v1"
        ):
            raise ValueError("objective ancestral child-budget policy is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "candidate_caps": list(self.candidate_caps),
            "pilot_order": list(self.pilot_order),
            "reference_cap": self.reference_cap,
            "minimum_gain_retention": self.minimum_gain_retention,
            "root_parity_tolerance": self.root_parity_tolerance,
            "selection_rule": self.selection_rule,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeObjectiveAncestralChildBudgetCalibrationIR:
    """One pilot candidate result used by the frozen selection decision."""

    cap: int
    root_lower: float
    worst_active_lower: float
    accepted_nodes: int
    lineage_valid: bool
    result_hash: str

    def validate(self) -> None:
        if (
            self.cap not in CHILD_BUDGET_CANDIDATE_CAPS
            or not math.isfinite(self.root_lower)
            or not math.isfinite(self.worst_active_lower)
            or self.accepted_nodes < 1
            or self.lineage_valid is not True
            or not _is_sha256(self.result_hash)
        ):
            raise ValueError("objective ancestral child-budget calibration is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "cap": self.cap,
            "root_lower": self.root_lower,
            "worst_active_lower": self.worst_active_lower,
            "accepted_nodes": self.accepted_nodes,
            "lineage_valid": self.lineage_valid,
            "result_hash": self.result_hash,
        }


@dataclass(frozen=True)
class NativeObjectiveAncestralChildBudgetDecisionIR:
    """Pilot candidate or evidence-bound deterministic Pareto selection."""

    policy_hash: str
    selected_cap: int
    selection_mode: str
    calibration_evidence_hash: Optional[str] = None
    root_global_worst_active_lower: Optional[float] = None
    calibration_rows: Tuple[NativeObjectiveAncestralChildBudgetCalibrationIR, ...] = ()
    reference_gain: Optional[float] = None
    selected_gain_retention: Optional[float] = None
    schema_version: str = OBJECTIVE_ANCESTRAL_CHILD_BUDGET_DECISION_IR_SCHEMA_VERSION

    def validate_against(
        self, policy: NativeObjectiveAncestralChildBudgetPolicyIR
    ) -> None:
        policy.validate()
        if (
            self.schema_version
            != OBJECTIVE_ANCESTRAL_CHILD_BUDGET_DECISION_IR_SCHEMA_VERSION
            or self.policy_hash != policy.stable_hash()
            or self.selected_cap not in policy.candidate_caps
            or self.selection_mode
            not in {"calibration_candidate", "frozen_pareto_selection"}
        ):
            raise ValueError("objective ancestral child-budget decision is invalid")
        if self.selection_mode == "calibration_candidate":
            if (
                self.calibration_evidence_hash is not None
                or self.root_global_worst_active_lower is not None
                or self.calibration_rows
                or self.reference_gain is not None
                or self.selected_gain_retention is not None
            ):
                raise ValueError("child-budget calibration candidate invents evidence")
            return
        if (
            not _is_sha256(self.calibration_evidence_hash)
            or self.root_global_worst_active_lower is None
            or not math.isfinite(self.root_global_worst_active_lower)
            or tuple(row.cap for row in self.calibration_rows) != policy.candidate_caps
            or self.reference_gain is None
            or not math.isfinite(self.reference_gain)
            or self.reference_gain <= 1e-4
            or self.selected_gain_retention is None
            or not math.isfinite(self.selected_gain_retention)
            or self.selected_gain_retention < policy.minimum_gain_retention
        ):
            raise ValueError("objective ancestral frozen child-budget evidence differs")
        for row in self.calibration_rows:
            row.validate()
        reference = next(
            row for row in self.calibration_rows if row.cap == policy.reference_cap
        )
        expected_reference_gain = (
            reference.worst_active_lower - self.root_global_worst_active_lower
        )
        eligible: list[tuple[int, float]] = []
        for row in self.calibration_rows:
            retention = (
                row.worst_active_lower - self.root_global_worst_active_lower
            ) / expected_reference_gain
            if (
                abs(row.root_lower - reference.root_lower)
                <= policy.root_parity_tolerance
                and retention + 1e-12 >= policy.minimum_gain_retention
            ):
                eligible.append((row.cap, retention))
        if not eligible:
            raise ValueError("objective ancestral child-budget has no eligible cap")
        expected_cap, expected_retention = min(eligible, key=lambda item: item[0])
        if (
            self.selected_cap != expected_cap
            or abs(self.reference_gain - expected_reference_gain) > 1e-9
            or abs(self.selected_gain_retention - expected_retention) > 1e-12
        ):
            raise ValueError("objective ancestral child-budget winner differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_hash": self.policy_hash,
            "selected_cap": self.selected_cap,
            "selection_mode": self.selection_mode,
            "calibration_evidence_hash": self.calibration_evidence_hash,
            "root_global_worst_active_lower": self.root_global_worst_active_lower,
            "calibration_rows": [row.to_dict() for row in self.calibration_rows],
            "reference_gain": self.reference_gain,
            "selected_gain_retention": self.selected_gain_retention,
        }

    def stable_hash(self, policy: NativeObjectiveAncestralChildBudgetPolicyIR) -> str:
        self.validate_against(policy)
        return _canonical_hash(self.to_dict())


def compile_frozen_child_budget_decision(
    policy: NativeObjectiveAncestralChildBudgetPolicyIR,
    *,
    calibration_evidence_hash: str,
    root_global_worst_active_lower: float,
    calibration_rows: Tuple[NativeObjectiveAncestralChildBudgetCalibrationIR, ...],
) -> NativeObjectiveAncestralChildBudgetDecisionIR:
    policy.validate()
    if tuple(row.cap for row in calibration_rows) != policy.candidate_caps:
        raise ValueError("objective ancestral calibration cap coverage differs")
    reference = next(row for row in calibration_rows if row.cap == policy.reference_cap)
    reference_gain = reference.worst_active_lower - root_global_worst_active_lower
    if reference_gain <= 1e-4:
        raise ValueError("objective ancestral reference cap has no strict gain")
    eligible = tuple(
        (
            row.cap,
            (row.worst_active_lower - root_global_worst_active_lower) / reference_gain,
        )
        for row in calibration_rows
        if abs(row.root_lower - reference.root_lower) <= policy.root_parity_tolerance
        and (row.worst_active_lower - root_global_worst_active_lower) / reference_gain
        + 1e-12
        >= policy.minimum_gain_retention
    )
    if not eligible:
        raise ValueError("objective ancestral child-budget has no eligible cap")
    selected_cap, retention = min(eligible, key=lambda item: item[0])
    decision = NativeObjectiveAncestralChildBudgetDecisionIR(
        policy_hash=policy.stable_hash(),
        selected_cap=selected_cap,
        selection_mode="frozen_pareto_selection",
        calibration_evidence_hash=calibration_evidence_hash,
        root_global_worst_active_lower=root_global_worst_active_lower,
        calibration_rows=calibration_rows,
        reference_gain=reference_gain,
        selected_gain_retention=retention,
    )
    decision.validate_against(policy)
    return decision


@dataclass(frozen=True)
class NativeObjectiveAncestralChildBudgetPlanIR:
    """NRIR-32-compatible Plan protocol with explicit child-budget ownership."""

    plan_id: str
    primal_graph_hash: str
    input_bounds_hash: str
    objective_hash: str
    threshold_hash: str
    root_refinement_plan_hash: str
    root_refinement_semantic_trace_hash: str
    root_intermediate_bounds_hash: str
    optimizer_policy_hash: str
    search_budget: NativeBabSearchBudgetIR
    child_refinement_policy: NativeIntermediateRefinementPolicyIR
    child_budget_policy: NativeObjectiveAncestralChildBudgetPolicyIR
    child_budget_decision: NativeObjectiveAncestralChildBudgetDecisionIR
    whole_query_timeout_ns: int = 60 * 1_000_000_000
    semantics_owner: str = "boundflow_native_objective_ancestral_child_budget"
    schema_version: str = OBJECTIVE_ANCESTRAL_CHILD_BUDGET_PLAN_IR_SCHEMA_VERSION

    def validate(self) -> None:
        self.search_budget.validate()
        self.child_refinement_policy.validate()
        self.child_budget_policy.validate()
        self.child_budget_decision.validate_against(self.child_budget_policy)
        hashes = (
            self.primal_graph_hash,
            self.input_bounds_hash,
            self.objective_hash,
            self.threshold_hash,
            self.root_refinement_plan_hash,
            self.root_refinement_semantic_trace_hash,
            self.root_intermediate_bounds_hash,
            self.optimizer_policy_hash,
        )
        if (
            self.schema_version
            != OBJECTIVE_ANCESTRAL_CHILD_BUDGET_PLAN_IR_SCHEMA_VERSION
            or not self.plan_id
            or any(not _is_sha256(value) for value in hashes)
            or (self.search_budget.max_nodes, self.search_budget.max_depth) != (31, 4)
            or self.child_refinement_policy.passes != 1
            or self.child_refinement_policy.max_neurons_per_relu
            != self.child_budget_decision.selected_cap
            or self.child_refinement_policy.backward_chunk_size
            != min(32, self.child_budget_decision.selected_cap)
            or self.child_refinement_policy.candidate_policy_id
            != "objective_influence_width_per_relu_v1"
            or self.child_refinement_policy.refinement_method
            != "selected_plain_crown_v1"
            or self.whole_query_timeout_ns != 60 * 1_000_000_000
            or self.semantics_owner
            != "boundflow_native_objective_ancestral_child_budget"
        ):
            raise ValueError("objective ancestral child-budget Plan IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "primal_graph_hash": self.primal_graph_hash,
            "input_bounds_hash": self.input_bounds_hash,
            "objective_hash": self.objective_hash,
            "threshold_hash": self.threshold_hash,
            "root_refinement_plan_hash": self.root_refinement_plan_hash,
            "root_refinement_semantic_trace_hash": (
                self.root_refinement_semantic_trace_hash
            ),
            "root_intermediate_bounds_hash": self.root_intermediate_bounds_hash,
            "optimizer_policy_hash": self.optimizer_policy_hash,
            "search_budget": self.search_budget.to_dict(),
            "child_refinement_policy": self.child_refinement_policy.to_dict(),
            "child_budget_policy": self.child_budget_policy.to_dict(),
            "child_budget_decision": self.child_budget_decision.to_dict(),
            "whole_query_timeout_ns": self.whole_query_timeout_ns,
            "task_template": [kind.value for kind in ObjectiveAncestralQueueTaskKind],
            "node_dispatch": "serial_dynamic_parent_before_child",
            "source_consumption": "sound_constraint_only",
            "deadline_enforcement": "whole_query_cooperative_stage_boundaries",
            "semantics_owner": self.semantics_owner,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


__all__ = [
    "CHILD_BUDGET_CANDIDATE_CAPS",
    "CHILD_BUDGET_PILOT_ORDER",
    "NativeObjectiveAncestralChildBudgetCalibrationIR",
    "NativeObjectiveAncestralChildBudgetDecisionIR",
    "NativeObjectiveAncestralChildBudgetPlanIR",
    "NativeObjectiveAncestralChildBudgetPolicyIR",
    "compile_frozen_child_budget_decision",
]
