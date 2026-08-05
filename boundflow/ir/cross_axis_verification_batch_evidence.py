"""Typed Phase-A evidence for NRIR-43 cross-axis verification batching."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,missing-class-docstring,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Tuple


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
class NativeCrossAxisVerificationBatchEvidencePlanIR:
    plan_id: str
    source_phase_a_hash: str
    source_phase_b_hash: str
    clause_ordinals: Tuple[int, ...] = (2, 3)
    paired_orders: Tuple[Tuple[str, ...], ...] = (
        ("nrir42", "cross_axis"),
        ("cross_axis", "nrir42"),
        ("nrir42", "cross_axis"),
    )
    required_nodes: int = 31
    required_sibling_groups: int = 15
    nrir42_scorer_launches_per_clause: int = 31
    cross_axis_scorer_launches_per_clause: int = 16
    maximum_queue_median_ratio: float = 0.85
    torch_threads: int = 8
    performance_claimed: bool = False
    schema_version: str = "boundflow.cross-axis-verification-batch-evidence-plan/v1"

    def validate(self) -> None:
        if (
            self.schema_version
            != "boundflow.cross-axis-verification-batch-evidence-plan/v1"
            or not self.plan_id
            or not _is_sha256(self.source_phase_a_hash)
            or not _is_sha256(self.source_phase_b_hash)
            or self.clause_ordinals != (2, 3)
            or self.paired_orders
            != (
                ("nrir42", "cross_axis"),
                ("cross_axis", "nrir42"),
                ("nrir42", "cross_axis"),
            )
            or self.required_nodes != 31
            or self.required_sibling_groups != 15
            or self.nrir42_scorer_launches_per_clause != 31
            or self.cross_axis_scorer_launches_per_clause != 16
            or not math.isclose(self.maximum_queue_median_ratio, 0.85)
            or self.torch_threads != 8
            or self.performance_claimed is not False
        ):
            raise ValueError("cross-axis verification evidence Plan is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "source_phase_a_hash": self.source_phase_a_hash,
            "source_phase_b_hash": self.source_phase_b_hash,
            "clause_ordinals": list(self.clause_ordinals),
            "paired_orders": [list(item) for item in self.paired_orders],
            "required_nodes": self.required_nodes,
            "required_sibling_groups": self.required_sibling_groups,
            "nrir42_scorer_launches_per_clause": (
                self.nrir42_scorer_launches_per_clause
            ),
            "cross_axis_scorer_launches_per_clause": (
                self.cross_axis_scorer_launches_per_clause
            ),
            "maximum_queue_median_ratio": self.maximum_queue_median_ratio,
            "torch_threads": self.torch_threads,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeCrossAxisVerificationBatchRowIR:
    plan_hash: str
    repeat_index: int
    original_clause_index: int
    mode: str
    order_position: int
    queue_elapsed_ns: int
    accepted_nodes: int
    sibling_group_count: int
    scorer_launch_count: int
    scorer_node_widths: Tuple[int, ...]
    scorer_child_domain_widths: Tuple[int, ...]
    queue_semantic_hash: str
    branch_semantic_hash: str
    state_semantic_hash: str
    refinement_semantic_hash: str
    cross_batch_trace_hashes: Tuple[str, ...]
    performance_claimed: bool = False
    schema_version: str = "boundflow.cross-axis-verification-batch-row/v1"

    def validate(self, *, plan: NativeCrossAxisVerificationBatchEvidencePlanIR) -> None:
        plan.validate()
        hashes = (
            self.plan_hash,
            self.queue_semantic_hash,
            self.branch_semantic_hash,
            self.state_semantic_hash,
            self.refinement_semantic_hash,
            *self.cross_batch_trace_hashes,
        )
        expected_launches = (
            plan.nrir42_scorer_launches_per_clause
            if self.mode == "nrir42"
            else plan.cross_axis_scorer_launches_per_clause
        )
        if (
            self.schema_version != "boundflow.cross-axis-verification-batch-row/v1"
            or self.plan_hash != plan.stable_hash()
            or self.repeat_index not in range(len(plan.paired_orders))
            or self.original_clause_index not in plan.clause_ordinals
            or self.mode not in {"nrir42", "cross_axis"}
            or self.order_position not in (0, 1)
            or self.queue_elapsed_ns <= 0
            or self.accepted_nodes != plan.required_nodes
            or self.sibling_group_count != plan.required_sibling_groups
            or self.scorer_launch_count != expected_launches
            or len(self.scorer_node_widths) != expected_launches
            or len(self.scorer_child_domain_widths) != expected_launches
            or any(width < 1 for width in self.scorer_node_widths)
            or any(width < 2 for width in self.scorer_child_domain_widths)
            or any(not _is_sha256(value) for value in hashes)
            or (
                self.mode == "nrir42"
                and (
                    self.scorer_node_widths != (1,) * expected_launches
                    or self.cross_batch_trace_hashes
                )
            )
            or (
                self.mode == "cross_axis"
                and (
                    self.scorer_node_widths != (1, *([2] * 15))
                    or len(self.cross_batch_trace_hashes) != expected_launches
                )
            )
            or self.performance_claimed is not False
        ):
            raise ValueError("cross-axis verification evidence row differs")

    def to_dict(
        self, *, plan: NativeCrossAxisVerificationBatchEvidencePlanIR
    ) -> dict[str, object]:
        self.validate(plan=plan)
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "repeat_index": self.repeat_index,
            "original_clause_index": self.original_clause_index,
            "mode": self.mode,
            "order_position": self.order_position,
            "queue_elapsed_ns": self.queue_elapsed_ns,
            "accepted_nodes": self.accepted_nodes,
            "sibling_group_count": self.sibling_group_count,
            "scorer_launch_count": self.scorer_launch_count,
            "scorer_node_widths": list(self.scorer_node_widths),
            "scorer_child_domain_widths": list(self.scorer_child_domain_widths),
            "queue_semantic_hash": self.queue_semantic_hash,
            "branch_semantic_hash": self.branch_semantic_hash,
            "state_semantic_hash": self.state_semantic_hash,
            "refinement_semantic_hash": self.refinement_semantic_hash,
            "cross_batch_trace_hashes": list(self.cross_batch_trace_hashes),
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(
        self, *, plan: NativeCrossAxisVerificationBatchEvidencePlanIR
    ) -> str:
        return _canonical_hash(self.to_dict(plan=plan))


@dataclass(frozen=True)
class NativeCrossAxisVerificationBatchParityIR:
    plan_hash: str
    repeat_index: int
    original_clause_index: int
    queue_exact: bool
    branch_exact: bool
    state_exact: bool
    refinement_exact: bool
    all_exact: bool
    nrir42_raw_hash: str
    cross_axis_raw_hash: str

    def validate(self, *, plan: NativeCrossAxisVerificationBatchEvidencePlanIR) -> None:
        exact = (
            self.queue_exact
            and self.branch_exact
            and self.state_exact
            and self.refinement_exact
        )
        if (
            self.plan_hash != plan.stable_hash()
            or self.repeat_index not in range(len(plan.paired_orders))
            or self.original_clause_index not in plan.clause_ordinals
            or self.all_exact is not exact
            or not _is_sha256(self.nrir42_raw_hash)
            or not _is_sha256(self.cross_axis_raw_hash)
        ):
            raise ValueError("cross-axis verification parity differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_hash": self.plan_hash,
            "repeat_index": self.repeat_index,
            "original_clause_index": self.original_clause_index,
            "queue_exact": self.queue_exact,
            "branch_exact": self.branch_exact,
            "state_exact": self.state_exact,
            "refinement_exact": self.refinement_exact,
            "all_exact": self.all_exact,
            "nrir42_raw_hash": self.nrir42_raw_hash,
            "cross_axis_raw_hash": self.cross_axis_raw_hash,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeCrossAxisVerificationBatchClauseMetricIR:
    original_clause_index: int
    nrir42_median_ns: int
    cross_axis_median_ns: int
    nrir42_mad_ns: int
    cross_axis_mad_ns: int
    median_ratio: float
    median_improvement_ns: int
    improvement_exceeds_pooled_mad: bool
    timing_gate_passed: bool

    def validate(self, *, plan: NativeCrossAxisVerificationBatchEvidencePlanIR) -> None:
        expected_ratio = self.cross_axis_median_ns / self.nrir42_median_ns
        expected_improvement = self.nrir42_median_ns - self.cross_axis_median_ns
        expected_mad = expected_improvement > max(
            self.nrir42_mad_ns, self.cross_axis_mad_ns
        )
        expected_gate = (
            expected_ratio <= plan.maximum_queue_median_ratio and expected_mad
        )
        if (
            self.original_clause_index not in plan.clause_ordinals
            or min(self.nrir42_median_ns, self.cross_axis_median_ns) <= 0
            or min(self.nrir42_mad_ns, self.cross_axis_mad_ns) < 0
            or not math.isclose(self.median_ratio, expected_ratio, rel_tol=1e-12)
            or self.median_improvement_ns != expected_improvement
            or self.improvement_exceeds_pooled_mad is not expected_mad
            or self.timing_gate_passed is not expected_gate
        ):
            raise ValueError("cross-axis verification clause metric differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "original_clause_index": self.original_clause_index,
            "nrir42_median_ns": self.nrir42_median_ns,
            "cross_axis_median_ns": self.cross_axis_median_ns,
            "nrir42_mad_ns": self.nrir42_mad_ns,
            "cross_axis_mad_ns": self.cross_axis_mad_ns,
            "median_ratio": self.median_ratio,
            "median_improvement_ns": self.median_improvement_ns,
            "improvement_exceeds_pooled_mad": self.improvement_exceeds_pooled_mad,
            "timing_gate_passed": self.timing_gate_passed,
        }


@dataclass(frozen=True)
class NativeCrossAxisVerificationBatchDecisionIR:
    plan_hash: str
    clause_metrics: Tuple[NativeCrossAxisVerificationBatchClauseMetricIR, ...]
    parity_passed: bool
    launch_gate_passed: bool
    timing_gate_passed: bool
    phase_a_go: bool
    next_route: str
    reason: str
    performance_claimed: bool = False

    def validate(self, *, plan: NativeCrossAxisVerificationBatchEvidencePlanIR) -> None:
        for metric in self.clause_metrics:
            metric.validate(plan=plan)
        expected_timing = all(item.timing_gate_passed for item in self.clause_metrics)
        expected_go = self.parity_passed and self.launch_gate_passed and expected_timing
        if (
            self.plan_hash != plan.stable_hash()
            or tuple(item.original_clause_index for item in self.clause_metrics)
            != plan.clause_ordinals
            or self.timing_gate_passed is not expected_timing
            or self.phase_a_go is not expected_go
            or self.next_route
            != ("run_phase_b" if expected_go else "stop_cross_axis_batching")
            or not self.reason
            or self.performance_claimed is not False
        ):
            raise ValueError("cross-axis verification decision differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_hash": self.plan_hash,
            "clause_metrics": [item.to_dict() for item in self.clause_metrics],
            "parity_passed": self.parity_passed,
            "launch_gate_passed": self.launch_gate_passed,
            "timing_gate_passed": self.timing_gate_passed,
            "phase_a_go": self.phase_a_go,
            "next_route": self.next_route,
            "reason": self.reason,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


__all__ = [
    "NativeCrossAxisVerificationBatchClauseMetricIR",
    "NativeCrossAxisVerificationBatchDecisionIR",
    "NativeCrossAxisVerificationBatchEvidencePlanIR",
    "NativeCrossAxisVerificationBatchParityIR",
    "NativeCrossAxisVerificationBatchRowIR",
]
