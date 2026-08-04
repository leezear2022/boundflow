"""Native reconstruction and decision logic for NRIR-41 attribution."""

# pylint: disable=too-many-locals,too-many-arguments,duplicate-code

from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Any, Mapping, Sequence

from ..ir.objective_branch_cost_attribution import (
    NativeObjectiveBranchCostAttributionPlanIR,
    NativeObjectiveBranchCostDecisionIR,
    NativeObjectiveBranchPrefixAttributionIR,
    NativeObjectiveBranchProfilePhaseIR,
    NativeObjectiveBranchWallAttributionIR,
)
from .native_shared_parametric_ancestral import (
    NativeSharedParametricAncestralExecution,
)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compile_native_objective_branch_cost_plan(
    *, plan_id: str, source_pilot_hash: str, source_formal_hash: str
) -> NativeObjectiveBranchCostAttributionPlanIR:
    """Compile the frozen source identities and preregistered thresholds."""

    plan = NativeObjectiveBranchCostAttributionPlanIR(
        plan_id=plan_id,
        source_pilot_hash=source_pilot_hash,
        source_formal_hash=source_formal_hash,
    )
    plan.validate()
    return plan


def reconstruct_native_objective_branch_prefixes(
    plan: NativeObjectiveBranchCostAttributionPlanIR,
    source_clauses: Sequence[Mapping[str, Any]],
) -> tuple[NativeObjectiveBranchPrefixAttributionIR, ...]:
    """Reconstruct active frontiers from frozen ordered evaluation rows."""

    plan.validate()
    clause_by_ordinal = {
        int(value["original_clause_index"]): value for value in source_clauses
    }
    if set(clause_by_ordinal) != set(plan.clause_ordinals):
        raise ValueError("objective-branch prefix clause coverage differs")
    result: list[NativeObjectiveBranchPrefixAttributionIR] = []
    modes = (
        ("control", plan.control_policy_id),
        ("candidate", plan.candidate_policy_id),
    )
    for ordinal in plan.clause_ordinals:
        clause = clause_by_ordinal[ordinal]
        for source_key, policy_id in modes:
            source = clause.get(source_key)
            if not isinstance(source, dict):
                raise TypeError("objective-branch prefix source differs")
            rows = source.get("evaluations")
            if not isinstance(rows, list) or len(rows) != plan.required_nodes:
                raise ValueError("objective-branch prefix row coverage differs")
            seen: set[str] = set()
            for index, row in enumerate(rows):
                node_id = row.get("node_id")
                parent_id = row.get("parent_node_id")
                if (
                    not isinstance(node_id, str)
                    or node_id in seen
                    or (index == 0) != (parent_id is None)
                    or (parent_id is not None and parent_id not in seen)
                ):
                    raise ValueError("objective-branch prefix parent lineage differs")
                seen.add(node_id)
            for accepted_nodes in plan.prefix_node_counts:
                prefix = rows[:accepted_nodes]
                parent_ids = {
                    row["parent_node_id"]
                    for row in prefix
                    if row.get("parent_node_id") is not None
                }
                active = tuple(
                    row for row in prefix if row["node_id"] not in parent_ids
                )
                lowers = [float(row["lower"]) for row in active]
                item = NativeObjectiveBranchPrefixAttributionIR(
                    plan_hash=plan.stable_hash(),
                    original_clause_index=ordinal,
                    policy_id=policy_id,
                    accepted_nodes=accepted_nodes,
                    active_node_ids=tuple(str(row["node_id"]) for row in active),
                    active_evaluation_hashes=tuple(
                        str(row["evaluation_hash"]) for row in active
                    ),
                    active_count=len(active),
                    worst_active_lower=min(lowers),
                    median_active_lower=float(statistics.median(lowers)),
                    source_rows_hash=_canonical_hash(prefix),
                )
                item.validate()
                result.append(item)
            final = result[-1]
            if not math.isclose(
                final.worst_active_lower,
                float(source["worst_active_lower"]),
                rel_tol=1e-7,
                abs_tol=1e-7,
            ) or not math.isclose(
                final.median_active_lower,
                float(source["median_active_lower"]),
                rel_tol=1e-7,
                abs_tol=1e-7,
            ):
                raise ValueError("objective-branch frozen final summary differs")
    return tuple(result)


def native_objective_branch_wall_row(
    plan: NativeObjectiveBranchCostAttributionPlanIR,
    execution: NativeSharedParametricAncestralExecution,
    *,
    repeat_index: int,
    original_clause_index: int,
    policy_id: str,
    order_position: int,
    branch_execution_count: int,
) -> NativeObjectiveBranchWallAttributionIR:
    """Project one exact 31-node production execution into wall-time IR."""

    plan.validate()
    evaluations = execution.queue.trace.evaluations
    parent_ids = {
        evaluation.node.parent_node_id
        for evaluation in evaluations
        if evaluation.node.parent_node_id is not None
    }
    active = tuple(
        evaluation
        for evaluation in evaluations
        if evaluation.node.node_id not in parent_ids
    )
    if not active:
        raise ValueError("objective-branch wall active frontier is empty")
    cache_misses = sum(
        item.cache_event.outcome == "miss_compiled"
        for item in execution.compiler_batches
    )
    cache_hits = sum(
        item.cache_event.outcome == "hit_exact_contract"
        for item in execution.compiler_batches
    )
    item = NativeObjectiveBranchWallAttributionIR(
        plan_hash=plan.stable_hash(),
        repeat_index=repeat_index,
        original_clause_index=original_clause_index,
        policy_id=policy_id,
        order_position=order_position,
        execution_hash=_canonical_hash(execution.to_dict()),
        queue_trace_hash=execution.queue.trace.stable_hash(),
        root_lower=evaluations[0].lower,
        worst_active_lower=min(evaluation.lower for evaluation in active),
        median_active_lower=float(
            statistics.median(evaluation.lower for evaluation in active)
        ),
        accepted_nodes=len(execution.queue.trace.evaluations),
        sibling_group_count=len(execution.batch_commits) - 1,
        source_elapsed_ns=execution.trace.source_elapsed_ns,
        queue_elapsed_ns=execution.trace.queue_elapsed_ns,
        whole_elapsed_ns=execution.trace.whole_elapsed_ns,
        cache_miss_count=cache_misses,
        cache_hit_count=cache_hits,
        branch_execution_count=branch_execution_count,
    )
    item.validate()
    return item


def native_objective_branch_profile_rows(
    plan: NativeObjectiveBranchCostAttributionPlanIR,
    *,
    original_clause_index: int,
    stats: Mapping[tuple[str, int, str], tuple[int, int, float, float, object]],
    profile_queue_elapsed_ns: int,
) -> tuple[NativeObjectiveBranchProfilePhaseIR, ...]:
    """Normalize cProfile rows for the exact production branch call path."""

    plan.validate()
    phases = (
        ("branch_program", "execute_native_objective_branch_program"),
        ("enumerate_candidates", "_enumerate_candidates"),
        ("materialize_children", "_materialize_child_splits"),
        ("evaluate_child_bounds", "_evaluate_child_lowers"),
    )
    result: list[NativeObjectiveBranchProfilePhaseIR] = []
    for phase_id, function_name in phases:
        matches = [values for key, values in stats.items() if key[2] == function_name]
        if len(matches) != 1:
            raise ValueError(f"objective-branch profile phase differs: {phase_id}")
        primitive_calls, _total_calls, total_seconds, cumulative_seconds, _callers = (
            matches[0]
        )
        item = NativeObjectiveBranchProfilePhaseIR(
            plan_hash=plan.stable_hash(),
            original_clause_index=original_clause_index,
            phase_id=phase_id,
            primitive_calls=int(primitive_calls),
            total_ns=round(total_seconds * 1e9),
            cumulative_ns=round(cumulative_seconds * 1e9),
            profile_queue_elapsed_ns=profile_queue_elapsed_ns,
        )
        item.validate()
        result.append(item)
    expected_calls = {
        "branch_program": 31,
        "enumerate_candidates": 341,
        "materialize_children": 31,
        "evaluate_child_bounds": 31,
    }
    if any(item.primitive_calls != expected_calls[item.phase_id] for item in result):
        raise ValueError(
            "objective-branch profile call coverage differs: "
            f"{[(item.phase_id, item.primitive_calls) for item in result]}"
        )
    return tuple(result)


def derive_native_objective_branch_cost_decision(
    plan: NativeObjectiveBranchCostAttributionPlanIR,
    prefixes: Sequence[NativeObjectiveBranchPrefixAttributionIR],
    walls: Sequence[NativeObjectiveBranchWallAttributionIR],
    profiles: Sequence[NativeObjectiveBranchProfilePhaseIR],
) -> NativeObjectiveBranchCostDecisionIR:
    """Derive the preregistered causal route from complete evidence."""

    plan.validate()
    for collection in (prefixes, walls, profiles):
        for item in collection:
            item.validate()
            if item.plan_hash != plan.stable_hash():
                raise ValueError("objective-branch cost evidence Plan differs")
    prefix_by_key = {
        (item.original_clause_index, item.policy_id, item.accepted_nodes): item
        for item in prefixes
    }
    expected_prefixes = {
        (ordinal, policy, nodes)
        for ordinal in plan.clause_ordinals
        for policy in (plan.control_policy_id, plan.candidate_policy_id)
        for nodes in plan.prefix_node_counts
    }
    if set(prefix_by_key) != expected_prefixes:
        raise ValueError("objective-branch cost prefix coverage differs")
    improvements: list[tuple[str, float]] = []
    for ordinal in plan.clause_ordinals:
        for nodes in plan.prefix_node_counts:
            control = prefix_by_key[(ordinal, plan.control_policy_id, nodes)]
            candidate = prefix_by_key[(ordinal, plan.candidate_policy_id, nodes)]
            improvements.append(
                (
                    f"clause_{ordinal}_nodes_{nodes}",
                    candidate.worst_active_lower - control.worst_active_lower,
                )
            )
    frontier_order_retained = all(value >= 0.0 for _key, value in improvements) and all(
        dict(improvements)[f"clause_{ordinal}_nodes_31"]
        >= plan.minimum_frontier_improvement
        for ordinal in plan.clause_ordinals
    )
    wall_by_key: dict[tuple[int, str], list[int]] = {}
    for item in walls:
        wall_by_key.setdefault((item.original_clause_index, item.policy_id), []).append(
            item.queue_elapsed_ns
        )
    expected_walls = {
        (ordinal, policy)
        for ordinal in plan.clause_ordinals
        for policy in (plan.control_policy_id, plan.candidate_policy_id)
    }
    if set(wall_by_key) != expected_walls or any(
        len(values) != 3 for values in wall_by_key.values()
    ):
        raise ValueError("objective-branch wall repeat coverage differs")
    ratios = tuple(
        (
            f"clause_{ordinal}",
            float(
                statistics.median(wall_by_key[(ordinal, plan.candidate_policy_id)])
                / statistics.median(wall_by_key[(ordinal, plan.control_policy_id)])
            ),
        )
        for ordinal in plan.clause_ordinals
    )
    profile_by_key = {
        (item.original_clause_index, item.phase_id): item for item in profiles
    }
    if len(profile_by_key) != 8:
        raise ValueError("objective-branch profile coverage differs")
    shares = tuple(
        (
            f"clause_{ordinal}",
            profile_by_key[(ordinal, "branch_program")].cumulative_ns
            / profile_by_key[(ordinal, "branch_program")].profile_queue_elapsed_ns,
        )
        for ordinal in plan.clause_ordinals
    )
    scoring_cost_dominant = all(
        value >= plan.minimum_queue_ratio for _key, value in ratios
    ) and all(value >= plan.minimum_branch_program_share for _key, value in shares)
    if not frontier_order_retained:
        next_route = "freeze_objective_branch_production"
        reason = "same_prefix_frontier_gate_failed"
    elif scoring_cost_dominant:
        next_route = "optimize_scorer_ownership"
        reason = "frontier_retained_and_scoring_cost_dominant"
    else:
        next_route = "attribute_atomic_tail_scheduling"
        reason = "frontier_retained_without_dominant_scoring_cost"
    decision = NativeObjectiveBranchCostDecisionIR(
        plan_hash=plan.stable_hash(),
        frontier_improvements=tuple(improvements),
        queue_ratios=ratios,
        branch_program_shares=shares,
        frontier_order_retained=frontier_order_retained,
        scoring_cost_dominant=scoring_cost_dominant,
        next_route=next_route,
        reason=reason,
    )
    decision.validate()
    return decision


__all__ = [
    "compile_native_objective_branch_cost_plan",
    "derive_native_objective_branch_cost_decision",
    "native_objective_branch_profile_rows",
    "native_objective_branch_wall_row",
    "reconstruct_native_objective_branch_prefixes",
]
