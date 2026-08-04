"""NRIR-41 objective-branch production cost attribution contracts."""

# pylint: disable=duplicate-code

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from boundflow.ir.objective_branch_cost_attribution import (
    NativeObjectiveBranchCostAttributionPlanIR,
    NativeObjectiveBranchProfilePhaseIR,
    NativeObjectiveBranchWallAttributionIR,
    lower_native_objective_branch_cost_schedule,
)
from boundflow.runtime.native_objective_branch_cost_attribution import (
    compile_native_objective_branch_cost_plan,
    derive_native_objective_branch_cost_decision,
    reconstruct_native_objective_branch_prefixes,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT = (
    ROOT
    / "artifacts/objective-branch-shared-evaluator"
    / "vnncomp21-resnet2b-property0-cpu-pilot-v1/pilot.json"
)
FORMAL = (
    ROOT
    / "artifacts/objective-branch-whole-query"
    / "vnncomp21-resnet2b-property0-three-repeat-cpu-formal-v1/formal.json"
)
SHA = "a" * 64


def _sources() -> tuple[dict[str, Any], dict[str, Any]]:
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    return pilot, formal


def _plan() -> NativeObjectiveBranchCostAttributionPlanIR:
    pilot, formal = _sources()
    return compile_native_objective_branch_cost_plan(
        plan_id="objective-branch-cost-test",
        source_pilot_hash=pilot["pilot_hash"],
        source_formal_hash=formal["formal_payload_hash"],
    )


def _wall_rows(
    plan: NativeObjectiveBranchCostAttributionPlanIR,
) -> tuple[NativeObjectiveBranchWallAttributionIR, ...]:
    result = []
    for repeat in range(3):
        order = plan.paired_orders[repeat]
        for ordinal in plan.clause_ordinals:
            for position, mode in enumerate(order):
                objective = mode == "objective"
                result.append(
                    NativeObjectiveBranchWallAttributionIR(
                        plan_hash=plan.stable_hash(),
                        repeat_index=repeat,
                        original_clause_index=ordinal,
                        policy_id=(
                            plan.candidate_policy_id
                            if objective
                            else plan.control_policy_id
                        ),
                        order_position=position,
                        execution_hash=SHA,
                        queue_trace_hash=SHA,
                        root_lower=-100.0,
                        worst_active_lower=-30.0 if objective else -35.0,
                        median_active_lower=-25.0,
                        accepted_nodes=31,
                        sibling_group_count=15,
                        source_elapsed_ns=1,
                        queue_elapsed_ns=150 if objective else 100,
                        whole_elapsed_ns=160 if objective else 110,
                        cache_miss_count=1,
                        cache_hit_count=15,
                        branch_execution_count=31 if objective else 0,
                    )
                )
    return tuple(result)


def _profiles(
    plan: NativeObjectiveBranchCostAttributionPlanIR,
) -> tuple[NativeObjectiveBranchProfilePhaseIR, ...]:
    result = []
    for ordinal in plan.clause_ordinals:
        for phase in (
            "branch_program",
            "enumerate_candidates",
            "materialize_children",
            "evaluate_child_bounds",
        ):
            result.append(
                NativeObjectiveBranchProfilePhaseIR(
                    plan_hash=plan.stable_hash(),
                    original_clause_index=ordinal,
                    phase_id=phase,
                    primitive_calls=31,
                    total_ns=10,
                    cumulative_ns=30 if phase == "branch_program" else 20,
                    profile_queue_elapsed_ns=100,
                )
            )
    return tuple(result)


def test_frozen_prefix_reconstruction_retains_objective_order() -> None:
    """All preregistered same-node prefixes favor the objective policy."""

    pilot, _formal = _sources()
    plan = _plan()
    prefixes = reconstruct_native_objective_branch_prefixes(plan, pilot["clauses"])
    decision = derive_native_objective_branch_cost_decision(
        plan, prefixes, _wall_rows(plan), _profiles(plan)
    )
    task_ir, schedule = lower_native_objective_branch_cost_schedule(
        plan, prefixes, _wall_rows(plan), _profiles(plan), decision
    )

    assert len(prefixes) == 16
    assert decision.frontier_order_retained is True
    assert decision.scoring_cost_dominant is True
    assert decision.next_route == "optimize_scorer_ownership"
    assert min(dict(decision.frontier_improvements).values()) > 0.0
    schedule.validate_against(task_ir)


def test_prefix_reconstruction_rejects_parent_order_tamper() -> None:
    """A child cannot appear before its frozen parent, even after rehashing."""

    pilot, _formal = _sources()
    tampered = deepcopy(pilot["clauses"])
    tampered[0]["candidate"]["evaluations"][1]["parent_node_id"] = "missing"
    with pytest.raises(ValueError, match="parent lineage differs"):
        reconstruct_native_objective_branch_prefixes(_plan(), tampered)
