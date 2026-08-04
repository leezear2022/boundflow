"""Unit contracts for NRIR-38 frontier-attribution IR."""

# pylint: disable=missing-function-docstring

from dataclasses import replace

import pytest

from boundflow.ir.frontier_tightness_attribution import (
    NativeFrontierCandidateNodeIR,
    NativeFrontierNodeAttributionIR,
    NativeFrontierTightnessAttributionPlanIR,
    NativeFrontierTightnessScheduleIR,
    lower_native_frontier_tightness_attribution_schedule,
)
from boundflow.runtime.native_frontier_tightness_attribution import _decision

HASH = "1" * 64
HASH2 = "2" * 64


def _plan() -> NativeFrontierTightnessAttributionPlanIR:
    return NativeFrontierTightnessAttributionPlanIR(
        plan_id="nrir38:test",
        source_execution_hash=HASH,
        source_plan_hash=HASH,
        source_queue_trace_hash=HASH,
        objective_hash=HASH,
        threshold_hash=HASH,
        original_clause_index=2,
        active_node_split_hashes=tuple((f"n{index:02d}", HASH) for index in range(16)),
        baseline_optimizer_policy_hash=HASH,
        candidate_optimizer_policy_hash=HASH2,
        baseline_optimizer_steps=5,
        candidate_optimizer_steps=15,
        required_active_depth=4,
        required_active_nodes=16,
    )


def _node_rows() -> tuple[NativeFrontierNodeAttributionIR, ...]:
    return tuple(
        NativeFrontierNodeAttributionIR(
            node_id=f"n{index:02d}",
            parent_node_id="parent",
            split_state_hash=HASH,
            evaluation_hash=HASH,
            depth=4,
            active=True,
            lower=-10.0 + index / 100.0,
            upper=1.0,
            proof_deficit=10.0 - index / 100.0,
            parent_lower_gain=2.0,
            refinement_plan_hash=HASH,
            refinement_semantic_trace_hash=HASH,
            final_intermediate_bounds_hash=HASH,
            selected_target_count=128,
            tightened_neuron_count=64,
            width_reduction_sum=3.0,
            initial_ambiguous_count=100,
            final_ambiguous_count=90,
            alpha_count=100,
            alpha_boundary_count=95,
            alpha_interior_count=5,
            beta_count=100,
            beta_positive_count=2,
        )
        for index in range(16)
    )


def _candidate_rows() -> tuple[NativeFrontierCandidateNodeIR, ...]:
    return tuple(
        NativeFrontierCandidateNodeIR(
            node_id=f"n{index:02d}",
            sibling_batch_index=index // 2,
            split_state_hash=HASH,
            source_evaluation_hash=HASH,
            source_refinement_hash=HASH,
            baseline_refinement_hash=HASH,
            candidate_refinement_hash=HASH,
            baseline_selected_state_hash=HASH,
            candidate_selected_state_hash=HASH2,
            source_lower=-10.0 + index / 100.0,
            source_upper=1.0,
            replay_lower=-10.0 + index / 100.0,
            replay_upper=1.0,
            candidate_lower=-9.9 + index / 100.0,
            candidate_upper=1.0,
            replay_lower_diff=0.0,
            replay_upper_diff=0.0,
            candidate_lower_delta=0.1,
        )
        for index in range(16)
    )


def test_frontier_tightness_lowers_seven_stage_schedule() -> None:
    plan = _plan()
    rows = _node_rows()
    candidates = _candidate_rows()
    decision = _decision(plan, candidates, source_coverage_passed=True)

    task_ir, schedule = lower_native_frontier_tightness_attribution_schedule(
        plan, rows, candidates, decision
    )

    assert len(task_ir.tasks) == 7
    assert [task.kind.value for task in task_ir.tasks] == [
        "admit_source",
        "enumerate_frontier",
        "summarize_source",
        "replay_baseline",
        "evaluate_candidate",
        "decide",
        "emit",
    ]
    assert decision.go is False
    assert decision.reason == "candidate_worst_improvement_below_gate"
    schedule.validate_against(task_ir)


def test_frontier_tightness_plan_rejects_non_preregistered_steps() -> None:
    with pytest.raises(ValueError, match="Plan IR differs"):
        replace(_plan(), candidate_optimizer_steps=16).validate()


def test_frontier_tightness_candidate_rejects_refinement_drift() -> None:
    with pytest.raises(ValueError, match="candidate node evidence differs"):
        replace(_candidate_rows()[0], candidate_refinement_hash=HASH2).validate()


def test_frontier_tightness_schedule_rejects_task_hash_drift() -> None:
    plan = _plan()
    rows = _node_rows()
    candidates = _candidate_rows()
    decision = _decision(plan, candidates, source_coverage_passed=True)
    task_ir, schedule = lower_native_frontier_tightness_attribution_schedule(
        plan, rows, candidates, decision
    )
    tampered_action = replace(schedule.actions[3], task_hash=HASH2)
    tampered = NativeFrontierTightnessScheduleIR(
        plan_hash=schedule.plan_hash,
        task_ir_hash=schedule.task_ir_hash,
        actions=(*schedule.actions[:3], tampered_action, *schedule.actions[4:]),
    )

    with pytest.raises(ValueError, match="Schedule/Task binding differs"):
        tampered.validate_against(task_ir)
