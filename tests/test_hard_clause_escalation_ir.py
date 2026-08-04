"""Typed Plan/Decision/Task/Schedule tests for hard-clause escalation."""

from dataclasses import replace

import pytest

from boundflow.ir.hard_clause_escalation import (
    NativeHardClauseEscalationDecisionIR,
    NativeHardClauseEscalationPlanIR,
    lower_native_hard_clause_escalation_ir,
)
from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
from boundflow.ir.search_scaling import NativeBabSearchBudgetIR


def _plan() -> NativeHardClauseEscalationPlanIR:
    return NativeHardClauseEscalationPlanIR(
        plan_id="hard-clause-test",
        primal_graph_hash="1" * 64,
        input_bounds_hash="2" * 64,
        objective_matrix_hash="3" * 64,
        thresholds_hash="4" * 64,
        clause_count=9,
        whole_query_timeout_ns=60_000_000_000,
        baseline_budget=NativeBabSearchBudgetIR("baseline-n7d2", 7, 2),
        escalation_budget=NativeBabSearchBudgetIR("escalation-n31d4", 31, 4),
        refinement_policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=128,
            backward_chunk_size=32,
        ),
        search_policy_hash="5" * 64,
        optimizer_policy_hash="6" * 64,
    )


def test_hard_clause_escalation_lowers_exact_guarded_stage_order() -> None:
    plan = _plan()
    task_ir, schedule = lower_native_hard_clause_escalation_ir(plan)

    assert len(task_ir.tasks) == 8
    assert tuple(item.kind.value for item in task_ir.tasks) == (
        "execute_baseline",
        "admit_hard_clauses",
        "compile_refinement",
        "execute_refinement",
        "project_hard_clauses",
        "execute_escalation",
        "aggregate_verdicts",
        "emit_result",
    )
    assert [item.guard for item in task_ir.tasks[2:6]] == [
        "escalated_clauses_nonempty"
    ] * 4
    assert schedule.actions[-2].task_id == f"{plan.plan_id}:aggregate"
    schedule.validate_against(task_ir)


def test_hard_clause_decision_admits_exact_unresolved_ordinals() -> None:
    plan = _plan()
    decision = NativeHardClauseEscalationDecisionIR(
        decision_id="hard-clause-test:decision",
        plan_hash=plan.stable_hash(),
        baseline_query_trace_hash="7" * 64,
        clause_count=9,
        baseline_completed_clause_indices=tuple(range(9)),
        baseline_verified_clause_indices=(0, 1, 2, 4, 5, 6),
        baseline_unresolved_clause_indices=(3, 7, 8),
        baseline_pending_clause_indices=(),
        baseline_unsafe_clause_index=None,
        escalated_clause_indices=(3, 7, 8),
        reason="escalate_exact_unresolved",
    )

    decision.validate()
    assert decision.escalated_clause_indices == (3, 7, 8)

    tampered = replace(decision, escalated_clause_indices=(3, 8))
    with pytest.raises(ValueError, match="admission differs"):
        tampered.validate()


def test_hard_clause_plan_rejects_post_registration_budget_tuning() -> None:
    tampered = replace(
        _plan(),
        escalation_budget=NativeBabSearchBudgetIR("escalation-n63d5", 63, 5),
    )

    with pytest.raises(ValueError, match="Plan IR is invalid"):
        tampered.validate()
