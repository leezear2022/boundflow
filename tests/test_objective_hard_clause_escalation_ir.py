"""Static IR tests for objective-directed hard-clause escalation."""

from dataclasses import replace

import pytest

from boundflow.ir.objective_hard_clause_escalation import (
    NativeObjectiveHardClauseEscalationPlanIR,
    lower_native_objective_hard_clause_escalation_ir,
)
from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR


def _plan() -> NativeObjectiveHardClauseEscalationPlanIR:
    return NativeObjectiveHardClauseEscalationPlanIR(
        plan_id="objective-hard-test",
        base_plan_hash="1" * 64,
        base_task_ir_hash="2" * 64,
        base_schedule_hash="3" * 64,
        clause_count=3,
        objective_refinement_policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=128,
            backward_chunk_size=32,
            candidate_policy_id="objective_influence_width_per_relu_v1",
        ),
    )


def test_objective_escalation_unrolls_exact_original_clause_tasks() -> None:
    plan = _plan()
    task_ir, schedule = lower_native_objective_hard_clause_escalation_ir(plan)

    assert len(task_ir.tasks) == 15
    assert [task.original_clause_index for task in task_ir.tasks[4:13]] == [
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
    ]
    assert [task.guard for task in task_ir.tasks[4:13]] == ["clause_is_admitted"] * 9
    assert task_ir.tasks[-2].dependency_task_ids[-3:] == (
        "objective-hard-test:clause-0000:execute-query",
        "objective-hard-test:clause-0001:execute-query",
        "objective-hard-test:clause-0002:execute-query",
    )
    schedule.validate_against(task_ir)


def test_objective_escalation_rejects_non_objective_policy() -> None:
    tampered = replace(
        _plan(),
        objective_refinement_policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=128,
            backward_chunk_size=32,
        ),
    )

    with pytest.raises(ValueError, match="Plan IR is invalid"):
        tampered.validate()
