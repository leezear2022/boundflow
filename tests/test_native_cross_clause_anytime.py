"""Typed cross-clause floor plus anytime escalation tests."""

# pylint: disable=missing-function-docstring,duplicate-code

from dataclasses import replace

import pytest
import torch

from boundflow.ir.cross_clause_anytime import (
    NativeCrossClauseAnytimeAggregateIR,
    NativeCrossClauseAnytimeDecisionIR,
    NativeCrossClauseAnytimePlanIR,
    NativeCrossClauseAnytimeTaskKind,
    lower_native_cross_clause_anytime_ir,
)
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_candidate_search import (
    NativeProjectedGradientSearchPolicy,
)
from boundflow.runtime.native_cross_clause_anytime import (
    compile_native_cross_clause_anytime_program,
    execute_native_cross_clause_anytime_program,
)
from boundflow.runtime.task_executor import InputSpec


def _plan() -> NativeCrossClauseAnytimePlanIR:
    return NativeCrossClauseAnytimePlanIR(
        plan_id="cross-clause-anytime-test",
        floor_plan_hash="1" * 64,
        floor_task_ir_hash="2" * 64,
        floor_schedule_hash="3" * 64,
        objective_matrix_hash="4" * 64,
        thresholds_hash="5" * 64,
        search_policy_hash="6" * 64,
        optimizer_policy_hash="7" * 64,
    )


def _admitted_decision(plan: NativeCrossClauseAnytimePlanIR):
    return NativeCrossClauseAnytimeDecisionIR(
        plan_hash=plan.stable_hash(),
        floor_trace_hash="8" * 64,
        floor_completed_original_clause_indices=tuple(range(9)),
        floor_status="unknown",
        floor_verified_clause_indices=(),
        floor_unresolved_clause_indices=tuple(range(9)),
        floor_unsafe_clause_index=None,
        admitted_original_clause_index=0,
        root_refinement_plan_hash="9" * 64,
        root_refinement_semantic_trace_hash="a" * 64,
        root_final_intermediate_bounds_hash="b" * 64,
        admitted=True,
        reason="floor_complete_unresolved_clause_admitted",
    )


def _toy() -> tuple[BFTaskModule, InputSpec, torch.Tensor, torch.Tensor]:
    module = BFTaskModule(
        tasks=[
            BoundTask(
                task_id="cross-clause-anytime-toy",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp("linear", "linear1", ["input", "W1", "b1"], ["h1"]),
                    TaskOp("relu", "relu1", ["h1"], ["r1"]),
                    TaskOp("linear", "linear2", ["r1", "W2", "b2"], ["out"]),
                ],
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id="cross-clause-anytime-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0, -0.5], [-0.25, 0.75]]),
                "b1": torch.tensor([0.1, -0.2]),
                "W2": torch.tensor([[0.75, -1.0], [-0.5, 0.25]]),
                "b2": torch.tensor([0.15, -0.1]),
            }
        },
    )
    spec = InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.3, -0.6]]),
        upper=torch.tensor([[0.7, 0.4]]),
    )
    objective = torch.tensor([[[1.0, -1.0]]]).repeat(1, 9, 1)
    thresholds = torch.full((9,), -1e6)
    return module, spec, objective, thresholds


def test_cross_clause_anytime_lowers_six_guarded_stages() -> None:
    plan = _plan()
    task_ir, schedule = lower_native_cross_clause_anytime_ir(plan)

    assert tuple(task.kind for task in task_ir.tasks) == tuple(
        NativeCrossClauseAnytimeTaskKind
    )
    assert tuple(task.guard for task in task_ir.tasks) == (
        "always",
        "always",
        "decision_admitted_before_deadline",
        "decision_admitted_before_deadline",
        "always",
        "always",
    )
    assert task_ir.tasks[4].dependency_task_ids == (
        task_ir.tasks[1].task_id,
        task_ir.tasks[3].task_id,
    )
    schedule.validate_against(task_ir)


def test_cross_clause_anytime_decision_rejects_source_tamper() -> None:
    plan = _plan()
    decision = _admitted_decision(plan)
    decision.validate_against(plan)

    with pytest.raises(ValueError, match="admitted Decision differs"):
        replace(decision, root_refinement_plan_hash=None).validate_against(plan)


def test_cross_clause_anytime_aggregate_is_floor_monotone() -> None:
    plan = _plan()
    decision = _admitted_decision(plan)
    aggregate = NativeCrossClauseAnytimeAggregateIR(
        plan_hash=plan.stable_hash(),
        decision_hash=decision.stable_hash(plan),
        floor_trace_hash=decision.floor_trace_hash,
        packed_queue_trace_hash="c" * 64,
        packed_status="unknown",
        floor_status="unknown",
        floor_verified_clause_indices=(),
        floor_unresolved_clause_indices=tuple(range(9)),
        floor_unsafe_clause_index=None,
        final_status="unknown",
        final_verified_clause_indices=(),
        final_unresolved_clause_indices=tuple(range(9)),
        final_unsafe_clause_index=None,
    )
    aggregate.validate_against(plan, decision)

    with pytest.raises(ValueError, match="non-monotone"):
        replace(
            aggregate,
            final_status="verified",
            final_verified_clause_indices=tuple(range(9)),
            final_unresolved_clause_indices=(),
        ).validate_against(plan, decision)


def test_cross_clause_anytime_aggregate_rejects_packed_result_without_admission() -> (
    None
):
    plan = _plan()
    decision = NativeCrossClauseAnytimeDecisionIR(
        plan_hash=plan.stable_hash(),
        floor_trace_hash="8" * 64,
        floor_completed_original_clause_indices=tuple(range(9)),
        floor_status="verified",
        floor_verified_clause_indices=tuple(range(9)),
        floor_unresolved_clause_indices=(),
        floor_unsafe_clause_index=None,
        admitted_original_clause_index=None,
        root_refinement_plan_hash=None,
        root_refinement_semantic_trace_hash=None,
        root_final_intermediate_bounds_hash=None,
        admitted=False,
        reason="floor_already_verified",
    )
    aggregate = NativeCrossClauseAnytimeAggregateIR(
        plan_hash=plan.stable_hash(),
        decision_hash=decision.stable_hash(plan),
        floor_trace_hash=decision.floor_trace_hash,
        packed_queue_trace_hash="c" * 64,
        packed_status="unknown",
        floor_status="verified",
        floor_verified_clause_indices=tuple(range(9)),
        floor_unresolved_clause_indices=(),
        floor_unsafe_clause_index=None,
        final_status="verified",
        final_verified_clause_indices=tuple(range(9)),
        final_unresolved_clause_indices=(),
        final_unsafe_clause_index=None,
    )

    with pytest.raises(ValueError, match="aggregate is invalid"):
        aggregate.validate_against(plan, decision)


def test_cross_clause_anytime_runtime_preserves_verified_floor() -> None:
    module, spec, objectives, thresholds = _toy()
    search = NativeProjectedGradientSearchPolicy(steps=1, step_size=0.01)
    optimizer = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)
    program = compile_native_cross_clause_anytime_program(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        plan_id="cross-clause-anytime-toy",
        search_policy=search,
        optimizer_policy=optimizer,
    )
    execution = execute_native_cross_clause_anytime_program(
        program,
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="cross-clause-anytime-toy",
        search_policy=search,
        optimizer_policy=optimizer,
    )

    assert execution.floor.trace.final_status == "verified"
    assert execution.decision.reason == "floor_already_verified"
    assert execution.packed is None
    assert execution.aggregate.final_status == "verified"
    assert execution.aggregate.final_verified_clause_indices == tuple(range(9))
    assert [action.executed for action in execution.trace.actions] == [
        True,
        True,
        False,
        False,
        True,
        True,
    ]
    assert execution.trace.performance_claimed is False
