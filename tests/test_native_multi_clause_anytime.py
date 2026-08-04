"""Typed multi-clause priority and time-slice tests."""

# pylint: disable=missing-function-docstring,duplicate-code,too-many-arguments

from dataclasses import replace

import pytest
import torch

from boundflow.ir.multi_clause_anytime import (
    NativeMultiClauseAnytimeAggregateIR,
    NativeMultiClauseAnytimeCandidateIR,
    NativeMultiClauseAnytimeDecisionIR,
    NativeMultiClauseAnytimeOutcomeIR,
    NativeMultiClauseAnytimePlanIR,
    NativeMultiClauseAnytimePolicyIR,
    NativeMultiClauseAnytimeSliceIR,
    lower_native_multi_clause_anytime_ir,
)
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_candidate_search import (
    NativeProjectedGradientSearchPolicy,
)
from boundflow.runtime.native_multi_clause_anytime import (
    _OneShotSliceClock,
    _canonical_hash,
    compile_native_multi_clause_anytime_program,
    execute_native_multi_clause_anytime_program,
)
from boundflow.runtime.task_executor import InputSpec


def _plan() -> NativeMultiClauseAnytimePlanIR:
    return NativeMultiClauseAnytimePlanIR(
        plan_id="multi-clause-anytime-test",
        floor_plan_hash="1" * 64,
        floor_task_ir_hash="2" * 64,
        floor_schedule_hash="3" * 64,
        objective_matrix_hash="4" * 64,
        thresholds_hash="5" * 64,
        search_policy_hash="6" * 64,
        optimizer_policy_hash="7" * 64,
        allocation_policy=NativeMultiClauseAnytimePolicyIR(),
    )


def _candidate(ordinal: int, lower: float) -> NativeMultiClauseAnytimeCandidateIR:
    return NativeMultiClauseAnytimeCandidateIR(
        original_clause_index=ordinal,
        threshold=0.0,
        root_lower=lower,
        root_upper=lower + 10.0,
        root_lower_margin=lower,
        root_refinement_plan_hash=f"{ordinal + 1:x}" * 64,
        root_refinement_semantic_trace_hash=f"{ordinal + 2:x}" * 64,
        root_final_intermediate_bounds_hash=f"{ordinal + 3:x}" * 64,
    )


def _decision(plan: NativeMultiClauseAnytimePlanIR):
    candidates = (
        _candidate(0, -204.0),
        _candidate(2, -139.0),
        _candidate(3, -152.0),
        _candidate(4, -163.0),
    )
    return NativeMultiClauseAnytimeDecisionIR(
        plan_hash=plan.stable_hash(),
        floor_trace_hash="8" * 64,
        floor_completed_original_clause_indices=tuple(range(9)),
        floor_status="unknown",
        floor_verified_clause_indices=(),
        floor_unresolved_clause_indices=tuple(range(9)),
        floor_unsafe_clause_index=None,
        candidates=candidates,
        ranked_original_clause_indices=(2, 3, 4, 0),
        selected_original_clause_indices=(2, 3),
        reason="ranked_unresolved_candidates_selected",
    )


def _slice(
    plan: NativeMultiClauseAnytimePlanIR,
    decision: NativeMultiClauseAnytimeDecisionIR,
    *,
    position: int,
    ordinal: int,
    dispatch_ns: int,
    status: str,
) -> NativeMultiClauseAnytimeSliceIR:
    remaining_count = 2 - position
    remaining_ns = plan.whole_query_timeout_ns - dispatch_ns
    allocated_ns = remaining_ns // remaining_count
    candidate = decision.candidate(ordinal)
    return NativeMultiClauseAnytimeSliceIR(
        plan_hash=plan.stable_hash(),
        decision_hash=decision.stable_hash(plan),
        priority_position=position,
        original_clause_index=ordinal,
        dispatch_started_elapsed_ns=dispatch_ns,
        remaining_before_ns=remaining_ns,
        remaining_selected_count=remaining_count,
        allocated_slice_ns=allocated_ns,
        slice_cutoff_elapsed_ns=dispatch_ns + allocated_ns,
        finished_elapsed_ns=dispatch_ns + allocated_ns + 1,
        source_refinement_plan_hash=candidate.root_refinement_plan_hash,
        source_refinement_semantic_trace_hash=(
            candidate.root_refinement_semantic_trace_hash
        ),
        source_final_intermediate_bounds_hash=(
            candidate.root_final_intermediate_bounds_hash
        ),
        packed_plan_hash="a" * 64,
        packed_queue_trace_hash="b" * 64,
        packed_verdict_trace_hash="c" * 64,
        packed_status=status,
        accepted_nodes=3,
        sibling_group_count=1,
        cutoff_signaled=True,
        reason=f"packed_slice_{status}",
    )


def _toy() -> tuple[BFTaskModule, InputSpec, torch.Tensor, torch.Tensor]:
    module = BFTaskModule(
        tasks=[
            BoundTask(
                task_id="multi-clause-anytime-toy",
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
        entry_task_id="multi-clause-anytime-toy",
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


def test_multi_clause_anytime_lowers_two_guarded_priority_slots() -> None:
    plan = _plan()
    task_ir, schedule = lower_native_multi_clause_anytime_ir(plan)

    assert len(task_ir.tasks) == 8
    assert tuple(task.kind.value for task in task_ir.tasks) == (
        "execute_floor",
        "rank_candidates",
        "compile_packed_plan",
        "execute_packed_slice",
        "compile_packed_plan",
        "execute_packed_slice",
        "aggregate_original_ordinals",
        "emit_result",
    )
    assert tuple(task.priority_position for task in task_ir.tasks) == (
        None,
        None,
        0,
        0,
        1,
        1,
        None,
        None,
    )
    assert task_ir.tasks[6].dependency_task_ids == (
        task_ir.tasks[1].task_id,
        task_ir.tasks[3].task_id,
        task_ir.tasks[5].task_id,
    )
    schedule.validate_against(task_ir)


def test_multi_clause_anytime_priority_is_margin_desc_then_ordinal() -> None:
    plan = _plan()
    decision = _decision(plan)
    decision.validate_against(plan)
    assert decision.selected_original_clause_indices == (2, 3)

    with pytest.raises(ValueError, match="Decision IR is invalid"):
        replace(decision, ranked_original_clause_indices=(3, 2, 4, 0)).validate_against(
            plan
        )


def test_multi_clause_anytime_slice_rejects_allocation_tamper() -> None:
    plan = _plan()
    decision = _decision(plan)
    slice_ir = _slice(
        plan,
        decision,
        position=0,
        ordinal=2,
        dispatch_ns=20_000_000_000,
        status="unknown",
    )
    slice_ir.validate_against(plan, decision)

    with pytest.raises(ValueError, match="slice is invalid"):
        replace(
            slice_ir, allocated_slice_ns=slice_ir.allocated_slice_ns + 1
        ).validate_against(plan, decision)


def test_multi_clause_anytime_aggregate_is_monotone_over_two_results() -> None:
    plan = _plan()
    decision = _decision(plan)
    first = _slice(
        plan,
        decision,
        position=0,
        ordinal=2,
        dispatch_ns=20_000_000_000,
        status="verified",
    )
    second = _slice(
        plan,
        decision,
        position=1,
        ordinal=3,
        dispatch_ns=45_000_000_000,
        status="unknown",
    )
    slices = (first, second)
    outcomes = tuple(
        NativeMultiClauseAnytimeOutcomeIR(
            original_clause_index=item.original_clause_index,
            packed_queue_trace_hash=item.packed_queue_trace_hash or "",
            packed_verdict_trace_hash=item.packed_verdict_trace_hash or "",
            status=item.packed_status or "unknown",
        )
        for item in slices
    )
    aggregate = NativeMultiClauseAnytimeAggregateIR(
        plan_hash=plan.stable_hash(),
        decision_hash=decision.stable_hash(plan),
        floor_trace_hash=decision.floor_trace_hash,
        slice_hashes=tuple(item.stable_hash(plan, decision) for item in slices),
        outcomes=outcomes,
        floor_status="unknown",
        floor_verified_clause_indices=(),
        floor_unresolved_clause_indices=tuple(range(9)),
        floor_unsafe_clause_index=None,
        final_status="unknown",
        final_verified_clause_indices=(2,),
        final_unresolved_clause_indices=(0, 1, 3, 4, 5, 6, 7, 8),
        final_unsafe_clause_index=None,
    )
    aggregate.validate_against(plan, decision, slices)

    with pytest.raises(ValueError, match="non-monotone"):
        replace(
            aggregate,
            final_verified_clause_indices=(2, 3),
            final_unresolved_clause_indices=(0, 1, 4, 5, 6, 7, 8),
        ).validate_against(plan, decision, slices)


def test_multi_clause_anytime_runtime_preserves_verified_floor() -> None:
    module, spec, objectives, thresholds = _toy()
    search = NativeProjectedGradientSearchPolicy(steps=1, step_size=0.01)
    optimizer = NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)
    program = compile_native_multi_clause_anytime_program(
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        plan_id="multi-clause-anytime-toy",
        search_policy=search,
        optimizer_policy=optimizer,
    )
    execution = execute_native_multi_clause_anytime_program(
        program,
        module,
        spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id="multi-clause-anytime-toy",
        search_policy=search,
        optimizer_policy=optimizer,
    )

    assert execution.floor.trace.final_status == "verified"
    assert execution.decision.reason == "floor_not_eligible_for_multi_clause_anytime"
    assert not execution.packed_executions
    assert execution.aggregate.final_status == "verified"
    assert execution.aggregate.final_verified_clause_indices == tuple(range(9))
    assert [action.executed for action in execution.trace.actions] == [
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    assert execution.trace.performance_claimed is False

    tampered_action = replace(execution.trace.actions[0], output_hash="0" * 64)
    tampered_trace = replace(
        execution.trace,
        actions=(tampered_action, *execution.trace.actions[1:]),
        semantic_signature_hash="",
    )
    tampered_trace = replace(
        tampered_trace,
        semantic_signature_hash=_canonical_hash(tampered_trace.semantic_dict()),
    )
    with pytest.raises(ValueError, match="action/result binding"):
        replace(execution, trace=tampered_trace).validate_against(
            module,
            spec,
            linear_spec_C=objectives,
            thresholds=thresholds,
            search_policy=search,
            optimizer_policy=optimizer,
        )


def test_multi_clause_anytime_cutoff_signal_cannot_be_consumed_by_start_read() -> None:
    clock = _OneShotSliceClock(
        lambda: 10,
        cutoff_ns=10,
        global_deadline_ns=20,
    )

    assert clock() == 10
    assert clock() == 21
    assert clock() == 10
