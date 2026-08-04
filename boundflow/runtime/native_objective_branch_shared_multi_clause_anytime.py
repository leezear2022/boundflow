"""NRIR-36 control composed with the objective-branch shared production queue."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=duplicate-code,protected-access,missing-function-docstring

from __future__ import annotations

from dataclasses import replace
import time
from typing import Callable, Optional

import torch

from ..ir.multi_clause_anytime import NativeMultiClauseAnytimeSliceIR
from ..ir.shared_parametric_ancestral import NativeSharedParametricAncestralPlanIR
from ..ir.task import BFTaskModule
from .complete_verifier_query import _normalize_objective_matrix
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_candidate_search import NativeProjectedGradientSearchPolicy
from .native_intermediate_refinement import (
    intermediate_bounds_hash,
    intermediate_refinement_semantic_trace_hash,
)
from .native_multi_clause_anytime import (
    NativeMultiClauseAnytimeActionTrace,
    NativeMultiClauseAnytimeProgram,
    NativeMultiClauseAnytimeTrace,
    _OneShotSliceClock,
    _accepted_child,
    _action_trace,
    _aggregate_from_slices,
    _canonical_hash,
    _decision_from_floor,
    _skipped_action,
)
from .native_objective_branch_score import NativeObjectiveBranchPolicy
from .native_objective_branch_shared_evaluator import (
    compile_native_objective_branch_shared_plan,
)
from .native_objective_branch_shared_production_queue import (
    execute_native_objective_branch_shared_production_queue,
)
from .native_objective_hard_clause_escalation import (
    execute_native_objective_hard_clause_escalation_program,
)
from .native_parametric_optimizer import NativeParametricOptimizerTemplateCache
from .native_property_verdict import (
    NativePropertyVerdictExecution,
    derive_native_property_verdict,
)
from .native_shared_parametric_ancestral import NativeSharedParametricAncestralExecution
from .native_shared_parametric_multi_clause_anytime import (
    NativeSharedParametricMultiClauseExecution,
    NativeSharedParametricMultiClausePackedExecution,
)
from .task_executor import InputSpec

ClockNs = Callable[[], int]


def execute_native_objective_branch_shared_multi_clause_anytime_program(  # pylint: disable=too-many-statements
    program: NativeMultiClauseAnytimeProgram,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
    query_id: str,
    search_policy: NativeProjectedGradientSearchPolicy,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    branch_policy: NativeObjectiveBranchPolicy,
    clock_ns: ClockNs = time.monotonic_ns,
) -> NativeSharedParametricMultiClauseExecution:
    """Execute frozen top-2 control with objective scoring inside each slice."""

    if not query_id:
        raise ValueError("objective-branch multi-clause query ID must be non-empty")
    branch_policy.validate()
    objectives = _normalize_objective_matrix(linear_spec_C)
    program.validate_against(
        module,
        input_spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    started_ns = clock_ns()
    global_deadline_ns = started_ns + program.plan.whole_query_timeout_ns
    actions: list[NativeMultiClauseAnytimeActionTrace] = []
    action_started_ns = clock_ns()
    floor_clock_first = True

    def floor_clock_ns() -> int:
        nonlocal floor_clock_first
        if floor_clock_first:
            floor_clock_first = False
            return started_ns
        return clock_ns()

    floor = execute_native_objective_hard_clause_escalation_program(
        program.floor_program,
        module,
        input_spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id=f"{query_id}:floor-nrir31",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
        clock_ns=floor_clock_ns,
    )
    action_finished_ns = clock_ns()
    actions.append(
        _action_trace(
            program,
            0,
            executed=True,
            reason="nrir31_floor_completed",
            elapsed_ns=max(0, action_finished_ns - action_started_ns),
            output=floor.trace.semantic_signature_hash,
        )
    )
    action_started_ns = clock_ns()
    decision = _decision_from_floor(program.plan, floor, thresholds)
    action_finished_ns = clock_ns()
    actions.append(
        _action_trace(
            program,
            1,
            executed=True,
            reason=decision.reason,
            elapsed_ns=max(0, action_finished_ns - action_started_ns),
            output=decision.to_dict(program.plan),
        )
    )
    compiler_cache = NativeParametricOptimizerTemplateCache()
    packed_executions: list[NativeSharedParametricMultiClausePackedExecution] = []
    prior_unsafe = False
    selected = decision.selected_original_clause_indices
    for position in range(program.plan.allocation_policy.max_selected_clauses):
        compile_action = 2 + 2 * position
        execute_action = compile_action + 1
        if position >= len(selected):
            reason = "priority_slot_not_selected"
            actions.append(_skipped_action(program, compile_action, reason))
            actions.append(_skipped_action(program, execute_action, reason))
            continue
        ordinal = selected[position]
        child = _accepted_child(floor, ordinal)
        assert child is not None
        dispatch_ns = clock_ns()
        dispatch_elapsed_ns = max(0, dispatch_ns - started_ns)
        remaining_before_ns = max(0, global_deadline_ns - dispatch_ns)
        remaining_selected_count = len(selected) - position
        allocated_slice_ns = remaining_before_ns // remaining_selected_count
        cutoff_ns = min(global_deadline_ns, dispatch_ns + allocated_slice_ns)
        objective = objectives[:, ordinal : ordinal + 1, :].contiguous()
        threshold = thresholds[ordinal : ordinal + 1].contiguous()
        packed_plan: Optional[NativeSharedParametricAncestralPlanIR] = None
        composite_plan = None
        packed: Optional[NativeSharedParametricAncestralExecution] = None
        verdict: Optional[NativePropertyVerdictExecution] = None
        cutoff_signaled = False
        reason = "packed_slice_not_started"
        if prior_unsafe:
            reason = "prior_slice_unsafe"
            actions.append(_skipped_action(program, compile_action, reason))
            actions.append(_skipped_action(program, execute_action, reason))
        elif allocated_slice_ns == 0 or dispatch_ns >= cutoff_ns:
            reason = "slice_deadline_before_compile"
            actions.append(_skipped_action(program, compile_action, reason))
            actions.append(_skipped_action(program, execute_action, reason))
        else:
            compile_started_ns = clock_ns()
            composite_plan = compile_native_objective_branch_shared_plan(
                module,
                input_spec,
                linear_spec_C=objective,
                threshold=threshold,
                root_refinement=child.refinement,
                optimizer_policy=optimizer_policy,
                branch_policy=branch_policy,
                plan_id=(
                    f"{query_id}:clause:{ordinal:04d}:priority:{position}:"
                    "objective-branch-shared-nrir40"
                ),
            )
            packed_plan = composite_plan.shared_plan
            compile_finished_ns = clock_ns()
            actions.append(
                _action_trace(
                    program,
                    compile_action,
                    executed=True,
                    reason="shared_parametric_plan_compiled",
                    elapsed_ns=max(0, compile_finished_ns - compile_started_ns),
                    output=packed_plan.stable_hash(),
                )
            )
            if compile_finished_ns >= cutoff_ns:
                reason = "slice_deadline_after_compile"
                actions.append(_skipped_action(program, execute_action, reason))
            else:
                execute_started_ns = clock_ns()
                slice_clock = _OneShotSliceClock(
                    clock_ns,
                    cutoff_ns=cutoff_ns,
                    global_deadline_ns=global_deadline_ns,
                )
                try:
                    packed = execute_native_objective_branch_shared_production_queue(
                        composite_plan,
                        module,
                        input_spec,
                        linear_spec_C=objective,
                        threshold=threshold,
                        root_refinement=child.refinement,
                        optimizer_policy=optimizer_policy,
                        branch_policy=branch_policy,
                        compiler_cache=compiler_cache,
                        query_id=(
                            f"{query_id}:clause:{ordinal:04d}:priority:{position}"
                        ),
                        whole_query_started_ns=started_ns,
                        clock_ns=slice_clock,
                    )
                except RuntimeError as error:
                    if str(error) != (
                        "objective-branch production deadline expired at root"
                    ):
                        raise
                execute_finished_ns = clock_ns()
                cutoff_signaled = slice_clock.signaled
                if packed is None:
                    reason = "slice_deadline_during_packed_root"
                    actions.append(_skipped_action(program, execute_action, reason))
                else:
                    root_id = packed.queue.trace.evaluations[0].node.node_id
                    verdict = derive_native_property_verdict(
                        module,
                        input_spec,
                        linear_spec_C=objective,
                        queue_execution=packed.queue,
                        candidate_counterexamples=(
                            (root_id, child.query.clauses[0].search.best_input),
                        ),
                    )
                    reason = f"packed_slice_{verdict.trace.status}"
                    prior_unsafe = verdict.trace.status == "unsafe"
                    actions.append(
                        _action_trace(
                            program,
                            execute_action,
                            executed=True,
                            reason=reason,
                            elapsed_ns=max(0, execute_finished_ns - execute_started_ns),
                            output=packed.trace.semantic_signature_hash,
                        )
                    )
        finished_ns = clock_ns()
        packed_result = packed is not None and verdict is not None
        slice_ir = NativeMultiClauseAnytimeSliceIR(
            plan_hash=program.plan.stable_hash(),
            decision_hash=decision.stable_hash(program.plan),
            priority_position=position,
            original_clause_index=ordinal,
            dispatch_started_elapsed_ns=dispatch_elapsed_ns,
            remaining_before_ns=remaining_before_ns,
            remaining_selected_count=remaining_selected_count,
            allocated_slice_ns=allocated_slice_ns,
            slice_cutoff_elapsed_ns=max(0, cutoff_ns - started_ns),
            finished_elapsed_ns=max(0, finished_ns - started_ns),
            source_refinement_plan_hash=child.refinement.program.plan.stable_hash(),
            source_refinement_semantic_trace_hash=(
                intermediate_refinement_semantic_trace_hash(child.refinement)
            ),
            source_final_intermediate_bounds_hash=intermediate_bounds_hash(
                child.refinement.relu_pre
            ),
            packed_plan_hash=(
                packed_plan.stable_hash() if packed_result and packed_plan else None
            ),
            packed_queue_trace_hash=(
                packed.queue.trace.stable_hash() if packed_result and packed else None
            ),
            packed_verdict_trace_hash=(
                verdict.trace.stable_hash(packed.queue.trace)
                if packed_result and packed and verdict
                else None
            ),
            packed_status=verdict.trace.status if packed_result and verdict else None,
            accepted_nodes=(
                len(packed.queue.trace.evaluations) if packed_result and packed else 0
            ),
            sibling_group_count=(
                len(packed.batch_commits) - 1 if packed_result and packed else 0
            ),
            cutoff_signaled=cutoff_signaled,
            reason=reason,
        )
        slice_ir.validate_against(program.plan, decision)
        packed_executions.append(
            NativeSharedParametricMultiClausePackedExecution(
                slice_ir=slice_ir,
                packed_plan=packed_plan,
                packed=packed,
                verdict=verdict,
            )
        )
    slices = tuple(item.slice_ir for item in packed_executions)
    aggregate = _aggregate_from_slices(program.plan, decision, slices)
    actions.append(
        _action_trace(
            program,
            6,
            executed=True,
            reason="original_ordinals_aggregated_monotonically",
            elapsed_ns=0,
            output=aggregate.to_dict(program.plan, decision, slices),
        )
    )
    actions.append(
        _action_trace(
            program,
            7,
            executed=True,
            reason="result_emitted",
            elapsed_ns=0,
            output=aggregate.stable_hash(program.plan, decision, slices),
        )
    )
    finished_ns = clock_ns()
    compiler_cache.validate()
    trace = NativeMultiClauseAnytimeTrace(
        query_id=query_id,
        plan_hash=program.plan.stable_hash(),
        task_ir_hash=program.task_ir.stable_hash(),
        schedule_hash=program.schedule.stable_hash(program.task_ir),
        floor_trace_hash=floor.trace.semantic_signature_hash,
        decision_hash=decision.stable_hash(program.plan),
        slice_hashes=tuple(item.stable_hash(program.plan, decision) for item in slices),
        aggregate_hash=aggregate.stable_hash(program.plan, decision, slices),
        actions=tuple(actions),
        fallback_reasons=tuple(item.reason for item in slices),
        elapsed_ns=max(0, finished_ns - started_ns),
        deadline_ns=program.plan.whole_query_timeout_ns,
        semantic_signature_hash="",
    )
    trace = replace(
        trace, semantic_signature_hash=_canonical_hash(trace.semantic_dict())
    )
    execution = NativeSharedParametricMultiClauseExecution(
        program=program,
        floor=floor,
        decision=decision,
        packed_executions=tuple(packed_executions),
        aggregate=aggregate,
        cache_events=compiler_cache.events,
        template_hashes=tuple(item.template_hash for item in compiler_cache.templates),
        trace=trace,
    )
    execution.validate_against(
        module,
        input_spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    for item in execution.packed_executions:
        if item.packed is not None and (
            item.packed.queue.objective_branch_policy != branch_policy
        ):
            raise ValueError("objective-branch multi-clause policy was erased")
    return execution


__all__ = [
    "execute_native_objective_branch_shared_multi_clause_anytime_program",
]
