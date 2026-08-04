"""NRIR-36 control composed with the NRIR-37 shared-parametric evaluator."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=duplicate-code,too-many-lines,protected-access

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Callable, Optional, Tuple

import torch

from ..ir.multi_clause_anytime import (
    NativeMultiClauseAnytimeAggregateIR,
    NativeMultiClauseAnytimeDecisionIR,
    NativeMultiClauseAnytimeSliceIR,
)
from ..ir.parametric_optimizer import NativeParametricOptimizerCacheEventIR
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
from .native_objective_hard_clause_escalation import (
    NativeObjectiveHardClauseEscalationExecution,
    execute_native_objective_hard_clause_escalation_program,
)
from .native_parametric_optimizer import NativeParametricOptimizerTemplateCache
from .native_property_verdict import (
    NativePropertyVerdictExecution,
    derive_native_property_verdict,
)
from .native_shared_parametric_ancestral import (
    NativeSharedParametricAncestralExecution,
    compile_native_shared_parametric_ancestral_plan,
    execute_native_shared_parametric_ancestral_queue,
    validate_native_shared_parametric_ancestral_plan,
)
from .task_executor import InputSpec

ClockNs = Callable[[], int]


@dataclass(frozen=True)
class NativeSharedParametricMultiClausePackedExecution:
    """One selected priority slice executed by the shared evaluator."""

    slice_ir: NativeMultiClauseAnytimeSliceIR
    packed_plan: Optional[NativeSharedParametricAncestralPlanIR]
    packed: Optional[NativeSharedParametricAncestralExecution]
    verdict: Optional[NativePropertyVerdictExecution]

    def to_dict(
        self,
        program: NativeMultiClauseAnytimeProgram,
        decision: NativeMultiClauseAnytimeDecisionIR,
    ) -> dict[str, object]:
        return {
            "slice": self.slice_ir.to_dict(program.plan, decision),
            "packed_plan": (
                None if self.packed_plan is None else self.packed_plan.to_dict()
            ),
            "packed": None if self.packed is None else self.packed.to_dict(),
            "verdict": None if self.verdict is None else self.verdict.trace.to_dict(),
        }


@dataclass(frozen=True)
class NativeSharedParametricMultiClauseExecution:
    """Frozen floor/control plus one query-owned shared compiler cache."""

    program: NativeMultiClauseAnytimeProgram
    floor: NativeObjectiveHardClauseEscalationExecution
    decision: NativeMultiClauseAnytimeDecisionIR
    packed_executions: Tuple[NativeSharedParametricMultiClausePackedExecution, ...]
    aggregate: NativeMultiClauseAnytimeAggregateIR
    cache_events: Tuple[NativeParametricOptimizerCacheEventIR, ...]
    template_hashes: Tuple[str, ...]
    trace: NativeMultiClauseAnytimeTrace

    def validate_against(  # pylint: disable=too-many-statements
        self,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        linear_spec_C: torch.Tensor,
        thresholds: torch.Tensor,
        search_policy: NativeProjectedGradientSearchPolicy,
        optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    ) -> None:
        objectives = _normalize_objective_matrix(linear_spec_C)
        self.program.validate_against(
            module,
            input_spec,
            linear_spec_C=objectives,
            thresholds=thresholds,
            search_policy=search_policy,
            optimizer_policy=optimizer_policy,
        )
        self.floor.validate_against(
            module,
            input_spec,
            linear_spec_C=objectives,
            thresholds=thresholds,
            search_policy=search_policy,
            optimizer_policy=optimizer_policy,
        )
        if self.floor.program != self.program.floor_program:
            raise ValueError("shared multi-clause floor program differs")
        expected_decision = _decision_from_floor(
            self.program.plan, self.floor, thresholds
        )
        if self.decision != expected_decision:
            raise ValueError("shared multi-clause Decision/floor binding differs")
        selected = self.decision.selected_original_clause_indices
        if (
            len(self.packed_executions) != len(selected)
            or tuple(
                item.slice_ir.original_clause_index for item in self.packed_executions
            )
            != selected
        ):
            raise ValueError("shared multi-clause selected slice coverage differs")
        committed_event_hashes: list[str] = []
        for position, item in enumerate(self.packed_executions):
            slice_ir = item.slice_ir
            slice_ir.validate_against(self.program.plan, self.decision)
            if slice_ir.priority_position != position:
                raise ValueError("shared multi-clause priority position differs")
            ordinal = slice_ir.original_clause_index
            child = _accepted_child(self.floor, ordinal)
            if child is None:
                raise ValueError("shared multi-clause floor source is absent")
            objective = objectives[:, ordinal : ordinal + 1, :].contiguous()
            threshold = thresholds[ordinal : ordinal + 1].contiguous()
            if item.packed_plan is not None:
                validate_native_shared_parametric_ancestral_plan(
                    item.packed_plan,
                    module,
                    input_spec,
                    linear_spec_C=objective,
                    threshold=threshold,
                    root_refinement=child.refinement,
                    optimizer_policy=optimizer_policy,
                )
            if item.packed is not None:
                if item.packed_plan is None or item.verdict is None:
                    raise ValueError("shared multi-clause packed result is incomplete")
                if item.packed.plan != item.packed_plan:
                    raise ValueError("shared multi-clause packed plan differs")
                item.packed.validate_against(
                    module,
                    input_spec,
                    linear_spec_C=objective,
                    threshold=threshold,
                    root_refinement=child.refinement,
                    optimizer_policy=optimizer_policy,
                )
                item.verdict.validate_against(
                    module,
                    input_spec,
                    linear_spec_C=objective,
                    queue_execution=item.packed.queue,
                )
                committed_event_hashes.extend(
                    compiler.cache_event.stable_hash()
                    for compiler in item.packed.compiler_batches
                )
                if (
                    slice_ir.packed_plan_hash != item.packed_plan.stable_hash()
                    or slice_ir.packed_queue_trace_hash
                    != item.packed.queue.trace.stable_hash()
                    or slice_ir.packed_verdict_trace_hash
                    != item.verdict.trace.stable_hash(item.packed.queue.trace)
                    or slice_ir.packed_status != item.verdict.trace.status
                    or slice_ir.accepted_nodes
                    != len(item.packed.queue.trace.evaluations)
                    or slice_ir.sibling_group_count
                    != len(item.packed.batch_commits) - 1
                    or item.packed.trace.source_elapsed_ns
                    < slice_ir.dispatch_started_elapsed_ns
                    or item.packed.trace.whole_elapsed_ns
                    < item.packed.trace.source_elapsed_ns
                    or item.packed.trace.selected_native_reexecution is not False
                ):
                    raise ValueError("shared multi-clause slice/packed binding differs")
            elif item.verdict is not None:
                raise ValueError("shared multi-clause verdict lacks a queue")
            elif any(
                value is not None
                for value in (
                    slice_ir.packed_plan_hash,
                    slice_ir.packed_queue_trace_hash,
                    slice_ir.packed_verdict_trace_hash,
                    slice_ir.packed_status,
                )
            ):
                raise ValueError("shared multi-clause skipped slice retains result")
        slices = tuple(item.slice_ir for item in self.packed_executions)
        expected_aggregate = _aggregate_from_slices(
            self.program.plan, self.decision, slices
        )
        if self.aggregate != expected_aggregate:
            raise ValueError("shared multi-clause aggregate/result differs")
        for event in self.cache_events:
            event.validate()
        if self.cache_events:
            event_hashes = tuple(event.stable_hash() for event in self.cache_events)
            if (
                tuple(event.event_index for event in self.cache_events)
                != tuple(range(len(self.cache_events)))
                or len(self.template_hashes) != 1
                or any(
                    event.template_hash != self.template_hashes[0]
                    for event in self.cache_events
                )
                or sum(event.outcome == "miss_compiled" for event in self.cache_events)
                != 1
                or any(
                    event.outcome != "hit_exact_contract"
                    for event in self.cache_events[1:]
                )
                or any(value not in event_hashes for value in committed_event_hashes)
            ):
                raise ValueError("shared multi-clause cache ownership differs")
        elif self.template_hashes or committed_event_hashes:
            raise ValueError("shared multi-clause empty cache accounting differs")
        _validate_action_bindings(self)
        self.trace.validate_against(self.program, self.decision, slices, self.aggregate)

    def to_dict(self) -> dict[str, object]:
        return {
            "program": self.program.to_dict(),
            "floor": self.floor.trace.to_dict(),
            "decision": self.decision.to_dict(self.program.plan),
            "packed_executions": [
                item.to_dict(self.program, self.decision)
                for item in self.packed_executions
            ],
            "aggregate": self.aggregate.to_dict(
                self.program.plan,
                self.decision,
                tuple(item.slice_ir for item in self.packed_executions),
            ),
            "cache_events": [item.to_dict() for item in self.cache_events],
            "template_hashes": list(self.template_hashes),
            "cache_scope": "one_query_cross_batch_cross_clause_v1",
            "selected_native_reexecution": False,
            "performance_claimed": False,
            "trace": self.trace.to_dict(),
        }


def _validate_action_bindings(
    execution: NativeSharedParametricMultiClauseExecution,
) -> None:
    program = execution.program
    decision = execution.decision
    slices = tuple(item.slice_ir for item in execution.packed_executions)
    expected: list[tuple[bool, str, object]] = [
        (True, "nrir31_floor_completed", execution.floor.trace.semantic_signature_hash),
        (True, decision.reason, decision.to_dict(program.plan)),
    ]
    for position in range(program.plan.allocation_policy.max_selected_clauses):
        if position >= len(execution.packed_executions):
            reason = "priority_slot_not_selected"
            expected.extend(
                (
                    (False, reason, {"skipped": reason}),
                    (False, reason, {"skipped": reason}),
                )
            )
            continue
        item = execution.packed_executions[position]
        reason = item.slice_ir.reason
        expected.append(
            (
                item.packed_plan is not None,
                (
                    "shared_parametric_plan_compiled"
                    if item.packed_plan is not None
                    else reason
                ),
                (
                    item.packed_plan.stable_hash()
                    if item.packed_plan is not None
                    else {"skipped": reason}
                ),
            )
        )
        expected.append(
            (
                item.packed is not None,
                reason,
                (
                    item.packed.trace.semantic_signature_hash
                    if item.packed is not None
                    else {"skipped": reason}
                ),
            )
        )
    expected.extend(
        (
            (
                True,
                "original_ordinals_aggregated_monotonically",
                execution.aggregate.to_dict(program.plan, decision, slices),
            ),
            (
                True,
                "result_emitted",
                execution.aggregate.stable_hash(program.plan, decision, slices),
            ),
        )
    )
    if len(execution.trace.actions) != len(expected) or any(
        action.executed != executed
        or action.reason != reason
        or action.output_hash != _canonical_hash(output)
        for action, (executed, reason, output) in zip(execution.trace.actions, expected)
    ):
        raise ValueError("shared multi-clause action/result binding differs")


def execute_native_shared_parametric_multi_clause_anytime_program(  # pylint: disable=too-many-statements
    program: NativeMultiClauseAnytimeProgram,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
    query_id: str,
    search_policy: NativeProjectedGradientSearchPolicy,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    clock_ns: ClockNs = time.monotonic_ns,
) -> NativeSharedParametricMultiClauseExecution:
    """Execute frozen top-2 control with one NRIR-37 compiler-cache owner."""

    if not query_id:
        raise ValueError("shared multi-clause query ID must be non-empty")
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
            packed_plan = compile_native_shared_parametric_ancestral_plan(
                module,
                input_spec,
                linear_spec_C=objective,
                threshold=threshold,
                root_refinement=child.refinement,
                optimizer_policy=optimizer_policy,
                plan_id=(
                    f"{query_id}:clause:{ordinal:04d}:priority:{position}:"
                    "shared-parametric-nrir37"
                ),
            )
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
                    packed = execute_native_shared_parametric_ancestral_queue(
                        packed_plan,
                        module,
                        input_spec,
                        linear_spec_C=objective,
                        threshold=threshold,
                        root_refinement=child.refinement,
                        optimizer_policy=optimizer_policy,
                        compiler_cache=compiler_cache,
                        query_id=(
                            f"{query_id}:clause:{ordinal:04d}:priority:{position}"
                        ),
                        whole_query_started_ns=started_ns,
                        clock_ns=slice_clock,
                    )
                except RuntimeError as error:
                    if str(error) != "shared-parametric deadline expired at root":
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
    return execution


__all__ = [
    "NativeSharedParametricMultiClauseExecution",
    "NativeSharedParametricMultiClausePackedExecution",
    "execute_native_shared_parametric_multi_clause_anytime_program",
]
