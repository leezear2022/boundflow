"""Runtime for ranked multi-clause packed work under one global deadline."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=duplicate-code,too-many-lines,too-few-public-methods
# pylint: disable=too-many-positional-arguments

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import time
from typing import Callable, Optional, Tuple

import torch

from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.multi_clause_anytime import (
    NativeMultiClauseAnytimeAggregateIR,
    NativeMultiClauseAnytimeCandidateIR,
    NativeMultiClauseAnytimeDecisionIR,
    NativeMultiClauseAnytimeOutcomeIR,
    NativeMultiClauseAnytimePlanIR,
    NativeMultiClauseAnytimePolicyIR,
    NativeMultiClauseAnytimeScheduleIR,
    NativeMultiClauseAnytimeSliceIR,
    NativeMultiClauseAnytimeTaskIRModule,
    NativeMultiClauseAnytimeTaskKind,
    lower_native_multi_clause_anytime_ir,
)
from ..ir.objective_ancestral_sibling_pack import (
    NativeObjectiveAncestralSiblingPackPlanIR,
)
from ..ir.task import BFTaskModule
from .complete_verifier_query import _normalize_objective_matrix
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_candidate_search import NativeProjectedGradientSearchPolicy
from .native_intermediate_refinement import (
    intermediate_bounds_hash,
    intermediate_refinement_semantic_trace_hash,
)
from .native_objective_ancestral_sibling_pack import (
    NativeObjectiveAncestralSiblingPackExecution,
    compile_native_objective_ancestral_sibling_pack_plan,
    execute_native_objective_ancestral_sibling_pack_queue,
    validate_native_objective_ancestral_sibling_pack_plan,
)
from .native_objective_hard_clause_escalation import (
    NativeObjectiveHardClauseEscalationClauseExecution,
    NativeObjectiveHardClauseEscalationExecution,
    NativeObjectiveHardClauseEscalationProgram,
    compile_native_objective_hard_clause_escalation_program,
    execute_native_objective_hard_clause_escalation_program,
)
from .native_property_verdict import (
    NativePropertyVerdictExecution,
    derive_native_property_verdict,
)
from .task_executor import InputSpec

NATIVE_MULTI_CLAUSE_ANYTIME_ACTION_TRACE_SCHEMA_VERSION = (
    "boundflow.native-multi-clause-anytime-action-trace/v1"
)
NATIVE_MULTI_CLAUSE_ANYTIME_TRACE_SCHEMA_VERSION = (
    "boundflow.native-multi-clause-anytime-trace/v1"
)
ClockNs = Callable[[], int]


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class NativeMultiClauseAnytimeProgram:
    """Frozen NRIR-31 floor plus typed priority and slice control."""

    floor_program: NativeObjectiveHardClauseEscalationProgram
    plan: NativeMultiClauseAnytimePlanIR
    task_ir: NativeMultiClauseAnytimeTaskIRModule
    schedule: NativeMultiClauseAnytimeScheduleIR

    def validate_against(
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
        self.floor_program.validate_against(
            module,
            input_spec,
            linear_spec_C=objectives,
            thresholds=thresholds,
            search_policy=search_policy,
            optimizer_policy=optimizer_policy,
        )
        self.plan.validate()
        self.schedule.validate_against(self.task_ir)
        if (
            self.plan.floor_plan_hash != self.floor_program.plan.stable_hash()
            or self.plan.floor_task_ir_hash != self.floor_program.task_ir.stable_hash()
            or self.plan.floor_schedule_hash
            != self.floor_program.schedule.stable_hash(self.floor_program.task_ir)
            or self.plan.objective_matrix_hash != tensor_content_hash(objectives)
            or self.plan.thresholds_hash != tensor_content_hash(thresholds)
            or self.plan.search_policy_hash != search_policy.stable_hash()
            or self.plan.optimizer_policy_hash != optimizer_policy.stable_hash()
            or self.plan.clause_count != int(objectives.shape[1])
            or self.task_ir.plan_hash != self.plan.stable_hash()
        ):
            raise ValueError("multi-clause anytime program/query binding differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "floor_program": self.floor_program.to_dict(),
            "plan": self.plan.to_dict(),
            "plan_hash": self.plan.stable_hash(),
            "task_ir": self.task_ir.to_dict(),
            "task_ir_hash": self.task_ir.stable_hash(),
            "schedule": self.schedule.to_dict(),
            "schedule_hash": self.schedule.stable_hash(self.task_ir),
        }


@dataclass(frozen=True)
class NativeMultiClauseAnytimeActionTrace:
    """One executed or guarded-skip scheduler action."""

    sequence: int
    action_id: str
    task_id: str
    kind: NativeMultiClauseAnytimeTaskKind
    guard: str
    priority_position: Optional[int]
    executed: bool
    reason: str
    elapsed_ns: int
    output_hash: str
    schema_version: str = NATIVE_MULTI_CLAUSE_ANYTIME_ACTION_TRACE_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version
            != NATIVE_MULTI_CLAUSE_ANYTIME_ACTION_TRACE_SCHEMA_VERSION
            or self.sequence < 0
            or not self.action_id
            or not self.task_id
            or self.guard not in {"always", "selected_slot_available_before_deadline"}
            or self.priority_position not in {None, 0, 1}
            or not self.reason
            or self.elapsed_ns < 0
            or not _is_sha256(self.output_hash)
            or (not self.executed and self.elapsed_ns != 0)
        ):
            raise ValueError("multi-clause anytime action trace is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "guard": self.guard,
            "priority_position": self.priority_position,
            "executed": self.executed,
            "reason": self.reason,
            "elapsed_ns": self.elapsed_ns,
            "output_hash": self.output_hash,
        }


@dataclass(frozen=True)
class NativeMultiClauseAnytimeTrace:
    """Replay identity for floor, ranking, slices, and final aggregate."""

    query_id: str
    plan_hash: str
    task_ir_hash: str
    schedule_hash: str
    floor_trace_hash: str
    decision_hash: str
    slice_hashes: Tuple[str, ...]
    aggregate_hash: str
    actions: Tuple[NativeMultiClauseAnytimeActionTrace, ...]
    fallback_reasons: Tuple[str, ...]
    elapsed_ns: int
    deadline_ns: int
    semantic_signature_hash: str
    performance_claimed: bool = False
    schema_version: str = NATIVE_MULTI_CLAUSE_ANYTIME_TRACE_SCHEMA_VERSION

    def semantic_dict(self) -> dict[str, object]:
        return {
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "schedule_hash": self.schedule_hash,
            "floor_trace_hash": self.floor_trace_hash,
            "decision_hash": self.decision_hash,
            "slice_hashes": list(self.slice_hashes),
            "aggregate_hash": self.aggregate_hash,
            "actions": [action.to_dict() for action in self.actions],
            "fallback_reasons": list(self.fallback_reasons),
        }

    def validate_against(
        self,
        program: NativeMultiClauseAnytimeProgram,
        decision: NativeMultiClauseAnytimeDecisionIR,
        slices: Tuple[NativeMultiClauseAnytimeSliceIR, ...],
        aggregate: NativeMultiClauseAnytimeAggregateIR,
    ) -> None:
        program.schedule.validate_against(program.task_ir)
        decision.validate_against(program.plan)
        aggregate.validate_against(program.plan, decision, slices)
        if (
            self.schema_version != NATIVE_MULTI_CLAUSE_ANYTIME_TRACE_SCHEMA_VERSION
            or not self.query_id
            or self.plan_hash != program.plan.stable_hash()
            or self.task_ir_hash != program.task_ir.stable_hash()
            or self.schedule_hash != program.schedule.stable_hash(program.task_ir)
            or self.floor_trace_hash != decision.floor_trace_hash
            or self.decision_hash != decision.stable_hash(program.plan)
            or self.slice_hashes
            != tuple(item.stable_hash(program.plan, decision) for item in slices)
            or self.aggregate_hash
            != aggregate.stable_hash(program.plan, decision, slices)
            or len(self.actions) != len(program.schedule.actions)
            or len(self.fallback_reasons) != len(slices)
            or self.elapsed_ns < 0
            or self.deadline_ns != program.plan.whole_query_timeout_ns
            or self.performance_claimed is not False
            or self.semantic_signature_hash != _canonical_hash(self.semantic_dict())
        ):
            raise ValueError("multi-clause anytime trace differs")
        for expected, actual in zip(program.schedule.actions, self.actions):
            actual.validate()
            if (
                expected.sequence != actual.sequence
                or expected.action_id != actual.action_id
                or expected.task_id != actual.task_id
                or expected.kind != actual.kind
                or expected.guard != actual.guard
                or expected.priority_position != actual.priority_position
            ):
                raise ValueError("multi-clause anytime runtime/Schedule differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            **self.semantic_dict(),
            "elapsed_ns": self.elapsed_ns,
            "deadline_ns": self.deadline_ns,
            "deadline_enforcement": "one_global_clock_dynamic_equal_remaining",
            "semantic_signature_hash": self.semantic_signature_hash,
            "performance_claimed": self.performance_claimed,
        }


@dataclass(frozen=True)
class NativeMultiClauseAnytimePackedExecution:
    """One selected clause's allocation and optional packed execution."""

    slice_ir: NativeMultiClauseAnytimeSliceIR
    packed_plan: Optional[NativeObjectiveAncestralSiblingPackPlanIR]
    packed: Optional[NativeObjectiveAncestralSiblingPackExecution]
    verdict: Optional[NativePropertyVerdictExecution]

    def to_dict(
        self,
        plan: NativeMultiClauseAnytimePlanIR,
        decision: NativeMultiClauseAnytimeDecisionIR,
    ) -> dict[str, object]:
        return {
            "slice": self.slice_ir.to_dict(plan, decision),
            "packed_plan": (
                None if self.packed_plan is None else self.packed_plan.to_dict()
            ),
            "packed": None if self.packed is None else self.packed.to_dict(),
            "verdict": (None if self.verdict is None else self.verdict.trace.to_dict()),
        }


@dataclass(frozen=True)
class NativeMultiClauseAnytimeExecution:
    """Typed floor, ranked packed slices, and monotone final result."""

    program: NativeMultiClauseAnytimeProgram
    floor: NativeObjectiveHardClauseEscalationExecution
    decision: NativeMultiClauseAnytimeDecisionIR
    packed_executions: Tuple[NativeMultiClauseAnytimePackedExecution, ...]
    aggregate: NativeMultiClauseAnytimeAggregateIR
    trace: NativeMultiClauseAnytimeTrace

    def validate_against(
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
            raise ValueError("multi-clause anytime floor program differs")
        expected_decision = _decision_from_floor(
            self.program.plan, self.floor, thresholds
        )
        if self.decision != expected_decision:
            raise ValueError("multi-clause anytime Decision/floor binding differs")
        selected = self.decision.selected_original_clause_indices
        if (
            len(self.packed_executions) != len(selected)
            or tuple(
                item.slice_ir.original_clause_index for item in self.packed_executions
            )
            != selected
        ):
            raise ValueError("multi-clause anytime selected slice coverage differs")
        for position, item in enumerate(self.packed_executions):
            _validate_packed_execution(
                item,
                position,
                self.program,
                self.floor,
                self.decision,
                module,
                input_spec,
                objectives,
                thresholds,
                optimizer_policy,
            )
        slices = tuple(item.slice_ir for item in self.packed_executions)
        expected_aggregate = _aggregate_from_slices(
            self.program.plan, self.decision, slices
        )
        if self.aggregate != expected_aggregate:
            raise ValueError("multi-clause anytime aggregate/result differs")
        _validate_action_bindings(self)
        self.trace.validate_against(self.program, self.decision, slices, self.aggregate)

    def to_dict(self) -> dict[str, object]:
        return {
            "program": self.program.to_dict(),
            "floor": self.floor.trace.to_dict(),
            "decision": self.decision.to_dict(self.program.plan),
            "packed_executions": [
                item.to_dict(self.program.plan, self.decision)
                for item in self.packed_executions
            ],
            "aggregate": self.aggregate.to_dict(
                self.program.plan,
                self.decision,
                tuple(item.slice_ir for item in self.packed_executions),
            ),
            "trace": self.trace.to_dict(),
        }


class _OneShotSliceClock:
    """Expose one synthetic global expiry when an allocation is exhausted."""

    def __init__(
        self,
        source: ClockNs,
        *,
        cutoff_ns: int,
        global_deadline_ns: int,
    ) -> None:
        self._source = source
        self._cutoff_ns = cutoff_ns
        self._global_deadline_ns = global_deadline_ns
        self._calls = 0
        self.signaled = False

    def __call__(self) -> int:
        observed = self._source()
        if self._calls > 0 and not self.signaled and observed >= self._cutoff_ns:
            self.signaled = True
            return self._global_deadline_ns + 1
        self._calls += 1
        return observed


def compile_native_multi_clause_anytime_program(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
    plan_id: str,
    search_policy: NativeProjectedGradientSearchPolicy,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
) -> NativeMultiClauseAnytimeProgram:
    """Compile the exact NRIR-31 floor and two guarded priority slots."""

    objectives = _normalize_objective_matrix(linear_spec_C)
    floor = compile_native_objective_hard_clause_escalation_program(
        module,
        input_spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        plan_id=f"{plan_id}:floor-nrir31",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    plan = NativeMultiClauseAnytimePlanIR(
        plan_id=plan_id,
        floor_plan_hash=floor.plan.stable_hash(),
        floor_task_ir_hash=floor.task_ir.stable_hash(),
        floor_schedule_hash=floor.schedule.stable_hash(floor.task_ir),
        objective_matrix_hash=tensor_content_hash(objectives),
        thresholds_hash=tensor_content_hash(thresholds),
        search_policy_hash=search_policy.stable_hash(),
        optimizer_policy_hash=optimizer_policy.stable_hash(),
        allocation_policy=NativeMultiClauseAnytimePolicyIR(),
        clause_count=int(objectives.shape[1]),
    )
    task_ir, schedule = lower_native_multi_clause_anytime_ir(plan)
    program = NativeMultiClauseAnytimeProgram(floor, plan, task_ir, schedule)
    program.validate_against(
        module,
        input_spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    return program


def _accepted_child(
    floor: NativeObjectiveHardClauseEscalationExecution,
    original_clause_index: int,
) -> Optional[NativeObjectiveHardClauseEscalationClauseExecution]:
    return next(
        (
            child
            for child in floor.clause_executions
            if child.original_clause_index == original_clause_index
            and child.accepted_before_deadline
            and bool(child.query.clauses)
        ),
        None,
    )


def _candidate_from_child(
    child: NativeObjectiveHardClauseEscalationClauseExecution,
    threshold: torch.Tensor,
) -> NativeMultiClauseAnytimeCandidateIR:
    root = child.query.clauses[0].queue.trace.evaluations[0]
    threshold_value = float(threshold.item())
    return NativeMultiClauseAnytimeCandidateIR(
        original_clause_index=child.original_clause_index,
        threshold=threshold_value,
        root_lower=root.lower,
        root_upper=root.upper,
        root_lower_margin=root.lower - threshold_value,
        root_refinement_plan_hash=child.refinement.program.plan.stable_hash(),
        root_refinement_semantic_trace_hash=(
            intermediate_refinement_semantic_trace_hash(child.refinement)
        ),
        root_final_intermediate_bounds_hash=intermediate_bounds_hash(
            child.refinement.relu_pre
        ),
    )


def _decision_from_floor(
    plan: NativeMultiClauseAnytimePlanIR,
    floor: NativeObjectiveHardClauseEscalationExecution,
    thresholds: torch.Tensor,
) -> NativeMultiClauseAnytimeDecisionIR:
    trace = floor.trace
    candidates = tuple(
        _candidate_from_child(child, thresholds[index : index + 1])
        for index in trace.final_unresolved_clause_indices
        for child in (_accepted_child(floor, index),)
        if child is not None
    )
    ranked = tuple(
        item.original_clause_index
        for item in sorted(
            candidates,
            key=lambda item: (-item.root_lower_margin, item.original_clause_index),
        )
    )
    admitted = (
        trace.final_status == "unknown"
        and trace.completed_objective_clause_indices == tuple(range(plan.clause_count))
        and trace.final_unsafe_clause_index is None
        and bool(candidates)
    )
    decision = NativeMultiClauseAnytimeDecisionIR(
        plan_hash=plan.stable_hash(),
        floor_trace_hash=trace.semantic_signature_hash,
        floor_completed_original_clause_indices=(
            trace.completed_objective_clause_indices
        ),
        floor_status=trace.final_status,
        floor_verified_clause_indices=trace.final_verified_clause_indices,
        floor_unresolved_clause_indices=trace.final_unresolved_clause_indices,
        floor_unsafe_clause_index=trace.final_unsafe_clause_index,
        candidates=candidates,
        ranked_original_clause_indices=ranked,
        selected_original_clause_indices=(
            ranked[: plan.allocation_policy.max_selected_clauses] if admitted else ()
        ),
        reason=(
            "ranked_unresolved_candidates_selected"
            if admitted
            else "floor_not_eligible_for_multi_clause_anytime"
        ),
    )
    decision.validate_against(plan)
    return decision


def _aggregate_from_slices(
    plan: NativeMultiClauseAnytimePlanIR,
    decision: NativeMultiClauseAnytimeDecisionIR,
    slices: Tuple[NativeMultiClauseAnytimeSliceIR, ...],
) -> NativeMultiClauseAnytimeAggregateIR:
    outcomes = tuple(
        NativeMultiClauseAnytimeOutcomeIR(
            original_clause_index=item.original_clause_index,
            packed_queue_trace_hash=item.packed_queue_trace_hash or "",
            packed_verdict_trace_hash=item.packed_verdict_trace_hash or "",
            status=item.packed_status or "unknown",
        )
        for item in slices
        if item.packed_queue_trace_hash is not None
    )
    verified = set(decision.floor_verified_clause_indices)
    unresolved = list(decision.floor_unresolved_clause_indices)
    unsafe = decision.floor_unsafe_clause_index
    status = decision.floor_status
    for outcome in outcomes:
        if outcome.status == "verified":
            verified.add(outcome.original_clause_index)
            unresolved = [
                item for item in unresolved if item != outcome.original_clause_index
            ]
        elif outcome.status == "unsafe":
            status = "unsafe"
            unsafe = outcome.original_clause_index
            break
    if status != "unsafe":
        status = "verified" if len(verified) == plan.clause_count else "unknown"
        unsafe = None
    aggregate = NativeMultiClauseAnytimeAggregateIR(
        plan_hash=plan.stable_hash(),
        decision_hash=decision.stable_hash(plan),
        floor_trace_hash=decision.floor_trace_hash,
        slice_hashes=tuple(item.stable_hash(plan, decision) for item in slices),
        outcomes=outcomes,
        floor_status=decision.floor_status,
        floor_verified_clause_indices=decision.floor_verified_clause_indices,
        floor_unresolved_clause_indices=decision.floor_unresolved_clause_indices,
        floor_unsafe_clause_index=decision.floor_unsafe_clause_index,
        final_status=status,
        final_verified_clause_indices=tuple(sorted(verified)),
        final_unresolved_clause_indices=tuple(unresolved),
        final_unsafe_clause_index=unsafe,
    )
    aggregate.validate_against(plan, decision, slices)
    return aggregate


def _validate_packed_execution(
    item: NativeMultiClauseAnytimePackedExecution,
    position: int,
    program: NativeMultiClauseAnytimeProgram,
    floor: NativeObjectiveHardClauseEscalationExecution,
    decision: NativeMultiClauseAnytimeDecisionIR,
    module: BFTaskModule,
    input_spec: InputSpec,
    objectives: torch.Tensor,
    thresholds: torch.Tensor,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
) -> None:
    slice_ir = item.slice_ir
    slice_ir.validate_against(program.plan, decision)
    if slice_ir.priority_position != position:
        raise ValueError("multi-clause anytime slice priority position differs")
    ordinal = slice_ir.original_clause_index
    child = _accepted_child(floor, ordinal)
    if child is None:
        raise ValueError("multi-clause anytime selected floor source is absent")
    objective = objectives[:, ordinal : ordinal + 1, :].contiguous()
    threshold = thresholds[ordinal : ordinal + 1].contiguous()
    if item.packed_plan is not None:
        validate_native_objective_ancestral_sibling_pack_plan(
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
            raise ValueError("multi-clause anytime packed result is incomplete")
        if item.packed.plan != item.packed_plan:
            raise ValueError("multi-clause anytime packed plan differs")
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
        if (
            slice_ir.packed_plan_hash != item.packed_plan.stable_hash()
            or slice_ir.packed_queue_trace_hash != item.packed.queue.trace.stable_hash()
            or slice_ir.packed_verdict_trace_hash
            != item.verdict.trace.stable_hash(item.packed.queue.trace)
            or slice_ir.packed_status != item.verdict.trace.status
            or slice_ir.accepted_nodes != len(item.packed.queue.trace.evaluations)
            or slice_ir.sibling_group_count != len(item.packed.sibling_groups)
            or item.packed.trace.source_elapsed_ns
            < slice_ir.dispatch_started_elapsed_ns
            or item.packed.trace.whole_elapsed_ns < item.packed.trace.source_elapsed_ns
        ):
            raise ValueError("multi-clause anytime slice/packed binding differs")
    elif item.verdict is not None:
        raise ValueError("multi-clause anytime verdict lacks packed queue")
    elif any(
        value is not None
        for value in (
            slice_ir.packed_plan_hash,
            slice_ir.packed_queue_trace_hash,
            slice_ir.packed_verdict_trace_hash,
            slice_ir.packed_status,
        )
    ):
        raise ValueError("multi-clause anytime skipped slice retains result")


def _validate_action_bindings(execution: NativeMultiClauseAnytimeExecution) -> None:
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
                "packed_plan_compiled" if item.packed_plan is not None else reason,
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
    actual = execution.trace.actions
    if len(actual) != len(expected) or any(
        action.executed != executed
        or action.reason != reason
        or action.output_hash != _canonical_hash(output)
        for action, (executed, reason, output) in zip(actual, expected)
    ):
        raise ValueError("multi-clause anytime action/result binding differs")


def _action_trace(
    program: NativeMultiClauseAnytimeProgram,
    index: int,
    *,
    executed: bool,
    reason: str,
    elapsed_ns: int,
    output: object,
) -> NativeMultiClauseAnytimeActionTrace:
    action = program.schedule.actions[index]
    result = NativeMultiClauseAnytimeActionTrace(
        sequence=action.sequence,
        action_id=action.action_id,
        task_id=action.task_id,
        kind=action.kind,
        guard=action.guard,
        priority_position=action.priority_position,
        executed=executed,
        reason=reason,
        elapsed_ns=elapsed_ns,
        output_hash=_canonical_hash(output),
    )
    result.validate()
    return result


def _skipped_action(
    program: NativeMultiClauseAnytimeProgram,
    index: int,
    reason: str,
) -> NativeMultiClauseAnytimeActionTrace:
    return _action_trace(
        program,
        index,
        executed=False,
        reason=reason,
        elapsed_ns=0,
        output={"skipped": reason},
    )


def execute_native_multi_clause_anytime_program(
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
) -> NativeMultiClauseAnytimeExecution:
    """Execute ranked clauses with dynamically recomputed equal remaining slices."""

    if not query_id:
        raise ValueError("multi-clause anytime query ID must be non-empty")
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

    packed_executions: list[NativeMultiClauseAnytimePackedExecution] = []
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
        packed_plan: Optional[NativeObjectiveAncestralSiblingPackPlanIR] = None
        packed: Optional[NativeObjectiveAncestralSiblingPackExecution] = None
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
            packed_plan = compile_native_objective_ancestral_sibling_pack_plan(
                module,
                input_spec,
                linear_spec_C=objective,
                threshold=threshold,
                root_refinement=child.refinement,
                optimizer_policy=optimizer_policy,
                plan_id=(
                    f"{query_id}:clause:{ordinal:04d}:priority:{position}:packed-nrir34"
                ),
            )
            compile_finished_ns = clock_ns()
            actions.append(
                _action_trace(
                    program,
                    compile_action,
                    executed=True,
                    reason="packed_plan_compiled",
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
                    packed = execute_native_objective_ancestral_sibling_pack_queue(
                        packed_plan,
                        module,
                        input_spec,
                        linear_spec_C=objective,
                        threshold=threshold,
                        root_refinement=child.refinement,
                        optimizer_policy=optimizer_policy,
                        query_id=(
                            f"{query_id}:clause:{ordinal:04d}:priority:{position}"
                        ),
                        whole_query_started_ns=started_ns,
                        clock_ns=slice_clock,
                    )
                except RuntimeError as error:
                    if str(error) != "sibling-pack deadline expired at root":
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
            packed_status=(verdict.trace.status if packed_result and verdict else None),
            accepted_nodes=(
                len(packed.queue.trace.evaluations) if packed_result and packed else 0
            ),
            sibling_group_count=(
                len(packed.sibling_groups) if packed_result and packed else 0
            ),
            cutoff_signaled=cutoff_signaled,
            reason=reason,
        )
        slice_ir.validate_against(program.plan, decision)
        packed_executions.append(
            NativeMultiClauseAnytimePackedExecution(
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
    execution = NativeMultiClauseAnytimeExecution(
        program=program,
        floor=floor,
        decision=decision,
        packed_executions=tuple(packed_executions),
        aggregate=aggregate,
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
    "NativeMultiClauseAnytimeActionTrace",
    "NativeMultiClauseAnytimeExecution",
    "NativeMultiClauseAnytimePackedExecution",
    "NativeMultiClauseAnytimeProgram",
    "NativeMultiClauseAnytimeTrace",
    "compile_native_multi_clause_anytime_program",
    "execute_native_multi_clause_anytime_program",
]
