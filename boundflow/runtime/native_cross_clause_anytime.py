"""Runtime for an NRIR-31 floor plus optional NRIR-34 anytime escalation."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import time
from typing import Callable, Optional, Tuple

import torch

from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.cross_clause_anytime import (
    NativeCrossClauseAnytimeAggregateIR,
    NativeCrossClauseAnytimeDecisionIR,
    NativeCrossClauseAnytimePlanIR,
    NativeCrossClauseAnytimeScheduleIR,
    NativeCrossClauseAnytimeTaskIRModule,
    NativeCrossClauseAnytimeTaskKind,
    lower_native_cross_clause_anytime_ir,
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

NATIVE_CROSS_CLAUSE_ANYTIME_ACTION_TRACE_SCHEMA_VERSION = (
    "boundflow.native-cross-clause-anytime-action-trace/v1"
)
NATIVE_CROSS_CLAUSE_ANYTIME_TRACE_SCHEMA_VERSION = (
    "boundflow.native-cross-clause-anytime-trace/v1"
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
class NativeCrossClauseAnytimeProgram:
    """Frozen floor program and first-class cross-clause control IR."""

    floor_program: NativeObjectiveHardClauseEscalationProgram
    plan: NativeCrossClauseAnytimePlanIR
    task_ir: NativeCrossClauseAnytimeTaskIRModule
    schedule: NativeCrossClauseAnytimeScheduleIR

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
            raise ValueError("cross-clause anytime program/query binding differs")

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
class NativeCrossClauseAnytimeActionTrace:
    """One executed or guarded-skip cross-clause schedule action."""

    sequence: int
    action_id: str
    task_id: str
    kind: NativeCrossClauseAnytimeTaskKind
    guard: str
    executed: bool
    reason: str
    elapsed_ns: int
    output_hash: str
    schema_version: str = NATIVE_CROSS_CLAUSE_ANYTIME_ACTION_TRACE_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version
            != NATIVE_CROSS_CLAUSE_ANYTIME_ACTION_TRACE_SCHEMA_VERSION
            or self.sequence < 0
            or not self.action_id
            or not self.task_id
            or self.guard not in {"always", "decision_admitted_before_deadline"}
            or not self.reason
            or self.elapsed_ns < 0
            or not _is_sha256(self.output_hash)
            or (not self.executed and self.elapsed_ns != 0)
        ):
            raise ValueError("cross-clause anytime action trace is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "guard": self.guard,
            "executed": self.executed,
            "reason": self.reason,
            "elapsed_ns": self.elapsed_ns,
            "output_hash": self.output_hash,
        }


@dataclass(frozen=True)
class NativeCrossClauseAnytimeTrace:
    """Replay identity for the floor, admission, packed work, and aggregate."""

    query_id: str
    plan_hash: str
    task_ir_hash: str
    schedule_hash: str
    floor_trace_hash: str
    decision_hash: str
    packed_plan_hash: Optional[str]
    packed_trace_hash: Optional[str]
    packed_verdict_hash: Optional[str]
    aggregate_hash: str
    actions: Tuple[NativeCrossClauseAnytimeActionTrace, ...]
    fallback_reason: str
    elapsed_ns: int
    deadline_ns: int
    semantic_signature_hash: str
    performance_claimed: bool = False
    schema_version: str = NATIVE_CROSS_CLAUSE_ANYTIME_TRACE_SCHEMA_VERSION

    def semantic_dict(self) -> dict[str, object]:
        return {
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "schedule_hash": self.schedule_hash,
            "floor_trace_hash": self.floor_trace_hash,
            "decision_hash": self.decision_hash,
            "packed_plan_hash": self.packed_plan_hash,
            "packed_trace_hash": self.packed_trace_hash,
            "packed_verdict_hash": self.packed_verdict_hash,
            "aggregate_hash": self.aggregate_hash,
            "actions": [action.to_dict() for action in self.actions],
            "fallback_reason": self.fallback_reason,
        }

    def validate_against(
        self,
        program: NativeCrossClauseAnytimeProgram,
        decision: NativeCrossClauseAnytimeDecisionIR,
        aggregate: NativeCrossClauseAnytimeAggregateIR,
    ) -> None:
        program.schedule.validate_against(program.task_ir)
        decision.validate_against(program.plan)
        aggregate.validate_against(program.plan, decision)
        packed_values = (
            self.packed_plan_hash,
            self.packed_trace_hash,
            self.packed_verdict_hash,
        )
        if (
            self.schema_version != NATIVE_CROSS_CLAUSE_ANYTIME_TRACE_SCHEMA_VERSION
            or not self.query_id
            or self.plan_hash != program.plan.stable_hash()
            or self.task_ir_hash != program.task_ir.stable_hash()
            or self.schedule_hash != program.schedule.stable_hash(program.task_ir)
            or self.floor_trace_hash != decision.floor_trace_hash
            or self.decision_hash != decision.stable_hash(program.plan)
            or self.aggregate_hash != aggregate.stable_hash(program.plan, decision)
            or len(self.actions) != len(program.schedule.actions)
            or any(
                value is not None and not _is_sha256(value) for value in packed_values
            )
            or (self.packed_trace_hash is None) != (self.packed_verdict_hash is None)
            or self.packed_trace_hash is not None
            and self.packed_plan_hash is None
            or not decision.admitted
            and any(value is not None for value in packed_values)
            or not self.fallback_reason
            or self.elapsed_ns < 0
            or self.deadline_ns != program.plan.whole_query_timeout_ns
            or self.performance_claimed is not False
            or self.semantic_signature_hash != _canonical_hash(self.semantic_dict())
        ):
            raise ValueError("cross-clause anytime trace differs")
        for expected, actual in zip(program.schedule.actions, self.actions):
            actual.validate()
            if (
                expected.sequence != actual.sequence
                or expected.action_id != actual.action_id
                or expected.task_id != actual.task_id
                or expected.kind != actual.kind
                or expected.guard != actual.guard
            ):
                raise ValueError("cross-clause anytime runtime/Schedule differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            **self.semantic_dict(),
            "elapsed_ns": self.elapsed_ns,
            "deadline_ns": self.deadline_ns,
            "deadline_enforcement": "single_global_cooperative_stage_boundaries",
            "semantic_signature_hash": self.semantic_signature_hash,
            "performance_claimed": self.performance_claimed,
        }


@dataclass(frozen=True)
class NativeCrossClauseAnytimeExecution:
    """Typed floor, optional packed escalation, and monotone final aggregate."""

    program: NativeCrossClauseAnytimeProgram
    floor: NativeObjectiveHardClauseEscalationExecution
    decision: NativeCrossClauseAnytimeDecisionIR
    packed_plan: Optional[NativeObjectiveAncestralSiblingPackPlanIR]
    packed: Optional[NativeObjectiveAncestralSiblingPackExecution]
    packed_verdict: Optional[NativePropertyVerdictExecution]
    aggregate: NativeCrossClauseAnytimeAggregateIR
    trace: NativeCrossClauseAnytimeTrace

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
            raise ValueError("cross-clause anytime floor program differs")
        self.decision.validate_against(self.program.plan)
        _validate_decision_source(self.decision, self.floor, self.program.plan)
        optional_values = (self.packed_plan, self.packed, self.packed_verdict)
        if not self.decision.admitted and any(
            value is not None for value in optional_values
        ):
            raise ValueError(
                "cross-clause anytime skipped decision executed packed work"
            )
        if self.packed is not None:
            if self.packed_plan is None or self.packed_verdict is None:
                raise ValueError("cross-clause anytime packed result is incomplete")
            child = _accepted_clause_source(self.floor, self.program.plan)
            assert child is not None
            objective = objectives[:, 0:1, :].contiguous()
            threshold = thresholds[0:1].contiguous()
            if self.packed.plan != self.packed_plan:
                raise ValueError("cross-clause anytime packed plan differs")
            self.packed.validate_against(
                module,
                input_spec,
                linear_spec_C=objective,
                threshold=threshold,
                root_refinement=child.refinement,
                optimizer_policy=optimizer_policy,
            )
            if (
                self.packed.trace.source_elapsed_ns < self.floor.trace.elapsed_ns
                or self.packed.trace.whole_elapsed_ns
                < self.packed.trace.source_elapsed_ns
                or self.trace.elapsed_ns < self.packed.trace.source_elapsed_ns
            ):
                raise ValueError("cross-clause anytime packed queue reset global time")
            self.packed_verdict.validate_against(
                module,
                input_spec,
                linear_spec_C=objective,
                queue_execution=self.packed.queue,
            )
        elif self.packed_verdict is not None:
            raise ValueError("cross-clause anytime verdict lacks packed queue")
        expected_packed_plan_hash = (
            None if self.packed_plan is None else self.packed_plan.stable_hash()
        )
        expected_packed_trace_hash = (
            None if self.packed is None else self.packed.trace.semantic_signature_hash
        )
        expected_packed_verdict_hash = (
            None
            if self.packed is None or self.packed_verdict is None
            else self.packed_verdict.trace.stable_hash(self.packed.queue.trace)
        )
        if (
            self.trace.packed_plan_hash != expected_packed_plan_hash
            or self.trace.packed_trace_hash != expected_packed_trace_hash
            or self.trace.packed_verdict_hash != expected_packed_verdict_hash
        ):
            raise ValueError("cross-clause anytime trace/packed result binding differs")
        expected = _aggregate_from_results(
            self.program.plan,
            self.decision,
            self.packed,
            self.packed_verdict,
        )
        if self.aggregate != expected:
            raise ValueError("cross-clause anytime aggregate/result differs")
        self.trace.validate_against(self.program, self.decision, self.aggregate)

    def to_dict(self) -> dict[str, object]:
        return {
            "program": self.program.to_dict(),
            "floor": self.floor.trace.to_dict(),
            "decision": self.decision.to_dict(self.program.plan),
            "packed_plan": (
                None if self.packed_plan is None else self.packed_plan.to_dict()
            ),
            "packed": None if self.packed is None else self.packed.to_dict(),
            "packed_verdict": (
                None
                if self.packed_verdict is None
                else self.packed_verdict.trace.to_dict()
            ),
            "aggregate": self.aggregate.to_dict(self.program.plan, self.decision),
            "trace": self.trace.to_dict(),
        }


def compile_native_cross_clause_anytime_program(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
    plan_id: str,
    search_policy: NativeProjectedGradientSearchPolicy,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
) -> NativeCrossClauseAnytimeProgram:
    """Compile the exact NRIR-31 floor and guarded NRIR-34 escalation stages."""

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
    plan = NativeCrossClauseAnytimePlanIR(
        plan_id=plan_id,
        floor_plan_hash=floor.plan.stable_hash(),
        floor_task_ir_hash=floor.task_ir.stable_hash(),
        floor_schedule_hash=floor.schedule.stable_hash(floor.task_ir),
        objective_matrix_hash=tensor_content_hash(objectives),
        thresholds_hash=tensor_content_hash(thresholds),
        search_policy_hash=search_policy.stable_hash(),
        optimizer_policy_hash=optimizer_policy.stable_hash(),
        clause_count=int(objectives.shape[1]),
    )
    task_ir, schedule = lower_native_cross_clause_anytime_ir(plan)
    program = NativeCrossClauseAnytimeProgram(floor, plan, task_ir, schedule)
    program.validate_against(
        module,
        input_spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    return program


def _accepted_clause_source(
    floor: NativeObjectiveHardClauseEscalationExecution,
    plan: NativeCrossClauseAnytimePlanIR,
) -> Optional[NativeObjectiveHardClauseEscalationClauseExecution]:
    return next(
        (
            child
            for child in floor.clause_executions
            if child.original_clause_index == plan.packed_original_clause_index
            and child.accepted_before_deadline
            and bool(child.query.clauses)
        ),
        None,
    )


def _decision_from_floor(
    plan: NativeCrossClauseAnytimePlanIR,
    floor: NativeObjectiveHardClauseEscalationExecution,
    *,
    deadline_at_ns: int,
    observed_ns: int,
) -> NativeCrossClauseAnytimeDecisionIR:
    trace = floor.trace
    child = _accepted_clause_source(floor, plan)
    reason = "floor_complete_unresolved_clause_admitted"
    admitted = True
    if trace.final_status != "unknown":
        admitted, reason = False, f"floor_already_{trace.final_status}"
    elif trace.completed_objective_clause_indices != tuple(range(plan.clause_count)):
        admitted, reason = False, "floor_original_ordinal_accounting_incomplete"
    elif trace.final_unsafe_clause_index is not None:
        admitted, reason = False, "floor_unsafe_witness_present"
    elif plan.packed_original_clause_index not in trace.final_unresolved_clause_indices:
        admitted, reason = False, "packed_clause_not_unresolved"
    elif child is None:
        admitted, reason = False, "floor_clause_source_not_accepted"
    elif observed_ns >= deadline_at_ns:
        admitted, reason = False, "whole_query_deadline_exhausted"
    decision = NativeCrossClauseAnytimeDecisionIR(
        plan_hash=plan.stable_hash(),
        floor_trace_hash=trace.semantic_signature_hash,
        floor_completed_original_clause_indices=(
            trace.completed_objective_clause_indices
        ),
        floor_status=trace.final_status,
        floor_verified_clause_indices=trace.final_verified_clause_indices,
        floor_unresolved_clause_indices=trace.final_unresolved_clause_indices,
        floor_unsafe_clause_index=trace.final_unsafe_clause_index,
        admitted_original_clause_index=(
            plan.packed_original_clause_index if admitted else None
        ),
        root_refinement_plan_hash=(
            child.refinement.program.plan.stable_hash()
            if admitted and child is not None
            else None
        ),
        root_refinement_semantic_trace_hash=(
            intermediate_refinement_semantic_trace_hash(child.refinement)
            if admitted and child is not None
            else None
        ),
        root_final_intermediate_bounds_hash=(
            intermediate_bounds_hash(child.refinement.relu_pre)
            if admitted and child is not None
            else None
        ),
        admitted=admitted,
        reason=reason,
    )
    decision.validate_against(plan)
    return decision


def _validate_decision_source(
    decision: NativeCrossClauseAnytimeDecisionIR,
    floor: NativeObjectiveHardClauseEscalationExecution,
    plan: NativeCrossClauseAnytimePlanIR,
) -> None:
    trace = floor.trace
    if (
        decision.floor_trace_hash != trace.semantic_signature_hash
        or decision.floor_completed_original_clause_indices
        != trace.completed_objective_clause_indices
        or decision.floor_status != trace.final_status
        or decision.floor_verified_clause_indices != trace.final_verified_clause_indices
        or decision.floor_unresolved_clause_indices
        != trace.final_unresolved_clause_indices
        or decision.floor_unsafe_clause_index != trace.final_unsafe_clause_index
    ):
        raise ValueError("cross-clause anytime Decision/floor binding differs")
    if not decision.admitted:
        return
    child = _accepted_clause_source(floor, plan)
    if child is None or (
        decision.root_refinement_plan_hash
        != child.refinement.program.plan.stable_hash()
        or decision.root_refinement_semantic_trace_hash
        != intermediate_refinement_semantic_trace_hash(child.refinement)
        or decision.root_final_intermediate_bounds_hash
        != intermediate_bounds_hash(child.refinement.relu_pre)
    ):
        raise ValueError("cross-clause anytime admitted source differs")


def _aggregate_from_results(
    plan: NativeCrossClauseAnytimePlanIR,
    decision: NativeCrossClauseAnytimeDecisionIR,
    packed: Optional[NativeObjectiveAncestralSiblingPackExecution],
    verdict: Optional[NativePropertyVerdictExecution],
) -> NativeCrossClauseAnytimeAggregateIR:
    packed_status = None if verdict is None else verdict.trace.status
    packed_trace_hash = None if packed is None else packed.queue.trace.stable_hash()
    final_status = decision.floor_status
    verified = decision.floor_verified_clause_indices
    unresolved = decision.floor_unresolved_clause_indices
    unsafe = decision.floor_unsafe_clause_index
    if packed_status == "verified":
        verified = tuple(sorted({*verified, plan.packed_original_clause_index}))
        unresolved = tuple(
            item for item in unresolved if item != plan.packed_original_clause_index
        )
        final_status = "verified" if len(verified) == plan.clause_count else "unknown"
        unsafe = None
    elif packed_status == "unsafe":
        final_status = "unsafe"
        unsafe = plan.packed_original_clause_index
    aggregate = NativeCrossClauseAnytimeAggregateIR(
        plan_hash=plan.stable_hash(),
        decision_hash=decision.stable_hash(plan),
        floor_trace_hash=decision.floor_trace_hash,
        packed_queue_trace_hash=packed_trace_hash,
        packed_status=packed_status,
        floor_status=decision.floor_status,
        floor_verified_clause_indices=decision.floor_verified_clause_indices,
        floor_unresolved_clause_indices=decision.floor_unresolved_clause_indices,
        floor_unsafe_clause_index=decision.floor_unsafe_clause_index,
        final_status=final_status,
        final_verified_clause_indices=verified,
        final_unresolved_clause_indices=unresolved,
        final_unsafe_clause_index=unsafe,
    )
    aggregate.validate_against(plan, decision)
    return aggregate


def _action_trace(
    program: NativeCrossClauseAnytimeProgram,
    index: int,
    *,
    executed: bool,
    reason: str,
    elapsed_ns: int,
    output: object,
) -> NativeCrossClauseAnytimeActionTrace:
    action = program.schedule.actions[index]
    result = NativeCrossClauseAnytimeActionTrace(
        sequence=action.sequence,
        action_id=action.action_id,
        task_id=action.task_id,
        kind=action.kind,
        guard=action.guard,
        executed=executed,
        reason=reason,
        elapsed_ns=elapsed_ns,
        output_hash=_canonical_hash(output),
    )
    result.validate()
    return result


def execute_native_cross_clause_anytime_program(
    program: NativeCrossClauseAnytimeProgram,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
    query_id: str,
    search_policy: NativeProjectedGradientSearchPolicy,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    clock_ns: ClockNs = time.monotonic_ns,
) -> NativeCrossClauseAnytimeExecution:
    """Execute the floor first, then spend only its remaining global budget."""

    if not query_id:
        raise ValueError("cross-clause anytime query ID must be non-empty")
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
    deadline_at_ns = started_ns + program.plan.whole_query_timeout_ns
    actions: list[NativeCrossClauseAnytimeActionTrace] = []

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
    decision = _decision_from_floor(
        program.plan,
        floor,
        deadline_at_ns=deadline_at_ns,
        observed_ns=action_started_ns,
    )
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

    packed_plan: Optional[NativeObjectiveAncestralSiblingPackPlanIR] = None
    packed: Optional[NativeObjectiveAncestralSiblingPackExecution] = None
    verdict: Optional[NativePropertyVerdictExecution] = None
    if decision.admitted and clock_ns() < deadline_at_ns:
        child = _accepted_clause_source(floor, program.plan)
        assert child is not None
        objective = objectives[:, 0:1, :].contiguous()
        threshold = thresholds[0:1].contiguous()
        action_started_ns = clock_ns()
        packed_plan = compile_native_objective_ancestral_sibling_pack_plan(
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=child.refinement,
            optimizer_policy=optimizer_policy,
            plan_id=f"{query_id}:clause:0000:packed-nrir34",
        )
        action_finished_ns = clock_ns()
        actions.append(
            _action_trace(
                program,
                2,
                executed=True,
                reason="packed_plan_compiled",
                elapsed_ns=max(0, action_finished_ns - action_started_ns),
                output=packed_plan.stable_hash(),
            )
        )
        if action_finished_ns < deadline_at_ns:
            action_started_ns = clock_ns()
            try:
                packed = execute_native_objective_ancestral_sibling_pack_queue(
                    packed_plan,
                    module,
                    input_spec,
                    linear_spec_C=objective,
                    threshold=threshold,
                    root_refinement=child.refinement,
                    optimizer_policy=optimizer_policy,
                    query_id=f"{query_id}:clause:0000:packed-nrir34",
                    whole_query_started_ns=started_ns,
                    clock_ns=clock_ns,
                )
            except RuntimeError as error:
                if str(error) != "sibling-pack deadline expired at root":
                    raise
            action_finished_ns = clock_ns()
            if packed is not None:
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
                reason = f"packed_queue_{verdict.trace.status}"
                output: object = packed.trace.semantic_signature_hash
            else:
                reason = "deadline_during_packed_root"
                output = {"skipped": reason}
            actions.append(
                _action_trace(
                    program,
                    3,
                    executed=packed is not None,
                    reason=reason,
                    elapsed_ns=(
                        max(0, action_finished_ns - action_started_ns)
                        if packed is not None
                        else 0
                    ),
                    output=output,
                )
            )
        else:
            actions.append(
                _action_trace(
                    program,
                    3,
                    executed=False,
                    reason="deadline_after_packed_compile",
                    elapsed_ns=0,
                    output={"skipped": "deadline_after_packed_compile"},
                )
            )
    else:
        reason = (
            decision.reason
            if not decision.admitted
            else "deadline_before_packed_compile"
        )
        for index in (2, 3):
            actions.append(
                _action_trace(
                    program,
                    index,
                    executed=False,
                    reason=reason,
                    elapsed_ns=0,
                    output={"skipped": reason},
                )
            )

    aggregate = _aggregate_from_results(program.plan, decision, packed, verdict)
    actions.append(
        _action_trace(
            program,
            4,
            executed=True,
            reason="original_ordinals_aggregated_monotonically",
            elapsed_ns=0,
            output=aggregate.to_dict(program.plan, decision),
        )
    )
    actions.append(
        _action_trace(
            program,
            5,
            executed=True,
            reason="result_emitted",
            elapsed_ns=0,
            output=aggregate.stable_hash(program.plan, decision),
        )
    )
    finished_ns = clock_ns()
    if verdict is not None:
        fallback_reason = (
            "none" if verdict.trace.status != "unknown" else verdict.trace.reason
        )
    elif packed_plan is not None:
        fallback_reason = "deadline_before_packed_result"
    else:
        fallback_reason = decision.reason
    trace = NativeCrossClauseAnytimeTrace(
        query_id=query_id,
        plan_hash=program.plan.stable_hash(),
        task_ir_hash=program.task_ir.stable_hash(),
        schedule_hash=program.schedule.stable_hash(program.task_ir),
        floor_trace_hash=floor.trace.semantic_signature_hash,
        decision_hash=decision.stable_hash(program.plan),
        packed_plan_hash=(None if packed_plan is None else packed_plan.stable_hash()),
        packed_trace_hash=(
            None if packed is None else packed.trace.semantic_signature_hash
        ),
        packed_verdict_hash=(
            None
            if verdict is None or packed is None
            else verdict.trace.stable_hash(packed.queue.trace)
        ),
        aggregate_hash=aggregate.stable_hash(program.plan, decision),
        actions=tuple(actions),
        fallback_reason=fallback_reason,
        elapsed_ns=max(0, finished_ns - started_ns),
        deadline_ns=program.plan.whole_query_timeout_ns,
        semantic_signature_hash="",
    )
    trace = replace(
        trace, semantic_signature_hash=_canonical_hash(trace.semantic_dict())
    )
    execution = NativeCrossClauseAnytimeExecution(
        program=program,
        floor=floor,
        decision=decision,
        packed_plan=packed_plan,
        packed=packed,
        packed_verdict=verdict,
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
    "NativeCrossClauseAnytimeActionTrace",
    "NativeCrossClauseAnytimeExecution",
    "NativeCrossClauseAnytimeProgram",
    "NativeCrossClauseAnytimeTrace",
    "compile_native_cross_clause_anytime_program",
    "execute_native_cross_clause_anytime_program",
]
