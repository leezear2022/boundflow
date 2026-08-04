"""Runtime for shared-source, per-clause objective-directed escalation."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=protected-access,duplicate-code
# pylint: disable=too-many-lines

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Callable, Literal, Optional, Tuple

import torch

from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.bound import IntermediateBoundSource
from ..ir.hard_clause_escalation import NativeHardClauseEscalationDecisionIR
from ..ir.objective_hard_clause_escalation import (
    NativeObjectiveHardClauseEscalationPlanIR,
    NativeObjectiveHardClauseEscalationScheduleIR,
    NativeObjectiveHardClauseEscalationTaskIRModule,
    ObjectiveHardClauseEscalationTaskKind,
    lower_native_objective_hard_clause_escalation_ir,
)
from ..ir.refinement import NativeIntermediateRefinementPolicyIR
from ..ir.task import BFTaskModule
from .complete_verifier_query import (
    CompleteVerifierQueryPolicy,
    _normalize_objective_matrix,
)
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_candidate_search import NativeProjectedGradientSearchPolicy
from .native_hard_clause_escalation import (
    NativeHardClauseEscalationProgram,
    _decision_from_baseline,
    compile_native_hard_clause_escalation_program,
)
from .native_intermediate_refinement import (
    NativeIntermediateRefinementExecution,
    NativeIntermediateRefinementProgram,
    compile_native_intermediate_refinement_program,
    execute_native_intermediate_refinement_program,
    intermediate_refinement_semantic_trace_hash,
)
from .native_parametric_production_complete_query import (
    NativeParametricCompleteVerifierQueryExecution,
    execute_native_parametric_production_complete_verifier_query,
)
from .native_relu_split_bab_runtime import NativeReluSplitBabConfig
from .task_executor import InputSpec

NATIVE_OBJECTIVE_HARD_CLAUSE_ESCALATION_ACTION_TRACE_SCHEMA_VERSION = (
    "boundflow.native-objective-hard-clause-escalation-action-trace/v1"
)
NATIVE_OBJECTIVE_HARD_CLAUSE_ESCALATION_CLAUSE_TRACE_SCHEMA_VERSION = (
    "boundflow.native-objective-hard-clause-escalation-clause-trace/v1"
)
NATIVE_OBJECTIVE_HARD_CLAUSE_ESCALATION_TRACE_SCHEMA_VERSION = (
    "boundflow.native-objective-hard-clause-escalation-trace/v1"
)
ClockNs = Callable[[], int]
FinalStatus = Literal["verified", "unsafe", "unknown"]


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
class NativeObjectiveHardClauseEscalationProgram:
    """Frozen NRIR-30 program plus the additive objective stage stack."""

    base_program: NativeHardClauseEscalationProgram
    plan: NativeObjectiveHardClauseEscalationPlanIR
    task_ir: NativeObjectiveHardClauseEscalationTaskIRModule
    schedule: NativeObjectiveHardClauseEscalationScheduleIR

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
        self.base_program.validate_against(
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
            thresholds=thresholds,
            search_policy=search_policy,
            optimizer_policy=optimizer_policy,
        )
        self.plan.validate()
        self.schedule.validate_against(self.task_ir)
        if (
            self.plan.base_plan_hash != self.base_program.plan.stable_hash()
            or self.plan.base_task_ir_hash != self.base_program.task_ir.stable_hash()
            or self.plan.base_schedule_hash
            != self.base_program.schedule.stable_hash(self.base_program.task_ir)
            or self.plan.clause_count != self.base_program.plan.clause_count
            or self.task_ir.plan_hash != self.plan.stable_hash()
        ):
            raise ValueError("objective hard-clause escalation base binding differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "base_program": self.base_program.to_dict(),
            "plan": self.plan.to_dict(),
            "plan_hash": self.plan.stable_hash(),
            "task_ir": self.task_ir.to_dict(),
            "task_ir_hash": self.task_ir.stable_hash(),
            "schedule": self.schedule.to_dict(),
            "schedule_hash": self.schedule.stable_hash(self.task_ir),
        }


@dataclass(frozen=True)
class NativeObjectiveHardClauseEscalationActionTrace:
    """One executed or guarded-skip schedule action."""

    sequence: int
    action_id: str
    task_id: str
    kind: ObjectiveHardClauseEscalationTaskKind
    guard: str
    original_clause_index: Optional[int]
    executed: bool
    reason: str
    elapsed_ns: int
    output_hash: str
    schema_version: str = (
        NATIVE_OBJECTIVE_HARD_CLAUSE_ESCALATION_ACTION_TRACE_SCHEMA_VERSION
    )

    def validate(self) -> None:
        if (
            self.schema_version
            != NATIVE_OBJECTIVE_HARD_CLAUSE_ESCALATION_ACTION_TRACE_SCHEMA_VERSION
            or self.sequence < 0
            or not self.action_id
            or not self.task_id
            or self.guard
            not in {"always", "admitted_clauses_nonempty", "clause_is_admitted"}
            or not self.reason
            or self.elapsed_ns < 0
            or not _is_sha256(self.output_hash)
            or (not self.executed and self.elapsed_ns != 0)
        ):
            raise ValueError("objective hard-clause action trace is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "guard": self.guard,
            "original_clause_index": self.original_clause_index,
            "executed": self.executed,
            "reason": self.reason,
            "elapsed_ns": self.elapsed_ns,
            "output_hash": self.output_hash,
        }


@dataclass(frozen=True)
class NativeObjectiveHardClauseEscalationClauseExecution:
    """Typed objective refinement and scalar query for one original clause."""

    original_clause_index: int
    refinement_program: NativeIntermediateRefinementProgram
    refinement: NativeIntermediateRefinementExecution
    query: NativeParametricCompleteVerifierQueryExecution
    accepted_before_deadline: bool

    def validate_against(
        self,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        objective: torch.Tensor,
        threshold: torch.Tensor,
        shared_refinement: NativeIntermediateRefinementExecution,
    ) -> None:
        self.refinement_program.validate(module, input_spec)
        self.refinement.validate(module, input_spec)
        self.query.validate_against(module, input_spec, linear_spec_C=objective)
        if (
            self.refinement.program != self.refinement_program
            or self.refinement_program.plan.objective_hash
            != tensor_content_hash(objective)
            or self.refinement_program.plan.source_refinement_plan_hash
            != shared_refinement.program.plan.stable_hash()
            or self.refinement_program.plan.source_refinement_semantic_trace_hash
            != intermediate_refinement_semantic_trace_hash(shared_refinement)
            or self.query.trace.thresholds != (float(threshold.item()),)
            or bool(self.query.trace.pending_clause_indices)
            and self.accepted_before_deadline
        ):
            raise ValueError("objective hard-clause child binding differs")


@dataclass(frozen=True)
class NativeObjectiveHardClauseEscalationClauseTrace:
    """Deterministic source lineage and scalar-query result."""

    original_clause_index: int
    objective_hash: str
    source_refinement_plan_hash: str
    source_refinement_semantic_trace_hash: str
    refinement_plan_hash: str
    refinement_semantic_trace_hash: str
    query_trace_hash: str
    accepted_before_deadline: bool
    status: FinalStatus
    root_lower: Optional[float]
    root_upper: Optional[float]
    schema_version: str = (
        NATIVE_OBJECTIVE_HARD_CLAUSE_ESCALATION_CLAUSE_TRACE_SCHEMA_VERSION
    )

    def validate(self, *, clause_count: int) -> None:
        hashes = (
            self.objective_hash,
            self.source_refinement_plan_hash,
            self.source_refinement_semantic_trace_hash,
            self.refinement_plan_hash,
            self.refinement_semantic_trace_hash,
            self.query_trace_hash,
        )
        if (
            self.schema_version
            != NATIVE_OBJECTIVE_HARD_CLAUSE_ESCALATION_CLAUSE_TRACE_SCHEMA_VERSION
            or not 0 <= self.original_clause_index < clause_count
            or any(not _is_sha256(value) for value in hashes)
            or (self.root_lower is None) != (self.root_upper is None)
            or (self.accepted_before_deadline and self.root_lower is None)
        ):
            raise ValueError("objective hard-clause clause trace is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "original_clause_index": self.original_clause_index,
            "objective_hash": self.objective_hash,
            "source_refinement_plan_hash": self.source_refinement_plan_hash,
            "source_refinement_semantic_trace_hash": (
                self.source_refinement_semantic_trace_hash
            ),
            "refinement_plan_hash": self.refinement_plan_hash,
            "refinement_semantic_trace_hash": self.refinement_semantic_trace_hash,
            "query_trace_hash": self.query_trace_hash,
            "accepted_before_deadline": self.accepted_before_deadline,
            "status": self.status,
            "root_lower": self.root_lower,
            "root_upper": self.root_upper,
        }


def _clause_trace(
    child: NativeObjectiveHardClauseEscalationClauseExecution,
) -> NativeObjectiveHardClauseEscalationClauseTrace:
    query = child.query
    if query.clauses:
        status: FinalStatus = query.clauses[0].trace.status
        root = query.clauses[0].queue.trace.evaluations[0]
        root_lower: Optional[float] = root.lower
        root_upper: Optional[float] = root.upper
    else:
        status = "unknown"
        root_lower = None
        root_upper = None
    program = child.refinement_program
    return NativeObjectiveHardClauseEscalationClauseTrace(
        original_clause_index=child.original_clause_index,
        objective_hash=program.plan.objective_hash or "",
        source_refinement_plan_hash=program.plan.source_refinement_plan_hash or "",
        source_refinement_semantic_trace_hash=(
            program.plan.source_refinement_semantic_trace_hash or ""
        ),
        refinement_plan_hash=program.plan.stable_hash(),
        refinement_semantic_trace_hash=intermediate_refinement_semantic_trace_hash(
            child.refinement
        ),
        query_trace_hash=query.trace.stable_hash(),
        accepted_before_deadline=child.accepted_before_deadline,
        status=status,
        root_lower=root_lower,
        root_upper=root_upper,
    )


def _aggregate_semantics(
    clause_count: int,
    decision: NativeHardClauseEscalationDecisionIR,
    clauses: Tuple[NativeObjectiveHardClauseEscalationClauseExecution, ...],
) -> dict[str, object]:
    statuses = {index: "unknown" for index in range(clause_count)}
    for index in decision.baseline_verified_clause_indices:
        statuses[index] = "verified"
    if decision.baseline_unsafe_clause_index is not None:
        statuses[decision.baseline_unsafe_clause_index] = "unsafe"
    completed: list[int] = []
    for child in clauses:
        if not child.accepted_before_deadline or not child.query.clauses:
            continue
        completed.append(child.original_clause_index)
        statuses[child.original_clause_index] = child.query.clauses[0].trace.status
    verified = tuple(
        index for index in range(clause_count) if statuses[index] == "verified"
    )
    unresolved = tuple(
        index for index in range(clause_count) if statuses[index] == "unknown"
    )
    unsafe_values = tuple(
        index for index in range(clause_count) if statuses[index] == "unsafe"
    )
    unsafe = None if not unsafe_values else unsafe_values[0]
    if unsafe is not None:
        final_status: FinalStatus = "unsafe"
    elif len(verified) == clause_count:
        final_status = "verified"
    else:
        final_status = "unknown"
    return {
        "completed_objective_clause_indices": tuple(completed),
        "final_status": final_status,
        "final_verified_clause_indices": verified,
        "final_unresolved_clause_indices": unresolved,
        "final_unsafe_clause_index": unsafe,
    }


@dataclass(frozen=True)
class NativeObjectiveHardClauseEscalationTrace:
    """Whole-query aggregate trace."""

    query_id: str
    plan_hash: str
    task_ir_hash: str
    schedule_hash: str
    decision: NativeHardClauseEscalationDecisionIR
    actions: Tuple[NativeObjectiveHardClauseEscalationActionTrace, ...]
    clause_traces: Tuple[NativeObjectiveHardClauseEscalationClauseTrace, ...]
    completed_objective_clause_indices: Tuple[int, ...]
    final_status: FinalStatus
    final_verified_clause_indices: Tuple[int, ...]
    final_unresolved_clause_indices: Tuple[int, ...]
    final_unsafe_clause_index: Optional[int]
    fallback_reason: str
    elapsed_ns: int
    deadline_ns: int
    semantic_signature_hash: str
    performance_claimed: bool = False
    schema_version: str = NATIVE_OBJECTIVE_HARD_CLAUSE_ESCALATION_TRACE_SCHEMA_VERSION

    def semantic_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision.to_dict(),
            "clause_traces": [trace.to_dict() for trace in self.clause_traces],
            "completed_objective_clause_indices": list(
                self.completed_objective_clause_indices
            ),
            "final_status": self.final_status,
            "final_verified_clause_indices": list(self.final_verified_clause_indices),
            "final_unresolved_clause_indices": list(
                self.final_unresolved_clause_indices
            ),
            "final_unsafe_clause_index": self.final_unsafe_clause_index,
            "fallback_reason": self.fallback_reason,
        }

    def validate_against(
        self, program: NativeObjectiveHardClauseEscalationProgram
    ) -> None:
        program.schedule.validate_against(program.task_ir)
        sequences = (
            self.completed_objective_clause_indices,
            self.final_verified_clause_indices,
            self.final_unresolved_clause_indices,
        )
        if (
            self.schema_version
            != NATIVE_OBJECTIVE_HARD_CLAUSE_ESCALATION_TRACE_SCHEMA_VERSION
            or not self.query_id
            or self.plan_hash != program.plan.stable_hash()
            or self.task_ir_hash != program.task_ir.stable_hash()
            or self.schedule_hash != program.schedule.stable_hash(program.task_ir)
            or self.decision.plan_hash != program.base_program.plan.stable_hash()
            or len(self.actions) != len(program.schedule.actions)
            or any(tuple(sorted(set(values))) != values for values in sequences)
            or not set(self.decision.baseline_verified_clause_indices)
            <= set(self.final_verified_clause_indices)
            or not self.fallback_reason
            or self.elapsed_ns < 0
            or self.deadline_ns != program.base_program.plan.whole_query_timeout_ns
            or self.performance_claimed is not False
            or self.semantic_signature_hash != _canonical_hash(self.semantic_dict())
        ):
            raise ValueError("objective hard-clause aggregate trace differs")
        for expected, actual in zip(program.schedule.actions, self.actions):
            actual.validate()
            if (
                expected.sequence != actual.sequence
                or expected.action_id != actual.action_id
                or expected.task_id != actual.task_id
                or expected.kind != actual.kind
                or expected.guard != actual.guard
                or expected.original_clause_index != actual.original_clause_index
            ):
                raise ValueError("objective hard-clause runtime/Schedule differs")
        for clause_trace in self.clause_traces:
            clause_trace.validate(clause_count=program.plan.clause_count)
        if self.final_status == "verified":
            if len(self.final_verified_clause_indices) != program.plan.clause_count:
                raise ValueError("objective hard-clause verified closure differs")
        elif self.final_status == "unsafe":
            if self.final_unsafe_clause_index is None:
                raise ValueError("objective hard-clause unsafe witness differs")
        elif (
            self.final_unsafe_clause_index is not None
            or not self.final_unresolved_clause_indices
        ):
            raise ValueError("objective hard-clause unknown accounting differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "schedule_hash": self.schedule_hash,
            "actions": [action.to_dict() for action in self.actions],
            **self.semantic_dict(),
            "elapsed_ns": self.elapsed_ns,
            "deadline_ns": self.deadline_ns,
            "deadline_enforcement": "whole_query_cooperative_stage_boundaries",
            "semantic_signature_hash": self.semantic_signature_hash,
            "performance_claimed": self.performance_claimed,
        }


@dataclass(frozen=True)
class NativeObjectiveHardClauseEscalationExecution:
    """Typed parent, shared source, objective children, and aggregate proof."""

    program: NativeObjectiveHardClauseEscalationProgram
    baseline: NativeParametricCompleteVerifierQueryExecution
    shared_refinement_program: Optional[NativeIntermediateRefinementProgram]
    shared_refinement: Optional[NativeIntermediateRefinementExecution]
    clause_executions: Tuple[NativeObjectiveHardClauseEscalationClauseExecution, ...]
    trace: NativeObjectiveHardClauseEscalationTrace

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
        self.program.validate_against(
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
            thresholds=thresholds,
            search_policy=search_policy,
            optimizer_policy=optimizer_policy,
        )
        objectives = _normalize_objective_matrix(linear_spec_C)
        self.baseline.validate_against(module, input_spec, linear_spec_C=objectives)
        decision = _decision_from_baseline(
            self.program.base_program.plan, self.baseline
        )
        if self.trace.decision != decision:
            raise ValueError("objective hard-clause baseline/Decision differs")
        if (self.shared_refinement_program is None) != (self.shared_refinement is None):
            raise ValueError("objective hard-clause shared state differs")
        if (
            self.shared_refinement is not None
            and self.shared_refinement_program is not None
        ):
            self.shared_refinement.validate(module, input_spec)
            if (
                self.shared_refinement.program != self.shared_refinement_program
                or self.shared_refinement_program.plan.policy
                != self.program.base_program.plan.refinement_policy
                or self.shared_refinement_program.plan.objective_hash is not None
                or self.shared_refinement_program.plan.source_refinement_plan_hash
                is not None
            ):
                raise ValueError("objective hard-clause shared program differs")
        ordinals = tuple(
            child.original_clause_index for child in self.clause_executions
        )
        if len(ordinals) != len(set(ordinals)) or not set(ordinals) <= set(
            decision.escalated_clause_indices
        ):
            raise ValueError("objective hard-clause child ordinal coverage differs")
        if self.clause_executions and self.shared_refinement is None:
            raise ValueError("objective hard-clause child lacks shared source")
        for child in self.clause_executions:
            assert self.shared_refinement is not None
            ordinal = child.original_clause_index
            child.validate_against(
                module,
                input_spec,
                objective=objectives[:, ordinal : ordinal + 1, :].contiguous(),
                threshold=thresholds[ordinal : ordinal + 1].contiguous(),
                shared_refinement=self.shared_refinement,
            )
        expected_traces = tuple(
            _clause_trace(child) for child in self.clause_executions
        )
        expected_aggregate = _aggregate_semantics(
            self.program.plan.clause_count, decision, self.clause_executions
        )
        if self.trace.clause_traces != expected_traces or any(
            getattr(self.trace, key) != value
            for key, value in expected_aggregate.items()
        ):
            raise ValueError("objective hard-clause child/aggregate semantics differ")
        self.trace.validate_against(self.program)


def compile_native_objective_hard_clause_escalation_program(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
    plan_id: str,
    search_policy: NativeProjectedGradientSearchPolicy,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
) -> NativeObjectiveHardClauseEscalationProgram:
    """Compile an additive objective stage over an exact NRIR-30 program."""

    base = compile_native_hard_clause_escalation_program(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        thresholds=thresholds,
        plan_id=f"{plan_id}:base-nrir30",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    plan = NativeObjectiveHardClauseEscalationPlanIR(
        plan_id=plan_id,
        base_plan_hash=base.plan.stable_hash(),
        base_task_ir_hash=base.task_ir.stable_hash(),
        base_schedule_hash=base.schedule.stable_hash(base.task_ir),
        clause_count=base.plan.clause_count,
        objective_refinement_policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=128,
            backward_chunk_size=32,
            candidate_policy_id="objective_influence_width_per_relu_v1",
        ),
    )
    task_ir, schedule = lower_native_objective_hard_clause_escalation_ir(plan)
    program = NativeObjectiveHardClauseEscalationProgram(base, plan, task_ir, schedule)
    program.validate_against(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        thresholds=thresholds,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    return program


def _action_trace(
    program: NativeObjectiveHardClauseEscalationProgram,
    index: int,
    *,
    executed: bool,
    reason: str,
    elapsed_ns: int,
    output: object,
) -> NativeObjectiveHardClauseEscalationActionTrace:
    action = program.schedule.actions[index]
    trace = NativeObjectiveHardClauseEscalationActionTrace(
        sequence=action.sequence,
        action_id=action.action_id,
        task_id=action.task_id,
        kind=action.kind,
        guard=action.guard,
        original_clause_index=action.original_clause_index,
        executed=executed,
        reason=reason,
        elapsed_ns=elapsed_ns,
        output_hash=_canonical_hash(output),
    )
    trace.validate()
    return trace


def execute_native_objective_hard_clause_escalation_program(
    program: NativeObjectiveHardClauseEscalationProgram,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
    query_id: str,
    search_policy: NativeProjectedGradientSearchPolicy,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    clock_ns: ClockNs = time.monotonic_ns,
) -> NativeObjectiveHardClauseEscalationExecution:
    """Execute baseline/shared source/per-clause children under one deadline."""

    if not query_id:
        raise ValueError("objective hard-clause query ID must be non-empty")
    program.validate_against(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        thresholds=thresholds,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    objectives = _normalize_objective_matrix(linear_spec_C)
    started_ns = clock_ns()
    deadline_ns = started_ns + program.base_program.plan.whole_query_timeout_ns
    actions: list[NativeObjectiveHardClauseEscalationActionTrace] = []

    action_started_ns = clock_ns()
    baseline = execute_native_parametric_production_complete_verifier_query(
        module,
        input_spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        query_id=f"{query_id}:baseline",
        query_policy=CompleteVerifierQueryPolicy(
            timeout_ns=max(1, deadline_ns - action_started_ns)
        ),
        search_policy=search_policy,
        queue_config=NativeReluSplitBabConfig(
            max_nodes=program.base_program.plan.baseline_budget.max_nodes,
            max_depth=program.base_program.plan.baseline_budget.max_depth,
            expansion_batch_size=2,
            max_eval_batch_size=4,
        ),
        optimizer_policy=optimizer_policy,
        clock_ns=clock_ns,
    )
    action_finished_ns = clock_ns()
    actions.append(
        _action_trace(
            program,
            0,
            executed=True,
            reason="baseline_completed",
            elapsed_ns=max(0, action_finished_ns - action_started_ns),
            output=baseline.trace.stable_hash(),
        )
    )
    action_started_ns = clock_ns()
    decision = _decision_from_baseline(program.base_program.plan, baseline)
    action_finished_ns = clock_ns()
    actions.append(
        _action_trace(
            program,
            1,
            executed=True,
            reason=decision.reason,
            elapsed_ns=max(0, action_finished_ns - action_started_ns),
            output=decision.to_dict(),
        )
    )

    shared_program: Optional[NativeIntermediateRefinementProgram] = None
    shared: Optional[NativeIntermediateRefinementExecution] = None
    admitted = set(decision.escalated_clause_indices)
    fallback_reason = "no_escalation_needed"
    if admitted and clock_ns() < deadline_ns:
        action_started_ns = clock_ns()
        shared_program = compile_native_intermediate_refinement_program(
            module,
            input_spec,
            policy=program.base_program.plan.refinement_policy,
            plan_id=f"{query_id}:shared-refinement",
        )
        action_finished_ns = clock_ns()
        actions.append(
            _action_trace(
                program,
                2,
                executed=True,
                reason="shared_refinement_compiled",
                elapsed_ns=max(0, action_finished_ns - action_started_ns),
                output=shared_program.hashes(),
            )
        )
        if action_finished_ns < deadline_ns:
            action_started_ns = clock_ns()
            shared = execute_native_intermediate_refinement_program(
                shared_program, module, input_spec
            )
            action_finished_ns = clock_ns()
            actions.append(
                _action_trace(
                    program,
                    3,
                    executed=True,
                    reason="shared_refinement_completed",
                    elapsed_ns=max(0, action_finished_ns - action_started_ns),
                    output=shared.trace.stable_hash(),
                )
            )
            if action_finished_ns >= deadline_ns:
                fallback_reason = "deadline_after_shared_refinement"
        else:
            actions.append(
                _action_trace(
                    program,
                    3,
                    executed=False,
                    reason="deadline_after_shared_compile",
                    elapsed_ns=0,
                    output={"skipped": True},
                )
            )
            fallback_reason = "deadline_after_shared_compile"
    else:
        reason = (
            "deadline_before_shared_refinement" if admitted else "no_escalation_needed"
        )
        for index in (2, 3):
            actions.append(
                _action_trace(
                    program,
                    index,
                    executed=False,
                    reason=reason,
                    elapsed_ns=0,
                    output={"skipped": True},
                )
            )
        fallback_reason = reason

    children: list[NativeObjectiveHardClauseEscalationClauseExecution] = []
    for ordinal in range(program.plan.clause_count):
        base_index = 4 + ordinal * 3
        if ordinal not in admitted:
            for offset in range(3):
                actions.append(
                    _action_trace(
                        program,
                        base_index + offset,
                        executed=False,
                        reason="clause_not_admitted",
                        elapsed_ns=0,
                        output={"skipped": True},
                    )
                )
            continue
        if shared is None or clock_ns() >= deadline_ns:
            reason = "deadline_before_objective_refinement"
            for offset in range(3):
                actions.append(
                    _action_trace(
                        program,
                        base_index + offset,
                        executed=False,
                        reason=reason,
                        elapsed_ns=0,
                        output={"skipped": True},
                    )
                )
            fallback_reason = reason
            continue
        objective = objectives[:, ordinal : ordinal + 1, :].contiguous()
        threshold = thresholds[ordinal : ordinal + 1].contiguous()
        action_started_ns = clock_ns()
        child_program = compile_native_intermediate_refinement_program(
            module,
            input_spec,
            policy=program.plan.objective_refinement_policy,
            plan_id=f"{query_id}:clause:{ordinal:04d}:objective-refinement",
            linear_spec_C=objective,
            source_refinement_execution=shared,
        )
        action_finished_ns = clock_ns()
        actions.append(
            _action_trace(
                program,
                base_index,
                executed=True,
                reason="objective_refinement_compiled",
                elapsed_ns=max(0, action_finished_ns - action_started_ns),
                output=child_program.hashes(),
            )
        )
        if action_finished_ns >= deadline_ns:
            for offset in (1, 2):
                actions.append(
                    _action_trace(
                        program,
                        base_index + offset,
                        executed=False,
                        reason="deadline_after_objective_compile",
                        elapsed_ns=0,
                        output={"skipped": True},
                    )
                )
            fallback_reason = "deadline_after_objective_compile"
            continue
        action_started_ns = clock_ns()
        child_refinement = execute_native_intermediate_refinement_program(
            child_program, module, input_spec
        )
        action_finished_ns = clock_ns()
        actions.append(
            _action_trace(
                program,
                base_index + 1,
                executed=True,
                reason="objective_refinement_completed",
                elapsed_ns=max(0, action_finished_ns - action_started_ns),
                output=child_refinement.trace.stable_hash(),
            )
        )
        if action_finished_ns >= deadline_ns:
            actions.append(
                _action_trace(
                    program,
                    base_index + 2,
                    executed=False,
                    reason="deadline_after_objective_refinement",
                    elapsed_ns=0,
                    output={"skipped": True},
                )
            )
            fallback_reason = "deadline_after_objective_refinement"
            continue
        action_started_ns = clock_ns()
        child_query = execute_native_parametric_production_complete_verifier_query(
            module,
            input_spec,
            linear_spec_C=objective,
            thresholds=threshold,
            query_id=f"{query_id}:clause:{ordinal:04d}:query",
            query_policy=CompleteVerifierQueryPolicy(
                timeout_ns=max(1, deadline_ns - action_started_ns)
            ),
            search_policy=search_policy,
            queue_config=NativeReluSplitBabConfig(
                max_nodes=program.base_program.plan.escalation_budget.max_nodes,
                max_depth=program.base_program.plan.escalation_budget.max_depth,
                expansion_batch_size=2,
                max_eval_batch_size=4,
            ),
            optimizer_policy=optimizer_policy,
            relu_pre_override=child_refinement.relu_pre,
            intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
            clock_ns=clock_ns,
        )
        action_finished_ns = clock_ns()
        accepted = (
            action_finished_ns <= deadline_ns
            and not child_query.trace.pending_clause_indices
            and bool(child_query.clauses)
        )
        child = NativeObjectiveHardClauseEscalationClauseExecution(
            original_clause_index=ordinal,
            refinement_program=child_program,
            refinement=child_refinement,
            query=child_query,
            accepted_before_deadline=accepted,
        )
        children.append(child)
        actions.append(
            _action_trace(
                program,
                base_index + 2,
                executed=True,
                reason=(
                    "objective_query_completed"
                    if accepted
                    else "objective_query_discarded_or_pending"
                ),
                elapsed_ns=max(0, action_finished_ns - action_started_ns),
                output=child_query.trace.stable_hash(),
            )
        )
        if not accepted:
            fallback_reason = "deadline_during_objective_query"

    aggregate = _aggregate_semantics(
        program.plan.clause_count, decision, tuple(children)
    )
    aggregate_index = len(program.schedule.actions) - 2
    actions.append(
        _action_trace(
            program,
            aggregate_index,
            executed=True,
            reason="verdicts_aggregated",
            elapsed_ns=0,
            output=aggregate,
        )
    )
    completed_objective_clause_indices = tuple(
        child.original_clause_index
        for child in children
        if child.accepted_before_deadline and child.query.clauses
    )
    if admitted and set(completed_objective_clause_indices) == admitted:
        fallback_reason = "none"
    actions.append(
        _action_trace(
            program,
            aggregate_index + 1,
            executed=True,
            reason="result_emitted",
            elapsed_ns=0,
            output=aggregate,
        )
    )
    clause_traces = tuple(_clause_trace(child) for child in children)
    trace = NativeObjectiveHardClauseEscalationTrace(
        query_id=query_id,
        plan_hash=program.plan.stable_hash(),
        task_ir_hash=program.task_ir.stable_hash(),
        schedule_hash=program.schedule.stable_hash(program.task_ir),
        decision=decision,
        actions=tuple(actions),
        clause_traces=clause_traces,
        completed_objective_clause_indices=aggregate[
            "completed_objective_clause_indices"
        ],  # type: ignore[arg-type]
        final_status=aggregate["final_status"],  # type: ignore[arg-type]
        final_verified_clause_indices=aggregate[
            "final_verified_clause_indices"
        ],  # type: ignore[arg-type]
        final_unresolved_clause_indices=aggregate[
            "final_unresolved_clause_indices"
        ],  # type: ignore[arg-type]
        final_unsafe_clause_index=aggregate[
            "final_unsafe_clause_index"
        ],  # type: ignore[arg-type]
        fallback_reason=fallback_reason,
        elapsed_ns=max(0, clock_ns() - started_ns),
        deadline_ns=program.base_program.plan.whole_query_timeout_ns,
        semantic_signature_hash="",
    )
    trace = NativeObjectiveHardClauseEscalationTrace(
        **{
            **trace.__dict__,
            "semantic_signature_hash": _canonical_hash(trace.semantic_dict()),
        }
    )
    execution = NativeObjectiveHardClauseEscalationExecution(
        program=program,
        baseline=baseline,
        shared_refinement_program=shared_program,
        shared_refinement=shared,
        clause_executions=tuple(children),
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
    "NativeObjectiveHardClauseEscalationActionTrace",
    "NativeObjectiveHardClauseEscalationClauseExecution",
    "NativeObjectiveHardClauseEscalationClauseTrace",
    "NativeObjectiveHardClauseEscalationExecution",
    "NativeObjectiveHardClauseEscalationProgram",
    "NativeObjectiveHardClauseEscalationTrace",
    "compile_native_objective_hard_clause_escalation_program",
    "execute_native_objective_hard_clause_escalation_program",
]
