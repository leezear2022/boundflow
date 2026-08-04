"""Runtime orchestration for typed unresolved-clause verifier escalation."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=protected-access,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Callable, Literal, Optional, Tuple

import torch

from ..frontends.plain_crown_bound_ir import (
    plain_crown_primal_graph_hash,
    tensor_content_hash,
)
from ..ir.bound import IntermediateBoundSource
from ..ir.hard_clause_escalation import (
    HARD_CLAUSE_ESCALATION_DECISION_IR_SCHEMA_VERSION,
    HardClauseEscalationTaskKind,
    NativeHardClauseEscalationDecisionIR,
    NativeHardClauseEscalationPlanIR,
    NativeHardClauseEscalationScheduleIR,
    NativeHardClauseEscalationTaskIRModule,
    lower_native_hard_clause_escalation_ir,
)
from ..ir.refinement import NativeIntermediateRefinementPolicyIR
from ..ir.search_scaling import NativeBabSearchBudgetIR
from ..ir.task import BFTaskModule
from .complete_verifier_query import (
    CompleteVerifierQueryPolicy,
    _normalize_objective_matrix,
)
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_candidate_search import NativeProjectedGradientSearchPolicy
from .native_intermediate_refinement import (
    NativeIntermediateRefinementExecution,
    NativeIntermediateRefinementProgram,
    _input_bounds_hash,
    compile_native_intermediate_refinement_program,
    execute_native_intermediate_refinement_program,
)
from .native_parametric_production_complete_query import (
    NativeParametricCompleteVerifierQueryExecution,
    execute_native_parametric_production_complete_verifier_query,
)
from .native_relu_split_bab_runtime import NativeReluSplitBabConfig
from .task_executor import InputSpec

NATIVE_HARD_CLAUSE_ESCALATION_ACTION_TRACE_SCHEMA_VERSION = (
    "boundflow.native-hard-clause-escalation-action-trace/v1"
)
NATIVE_HARD_CLAUSE_ESCALATION_TRACE_SCHEMA_VERSION = (
    "boundflow.native-hard-clause-escalation-trace/v1"
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
class NativeHardClauseEscalationProgram:
    """Compiled static Plan/Task/Schedule stack for one exact query."""

    plan: NativeHardClauseEscalationPlanIR
    task_ir: NativeHardClauseEscalationTaskIRModule
    schedule: NativeHardClauseEscalationScheduleIR

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
        self.plan.validate()
        self.schedule.validate_against(self.task_ir)
        objectives = _normalize_objective_matrix(linear_spec_C)
        if (
            self.task_ir.plan_hash != self.plan.stable_hash()
            or self.plan.primal_graph_hash != plain_crown_primal_graph_hash(module)
            or self.plan.input_bounds_hash != _input_bounds_hash(input_spec)
            or self.plan.objective_matrix_hash != tensor_content_hash(objectives)
            or self.plan.thresholds_hash != tensor_content_hash(thresholds)
            or self.plan.clause_count != int(objectives.shape[1])
            or thresholds.dim() != 1
            or int(thresholds.shape[0]) != self.plan.clause_count
            or self.plan.search_policy_hash != search_policy.stable_hash()
            or self.plan.optimizer_policy_hash != optimizer_policy.stable_hash()
        ):
            raise ValueError("hard-clause escalation program/query binding differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "plan_hash": self.plan.stable_hash(),
            "task_ir": self.task_ir.to_dict(),
            "task_ir_hash": self.task_ir.stable_hash(),
            "schedule": self.schedule.to_dict(),
            "schedule_hash": self.schedule.stable_hash(self.task_ir),
        }


@dataclass(frozen=True)
class NativeHardClauseEscalationActionTrace:
    """One actual or guarded-skip Schedule dispatch."""

    sequence: int
    action_id: str
    task_id: str
    kind: HardClauseEscalationTaskKind
    guard: str
    executed: bool
    reason: str
    elapsed_ns: int
    output_hash: str
    schema_version: str = NATIVE_HARD_CLAUSE_ESCALATION_ACTION_TRACE_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version
            != NATIVE_HARD_CLAUSE_ESCALATION_ACTION_TRACE_SCHEMA_VERSION
            or self.sequence < 0
            or not self.action_id
            or not self.task_id
            or self.guard not in {"always", "escalated_clauses_nonempty"}
            or not self.reason
            or self.elapsed_ns < 0
            or not _is_sha256(self.output_hash)
            or (not self.executed and self.elapsed_ns != 0)
        ):
            raise ValueError("hard-clause escalation action trace is invalid")

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
class NativeHardClauseEscalationTrace:
    """Aggregate proof/control trace with original clause ordinal ownership."""

    query_id: str
    plan_hash: str
    task_ir_hash: str
    schedule_hash: str
    decision: NativeHardClauseEscalationDecisionIR
    actions: Tuple[NativeHardClauseEscalationActionTrace, ...]
    escalation_original_clause_indices: Tuple[int, ...]
    escalation_completed_original_clause_indices: Tuple[int, ...]
    escalation_verified_original_clause_indices: Tuple[int, ...]
    escalation_pending_original_clause_indices: Tuple[int, ...]
    final_status: FinalStatus
    final_verified_clause_indices: Tuple[int, ...]
    final_unresolved_clause_indices: Tuple[int, ...]
    final_unsafe_clause_index: Optional[int]
    fallback_reason: str
    elapsed_ns: int
    deadline_ns: int
    semantic_signature_hash: str
    performance_claimed: bool = False
    schema_version: str = NATIVE_HARD_CLAUSE_ESCALATION_TRACE_SCHEMA_VERSION

    def semantic_dict(self) -> dict[str, object]:
        return {
            "decision_semantics": {
                "baseline_completed_clause_indices": list(
                    self.decision.baseline_completed_clause_indices
                ),
                "baseline_verified_clause_indices": list(
                    self.decision.baseline_verified_clause_indices
                ),
                "baseline_unresolved_clause_indices": list(
                    self.decision.baseline_unresolved_clause_indices
                ),
                "baseline_pending_clause_indices": list(
                    self.decision.baseline_pending_clause_indices
                ),
                "baseline_unsafe_clause_index": (
                    self.decision.baseline_unsafe_clause_index
                ),
                "escalated_clause_indices": list(
                    self.decision.escalated_clause_indices
                ),
                "reason": self.decision.reason,
            },
            "escalation_original_clause_indices": list(
                self.escalation_original_clause_indices
            ),
            "escalation_completed_original_clause_indices": list(
                self.escalation_completed_original_clause_indices
            ),
            "escalation_verified_original_clause_indices": list(
                self.escalation_verified_original_clause_indices
            ),
            "escalation_pending_original_clause_indices": list(
                self.escalation_pending_original_clause_indices
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
        self,
        program: NativeHardClauseEscalationProgram,
    ) -> None:
        self.decision.validate()
        program.schedule.validate_against(program.task_ir)
        clause_indices = set(range(program.plan.clause_count))
        verified = set(self.final_verified_clause_indices)
        unresolved = set(self.final_unresolved_clause_indices)
        sequences = (
            self.escalation_original_clause_indices,
            self.escalation_completed_original_clause_indices,
            self.escalation_verified_original_clause_indices,
            self.escalation_pending_original_clause_indices,
            self.final_verified_clause_indices,
            self.final_unresolved_clause_indices,
        )
        if (
            self.schema_version != NATIVE_HARD_CLAUSE_ESCALATION_TRACE_SCHEMA_VERSION
            or not self.query_id
            or self.plan_hash != program.plan.stable_hash()
            or self.task_ir_hash != program.task_ir.stable_hash()
            or self.schedule_hash != program.schedule.stable_hash(program.task_ir)
            or self.decision.plan_hash != self.plan_hash
            or len(self.actions) != len(program.schedule.actions)
            or any(tuple(sorted(set(values))) != values for values in sequences)
            or not verified <= clause_indices
            or not unresolved <= clause_indices
            or verified & unresolved
            or not set(self.decision.baseline_verified_clause_indices) <= verified
            or self.elapsed_ns < 0
            or self.deadline_ns != program.plan.whole_query_timeout_ns
            or not self.fallback_reason
            or self.performance_claimed is not False
            or self.semantic_signature_hash != _canonical_hash(self.semantic_dict())
        ):
            raise ValueError("hard-clause escalation aggregate trace differs")
        for expected, action in zip(program.schedule.actions, self.actions):
            action.validate()
            if (
                action.sequence != expected.sequence
                or action.action_id != expected.action_id
                or action.task_id != expected.task_id
                or action.kind != expected.kind
                or action.guard != expected.guard
            ):
                raise ValueError("hard-clause escalation runtime/Schedule differs")
        if self.final_status == "verified":
            if (
                verified != clause_indices
                or unresolved
                or self.final_unsafe_clause_index is not None
            ):
                raise ValueError("hard-clause escalation verified closure differs")
        elif self.final_status == "unsafe":
            if self.final_unsafe_clause_index is None:
                raise ValueError("hard-clause escalation unsafe witness differs")
        elif self.final_unsafe_clause_index is not None or not unresolved:
            raise ValueError("hard-clause escalation unknown accounting differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "schedule_hash": self.schedule_hash,
            "decision": self.decision.to_dict(),
            "decision_hash": self.decision.stable_hash(),
            "actions": [item.to_dict() for item in self.actions],
            **self.semantic_dict(),
            "elapsed_ns": self.elapsed_ns,
            "deadline_ns": self.deadline_ns,
            "deadline_enforcement": "whole_query_cooperative_stage_boundaries",
            "semantic_signature_hash": self.semantic_signature_hash,
            "performance_claimed": self.performance_claimed,
        }


@dataclass(frozen=True)
class NativeHardClauseEscalationExecution:
    """Typed staged execution plus all proof-producing child executions."""

    program: NativeHardClauseEscalationProgram
    baseline: NativeParametricCompleteVerifierQueryExecution
    refinement_program: Optional[NativeIntermediateRefinementProgram]
    refinement: Optional[NativeIntermediateRefinementExecution]
    escalation: Optional[NativeParametricCompleteVerifierQueryExecution]
    trace: NativeHardClauseEscalationTrace

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
        expected_decision = _decision_from_baseline(self.program.plan, self.baseline)
        if self.trace.decision != expected_decision:
            raise ValueError("hard-clause escalation baseline/Decision differs")
        escalated = expected_decision.escalated_clause_indices
        if self.escalation is None:
            if (self.refinement is None) != (self.refinement_program is None):
                raise ValueError("hard-clause escalation partial child state differs")
            if self.refinement is not None and self.refinement_program is not None:
                self.refinement.validate(module, input_spec)
                if self.refinement.program != self.refinement_program:
                    raise ValueError(
                        "hard-clause escalation fallback refinement program differs"
                    )
        else:
            if (
                self.refinement is None
                or self.refinement_program is None
                or not escalated
                or self.trace.escalation_original_clause_indices != escalated
            ):
                raise ValueError("hard-clause escalation child coverage differs")
            self.refinement.validate(module, input_spec)
            if self.refinement.program != self.refinement_program:
                raise ValueError("hard-clause escalation refinement program differs")
            projected = objectives[:, list(escalated), :].contiguous()
            self.escalation.validate_against(
                module, input_spec, linear_spec_C=projected
            )
        expected_aggregate = _aggregate_semantics(
            self.program.plan,
            expected_decision,
            self.escalation,
            escalation_accepted=(
                self.escalation is not None and self.trace.fallback_reason == "none"
            ),
        )
        if any(
            getattr(self.trace, name) != value
            for name, value in expected_aggregate.items()
        ):
            raise ValueError("hard-clause escalation child/aggregate semantics differ")
        self.trace.validate_against(self.program)


def compile_native_hard_clause_escalation_program(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
    plan_id: str,
    search_policy: NativeProjectedGradientSearchPolicy,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
) -> NativeHardClauseEscalationProgram:
    """Compile the frozen two-stage query policy into first-class IR."""

    module.validate()
    search_policy.validate()
    optimizer_policy.validate()
    objectives = _normalize_objective_matrix(linear_spec_C)
    if (
        not torch.is_tensor(thresholds)
        or not torch.is_floating_point(thresholds)
        or thresholds.dim() != 1
        or int(thresholds.shape[0]) != int(objectives.shape[1])
        or not bool(torch.isfinite(thresholds).all())
    ):
        raise ValueError("hard-clause escalation thresholds are invalid")
    plan = NativeHardClauseEscalationPlanIR(
        plan_id=plan_id,
        primal_graph_hash=plain_crown_primal_graph_hash(module),
        input_bounds_hash=_input_bounds_hash(input_spec),
        objective_matrix_hash=tensor_content_hash(objectives),
        thresholds_hash=tensor_content_hash(thresholds),
        clause_count=int(objectives.shape[1]),
        whole_query_timeout_ns=60 * 1_000_000_000,
        baseline_budget=NativeBabSearchBudgetIR("baseline-n7d2", 7, 2),
        escalation_budget=NativeBabSearchBudgetIR("escalation-n31d4", 31, 4),
        refinement_policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=128,
            backward_chunk_size=32,
        ),
        search_policy_hash=search_policy.stable_hash(),
        optimizer_policy_hash=optimizer_policy.stable_hash(),
    )
    task_ir, schedule = lower_native_hard_clause_escalation_ir(plan)
    program = NativeHardClauseEscalationProgram(plan, task_ir, schedule)
    program.validate_against(
        module,
        input_spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    return program


def _decision_from_baseline(
    plan: NativeHardClauseEscalationPlanIR,
    baseline: NativeParametricCompleteVerifierQueryExecution,
) -> NativeHardClauseEscalationDecisionIR:
    trace = baseline.trace
    completed = tuple(item.clause_index for item in trace.completed_clauses)
    verified = tuple(
        item.clause_index
        for item in trace.completed_clauses
        if item.status == "verified"
    )
    unresolved = trace.unresolved_clause_indices
    pending = trace.pending_clause_indices
    if trace.unsafe_clause_index is not None:
        escalated: Tuple[int, ...] = ()
        reason = "baseline_unsafe_short_circuit"
    elif pending:
        escalated = ()
        reason = "baseline_deadline_pending"
    elif unresolved:
        escalated = unresolved
        reason = "escalate_exact_unresolved"
    else:
        escalated = ()
        reason = "baseline_verified_all"
    decision = NativeHardClauseEscalationDecisionIR(
        decision_id=f"{plan.plan_id}:decision",
        plan_hash=plan.stable_hash(),
        baseline_query_trace_hash=trace.stable_hash(),
        clause_count=plan.clause_count,
        baseline_completed_clause_indices=completed,
        baseline_verified_clause_indices=verified,
        baseline_unresolved_clause_indices=unresolved,
        baseline_pending_clause_indices=pending,
        baseline_unsafe_clause_index=trace.unsafe_clause_index,
        escalated_clause_indices=escalated,
        reason=reason,
        schema_version=HARD_CLAUSE_ESCALATION_DECISION_IR_SCHEMA_VERSION,
    )
    decision.validate()
    return decision


def _action_trace(
    program: NativeHardClauseEscalationProgram,
    index: int,
    *,
    executed: bool,
    reason: str,
    elapsed_ns: int,
    output: object,
) -> NativeHardClauseEscalationActionTrace:
    action = program.schedule.actions[index]
    trace = NativeHardClauseEscalationActionTrace(
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
    trace.validate()
    return trace


def _aggregate_semantics(
    plan: NativeHardClauseEscalationPlanIR,
    decision: NativeHardClauseEscalationDecisionIR,
    escalation: Optional[NativeParametricCompleteVerifierQueryExecution],
    *,
    escalation_accepted: bool,
) -> dict[str, object]:
    statuses: dict[int, str] = {index: "unknown" for index in range(plan.clause_count)}
    for index in decision.baseline_verified_clause_indices:
        statuses[index] = "verified"
    if decision.baseline_unsafe_clause_index is not None:
        statuses[decision.baseline_unsafe_clause_index] = "unsafe"
    completed_original: list[int] = []
    escalation_verified: list[int] = []
    escalation_pending: list[int] = []
    escalation_unsafe: Optional[int] = None
    if escalation is not None and escalation_accepted:
        ordinal_map = decision.escalated_clause_indices
        for clause in escalation.trace.completed_clauses:
            original = ordinal_map[clause.clause_index]
            completed_original.append(original)
            statuses[original] = clause.status
            if clause.status == "verified":
                escalation_verified.append(original)
            elif clause.status == "unsafe":
                escalation_unsafe = original
        escalation_pending.extend(
            ordinal_map[index] for index in escalation.trace.pending_clause_indices
        )
    unsafe_indices = sorted(
        index for index, status in statuses.items() if status == "unsafe"
    )
    verified = tuple(
        sorted(index for index, status in statuses.items() if status == "verified")
    )
    unresolved = tuple(
        sorted(index for index, status in statuses.items() if status == "unknown")
    )
    unsafe = None if not unsafe_indices else unsafe_indices[0]
    if unsafe is not None:
        final_status: FinalStatus = "unsafe"
    elif len(verified) == plan.clause_count:
        final_status = "verified"
    else:
        final_status = "unknown"
    return {
        "escalation_completed_original_clause_indices": tuple(
            sorted(completed_original)
        ),
        "escalation_verified_original_clause_indices": tuple(
            sorted(escalation_verified)
        ),
        "escalation_pending_original_clause_indices": tuple(sorted(escalation_pending)),
        "final_status": final_status,
        "final_verified_clause_indices": verified,
        "final_unresolved_clause_indices": unresolved,
        "final_unsafe_clause_index": (
            escalation_unsafe if escalation_unsafe is not None else unsafe
        ),
    }


def execute_native_hard_clause_escalation_program(
    program: NativeHardClauseEscalationProgram,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
    query_id: str,
    search_policy: NativeProjectedGradientSearchPolicy,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    clock_ns: ClockNs = time.monotonic_ns,
) -> NativeHardClauseEscalationExecution:
    """Execute one whole-deadline baseline/refine/escalate/aggregate program."""

    if not query_id:
        raise ValueError("hard-clause escalation query ID must be non-empty")
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
    deadline_ns = started_ns + program.plan.whole_query_timeout_ns
    actions: list[NativeHardClauseEscalationActionTrace] = []

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
            max_nodes=program.plan.baseline_budget.max_nodes,
            max_depth=program.plan.baseline_budget.max_depth,
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
    decision = _decision_from_baseline(program.plan, baseline)
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

    refinement_program: Optional[NativeIntermediateRefinementProgram] = None
    refinement: Optional[NativeIntermediateRefinementExecution] = None
    escalation: Optional[NativeParametricCompleteVerifierQueryExecution] = None
    escalation_accepted = False
    fallback_reason = "no_escalation_needed"
    guarded = bool(decision.escalated_clause_indices)
    if guarded and clock_ns() < deadline_ns:
        action_started_ns = clock_ns()
        refinement_program = compile_native_intermediate_refinement_program(
            module,
            input_spec,
            policy=program.plan.refinement_policy,
            plan_id=f"{query_id}:shared-refinement",
        )
        action_finished_ns = clock_ns()
        actions.append(
            _action_trace(
                program,
                2,
                executed=True,
                reason="refinement_compiled",
                elapsed_ns=max(0, action_finished_ns - action_started_ns),
                output=refinement_program.hashes(),
            )
        )

        action_started_ns = clock_ns()
        refinement = execute_native_intermediate_refinement_program(
            refinement_program, module, input_spec
        )
        action_finished_ns = clock_ns()
        actions.append(
            _action_trace(
                program,
                3,
                executed=True,
                reason="refinement_completed",
                elapsed_ns=max(0, action_finished_ns - action_started_ns),
                output=refinement.trace.stable_hash(),
            )
        )

        action_started_ns = clock_ns()
        ordinal_map = decision.escalated_clause_indices
        projected_objectives = objectives[:, list(ordinal_map), :].contiguous()
        projected_thresholds = thresholds[list(ordinal_map)].contiguous()
        action_finished_ns = clock_ns()
        actions.append(
            _action_trace(
                program,
                4,
                executed=True,
                reason="hard_clauses_projected",
                elapsed_ns=max(0, action_finished_ns - action_started_ns),
                output={
                    "ordinals": list(ordinal_map),
                    "objectives_hash": tensor_content_hash(projected_objectives),
                    "thresholds_hash": tensor_content_hash(projected_thresholds),
                },
            )
        )

        if action_finished_ns < deadline_ns:
            action_started_ns = clock_ns()
            escalation = execute_native_parametric_production_complete_verifier_query(
                module,
                input_spec,
                linear_spec_C=projected_objectives,
                thresholds=projected_thresholds,
                query_id=f"{query_id}:escalation",
                query_policy=CompleteVerifierQueryPolicy(
                    timeout_ns=max(1, deadline_ns - action_started_ns)
                ),
                search_policy=search_policy,
                queue_config=NativeReluSplitBabConfig(
                    max_nodes=program.plan.escalation_budget.max_nodes,
                    max_depth=program.plan.escalation_budget.max_depth,
                    expansion_batch_size=2,
                    max_eval_batch_size=4,
                ),
                optimizer_policy=optimizer_policy,
                relu_pre_override=refinement.relu_pre,
                intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
                clock_ns=clock_ns,
            )
            action_finished_ns = clock_ns()
            escalation_accepted = action_finished_ns <= deadline_ns
            fallback_reason = (
                "none" if escalation_accepted else "deadline_after_escalation"
            )
            actions.append(
                _action_trace(
                    program,
                    5,
                    executed=True,
                    reason=(
                        "escalation_completed"
                        if escalation_accepted
                        else "escalation_discarded_after_deadline"
                    ),
                    elapsed_ns=max(0, action_finished_ns - action_started_ns),
                    output=escalation.trace.stable_hash(),
                )
            )
        else:
            fallback_reason = "deadline_after_refinement"
            actions.append(
                _action_trace(
                    program,
                    5,
                    executed=False,
                    reason="deadline_after_refinement",
                    elapsed_ns=0,
                    output={"skipped": True},
                )
            )
    else:
        fallback_reason = (
            "deadline_before_refinement" if guarded else "no_escalation_needed"
        )
        for index in range(2, 6):
            actions.append(
                _action_trace(
                    program,
                    index,
                    executed=False,
                    reason=fallback_reason,
                    elapsed_ns=0,
                    output={"skipped": True},
                )
            )

    aggregate_started_ns = clock_ns()
    aggregate = _aggregate_semantics(
        program.plan,
        decision,
        escalation,
        escalation_accepted=escalation_accepted,
    )
    aggregate_finished_ns = clock_ns()
    actions.append(
        _action_trace(
            program,
            6,
            executed=True,
            reason="verdicts_aggregated",
            elapsed_ns=max(0, aggregate_finished_ns - aggregate_started_ns),
            output=aggregate,
        )
    )
    actions.append(
        _action_trace(
            program,
            7,
            executed=True,
            reason="result_emitted",
            elapsed_ns=0,
            output=aggregate,
        )
    )
    elapsed_ns = max(0, clock_ns() - started_ns)
    trace = NativeHardClauseEscalationTrace(
        query_id=query_id,
        plan_hash=program.plan.stable_hash(),
        task_ir_hash=program.task_ir.stable_hash(),
        schedule_hash=program.schedule.stable_hash(program.task_ir),
        decision=decision,
        actions=tuple(actions),
        escalation_original_clause_indices=(
            decision.escalated_clause_indices if escalation is not None else ()
        ),
        escalation_completed_original_clause_indices=aggregate[
            "escalation_completed_original_clause_indices"
        ],  # type: ignore[arg-type]
        escalation_verified_original_clause_indices=aggregate[
            "escalation_verified_original_clause_indices"
        ],  # type: ignore[arg-type]
        escalation_pending_original_clause_indices=aggregate[
            "escalation_pending_original_clause_indices"
        ],  # type: ignore[arg-type]
        final_status=aggregate["final_status"],  # type: ignore[arg-type]
        final_verified_clause_indices=aggregate[
            "final_verified_clause_indices"
        ],  # type: ignore[arg-type]
        final_unresolved_clause_indices=aggregate[
            "final_unresolved_clause_indices"
        ],  # type: ignore[arg-type]
        final_unsafe_clause_index=aggregate["final_unsafe_clause_index"],  # type: ignore[arg-type]
        fallback_reason=fallback_reason,
        elapsed_ns=elapsed_ns,
        deadline_ns=program.plan.whole_query_timeout_ns,
        semantic_signature_hash="",
    )
    trace = NativeHardClauseEscalationTrace(
        **{
            **trace.__dict__,
            "semantic_signature_hash": _canonical_hash(trace.semantic_dict()),
        }
    )
    trace.validate_against(program)
    execution = NativeHardClauseEscalationExecution(
        program=program,
        baseline=baseline,
        refinement_program=refinement_program,
        refinement=refinement,
        escalation=escalation,
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
    "NATIVE_HARD_CLAUSE_ESCALATION_ACTION_TRACE_SCHEMA_VERSION",
    "NATIVE_HARD_CLAUSE_ESCALATION_TRACE_SCHEMA_VERSION",
    "NativeHardClauseEscalationActionTrace",
    "NativeHardClauseEscalationExecution",
    "NativeHardClauseEscalationProgram",
    "NativeHardClauseEscalationTrace",
    "compile_native_hard_clause_escalation_program",
    "execute_native_hard_clause_escalation_program",
]
