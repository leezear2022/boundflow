"""Runtime-owned ranking-floor root projection for NRIR44."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=protected-access,duplicate-code,too-many-lines

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Callable, Optional

import torch

from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.bound import IntermediateBoundSource
from ..ir.root_projection_floor import (
    NativeRootProjectionClauseOwnerIR,
    NativeRootProjectionClauseTraceIR,
    NativeRootProjectionFloorInstanceIR,
    NativeRootProjectionFloorPlanIR,
    NativeRootProjectionFloorScheduleIR,
    NativeRootProjectionFloorTaskIRModule,
    NativeRootProjectionFloorTraceIR,
    lower_native_root_projection_floor_ir,
)
from ..ir.task import BFTaskModule
from .complete_verifier_query import (
    CompleteVerifierQueryPolicy,
    _normalize_objective_matrix,
)
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_candidate_search import NativeProjectedGradientSearchPolicy
from .native_hard_clause_escalation import _decision_from_baseline
from .native_intermediate_refinement import (
    NativeIntermediateRefinementExecution,
    NativeIntermediateRefinementProgram,
    compile_native_intermediate_refinement_program,
    execute_native_intermediate_refinement_program,
    intermediate_refinement_semantic_trace_hash,
)
from .native_objective_hard_clause_escalation import (
    NativeObjectiveHardClauseEscalationActionTrace,
    NativeObjectiveHardClauseEscalationClauseExecution,
    NativeObjectiveHardClauseEscalationExecution,
    NativeObjectiveHardClauseEscalationProgram,
    NativeObjectiveHardClauseEscalationTrace,
    _action_trace,
    _aggregate_semantics,
    _clause_trace,
    compile_native_objective_hard_clause_escalation_program,
)
from .native_parametric_production_complete_query import (
    execute_native_parametric_production_complete_verifier_query,
)
from .native_relu_split_bab_runtime import NativeReluSplitBabConfig
from .task_executor import InputSpec

ClockNs = Callable[[], int]


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NativeRootProjectionFloorProgram:
    """Source NRIR30/31 floor plus first-class ranking projection IR."""

    source_program: NativeObjectiveHardClauseEscalationProgram
    plan: NativeRootProjectionFloorPlanIR
    instance: NativeRootProjectionFloorInstanceIR
    task_ir: NativeRootProjectionFloorTaskIRModule
    schedule: NativeRootProjectionFloorScheduleIR

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
        self.source_program.validate_against(
            module,
            input_spec,
            linear_spec_C=objectives,
            thresholds=thresholds,
            search_policy=search_policy,
            optimizer_policy=optimizer_policy,
        )
        self.schedule.validate(
            plan=self.plan,
            instance=self.instance,
            task_module=self.task_ir,
        )
        owners = tuple(
            NativeRootProjectionClauseOwnerIR(
                original_clause_index=index,
                objective_hash=tensor_content_hash(
                    objectives[:, index : index + 1, :].contiguous()
                ),
                threshold_hash=tensor_content_hash(
                    thresholds[index : index + 1].contiguous()
                ),
            )
            for index in range(self.plan.clause_count)
        )
        if (
            self.plan.source_plan_hash != self.source_program.plan.stable_hash()
            or self.plan.source_task_ir_hash
            != self.source_program.task_ir.stable_hash()
            or self.plan.source_schedule_hash
            != self.source_program.schedule.stable_hash(self.source_program.task_ir)
            or self.instance.objective_matrix_hash != tensor_content_hash(objectives)
            or self.instance.thresholds_hash != tensor_content_hash(thresholds)
            or self.instance.clause_owners != owners
        ):
            raise ValueError("root-projection floor source/instance binding differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_program": self.source_program.to_dict(),
            "plan": self.plan.to_dict(),
            "instance": self.instance.to_dict(plan=self.plan),
            "task_ir": self.task_ir.to_dict(plan=self.plan, instance=self.instance),
            "schedule": self.schedule.to_dict(
                plan=self.plan,
                instance=self.instance,
                task_module=self.task_ir,
            ),
        }


@dataclass(frozen=True)
class NativeRootProjectionFloorExecution:
    """Projected source execution and its consumer-owned semantic trace."""

    program: NativeRootProjectionFloorProgram
    source_execution: NativeObjectiveHardClauseEscalationExecution
    projection_trace: NativeRootProjectionFloorTraceIR

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
        self.source_execution.validate_against(
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
            thresholds=thresholds,
            search_policy=search_policy,
            optimizer_policy=optimizer_policy,
        )
        if self.source_execution.program != self.program.source_program:
            raise ValueError("root-projection floor execution/source differs")
        expected = _projection_trace(self.program, self.source_execution)
        if self.projection_trace != expected:
            raise ValueError("root-projection floor execution/Trace differs")
        self.projection_trace.validate(
            plan=self.program.plan,
            instance=self.program.instance,
            task_module=self.program.task_ir,
            schedule=self.program.schedule,
        )


def compile_native_root_projection_floor_program(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
    plan_id: str,
    search_policy: NativeProjectedGradientSearchPolicy,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
) -> NativeRootProjectionFloorProgram:
    """Compile the ranking-only liveness projection over the frozen source floor."""

    objectives = _normalize_objective_matrix(linear_spec_C)
    source = compile_native_objective_hard_clause_escalation_program(
        module,
        input_spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        plan_id=f"{plan_id}:source-nrir31",
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    program = lower_native_root_projection_floor_program(
        source,
        linear_spec_C=objectives,
        thresholds=thresholds,
        plan_id=plan_id,
    )
    program.validate_against(
        module,
        input_spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    return program


def lower_native_root_projection_floor_program(
    source: NativeObjectiveHardClauseEscalationProgram,
    *,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
    plan_id: str,
) -> NativeRootProjectionFloorProgram:
    """Bind projection IR to an already compiled NRIR31 floor program."""

    objectives = _normalize_objective_matrix(linear_spec_C)
    plan = NativeRootProjectionFloorPlanIR(
        plan_id=plan_id,
        source_plan_hash=source.plan.stable_hash(),
        source_task_ir_hash=source.task_ir.stable_hash(),
        source_schedule_hash=source.schedule.stable_hash(source.task_ir),
        clause_count=source.plan.clause_count,
        consumed_result_fields=(
            "root.lower",
            "root.upper",
            "root.branch_candidate",
            "query.status",
            "counterexample.evidence",
        ),
    )
    owners = tuple(
        NativeRootProjectionClauseOwnerIR(
            original_clause_index=index,
            objective_hash=tensor_content_hash(
                objectives[:, index : index + 1, :].contiguous()
            ),
            threshold_hash=tensor_content_hash(
                thresholds[index : index + 1].contiguous()
            ),
        )
        for index in range(plan.clause_count)
    )
    instance = NativeRootProjectionFloorInstanceIR.create(
        plan=plan,
        objective_matrix_hash=tensor_content_hash(objectives),
        thresholds_hash=tensor_content_hash(thresholds),
        clause_owners=owners,
    )
    task_ir, schedule = lower_native_root_projection_floor_ir(plan, instance)
    program = NativeRootProjectionFloorProgram(
        source_program=source,
        plan=plan,
        instance=instance,
        task_ir=task_ir,
        schedule=schedule,
    )
    return program


def _execute_native_root_projection_source(
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
                max_nodes=1,
                max_depth=0,
                expansion_batch_size=1,
                max_eval_batch_size=1,
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


def _projection_trace(
    program: NativeRootProjectionFloorProgram,
    source: NativeObjectiveHardClauseEscalationExecution,
) -> NativeRootProjectionFloorTraceIR:
    if source.shared_refinement is None:
        raise ValueError("root-projection floor requires shared refinement")
    clause_traces = []
    for child in source.clause_executions:
        if len(child.query.clauses) != 1:
            raise ValueError("root-projection child query must own one clause")
        queue = child.query.clauses[0].queue.trace
        if (
            len(queue.evaluations) != 1
            or len(queue.decisions) != 1
            or queue.evaluations[0].node.depth != 0
        ):
            raise ValueError("root-projection child escaped the root-only budget")
        root = queue.evaluations[0]
        clause_traces.append(
            NativeRootProjectionClauseTraceIR(
                original_clause_index=child.original_clause_index,
                objective_hash=child.refinement_program.plan.objective_hash or "",
                refinement_plan_hash=child.refinement_program.plan.stable_hash(),
                refinement_trace_hash=intermediate_refinement_semantic_trace_hash(
                    child.refinement
                ),
                query_trace_hash=child.query.trace.stable_hash(),
                root_evaluation_hash=_canonical_hash(root.to_dict()),
                evaluation_count=len(queue.evaluations),
                decision_count=len(queue.decisions),
                child_node_count=sum(
                    evaluation.node.depth > 0 for evaluation in queue.evaluations
                ),
                status=child.query.clauses[0].trace.status,
            )
        )
    trace = NativeRootProjectionFloorTraceIR(
        plan_hash=program.plan.stable_hash(),
        instance_hash=program.instance.stable_hash(plan=program.plan),
        task_module_hash=program.task_ir.stable_hash(
            plan=program.plan, instance=program.instance
        ),
        schedule_hash=program.schedule.stable_hash(
            plan=program.plan,
            instance=program.instance,
            task_module=program.task_ir,
        ),
        source_floor_trace_hash=source.trace.semantic_signature_hash,
        baseline_trace_hash=source.baseline.trace.stable_hash(),
        shared_refinement_trace_hash=intermediate_refinement_semantic_trace_hash(
            source.shared_refinement
        ),
        clause_traces=tuple(clause_traces),
        completed_original_clause_indices=tuple(
            child.original_clause_index for child in source.clause_executions
        ),
        final_status=source.trace.final_status,
    )
    trace.validate(
        plan=program.plan,
        instance=program.instance,
        task_module=program.task_ir,
        schedule=program.schedule,
    )
    return trace


def execute_native_root_projection_floor_program(
    program: NativeRootProjectionFloorProgram,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
    query_id: str,
    search_policy: NativeProjectedGradientSearchPolicy,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    clock_ns: ClockNs = time.monotonic_ns,
) -> NativeRootProjectionFloorExecution:
    """Execute the ranking-only root projection under the source deadline."""

    source = _execute_native_root_projection_source(
        program.source_program,
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        thresholds=thresholds,
        query_id=query_id,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
        clock_ns=clock_ns,
    )
    execution = NativeRootProjectionFloorExecution(
        program=program,
        source_execution=source,
        projection_trace=_projection_trace(program, source),
    )
    execution.validate_against(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        thresholds=thresholds,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
    )
    return execution


__all__ = [
    "NativeRootProjectionFloorExecution",
    "NativeRootProjectionFloorProgram",
    "compile_native_root_projection_floor_program",
    "execute_native_root_projection_floor_program",
    "lower_native_root_projection_floor_program",
]
