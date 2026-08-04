"""Global-deadline complete-query control for sibling-packed queues."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Callable, Optional, Tuple

import torch

from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.refinement import NativeIntermediateRefinementPolicyIR
from ..ir.task import BFTaskModule
from .complete_verifier_query import (
    CompleteQueryStatus,
    CompleteVerifierClauseTrace,
    CompleteVerifierQueryPolicy,
    CompleteVerifierQueryTrace,
    _config_hash,
    _normalize_objective_matrix,
)
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_candidate_search import (
    NativeCandidateSearchExecution,
    NativeProjectedGradientSearchPolicy,
    search_native_box_counterexample,
)
from .native_intermediate_refinement import (
    NativeIntermediateRefinementExecution,
    compile_native_intermediate_refinement_program,
    execute_native_intermediate_refinement_program,
)
from .native_objective_ancestral_queue import _normalize_objective
from .native_objective_ancestral_sibling_pack import (
    NativeObjectiveAncestralSiblingPackExecution,
    compile_native_objective_ancestral_sibling_pack_plan,
    execute_native_objective_ancestral_sibling_pack_queue,
)
from .native_property_verdict import (
    NativePropertyVerdictExecution,
    derive_native_property_verdict,
)
from .native_relu_split_bab_runtime import NativeReluSplitBabConfig
from .task_executor import InputSpec

ClockNs = Callable[[], int]


@dataclass(frozen=True)
class NativeObjectiveAncestralSiblingPackClauseExecution:
    """Search, typed sibling-packed queue, and sound verdict for one clause."""

    trace: CompleteVerifierClauseTrace
    search: NativeCandidateSearchExecution
    queue: NativeObjectiveAncestralSiblingPackExecution
    verdict: NativePropertyVerdictExecution

    def validate_against(
        self,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        linear_spec_C: torch.Tensor,
        threshold: torch.Tensor,
        optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    ) -> None:
        objective = _normalize_objective(linear_spec_C)
        self.trace.validate()
        self.search.validate_against(module, input_spec, linear_spec_C=objective)
        root_refinement = self.queue.node_refinements[0].execution
        self.queue.validate_against(
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root_refinement,
            optimizer_policy=optimizer_policy,
        )
        self.verdict.validate_against(
            module,
            input_spec,
            linear_spec_C=objective,
            queue_execution=self.queue.queue,
        )
        if (
            self.trace.queue_execution_mode != "audit_validation"
            or self.trace.objective_hash != tensor_content_hash(objective)
            or self.trace.threshold != float(threshold.reshape(-1)[0].item())
            or self.trace.search_trace_hash != self.search.trace.stable_hash()
            or self.trace.queue_trace_hash != self.queue.queue.trace.stable_hash()
            or self.trace.verdict_trace_hash
            != self.verdict.trace.stable_hash(self.queue.queue.trace)
            or self.trace.status != self.verdict.trace.status
            or self.trace.search_counterexample_found
            != self.search.trace.counterexample_found
        ):
            raise ValueError("sibling-pack complete-query clause identity differs")


@dataclass(frozen=True)
class NativeObjectiveAncestralSiblingPackCompleteQueryExecution:
    """Global-deadline conjunction over typed sibling-packed clause queues."""

    trace: CompleteVerifierQueryTrace
    clauses: Tuple[NativeObjectiveAncestralSiblingPackClauseExecution, ...]
    search_policy: NativeProjectedGradientSearchPolicy
    queue_config: NativeReluSplitBabConfig
    optimizer_policy: NativeAlphaBetaOptimizerPolicy

    def validate_against(
        self,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        linear_spec_C: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> None:
        self.trace.validate()
        objectives = _normalize_objective_matrix(linear_spec_C)
        if (
            tensor_content_hash(objectives) != self.trace.objective_matrix_hash
            or tuple(float(item) for item in thresholds.detach().cpu())
            != self.trace.thresholds
            or len(self.clauses) != len(self.trace.completed_clauses)
            or tuple(item.trace for item in self.clauses)
            != self.trace.completed_clauses
            or self.search_policy.stable_hash() != self.trace.search_policy_hash
            or _config_hash(self.queue_config) != self.trace.queue_config_hash
            or self.optimizer_policy.stable_hash() != self.trace.optimizer_policy_hash
        ):
            raise ValueError("sibling-pack complete-query coverage differs")
        for clause in self.clauses:
            index = clause.trace.clause_index
            objective = objectives[:, index : index + 1, :].contiguous()
            threshold = thresholds[index : index + 1].contiguous()
            clause.validate_against(
                module,
                input_spec,
                linear_spec_C=objective,
                threshold=threshold,
                optimizer_policy=self.optimizer_policy,
            )
            if (
                clause.search.trace.policy != self.search_policy
                or clause.queue.queue.trace.config
                != replace(self.queue_config, threshold=clause.trace.threshold)
            ):
                raise ValueError("sibling-pack complete-query policy binding differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "trace": self.trace.to_dict(),
            "trace_hash": self.trace.stable_hash(),
            "clauses": [
                {
                    "trace": item.trace.to_dict(),
                    "queue": item.queue.to_dict(),
                    "verdict": item.verdict.trace.to_dict(),
                }
                for item in self.clauses
            ],
        }


def build_native_objective_ancestral_root_source(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    source_id: str,
) -> tuple[
    NativeIntermediateRefinementExecution,
    NativeIntermediateRefinementExecution,
]:
    """Build the frozen shared then objective-directed typed root source."""

    if not source_id:
        raise ValueError("objective ancestral root source ID must be non-empty")
    objective = _normalize_objective(linear_spec_C)
    shared_program = compile_native_intermediate_refinement_program(
        module,
        input_spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=128,
            backward_chunk_size=32,
        ),
        plan_id=f"{source_id}:shared",
    )
    shared = execute_native_intermediate_refinement_program(
        shared_program, module, input_spec
    )
    root_program = compile_native_intermediate_refinement_program(
        module,
        input_spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1,
            max_neurons_per_relu=128,
            backward_chunk_size=32,
            candidate_policy_id="objective_influence_width_per_relu_v1",
        ),
        plan_id=f"{source_id}:objective-root",
        linear_spec_C=objective,
        source_refinement_execution=shared,
    )
    root = execute_native_intermediate_refinement_program(
        root_program, module, input_spec
    )
    return shared, root


def execute_native_objective_ancestral_sibling_pack_complete_query(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
    query_id: str,
    query_policy: CompleteVerifierQueryPolicy,
    search_policy: NativeProjectedGradientSearchPolicy,
    queue_config: NativeReluSplitBabConfig,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    clock_ns: ClockNs = time.monotonic_ns,
) -> NativeObjectiveAncestralSiblingPackCompleteQueryExecution:
    """Run clauses in ordinal order under one cooperative 60-second deadline."""

    if not query_id:
        raise ValueError("sibling-pack complete-query ID must be non-empty")
    module.validate()
    query_policy.validate()
    search_policy.validate()
    queue_config.validate()
    optimizer_policy.validate()
    objectives = _normalize_objective_matrix(linear_spec_C)
    if (
        query_policy.timeout_ns != 60 * 1_000_000_000
        or queue_config.max_nodes != 31
        or queue_config.max_depth != 4
        or queue_config.expansion_batch_size != 1
        or queue_config.max_eval_batch_size != 2
        or not torch.is_tensor(thresholds)
        or not torch.is_floating_point(thresholds)
        or thresholds.dim() != 1
        or int(thresholds.shape[0]) != int(objectives.shape[1])
        or not bool(torch.isfinite(thresholds).all())
    ):
        raise ValueError("sibling-pack complete-query protocol differs")
    threshold_values = tuple(float(value) for value in thresholds.detach().cpu())
    clause_count = len(threshold_values)
    started_ns = clock_ns()
    deadline_ns = started_ns + query_policy.timeout_ns
    clauses: list[NativeObjectiveAncestralSiblingPackClauseExecution] = []
    pending: tuple[int, ...] = ()
    skipped: tuple[int, ...] = ()
    unsafe_index: Optional[int] = None

    for clause_index in range(clause_count):
        if clock_ns() >= deadline_ns:
            pending = tuple(range(clause_index, clause_count))
            break
        objective = objectives[:, clause_index : clause_index + 1, :].contiguous()
        threshold = thresholds[clause_index : clause_index + 1].contiguous()
        search = search_native_box_counterexample(
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold_values[clause_index],
            policy=search_policy,
        )
        if clock_ns() >= deadline_ns:
            pending = tuple(range(clause_index, clause_count))
            break
        _shared, root = build_native_objective_ancestral_root_source(
            module,
            input_spec,
            linear_spec_C=objective,
            source_id=f"{query_id}:clause:{clause_index:04d}:source",
        )
        plan = compile_native_objective_ancestral_sibling_pack_plan(
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=optimizer_policy,
            plan_id=f"{query_id}:clause:{clause_index:04d}:plan",
        )
        queue = execute_native_objective_ancestral_sibling_pack_queue(
            plan,
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=root,
            optimizer_policy=optimizer_policy,
            query_id=f"{query_id}:clause:{clause_index:04d}",
            whole_query_started_ns=started_ns,
            clock_ns=clock_ns,
        )
        root_id = queue.queue.trace.evaluations[0].node.node_id
        verdict = derive_native_property_verdict(
            module,
            input_spec,
            linear_spec_C=objective,
            queue_execution=queue.queue,
            candidate_counterexamples=((root_id, search.best_input),),
        )
        clause_trace = CompleteVerifierClauseTrace(
            clause_index=clause_index,
            objective_hash=tensor_content_hash(objective),
            threshold=threshold_values[clause_index],
            search_trace_hash=search.trace.stable_hash(),
            queue_trace_hash=queue.queue.trace.stable_hash(),
            verdict_trace_hash=verdict.trace.stable_hash(queue.queue.trace),
            status=verdict.trace.status,
            search_counterexample_found=search.trace.counterexample_found,
        )
        clause = NativeObjectiveAncestralSiblingPackClauseExecution(
            trace=clause_trace,
            search=search,
            queue=queue,
            verdict=verdict,
        )
        clause.validate_against(
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            optimizer_policy=optimizer_policy,
        )
        clauses.append(clause)
        if verdict.trace.status == "unsafe":
            unsafe_index = clause_index
            skipped = tuple(range(clause_index + 1, clause_count))
            break

    unresolved = tuple(
        item.trace.clause_index for item in clauses if item.trace.status == "unknown"
    )
    if unsafe_index is not None:
        status: CompleteQueryStatus = "unsafe"
        reason = "concrete_counterexample_clause"
    elif len(clauses) < clause_count:
        if not pending:
            pending = tuple(range(len(clauses), clause_count))
        status = "unknown"
        reason = "query_deadline_exhausted"
    elif unresolved:
        status = "unknown"
        reason = "one_or_more_clauses_unresolved"
    else:
        status = "verified"
        reason = "all_clauses_verified"
    trace = CompleteVerifierQueryTrace(
        query_id=query_id,
        status=status,
        reason=reason,
        objective_matrix_hash=tensor_content_hash(objectives),
        thresholds=threshold_values,
        query_policy=query_policy,
        search_policy_hash=search_policy.stable_hash(),
        optimizer_policy_hash=optimizer_policy.stable_hash(),
        queue_config_hash=_config_hash(queue_config),
        completed_clauses=tuple(item.trace for item in clauses),
        unresolved_clause_indices=unresolved,
        pending_clause_indices=pending,
        skipped_after_unsafe_clause_indices=skipped,
        unsafe_clause_index=unsafe_index,
        elapsed_ns=max(0, clock_ns() - started_ns),
    )
    execution = NativeObjectiveAncestralSiblingPackCompleteQueryExecution(
        trace=trace,
        clauses=tuple(clauses),
        search_policy=search_policy,
        queue_config=queue_config,
        optimizer_policy=optimizer_policy,
    )
    execution.validate_against(
        module,
        input_spec,
        linear_spec_C=objectives,
        thresholds=thresholds,
    )
    return execution


__all__ = [
    "NativeObjectiveAncestralSiblingPackClauseExecution",
    "NativeObjectiveAncestralSiblingPackCompleteQueryExecution",
    "build_native_objective_ancestral_root_source",
    "execute_native_objective_ancestral_sibling_pack_complete_query",
]
