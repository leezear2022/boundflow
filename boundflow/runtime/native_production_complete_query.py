"""Complete-query control over the production prepared verifier queue."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes
# pylint: disable=missing-function-docstring,duplicate-code,too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Mapping, Optional

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.bound import IntermediateBoundSource
from ..ir.task import BFTaskModule
from .complete_verifier_query import (
    ClockNs,
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
from .native_production_verifier import (
    NativeProductionReluSplitBabExecution,
    execute_native_production_relu_split_bab,
)
from .native_property_verdict import (
    NativePropertyVerdictExecution,
    derive_native_property_verdict,
)
from .native_relu_split_bab_runtime import NativeReluSplitBabConfig
from .task_executor import InputSpec


@dataclass(frozen=True)
class NativeProductionCompleteVerifierClauseExecution:
    """Search, production queue, and sound verdict for one clause."""

    trace: CompleteVerifierClauseTrace
    search: NativeCandidateSearchExecution
    queue: NativeProductionReluSplitBabExecution
    verdict: NativePropertyVerdictExecution

    def validate_against(
        self,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        linear_spec_C: torch.Tensor,
    ) -> None:
        self.trace.validate()
        self.search.validate_against(module, input_spec, linear_spec_C=linear_spec_C)
        self.queue.validate()
        self.verdict.validate_against(
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
            queue_execution=self.queue,
        )
        if (
            self.trace.queue_execution_mode != "production_prepared"
            or self.trace.objective_hash != tensor_content_hash(linear_spec_C)
            or self.trace.threshold != self.queue.trace.config.threshold
            or self.trace.search_trace_hash != self.search.trace.stable_hash()
            or self.trace.queue_trace_hash != self.queue.trace.stable_hash()
            or self.trace.verdict_trace_hash
            != self.verdict.trace.stable_hash(self.queue.trace)
            or self.trace.status != self.verdict.trace.status
            or self.trace.search_counterexample_found
            != self.search.trace.counterexample_found
        ):
            raise ValueError("production complete verifier clause identity differs")


@dataclass(frozen=True)
class NativeProductionCompleteVerifierQueryExecution:
    """Production complete-query trace and its completed clause executions."""

    trace: CompleteVerifierQueryTrace
    clauses: tuple[NativeProductionCompleteVerifierClauseExecution, ...]
    search_policy: NativeProjectedGradientSearchPolicy
    queue_config: NativeReluSplitBabConfig
    optimizer_policy: NativeAlphaBetaOptimizerPolicy

    def validate_against(
        self,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        linear_spec_C: torch.Tensor,
    ) -> None:
        self.trace.validate()
        objectives = _normalize_objective_matrix(linear_spec_C)
        if (
            self.trace.execution_mode != "production_prepared"
            or tensor_content_hash(objectives) != self.trace.objective_matrix_hash
            or len(self.clauses) != len(self.trace.completed_clauses)
            or tuple(item.trace for item in self.clauses)
            != self.trace.completed_clauses
            or self.search_policy.stable_hash() != self.trace.search_policy_hash
            or _config_hash(self.queue_config) != self.trace.queue_config_hash
            or self.optimizer_policy.stable_hash() != self.trace.optimizer_policy_hash
        ):
            raise ValueError("production complete verifier query coverage differs")
        for clause in self.clauses:
            objective = objectives[:, clause.trace.clause_index, :].contiguous()
            clause.validate_against(module, input_spec, linear_spec_C=objective)
            if (
                clause.search.trace.policy != self.search_policy
                or clause.queue.trace.config
                != replace(
                    self.queue_config,
                    threshold=self.trace.thresholds[clause.trace.clause_index],
                )
                or clause.queue.trace.optimizer_policy != self.optimizer_policy
            ):
                raise ValueError("production complete verifier policy binding differs")


def execute_native_production_complete_verifier_query(
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
    relu_pre_override: Optional[Mapping[str, IntervalState]] = None,
    intermediate_bound_source: IntermediateBoundSource = (
        IntermediateBoundSource.LOCAL_FORWARD
    ),
    clock_ns: ClockNs = time.monotonic_ns,
) -> NativeProductionCompleteVerifierQueryExecution:
    """Execute all conjunction clauses through the typed production queue."""

    if not query_id:
        raise ValueError("production complete verifier query ID must be non-empty")
    module.validate()
    query_policy.validate()
    search_policy.validate()
    queue_config.validate()
    optimizer_policy.validate()
    if not isinstance(intermediate_bound_source, IntermediateBoundSource):
        raise TypeError(
            "production complete verifier intermediate-bound source is invalid"
        )
    if (relu_pre_override is None) != (
        intermediate_bound_source == IntermediateBoundSource.LOCAL_FORWARD
    ):
        raise ValueError(
            "production complete verifier intermediate semantics/provenance differ"
        )
    objectives = _normalize_objective_matrix(linear_spec_C)
    if (
        not torch.is_tensor(thresholds)
        or not torch.is_floating_point(thresholds)
        or thresholds.dim() != 1
        or int(thresholds.shape[0]) != int(objectives.shape[1])
        or not bool(torch.isfinite(thresholds).all())
    ):
        raise ValueError(
            "production complete verifier thresholds must be finite shape [S]"
        )
    threshold_values = tuple(float(value) for value in thresholds.detach().cpu())
    clause_count = len(threshold_values)
    start_ns = clock_ns() if query_policy.timeout_ns is not None else 0
    deadline_ns = (
        None if query_policy.timeout_ns is None else start_ns + query_policy.timeout_ns
    )
    observed_ns = start_ns

    def deadline_expired() -> bool:
        nonlocal observed_ns
        if deadline_ns is None:
            return False
        observed_ns = clock_ns()
        return observed_ns >= deadline_ns

    clauses: list[NativeProductionCompleteVerifierClauseExecution] = []
    pending: tuple[int, ...] = ()
    skipped: tuple[int, ...] = ()
    unsafe_index: Optional[int] = None

    for clause_index in range(clause_count):
        if deadline_expired():
            pending = tuple(range(clause_index, clause_count))
            break
        objective = objectives[:, clause_index, :].contiguous()
        threshold = threshold_values[clause_index]
        search = search_native_box_counterexample(
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            policy=search_policy,
        )
        if deadline_expired():
            pending = tuple(range(clause_index, clause_count))
            break
        clause_config = replace(queue_config, threshold=threshold)
        queue = execute_native_production_relu_split_bab(
            module,
            input_spec,
            linear_spec_C=objective,
            run_id=f"{query_id}:clause:{clause_index:04d}",
            config=clause_config,
            optimizer_policy=optimizer_policy,
            relu_pre_override=relu_pre_override,
            intermediate_bound_source=intermediate_bound_source,
        )
        if deadline_expired():
            pending = tuple(range(clause_index, clause_count))
            break
        root_id = queue.trace.evaluations[0].node.node_id
        verdict = derive_native_property_verdict(
            module,
            input_spec,
            linear_spec_C=objective,
            queue_execution=queue,
            candidate_counterexamples=((root_id, search.best_input),),
        )
        clause_trace = CompleteVerifierClauseTrace(
            clause_index=clause_index,
            objective_hash=tensor_content_hash(objective),
            threshold=threshold,
            search_trace_hash=search.trace.stable_hash(),
            queue_trace_hash=queue.trace.stable_hash(),
            verdict_trace_hash=verdict.trace.stable_hash(queue.trace),
            status=verdict.trace.status,
            search_counterexample_found=search.trace.counterexample_found,
            queue_execution_mode="production_prepared",
        )
        clause = NativeProductionCompleteVerifierClauseExecution(
            trace=clause_trace,
            search=search,
            queue=queue,
            verdict=verdict,
        )
        clause.validate_against(module, input_spec, linear_spec_C=objective)
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
    elif pending:
        status = "unknown"
        reason = "query_deadline_exhausted"
    elif unresolved:
        status = "unknown"
        reason = "one_or_more_clauses_unresolved"
    else:
        status = "verified"
        reason = "all_clauses_verified"
    elapsed_ns = 0 if deadline_ns is None else max(0, observed_ns - start_ns)
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
        elapsed_ns=elapsed_ns,
        execution_mode="production_prepared",
    )
    execution = NativeProductionCompleteVerifierQueryExecution(
        trace=trace,
        clauses=tuple(clauses),
        search_policy=search_policy,
        queue_config=queue_config,
        optimizer_policy=optimizer_policy,
    )
    execution.validate_against(module, input_spec, linear_spec_C=objectives)
    return execution


__all__ = [
    "NativeProductionCompleteVerifierClauseExecution",
    "NativeProductionCompleteVerifierQueryExecution",
    "execute_native_production_complete_verifier_query",
]
