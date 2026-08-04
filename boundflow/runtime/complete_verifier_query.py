"""Multi-clause complete verifier query control over native BaB verdicts."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=invalid-name,too-many-statements,too-many-branches,duplicate-code

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import time
from typing import Callable, Literal, Mapping, Optional

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.bound import IntermediateBoundSource
from ..ir.task import BFTaskModule
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_candidate_search import (
    NativeCandidateSearchExecution,
    NativeProjectedGradientSearchPolicy,
    search_native_box_counterexample,
)
from .native_optimized_relu_split_bab_runtime import (
    NativeOptimizedReluSplitBabExecution,
    execute_native_optimized_relu_split_bab,
)
from .native_property_verdict import (
    NativePropertyVerdictExecution,
    derive_native_property_verdict,
)
from .native_relu_split_bab_runtime import NativeReluSplitBabConfig
from .task_executor import InputSpec

COMPLETE_VERIFIER_QUERY_SCHEMA_VERSION = "boundflow.complete-verifier-query/v1"
COMPLETE_VERIFIER_QUERY_COMPILER_VERSION = "boundflow.complete-verifier-query/v1"
CompleteQueryStatus = Literal["verified", "unsafe", "unknown"]
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
class CompleteVerifierQueryPolicy:
    """Query-level clause order, short-circuit, and cooperative deadline policy."""

    timeout_ns: Optional[int] = None

    def validate(self) -> None:
        if self.timeout_ns is not None and self.timeout_ns < 1:
            raise ValueError("complete verifier query timeout must be positive")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "timeout_ns": self.timeout_ns,
            "clause_order": "ascending_index",
            "property_aggregation": "conjunction",
            "unsafe_short_circuit": True,
            "deadline_enforcement": "cooperative_stage_boundaries",
        }


@dataclass(frozen=True)
class CompleteVerifierClauseTrace:
    """One completed search -> queue -> sound-verdict clause pipeline."""

    clause_index: int
    objective_hash: str
    threshold: float
    search_trace_hash: str
    queue_trace_hash: str
    verdict_trace_hash: str
    status: CompleteQueryStatus
    search_counterexample_found: bool
    queue_execution_mode: str = "audit_validation"

    def validate(self) -> None:
        if (
            self.clause_index < 0
            or not _is_sha256(self.objective_hash)
            or not _is_sha256(self.search_trace_hash)
            or not _is_sha256(self.queue_trace_hash)
            or not _is_sha256(self.verdict_trace_hash)
            or self.status not in {"verified", "unsafe", "unknown"}
            or self.queue_execution_mode
            not in {"audit_validation", "production_prepared"}
            or not torch.isfinite(torch.tensor(self.threshold)).item()
            or self.search_counterexample_found != (self.status == "unsafe")
        ):
            raise ValueError("complete verifier clause trace is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "clause_index": self.clause_index,
            "objective_hash": self.objective_hash,
            "threshold": self.threshold,
            "search_trace_hash": self.search_trace_hash,
            "queue_trace_hash": self.queue_trace_hash,
            "verdict_trace_hash": self.verdict_trace_hash,
            "status": self.status,
            "search_counterexample_found": self.search_counterexample_found,
        }
        if self.queue_execution_mode != "audit_validation":
            payload["queue_execution_mode"] = self.queue_execution_mode
        return payload


@dataclass(frozen=True)
class CompleteVerifierClauseExecution:
    """Non-serialized executions backing one completed clause trace."""

    trace: CompleteVerifierClauseTrace
    search: NativeCandidateSearchExecution
    queue: NativeOptimizedReluSplitBabExecution
    verdict: NativePropertyVerdictExecution

    def validate_against(
        self,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        linear_spec_C: torch.Tensor,
    ) -> None:
        self.trace.validate()
        self.search.validate_against(
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
        )
        self.queue.validate()
        self.verdict.validate_against(
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
            queue_execution=self.queue,
        )
        if (
            self.trace.objective_hash != tensor_content_hash(linear_spec_C)
            or self.trace.threshold != self.queue.trace.config.threshold
            or self.trace.search_trace_hash != self.search.trace.stable_hash()
            or self.trace.queue_trace_hash != self.queue.trace.stable_hash()
            or self.trace.verdict_trace_hash
            != self.verdict.trace.stable_hash(self.queue.trace)
            or self.trace.status != self.verdict.trace.status
            or self.trace.search_counterexample_found
            != self.search.trace.counterexample_found
        ):
            raise ValueError("complete verifier clause execution identity differs")


@dataclass(frozen=True)
class CompleteVerifierQueryTrace:
    """Replayable conjunction result with unresolved/pending clause accounting."""

    query_id: str
    status: CompleteQueryStatus
    reason: str
    objective_matrix_hash: str
    thresholds: tuple[float, ...]
    query_policy: CompleteVerifierQueryPolicy
    search_policy_hash: str
    optimizer_policy_hash: str
    queue_config_hash: str
    completed_clauses: tuple[CompleteVerifierClauseTrace, ...]
    unresolved_clause_indices: tuple[int, ...]
    pending_clause_indices: tuple[int, ...]
    skipped_after_unsafe_clause_indices: tuple[int, ...]
    unsafe_clause_index: Optional[int]
    elapsed_ns: int
    performance_claimed: bool = False
    schema_version: str = COMPLETE_VERIFIER_QUERY_SCHEMA_VERSION
    execution_mode: str = "audit_validation"

    def validate(self) -> None:  # pylint: disable=too-many-statements
        self.query_policy.validate()
        clause_count = len(self.thresholds)
        completed_indices = tuple(item.clause_index for item in self.completed_clauses)
        all_indices = set(range(clause_count))
        unresolved = set(self.unresolved_clause_indices)
        pending = set(self.pending_clause_indices)
        skipped = set(self.skipped_after_unsafe_clause_indices)
        if (
            self.schema_version != COMPLETE_VERIFIER_QUERY_SCHEMA_VERSION
            or not self.query_id
            or self.status not in {"verified", "unsafe", "unknown"}
            or not self.reason
            or not _is_sha256(self.objective_matrix_hash)
            or not _is_sha256(self.search_policy_hash)
            or not _is_sha256(self.optimizer_policy_hash)
            or not _is_sha256(self.queue_config_hash)
            or clause_count < 1
            or not all(
                torch.isfinite(torch.tensor(value)).item() for value in self.thresholds
            )
            or completed_indices != tuple(range(len(completed_indices)))
            or any(
                item.threshold != self.thresholds[item.clause_index]
                for item in self.completed_clauses
            )
            or len(unresolved) != len(self.unresolved_clause_indices)
            or len(pending) != len(self.pending_clause_indices)
            or len(skipped) != len(self.skipped_after_unsafe_clause_indices)
            or not unresolved <= set(completed_indices)
            or not pending <= all_indices
            or not skipped <= all_indices
            or pending & skipped
            or self.elapsed_ns < 0
            or (self.query_policy.timeout_ns is None and self.elapsed_ns != 0)
            or self.performance_claimed is not False
            or self.execution_mode not in {"audit_validation", "production_prepared"}
            or any(
                item.queue_execution_mode != self.execution_mode
                for item in self.completed_clauses
            )
        ):
            raise ValueError(
                "complete verifier query trace header/accounting is invalid"
            )
        for clause in self.completed_clauses:
            clause.validate()
        expected_unresolved = {
            item.clause_index
            for item in self.completed_clauses
            if item.status == "unknown"
        }
        if unresolved != expected_unresolved:
            raise ValueError("complete verifier unresolved clause accounting differs")

        if self.status == "verified":
            if (
                self.reason != "all_clauses_verified"
                or len(self.completed_clauses) != clause_count
                or any(item.status != "verified" for item in self.completed_clauses)
                or unresolved
                or pending
                or skipped
                or self.unsafe_clause_index is not None
            ):
                raise ValueError("verified query lacks all-clause proof closure")
        elif self.status == "unsafe":
            if (
                self.reason != "concrete_counterexample_clause"
                or self.unsafe_clause_index is None
                or not self.completed_clauses
                or self.completed_clauses[-1].clause_index != self.unsafe_clause_index
                or self.completed_clauses[-1].status != "unsafe"
                or pending
                or skipped != set(range(self.unsafe_clause_index + 1, clause_count))
            ):
                raise ValueError("unsafe query lacks exact short-circuit accounting")
        elif (
            self.unsafe_clause_index is not None
            or skipped
            or not (unresolved or pending)
            or self.reason
            != (
                "query_deadline_exhausted"
                if pending
                else "one_or_more_clauses_unresolved"
            )
        ):
            raise ValueError("unknown query lacks unresolved/deadline accounting")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "compiler_version": COMPLETE_VERIFIER_QUERY_COMPILER_VERSION,
            "query_id": self.query_id,
            "status": self.status,
            "reason": self.reason,
            "performance_claimed": self.performance_claimed,
            "objective_matrix_hash": self.objective_matrix_hash,
            "thresholds": list(self.thresholds),
            "query_policy": self.query_policy.to_dict(),
            "search_policy_hash": self.search_policy_hash,
            "optimizer_policy_hash": self.optimizer_policy_hash,
            "queue_config_hash": self.queue_config_hash,
            "completed_clauses": [item.to_dict() for item in self.completed_clauses],
            "unresolved_clause_indices": list(self.unresolved_clause_indices),
            "pending_clause_indices": list(self.pending_clause_indices),
            "skipped_after_unsafe_clause_indices": list(
                self.skipped_after_unsafe_clause_indices
            ),
            "unsafe_clause_index": self.unsafe_clause_index,
            "elapsed_ns": self.elapsed_ns,
            "deadline_is_cooperative": self.query_policy.timeout_ns is not None,
        }
        if self.execution_mode != "audit_validation":
            payload["execution_mode"] = self.execution_mode
        return payload

    def stable_hash(self) -> str:
        self.validate()
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class CompleteVerifierQueryExecution:
    """Query trace plus non-serialized completed clause executions."""

    trace: CompleteVerifierQueryTrace
    clauses: tuple[CompleteVerifierClauseExecution, ...]
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
            tensor_content_hash(objectives) != self.trace.objective_matrix_hash
            or len(self.clauses) != len(self.trace.completed_clauses)
            or tuple(item.trace for item in self.clauses)
            != self.trace.completed_clauses
            or self.search_policy.stable_hash() != self.trace.search_policy_hash
            or _config_hash(self.queue_config) != self.trace.queue_config_hash
            or self.optimizer_policy.stable_hash() != self.trace.optimizer_policy_hash
        ):
            raise ValueError("complete verifier query execution coverage differs")
        for clause in self.clauses:
            objective = objectives[:, clause.trace.clause_index, :].contiguous()
            clause.validate_against(
                module,
                input_spec,
                linear_spec_C=objective,
            )
            if (
                clause.search.trace.policy != self.search_policy
                or clause.queue.trace.config
                != replace(
                    self.queue_config,
                    threshold=self.trace.thresholds[clause.trace.clause_index],
                )
                or clause.queue.trace.optimizer_policy != self.optimizer_policy
            ):
                raise ValueError("complete verifier clause policy binding differs")


def _normalize_objective_matrix(linear_spec_C: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(linear_spec_C) or not torch.is_floating_point(linear_spec_C):
        raise TypeError("complete verifier objective matrix must be floating point")
    objectives = linear_spec_C.detach().contiguous()
    if objectives.dim() == 2:
        objectives = objectives.unsqueeze(0).contiguous()
    if (
        objectives.dim() != 3
        or int(objectives.shape[0]) != 1
        or int(objectives.shape[1]) < 1
    ):
        raise ValueError("complete verifier objective matrix must have shape [1,S,O]")
    if not bool(torch.isfinite(objectives).all()):
        raise ValueError("complete verifier objective matrix must be finite")
    return objectives


def _config_hash(config: NativeReluSplitBabConfig) -> str:
    payload = config.to_dict()
    payload.pop("threshold")
    return _canonical_hash(payload)


def execute_complete_verifier_query(
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
) -> CompleteVerifierQueryExecution:
    """Execute conjunction clauses sequentially with sound unsafe short-circuit."""

    if not query_id:
        raise ValueError("complete verifier query ID must be non-empty")
    module.validate()
    query_policy.validate()
    search_policy.validate()
    queue_config.validate()
    optimizer_policy.validate()
    if not isinstance(intermediate_bound_source, IntermediateBoundSource):
        raise TypeError("complete verifier intermediate-bound source is invalid")
    if (relu_pre_override is None) != (
        intermediate_bound_source == IntermediateBoundSource.LOCAL_FORWARD
    ):
        raise ValueError("complete verifier intermediate semantics/provenance differ")
    objectives = _normalize_objective_matrix(linear_spec_C)
    if (
        not torch.is_tensor(thresholds)
        or not torch.is_floating_point(thresholds)
        or thresholds.dim() != 1
        or int(thresholds.shape[0]) != int(objectives.shape[1])
        or not bool(torch.isfinite(thresholds).all())
    ):
        raise ValueError("complete verifier thresholds must be finite shape [S]")
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

    clause_executions: list[CompleteVerifierClauseExecution] = []
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
        queue = execute_native_optimized_relu_split_bab(
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
        )
        clause_execution = CompleteVerifierClauseExecution(
            trace=clause_trace,
            search=search,
            queue=queue,
            verdict=verdict,
        )
        clause_execution.validate_against(
            module,
            input_spec,
            linear_spec_C=objective,
        )
        clause_executions.append(clause_execution)
        if verdict.trace.status == "unsafe":
            unsafe_index = clause_index
            skipped = tuple(range(clause_index + 1, clause_count))
            break

    unresolved = tuple(
        item.trace.clause_index
        for item in clause_executions
        if item.trace.status == "unknown"
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
        completed_clauses=tuple(item.trace for item in clause_executions),
        unresolved_clause_indices=unresolved,
        pending_clause_indices=pending,
        skipped_after_unsafe_clause_indices=skipped,
        unsafe_clause_index=unsafe_index,
        elapsed_ns=elapsed_ns,
    )
    execution = CompleteVerifierQueryExecution(
        trace=trace,
        clauses=tuple(clause_executions),
        search_policy=search_policy,
        queue_config=queue_config,
        optimizer_policy=optimizer_policy,
    )
    execution.validate_against(
        module,
        input_spec,
        linear_spec_C=objectives,
    )
    return execution


__all__ = [
    "COMPLETE_VERIFIER_QUERY_COMPILER_VERSION",
    "COMPLETE_VERIFIER_QUERY_SCHEMA_VERSION",
    "CompleteQueryStatus",
    "CompleteVerifierClauseExecution",
    "CompleteVerifierClauseTrace",
    "CompleteVerifierQueryExecution",
    "CompleteVerifierQueryPolicy",
    "CompleteVerifierQueryTrace",
    "execute_complete_verifier_query",
]
