"""Prepared production root-bound conjunction over optimizer Task/Schedule IR."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,duplicate-code,invalid-name

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal, Mapping, Optional

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.bound import IntermediateBoundSource
from ..ir.task import BFTaskModule
from .crown_ibp import _forward_ibp_trace_mlp
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .native_alpha_beta_optimizer_schedule import (
    NativePreparedOptimizerProgram,
    NativeProductionOptimizerResult,
    compile_native_alpha_beta_optimizer_program,
    execute_prepared_native_alpha_beta_optimizer_program,
)
from .native_candidate_search import (
    NativeCandidateSearchExecution,
    NativeProjectedGradientSearchPolicy,
    search_native_box_counterexample,
)
from .task_executor import InputSpec

NATIVE_PREPARED_COMPLETE_QUERY_SCHEMA_VERSION = (
    "boundflow.native-prepared-complete-query/v1"
)
NATIVE_PREPARED_COMPLETE_QUERY_COMPILER_VERSION = (
    "boundflow.native-prepared-complete-query/v1"
)
PreparedClauseStatus = Literal["verified", "unsafe", "unknown"]
PreparedQueryStatus = Literal["verified", "unsafe", "unknown"]


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalize_objectives(linear_spec_C: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(linear_spec_C) or not torch.is_floating_point(linear_spec_C):
        raise TypeError("prepared complete query objectives must be floating point")
    objectives = linear_spec_C.detach().contiguous()
    if objectives.dim() == 2:
        objectives = objectives.unsqueeze(0).contiguous()
    if (
        objectives.dim() != 3
        or int(objectives.shape[0]) != 1
        or int(objectives.shape[1]) < 1
        or not bool(torch.isfinite(objectives).all())
    ):
        raise ValueError("prepared complete query objectives require finite [1,S,O]")
    return objectives


def _normalize_thresholds(
    thresholds: torch.Tensor, *, clause_count: int
) -> tuple[float, ...]:
    if (
        not torch.is_tensor(thresholds)
        or not torch.is_floating_point(thresholds)
        or thresholds.dim() != 1
        or int(thresholds.shape[0]) != clause_count
        or not bool(torch.isfinite(thresholds).all())
    ):
        raise ValueError("prepared complete query thresholds require finite [S]")
    return tuple(float(value) for value in thresholds.detach().cpu())


@dataclass(frozen=True)
class NativePreparedRootClause:
    """One exact objective and its prevalidated optimizer program."""

    clause_index: int
    threshold: float
    objective_hash: str
    prepared_optimizer: NativePreparedOptimizerProgram

    def validate(self, *, query_id: str) -> None:
        expected_plan_id = f"{query_id}:clause:{self.clause_index:04d}:prepared-root"
        if (
            self.clause_index < 0
            or not torch.isfinite(torch.tensor(self.threshold)).item()
            or not _is_sha256(self.objective_hash)
            or self.prepared_optimizer.program.plan.plan_id != expected_plan_id
        ):
            raise ValueError("prepared complete query clause differs")


@dataclass(frozen=True)
class NativePreparedCompleteQuery:
    """Exact multi-clause production capsule prepared outside steady state."""

    query_id: str
    module: BFTaskModule
    objective_matrix_hash: str
    thresholds: tuple[float, ...]
    search_policy: NativeProjectedGradientSearchPolicy
    optimizer_policy: NativeAlphaBetaOptimizerPolicy
    intermediate_bound_source: IntermediateBoundSource
    clauses: tuple[NativePreparedRootClause, ...]
    schema_version: str = NATIVE_PREPARED_COMPLETE_QUERY_SCHEMA_VERSION

    def validate(self) -> None:
        self.search_policy.validate()
        self.optimizer_policy.validate()
        if (
            self.schema_version != NATIVE_PREPARED_COMPLETE_QUERY_SCHEMA_VERSION
            or not self.query_id
            or not _is_sha256(self.objective_matrix_hash)
            or not isinstance(self.intermediate_bound_source, IntermediateBoundSource)
            or not self.clauses
            or len(self.thresholds) != len(self.clauses)
            or tuple(item.clause_index for item in self.clauses)
            != tuple(range(len(self.clauses)))
            or tuple(item.threshold for item in self.clauses) != self.thresholds
        ):
            raise ValueError("prepared complete query capsule differs")
        for clause in self.clauses:
            clause.validate(query_id=self.query_id)
            if clause.prepared_optimizer.program.policy != self.optimizer_policy:
                raise ValueError("prepared complete query optimizer policy differs")

    def require_identity(
        self,
        module: BFTaskModule,
        *,
        linear_spec_C: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> torch.Tensor:
        objectives = _normalize_objectives(linear_spec_C)
        threshold_values = _normalize_thresholds(
            thresholds, clause_count=int(objectives.shape[1])
        )
        if (
            module is not self.module
            or tensor_content_hash(objectives) != self.objective_matrix_hash
            or threshold_values != self.thresholds
            or int(objectives.shape[1]) != len(self.clauses)
        ):
            raise ValueError("prepared complete query exact identity differs")
        return objectives


@dataclass(frozen=True)
class NativePreparedClauseTrace:
    """Production clause summary without audit action/tensor hash chains."""

    clause_index: int
    objective_hash: str
    threshold: float
    status: PreparedClauseStatus
    search_trace_hash: str
    candidate_best: float
    counterexample_input_hash: Optional[str]
    lower: float
    upper: float
    lower_hash: str
    upper_hash: str
    best_iteration: int
    prepared_program_hashes: tuple[tuple[str, str], ...]
    audit_hash_chain_constructed: bool = False
    selected_native_reexecution: bool = False

    def validate(self) -> None:
        program_hashes = dict(self.prepared_program_hashes)
        if (
            self.clause_index < 0
            or not _is_sha256(self.objective_hash)
            or not _is_sha256(self.search_trace_hash)
            or not _is_sha256(self.lower_hash)
            or not _is_sha256(self.upper_hash)
            or set(program_hashes)
            != {
                "optimizer_plan_hash",
                "optimizer_task_module_hash",
                "optimizer_schedule_hash",
            }
            or len(program_hashes) != len(self.prepared_program_hashes)
            or any(not _is_sha256(value) for value in program_hashes.values())
            or self.status not in {"verified", "unsafe", "unknown"}
            or self.best_iteration < 0
            or self.lower > self.upper
            or not all(
                torch.isfinite(torch.tensor(value)).item()
                for value in (
                    self.threshold,
                    self.candidate_best,
                    self.lower,
                    self.upper,
                )
            )
            or self.audit_hash_chain_constructed is not False
            or self.selected_native_reexecution is not False
        ):
            raise ValueError("prepared production clause trace differs")
        expected = (
            "unsafe"
            if self.counterexample_input_hash is not None
            else ("verified" if self.lower >= self.threshold else "unknown")
        )
        if self.status != expected or (
            self.counterexample_input_hash is not None
            and (
                not _is_sha256(self.counterexample_input_hash)
                or self.candidate_best >= self.threshold
            )
        ):
            raise ValueError("prepared production clause status differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "clause_index": self.clause_index,
            "objective_hash": self.objective_hash,
            "threshold": self.threshold,
            "status": self.status,
            "search_trace_hash": self.search_trace_hash,
            "candidate_best": self.candidate_best,
            "counterexample_input_hash": self.counterexample_input_hash,
            "lower": self.lower,
            "upper": self.upper,
            "lower_hash": self.lower_hash,
            "upper_hash": self.upper_hash,
            "best_iteration": self.best_iteration,
            "prepared_program_hashes": dict(self.prepared_program_hashes),
            "audit_hash_chain_constructed": self.audit_hash_chain_constructed,
            "selected_native_reexecution": self.selected_native_reexecution,
        }


@dataclass(frozen=True)
class NativePreparedCompleteQueryTrace:
    """Typed production query result with explicit audit omissions."""

    query_id: str
    status: PreparedQueryStatus
    reason: str
    objective_matrix_hash: str
    thresholds: tuple[float, ...]
    search_policy_hash: str
    optimizer_policy_hash: str
    intermediate_bound_source: IntermediateBoundSource
    completed_clauses: tuple[NativePreparedClauseTrace, ...]
    unresolved_clause_indices: tuple[int, ...]
    skipped_after_unsafe_clause_indices: tuple[int, ...]
    unsafe_clause_index: Optional[int]
    performance_claimed: bool = False
    schema_version: str = NATIVE_PREPARED_COMPLETE_QUERY_SCHEMA_VERSION

    def validate(self) -> None:
        for clause in self.completed_clauses:
            clause.validate()
        completed = tuple(item.clause_index for item in self.completed_clauses)
        unresolved = tuple(
            item.clause_index
            for item in self.completed_clauses
            if item.status == "unknown"
        )
        if (
            self.schema_version != NATIVE_PREPARED_COMPLETE_QUERY_SCHEMA_VERSION
            or not self.query_id
            or self.status not in {"verified", "unsafe", "unknown"}
            or not self.reason
            or not _is_sha256(self.objective_matrix_hash)
            or not _is_sha256(self.search_policy_hash)
            or not _is_sha256(self.optimizer_policy_hash)
            or not isinstance(self.intermediate_bound_source, IntermediateBoundSource)
            or not self.thresholds
            or completed != tuple(range(len(completed)))
            or unresolved != self.unresolved_clause_indices
            or self.performance_claimed is not False
        ):
            raise ValueError("prepared production query trace differs")
        clause_count = len(self.thresholds)
        if self.status == "verified":
            if (
                self.reason != "all_root_clauses_verified"
                or len(completed) != clause_count
                or unresolved
                or self.skipped_after_unsafe_clause_indices
                or self.unsafe_clause_index is not None
                or any(item.status != "verified" for item in self.completed_clauses)
            ):
                raise ValueError("prepared verified query is not closed")
        elif self.status == "unsafe":
            expected_skipped = tuple(
                range((self.unsafe_clause_index or 0) + 1, clause_count)
            )
            if (
                self.reason != "concrete_counterexample_replayed"
                or self.unsafe_clause_index is None
                or self.completed_clauses[-1].clause_index != self.unsafe_clause_index
                or self.completed_clauses[-1].status != "unsafe"
                or self.skipped_after_unsafe_clause_indices != expected_skipped
            ):
                raise ValueError("prepared unsafe query accounting differs")
        elif (
            self.reason != "one_or_more_root_clauses_unresolved"
            or len(completed) != clause_count
            or not unresolved
            or self.skipped_after_unsafe_clause_indices
            or self.unsafe_clause_index is not None
        ):
            raise ValueError("prepared unknown query accounting differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "compiler_version": NATIVE_PREPARED_COMPLETE_QUERY_COMPILER_VERSION,
            "query_id": self.query_id,
            "status": self.status,
            "reason": self.reason,
            "performance_claimed": self.performance_claimed,
            "execution_mode": "prepared_production_root_only",
            "objective_matrix_hash": self.objective_matrix_hash,
            "thresholds": list(self.thresholds),
            "search_policy_hash": self.search_policy_hash,
            "optimizer_policy_hash": self.optimizer_policy_hash,
            "intermediate_bound_source": self.intermediate_bound_source.value,
            "completed_clauses": [item.to_dict() for item in self.completed_clauses],
            "unresolved_clause_indices": list(self.unresolved_clause_indices),
            "skipped_after_unsafe_clause_indices": list(
                self.skipped_after_unsafe_clause_indices
            ),
            "unsafe_clause_index": self.unsafe_clause_index,
            "audit_hash_chain_constructed": False,
            "selected_native_reexecution": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativePreparedCompleteQueryExecution:
    """Production trace plus ephemeral search and optimizer results."""

    trace: NativePreparedCompleteQueryTrace
    searches: tuple[NativeCandidateSearchExecution, ...]
    optimizer_results: tuple[NativeProductionOptimizerResult, ...]


def prepare_native_root_complete_query(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
    query_id: str,
    search_policy: NativeProjectedGradientSearchPolicy,
    optimizer_policy: NativeAlphaBetaOptimizerPolicy,
    relu_pre_override: Optional[Mapping[str, IntervalState]] = None,
    intermediate_bound_source: IntermediateBoundSource = (
        IntermediateBoundSource.LOCAL_FORWARD
    ),
) -> NativePreparedCompleteQuery:
    """Compile and validate every root objective outside steady-state execution."""

    if not query_id:
        raise ValueError("prepared complete query ID must be non-empty")
    module.validate()
    search_policy.validate()
    optimizer_policy.validate()
    if (relu_pre_override is None) != (
        intermediate_bound_source == IntermediateBoundSource.LOCAL_FORWARD
    ):
        raise ValueError("prepared complete query semantics/provenance differ")
    objectives = _normalize_objectives(linear_spec_C)
    threshold_values = _normalize_thresholds(
        thresholds, clause_count=int(objectives.shape[1])
    )
    _root_env, root_pre = _forward_ibp_trace_mlp(module, input_spec)
    root_splits = {
        name: torch.zeros_like(value.lower, dtype=torch.int8)
        for name, value in root_pre.items()
    }
    clauses: list[NativePreparedRootClause] = []
    for clause_index, threshold in enumerate(threshold_values):
        objective = objectives[:, clause_index, :].contiguous()
        program = compile_native_alpha_beta_optimizer_program(
            module,
            input_spec,
            linear_spec_C=objective,
            relu_split_state=root_splits,
            policy=optimizer_policy,
            program_id=f"{query_id}:clause:{clause_index:04d}:prepared-root",
            relu_pre_override=relu_pre_override,
            intermediate_bound_source=intermediate_bound_source,
        )
        prepared = NativePreparedOptimizerProgram.prepare(
            program,
            module,
            input_spec,
            linear_spec_C=objective,
            intermediate_bound_source=intermediate_bound_source,
        )
        clauses.append(
            NativePreparedRootClause(
                clause_index=clause_index,
                threshold=threshold,
                objective_hash=tensor_content_hash(objective),
                prepared_optimizer=prepared,
            )
        )
    result = NativePreparedCompleteQuery(
        query_id=query_id,
        module=module,
        objective_matrix_hash=tensor_content_hash(objectives),
        thresholds=threshold_values,
        search_policy=search_policy,
        optimizer_policy=optimizer_policy,
        intermediate_bound_source=intermediate_bound_source,
        clauses=tuple(clauses),
    )
    result.validate()
    return result


def execute_native_prepared_complete_query(
    prepared: NativePreparedCompleteQuery,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
) -> NativePreparedCompleteQueryExecution:
    """Execute candidate and prepared root bound stages with sound aggregation."""

    objectives = prepared.require_identity(
        module, linear_spec_C=linear_spec_C, thresholds=thresholds
    )
    traces: list[NativePreparedClauseTrace] = []
    searches: list[NativeCandidateSearchExecution] = []
    results: list[NativeProductionOptimizerResult] = []
    unsafe_index: Optional[int] = None
    skipped: tuple[int, ...] = ()
    for clause in prepared.clauses:
        objective = objectives[:, clause.clause_index, :].contiguous()
        search = search_native_box_counterexample(
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=clause.threshold,
            policy=prepared.search_policy,
        )
        production = execute_prepared_native_alpha_beta_optimizer_program(
            clause.prepared_optimizer,
            module,
            input_spec,
            linear_spec_C=objective,
            intermediate_bound_source=prepared.intermediate_bound_source,
        )
        lower = production.bounds.lower
        upper = production.bounds.upper
        if tuple(lower.shape) != (1, 1) or upper.shape != lower.shape:
            raise ValueError("prepared complete query requires one scalar per clause")
        status: PreparedClauseStatus = (
            "unsafe"
            if search.trace.counterexample_found
            else (
                "verified"
                if float(lower[0, 0].item()) >= clause.threshold
                else "unknown"
            )
        )
        trace = NativePreparedClauseTrace(
            clause_index=clause.clause_index,
            objective_hash=clause.objective_hash,
            threshold=clause.threshold,
            status=status,
            search_trace_hash=search.trace.stable_hash(),
            candidate_best=search.trace.best_objective_value,
            counterexample_input_hash=(
                tensor_content_hash(search.best_input) if status == "unsafe" else None
            ),
            lower=float(lower[0, 0].item()),
            upper=float(upper[0, 0].item()),
            lower_hash=tensor_content_hash(lower),
            upper_hash=tensor_content_hash(upper),
            best_iteration=production.best_iteration_by_domain[0],
            prepared_program_hashes=clause.prepared_optimizer.program_hashes,
        )
        trace.validate()
        traces.append(trace)
        searches.append(search)
        results.append(production)
        if status == "unsafe":
            unsafe_index = clause.clause_index
            skipped = tuple(range(clause.clause_index + 1, len(prepared.clauses)))
            break
    unresolved = tuple(item.clause_index for item in traces if item.status == "unknown")
    if unsafe_index is not None:
        query_status: PreparedQueryStatus = "unsafe"
        reason = "concrete_counterexample_replayed"
    elif unresolved:
        query_status = "unknown"
        reason = "one_or_more_root_clauses_unresolved"
    else:
        query_status = "verified"
        reason = "all_root_clauses_verified"
    query_trace = NativePreparedCompleteQueryTrace(
        query_id=prepared.query_id,
        status=query_status,
        reason=reason,
        objective_matrix_hash=prepared.objective_matrix_hash,
        thresholds=prepared.thresholds,
        search_policy_hash=prepared.search_policy.stable_hash(),
        optimizer_policy_hash=prepared.optimizer_policy.stable_hash(),
        intermediate_bound_source=prepared.intermediate_bound_source,
        completed_clauses=tuple(traces),
        unresolved_clause_indices=unresolved,
        skipped_after_unsafe_clause_indices=skipped,
        unsafe_clause_index=unsafe_index,
    )
    query_trace.validate()
    return NativePreparedCompleteQueryExecution(
        trace=query_trace,
        searches=tuple(searches),
        optimizer_results=tuple(results),
    )


__all__ = [
    "NATIVE_PREPARED_COMPLETE_QUERY_COMPILER_VERSION",
    "NATIVE_PREPARED_COMPLETE_QUERY_SCHEMA_VERSION",
    "NativePreparedClauseTrace",
    "NativePreparedCompleteQuery",
    "NativePreparedCompleteQueryExecution",
    "NativePreparedCompleteQueryTrace",
    "NativePreparedRootClause",
    "execute_native_prepared_complete_query",
    "prepare_native_root_complete_query",
]
