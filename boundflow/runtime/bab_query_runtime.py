"""Synchronous same-solver adapter over the PR-13 query BatchManager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..ir.task import BFTaskModule
from .bab_query import (
    BoundQueryRequest,
    BoundQueryResult,
    QueryBatch,
    QueryCompatibilityKey,
    execute_query_batch_reference,
)
from .compiler_query_runtime import (
    CompilerBoundQueryRequest,
    CompilerQueryCapabilityError,
    TypedCompilerQueryResult,
    TypedCompilerQueryRuntime,
)
from .query_batcher import BatchPolicy, DynamicBatchManager
from .query_executor import execute_alpha_beta_query_batch


@dataclass(frozen=True)
class SameSolverRuntimeConfig:
    """Runtime limits that do not alter the host solver algorithm."""

    max_batch_size: int
    memory_budget_bytes: int
    max_wait_us: int = 0
    minimum_fill_ratio: float = 1.0
    allow_legacy_alpha_beta: bool = False

    def batch_policy(self) -> BatchPolicy:
        """Build and validate the shared scheduler policy."""

        policy = BatchPolicy(
            max_batch_size=self.max_batch_size,
            memory_budget_bytes=self.memory_budget_bytes,
            max_wait_us=self.max_wait_us,
            minimum_fill_ratio=self.minimum_fill_ratio,
        )
        policy.validate()
        return policy


class SameSolverQueryRuntime:
    """Execute solver-provided query groups without owning search control flow."""

    def __init__(self, config: SameSolverRuntimeConfig) -> None:
        self.config = config
        self.batch_manager = DynamicBatchManager(config.batch_policy())
        self._dispatch_plan_cache: dict[QueryCompatibilityKey, str] = {}
        self._dispatch_plan_cache_hits = 0
        self._dispatch_plan_cache_misses = 0
        self._dense_capability_dispatches = 0

    def _resolve_dispatch_plan(self, key: QueryCompatibilityKey) -> str:
        """Cache capability routing separately from compiled-kernel caching."""

        cached = self._dispatch_plan_cache.get(key)
        if cached is not None:
            self._dispatch_plan_cache_hits += 1
            return cached
        self._dispatch_plan_cache_misses += 1
        capability = key.backend_capability_class
        if capability not in {"alpha_beta_dense_split", "alpha_dense"}:
            raise ValueError(
                f"same-solver runtime rejects unsupported capability: {capability}"
            )
        self._dispatch_plan_cache[key] = capability
        return capability

    def execute(
        self,
        module: BFTaskModule,
        requests: Sequence[BoundQueryRequest],
        *,
        now_us: Optional[int] = None,
    ) -> list[tuple[str, BoundQueryResult]]:
        """Synchronously execute one solver-selected group in original order."""

        if not requests:
            return []
        if not self.config.allow_legacy_alpha_beta:
            raise CompilerQueryCapabilityError(
                "legacy PR-13 alpha/beta executor is historical opt-in only; "
                "set allow_legacy_alpha_beta=True for validated-reduced replay. "
                "PR-14 external mismatch remains NO-GO for compiler fallback"
            )
        for request in requests:
            self.batch_manager.submit(request, now_us=now_us)
        results: list[tuple[str, BoundQueryResult]] = []
        for batch in self.batch_manager.pop_ready(now_us=now_us, force=True):
            capability = self._resolve_dispatch_plan(batch.key)
            if capability == "alpha_beta_dense_split":

                def executor(candidate):
                    return execute_alpha_beta_query_batch(module, candidate)

            elif capability == "alpha_dense":

                def executor(candidate):
                    return execute_query_batch_reference(module, candidate)

            else:
                raise AssertionError(f"unreachable dispatch capability: {capability}")
            self._dense_capability_dispatches += 1
            results.extend(
                self.batch_manager.execute_batch_with_oom_retry(batch, executor)
            )
        expected_ids = [request.query.query_id for request in requests]
        result_by_id = dict(results)
        if set(result_by_id) != set(expected_ids):
            raise ValueError("same-solver runtime lost or invented query results")
        return [(query_id, result_by_id[query_id]) for query_id in expected_ids]

    def audit(self) -> dict[str, object]:
        """Return scheduler/result accounting for solver-level comparison."""

        audit = dict(self.batch_manager.audit())
        audit.update(
            {
                "dispatch_plan_cache_hits": self._dispatch_plan_cache_hits,
                "dispatch_plan_cache_misses": self._dispatch_plan_cache_misses,
                "dispatch_plan_cache_entries": len(self._dispatch_plan_cache),
                "dense_capability_dispatches": self._dense_capability_dispatches,
                # α/β and split queries are intentionally not eligible for the
                # PR-12 plain-CROWN compiler/Planner in this reduced closure.
                "compiled_plan_cache_applicable": False,
                "pr12_planner_dispatches": 0,
                "legacy_alpha_beta_opt_in": self.config.allow_legacy_alpha_beta,
                "typed_compiler_ir_consumed": False,
            }
        )
        return audit


class CompilerSameSolverQueryRuntime:
    """PR-13 batch scheduling whose only executor is typed compiler IR."""

    def __init__(
        self,
        config: SameSolverRuntimeConfig,
        compiler_runtime: TypedCompilerQueryRuntime,
    ) -> None:
        self.config = config
        self.compiler_runtime = compiler_runtime
        self.batch_manager = DynamicBatchManager(config.batch_policy())
        self._compatibility_dispatches = 0

    def execute(
        self,
        requests: Sequence[CompilerBoundQueryRequest],
        *,
        now_us: Optional[int] = None,
    ) -> list[tuple[str, TypedCompilerQueryResult]]:
        """Batch compatible query identities and restore exact caller order."""

        if not requests:
            return []
        by_id: dict[str, CompilerBoundQueryRequest] = {}
        for request in requests:
            request.validate()
            query_id = request.query_request.query.query_id
            if query_id in by_id:
                raise ValueError(f"duplicate compiler query submission: {query_id}")
            by_id[query_id] = request
            self.batch_manager.submit(request.query_request, now_us=now_us)
        typed_results: dict[str, TypedCompilerQueryResult] = {}
        for batch in self.batch_manager.pop_ready(now_us=now_us, force=True):
            if batch.key.backend_capability_class != "plain_crown_typed_ir":
                raise ValueError(
                    "compiler same-solver runtime rejects non-typed capability: "
                    f"{batch.key.backend_capability_class}"
                )

            def executor(
                candidate: QueryBatch,
            ) -> list[tuple[str, BoundQueryResult]]:
                compiler_requests = tuple(
                    by_id[item.query.query_id] for item in candidate.requests
                )
                results = self.compiler_runtime.execute_bound_queries(compiler_requests)
                for result in results:
                    typed_results[result.query_id] = result
                return [
                    (
                        result.query_id,
                        BoundQueryResult(
                            status="ok",
                            lower=result.bounds.lower,
                            upper=result.bounds.upper,
                            branch=None,
                            alpha_state_version=None,
                            beta_state_version=None,
                        ),
                    )
                    for result in results
                ]

            self._compatibility_dispatches += 1
            self.batch_manager.execute_batch_with_oom_retry(batch, executor)
        expected_ids = tuple(
            request.query_request.query.query_id for request in requests
        )
        if set(typed_results) != set(expected_ids):
            raise ValueError("compiler same-solver runtime lost or invented results")
        return [(query_id, typed_results[query_id]) for query_id in expected_ids]

    def audit(self) -> dict[str, object]:
        """Return PR-13 scheduler and typed compiler accounting together."""

        return {
            **self.batch_manager.audit(),
            "compatibility_dispatches": self._compatibility_dispatches,
            "compiler": self.compiler_runtime.audit(),
            "legacy_executor_dispatches": 0,
        }


__all__ = [
    "CompilerSameSolverQueryRuntime",
    "SameSolverQueryRuntime",
    "SameSolverRuntimeConfig",
]
