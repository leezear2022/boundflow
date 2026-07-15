"""Compatibility-, deadline-, and memory-aware BaB query batching."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
import math
import time
from typing import Callable, Deque, Mapping, Optional, Protocol, Sequence

import torch

from .bab_query import (
    BoundQueryRequest,
    BoundQueryResult,
    QueryBatch,
    QueryCompatibilityKey,
    build_query_batch,
)


class QueryBatchExecutor(Protocol):  # pylint: disable=too-few-public-methods
    """Physical executor used after the host scheduler forms a batch."""

    def __call__(self, batch: QueryBatch, /) -> Sequence[tuple[str, BoundQueryResult]]:
        """Execute a batch and return uniquely identified results."""


@dataclass(frozen=True)
class BatchPolicy:
    """Deterministic first-fit batching limits."""

    max_batch_size: int
    memory_budget_bytes: int
    max_wait_us: int
    minimum_fill_ratio: float = 1.0

    def validate(self) -> None:
        """Reject limits that could lose work or create unbounded queues."""

        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive")
        if self.max_wait_us < 0:
            raise ValueError("max_wait_us must be non-negative")
        if not 0.0 < self.minimum_fill_ratio <= 1.0:
            raise ValueError("minimum_fill_ratio must be in (0, 1]")

    @property
    def minimum_fill_count(self) -> int:
        """Return the smallest fill that permits an early dispatch."""

        return max(1, math.ceil(self.max_batch_size * self.minimum_fill_ratio))


@dataclass(frozen=True)
class PendingQuery:
    """Owned request plus host scheduling timestamps."""

    request: BoundQueryRequest
    submit_sequence: int
    submitted_us: int
    deadline_us: Optional[int]
    estimated_bytes: int


@dataclass
class BatchRuntimeMetrics:  # pylint: disable=too-many-instance-attributes
    """Observable counters and samples required by PR-13B/D."""

    submitted_queries: int = 0
    emitted_queries: int = 0
    completed_queries: int = 0
    emitted_batches: int = 0
    duplicate_submissions: int = 0
    invalid_results: int = 0
    timeout_flushes: int = 0
    deadline_flushes: int = 0
    fill_flushes: int = 0
    force_flushes: int = 0
    memory_limited_batches: int = 0
    oversize_singletons: int = 0
    oom_events: int = 0
    oom_splits: int = 0
    queue_wait_us: list[int] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    execution_us: list[int] = field(default_factory=list)

    def snapshot(
        self, *, pending_queries: int, max_batch_size: int
    ) -> dict[str, object]:
        """Return JSON-safe counters and p50/p90/p99 distributions."""

        return {
            "submitted_queries": self.submitted_queries,
            "emitted_queries": self.emitted_queries,
            "completed_queries": self.completed_queries,
            "pending_queries": int(pending_queries),
            "emitted_batches": self.emitted_batches,
            "duplicate_submissions": self.duplicate_submissions,
            "invalid_results": self.invalid_results,
            "timeout_flushes": self.timeout_flushes,
            "deadline_flushes": self.deadline_flushes,
            "fill_flushes": self.fill_flushes,
            "force_flushes": self.force_flushes,
            "memory_limited_batches": self.memory_limited_batches,
            "oversize_singletons": self.oversize_singletons,
            "oom_events": self.oom_events,
            "oom_splits": self.oom_splits,
            "average_batch_size": _mean(self.batch_sizes),
            "average_batch_fill_ratio": (
                0.0
                if not self.batch_sizes
                else _mean(self.batch_sizes) / float(max_batch_size)
            ),
            "queue_wait_us_p50": _percentile(self.queue_wait_us, 0.50),
            "queue_wait_us_p90": _percentile(self.queue_wait_us, 0.90),
            "queue_wait_us_p99": _percentile(self.queue_wait_us, 0.99),
            "execution_us_p50": _percentile(self.execution_us, 0.50),
            "execution_us_p90": _percentile(self.execution_us, 0.90),
            "execution_us_p99": _percentile(self.execution_us, 0.99),
        }


def _mean(values: Sequence[int]) -> float:
    return 0.0 if not values else float(sum(values)) / float(len(values))


def _percentile(values: Sequence[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return float(ordered[index])


def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel()) * int(value.element_size())


def estimate_request_bytes(request: BoundQueryRequest) -> int:
    """Estimate owned dynamic tensor bytes without claiming allocator peak."""

    payload = request.payload
    total = _tensor_bytes(payload.input_spec.center)
    if payload.linear_spec_c is not None:
        total += _tensor_bytes(payload.linear_spec_c)
    for values in (
        payload.split_by_relu_input,
        payload.warm_alpha_by_relu_input,
        payload.warm_beta_by_relu_input,
    ):
        total += sum(_tensor_bytes(tensor) for tensor in values.values())
    return total


def _is_oom(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


class DynamicBatchManager:  # pylint: disable=too-many-instance-attributes
    """Bucket compatible queries and dispatch under wait/memory constraints."""

    def __init__(
        self,
        policy: BatchPolicy,
        *,
        estimator: Callable[[BoundQueryRequest], int] = estimate_request_bytes,
        clock_us: Callable[[], int] = lambda: time.monotonic_ns() // 1_000,
    ) -> None:
        policy.validate()
        self.policy = policy
        self._estimator = estimator
        self._clock_us = clock_us
        self._buckets: "OrderedDict[QueryCompatibilityKey, Deque[PendingQuery]]" = (
            OrderedDict()
        )
        self._submitted_ids: set[str] = set()
        self._completed_ids: set[str] = set()
        self._next_sequence = 0
        self.metrics = BatchRuntimeMetrics()

    @property
    def pending_count(self) -> int:
        """Return queued requests not yet emitted."""

        return sum(len(bucket) for bucket in self._buckets.values())

    @property
    def next_wakeup_us(self) -> Optional[int]:
        """Return the earliest timeout/deadline at which the host should poll."""

        wakeups: list[int] = []
        for bucket in self._buckets.values():
            if not bucket:
                continue
            wakeups.append(bucket[0].submitted_us + self.policy.max_wait_us)
            wakeups.extend(
                pending.deadline_us
                for pending in bucket
                if pending.deadline_us is not None
            )
        return min(wakeups) if wakeups else None

    def submit(
        self,
        request: BoundQueryRequest,
        *,
        deadline_us: Optional[int] = None,
        now_us: Optional[int] = None,
    ) -> None:
        """Append one already-owned immutable request to its exact bucket."""

        request.query.validate()
        query_id = request.query.query_id
        if query_id in self._submitted_ids:
            self.metrics.duplicate_submissions += 1
            raise ValueError(f"duplicate query submission: {query_id}")
        submitted_us = self._clock_us() if now_us is None else int(now_us)
        if deadline_us is not None and int(deadline_us) < submitted_us:
            raise ValueError("deadline_us cannot precede submission")
        estimated_bytes = int(self._estimator(request))
        if estimated_bytes < 0:
            raise ValueError("request byte estimate must be non-negative")
        pending = PendingQuery(
            request=request,
            submit_sequence=self._next_sequence,
            submitted_us=submitted_us,
            deadline_us=None if deadline_us is None else int(deadline_us),
            estimated_bytes=estimated_bytes,
        )
        self._buckets.setdefault(request.query.compatibility_key, deque()).append(
            pending
        )
        self._submitted_ids.add(query_id)
        self._next_sequence += 1
        self.metrics.submitted_queries += 1

    def pop_ready(
        self,
        *,
        now_us: Optional[int] = None,
        force: bool = False,
    ) -> list[QueryBatch]:
        """Emit all currently ready first-fit batches in deterministic order."""

        now = self._clock_us() if now_us is None else int(now_us)
        emitted: list[QueryBatch] = []
        for key in list(self._buckets):
            bucket = self._buckets[key]
            while bucket:
                reason = self._flush_reason(bucket, now_us=now, force=force)
                if reason is None:
                    break
                pending_batch, memory_limited = self._take_first_fit(bucket)
                if not pending_batch:
                    raise AssertionError("ready bucket produced an empty batch")
                if reason == "force":
                    self.metrics.force_flushes += 1
                elif reason == "deadline":
                    self.metrics.deadline_flushes += 1
                elif reason == "timeout":
                    self.metrics.timeout_flushes += 1
                else:
                    self.metrics.fill_flushes += 1
                if memory_limited:
                    self.metrics.memory_limited_batches += 1
                estimated_peak_bytes = sum(
                    pending.estimated_bytes for pending in pending_batch
                )
                if (
                    len(pending_batch) == 1
                    and estimated_peak_bytes > self.policy.memory_budget_bytes
                ):
                    self.metrics.oversize_singletons += 1
                batch = build_query_batch(
                    [pending.request for pending in pending_batch],
                    estimated_peak_bytes=estimated_peak_bytes,
                    memory_budget_bytes=self.policy.memory_budget_bytes,
                )
                emitted.append(batch)
                self.metrics.emitted_batches += 1
                self.metrics.emitted_queries += len(pending_batch)
                self.metrics.batch_sizes.append(len(pending_batch))
                self.metrics.queue_wait_us.extend(
                    max(0, now - pending.submitted_us) for pending in pending_batch
                )
            if not bucket:
                del self._buckets[key]
        return emitted

    def _flush_reason(
        self,
        bucket: Deque[PendingQuery],
        *,
        now_us: int,
        force: bool,
    ) -> Optional[str]:
        if force:
            return "force"
        if any(
            pending.deadline_us is not None and pending.deadline_us <= now_us
            for pending in bucket
        ):
            return "deadline"
        if now_us - bucket[0].submitted_us >= self.policy.max_wait_us:
            return "timeout"
        estimated_prefix = 0
        for pending in list(bucket)[: self.policy.max_batch_size]:
            estimated_prefix += pending.estimated_bytes
        if estimated_prefix >= self.policy.memory_budget_bytes:
            return "fill"
        if len(bucket) >= self.policy.minimum_fill_count:
            return "fill"
        return None

    def _take_first_fit(
        self, bucket: Deque[PendingQuery]
    ) -> tuple[list[PendingQuery], bool]:
        selected: list[PendingQuery] = []
        total_bytes = 0
        memory_limited = False
        while bucket and len(selected) < self.policy.max_batch_size:
            candidate = bucket[0]
            next_bytes = total_bytes + candidate.estimated_bytes
            if selected and next_bytes > self.policy.memory_budget_bytes:
                memory_limited = True
                break
            selected.append(bucket.popleft())
            total_bytes = next_bytes
            if total_bytes > self.policy.memory_budget_bytes:
                memory_limited = True
                break
        return selected, memory_limited

    def execute_batch_with_oom_retry(
        self,
        batch: QueryBatch,
        executor: QueryBatchExecutor,
    ) -> list[tuple[str, BoundQueryResult]]:
        """Execute and deterministically bisect OOM batches while preserving order."""

        started_us = self._clock_us()
        results = self._execute_recursive(batch, executor)
        finished_us = self._clock_us()
        self.metrics.execution_us.append(max(0, finished_us - started_us))
        expected_ids = [request.query.query_id for request in batch.requests]
        result_by_id: dict[str, BoundQueryResult] = {}
        for query_id, result in results:
            if query_id not in expected_ids or query_id in result_by_id:
                self.metrics.invalid_results += 1
                raise ValueError(f"unexpected or duplicate result: {query_id}")
            result_by_id[query_id] = result
        missing = [
            query_id for query_id in expected_ids if query_id not in result_by_id
        ]
        if missing:
            self.metrics.invalid_results += 1
            raise ValueError(f"missing batch results: {missing}")
        restored = [(query_id, result_by_id[query_id]) for query_id in expected_ids]
        for query_id, _result in restored:
            if query_id in self._completed_ids:
                self.metrics.invalid_results += 1
                raise ValueError(f"query completed twice: {query_id}")
            self._completed_ids.add(query_id)
        self.metrics.completed_queries += len(restored)
        return restored

    def _execute_recursive(
        self,
        batch: QueryBatch,
        executor: QueryBatchExecutor,
    ) -> list[tuple[str, BoundQueryResult]]:
        try:
            return list(executor(batch))
        except Exception as error:  # pylint: disable=broad-exception-caught
            if not _is_oom(error) or len(batch.requests) <= 1:
                raise
            self.metrics.oom_events += 1
            self.metrics.oom_splits += 1
            midpoint = len(batch.requests) // 2
            left = self._sub_batch(batch, batch.requests[:midpoint])
            right = self._sub_batch(batch, batch.requests[midpoint:])
            return self._execute_recursive(left, executor) + self._execute_recursive(
                right, executor
            )

    def _sub_batch(
        self,
        parent: QueryBatch,
        requests: Sequence[BoundQueryRequest],
    ) -> QueryBatch:
        return build_query_batch(
            requests,
            estimated_peak_bytes=sum(self._estimator(item) for item in requests),
            memory_budget_bytes=parent.memory_budget_bytes,
        )

    def audit(self) -> Mapping[str, object]:
        """Return no-loss/no-duplicate accounting for closure checks."""

        emitted_not_completed = (
            self.metrics.emitted_queries - self.metrics.completed_queries
        )
        return {
            **self.metrics.snapshot(
                pending_queries=self.pending_count,
                max_batch_size=self.policy.max_batch_size,
            ),
            "emitted_not_completed": emitted_not_completed,
            "submitted_id_count": len(self._submitted_ids),
            "completed_id_count": len(self._completed_ids),
            "query_loss": self.metrics.submitted_queries
            - self.pending_count
            - self.metrics.completed_queries,
        }


__all__ = [
    "BatchPolicy",
    "BatchRuntimeMetrics",
    "DynamicBatchManager",
    "PendingQuery",
    "QueryBatchExecutor",
    "estimate_request_bytes",
]
