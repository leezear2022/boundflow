"""PR-13B deterministic dynamic BatchManager correctness tests."""

from dataclasses import replace

import pytest
import torch

from boundflow.planner.materialization import BoundMethod, OptimizationStage
from boundflow.runtime.bab_query import (
    BoundQuery,
    BoundQueryPayload,
    BoundQueryRequest,
    BoundQueryResult,
    QueryCompatibilityKey,
)
from boundflow.runtime.query_batcher import BatchPolicy, DynamicBatchManager
from boundflow.runtime.task_executor import InputSpec


def _key(*, capability: str = "alpha_beta_dense_split") -> QueryCompatibilityKey:
    return QueryCompatibilityKey(
        model_structure_hash="model",
        weight_version="weights",
        bound_method=BoundMethod.ALPHA_BETA_CROWN.value,
        optimization_stage=OptimizationStage.BAB_NODE_EVAL.value,
        requires_grad=True,
        input_value_name="input",
        input_shape=(1, 2),
        spec_shape=(),
        split_tensor_shapes=(("h1", (2,)),),
        dtype="torch.float32",
        device="cpu",
        perturbation_signature="lp(p=inf,eps=1.0)",
        execution_options_hash="options",
        backend_capability_class=capability,
        numeric_policy="fp32_strict",
    )


def _request(
    index: int, *, key: QueryCompatibilityKey | None = None
) -> BoundQueryRequest:
    compatibility = _key() if key is None else key
    query = BoundQuery(
        query_id=f"q{index}",
        parent_query_id=None if index == 0 else "q0",
        sequence_number=index,
        example_idx=0,
        model_structure_hash="model",
        weight_version="weights",
        input_region_hash=f"input-{index}",
        output_spec_hash="none",
        split_signature=f"split-{index}",
        bound_method=BoundMethod.ALPHA_BETA_CROWN,
        optimization_stage=OptimizationStage.BAB_NODE_EVAL,
        requires_grad=True,
        alpha_state_version=None,
        beta_state_version=None,
        cuts_version=None,
        dtype="torch.float32",
        device="cpu",
        numeric_policy="fp32_strict",
        requested_outputs=("bounds",),
        compatibility_key=compatibility,
        execution_options={},
    )
    payload = BoundQueryPayload(
        input_spec=InputSpec.linf(
            value_name="input",
            center=torch.tensor([[float(index), 0.0]], dtype=torch.float32),
            eps=1.0,
        ),
        linear_spec_c=None,
        split_by_relu_input={"h1": torch.zeros(2, dtype=torch.int8)},
        warm_alpha_by_relu_input={},
        warm_beta_by_relu_input={},
    )
    return BoundQueryRequest(query=query, payload=payload)


def _result(index: int) -> BoundQueryResult:
    value = torch.tensor([[float(index)]], dtype=torch.float32)
    return BoundQueryResult(
        status="ok",
        lower=value,
        upper=value + 1.0,
        branch=None,
        alpha_state_version=None,
        beta_state_version=None,
    )


def test_pr13b_partial_timeout_flush_and_order_restoration() -> None:
    """A partial bucket must flush on timeout and restore executor output order."""

    now = [0]
    manager = DynamicBatchManager(
        BatchPolicy(
            max_batch_size=4,
            memory_budget_bytes=1 << 20,
            max_wait_us=100,
            minimum_fill_ratio=1.0,
        ),
        clock_us=lambda: now[0],
    )
    manager.submit(_request(0), now_us=0)
    manager.submit(_request(1), now_us=0)
    assert not manager.pop_ready(now_us=99)
    batches = manager.pop_ready(now_us=100)
    assert len(batches) == 1
    assert len(batches[0].requests) == 2

    def reverse_executor(batch):
        return [
            (request.query.query_id, _result(int(request.query.query_id[1:])))
            for request in reversed(batch.requests)
        ]

    restored = manager.execute_batch_with_oom_retry(batches[0], reverse_executor)
    assert [query_id for query_id, _ in restored] == ["q0", "q1"]
    audit = manager.audit()
    assert audit["query_loss"] == 0
    assert audit["queue_wait_us_p99"] == 100.0
    assert audit["average_batch_fill_ratio"] == 0.5


def test_pr13b_compatibility_buckets_and_deadline_flush() -> None:
    """Different capability keys never mix and an absolute deadline flushes early."""

    manager = DynamicBatchManager(
        BatchPolicy(
            max_batch_size=4,
            memory_budget_bytes=1 << 20,
            max_wait_us=1_000,
        )
    )
    dense = _request(0)
    plain_key = replace(_key(), backend_capability_class="plain_crown_fused")
    plain = _request(1, key=plain_key)
    manager.submit(dense, deadline_us=50, now_us=0)
    manager.submit(plain, deadline_us=50, now_us=0)

    assert manager.next_wakeup_us == 50
    assert not manager.pop_ready(now_us=49)
    batches = manager.pop_ready(now_us=50)
    assert len(batches) == 2
    assert all(len(batch.requests) == 1 for batch in batches)
    assert batches[0].key != batches[1].key
    assert manager.metrics.deadline_flushes == 2
    assert manager.next_wakeup_us is None


def test_pr13b_budget_forms_multiple_deterministic_batches() -> None:
    """Memory pressure lowers effective batch size without dropping the remainder."""

    manager = DynamicBatchManager(
        BatchPolicy(
            max_batch_size=8,
            memory_budget_bytes=20,
            max_wait_us=1_000,
            minimum_fill_ratio=0.5,
        ),
        estimator=lambda _request: 10,
    )
    for index in range(4):
        manager.submit(_request(index), now_us=0)

    batches = manager.pop_ready(now_us=0)
    assert [[item.query.query_id for item in batch.requests] for batch in batches] == [
        ["q0", "q1"],
        ["q2", "q3"],
    ]
    assert all(batch.estimated_peak_bytes <= 20 for batch in batches)
    assert manager.pending_count == 0
    assert manager.metrics.memory_limited_batches == 1


def test_pr13b_oom_bisection_is_deterministic_and_lossless() -> None:
    """OOM retries split in halves and restore the original five-query order."""

    manager = DynamicBatchManager(
        BatchPolicy(
            max_batch_size=8,
            memory_budget_bytes=1 << 20,
            max_wait_us=1_000,
        )
    )
    for index in range(5):
        manager.submit(_request(index), now_us=0)
    batch = manager.pop_ready(now_us=0, force=True)[0]
    physical_batches: list[list[str]] = []

    def limited_executor(candidate):
        query_ids = [request.query.query_id for request in candidate.requests]
        physical_batches.append(query_ids)
        if len(candidate.requests) > 2:
            raise RuntimeError("CUDA out of memory in deterministic test")
        return [
            (query_id, _result(int(query_id[1:]))) for query_id in reversed(query_ids)
        ]

    restored = manager.execute_batch_with_oom_retry(batch, limited_executor)
    assert [query_id for query_id, _ in restored] == [f"q{i}" for i in range(5)]
    assert physical_batches == [
        ["q0", "q1", "q2", "q3", "q4"],
        ["q0", "q1"],
        ["q2", "q3", "q4"],
        ["q2"],
        ["q3", "q4"],
    ]
    assert manager.metrics.oom_events == 2
    assert manager.metrics.oom_splits == 2
    assert manager.audit()["query_loss"] == 0


def test_pr13b_rejects_duplicate_submission_and_missing_result() -> None:
    """Duplicate inputs and incomplete executor outputs are explicit hard errors."""

    manager = DynamicBatchManager(
        BatchPolicy(
            max_batch_size=2,
            memory_budget_bytes=1 << 20,
            max_wait_us=0,
        )
    )
    request = _request(0)
    manager.submit(request, now_us=0)
    with pytest.raises(ValueError, match="duplicate query submission"):
        manager.submit(request, now_us=0)
    batch = manager.pop_ready(now_us=0)[0]
    with pytest.raises(ValueError, match="missing batch results"):
        manager.execute_batch_with_oom_retry(batch, lambda _batch: [])
