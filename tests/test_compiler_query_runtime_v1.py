"""IR-4D typed query entry and exact runtime-state closure."""

# Test fixtures intentionally mirror the standalone fresh-process artifact.
# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from boundflow.ir.plan import PlanCost, StateAction, StateCandidate
from boundflow.planner.materialization import BoundMethod
from boundflow.planner.plan_ir_selector import PlanSelectionContext
from boundflow.runtime.bab_query import BoundQueryRequest, make_bound_query
from boundflow.runtime.bound_ir_interpreter import execute_plain_crown_bound_ir
from boundflow.runtime.bound_state_store import (
    BoundRuntimeStatePayload,
    BoundRuntimeStateStore,
)
from boundflow.runtime.compiler_query_runtime import (
    CompilerBoundQueryRequest,
    CompilerQueryCapabilityError,
    CompilerRuntimeContext,
    TypedCompilerQueryPayload,
    TypedCompilerQueryRequest,
    TypedCompilerQueryRuntime,
)
from boundflow.runtime.bab_query_runtime import (
    CompilerSameSolverQueryRuntime,
    SameSolverQueryRuntime,
    SameSolverRuntimeConfig,
)
from boundflow.runtime.task_executor import InputSpec
from scripts.run_plan_ir_v1_reference_artifact import (
    ReferenceSmokeWorkload,
    build_reference_smoke_workload,
)


def _request(
    workload: ReferenceSmokeWorkload,
    query_id: str,
    sequence_number: int,
) -> TypedCompilerQueryRequest:
    return TypedCompilerQueryRequest(
        query_id=query_id,
        sequence_number=sequence_number,
        payload=TypedCompilerQueryPayload(
            legacy_task_module=workload.task_module,
            bound_module=workload.bound_module,
            template=workload.template,
            input_spec=workload.input_spec,
            relu_pre=workload.relu_pre,
        ),
    )


def _compiler_bound_query_request(
    workload: ReferenceSmokeWorkload,
    query_id: str,
    sequence_number: int,
    *,
    method: BoundMethod = BoundMethod.CROWN,
) -> CompilerBoundQueryRequest:
    query, dynamic_payload = make_bound_query(
        module=workload.task_module,
        query_id=query_id,
        parent_query_id=None,
        sequence_number=sequence_number,
        example_idx=0,
        input_spec=workload.input_spec,
        linear_spec_c=None,
        split_by_relu_input={},
        warm_alpha_by_relu_input={},
        warm_beta_by_relu_input={},
        bound_method=method,
        execution_options={},
    )
    return CompilerBoundQueryRequest(
        query_request=BoundQueryRequest(query=query, payload=dynamic_payload),
        compiler_payload=_request(workload, query_id, sequence_number).payload,
    )


def _state_cost(latency: float) -> PlanCost:
    return PlanCost(
        predicted_latency_ms=latency,
        predicted_peak_bytes=0,
        compile_cost_ms=0.0,
        setup_cost_ms=0.0,
        confidence=1.0,
        risk_tags=("ir4d_runtime_state",),
    )


def _stateful_workload() -> tuple[ReferenceSmokeWorkload, tuple[str, ...]]:
    workload = build_reference_smoke_workload()
    middle_op = workload.bound_module.graph.ops[1]
    candidates: list[StateCandidate] = []
    for index, value_id in enumerate(middle_op.outputs):
        value = next(
            item
            for item in workload.bound_module.graph.values
            if item.value_id == value_id
        )
        state_id = f"middle-output:{index}"
        for action, latency in (
            (StateAction.REUSE, 0.0),
            (StateAction.CACHE, 0.1),
        ):
            candidates.append(
                StateCandidate(
                    candidate_id=f"state:{state_id}:{action.value}",
                    state_id=state_id,
                    source_value_id=value_id,
                    action=action,
                    state_version=value.state_version or "",
                    size_bytes=int(
                        torch.empty(
                            tuple(int(dim) for dim in value.tensor_type.shape)
                        ).numel()
                        * 4
                    ),
                    static_legal=True,
                    rejection_reasons=(),
                    cost=_state_cost(latency),
                )
            )
    return (
        replace(
            workload,
            template=replace(
                workload.template,
                state_candidates=tuple(candidates),
            ),
        ),
        middle_op.outputs,
    )


def test_typed_compiler_query_preserves_order_and_reuses_plan() -> None:
    workload = build_reference_smoke_workload()
    requests = (
        _request(workload, "query:z", 9),
        _request(workload, "query:a", 3),
    )
    runtime = TypedCompilerQueryRuntime(
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
    )
    results = runtime.execute(requests)
    direct = execute_plain_crown_bound_ir(
        workload.bound_module,
        task_module=workload.task_module,
        input_spec=workload.input_spec,
        relu_pre=workload.relu_pre,
    )

    assert [result.query_id for result in results] == ["query:z", "query:a"]
    assert [result.sequence_number for result in results] == [9, 3]
    for result in results:
        torch.testing.assert_close(result.bounds.lower, direct.lower)
        torch.testing.assert_close(result.bounds.upper, direct.upper)
    assert runtime.audit()["plan_cache_misses"] == 1
    assert runtime.audit()["plan_cache_hits"] == 1
    assert runtime.audit()["physical_cross_query_batching_claimed"] is False


def test_typed_compiler_query_binds_dynamic_plan_context_to_instance() -> None:
    workload = build_reference_smoke_workload()
    runtime = TypedCompilerQueryRuntime(
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
    )
    cold = replace(
        _request(workload, "query:cold", 0),
        runtime_context=CompilerRuntimeContext(
            available_memory_bytes=1 << 29,
            memory_budget_bytes=1 << 28,
            plan_selection=PlanSelectionContext(
                query_distribution_id="cold-single",
                expected_query_count=1,
            ),
        ),
    )
    repeated = replace(
        _request(workload, "query:repeated", 1),
        runtime_context=CompilerRuntimeContext(
            available_memory_bytes=1 << 29,
            memory_budget_bytes=1 << 28,
            plan_selection=PlanSelectionContext(
                query_distribution_id="repeated-64",
                expected_query_count=64,
            ),
        ),
    )

    cold_result, repeated_result = runtime.execute((cold, repeated))

    torch.testing.assert_close(cold_result.bounds.lower, repeated_result.bounds.lower)
    torch.testing.assert_close(cold_result.bounds.upper, repeated_result.bounds.upper)
    assert cold_result.plan_instance_hash != repeated_result.plan_instance_hash
    assert cold_result.plan_instance.memory_budget_bytes == 1 << 28
    assert repeated_result.plan_instance.memory_budget_bytes == 1 << 28
    assert runtime.audit()["plan_cache_misses"] == 2
    assert {
        item.value
        for item in repeated_result.plan_instance.provenance
        if item.key == "query_distribution_id"
    } == {"repeated-64"}


def test_pr13_batch_manager_dispatches_only_typed_compiler_requests() -> None:
    workload = build_reference_smoke_workload()
    compiler = TypedCompilerQueryRuntime(
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
    )
    runtime = CompilerSameSolverQueryRuntime(
        SameSolverRuntimeConfig(
            max_batch_size=2,
            memory_budget_bytes=1 << 20,
            minimum_fill_ratio=1.0,
        ),
        compiler,
    )
    requests = (
        _compiler_bound_query_request(workload, "bound-query:z", 8),
        _compiler_bound_query_request(workload, "bound-query:a", 1),
    )

    results = runtime.execute(requests, now_us=100)

    assert [query_id for query_id, _result in results] == [
        "bound-query:z",
        "bound-query:a",
    ]
    audit = runtime.audit()
    assert audit["emitted_batches"] == 1
    assert audit["completed_queries"] == 2
    assert audit["legacy_executor_dispatches"] == 0
    assert audit["compiler"]["plan_cache_misses"] == 1
    assert audit["compiler"]["plan_cache_hits"] == 1


def test_pr13_compiler_adapter_rejects_alpha_and_payload_mismatch() -> None:
    workload = build_reference_smoke_workload()
    alpha_request = _compiler_bound_query_request(
        workload,
        "bound-query:alpha",
        0,
        method=BoundMethod.ALPHA_CROWN,
    )
    with pytest.raises(CompilerQueryCapabilityError, match="plain_crown_typed_ir"):
        alpha_request.validate()
    legacy_runtime = SameSolverQueryRuntime(
        SameSolverRuntimeConfig(
            max_batch_size=1,
            memory_budget_bytes=1 << 20,
        )
    )
    with pytest.raises(
        CompilerQueryCapabilityError, match="historical opt-in only.*NO-GO"
    ):
        legacy_runtime.execute(
            workload.task_module,
            (alpha_request.query_request,),
            now_us=0,
        )

    plain_request = _compiler_bound_query_request(workload, "bound-query:mismatch", 1)
    mismatched_input = InputSpec.linf(
        value_name=workload.input_spec.value_name,
        center=torch.ones_like(workload.input_spec.center),
        eps=0.1,
    )
    mismatched = replace(
        plain_request,
        compiler_payload=replace(
            plain_request.compiler_payload,
            input_spec=mismatched_input,
        ),
    )
    with pytest.raises(ValueError, match="input payload differs"):
        mismatched.validate()


def test_runtime_state_cache_then_exact_reuse_skips_middle_task() -> None:
    workload, middle_outputs = _stateful_workload()
    state_store = BoundRuntimeStateStore()
    runtime = TypedCompilerQueryRuntime(
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
        state_store=state_store,
    )

    first, second = runtime.execute(
        (
            _request(workload, "query:cache", 0),
            _request(workload, "query:reuse", 1),
        )
    )

    torch.testing.assert_close(first.bounds.lower, second.bounds.lower)
    torch.testing.assert_close(first.bounds.upper, second.bounds.upper)
    assert all(
        event.backend_candidate_id != "state-reuse" for event in first.trace.events
    )
    reused = [
        event
        for event in second.trace.events
        if event.backend_candidate_id == "state-reuse"
    ]
    assert len(reused) == 1
    assert {value_id for value_id, _value_hash in reused[0].output_value_hashes} == set(
        middle_outputs
    )
    assert state_store.audit() == {
        "entries": 4,
        "load_hits": 4,
        "load_misses": 0,
        "stores": 4,
        "invalidations": 0,
    }
    assert runtime.audit()["plan_cache_misses"] == 2


def test_stale_runtime_state_is_not_selected_for_reuse() -> None:
    workload, middle_outputs = _stateful_workload()
    state_store = BoundRuntimeStateStore()
    state_id = "middle-output:0"
    value = next(
        item
        for item in workload.bound_module.graph.values
        if item.value_id == middle_outputs[0]
    )
    stale = torch.zeros(
        tuple(int(dim) for dim in value.tensor_type.shape),
        dtype=workload.input_spec.center.dtype,
    )
    state_store.put(
        BoundRuntimeStatePayload.create(
            state_id=state_id,
            source_value_id=value.value_id,
            state_version="stale-version",
            bound_module_hash=workload.bound_module.stable_hash(),
            value=stale,
        )
    )
    runtime = TypedCompilerQueryRuntime(
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
        state_store=state_store,
    )

    result = runtime.execute((_request(workload, "query:stale", 0),))[0]

    assert all(
        event.backend_candidate_id != "state-reuse" for event in result.trace.events
    )
    assert state_store.audit()["load_hits"] == 0
    assert state_store.audit()["stores"] == 4


def test_runtime_state_payload_detects_post_creation_mutation() -> None:
    workload = build_reference_smoke_workload()
    value = torch.zeros(1)
    payload = BoundRuntimeStatePayload.create(
        state_id="state",
        source_value_id=workload.bound_module.graph.inputs[0],
        state_version="plain-crown-v1",
        bound_module_hash=workload.bound_module.stable_hash(),
        value=value,
    )
    payload.value.add_(1.0)

    with pytest.raises(ValueError, match="content hash mismatch"):
        payload.validate()


def test_compiler_query_artifact_replays_in_fresh_process(tmp_path: Path) -> None:
    artifact = tmp_path / "compiler-query-runtime-v1.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    subprocess.run(
        (
            sys.executable,
            "scripts/run_compiler_query_runtime_v1_artifact.py",
            "generate",
            "--out",
            str(artifact),
        ),
        check=True,
        env=env,
    )
    subprocess.run(
        (
            sys.executable,
            "scripts/run_compiler_query_runtime_v1_artifact.py",
            "replay",
            "--artifact",
            str(artifact),
        ),
        check=True,
        env=env,
    )


@pytest.mark.parametrize(
    "method, capability",
    (
        (BoundMethod.ALPHA_CROWN, "alpha_dense"),
        (BoundMethod.ALPHA_BETA_CROWN, "alpha_beta_dense_split"),
    ),
)
def test_legacy_pr13_query_is_explicit_compiler_no_go(
    method: BoundMethod, capability: str
) -> None:
    workload = build_reference_smoke_workload()
    query, payload = make_bound_query(
        module=workload.task_module,
        query_id=f"legacy:{method.value}",
        parent_query_id=None,
        sequence_number=0,
        example_idx=0,
        input_spec=workload.input_spec,
        linear_spec_c=None,
        split_by_relu_input={},
        warm_alpha_by_relu_input={},
        warm_beta_by_relu_input={},
        bound_method=method,
        execution_options={},
    )
    runtime = TypedCompilerQueryRuntime(
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
    )

    with pytest.raises(
        CompilerQueryCapabilityError,
        match=rf"capability={capability}.*NO-GO",
    ):
        runtime.reject_legacy_bab_request(
            BoundQueryRequest(query=query, payload=payload)
        )
