"""NRIR-2 budget-selected storage plans with runtime lifetime enforcement."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.planner.plan_ir_selector import NoFeasiblePlanError
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_verifier_ir_integration import (
    compile_native_plain_crown_memory_query,
    execute_native_plain_crown_memory_query,
)
from tests.test_task_ir_v1 import _semantic_case


def _inputs():
    legacy_module, input_spec = _semantic_case("residual")
    interval_env, relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    linear_spec = torch.tensor(
        [[1.0, -1.0, 0.5], [-0.5, 0.25, 1.0]], dtype=torch.float32
    )
    return legacy_module, input_spec, interval_env, relu_pre, linear_spec


def _compile(*, memory_budget_bytes: int):
    legacy_module, input_spec, interval_env, relu_pre, linear_spec = _inputs()
    compilation = compile_native_plain_crown_memory_query(
        legacy_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec,
        intermediate_bounds_hash="a" * 64,
        query_id="residual-native-memory-0",
        available_memory_bytes=1 << 30,
        memory_budget_bytes=memory_budget_bytes,
    )
    return compilation, legacy_module, input_spec, relu_pre, linear_spec


def test_budget_switches_real_storage_schedule_and_runtime_residency() -> None:
    high, legacy_module, input_spec, relu_pre, linear_spec = _compile(
        memory_budget_bytes=1 << 30
    )
    storage = {
        candidate.candidate_id: candidate
        for candidate in high.template.storage_candidates
    }
    retain = storage["storage:native-retain-all-v1"]
    reuse = storage["storage:native-lifetime-reuse-v1"]
    assert reuse.cost.predicted_peak_bytes < retain.cost.predicted_peak_bytes
    assert high.instance.storage_decision.candidate_id == retain.candidate_id

    low, *_ = _compile(memory_budget_bytes=reuse.cost.predicted_peak_bytes)
    assert low.instance.storage_decision.candidate_id == reuse.candidate_id
    assert high.bound_module.stable_hash() == low.bound_module.stable_hash()
    assert high.template.stable_hash(
        bound_module=high.bound_module
    ) == low.template.stable_hash(bound_module=low.bound_module)
    assert high.hashes()["plan_instance_hash"] != low.hashes()["plan_instance_hash"]
    assert high.hashes()["schedule_hash"] != low.hashes()["schedule_hash"]

    high_result, _high_task_trace, high_storage_trace = (
        execute_native_plain_crown_memory_query(
            high,
            legacy_task_module=legacy_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=linear_spec,
        )
    )
    low_result, _low_task_trace, low_storage_trace = (
        execute_native_plain_crown_memory_query(
            low,
            legacy_task_module=legacy_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=linear_spec,
        )
    )
    torch.testing.assert_close(low_result.lower, high_result.lower)
    torch.testing.assert_close(low_result.upper, high_result.upper)
    assert high_storage_trace.storage_candidate_id == retain.candidate_id
    assert low_storage_trace.storage_candidate_id == reuse.candidate_id
    assert high_storage_trace.observed_peak_live_bytes == (
        high_storage_trace.planned_peak_bytes
    )
    assert low_storage_trace.observed_peak_live_bytes < (
        high_storage_trace.observed_peak_live_bytes
    )
    assert not any(event.evicted_value_ids for event in high_storage_trace.events[:-1])
    assert any(event.evicted_value_ids for event in low_storage_trace.events[:-1])
    assert len(high_storage_trace.stable_hash()) == 64
    assert len(low_storage_trace.stable_hash()) == 64


def test_budget_below_lifetime_reuse_peak_fails_closed() -> None:
    high, *_ = _compile(memory_budget_bytes=1 << 30)
    reuse = next(
        candidate
        for candidate in high.template.storage_candidates
        if candidate.candidate_id == "storage:native-lifetime-reuse-v1"
    )
    with pytest.raises(NoFeasiblePlanError, match="memory_budget_exceeded"):
        _compile(memory_budget_bytes=reuse.cost.predicted_peak_bytes - 1)


def test_storage_trace_rejects_observed_residency_above_plan() -> None:
    compilation, legacy_module, input_spec, relu_pre, linear_spec = _compile(
        memory_budget_bytes=1 << 30
    )
    _result, _task_trace, storage_trace = execute_native_plain_crown_memory_query(
        compilation,
        legacy_task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec,
    )
    with pytest.raises(ValueError, match="summary is invalid"):
        replace(
            storage_trace,
            observed_peak_live_bytes=storage_trace.planned_peak_bytes + 1,
        ).validate()
