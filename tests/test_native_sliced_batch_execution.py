"""Native source Plan spec slicing and real child-stack execution."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.ir.plan import PlanProvenance
from boundflow.ir.schedule import BatchLoopAction
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_sliced_batch_integration import (
    compile_native_plain_crown_sliced_batch_query,
    execute_native_plain_crown_sliced_batch_query,
)
from tests.test_task_ir_v1 import _semantic_case


def _compile_pair():
    legacy_module, input_spec = _semantic_case("residual")
    interval_env, relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    linear_spec = torch.tensor(
        [[1.0, -1.0, 0.5], [-0.5, 0.25, 1.0]], dtype=torch.float32
    )
    kwargs = {
        "interval_env": interval_env,
        "relu_pre": relu_pre,
        "linear_spec_C": linear_spec,
        "intermediate_bounds_hash": "c" * 64,
        "query_id": "native-spec-batch-residual",
        "available_memory_bytes": 1 << 30,
        "memory_budget_bytes": 1 << 30,
        "spec_slice_candidate_size": 1,
    }
    full = compile_native_plain_crown_sliced_batch_query(
        legacy_module,
        input_spec,
        max_spec_batch_size=2,
        **kwargs,
    )
    sliced = compile_native_plain_crown_sliced_batch_query(
        legacy_module,
        input_spec,
        max_spec_batch_size=1,
        **kwargs,
    )
    return full, sliced, legacy_module, input_spec, relu_pre, linear_spec


def test_spec_batch_decision_builds_and_executes_real_child_stacks() -> None:
    full, sliced, legacy_module, input_spec, relu_pre, linear_spec = _compile_pair()

    assert full.source_instance.batch_decision.candidate_id == "batch:full-query"
    assert sliced.source_instance.batch_decision.candidate_id == (
        "batch:native-spec-sliced-v1:0001"
    )
    assert full.source_build.module.stable_hash() == (
        sliced.source_build.module.stable_hash()
    )
    assert full.source_template.stable_hash(
        bound_module=full.source_build.module
    ) == sliced.source_template.stable_hash(bound_module=sliced.source_build.module)
    assert full.source_instance.stable_hash(
        template=full.source_template, bound_module=full.source_build.module
    ) != sliced.source_instance.stable_hash(
        template=sliced.source_template, bound_module=sliced.source_build.module
    )

    full_loop = next(
        item
        for item in full.source_schedule.actions
        if isinstance(item, BatchLoopAction)
    )
    sliced_loop = next(
        item
        for item in sliced.source_schedule.actions
        if isinstance(item, BatchLoopAction)
    )
    assert full_loop.axis == "domain"
    assert sliced_loop.axis == "spec"
    assert tuple(
        (item.start_index, item.stop_index) for item in sliced_loop.slices
    ) == ((0, 1), (1, 2))
    assert len(full.child_compilations) == 1
    assert len(sliced.child_compilations) == 2
    assert all(
        len(child.task_module.tasks) == len(child.bound_module.graph.ops) > 1
        for child in sliced.child_compilations
    )

    full_result, full_trace = execute_native_plain_crown_sliced_batch_query(
        full,
        legacy_task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec,
    )
    sliced_result, sliced_trace = execute_native_plain_crown_sliced_batch_query(
        sliced,
        legacy_task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec,
    )
    torch.testing.assert_close(
        full_result.lower, sliced_result.lower, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        full_result.upper, sliced_result.upper, atol=0.0, rtol=0.0
    )
    assert len(full_trace.child_query_ids) == 1
    assert len(sliced_trace.child_query_ids) == 2
    assert full_trace.result_lower_hash == sliced_trace.result_lower_hash
    assert full.binding_trace.source_linear_spec_hash == (
        sliced.binding_trace.source_linear_spec_hash
    )


def test_spec_batch_compilation_is_deterministic() -> None:
    _full, sliced, legacy_module, input_spec, relu_pre, linear_spec = _compile_pair()
    interval_env, _local = _forward_ibp_trace_mlp(legacy_module, input_spec)
    repeated = compile_native_plain_crown_sliced_batch_query(
        legacy_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec,
        intermediate_bounds_hash="c" * 64,
        query_id="native-spec-batch-residual",
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
        spec_slice_candidate_size=1,
        max_spec_batch_size=1,
    )
    assert repeated.hashes() == sliced.hashes()
    assert repeated.binding_trace.canonical_json() == (
        sliced.binding_trace.canonical_json()
    )


def test_spec_batch_rejects_objective_schedule_and_binding_tamper() -> None:
    _full, sliced, legacy_module, input_spec, relu_pre, linear_spec = _compile_pair()
    changed_objective = linear_spec.clone()
    changed_objective[0, 0] += 1.0
    with pytest.raises(ValueError, match="objective hash"):
        execute_native_plain_crown_sliced_batch_query(
            sliced,
            legacy_task_module=legacy_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=changed_objective,
        )

    first = sliced.binding_trace.slices[0]
    broken_binding = replace(
        sliced.binding_trace,
        slices=(replace(first, stop_index=2), *sliced.binding_trace.slices[1:]),
    )
    with pytest.raises(ValueError, match="overlap"):
        replace(sliced, binding_trace=broken_binding).validate()

    loop = next(
        item
        for item in sliced.source_schedule.actions
        if isinstance(item, BatchLoopAction)
    )
    broken_loop = replace(
        loop,
        slices=(
            replace(loop.slices[0], stop_index=2),
            *loop.slices[1:],
        ),
    )
    broken_schedule = replace(
        sliced.source_schedule,
        actions=tuple(
            broken_loop if item is loop else item
            for item in sliced.source_schedule.actions
        ),
    )
    with pytest.raises(ValueError, match="overlaps objective slices"):
        replace(sliced, source_schedule=broken_schedule).validate()


def test_plan_instance_enforces_recorded_spec_batch_limit() -> None:
    _full, sliced, *_rest = _compile_pair()
    provenance = tuple(
        PlanProvenance(item.key, "0") if item.key == "max_spec_batch_size" else item
        for item in sliced.source_instance.provenance
    )
    broken = replace(sliced.source_instance, provenance=provenance)
    with pytest.raises(ValueError, match="query-time batch limit"):
        broken.validate(
            template=sliced.source_template,
            bound_module=sliced.source_build.module,
        )
