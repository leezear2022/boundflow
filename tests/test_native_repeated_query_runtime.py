"""Native repeated-query packing, cache, lineage, and serial reference."""

# pylint: disable=missing-function-docstring,too-many-locals
# pylint: disable=assignment-from-no-return,unpacking-non-sequence

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.planner.representation_plan_binding import (
    DENSE_POLICY_ID,
    STRUCTURED_AFFINE_POLICY_ID,
)
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_repeated_query_runtime import (
    NativeRepeatedQueryCompilationCache,
    NativeRepeatedQuerySpec,
    compile_native_repeated_query_stream,
    execute_native_repeated_query_serial_reference,
    execute_native_repeated_query_stream,
)
from boundflow.runtime.task_executor import InputSpec
from tests.test_task_ir_v1 import _semantic_case


def _runtime_case(*, memory_budget_bytes: int = 1 << 30):
    legacy_module, input_spec = _semantic_case("residual")
    lower, upper = input_spec.perturbation.bounding_box(input_spec.center)
    input_spec = InputSpec.box(
        value_name=input_spec.value_name,
        lower=lower[:1],
        upper=upper[:1],
    )
    interval_env, relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    rows = torch.tensor(
        [
            [1.0, -1.0, 0.5],
            [-0.5, 0.25, 1.0],
            [0.75, 0.5, -1.0],
            [-1.0, 0.125, 0.25],
        ],
        dtype=torch.float32,
    )
    query_specs = tuple(
        NativeRepeatedQuerySpec(f"property-{index:02d}", rows[index : index + 1])
        for index in range(4)
    )
    cache = NativeRepeatedQueryCompilationCache()
    kwargs = {
        "interval_env": interval_env,
        "relu_pre": relu_pre,
        "intermediate_bounds_hash": "e" * 64,
        "stream_id": "native-repeated-query-residual",
        "workload_identity_hash": "a" * 64,
        "state_identity_hash": "b" * 64,
        "available_memory_bytes": 1 << 30,
        "memory_budget_bytes": memory_budget_bytes,
        "spec_slice_candidate_size": 2,
        "max_spec_batch_size": 2,
    }
    return (
        cache,
        legacy_module,
        input_spec,
        interval_env,
        relu_pre,
        query_specs,
        kwargs,
    )


def test_repeated_query_cache_miss_hit_and_exact_layout() -> None:
    cache, legacy, input_spec, _interval, _relu_pre, specs, kwargs = _runtime_case()
    first = compile_native_repeated_query_stream(
        cache, legacy, input_spec, specs, **kwargs
    )
    second = compile_native_repeated_query_stream(
        cache, legacy, input_spec, specs, **kwargs
    )
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert cache.miss_count == cache.hit_count == len(cache.entries) == 1
    assert second.compilation is first.compilation
    assert first.compilation.layout_trace.representation_policy_id == DENSE_POLICY_ID
    assert tuple(
        (item.query_id, item.start_index, item.stop_index)
        for item in first.compilation.layout_trace.query_ranges
    ) == (
        ("property-00", 0, 1),
        ("property-01", 1, 2),
        ("property-02", 2, 3),
        ("property-03", 3, 4),
    )
    assert len(first.compilation.joint_compilation.child_compilations) == 2


def test_repeated_query_packed_two_children_matches_four_serial_queries() -> None:
    cache, legacy, input_spec, interval_env, relu_pre, specs, kwargs = _runtime_case()
    compiled = compile_native_repeated_query_stream(
        cache, legacy, input_spec, specs, **kwargs
    )
    packed, packed_trace = execute_native_repeated_query_stream(
        compiled,
        legacy_task_module=legacy,
        input_spec=input_spec,
        relu_pre=relu_pre,
    )
    serial, serial_trace = execute_native_repeated_query_serial_reference(
        compiled.compilation,
        legacy_task_module=legacy,
        input_spec=input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
    )
    assert packed_trace.packed_child_stack_count == 2
    assert serial_trace.serial_child_stack_count == 4
    assert (
        tuple(item.query_id for item in packed)
        == tuple(item.query_id for item in serial)
        == tuple(item.query_id for item in specs)
    )
    for packed_item, serial_item in zip(packed, serial):
        torch.testing.assert_close(
            packed_item.result.lower, serial_item.result.lower, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            packed_item.result.upper, serial_item.result.upper, atol=0.0, rtol=0.0
        )


def test_repeated_query_structured_policy_is_preserved_in_packed_and_serial() -> None:
    cache, legacy, input_spec, interval_env, relu_pre, specs, kwargs = _runtime_case()
    dense = compile_native_repeated_query_stream(
        cache, legacy, input_spec, specs, **kwargs
    )
    storage = {
        item.candidate_id: item
        for item in dense.compilation.joint_compilation.source_template.storage_candidates
    }
    reuse_budget = storage["storage:native-lifetime-reuse-v1"].cost.predicted_peak_bytes
    structured_kwargs = {**kwargs, "memory_budget_bytes": reuse_budget}
    structured = compile_native_repeated_query_stream(
        cache, legacy, input_spec, specs, **structured_kwargs
    )
    packed, _trace = execute_native_repeated_query_stream(
        structured,
        legacy_task_module=legacy,
        input_spec=input_spec,
        relu_pre=relu_pre,
    )
    serial, serial_trace = execute_native_repeated_query_serial_reference(
        structured.compilation,
        legacy_task_module=legacy,
        input_spec=input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
    )
    assert structured.compilation.layout_trace.representation_policy_id == (
        STRUCTURED_AFFINE_POLICY_ID
    )
    assert serial_trace.representation_policy_id == STRUCTURED_AFFINE_POLICY_ID
    for packed_item, serial_item in zip(packed, serial):
        torch.testing.assert_close(
            packed_item.result.lower, serial_item.result.lower, atol=0.0, rtol=0.0
        )


def test_repeated_query_cache_key_tracks_content_order_and_state() -> None:
    cache, legacy, input_spec, _interval, _relu_pre, specs, kwargs = _runtime_case()
    original = compile_native_repeated_query_stream(
        cache, legacy, input_spec, specs, **kwargs
    )
    changed_tensor = specs[0].linear_spec_C.clone()
    changed_tensor[0, 0] += 1.0
    changed_specs = (
        replace(specs[0], linear_spec_C=changed_tensor),
        *specs[1:],
    )
    changed = compile_native_repeated_query_stream(
        cache, legacy, input_spec, changed_specs, **kwargs
    )
    reordered = compile_native_repeated_query_stream(
        cache, legacy, input_spec, tuple(reversed(specs)), **kwargs
    )
    changed_state = compile_native_repeated_query_stream(
        cache,
        legacy,
        input_spec,
        specs,
        **{**kwargs, "state_identity_hash": "c" * 64},
    )
    assert all(
        item.cache_hit is False
        for item in (original, changed, reordered, changed_state)
    )
    assert (
        len(
            {
                original.compilation.cache_key,
                changed.compilation.cache_key,
                reordered.compilation.cache_key,
                changed_state.compilation.cache_key,
            }
        )
        == 4
    )
    assert cache.miss_count == len(cache.entries) == 4


def test_repeated_query_rejects_runtime_and_layout_tamper() -> None:
    cache, legacy, input_spec, _interval, relu_pre, specs, kwargs = _runtime_case()
    compiled = compile_native_repeated_query_stream(
        cache, legacy, input_spec, specs, **kwargs
    )
    changed_packed = compiled.compilation.packed_objective.clone()
    changed_packed[0, 0, 0] += 1.0
    broken_objective = replace(compiled.compilation, packed_objective=changed_packed)
    with pytest.raises(ValueError, match="packed objective hash"):
        replace(compiled, compilation=broken_objective).validate()

    first_range = compiled.compilation.layout_trace.query_ranges[0]
    broken_layout = replace(
        compiled.compilation.layout_trace,
        query_ranges=(
            replace(first_range, stop_index=2),
            *compiled.compilation.layout_trace.query_ranges[1:],
        ),
    )
    with pytest.raises(ValueError, match="overlap"):
        replace(compiled.compilation, layout_trace=broken_layout).validate()

    changed_joint = replace(
        compiled.compilation.joint_compilation.binding_trace,
        source_linear_spec_hash="f" * 64,
    )
    with pytest.raises(ValueError):
        replace(
            compiled.compilation,
            joint_compilation=replace(
                compiled.compilation.joint_compilation,
                binding_trace=changed_joint,
            ),
        ).validate()

    packed, trace = execute_native_repeated_query_stream(
        compiled,
        legacy_task_module=legacy,
        input_spec=input_spec,
        relu_pre=relu_pre,
    )
    assert len(packed) == 4
    assert trace.cache_hit is False
