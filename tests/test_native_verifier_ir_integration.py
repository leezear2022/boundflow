"""Native real-network Bound→Plan→Task→Schedule ownership contracts."""

from __future__ import annotations

import torch

from boundflow.ir.bound import (
    BoundOpKind,
    IntermediateBoundSource,
    ReluLowerSlopePolicy,
    ReluRelaxationAttrs,
)
from boundflow.ir.schedule import LaunchAction
from boundflow.runtime.bound_ir_interpreter import execute_plain_crown_bound_ir
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_verifier_ir_integration import (
    compile_native_plain_crown_query,
    execute_native_plain_crown_query,
)
from tests.test_task_ir_v1 import _semantic_case


def test_native_residual_query_owns_all_compiler_layers_and_semantics() -> None:
    legacy_module, input_spec = _semantic_case("residual")
    interval_env, external_relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    linear_spec = torch.tensor(
        [[1.0, -1.0, 0.5], [-0.5, 0.25, 1.0]], dtype=torch.float32
    )

    compiled = compile_native_plain_crown_query(
        legacy_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=external_relu_pre,
        linear_spec_C=linear_spec,
        intermediate_bounds_hash="a" * 64,
        query_id="residual-native-0",
        available_memory_bytes=1 << 30,
    )
    repeated = compile_native_plain_crown_query(
        legacy_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=external_relu_pre,
        linear_spec_C=linear_spec,
        intermediate_bounds_hash="a" * 64,
        query_id="residual-native-0",
        available_memory_bytes=1 << 30,
    )

    assert compiled.hashes() == repeated.hashes()
    assert len(compiled.hashes()) == 5
    assert len(compiled.task_module.tasks) == len(compiled.bound_module.graph.ops)
    assert sum(
        isinstance(action, LaunchAction) for action in compiled.schedule.actions
    ) == len(compiled.bound_module.graph.ops)
    assert all(
        op.kind != BoundOpKind.EXTERNAL_VERIFIER_CALL
        for op in compiled.bound_module.graph.ops
    )
    relu_attrs = tuple(
        op.attrs
        for op in compiled.bound_module.graph.ops
        if op.kind == BoundOpKind.RELU_RELAXATION
    )
    assert relu_attrs
    assert all(isinstance(attrs, ReluRelaxationAttrs) for attrs in relu_attrs)
    assert all(
        attrs.intermediate_bound_source == IntermediateBoundSource.EXTERNAL_VERIFIER
        and attrs.lower_slope_policy == ReluLowerSlopePolicy.ADAPTIVE
        for attrs in relu_attrs
        if isinstance(attrs, ReluRelaxationAttrs)
    )

    actual, trace = execute_native_plain_crown_query(
        compiled,
        legacy_task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=external_relu_pre,
        linear_spec_C=linear_spec,
    )
    expected = execute_plain_crown_bound_ir(
        compiled.bound_module,
        task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=external_relu_pre,
        linear_spec_C=linear_spec,
    )

    torch.testing.assert_close(actual.lower, expected.lower)
    torch.testing.assert_close(actual.upper, expected.upper)
    assert len(trace.events) == len(compiled.task_module.tasks)
    assert trace.task_module_hash == compiled.hashes()["task_module_hash"]
    assert trace.schedule_hash == compiled.hashes()["schedule_hash"]
