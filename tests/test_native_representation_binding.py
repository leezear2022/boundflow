"""Native representation Plan-to-Bound semantic binding contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.frontends.plain_crown_bound_ir import relu_split_state_hash
from boundflow.ir.bound import BoundOpKind, BoundRepresentation
from boundflow.ir.schedule import LaunchAction, MaterializeAction
from boundflow.planner.plan_ir_selector import NoFeasiblePlanError
from boundflow.planner.representation_plan_binding import (
    DENSE_POLICY_ID,
    STRUCTURED_AFFINE_POLICY_ID,
    BoundRepresentationBinding,
    bind_native_representation_plan,
)
from boundflow.runtime.bound_ir_interpreter import execute_plain_crown_bound_ir
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_verifier_ir_integration import (
    NativePlainCrownRepresentationCompilation,
    compile_native_plain_crown_representation_query,
    execute_native_plain_crown_representation_query,
)
from tests.test_task_ir_v1 import _semantic_case


def _compile_pair() -> tuple[
    NativePlainCrownRepresentationCompilation,
    NativePlainCrownRepresentationCompilation,
    object,
    object,
    object,
    torch.Tensor,
]:
    legacy_module, input_spec = _semantic_case("residual")
    interval_env, relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    linear_spec = torch.tensor(
        [[1.0, -1.0, 0.5], [-0.5, 0.25, 1.0]], dtype=torch.float32
    )
    kwargs = {
        "interval_env": interval_env,
        "relu_pre": relu_pre,
        "linear_spec_C": linear_spec,
        "intermediate_bounds_hash": "b" * 64,
        "query_id": "native-representation-residual",
        "available_memory_bytes": 1 << 30,
    }
    dense = compile_native_plain_crown_representation_query(
        legacy_module,
        input_spec,
        memory_budget_bytes=1 << 30,
        **kwargs,
    )
    reuse_peak = next(
        candidate.cost.predicted_peak_bytes
        for candidate in dense.source_template.storage_candidates
        if candidate.candidate_id == "storage:native-lifetime-reuse-v1"
    )
    structured = compile_native_plain_crown_representation_query(
        legacy_module,
        input_spec,
        memory_budget_bytes=reuse_peak,
        **kwargs,
    )
    return (
        dense,
        structured,
        legacy_module,
        input_spec,
        relu_pre,
        linear_spec,
    )


def test_budget_selects_real_dense_or_structured_execution_program() -> None:
    dense, structured, legacy_module, input_spec, relu_pre, linear_spec = (
        _compile_pair()
    )

    assert dense.binding.trace.policy_id == DENSE_POLICY_ID
    assert structured.binding.trace.policy_id == STRUCTURED_AFFINE_POLICY_ID
    assert dense.source_instance.storage_decision.candidate_id == (
        "storage:native-retain-all-v1"
    )
    assert structured.source_instance.storage_decision.candidate_id == (
        "storage:native-lifetime-reuse-v1"
    )
    assert dense.source_bound_module.stable_hash() == (
        structured.source_bound_module.stable_hash()
    )
    assert dense.source_template.stable_hash(
        bound_module=dense.source_bound_module
    ) == structured.source_template.stable_hash(
        bound_module=structured.source_bound_module
    )
    assert dense.bound_module is dense.source_bound_module
    assert structured.bound_module.stable_hash() != (
        structured.source_bound_module.stable_hash()
    )
    assert len(dense.binding.trace.events) == 0

    transition_ops = tuple(
        op
        for op in structured.bound_module.graph.ops
        if op.kind in {BoundOpKind.REPRESENTATION_CAST, BoundOpKind.MATERIALIZE}
    )
    assert len(transition_ops) == len(structured.binding.trace.events) > 0
    assert {event.execution_op_id for event in structured.binding.trace.events} == {
        op.op_id for op in transition_ops
    }
    source_actions = tuple(
        action
        for action in structured.source_schedule.actions
        if isinstance(action, MaterializeAction)
    )
    assert {
        (event.transition_candidate_id, event.schedule_action_id)
        for event in structured.binding.trace.events
    } == {
        (action.transition_candidate_id, action.action_id) for action in source_actions
    }
    assert len(structured.task_module.tasks) == len(structured.bound_module.graph.ops)
    assert sum(
        isinstance(action, LaunchAction) for action in structured.schedule.actions
    ) == len(structured.bound_module.graph.ops)

    dense_result, _dense_trace = execute_native_plain_crown_representation_query(
        dense,
        legacy_task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec,
    )
    structured_result, structured_trace = (
        execute_native_plain_crown_representation_query(
            structured,
            legacy_task_module=legacy_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=linear_spec,
        )
    )
    torch.testing.assert_close(
        dense_result.lower, structured_result.lower, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        dense_result.upper, structured_result.upper, atol=0.0, rtol=0.0
    )
    assert {event.execution_op_id for event in structured.binding.trace.events} <= {
        op_id for event in structured_trace.events for op_id in event.op_ids
    }


def test_native_stack_executes_first_class_relu_split_input() -> None:
    legacy_module, input_spec = _semantic_case("cnn")
    _root_env, root_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    splits = {
        name: torch.zeros_like(pre.lower, dtype=torch.int8)
        for name, pre in root_pre.items()
    }
    ambiguous = (
        (root_pre["conv_out"].lower < 0) & (root_pre["conv_out"].upper > 0)
    ).nonzero()
    assert int(ambiguous.shape[0]) > 0
    splits["conv_out"][tuple(ambiguous[0].tolist())] = 1
    interval_env, relu_pre = _forward_ibp_trace_mlp(
        legacy_module, input_spec, relu_split_state=splits
    )
    linear_spec = torch.tensor([[1.0, -1.0, 0.5]])
    compilation = compile_native_plain_crown_representation_query(
        legacy_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec,
        intermediate_bounds_hash="d" * 64,
        query_id="native-representation-split-cnn",
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
        relu_split_state=splits,
        split_state_hash=relu_split_state_hash(splits),
    )
    result, trace = execute_native_plain_crown_representation_query(
        compilation,
        legacy_task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec,
        relu_split_state=splits,
    )
    expected = execute_plain_crown_bound_ir(
        compilation.source_bound_module,
        task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec,
        relu_split_state=splits,
    )

    torch.testing.assert_close(result.lower, expected.lower, atol=0.0, rtol=0.0)
    torch.testing.assert_close(result.upper, expected.upper, atol=0.0, rtol=0.0)
    assert compilation.source_bound_module.domain.split_state_present
    assert compilation.bound_module.domain.split_state_present
    assert compilation.source_template.workload.split_state_present
    assert compilation.execution_template.workload.split_state_present
    assert all(
        capability.supports_split_state
        for capability in compilation.execution_template.capabilities
    )
    assert any(
        "int8" in capability.supported_dtypes
        for capability in (
            *compilation.source_template.capabilities,
            *compilation.execution_template.capabilities,
        )
    )
    assert len(trace.events) == len(compilation.task_module.tasks)

    tampered = {name: tensor.clone() for name, tensor in splits.items()}
    tampered["conv_out"].zero_()
    with pytest.raises(ValueError, match="payload hash"):
        execute_native_plain_crown_representation_query(
            compilation,
            legacy_task_module=legacy_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=linear_spec,
            relu_split_state=tampered,
        )


def test_structured_execution_storage_records_dense_equivalent_size() -> None:
    _dense, structured, *_rest = _compile_pair()
    values = {value.value_id: value for value in structured.bound_module.graph.values}
    storage = structured.execution_template.storage_candidates[0]
    structured_bindings = tuple(
        binding
        for binding in storage.bindings
        if values[binding.value_id].representation == BoundRepresentation.STRUCTURED
    )
    assert structured_bindings
    assert all(
        binding.representation == BoundRepresentation.STRUCTURED
        and binding.size_bytes >= binding.logical_size_bytes
        for binding in structured_bindings
    )
    assert all(
        "performance_claim" in item.key
        for item in structured.execution_template.provenance
        if item.value == "forbidden"
    )


def test_representation_binding_is_deterministic_and_fails_below_minimum() -> None:
    dense, structured, legacy_module, input_spec, relu_pre, linear_spec = (
        _compile_pair()
    )
    repeated = compile_native_plain_crown_representation_query(
        legacy_module,
        input_spec,
        interval_env=_forward_ibp_trace_mlp(legacy_module, input_spec)[0],
        relu_pre=relu_pre,
        linear_spec_C=linear_spec,
        intermediate_bounds_hash="b" * 64,
        query_id="native-representation-residual",
        available_memory_bytes=1 << 30,
        memory_budget_bytes=structured.source_instance.memory_budget_bytes,
    )
    assert repeated.hashes() == structured.hashes()
    assert repeated.binding.trace.canonical_json() == (
        structured.binding.trace.canonical_json()
    )
    with pytest.raises(NoFeasiblePlanError) as error:
        compile_native_plain_crown_representation_query(
            legacy_module,
            input_spec,
            interval_env=_forward_ibp_trace_mlp(legacy_module, input_spec)[0],
            relu_pre=relu_pre,
            linear_spec_C=linear_spec,
            intermediate_bounds_hash="b" * 64,
            query_id="native-representation-residual",
            available_memory_bytes=1 << 30,
            memory_budget_bytes=structured.source_instance.memory_budget_bytes - 1,
        )
    assert any(
        failure.reason == "memory_budget_exceeded" for failure in error.value.failures
    )
    assert dense.hashes()["source_plan_instance_hash"] != (
        structured.hashes()["source_plan_instance_hash"]
    )


def test_binding_rejects_schedule_action_and_trace_tampering() -> None:
    _dense, structured, *_rest = _compile_pair()
    action = next(
        item
        for item in structured.source_schedule.actions
        if isinstance(item, MaterializeAction)
    )
    tampered_action = replace(
        action,
        target_representation=(
            BoundRepresentation.STRUCTURED
            if action.target_representation == BoundRepresentation.DENSE
            else BoundRepresentation.DENSE
        ),
    )
    tampered_schedule = replace(
        structured.source_schedule,
        actions=tuple(
            tampered_action if item is action else item
            for item in structured.source_schedule.actions
        ),
    )
    with pytest.raises(ValueError):
        bind_native_representation_plan(
            structured.source_bound_module,
            template=structured.source_template,
            instance=structured.source_instance,
            schedule=tampered_schedule,
        )

    event = structured.binding.trace.events[0]
    tampered_trace = replace(
        structured.binding.trace,
        events=(replace(event, execution_op_id="missing.transition"),)
        + structured.binding.trace.events[1:],
    )
    tampered_binding = BoundRepresentationBinding(
        structured.bound_module, tampered_trace
    )
    tampered_compilation = replace(structured, binding=tampered_binding)
    with pytest.raises(ValueError, match="not a Task op"):
        tampered_compilation.validate()

    hash_tampered = replace(
        structured,
        binding=BoundRepresentationBinding(
            structured.bound_module,
            replace(
                structured.binding.trace,
                source_plan_template_hash="0" * 64,
            ),
        ),
    )
    with pytest.raises(ValueError, match="source binding identity"):
        hash_tampered.validate()
