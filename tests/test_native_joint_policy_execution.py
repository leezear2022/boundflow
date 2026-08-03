"""Joint native representation and spec-batch policy execution."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.ir.plan import PlanProvenance
from boundflow.ir.schedule import BatchLoopAction
from boundflow.planner.representation_plan_binding import (
    DENSE_POLICY_ID,
    STRUCTURED_AFFINE_POLICY_ID,
)
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_joint_policy_integration import (
    compile_native_plain_crown_joint_policy_query,
    execute_native_plain_crown_joint_policy_query,
)
from tests.test_task_ir_v1 import _semantic_case


def _compile_matrix():
    legacy_module, input_spec = _semantic_case("residual")
    interval_env, relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    linear_spec = torch.tensor(
        [[1.0, -1.0, 0.5], [-0.5, 0.25, 1.0]], dtype=torch.float32
    )
    common = {
        "interval_env": interval_env,
        "relu_pre": relu_pre,
        "linear_spec_C": linear_spec,
        "intermediate_bounds_hash": "d" * 64,
        "query_id": "native-joint-policy-residual",
        "available_memory_bytes": 1 << 30,
        "spec_slice_candidate_size": 1,
    }
    dense_full = compile_native_plain_crown_joint_policy_query(
        legacy_module,
        input_spec,
        memory_budget_bytes=1 << 30,
        max_spec_batch_size=2,
        **common,
    )
    storage = {
        item.candidate_id: item
        for item in dense_full.source_template.storage_candidates
    }
    reuse_budget = storage["storage:native-lifetime-reuse-v1"].cost.predicted_peak_bytes
    dense_sliced = compile_native_plain_crown_joint_policy_query(
        legacy_module,
        input_spec,
        memory_budget_bytes=1 << 30,
        max_spec_batch_size=1,
        **common,
    )
    structured_full = compile_native_plain_crown_joint_policy_query(
        legacy_module,
        input_spec,
        memory_budget_bytes=reuse_budget,
        max_spec_batch_size=2,
        **common,
    )
    structured_sliced = compile_native_plain_crown_joint_policy_query(
        legacy_module,
        input_spec,
        memory_budget_bytes=reuse_budget,
        max_spec_batch_size=1,
        **common,
    )
    return (
        dense_full,
        dense_sliced,
        structured_full,
        structured_sliced,
        legacy_module,
        input_spec,
        relu_pre,
        linear_spec,
    )


def test_joint_policy_builds_four_source_decisions_and_real_children() -> None:
    dense_full, dense_sliced, structured_full, structured_sliced, *_ = _compile_matrix()
    compilations = (
        dense_full,
        dense_sliced,
        structured_full,
        structured_sliced,
    )
    source = dense_full.source_build.module
    template_hashes = {
        item.source_template.stable_hash(bound_module=item.source_build.module)
        for item in compilations
    }
    assert len({item.source_build.module.stable_hash() for item in compilations}) == 1
    assert len(template_hashes) == 1
    assert (
        len({item.hashes()["source_plan_instance_hash"] for item in compilations}) == 4
    )
    assert len({item.hashes()["source_schedule_hash"] for item in compilations}) == 4
    assert source.stable_hash() == dense_full.hashes()["source_bound_module_hash"]

    assert dense_full.binding_trace.selected_representation_policy_id == DENSE_POLICY_ID
    assert (
        dense_sliced.binding_trace.selected_representation_policy_id == DENSE_POLICY_ID
    )
    assert structured_full.binding_trace.selected_representation_policy_id == (
        STRUCTURED_AFFINE_POLICY_ID
    )
    assert structured_sliced.binding_trace.selected_representation_policy_id == (
        STRUCTURED_AFFINE_POLICY_ID
    )
    assert dense_full.source_instance.batch_decision.candidate_id == "batch:full-query"
    assert structured_full.source_instance.batch_decision.candidate_id == (
        "batch:full-query"
    )
    assert dense_sliced.source_instance.batch_decision.candidate_id.endswith("0001")
    assert structured_sliced.source_instance.batch_decision.candidate_id.endswith(
        "0001"
    )
    assert (
        len(dense_full.child_compilations)
        == len(structured_full.child_compilations)
        == 1
    )
    assert (
        len(dense_sliced.child_compilations)
        == len(structured_sliced.child_compilations)
        == 2
    )
    for compilation in compilations:
        assert all(
            child.binding.trace.policy_id
            == compilation.binding_trace.selected_representation_policy_id
            for child in compilation.child_compilations
        )
        assert all(
            child.source_instance.storage_decision.candidate_id
            == compilation.source_instance.storage_decision.candidate_id
            for child in compilation.child_compilations
        )


def test_joint_policy_four_paths_execute_with_equal_semantics() -> None:
    (
        dense_full,
        dense_sliced,
        structured_full,
        structured_sliced,
        legacy_module,
        input_spec,
        relu_pre,
        linear_spec,
    ) = _compile_matrix()
    results = []
    traces = []
    for compilation in (
        dense_full,
        dense_sliced,
        structured_full,
        structured_sliced,
    ):
        result, trace = execute_native_plain_crown_joint_policy_query(
            compilation,
            legacy_task_module=legacy_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=linear_spec,
        )
        results.append(result)
        traces.append(trace)
    for result in results[1:]:
        torch.testing.assert_close(result.lower, results[0].lower, atol=0.0, rtol=0.0)
        torch.testing.assert_close(result.upper, results[0].upper, atol=0.0, rtol=0.0)
    assert len({trace.binding_hash for trace in traces}) == 4
    assert tuple(trace.representation_policy_id for trace in traces) == (
        DENSE_POLICY_ID,
        DENSE_POLICY_ID,
        STRUCTURED_AFFINE_POLICY_ID,
        STRUCTURED_AFFINE_POLICY_ID,
    )


def test_joint_policy_rejects_objective_range_policy_and_provenance_tamper() -> None:
    (
        _dense_full,
        _dense_sliced,
        _structured_full,
        structured_sliced,
        legacy_module,
        input_spec,
        relu_pre,
        linear_spec,
    ) = _compile_matrix()
    changed = linear_spec.clone()
    changed[0, 0] += 1.0
    with pytest.raises(ValueError, match="objective hash"):
        execute_native_plain_crown_joint_policy_query(
            structured_sliced,
            legacy_task_module=legacy_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=changed,
        )

    first = structured_sliced.binding_trace.slices[0]
    broken_range = replace(
        structured_sliced.binding_trace,
        slices=(
            replace(first, stop_index=2),
            *structured_sliced.binding_trace.slices[1:],
        ),
    )
    with pytest.raises(ValueError, match="overlap"):
        replace(structured_sliced, binding_trace=broken_range).validate()

    child = structured_sliced.child_compilations[0]
    broken_child = replace(
        child.binding.trace,
        policy_id=DENSE_POLICY_ID,
    )
    with pytest.raises(ValueError):
        replace(
            structured_sliced,
            child_compilations=(
                replace(child, binding=replace(child.binding, trace=broken_child)),
                *structured_sliced.child_compilations[1:],
            ),
        ).validate()

    provenance = tuple(
        (
            PlanProvenance(item.key, "storage:native-retain-all-v1")
            if item.key == "required_storage_candidate_id"
            else item
        )
        for item in child.source_instance.provenance
    )
    with pytest.raises(ValueError, match="required query policy"):
        replace(child.source_instance, provenance=provenance).validate(
            template=child.source_template,
            bound_module=child.source_bound_module,
        )


def test_joint_policy_sliced_schedule_owns_exact_ranges() -> None:
    _dense_full, dense_sliced, _structured_full, structured_sliced, *_ = (
        _compile_matrix()
    )
    for compilation in (dense_sliced, structured_sliced):
        loop = next(
            item
            for item in compilation.source_schedule.actions
            if isinstance(item, BatchLoopAction)
        )
        assert loop.axis == "spec"
        assert tuple((item.start_index, item.stop_index) for item in loop.slices) == (
            (0, 1),
            (1, 2),
        )
