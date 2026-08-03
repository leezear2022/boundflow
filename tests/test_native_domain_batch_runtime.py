"""Native BaB-style input-domain batching and exact child state ownership."""

# pylint: disable=missing-function-docstring,too-many-locals
# pylint: disable=assignment-from-no-return,unpacking-non-sequence,no-member

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.domains.interval import IntervalState
from boundflow.ir.schedule import BatchLoopAction
from boundflow.runtime.native_domain_batch_runtime import (
    PARENT_STATE_VALIDITY,
    build_deterministic_box_domain_queries,
    compile_native_domain_batch_query,
    execute_native_domain_batch_query,
    execute_native_domain_serial_reference,
)
from boundflow.runtime.task_executor import InputSpec
from tests.test_task_ir_v1 import _semantic_case


def _domain_case(*, max_domain_batch_size: int = 4):
    legacy_module, original_spec = _semantic_case("residual")
    lower, upper = original_spec.perturbation.bounding_box(original_spec.center)
    root_spec = InputSpec.box(
        value_name=original_spec.value_name,
        lower=lower[:1],
        upper=upper[:1],
    )
    query_specs = build_deterministic_box_domain_queries(
        root_spec,
        root_query_id="toy-bab-root",
        split_depth=3,
    )
    objective = torch.tensor([[1.0, -1.0, 0.5]], dtype=torch.float32)
    compilation = compile_native_domain_batch_query(
        legacy_module,
        query_specs,
        linear_spec_C=objective,
        query_id="toy-native-domain-batch",
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
        domain_batch_candidate_size=4,
        max_domain_batch_size=max_domain_batch_size,
    )
    return legacy_module, root_spec, query_specs, objective, compilation


def test_domain_split_builds_eight_strict_ordered_leaf_boxes() -> None:
    _legacy, root_spec, query_specs, _objective, _compilation = _domain_case()
    root_lower, root_upper = root_spec.perturbation.bounding_box(root_spec.center)
    assert len(query_specs) == 8
    assert tuple(item.query_id for item in query_specs) == tuple(
        f"toy-bab-root:d03:n{index:04d}" for index in range(8)
    )
    assert len({item.parent_query_id for item in query_specs}) == 4
    assert all(item.depth == 3 for item in query_specs)
    assert tuple(item.branch_ordinal for item in query_specs) == (0, 1) * 4
    for item in query_specs:
        lower, upper = item.input_spec.perturbation.bounding_box(item.input_spec.center)
        assert bool((lower >= root_lower).all())
        assert bool((upper <= root_upper).all())
        assert not (torch.equal(lower, root_lower) and torch.equal(upper, root_upper))


def test_domain_plan_selects_two_packed_slices_or_one_full_slice() -> None:
    *_prefix, packed = _domain_case(max_domain_batch_size=4)
    *_prefix, full = _domain_case(max_domain_batch_size=8)
    assert packed.binding_trace.selected_batch_candidate_id == (
        "batch:native-domain-sliced-v1:0004"
    )
    assert full.binding_trace.selected_batch_candidate_id == "batch:full-query"
    assert len(packed.child_compilations) == 2
    assert len(full.child_compilations) == 1
    assert (
        packed.hashes()["source_bound_module_hash"]
        == full.hashes()["source_bound_module_hash"]
    )
    assert (
        packed.hashes()["source_plan_template_hash"]
        == full.hashes()["source_plan_template_hash"]
    )
    assert (
        packed.hashes()["source_plan_instance_hash"]
        != full.hashes()["source_plan_instance_hash"]
    )
    loop = next(
        item
        for item in packed.source_schedule.actions
        if isinstance(item, BatchLoopAction)
    )
    assert loop.axis == "domain"
    assert tuple(len(item.query_ids) for item in loop.slices) == (4, 4)


def test_domain_packed_two_children_matches_eight_serial_exactly() -> None:
    legacy, _root, query_specs, _objective, compilation = _domain_case()
    packed, packed_trace = execute_native_domain_batch_query(
        compilation, legacy_task_module=legacy
    )
    serial, serial_trace = execute_native_domain_serial_reference(
        compilation,
        legacy_task_module=legacy,
        available_memory_bytes=1 << 30,
    )
    assert packed_trace.packed_child_stack_count == 2
    assert serial_trace.serial_child_stack_count == 8
    assert packed_trace.parent_state_consumed_as_exact is False
    assert serial_trace.parent_state_consumed_as_exact is False
    assert tuple(item.query_id for item in packed) == tuple(
        item.query_id for item in query_specs
    )
    assert tuple(item.parent_query_id for item in packed) == tuple(
        item.parent_query_id for item in query_specs
    )
    for packed_item, serial_item in zip(packed, serial):
        torch.testing.assert_close(
            packed_item.result.lower,
            serial_item.result.lower,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            packed_item.result.upper,
            serial_item.result.upper,
            atol=0.0,
            rtol=0.0,
        )


def test_domain_child_states_are_exact_and_parent_is_warm_start_only() -> None:
    _legacy, _root, query_specs, _objective, compilation = _domain_case()
    states = compilation.binding_trace.query_states
    assert len(states) == len(query_specs) == 8
    assert len({item.exact_state_hash for item in states}) == 8
    assert all(item.parent_state_validity == PARENT_STATE_VALIDITY for item in states)
    assert all(item.parent_state_consumed_as_exact is False for item in states)
    assert all(item.parent_state_hash != item.exact_state_hash for item in states)
    assert tuple(item.query_id for item in states) == tuple(
        item.query_id for item in query_specs
    )
    assert tuple(
        item.child_exact_state_hash for item in compilation.binding_trace.slices
    ) == tuple(item.exact_state_hash for item in compilation.child_payloads)


def test_domain_runtime_rejects_lineage_state_and_range_tamper() -> None:
    _legacy, _root, _query_specs, _objective, compilation = _domain_case()
    first_state = compilation.binding_trace.query_states[0]
    broken_state = replace(first_state, parent_state_consumed_as_exact=True)
    with pytest.raises(ValueError, match="promoted"):
        replace(
            compilation,
            binding_trace=replace(
                compilation.binding_trace,
                query_states=(
                    broken_state,
                    *compilation.binding_trace.query_states[1:],
                ),
            ),
        ).validate()

    first_slice = compilation.binding_trace.slices[0]
    broken_slice = replace(first_slice, stop_index=5)
    with pytest.raises(ValueError, match="slice"):
        replace(
            compilation,
            binding_trace=replace(
                compilation.binding_trace,
                slices=(broken_slice, *compilation.binding_trace.slices[1:]),
            ),
        ).validate()

    payload = compilation.child_payloads[0]
    first_key = sorted(payload.relu_pre)[0]
    original = payload.relu_pre[first_key]
    changed_lower = original.lower.clone()
    changed_lower.reshape(-1)[0] -= 1.0
    changed_relu = {
        **payload.relu_pre,
        first_key: IntervalState(changed_lower, original.upper.clone()),
    }
    with pytest.raises(ValueError, match="exact state"):
        replace(
            compilation,
            child_payloads=(
                replace(payload, relu_pre=changed_relu),
                *compilation.child_payloads[1:],
            ),
        ).validate()


def test_domain_split_rejects_non_box_and_insufficient_width() -> None:
    linf = InputSpec.linf(value_name="input", center=torch.zeros(1, 3), eps=0.1)
    with pytest.raises(TypeError, match="BoxPerturbation"):
        build_deterministic_box_domain_queries(
            linf, root_query_id="root", split_depth=2
        )
    point = InputSpec.box(
        value_name="input",
        lower=torch.zeros(1, 3),
        upper=torch.zeros(1, 3),
    )
    with pytest.raises(ValueError, match="positive-width"):
        build_deterministic_box_domain_queries(
            point, root_query_id="root", split_depth=1
        )
