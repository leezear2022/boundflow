"""Optimizer Schedule integration with native ReLU-split BaB queue."""

# pylint: disable=missing-function-docstring,redefined-outer-name,too-many-locals

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.ir.bound import IntermediateBoundSource
from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
    NATIVE_REEXECUTION_ATOL,
    NATIVE_REEXECUTION_TRACE_MAX_ABS_DIFF,
    NativeOptimizedReluSplitBabTrace,
    PerChildRefinementStrategy,
    compare_native_optimized_bab_states,
    execute_native_optimized_relu_split_bab,
    run_native_optimized_relu_split_bab,
)
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from boundflow.runtime.task_executor import InputSpec


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="native-optimized-bab-toy",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp("linear", "linear1", ["input", "W1", "b1"], ["h1"]),
                    TaskOp("relu", "relu1", ["h1"], ["r1"]),
                    TaskOp("linear", "linear2", ["r1", "W2", "b2"], ["out"]),
                ],
                input_values=["input"],
                output_values=["out"],
            )
        ],
        entry_task_id="native-optimized-bab-toy",
        bindings={
            "params": {
                "W1": torch.tensor(
                    [[1.0, -0.5, 0.25], [-0.25, 0.75, 1.0], [0.5, 0.5, -1.0]]
                ),
                "b1": torch.tensor([0.1, -0.2, 0.05]),
                "W2": torch.tensor([[0.75, -1.0, 0.5], [-0.5, 0.25, 1.25]]),
                "b2": torch.tensor([0.15, -0.1]),
            }
        },
    )


def _spec() -> InputSpec:
    return InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.3, -0.6, -0.1]]),
        upper=torch.tensor([[0.7, 0.4, 0.9]]),
    )


def _policy() -> NativeAlphaBetaOptimizerPolicy:
    return NativeAlphaBetaOptimizerPolicy(
        steps=1,
        lr=0.1,
        alpha_init=0.5,
        beta_init=0.0,
    )


def _per_child_policy() -> NativeIntermediateRefinementPolicyIR:
    return NativeIntermediateRefinementPolicyIR(
        passes=1,
        max_neurons_per_relu=2,
        backward_chunk_size=2,
        candidate_policy_id="objective_influence_width_per_relu_v1",
    )


def _run_per_child(
    *,
    batch_size: int,
    strategy: PerChildRefinementStrategy = "independent_exact_split_v1",
):
    return execute_native_optimized_relu_split_bab(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[1.0, -1.0]]),
        run_id="native-per-child-refinement-toy",
        config=NativeReluSplitBabConfig(
            max_nodes=7,
            max_depth=2,
            expansion_batch_size=2,
            max_eval_batch_size=batch_size,
            threshold=1e6,
        ),
        optimizer_policy=_policy(),
        intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
        per_child_refinement_policy=_per_child_policy(),
        per_child_refinement_strategy=strategy,
    )


def _run(*, batch_size: int, max_nodes: int = 15) -> NativeOptimizedReluSplitBabTrace:
    return run_native_optimized_relu_split_bab(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[1.0, -1.0]]),
        run_id="native-optimized-bab-toy",
        config=NativeReluSplitBabConfig(
            max_nodes=max_nodes,
            max_depth=3,
            expansion_batch_size=2,
            max_eval_batch_size=batch_size,
            threshold=1e6,
        ),
        optimizer_policy=_policy(),
    )


@pytest.fixture(scope="module")
def complete_traces() -> (
    tuple[NativeOptimizedReluSplitBabTrace, NativeOptimizedReluSplitBabTrace]
):
    return _run(batch_size=4), _run(batch_size=1)


def test_packed_and_serial_optimizer_queues_match(
    complete_traces: tuple[
        NativeOptimizedReluSplitBabTrace, NativeOptimizedReluSplitBabTrace
    ],
) -> None:
    packed, serial = complete_traces

    assert packed.status == serial.status == "complete"
    assert packed.final_frontier_node_ids == serial.final_frontier_node_ids == ()
    assert len(packed.evaluations) == len(serial.evaluations) == 15
    assert len(packed.decisions) == len(serial.decisions) == 15
    assert packed.native_stack_count == 5
    assert serial.native_stack_count == 15
    assert packed.logical_queue_signature() == serial.logical_queue_signature()
    assert all(
        actual.lower == expected.lower
        and actual.upper == expected.upper
        and actual.selected_state_hash == expected.selected_state_hash
        and actual.node.split_state_hash == expected.node.split_state_hash
        for actual, expected in zip(packed.evaluations, serial.evaluations)
    )


def test_parent_state_is_warm_initialization_only_and_all_ir_stacks_execute(
    complete_traces: tuple[
        NativeOptimizedReluSplitBabTrace, NativeOptimizedReluSplitBabTrace
    ],
) -> None:
    packed, _serial = complete_traces
    by_id = {item.node.node_id: item for item in packed.evaluations}

    assert packed.native_stacks[0].warm_start_kind == "none"
    assert packed.native_stacks[0].warm_source_state_hash is None
    assert all(
        item.warm_start_kind == "monotonic_split_refinement"
        and item.parent_state_consumed_as_exact is False
        and item.parent_selected_state_hash
        == by_id[item.node.parent_node_id or ""].selected_state_hash
        for item in packed.evaluations[1:]
    )
    assert all(
        stack.optimizer_action_count == 8
        and stack.optimizer_evaluation_count == 2
        and stack.optimizer_backward_count == 1
        and stack.optimizer_projection_count == 1
        and stack.native_task_count == stack.native_schedule_launch_count
        and stack.native_task_count == stack.native_task_trace_event_count
        and stack.selected_native_lower_max_abs_diff <= NATIVE_REEXECUTION_ATOL
        and stack.selected_native_upper_max_abs_diff <= NATIVE_REEXECUTION_ATOL
        for stack in packed.native_stacks
    )
    assert all(stack.beta_gradient_l1 > 0.0 for stack in packed.native_stacks[1:])


def test_node_budget_preserves_optimized_frontier() -> None:
    trace = _run(batch_size=4, max_nodes=7)

    assert trace.status == "budget_exhausted"
    assert trace.termination_reason == "node_budget_exhausted"
    assert len(trace.evaluations) == 7
    assert len(trace.decisions) == 3
    assert len(trace.final_frontier_node_ids) == 4
    assert trace.native_stack_count == 3
    trace.validate()


def test_default_queue_payload_omits_per_child_extension(
    complete_traces: tuple[
        NativeOptimizedReluSplitBabTrace, NativeOptimizedReluSplitBabTrace
    ],
) -> None:
    packed, _serial = complete_traces
    payload = packed.to_dict()

    assert "per_child_refinement_policy" not in payload
    assert "per_child_refinements" not in payload
    assert all(
        "intermediate_refinement_trace_hash" not in evaluation
        for evaluation in payload["evaluations"]
    )


def test_trace_rejects_parent_optimizer_and_native_tampering(
    complete_traces: tuple[
        NativeOptimizedReluSplitBabTrace, NativeOptimizedReluSplitBabTrace
    ],
) -> None:
    packed, _serial = complete_traces
    child_index = next(
        index for index, item in enumerate(packed.evaluations) if item.node.depth == 1
    )
    child = packed.evaluations[child_index]
    evaluations = list(packed.evaluations)
    evaluations[child_index] = replace(child, parent_selected_state_hash="f" * 64)
    with pytest.raises(ValueError, match="parent state link"):
        replace(packed, evaluations=tuple(evaluations)).validate()

    stack = packed.native_stacks[1]
    stacks = list(packed.native_stacks)
    stacks[1] = replace(stack, optimizer_action_count=7)
    with pytest.raises(ValueError, match="stack trace"):
        replace(packed, native_stacks=tuple(stacks)).validate()

    stacks[1] = replace(stack, selected_native_lower_max_abs_diff=float("nan"))
    with pytest.raises(ValueError, match="stack trace"):
        replace(packed, native_stacks=tuple(stacks)).validate()

    stacks[1] = replace(
        stack,
        selected_native_lower_max_abs_diff=(
            NATIVE_REEXECUTION_TRACE_MAX_ABS_DIFF * 2.0
        ),
    )
    with pytest.raises(ValueError, match="stack trace"):
        replace(packed, native_stacks=tuple(stacks)).validate()


def test_invalid_policy_and_input_fail_closed() -> None:
    with pytest.raises(ValueError, match="optimizer policy"):
        run_native_optimized_relu_split_bab(
            _module(),
            _spec(),
            linear_spec_C=torch.tensor([[1.0, -1.0]]),
            run_id="invalid-policy",
            config=NativeReluSplitBabConfig(
                max_nodes=1,
                max_depth=0,
                expansion_batch_size=1,
                max_eval_batch_size=1,
            ),
            optimizer_policy=NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.0),
        )


def test_external_intermediate_semantics_survive_child_batches() -> None:
    module = _module()
    spec = _spec()
    _interval_env, external = _forward_ibp_trace_mlp(module, spec)
    trace = run_native_optimized_relu_split_bab(
        module,
        spec,
        linear_spec_C=torch.tensor([[1.0, -1.0]]),
        run_id="native-optimized-bab-external",
        config=NativeReluSplitBabConfig(
            max_nodes=7,
            max_depth=2,
            expansion_batch_size=1,
            max_eval_batch_size=4,
            threshold=1e6,
        ),
        optimizer_policy=NativeAlphaBetaOptimizerPolicy(
            steps=1,
            lr=0.1,
            alpha_initialization_mode="adaptive",
        ),
        relu_pre_override=external,
        intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
    )

    assert len(trace.evaluations) == 7
    assert trace.native_stacks[0].warm_start_kind == "none"
    assert all(
        stack.warm_start_kind == "monotonic_split_refinement"
        for stack in trace.native_stacks[1:]
    )
    assert all(
        stack.selected_native_lower_max_abs_diff <= NATIVE_REEXECUTION_ATOL
        and stack.selected_native_upper_max_abs_diff <= NATIVE_REEXECUTION_ATOL
        for stack in trace.native_stacks
    )
    trace.validate()


def test_external_intermediate_queue_provenance_mismatch_fails_closed() -> None:
    module = _module()
    spec = _spec()
    _interval_env, external = _forward_ibp_trace_mlp(module, spec)
    config = NativeReluSplitBabConfig(
        max_nodes=1,
        max_depth=0,
        expansion_batch_size=1,
        max_eval_batch_size=1,
    )
    with pytest.raises(ValueError, match="semantics/provenance differ"):
        run_native_optimized_relu_split_bab(
            module,
            spec,
            linear_spec_C=torch.tensor([[1.0, -1.0]]),
            run_id="native-optimized-bab-external-missing",
            config=config,
            optimizer_policy=_policy(),
            intermediate_bound_source=IntermediateBoundSource.EXTERNAL_VERIFIER,
        )
    with pytest.raises(ValueError, match="semantics/provenance differ"):
        run_native_optimized_relu_split_bab(
            module,
            spec,
            linear_spec_C=torch.tensor([[1.0, -1.0]]),
            run_id="native-optimized-bab-external-wrong-owner",
            config=config,
            optimizer_policy=_policy(),
            relu_pre_override=external,
        )


@pytest.fixture(scope="module")
def per_child_executions():
    return _run_per_child(batch_size=4), _run_per_child(batch_size=1)


@pytest.fixture(scope="module")
def ancestral_per_child_executions():
    return (
        _run_per_child(batch_size=4, strategy="ancestral_constraint_carry_v1"),
        _run_per_child(batch_size=1, strategy="ancestral_constraint_carry_v1"),
    )


def test_per_child_refinement_packed_and_serial_semantics_match(
    per_child_executions,
) -> None:
    packed, serial = per_child_executions
    packed_trace = packed.trace
    serial_trace = serial.trace

    assert packed_trace.status == serial_trace.status == "complete"
    assert len(packed_trace.evaluations) == len(serial_trace.evaluations) == 7
    assert packed_trace.logical_queue_signature() == (
        serial_trace.logical_queue_signature()
    )
    assert [item.to_dict() for item in packed_trace.per_child_refinements] == [
        item.to_dict() for item in serial_trace.per_child_refinements
    ]
    assert all(
        actual.lower == pytest.approx(expected.lower, abs=1e-6)
        and actual.upper == pytest.approx(expected.upper, abs=1e-6)
        and actual.node.split_state_hash == expected.node.split_state_hash
        for actual, expected in zip(packed_trace.evaluations, serial_trace.evaluations)
    )
    state_comparison = compare_native_optimized_bab_states(packed, serial)
    assert state_comparison["split_tensors_exact"] is True
    assert state_comparison["stable_scope_fields_equal"] is True
    assert state_comparison["intermediate_scope_hashes_equal"] is True
    assert state_comparison["alpha_max_abs_diff"] <= 1e-6
    assert state_comparison["beta_max_abs_diff"] <= 1e-6


def test_per_child_refinement_lineage_is_exact_and_parent_is_warm_only(
    per_child_executions,
) -> None:
    packed, _serial = per_child_executions
    trace = packed.trace
    refinements = dict(packed.per_child_refinement_executions)
    records = {item.node_id: item for item in trace.per_child_refinements}
    evaluations = {item.node.node_id: item for item in trace.evaluations}

    assert tuple(refinements) == tuple(evaluations)
    for node_id, execution in refinements.items():
        evaluation = evaluations[node_id]
        record = records[node_id]
        assert execution.program.plan.split_state_hash == (
            evaluation.node.split_state_hash
        )
        assert execution.program.plan.initial_intermediate_bounds_hash == (
            record.initial_intermediate_bounds_hash
        )
        assert execution.trace.final_intermediate_bounds_hash == (
            record.final_intermediate_bounds_hash
        )
        assert record.parent_refinement_consumed_as_exact is False
        for name, initial in execution.program.initial_relu_pre.items():
            refined = execution.relu_pre[name]
            assert bool((refined.lower >= initial.lower).all())
            assert bool((refined.upper <= initial.upper).all())
        if evaluation.node.depth > 0:
            parent = evaluations[evaluation.node.parent_node_id or ""]
            assert evaluation.warm_start_kind == "monotonic_split_refinement"
            assert evaluation.parent_selected_state_hash == parent.selected_state_hash
            assert evaluation.parent_state_consumed_as_exact is False

    assert any(
        records[item.node.node_id].initial_intermediate_bounds_hash
        != records[item.node.parent_node_id or ""].final_intermediate_bounds_hash
        for item in trace.evaluations
        if item.node.depth > 0
    )


def test_per_child_root_matches_same_policy_root_global_reuse(
    per_child_executions,
) -> None:
    per_child, _serial = per_child_executions
    root = per_child.trace.evaluations[0]
    root_refinement = per_child.per_child_refinement_executions[0][1]
    root_global = run_native_optimized_relu_split_bab(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[1.0, -1.0]]),
        run_id="native-per-child-refinement-toy",
        config=NativeReluSplitBabConfig(
            max_nodes=7,
            max_depth=2,
            expansion_batch_size=2,
            max_eval_batch_size=4,
            threshold=1e6,
        ),
        optimizer_policy=_policy(),
        relu_pre_override=root_refinement.relu_pre,
        intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
    )

    assert root.lower == pytest.approx(root_global.evaluations[0].lower, abs=1e-6)
    assert root.upper == pytest.approx(root_global.evaluations[0].upper, abs=1e-6)
    assert root.node.split_state_hash == (
        root_global.evaluations[0].node.split_state_hash
    )


def test_per_child_refinement_trace_tampering_fails_closed(
    per_child_executions,
) -> None:
    packed, _serial = per_child_executions
    trace = packed.trace
    records = list(trace.per_child_refinements)
    records[1] = replace(records[1], node_split_state_hash="f" * 64)
    with pytest.raises(ValueError, match="node/refinement binding"):
        replace(trace, per_child_refinements=tuple(records)).validate()

    evaluations = list(trace.evaluations)
    evaluations[1] = replace(
        evaluations[1], intermediate_refinement_trace_hash="f" * 64
    )
    with pytest.raises(ValueError, match="node/refinement binding"):
        replace(trace, evaluations=tuple(evaluations)).validate()


def test_per_child_refinement_admission_fails_closed() -> None:
    config = NativeReluSplitBabConfig(
        max_nodes=1,
        max_depth=0,
        expansion_batch_size=1,
        max_eval_batch_size=1,
    )
    with pytest.raises(ValueError, match="semantics/provenance differ"):
        execute_native_optimized_relu_split_bab(
            _module(),
            _spec(),
            linear_spec_C=torch.tensor([[1.0, -1.0]]),
            run_id="per-child-wrong-provenance",
            config=config,
            optimizer_policy=_policy(),
            per_child_refinement_policy=_per_child_policy(),
        )
    with pytest.raises(ValueError, match="must be objective-directed"):
        execute_native_optimized_relu_split_bab(
            _module(),
            _spec(),
            linear_spec_C=torch.tensor([[1.0, -1.0]]),
            run_id="per-child-wrong-policy",
            config=config,
            optimizer_policy=_policy(),
            intermediate_bound_source=IntermediateBoundSource.NATIVE_REFINED,
            per_child_refinement_policy=NativeIntermediateRefinementPolicyIR(
                passes=1,
                max_neurons_per_relu=2,
                backward_chunk_size=2,
            ),
        )
    with pytest.raises(ValueError, match="strategy requires a policy"):
        execute_native_optimized_relu_split_bab(
            _module(),
            _spec(),
            linear_spec_C=torch.tensor([[1.0, -1.0]]),
            run_id="ancestral-missing-policy",
            config=config,
            optimizer_policy=_policy(),
            per_child_refinement_strategy="ancestral_constraint_carry_v1",
        )


def test_ancestral_constraint_packed_and_serial_semantics_match(
    ancestral_per_child_executions,
) -> None:
    packed, serial = ancestral_per_child_executions

    assert packed.trace.logical_queue_signature() == (
        serial.trace.logical_queue_signature()
    )
    assert [item.to_dict() for item in packed.trace.per_child_refinements] == [
        item.to_dict() for item in serial.trace.per_child_refinements
    ]
    assert packed.trace.to_dict()["per_child_refinement_strategy"] == (
        "ancestral_constraint_carry_v1"
    )
    state_comparison = compare_native_optimized_bab_states(packed, serial)
    assert state_comparison["split_tensors_exact"] is True
    assert state_comparison["stable_scope_fields_equal"] is True
    assert state_comparison["intermediate_scope_hashes_equal"] is True
    assert state_comparison["alpha_max_abs_diff"] <= 1e-6
    assert state_comparison["beta_max_abs_diff"] <= 1e-6


def test_ancestral_constraint_lineage_and_double_monotonicity(
    ancestral_per_child_executions,
) -> None:
    packed, _serial = ancestral_per_child_executions
    records = {item.node_id: item for item in packed.trace.per_child_refinements}
    executions = dict(packed.per_child_refinement_executions)
    evaluations = {item.node.node_id: item for item in packed.trace.evaluations}

    for node_id, execution in executions.items():
        evaluation = evaluations[node_id]
        record = records[node_id]
        _local_env, local_pre = _forward_ibp_trace_mlp(
            _module(),
            _spec(),
            relu_split_state=dict(execution.program.split_state),
        )
        for name, local in local_pre.items():
            initial = execution.program.initial_relu_pre[name]
            final = execution.relu_pre[name]
            assert bool((initial.lower >= local.lower).all())
            assert bool((initial.upper <= local.upper).all())
            assert bool((final.lower >= initial.lower).all())
            assert bool((final.upper <= initial.upper).all())
        if evaluation.node.depth == 0:
            assert record.source_parent_node_id is None
            continue
        parent_id = evaluation.node.parent_node_id or ""
        parent_record = records[parent_id]
        assert record.source_parent_node_id == parent_id
        assert record.source_consumption == "sound_constraint_only"
        assert record.source_intermediate_constraints_hash == (
            parent_record.final_intermediate_bounds_hash
        )
        assert record.source_refinement_plan_hash == (
            parent_record.refinement_plan_hash
        )
        assert record.source_refinement_semantic_trace_hash == (
            parent_record.refinement_semantic_trace_hash
        )
        assert record.parent_refinement_consumed_as_exact is False


def test_ancestral_constraint_parent_lineage_tampering_fails_closed(
    ancestral_per_child_executions,
) -> None:
    packed, _serial = ancestral_per_child_executions
    trace = packed.trace
    records = list(trace.per_child_refinements)
    child_index = next(
        index for index, item in enumerate(trace.evaluations) if item.node.depth == 1
    )
    records[child_index] = replace(
        records[child_index], source_intermediate_constraints_hash="f" * 64
    )
    evaluations = list(trace.evaluations)
    evaluations[child_index] = replace(
        evaluations[child_index],
        intermediate_refinement_trace_hash=records[child_index].stable_hash(),
    )
    with pytest.raises(ValueError, match="parent lineage differs"):
        replace(
            trace,
            evaluations=tuple(evaluations),
            per_child_refinements=tuple(records),
        ).validate()
