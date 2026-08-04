"""Sound property verdict and concrete witness replay tests."""

# pylint: disable=missing-function-docstring,redefined-outer-name,duplicate-code

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
    NativeOptimizedReluSplitBabExecution,
    execute_native_optimized_relu_split_bab,
)
from boundflow.runtime.native_property_verdict import (
    NativePropertyVerdictExecution,
    derive_native_property_verdict,
)
from boundflow.runtime.native_relu_split_bab_runtime import NativeReluSplitBabConfig
from boundflow.runtime.task_executor import (
    InputSpec,
    execute_task_module_concrete,
)


def _module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="native-property-toy",
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
        entry_task_id="native-property-toy",
        bindings={
            "params": {
                "W1": torch.tensor([[1.0]]),
                "b1": torch.tensor([0.0]),
                "W2": torch.tensor([[1.0]]),
                "b2": torch.tensor([0.0]),
            }
        },
    )


def _spec() -> InputSpec:
    return InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-1.0]]),
        upper=torch.tensor([[1.0]]),
    )


def _policy() -> NativeAlphaBetaOptimizerPolicy:
    return NativeAlphaBetaOptimizerPolicy(steps=1, lr=0.1)


def _queue(
    *, threshold: float, max_nodes: int = 1, max_depth: int = 3
) -> NativeOptimizedReluSplitBabExecution:
    return execute_native_optimized_relu_split_bab(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[1.0]]),
        run_id=f"native-property-toy-{threshold}-{max_nodes}-{max_depth}",
        config=NativeReluSplitBabConfig(
            max_nodes=max_nodes,
            max_depth=max_depth,
            expansion_batch_size=1,
            max_eval_batch_size=1,
            threshold=threshold,
        ),
        optimizer_policy=_policy(),
    )


@pytest.fixture(scope="module")
def verified_queue() -> NativeOptimizedReluSplitBabExecution:
    return _queue(threshold=-2.0, max_depth=0)


@pytest.fixture(scope="module")
def budget_queue() -> NativeOptimizedReluSplitBabExecution:
    return _queue(threshold=0.5)


@pytest.fixture(scope="module")
def depth_queue() -> NativeOptimizedReluSplitBabExecution:
    return _queue(threshold=0.5, max_depth=0)


@pytest.fixture(scope="module")
def split_queue() -> NativeOptimizedReluSplitBabExecution:
    return _queue(threshold=1.0, max_nodes=3)


def test_concrete_task_ir_execution_exposes_primal_values() -> None:
    execution = execute_task_module_concrete(_module(), torch.tensor([[0.25]]))

    assert execution.output_value == "out"
    assert torch.equal(execution.output, torch.tensor([[0.25]]))
    assert tuple(execution.value_map()) == ("input", "h1", "r1", "out")
    assert torch.equal(execution.value_map()["h1"], torch.tensor([[0.25]]))


def test_verified_requires_closed_sound_prune_proof(
    verified_queue: NativeOptimizedReluSplitBabExecution,
) -> None:
    result = derive_native_property_verdict(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[1.0]]),
        queue_execution=verified_queue,
    )

    assert result.trace.status == "verified"
    assert result.trace.reason == "all_leaves_soundly_pruned"
    assert result.trace.sound_pruned_leaf_node_ids == (
        verified_queue.trace.evaluations[0].node.node_id,
    )
    assert not result.trace.unresolved_leaf_node_ids
    assert result.counterexample_input is None


def test_unproven_prune_reason_cannot_inflate_to_verified(
    verified_queue: NativeOptimizedReluSplitBabExecution,
) -> None:
    decision = verified_queue.trace.decisions[0]
    tampered_queue = replace(
        verified_queue,
        trace=replace(
            verified_queue.trace,
            decisions=(replace(decision, reason="unproven_prune"),),
        ),
    )
    result = derive_native_property_verdict(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[1.0]]),
        queue_execution=tampered_queue,
    )

    assert result.trace.status == "unknown"
    assert result.trace.reason == "unproven_prune_open"
    assert not result.trace.sound_pruned_leaf_node_ids
    assert result.trace.unresolved_leaf_node_ids == (
        tampered_queue.trace.evaluations[0].node.node_id,
    )


def test_budget_and_depth_terminal_are_unknown(
    budget_queue: NativeOptimizedReluSplitBabExecution,
    depth_queue: NativeOptimizedReluSplitBabExecution,
) -> None:
    budget = derive_native_property_verdict(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[1.0]]),
        queue_execution=budget_queue,
    )
    depth = derive_native_property_verdict(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[1.0]]),
        queue_execution=depth_queue,
    )

    assert budget.trace.status == "unknown"
    assert budget.trace.reason == "node_budget_frontier_open"
    assert budget.trace.unresolved_leaf_node_ids == (
        budget_queue.trace.evaluations[0].node.node_id,
    )
    assert depth.trace.status == "unknown"
    assert depth.trace.reason == "configured_depth_terminal_open"
    assert depth.trace.unresolved_leaf_node_ids == (
        depth_queue.trace.evaluations[0].node.node_id,
    )


def test_unsafe_requires_reexecuted_concrete_counterexample(
    budget_queue: NativeOptimizedReluSplitBabExecution,
) -> None:
    root_id = budget_queue.trace.evaluations[0].node.node_id
    result = derive_native_property_verdict(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[1.0]]),
        queue_execution=budget_queue,
        candidate_counterexamples=((root_id, torch.tensor([[0.0]])),),
    )

    assert result.trace.status == "unsafe"
    assert result.trace.reason == "concrete_counterexample_reexecuted"
    assert result.trace.counterexample is not None
    assert result.trace.counterexample.objective_value == 0.0
    assert result.trace.counterexample.objective_margin == -0.5
    assert result.trace.counterexample.satisfied_split_count == 0
    assert result.trace.counterexample.split_constraint_min_margin is None
    result.validate_against(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[1.0]]),
        queue_execution=budget_queue,
    )


def test_counterexample_replay_checks_nonroot_relu_split_path(
    split_queue: NativeOptimizedReluSplitBabExecution,
) -> None:
    active_child = next(
        item for item in split_queue.trace.evaluations if item.node.branch_value == 1
    )
    result = derive_native_property_verdict(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[1.0]]),
        queue_execution=split_queue,
        candidate_counterexamples=((active_child.node.node_id, torch.tensor([[0.5]])),),
    )

    witness = result.trace.counterexample
    assert result.trace.status == "unsafe"
    assert witness is not None
    assert witness.satisfied_split_count == 1
    assert witness.split_constraint_min_margin == 0.5

    with pytest.raises(ValueError, match="split path"):
        derive_native_property_verdict(
            _module(),
            _spec(),
            linear_spec_C=torch.tensor([[1.0]]),
            queue_execution=split_queue,
            candidate_counterexamples=(
                (active_child.node.node_id, torch.tensor([[-0.5]])),
            ),
        )


def test_verdict_and_witness_tampering_fail_closed(
    budget_queue: NativeOptimizedReluSplitBabExecution,
) -> None:
    root_id = budget_queue.trace.evaluations[0].node.node_id
    result = derive_native_property_verdict(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[1.0]]),
        queue_execution=budget_queue,
        candidate_counterexamples=((root_id, torch.tensor([[0.0]])),),
    )
    witness = result.trace.counterexample
    assert witness is not None
    tampered_witness = replace(
        witness,
        objective_value=0.1,
        objective_margin=-0.4,
    )
    tampered = NativePropertyVerdictExecution(
        trace=replace(result.trace, counterexample=tampered_witness),
        counterexample_input=result.counterexample_input,
    )
    with pytest.raises(ValueError, match="replay differs"):
        tampered.validate_against(
            _module(),
            _spec(),
            linear_spec_C=torch.tensor([[1.0]]),
            queue_execution=budget_queue,
        )

    unknown = derive_native_property_verdict(
        _module(),
        _spec(),
        linear_spec_C=torch.tensor([[1.0]]),
        queue_execution=budget_queue,
    )
    with pytest.raises(ValueError, match="does not match|verified verdict"):
        replace(
            unknown.trace,
            status="verified",
            reason="all_leaves_soundly_pruned",
        ).validate(budget_queue.trace)


def test_mismatched_objective_and_out_of_box_candidate_fail_closed(
    budget_queue: NativeOptimizedReluSplitBabExecution,
) -> None:
    root_id = budget_queue.trace.evaluations[0].node.node_id
    with pytest.raises(ValueError, match="identity differs"):
        derive_native_property_verdict(
            _module(),
            _spec(),
            linear_spec_C=torch.tensor([[-1.0]]),
            queue_execution=budget_queue,
        )
    with pytest.raises(ValueError, match="outside"):
        derive_native_property_verdict(
            _module(),
            _spec(),
            linear_spec_C=torch.tensor([[1.0]]),
            queue_execution=budget_queue,
            candidate_counterexamples=((root_id, torch.tensor([[2.0]])),),
        )
