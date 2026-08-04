"""Optimizer Schedule integration with native ReLU-split BaB queue."""

# pylint: disable=missing-function-docstring,redefined-outer-name

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizerPolicy,
)
from boundflow.runtime.native_optimized_relu_split_bab_runtime import (
    NATIVE_REEXECUTION_ATOL,
    NATIVE_REEXECUTION_TRACE_MAX_ABS_DIFF,
    NativeOptimizedReluSplitBabTrace,
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
