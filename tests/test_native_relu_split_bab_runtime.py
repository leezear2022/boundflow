"""Native first-class ReLU-split queue and node-batching contracts."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.native_relu_split_bab_runtime import (
    NativeReluSplitBabConfig,
    run_native_relu_split_bab,
)
from boundflow.runtime.task_executor import InputSpec


def _toy_module() -> BFTaskModule:
    return BFTaskModule(
        tasks=[
            BoundTask(
                task_id="native-relu-split-toy",
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
        entry_task_id="native-relu-split-toy",
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


def _toy_spec() -> InputSpec:
    return InputSpec.box(
        value_name="input",
        lower=torch.tensor([[-0.3, -0.6, -0.1]]),
        upper=torch.tensor([[0.7, 0.4, 0.9]]),
    )


def _run(*, max_eval_batch_size: int, max_nodes: int = 15):
    return run_native_relu_split_bab(
        _toy_module(),
        _toy_spec(),
        linear_spec_C=torch.tensor([[1.0, -1.0]]),
        run_id="native-relu-split-toy",
        config=NativeReluSplitBabConfig(
            max_nodes=max_nodes,
            max_depth=3,
            expansion_batch_size=2,
            max_eval_batch_size=max_eval_batch_size,
            threshold=1e6,
        ),
    )


def test_packed_and_serial_node_stacks_close_same_complete_queue() -> None:
    packed = _run(max_eval_batch_size=4)
    serial = _run(max_eval_batch_size=1)

    assert packed.status == serial.status == "complete"
    assert packed.final_frontier_node_ids == serial.final_frontier_node_ids == ()
    assert len(packed.evaluations) == len(serial.evaluations) == 15
    assert len(packed.decisions) == len(serial.decisions) == 15
    assert packed.native_stack_count == 5
    assert serial.native_stack_count == 15
    assert packed.logical_queue_signature() == serial.logical_queue_signature()
    assert all(
        actual.node.split_state_hash == expected.node.split_state_hash
        and actual.exact_state_hash == expected.exact_state_hash
        and actual.lower == expected.lower
        and actual.upper == expected.upper
        for actual, expected in zip(packed.evaluations, serial.evaluations)
    )
    assert all(
        evaluation.parent_state_consumed_as_exact is False
        and (
            evaluation.parent_exact_state_hash is None
            or evaluation.parent_exact_state_hash != evaluation.exact_state_hash
        )
        for evaluation in packed.evaluations
    )
    assert packed.performance_claimed is False
    assert packed.property_status == "not_claimed"


def test_node_budget_stops_with_evaluated_replayable_frontier() -> None:
    trace = _run(max_eval_batch_size=4, max_nodes=7)

    assert trace.status == "budget_exhausted"
    assert trace.termination_reason == "node_budget_exhausted"
    assert len(trace.evaluations) == 7
    assert len(trace.decisions) == 3
    assert len(trace.final_frontier_node_ids) == 4
    assert set(trace.final_frontier_node_ids).isdisjoint(
        decision.node_id for decision in trace.decisions
    )
    trace.validate()


def test_queue_trace_rejects_parent_and_branch_tampering() -> None:
    trace = _run(max_eval_batch_size=4)
    child_index = next(
        index
        for index, evaluation in enumerate(trace.evaluations)
        if evaluation.node.depth == 1
    )
    child = trace.evaluations[child_index]
    wrong_parent = replace(
        child,
        node=replace(child.node, parent_node_id="missing-parent"),
    )
    tampered_evaluations = list(trace.evaluations)
    tampered_evaluations[child_index] = wrong_parent
    with pytest.raises(ValueError, match="parent"):
        replace(trace, evaluations=tuple(tampered_evaluations)).validate()

    expansion_index = next(
        index
        for index, decision in enumerate(trace.decisions)
        if decision.kind == "expand"
    )
    expansion = trace.decisions[expansion_index]
    tampered_decisions = list(trace.decisions)
    tampered_decisions[expansion_index] = replace(
        expansion,
        child_node_ids=tuple(reversed(expansion.child_node_ids)),
    )
    with pytest.raises(ValueError, match="branch"):
        replace(trace, decisions=tuple(tampered_decisions)).validate()


def test_queue_config_and_input_fail_closed() -> None:
    with pytest.raises(ValueError, match="config"):
        NativeReluSplitBabConfig(
            max_nodes=0,
            max_depth=1,
            expansion_batch_size=1,
            max_eval_batch_size=1,
        ).validate()
    linf = InputSpec.linf(value_name="input", center=torch.zeros(1, 3), eps=0.1)
    with pytest.raises(NotImplementedError, match="box input"):
        run_native_relu_split_bab(
            _toy_module(),
            linf,
            linear_spec_C=torch.tensor([[1.0, -1.0]]),
            run_id="invalid-input",
            config=NativeReluSplitBabConfig(
                max_nodes=1,
                max_depth=0,
                expansion_batch_size=1,
                max_eval_batch_size=1,
            ),
        )
