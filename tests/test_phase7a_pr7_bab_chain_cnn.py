import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.bab import (
    BabConfig,
    NodeEvalCache,
    ReluSplitState,
    _QueueItem,
    _pick_branch,
    prune_infeasible_first_layer_items,
    solve_bab_mlp,
)
from boundflow.runtime.task_executor import InputSpec


def _make_small_conv_branch_module() -> BFTaskModule:
    w1 = torch.tensor([[[[1.0]]]], dtype=torch.float32)
    b1 = torch.tensor([0.0], dtype=torch.float32)
    w2 = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    b2 = torch.tensor([0.0], dtype=torch.float32)
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="conv2d",
                name="conv1",
                inputs=["input", "W1", "b1"],
                outputs=["h1"],
                attrs={"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="flatten", name="flatten1", inputs=["r1"], outputs=["flat"], attrs={"start_dim": 1, "end_dim": -1}),
            TaskOp(op_type="linear", name="linear1", inputs=["flat", "W2", "b2"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2}},
    )


def _make_single_relu_conv_module(*, out_bias: float = 0.0) -> BFTaskModule:
    w1 = torch.tensor([[[[1.0]]]], dtype=torch.float32)
    b1 = torch.tensor([0.0], dtype=torch.float32)
    w2 = torch.tensor([[1.0]], dtype=torch.float32)
    b2 = torch.tensor([float(out_bias)], dtype=torch.float32)
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="conv2d",
                name="conv1",
                inputs=["input", "W1", "b1"],
                outputs=["h1"],
                attrs={"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="flatten", name="flatten1", inputs=["r1"], outputs=["flat"], attrs={"start_dim": 1, "end_dim": -1}),
            TaskOp(op_type="linear", name="linear1", inputs=["flat", "W2", "b2"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2}},
    )


def _make_contradictory_first_layer_conv_module() -> BFTaskModule:
    w1 = torch.tensor([[[[1.0]]], [[[-1.0]]]], dtype=torch.float32)
    b1 = torch.tensor([-0.2, -0.2], dtype=torch.float32)
    w2 = torch.ones((1, 2), dtype=torch.float32)
    b2 = torch.zeros((1,), dtype=torch.float32)
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="conv2d",
                name="conv1",
                inputs=["input", "W1", "b1"],
                outputs=["h1"],
                attrs={"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="flatten", name="flatten1", inputs=["r1"], outputs=["flat"], attrs={"start_dim": 1, "end_dim": -1}),
            TaskOp(op_type="linear", name="linear1", inputs=["flat", "W2", "b2"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2}},
    )


def test_phase7a_pr7_conv_split_state_and_branch_pick_use_flat_idx() -> None:
    module = _make_small_conv_branch_module()
    spec = InputSpec.linf(value_name="input", center=torch.zeros((1, 1, 2, 2), dtype=torch.float32), eps=1.0)

    split = ReluSplitState.empty(module, device=spec.center.device, input_spec=spec)
    assert tuple(split.by_relu_input["h1"].shape) == (1, 2, 2)

    split2 = split.with_split(relu_input="h1", neuron_idx=1, split_value=1)
    flat = split2.by_relu_input["h1"].reshape(-1)
    assert int(flat[1].item()) == 1
    assert int(flat.sum().item()) == 1

    branch = _pick_branch(module, spec, split_state=split)
    assert branch is not None
    assert branch[0] == "h1"
    assert 0 <= int(branch[1]) < 4


def test_phase7a_pr7_bab_conv_alpha_beta_single_and_node_batch_match() -> None:
    module = _make_single_relu_conv_module()
    spec = InputSpec.linf(value_name="input", center=torch.tensor([[[[0.0]]]], dtype=torch.float32), eps=1.0)

    cfg1 = BabConfig(
        max_nodes=32,
        oracle="alpha_beta",
        node_batch_size=1,
        alpha_steps=40,
        alpha_lr=0.2,
        alpha_init=0.5,
        beta_init=0.0,
        threshold=0.0,
        tol=1e-8,
    )
    res1 = solve_bab_mlp(module, spec, config=cfg1)

    cfg4 = BabConfig(
        max_nodes=32,
        oracle="alpha_beta",
        node_batch_size=4,
        alpha_steps=40,
        alpha_lr=0.2,
        alpha_init=0.5,
        beta_init=0.0,
        threshold=0.0,
        tol=1e-8,
    )
    res4 = solve_bab_mlp(module, spec, config=cfg4)

    assert res1.status == "proven"
    assert res4.status == "proven"
    assert res4.best_lower >= -1e-6


def test_phase7a_pr7_conv_batch_partial_prune_supports_mixed_examples() -> None:
    module = _make_contradictory_first_layer_conv_module()
    spec = InputSpec.linf(value_name="input", center=torch.zeros((2, 1, 1, 1), dtype=torch.float32), eps=1.0)
    infeasible = ReluSplitState(by_relu_input={"h1": torch.tensor([[[1]], [[1]]], dtype=torch.int8)})
    feasible = ReluSplitState(by_relu_input={"h1": torch.tensor([[[0]], [[0]]], dtype=torch.int8)})

    cfg = BabConfig(oracle="alpha_beta", enable_batch_infeasible_prune=True)
    cache_by_example = {
        0: NodeEvalCache(
            module=module,
            input_spec=InputSpec.linf(value_name="input", center=spec.center[0:1], eps=1.0),
            linear_spec_C=None,
            cfg=cfg,
        ),
        1: NodeEvalCache(
            module=module,
            input_spec=InputSpec.linf(value_name="input", center=spec.center[1:2], eps=1.0),
            linear_spec_C=None,
            cfg=cfg,
        ),
    }

    items = [
        (0, _QueueItem(priority=0.0, node_id=0, example_idx=0, split_state=infeasible)),
        (1, _QueueItem(priority=0.0, node_id=1, example_idx=1, split_state=feasible)),
    ]
    kept, pruned = prune_infeasible_first_layer_items(module, spec, items=items, cache_by_example=cache_by_example, cfg=cfg)

    assert pruned == [0]
    assert [i for i, _ in kept] == [1]


def test_phase7a_pr7_bab_conv_alpha_oracle_fail_fast() -> None:
    module = _make_single_relu_conv_module()
    spec = InputSpec.linf(value_name="input", center=torch.tensor([[[[0.0]]]], dtype=torch.float32), eps=1.0)
    cfg = BabConfig(oracle="alpha", max_nodes=8)

    with pytest.raises(NotImplementedError, match="alpha-only BaB does not yet support conv graphs"):
        solve_bab_mlp(module, spec, config=cfg)
