import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.bab import BabConfig, solve_bab_mlp
from boundflow.runtime.task_executor import InputSpec


def _make_single_relu_conv_module(*, out_bias: float) -> BFTaskModule:
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


def test_phase7a_pr7_bab_per_example_budget_is_independent() -> None:
    module = _make_single_relu_conv_module(out_bias=-0.1)
    x0 = torch.tensor([[[[1.0]]], [[[0.0]]]], dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=0.25)
    cfg = BabConfig(
        max_nodes=1,
        oracle="alpha_beta",
        node_batch_size=4,
        alpha_steps=8,
        alpha_lr=0.2,
        alpha_init=0.5,
        beta_init=0.0,
        threshold=0.0,
        tol=1e-8,
    )

    res = solve_bab_mlp(module, spec, config=cfg)

    assert len(res.per_example) == 2
    assert res.nodes_visited == 2
    assert res.per_example[0].nodes_visited == 1
    assert res.per_example[1].nodes_visited == 1
    assert res.per_example[0].status == "proven"
    assert res.per_example[1].status == "unknown"
    assert res.status == "unknown"


def test_phase7a_pr7_bab_batch_node_batch_mixes_examples_in_one_oracle_call(monkeypatch) -> None:
    import boundflow.runtime.bab as bab_mod

    module = _make_single_relu_conv_module(out_bias=-0.1)
    x0 = torch.tensor([[[[1.0]]], [[[0.0]]]], dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=0.25)
    cfg = BabConfig(
        max_nodes=1,
        oracle="alpha_beta",
        node_batch_size=4,
        alpha_steps=0,
        alpha_lr=0.2,
        alpha_init=0.5,
        beta_init=0.0,
        threshold=0.0,
        tol=1e-8,
    )

    orig = bab_mod.run_alpha_beta_crown_mlp
    seen_batch_sizes = []

    def _wrapped(*args, **kwargs):
        input_spec = args[1]
        seen_batch_sizes.append(int(input_spec.center.shape[0]))
        return orig(*args, **kwargs)

    monkeypatch.setattr(bab_mod, "run_alpha_beta_crown_mlp", _wrapped)
    res = solve_bab_mlp(module, spec, config=cfg)

    assert len(res.per_example) == 2
    assert any(bsz > 1 for bsz in seen_batch_sizes)


def test_phase7a_pr7_bab_aggregate_result_matches_per_example() -> None:
    module = _make_single_relu_conv_module(out_bias=-0.1)
    x0 = torch.tensor([[[[1.0]]], [[[0.0]]]], dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=0.25)
    cfg = BabConfig(
        max_nodes=8,
        oracle="alpha_beta",
        node_batch_size=4,
        alpha_steps=20,
        alpha_lr=0.2,
        alpha_init=0.5,
        beta_init=0.0,
        threshold=0.0,
        tol=1e-8,
    )

    res = solve_bab_mlp(module, spec, config=cfg)

    assert len(res.per_example) == 2
    assert [r.status for r in res.per_example] == ["proven", "unsafe"]
    assert res.status == "unsafe"
    assert res.best_lower == min(r.best_lower for r in res.per_example)
    assert res.best_upper == max(r.best_upper for r in res.per_example)
