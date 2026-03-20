import pytest
import torch
import torch.nn.functional as F

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.alpha_beta_crown import run_alpha_beta_crown_mlp
from boundflow.runtime.alpha_crown import run_alpha_crown_mlp
from boundflow.runtime.bab import BabConfig, solve_bab_mlp
from boundflow.runtime.crown_ibp import get_crown_ibp_mlp_stats, run_crown_ibp_mlp
from boundflow.runtime.task_executor import InputSpec


def _sample_linf_ball(*, x0: torch.Tensor, eps: float, n: int) -> torch.Tensor:
    noise = torch.rand((n,) + tuple(x0.shape), device=x0.device, dtype=x0.dtype) * 2.0 - 1.0
    return x0.unsqueeze(0) + float(eps) * noise


def _make_identity_residual_mlp(
    *,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
) -> BFTaskModule:
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(op_type="linear", name="linear1", inputs=["input", "W1", "b1"], outputs=["h1"]),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="linear", name="linear2", inputs=["r1", "W2", "b2"], outputs=["h2"]),
            TaskOp(op_type="add", name="add0", inputs=["input", "h2"], outputs=["sum0"]),
            TaskOp(op_type="relu", name="relu2", inputs=["sum0"], outputs=["r2"]),
            TaskOp(op_type="linear", name="linear3", inputs=["r2", "W3", "b3"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2, "W3": w3, "b3": b3}},
    )


def _eval_identity_residual_mlp(
    xs: torch.Tensor,
    *,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
) -> torch.Tensor:
    flat_x = xs.flatten(0, 1)
    h1 = flat_x.matmul(w1.t()) + b1
    r1 = torch.relu(h1)
    h2 = r1.matmul(w2.t()) + b2
    s0 = flat_x + h2
    r2 = torch.relu(s0)
    out = r2.matmul(w3.t()) + b3
    return out.view(xs.shape[0], xs.shape[1], -1)


def _make_projection_skip_cnn(
    *,
    w_main: torch.Tensor,
    b_main: torch.Tensor,
    w_skip: torch.Tensor,
    b_skip: torch.Tensor,
    w_out: torch.Tensor,
    b_out: torch.Tensor,
) -> BFTaskModule:
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="conv2d",
                name="conv_main",
                inputs=["input", "Wm", "bm"],
                outputs=["h_main"],
                attrs={"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp(op_type="relu", name="relu_main", inputs=["h_main"], outputs=["r_main"]),
            TaskOp(
                op_type="conv2d",
                name="conv_skip",
                inputs=["input", "Ws", "bs"],
                outputs=["h_skip"],
                attrs={"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp(op_type="add", name="add0", inputs=["r_main", "h_skip"], outputs=["sum0"]),
            TaskOp(op_type="relu", name="relu_out", inputs=["sum0"], outputs=["r_out"]),
            TaskOp(op_type="flatten", name="flatten0", inputs=["r_out"], outputs=["flat"], attrs={"start_dim": 1, "end_dim": -1}),
            TaskOp(op_type="linear", name="linear_out", inputs=["flat", "Wo", "bo"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"Wm": w_main, "bm": b_main, "Ws": w_skip, "bs": b_skip, "Wo": w_out, "bo": b_out}},
    )


def _make_feature_concat_mlp(
    *,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
) -> BFTaskModule:
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(op_type="linear", name="linear1", inputs=["input", "W1", "b1"], outputs=["h1"]),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="linear", name="linear2", inputs=["input", "W2", "b2"], outputs=["h2"]),
            TaskOp(op_type="relu", name="relu2", inputs=["h2"], outputs=["r2"]),
            TaskOp(op_type="concat", name="concat0", inputs=["r1", "r2"], outputs=["cat0"], attrs={"axis": 1}),
            TaskOp(op_type="linear", name="linear3", inputs=["cat0", "W3", "b3"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2, "W3": w3, "b3": b3}},
    )


def _make_channel_concat_cnn(
    *,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
) -> BFTaskModule:
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
            TaskOp(
                op_type="conv2d",
                name="conv2",
                inputs=["input", "W2", "b2"],
                outputs=["h2"],
                attrs={"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp(op_type="relu", name="relu2", inputs=["h2"], outputs=["r2"]),
            TaskOp(op_type="concat", name="concat0", inputs=["r1", "r2"], outputs=["cat0"], attrs={"axis": 1}),
            TaskOp(op_type="flatten", name="flatten0", inputs=["cat0"], outputs=["flat"], attrs={"start_dim": 1, "end_dim": -1}),
            TaskOp(op_type="linear", name="linear0", inputs=["flat", "W3", "b3"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2, "W3": w3, "b3": b3}},
    )


def _make_bad_concat_axis_mlp(
    *,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
) -> BFTaskModule:
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(op_type="linear", name="linear1", inputs=["input", "W1", "b1"], outputs=["h1"]),
            TaskOp(op_type="linear", name="linear2", inputs=["input", "W2", "b2"], outputs=["h2"]),
            TaskOp(op_type="concat", name="concat0", inputs=["h1", "h2"], outputs=["cat0"], attrs={"axis": 0}),
            TaskOp(op_type="linear", name="linear3", inputs=["cat0", "W3", "b3"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2, "W3": w3, "b3": b3}},
    )


def _make_broadcast_add_mlp(
    *,
    w1: torch.Tensor,
    b1: torch.Tensor,
    b_add: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> BFTaskModule:
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(op_type="linear", name="linear1", inputs=["input", "W1", "b1"], outputs=["h1"]),
            TaskOp(op_type="add", name="add0", inputs=["h1", "b_add"], outputs=["sum0"]),
            TaskOp(op_type="linear", name="linear2", inputs=["sum0", "W2", "b2"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "b_add": b_add, "W2": w2, "b2": b2}},
    )


def test_phase7a_pr8_general_dag_stats_support_residual_and_concat() -> None:
    torch.manual_seed(0)
    residual = _make_identity_residual_mlp(
        w1=torch.randn(5, 4, dtype=torch.float32),
        b1=torch.randn(5, dtype=torch.float32),
        w2=torch.randn(4, 5, dtype=torch.float32),
        b2=torch.randn(4, dtype=torch.float32),
        w3=torch.randn(3, 4, dtype=torch.float32),
        b3=torch.randn(3, dtype=torch.float32),
    )
    concat = _make_feature_concat_mlp(
        w1=torch.randn(3, 4, dtype=torch.float32),
        b1=torch.randn(3, dtype=torch.float32),
        w2=torch.randn(2, 4, dtype=torch.float32),
        b2=torch.randn(2, dtype=torch.float32),
        w3=torch.randn(2, 5, dtype=torch.float32),
        b3=torch.randn(2, dtype=torch.float32),
    )

    assert get_crown_ibp_mlp_stats(residual).supported is True
    assert get_crown_ibp_mlp_stats(concat).supported is True


def test_phase7a_pr8_identity_residual_general_dag_runs_full_solver_stack_and_is_sound() -> None:
    torch.manual_seed(0)
    x0 = torch.randn(2, 4, dtype=torch.float32)
    eps = 0.1
    w1 = torch.randn(5, 4, dtype=torch.float32)
    b1 = torch.randn(5, dtype=torch.float32)
    w2 = torch.randn(4, 5, dtype=torch.float32)
    b2 = torch.randn(4, dtype=torch.float32)
    w3 = torch.randn(3, 4, dtype=torch.float32)
    b3 = torch.randn(3, dtype=torch.float32)
    module = _make_identity_residual_mlp(w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
    spec = InputSpec.linf(value_name="input", center=x0, eps=eps)

    crown = run_crown_ibp_mlp(module, spec)
    alpha_bounds, _, _ = run_alpha_crown_mlp(module, spec, steps=8, lr=0.2)
    alpha_beta_bounds, _, _, ab_stats = run_alpha_beta_crown_mlp(module, spec, steps=8, lr=0.2)
    bab_result = solve_bab_mlp(module, spec, config=BabConfig(oracle="alpha_beta", max_nodes=32, alpha_steps=8, node_batch_size=2))

    xs = _sample_linf_ball(x0=x0, eps=eps, n=96)
    ys = _eval_identity_residual_mlp(xs, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)

    assert (ys >= crown.lower.unsqueeze(0) - 1e-5).all()
    assert (ys <= crown.upper.unsqueeze(0) + 1e-5).all()
    assert (alpha_bounds.lower >= crown.lower - 1e-5).all()
    assert (alpha_beta_bounds.lower >= alpha_bounds.lower - 1e-5).all()
    assert ab_stats.feasibility == "unknown"
    assert bab_result.status in {"proven", "unknown", "unsafe"}


def test_phase7a_pr8_projection_skip_conv_and_concat_variants_run() -> None:
    torch.manual_seed(0)
    spec = InputSpec.linf(value_name="input", center=torch.randn(1, 1, 2, 2, dtype=torch.float32), eps=0.1)
    proj = _make_projection_skip_cnn(
        w_main=torch.randn(1, 1, 1, 1, dtype=torch.float32),
        b_main=torch.randn(1, dtype=torch.float32),
        w_skip=torch.randn(1, 1, 1, 1, dtype=torch.float32),
        b_skip=torch.randn(1, dtype=torch.float32),
        w_out=torch.randn(2, 4, dtype=torch.float32),
        b_out=torch.randn(2, dtype=torch.float32),
    )
    cat_feat = _make_feature_concat_mlp(
        w1=torch.randn(3, 4, dtype=torch.float32),
        b1=torch.randn(3, dtype=torch.float32),
        w2=torch.randn(2, 4, dtype=torch.float32),
        b2=torch.randn(2, dtype=torch.float32),
        w3=torch.randn(2, 5, dtype=torch.float32),
        b3=torch.randn(2, dtype=torch.float32),
    )
    cat_ch = _make_channel_concat_cnn(
        w1=torch.randn(1, 1, 1, 1, dtype=torch.float32),
        b1=torch.randn(1, dtype=torch.float32),
        w2=torch.randn(2, 1, 1, 1, dtype=torch.float32),
        b2=torch.randn(2, dtype=torch.float32),
        w3=torch.randn(2, 12, dtype=torch.float32),
        b3=torch.randn(2, dtype=torch.float32),
    )

    _ = run_crown_ibp_mlp(proj, spec)
    _ = run_crown_ibp_mlp(cat_feat, InputSpec.linf(value_name="input", center=torch.randn(1, 4, dtype=torch.float32), eps=0.1))
    _ = run_crown_ibp_mlp(cat_ch, spec)


def test_phase7a_pr8_deeper_general_dag_split_does_not_trigger_new_detector_certificate() -> None:
    torch.manual_seed(0)
    module = _make_projection_skip_cnn(
        w_main=torch.randn(1, 1, 1, 1, dtype=torch.float32),
        b_main=torch.randn(1, dtype=torch.float32),
        w_skip=torch.randn(1, 1, 1, 1, dtype=torch.float32),
        b_skip=torch.randn(1, dtype=torch.float32),
        w_out=torch.randn(2, 4, dtype=torch.float32),
        b_out=torch.randn(2, dtype=torch.float32),
    )
    spec = InputSpec.linf(value_name="input", center=torch.zeros((1, 1, 2, 2), dtype=torch.float32), eps=0.2)
    split = {"sum0": torch.ones((1, 2, 2), dtype=torch.int8)}

    _bounds, _alpha, _beta, stats = run_alpha_beta_crown_mlp(module, spec, relu_split_state=split, steps=4, lr=0.2)

    assert stats.feasibility == "unknown"


def test_phase7a_pr8_general_dag_rejects_broadcast_add_and_bad_concat_axis() -> None:
    torch.manual_seed(0)
    add_bad = _make_broadcast_add_mlp(
        w1=torch.randn(4, 4, dtype=torch.float32),
        b1=torch.randn(4, dtype=torch.float32),
        b_add=torch.randn(4, dtype=torch.float32),
        w2=torch.randn(2, 4, dtype=torch.float32),
        b2=torch.randn(2, dtype=torch.float32),
    )
    concat_bad = _make_bad_concat_axis_mlp(
        w1=torch.randn(3, 4, dtype=torch.float32),
        b1=torch.randn(3, dtype=torch.float32),
        w2=torch.randn(2, 4, dtype=torch.float32),
        b2=torch.randn(2, dtype=torch.float32),
        w3=torch.randn(2, 5, dtype=torch.float32),
        b3=torch.randn(2, dtype=torch.float32),
    )

    with pytest.raises((NotImplementedError, ValueError), match="add|broadcast|shape"):
        run_crown_ibp_mlp(add_bad, InputSpec.linf(value_name="input", center=torch.randn(1, 4, dtype=torch.float32), eps=0.1))

    with pytest.raises((NotImplementedError, ValueError), match="concat|axis"):
        run_crown_ibp_mlp(concat_bad, InputSpec.linf(value_name="input", center=torch.randn(1, 4, dtype=torch.float32), eps=0.1))
