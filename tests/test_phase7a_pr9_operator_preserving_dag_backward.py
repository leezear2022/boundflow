import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.alpha_beta_crown import run_alpha_beta_crown_mlp
from boundflow.runtime.alpha_crown import run_alpha_crown_mlp
from boundflow.runtime.crown_ibp import _split_bias_once, run_crown_ibp_mlp
from boundflow.runtime.linear_operator import DenseLinearOperator, ReshapeInputLinearOperator, RightMatmulLinearOperator
from boundflow.runtime.task_executor import InputSpec


def _sample_linf_ball(*, x0: torch.Tensor, eps: float, n: int) -> torch.Tensor:
    noise = torch.rand((n,) + tuple(x0.shape), device=x0.device, dtype=x0.dtype) * 2.0 - 1.0
    return x0.unsqueeze(0) + float(eps) * noise


def _make_linear_add_module(
    *,
    w_left: torch.Tensor,
    b_left: torch.Tensor,
    w_right: torch.Tensor,
    b_right: torch.Tensor,
    w_out: torch.Tensor,
    b_out: torch.Tensor,
) -> BFTaskModule:
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(op_type="linear", name="left", inputs=["input", "Wl", "bl"], outputs=["left"]),
            TaskOp(op_type="linear", name="right", inputs=["input", "Wr", "br"], outputs=["right"]),
            TaskOp(op_type="add", name="add0", inputs=["left", "right"], outputs=["sum0"]),
            TaskOp(op_type="linear", name="out", inputs=["sum0", "Wo", "bo"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"Wl": w_left, "bl": b_left, "Wr": w_right, "br": b_right, "Wo": w_out, "bo": b_out}},
    )


def _eval_linear_add(xs: torch.Tensor, *, w_left: torch.Tensor, b_left: torch.Tensor, w_right: torch.Tensor, b_right: torch.Tensor, w_out: torch.Tensor, b_out: torch.Tensor) -> torch.Tensor:
    left = xs.matmul(w_left.t()) + b_left
    right = xs.matmul(w_right.t()) + b_right
    return (left + right).matmul(w_out.t()) + b_out


def _make_linear_concat_module(
    *,
    w_left: torch.Tensor,
    b_left: torch.Tensor,
    w_right: torch.Tensor,
    b_right: torch.Tensor,
    w_out: torch.Tensor,
    b_out: torch.Tensor,
) -> BFTaskModule:
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(op_type="linear", name="left", inputs=["input", "Wl", "bl"], outputs=["left"]),
            TaskOp(op_type="linear", name="right", inputs=["input", "Wr", "br"], outputs=["right"]),
            TaskOp(op_type="concat", name="concat0", inputs=["left", "right"], outputs=["cat0"], attrs={"axis": 1}),
            TaskOp(op_type="linear", name="out", inputs=["cat0", "Wo", "bo"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"Wl": w_left, "bl": b_left, "Wr": w_right, "br": b_right, "Wo": w_out, "bo": b_out}},
    )


def _eval_linear_concat(xs: torch.Tensor, *, w_left: torch.Tensor, b_left: torch.Tensor, w_right: torch.Tensor, b_right: torch.Tensor, w_out: torch.Tensor, b_out: torch.Tensor) -> torch.Tensor:
    left = xs.matmul(w_left.t()) + b_left
    right = xs.matmul(w_right.t()) + b_right
    cat = torch.cat([left, right], dim=-1)
    return cat.matmul(w_out.t()) + b_out


def _make_residual_relu_module(
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


def test_run_crown_ibp_mlp_add_backward_uses_operator_add(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    module = _make_linear_add_module(
        w_left=torch.randn(5, 4, dtype=torch.float32),
        b_left=torch.randn(5, dtype=torch.float32),
        w_right=torch.randn(5, 4, dtype=torch.float32),
        b_right=torch.randn(5, dtype=torch.float32),
        w_out=torch.randn(3, 5, dtype=torch.float32),
        b_out=torch.randn(3, dtype=torch.float32),
    )
    x0 = torch.randn(2, 4, dtype=torch.float32)
    counts = {"dense": 0, "right": 0}

    orig_dense = DenseLinearOperator.add
    orig_right = RightMatmulLinearOperator.add

    def wrapped_dense(self, other):
        counts["dense"] += 1
        return orig_dense(self, other)

    def wrapped_right(self, other):
        counts["right"] += 1
        return orig_right(self, other)

    monkeypatch.setattr(DenseLinearOperator, "add", wrapped_dense)
    monkeypatch.setattr(RightMatmulLinearOperator, "add", wrapped_right)

    bounds = run_crown_ibp_mlp(module, InputSpec.linf(value_name="input", center=x0, eps=0.1))

    assert tuple(bounds.lower.shape) == (2, 3)
    assert tuple(bounds.upper.shape) == (2, 3)
    assert counts["dense"] + counts["right"] >= 2


def test_run_crown_ibp_mlp_concat_backward_uses_operator_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    module = _make_linear_concat_module(
        w_left=torch.randn(3, 4, dtype=torch.float32),
        b_left=torch.randn(3, dtype=torch.float32),
        w_right=torch.randn(2, 4, dtype=torch.float32),
        b_right=torch.randn(2, dtype=torch.float32),
        w_out=torch.randn(2, 5, dtype=torch.float32),
        b_out=torch.randn(2, dtype=torch.float32),
    )
    x0 = torch.randn(2, 4, dtype=torch.float32)
    counts = {"dense": 0, "right": 0, "reshape": 0}

    orig_dense = DenseLinearOperator.slice_input
    orig_right = RightMatmulLinearOperator.slice_input
    orig_reshape = ReshapeInputLinearOperator.slice_input

    def wrapped_dense(self, new_input_shape, *, start, stop):
        counts["dense"] += 1
        return orig_dense(self, new_input_shape, start=start, stop=stop)

    def wrapped_right(self, new_input_shape, *, start, stop):
        counts["right"] += 1
        return orig_right(self, new_input_shape, start=start, stop=stop)

    def wrapped_reshape(self, new_input_shape, *, start, stop):
        counts["reshape"] += 1
        return orig_reshape(self, new_input_shape, start=start, stop=stop)

    monkeypatch.setattr(DenseLinearOperator, "slice_input", wrapped_dense)
    monkeypatch.setattr(RightMatmulLinearOperator, "slice_input", wrapped_right)
    monkeypatch.setattr(ReshapeInputLinearOperator, "slice_input", wrapped_reshape)

    bounds = run_crown_ibp_mlp(module, InputSpec.linf(value_name="input", center=x0, eps=0.1))

    assert tuple(bounds.lower.shape) == (2, 2)
    assert tuple(bounds.upper.shape) == (2, 2)
    assert counts["dense"] + counts["right"] + counts["reshape"] >= 2


def test_operator_preserving_dag_backward_keeps_plain_crown_sound() -> None:
    torch.manual_seed(0)
    x0 = torch.randn(2, 4, dtype=torch.float32)
    eps = 0.12
    module = _make_linear_add_module(
        w_left=torch.randn(5, 4, dtype=torch.float32),
        b_left=torch.randn(5, dtype=torch.float32),
        w_right=torch.randn(5, 4, dtype=torch.float32),
        b_right=torch.randn(5, dtype=torch.float32),
        w_out=torch.randn(3, 5, dtype=torch.float32),
        b_out=torch.randn(3, dtype=torch.float32),
    )

    bounds = run_crown_ibp_mlp(module, InputSpec.linf(value_name="input", center=x0, eps=eps))
    ys = _eval_linear_add(
        _sample_linf_ball(x0=x0, eps=eps, n=256),
        w_left=module.bindings["params"]["Wl"],
        b_left=module.bindings["params"]["bl"],
        w_right=module.bindings["params"]["Wr"],
        b_right=module.bindings["params"]["br"],
        w_out=module.bindings["params"]["Wo"],
        b_out=module.bindings["params"]["bo"],
    )

    assert (ys >= bounds.lower.unsqueeze(0) - 1e-5).all()
    assert (ys <= bounds.upper.unsqueeze(0) + 1e-5).all()


def test_operator_preserving_dag_backward_keeps_alpha_paths_running() -> None:
    torch.manual_seed(0)
    x0 = torch.randn(2, 4, dtype=torch.float32)
    module = _make_residual_relu_module(
        w1=torch.randn(5, 4, dtype=torch.float32),
        b1=torch.randn(5, dtype=torch.float32),
        w2=torch.randn(4, 5, dtype=torch.float32),
        b2=torch.randn(4, dtype=torch.float32),
        w3=torch.randn(3, 4, dtype=torch.float32),
        b3=torch.randn(3, dtype=torch.float32),
    )
    spec = InputSpec.linf(value_name="input", center=x0, eps=0.1)

    alpha_bounds, _alpha_state, _alpha_stats = run_alpha_crown_mlp(module, spec, steps=1, lr=0.1)
    beta_bounds, _alpha_state2, _beta_state, _beta_stats = run_alpha_beta_crown_mlp(module, spec, steps=1, lr=0.1)

    assert tuple(alpha_bounds.lower.shape) == (2, 3)
    assert tuple(alpha_bounds.upper.shape) == (2, 3)
    assert tuple(beta_bounds.lower.shape) == (2, 3)
    assert tuple(beta_bounds.upper.shape) == (2, 3)


def test_split_bias_once_zero_children_returns_empty_list() -> None:
    state = _split_bias_once(
        type("State", (), {
            "b_u": torch.randn(2, 3, dtype=torch.float32),
            "b_l": torch.randn(2, 3, dtype=torch.float32),
        })(),
        num_children=0,
    )

    assert state == []
