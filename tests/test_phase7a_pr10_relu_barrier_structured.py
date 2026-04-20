import pytest
import torch

from boundflow.domains.interval import IntervalState
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import AffineBackwardState, _backprop_relu_step, _relu_relax, run_crown_ibp_mlp
from boundflow.runtime.linear_operator import (
    AddLinearOperator,
    DenseLinearOperator,
    RepeatedRowLinearOperator,
    RightMatmulLinearOperator,
    ScaledInputLinearOperator,
)
from boundflow.runtime.task_executor import InputSpec


def _make_toy_relu_chain_module() -> BFTaskModule:
    w1 = torch.tensor([[1.0]], dtype=torch.float32)
    b1 = torch.tensor([0.0], dtype=torch.float32)
    w2 = torch.tensor([[1.0]], dtype=torch.float32)
    b2 = torch.tensor([0.0], dtype=torch.float32)
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(op_type="linear", name="linear1", inputs=["input", "W1", "b1"], outputs=["h1"]),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="linear", name="linear2", inputs=["r1", "W2", "b2"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2}},
    )


def _dense_relu_step_reference(
    state: AffineBackwardState,
    *,
    pre: IntervalState,
    relu_alpha: dict[str, torch.Tensor] | None = None,
    relu_pre_add_coeff_u: dict[str, torch.Tensor] | None = None,
    relu_pre_add_coeff_l: dict[str, torch.Tensor] | None = None,
    x_name: str = "h1",
) -> AffineBackwardState:
    batch = int(pre.lower.shape[0])
    input_shape = tuple(int(dim) for dim in pre.lower.shape[1:])
    pre_flat = IntervalState(lower=pre.lower.reshape(batch, -1), upper=pre.upper.reshape(batch, -1))
    A_u = state.A_u.to_dense()
    A_l = state.A_l.to_dense()
    b_u = state.b_u.clone()
    b_l = state.b_l.clone()

    alpha_u, beta_u, alpha_l, beta_l = _relu_relax(pre_flat.lower, pre_flat.upper)
    if relu_alpha is not None and x_name in relu_alpha:
        alpha_raw = relu_alpha[x_name]
        alpha = alpha_raw if torch.is_tensor(alpha_raw) else torch.as_tensor(alpha_raw)
        alpha = alpha.to(device=pre.lower.device, dtype=pre.lower.dtype).reshape(batch, -1)
        amb = (pre_flat.lower < 0) & (pre_flat.upper > 0)
        if amb.any():
            alpha_l = torch.where(amb, alpha, alpha_l)

    sel_alpha_u = torch.where(A_u >= 0, alpha_u.unsqueeze(1), alpha_l.unsqueeze(1))
    sel_beta_u = torch.where(A_u >= 0, beta_u.unsqueeze(1), beta_l.unsqueeze(1))
    b_u = b_u + (A_u * sel_beta_u).sum(dim=2)
    A_u = A_u * sel_alpha_u
    if relu_pre_add_coeff_u is not None and x_name in relu_pre_add_coeff_u:
        add_u = relu_pre_add_coeff_u[x_name]
        add_u = add_u if torch.is_tensor(add_u) else torch.as_tensor(add_u, device=A_u.device, dtype=A_u.dtype)
        add_u = add_u.to(device=A_u.device, dtype=A_u.dtype).reshape(batch, -1)
        A_u = A_u + add_u.unsqueeze(1)

    sel_alpha_l = torch.where(A_l >= 0, alpha_l.unsqueeze(1), alpha_u.unsqueeze(1))
    sel_beta_l = torch.where(A_l >= 0, beta_l.unsqueeze(1), beta_u.unsqueeze(1))
    b_l = b_l + (A_l * sel_beta_l).sum(dim=2)
    A_l = A_l * sel_alpha_l
    if relu_pre_add_coeff_l is not None and x_name in relu_pre_add_coeff_l:
        add_l = relu_pre_add_coeff_l[x_name]
        add_l = add_l if torch.is_tensor(add_l) else torch.as_tensor(add_l, device=A_l.device, dtype=A_l.dtype)
        add_l = add_l.to(device=A_l.device, dtype=A_l.dtype).reshape(batch, -1)
        A_l = A_l + add_l.unsqueeze(1)

    return AffineBackwardState(
        A_u=DenseLinearOperator(A_u, input_shape=input_shape),
        A_l=DenseLinearOperator(A_l, input_shape=input_shape),
        b_u=b_u,
        b_l=b_l,
    )


def _relu_relax_pullback_reference(
    op,
    *,
    pos_slope: torch.Tensor,
    neg_slope: torch.Tensor,
    pos_bias: torch.Tensor,
    neg_bias: torch.Tensor,
):
    pos, neg = op.split_pos_neg()
    delta_b = pos.contract_input(pos_bias) + neg.contract_input(neg_bias)
    A_out = ScaledInputLinearOperator(pos, pos_slope).add(ScaledInputLinearOperator(neg, neg_slope))
    return A_out, delta_b


def test_split_pos_neg_dense_matches_reference() -> None:
    coeffs = torch.tensor(
        [[[1.0, -2.0, 0.5], [-0.3, 0.0, 4.0]]],
        dtype=torch.float32,
    )
    op = DenseLinearOperator(coeffs)

    pos, neg = op.split_pos_neg()

    assert torch.allclose(pos.to_dense() + neg.to_dense(), coeffs, atol=1e-6, rtol=1e-6)
    assert bool((pos.to_dense() >= 0).all().item())
    assert bool((neg.to_dense() <= 0).all().item())



def test_scaled_input_linear_operator_matches_dense_reference() -> None:
    torch.manual_seed(0)
    base = DenseLinearOperator(torch.randn(2, 3, 4, dtype=torch.float32))
    scale = torch.tensor([[0.2, 0.5, 1.0, 0.3], [1.0, 0.4, 0.7, 0.9]], dtype=torch.float32)
    op = ScaledInputLinearOperator(base, scale)
    dense = base.to_dense() * scale.unsqueeze(1)
    center = torch.randn(2, 4, dtype=torch.float32)

    assert torch.allclose(op.to_dense(), dense, atol=1e-6, rtol=1e-6)
    assert torch.allclose(op.center_term(center), torch.einsum("bki,bi->bk", dense, center), atol=1e-6, rtol=1e-6)
    assert torch.allclose(op.contract_input(center), torch.einsum("bki,bi->bk", dense, center), atol=1e-6, rtol=1e-6)
    assert torch.allclose(op.row_abs_sum(), dense.abs().sum(dim=2), atol=1e-6, rtol=1e-6)
    assert torch.allclose(op.row_l2_norm(), torch.linalg.vector_norm(dense, ord=2, dim=2), atol=1e-6, rtol=1e-6)
    assert torch.allclose(op.row_abs_max(), dense.abs().amax(dim=2), atol=1e-6, rtol=1e-6)



def test_repeated_row_linear_operator_matches_dense_reference() -> None:
    coeffs = torch.tensor([[1.0, -2.0, 0.5, 3.0], [-1.5, 0.2, 0.0, 1.0]], dtype=torch.float32)
    op = RepeatedRowLinearOperator(coeffs=coeffs, spec_dim_size=3, input_shape=(4,), batch_size=2)
    dense = coeffs.unsqueeze(1).expand(2, 3, 4)
    center = torch.randn(2, 4, dtype=torch.float32)

    assert torch.allclose(op.to_dense(), dense, atol=1e-6, rtol=1e-6)
    assert torch.allclose(op.center_term(center), torch.einsum("bki,bi->bk", dense, center), atol=1e-6, rtol=1e-6)
    assert torch.allclose(op.contract_input(center), torch.einsum("bki,bi->bk", dense, center), atol=1e-6, rtol=1e-6)
    assert torch.allclose(op.row_abs_sum(), dense.abs().sum(dim=2), atol=1e-6, rtol=1e-6)
    assert torch.allclose(op.row_l2_norm(), torch.linalg.vector_norm(dense, ord=2, dim=2), atol=1e-6, rtol=1e-6)
    assert torch.allclose(op.row_abs_max(), dense.abs().amax(dim=2), atol=1e-6, rtol=1e-6)



def test_backprop_relu_step_preserves_operator_form_and_matches_dense_reference() -> None:
    torch.manual_seed(0)
    base = DenseLinearOperator(torch.randn(1, 2, 3, dtype=torch.float32))
    rhs = torch.randn(3, 4, dtype=torch.float32)
    state = AffineBackwardState(
        A_u=base.matmul_right(rhs),
        A_l=base.matmul_right(rhs),
        b_u=torch.randn(1, 2, dtype=torch.float32),
        b_l=torch.randn(1, 2, dtype=torch.float32),
    )
    pre = IntervalState(
        lower=torch.tensor([[-1.0, -0.5, -0.2, -0.1]], dtype=torch.float32),
        upper=torch.tensor([[0.8, 0.6, 1.5, 2.0]], dtype=torch.float32),
    )

    out = _backprop_relu_step(
        state,
        pre=pre,
        x_name="h1",
        relu_alpha=None,
        relu_pre_add_coeff_u=None,
        relu_pre_add_coeff_l=None,
        device=pre.lower.device,
        dtype=pre.lower.dtype,
        caller="test",
    )
    ref = _dense_relu_step_reference(state, pre=pre)

    assert isinstance(out.A_u, AddLinearOperator)
    assert isinstance(out.A_l, AddLinearOperator)
    assert not isinstance(out.A_u, DenseLinearOperator)
    assert not isinstance(out.A_l, DenseLinearOperator)
    assert torch.allclose(out.A_u.to_dense(), ref.A_u.to_dense(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(out.A_l.to_dense(), ref.A_l.to_dense(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(out.b_u, ref.b_u, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out.b_l, ref.b_l, atol=1e-6, rtol=1e-6)


def test_relu_relax_pullback_matches_split_reference_across_operator_forms() -> None:
    torch.manual_seed(0)
    dense = DenseLinearOperator(torch.randn(2, 3, 6, dtype=torch.float32), input_shape=(6,))
    ops = {
        "dense": dense,
        "reshape": dense.reshape_input((2, 3)),
        "slice": dense.reshape_input((2, 3)).slice_input((1, 3), start=1, stop=2),
        "add": dense.add(DenseLinearOperator(torch.randn(2, 3, 6, dtype=torch.float32), input_shape=(6,))),
        "right_matmul": dense.matmul_right(torch.randn(6, 5, dtype=torch.float32)),
    }

    for name, op in ops.items():
        pos_slope = torch.rand((2, *op.input_shape), dtype=torch.float32)
        neg_slope = torch.rand((2, *op.input_shape), dtype=torch.float32)
        pos_bias = torch.rand((2, *op.input_shape), dtype=torch.float32)
        neg_bias = torch.rand((2, *op.input_shape), dtype=torch.float32)

        out_op, out_delta_b = op.relu_relax_pullback(
            pos_slope=pos_slope,
            neg_slope=neg_slope,
            pos_bias=pos_bias,
            neg_bias=neg_bias,
        )
        ref_op, ref_delta_b = _relu_relax_pullback_reference(
            op,
            pos_slope=pos_slope,
            neg_slope=neg_slope,
            pos_bias=pos_bias,
            neg_bias=neg_bias,
        )

        assert tuple(out_op.input_shape) == tuple(op.input_shape), name
        assert tuple(out_delta_b.shape) == (2, 3), name
        assert torch.allclose(out_op.to_dense(), ref_op.to_dense(), atol=1e-6, rtol=1e-6), name
        assert torch.allclose(out_delta_b, ref_delta_b, atol=1e-6, rtol=1e-6), name


def test_backprop_relu_step_uses_relu_relax_pullback_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    base = DenseLinearOperator(torch.randn(1, 2, 3, dtype=torch.float32))
    rhs = torch.randn(3, 4, dtype=torch.float32)
    state = AffineBackwardState(
        A_u=base.matmul_right(rhs),
        A_l=base.matmul_right(rhs),
        b_u=torch.randn(1, 2, dtype=torch.float32),
        b_l=torch.randn(1, 2, dtype=torch.float32),
    )
    pre = IntervalState(
        lower=torch.tensor([[-1.0, -0.5, -0.2, -0.1]], dtype=torch.float32),
        upper=torch.tensor([[0.8, 0.6, 1.5, 2.0]], dtype=torch.float32),
    )
    calls = {"count": 0}
    orig_pullback = RightMatmulLinearOperator.relu_relax_pullback

    def wrapped_pullback(self, **kwargs):
        calls["count"] += 1
        return orig_pullback(self, **kwargs)

    monkeypatch.setattr(RightMatmulLinearOperator, "relu_relax_pullback", wrapped_pullback)

    _ = _backprop_relu_step(
        state,
        pre=pre,
        x_name="h1",
        relu_alpha=None,
        relu_pre_add_coeff_u=None,
        relu_pre_add_coeff_l=None,
        device=pre.lower.device,
        dtype=pre.lower.dtype,
        caller="test",
    )

    assert calls["count"] == 2


def test_right_matmul_relu_relax_pullback_does_not_use_split_pos_neg_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    base = DenseLinearOperator(torch.randn(2, 3, 4, dtype=torch.float32))
    op = base.matmul_right(torch.randn(4, 5, dtype=torch.float32))
    assert isinstance(op, RightMatmulLinearOperator)
    pos_slope = torch.rand(2, 5, dtype=torch.float32)
    neg_slope = torch.rand(2, 5, dtype=torch.float32)
    pos_bias = torch.rand(2, 5, dtype=torch.float32)
    neg_bias = torch.rand(2, 5, dtype=torch.float32)
    ref_op, ref_delta_b = _relu_relax_pullback_reference(
        op,
        pos_slope=pos_slope,
        neg_slope=neg_slope,
        pos_bias=pos_bias,
        neg_bias=neg_bias,
    )

    import boundflow.runtime.linear_operator as linear_operator_mod

    def fail(_op):
        raise AssertionError("_split_pos_neg_dense should not be used by RightMatmulLinearOperator.relu_relax_pullback")

    monkeypatch.setattr(linear_operator_mod, "_split_pos_neg_dense", fail)

    out_op, delta_b = op.relu_relax_pullback(
        pos_slope=pos_slope,
        neg_slope=neg_slope,
        pos_bias=pos_bias,
        neg_bias=neg_bias,
    )

    assert torch.allclose(out_op.to_dense(), ref_op.to_dense(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(delta_b, ref_delta_b, atol=1e-6, rtol=1e-6)



def test_relu_alpha_grad_flows_through_structured_operator() -> None:
    module = _make_toy_relu_chain_module()
    x0 = torch.tensor([[0.0]], dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=1.0)
    alpha = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

    bounds = run_crown_ibp_mlp(module, spec, relu_alpha={"h1": alpha})
    loss = -bounds.lower.mean()
    loss.backward()

    assert alpha.grad is not None
    assert torch.isfinite(alpha.grad).all()
    assert float(alpha.grad.abs().sum().item()) > 0.0



def test_relu_pre_add_coeff_l_uses_structured_operator_and_keeps_beta_grad() -> None:
    torch.manual_seed(0)
    base = DenseLinearOperator(torch.randn(1, 2, 3, dtype=torch.float32))
    rhs = torch.randn(3, 4, dtype=torch.float32)
    state = AffineBackwardState(
        A_u=base.matmul_right(rhs),
        A_l=base.matmul_right(rhs),
        b_u=torch.randn(1, 2, dtype=torch.float32),
        b_l=torch.randn(1, 2, dtype=torch.float32),
    )
    pre = IntervalState(
        lower=torch.tensor([[-1.0, -0.5, -0.2, -0.1]], dtype=torch.float32),
        upper=torch.tensor([[0.8, 0.6, 1.5, 2.0]], dtype=torch.float32),
    )
    beta = torch.nn.Parameter(torch.full((1, 4), 0.1, dtype=torch.float32))

    out = _backprop_relu_step(
        state,
        pre=pre,
        x_name="h1",
        relu_alpha=None,
        relu_pre_add_coeff_u=None,
        relu_pre_add_coeff_l={"h1": -beta},
        device=pre.lower.device,
        dtype=pre.lower.dtype,
        caller="test",
    )
    ref = _dense_relu_step_reference(state, pre=pre, relu_pre_add_coeff_l={"h1": -beta})
    loss = out.A_l.to_dense().sum() + out.b_l.sum()
    loss.backward()

    assert isinstance(out.A_l, AddLinearOperator)
    assert torch.allclose(out.A_l.to_dense(), ref.A_l.to_dense(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(out.b_l, ref.b_l, atol=1e-6, rtol=1e-6)
    assert beta.grad is not None
    assert torch.isfinite(beta.grad).all()
    assert float(beta.grad.abs().sum().item()) > 0.0
