import torch

from boundflow.domains.interval import IntervalState
from boundflow.runtime.crown_ibp import (
    AffineBackwardState,
    _backprop_relu_step,
    _backprop_relu_step_dense_reference,
)
from boundflow.runtime.linear_operator import DenseLinearOperator


def _flat_case() -> (
    tuple[IntervalState, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    pre = IntervalState(
        lower=torch.tensor(
            [[0.2, -2.0, -1.0, -0.5], [0.1, -1.5, -0.8, -0.2]], dtype=torch.float64
        ),
        upper=torch.tensor(
            [[1.0, -0.1, 2.0, 0.5], [0.9, -0.2, 1.2, 0.7]], dtype=torch.float64
        ),
    )
    torch.manual_seed(3)
    A_u = torch.randn(2, 3, 4, dtype=torch.float64)
    A_l = torch.randn(2, 3, 4, dtype=torch.float64)
    b_u = torch.randn(2, 3, dtype=torch.float64)
    b_l = torch.randn(2, 3, dtype=torch.float64)
    return pre, A_u, A_l, b_u, b_l


def test_dense_relu_reference_matches_current_wrapper_with_alpha_and_beta_coefficients() -> (
    None
):
    pre, A_u, A_l, b_u, b_l = _flat_case()
    alpha = torch.tensor([0.2, 0.3, 0.6, 0.8], dtype=torch.float64)
    add_u = torch.tensor(
        [[0.0, 0.0, 0.1, -0.2], [0.0, 0.0, 0.3, -0.4]], dtype=torch.float64
    )
    add_l = -add_u

    reference = _backprop_relu_step_dense_reference(
        A_u,
        A_l,
        b_u,
        b_l,
        pre=pre,
        x_name="h1",
        relu_alpha={"h1": alpha},
        relu_pre_add_coeff_u={"h1": add_u},
        relu_pre_add_coeff_l={"h1": add_l},
        device=A_u.device,
        dtype=A_u.dtype,
        caller="test",
    )
    wrapped = _backprop_relu_step(
        AffineBackwardState(
            A_u=DenseLinearOperator(A_u),
            A_l=DenseLinearOperator(A_l),
            b_u=b_u,
            b_l=b_l,
        ),
        pre=pre,
        x_name="h1",
        relu_alpha={"h1": alpha},
        relu_pre_add_coeff_u={"h1": add_u},
        relu_pre_add_coeff_l={"h1": add_l},
        device=A_u.device,
        dtype=A_u.dtype,
        caller="test",
    )

    assert torch.equal(wrapped.A_u.to_dense(), reference.A_u)
    assert torch.equal(wrapped.A_l.to_dense(), reference.A_l)
    assert torch.equal(wrapped.b_u, reference.b_u)
    assert torch.equal(wrapped.b_l, reference.b_l)


def test_dense_relu_reference_alpha_gradient_matches_sign_selection_formula() -> None:
    pre, A_u, A_l, b_u, b_l = _flat_case()
    alpha = torch.tensor([0.2, 0.3, 0.6, 0.8], dtype=torch.float64, requires_grad=True)
    result = _backprop_relu_step_dense_reference(
        A_u,
        A_l,
        b_u,
        b_l,
        pre=pre,
        x_name="h1",
        relu_alpha={"h1": alpha},
        relu_pre_add_coeff_u=None,
        relu_pre_add_coeff_l=None,
        device=A_u.device,
        dtype=A_u.dtype,
        caller="test",
    )
    (
        result.A_u.sum() + result.A_l.sum() + result.b_u.sum() + result.b_l.sum()
    ).backward()

    ambiguous = (pre.lower < 0) & (pre.upper > 0)
    expected = (
        torch.where((A_u < 0) & ambiguous.unsqueeze(1), A_u, torch.zeros_like(A_u))
        + torch.where((A_l >= 0) & ambiguous.unsqueeze(1), A_l, torch.zeros_like(A_l))
    ).sum(dim=(0, 1))
    assert alpha.grad is not None
    assert torch.isfinite(alpha.grad).all()
    assert torch.allclose(alpha.grad, expected, atol=1e-12, rtol=1e-12)
    assert torch.equal(alpha.grad[:2], torch.zeros(2, dtype=torch.float64))
    assert float(alpha.grad[2:].abs().sum()) > 0.0


def test_dense_relu_reference_supports_batched_nchw_alpha_gradient() -> None:
    torch.manual_seed(5)
    pre = IntervalState(
        lower=-torch.rand(2, 2, 2, 2, dtype=torch.float64),
        upper=torch.rand(2, 2, 2, 2, dtype=torch.float64),
    )
    A_u = torch.randn(2, 4, 8, dtype=torch.float64)
    A_l = torch.randn(2, 4, 8, dtype=torch.float64)
    alpha = torch.full((2, 2, 2), 0.5, dtype=torch.float64, requires_grad=True)
    result = _backprop_relu_step_dense_reference(
        A_u,
        A_l,
        torch.zeros(2, 4, dtype=torch.float64),
        torch.zeros(2, 4, dtype=torch.float64),
        pre=pre,
        x_name="conv_pre",
        relu_alpha={"conv_pre": alpha},
        relu_pre_add_coeff_u=None,
        relu_pre_add_coeff_l=None,
        device=A_u.device,
        dtype=A_u.dtype,
        caller="test",
    )
    loss = result.A_l.sum() + result.b_l.sum()
    loss.backward()

    assert tuple(result.A_u.shape) == (2, 4, 8)
    assert tuple(result.A_l.shape) == (2, 4, 8)
    assert alpha.grad is not None
    assert tuple(alpha.grad.shape) == (2, 2, 2)
    assert torch.isfinite(alpha.grad).all()
    assert float(alpha.grad.abs().sum()) > 0.0
