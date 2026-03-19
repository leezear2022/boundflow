import pytest
import torch

from boundflow.runtime.linear_operator import Conv2dLinearOperator, DenseLinearOperator
from boundflow.runtime.perturbation import LpBallPerturbation


def _make_single_conv_operator() -> Conv2dLinearOperator:
    torch.manual_seed(0)
    batch = 2
    specs = 4
    input_shape = (2, 5, 4)
    weight = torch.randn(3, 2, 3, 2, dtype=torch.float32)
    coeffs = torch.randn(batch, specs, 3, 3, 3, dtype=torch.float32)
    op = DenseLinearOperator(coeffs).conv2d_right(
        weight,
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 1),
        groups=1,
        input_shape=input_shape,
    )
    assert isinstance(op, Conv2dLinearOperator)
    return op


def _make_nested_conv_operator() -> Conv2dLinearOperator:
    torch.manual_seed(0)
    base = DenseLinearOperator(torch.randn(2, 3, 4, 3, 3, dtype=torch.float32))
    op1 = base.conv2d_right(
        torch.randn(4, 2, 3, 2, dtype=torch.float32),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(1, 1),
        groups=1,
        input_shape=(2, 5, 4),
    )
    op2 = op1.conv2d_right(
        torch.randn(2, 1, 2, 2, dtype=torch.float32),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        input_shape=(1, 6, 5),
    )
    assert isinstance(op2, Conv2dLinearOperator)
    return op2


def test_conv2d_lazy_row_norms_match_dense_single_layer() -> None:
    op = _make_single_conv_operator()
    dense = op.to_dense()

    assert torch.allclose(op.row_abs_sum(), dense.abs().sum(dim=2), atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.row_l2_norm(), torch.linalg.vector_norm(dense, ord=2, dim=2), atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.row_abs_max(), dense.abs().amax(dim=2), atol=1e-5, rtol=1e-5)


def test_conv2d_lazy_row_norms_match_dense_nested_conv_path() -> None:
    op = _make_nested_conv_operator()
    dense = op.to_dense()

    assert torch.allclose(op.row_abs_sum(), dense.abs().sum(dim=2), atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.row_l2_norm(), torch.linalg.vector_norm(dense, ord=2, dim=2), atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.row_abs_max(), dense.abs().amax(dim=2), atol=1e-5, rtol=1e-5)


def test_conv2d_lazy_row_norms_do_not_call_conv_to_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    op = _make_nested_conv_operator()
    dense = op.to_dense()
    expected_l1 = dense.abs().sum(dim=2)
    expected_l2 = torch.linalg.vector_norm(dense, ord=2, dim=2)
    expected_linf = dense.abs().amax(dim=2)

    def _fail(self: Conv2dLinearOperator) -> torch.Tensor:
        raise AssertionError("Conv2dLinearOperator.to_dense should not be used by row norms")

    monkeypatch.setattr(Conv2dLinearOperator, "to_dense", _fail)

    assert torch.allclose(op.row_abs_sum(), expected_l1, atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.row_l2_norm(), expected_l2, atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.row_abs_max(), expected_linf, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("p", ["inf", 2, 1])
def test_conv_operator_concretize_affine_matches_dense_reference(p: str | int) -> None:
    op = _make_single_conv_operator()
    dense = op.to_dense()
    center = torch.randn(2, *op.input_shape, dtype=torch.float32)
    bias = torch.randn(2, op.spec_dim, dtype=torch.float32)
    perturbation = LpBallPerturbation(p=p, eps=0.2)

    lb_op, ub_op = perturbation.concretize_affine(center=center, A=op, b=bias)
    lb_dense, ub_dense = perturbation.concretize_affine(center=center, A=dense.view(2, op.spec_dim, *op.input_shape), b=bias)

    assert torch.allclose(lb_op, lb_dense, atol=1e-5, rtol=1e-5)
    assert torch.allclose(ub_op, ub_dense, atol=1e-5, rtol=1e-5)
