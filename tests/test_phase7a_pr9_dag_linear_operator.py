import pytest
import torch

from boundflow.runtime.linear_operator import (
    Conv2dLinearOperator,
    DenseLinearOperator,
    RightMatmulLinearOperator,
    SliceInputLinearOperator,
)


def _make_single_conv_operator() -> Conv2dLinearOperator:
    torch.manual_seed(0)
    coeffs = torch.randn(2, 4, 3, 3, 3, dtype=torch.float32)
    weight = torch.randn(3, 2, 2, 2, dtype=torch.float32)
    op = DenseLinearOperator(coeffs).conv2d_right(
        weight,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        input_shape=(2, 4, 4),
    )
    assert isinstance(op, Conv2dLinearOperator)
    return op


def test_add_linear_operator_matches_dense_reference() -> None:
    torch.manual_seed(0)
    coeffs_a = torch.randn(2, 3, 5, dtype=torch.float32)
    coeffs_b = torch.randn(2, 3, 5, dtype=torch.float32)
    center = torch.randn(2, 5, dtype=torch.float32)
    vec = torch.randn(2, 5, dtype=torch.float32)

    op = DenseLinearOperator(coeffs_a).add(DenseLinearOperator(coeffs_b))
    dense = coeffs_a + coeffs_b

    assert torch.allclose(op.to_dense(), dense)
    assert torch.allclose(op.center_term(center), torch.einsum("bki,bi->bk", dense, center))
    assert torch.allclose(op.contract_input(vec), torch.einsum("bki,bi->bk", dense, vec))
    assert torch.allclose(op.row_abs_sum(), dense.abs().sum(dim=2))
    assert torch.allclose(op.row_l2_norm(), torch.linalg.vector_norm(dense, ord=2, dim=2))
    assert torch.allclose(op.row_abs_max(), dense.abs().amax(dim=2))


def test_slice_input_linear_operator_matches_dense_reference_flat() -> None:
    torch.manual_seed(0)
    coeffs = torch.randn(2, 3, 7, dtype=torch.float32)
    center = torch.randn(2, 4, dtype=torch.float32)
    vec = torch.randn(2, 4, dtype=torch.float32)

    op = DenseLinearOperator(coeffs).slice_input((4,), start=2, stop=6)
    dense = coeffs[:, :, 2:6]

    assert torch.allclose(op.to_dense(), dense)
    assert torch.allclose(op.center_term(center), torch.einsum("bki,bi->bk", dense, center))
    assert torch.allclose(op.contract_input(vec), torch.einsum("bki,bi->bk", dense, vec))
    assert torch.allclose(op.row_abs_sum(), dense.abs().sum(dim=2))
    assert torch.allclose(op.row_l2_norm(), torch.linalg.vector_norm(dense, ord=2, dim=2))
    assert torch.allclose(op.row_abs_max(), dense.abs().amax(dim=2))


def test_slice_input_linear_operator_matches_dense_reference_channel_concat() -> None:
    op = _make_single_conv_operator()
    center = torch.randn(2, 1, 4, 4, dtype=torch.float32)
    vec = torch.randn(2, 1, 4, 4, dtype=torch.float32)

    sliced = op.slice_input((1, 4, 4), start=1, stop=2)
    dense = op.to_dense().reshape(2, op.spec_dim, *op.input_shape)[:, :, 1:2, :, :].reshape(2, op.spec_dim, -1)

    assert torch.allclose(sliced.to_dense(), dense, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sliced.center_term(center), torch.einsum("bki,bi->bk", dense, center.reshape(2, -1)), atol=1e-5, rtol=1e-5)
    assert torch.allclose(sliced.contract_input(vec), torch.einsum("bki,bi->bk", dense, vec.reshape(2, -1)), atol=1e-5, rtol=1e-5)
    assert torch.allclose(sliced.row_abs_sum(), dense.abs().sum(dim=2), atol=1e-5, rtol=1e-5)
    assert torch.allclose(sliced.row_l2_norm(), torch.linalg.vector_norm(dense, ord=2, dim=2), atol=1e-5, rtol=1e-5)
    assert torch.allclose(sliced.row_abs_max(), dense.abs().amax(dim=2), atol=1e-5, rtol=1e-5)


def test_slice_input_linear_operator_split_pos_neg_matches_dense_reference() -> None:
    torch.manual_seed(0)
    coeffs = torch.randn(2, 3, 20, dtype=torch.float32)

    op = DenseLinearOperator(coeffs, input_shape=(20,)).reshape_input((4, 5)).slice_input((2, 5), start=1, stop=3)
    assert isinstance(op, SliceInputLinearOperator)

    pos, neg = op.split_pos_neg()
    dense = op.to_dense().reshape(2, 3, *op.input_shape)
    dense_pos = dense.clamp_min(0.0)
    dense_neg = dense.clamp_max(0.0)

    assert tuple(pos.input_shape) == tuple(op.input_shape)
    assert tuple(neg.input_shape) == tuple(op.input_shape)
    assert torch.allclose(pos.to_dense().reshape(2, 3, *op.input_shape), dense_pos)
    assert torch.allclose(neg.to_dense().reshape(2, 3, *op.input_shape), dense_neg)
    assert torch.allclose(pos.to_dense() + neg.to_dense(), op.to_dense())
    assert (pos.to_dense() >= 0.0).all()
    assert (neg.to_dense() <= 0.0).all()


def test_add_linear_operator_row_norms_do_not_call_conv_to_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    op_a = _make_single_conv_operator()
    op_b = _make_single_conv_operator()
    op = op_a.add(op_b)
    dense = op.to_dense()
    expected_l1 = dense.abs().sum(dim=2)
    expected_l2 = torch.linalg.vector_norm(dense, ord=2, dim=2)
    expected_linf = dense.abs().amax(dim=2)

    def _fail(self: Conv2dLinearOperator) -> torch.Tensor:
        raise AssertionError("Conv2dLinearOperator.to_dense should not be used by AddLinearOperator row norms")

    monkeypatch.setattr(Conv2dLinearOperator, "to_dense", _fail)

    assert torch.allclose(op.row_abs_sum(), expected_l1, atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.row_l2_norm(), expected_l2, atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.row_abs_max(), expected_linf, atol=1e-5, rtol=1e-5)


def test_slice_input_linear_operator_row_norms_do_not_call_conv_to_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    op = _make_single_conv_operator()
    dense = op.to_dense().reshape(2, op.spec_dim, *op.input_shape)[:, :, 0:1, :, :].reshape(2, op.spec_dim, -1)
    sliced = op.slice_input((1, 4, 4), start=0, stop=1)
    expected_l1 = dense.abs().sum(dim=2)
    expected_l2 = torch.linalg.vector_norm(dense, ord=2, dim=2)
    expected_linf = dense.abs().amax(dim=2)

    def _fail(self: Conv2dLinearOperator) -> torch.Tensor:
        raise AssertionError("Conv2dLinearOperator.to_dense should not be used by SliceInputLinearOperator row norms")

    monkeypatch.setattr(Conv2dLinearOperator, "to_dense", _fail)

    assert torch.allclose(sliced.row_abs_sum(), expected_l1, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sliced.row_l2_norm(), expected_l2, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sliced.row_abs_max(), expected_linf, atol=1e-5, rtol=1e-5)


def test_add_linear_operator_fuses_right_matmul_chain() -> None:
    torch.manual_seed(0)
    coeffs_a = torch.randn(2, 3, 4, dtype=torch.float32)
    coeffs_b = torch.randn(2, 3, 4, dtype=torch.float32)
    rhs = torch.randn(4, 5, dtype=torch.float32)

    op = DenseLinearOperator(coeffs_a).add(DenseLinearOperator(coeffs_b)).matmul_right(rhs)

    assert isinstance(op, RightMatmulLinearOperator)
    dense = torch.einsum("bki,ij->bkj", coeffs_a + coeffs_b, rhs)
    assert torch.allclose(op.to_dense(), dense)
