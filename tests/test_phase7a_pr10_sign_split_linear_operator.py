import torch

from boundflow.runtime.linear_operator import (
    AddLinearOperator,
    Conv2dLinearOperator,
    DenseLinearOperator,
    RightMatmulLinearOperator,
    SignSplitLinearOperator,
)
from boundflow.runtime.materialization import dump_operator_tree, trace_materializations


def _make_flat_operator() -> (
    tuple[SignSplitLinearOperator, torch.Tensor, torch.Tensor, torch.Tensor]
):
    torch.manual_seed(7)
    coeffs = torch.randn(2, 3, 5, dtype=torch.float64)
    positive = torch.rand(2, 5, dtype=torch.float64, requires_grad=True)
    negative = torch.rand(2, 5, dtype=torch.float64, requires_grad=True)
    operator = SignSplitLinearOperator(
        base=DenseLinearOperator(coeffs),
        positive_scale=positive,
        negative_scale=negative,
        source_value="h1",
        bound_direction="upper",
    )
    return operator, coeffs, positive, negative


def test_sign_split_to_dense_and_scale_gradients_are_exact() -> None:
    operator, coeffs, positive, negative = _make_flat_operator()
    expected = coeffs.clamp_min(0) * positive.unsqueeze(1) + coeffs.clamp_max(
        0
    ) * negative.unsqueeze(1)

    actual = operator.to_dense()
    assert torch.equal(actual, expected)
    actual.sum().backward()

    assert positive.grad is not None and negative.grad is not None
    assert torch.equal(positive.grad, coeffs.clamp_min(0).sum(dim=1))
    assert torch.equal(negative.grad, coeffs.clamp_max(0).sum(dim=1))


def test_sign_split_composition_wraps_without_pushing_sign_through_matmul_or_add() -> (
    None
):
    operator, _coeffs, _positive, _negative = _make_flat_operator()
    rhs = torch.randn(5, 4, dtype=torch.float64)
    matmul = operator.matmul_right(rhs)
    added = operator.add(operator)

    assert isinstance(matmul, RightMatmulLinearOperator)
    assert matmul.base is operator
    assert torch.allclose(matmul.to_dense(), operator.to_dense().matmul(rhs))
    assert isinstance(added, AddLinearOperator)
    assert added.lhs is operator and added.rhs is operator
    assert torch.equal(added.to_dense(), operator.to_dense() * 2)


def test_sign_split_conv_composition_keeps_transform_outside_conv_algebra() -> None:
    torch.manual_seed(9)
    coeffs = torch.randn(1, 2, 3, 4, 4, dtype=torch.float64)
    positive = torch.rand(1, 3, 4, 4, dtype=torch.float64)
    negative = torch.rand(1, 3, 4, 4, dtype=torch.float64)
    operator = SignSplitLinearOperator(
        base=DenseLinearOperator(coeffs, input_shape=(3, 4, 4)),
        positive_scale=positive,
        negative_scale=negative,
        source_value="conv_pre",
        bound_direction="lower",
    )
    weight = torch.randn(3, 2, 3, 3, dtype=torch.float64)
    composed = operator.conv2d_right(weight, padding=1, input_shape=(2, 4, 4))
    expected = DenseLinearOperator(
        operator.to_dense(), input_shape=(3, 4, 4)
    ).conv2d_right(
        weight,
        padding=1,
        input_shape=(2, 4, 4),
    )

    assert isinstance(composed, Conv2dLinearOperator)
    assert composed.base is operator
    assert torch.allclose(
        composed.to_dense(), expected.to_dense(), atol=1e-12, rtol=1e-12
    )


def test_sign_split_reductions_match_dense_and_trace_ephemeral_materialization() -> (
    None
):
    operator, _coeffs, _positive, _negative = _make_flat_operator()
    center = torch.randn(2, 5, dtype=torch.float64)
    dense = operator.to_dense()

    with trace_materializations(run_id="test", query_id="sign-split") as trace:
        center_term = operator.center_term(center)
        row_l1 = operator.row_abs_sum()
        row_l2 = operator.row_l2_norm()
        row_linf = operator.row_abs_max()

    assert torch.allclose(center_term, (dense * center.unsqueeze(1)).sum(dim=2))
    assert torch.equal(row_l1, dense.abs().sum(dim=2))
    assert torch.equal(row_l2, torch.linalg.vector_norm(dense, ord=2, dim=2))
    assert torch.equal(row_linf, dense.abs().amax(dim=2))
    assert len(trace.events) == 4
    assert {event.persistent_or_ephemeral for event in trace.events} == {"ephemeral"}
    assert {event.reason for event in trace.events} == {
        "sign_split_center_term",
        "sign_split_row_l1",
        "sign_split_row_l2",
        "sign_split_row_linf",
    }


def test_sign_split_operator_tree_dump_is_deterministic_and_value_free() -> None:
    operator, _coeffs, _positive, _negative = _make_flat_operator()
    composed = operator.matmul_right(torch.randn(5, 4, dtype=torch.float64)).add(
        operator.matmul_right(torch.randn(5, 4, dtype=torch.float64))
    )

    first = dump_operator_tree(composed)
    second = dump_operator_tree(composed)

    assert first == second
    assert first["root"] == 0
    assert [node["id"] for node in first["nodes"]] == list(range(len(first["nodes"])))
    assert {node["operator_type"] for node in first["nodes"]} >= {
        "AddLinearOperator",
        "RightMatmulLinearOperator",
        "SignSplitLinearOperator",
        "DenseLinearOperator",
    }
    rendered = str(first)
    assert "positive_scale_shape" in rendered
    assert "tensor(" not in rendered
