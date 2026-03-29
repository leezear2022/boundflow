import math

import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import _backprop_permute_step, get_crown_ibp_mlp_stats, run_crown_ibp_mlp
from boundflow.runtime.linear_operator import DenseLinearOperator, ReindexInputLinearOperator
from boundflow.runtime.task_executor import InputSpec


def _make_gather_index(input_shape: tuple[int, ...], dims_with_batch: tuple[int, ...]) -> torch.Tensor:
    flat = torch.arange(math.prod(input_shape), dtype=torch.long).reshape(input_shape)
    dims_no_batch = tuple(int(dim) - 1 for dim in dims_with_batch[1:])
    return flat.permute(*dims_no_batch).reshape(-1)


def _sample_linf_ball(*, x0: torch.Tensor, eps: float, n: int) -> torch.Tensor:
    noise = torch.rand((n,) + tuple(x0.shape), device=x0.device, dtype=x0.dtype) * 2.0 - 1.0
    return x0.unsqueeze(0) + float(eps) * noise


def _make_permute_reshape_linear_module(*, weight: torch.Tensor, bias: torch.Tensor) -> BFTaskModule:
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(op_type="permute", name="perm0", inputs=["input"], outputs=["p0"], attrs={"dims": [0, 2, 3, 1], "layout_only": True}),
            TaskOp(op_type="reshape", name="reshape0", inputs=["p0"], outputs=["flat"], attrs={"shape": [2, -1]}),
            TaskOp(op_type="linear", name="linear0", inputs=["flat", "W", "b"], outputs=["out"], attrs={"op": "linear"}),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W": weight, "b": bias}},
    )


def _eval_permute_reshape_linear(xs: torch.Tensor, *, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    flat = xs.permute(0, 2, 3, 1).reshape(xs.shape[0], -1)
    return flat.matmul(weight.t()) + bias


def test_reindex_input_linear_operator_matches_dense_reference() -> None:
    torch.manual_seed(0)
    input_shape = (2, 3, 4)
    dims_with_batch = (0, 2, 3, 1)
    base_shape = (3, 4, 2)
    gather_index = _make_gather_index(input_shape, dims_with_batch)
    scatter_index = torch.empty_like(gather_index)
    scatter_index[gather_index] = torch.arange(int(gather_index.numel()), dtype=torch.long)

    base = DenseLinearOperator(torch.randn(2, 5, *base_shape, dtype=torch.float32), input_shape=base_shape)
    op = ReindexInputLinearOperator(base=base, input_shape=input_shape, gather_index=gather_index)
    dense = base.to_dense().index_select(2, scatter_index)
    center = torch.randn(2, *input_shape, dtype=torch.float32)
    vec = torch.randn(2, *input_shape, dtype=torch.float32)

    assert torch.equal(op.gather_index.cpu(), gather_index)
    assert torch.equal(op.scatter_index.cpu(), scatter_index)
    assert torch.allclose(op.to_dense(), dense, atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.center_term(center), torch.einsum("bki,bi->bk", dense, center.reshape(2, -1)), atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.contract_input(vec), torch.einsum("bki,bi->bk", dense, vec.reshape(2, -1)), atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.row_abs_sum(), dense.abs().sum(dim=2), atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.row_l2_norm(), torch.linalg.vector_norm(dense, ord=2, dim=2), atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.row_abs_max(), dense.abs().amax(dim=2), atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.row_abs_sum(), base.row_abs_sum(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.row_l2_norm(), base.row_l2_norm(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(op.row_abs_max(), base.row_abs_max(), atol=1e-5, rtol=1e-5)


def test_reindex_input_linear_operator_split_pos_neg_matches_dense_reference() -> None:
    torch.manual_seed(0)
    input_shape = (2, 2, 3)
    dims_with_batch = (0, 3, 1, 2)
    gather_index = _make_gather_index(input_shape, dims_with_batch)
    scatter_index = torch.empty_like(gather_index)
    scatter_index[gather_index] = torch.arange(int(gather_index.numel()), dtype=torch.long)

    base = DenseLinearOperator(torch.randn(2, 4, 3, 2, 2, dtype=torch.float32), input_shape=(3, 2, 2))
    op = ReindexInputLinearOperator(base=base, input_shape=input_shape, gather_index=gather_index)
    dense = base.to_dense().index_select(2, scatter_index)
    pos, neg = op.split_pos_neg()

    assert isinstance(pos, ReindexInputLinearOperator)
    assert isinstance(neg, ReindexInputLinearOperator)
    assert torch.allclose(pos.to_dense(), dense.clamp_min(0.0), atol=1e-5, rtol=1e-5)
    assert torch.allclose(neg.to_dense(), dense.clamp_max(0.0), atol=1e-5, rtol=1e-5)


def test_backprop_permute_step_preserves_operator_form_and_matches_dense_reference() -> None:
    torch.manual_seed(0)
    input_shape = (2, 2, 3)
    dims_with_batch = (0, 2, 3, 1)
    output_shape = (2, 3, 2)
    gather_index = _make_gather_index(input_shape, dims_with_batch)
    scatter_index = torch.empty_like(gather_index)
    scatter_index[gather_index] = torch.arange(int(gather_index.numel()), dtype=torch.long)

    state = type("State", (), {
        "A_u": DenseLinearOperator(torch.randn(2, 4, *output_shape, dtype=torch.float32), input_shape=output_shape),
        "A_l": DenseLinearOperator(torch.randn(2, 4, *output_shape, dtype=torch.float32), input_shape=output_shape),
        "b_u": torch.randn(2, 4, dtype=torch.float32),
        "b_l": torch.randn(2, 4, dtype=torch.float32),
    })()
    contrib = _backprop_permute_step(
        state,
        input_shape=input_shape,
        dims_with_batch=dims_with_batch,
        device=state.A_u.device,
        caller="test_backprop_permute_step",
    )

    assert isinstance(contrib.A_u, ReindexInputLinearOperator)
    assert isinstance(contrib.A_l, ReindexInputLinearOperator)
    assert torch.allclose(contrib.A_u.to_dense(), state.A_u.to_dense().index_select(2, scatter_index), atol=1e-5, rtol=1e-5)
    assert torch.allclose(contrib.A_l.to_dense(), state.A_l.to_dense().index_select(2, scatter_index), atol=1e-5, rtol=1e-5)
    assert torch.allclose(contrib.b_u, state.b_u)
    assert torch.allclose(contrib.b_l, state.b_l)


def test_shared_crown_supports_permute_reshape_linear_soundly() -> None:
    torch.manual_seed(0)
    weight = torch.randn(3, 8, dtype=torch.float32)
    bias = torch.randn(3, dtype=torch.float32)
    module = _make_permute_reshape_linear_module(weight=weight, bias=bias)
    x0 = torch.randn(2, 2, 2, 2, dtype=torch.float32)
    eps = 0.08

    stats = get_crown_ibp_mlp_stats(module)
    bounds = run_crown_ibp_mlp(module, InputSpec.linf(value_name="input", center=x0, eps=eps))
    ys = _eval_permute_reshape_linear(
        _sample_linf_ball(x0=x0, eps=eps, n=256).reshape(-1, *x0.shape[1:]),
        weight=weight,
        bias=bias,
    ).reshape(256, x0.shape[0], -1)

    assert stats.supported is True
    assert tuple(bounds.lower.shape) == (2, 3)
    assert tuple(bounds.upper.shape) == (2, 3)
    assert (ys >= bounds.lower.unsqueeze(0) - 1e-5).all()
    assert (ys <= bounds.upper.unsqueeze(0) + 1e-5).all()
