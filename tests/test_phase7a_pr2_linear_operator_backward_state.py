import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.linear_operator import DenseLinearOperator, RightMatmulLinearOperator
from boundflow.runtime.task_executor import InputSpec


def _make_linear_chain_module(*, weights: list[torch.Tensor], biases: list[torch.Tensor]) -> BFTaskModule:
    if len(weights) != len(biases):
        raise ValueError("weights and biases must have the same length")
    if not weights:
        raise ValueError("weights must be non-empty")
    ops: list[TaskOp] = []
    params: dict[str, torch.Tensor] = {}
    cur = "input"
    for idx, (w, b) in enumerate(zip(weights, biases), start=1):
        w_name = f"W{idx}"
        b_name = f"b{idx}"
        out = "out" if idx == len(weights) else f"h{idx}"
        ops.append(TaskOp(op_type="linear", name=f"linear{idx}", inputs=[cur, w_name, b_name], outputs=[out]))
        params[w_name] = w
        params[b_name] = b
        cur = out
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=ops,
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(tasks=[task], entry_task_id="t0", bindings={"params": params})


def _make_linear_linear_relu_linear_module(
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
            TaskOp(op_type="linear", name="linear2", inputs=["h1", "W2", "b2"], outputs=["h2"]),
            TaskOp(op_type="relu", name="relu1", inputs=["h2"], outputs=["r1"]),
            TaskOp(op_type="linear", name="linear3", inputs=["r1", "W3", "b3"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2, "W3": w3, "b3": b3}},
    )


def _sample_linf_ball(*, x0: torch.Tensor, eps: float, n: int) -> torch.Tensor:
    noise = torch.rand((n,) + tuple(x0.shape), device=x0.device, dtype=x0.dtype) * 2.0 - 1.0
    return x0.unsqueeze(0) + float(eps) * noise


def _eval_linear_linear_relu_linear(
    xs: torch.Tensor,
    *,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
) -> torch.Tensor:
    h1 = xs.matmul(w1.t()) + b1
    h2 = h1.matmul(w2.t()) + b2
    r1 = torch.relu(h2)
    return r1.matmul(w3.t()) + b3


def test_right_matmul_linear_operator_matches_dense_reference() -> None:
    torch.manual_seed(0)
    coeffs = torch.randn(2, 3, 4, dtype=torch.float32)
    rhs = torch.randn(4, 5, dtype=torch.float32)
    center = torch.randn(2, 5, dtype=torch.float32)
    vec = torch.randn(5, dtype=torch.float32)

    op = DenseLinearOperator(coeffs).matmul_right(rhs)
    assert isinstance(op, RightMatmulLinearOperator)

    dense = torch.einsum("bko,oi->bki", coeffs, rhs)
    assert torch.allclose(op.to_dense(), dense)
    assert torch.allclose(op.center_term(center), torch.einsum("bki,bi->bk", dense, center))
    assert torch.allclose(op.contract_last_dim(vec), torch.einsum("bki,i->bk", dense, vec))
    assert torch.allclose(op.row_abs_sum(), dense.abs().sum(dim=2))
    assert torch.allclose(op.row_l2_norm(), torch.linalg.vector_norm(dense, ord=2, dim=2))
    assert torch.allclose(op.row_abs_max(), dense.abs().amax(dim=2))


def test_right_matmul_linear_operator_fuses_nested_composition() -> None:
    torch.manual_seed(0)
    coeffs = torch.randn(2, 3, 4, dtype=torch.float32)
    rhs1 = torch.randn(4, 6, dtype=torch.float32)
    rhs2 = torch.randn(6, 5, dtype=torch.float32)

    op = DenseLinearOperator(coeffs).matmul_right(rhs1).matmul_right(rhs2)

    assert isinstance(op, RightMatmulLinearOperator)
    assert isinstance(op.base, DenseLinearOperator)
    expected_rhs = rhs1.matmul(rhs2)
    assert torch.allclose(op.rhs, expected_rhs)
    assert torch.allclose(op.to_dense(), torch.einsum("bko,oi->bki", coeffs, expected_rhs))


def test_right_matmul_linear_operator_rejects_invalid_rhs_and_vec() -> None:
    coeffs = torch.randn(2, 3, 4, dtype=torch.float32)
    op = DenseLinearOperator(coeffs)

    with pytest.raises(TypeError, match="floating tensor"):
        op.matmul_right(torch.ones(4, 5, dtype=torch.int64))
    with pytest.raises(ValueError, match="rank-2 matrix"):
        op.matmul_right(torch.randn(4, dtype=torch.float32))
    with pytest.raises(ValueError, match="first dim 4"):
        op.matmul_right(torch.randn(5, 6, dtype=torch.float32))
    with pytest.raises(TypeError, match="dtype mismatch"):
        op.contract_last_dim(torch.randn(4, dtype=torch.float64))


def test_run_crown_ibp_mlp_linear_chain_uses_operator_matmul_right(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    weights = [
        torch.randn(7, 5, dtype=torch.float32),
        torch.randn(6, 7, dtype=torch.float32),
        torch.randn(4, 6, dtype=torch.float32),
    ]
    biases = [
        torch.randn(7, dtype=torch.float32),
        torch.randn(6, dtype=torch.float32),
        torch.randn(4, dtype=torch.float32),
    ]
    module = _make_linear_chain_module(weights=weights, biases=biases)
    x0 = torch.randn(2, 5, dtype=torch.float32)
    counts = {"dense": 0, "right": 0}

    orig_dense = DenseLinearOperator.matmul_right
    orig_right = RightMatmulLinearOperator.matmul_right

    def wrapped_dense(self, rhs: torch.Tensor):
        counts["dense"] += 1
        return orig_dense(self, rhs)

    def wrapped_right(self, rhs: torch.Tensor):
        counts["right"] += 1
        return orig_right(self, rhs)

    monkeypatch.setattr(DenseLinearOperator, "matmul_right", wrapped_dense)
    monkeypatch.setattr(RightMatmulLinearOperator, "matmul_right", wrapped_right)

    bounds = run_crown_ibp_mlp(module, InputSpec.linf(value_name="input", center=x0, eps=0.2))

    assert tuple(bounds.lower.shape) == (2, 4)
    assert tuple(bounds.upper.shape) == (2, 4)
    assert counts["dense"] + counts["right"] == 2 * len(weights)
    assert counts["dense"] == 2


def test_run_crown_ibp_mlp_mixed_linear_segment_and_relu_remains_sound() -> None:
    torch.manual_seed(0)
    x0 = torch.randn(2, 4, dtype=torch.float32)
    w1 = torch.randn(5, 4, dtype=torch.float32)
    b1 = torch.randn(5, dtype=torch.float32)
    w2 = torch.randn(6, 5, dtype=torch.float32)
    b2 = torch.randn(6, dtype=torch.float32)
    w3 = torch.randn(3, 6, dtype=torch.float32)
    b3 = torch.randn(3, dtype=torch.float32)
    eps = 0.15

    module = _make_linear_linear_relu_linear_module(w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
    bounds = run_crown_ibp_mlp(module, InputSpec.linf(value_name="input", center=x0, eps=eps))

    xs = _sample_linf_ball(x0=x0, eps=eps, n=256)
    ys = _eval_linear_linear_relu_linear(xs, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
    lb = bounds.lower.unsqueeze(0)
    ub = bounds.upper.unsqueeze(0)
    assert (ys >= lb - 1e-5).all()
    assert (ys <= ub + 1e-5).all()
