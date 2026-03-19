import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.alpha_beta_crown import check_first_layer_infeasible_split
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.linear_operator import DenseLinearOperator, LinearOperator
from boundflow.runtime.perturbation import LpBallPerturbation
from boundflow.runtime.task_executor import InputSpec


def _make_linear_module(*, weight: torch.Tensor, bias: torch.Tensor | None = None) -> BFTaskModule:
    params = {"W": weight}
    op_inputs = ["input", "W"]
    if bias is not None:
        params["b"] = bias
        op_inputs.append("b")
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[TaskOp(op_type="linear", name="linear0", inputs=op_inputs, outputs=["out"])],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(tasks=[task], entry_task_id="t0", bindings={"params": params})


def _make_relu_mlp_module(*, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor) -> BFTaskModule:
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


@pytest.mark.parametrize("p", ["inf", 2, 1])
def test_dense_linear_operator_matches_tensor_concretize(p: str | int) -> None:
    torch.manual_seed(0)
    batch = 3
    inputs = 5
    outputs = 4
    center = torch.randn(batch, inputs, dtype=torch.float32)
    A = torch.randn(batch, outputs, inputs, dtype=torch.float32)
    b = torch.randn(batch, outputs, dtype=torch.float32)
    perturbation = LpBallPerturbation(p=p, eps=0.3)

    lb_dense, ub_dense = perturbation.concretize_affine(center=center, A=A, b=b)
    lb_op, ub_op = perturbation.concretize_affine(center=center, A=DenseLinearOperator(A), b=b)

    assert torch.allclose(lb_dense, lb_op)
    assert torch.allclose(ub_dense, ub_op)


def test_dense_linear_operator_rejects_non_floating_coeffs() -> None:
    coeffs = torch.ones(2, 3, 4, dtype=torch.int64)
    with pytest.raises(TypeError, match="floating coeffs"):
        DenseLinearOperator(coeffs)


def test_dense_linear_operator_round_trip_dense() -> None:
    coeffs = torch.randn(2, 3, 4, dtype=torch.float32)
    op = DenseLinearOperator(coeffs)
    assert torch.equal(op.to_dense(), coeffs)


def test_concretize_affine_rejects_center_shape_mismatch() -> None:
    perturbation = LpBallPerturbation(p="inf", eps=0.2)
    center = torch.randn(2, 4, dtype=torch.float32)
    op = DenseLinearOperator(torch.randn(3, 5, 4, dtype=torch.float32))
    with pytest.raises(ValueError, match="affine shape mismatch"):
        perturbation.concretize_affine(center=center, A=op)


def test_concretize_affine_rejects_dtype_mismatch() -> None:
    perturbation = LpBallPerturbation(p="inf", eps=0.2)
    center = torch.randn(2, 4, dtype=torch.float32)
    op = DenseLinearOperator(torch.randn(2, 5, 4, dtype=torch.float64))
    with pytest.raises(TypeError, match="affine dtype mismatch"):
        perturbation.concretize_affine(center=center, A=op)


def test_concretize_affine_rejects_device_mismatch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    perturbation = LpBallPerturbation(p="inf", eps=0.2)
    center = torch.randn(2, 4, dtype=torch.float32, device="cpu")
    op = DenseLinearOperator(torch.randn(2, 5, 4, dtype=torch.float32, device="cuda"))
    with pytest.raises(ValueError, match="affine device mismatch"):
        perturbation.concretize_affine(center=center, A=op)


def test_run_crown_ibp_mlp_uses_linear_operator(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    x0 = torch.randn(2, 4, dtype=torch.float32)
    module = _make_relu_mlp_module(
        w1=torch.randn(6, 4, dtype=torch.float32),
        b1=torch.randn(6, dtype=torch.float32),
        w2=torch.randn(3, 6, dtype=torch.float32),
        b2=torch.randn(3, dtype=torch.float32),
    )
    seen: list[object] = []
    original = LpBallPerturbation.concretize_affine

    def wrapped(self, *, center: torch.Tensor, A: torch.Tensor | LinearOperator, b: torch.Tensor | None = None):
        seen.append(A)
        return original(self, center=center, A=A, b=b)

    monkeypatch.setattr(LpBallPerturbation, "concretize_affine", wrapped)
    run_crown_ibp_mlp(module, InputSpec.linf(value_name="input", center=x0, eps=0.1))

    assert any(isinstance(item, LinearOperator) for item in seen)


def test_alpha_beta_first_layer_infeasible_helper_uses_dense_linear_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    module = _make_relu_mlp_module(
        w1=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        b1=torch.tensor([-0.1], dtype=torch.float32),
        w2=torch.tensor([[1.0]], dtype=torch.float32),
        b2=torch.tensor([0.0], dtype=torch.float32),
    )
    spec = InputSpec.linf(value_name="input", center=torch.zeros(1, 2, dtype=torch.float32), eps=0.2)
    seen: list[object] = []
    original = LpBallPerturbation.concretize_affine

    def wrapped(self, *, center: torch.Tensor, A: torch.Tensor | DenseLinearOperator, b: torch.Tensor | None = None):
        seen.append(A)
        return original(self, center=center, A=A, b=b)

    monkeypatch.setattr(LpBallPerturbation, "concretize_affine", wrapped)
    stats = check_first_layer_infeasible_split(
        module,
        spec,
        relu_split_state={"h1": torch.tensor([1], dtype=torch.int8)},
    )

    assert stats.feasibility in {"unknown", "infeasible"}
    assert any(isinstance(item, DenseLinearOperator) for item in seen)
