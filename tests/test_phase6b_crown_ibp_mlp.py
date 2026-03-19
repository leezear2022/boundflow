import pytest
import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import get_crown_ibp_mlp_stats, run_crown_ibp_mlp
from boundflow.runtime.task_executor import InputSpec, LinfInputSpec, PythonTaskExecutor
from boundflow.runtime.perturbation import LpBallPerturbation


def _make_mlp_module(*, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor) -> BFTaskModule:
    # input -> linear1 -> relu1 -> linear2 -> out
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


def _sample_linf_ball(*, x0: torch.Tensor, eps: float, n: int) -> torch.Tensor:
    # Returns [n,B,I]
    u = torch.rand((n,) + tuple(x0.shape), device=x0.device, dtype=x0.dtype) * 2.0 - 1.0
    return x0.unsqueeze(0) + float(eps) * u


def _sample_l2_ball(*, x0: torch.Tensor, eps: float, n: int) -> torch.Tensor:
    # Returns [n,B,I]
    b, d = int(x0.shape[0]), int(x0.shape[1])
    g = torch.randn(n, b, d, device=x0.device, dtype=x0.dtype)
    g_norm = torch.linalg.vector_norm(g, ord=2, dim=-1, keepdim=True).clamp_min(1e-12)
    u = g / g_norm
    r = torch.rand(n, b, 1, device=x0.device, dtype=x0.dtype).pow(1.0 / d) * float(eps)
    return x0.unsqueeze(0) + u * r


def _sample_l1_ball(*, x0: torch.Tensor, eps: float, n: int) -> torch.Tensor:
    # Returns [n,B,I] with ||δ||_1 <= eps.
    b, d = int(x0.shape[0]), int(x0.shape[1])
    g = torch.randn(n, b, d, device=x0.device, dtype=x0.dtype)
    sign = torch.sign(g)
    sign[sign == 0] = 1.0
    a = g.abs()
    a_sum = a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    u = a / a_sum  # sum(u)=1
    r = torch.rand(n, b, 1, device=x0.device, dtype=x0.dtype).pow(1.0 / d) * float(eps)
    delta = sign * u * r
    return x0.unsqueeze(0) + delta


def _eval_mlp(xs: torch.Tensor, *, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    # xs: [n,B,I] -> ys: [n,B,O]
    h1 = xs.matmul(w1.t()) + b1
    r1 = torch.relu(h1)
    out = r1.matmul(w2.t()) + b2
    return out


def test_crown_ibp_mlp_linf_sound_and_upper_not_worse_than_ibp() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    batch = 3
    in_dim = 16
    hidden = 32
    out_dim = 8
    eps = 0.2

    x0 = torch.randn(batch, in_dim, device=device, dtype=dtype)
    w1 = torch.randn(hidden, in_dim, device=device, dtype=dtype)
    b1 = torch.randn(hidden, device=device, dtype=dtype)
    w2 = torch.randn(out_dim, hidden, device=device, dtype=dtype)
    b2 = torch.randn(out_dim, device=device, dtype=dtype)

    module = _make_mlp_module(w1=w1, b1=b1, w2=w2, b2=b2)
    ibp = PythonTaskExecutor().run_ibp(module, LinfInputSpec(value_name="input", center=x0, eps=eps))
    crown = run_crown_ibp_mlp(module, InputSpec.linf(value_name="input", center=x0, eps=eps))

    assert (crown.upper <= ibp.upper + 1e-5).all()

    xs = _sample_linf_ball(x0=x0, eps=eps, n=256)
    ys = _eval_mlp(xs, w1=w1, b1=b1, w2=w2, b2=b2)
    lb = crown.lower.unsqueeze(0)
    ub = crown.upper.unsqueeze(0)
    assert (ys >= lb - 1e-5).all()
    assert (ys <= ub + 1e-5).all()


def test_crown_ibp_mlp_l2_soundness() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    batch = 2
    in_dim = 24
    hidden = 16
    out_dim = 6
    eps = 0.5

    x0 = torch.randn(batch, in_dim, device=device, dtype=dtype)
    w1 = torch.randn(hidden, in_dim, device=device, dtype=dtype)
    b1 = torch.randn(hidden, device=device, dtype=dtype)
    w2 = torch.randn(out_dim, hidden, device=device, dtype=dtype)
    b2 = torch.randn(out_dim, device=device, dtype=dtype)

    module = _make_mlp_module(w1=w1, b1=b1, w2=w2, b2=b2)
    crown = run_crown_ibp_mlp(module, InputSpec.l2(value_name="input", center=x0, eps=eps))

    xs = _sample_l2_ball(x0=x0, eps=eps, n=256)
    ys = _eval_mlp(xs, w1=w1, b1=b1, w2=w2, b2=b2)
    lb = crown.lower.unsqueeze(0)
    ub = crown.upper.unsqueeze(0)
    assert (ys >= lb - 1e-5).all()
    assert (ys <= ub + 1e-5).all()


def test_crown_ibp_mlp_l1_soundness() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    batch = 2
    in_dim = 20
    hidden = 12
    out_dim = 4
    eps = 0.6

    x0 = torch.randn(batch, in_dim, device=device, dtype=dtype)
    w1 = torch.randn(hidden, in_dim, device=device, dtype=dtype)
    b1 = torch.randn(hidden, device=device, dtype=dtype)
    w2 = torch.randn(out_dim, hidden, device=device, dtype=dtype)
    b2 = torch.randn(out_dim, device=device, dtype=dtype)

    module = _make_mlp_module(w1=w1, b1=b1, w2=w2, b2=b2)
    spec = InputSpec(value_name="input", center=x0, perturbation=LpBallPerturbation(p=1, eps=eps))
    crown = run_crown_ibp_mlp(module, spec)

    xs = _sample_l1_ball(x0=x0, eps=eps, n=256)
    ys = _eval_mlp(xs, w1=w1, b1=b1, w2=w2, b2=b2)
    lb = crown.lower.unsqueeze(0)
    ub = crown.upper.unsqueeze(0)
    assert (ys >= lb - 1e-5).all()
    assert (ys <= ub + 1e-5).all()


def test_crown_ibp_mlp_bruteforce_toy_linf_catches_sign_errors() -> None:
    # Small brute-force grid check intended to catch sign-selection mistakes in backward ReLU relaxation.
    device = torch.device("cpu")
    dtype = torch.float64
    eps = 0.3
    x0 = torch.tensor([[0.2, -0.1]], device=device, dtype=dtype)  # [B=1,I=2]

    # Hand-crafted weights to include negative coefficients through the last layer.
    w1 = torch.tensor([[1.3, -0.7], [-0.4, 0.9]], device=device, dtype=dtype)  # [H=2,I=2]
    b1 = torch.tensor([0.05, -0.02], device=device, dtype=dtype)
    w2 = torch.tensor([[-1.2, 0.8], [0.6, -0.5]], device=device, dtype=dtype)  # [O=2,H=2]
    b2 = torch.tensor([0.1, -0.3], device=device, dtype=dtype)
    module = _make_mlp_module(w1=w1, b1=b1, w2=w2, b2=b2)
    crown = run_crown_ibp_mlp(module, InputSpec.linf(value_name="input", center=x0, eps=eps))

    grid = torch.linspace(-eps, eps, steps=41, device=device, dtype=dtype)
    xs = torch.stack([x0[0] + torch.stack([dx, dy]) for dx in grid for dy in grid], dim=0).unsqueeze(1)  # [N,1,2]
    ys = _eval_mlp(xs, w1=w1, b1=b1, w2=w2, b2=b2)  # [N,1,2]

    lb = crown.lower.unsqueeze(0)  # [1,1,2] -> broadcast
    ub = crown.upper.unsqueeze(0)
    assert (ys >= lb - 1e-9).all()
    assert (ys <= ub + 1e-9).all()


def test_crown_ibp_mlp_multi_spec_matches_serial() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    batch = 2
    in_dim = 10
    hidden = 12
    out_dim = 5
    specs = 4
    eps = 0.15

    x0 = torch.randn(batch, in_dim, device=device, dtype=dtype)
    w1 = torch.randn(hidden, in_dim, device=device, dtype=dtype)
    b1 = torch.randn(hidden, device=device, dtype=dtype)
    w2 = torch.randn(out_dim, hidden, device=device, dtype=dtype)
    b2 = torch.randn(out_dim, device=device, dtype=dtype)
    module = _make_mlp_module(w1=w1, b1=b1, w2=w2, b2=b2)

    C = torch.randn(batch, specs, out_dim, device=device, dtype=dtype)
    spec = InputSpec.linf(value_name="input", center=x0, eps=eps)
    batched = run_crown_ibp_mlp(module, spec, linear_spec_C=C)
    assert tuple(batched.lower.shape) == (batch, specs)
    assert tuple(batched.upper.shape) == (batch, specs)

    for s in range(specs):
        serial = run_crown_ibp_mlp(module, spec, linear_spec_C=C[:, s : s + 1, :])
        assert torch.allclose(serial.lower.squeeze(1), batched.lower[:, s], atol=1e-5, rtol=1e-5)
        assert torch.allclose(serial.upper.squeeze(1), batched.upper[:, s], atol=1e-5, rtol=1e-5)


def test_crown_ibp_mlp_C_broadcast_S_O_matches_batched() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    batch = 3
    in_dim = 8
    hidden = 9
    out_dim = 4
    specs = 3
    eps = 0.1

    x0 = torch.randn(batch, in_dim, device=device, dtype=dtype)
    w1 = torch.randn(hidden, in_dim, device=device, dtype=dtype)
    b1 = torch.randn(hidden, device=device, dtype=dtype)
    w2 = torch.randn(out_dim, hidden, device=device, dtype=dtype)
    b2 = torch.randn(out_dim, device=device, dtype=dtype)
    module = _make_mlp_module(w1=w1, b1=b1, w2=w2, b2=b2)
    spec = InputSpec.linf(value_name="input", center=x0, eps=eps)

    C_so = torch.randn(specs, out_dim, device=device, dtype=dtype)
    C_bso = C_so.unsqueeze(0).expand(batch, specs, out_dim).clone()
    out_so = run_crown_ibp_mlp(module, spec, linear_spec_C=C_so)
    out_bso = run_crown_ibp_mlp(module, spec, linear_spec_C=C_bso)
    assert torch.allclose(out_so.lower, out_bso.lower, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_so.upper, out_bso.upper, atol=1e-6, rtol=1e-6)


def test_crown_ibp_mlp_rejects_non_chain_graph() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    batch = 1
    in_dim = 4
    hidden = 6
    out_dim = 3
    eps = 0.2

    x0 = torch.randn(batch, in_dim, device=device, dtype=dtype)
    w1 = torch.randn(hidden, in_dim, device=device, dtype=dtype)
    b1 = torch.randn(hidden, device=device, dtype=dtype)
    w2 = torch.randn(out_dim, hidden, device=device, dtype=dtype)
    b2 = torch.randn(out_dim, device=device, dtype=dtype)

    # input -> linear1 -> relu1, and also linear2 takes linear1 output directly (skip), which violates chain structure.
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(op_type="linear", name="linear1", inputs=["input", "W1", "b1"], outputs=["h1"]),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="linear", name="linear2", inputs=["h1", "W2", "b2"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    module = BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2}},
    )
    stats = get_crown_ibp_mlp_stats(module)
    assert stats.supported is False
    assert "non-chain" in stats.reason

    with pytest.raises(NotImplementedError, match="chain-structured"):
        run_crown_ibp_mlp(module, InputSpec.linf(value_name="input", center=x0, eps=eps))
