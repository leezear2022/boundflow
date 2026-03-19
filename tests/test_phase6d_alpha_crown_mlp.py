import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.alpha_crown import run_alpha_crown_mlp
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.task_executor import InputSpec


def _make_toy_relu_chain_module() -> BFTaskModule:
    # input -> linear1 -> relu1 -> linear2 -> out
    # Designed so the pre-activation interval crosses 0, and the sound lower bound for ReLU is 0.
    w1 = torch.tensor([[1.0]], dtype=torch.float32)  # [H=1,I=1]
    b1 = torch.tensor([0.0], dtype=torch.float32)
    w2 = torch.tensor([[1.0]], dtype=torch.float32)  # [O=1,H=1]
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


def _eval_toy(xs: torch.Tensor) -> torch.Tensor:
    # xs: [N,B,1] -> ys: [N,B,1]
    h = xs
    r = torch.relu(h)
    return r


def test_phase6d_alpha_crown_optim_improves_lower_from_bad_init_and_is_sound() -> None:
    module = _make_toy_relu_chain_module()
    device = torch.device("cpu")
    x0 = torch.tensor([[0.0]], device=device, dtype=torch.float32)
    eps = 1.0
    spec = InputSpec.linf(value_name="input", center=x0, eps=eps)

    # alpha_init=0.5 is intentionally suboptimal for unstable ReLU in this toy:
    # it yields a negative lower bound, while the tight bound is 0.
    b0, s0, _ = run_alpha_crown_mlp(module, spec, steps=0, lr=0.2, alpha_init=0.5, objective="lower")
    b1, s1, _ = run_alpha_crown_mlp(module, spec, steps=30, lr=0.2, alpha_init=0.5, objective="lower")

    assert float(b0.lower.item()) <= float(b1.lower.item()) + 1e-6
    assert float(b1.lower.item()) >= -1e-4  # should approach 0
    assert float(b1.upper.item()) >= float(b1.lower.item())

    # alpha must be in [0,1].
    alpha = next(iter(s1.alpha_by_relu_input.values()))
    assert float(alpha.min().item()) >= 0.0
    assert float(alpha.max().item()) <= 1.0

    # Soundness check by sampling points in the Linf ball.
    torch.manual_seed(0)
    xs = (torch.rand(256, 1, 1, device=device, dtype=torch.float32) * 2.0 - 1.0) * eps + x0.unsqueeze(0)
    ys = _eval_toy(xs)
    lb = b1.lower.unsqueeze(0)
    ub = b1.upper.unsqueeze(0)
    assert (ys >= lb - 1e-5).all()
    assert (ys <= ub + 1e-5).all()


def test_phase6d_alpha_crown_warm_start_not_worse() -> None:
    module = _make_toy_relu_chain_module()
    device = torch.device("cpu")
    x0 = torch.tensor([[0.0]], device=device, dtype=torch.float32)
    eps = 1.0
    spec = InputSpec.linf(value_name="input", center=x0, eps=eps)

    b_a, s_a, _ = run_alpha_crown_mlp(module, spec, steps=5, lr=0.2, alpha_init=0.5, objective="lower")
    b_b, s_b, _ = run_alpha_crown_mlp(
        module, spec, steps=5, lr=0.2, alpha_init=0.5, objective="lower", warm_start=s_a
    )

    assert float(b_b.lower.item()) >= float(b_a.lower.item()) - 1e-6
    alpha_b = next(iter(s_b.alpha_by_relu_input.values()))
    assert float(alpha_b.min().item()) >= 0.0
    assert float(alpha_b.max().item()) <= 1.0


def test_phase6d_alpha_crown_relu_alpha_has_gradient() -> None:
    module = _make_toy_relu_chain_module()
    device = torch.device("cpu")
    x0 = torch.tensor([[0.0]], device=device, dtype=torch.float32)
    eps = 1.0
    spec = InputSpec.linf(value_name="input", center=x0, eps=eps)

    alpha = torch.nn.Parameter(torch.tensor([0.5], device=device, dtype=torch.float32))
    bounds = run_crown_ibp_mlp(module, spec, relu_alpha={"h1": alpha})
    loss = -bounds.lower.mean()
    loss.backward()

    assert alpha.grad is not None
    assert torch.isfinite(alpha.grad).all()
    assert float(alpha.grad.abs().sum().item()) > 0.0
