import torch
import torch.nn.functional as F

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.alpha_crown import run_alpha_crown_mlp
from boundflow.runtime.bab import solve_bab_mlp
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.task_executor import InputSpec


def _make_small_cnn_module() -> BFTaskModule:
    w1 = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    b1 = torch.zeros((1,), dtype=torch.float32)
    w2 = torch.ones((1, 4), dtype=torch.float32)
    b2 = torch.zeros((1,), dtype=torch.float32)
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="conv2d",
                name="conv1",
                inputs=["input", "W1", "b1"],
                outputs=["h1"],
                attrs={"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="flatten", name="flatten", inputs=["r1"], outputs=["flat"], attrs={"start_dim": 1, "end_dim": -1}),
            TaskOp(op_type="linear", name="linear1", inputs=["flat", "W2", "b2"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2}},
    )


def _eval_small_cnn(xs: torch.Tensor) -> torch.Tensor:
    flat_x = xs.flatten(0, 1)
    h1 = F.conv2d(flat_x, torch.ones((1, 1, 1, 1), dtype=xs.dtype, device=xs.device), bias=None)
    r1 = torch.relu(h1)
    flat = r1.flatten(1)
    out = flat.matmul(torch.ones((4, 1), dtype=xs.dtype, device=xs.device)).reshape(xs.shape[0], xs.shape[1], 1)
    return out


def _sample_linf_ball(*, x0: torch.Tensor, eps: float, n: int) -> torch.Tensor:
    noise = torch.rand((n,) + tuple(x0.shape), device=x0.device, dtype=x0.dtype) * 2.0 - 1.0
    return x0.unsqueeze(0) + float(eps) * noise


def test_phase7a_pr5_alpha_crown_cnn_improves_lower_and_is_sound() -> None:
    module = _make_small_cnn_module()
    x0 = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=1.0)

    b0, _s0, _ = run_alpha_crown_mlp(module, spec, steps=0, lr=0.2, alpha_init=0.5, objective="lower")
    b1, s1, _ = run_alpha_crown_mlp(module, spec, steps=30, lr=0.2, alpha_init=0.5, objective="lower")

    assert float(b1.lower.item()) >= float(b0.lower.item()) - 1e-6
    assert float(b1.lower.item()) >= -1e-4
    assert float(b1.upper.item()) >= float(b1.lower.item())

    alpha = s1.alpha_by_relu_input["h1"]
    assert tuple(alpha.shape) == (1, 2, 2)
    assert float(alpha.min().item()) >= 0.0
    assert float(alpha.max().item()) <= 1.0

    torch.manual_seed(0)
    xs = _sample_linf_ball(x0=x0, eps=1.0, n=256)
    ys = _eval_small_cnn(xs)
    assert (ys >= b1.lower.unsqueeze(0) - 1e-5).all()
    assert (ys <= b1.upper.unsqueeze(0) + 1e-5).all()


def test_phase7a_pr5_alpha_crown_cnn_warm_start_not_worse() -> None:
    module = _make_small_cnn_module()
    x0 = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=1.0)

    b_a, s_a, _ = run_alpha_crown_mlp(module, spec, steps=8, lr=0.2, alpha_init=0.5, objective="lower")
    b_b, s_b, _ = run_alpha_crown_mlp(
        module,
        spec,
        steps=8,
        lr=0.2,
        alpha_init=0.5,
        objective="lower",
        warm_start=s_a,
    )

    assert float(b_b.lower.item()) >= float(b_a.lower.item()) - 1e-6
    assert tuple(s_b.alpha_by_relu_input["h1"].shape) == (1, 2, 2)


def test_phase7a_pr5_cnn_relu_alpha_has_gradient() -> None:
    module = _make_small_cnn_module()
    x0 = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=1.0)

    alpha = torch.nn.Parameter(torch.full((1, 2, 2), 0.5, dtype=torch.float32))
    bounds = run_crown_ibp_mlp(module, spec, relu_alpha={"h1": alpha})
    loss = -bounds.lower.mean()
    loss.backward()

    assert alpha.grad is not None
    assert torch.isfinite(alpha.grad).all()
    assert float(alpha.grad.abs().sum().item()) > 0.0


def test_phase7a_pr5_structured_alpha_matches_flat_alpha() -> None:
    module = _make_small_cnn_module()
    x0 = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=1.0)
    alpha_struct = torch.tensor([[[0.1, 0.3], [0.6, 0.9]]], dtype=torch.float32)
    alpha_flat = alpha_struct.flatten()

    bounds_struct = run_crown_ibp_mlp(module, spec, relu_alpha={"h1": alpha_struct})
    bounds_flat = run_crown_ibp_mlp(module, spec, relu_alpha={"h1": alpha_flat})

    assert torch.allclose(bounds_struct.lower, bounds_flat.lower, atol=1e-6, rtol=1e-6)
    assert torch.allclose(bounds_struct.upper, bounds_flat.upper, atol=1e-6, rtol=1e-6)


def test_phase7a_pr5_conv_split_state_runs_after_pr6() -> None:
    module = _make_small_cnn_module()
    x0 = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=1.0)
    split = {"h1": torch.tensor([[[1, 0], [0, 0]]], dtype=torch.int8)}

    bounds, state, _stats = run_alpha_crown_mlp(module, spec, steps=0, relu_split_state=split)

    assert tuple(bounds.lower.shape) == (1, 1)
    assert tuple(state.alpha_by_relu_input["h1"].shape) == (1, 2, 2)


def test_phase7a_pr5_bab_conv_fail_fast() -> None:
    import pytest

    module = _make_small_cnn_module()
    x0 = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    spec = InputSpec.linf(value_name="input", center=x0, eps=1.0)

    with pytest.raises(NotImplementedError, match="alpha-only BaB does not yet support conv graphs"):
        solve_bab_mlp(module, spec)
