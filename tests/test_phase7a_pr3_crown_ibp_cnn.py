import pytest
import torch
import torch.nn.functional as F

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.crown_ibp import get_crown_ibp_mlp_stats, run_crown_ibp_mlp
from boundflow.runtime.task_executor import InputSpec, LinfInputSpec, PythonTaskExecutor


def _make_cnn_module(
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
            TaskOp(
                op_type="conv2d",
                name="conv1",
                inputs=["input", "W1", "b1"],
                outputs=["h1"],
                attrs={"stride": (1, 1), "padding": (1, 1), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(
                op_type="conv2d",
                name="conv2",
                inputs=["r1", "W2", "b2"],
                outputs=["h2"],
                attrs={"stride": (2, 2), "padding": (1, 1), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp(op_type="relu", name="relu2", inputs=["h2"], outputs=["r2"]),
            TaskOp(op_type="flatten", name="flatten", inputs=["r2"], outputs=["flat"], attrs={"start_dim": 1, "end_dim": -1}),
            TaskOp(op_type="linear", name="linear", inputs=["flat", "W3", "b3"], outputs=["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id="t0",
        bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2, "W3": w3, "b3": b3}},
    )


def _make_non_chain_conv_module(*, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor) -> BFTaskModule:
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                op_type="conv2d",
                name="conv1",
                inputs=["input", "W1", "b1"],
                outputs=["h1"],
                attrs={"stride": (1, 1), "padding": (1, 1), "dilation": (1, 1), "groups": 1},
            ),
            TaskOp(op_type="relu", name="relu1", inputs=["h1"], outputs=["r1"]),
            TaskOp(op_type="linear", name="linear", inputs=["h1", "W2", "b2"], outputs=["out"]),
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
    noise = torch.rand((n,) + tuple(x0.shape), device=x0.device, dtype=x0.dtype) * 2.0 - 1.0
    return x0.unsqueeze(0) + float(eps) * noise


def _sample_l2_ball(*, x0: torch.Tensor, eps: float, n: int) -> torch.Tensor:
    batch = int(x0.shape[0])
    dim = int(x0[0].numel())
    g = torch.randn((n, batch, dim), device=x0.device, dtype=x0.dtype)
    g_norm = torch.linalg.vector_norm(g, ord=2, dim=-1, keepdim=True).clamp_min(1e-12)
    u = g / g_norm
    r = torch.rand((n, batch, 1), device=x0.device, dtype=x0.dtype).pow(1.0 / dim) * float(eps)
    return x0.unsqueeze(0) + (u * r).view((n,) + tuple(x0.shape))


def _eval_cnn(
    xs: torch.Tensor,
    *,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
) -> torch.Tensor:
    flat_x = xs.flatten(0, 1)
    h1 = F.conv2d(flat_x, w1, bias=b1, stride=1, padding=1)
    r1 = torch.relu(h1)
    h2 = F.conv2d(r1, w2, bias=b2, stride=2, padding=1)
    r2 = torch.relu(h2)
    flat = r2.flatten(1)
    out = flat.matmul(w3.t()) + b3
    return out.view(xs.shape[0], xs.shape[1], -1)


def test_crown_ibp_cnn_linf_sound_and_upper_not_worse_than_ibp() -> None:
    torch.manual_seed(0)
    x0 = torch.randn(2, 1, 5, 5, dtype=torch.float32)
    w1 = torch.randn(2, 1, 3, 3, dtype=torch.float32)
    b1 = torch.randn(2, dtype=torch.float32)
    w2 = torch.randn(3, 2, 3, 3, dtype=torch.float32)
    b2 = torch.randn(3, dtype=torch.float32)
    w3 = torch.randn(4, 27, dtype=torch.float32)
    b3 = torch.randn(4, dtype=torch.float32)
    eps = 0.12
    module = _make_cnn_module(w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)

    ibp = PythonTaskExecutor().run_ibp(module, LinfInputSpec(value_name="input", center=x0, eps=eps))
    crown = run_crown_ibp_mlp(module, InputSpec.linf(value_name="input", center=x0, eps=eps))

    xs = _sample_linf_ball(x0=x0, eps=eps, n=192)
    ys = _eval_cnn(xs, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)

    assert (crown.upper <= ibp.upper + 1e-5).all()
    assert (ys >= crown.lower.unsqueeze(0) - 1e-5).all()
    assert (ys <= crown.upper.unsqueeze(0) + 1e-5).all()


def test_crown_ibp_cnn_l2_soundness() -> None:
    torch.manual_seed(0)
    x0 = torch.randn(2, 1, 5, 5, dtype=torch.float32)
    w1 = torch.randn(2, 1, 3, 3, dtype=torch.float32)
    b1 = torch.randn(2, dtype=torch.float32)
    w2 = torch.randn(3, 2, 3, 3, dtype=torch.float32)
    b2 = torch.randn(3, dtype=torch.float32)
    w3 = torch.randn(4, 27, dtype=torch.float32)
    b3 = torch.randn(4, dtype=torch.float32)
    eps = 0.35
    module = _make_cnn_module(w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)

    crown = run_crown_ibp_mlp(module, InputSpec.l2(value_name="input", center=x0, eps=eps))
    xs = _sample_l2_ball(x0=x0, eps=eps, n=160)
    ys = _eval_cnn(xs, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)

    assert (ys >= crown.lower.unsqueeze(0) - 1e-5).all()
    assert (ys <= crown.upper.unsqueeze(0) + 1e-5).all()


def test_crown_ibp_cnn_multi_spec_matches_serial() -> None:
    torch.manual_seed(0)
    x0 = torch.randn(2, 1, 5, 5, dtype=torch.float32)
    w1 = torch.randn(2, 1, 3, 3, dtype=torch.float32)
    b1 = torch.randn(2, dtype=torch.float32)
    w2 = torch.randn(3, 2, 3, 3, dtype=torch.float32)
    b2 = torch.randn(3, dtype=torch.float32)
    w3 = torch.randn(4, 27, dtype=torch.float32)
    b3 = torch.randn(4, dtype=torch.float32)
    module = _make_cnn_module(w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
    spec = InputSpec.linf(value_name="input", center=x0, eps=0.08)
    C = torch.randn(2, 3, 4, dtype=torch.float32)

    batched = run_crown_ibp_mlp(module, spec, linear_spec_C=C)
    for idx in range(int(C.shape[1])):
        serial = run_crown_ibp_mlp(module, spec, linear_spec_C=C[:, idx : idx + 1, :])
        assert torch.allclose(serial.lower.squeeze(1), batched.lower[:, idx], atol=1e-5, rtol=1e-5)
        assert torch.allclose(serial.upper.squeeze(1), batched.upper[:, idx], atol=1e-5, rtol=1e-5)


def test_crown_ibp_stats_supports_chain_cnn_and_rejects_skip_like_graph() -> None:
    torch.manual_seed(0)
    module = _make_cnn_module(
        w1=torch.randn(2, 1, 3, 3, dtype=torch.float32),
        b1=torch.randn(2, dtype=torch.float32),
        w2=torch.randn(3, 2, 3, 3, dtype=torch.float32),
        b2=torch.randn(3, dtype=torch.float32),
        w3=torch.randn(4, 27, dtype=torch.float32),
        b3=torch.randn(4, dtype=torch.float32),
    )
    stats = get_crown_ibp_mlp_stats(module)
    assert stats.supported is True

    bad = _make_non_chain_conv_module(
        w1=torch.randn(2, 1, 3, 3, dtype=torch.float32),
        b1=torch.randn(2, dtype=torch.float32),
        w2=torch.randn(4, 50, dtype=torch.float32),
        b2=torch.randn(4, dtype=torch.float32),
    )
    bad_stats = get_crown_ibp_mlp_stats(bad)
    assert bad_stats.supported is False
    assert "non-chain" in bad_stats.reason
