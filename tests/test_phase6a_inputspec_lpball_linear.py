import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.task_executor import InputSpec, LinfInputSpec, PythonTaskExecutor


def _make_linear_module(*, weight: torch.Tensor, bias: torch.Tensor | None = None) -> BFTaskModule:
    params = {"W": weight}
    inputs = ["input"]
    op_inputs = ["input", "W"]
    if bias is not None:
        params["b"] = bias
        op_inputs.append("b")
    task = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[TaskOp(op_type="linear", name="linear0", inputs=op_inputs, outputs=["out"])],
        input_values=inputs,
        output_values=["out"],
    )
    return BFTaskModule(tasks=[task], entry_task_id="t0", bindings={"params": params})


def _sample_l2_ball(*, batch: int, dim: int, eps: float, n: int, device: torch.device) -> torch.Tensor:
    g = torch.randn(n, batch, dim, device=device)
    g_norm = torch.linalg.vector_norm(g, ord=2, dim=-1, keepdim=True).clamp_min(1e-12)
    u = g / g_norm
    r = torch.rand(n, batch, 1, device=device).pow(1.0 / dim) * float(eps)
    return u * r


def test_inputspec_l2_linear_concretize_soundness() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    batch = 4
    in_dim = 32
    out_dim = 16
    eps = 0.3

    x0 = torch.randn(batch, in_dim, device=device)
    W = torch.randn(out_dim, in_dim, device=device)
    b = torch.randn(out_dim, device=device)

    module = _make_linear_module(weight=W, bias=b)
    out = PythonTaskExecutor().run_ibp(module, InputSpec.l2(value_name="input", center=x0, eps=eps))

    deltas = _sample_l2_ball(batch=batch, dim=in_dim, eps=eps, n=256, device=device)
    xs = x0.unsqueeze(0) + deltas
    ys = xs.matmul(W.t()) + b

    lb = out.lower.unsqueeze(0)
    ub = out.upper.unsqueeze(0)
    assert (ys >= lb - 1e-5).all()
    assert (ys <= ub + 1e-5).all()


def test_inputspec_linf_matches_legacy_linfinputspec_linear() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    batch = 2
    in_dim = 8
    out_dim = 5
    eps = 0.1

    x0 = torch.randn(batch, in_dim, device=device)
    W = torch.randn(out_dim, in_dim, device=device)
    b = torch.randn(out_dim, device=device)
    module = _make_linear_module(weight=W, bias=b)

    out_legacy = PythonTaskExecutor().run_ibp(module, LinfInputSpec(value_name="input", center=x0, eps=eps))
    out_new = PythonTaskExecutor().run_ibp(module, InputSpec.linf(value_name="input", center=x0, eps=eps))

    assert torch.allclose(out_legacy.lower, out_new.lower)
    assert torch.allclose(out_legacy.upper, out_new.upper)

