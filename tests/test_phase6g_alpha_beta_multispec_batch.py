import torch

from boundflow.domains.interval import IntervalDomain
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.alpha_beta_crown import run_alpha_beta_crown_mlp
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.task_executor import InputSpec


def _make_mlp_module(*, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor) -> BFTaskModule:
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


def test_phase6g_alpha_beta_multispec_matches_serial_steps0() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    batch = 3
    in_dim = 8
    hidden = 7
    out_dim = 5
    specs = 6

    x0 = torch.randn(batch, in_dim, device=device, dtype=dtype)
    w1 = torch.randn(hidden, in_dim, device=device, dtype=dtype)
    b1 = torch.randn(hidden, device=device, dtype=dtype)
    w2 = torch.randn(out_dim, hidden, device=device, dtype=dtype)
    b2 = torch.randn(out_dim, device=device, dtype=dtype)
    module = _make_mlp_module(w1=w1, b1=b1, w2=w2, b2=b2)
    spec = InputSpec.linf(value_name="input", center=x0, eps=0.2)

    C = torch.randn(batch, specs, out_dim, device=device, dtype=dtype)
    batched, _alpha, _beta, stats = run_alpha_beta_crown_mlp(
        module,
        spec,
        linear_spec_C=C,
        steps=0,
        lr=0.2,
        alpha_init=0.5,
        beta_init=0.0,
        objective="lower",
    )
    assert stats.feasibility == "unknown"
    assert tuple(batched.lower.shape) == (batch, specs)
    assert tuple(batched.upper.shape) == (batch, specs)

    for s in range(specs):
        one, _a, _b, st = run_alpha_beta_crown_mlp(
            module,
            spec,
            linear_spec_C=C[:, s : s + 1, :],
            steps=0,
            lr=0.2,
            alpha_init=0.5,
            beta_init=0.0,
            objective="lower",
        )
        assert st.feasibility == "unknown"
        assert torch.allclose(one.lower.squeeze(1), batched.lower[:, s], atol=1e-6, rtol=1e-6)
        assert torch.allclose(one.upper.squeeze(1), batched.upper[:, s], atol=1e-6, rtol=1e-6)


def test_phase6g_alpha_beta_forward_work_independent_of_specs(monkeypatch) -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    batch = 2
    in_dim = 12
    hidden = 16
    out_dim = 6

    x0 = torch.randn(batch, in_dim, device=device, dtype=dtype)
    w1 = torch.randn(hidden, in_dim, device=device, dtype=dtype)
    b1 = torch.randn(hidden, device=device, dtype=dtype)
    w2 = torch.randn(out_dim, hidden, device=device, dtype=dtype)
    b2 = torch.randn(out_dim, device=device, dtype=dtype)
    module = _make_mlp_module(w1=w1, b1=b1, w2=w2, b2=b2)
    spec = InputSpec.linf(value_name="input", center=x0, eps=0.1)

    orig_affine = IntervalDomain.affine_transformer
    orig_relu = IntervalDomain.relu_transformer
    counts = {"affine": 0, "relu": 0}

    def _affine(self, *args, **kwargs):
        counts["affine"] += 1
        return orig_affine(self, *args, **kwargs)

    def _relu(self, *args, **kwargs):
        counts["relu"] += 1
        return orig_relu(self, *args, **kwargs)

    monkeypatch.setattr(IntervalDomain, "affine_transformer", _affine)
    monkeypatch.setattr(IntervalDomain, "relu_transformer", _relu)

    with torch.inference_mode():
        counts["affine"] = 0
        counts["relu"] = 0
        _ = run_alpha_beta_crown_mlp(
            module,
            spec,
            linear_spec_C=torch.randn(batch, 1, out_dim, device=device, dtype=dtype),
            steps=0,
            lr=0.2,
            alpha_init=0.5,
            beta_init=0.0,
        )
        c1 = dict(counts)

        counts["affine"] = 0
        counts["relu"] = 0
        _ = run_alpha_beta_crown_mlp(
            module,
            spec,
            linear_spec_C=torch.randn(batch, 32, out_dim, device=device, dtype=dtype),
            steps=0,
            lr=0.2,
            alpha_init=0.5,
            beta_init=0.0,
        )
        c32 = dict(counts)

    assert c1["affine"] == c32["affine"]
    assert c1["relu"] == c32["relu"]
    assert c1["affine"] > 0 and c1["relu"] > 0


def test_phase6g_multispec_gradients_alpha_and_beta() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    batch = 2
    in_dim = 6
    hidden = 4
    out_dim = 3
    specs = 5

    x0 = torch.randn(batch, in_dim, device=device, dtype=dtype)
    w1 = torch.randn(hidden, in_dim, device=device, dtype=dtype)
    b1 = torch.randn(hidden, device=device, dtype=dtype)
    w2 = torch.randn(out_dim, hidden, device=device, dtype=dtype)
    b2 = torch.randn(out_dim, device=device, dtype=dtype)
    module = _make_mlp_module(w1=w1, b1=b1, w2=w2, b2=b2)
    input_spec = InputSpec.linf(value_name="input", center=x0, eps=0.3)
    C = torch.randn(batch, specs, out_dim, device=device, dtype=dtype)

    alpha = torch.nn.Parameter(torch.full((hidden,), 0.5, device=device, dtype=dtype))
    beta = torch.nn.Parameter(torch.full((hidden,), 0.1, device=device, dtype=dtype))

    bounds = run_crown_ibp_mlp(
        module,
        input_spec,
        linear_spec_C=C,
        relu_alpha={"h1": alpha},
        relu_pre_add_coeff_l={"h1": -beta},
    )
    # Encourage worst-case spec lower to be tight.
    loss = -bounds.lower.min(dim=1).values.mean()
    loss.backward()

    assert alpha.grad is not None
    assert beta.grad is not None
    assert torch.isfinite(alpha.grad).all()
    assert torch.isfinite(beta.grad).all()
    assert float(alpha.grad.abs().sum().item()) > 0.0
    assert float(beta.grad.abs().sum().item()) > 0.0
