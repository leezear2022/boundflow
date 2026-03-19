import torch

from boundflow.domains.interval import IntervalDomain
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
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


def test_phase6c_forward_ibp_work_independent_of_specs(monkeypatch) -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32
    batch = 2
    in_dim = 12
    hidden = 16
    out_dim = 6
    eps = 0.1

    x0 = torch.randn(batch, in_dim, device=device, dtype=dtype)
    w1 = torch.randn(hidden, in_dim, device=device, dtype=dtype)
    b1 = torch.randn(hidden, device=device, dtype=dtype)
    w2 = torch.randn(out_dim, hidden, device=device, dtype=dtype)
    b2 = torch.randn(out_dim, device=device, dtype=dtype)
    module = _make_mlp_module(w1=w1, b1=b1, w2=w2, b2=b2)
    spec = InputSpec.linf(value_name="input", center=x0, eps=eps)

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
        # S=1
        counts["affine"] = 0
        counts["relu"] = 0
        out1 = run_crown_ibp_mlp(module, spec, linear_spec_C=torch.randn(batch, 1, out_dim, device=device, dtype=dtype))
        c1 = dict(counts)

        # S=32 (forward IBP should still run once; backward cost scales but does not affect these counters)
        counts["affine"] = 0
        counts["relu"] = 0
        out32 = run_crown_ibp_mlp(
            module, spec, linear_spec_C=torch.randn(batch, 32, out_dim, device=device, dtype=dtype)
        )
        c32 = dict(counts)

    assert tuple(out1.lower.shape) == (batch, 1)
    assert tuple(out32.lower.shape) == (batch, 32)
    assert c1["affine"] == c32["affine"]
    assert c1["relu"] == c32["relu"]
    assert c1["affine"] > 0 and c1["relu"] > 0
