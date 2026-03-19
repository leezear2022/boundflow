import torch

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


def test_phase6g_node_batch_gradients_do_not_mix_between_nodes() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64
    nodes = 4
    in_dim = 5
    hidden = 3
    out_dim = 2

    x0 = torch.zeros(nodes, in_dim, device=device, dtype=dtype)
    spec = InputSpec.linf(value_name="input", center=x0, eps=0.2)

    w1 = torch.randn(hidden, in_dim, device=device, dtype=dtype)
    b1 = torch.randn(hidden, device=device, dtype=dtype)
    w2 = torch.randn(out_dim, hidden, device=device, dtype=dtype)
    b2 = torch.randn(out_dim, device=device, dtype=dtype)
    module = _make_mlp_module(w1=w1, b1=b1, w2=w2, b2=b2)

    C = torch.randn(nodes, 1, out_dim, device=device, dtype=dtype)
    alpha = torch.nn.Parameter(torch.full((nodes, hidden), 0.5, device=device, dtype=dtype))
    beta = torch.nn.Parameter(torch.full((nodes, hidden), 0.1, device=device, dtype=dtype))

    bounds = run_crown_ibp_mlp(
        module,
        spec,
        linear_spec_C=C,
        relu_alpha={"h1": alpha},
        relu_pre_add_coeff_l={"h1": -beta},
    )
    loss = -bounds.lower[0].mean()
    loss.backward()

    assert alpha.grad is not None
    assert beta.grad is not None
    assert torch.isfinite(alpha.grad).all()
    assert torch.isfinite(beta.grad).all()
    assert float(alpha.grad[0].abs().sum().item()) > 0.0
    assert float(beta.grad[0].abs().sum().item()) > 0.0
    assert float(alpha.grad[1:].abs().sum().item()) == 0.0
    assert float(beta.grad[1:].abs().sum().item()) == 0.0

