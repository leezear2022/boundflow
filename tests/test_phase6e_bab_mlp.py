import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.bab import BabConfig, solve_bab_mlp
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.task_executor import InputSpec


def _make_single_relu_module() -> BFTaskModule:
    # input -> linear (identity) -> relu -> linear (identity) -> out
    w1 = torch.tensor([[1.0]], dtype=torch.float32)
    b1 = torch.tensor([0.0], dtype=torch.float32)
    w2 = torch.tensor([[1.0]], dtype=torch.float32)
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
    return BFTaskModule(tasks=[task], entry_task_id="t0", bindings={"params": {"W1": w1, "b1": b1, "W2": w2, "b2": b2}})


def test_phase6e_split_constraints_tighten_bounds() -> None:
    module = _make_single_relu_module()
    x0 = torch.tensor([[0.0]], dtype=torch.float32)
    eps = 1.0
    spec = InputSpec.linf(value_name="input", center=x0, eps=eps)

    base = run_crown_ibp_mlp(module, spec, relu_alpha={"h1": torch.tensor([0.5])})

    # The split constraints h1<=0 / h1>=0 restrict the feasible input interval for this toy:
    # x in [-1,0] and x in [0,1] respectively. Represent them as Linf balls in 1D.
    inactive_spec = InputSpec.linf(value_name="input", center=torch.tensor([[-0.5]], dtype=torch.float32), eps=0.5)
    active_spec = InputSpec.linf(value_name="input", center=torch.tensor([[0.5]], dtype=torch.float32), eps=0.5)
    inactive = run_crown_ibp_mlp(module, inactive_spec, relu_alpha={"h1": torch.tensor([0.5])})
    active = run_crown_ibp_mlp(module, active_spec, relu_alpha={"h1": torch.tensor([0.5])})

    # Splitting restricts the feasible region, so bounds should not become wider.
    assert float(inactive.lower.item()) >= float(base.lower.item()) - 1e-6
    assert float(inactive.upper.item()) <= float(base.upper.item()) + 1e-6
    assert float(active.lower.item()) >= float(base.lower.item()) - 1e-6
    assert float(active.upper.item()) <= float(base.upper.item()) + 1e-6


def test_phase6e_bab_proves_nonneg_with_suboptimal_alpha_init() -> None:
    # With alpha_init=0.5 and steps=0, the root relaxation lower bound is negative,
    # but BaB splitting makes ReLU stable and proves y>=0.
    module = _make_single_relu_module()
    x0 = torch.tensor([[0.0]], dtype=torch.float32)
    eps = 1.0
    spec = InputSpec.linf(value_name="input", center=x0, eps=eps)

    cfg = BabConfig(
        max_nodes=16,
        oracle="alpha",
        use_1d_linf_input_restriction_patch=True,
        alpha_steps=0,
        alpha_lr=0.2,
        alpha_init=0.5,
        threshold=0.0,
        tol=1e-8,
    )
    res = solve_bab_mlp(module, spec, config=cfg)
    assert res.status == "proven"
    assert res.nodes_visited >= 2
