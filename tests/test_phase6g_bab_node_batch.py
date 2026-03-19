import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.bab import BabConfig, solve_bab_mlp
from boundflow.runtime.task_executor import InputSpec


def _make_single_relu_module() -> BFTaskModule:
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


def test_phase6g_bab_node_batch_matches_k1_on_1d_toy() -> None:
    module = _make_single_relu_module()
    spec = InputSpec.linf(value_name="input", center=torch.tensor([[0.0]], dtype=torch.float32), eps=1.0)

    cfg1 = BabConfig(
        max_nodes=32,
        oracle="alpha_beta",
        node_batch_size=1,
        alpha_steps=40,
        alpha_lr=0.2,
        alpha_init=0.5,
        beta_init=0.0,
        threshold=0.0,
        tol=1e-8,
    )
    res1 = solve_bab_mlp(module, spec, config=cfg1)
    assert res1.status == "proven"

    cfg4 = BabConfig(
        max_nodes=32,
        oracle="alpha_beta",
        node_batch_size=4,
        alpha_steps=40,
        alpha_lr=0.2,
        alpha_init=0.5,
        beta_init=0.0,
        threshold=0.0,
        tol=1e-8,
    )
    res4 = solve_bab_mlp(module, spec, config=cfg4)
    assert res4.status == "proven"
    assert res4.best_lower >= -1e-6

