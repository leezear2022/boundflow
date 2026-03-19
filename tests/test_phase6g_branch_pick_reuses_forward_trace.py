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


def test_phase6g_branch_pick_does_not_run_second_forward(monkeypatch) -> None:
    module = _make_single_relu_module()
    spec = InputSpec.linf(value_name="input", center=torch.tensor([[0.0]], dtype=torch.float32), eps=1.0)

    import boundflow.runtime.alpha_beta_crown as ab
    import boundflow.runtime.crown_ibp as ci

    counts = {"oracle_forward": 0, "branch_forward": 0}
    orig_ab_forward = ab._forward_ibp_trace_mlp
    orig_ci_forward = ci._forward_ibp_trace_mlp

    def _ab_forward(*args, **kwargs):
        counts["oracle_forward"] += 1
        return orig_ab_forward(*args, **kwargs)

    def _ci_forward(*args, **kwargs):
        counts["branch_forward"] += 1
        return orig_ci_forward(*args, **kwargs)

    monkeypatch.setattr(ab, "_forward_ibp_trace_mlp", _ab_forward)
    monkeypatch.setattr(ci, "_forward_ibp_trace_mlp", _ci_forward)

    # max_nodes=1 forces: eval root once + pick branch once (then stop).
    cfg = BabConfig(
        max_nodes=1,
        oracle="alpha_beta",
        node_batch_size=1,
        enable_node_eval_cache=False,
        alpha_steps=0,
        alpha_lr=0.2,
        alpha_init=0.5,
        beta_init=0.0,
        threshold=0.0,
        tol=1e-8,
    )
    _res = solve_bab_mlp(module, spec, config=cfg)

    assert counts["oracle_forward"] == 1
    # Branch picking should reuse oracle's forward trace, so it must not trigger another forward trace call.
    assert counts["branch_forward"] == 0

