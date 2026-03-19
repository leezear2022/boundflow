import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.bab import BabConfig, NodeEvalCache, ReluSplitState, eval_bab_alpha_beta_node
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


def test_phase6g_node_eval_cache_hit_and_miss(monkeypatch) -> None:
    module = _make_single_relu_module()
    spec = InputSpec.linf(value_name="input", center=torch.tensor([[0.0]], dtype=torch.float32), eps=1.0)
    split0 = ReluSplitState.empty(module, device=spec.center.device)

    cfg = BabConfig(oracle="alpha_beta", alpha_steps=0, alpha_lr=0.2, alpha_init=0.5, beta_init=0.0)
    cache = NodeEvalCache(module=module, input_spec=spec, linear_spec_C=None, cfg=cfg)

    import boundflow.runtime.bab as bab_mod

    calls = {"n": 0}
    orig = bab_mod.run_alpha_beta_crown_mlp

    def _wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(bab_mod, "run_alpha_beta_crown_mlp", _wrapped)

    _b0, _a0, _beta0, _st0, _branch0, hit0 = eval_bab_alpha_beta_node(
        module,
        spec,
        linear_spec_C=None,
        split_state=split0,
        warm_start_alpha=None,
        warm_start_beta=None,
        cfg=cfg,
        cache=cache,
    )
    assert hit0 is False
    assert calls["n"] == 1

    _b1, _a1, _beta1, _st1, _branch1, hit1 = eval_bab_alpha_beta_node(
        module,
        spec,
        linear_spec_C=None,
        split_state=split0,
        warm_start_alpha=None,
        warm_start_beta=None,
        cfg=cfg,
        cache=cache,
    )
    assert hit1 is True
    assert calls["n"] == 1

    split1 = split0.with_split(relu_input="h1", neuron_idx=0, split_value=+1)
    _b2, _a2, _beta2, _st2, _branch2, hit2 = eval_bab_alpha_beta_node(
        module,
        spec,
        linear_spec_C=None,
        split_state=split1,
        warm_start_alpha=None,
        warm_start_beta=None,
        cfg=cfg,
        cache=cache,
    )
    assert hit2 is False
    assert calls["n"] == 2
