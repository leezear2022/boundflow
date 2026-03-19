import torch

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.runtime.alpha_beta_crown import check_first_layer_infeasible_split
from boundflow.runtime.bab import BabConfig, NodeEvalCache, ReluSplitState, _QueueItem, prune_infeasible_first_layer_items
from boundflow.runtime.task_executor import InputSpec


def _make_first_layer_three_neuron_module(*, w: torch.Tensor, b: torch.Tensor) -> BFTaskModule:
    hidden, in_dim = int(w.shape[0]), int(w.shape[1])
    assert tuple(b.shape) == (hidden,)
    w2 = torch.eye(hidden, dtype=w.dtype)
    b2 = torch.zeros(hidden, dtype=w.dtype)
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
    return BFTaskModule(tasks=[task], entry_task_id="t0", bindings={"params": {"W1": w, "b1": b, "W2": w2, "b2": b2}})


def test_phase6g_batch_prune_mixed_infeasible_and_feasible_nodes() -> None:
    # Same clean non-pairwise infeasible construction as Phase 6F, but exercised via PR-3C prune helper.
    t = 0.49
    a1 = torch.tensor([1.0, 0.0], dtype=torch.float32)
    a2 = torch.tensor([-0.5, 0.8660254], dtype=torch.float32)
    a3 = torch.tensor([-0.5, -0.8660254], dtype=torch.float32)
    w = torch.stack([a1, a2, a3], dim=0)
    b = torch.full((3,), -t, dtype=torch.float32)
    module = _make_first_layer_three_neuron_module(w=w, b=b)
    spec = InputSpec.l2(value_name="input", center=torch.zeros(1, 2, dtype=torch.float32), eps=1.0)

    infeasible = ReluSplitState(by_relu_input={"h1": torch.tensor([+1, +1, +1], dtype=torch.int8)})
    feasible = ReluSplitState(by_relu_input={"h1": torch.tensor([0, 0, 0], dtype=torch.int8)})

    st_inf = check_first_layer_infeasible_split(module, spec, relu_split_state=infeasible.by_relu_input)
    assert st_inf.feasibility == "infeasible"
    st_ok = check_first_layer_infeasible_split(module, spec, relu_split_state=feasible.by_relu_input)
    assert st_ok.feasibility == "unknown"

    cfg = BabConfig(oracle="alpha_beta", enable_batch_infeasible_prune=True)
    cache = NodeEvalCache(module=module, input_spec=spec, linear_spec_C=None, cfg=cfg)

    items = [
        (0, _QueueItem(priority=0.0, node_id=0, split_state=infeasible)),
        (1, _QueueItem(priority=0.0, node_id=1, split_state=feasible)),
    ]
    kept, pruned = prune_infeasible_first_layer_items(module, spec, items=items, cache=cache, cfg=cfg)
    assert pruned == [0]
    assert [i for i, _ in kept] == [1]

    # Cache should remember the infeasible node as infeasible.
    v = cache.get(split_state=infeasible)
    assert v is not None
    assert getattr(v.stats, "feasibility", None) == "infeasible"

