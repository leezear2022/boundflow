import torch

from boundflow.ir.task import BFTaskModule, BoundTask, StoragePlan, TaskKind, TaskLowering, TaskOp
from boundflow.ir.task_graph import TaskDepEdge, TaskGraph
from boundflow.runtime.scheduler import run_ibp_scheduled
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor


def _make_two_task_relu_chain_module() -> BFTaskModule:
    t0 = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[TaskOp(op_type="relu", name="relu0", inputs=["input"], outputs=["h0"], attrs={})],
        input_values=["input"],
        output_values=["h0"],
        params=[],
        batch_axes={},
        memory_plan={},
        lowering=TaskLowering.TVM_TIR,
    )
    t1 = BoundTask(
        task_id="t1",
        kind=TaskKind.INTERVAL_IBP,
        ops=[TaskOp(op_type="relu", name="relu1", inputs=["h0"], outputs=["out"], attrs={})],
        input_values=["h0"],
        output_values=["out"],
        params=[],
        batch_axes={},
        memory_plan={},
        lowering=TaskLowering.TVM_TIR,
    )
    g = TaskGraph(task_ids=["t0", "t1"], edges=[TaskDepEdge(src_task_id="t0", dst_task_id="t1", values=["h0"])])
    return BFTaskModule(
        tasks=[t0, t1],
        entry_task_id="t0",
        bindings={"params": {}},
        storage_plan=StoragePlan(),
        task_graph=g,
    )


def _make_single_task_relu2_module() -> BFTaskModule:
    t = BoundTask(
        task_id="t0",
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(op_type="relu", name="relu0", inputs=["input"], outputs=["h0"], attrs={}),
            TaskOp(op_type="relu", name="relu1", inputs=["h0"], outputs=["out"], attrs={}),
        ],
        input_values=["input"],
        output_values=["out"],
        params=[],
        batch_axes={},
        memory_plan={},
        lowering=TaskLowering.TVM_TIR,
    )
    return BFTaskModule(tasks=[t], entry_task_id="t0", bindings={"params": {}}, storage_plan=StoragePlan())


def test_taskgraph_validate_and_toposort():
    module = _make_two_task_relu_chain_module()
    module.validate()
    tasks_by_id = {t.task_id: t for t in module.tasks}
    order = module.task_graph.topo_sort(tasks_by_id=tasks_by_id, entry_task_id=module.entry_task_id)  # type: ignore[union-attr]
    assert order == ["t0", "t1"]


def test_scheduler_matches_single_task_execution():
    torch.manual_seed(0)
    x0 = torch.randn(4, 8)
    eps = 0.1
    input_spec = LinfInputSpec(value_name="input", center=x0, eps=eps)

    module_multi = _make_two_task_relu_chain_module()
    module_single = _make_single_task_relu2_module()

    exe = PythonTaskExecutor()
    out_sched = run_ibp_scheduled(module_multi, input_spec, executor=exe, output_value="out")
    out_single = exe.run_ibp(module_single, input_spec, output_value="out")

    assert torch.allclose(out_sched.lower, out_single.lower)
    assert torch.allclose(out_sched.upper, out_single.upper)
