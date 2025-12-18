import torch

from boundflow.ir.task import BFTaskModule, BoundTask, BufferSpec, StoragePlan, TaskKind, TaskLowering, TaskOp
from boundflow.ir.task_graph import TaskBufferDep, TaskDepEdge, TaskGraph
from boundflow.runtime.scheduler import run_ibp_scheduled
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor


def _storage_plan_for_values(values):
    buffers = {}
    v2b = {}
    for v in values:
        bid = f"buf_{v}"
        buffers[bid] = BufferSpec(buffer_id=bid, dtype="float32", shape=[4, 8], scope="global")
        v2b[v] = bid
    return StoragePlan(buffers=buffers, value_to_buffer=v2b)


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
    storage_plan = _storage_plan_for_values(["input", "h0", "out"])
    t0.input_buffers = [storage_plan.value_to_buffer["input"]]
    t0.output_buffers = [storage_plan.value_to_buffer["h0"]]
    t1.input_buffers = [storage_plan.value_to_buffer["h0"]]
    t1.output_buffers = [storage_plan.value_to_buffer["out"]]
    g = TaskGraph(
        task_ids=["t0", "t1"],
        edges=[
            TaskDepEdge(
                src_task_id="t0",
                dst_task_id="t1",
                deps=[
                    TaskBufferDep(
                        src_value="h0",
                        src_buffer_id=storage_plan.value_to_buffer["h0"],
                        dst_value="h0",
                        dst_buffer_id=storage_plan.value_to_buffer["h0"],
                    )
                ],
            )
        ],
    )
    return BFTaskModule(
        tasks=[t0, t1],
        entry_task_id="t0",
        bindings={"params": {}},
        storage_plan=storage_plan,
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
    sp = _storage_plan_for_values(["input", "h0", "out"])
    t.input_buffers = [sp.value_to_buffer["input"]]
    t.output_buffers = [sp.value_to_buffer["out"]]
    return BFTaskModule(
        tasks=[t],
        entry_task_id="t0",
        bindings={"params": {}},
        storage_plan=sp,
    )


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


def test_scheduler_branch_and_merge():
    # t0: input -> h0
    # t1: h0 -> a
    # t2: h0 -> b
    # t3: a,b -> out
    values = ["input", "h0", "a", "b", "out"]
    sp = _storage_plan_for_values(values)

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
        ops=[TaskOp(op_type="relu", name="relu1", inputs=["h0"], outputs=["a"], attrs={})],
        input_values=["h0"],
        output_values=["a"],
        params=[],
        batch_axes={},
        memory_plan={},
        lowering=TaskLowering.TVM_TIR,
    )
    t2 = BoundTask(
        task_id="t2",
        kind=TaskKind.INTERVAL_IBP,
        ops=[TaskOp(op_type="relu", name="relu2", inputs=["h0"], outputs=["b"], attrs={})],
        input_values=["h0"],
        output_values=["b"],
        params=[],
        batch_axes={},
        memory_plan={},
        lowering=TaskLowering.TVM_TIR,
    )
    t3 = BoundTask(
        task_id="t3",
        kind=TaskKind.INTERVAL_IBP,
        ops=[TaskOp(op_type="add", name="add0", inputs=["a", "b"], outputs=["out"], attrs={})],
        input_values=["a", "b"],
        output_values=["out"],
        params=[],
        batch_axes={},
        memory_plan={},
        lowering=TaskLowering.TVM_TIR,
    )
    g = TaskGraph(
        task_ids=["t0", "t1", "t2", "t3"],
        edges=[
            TaskDepEdge(
                src_task_id="t0",
                dst_task_id="t1",
                deps=[
                    TaskBufferDep("h0", sp.value_to_buffer["h0"], "h0", sp.value_to_buffer["h0"]),
                ],
            ),
            TaskDepEdge(
                src_task_id="t0",
                dst_task_id="t2",
                deps=[
                    TaskBufferDep("h0", sp.value_to_buffer["h0"], "h0", sp.value_to_buffer["h0"]),
                ],
            ),
            TaskDepEdge(
                src_task_id="t1",
                dst_task_id="t3",
                deps=[
                    TaskBufferDep("a", sp.value_to_buffer["a"], "a", sp.value_to_buffer["a"]),
                ],
            ),
            TaskDepEdge(
                src_task_id="t2",
                dst_task_id="t3",
                deps=[
                    TaskBufferDep("b", sp.value_to_buffer["b"], "b", sp.value_to_buffer["b"]),
                ],
            ),
        ],
    )
    module = BFTaskModule(
        tasks=[t0, t1, t2, t3],
        entry_task_id="t0",
        bindings={"params": {}},
        storage_plan=sp,
        task_graph=g,
    )
    t0.input_buffers = [sp.value_to_buffer["input"]]
    t0.output_buffers = [sp.value_to_buffer["h0"]]
    t1.input_buffers = [sp.value_to_buffer["h0"]]
    t1.output_buffers = [sp.value_to_buffer["a"]]
    t2.input_buffers = [sp.value_to_buffer["h0"]]
    t2.output_buffers = [sp.value_to_buffer["b"]]
    t3.input_buffers = [sp.value_to_buffer["a"], sp.value_to_buffer["b"]]
    t3.output_buffers = [sp.value_to_buffer["out"]]

    single = BFTaskModule(
        tasks=[
            BoundTask(
                task_id="t0",
                kind=TaskKind.INTERVAL_IBP,
                ops=[
                    TaskOp(op_type="relu", name="relu0", inputs=["input"], outputs=["h0"], attrs={}),
                    TaskOp(op_type="relu", name="relu1", inputs=["h0"], outputs=["a"], attrs={}),
                    TaskOp(op_type="relu", name="relu2", inputs=["h0"], outputs=["b"], attrs={}),
                    TaskOp(op_type="add", name="add0", inputs=["a", "b"], outputs=["out"], attrs={}),
                ],
                input_values=["input"],
                output_values=["out"],
                params=[],
                batch_axes={},
                memory_plan={},
                lowering=TaskLowering.TVM_TIR,
            )
        ],
        entry_task_id="t0",
        bindings={"params": {}},
        storage_plan=sp,
    )
    single.tasks[0].input_buffers = [sp.value_to_buffer["input"]]
    single.tasks[0].output_buffers = [sp.value_to_buffer["out"]]

    torch.manual_seed(0)
    x0 = torch.randn(4, 8)
    eps = 0.1
    spec = LinfInputSpec(value_name="input", center=x0, eps=eps)
    out_sched = run_ibp_scheduled(module, spec, output_value="out")
    out_single = PythonTaskExecutor().run_ibp(single, spec, output_value="out")
    assert torch.allclose(out_sched.lower, out_single.lower)
    assert torch.allclose(out_sched.upper, out_single.upper)
