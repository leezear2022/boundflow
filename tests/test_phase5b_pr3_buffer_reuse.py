import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.ir.liveness import buffer_size_bytes
from boundflow.ir.task import BFTaskModule, BoundTask, BufferSpec, StoragePlan, TaskKind, TaskLowering, TaskOp
from boundflow.ir.task_graph import TaskBufferDep, TaskDepEdge, TaskGraph
from boundflow.planner import plan_interval_ibp_v0, plan_interval_ibp_v2
from boundflow.planner.interval_v2 import IntervalV2PartitionConfig
from boundflow.planner.passes import apply_conservative_buffer_reuse
from boundflow.runtime.scheduler import run_ibp_scheduled
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor


def test_phase5b_pr3_reuse_matches_v0_on_mlp():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    x0 = torch.randn(4, 16)
    eps = 0.05

    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    m0 = plan_interval_ibp_v0(program)
    out0 = PythonTaskExecutor().run_ibp(m0, LinfInputSpec(value_name="input", center=x0, eps=eps))
    out_name = m0.get_entry_task().output_values[0]

    m2 = plan_interval_ibp_v2(program, config=IntervalV2PartitionConfig(min_tasks=2, enable_storage_reuse=True))
    assert m2.task_graph is not None
    assert len(m2.tasks) >= 2
    assert m2.storage_plan.num_physical_buffers() <= m2.storage_plan.num_logical_buffers()

    out2 = run_ibp_scheduled(
        m2, LinfInputSpec(value_name="input", center=x0, eps=eps), executor=PythonTaskExecutor(), output_value=out_name
    )
    assert torch.allclose(out2.lower, out0.lower, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out2.upper, out0.upper, rtol=1e-5, atol=1e-6)


def _storage_plan_for_values(values, *, shape=(4, 8), dtype="float32"):
    buffers = {}
    v2b = {}
    for v in values:
        bid = f"buf_{v}"
        buffers[bid] = BufferSpec(buffer_id=bid, dtype=dtype, shape=list(shape), scope="global")
        v2b[v] = bid
    return StoragePlan(buffers=buffers, value_to_buffer=v2b)


def _make_chain_module(*, num_tasks: int = 6, num_internal: int = 4) -> BFTaskModule:
    if num_tasks < 2:
        raise ValueError("num_tasks must be >= 2")
    if num_internal < 1:
        raise ValueError("num_internal must be >= 1")

    values = ["input"]
    for i in range(num_tasks):
        for j in range(num_internal):
            values.append(f"t{i}_tmp{j}")
        values.append(f"h{i}")
    sp = _storage_plan_for_values(values)

    tasks = []
    edges = []
    for i in range(num_tasks):
        in_v = "input" if i == 0 else f"h{i-1}"
        out_v = f"h{i}"
        ops = []
        prev = in_v
        for j in range(num_internal):
            tmp = f"t{i}_tmp{j}"
            ops.append(TaskOp(op_type="relu", name=f"relu_{i}_{j}", inputs=[prev], outputs=[tmp], attrs={}))
            prev = tmp
        ops.append(TaskOp(op_type="relu", name=f"relu_{i}_out", inputs=[prev], outputs=[out_v], attrs={}))

        t = BoundTask(
            task_id=f"t{i}",
            kind=TaskKind.INTERVAL_IBP,
            ops=ops,
            input_values=[in_v],
            output_values=[out_v],
            params=[],
            batch_axes={},
            memory_plan={},
            lowering=TaskLowering.TVM_TIR,
        )
        t.input_buffers = [sp.value_to_buffer[in_v]]
        t.output_buffers = [sp.value_to_buffer[out_v]]
        tasks.append(t)

        if i > 0:
            edges.append(
                TaskDepEdge(
                    src_task_id=f"t{i-1}",
                    dst_task_id=f"t{i}",
                    deps=[
                        TaskBufferDep(
                            src_value=in_v,
                            src_buffer_id=sp.value_to_buffer[in_v],
                            dst_value=in_v,
                            dst_buffer_id=sp.value_to_buffer[in_v],
                        )
                    ],
                )
            )

    module = BFTaskModule(
        tasks=tasks,
        entry_task_id="t0",
        bindings={"params": {}},
        storage_plan=sp,
        task_graph=TaskGraph(task_ids=[t.task_id for t in tasks], edges=edges),
    )
    module.validate()
    return module


def test_phase5b_pr3_reuse_reduces_physical_buffers_and_preserves_output():
    torch.manual_seed(0)
    x0 = torch.randn(4, 8)
    eps = 0.1
    spec = LinfInputSpec(value_name="input", center=x0, eps=eps)

    module = _make_chain_module(num_tasks=6, num_internal=4)
    out_before = run_ibp_scheduled(module, spec, executor=PythonTaskExecutor(), output_value="h5")

    logical = module.storage_plan.num_logical_buffers()
    physical_before = module.storage_plan.num_physical_buffers()
    assert physical_before == logical
    logical_bytes_before = sum((buffer_size_bytes(s) or 0) for s in module.storage_plan.buffers.values())

    apply_conservative_buffer_reuse(module)
    physical_after = module.storage_plan.num_physical_buffers()
    assert physical_after < physical_before
    physical_bytes_after = sum((buffer_size_bytes(s) or 0) for s in module.storage_plan.physical_buffers.values())
    assert physical_bytes_after < logical_bytes_before

    out_after = run_ibp_scheduled(module, spec, executor=PythonTaskExecutor(), output_value="h5")
    assert torch.allclose(out_after.lower, out_before.lower)
    assert torch.allclose(out_after.upper, out_before.upper)
