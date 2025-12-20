import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner.interval_v2 import IntervalV2PartitionConfig, plan_interval_ibp_v2
from boundflow.runtime.scheduler import run_ibp_scheduled
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor
from boundflow.runtime.tvm_executor import MemoryPlanMode, TVMExecutorOptions, TVMTaskExecutor


def test_pr12_static_plan_modes_build_and_match_python():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
    x0 = torch.randn(4, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)

    module = plan_interval_ibp_v2(program, config=IntervalV2PartitionConfig(min_tasks=1))
    assert module.task_graph is not None

    input_spec = LinfInputSpec(value_name=program.graph.inputs[0], center=x0, eps=0.1)
    py = run_ibp_scheduled(module, input_spec, executor=PythonTaskExecutor())

    for mode in (MemoryPlanMode.DEFAULT, MemoryPlanMode.DISABLE_STATIC_PLAN):
        tvm_exec = TVMTaskExecutor(
            options=TVMExecutorOptions(
                target="llvm",
                kernel_style="relax",
                enable_task_relax_ops=True,
                enable_task_fusion_pipeline=True,
                task_fuse_opt_level=2,
                memory_plan_mode=mode,
            )
        )
        tvm = run_ibp_scheduled(module, input_spec, executor=tvm_exec)

        assert torch.allclose(py.lower, tvm.lower, atol=1e-5, rtol=1e-5)
        assert torch.allclose(py.upper, tvm.upper, atol=1e-5, rtol=1e-5)

        compile_stats = tvm_exec.get_compile_stats()
        assert compile_stats
        one = next(iter(compile_stats.values()))
        assert one.get("memory_plan_mode") in ("default", "disable_static_plan")
        mem = one.get("memory_stats") or {}
        assert "alloc_storage" in mem
        assert "alloc_storage_total_bytes" in mem
