import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner.interval_v2 import IntervalV2PartitionConfig, plan_interval_ibp_v2
from boundflow.runtime.scheduler import run_ibp_scheduled
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor
from boundflow.runtime.tvm_executor import TVMExecutorOptions, TVMTaskExecutor


def test_pr11a_tvm_task_relax_ops_matches_python_on_linear_relu():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
    x0 = torch.randn(4, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)

    module = plan_interval_ibp_v2(program, config=IntervalV2PartitionConfig(min_tasks=1))
    assert module.task_graph is not None

    input_spec = LinfInputSpec(value_name=program.graph.inputs[0], center=x0, eps=0.1)
    py = run_ibp_scheduled(module, input_spec, executor=PythonTaskExecutor())

    tvm_exec = TVMTaskExecutor(
        options=TVMExecutorOptions(
            target="llvm",
            kernel_style="relax",
            enable_task_relax_ops=True,
            enable_task_fusion_pipeline=True,
            task_fuse_opt_level=2,
        )
    )
    tvm = run_ibp_scheduled(module, input_spec, executor=tvm_exec)

    assert torch.allclose(py.lower, tvm.lower, atol=1e-5, rtol=1e-5)
    assert torch.allclose(py.upper, tvm.upper, atol=1e-5, rtol=1e-5)

    compile_stats = tvm_exec.get_compile_stats()
    assert compile_stats
    assert any((v or {}).get("kind") == "task_relax_ops" for v in compile_stats.values())
    one = next(iter(compile_stats.values()))
    ir_stats = one.get("ir_stats") or {}
    if "after_legalize" in ir_stats and "after_fuse_tir" in ir_stats:
        assert int(ir_stats["after_fuse_tir"].get("call_tir", 0)) <= int(ir_stats["after_legalize"].get("call_tir", 0))
