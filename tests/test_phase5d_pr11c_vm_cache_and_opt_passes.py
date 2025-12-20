import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner.interval_v2 import IntervalV2PartitionConfig, plan_interval_ibp_v2
from boundflow.runtime.scheduler import run_ibp_scheduled
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor
from boundflow.runtime.tvm_executor import TVMExecutorOptions, TVMTaskExecutor


def test_pr11c_vm_cache_and_opt_passes_do_not_change_semantics():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
    x0 = torch.randn(4, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    module = plan_interval_ibp_v2(program, config=IntervalV2PartitionConfig(min_tasks=1))

    input_spec = LinfInputSpec(value_name=program.graph.inputs[0], center=x0, eps=0.1)
    py = run_ibp_scheduled(module, input_spec, executor=PythonTaskExecutor())

    tvm_exec = TVMTaskExecutor(
        options=TVMExecutorOptions(
            target="llvm",
            kernel_style="relax",
            enable_task_relax_ops=True,
            enable_task_fusion_pipeline=True,
            task_fuse_opt_level=2,
            enable_vm_cache=True,
            enable_vm_packed_func_cache=True,
            task_vm_opt_passes=("ExpandTupleArguments", "RemoveUnusedParameters", "InlinePrivateFunctions", "CallTIRRewrite"),
        )
    )

    # Run twice to exercise VM caching.
    tvm1 = run_ibp_scheduled(module, input_spec, executor=tvm_exec)
    tvm2 = run_ibp_scheduled(module, input_spec, executor=tvm_exec)

    assert torch.allclose(py.lower, tvm1.lower, atol=1e-5, rtol=1e-5)
    assert torch.allclose(py.upper, tvm1.upper, atol=1e-5, rtol=1e-5)
    assert torch.allclose(tvm1.lower, tvm2.lower, atol=1e-6, rtol=1e-6)
    assert torch.allclose(tvm1.upper, tvm2.upper, atol=1e-6, rtol=1e-6)

