import pytest
import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner.interval_v2 import IntervalV2PartitionConfig, plan_interval_ibp_v2
from boundflow.runtime.scheduler import run_ibp_scheduled
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor
from boundflow.runtime.tvm_executor import TVMExecutorOptions, TVMTaskExecutor


@pytest.mark.parametrize("kernel_style", ["relax", "call_tir"])
def test_pr9_tvm_executor_run_ibp_task_matches_python_on_single_linear(kernel_style: str):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 8))
    x0 = torch.randn(4, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)

    module = plan_interval_ibp_v2(program, config=IntervalV2PartitionConfig(min_tasks=1))
    assert module.task_graph is not None  # ensure scheduled path is used

    input_spec = LinfInputSpec(value_name=program.graph.inputs[0], center=x0, eps=0.1)

    py = run_ibp_scheduled(module, input_spec, executor=PythonTaskExecutor())
    tvm_exec = TVMTaskExecutor(options=TVMExecutorOptions(target="llvm", kernel_style=kernel_style))
    tvm = run_ibp_scheduled(module, input_spec, executor=tvm_exec)

    assert torch.allclose(py.lower, tvm.lower, atol=1e-5, rtol=1e-5)
    assert torch.allclose(py.upper, tvm.upper, atol=1e-5, rtol=1e-5)

