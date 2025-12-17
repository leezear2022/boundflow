import pytest
import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor


def test_tvm_executor_matches_python_on_mnist_cnn():
    tvm = pytest.importorskip("tvm")
    if not tvm.runtime.enabled("llvm"):
        pytest.skip("tvm llvm backend not enabled")

    auto_LiRPA = pytest.importorskip("auto_LiRPA")
    Flatten = pytest.importorskip("auto_LiRPA.utils").Flatten

    from boundflow.runtime.tvm_executor import TVMExecutorOptions, TVMTaskExecutor

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32 * 7 * 7, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )
    x0 = torch.randn(2, 1, 28, 28)
    eps = 0.3

    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    task_module = plan_interval_ibp_v0(program)
    input_spec = LinfInputSpec(value_name="input", center=x0, eps=eps)

    ref = PythonTaskExecutor().run_ibp(task_module, input_spec)
    exe = TVMTaskExecutor(options=TVMExecutorOptions(target="llvm"))
    out = exe.run_ibp(task_module, input_spec)

    assert torch.allclose(out.lower, ref.lower, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out.upper, ref.upper, rtol=1e-5, atol=1e-6)

    assert exe.last_stats is not None
    assert "conv2d" in exe.last_stats.tvm_ops

