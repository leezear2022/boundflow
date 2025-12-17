import pytest
import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor


def test_tvm_executor_matches_python_on_mlp():
    tvm = pytest.importorskip("tvm")

    # Require a CPU target at least.
    if not tvm.runtime.enabled("llvm"):
        pytest.skip("tvm llvm backend not enabled")

    from boundflow.runtime.tvm_executor import TVMTaskExecutor

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    x0 = torch.randn(4, 16)
    eps = 0.05

    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    task_module = plan_interval_ibp_v0(program)
    input_spec = LinfInputSpec(value_name="input", center=x0, eps=eps)

    ref = PythonTaskExecutor().run_ibp(task_module, input_spec)
    out = TVMTaskExecutor().run_ibp(task_module, input_spec)

    assert torch.allclose(out.lower, ref.lower, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out.upper, ref.upper, rtol=1e-5, atol=1e-6)

