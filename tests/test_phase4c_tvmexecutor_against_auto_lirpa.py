import pytest
import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor


def test_auto_lirpa_python_tvm_all_match_on_mlp_ibp():
    tvm = pytest.importorskip("tvm")
    auto_LiRPA = pytest.importorskip("auto_LiRPA")

    if not tvm.runtime.enabled("llvm"):
        pytest.skip("tvm llvm backend not enabled")

    from boundflow.runtime.tvm_executor import TVMExecutorOptions, TVMTaskExecutor

    BoundedModule = auto_LiRPA.BoundedModule
    BoundedTensor = auto_LiRPA.BoundedTensor
    PerturbationLpNorm = pytest.importorskip("auto_LiRPA.perturbations").PerturbationLpNorm

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    x0 = torch.randn(4, 16)
    eps = 0.05

    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    task_module = plan_interval_ibp_v0(program)
    input_spec = LinfInputSpec(value_name="input", center=x0, eps=eps)

    out_py = PythonTaskExecutor().run_ibp(task_module, input_spec)
    out_tvm = TVMTaskExecutor(options=TVMExecutorOptions(target="llvm")).run_ibp(task_module, input_spec)

    lirpa_model = BoundedModule(model, torch.empty_like(x0), device=x0.device)
    bounded_x = BoundedTensor(x0, PerturbationLpNorm(norm=float("inf"), eps=eps))
    lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method="IBP")

    # PythonTaskExecutor (reference) should match auto_LiRPA (ground truth).
    assert torch.allclose(out_py.lower, lb, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out_py.upper, ub, rtol=1e-5, atol=1e-6)

    # TVMTaskExecutor should match PythonTaskExecutor.
    assert torch.allclose(out_tvm.lower, out_py.lower, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out_tvm.upper, out_py.upper, rtol=1e-5, atol=1e-6)

