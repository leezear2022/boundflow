import pytest
import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.ir.spec import LinearSpec
from boundflow.planner import plan_interval_ibp_with_linear_spec
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor


def test_task_pipeline_matches_auto_lirpa_with_C_matrix():
    auto_LiRPA = pytest.importorskip("auto_LiRPA")
    BoundedModule = auto_LiRPA.BoundedModule
    BoundedTensor = auto_LiRPA.BoundedTensor
    PerturbationLpNorm = pytest.importorskip("auto_LiRPA.perturbations").PerturbationLpNorm

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 5))
    x0 = torch.randn(2, 4)
    eps = 0.1

    # Random linear property C: [B, S, O]
    torch.manual_seed(1)
    C = torch.randn(x0.shape[0], 3, 5)

    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    task_module = plan_interval_ibp_with_linear_spec(program, LinearSpec(C=C))
    bf_out = PythonTaskExecutor().run_ibp(task_module, LinfInputSpec(value_name="input", center=x0, eps=eps))

    lirpa_model = BoundedModule(model, torch.empty_like(x0), device=x0.device)
    ptb = PerturbationLpNorm(norm=float("inf"), eps=eps)
    bounded_x = BoundedTensor(x0, ptb)
    lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method="IBP", C=C)

    assert bf_out.lower.shape == lb.shape
    assert bf_out.upper.shape == ub.shape
    assert torch.allclose(bf_out.lower, lb, rtol=1e-5, atol=1e-6)
    assert torch.allclose(bf_out.upper, ub, rtol=1e-5, atol=1e-6)

