import pytest
import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.runtime import LinfInputSpec, PythonInterpreter


def test_ibp_matches_auto_lirpa_on_mlp():
    auto_LiRPA = pytest.importorskip("auto_LiRPA")
    BoundedModule = auto_LiRPA.BoundedModule
    BoundedTensor = auto_LiRPA.BoundedTensor
    PerturbationLpNorm = pytest.importorskip("auto_LiRPA.perturbations").PerturbationLpNorm

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    x0 = torch.randn(2, 4)
    eps = 0.1

    # BoundFlow (reference) IBP
    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    bf_out = PythonInterpreter().run_ibp(program, LinfInputSpec(value_name="input", center=x0, eps=eps))

    # auto_LiRPA IBP ground truth
    lirpa_model = BoundedModule(model, torch.empty_like(x0), device=x0.device)
    ptb = PerturbationLpNorm(norm=float("inf"), eps=eps)
    bounded_x = BoundedTensor(x0, ptb)
    lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method="IBP")

    assert torch.allclose(bf_out.lower, lb, rtol=1e-5, atol=1e-6)
    assert torch.allclose(bf_out.upper, ub, rtol=1e-5, atol=1e-6)

