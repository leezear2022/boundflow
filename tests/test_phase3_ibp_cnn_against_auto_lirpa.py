import pytest
import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.runtime import LinfInputSpec, PythonInterpreter


def test_ibp_matches_auto_lirpa_on_mnist_cnn():
    auto_LiRPA = pytest.importorskip("auto_LiRPA")
    BoundedModule = auto_LiRPA.BoundedModule
    BoundedTensor = auto_LiRPA.BoundedTensor
    PerturbationLpNorm = pytest.importorskip("auto_LiRPA.perturbations").PerturbationLpNorm
    Flatten = pytest.importorskip("auto_LiRPA.utils").Flatten

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

    # BoundFlow IBP
    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    bf_out = PythonInterpreter().run_ibp(program, LinfInputSpec(value_name="input", center=x0, eps=eps))

    # auto_LiRPA IBP ground truth
    lirpa_model = BoundedModule(model, torch.empty_like(x0), device=x0.device)
    ptb = PerturbationLpNorm(norm=float("inf"), eps=eps)
    bounded_x = BoundedTensor(x0, ptb)
    lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method="IBP")

    assert torch.allclose(bf_out.lower, lb, rtol=1e-5, atol=1e-6)
    assert torch.allclose(bf_out.upper, ub, rtol=1e-5, atol=1e-6)

