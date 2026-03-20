import tempfile

import pytest
import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.task_executor import InputSpec, LinfInputSpec, PythonTaskExecutor


def _export_to_onnx(model: nn.Module, x0: torch.Tensor) -> "object":
    onnx = pytest.importorskip("onnx")
    model.eval()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    torch.onnx.export(
        model,
        x0,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )
    return onnx.load(path)


def _run_ibp_from_program(program, x0: torch.Tensor, eps: float):
    task_module = plan_interval_ibp_v0(program)
    return PythonTaskExecutor().run_ibp(task_module, LinfInputSpec(value_name=program.graph.inputs[0], center=x0, eps=eps))


def _run_crown_from_program(program, x0: torch.Tensor, eps: float):
    task_module = plan_interval_ibp_v0(program)
    return run_crown_ibp_mlp(task_module, InputSpec.linf(value_name=program.graph.inputs[0], center=x0, eps=eps))


class ResidualAddNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Conv2d(1, 2, 1)
        self.skip = nn.Conv2d(1, 2, 1)
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = torch.relu(self.main(x))
        skip = self.skip(x)
        out = torch.relu(main + skip)
        return self.head(out.flatten(1))


class ConcatNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Conv2d(1, 2, 1)
        self.right = nn.Conv2d(1, 1, 1)
        self.head = nn.Linear(12, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = torch.relu(self.left(x))
        right = torch.relu(self.right(x))
        cat = torch.cat([left, right], dim=1)
        return self.head(cat.flatten(1))


def test_phase7a_pr8_torch_and_onnx_frontends_match_on_residual_add_ibp_and_crown() -> None:
    pytest.importorskip("onnx")
    from boundflow.frontends.onnx.frontend import import_onnx

    torch.manual_seed(0)
    model = ResidualAddNet().eval()
    x0 = torch.randn(2, 1, 2, 2)
    eps = 0.05

    program_torch = import_torch(model, (x0,), export_mode="export", normalize=True)
    onnx_model = _export_to_onnx(model, x0)
    program_onnx = import_onnx(onnx_model, do_shape_infer=True, normalize=True)

    out_torch = _run_ibp_from_program(program_torch, x0, eps)
    out_onnx = _run_ibp_from_program(program_onnx, x0, eps)
    crown_torch = _run_crown_from_program(program_torch, x0, eps)
    crown_onnx = _run_crown_from_program(program_onnx, x0, eps)

    assert torch.allclose(out_onnx.lower, out_torch.lower, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out_onnx.upper, out_torch.upper, rtol=1e-5, atol=1e-6)
    assert torch.allclose(crown_onnx.lower, crown_torch.lower, rtol=1e-5, atol=1e-6)
    assert torch.allclose(crown_onnx.upper, crown_torch.upper, rtol=1e-5, atol=1e-6)


def test_phase7a_pr8_torch_and_onnx_frontends_match_on_concat_ibp_and_crown() -> None:
    pytest.importorskip("onnx")
    from boundflow.frontends.onnx.frontend import import_onnx

    torch.manual_seed(0)
    model = ConcatNet().eval()
    x0 = torch.randn(2, 1, 2, 2)
    eps = 0.05

    program_torch = import_torch(model, (x0,), export_mode="export", normalize=True)
    onnx_model = _export_to_onnx(model, x0)
    program_onnx = import_onnx(onnx_model, do_shape_infer=True, normalize=True)

    out_torch = _run_ibp_from_program(program_torch, x0, eps)
    out_onnx = _run_ibp_from_program(program_onnx, x0, eps)
    crown_torch = _run_crown_from_program(program_torch, x0, eps)
    crown_onnx = _run_crown_from_program(program_onnx, x0, eps)

    assert torch.allclose(out_onnx.lower, out_torch.lower, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out_onnx.upper, out_torch.upper, rtol=1e-5, atol=1e-6)
    assert torch.allclose(crown_onnx.lower, crown_torch.lower, rtol=1e-5, atol=1e-6)
    assert torch.allclose(crown_onnx.upper, crown_torch.upper, rtol=1e-5, atol=1e-6)
