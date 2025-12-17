import tempfile

import pytest
import torch
import torch.nn as nn

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor


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
    return PythonTaskExecutor().run_ibp(task_module, LinfInputSpec(value_name="input", center=x0, eps=eps))


def test_onnx_import_matches_torch_import_on_mlp_ibp():
    pytest.importorskip("onnx")

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    x0 = torch.randn(4, 16)
    eps = 0.05

    program_torch = import_torch(model, (x0,), export_mode="export", normalize=True)
    onnx_model = _export_to_onnx(model, x0)
    program_onnx = import_onnx(onnx_model, do_shape_infer=True, normalize=True)

    out_torch = _run_ibp_from_program(program_torch, x0, eps)
    out_onnx = _run_ibp_from_program(program_onnx, x0, eps)

    assert torch.allclose(out_onnx.lower, out_torch.lower, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out_onnx.upper, out_torch.upper, rtol=1e-5, atol=1e-6)


def test_onnx_import_matches_torch_import_on_mnist_cnn_ibp():
    onnx = pytest.importorskip("onnx")
    auto_LiRPA = pytest.importorskip("auto_LiRPA")
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

    program_torch = import_torch(model, (x0,), export_mode="export", normalize=True)
    onnx_model = _export_to_onnx(model, x0)
    # sanity: ensure ONNX model loads
    assert isinstance(onnx_model, onnx.ModelProto)

    program_onnx = import_onnx(onnx_model, do_shape_infer=True, normalize=True)

    out_torch = _run_ibp_from_program(program_torch, x0, eps)
    out_onnx = _run_ibp_from_program(program_onnx, x0, eps)

    assert torch.allclose(out_onnx.lower, out_torch.lower, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out_onnx.upper, out_torch.upper, rtol=1e-5, atol=1e-6)

