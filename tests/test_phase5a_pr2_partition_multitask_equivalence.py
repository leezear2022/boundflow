import pytest
import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner import plan_interval_ibp_v0, plan_interval_ibp_v2
from boundflow.runtime.scheduler import run_ibp_scheduled
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor


def test_planner_v2_multitask_matches_v0_single_task_on_mlp():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    x0 = torch.randn(4, 16)
    eps = 0.05

    program = import_torch(model, (x0,), export_mode="export", normalize=True)

    m0 = plan_interval_ibp_v0(program)
    out0 = PythonTaskExecutor().run_ibp(m0, LinfInputSpec(value_name="input", center=x0, eps=eps))
    out_name = m0.get_entry_task().output_values[0]

    m2 = plan_interval_ibp_v2(program)
    assert m2.task_graph is not None
    assert len(m2.tasks) >= 2
    out2 = run_ibp_scheduled(
        m2, LinfInputSpec(value_name="input", center=x0, eps=eps), executor=PythonTaskExecutor(), output_value=out_name
    )

    assert torch.allclose(out2.lower, out0.lower, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out2.upper, out0.upper, rtol=1e-5, atol=1e-6)


def test_planner_v2_multitask_matches_v0_single_task_on_mnist_cnn():
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

    program = import_torch(model, (x0,), export_mode="export", normalize=True)

    m0 = plan_interval_ibp_v0(program)
    out0 = PythonTaskExecutor().run_ibp(m0, LinfInputSpec(value_name="input", center=x0, eps=eps))
    out_name = m0.get_entry_task().output_values[0]

    m2 = plan_interval_ibp_v2(program)
    assert m2.task_graph is not None
    assert len(m2.tasks) >= 2
    out2 = run_ibp_scheduled(
        m2, LinfInputSpec(value_name="input", center=x0, eps=eps), executor=PythonTaskExecutor(), output_value=out_name
    )

    assert torch.allclose(out2.lower, out0.lower, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out2.upper, out0.upper, rtol=1e-5, atol=1e-6)
