import pytest
import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner.interval_v2 import IntervalV2PartitionConfig
from boundflow.planner import plan_interval_ibp_v0, plan_interval_ibp_v2
from boundflow.runtime.scheduler import run_ibp_scheduled
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor
from boundflow.ir.primal import BFPrimalGraph, BFPrimalProgram, Node, Source, TensorType, Value, ValueKind


def test_planner_v2_multitask_matches_v0_single_task_on_mlp():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    x0 = torch.randn(4, 16)
    eps = 0.05

    program = import_torch(model, (x0,), export_mode="export", normalize=True)

    m0 = plan_interval_ibp_v0(program)
    out0 = PythonTaskExecutor().run_ibp(m0, LinfInputSpec(value_name="input", center=x0, eps=eps))
    out_name = m0.get_entry_task().output_values[0]

    # v2 默认不强制切分；这里显式要求 min_tasks>=2 以覆盖 multi-task 行为。
    m2 = plan_interval_ibp_v2(program, config=IntervalV2PartitionConfig(min_tasks=2))
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

    m2 = plan_interval_ibp_v2(program, config=IntervalV2PartitionConfig(min_tasks=2))
    assert m2.task_graph is not None
    assert len(m2.tasks) >= 2
    out2 = run_ibp_scheduled(
        m2, LinfInputSpec(value_name="input", center=x0, eps=eps), executor=PythonTaskExecutor(), output_value=out_name
    )

    assert torch.allclose(out2.lower, out0.lower, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out2.upper, out0.upper, rtol=1e-5, atol=1e-6)


def test_planner_v2_handles_branch_merge_graph():
    # Manual primal graph with a branch+merge pattern:
    # a = relu(x)
    # b = relu(x)
    # out = add(a, b)
    graph = BFPrimalGraph()
    graph.inputs = ["x"]
    graph.outputs = ["out"]
    graph.values["x"] = Value(name="x", type=TensorType(shape=[4, 8], dtype="float32"), kind=ValueKind.INPUT)
    graph.values["a"] = Value(name="a", type=TensorType(shape=[4, 8], dtype="float32"))
    graph.values["b"] = Value(name="b", type=TensorType(shape=[4, 8], dtype="float32"))
    graph.values["out"] = Value(name="out", type=TensorType(shape=[4, 8], dtype="float32"))
    graph.nodes = [
        Node(op_type="relu", name="relu_a", inputs=["x"], outputs=["a"], attrs={}),
        Node(op_type="relu", name="relu_b", inputs=["x"], outputs=["b"], attrs={}),
        Node(op_type="add", name="add0", inputs=["a", "b"], outputs=["out"], attrs={}),
    ]
    program = BFPrimalProgram(source=Source.TORCH_EXPORT, graph=graph, params={}, tensor_meta={})

    torch.manual_seed(0)
    x0 = torch.randn(4, 8)
    eps = 0.1
    spec = LinfInputSpec(value_name="x", center=x0, eps=eps)

    m0 = plan_interval_ibp_v0(program)
    out0 = PythonTaskExecutor().run_ibp(m0, spec, output_value="out")

    m2 = plan_interval_ibp_v2(program, config=IntervalV2PartitionConfig(min_tasks=2))
    assert m2.task_graph is not None
    assert len(m2.tasks) >= 2
    out2 = run_ibp_scheduled(m2, spec, executor=PythonTaskExecutor(), output_value="out")

    assert torch.allclose(out2.lower, out0.lower)
    assert torch.allclose(out2.upper, out0.upper)
