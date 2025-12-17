import torch

from boundflow.ir.primal import BFPrimalGraph, BFPrimalProgram, Node, Source, TensorType, Value, ValueKind
from boundflow.planner import plan_interval_ibp_v0


def test_planner_simplifies_consecutive_inverse_permutes():
    graph = BFPrimalGraph()
    graph.values["x"] = Value(name="x", type=TensorType(shape=[2, 3, 4], dtype="float32"), kind=ValueKind.INPUT)
    graph.values["p1"] = Value(name="p1", type=TensorType(shape=[2, 4, 3], dtype="float32"))
    graph.values["p2"] = Value(name="p2", type=TensorType(shape=[2, 3, 4], dtype="float32"))
    graph.values["y"] = Value(name="y", type=TensorType(shape=[2, 3, 4], dtype="float32"))
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.nodes = [
        Node(op_type="permute", name="perm1", inputs=["x"], outputs=["p1"], attrs={"dims": [0, 2, 1]}),
        Node(op_type="permute", name="perm2", inputs=["p1"], outputs=["p2"], attrs={"dims": [0, 2, 1]}),
        Node(op_type="relu", name="relu0", inputs=["p2"], outputs=["y"], attrs={}),
    ]
    program = BFPrimalProgram(source=Source.TORCH_EXPORT, graph=graph, params={}, tensor_meta={})

    task_module = plan_interval_ibp_v0(program)
    task = task_module.get_entry_task()

    assert all(op.op_type != "permute" for op in task.ops)
    assert task.ops[0].op_type == "relu"
    assert task.ops[0].inputs == ["x"]


def test_planner_eliminates_identity_permute():
    graph = BFPrimalGraph()
    graph.values["x"] = Value(name="x", type=TensorType(shape=[2, 3, 4], dtype="float32"), kind=ValueKind.INPUT)
    graph.values["p"] = Value(name="p", type=TensorType(shape=[2, 3, 4], dtype="float32"))
    graph.values["y"] = Value(name="y", type=TensorType(shape=[2, 3, 4], dtype="float32"))
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.nodes = [
        Node(op_type="permute", name="perm0", inputs=["x"], outputs=["p"], attrs={"dims": [0, 1, 2]}),
        Node(op_type="relu", name="relu0", inputs=["p"], outputs=["y"], attrs={}),
    ]
    program = BFPrimalProgram(source=Source.TORCH_EXPORT, graph=graph, params={}, tensor_meta={})

    task_module = plan_interval_ibp_v0(program)
    task = task_module.get_entry_task()

    assert all(op.op_type != "permute" for op in task.ops)
    assert task.ops[0].op_type == "relu"
    assert task.ops[0].inputs == ["x"]


def test_planner_keeps_non_identity_permute():
    graph = BFPrimalGraph()
    graph.values["x"] = Value(name="x", type=TensorType(shape=[2, 3, 4], dtype="float32"), kind=ValueKind.INPUT)
    graph.values["p"] = Value(name="p", type=TensorType(shape=[2, 4, 3], dtype="float32"))
    graph.values["y"] = Value(name="y", type=TensorType(shape=[2, 4, 3], dtype="float32"))
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.nodes = [
        Node(op_type="permute", name="perm0", inputs=["x"], outputs=["p"], attrs={"dims": [0, 2, 1]}),
        Node(op_type="relu", name="relu0", inputs=["p"], outputs=["y"], attrs={}),
    ]
    program = BFPrimalProgram(source=Source.TORCH_EXPORT, graph=graph, params={}, tensor_meta={})

    task_module = plan_interval_ibp_v0(program)
    task = task_module.get_entry_task()

    assert any(op.op_type == "permute" for op in task.ops)
    perm = next(op for op in task.ops if op.op_type == "permute")
    assert perm.attrs["dims"] == [0, 2, 1]
    assert perm.attrs.get("layout_only") is True

