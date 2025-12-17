import pytest

from boundflow.ir.primal import BFPrimalGraph, Node, TensorType, Value


def test_primal_graph_validate_ok():
    graph = BFPrimalGraph(
        nodes=[Node(op_type="relu", name="relu1", inputs=["x"], outputs=["y"])],
        inputs=["x"],
        outputs=["y"],
        values={
            "x": Value(name="x", type=TensorType(shape=[1, 3], dtype="float32")),
            "y": Value(name="y", type=TensorType(shape=[1, 3], dtype="float32")),
        },
    )
    graph.validate()
    assert "relu1" in graph.node_map


def test_primal_graph_validate_missing_value_meta_raises():
    graph = BFPrimalGraph(
        nodes=[Node(op_type="relu", name="relu1", inputs=["x"], outputs=["y"])],
        inputs=["x"],
        outputs=["y"],
        values={
            "x": Value(name="x", type=TensorType(shape=[1, 3], dtype="float32")),
        },
    )
    with pytest.raises(ValueError, match="missing meta"):
        graph.validate()
