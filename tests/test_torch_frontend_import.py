import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.ir.primal import ValueKind


def test_import_torch_export_mlp():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    program = import_torch(model, (torch.randn(5, 4),), export_mode="export", normalize=True)

    assert len(program.graph.nodes) > 0
    assert program.graph.inputs == ["input"]
    assert len(program.graph.outputs) == 1
    program.graph.validate()

    op_types = [n.op_type for n in program.graph.nodes]
    assert "linear" in op_types
    assert "relu" in op_types

    assert program.graph.values["input"].kind == ValueKind.INPUT
    assert all(program.graph.values[name].kind == ValueKind.PARAM for name in program.params.keys())
    assert len(program.params) > 0

