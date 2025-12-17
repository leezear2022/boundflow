from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Source(Enum):
    TORCH_EXPORT = "torch_export"
    ONNX = "onnx"


class ValueKind(Enum):
    INPUT = "input"
    PARAM = "param"
    CONST = "const"
    INTERMEDIATE = "intermediate"


@dataclass(frozen=True)
class TensorType:
    shape: List[Optional[int]]
    dtype: str
    layout: Optional[str] = None


@dataclass
class Value:
    """
    Value 表示数据边（tensor/SSA value），用来承载 shape/dtype 等元信息。
    `name` 是图内唯一标识，Node 的 inputs/outputs 引用该 name。
    """

    name: str
    type: TensorType
    kind: ValueKind = ValueKind.INTERMEDIATE
    debug_name: Optional[str] = None


@dataclass
class Node:
    """
    Node 表示算子/计算（operation）。
    inputs/outputs 均为 value 名称，而不是 node 名称。
    """

    op_type: str
    name: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BFPrimalGraph:
    nodes: List[Node] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)  # input value names
    outputs: List[str] = field(default_factory=list)  # output value names
    values: Dict[str, Value] = field(default_factory=dict)

    node_map: Dict[str, Node] = field(default_factory=dict, init=False)

    def rebuild_index(self) -> None:
        self.node_map = {}
        for node in self.nodes:
            if node.name in self.node_map:
                raise ValueError(f"duplicate node name: {node.name}")
            self.node_map[node.name] = node

    def validate(self) -> None:
        if not self.node_map or len(self.node_map) != len(self.nodes):
            self.rebuild_index()

        if len(set(self.values.keys())) != len(self.values):
            raise ValueError("duplicate value name in values dict")

        for value_name in self.inputs + self.outputs:
            if value_name not in self.values:
                raise ValueError(f"graph input/output value missing meta: {value_name}")

        for node in self.nodes:
            for value_name in node.inputs + node.outputs:
                if value_name not in self.values:
                    raise ValueError(f"node '{node.name}' references unknown value: {value_name}")


@dataclass
class BFPrimalProgram:
    source: Source
    graph: BFPrimalGraph
    params: Dict[str, Any]  # param/const tensors; keys are value names
    tensor_meta: Dict[str, Any]  # reserved for frontend-specific meta
    debug_map: Dict[str, str] = field(default_factory=dict)  # id -> source location
