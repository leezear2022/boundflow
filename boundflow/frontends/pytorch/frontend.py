from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.fx

from ...frontends.normalize import normalize_primal_graph
from ...ir.primal import (
    BFPrimalGraph,
    BFPrimalProgram,
    Node,
    Source,
    TensorType,
    Value,
    ValueKind,
)


def import_torch(
    module: torch.nn.Module,
    example_inputs: Tuple[Any, ...],
    *,
    strict: bool = False,
    normalize: bool = True,
    export_mode: str = "export",
) -> BFPrimalProgram:
    """
    Import a PyTorch module into BoundFlow Primal IR.

    Args:
        module: The PyTorch nn.Module.
        example_inputs: Example inputs for tracing/export.
        strict: Whether to enforce strict export checks.
        normalize: Whether to run normalization to primitive ops.
        export_mode: "export" (torch.export) or "fx" (torch.fx.symbolic_trace).
    """
    if export_mode not in ("export", "fx"):
        raise ValueError(f"unsupported export_mode: {export_mode}")

    if export_mode == "fx":
        traced = torch.fx.symbolic_trace(module)
        graph = _convert_fx_graph(
            traced,
            user_inputs=[f"arg{i}" for i in range(len(example_inputs))],
            inputs_to_parameters={},
            state_dict=dict(module.state_dict()),
        )
        program = BFPrimalProgram(
            source=Source.TORCH_EXPORT,
            graph=graph,
            params={},
            tensor_meta={"export_mode": "fx"},
        )
        if normalize:
            program.graph = normalize_primal_graph(program.graph)
        return program

    exported = torch.export.export(module, example_inputs, strict=strict)
    graph_signature = exported.graph_signature
    state_dict = dict(exported.state_dict)
    graph = _convert_fx_graph(
        exported.graph_module,
        user_inputs=list(graph_signature.user_inputs),
        inputs_to_parameters=dict(graph_signature.inputs_to_parameters),
        state_dict=state_dict,
    )
    program = BFPrimalProgram(
        source=Source.TORCH_EXPORT,
        graph=graph,
        params=_extract_params(dict(graph_signature.inputs_to_parameters), state_dict),
        tensor_meta={
            "export_mode": "export",
            "graph_signature": _safe_graph_signature_dict(graph_signature),
        },
    )
    if normalize:
        program.graph = normalize_primal_graph(program.graph)
    return program


def _safe_graph_signature_dict(graph_signature: Any) -> Dict[str, Any]:
    try:
        return asdict(graph_signature)  # type: ignore[arg-type]
    except Exception:
        return {
            "inputs_to_parameters": getattr(graph_signature, "inputs_to_parameters", None),
            "inputs_to_buffers": getattr(graph_signature, "inputs_to_buffers", None),
            "user_inputs": getattr(graph_signature, "user_inputs", None),
            "user_outputs": getattr(graph_signature, "user_outputs", None),
        }


def _extract_params(inputs_to_parameters: Dict[str, str], state_dict: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for placeholder_name, target_name in inputs_to_parameters.items():
        if target_name in state_dict:
            params[placeholder_name] = state_dict[target_name]
    return params


def _convert_fx_graph(
    graph_module: torch.fx.GraphModule,
    *,
    user_inputs: List[str],
    inputs_to_parameters: Dict[str, str],
    state_dict: Dict[str, Any],
) -> BFPrimalGraph:
    fx_graph = graph_module.graph
    primal_graph = BFPrimalGraph()

    for fx_node in fx_graph.nodes:
        if fx_node.op == "placeholder":
            kind = ValueKind.INPUT
            if fx_node.name in inputs_to_parameters:
                kind = ValueKind.PARAM
            elif fx_node.name not in user_inputs:
                kind = ValueKind.INTERMEDIATE

            primal_graph.values[fx_node.name] = Value(
                name=fx_node.name,
                type=_tensor_type_from_fx_meta(fx_node),
                kind=kind,
                debug_name=str(fx_node.target),
            )
            continue

        if fx_node.op == "get_attr":
            const_value = _get_attr_value(graph_module, fx_node.target)
            primal_graph.values[fx_node.name] = Value(
                name=fx_node.name,
                type=_tensor_type_from_fx_meta(fx_node, fallback_value=const_value),
                kind=ValueKind.CONST,
                debug_name=str(fx_node.target),
            )
            state_dict[fx_node.name] = const_value
            continue

        if fx_node.op == "call_function":
            op_type = _map_torch_target_to_primitive(fx_node.target)
            inputs = _collect_fx_value_uses(fx_node.args) + _collect_fx_value_uses(fx_node.kwargs)
            attrs = _extract_attrs_for_call_function(op_type, fx_node)
            primal_graph.nodes.append(
                Node(
                    op_type=op_type,
                    name=fx_node.name,
                    inputs=inputs,
                    outputs=[fx_node.name],
                    attrs=attrs,
                )
            )
            primal_graph.values[fx_node.name] = Value(
                name=fx_node.name,
                type=_tensor_type_from_fx_meta(fx_node),
                kind=ValueKind.INTERMEDIATE,
                debug_name=_debug_target_name(fx_node.target),
            )
            continue

        if fx_node.op == "call_method":
            op_type = f"call_method::{fx_node.target}"
            inputs = _collect_fx_value_uses(fx_node.args) + _collect_fx_value_uses(fx_node.kwargs)
            primal_graph.nodes.append(
                Node(
                    op_type=op_type,
                    name=fx_node.name,
                    inputs=inputs,
                    outputs=[fx_node.name],
                    attrs={},
                )
            )
            primal_graph.values[fx_node.name] = Value(
                name=fx_node.name,
                type=_tensor_type_from_fx_meta(fx_node),
                kind=ValueKind.INTERMEDIATE,
                debug_name=str(fx_node.target),
            )
            continue

        if fx_node.op == "output":
            primal_graph.outputs = _collect_fx_value_uses(fx_node.args)
            continue

        raise NotImplementedError(f"unsupported fx node op: {fx_node.op}")

    primal_graph.inputs = list(user_inputs)
    primal_graph.validate()
    return primal_graph


def _get_attr_value(graph_module: torch.fx.GraphModule, target: Any) -> Any:
    value: Any = graph_module
    for atom in str(target).split("."):
        value = getattr(value, atom)
    return value


def _tensor_type_from_fx_meta(fx_node: torch.fx.Node, fallback_value: Optional[Any] = None) -> TensorType:
    tensor_meta = fx_node.meta.get("tensor_meta")
    if tensor_meta is not None:
        shape = [int(d) if d is not None else None for d in list(tensor_meta.shape)]
        dtype = str(tensor_meta.dtype).replace("torch.", "")
        return TensorType(shape=shape, dtype=dtype)

    if fallback_value is not None and hasattr(fallback_value, "shape") and hasattr(fallback_value, "dtype"):
        shape = [int(d) if d is not None else None for d in list(fallback_value.shape)]
        dtype = str(fallback_value.dtype).replace("torch.", "")
        return TensorType(shape=shape, dtype=dtype)

    return TensorType(shape=[], dtype="unknown")


def _collect_fx_value_uses(obj: Any) -> List[str]:
    results: List[str] = []

    def visit(x: Any) -> None:
        if isinstance(x, torch.fx.Node):
            results.append(x.name)
            return
        if isinstance(x, (list, tuple)):
            for item in x:
                visit(item)
            return
        if isinstance(x, dict):
            for item in x.values():
                visit(item)
            return

    visit(obj)
    return results


def _debug_target_name(target: Any) -> str:
    try:
        return str(target)
    except Exception:
        return getattr(target, "__name__", "<unknown>")


def _map_torch_target_to_primitive(target: Any) -> str:
    name = _debug_target_name(target)
    mapping = {
        "aten.linear.default": "linear",
        "aten.relu.default": "relu",
        "aten.add.Tensor": "add",
        "aten.conv2d.default": "conv2d",
        "aten.matmul.default": "matmul",
        "aten.flatten.using_ints": "flatten",
        "aten.reshape.default": "reshape",
        "aten.view.default": "reshape",
        "aten.permute.default": "transpose",
    }
    return mapping.get(name, name)


def _extract_attrs_for_call_function(op_type: str, fx_node: torch.fx.Node) -> Dict[str, Any]:
    """
    从 call_function 节点的 args/kwargs 提取必要的 attrs（常量参数）。
    v0.1 先覆盖 conv2d/flatten，其它算子保持空 attrs。
    """
    if op_type == "conv2d":
        # aten.conv2d.default(input, weight, bias, stride, padding, dilation?, groups?)
        args = list(fx_node.args)
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }

    if op_type == "flatten":
        # aten.flatten.using_ints(input, start_dim, end_dim?)
        args = list(fx_node.args)
        start_dim = int(args[1]) if len(args) > 1 else 0
        end_dim = int(args[2]) if len(args) > 2 else -1
        return {"start_dim": start_dim, "end_dim": end_dim}

    if op_type == "transpose":
        # aten.permute.default(input, dims)
        args = list(fx_node.args)
        dims = args[1] if len(args) > 1 else None
        if not isinstance(dims, (list, tuple)):
            raise ValueError(f"permute dims must be list/tuple, got {type(dims)}: {dims}")
        return {"dims": [int(d) for d in dims]}

    return {}
