from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from onnx import numpy_helper

import torch

from ..normalize import normalize_primal_graph
from ...ir.primal import BFPrimalGraph, BFPrimalProgram, Node, Source, TensorType, Value, ValueKind

def import_onnx(
    model_or_path: Union[str, onnx.ModelProto],
    *,
    do_shape_infer: bool = True,
    input_shapes: Optional[List[List[int]]] = None,
    normalize: bool = True
) -> BFPrimalProgram:
    """
    Import an ONNX model into BoundFlow Primal IR.
    
    Args:
        model_or_path: Path to .onnx file or loaded ModelProto.
        do_shape_infer: Whether to run onnx.shape_inference.
        input_shapes: Optional manual shape override.
        normalize: Whether to run normalization to primitive ops.
    """
    if isinstance(model_or_path, str):
        model = onnx.load(model_or_path)
        model_path = model_or_path
    else:
        model = model_or_path
        model_path = None

    if do_shape_infer:
        model = onnx.shape_inference.infer_shapes(model)

    graph = _convert_onnx_graph(model, input_shapes=input_shapes)
    program = BFPrimalProgram(
        source=Source.ONNX,
        graph=graph.graph,
        params=graph.params,
        tensor_meta={
            "onnx_path": model_path,
            "ir_version": getattr(model, "ir_version", None),
            "opset_import": [(op.domain, int(op.version)) for op in getattr(model, "opset_import", [])],
        },
    )
    if normalize:
        program.graph = normalize_primal_graph(program.graph)
    return program


@dataclass
class _Converted:
    graph: BFPrimalGraph
    params: Dict[str, Any]


def _convert_onnx_graph(model: onnx.ModelProto, *, input_shapes: Optional[List[List[int]]]) -> _Converted:
    onnx_graph = model.graph
    primal = BFPrimalGraph()
    params: Dict[str, Any] = {}

    type_map = _collect_value_types(onnx_graph)

    initializer_arrays: Dict[str, np.ndarray] = {}
    for init in onnx_graph.initializer:
        arr = numpy_helper.to_array(init)
        initializer_arrays[init.name] = arr

    # Optional manual shape override (only for user inputs).
    if input_shapes is not None:
        user_inputs = [i for i in onnx_graph.input if i.name not in initializer_arrays]
        if len(input_shapes) != len(user_inputs):
            raise ValueError(f"input_shapes has {len(input_shapes)} entries, but model has {len(user_inputs)} user inputs")
        for vi, shape in zip(user_inputs, input_shapes):
            old = type_map.get(vi.name)
            dtype = old.dtype if old is not None else "float32"
            type_map[vi.name] = TensorType(shape=[int(d) for d in shape], dtype=dtype, layout=None)

    # Initializers as params.
    for name, arr in initializer_arrays.items():
        tensor = torch.from_numpy(np.array(arr))
        params[name] = tensor
        primal.values[name] = Value(
            name=name,
            type=type_map.get(name) or TensorType(shape=list(arr.shape), dtype=str(arr.dtype)),
            kind=ValueKind.PARAM,
            debug_name="initializer",
        )

    # Graph inputs (excluding initializers).
    for vi in onnx_graph.input:
        if vi.name in initializer_arrays:
            continue
        primal.inputs.append(vi.name)
        primal.values[vi.name] = Value(
            name=vi.name,
            type=type_map.get(vi.name) or TensorType(shape=[], dtype="unknown"),
            kind=ValueKind.INPUT,
            debug_name="input",
        )

    alias: Dict[str, str] = {}

    def resolve(name: str) -> str:
        while name in alias:
            name = alias[name]
        return name

    def ensure_value(name: str, *, kind: ValueKind = ValueKind.INTERMEDIATE) -> None:
        if name in primal.values:
            return
        t = type_map.get(name)
        if t is None and name in initializer_arrays:
            arr = initializer_arrays[name]
            t = TensorType(shape=list(arr.shape), dtype=str(arr.dtype))
        primal.values[name] = Value(name=name, type=t or TensorType(shape=[], dtype="unknown"), kind=kind)

    def const_name(base: str) -> str:
        candidate = base
        idx = 0
        while candidate in primal.values or candidate in params:
            idx += 1
            candidate = f"{base}_{idx}"
        return candidate

    for idx, node in enumerate(onnx_graph.node):
        op = node.op_type
        node_name = node.name or f"{op.lower()}_{idx}"

        inputs = [resolve(i) for i in node.input if i]
        outputs = [o for o in node.output if o]

        if op == "Identity":
            if len(inputs) == 1 and len(outputs) == 1:
                alias[outputs[0]] = inputs[0]
                continue
            raise NotImplementedError(f"unsupported Identity arity: inputs={len(inputs)} outputs={len(outputs)}")

        if op == "Constant":
            if len(outputs) != 1:
                raise NotImplementedError("Constant with multiple outputs not supported")
            value_attr = None
            for a in node.attribute:
                if a.name == "value":
                    value_attr = a
            if value_attr is None:
                raise NotImplementedError("Constant without 'value' attribute not supported")
            arr = numpy_helper.to_array(value_attr.t)
            cname = outputs[0]
            params[cname] = torch.from_numpy(np.array(arr))
            ensure_value(cname, kind=ValueKind.CONST)
            primal.values[cname].debug_name = "const"
            continue

        if op == "Reshape":
            if len(outputs) != 1 or len(inputs) < 2:
                raise NotImplementedError("Reshape expects 2 inputs and 1 output")
            data = inputs[0]
            shape_in = inputs[1]
            shape_tensor = params.get(shape_in)
            if shape_tensor is None:
                raise NotImplementedError("Reshape shape input must be constant/initializer in v0")
            shape_list = [int(x) for x in shape_tensor.detach().cpu().numpy().reshape(-1).tolist()]
            ensure_value(data)
            ensure_value(outputs[0])
            primal.nodes.append(
                Node(
                    op_type="reshape",
                    name=node_name,
                    inputs=[data],
                    outputs=[outputs[0]],
                    attrs={"shape": shape_list},
                )
            )
            continue

        if op == "Flatten":
            if len(outputs) != 1 or len(inputs) != 1:
                raise NotImplementedError("Flatten expects 1 input and 1 output")
            axis = 1
            for a in node.attribute:
                if a.name == "axis":
                    axis = int(a.i)
            ensure_value(inputs[0])
            ensure_value(outputs[0])
            primal.nodes.append(
                Node(
                    op_type="flatten",
                    name=node_name,
                    inputs=[inputs[0]],
                    outputs=[outputs[0]],
                    attrs={"start_dim": axis, "end_dim": -1},
                )
            )
            continue

        if op == "Transpose":
            if len(outputs) != 1 or len(inputs) != 1:
                raise NotImplementedError("Transpose expects 1 input and 1 output")
            perm = None
            for a in node.attribute:
                if a.name == "perm":
                    perm = [int(x) for x in a.ints]
            if perm is None:
                raise NotImplementedError("Transpose missing perm attribute")
            ensure_value(inputs[0])
            ensure_value(outputs[0])
            primal.nodes.append(
                Node(
                    op_type="permute",
                    name=node_name,
                    inputs=[inputs[0]],
                    outputs=[outputs[0]],
                    attrs={"dims": perm},
                )
            )
            continue

        if op == "Relu":
            if len(outputs) != 1 or len(inputs) != 1:
                raise NotImplementedError("Relu expects 1 input and 1 output")
            ensure_value(inputs[0])
            ensure_value(outputs[0])
            primal.nodes.append(Node(op_type="relu", name=node_name, inputs=[inputs[0]], outputs=[outputs[0]], attrs={}))
            continue

        if op in ("Add", "Mul"):
            if len(outputs) != 1 or len(inputs) != 2:
                raise NotImplementedError(f"{op} expects 2 inputs and 1 output")
            ensure_value(inputs[0])
            ensure_value(inputs[1])
            ensure_value(outputs[0])
            primal.nodes.append(
                Node(
                    op_type=op.lower(),
                    name=node_name,
                    inputs=[inputs[0], inputs[1]],
                    outputs=[outputs[0]],
                    attrs={},
                )
            )
            continue

        if op == "Conv":
            if len(outputs) != 1 or len(inputs) < 2:
                raise NotImplementedError("Conv expects 2 or 3 inputs and 1 output")
            x = inputs[0]
            w = inputs[1]
            b = inputs[2] if len(inputs) >= 3 else None

            strides = (1, 1)
            pads = (0, 0, 0, 0)
            dilations = (1, 1)
            groups = 1
            for a in node.attribute:
                if a.name == "strides":
                    strides = tuple(int(v) for v in a.ints)
                if a.name == "pads":
                    pads = tuple(int(v) for v in a.ints)
                if a.name == "dilations":
                    dilations = tuple(int(v) for v in a.ints)
                if a.name == "group":
                    groups = int(a.i)

            if len(strides) != 2 or len(dilations) != 2:
                raise NotImplementedError("Conv strides/dilations must be 2D")
            if len(pads) != 4:
                raise NotImplementedError("Conv pads must have 4 ints")
            pad_h0, pad_w0, pad_h1, pad_w1 = pads
            if pad_h0 != pad_h1 or pad_w0 != pad_w1:
                raise NotImplementedError("asymmetric padding not supported in v0")

            if b is None:
                # Optional bias: default to zeros.
                w_t = params.get(w)
                if w_t is None or not torch.is_tensor(w_t):
                    raise NotImplementedError("Conv without bias requires constant weight in v0")
                b_name = const_name(f"{w}_bias0")
                params[b_name] = torch.zeros((int(w_t.shape[0]),), dtype=w_t.dtype)
                ensure_value(b_name, kind=ValueKind.CONST)
                b = b_name
            else:
                ensure_value(b, kind=ValueKind.PARAM if b in params else ValueKind.INTERMEDIATE)

            ensure_value(x)
            ensure_value(w, kind=ValueKind.PARAM if w in params else ValueKind.INTERMEDIATE)
            ensure_value(outputs[0])
            primal.nodes.append(
                Node(
                    op_type="conv2d",
                    name=node_name,
                    inputs=[x, w, b],
                    outputs=[outputs[0]],
                    attrs={
                        "stride": [int(strides[0]), int(strides[1])],
                        "padding": [int(pad_h0), int(pad_w0)],
                        "dilation": [int(dilations[0]), int(dilations[1])],
                        "groups": int(groups),
                    },
                )
            )
            continue

        if op == "Gemm":
            if len(outputs) != 1 or len(inputs) < 2:
                raise NotImplementedError("Gemm expects at least 2 inputs and 1 output")
            A = inputs[0]
            B = inputs[1]
            C = inputs[2] if len(inputs) >= 3 else None
            alpha = 1.0
            beta = 1.0
            transA = 0
            transB = 0
            for a in node.attribute:
                if a.name == "alpha":
                    alpha = float(a.f)
                if a.name == "beta":
                    beta = float(a.f)
                if a.name == "transA":
                    transA = int(a.i)
                if a.name == "transB":
                    transB = int(a.i)
            if transA != 0:
                raise NotImplementedError("Gemm transA!=0 not supported")

            if B in params and torch.is_tensor(params[B]):
                w_t = params[B]
                if transB == 0:
                    w_t = w_t.t().contiguous()
                if alpha != 1.0:
                    w_t = w_t * float(alpha)
                if w_t is not params[B]:
                    new_name = const_name(f"{B}_for_linear")
                    params[new_name] = w_t
                    ensure_value(new_name, kind=ValueKind.CONST)
                    B = new_name
            else:
                if transB != 1:
                    raise NotImplementedError("Gemm weight must be initializer/const to handle transpose")

            if C is not None:
                if C in params and torch.is_tensor(params[C]) and beta != 1.0:
                    b_t = params[C] * float(beta)
                    new_b = const_name(f"{C}_scaled")
                    params[new_b] = b_t
                    ensure_value(new_b, kind=ValueKind.CONST)
                    C = new_b
            else:
                if beta != 1.0:
                    raise NotImplementedError("Gemm beta!=1 without bias not supported")

            ensure_value(A)
            ensure_value(B, kind=ValueKind.PARAM if B in params else ValueKind.INTERMEDIATE)
            if C is not None:
                ensure_value(C, kind=ValueKind.PARAM if C in params else ValueKind.INTERMEDIATE)
            ensure_value(outputs[0])
            op_inputs = [A, B] + ([C] if C is not None else [])
            primal.nodes.append(Node(op_type="linear", name=node_name, inputs=op_inputs, outputs=[outputs[0]], attrs={}))
            continue

        if op == "MatMul":
            if len(outputs) != 1 or len(inputs) != 2:
                raise NotImplementedError("MatMul expects 2 inputs and 1 output")
            A, B = inputs[0], inputs[1]
            # Try to recognize linear-like pattern: A @ (W^T), where B is [I,O].
            if B in params and torch.is_tensor(params[B]) and params[B].dim() == 2:
                w_t = params[B].t().contiguous()
                w_name = const_name(f"{B}_for_linear")
                params[w_name] = w_t
                ensure_value(w_name, kind=ValueKind.CONST)
                ensure_value(A)
                ensure_value(outputs[0])
                primal.nodes.append(
                    Node(op_type="linear", name=node_name, inputs=[A, w_name], outputs=[outputs[0]], attrs={})
                )
                continue
            raise NotImplementedError("MatMul only supported when second input is a constant 2D weight")

        raise NotImplementedError(f"unsupported ONNX op: {op}")

    primal.outputs = [resolve(o.name) for o in onnx_graph.output]
    for out_name in primal.outputs:
        ensure_value(out_name)

    primal.validate()
    return _Converted(graph=primal, params=params)


def _collect_value_types(onnx_graph: onnx.GraphProto) -> Dict[str, TensorType]:
    types: Dict[str, TensorType] = {}
    for vi in list(onnx_graph.value_info) + list(onnx_graph.input) + list(onnx_graph.output):
        if vi.type is None or not vi.type.HasField("tensor_type"):
            continue
        tt = vi.type.tensor_type
        if tt.elem_type is None:
            continue
        try:
            np_dtype = onnx.helper.tensor_dtype_to_np_dtype(int(tt.elem_type))
            dtype = str(np.dtype(np_dtype))
        except Exception:
            dtype = "unknown"
        shape: List[Optional[int]] = []
        if tt.shape is not None:
            for d in tt.shape.dim:
                if d.HasField("dim_value"):
                    shape.append(int(d.dim_value))
                else:
                    shape.append(None)
        types[vi.name] = TensorType(shape=shape, dtype=dtype, layout=None)
    return types
