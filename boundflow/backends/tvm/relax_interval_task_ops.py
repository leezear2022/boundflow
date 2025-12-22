from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple

from ...ir.task import BoundTask, StoragePlan, TaskOp


@dataclass(frozen=True)
class IntervalTaskLoweringSpec:
    """
    A small, json-able lowering spec describing how to call the compiled Relax function.
    """

    func_name: str
    input_values: List[str]
    param_values: List[str]
    output_values: List[str]
    output_flattened: bool = True  # outputs are returned as [o0_l,o0_u,o1_l,o1_u,...]


def _sanitize(name: str) -> str:
    # TVM Relax var names prefer [a-zA-Z0-9_].
    return re.sub(r"[^0-9a-zA-Z_]", "_", str(name))


def _spec_for_value(storage_plan: StoragePlan, value_name: str) -> Tuple[List[Optional[int]], str]:
    bid = storage_plan.value_to_buffer.get(value_name)
    if bid is None:
        raise KeyError(f"value not found in storage_plan.value_to_buffer: {value_name}")
    if bid not in storage_plan.buffers:
        raise KeyError(f"buffer not found in storage_plan.buffers: {bid} (value={value_name})")
    spec = storage_plan.buffers[bid]
    shape = list(spec.shape)
    dtype = str(spec.dtype).replace("torch.", "")
    if any(d is None for d in shape):
        raise ValueError(f"dynamic shape not supported in v0 relax task lowering: value={value_name} shape={shape}")
    return shape, dtype


def _expect_rank(shape: List[Optional[int]], rank: int, *, what: str) -> None:
    if len(shape) != rank:
        raise ValueError(f"{what} expects rank-{rank}, got shape={shape}")


def build_interval_task_relax_ops_ir_module(
    task: BoundTask,
    *,
    storage_plan: StoragePlan,
    target: str,
    func_name: str = "main",
) -> Tuple[object, IntervalTaskLoweringSpec]:
    """
    Lower a multi-op INTERVAL_IBP task into a single Relax function using Relax high-level ops only.

    v0 scope:
    - straight-line tasks (no control flow), ops executed in order
    - supports: linear / conv2d / relu / add / mul / permute / reshape / flatten
    - interval lanes are explicit pairs: each value is represented as (lower, upper)

    The produced Relax function signature is:
      (for each input value v: v_l, v_u) + (for each param p: p)  ->  Tuple([o0_l,o0_u,o1_l,o1_u,...])
    """
    _ = target  # reserved for future legalization customization
    task.validate()

    import tvm  # noqa: PLC0415
    from tvm import relax  # noqa: PLC0415

    # Inputs (interval lanes) and params (plain tensors).
    raw_input_values = list(task.input_values)
    raw_param_values = list(task.params or [])

    # Treat param/const-scoped values as plain params (not interval inputs),
    # even if they appear in task.input_values.
    param_scopes = {"param", "const"}
    inferred_params: List[str] = []
    for v in raw_input_values:
        bid = storage_plan.value_to_buffer.get(v)
        if bid is None:
            continue
        spec = storage_plan.buffers.get(bid)
        if spec is None:
            continue
        if str(spec.scope) in param_scopes:
            inferred_params.append(v)

    param_values = sorted({*raw_param_values, *inferred_params})
    input_values = [v for v in raw_input_values if v not in param_values]
    output_values = list(task.output_values)

    bb = relax.BlockBuilder()

    # Create vars for inputs/params.
    input_vars_lu: Dict[str, Tuple[Any, Any]] = {}
    param_vars: Dict[str, Any] = {}
    fn_params: List[Any] = []

    for v in input_values:
        shape, dtype = _spec_for_value(storage_plan, v)
        _expect_rank(shape, len(shape), what="input")  # no-op but keeps error message shape
        vl = relax.Var(f"{_sanitize(v)}_l", relax.TensorStructInfo(tuple(int(d) for d in shape), dtype))
        vu = relax.Var(f"{_sanitize(v)}_u", relax.TensorStructInfo(tuple(int(d) for d in shape), dtype))
        input_vars_lu[v] = (vl, vu)
        fn_params.extend([vl, vu])

    for p in param_values:
        shape, dtype = _spec_for_value(storage_plan, p)
        pv = relax.Var(_sanitize(p), relax.TensorStructInfo(tuple(int(d) for d in shape), dtype))
        param_vars[p] = pv
        fn_params.append(pv)

    with bb.function(func_name, fn_params):
        with bb.dataflow():
            # Map value_name -> (lower_expr, upper_expr)
            vals: Dict[str, Tuple[Any, Any]] = dict(input_vars_lu)

            def get_interval(name: str) -> Tuple[Any, Any]:
                if name in vals:
                    return vals[name]
                if name in param_vars:
                    t = param_vars[name]
                    return (t, t)
                raise KeyError(f"missing value in relax lowering env: {name}")

            def get_tensor(name: str) -> Any:
                if name in param_vars:
                    return param_vars[name]
                raise KeyError(f"missing param in relax lowering: {name}")

            for op in task.ops:
                if not isinstance(op, TaskOp):
                    raise TypeError(f"unexpected op type: {type(op)}")

                if op.op_type == "linear":
                    x_l, x_u = get_interval(op.inputs[0])
                    w = get_tensor(op.inputs[1])
                    b = get_tensor(op.inputs[2]) if len(op.inputs) >= 3 else None
                    if b is None:
                        # bias-free linear: treat bias as zeros.
                        # Shape inferred from weight out_features.
                        w_shape, w_dtype = _spec_for_value(storage_plan, op.inputs[1])
                        _expect_rank(w_shape, 2, what="linear weight")
                        O = int(w_shape[0])
                        b = bb.emit(relax.const(0, w_dtype))
                        b = bb.emit(relax.op.broadcast_to(bb.emit(relax.op.reshape(b, (1,))), (O,)))

                    w_pos = bb.emit(relax.op.maximum(w, relax.const(0, _spec_for_value(storage_plan, op.inputs[1])[1])))
                    w_neg = bb.emit(relax.op.minimum(w, relax.const(0, _spec_for_value(storage_plan, op.inputs[1])[1])))
                    w_pos_t = bb.emit(relax.op.permute_dims(w_pos, axes=[1, 0]))
                    w_neg_t = bb.emit(relax.op.permute_dims(w_neg, axes=[1, 0]))

                    mat_l = bb.emit(relax.op.add(relax.op.matmul(x_l, w_pos_t), relax.op.matmul(x_u, w_neg_t)))
                    mat_u = bb.emit(relax.op.add(relax.op.matmul(x_u, w_pos_t), relax.op.matmul(x_l, w_neg_t)))

                    # bias broadcast to [B, O]
                    x_shape, _ = _spec_for_value(storage_plan, op.inputs[0])
                    w_shape, _ = _spec_for_value(storage_plan, op.inputs[1])
                    B = int(x_shape[0])
                    O = int(w_shape[0])
                    b_2d = bb.emit(relax.op.reshape(b, (1, O)))
                    b_b = bb.emit(relax.op.broadcast_to(b_2d, (B, O)))
                    y_l = bb.emit(relax.op.add(mat_l, b_b))
                    y_u = bb.emit(relax.op.add(mat_u, b_b))

                    vals[op.outputs[0]] = (y_l, y_u)
                    continue

                if op.op_type == "conv2d":
                    # NCHW/OIHW only (v0).
                    x_l, x_u = get_interval(op.inputs[0])
                    w = get_tensor(op.inputs[1])
                    if len(op.inputs) < 3:
                        raise ValueError(f"conv2d expects bias tensor input in v0 relax task lowering: op='{op.name}'")
                    b = get_tensor(op.inputs[2])

                    def _as_int_tuple(x: Any, *, name: str) -> Tuple[int, ...]:
                        if isinstance(x, int):
                            return (int(x),)
                        if isinstance(x, (list, tuple)):
                            return tuple(int(d) for d in x)
                        raise ValueError(f"conv2d attr '{name}' must be int/list/tuple, got {type(x)}: {x}")

                    stride_raw = op.attrs.get("stride", 1)
                    padding_raw = op.attrs.get("padding", 0)
                    dilation_raw = op.attrs.get("dilation", 1)
                    groups = int(op.attrs.get("groups", 1))
                    if groups != 1:
                        raise NotImplementedError("conv2d relax task lowering (v0) only supports groups==1")

                    stride_t = _as_int_tuple(stride_raw, name="stride")
                    dilation_t = _as_int_tuple(dilation_raw, name="dilation")
                    padding_t = _as_int_tuple(padding_raw, name="padding")

                    if len(stride_t) == 1:
                        stride_t = (stride_t[0], stride_t[0])
                    if len(dilation_t) == 1:
                        dilation_t = (dilation_t[0], dilation_t[0])
                    if len(padding_t) == 1:
                        padding_t = (padding_t[0], padding_t[0])
                    if len(stride_t) != 2 or len(dilation_t) != 2 or len(padding_t) not in (2, 4):
                        raise ValueError(
                            f"conv2d attrs shape invalid for op '{op.name}': "
                            f"stride={stride_t}, padding={padding_t}, dilation={dilation_t}"
                        )

                    zero_w = relax.const(0, _spec_for_value(storage_plan, op.inputs[1])[1])
                    w_pos = bb.emit(relax.op.maximum(w, zero_w))
                    w_neg = bb.emit(relax.op.minimum(w, zero_w))

                    conv = relax.op.nn.conv2d
                    conv_l = bb.emit(
                        relax.op.add(
                            conv(
                                x_l,
                                w_pos,
                                strides=tuple(int(d) for d in stride_t),
                                padding=tuple(int(d) for d in padding_t),
                                dilation=tuple(int(d) for d in dilation_t),
                                groups=1,
                                data_layout="NCHW",
                                kernel_layout="OIHW",
                            ),
                            conv(
                                x_u,
                                w_neg,
                                strides=tuple(int(d) for d in stride_t),
                                padding=tuple(int(d) for d in padding_t),
                                dilation=tuple(int(d) for d in dilation_t),
                                groups=1,
                                data_layout="NCHW",
                                kernel_layout="OIHW",
                            ),
                        )
                    )
                    conv_u = bb.emit(
                        relax.op.add(
                            conv(
                                x_u,
                                w_pos,
                                strides=tuple(int(d) for d in stride_t),
                                padding=tuple(int(d) for d in padding_t),
                                dilation=tuple(int(d) for d in dilation_t),
                                groups=1,
                                data_layout="NCHW",
                                kernel_layout="OIHW",
                            ),
                            conv(
                                x_l,
                                w_neg,
                                strides=tuple(int(d) for d in stride_t),
                                padding=tuple(int(d) for d in padding_t),
                                dilation=tuple(int(d) for d in dilation_t),
                                groups=1,
                                data_layout="NCHW",
                                kernel_layout="OIHW",
                            ),
                        )
                    )

                    out_shape, _dtype = _spec_for_value(storage_plan, op.outputs[0])
                    _expect_rank(out_shape, 4, what="conv2d output")
                    N, CO, OH, OW = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]), int(out_shape[3]))
                    b_4d = bb.emit(relax.op.reshape(b, (1, CO, 1, 1)))
                    b_b = bb.emit(relax.op.broadcast_to(b_4d, (N, CO, OH, OW)))
                    y_l = bb.emit(relax.op.add(conv_l, b_b))
                    y_u = bb.emit(relax.op.add(conv_u, b_b))

                    vals[op.outputs[0]] = (y_l, y_u)
                    continue

                if op.op_type == "relu":
                    x_l, x_u = get_interval(op.inputs[0])
                    zero = relax.const(0, _spec_for_value(storage_plan, op.inputs[0])[1])
                    y_l = bb.emit(relax.op.maximum(x_l, zero))
                    y_u = bb.emit(relax.op.maximum(x_u, zero))
                    vals[op.outputs[0]] = (y_l, y_u)
                    continue

                if op.op_type == "add":
                    a_l, a_u = get_interval(op.inputs[0])
                    b_l, b_u = get_interval(op.inputs[1])
                    vals[op.outputs[0]] = (bb.emit(relax.op.add(a_l, b_l)), bb.emit(relax.op.add(a_u, b_u)))
                    continue

                if op.op_type == "mul":
                    a_l, a_u = get_interval(op.inputs[0])
                    b_l, b_u = get_interval(op.inputs[1])
                    c1 = bb.emit(relax.op.multiply(a_l, b_l))
                    c2 = bb.emit(relax.op.multiply(a_l, b_u))
                    c3 = bb.emit(relax.op.multiply(a_u, b_l))
                    c4 = bb.emit(relax.op.multiply(a_u, b_u))
                    lb = bb.emit(relax.op.minimum(bb.emit(relax.op.minimum(c1, c2)), bb.emit(relax.op.minimum(c3, c4))))
                    ub = bb.emit(relax.op.maximum(bb.emit(relax.op.maximum(c1, c2)), bb.emit(relax.op.maximum(c3, c4))))
                    vals[op.outputs[0]] = (lb, ub)
                    continue

                if op.op_type in ("permute", "transpose"):
                    x_l, x_u = get_interval(op.inputs[0])
                    axes = op.attrs.get("dims") or op.attrs.get("axes")
                    if not isinstance(axes, (list, tuple)):
                        raise ValueError(f"permute/transpose missing dims for op '{op.name}': {axes}")
                    axes = [int(a) for a in axes]
                    vals[op.outputs[0]] = (
                        bb.emit(relax.op.permute_dims(x_l, axes=axes)),
                        bb.emit(relax.op.permute_dims(x_u, axes=axes)),
                    )
                    continue

                if op.op_type == "reshape":
                    x_l, x_u = get_interval(op.inputs[0])
                    shape = op.attrs.get("shape")
                    if not isinstance(shape, (list, tuple)):
                        raise ValueError(f"reshape missing shape attr for op '{op.name}': {shape}")
                    # Use static reshape only in v0.
                    shape = tuple(int(d) for d in shape)
                    vals[op.outputs[0]] = (bb.emit(relax.op.reshape(x_l, shape)), bb.emit(relax.op.reshape(x_u, shape)))
                    continue

                if op.op_type == "flatten":
                    x_l, x_u = get_interval(op.inputs[0])
                    out_shape, _dtype = _spec_for_value(storage_plan, op.outputs[0])
                    shape = tuple(int(d) for d in out_shape)
                    vals[op.outputs[0]] = (bb.emit(relax.op.reshape(x_l, shape)), bb.emit(relax.op.reshape(x_u, shape)))
                    continue

                raise NotImplementedError(f"unsupported op_type for relax task lowering: {op.op_type}")

            # Build outputs as a flat tuple.
            out_exprs: List[Any] = []
            for ov in output_values:
                y_l, y_u = get_interval(ov)
                out_exprs.extend([y_l, y_u])
            out = bb.emit_output(relax.Tuple(out_exprs))
        bb.emit_func_output(out)

    mod = bb.get()
    _ = tvm
    spec = IntervalTaskLoweringSpec(
        func_name=str(func_name),
        input_values=list(input_values),
        param_values=list(param_values),
        output_values=list(output_values),
        output_flattened=True,
    )
    return mod, spec
