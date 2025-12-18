from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ...ir.task import BFTaskModule, BoundTask
from .interval_linear import IntervalLinearKey, build_interval_linear_primfunc


class RelaxLoweringMode(Enum):
    """
    Lowering mode for a single task.

    - RELAX_OPS: build a Relax function using high-level Relax ops only.
    - CALL_TIR: build a mixed IRModule (Relax wrapper + TIR PrimFunc) via call_tir.
    """

    RELAX_OPS = "relax_ops"
    CALL_TIR = "call_tir"


@dataclass(frozen=True)
class RelaxLoweringResult:
    ir_mod: object  # tvm.IRModule
    relax_func_name: str
    mode: RelaxLoweringMode


def build_interval_linear_relax_ir_module(
    key: IntervalLinearKey,
    *,
    mode: RelaxLoweringMode,
    relax_func_name: str = "main",
) -> object:
    """
    Build a Relax IRModule for interval-linear IBP from a shape key.

    This helper is intentionally key-driven so runtime (executor) can compile-cache without needing
    access to BFTaskModule/BoundTask.
    """
    import tvm  # noqa: PLC0415
    from tvm import relax  # noqa: PLC0415

    B, I, O = key.batch, key.in_features, key.out_features
    dtype = key.dtype

    bb = relax.BlockBuilder()

    x_l = relax.Var("x_l", relax.TensorStructInfo((B, I), dtype))
    x_u = relax.Var("x_u", relax.TensorStructInfo((B, I), dtype))
    w = relax.Var("w", relax.TensorStructInfo((O, I), dtype))
    b = relax.Var("b", relax.TensorStructInfo((O,), dtype))

    if mode == RelaxLoweringMode.CALL_TIR:
        tir_name = f"{relax_func_name}_interval_linear_tir"
        primfunc = build_interval_linear_primfunc(key).with_attr("global_symbol", tir_name)
        tir_gv = bb.add_func(primfunc, tir_name)

        with bb.function(relax_func_name, [x_l, x_u, w, b]):
            with bb.dataflow():
                args = relax.Tuple([x_l, x_u, w, b])
                out = bb.emit(
                    relax.call_tir(
                        tir_gv,
                        args,
                        out_sinfo=[
                            relax.TensorStructInfo((B, O), dtype),
                            relax.TensorStructInfo((B, O), dtype),
                        ],
                    )
                )
                out = bb.emit_output(out)
            bb.emit_func_output(out)

        return bb.get()

    if mode == RelaxLoweringMode.RELAX_OPS:
        with bb.function(relax_func_name, [x_l, x_u, w, b]):
            with bb.dataflow():
                zero = relax.const(0, dtype)
                w_pos = bb.emit(relax.op.maximum(w, zero))
                w_neg = bb.emit(relax.op.minimum(w, zero))
                w_pos_t = bb.emit(relax.op.permute_dims(w_pos, axes=[1, 0]))
                w_neg_t = bb.emit(relax.op.permute_dims(w_neg, axes=[1, 0]))

                mat_l = bb.emit(relax.op.add(relax.op.matmul(x_l, w_pos_t), relax.op.matmul(x_u, w_neg_t)))
                mat_u = bb.emit(relax.op.add(relax.op.matmul(x_u, w_pos_t), relax.op.matmul(x_l, w_neg_t)))

                b_2d = bb.emit(relax.op.reshape(b, (1, O)))
                b_b = bb.emit(relax.op.broadcast_to(b_2d, (B, O)))
                y_l = bb.emit(relax.op.add(mat_l, b_b))
                y_u = bb.emit(relax.op.add(mat_u, b_b))

                out = bb.emit_output(relax.Tuple([y_l, y_u]))
            bb.emit_func_output(out)

        return bb.get()

    raise ValueError(f"unsupported RelaxLoweringMode: {mode}")


def _torch_dtype_to_str(dtype: str) -> str:
    # StoragePlan/Value dtype strings are typically like "float32".
    # Keep as-is, but normalize "torch.float32" -> "float32" when needed.
    d = str(dtype)
    return d.replace("torch.", "")


def _infer_interval_linear_key(module: BFTaskModule, op_inputs: list[str], *, target: str) -> IntervalLinearKey:
    sp = module.storage_plan
    x_value, w_value = op_inputs[0], op_inputs[1]
    x_bid = sp.value_to_buffer.get(x_value)
    w_bid = sp.value_to_buffer.get(w_value)
    if x_bid is None or w_bid is None:
        raise KeyError(f"missing value_to_buffer for linear inputs: x={x_value} w={w_value}")
    x_spec = sp.buffers[x_bid]
    w_spec = sp.buffers[w_bid]
    if len(x_spec.shape) != 2 or len(w_spec.shape) != 2:
        raise ValueError(f"interval_linear expects x shape [B,I] and w shape [O,I], got {x_spec.shape} and {w_spec.shape}")
    B, I = x_spec.shape
    O, I2 = w_spec.shape
    if B is None or I is None or O is None or I2 is None:
        raise ValueError(f"interval_linear requires static B/I/O for v0 lowering, got x={x_spec.shape} w={w_spec.shape}")
    if int(I2) != int(I):
        raise ValueError(f"linear shape mismatch: x in_features={I} w in_features={I2}")
    dtype = _torch_dtype_to_str(x_spec.dtype)
    return IntervalLinearKey(batch=int(B), in_features=int(I), out_features=int(O), dtype=dtype, target=target)


def lower_interval_linear_task_to_relax_ir(
    task: BoundTask,
    module: BFTaskModule,
    *,
    target: str,
    mode: RelaxLoweringMode = RelaxLoweringMode.RELAX_OPS,
    relax_func_name: Optional[str] = None,
) -> RelaxLoweringResult:
    """
    Lower a single-task interval-IBP linear task to a Relax IRModule.

    v0: supports tasks with exactly one op: `linear(x, w, b) -> y`.
    Interval lanes are represented explicitly as two tensors: {x_l, x_u} -> {y_l, y_u}.
    """
    module.validate()
    task.validate()

    if len(task.ops) != 1:
        raise NotImplementedError(f"v0 relax lowering only supports single-op tasks, got {len(task.ops)} ops")
    op = task.ops[0]
    if op.op_type != "linear":
        raise NotImplementedError(f"v0 relax lowering only supports op_type='linear', got {op.op_type}")
    if len(op.inputs) < 2:
        raise ValueError(f"linear op expects inputs [x,w,(b)], got {op.inputs}")
    if len(op.outputs) != 1:
        raise ValueError(f"linear op expects exactly 1 output value, got {op.outputs}")

    if relax_func_name is None:
        relax_func_name = f"bf_{task.task_id}"

    key = _infer_interval_linear_key(module, list(op.inputs), target=target)

    mod = build_interval_linear_relax_ir_module(key, mode=mode, relax_func_name=relax_func_name)
    return RelaxLoweringResult(ir_mod=mod, relax_func_name=relax_func_name, mode=mode)
