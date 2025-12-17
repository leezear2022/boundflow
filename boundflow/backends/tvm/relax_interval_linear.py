from __future__ import annotations

from functools import lru_cache

from .interval_linear import IntervalLinearKey


@lru_cache(maxsize=128)
def build_relax_interval_linear_vm_exec(key: IntervalLinearKey):
    """
    Build a Relax VM executable for interval affine (Linear) bound propagation (IBP).

    Note:
    - We intentionally avoid writing TE/TIR by hand. TVM will still lower Relax ops to TIR internally.
    - This repository's TVM runtime uses `tvm.runtime.Tensor` (not `tvm.nd.NDArray`).
    """
    import tvm
    from tvm import relax

    B, I, O = key.batch, key.in_features, key.out_features
    dtype = key.dtype

    bb = relax.BlockBuilder()
    x_l = relax.Var("x_l", relax.TensorStructInfo((B, I), dtype))
    x_u = relax.Var("x_u", relax.TensorStructInfo((B, I), dtype))
    w = relax.Var("w", relax.TensorStructInfo((O, I), dtype))
    b = relax.Var("b", relax.TensorStructInfo((O,), dtype))

    with bb.function("main", [x_l, x_u, w, b]):
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

    mod = bb.get()
    _ = tvm  # keep import for type checkers
    return relax.build(mod, target=key.target)

