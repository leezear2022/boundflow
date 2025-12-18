from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple


@dataclass(frozen=True)
class IntervalLinearKey:
    batch: int
    in_features: int
    out_features: int
    dtype: str
    target: str


def _zero(dtype: str):
    import tvm

    return tvm.tir.const(0, dtype)


@lru_cache(maxsize=128)
def build_interval_linear_module(key: IntervalLinearKey):
    """
    Build a TVM module for interval affine (Linear) bound propagation (IBP):

    Given x_l, x_u in R^{B x I}, weight W in R^{O x I}, bias b in R^{O},
    compute y_l, y_u in R^{B x O}:

      W+ = max(W, 0), W- = min(W, 0)
      y_l = x_l @ W+^T + x_u @ W-^T + b
      y_u = x_u @ W+^T + x_l @ W-^T + b
    """
    import tvm
    from tvm import te

    B, I, O = key.batch, key.in_features, key.out_features
    dtype = key.dtype

    x_l = te.placeholder((B, I), name="x_l", dtype=dtype)
    x_u = te.placeholder((B, I), name="x_u", dtype=dtype)
    w = te.placeholder((O, I), name="w", dtype=dtype)
    b = te.placeholder((O,), name="b", dtype=dtype)

    k = te.reduce_axis((0, I), name="k")

    def w_pos(o, i):
        return tvm.tir.max(w[o, i], _zero(dtype))

    def w_neg(o, i):
        return tvm.tir.min(w[o, i], _zero(dtype))

    mat_l = te.compute(
        (B, O),
        lambda n, o: te.sum(x_l[n, k] * w_pos(o, k) + x_u[n, k] * w_neg(o, k), axis=k),
        name="mat_l",
    )
    mat_u = te.compute(
        (B, O),
        lambda n, o: te.sum(x_u[n, k] * w_pos(o, k) + x_l[n, k] * w_neg(o, k), axis=k),
        name="mat_u",
    )
    y_l = te.compute((B, O), lambda n, o: mat_l[n, o] + b[o], name="y_l")
    y_u = te.compute((B, O), lambda n, o: mat_u[n, o] + b[o], name="y_u")

    prim = te.create_prim_func([x_l, x_u, w, b, y_l, y_u])
    ir_mod = tvm.IRModule({"main": prim})

    # v0: build without custom scheduling for portability.
    # (We will revisit scheduling/fusion/autotune after TVMExecutor is in place.)
    rt_mod = tvm.build(ir_mod, target=key.target)
    return rt_mod["main"]


@lru_cache(maxsize=128)
def build_interval_linear_primfunc(key: IntervalLinearKey):
    """
    Build a TIR PrimFunc for interval affine (Linear) IBP.

    This is used by Relax `call_tir` lowering (Phase 5D PR#8+).
    """
    import tvm
    from tvm import te

    B, I, O = key.batch, key.in_features, key.out_features
    dtype = key.dtype

    x_l = te.placeholder((B, I), name="x_l", dtype=dtype)
    x_u = te.placeholder((B, I), name="x_u", dtype=dtype)
    w = te.placeholder((O, I), name="w", dtype=dtype)
    b = te.placeholder((O,), name="b", dtype=dtype)

    k = te.reduce_axis((0, I), name="k")

    def w_pos(o, i):
        return tvm.tir.max(w[o, i], _zero(dtype))

    def w_neg(o, i):
        return tvm.tir.min(w[o, i], _zero(dtype))

    mat_l = te.compute(
        (B, O),
        lambda n, o: te.sum(x_l[n, k] * w_pos(o, k) + x_u[n, k] * w_neg(o, k), axis=k),
        name="mat_l",
    )
    mat_u = te.compute(
        (B, O),
        lambda n, o: te.sum(x_u[n, k] * w_pos(o, k) + x_l[n, k] * w_neg(o, k), axis=k),
        name="mat_u",
    )
    y_l = te.compute((B, O), lambda n, o: mat_l[n, o] + b[o], name="y_l")
    y_u = te.compute((B, O), lambda n, o: mat_u[n, o] + b[o], name="y_u")

    return te.create_prim_func([x_l, x_u, w, b, y_l, y_u])
