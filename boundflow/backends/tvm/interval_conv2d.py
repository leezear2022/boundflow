from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class IntervalConv2dKey:
    batch: int
    in_channels: int
    in_h: int
    in_w: int
    out_channels: int
    k_h: int
    k_w: int
    stride_h: int
    stride_w: int
    pad_h: int
    pad_w: int
    dilation_h: int
    dilation_w: int
    groups: int
    dtype: str
    target: str


def _zero(dtype: str):
    import tvm

    return tvm.tir.const(0, dtype)


@lru_cache(maxsize=64)
def build_interval_conv2d_module(key: IntervalConv2dKey):
    """
    Build a TVM module for interval conv2d bound propagation (IBP), NCHW layout.

    Given x_l, x_u in R^{N x CI x H x W}, weight W in R^{CO x CI x KH x KW}, bias b in R^{CO},
    compute y_l, y_u in R^{N x CO x OH x OW}:

      W+ = max(W, 0), W- = min(W, 0)
      y_l = conv2d(x_l, W+) + conv2d(x_u, W-) + b
      y_u = conv2d(x_u, W+) + conv2d(x_l, W-) + b

    Notes:
    - groups is supported only for groups==1 in v0.
    - padding is zero-padding.
    """
    if key.groups != 1:
        raise NotImplementedError("interval conv2d tvm kernel (v0) only supports groups==1")

    import tvm
    from tvm import te

    N, CI, H, W = key.batch, key.in_channels, key.in_h, key.in_w
    CO, KH, KW = key.out_channels, key.k_h, key.k_w
    sh, sw = key.stride_h, key.stride_w
    ph, pw = key.pad_h, key.pad_w
    dh, dw = key.dilation_h, key.dilation_w
    dtype = key.dtype

    oh = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    ow = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError(f"invalid conv2d output shape: OH={oh}, OW={ow} from {key}")

    x_l = te.placeholder((N, CI, H, W), name="x_l", dtype=dtype)
    x_u = te.placeholder((N, CI, H, W), name="x_u", dtype=dtype)
    w = te.placeholder((CO, CI, KH, KW), name="w", dtype=dtype)
    b = te.placeholder((CO,), name="b", dtype=dtype)

    rc = te.reduce_axis((0, CI), name="rc")
    rkh = te.reduce_axis((0, KH), name="rkh")
    rkw = te.reduce_axis((0, KW), name="rkw")

    def w_pos(co, ci, kh, kw):
        return tvm.tir.max(w[co, ci, kh, kw], _zero(dtype))

    def w_neg(co, ci, kh, kw):
        return tvm.tir.min(w[co, ci, kh, kw], _zero(dtype))

    def load_padded(x, n, ci, ih, iw):
        in_bounds = tvm.tir.all(ih >= 0, ih < H, iw >= 0, iw < W)
        return tvm.tir.if_then_else(in_bounds, x[n, ci, ih, iw], _zero(dtype))

    mat_l = te.compute(
        (N, CO, oh, ow),
        lambda n, co, y, x: te.sum(
            load_padded(x_l, n, rc, y * sh + rkh * dh - ph, x * sw + rkw * dw - pw)
            * w_pos(co, rc, rkh, rkw)
            + load_padded(x_u, n, rc, y * sh + rkh * dh - ph, x * sw + rkw * dw - pw)
            * w_neg(co, rc, rkh, rkw),
            axis=[rc, rkh, rkw],
        ),
        name="mat_l",
    )
    mat_u = te.compute(
        (N, CO, oh, ow),
        lambda n, co, y, x: te.sum(
            load_padded(x_u, n, rc, y * sh + rkh * dh - ph, x * sw + rkw * dw - pw)
            * w_pos(co, rc, rkh, rkw)
            + load_padded(x_l, n, rc, y * sh + rkh * dh - ph, x * sw + rkw * dw - pw)
            * w_neg(co, rc, rkh, rkw),
            axis=[rc, rkh, rkw],
        ),
        name="mat_u",
    )
    y_l = te.compute((N, CO, oh, ow), lambda n, co, y, x: mat_l[n, co, y, x] + b[co], name="y_l")
    y_u = te.compute((N, CO, oh, ow), lambda n, co, y, x: mat_u[n, co, y, x] + b[co], name="y_u")

    prim = te.create_prim_func([x_l, x_u, w, b, y_l, y_u])
    ir_mod = tvm.IRModule({"main": prim})
    rt_mod = tvm.build(ir_mod, target=key.target)
    return rt_mod["main"]

