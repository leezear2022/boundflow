from __future__ import annotations

from functools import lru_cache

from .interval_conv2d import IntervalConv2dKey


@lru_cache(maxsize=64)
def build_relax_interval_conv2d_vm_exec(key: IntervalConv2dKey):
    """
    Build a Relax VM executable for interval conv2d bound propagation (IBP), NCHW layout.

    v0 limitations:
    - groups==1 only (same as the existing TE/TIR demo kernel).
    """
    if key.groups != 1:
        raise NotImplementedError("relax interval conv2d (v0) only supports groups==1")

    import tvm
    from tvm import relax

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

    bb = relax.BlockBuilder()
    x_l = relax.Var("x_l", relax.TensorStructInfo((N, CI, H, W), dtype))
    x_u = relax.Var("x_u", relax.TensorStructInfo((N, CI, H, W), dtype))
    w = relax.Var("w", relax.TensorStructInfo((CO, CI, KH, KW), dtype))
    b = relax.Var("b", relax.TensorStructInfo((CO,), dtype))

    with bb.function("main", [x_l, x_u, w, b]):
        with bb.dataflow():
            zero = relax.const(0, dtype)
            w_pos = bb.emit(relax.op.maximum(w, zero))
            w_neg = bb.emit(relax.op.minimum(w, zero))

            conv = relax.op.nn.conv2d
            conv_l = bb.emit(
                relax.op.add(
                    conv(
                        x_l,
                        w_pos,
                        strides=(sh, sw),
                        padding=(ph, pw),
                        dilation=(dh, dw),
                        groups=1,
                        data_layout="NCHW",
                        kernel_layout="OIHW",
                    ),
                    conv(
                        x_u,
                        w_neg,
                        strides=(sh, sw),
                        padding=(ph, pw),
                        dilation=(dh, dw),
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
                        strides=(sh, sw),
                        padding=(ph, pw),
                        dilation=(dh, dw),
                        groups=1,
                        data_layout="NCHW",
                        kernel_layout="OIHW",
                    ),
                    conv(
                        x_l,
                        w_neg,
                        strides=(sh, sw),
                        padding=(ph, pw),
                        dilation=(dh, dw),
                        groups=1,
                        data_layout="NCHW",
                        kernel_layout="OIHW",
                    ),
                )
            )

            b_4d = bb.emit(relax.op.reshape(b, (1, CO, 1, 1)))
            b_b = bb.emit(relax.op.broadcast_to(b_4d, (N, CO, oh, ow)))
            y_l = bb.emit(relax.op.add(conv_l, b_b))
            y_u = bb.emit(relax.op.add(conv_u, b_b))

            out = bb.emit_output(relax.Tuple([y_l, y_u]))
        bb.emit_func_output(out)

    mod = bb.get()
    _ = tvm
    return relax.build(mod, target=key.target)

