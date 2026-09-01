"""Frozen R3-1b2 compiled P-alpha VJP support kernels."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,invalid-name,protected-access
# pylint: disable=missing-function-docstring,too-many-locals,too-many-statements
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=chained-comparison,too-many-nested-blocks,too-many-branches
# pylint: disable=missing-class-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)
from boundflow.backends.tvm.r3_full_lower_forward import (
    R31B1_ARENA_CAPACITY,
    R31B1_THREADS,
    _conv_right_primfunc,
    _schedule_te_primfunc,
)

R31B2_CLEAR_SYMBOL = "boundflow_r31b2_clear_arenas"
R31B2_PACK_A24_SYMBOL = "boundflow_r31b2_pack_a24_sign"
R31B2_PACK_A20_SYMBOL = "boundflow_r31b2_pack_a20_sign"
R31B2_PACK_A18_SYMBOL = "boundflow_r31b2_pack_a18_sign"
R31B2_PACK_AINPUT_SYMBOL = "boundflow_r31b2_pack_ainput_sign"
R31B2_EFFECTIVE_PRE17_SYMBOL = "boundflow_r31b2_effective_pre17"
R31B2_EFFECTIVE_PRE23_SYMBOL = "boundflow_r31b2_effective_pre23"
R31B2_EFFECTIVE_PRE25_SYMBOL = "boundflow_r31b2_effective_pre25"
R31B2_CONV10_RIGHT_SYMBOL = "boundflow_r31b2_conv10_right"
R31B2_COMPRESSED_GRADIENT_SYMBOL = "boundflow_r31b2_compressed_gradient"

R31B2_EXPORTED_SYMBOLS = (
    R31B2_CLEAR_SYMBOL,
    R31B2_PACK_A24_SYMBOL,
    R31B2_PACK_A20_SYMBOL,
    R31B2_PACK_A18_SYMBOL,
    R31B2_PACK_AINPUT_SYMBOL,
    R31B2_EFFECTIVE_PRE17_SYMBOL,
    R31B2_EFFECTIVE_PRE23_SYMBOL,
    R31B2_EFFECTIVE_PRE25_SYMBOL,
    R31B2_CONV10_RIGHT_SYMBOL,
    R31B2_COMPRESSED_GRADIENT_SYMBOL,
)


@dataclass(frozen=True)
class CompiledR31B2PAlphaVJPV1:
    executable: DifferentiableLowerTIRExecutable
    module_hash: str
    device_source_hash: str
    device_source: str
    exported_symbols: tuple[str, ...]
    global_workspace_bytes: int
    tvm_version: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _selected_relu_value(
    tvm,
    value,
    sign,
    lower,
    upper,
    alpha,
    alpha_map,
    d_idx,
    feature,
):
    zero = tvm.tir.const(0.0, "float32")
    one = tvm.tir.const(1.0, "float32")
    epsilon = tvm.tir.const(1.1920928955078125e-07, "float32")
    lookup = alpha_map[feature]
    compact = alpha[0, 0, d_idx, tvm.tir.max(lookup, 0)]
    lower_alpha = tvm.tir.if_then_else(
        lookup >= 0, tvm.tir.min(tvm.tir.max(compact, zero), one), zero
    )
    ambiguous = tvm.tir.all(lower[d_idx, feature] < zero, upper[d_idx, feature] > zero)
    lower_slope = tvm.tir.if_then_else(
        ambiguous,
        lower_alpha,
        tvm.tir.if_then_else(lower[d_idx, feature] >= zero, one, zero),
    )
    upper_slope = tvm.tir.if_then_else(
        lower[d_idx, feature] >= zero,
        one,
        tvm.tir.if_then_else(
            upper[d_idx, feature] <= zero,
            zero,
            upper[d_idx, feature]
            / tvm.tir.max(upper[d_idx, feature] - lower[d_idx, feature], epsilon),
        ),
    )
    slope = tvm.tir.if_then_else(sign != 0, lower_slope, upper_slope)
    intercept = tvm.tir.if_then_else(
        tvm.tir.all(sign == 0, ambiguous),
        -lower[d_idx, feature] * upper_slope,
        zero,
    )
    return slope * value + intercept


def _clear_primfunc():
    from tvm.script import tir as T

    @T.prim_func
    def clear_arenas(
        scratch0: T.Buffer((R31B1_ARENA_CAPACITY,), "float32"),
        scratch1: T.Buffer((R31B1_ARENA_CAPACITY,), "float32"),
    ):
        T.func_attr(
            {
                "global_symbol": R31B2_CLEAR_SYMBOL,
                "boundflow.schema_version": "r3-1b2-p-alpha-vjp/v1",
            }
        )
        for block in T.thread_binding(144, thread="blockIdx.x"):
            for thread in T.thread_binding(R31B1_THREADS, thread="threadIdx.x"):
                flat = block * R31B1_THREADS + thread
                if flat < R31B1_ARENA_CAPACITY:
                    scratch0[flat] = T.float32(0)
                    scratch1[flat] = T.float32(0)

    return clear_arenas


def _pack_sign_primfunc(symbol: str, *, numel: int):
    import tvm
    from tvm import te

    source = te.placeholder((numel,), "float32", name="coefficient")
    bitmap = te.compute(
        (numel,),
        lambda flat: tvm.tir.if_then_else(
            source[flat] >= tvm.tir.const(0.0, "float32"),
            tvm.tir.const(1, "int8"),
            tvm.tir.const(0, "int8"),
        ),
        name="sign_bitmap",
    )
    return (
        te.create_prim_func([source, bitmap])
        .with_attr("global_symbol", symbol)
        .with_attr("boundflow.schema_version", "r3-1b2-p-alpha-vjp/v1")
    )


def _pack_a20_primfunc():
    from tvm.script import tir as T

    @T.prim_func
    def pack_a20(
        incoming_a23: T.Buffer((6144,), "float32"),
        weight4: T.Buffer((16, 16, 3, 3), "float32"),
        sign_a20: T.Buffer((6144,), "int8"),
    ):
        T.func_attr(
            {
                "global_symbol": R31B2_PACK_A20_SYMBOL,
                "boundflow.schema_version": "r3-1b2-p-alpha-vjp/v1",
            }
        )
        total = T.alloc_buffer((1,), "float32", scope="local")
        for block in T.thread_binding(48, thread="blockIdx.x"):
            for thread in T.thread_binding(R31B1_THREADS, thread="threadIdx.x"):
                flat = block * R31B1_THREADS + thread
                if flat < 6144:
                    d = flat // 1024
                    logical = flat % 1024
                    input_w = logical % 8
                    input_h = logical // 8 % 8
                    input_channel = logical // 64
                    total[0] = T.float32(0)
                    for out_channel, kernel_h, kernel_w in T.grid(16, 3, 3):
                        output_h = input_h + 1 - kernel_h
                        output_w = input_w + 1 - kernel_w
                        if (
                            0 <= output_h
                            and output_h < 8
                            and 0 <= output_w
                            and output_w < 8
                        ):
                            source = (
                                d * 1024 + out_channel * 64 + output_h * 8 + output_w
                            )
                            total[0] = (
                                total[0]
                                + incoming_a23[source]
                                * weight4[
                                    out_channel, input_channel, kernel_h, kernel_w
                                ]
                            )
                    sign_a20[flat] = T.if_then_else(
                        total[0] >= T.float32(0), T.int8(1), T.int8(0)
                    )

    return pack_a20


def _effective_pre17_primfunc():
    import tvm
    from tvm import te

    lower = te.placeholder((6, 3072), "float32", name="input_lower")
    upper = te.placeholder((6, 3072), "float32", name="input_upper")
    sign = te.placeholder((R31B1_ARENA_CAPACITY,), "int8", name="sign_ainput")
    weight = te.placeholder((8, 3, 3, 3), "float32", name="weight0")
    bias = te.placeholder((8,), "float32", name="bias0")
    reduce_conv = te.reduce_axis((0, 28), "reduce_conv")

    def value(flat):
        d_idx = flat // 2048
        logical = flat % 2048
        output_w = logical % 16
        output_h = logical // 16 % 16
        output_channel = logical // 256
        channel = tvm.tir.min(reduce_conv // 9, 2)
        kernel_h = reduce_conv // 3 % 3
        kernel_w = reduce_conv % 3
        input_h = output_h * 2 + kernel_h - 1
        input_w = output_w * 2 + kernel_w - 1
        valid = tvm.tir.all(
            reduce_conv < 27,
            input_h >= 0,
            input_h < 32,
            input_w >= 0,
            input_w < 32,
        )
        feature = channel * 1024 + input_h * 32 + input_w
        source = tvm.tir.if_then_else(
            sign[d_idx * 3072 + feature] != 0,
            lower[d_idx, feature],
            upper[d_idx, feature],
        )
        return te.sum(
            tvm.tir.if_then_else(
                reduce_conv == 27,
                bias[output_channel],
                tvm.tir.if_then_else(
                    valid,
                    source * weight[output_channel, channel, kernel_h, kernel_w],
                    tvm.tir.const(0.0, "float32"),
                ),
            ),
            axis=reduce_conv,
        )

    output = te.compute((12288,), value, name="effective_pre17")
    return (
        te.create_prim_func([lower, upper, sign, weight, bias, output])
        .with_attr("global_symbol", R31B2_EFFECTIVE_PRE17_SYMBOL)
        .with_attr("boundflow.schema_version", "r3-1b2-p-alpha-vjp/v1")
    )


def _effective_pre23_primfunc():  # pylint: disable=too-many-statements
    from tvm.script import tir as T

    @T.prim_func
    def effective_pre23(
        pre17: T.Buffer((12288,), "float32"),
        sign_a18: T.Buffer((12288,), "int8"),
        lower17: T.Buffer((6, 2048), "float32"),
        upper17: T.Buffer((6, 2048), "float32"),
        alpha17: T.Buffer((2, 1, 6, 164), "float32"),
        alpha_map17: T.Buffer((2048,), "int32"),
        weight2: T.Buffer((16, 8, 3, 3), "float32"),
        bias2: T.Buffer((16,), "float32"),
        sign_a20: T.Buffer((6144,), "int8"),
        lower19: T.Buffer((6, 1024), "float32"),
        upper19: T.Buffer((6, 1024), "float32"),
        alpha19: T.Buffer((2, 1, 6, 132), "float32"),
        alpha_map19: T.Buffer((1024,), "int32"),
        weight4: T.Buffer((16, 16, 3, 3), "float32"),
        bias4: T.Buffer((16,), "float32"),
        weight5: T.Buffer((16, 8, 1, 1), "float32"),
        bias5: T.Buffer((16,), "float32"),
        output: T.Buffer((6144,), "float32"),
    ):
        T.func_attr(
            {
                "global_symbol": R31B2_EFFECTIVE_PRE23_SYMBOL,
                "boundflow.schema_version": "r3-1b2-p-alpha-vjp/v1",
            }
        )
        total = T.alloc_buffer((1,), "float32", scope="local")
        pre19_value = T.alloc_buffer((1,), "float32", scope="local")
        for block in T.thread_binding(48, thread="blockIdx.x"):
            for thread in T.thread_binding(R31B1_THREADS, thread="threadIdx.x"):
                flat = block * R31B1_THREADS + thread
                if flat < 6144:
                    d = flat // 1024
                    logical = flat % 1024
                    out_w = logical % 8
                    out_h = logical // 8 % 8
                    out_channel = logical // 64
                    total[0] = bias4[out_channel] + bias5[out_channel]
                    shortcut_h = out_h * 2
                    shortcut_w = out_w * 2
                    for input_channel in range(8):
                        feature18 = input_channel * 256 + shortcut_h * 16 + shortcut_w
                        lower_alpha18 = T.if_then_else(
                            alpha_map17[feature18] >= 0,
                            T.min(
                                T.max(
                                    alpha17[
                                        0,
                                        0,
                                        d,
                                        T.max(alpha_map17[feature18], 0),
                                    ],
                                    T.float32(0),
                                ),
                                T.float32(1),
                            ),
                            T.float32(0),
                        )
                        ambiguous18 = lower17[d, feature18] < T.float32(0) and upper17[
                            d, feature18
                        ] > T.float32(0)
                        lower_slope18 = T.if_then_else(
                            ambiguous18,
                            lower_alpha18,
                            T.if_then_else(
                                lower17[d, feature18] >= T.float32(0),
                                T.float32(1),
                                T.float32(0),
                            ),
                        )
                        upper_slope18 = T.if_then_else(
                            lower17[d, feature18] >= T.float32(0),
                            T.float32(1),
                            T.if_then_else(
                                upper17[d, feature18] <= T.float32(0),
                                T.float32(0),
                                upper17[d, feature18]
                                / T.max(
                                    upper17[d, feature18] - lower17[d, feature18],
                                    T.float32(1.1920928955078125e-07),
                                ),
                            ),
                        )
                        slope18 = T.if_then_else(
                            sign_a18[d * 2048 + feature18] != T.int8(0),
                            lower_slope18,
                            upper_slope18,
                        )
                        intercept18 = T.if_then_else(
                            sign_a18[d * 2048 + feature18] == T.int8(0) and ambiguous18,
                            -lower17[d, feature18] * upper_slope18,
                            T.float32(0),
                        )
                        value18 = slope18 * pre17[d * 2048 + feature18] + intercept18
                        total[0] = (
                            total[0]
                            + value18 * weight5[out_channel, input_channel, 0, 0]
                        )
                    for mid_channel, kernel4_h, kernel4_w in T.grid(16, 3, 3):
                        mid_h = out_h + kernel4_h - 1
                        mid_w = out_w + kernel4_w - 1
                        if 0 <= mid_h and mid_h < 8 and 0 <= mid_w and mid_w < 8:
                            pre19_value[0] = bias2[mid_channel]
                            for input_channel, kernel2_h, kernel2_w in T.grid(8, 3, 3):
                                input_h = mid_h * 2 + kernel2_h - 1
                                input_w = mid_w * 2 + kernel2_w - 1
                                if (
                                    0 <= input_h
                                    and input_h < 16
                                    and 0 <= input_w
                                    and input_w < 16
                                ):
                                    feature18 = (
                                        input_channel * 256 + input_h * 16 + input_w
                                    )
                                    lookup18 = alpha_map17[feature18]
                                    lower_alpha18 = T.if_then_else(
                                        lookup18 >= 0,
                                        T.min(
                                            T.max(
                                                alpha17[0, 0, d, T.max(lookup18, 0)],
                                                T.float32(0),
                                            ),
                                            T.float32(1),
                                        ),
                                        T.float32(0),
                                    )
                                    ambiguous18 = lower17[d, feature18] < T.float32(
                                        0
                                    ) and upper17[d, feature18] > T.float32(0)
                                    lower_slope18 = T.if_then_else(
                                        ambiguous18,
                                        lower_alpha18,
                                        T.if_then_else(
                                            lower17[d, feature18] >= T.float32(0),
                                            T.float32(1),
                                            T.float32(0),
                                        ),
                                    )
                                    upper_slope18 = T.if_then_else(
                                        lower17[d, feature18] >= T.float32(0),
                                        T.float32(1),
                                        T.if_then_else(
                                            upper17[d, feature18] <= T.float32(0),
                                            T.float32(0),
                                            upper17[d, feature18]
                                            / T.max(
                                                upper17[d, feature18]
                                                - lower17[d, feature18],
                                                T.float32(1.1920928955078125e-07),
                                            ),
                                        ),
                                    )
                                    slope18 = T.if_then_else(
                                        sign_a18[d * 2048 + feature18] != T.int8(0),
                                        lower_slope18,
                                        upper_slope18,
                                    )
                                    intercept18 = T.if_then_else(
                                        sign_a18[d * 2048 + feature18] == T.int8(0)
                                        and ambiguous18,
                                        -lower17[d, feature18] * upper_slope18,
                                        T.float32(0),
                                    )
                                    value18 = (
                                        slope18 * pre17[d * 2048 + feature18]
                                        + intercept18
                                    )
                                    pre19_value[0] = (
                                        pre19_value[0]
                                        + value18
                                        * weight2[
                                            mid_channel,
                                            input_channel,
                                            kernel2_h,
                                            kernel2_w,
                                        ]
                                    )
                            feature20 = mid_channel * 64 + mid_h * 8 + mid_w
                            lookup20 = alpha_map19[feature20]
                            lower_alpha20 = T.if_then_else(
                                lookup20 >= 0,
                                T.min(
                                    T.max(
                                        alpha19[0, 0, d, T.max(lookup20, 0)],
                                        T.float32(0),
                                    ),
                                    T.float32(1),
                                ),
                                T.float32(0),
                            )
                            ambiguous20 = lower19[d, feature20] < T.float32(
                                0
                            ) and upper19[d, feature20] > T.float32(0)
                            lower_slope20 = T.if_then_else(
                                ambiguous20,
                                lower_alpha20,
                                T.if_then_else(
                                    lower19[d, feature20] >= T.float32(0),
                                    T.float32(1),
                                    T.float32(0),
                                ),
                            )
                            upper_slope20 = T.if_then_else(
                                lower19[d, feature20] >= T.float32(0),
                                T.float32(1),
                                T.if_then_else(
                                    upper19[d, feature20] <= T.float32(0),
                                    T.float32(0),
                                    upper19[d, feature20]
                                    / T.max(
                                        upper19[d, feature20] - lower19[d, feature20],
                                        T.float32(1.1920928955078125e-07),
                                    ),
                                ),
                            )
                            slope20 = T.if_then_else(
                                sign_a20[d * 1024 + feature20] != T.int8(0),
                                lower_slope20,
                                upper_slope20,
                            )
                            intercept20 = T.if_then_else(
                                sign_a20[d * 1024 + feature20] == T.int8(0)
                                and ambiguous20,
                                -lower19[d, feature20] * upper_slope20,
                                T.float32(0),
                            )
                            value20 = slope20 * pre19_value[0] + intercept20
                            total[0] = (
                                total[0]
                                + value20
                                * weight4[
                                    out_channel,
                                    mid_channel,
                                    kernel4_h,
                                    kernel4_w,
                                ]
                            )
                    output[flat] = total[0]

    return effective_pre23


def _effective_pre25_primfunc():
    import tvm
    from tvm import te

    pre23 = te.placeholder((6144,), "float32", name="pre23")
    sign = te.placeholder((6144,), "int8", name="sign_a24")
    lower = te.placeholder((6, 1024), "float32", name="lower23")
    upper = te.placeholder((6, 1024), "float32", name="upper23")
    alpha = te.placeholder((2, 1, 6, 121), "float32", name="alpha23")
    alpha_map = te.placeholder((1024,), "int32", name="alpha_map23")
    weight = te.placeholder((16, 16, 3, 3), "float32", name="weight8")
    bias = te.placeholder((16,), "float32", name="bias8")
    reduce_conv = te.reduce_axis((0, 145), "reduce_conv")

    def value(flat):
        d_idx = flat // 1024
        logical = flat % 1024
        out_w = logical % 8
        out_h = logical // 8 % 8
        out_channel = logical // 64
        input_channel = tvm.tir.min(reduce_conv // 9, 15)
        kernel_h = reduce_conv // 3 % 3
        kernel_w = reduce_conv % 3
        input_h = out_h + kernel_h - 1
        input_w = out_w + kernel_w - 1
        valid = tvm.tir.all(
            reduce_conv < 144,
            input_h >= 0,
            input_h < 8,
            input_w >= 0,
            input_w < 8,
        )
        feature = input_channel * 64 + input_h * 8 + input_w
        selected = _selected_relu_value(
            tvm,
            pre23[d_idx * 1024 + feature],
            sign[d_idx * 1024 + feature],
            lower,
            upper,
            alpha,
            alpha_map,
            d_idx,
            feature,
        )
        return te.sum(
            tvm.tir.if_then_else(
                reduce_conv == 144,
                bias[out_channel],
                tvm.tir.if_then_else(
                    valid,
                    selected * weight[out_channel, input_channel, kernel_h, kernel_w],
                    tvm.tir.const(0.0, "float32"),
                ),
            ),
            axis=reduce_conv,
        )

    output = te.compute((6144,), value, name="effective_pre25")
    return (
        te.create_prim_func(
            [pre23, sign, lower, upper, alpha, alpha_map, weight, bias, output]
        )
        .with_attr("global_symbol", R31B2_EFFECTIVE_PRE25_SYMBOL)
        .with_attr("boundflow.schema_version", "r3-1b2-p-alpha-vjp/v1")
    )


def _gradient_primfunc():
    import tvm
    from tvm import te

    a26 = te.placeholder((6144,), "float32", name="a26")
    pre25 = te.placeholder((6144,), "float32", name="pre25")
    lower = te.placeholder((6, 1024), "float32", name="lower25")
    upper = te.placeholder((6, 1024), "float32", name="upper25")
    indices = te.placeholder((86,), "int32", name="alpha_indices")
    grad_output = te.placeholder((6,), "float32", name="grad_output")

    def gradient(flat):
        direction = flat // 516
        remainder = flat % 516
        d_idx = remainder // 86
        feature = indices[remainder % 86]
        ambiguous = tvm.tir.all(
            lower[d_idx, feature] < tvm.tir.const(0.0, "float32"),
            upper[d_idx, feature] > tvm.tir.const(0.0, "float32"),
        )
        admitted = tvm.tir.all(
            direction == 0,
            ambiguous,
            a26[d_idx * 1024 + feature] >= tvm.tir.const(0.0, "float32"),
        )
        return tvm.tir.if_then_else(
            admitted,
            grad_output[d_idx]
            * a26[d_idx * 1024 + feature]
            * pre25[d_idx * 1024 + feature],
            tvm.tir.const(0.0, "float32"),
        )

    output = te.compute((1032,), gradient, name="compressed_gradient")
    return (
        te.create_prim_func([a26, pre25, lower, upper, indices, grad_output, output])
        .with_attr("global_symbol", R31B2_COMPRESSED_GRADIENT_SYMBOL)
        .with_attr("boundflow.schema_version", "r3-1b2-p-alpha-vjp/v1")
    )


def build_r31b2_p_alpha_vjp_tir_v1():
    import tvm

    conv10 = _conv_right_primfunc(
        R31B2_CONV10_RIGHT_SYMBOL,
        output_channels=16,
        input_channels=16,
        output_hw=(8, 8),
        input_hw=(8, 8),
        kernel_hw=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
    ).with_attr("boundflow.schema_version", "r3-1b2-p-alpha-vjp/v1")
    raw = {
        R31B2_CLEAR_SYMBOL: _clear_primfunc(),
        R31B2_PACK_A24_SYMBOL: _pack_sign_primfunc(R31B2_PACK_A24_SYMBOL, numel=6144),
        R31B2_PACK_A20_SYMBOL: _pack_a20_primfunc(),
        R31B2_PACK_A18_SYMBOL: _pack_sign_primfunc(R31B2_PACK_A18_SYMBOL, numel=12288),
        R31B2_PACK_AINPUT_SYMBOL: _pack_sign_primfunc(
            R31B2_PACK_AINPUT_SYMBOL, numel=R31B1_ARENA_CAPACITY
        ),
        R31B2_EFFECTIVE_PRE17_SYMBOL: _effective_pre17_primfunc(),
        R31B2_EFFECTIVE_PRE23_SYMBOL: _effective_pre23_primfunc(),
        R31B2_EFFECTIVE_PRE25_SYMBOL: _effective_pre25_primfunc(),
        R31B2_CONV10_RIGHT_SYMBOL: conv10,
        R31B2_COMPRESSED_GRADIENT_SYMBOL: _gradient_primfunc(),
    }
    scheduled = {
        R31B2_CLEAR_SYMBOL: raw[R31B2_CLEAR_SYMBOL],
        R31B2_PACK_A20_SYMBOL: raw[R31B2_PACK_A20_SYMBOL],
        R31B2_EFFECTIVE_PRE23_SYMBOL: raw[R31B2_EFFECTIVE_PRE23_SYMBOL],
    }
    for symbol, block, reduction in (
        (R31B2_PACK_A24_SYMBOL, "sign_bitmap", False),
        (R31B2_PACK_A18_SYMBOL, "sign_bitmap", False),
        (R31B2_PACK_AINPUT_SYMBOL, "sign_bitmap", False),
        (R31B2_EFFECTIVE_PRE17_SYMBOL, "effective_pre17", True),
        (R31B2_EFFECTIVE_PRE25_SYMBOL, "effective_pre25", True),
        (R31B2_COMPRESSED_GRADIENT_SYMBOL, "compressed_gradient", False),
    ):
        scheduled[symbol] = _schedule_te_primfunc(
            tvm, symbol, raw[symbol], ((block, reduction, R31B1_THREADS),)
        )
    scheduled[R31B2_CONV10_RIGHT_SYMBOL] = _schedule_te_primfunc(
        tvm,
        R31B2_CONV10_RIGHT_SYMBOL,
        raw[R31B2_CONV10_RIGHT_SYMBOL],
        (("conv_right_output", True, R31B1_THREADS), ("conv_right_bias", True, 1)),
    )
    return tvm.IRModule(scheduled)


def compile_r31b2_p_alpha_vjp_tir_v1(
    *, compute_capability: str = "sm_89"
) -> CompiledR31B2PAlphaVJPV1:
    import tvm

    module = build_r31b2_p_alpha_vjp_tir_v1()
    executable = tvm.compile(module, target=f"cuda -arch={compute_capability}")
    sources = tuple(imported.inspect_source() for imported in executable.mod.imports)
    if not sources:
        raise RuntimeError("R3-1b2 P-alpha VJP compile produced no CUDA source")
    source = "\n".join(sources)
    for symbol in R31B2_EXPORTED_SYMBOLS:
        if symbol not in source:
            raise RuntimeError(f"R3-1b2 compiled symbol is absent: {symbol}")
    return CompiledR31B2PAlphaVJPV1(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        module_hash=_sha256(tvm.ir.save_json(module)),
        device_source_hash=_sha256(source),
        device_source=source,
        exported_symbols=R31B2_EXPORTED_SYMBOLS,
        global_workspace_bytes=0,
        tvm_version=str(tvm.__version__),
    )


__all__ = [
    "CompiledR31B2PAlphaVJPV1",
    "R31B2_EXPORTED_SYMBOLS",
    "build_r31b2_p_alpha_vjp_tir_v1",
    "compile_r31b2_p_alpha_vjp_tir_v1",
]
