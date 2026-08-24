"""Frozen R3-1b1 lower-only ResNet2B recurrence for CUDA TVM TIR."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,invalid-name
# pylint: disable=missing-function-docstring,too-many-locals,too-many-statements
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=chained-comparison,too-many-nested-blocks,too-many-branches

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)

R31B1_ARENA_CAPACITY = 18_432
R31B1_THREADS = 128

R31B1_SEED_SYMBOL = "boundflow_r31b1_seed"
R31B1_LINEAR16_SYMBOL = "boundflow_r31b1_linear16"
R31B1_RELU31_BIAS_SYMBOL = "boundflow_r31b1_relu31_bias"
R31B1_RELU31_COEFF_SYMBOL = "boundflow_r31b1_relu31_coeff"
R31B1_LINEAR14_SYMBOL = "boundflow_r31b1_linear14"
R31B1_RELU28_BIAS_SYMBOL = "boundflow_r31b1_relu28_bias"
R31B1_RELU28_COEFF_SYMBOL = "boundflow_r31b1_relu28_coeff"
R31B1_RESIDUAL11_SYMBOL = "boundflow_r31b1_residual11"
R31B1_RELU23_BIAS_SYMBOL = "boundflow_r31b1_relu23_bias"
R31B1_RELU23_COEFF_SYMBOL = "boundflow_r31b1_relu23_coeff"
R31B1_RESIDUAL6_SYMBOL = "boundflow_r31b1_residual6"
R31B1_RELU17_BIAS_SYMBOL = "boundflow_r31b1_relu17_bias"
R31B1_RELU17_COEFF_SYMBOL = "boundflow_r31b1_relu17_coeff"
R31B1_CONV0_SYMBOL = "boundflow_r31b1_conv0"
R31B1_CONCRETIZE_SYMBOL = "boundflow_r31b1_concretize"

R31B1_EXPORTED_SYMBOLS = (
    R31B1_SEED_SYMBOL,
    R31B1_LINEAR16_SYMBOL,
    R31B1_RELU31_BIAS_SYMBOL,
    R31B1_RELU31_COEFF_SYMBOL,
    R31B1_LINEAR14_SYMBOL,
    R31B1_RELU28_BIAS_SYMBOL,
    R31B1_RELU28_COEFF_SYMBOL,
    R31B1_RESIDUAL11_SYMBOL,
    R31B1_RELU23_BIAS_SYMBOL,
    R31B1_RELU23_COEFF_SYMBOL,
    R31B1_RESIDUAL6_SYMBOL,
    R31B1_RELU17_BIAS_SYMBOL,
    R31B1_RELU17_COEFF_SYMBOL,
    R31B1_CONV0_SYMBOL,
    R31B1_CONCRETIZE_SYMBOL,
)


@dataclass(frozen=True)
class CompiledR31B1FullLowerForwardV1:
    """Tensor-free compiled module and its exact compiler identities."""

    executable: DifferentiableLowerTIRExecutable
    module_hash: str
    device_source_hash: str
    device_source: str
    exported_symbols: tuple[str, ...]
    global_workspace_bytes: int
    tvm_version: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relu_terms(tvm, incoming, lower, upper, alpha, alpha_map, d_idx, f_idx):
    zero = tvm.tir.const(0.0, "float32")
    one = tvm.tir.const(1.0, "float32")
    epsilon = tvm.tir.const(1.1920928955078125e-07, "float32")
    lookup = alpha_map[f_idx]
    compact = alpha[0, 0, d_idx, tvm.tir.max(lookup, 0)]
    lower_alpha = tvm.tir.if_then_else(
        lookup >= 0, tvm.tir.min(tvm.tir.max(compact, zero), one), zero
    )
    ambiguous = tvm.tir.all(lower[d_idx, f_idx] < zero, upper[d_idx, f_idx] > zero)
    lower_slope = tvm.tir.if_then_else(
        ambiguous,
        lower_alpha,
        tvm.tir.if_then_else(lower[d_idx, f_idx] >= zero, one, zero),
    )
    upper_slope = tvm.tir.if_then_else(
        lower[d_idx, f_idx] >= zero,
        one,
        tvm.tir.if_then_else(
            upper[d_idx, f_idx] <= zero,
            zero,
            upper[d_idx, f_idx]
            / tvm.tir.max(upper[d_idx, f_idx] - lower[d_idx, f_idx], epsilon),
        ),
    )
    selected_slope = tvm.tir.if_then_else(incoming >= zero, lower_slope, upper_slope)
    selected_intercept = tvm.tir.if_then_else(
        tvm.tir.all(incoming < zero, ambiguous),
        -lower[d_idx, f_idx] * upper_slope,
        zero,
    )
    return selected_slope, selected_intercept


def _seed_primfunc():
    import tvm
    from tvm import te

    objective = te.placeholder((6, 1, 10), "float32", name="objective")
    output = te.compute(
        (60,), lambda flat: objective[flat // 10, 0, flat % 10], name="seed_output"
    )
    bias = te.compute((6,), lambda _d: tvm.tir.const(0.0, "float32"), name="seed_bias")
    return (
        te.create_prim_func([objective, output, bias])
        .with_attr("global_symbol", R31B1_SEED_SYMBOL)
        .with_attr("boundflow.schema_version", "r3-1b1-full-lower-forward/v1")
    )


def _linear_primfunc(symbol: str, *, current_features: int, previous_features: int):
    import tvm
    from tvm import te

    incoming = te.placeholder((6 * current_features,), "float32", name="incoming_arena")
    weight = te.placeholder(
        (current_features, previous_features), "float32", name="weight"
    )
    operator_bias = te.placeholder((current_features,), "float32", name="operator_bias")
    bias_in = te.placeholder((6,), "float32", name="bias_in")
    reduce_current = te.reduce_axis((0, current_features), "reduce_current")
    output = te.compute(
        (6 * previous_features,),
        lambda flat: te.sum(
            incoming[(flat // previous_features) * current_features + reduce_current]
            * weight[reduce_current, flat % previous_features],
            axis=reduce_current,
        ),
        name="linear_output",
    )
    reduce_bias = te.reduce_axis((0, current_features + 1), "reduce_bias")
    bias_out = te.compute(
        (6,),
        lambda d_idx: te.sum(
            tvm.tir.if_then_else(
                reduce_bias == current_features,
                bias_in[d_idx],
                incoming[
                    d_idx * current_features
                    + tvm.tir.min(reduce_bias, current_features - 1)
                ]
                * operator_bias[tvm.tir.min(reduce_bias, current_features - 1)],
            ),
            axis=reduce_bias,
        ),
        name="linear_bias",
    )
    return (
        te.create_prim_func(
            [incoming, weight, operator_bias, bias_in, output, bias_out]
        )
        .with_attr("global_symbol", symbol)
        .with_attr("boundflow.schema_version", "r3-1b1-full-lower-forward/v1")
    )


def _relu_bias_primfunc(symbol: str, *, feature_count: int, alpha_width: int):
    import tvm
    from tvm import te

    incoming = te.placeholder((6 * feature_count,), "float32", name="incoming_arena")
    lower = te.placeholder((6, feature_count), "float32", name="lower")
    upper = te.placeholder((6, feature_count), "float32", name="upper")
    alpha = te.placeholder((2, 1, 6, alpha_width), "float32", name="alpha")
    alpha_map = te.placeholder((feature_count,), "int32", name="alpha_map")
    bias_in = te.placeholder((6,), "float32", name="bias_in")
    reduce_feature = te.reduce_axis((0, feature_count + 1), "reduce_feature")

    def contribution(d_idx):
        safe_feature = tvm.tir.min(reduce_feature, feature_count - 1)
        value = incoming[d_idx * feature_count + safe_feature]
        _slope, intercept = _relu_terms(
            tvm,
            value,
            lower,
            upper,
            alpha,
            alpha_map,
            d_idx,
            safe_feature,
        )
        return tvm.tir.if_then_else(
            reduce_feature == feature_count,
            bias_in[d_idx],
            value * intercept,
        )

    bias_out = te.compute(
        (6,),
        lambda d_idx: te.sum(contribution(d_idx), axis=reduce_feature),
        name="relu_bias",
    )
    return (
        te.create_prim_func(
            [incoming, lower, upper, alpha, alpha_map, bias_in, bias_out]
        )
        .with_attr("global_symbol", symbol)
        .with_attr("boundflow.schema_version", "r3-1b1-full-lower-forward/v1")
    )


def _relu_coeff_primfunc(
    symbol: str,
    *,
    feature_count: int,
    alpha_width: int,
    beta_active: bool = False,
):
    import tvm
    from tvm import te

    incoming = te.placeholder((6 * feature_count,), "float32", name="incoming_arena")
    lower = te.placeholder((6, feature_count), "float32", name="lower")
    upper = te.placeholder((6, feature_count), "float32", name="upper")
    alpha = te.placeholder((2, 1, 6, alpha_width), "float32", name="alpha")
    alpha_map = te.placeholder((feature_count,), "int32", name="alpha_map")
    beta = te.placeholder((6, 1), "float32", name="beta") if beta_active else None
    beta_map = (
        te.placeholder((6, feature_count), "int32", name="beta_map")
        if beta_active
        else None
    )
    split = (
        te.placeholder((6, feature_count), "int8", name="split")
        if beta_active
        else None
    )

    def coefficient(flat):
        d_idx = flat // feature_count
        f_idx = flat % feature_count
        value = incoming[flat]
        slope, _intercept = _relu_terms(
            tvm, value, lower, upper, alpha, alpha_map, d_idx, f_idx
        )
        beta_add = tvm.tir.const(0.0, "float32")
        if beta_active:
            assert beta is not None and beta_map is not None and split is not None
            location = beta_map[d_idx, f_idx]
            beta_add = tvm.tir.if_then_else(
                location >= 0,
                -beta[d_idx, tvm.tir.max(location, 0)]
                * tvm.tir.Cast("float32", split[d_idx, f_idx]),
                tvm.tir.const(0.0, "float32"),
            )
        return value * slope + beta_add

    output = te.compute((6 * feature_count,), coefficient, name="relu_coefficient")
    arguments = [incoming, lower, upper, alpha, alpha_map]
    if beta_active:
        assert beta is not None and beta_map is not None and split is not None
        arguments.extend((beta, beta_map, split))
    arguments.append(output)
    return (
        te.create_prim_func(arguments)
        .with_attr("global_symbol", symbol)
        .with_attr("boundflow.schema_version", "r3-1b1-full-lower-forward/v1")
    )


def _conv_right_primfunc(
    symbol: str,
    *,
    output_channels: int,
    input_channels: int,
    output_hw: tuple[int, int],
    input_hw: tuple[int, int],
    kernel_hw: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
):
    import tvm
    from tvm import te

    input_feature_count = input_channels * input_hw[0] * input_hw[1]
    output_feature_count = output_channels * output_hw[0] * output_hw[1]
    incoming = te.placeholder(
        (6 * output_feature_count,), "float32", name="incoming_arena"
    )
    weight = te.placeholder(
        (output_channels, input_channels, *kernel_hw), "float32", name="weight"
    )
    operator_bias = te.placeholder((output_channels,), "float32", name="operator_bias")
    bias_in = te.placeholder((6,), "float32", name="bias_in")
    reduction_extent = output_channels * kernel_hw[0] * kernel_hw[1]
    reduce_conv = te.reduce_axis((0, reduction_extent), "reduce_conv")

    def output_value(flat):
        d_idx = flat // input_feature_count
        logical = flat % input_feature_count
        input_w = logical % input_hw[1]
        input_h = logical // input_hw[1] % input_hw[0]
        input_channel = logical // (input_hw[0] * input_hw[1])
        kernel_w = reduce_conv % kernel_hw[1]
        kernel_h = reduce_conv // kernel_hw[1] % kernel_hw[0]
        output_channel = reduce_conv // (kernel_hw[0] * kernel_hw[1])
        numerator_h = input_h + padding[0] - kernel_h
        numerator_w = input_w + padding[1] - kernel_w
        output_h = numerator_h // stride[0]
        output_w = numerator_w // stride[1]
        valid = tvm.tir.all(
            numerator_h >= 0,
            numerator_w >= 0,
            numerator_h % stride[0] == 0,
            numerator_w % stride[1] == 0,
            output_h < output_hw[0],
            output_w < output_hw[1],
        )
        source_index = (
            d_idx * output_feature_count
            + output_channel * output_hw[0] * output_hw[1]
            + output_h * output_hw[1]
            + output_w
        )
        term = tvm.tir.if_then_else(
            valid,
            incoming[source_index]
            * weight[output_channel, input_channel, kernel_h, kernel_w],
            tvm.tir.const(0.0, "float32"),
        )
        return te.sum(term, axis=reduce_conv)

    output = te.compute(
        (6 * input_feature_count,), output_value, name="conv_right_output"
    )
    reduce_bias = te.reduce_axis((0, output_feature_count + 1), "reduce_bias")
    bias_out = te.compute(
        (6,),
        lambda d_idx: te.sum(
            tvm.tir.if_then_else(
                reduce_bias == output_feature_count,
                bias_in[d_idx],
                incoming[d_idx * output_feature_count + reduce_bias]
                * operator_bias[
                    tvm.tir.min(reduce_bias, output_feature_count - 1)
                    // (output_hw[0] * output_hw[1])
                ],
            ),
            axis=reduce_bias,
        ),
        name="conv_right_bias",
    )
    return (
        te.create_prim_func(
            [incoming, weight, operator_bias, bias_in, output, bias_out]
        )
        .with_attr("global_symbol", symbol)
        .with_attr("boundflow.schema_version", "r3-1b1-full-lower-forward/v1")
    )


def _residual11_primfunc():  # pylint: disable=too-many-statements
    from tvm.script import tir as T

    @T.prim_func
    def residual11(
        incoming: T.Buffer((R31B1_ARENA_CAPACITY,), "float32"),
        weight10: T.Buffer((16, 16, 3, 3), "float32"),
        bias10: T.Buffer((16,), "float32"),
        lower25: T.Buffer((6, 1024), "float32"),
        upper25: T.Buffer((6, 1024), "float32"),
        alpha25: T.Buffer((2, 1, 6, 86), "float32"),
        alpha_map25: T.Buffer((1024,), "int32"),
        weight8: T.Buffer((16, 16, 3, 3), "float32"),
        bias8: T.Buffer((16,), "float32"),
        bias_acc: T.Buffer((6,), "float32"),
        output: T.Buffer((R31B1_ARENA_CAPACITY,), "float32"),
    ):
        T.func_attr(
            {
                "global_symbol": R31B1_RESIDUAL11_SYMBOL,
                "boundflow.schema_version": "r3-1b1-full-lower-forward/v1",
            }
        )
        total = T.alloc_buffer((1,), "float32", scope="local")
        a25 = T.alloc_buffer((1,), "float32", scope="local")
        delta = T.alloc_buffer((1,), "float32", scope="local")
        a25_bias = T.alloc_buffer((1,), "float32", scope="local")
        for block in T.thread_binding(1, thread="blockIdx.x"):
            for thread in T.thread_binding(R31B1_THREADS, thread="threadIdx.x"):
                T.evaluate(block)
                for flat in T.serial(thread, 6144, step=R31B1_THREADS):
                    d = flat // 1024
                    logical = flat % 1024
                    input_w = logical % 8
                    input_h = logical // 8 % 8
                    input_channel = logical // 64
                    total[0] = incoming[flat]
                    for mid_channel, kernel8_h, kernel8_w in T.grid(16, 3, 3):
                        mid_h = input_h + 1 - kernel8_h
                        mid_w = input_w + 1 - kernel8_w
                        if 0 <= mid_h and mid_h < 8 and 0 <= mid_w and mid_w < 8:
                            a25[0] = T.float32(0)
                            for out_channel, kernel10_h, kernel10_w in T.grid(16, 3, 3):
                                out_h = mid_h + 1 - kernel10_h
                                out_w = mid_w + 1 - kernel10_w
                                if (
                                    0 <= out_h
                                    and out_h < 8
                                    and 0 <= out_w
                                    and out_w < 8
                                ):
                                    source = (
                                        d * 1024 + out_channel * 64 + out_h * 8 + out_w
                                    )
                                    a25[0] = (
                                        a25[0]
                                        + incoming[source]
                                        * weight10[
                                            out_channel,
                                            mid_channel,
                                            kernel10_h,
                                            kernel10_w,
                                        ]
                                    )
                            feature = mid_channel * 64 + mid_h * 8 + mid_w
                            lookup = alpha_map25[feature]
                            lower_alpha = T.if_then_else(
                                lookup >= 0,
                                T.min(
                                    T.max(
                                        alpha25[0, 0, d, T.max(lookup, 0)], T.float32(0)
                                    ),
                                    T.float32(1),
                                ),
                                T.float32(0),
                            )
                            ambiguous = lower25[d, feature] < T.float32(0) and upper25[
                                d, feature
                            ] > T.float32(0)
                            lower_slope = T.if_then_else(
                                ambiguous,
                                lower_alpha,
                                T.if_then_else(
                                    lower25[d, feature] >= T.float32(0),
                                    T.float32(1),
                                    T.float32(0),
                                ),
                            )
                            upper_slope = T.if_then_else(
                                lower25[d, feature] >= T.float32(0),
                                T.float32(1),
                                T.if_then_else(
                                    upper25[d, feature] <= T.float32(0),
                                    T.float32(0),
                                    upper25[d, feature]
                                    / T.max(
                                        upper25[d, feature] - lower25[d, feature],
                                        T.float32(1.1920928955078125e-07),
                                    ),
                                ),
                            )
                            slope = T.if_then_else(
                                a25[0] >= T.float32(0), lower_slope, upper_slope
                            )
                            total[0] = (
                                total[0]
                                + a25[0]
                                * slope
                                * weight8[
                                    mid_channel, input_channel, kernel8_h, kernel8_w
                                ]
                            )
                    output[flat] = total[0]
                if thread < 6:
                    d = thread
                    delta[0] = T.float32(0)
                    for out_channel, out_h, out_w in T.grid(16, 8, 8):
                        source = d * 1024 + out_channel * 64 + out_h * 8 + out_w
                        delta[0] = delta[0] + incoming[source] * bias10[out_channel]
                    for mid_channel, mid_h, mid_w in T.grid(16, 8, 8):
                        a25_bias[0] = T.float32(0)
                        for out_channel, kernel10_h, kernel10_w in T.grid(16, 3, 3):
                            out_h = mid_h + 1 - kernel10_h
                            out_w = mid_w + 1 - kernel10_w
                            if 0 <= out_h and out_h < 8 and 0 <= out_w and out_w < 8:
                                source = d * 1024 + out_channel * 64 + out_h * 8 + out_w
                                a25_bias[0] = (
                                    a25_bias[0]
                                    + incoming[source]
                                    * weight10[
                                        out_channel, mid_channel, kernel10_h, kernel10_w
                                    ]
                                )
                        feature = mid_channel * 64 + mid_h * 8 + mid_w
                        lookup = alpha_map25[feature]
                        lower_alpha = T.if_then_else(
                            lookup >= 0,
                            T.min(
                                T.max(alpha25[0, 0, d, T.max(lookup, 0)], T.float32(0)),
                                T.float32(1),
                            ),
                            T.float32(0),
                        )
                        ambiguous = lower25[d, feature] < T.float32(0) and upper25[
                            d, feature
                        ] > T.float32(0)
                        lower_slope = T.if_then_else(
                            ambiguous,
                            lower_alpha,
                            T.if_then_else(
                                lower25[d, feature] >= T.float32(0),
                                T.float32(1),
                                T.float32(0),
                            ),
                        )
                        upper_slope = T.if_then_else(
                            lower25[d, feature] >= T.float32(0),
                            T.float32(1),
                            T.if_then_else(
                                upper25[d, feature] <= T.float32(0),
                                T.float32(0),
                                upper25[d, feature]
                                / T.max(
                                    upper25[d, feature] - lower25[d, feature],
                                    T.float32(1.1920928955078125e-07),
                                ),
                            ),
                        )
                        slope = T.if_then_else(
                            a25_bias[0] >= T.float32(0), lower_slope, upper_slope
                        )
                        intercept = T.if_then_else(
                            a25_bias[0] < T.float32(0) and ambiguous,
                            -lower25[d, feature] * upper_slope,
                            T.float32(0),
                        )
                        delta[0] = (
                            delta[0]
                            + a25_bias[0] * intercept
                            + a25_bias[0] * slope * bias8[mid_channel]
                        )
                    bias_acc[d] = bias_acc[d] + delta[0]

    return residual11


def _residual6_primfunc():  # pylint: disable=too-many-statements
    from tvm.script import tir as T

    @T.prim_func
    def residual6(
        incoming: T.Buffer((R31B1_ARENA_CAPACITY,), "float32"),
        weight4: T.Buffer((16, 16, 3, 3), "float32"),
        bias4: T.Buffer((16,), "float32"),
        lower19: T.Buffer((6, 1024), "float32"),
        upper19: T.Buffer((6, 1024), "float32"),
        alpha19: T.Buffer((2, 1, 6, 132), "float32"),
        alpha_map19: T.Buffer((1024,), "int32"),
        weight2: T.Buffer((16, 8, 3, 3), "float32"),
        bias2: T.Buffer((16,), "float32"),
        weight5: T.Buffer((16, 8, 1, 1), "float32"),
        bias5: T.Buffer((16,), "float32"),
        bias_acc: T.Buffer((6,), "float32"),
        output: T.Buffer((R31B1_ARENA_CAPACITY,), "float32"),
    ):
        T.func_attr(
            {
                "global_symbol": R31B1_RESIDUAL6_SYMBOL,
                "boundflow.schema_version": "r3-1b1-full-lower-forward/v1",
            }
        )
        total = T.alloc_buffer((1,), "float32", scope="local")
        a20 = T.alloc_buffer((1,), "float32", scope="local")
        delta = T.alloc_buffer((1,), "float32", scope="local")
        a20_bias = T.alloc_buffer((1,), "float32", scope="local")
        for block in T.thread_binding(1, thread="blockIdx.x"):
            for thread in T.thread_binding(R31B1_THREADS, thread="threadIdx.x"):
                T.evaluate(block)
                for flat in T.serial(thread, 12288, step=R31B1_THREADS):
                    d = flat // 2048
                    logical = flat % 2048
                    input_w = logical % 16
                    input_h = logical // 16 % 16
                    input_channel = logical // 256
                    total[0] = T.float32(0)
                    for mid_channel, kernel2_h, kernel2_w in T.grid(16, 3, 3):
                        numerator_h = input_h + 1 - kernel2_h
                        numerator_w = input_w + 1 - kernel2_w
                        if (
                            numerator_h >= 0
                            and numerator_w >= 0
                            and numerator_h % 2 == 0
                            and numerator_w % 2 == 0
                        ):
                            mid_h = numerator_h // 2
                            mid_w = numerator_w // 2
                            if mid_h < 8 and mid_w < 8:
                                a20[0] = T.float32(0)
                                for out_channel, kernel4_h, kernel4_w in T.grid(
                                    16, 3, 3
                                ):
                                    out_h = mid_h + 1 - kernel4_h
                                    out_w = mid_w + 1 - kernel4_w
                                    if (
                                        0 <= out_h
                                        and out_h < 8
                                        and 0 <= out_w
                                        and out_w < 8
                                    ):
                                        source = (
                                            d * 1024
                                            + out_channel * 64
                                            + out_h * 8
                                            + out_w
                                        )
                                        a20[0] = (
                                            a20[0]
                                            + incoming[source]
                                            * weight4[
                                                out_channel,
                                                mid_channel,
                                                kernel4_h,
                                                kernel4_w,
                                            ]
                                        )
                                feature = mid_channel * 64 + mid_h * 8 + mid_w
                                lookup = alpha_map19[feature]
                                lower_alpha = T.if_then_else(
                                    lookup >= 0,
                                    T.min(
                                        T.max(
                                            alpha19[0, 0, d, T.max(lookup, 0)],
                                            T.float32(0),
                                        ),
                                        T.float32(1),
                                    ),
                                    T.float32(0),
                                )
                                ambiguous = lower19[d, feature] < T.float32(
                                    0
                                ) and upper19[d, feature] > T.float32(0)
                                lower_slope = T.if_then_else(
                                    ambiguous,
                                    lower_alpha,
                                    T.if_then_else(
                                        lower19[d, feature] >= T.float32(0),
                                        T.float32(1),
                                        T.float32(0),
                                    ),
                                )
                                upper_slope = T.if_then_else(
                                    lower19[d, feature] >= T.float32(0),
                                    T.float32(1),
                                    T.if_then_else(
                                        upper19[d, feature] <= T.float32(0),
                                        T.float32(0),
                                        upper19[d, feature]
                                        / T.max(
                                            upper19[d, feature] - lower19[d, feature],
                                            T.float32(1.1920928955078125e-07),
                                        ),
                                    ),
                                )
                                slope = T.if_then_else(
                                    a20[0] >= T.float32(0), lower_slope, upper_slope
                                )
                                total[0] = (
                                    total[0]
                                    + a20[0]
                                    * slope
                                    * weight2[
                                        mid_channel, input_channel, kernel2_h, kernel2_w
                                    ]
                                )
                    if input_h % 2 == 0 and input_w % 2 == 0:
                        out_h = input_h // 2
                        out_w = input_w // 2
                        for out_channel in range(16):
                            source = d * 1024 + out_channel * 64 + out_h * 8 + out_w
                            total[0] = (
                                total[0]
                                + incoming[source]
                                * weight5[out_channel, input_channel, 0, 0]
                            )
                    output[flat] = total[0]
                if thread < 6:
                    d = thread
                    delta[0] = T.float32(0)
                    for out_channel, out_h, out_w in T.grid(16, 8, 8):
                        source = d * 1024 + out_channel * 64 + out_h * 8 + out_w
                        delta[0] = delta[0] + incoming[source] * (
                            bias4[out_channel] + bias5[out_channel]
                        )
                    for mid_channel, mid_h, mid_w in T.grid(16, 8, 8):
                        a20_bias[0] = T.float32(0)
                        for out_channel, kernel4_h, kernel4_w in T.grid(16, 3, 3):
                            out_h = mid_h + 1 - kernel4_h
                            out_w = mid_w + 1 - kernel4_w
                            if 0 <= out_h and out_h < 8 and 0 <= out_w and out_w < 8:
                                source = d * 1024 + out_channel * 64 + out_h * 8 + out_w
                                a20_bias[0] = (
                                    a20_bias[0]
                                    + incoming[source]
                                    * weight4[
                                        out_channel, mid_channel, kernel4_h, kernel4_w
                                    ]
                                )
                        feature = mid_channel * 64 + mid_h * 8 + mid_w
                        lookup = alpha_map19[feature]
                        lower_alpha = T.if_then_else(
                            lookup >= 0,
                            T.min(
                                T.max(alpha19[0, 0, d, T.max(lookup, 0)], T.float32(0)),
                                T.float32(1),
                            ),
                            T.float32(0),
                        )
                        ambiguous = lower19[d, feature] < T.float32(0) and upper19[
                            d, feature
                        ] > T.float32(0)
                        lower_slope = T.if_then_else(
                            ambiguous,
                            lower_alpha,
                            T.if_then_else(
                                lower19[d, feature] >= T.float32(0),
                                T.float32(1),
                                T.float32(0),
                            ),
                        )
                        upper_slope = T.if_then_else(
                            lower19[d, feature] >= T.float32(0),
                            T.float32(1),
                            T.if_then_else(
                                upper19[d, feature] <= T.float32(0),
                                T.float32(0),
                                upper19[d, feature]
                                / T.max(
                                    upper19[d, feature] - lower19[d, feature],
                                    T.float32(1.1920928955078125e-07),
                                ),
                            ),
                        )
                        slope = T.if_then_else(
                            a20_bias[0] >= T.float32(0), lower_slope, upper_slope
                        )
                        intercept = T.if_then_else(
                            a20_bias[0] < T.float32(0) and ambiguous,
                            -lower19[d, feature] * upper_slope,
                            T.float32(0),
                        )
                        delta[0] = (
                            delta[0]
                            + a20_bias[0] * intercept
                            + a20_bias[0] * slope * bias2[mid_channel]
                        )
                    bias_acc[d] = bias_acc[d] + delta[0]

    return residual6


def _concretize_primfunc():
    import tvm
    from tvm import te

    incoming = te.placeholder((R31B1_ARENA_CAPACITY,), "float32", name="incoming_arena")
    input_lower = te.placeholder((6, 3072), "float32", name="input_lower")
    input_upper = te.placeholder((6, 3072), "float32", name="input_upper")
    bias = te.placeholder((6,), "float32", name="bias")
    reduce_input = te.reduce_axis((0, 3073), "reduce_input")
    lower = te.compute(
        (6,),
        lambda d_idx: te.sum(
            tvm.tir.if_then_else(
                reduce_input == 3072,
                bias[d_idx],
                tvm.tir.if_then_else(
                    incoming[d_idx * 3072 + tvm.tir.min(reduce_input, 3071)] >= 0,
                    incoming[d_idx * 3072 + tvm.tir.min(reduce_input, 3071)]
                    * input_lower[d_idx, tvm.tir.min(reduce_input, 3071)],
                    incoming[d_idx * 3072 + tvm.tir.min(reduce_input, 3071)]
                    * input_upper[d_idx, tvm.tir.min(reduce_input, 3071)],
                ),
            ),
            axis=reduce_input,
        ),
        name="concretized_lower",
    )
    return (
        te.create_prim_func([incoming, input_lower, input_upper, bias, lower])
        .with_attr("global_symbol", R31B1_CONCRETIZE_SYMBOL)
        .with_attr("boundflow.schema_version", "r3-1b1-full-lower-forward/v1")
    )


def _schedule_te_primfunc(tvm, symbol: str, primfunc, blocks):
    schedule = tvm.tir.Schedule(tvm.IRModule({symbol: primfunc}))
    for block_name, reduction, thread_extent in blocks:
        block = schedule.get_block(block_name, func_name=symbol)
        loops = schedule.get_loops(block)
        spatial = loops[:-1] if reduction else loops
        fused = schedule.fuse(*spatial) if len(spatial) > 1 else spatial[0]
        outer, inner = schedule.split(fused, factors=[None, thread_extent])
        schedule.bind(outer, "blockIdx.x")
        schedule.bind(inner, "threadIdx.x")
    return schedule.mod[symbol]


def build_r31b1_full_lower_forward_tir_v1():
    """Build all 12 recurrence steps; only residual regions use hand-written TIR."""

    import tvm

    raw = {
        R31B1_SEED_SYMBOL: _seed_primfunc(),
        R31B1_LINEAR16_SYMBOL: _linear_primfunc(
            R31B1_LINEAR16_SYMBOL, current_features=10, previous_features=100
        ),
        R31B1_RELU31_BIAS_SYMBOL: _relu_bias_primfunc(
            R31B1_RELU31_BIAS_SYMBOL, feature_count=100, alpha_width=27
        ),
        R31B1_RELU31_COEFF_SYMBOL: _relu_coeff_primfunc(
            R31B1_RELU31_COEFF_SYMBOL,
            feature_count=100,
            alpha_width=27,
            beta_active=True,
        ),
        R31B1_LINEAR14_SYMBOL: _linear_primfunc(
            R31B1_LINEAR14_SYMBOL, current_features=100, previous_features=1024
        ),
        R31B1_RELU28_BIAS_SYMBOL: _relu_bias_primfunc(
            R31B1_RELU28_BIAS_SYMBOL, feature_count=1024, alpha_width=178
        ),
        R31B1_RELU28_COEFF_SYMBOL: _relu_coeff_primfunc(
            R31B1_RELU28_COEFF_SYMBOL, feature_count=1024, alpha_width=178
        ),
        R31B1_RESIDUAL11_SYMBOL: _residual11_primfunc(),
        R31B1_RELU23_BIAS_SYMBOL: _relu_bias_primfunc(
            R31B1_RELU23_BIAS_SYMBOL, feature_count=1024, alpha_width=121
        ),
        R31B1_RELU23_COEFF_SYMBOL: _relu_coeff_primfunc(
            R31B1_RELU23_COEFF_SYMBOL, feature_count=1024, alpha_width=121
        ),
        R31B1_RESIDUAL6_SYMBOL: _residual6_primfunc(),
        R31B1_RELU17_BIAS_SYMBOL: _relu_bias_primfunc(
            R31B1_RELU17_BIAS_SYMBOL, feature_count=2048, alpha_width=164
        ),
        R31B1_RELU17_COEFF_SYMBOL: _relu_coeff_primfunc(
            R31B1_RELU17_COEFF_SYMBOL, feature_count=2048, alpha_width=164
        ),
        R31B1_CONV0_SYMBOL: _conv_right_primfunc(
            R31B1_CONV0_SYMBOL,
            output_channels=8,
            input_channels=3,
            output_hw=(16, 16),
            input_hw=(32, 32),
            kernel_hw=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        ),
        R31B1_CONCRETIZE_SYMBOL: _concretize_primfunc(),
    }
    scheduled = {
        R31B1_SEED_SYMBOL: _schedule_te_primfunc(
            tvm,
            R31B1_SEED_SYMBOL,
            raw[R31B1_SEED_SYMBOL],
            (("seed_output", False, R31B1_THREADS), ("seed_bias", False, 1)),
        ),
        R31B1_LINEAR16_SYMBOL: _schedule_te_primfunc(
            tvm,
            R31B1_LINEAR16_SYMBOL,
            raw[R31B1_LINEAR16_SYMBOL],
            (("linear_output", True, R31B1_THREADS), ("linear_bias", True, 1)),
        ),
        R31B1_RELU31_BIAS_SYMBOL: _schedule_te_primfunc(
            tvm,
            R31B1_RELU31_BIAS_SYMBOL,
            raw[R31B1_RELU31_BIAS_SYMBOL],
            (("relu_bias", True, 1),),
        ),
        R31B1_RELU31_COEFF_SYMBOL: _schedule_te_primfunc(
            tvm,
            R31B1_RELU31_COEFF_SYMBOL,
            raw[R31B1_RELU31_COEFF_SYMBOL],
            (("relu_coefficient", False, R31B1_THREADS),),
        ),
        R31B1_LINEAR14_SYMBOL: _schedule_te_primfunc(
            tvm,
            R31B1_LINEAR14_SYMBOL,
            raw[R31B1_LINEAR14_SYMBOL],
            (("linear_output", True, R31B1_THREADS), ("linear_bias", True, 1)),
        ),
        R31B1_RELU28_BIAS_SYMBOL: _schedule_te_primfunc(
            tvm,
            R31B1_RELU28_BIAS_SYMBOL,
            raw[R31B1_RELU28_BIAS_SYMBOL],
            (("relu_bias", True, 1),),
        ),
        R31B1_RELU28_COEFF_SYMBOL: _schedule_te_primfunc(
            tvm,
            R31B1_RELU28_COEFF_SYMBOL,
            raw[R31B1_RELU28_COEFF_SYMBOL],
            (("relu_coefficient", False, R31B1_THREADS),),
        ),
        R31B1_RELU23_BIAS_SYMBOL: _schedule_te_primfunc(
            tvm,
            R31B1_RELU23_BIAS_SYMBOL,
            raw[R31B1_RELU23_BIAS_SYMBOL],
            (("relu_bias", True, 1),),
        ),
        R31B1_RELU23_COEFF_SYMBOL: _schedule_te_primfunc(
            tvm,
            R31B1_RELU23_COEFF_SYMBOL,
            raw[R31B1_RELU23_COEFF_SYMBOL],
            (("relu_coefficient", False, R31B1_THREADS),),
        ),
        R31B1_RELU17_BIAS_SYMBOL: _schedule_te_primfunc(
            tvm,
            R31B1_RELU17_BIAS_SYMBOL,
            raw[R31B1_RELU17_BIAS_SYMBOL],
            (("relu_bias", True, 1),),
        ),
        R31B1_RELU17_COEFF_SYMBOL: _schedule_te_primfunc(
            tvm,
            R31B1_RELU17_COEFF_SYMBOL,
            raw[R31B1_RELU17_COEFF_SYMBOL],
            (("relu_coefficient", False, R31B1_THREADS),),
        ),
        R31B1_CONV0_SYMBOL: _schedule_te_primfunc(
            tvm,
            R31B1_CONV0_SYMBOL,
            raw[R31B1_CONV0_SYMBOL],
            (("conv_right_output", True, R31B1_THREADS), ("conv_right_bias", True, 1)),
        ),
        R31B1_CONCRETIZE_SYMBOL: _schedule_te_primfunc(
            tvm,
            R31B1_CONCRETIZE_SYMBOL,
            raw[R31B1_CONCRETIZE_SYMBOL],
            (("concretized_lower", True, 1),),
        ),
    }
    scheduled[R31B1_RESIDUAL11_SYMBOL] = raw[R31B1_RESIDUAL11_SYMBOL]
    scheduled[R31B1_RESIDUAL6_SYMBOL] = raw[R31B1_RESIDUAL6_SYMBOL]
    return tvm.IRModule(scheduled)


def compile_r31b1_full_lower_forward_tir_v1(
    *, compute_capability: str = "sm_89"
) -> CompiledR31B1FullLowerForwardV1:
    import tvm

    module = build_r31b1_full_lower_forward_tir_v1()
    executable = tvm.compile(module, target=f"cuda -arch={compute_capability}")
    sources = tuple(imported.inspect_source() for imported in executable.mod.imports)
    if not sources:
        raise RuntimeError("R3-1b1 full-lower compile produced no CUDA source")
    source = "\n".join(sources)
    for symbol in R31B1_EXPORTED_SYMBOLS:
        if symbol not in source:
            raise RuntimeError(f"R3-1b1 compiled symbol is absent: {symbol}")
    return CompiledR31B1FullLowerForwardV1(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        module_hash=_sha256(tvm.ir.save_json(module)),
        device_source_hash=_sha256(source),
        device_source=source,
        exported_symbols=R31B1_EXPORTED_SYMBOLS,
        global_workspace_bytes=0,
        tvm_version=str(tvm.__version__),
    )


__all__ = [
    "CompiledR31B1FullLowerForwardV1",
    "R31B1_ARENA_CAPACITY",
    "R31B1_EXPORTED_SYMBOLS",
    "build_r31b1_full_lower_forward_tir_v1",
    "compile_r31b1_full_lower_forward_tir_v1",
]
