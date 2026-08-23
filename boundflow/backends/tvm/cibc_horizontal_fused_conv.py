"""Manual TVM TIR port of the CIBC-parity P-anchor horizontal fusion."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,too-many-statements,too-many-locals
# pylint: disable=invalid-name,line-too-long,missing-function-docstring
# pylint: disable=missing-class-docstring,too-many-instance-attributes
# pylint: disable=import-outside-toplevel,too-many-arguments
# pylint: disable=too-many-positional-arguments,chained-comparison

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)

CIBC_TIR_FORWARD_SYMBOL = "boundflow_cibc_horizontal_forward_v2"
CIBC_TIR_BACKWARD_SYMBOL = "boundflow_cibc_horizontal_backward_v2"


@dataclass(frozen=True)
class CompiledCIBCHorizontalConvTIRV2:
    executable: DifferentiableLowerTIRExecutable
    module_hash: str
    device_source_hash: str
    device_source: str
    tvm_version: str
    exported_symbols: tuple[str, ...]
    global_workspace_bytes: int
    local_buffer_count: int


def build_cibc_horizontal_conv_tir_v2():
    """Build the fixed P-anchor one-forward/one-backward manual TIR module."""

    import tvm
    from tvm.script import tir as T

    @T.prim_func
    def forward(
        incoming: T.Buffer((6, 1, 16, 8, 8), "float32"),
        lower: T.Buffer((6, 16, 8, 8), "float32"),
        upper: T.Buffer((6, 16, 8, 8), "float32"),
        alpha: T.Buffer((6, 86), "float32"),
        incoming_bias: T.Buffer((6, 1), "float32"),
        weight: T.Buffer((16, 16, 3, 3), "float32"),
        operator_bias: T.Buffer((16,), "float32"),
        alpha_map: T.Buffer((16, 8, 8), "int32"),
        combined_output: T.Buffer((6150,), "float32"),
    ):
        T.func_attr(
            {
                "global_symbol": CIBC_TIR_FORWARD_SYMBOL,
                "tir.noalias": True,
                "boundflow.schema_version": "cibc-horizontal-forward/v2",
            }
        )
        accumulator = T.alloc_buffer((1,), "float32", scope="local")
        bias_reduced = T.alloc_buffer((1,), "float32", scope="local")
        for block_x in T.thread_binding(54, thread="blockIdx.x"):
            for thread_x in T.thread_binding(128, thread="threadIdx.x"):
                flat = block_x * 128 + thread_x
                if block_x < 48:
                    output_w = flat % 8
                    output_h = flat // 8 % 8
                    output_channel = flat // 64 % 16
                    output_spec = flat // 1024 % 1
                    output_domain = flat // 1024
                    accumulator[0] = T.float32(0)
                    for reduction in range(144):
                        input_channel = reduction // 9
                        kernel_h = reduction // 3 % 3
                        kernel_w = reduction % 3
                        input_h = output_h + 1 - kernel_h
                        input_w = output_w + 1 - kernel_w
                        if (
                            0 <= input_h
                            and input_h < 8
                            and 0 <= input_w
                            and input_w < 8
                        ):
                            incoming_value = incoming[
                                output_domain,
                                output_spec,
                                input_channel,
                                input_h,
                                input_w,
                            ]
                            lower_value = lower[
                                output_domain, input_channel, input_h, input_w
                            ]
                            upper_value = upper[
                                output_domain, input_channel, input_h, input_w
                            ]
                            denominator = T.max(
                                upper_value - lower_value,
                                T.float32(1.1920928955078125e-7),
                            )
                            upper_slope = T.if_then_else(
                                lower_value >= T.float32(0),
                                T.float32(1),
                                T.if_then_else(
                                    upper_value <= T.float32(0),
                                    T.float32(0),
                                    upper_value / denominator,
                                ),
                            )
                            alpha_ordinal = alpha_map[input_channel, input_h, input_w]
                            alpha_value = T.if_then_else(
                                alpha_ordinal >= 0,
                                alpha[output_domain, alpha_ordinal],
                                T.float32(0),
                            )
                            alpha_value_clamped = T.min(
                                T.max(alpha_value, T.float32(0)), T.float32(1)
                            )
                            ambiguous = T.And(
                                lower_value < T.float32(0),
                                upper_value > T.float32(0),
                            )
                            lower_slope = T.if_then_else(
                                ambiguous,
                                alpha_value_clamped,
                                T.if_then_else(
                                    lower_value >= T.float32(0),
                                    T.float32(1),
                                    T.float32(0),
                                ),
                            )
                            selected_slope = T.if_then_else(
                                incoming_value >= T.float32(0),
                                lower_slope,
                                upper_slope,
                            )
                            accumulator[0] = (
                                accumulator[0]
                                + incoming_value
                                * selected_slope
                                * weight[
                                    input_channel,
                                    output_channel,
                                    kernel_h,
                                    kernel_w,
                                ]
                            )
                    combined_output[flat] = accumulator[0]
                bias_domain = T.max(block_x - 48, 0)
                accumulator[0] = T.float32(0)
                for reduction_outer in range(8):
                    reduction = reduction_outer * 128 + thread_x
                    input_channel = reduction // 64
                    input_h = reduction // 8 % 8
                    input_w = reduction % 8
                    incoming_value = incoming[
                        bias_domain, 0, input_channel, input_h, input_w
                    ]
                    lower_value = lower[bias_domain, input_channel, input_h, input_w]
                    upper_value = upper[bias_domain, input_channel, input_h, input_w]
                    denominator = T.max(
                        upper_value - lower_value,
                        T.float32(1.1920928955078125e-7),
                    )
                    upper_slope = T.if_then_else(
                        lower_value >= T.float32(0),
                        T.float32(1),
                        T.if_then_else(
                            upper_value <= T.float32(0),
                            T.float32(0),
                            upper_value / denominator,
                        ),
                    )
                    alpha_ordinal = alpha_map[input_channel, input_h, input_w]
                    alpha_value = T.if_then_else(
                        alpha_ordinal >= 0,
                        alpha[bias_domain, alpha_ordinal],
                        T.float32(0),
                    )
                    alpha_value_clamped = T.min(
                        T.max(alpha_value, T.float32(0)), T.float32(1)
                    )
                    ambiguous = T.And(
                        lower_value < T.float32(0),
                        upper_value > T.float32(0),
                    )
                    lower_slope = T.if_then_else(
                        ambiguous,
                        alpha_value_clamped,
                        T.if_then_else(
                            lower_value >= T.float32(0),
                            T.float32(1),
                            T.float32(0),
                        ),
                    )
                    selected_slope = T.if_then_else(
                        incoming_value >= T.float32(0),
                        lower_slope,
                        upper_slope,
                    )
                    upper_intercept = T.if_then_else(
                        ambiguous,
                        -lower_value * upper_slope,
                        T.float32(0),
                    )
                    selected_intercept = T.if_then_else(
                        incoming_value >= T.float32(0),
                        T.float32(0),
                        upper_intercept,
                    )
                    accumulator[0] = accumulator[0] + incoming_value * (
                        selected_intercept
                        + selected_slope * operator_bias[input_channel]
                    )
                with T.attr(
                    T.comm_reducer(lambda left, right: left + right, [T.float32(0)]),
                    "reduce_scope",
                    T.reinterpret("handle", T.uint64(0)),
                ):
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accumulator[0],
                        T.bool(True),
                        bias_reduced[0],
                        thread_x,
                    )
                if T.And(48 <= block_x, thread_x == 0):
                    combined_output[6144 + block_x - 48] = (
                        incoming_bias[block_x - 48, 0] + bias_reduced[0]
                    )

    @T.prim_func
    def forward_parallel(
        incoming: T.Buffer((6, 1, 16, 8, 8), "float32"),
        lower: T.Buffer((6, 16, 8, 8), "float32"),
        upper: T.Buffer((6, 16, 8, 8), "float32"),
        alpha: T.Buffer((6, 86), "float32"),
        incoming_bias: T.Buffer((6, 1), "float32"),
        weight: T.Buffer((16, 16, 3, 3), "float32"),
        operator_bias: T.Buffer((16,), "float32"),
        alpha_map: T.Buffer((16, 8, 8), "int32"),
        combined_output: T.Buffer((6150,), "float32"),
    ):
        T.func_attr(
            {
                "global_symbol": CIBC_TIR_FORWARD_SYMBOL,
                "tir.noalias": True,
                "boundflow.schema_version": "cibc-horizontal-forward-parallel/v2",
            }
        )
        accumulator = T.alloc_buffer((1,), "float32", scope="local")
        bias_reduced = T.alloc_buffer((1,), "float32", scope="local")
        output_partial = T.alloc_buffer((128,), "float32", scope="shared")
        for block_x in T.thread_binding(198, thread="blockIdx.x"):
            for thread_x in T.thread_binding(128, thread="threadIdx.x"):
                reduction_lane = thread_x % 4
                output_flat = T.min(block_x * 32 + thread_x // 4, 6143)
                output_w = output_flat % 8
                output_h = output_flat // 8 % 8
                output_channel = output_flat // 64 % 16
                output_domain = output_flat // 1024
                accumulator[0] = T.float32(0)
                for reduction_outer in range(36):
                    reduction = reduction_outer * 4 + reduction_lane
                    input_channel = reduction // 9
                    kernel_h = reduction // 3 % 3
                    kernel_w = reduction % 3
                    input_h = output_h + 1 - kernel_h
                    input_w = output_w + 1 - kernel_w
                    if 0 <= input_h and input_h < 8 and 0 <= input_w and input_w < 8:
                        incoming_value = incoming[
                            output_domain, 0, input_channel, input_h, input_w
                        ]
                        lower_value = lower[
                            output_domain, input_channel, input_h, input_w
                        ]
                        upper_value = upper[
                            output_domain, input_channel, input_h, input_w
                        ]
                        denominator = T.max(
                            upper_value - lower_value,
                            T.float32(1.1920928955078125e-7),
                        )
                        upper_slope = T.if_then_else(
                            lower_value >= T.float32(0),
                            T.float32(1),
                            T.if_then_else(
                                upper_value <= T.float32(0),
                                T.float32(0),
                                upper_value / denominator,
                            ),
                        )
                        alpha_ordinal = alpha_map[input_channel, input_h, input_w]
                        alpha_value = T.if_then_else(
                            alpha_ordinal >= 0,
                            alpha[output_domain, alpha_ordinal],
                            T.float32(0),
                        )
                        alpha_value_clamped = T.min(
                            T.max(alpha_value, T.float32(0)), T.float32(1)
                        )
                        ambiguous = T.And(
                            lower_value < T.float32(0), upper_value > T.float32(0)
                        )
                        lower_slope = T.if_then_else(
                            ambiguous,
                            alpha_value_clamped,
                            T.if_then_else(
                                lower_value >= T.float32(0),
                                T.float32(1),
                                T.float32(0),
                            ),
                        )
                        selected_slope = T.if_then_else(
                            incoming_value >= T.float32(0),
                            lower_slope,
                            upper_slope,
                        )
                        accumulator[0] = (
                            accumulator[0]
                            + incoming_value
                            * selected_slope
                            * weight[
                                input_channel,
                                output_channel,
                                kernel_h,
                                kernel_w,
                            ]
                        )
                output_partial[thread_x] = accumulator[0]
                T.tvm_storage_sync("shared")
                if T.And(block_x < 192, reduction_lane == 0):
                    combined_output[output_flat] = (
                        output_partial[thread_x]
                        + output_partial[thread_x + 1]
                        + output_partial[thread_x + 2]
                        + output_partial[thread_x + 3]
                    )
                bias_domain = T.max(block_x - 192, 0)
                accumulator[0] = T.float32(0)
                for reduction_outer in range(8):
                    reduction = reduction_outer * 128 + thread_x
                    input_channel = reduction // 64
                    input_h = reduction // 8 % 8
                    input_w = reduction % 8
                    incoming_value = incoming[
                        bias_domain, 0, input_channel, input_h, input_w
                    ]
                    lower_value = lower[bias_domain, input_channel, input_h, input_w]
                    upper_value = upper[bias_domain, input_channel, input_h, input_w]
                    denominator = T.max(
                        upper_value - lower_value,
                        T.float32(1.1920928955078125e-7),
                    )
                    upper_slope = T.if_then_else(
                        lower_value >= T.float32(0),
                        T.float32(1),
                        T.if_then_else(
                            upper_value <= T.float32(0),
                            T.float32(0),
                            upper_value / denominator,
                        ),
                    )
                    alpha_ordinal = alpha_map[input_channel, input_h, input_w]
                    alpha_value = T.if_then_else(
                        alpha_ordinal >= 0,
                        alpha[bias_domain, alpha_ordinal],
                        T.float32(0),
                    )
                    alpha_value_clamped = T.min(
                        T.max(alpha_value, T.float32(0)), T.float32(1)
                    )
                    ambiguous = T.And(
                        lower_value < T.float32(0), upper_value > T.float32(0)
                    )
                    lower_slope = T.if_then_else(
                        ambiguous,
                        alpha_value_clamped,
                        T.if_then_else(
                            lower_value >= T.float32(0),
                            T.float32(1),
                            T.float32(0),
                        ),
                    )
                    selected_slope = T.if_then_else(
                        incoming_value >= T.float32(0), lower_slope, upper_slope
                    )
                    upper_intercept = T.if_then_else(
                        ambiguous, -lower_value * upper_slope, T.float32(0)
                    )
                    selected_intercept = T.if_then_else(
                        incoming_value >= T.float32(0),
                        T.float32(0),
                        upper_intercept,
                    )
                    accumulator[0] = accumulator[0] + incoming_value * (
                        selected_intercept
                        + selected_slope * operator_bias[input_channel]
                    )
                with T.attr(
                    T.comm_reducer(lambda left, right: left + right, [T.float32(0)]),
                    "reduce_scope",
                    T.reinterpret("handle", T.uint64(0)),
                ):
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accumulator[0],
                        T.bool(True),
                        bias_reduced[0],
                        thread_x,
                    )
                if T.And(192 <= block_x, thread_x == 0):
                    combined_output[6144 + block_x - 192] = (
                        incoming_bias[block_x - 192, 0] + bias_reduced[0]
                    )

    @T.prim_func
    def backward(
        incoming: T.Buffer((6, 1, 16, 8, 8), "float32"),
        lower: T.Buffer((6, 16, 8, 8), "float32"),
        upper: T.Buffer((6, 16, 8, 8), "float32"),
        alpha: T.Buffer((6, 86), "float32"),
        weight: T.Buffer((16, 16, 3, 3), "float32"),
        operator_bias: T.Buffer((16,), "float32"),
        output_a_gradient: T.Buffer((6, 1, 16, 8, 8), "float32"),
        output_bias_gradient: T.Buffer((6, 1), "float32"),
        alpha_map: T.Buffer((16, 8, 8), "int32"),
        alpha_channel: T.Buffer((86,), "int32"),
        alpha_height: T.Buffer((86,), "int32"),
        alpha_width: T.Buffer((86,), "int32"),
        combined_gradient: T.Buffer((6660,), "float32"),
    ):
        T.func_attr(
            {
                "global_symbol": CIBC_TIR_BACKWARD_SYMBOL,
                "tir.noalias": True,
                "boundflow.schema_version": "cibc-horizontal-backward/v2",
            }
        )
        accumulator = T.alloc_buffer((1,), "float32", scope="local")
        for block_x in T.thread_binding(53, thread="blockIdx.x"):
            for thread_x in T.thread_binding(128, thread="threadIdx.x"):
                flat = block_x * 128 + thread_x
                if flat < 6144:
                    input_w = flat % 8
                    input_h = flat // 8 % 8
                    input_channel = flat // 64 % 16
                    input_spec = flat // 1024 % 1
                    input_domain = flat // 1024
                    accumulator[0] = T.float32(0)
                    for reduction in range(144):
                        output_channel = reduction // 9
                        kernel_h = reduction // 3 % 3
                        kernel_w = reduction % 3
                        output_h = input_h - 1 + kernel_h
                        output_w = input_w - 1 + kernel_w
                        if (
                            0 <= output_h
                            and output_h < 8
                            and 0 <= output_w
                            and output_w < 8
                        ):
                            accumulator[0] = (
                                accumulator[0]
                                + output_a_gradient[
                                    input_domain,
                                    input_spec,
                                    output_channel,
                                    output_h,
                                    output_w,
                                ]
                                * weight[
                                    input_channel,
                                    output_channel,
                                    kernel_h,
                                    kernel_w,
                                ]
                            )
                    bias_gradient = output_bias_gradient[input_domain, input_spec]
                    adjoint = (
                        accumulator[0] + bias_gradient * operator_bias[input_channel]
                    )
                    incoming_value = incoming[
                        input_domain, input_spec, input_channel, input_h, input_w
                    ]
                    lower_value = lower[input_domain, input_channel, input_h, input_w]
                    upper_value = upper[input_domain, input_channel, input_h, input_w]
                    denominator = T.max(
                        upper_value - lower_value,
                        T.float32(1.1920928955078125e-7),
                    )
                    upper_slope = T.if_then_else(
                        lower_value >= T.float32(0),
                        T.float32(1),
                        T.if_then_else(
                            upper_value <= T.float32(0),
                            T.float32(0),
                            upper_value / denominator,
                        ),
                    )
                    alpha_ordinal = alpha_map[input_channel, input_h, input_w]
                    alpha_value = T.if_then_else(
                        alpha_ordinal >= 0,
                        alpha[input_domain, alpha_ordinal],
                        T.float32(0),
                    )
                    alpha_value_clamped = T.min(
                        T.max(alpha_value, T.float32(0)), T.float32(1)
                    )
                    ambiguous = T.And(
                        lower_value < T.float32(0), upper_value > T.float32(0)
                    )
                    lower_slope = T.if_then_else(
                        ambiguous,
                        alpha_value_clamped,
                        T.if_then_else(
                            lower_value >= T.float32(0),
                            T.float32(1),
                            T.float32(0),
                        ),
                    )
                    selected_slope = T.if_then_else(
                        incoming_value >= T.float32(0), lower_slope, upper_slope
                    )
                    upper_intercept = T.if_then_else(
                        ambiguous, -lower_value * upper_slope, T.float32(0)
                    )
                    selected_intercept = T.if_then_else(
                        incoming_value >= T.float32(0),
                        T.float32(0),
                        upper_intercept,
                    )
                    combined_gradient[flat] = (
                        adjoint * selected_slope + bias_gradient * selected_intercept
                    )
                if 6144 <= flat and flat < 6660:
                    alpha_flat = flat - 6144
                    alpha_ordinal = alpha_flat % 86
                    alpha_domain = alpha_flat // 86
                    input_channel = alpha_channel[alpha_ordinal]
                    input_h = alpha_height[alpha_ordinal]
                    input_w = alpha_width[alpha_ordinal]
                    accumulator[0] = T.float32(0)
                    for reduction in range(144):
                        output_channel = reduction // 9
                        kernel_h = reduction // 3 % 3
                        kernel_w = reduction % 3
                        output_h = input_h - 1 + kernel_h
                        output_w = input_w - 1 + kernel_w
                        if (
                            0 <= output_h
                            and output_h < 8
                            and 0 <= output_w
                            and output_w < 8
                        ):
                            accumulator[0] = (
                                accumulator[0]
                                + output_a_gradient[
                                    alpha_domain,
                                    0,
                                    output_channel,
                                    output_h,
                                    output_w,
                                ]
                                * weight[
                                    input_channel,
                                    output_channel,
                                    kernel_h,
                                    kernel_w,
                                ]
                            )
                    bias_gradient = output_bias_gradient[alpha_domain, 0]
                    adjoint = (
                        accumulator[0] + bias_gradient * operator_bias[input_channel]
                    )
                    incoming_value = incoming[
                        alpha_domain, 0, input_channel, input_h, input_w
                    ]
                    lower_value = lower[alpha_domain, input_channel, input_h, input_w]
                    upper_value = upper[alpha_domain, input_channel, input_h, input_w]
                    alpha_value = alpha[alpha_domain, alpha_ordinal]
                    alpha_owned = T.And(
                        lower_value < T.float32(0),
                        T.And(
                            upper_value > T.float32(0),
                            T.And(
                                incoming_value >= T.float32(0),
                                T.And(
                                    alpha_value >= T.float32(0),
                                    alpha_value <= T.float32(1),
                                ),
                            ),
                        ),
                    )
                    combined_gradient[flat] = T.if_then_else(
                        alpha_owned,
                        adjoint * incoming_value,
                        T.float32(0),
                    )

    return tvm.IRModule(
        {
            CIBC_TIR_FORWARD_SYMBOL: forward_parallel,
            CIBC_TIR_BACKWARD_SYMBOL: backward,
        }
    )


def _local_buffer_count(module) -> int:
    import tvm

    count = 0

    def visit(node) -> None:
        nonlocal count
        if isinstance(node, tvm.tir.Allocate) and str(node.storage_scope) == "local":
            count += 1

    for function in module.functions.values():
        tvm.tir.stmt_functor.post_order_visit(function.body, visit)
    return count


def compile_cibc_horizontal_conv_tir_v2(
    *,
    compute_capability: str,
) -> CompiledCIBCHorizontalConvTIRV2:
    import tvm

    module = build_cibc_horizontal_conv_tir_v2()
    executable = tvm.compile(module, target=f"cuda -arch={compute_capability}")
    sources = tuple(imported.inspect_source() for imported in executable.mod.imports)
    if len(sources) != 1:
        raise RuntimeError("CIBC horizontal TIR device module inventory differs")
    source = sources[0]
    names = tuple(
        symbol
        for symbol in (CIBC_TIR_FORWARD_SYMBOL, CIBC_TIR_BACKWARD_SYMBOL)
        if symbol in source
    )
    if names != (CIBC_TIR_FORWARD_SYMBOL, CIBC_TIR_BACKWARD_SYMBOL):
        raise RuntimeError("CIBC horizontal TIR exported symbols differ")
    return CompiledCIBCHorizontalConvTIRV2(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        module_hash=hashlib.sha256(
            tvm.ir.save_json(module).encode("utf-8")
        ).hexdigest(),
        device_source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        device_source=source,
        tvm_version=str(tvm.__version__),
        exported_symbols=names,
        global_workspace_bytes=0,
        local_buffer_count=_local_buffer_count(module),
    )


__all__ = [
    "CIBC_TIR_BACKWARD_SYMBOL",
    "CIBC_TIR_FORWARD_SYMBOL",
    "CompiledCIBCHorizontalConvTIRV2",
    "build_cibc_horizontal_conv_tir_v2",
    "compile_cibc_horizontal_conv_tir_v2",
]
