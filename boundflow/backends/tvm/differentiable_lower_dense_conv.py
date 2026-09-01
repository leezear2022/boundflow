"""B4-B2 B2-3 dense P-anchor ConvTranspose forward/backward CUDA TIR."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-locals
# pylint: disable=missing-function-docstring,too-many-arguments

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)
from boundflow.ir.differentiable_lower_dense_conv_tir import (
    DENSE_CONV_BACKWARD_SYMBOL,
    DENSE_CONV_FORWARD_SYMBOL,
    DENSE_CONV_WORKSPACE_INVENTORY,
    DifferentiableLowerDenseConvTIRScheduleV1,
    DifferentiableLowerDenseConvTIRTemplateV1,
)


@dataclass(frozen=True)
class CompiledDifferentiableLowerDenseConvTIR:
    """Compiled runtime plus exact compiler and workspace identities."""

    executable: DifferentiableLowerTIRExecutable
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    tvm_version: str
    observed_workspace_inventory: tuple[tuple[str, tuple[int, ...]], ...]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _dense_conv_forward_primfunc(
    template: DifferentiableLowerDenseConvTIRTemplateV1,
):
    import tvm
    from tvm import te

    domain = template.domain_count
    spec = template.spec_count
    channels = template.channels
    height = template.height
    width = template.width
    kernel_height = template.kernel_height
    kernel_width = template.kernel_width
    pad_h, pad_w = template.padding
    dilation_h, dilation_w = template.dilation
    dtype = "float32"
    incoming = te.placeholder(
        (domain, spec, channels, height, width), dtype, name="incoming_lower_a"
    )
    lower = te.placeholder(
        (domain, channels, height, width), dtype, name="preactivation_lower"
    )
    upper = te.placeholder(
        (domain, channels, height, width), dtype, name="preactivation_upper"
    )
    alpha = te.placeholder(
        (domain, channels, height, width), dtype, name="native_alpha"
    )
    incoming_bias = te.placeholder((domain, spec), dtype, name="incoming_lower_bias")
    weight = te.placeholder(
        (channels, channels, kernel_height, kernel_width),
        dtype,
        name="operator_weight",
    )
    operator_bias = te.placeholder((channels,), dtype, name="operator_bias")
    zero = tvm.tir.const(0.0, dtype)
    one = tvm.tir.const(1.0, dtype)
    epsilon = tvm.tir.const(1.1920928955078125e-07, dtype)

    def upper_slope(d_idx, c_idx, h_idx, w_idx):
        denominator = tvm.tir.max(
            upper[d_idx, c_idx, h_idx, w_idx] - lower[d_idx, c_idx, h_idx, w_idx],
            epsilon,
        )
        return tvm.tir.if_then_else(
            lower[d_idx, c_idx, h_idx, w_idx] >= zero,
            one,
            tvm.tir.if_then_else(
                upper[d_idx, c_idx, h_idx, w_idx] <= zero,
                zero,
                upper[d_idx, c_idx, h_idx, w_idx] / denominator,
            ),
        )

    def lower_slope(d_idx, c_idx, h_idx, w_idx):
        clamped = tvm.tir.min(tvm.tir.max(alpha[d_idx, c_idx, h_idx, w_idx], zero), one)
        return tvm.tir.if_then_else(
            tvm.tir.all(
                lower[d_idx, c_idx, h_idx, w_idx] < zero,
                upper[d_idx, c_idx, h_idx, w_idx] > zero,
            ),
            clamped,
            tvm.tir.if_then_else(lower[d_idx, c_idx, h_idx, w_idx] >= zero, one, zero),
        )

    def selected_slope(d_idx, s_idx, c_idx, h_idx, w_idx):
        return tvm.tir.if_then_else(
            incoming[d_idx, s_idx, c_idx, h_idx, w_idx] >= zero,
            lower_slope(d_idx, c_idx, h_idx, w_idx),
            upper_slope(d_idx, c_idx, h_idx, w_idx),
        )

    def selected_intercept(d_idx, s_idx, c_idx, h_idx, w_idx):
        ambiguous = -lower[d_idx, c_idx, h_idx, w_idx] * upper_slope(
            d_idx, c_idx, h_idx, w_idx
        )
        return tvm.tir.if_then_else(
            incoming[d_idx, s_idx, c_idx, h_idx, w_idx] >= zero,
            zero,
            tvm.tir.if_then_else(
                tvm.tir.all(
                    lower[d_idx, c_idx, h_idx, w_idx] < zero,
                    upper[d_idx, c_idx, h_idx, w_idx] > zero,
                ),
                ambiguous,
                zero,
            ),
        )

    relu_lower_a = te.compute(
        (domain, spec, channels, height, width),
        lambda d_idx, s_idx, c_idx, h_idx, w_idx: incoming[
            d_idx, s_idx, c_idx, h_idx, w_idx
        ]
        * selected_slope(d_idx, s_idx, c_idx, h_idx, w_idx),
        name="relu_lower_a",
    )
    reduce_ci = te.reduce_axis((0, channels), "reduce_input_channel")
    reduce_kh = te.reduce_axis((0, kernel_height), "reduce_kernel_height")
    reduce_kw = te.reduce_axis((0, kernel_width), "reduce_kernel_width")

    def output_element(d_idx, s_idx, co_idx, oh_idx, ow_idx):
        ih_idx = oh_idx + pad_h - reduce_kh * dilation_h
        iw_idx = ow_idx + pad_w - reduce_kw * dilation_w
        return te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(
                    ih_idx >= 0,
                    ih_idx < height,
                    iw_idx >= 0,
                    iw_idx < width,
                ),
                relu_lower_a[d_idx, s_idx, reduce_ci, ih_idx, iw_idx]
                * weight[reduce_ci, co_idx, reduce_kh, reduce_kw],
                zero,
            ),
            axis=(reduce_ci, reduce_kh, reduce_kw),
        )

    output_lower_a = te.compute(
        (domain, spec, channels, height, width),
        output_element,
        name="output_lower_a",
    )
    reduce_bc = te.reduce_axis((0, channels), "reduce_bias_channel")
    reduce_bh = te.reduce_axis((0, height), "reduce_bias_height")
    reduce_bw = te.reduce_axis((0, width), "reduce_bias_width")
    output_bias_delta = te.compute(
        (domain, spec),
        lambda d_idx, s_idx: te.sum(
            incoming[d_idx, s_idx, reduce_bc, reduce_bh, reduce_bw]
            * selected_intercept(d_idx, s_idx, reduce_bc, reduce_bh, reduce_bw)
            + relu_lower_a[d_idx, s_idx, reduce_bc, reduce_bh, reduce_bw]
            * operator_bias[reduce_bc],
            axis=(reduce_bc, reduce_bh, reduce_bw),
        ),
        name="output_bias_delta",
    )
    output_bias = te.compute(
        (domain, spec),
        lambda d_idx, s_idx: incoming_bias[d_idx, s_idx]
        + output_bias_delta[d_idx, s_idx],
        name="output_bias",
    )
    return (
        te.create_prim_func(
            [
                incoming,
                lower,
                upper,
                alpha,
                incoming_bias,
                weight,
                operator_bias,
                output_lower_a,
                output_bias,
            ]
        )
        .with_attr("global_symbol", DENSE_CONV_FORWARD_SYMBOL)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "b4b2-dense-conv-forward/v1")
    )


def _dense_conv_backward_primfunc(
    template: DifferentiableLowerDenseConvTIRTemplateV1,
):
    import tvm
    from tvm import te

    domain = template.domain_count
    spec = template.spec_count
    channels = template.channels
    height = template.height
    width = template.width
    kernel_height = template.kernel_height
    kernel_width = template.kernel_width
    pad_h, pad_w = template.padding
    dilation_h, dilation_w = template.dilation
    dtype = "float32"
    incoming = te.placeholder(
        (domain, spec, channels, height, width), dtype, name="incoming_lower_a"
    )
    lower = te.placeholder(
        (domain, channels, height, width), dtype, name="preactivation_lower"
    )
    upper = te.placeholder(
        (domain, channels, height, width), dtype, name="preactivation_upper"
    )
    alpha = te.placeholder(
        (domain, channels, height, width), dtype, name="native_alpha"
    )
    weight = te.placeholder(
        (channels, channels, kernel_height, kernel_width),
        dtype,
        name="operator_weight",
    )
    operator_bias = te.placeholder((channels,), dtype, name="operator_bias")
    output_a_gradient = te.placeholder(
        (domain, spec, channels, height, width),
        dtype,
        name="output_lower_a_gradient",
    )
    output_bias_gradient = te.placeholder(
        (domain, spec), dtype, name="output_bias_gradient"
    )
    zero = tvm.tir.const(0.0, dtype)
    one = tvm.tir.const(1.0, dtype)
    epsilon = tvm.tir.const(1.1920928955078125e-07, dtype)

    reduce_co = te.reduce_axis((0, channels), "reduce_output_channel")
    reduce_kh = te.reduce_axis((0, kernel_height), "reduce_kernel_height")
    reduce_kw = te.reduce_axis((0, kernel_width), "reduce_kernel_width")

    def adjoint_element(d_idx, s_idx, ci_idx, ih_idx, iw_idx):
        oh_idx = ih_idx - pad_h + reduce_kh * dilation_h
        ow_idx = iw_idx - pad_w + reduce_kw * dilation_w
        return te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(
                    oh_idx >= 0,
                    oh_idx < height,
                    ow_idx >= 0,
                    ow_idx < width,
                ),
                output_a_gradient[d_idx, s_idx, reduce_co, oh_idx, ow_idx]
                * weight[ci_idx, reduce_co, reduce_kh, reduce_kw],
                zero,
            ),
            axis=(reduce_co, reduce_kh, reduce_kw),
        )

    adjoint_conv = te.compute(
        (domain, spec, channels, height, width),
        adjoint_element,
        name="adjoint_conv",
    )
    adjoint_relu = te.compute(
        (domain, spec, channels, height, width),
        lambda d_idx, s_idx, c_idx, h_idx, w_idx: adjoint_conv[
            d_idx, s_idx, c_idx, h_idx, w_idx
        ]
        + output_bias_gradient[d_idx, s_idx] * operator_bias[c_idx],
        name="adjoint_relu",
    )

    def upper_slope(d_idx, c_idx, h_idx, w_idx):
        denominator = tvm.tir.max(
            upper[d_idx, c_idx, h_idx, w_idx] - lower[d_idx, c_idx, h_idx, w_idx],
            epsilon,
        )
        return tvm.tir.if_then_else(
            lower[d_idx, c_idx, h_idx, w_idx] >= zero,
            one,
            tvm.tir.if_then_else(
                upper[d_idx, c_idx, h_idx, w_idx] <= zero,
                zero,
                upper[d_idx, c_idx, h_idx, w_idx] / denominator,
            ),
        )

    def lower_slope(d_idx, c_idx, h_idx, w_idx):
        clamped = tvm.tir.min(tvm.tir.max(alpha[d_idx, c_idx, h_idx, w_idx], zero), one)
        return tvm.tir.if_then_else(
            tvm.tir.all(
                lower[d_idx, c_idx, h_idx, w_idx] < zero,
                upper[d_idx, c_idx, h_idx, w_idx] > zero,
            ),
            clamped,
            tvm.tir.if_then_else(lower[d_idx, c_idx, h_idx, w_idx] >= zero, one, zero),
        )

    def selected_slope(d_idx, s_idx, c_idx, h_idx, w_idx):
        return tvm.tir.if_then_else(
            incoming[d_idx, s_idx, c_idx, h_idx, w_idx] >= zero,
            lower_slope(d_idx, c_idx, h_idx, w_idx),
            upper_slope(d_idx, c_idx, h_idx, w_idx),
        )

    def selected_intercept(d_idx, s_idx, c_idx, h_idx, w_idx):
        ambiguous = -lower[d_idx, c_idx, h_idx, w_idx] * upper_slope(
            d_idx, c_idx, h_idx, w_idx
        )
        return tvm.tir.if_then_else(
            incoming[d_idx, s_idx, c_idx, h_idx, w_idx] >= zero,
            zero,
            tvm.tir.if_then_else(
                tvm.tir.all(
                    lower[d_idx, c_idx, h_idx, w_idx] < zero,
                    upper[d_idx, c_idx, h_idx, w_idx] > zero,
                ),
                ambiguous,
                zero,
            ),
        )

    reduce_spec = te.reduce_axis((0, spec), "reduce_spec_alpha")
    native_alpha_gradient = te.compute(
        (domain, channels, height, width),
        lambda d_idx, c_idx, h_idx, w_idx: te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(
                    incoming[d_idx, reduce_spec, c_idx, h_idx, w_idx] >= zero,
                    lower[d_idx, c_idx, h_idx, w_idx] < zero,
                    upper[d_idx, c_idx, h_idx, w_idx] > zero,
                    alpha[d_idx, c_idx, h_idx, w_idx] >= zero,
                    alpha[d_idx, c_idx, h_idx, w_idx] <= one,
                ),
                adjoint_relu[d_idx, reduce_spec, c_idx, h_idx, w_idx]
                * incoming[d_idx, reduce_spec, c_idx, h_idx, w_idx],
                zero,
            ),
            axis=reduce_spec,
        ),
        name="native_alpha_gradient",
    )
    incoming_lower_a_gradient = te.compute(
        (domain, spec, channels, height, width),
        lambda d_idx, s_idx, c_idx, h_idx, w_idx: adjoint_relu[
            d_idx, s_idx, c_idx, h_idx, w_idx
        ]
        * selected_slope(d_idx, s_idx, c_idx, h_idx, w_idx)
        + output_bias_gradient[d_idx, s_idx]
        * selected_intercept(d_idx, s_idx, c_idx, h_idx, w_idx),
        name="incoming_lower_a_gradient",
    )
    return (
        te.create_prim_func(
            [
                incoming,
                lower,
                upper,
                alpha,
                weight,
                operator_bias,
                output_a_gradient,
                output_bias_gradient,
                native_alpha_gradient,
                incoming_lower_a_gradient,
            ]
        )
        .with_attr("global_symbol", DENSE_CONV_BACKWARD_SYMBOL)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "b4b2-dense-conv-backward/v1")
    )


def _allocated_buffer_inventory(module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    import tvm

    inventory: set[tuple[str, tuple[int, ...]]] = set()

    def visit(node) -> None:
        if isinstance(node, tvm.tir.Block):
            for buffer in node.alloc_buffers:
                shape = tuple(int(value) for value in buffer.shape)
                inventory.add((str(buffer.name), shape))

    for function in module.functions.values():
        tvm.tir.stmt_functor.post_order_visit(function.body, visit)
    return tuple(sorted(inventory))


def build_dense_conv_tir_modules(
    template: DifferentiableLowerDenseConvTIRTemplateV1,
    schedule_ir: DifferentiableLowerDenseConvTIRScheduleV1,
):
    """Build exact unscheduled and structurally checked correctness modules."""

    template.validate()
    schedule_ir.validate_against(template)
    import tvm

    unscheduled = tvm.IRModule(
        {
            template.forward_symbol: _dense_conv_forward_primfunc(template),
            template.backward_symbol: _dense_conv_backward_primfunc(template),
        }
    )
    inventories = {
        template.forward_symbol: (
            ("output_lower_a", 3),
            ("output_bias_delta", 3),
            ("output_bias", 0),
        ),
        template.backward_symbol: (
            ("adjoint_conv", 3),
            ("adjoint_relu", 0),
            ("native_alpha_gradient", 1),
            ("incoming_lower_a_gradient", 0),
        ),
    }
    scheduled_functions = {}
    for symbol, blocks in inventories.items():
        tir_schedule = tvm.tir.Schedule(tvm.IRModule({symbol: unscheduled[symbol]}))
        inline_name = (
            "relu_lower_a" if symbol == template.forward_symbol else "adjoint_relu"
        )
        tir_schedule.compute_inline(
            tir_schedule.get_block(inline_name, func_name=symbol)
        )
        for block_name, reduction_count in blocks:
            if block_name == inline_name:
                continue
            block = tir_schedule.get_block(block_name, func_name=symbol)
            loops = tir_schedule.get_loops(block)
            spatial = loops[:-reduction_count] if reduction_count else loops
            fused = tir_schedule.fuse(*spatial) if len(spatial) > 1 else spatial[0]
            block_loop, thread_loop = tir_schedule.split(
                fused, factors=[None, schedule_ir.thread_extent]
            )
            tir_schedule.bind(block_loop, "blockIdx.x")
            tir_schedule.bind(thread_loop, "threadIdx.x")
        scheduled_functions[symbol] = tir_schedule.mod[symbol]
    scheduled = tvm.IRModule(scheduled_functions)
    observed = _allocated_buffer_inventory(scheduled)
    if observed != DENSE_CONV_WORKSPACE_INVENTORY:
        raise RuntimeError(
            f"dense Conv TIR workspace differs: expected "
            f"{DENSE_CONV_WORKSPACE_INVENTORY}, observed {observed}"
        )
    return unscheduled, scheduled, observed


def compile_dense_conv_tir(
    template: DifferentiableLowerDenseConvTIRTemplateV1,
    schedule: DifferentiableLowerDenseConvTIRScheduleV1,
) -> CompiledDifferentiableLowerDenseConvTIR:
    """Compile both Conv semantic symbols and bind structural workspace evidence."""

    template.validate()
    schedule.validate_against(template)
    import tvm

    unscheduled, scheduled, observed = build_dense_conv_tir_modules(template, schedule)
    executable = tvm.compile(
        scheduled, target=f"{template.target} -arch={template.compute_capability}"
    )
    device_sources = tuple(module.inspect_source() for module in executable.mod.imports)
    if not device_sources:
        raise RuntimeError("dense Conv TIR compile produced no CUDA device source")
    return CompiledDifferentiableLowerDenseConvTIR(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        unscheduled_tir_hash=_sha256(tvm.ir.save_json(unscheduled)),
        scheduled_tir_hash=_sha256(tvm.ir.save_json(scheduled)),
        device_source_hash=_sha256("\n".join(device_sources)),
        tvm_version=str(tvm.__version__),
        observed_workspace_inventory=observed,
    )


__all__ = [
    "CompiledDifferentiableLowerDenseConvTIR",
    "build_dense_conv_tir_modules",
    "compile_dense_conv_tir",
]
