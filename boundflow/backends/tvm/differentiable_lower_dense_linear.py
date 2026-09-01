"""B4-B2 B2-1 dense S-anchor Linear forward/backward CUDA TIR."""

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
from boundflow.ir.differentiable_lower_dense_linear_tir import (
    DENSE_LINEAR_BACKWARD_SYMBOL,
    DENSE_LINEAR_FORWARD_SYMBOL,
    DifferentiableLowerDenseLinearTIRScheduleV1,
    DifferentiableLowerDenseLinearTIRTemplateV1,
)


@dataclass(frozen=True)
class CompiledDifferentiableLowerDenseLinearTIR:
    """Compiled runtime plus exact unscheduled/scheduled/device identities."""

    executable: DifferentiableLowerTIRExecutable
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    tvm_version: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _dense_linear_forward_primfunc(
    template: DifferentiableLowerDenseLinearTIRTemplateV1,
):
    import tvm
    from tvm import te

    domain = template.domain_count
    spec = template.spec_count
    current = template.current_features
    previous = template.previous_features
    dtype = "float32"
    incoming = te.placeholder((domain, spec, current), dtype, name="incoming_lower_a")
    lower = te.placeholder((domain, current), dtype, name="preactivation_lower")
    upper = te.placeholder((domain, current), dtype, name="preactivation_upper")
    alpha = te.placeholder((domain, current), dtype, name="native_alpha")
    beta = te.placeholder((domain, current), dtype, name="native_beta")
    split = te.placeholder((domain, current), dtype, name="dense_split_sign")
    incoming_bias = te.placeholder((domain, spec), dtype, name="incoming_lower_bias")
    weight = te.placeholder((current, previous), dtype, name="operator_weight")
    operator_bias = te.placeholder((current,), dtype, name="operator_bias")
    zero = tvm.tir.const(0.0, dtype)
    one = tvm.tir.const(1.0, dtype)
    epsilon = tvm.tir.const(1.1920928955078125e-07, dtype)

    def upper_slope(d_idx, c_idx):
        denominator = tvm.tir.max(upper[d_idx, c_idx] - lower[d_idx, c_idx], epsilon)
        return tvm.tir.if_then_else(
            lower[d_idx, c_idx] >= zero,
            one,
            tvm.tir.if_then_else(
                upper[d_idx, c_idx] <= zero,
                zero,
                upper[d_idx, c_idx] / denominator,
            ),
        )

    def lower_slope(d_idx, c_idx):
        clamped = tvm.tir.min(tvm.tir.max(alpha[d_idx, c_idx], zero), one)
        return tvm.tir.if_then_else(
            tvm.tir.all(lower[d_idx, c_idx] < zero, upper[d_idx, c_idx] > zero),
            clamped,
            tvm.tir.if_then_else(lower[d_idx, c_idx] >= zero, one, zero),
        )

    def selected_slope(d_idx, s_idx, c_idx):
        return tvm.tir.if_then_else(
            incoming[d_idx, s_idx, c_idx] >= zero,
            lower_slope(d_idx, c_idx),
            upper_slope(d_idx, c_idx),
        )

    def selected_intercept(d_idx, s_idx, c_idx):
        ambiguous_intercept = -lower[d_idx, c_idx] * upper_slope(d_idx, c_idx)
        return tvm.tir.if_then_else(
            incoming[d_idx, s_idx, c_idx] >= zero,
            zero,
            tvm.tir.if_then_else(
                tvm.tir.all(lower[d_idx, c_idx] < zero, upper[d_idx, c_idx] > zero),
                ambiguous_intercept,
                zero,
            ),
        )

    relu_lower_a = te.compute(
        (domain, spec, current),
        lambda d_idx, s_idx, c_idx: incoming[d_idx, s_idx, c_idx]
        * selected_slope(d_idx, s_idx, c_idx)
        - beta[d_idx, c_idx] * split[d_idx, c_idx],
        name="relu_lower_a",
    )
    reduce_a = te.reduce_axis((0, current), "reduce_output_a")
    output_lower_a = te.compute(
        (domain, spec, previous),
        lambda d_idx, s_idx, p_idx: te.sum(
            relu_lower_a[d_idx, s_idx, reduce_a] * weight[reduce_a, p_idx],
            axis=reduce_a,
        ),
        name="output_lower_a",
    )
    reduce_bias = te.reduce_axis((0, current), "reduce_output_bias")
    output_bias_delta = te.compute(
        (domain, spec),
        lambda d_idx, s_idx: te.sum(
            incoming[d_idx, s_idx, reduce_bias]
            * selected_intercept(d_idx, s_idx, reduce_bias)
            + relu_lower_a[d_idx, s_idx, reduce_bias] * operator_bias[reduce_bias],
            axis=reduce_bias,
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
                beta,
                split,
                incoming_bias,
                weight,
                operator_bias,
                output_lower_a,
                output_bias,
            ]
        )
        .with_attr("global_symbol", DENSE_LINEAR_FORWARD_SYMBOL)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "b4b2-dense-linear-forward/v1")
    )


def _dense_linear_backward_primfunc(
    template: DifferentiableLowerDenseLinearTIRTemplateV1,
):
    import tvm
    from tvm import te

    domain = template.domain_count
    spec = template.spec_count
    current = template.current_features
    previous = template.previous_features
    dtype = "float32"
    incoming = te.placeholder((domain, spec, current), dtype, name="incoming_lower_a")
    lower = te.placeholder((domain, current), dtype, name="preactivation_lower")
    upper = te.placeholder((domain, current), dtype, name="preactivation_upper")
    alpha = te.placeholder((domain, current), dtype, name="native_alpha")
    beta = te.placeholder((domain, current), dtype, name="native_beta")
    split = te.placeholder((domain, current), dtype, name="dense_split_sign")
    weight = te.placeholder((current, previous), dtype, name="operator_weight")
    operator_bias = te.placeholder((current,), dtype, name="operator_bias")
    output_a_gradient = te.placeholder(
        (domain, spec, previous), dtype, name="output_lower_a_gradient"
    )
    output_bias_gradient = te.placeholder(
        (domain, spec), dtype, name="output_bias_gradient"
    )
    zero = tvm.tir.const(0.0, dtype)
    one = tvm.tir.const(1.0, dtype)
    reduce_previous = te.reduce_axis((0, previous), "reduce_previous")
    adjoint_matmul = te.compute(
        (domain, spec, current),
        lambda d_idx, s_idx, c_idx: te.sum(
            output_a_gradient[d_idx, s_idx, reduce_previous]
            * weight[c_idx, reduce_previous],
            axis=reduce_previous,
        ),
        name="adjoint_matmul",
    )
    adjoint_relu = te.compute(
        (domain, spec, current),
        lambda d_idx, s_idx, c_idx: adjoint_matmul[d_idx, s_idx, c_idx]
        + output_bias_gradient[d_idx, s_idx] * operator_bias[c_idx],
        name="adjoint_relu",
    )
    reduce_spec_alpha = te.reduce_axis((0, spec), "reduce_spec_alpha")
    native_alpha_gradient = te.compute(
        (domain, current),
        lambda d_idx, c_idx: te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(
                    incoming[d_idx, reduce_spec_alpha, c_idx] >= zero,
                    lower[d_idx, c_idx] < zero,
                    upper[d_idx, c_idx] > zero,
                    alpha[d_idx, c_idx] >= zero,
                    alpha[d_idx, c_idx] <= one,
                ),
                adjoint_relu[d_idx, reduce_spec_alpha, c_idx]
                * incoming[d_idx, reduce_spec_alpha, c_idx],
                zero,
            ),
            axis=reduce_spec_alpha,
        ),
        name="native_alpha_gradient",
    )
    reduce_spec_beta = te.reduce_axis((0, spec), "reduce_spec_beta")
    native_beta_gradient = te.compute(
        (domain, current),
        lambda d_idx, c_idx: te.sum(
            -adjoint_relu[d_idx, reduce_spec_beta, c_idx] * split[d_idx, c_idx]
            + beta[d_idx, c_idx] * zero,
            axis=reduce_spec_beta,
        ),
        name="native_beta_gradient",
    )
    return (
        te.create_prim_func(
            [
                incoming,
                lower,
                upper,
                alpha,
                beta,
                split,
                weight,
                operator_bias,
                output_a_gradient,
                output_bias_gradient,
                native_alpha_gradient,
                native_beta_gradient,
            ]
        )
        .with_attr("global_symbol", DENSE_LINEAR_BACKWARD_SYMBOL)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "b4b2-dense-linear-backward/v1")
    )


def build_dense_linear_tir_modules(
    template: DifferentiableLowerDenseLinearTIRTemplateV1,
    schedule_ir: DifferentiableLowerDenseLinearTIRScheduleV1,
):
    """Build exact unscheduled and deterministic correctness-scheduled modules."""

    template.validate()
    schedule_ir.validate_against(template)
    import tvm

    forward = _dense_linear_forward_primfunc(template)
    backward = _dense_linear_backward_primfunc(template)
    unscheduled = tvm.IRModule(
        {
            template.forward_symbol: forward,
            template.backward_symbol: backward,
        }
    )
    scheduled_functions = {}
    block_inventories = {
        template.forward_symbol: (
            ("output_lower_a", True),
            ("output_bias_delta", True),
            ("output_bias", False),
        ),
        template.backward_symbol: (
            ("adjoint_matmul", True),
            ("adjoint_relu", False),
            ("native_alpha_gradient", True),
            ("native_beta_gradient", True),
        ),
    }
    for symbol, block_names in block_inventories.items():
        schedule = tvm.tir.Schedule(tvm.IRModule({symbol: unscheduled[symbol]}))
        if symbol == template.forward_symbol:
            schedule.compute_inline(
                schedule.get_block("relu_lower_a", func_name=symbol)
            )
        for block_name, has_reduction in block_names:
            block = schedule.get_block(block_name, func_name=symbol)
            loops = schedule.get_loops(block)
            spatial = loops[:-1] if has_reduction else loops
            fused = schedule.fuse(*spatial) if len(spatial) > 1 else spatial[0]
            block_loop, thread_loop = schedule.split(
                fused, factors=[None, schedule_ir.thread_extent]
            )
            schedule.bind(block_loop, "blockIdx.x")
            schedule.bind(thread_loop, "threadIdx.x")
        scheduled_functions[symbol] = schedule.mod[symbol]
    return unscheduled, tvm.IRModule(scheduled_functions)


def compile_dense_linear_tir(
    template: DifferentiableLowerDenseLinearTIRTemplateV1,
    schedule: DifferentiableLowerDenseLinearTIRScheduleV1,
) -> CompiledDifferentiableLowerDenseLinearTIR:
    """Compile both dense semantic symbols and hash all compiler identities."""

    template.validate()
    schedule.validate_against(template)
    import tvm

    unscheduled, scheduled = build_dense_linear_tir_modules(template, schedule)
    executable = tvm.compile(
        scheduled, target=f"{template.target} -arch={template.compute_capability}"
    )
    device_sources = tuple(module.inspect_source() for module in executable.mod.imports)
    if not device_sources:
        raise RuntimeError("dense Linear TIR compile produced no CUDA device source")
    return CompiledDifferentiableLowerDenseLinearTIR(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        unscheduled_tir_hash=_sha256(tvm.ir.save_json(unscheduled)),
        scheduled_tir_hash=_sha256(tvm.ir.save_json(scheduled)),
        device_source_hash=_sha256("\n".join(device_sources)),
        tvm_version=str(tvm.__version__),
    )


__all__ = [
    "CompiledDifferentiableLowerDenseLinearTIR",
    "build_dense_linear_tir_modules",
    "compile_dense_linear_tir",
]
