"""B4-B2 B2-2 S-anchor sparse-source Linear forward/backward CUDA TIR."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-locals
# pylint: disable=missing-function-docstring,too-many-arguments
# pylint: disable=too-many-positional-arguments

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)
from boundflow.ir.differentiable_lower_sparse_linear_tir import (
    SPARSE_LINEAR_BACKWARD_SYMBOL,
    SPARSE_LINEAR_FORWARD_SYMBOL,
    DifferentiableLowerSparseLinearTIRScheduleV1,
    DifferentiableLowerSparseLinearTIRTemplateV1,
)


@dataclass(frozen=True)
class CompiledDifferentiableLowerSparseLinearTIR:
    """Compiled runtime and independently hashable sparse-source identities."""

    executable: DifferentiableLowerTIRExecutable
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    tvm_version: str
    observed_workspace_names: tuple[str, ...]
    forbidden_workspace_count: int


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compressed_alpha_value(template, alpha, domain_index, current_index, zero):
    value = zero
    for ordinal, feature in reversed(tuple(enumerate(template.alpha_feature_indices))):
        import tvm

        value = tvm.tir.if_then_else(
            current_index == feature, alpha[domain_index, ordinal], value
        )
    return value


def _feature_for_ordinal(template, ordinal_index):
    import tvm

    value = tvm.tir.const(0, "int32")
    for ordinal, feature in reversed(tuple(enumerate(template.alpha_feature_indices))):
        value = tvm.tir.if_then_else(
            ordinal_index == ordinal, tvm.tir.const(feature, "int32"), value
        )
    return value


def _beta_location_for_domain(template, domain_index):
    import tvm

    value = tvm.tir.const(0, "int32")
    for domain, location in reversed(tuple(enumerate(template.beta_locations))):
        value = tvm.tir.if_then_else(
            domain_index == domain, tvm.tir.const(location, "int32"), value
        )
    return value


def _beta_sign_for_domain(template, domain_index, dtype):
    import tvm

    value = tvm.tir.const(0.0, dtype)
    for domain, sign in reversed(tuple(enumerate(template.beta_signs))):
        value = tvm.tir.if_then_else(
            domain_index == domain, tvm.tir.const(float(sign), dtype), value
        )
    return value


def _beta_pre_add(template, beta, domain_index, current_index, zero, dtype):
    import tvm

    location = _beta_location_for_domain(template, domain_index)
    sign = _beta_sign_for_domain(template, domain_index, dtype)
    return tvm.tir.if_then_else(
        current_index == location, -beta[domain_index, 0] * sign, zero
    )


def _sparse_linear_forward_primfunc(
    template: DifferentiableLowerSparseLinearTIRTemplateV1,
):
    import tvm
    from tvm import te

    domain = template.domain_count
    spec = template.spec_count
    current = template.current_features
    previous = template.previous_features
    alpha_features = template.compressed_alpha_features
    beta_entries = template.compressed_beta_entries
    dtype = "float32"
    incoming = te.placeholder((domain, spec, current), dtype, name="incoming_lower_a")
    lower = te.placeholder((domain, current), dtype, name="preactivation_lower")
    upper = te.placeholder((domain, current), dtype, name="preactivation_upper")
    alpha = te.placeholder((domain, alpha_features), dtype, name="compressed_alpha")
    beta = te.placeholder((domain, beta_entries), dtype, name="compressed_beta")
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
        source_alpha = _compressed_alpha_value(template, alpha, d_idx, c_idx, zero)
        clamped = tvm.tir.min(tvm.tir.max(source_alpha, zero), one)
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
        + _beta_pre_add(template, beta, d_idx, c_idx, zero, dtype),
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
                incoming_bias,
                weight,
                operator_bias,
                output_lower_a,
                output_bias,
            ]
        )
        .with_attr("global_symbol", SPARSE_LINEAR_FORWARD_SYMBOL)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "b4b2-sparse-linear-forward/v1")
    )


def _sparse_linear_backward_primfunc(
    template: DifferentiableLowerSparseLinearTIRTemplateV1,
):
    import tvm
    from tvm import te

    domain = template.domain_count
    spec = template.spec_count
    current = template.current_features
    previous = template.previous_features
    alpha_features = template.compressed_alpha_features
    beta_entries = template.compressed_beta_entries
    dtype = "float32"
    incoming = te.placeholder((domain, spec, current), dtype, name="incoming_lower_a")
    lower = te.placeholder((domain, current), dtype, name="preactivation_lower")
    upper = te.placeholder((domain, current), dtype, name="preactivation_upper")
    alpha = te.placeholder((domain, alpha_features), dtype, name="compressed_alpha")
    beta = te.placeholder((domain, beta_entries), dtype, name="compressed_beta")
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

    def alpha_gradient(d_idx, k_idx):
        feature = _feature_for_ordinal(template, k_idx)
        return te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(
                    incoming[d_idx, reduce_spec_alpha, feature] >= zero,
                    lower[d_idx, feature] < zero,
                    upper[d_idx, feature] > zero,
                    alpha[d_idx, k_idx] >= zero,
                    alpha[d_idx, k_idx] <= one,
                ),
                adjoint_relu[d_idx, reduce_spec_alpha, feature]
                * incoming[d_idx, reduce_spec_alpha, feature],
                zero,
            ),
            axis=reduce_spec_alpha,
        )

    compressed_alpha_gradient = te.compute(
        (domain, alpha_features), alpha_gradient, name="compressed_alpha_gradient"
    )
    reduce_spec_beta = te.reduce_axis((0, spec), "reduce_spec_beta")

    def beta_gradient(d_idx, q_idx):
        location = _beta_location_for_domain(template, d_idx)
        sign = _beta_sign_for_domain(template, d_idx, dtype)
        return te.sum(
            -adjoint_relu[d_idx, reduce_spec_beta, location] * sign
            + beta[d_idx, q_idx] * zero,
            axis=reduce_spec_beta,
        )

    compressed_beta_gradient = te.compute(
        (domain, beta_entries), beta_gradient, name="compressed_beta_gradient"
    )
    return (
        te.create_prim_func(
            [
                incoming,
                lower,
                upper,
                alpha,
                beta,
                weight,
                operator_bias,
                output_a_gradient,
                output_bias_gradient,
                compressed_alpha_gradient,
                compressed_beta_gradient,
            ]
        )
        .with_attr("global_symbol", SPARSE_LINEAR_BACKWARD_SYMBOL)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "b4b2-sparse-linear-backward/v1")
    )


def build_sparse_linear_tir_modules(
    template: DifferentiableLowerSparseLinearTIRTemplateV1,
    schedule_ir: DifferentiableLowerSparseLinearTIRScheduleV1,
):
    """Build the unscheduled and deterministic no-dense-state modules."""

    template.validate()
    schedule_ir.validate_against(template)
    import tvm

    forward = _sparse_linear_forward_primfunc(template)
    backward = _sparse_linear_backward_primfunc(template)
    unscheduled = tvm.IRModule(
        {template.forward_symbol: forward, template.backward_symbol: backward}
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
            ("compressed_alpha_gradient", True),
            ("compressed_beta_gradient", True),
        ),
    }
    for symbol, block_names in block_inventories.items():
        schedule = tvm.tir.Schedule(tvm.IRModule({symbol: unscheduled[symbol]}))
        if symbol == template.forward_symbol:
            schedule.compute_inline(
                schedule.get_block("relu_lower_a", func_name=symbol)
            )
        else:
            schedule.compute_inline(
                schedule.get_block("adjoint_relu", func_name=symbol)
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
    scheduled = tvm.IRModule(scheduled_functions)
    script = scheduled.script(show_meta=False)
    forbidden_count = sum(
        script.count(name) for name in schedule_ir.forbidden_global_workspaces
    )
    if forbidden_count:
        raise RuntimeError(
            "sparse-source Linear TIR materialized forbidden dense state"
        )
    return unscheduled, scheduled


def compile_sparse_linear_tir(
    template: DifferentiableLowerSparseLinearTIRTemplateV1,
    schedule: DifferentiableLowerSparseLinearTIRScheduleV1,
) -> CompiledDifferentiableLowerSparseLinearTIR:
    """Compile both sparse-source symbols and bind the workspace ledger."""

    template.validate()
    schedule.validate_against(template)
    import tvm

    unscheduled, scheduled = build_sparse_linear_tir_modules(template, schedule)
    executable = tvm.compile(
        scheduled, target=f"{template.target} -arch={template.compute_capability}"
    )
    device_sources = tuple(module.inspect_source() for module in executable.mod.imports)
    if not device_sources:
        raise RuntimeError("sparse-source Linear TIR compile produced no CUDA source")
    script = scheduled.script(show_meta=False)
    forbidden_count = sum(
        script.count(name) for name in schedule.forbidden_global_workspaces
    )
    observed_workspaces = tuple(
        name for name in schedule.workspace_names if script.count(name) > 0
    )
    if observed_workspaces != schedule.workspace_names:
        raise RuntimeError("sparse-source Linear workspace inventory differs")
    return CompiledDifferentiableLowerSparseLinearTIR(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        unscheduled_tir_hash=_sha256(tvm.ir.save_json(unscheduled)),
        scheduled_tir_hash=_sha256(tvm.ir.save_json(scheduled)),
        device_source_hash=_sha256("\n".join(device_sources)),
        tvm_version=str(tvm.__version__),
        observed_workspace_names=observed_workspaces,
        forbidden_workspace_count=forbidden_count,
    )


__all__ = [
    "CompiledDifferentiableLowerSparseLinearTIR",
    "build_sparse_linear_tir_modules",
    "compile_sparse_linear_tir",
]
