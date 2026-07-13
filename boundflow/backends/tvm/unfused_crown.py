"""Unfused TVM CROWN baselines with explicit ReLU-scaled coefficient outputs."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,too-many-arguments,too-many-locals
# pylint: disable=too-many-positional-arguments

from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Tuple

from .fused_crown_conv2d import FusedCrownConv2dSignature
from .fused_crown_linear import FusedCrownLinearKey

UNFUSED_CROWN_SCHEMA_VERSION = "boundflow.unfused_crown/v1"


def _schedule_blocks(module, spatial_counts: Mapping[str, int]):
    """Apply the same deterministic 128-thread spatial schedule to every stage."""

    import tvm  # pylint: disable=import-outside-toplevel

    schedule = tvm.tir.Schedule(module)
    for block_name, spatial_count in spatial_counts.items():
        block = schedule.get_block(block_name, func_name="main")
        loops = schedule.get_loops(block)
        fused = schedule.fuse(*loops[:spatial_count])
        block_loop, thread_loop = schedule.split(fused, factors=[None, 128])
        schedule.bind(block_loop, "blockIdx.x")
        schedule.bind(thread_loop, "threadIdx.x")
    return schedule.mod


def build_unfused_crown_linear_primfunc(key: FusedCrownLinearKey):
    """Build explicit scale, matmul and bias stages for the Linear baseline."""

    key.validate()
    import tvm  # pylint: disable=import-outside-toplevel
    from tvm import te  # pylint: disable=import-outside-toplevel

    domain, spec = key.domain_batch, key.spec_batch
    current, previous, dtype = key.current_features, key.previous_features, key.dtype
    coefficient_shape = (domain, spec, current)
    relaxation_shape = (domain, current)
    coeff_u = te.placeholder(coefficient_shape, dtype=dtype, name="coeff_u")
    coeff_l = te.placeholder(coefficient_shape, dtype=dtype, name="coeff_l")
    alpha_u = te.placeholder(relaxation_shape, dtype=dtype, name="alpha_u")
    beta_u = te.placeholder(relaxation_shape, dtype=dtype, name="beta_u")
    alpha_l = te.placeholder(relaxation_shape, dtype=dtype, name="alpha_l")
    beta_l = te.placeholder(relaxation_shape, dtype=dtype, name="beta_l")
    weight = te.placeholder((current, previous), dtype=dtype, name="weight")
    bias = te.placeholder((current,), dtype=dtype, name="bias")
    zero = tvm.tir.const(0.0, dtype)
    scaled_u = te.compute(
        coefficient_shape,
        lambda d_idx, s_idx, i_idx: tvm.tir.if_then_else(
            coeff_u[d_idx, s_idx, i_idx] >= zero,
            coeff_u[d_idx, s_idx, i_idx] * alpha_u[d_idx, i_idx],
            coeff_u[d_idx, s_idx, i_idx] * alpha_l[d_idx, i_idx],
        ),
        name="scaled_u",
    )
    scaled_l = te.compute(
        coefficient_shape,
        lambda d_idx, s_idx, i_idx: tvm.tir.if_then_else(
            coeff_l[d_idx, s_idx, i_idx] >= zero,
            coeff_l[d_idx, s_idx, i_idx] * alpha_l[d_idx, i_idx],
            coeff_l[d_idx, s_idx, i_idx] * alpha_u[d_idx, i_idx],
        ),
        name="scaled_l",
    )
    reduce_u = te.reduce_axis((0, current), "reduce_u")
    reduce_l = te.reduce_axis((0, current), "reduce_l")
    previous_u = te.compute(
        (domain, spec, previous),
        lambda d_idx, s_idx, j_idx: te.sum(
            scaled_u[d_idx, s_idx, reduce_u] * weight[reduce_u, j_idx],
            axis=reduce_u,
        ),
        name="previous_u",
    )
    previous_l = te.compute(
        (domain, spec, previous),
        lambda d_idx, s_idx, j_idx: te.sum(
            scaled_l[d_idx, s_idx, reduce_l] * weight[reduce_l, j_idx],
            axis=reduce_l,
        ),
        name="previous_l",
    )
    reduce_bu = te.reduce_axis((0, current), "reduce_bu")
    reduce_bl = te.reduce_axis((0, current), "reduce_bl")
    bias_delta_u = te.compute(
        (domain, spec),
        lambda d_idx, s_idx: te.sum(
            tvm.tir.if_then_else(
                coeff_u[d_idx, s_idx, reduce_bu] >= zero,
                coeff_u[d_idx, s_idx, reduce_bu] * beta_u[d_idx, reduce_bu],
                coeff_u[d_idx, s_idx, reduce_bu] * beta_l[d_idx, reduce_bu],
            )
            + scaled_u[d_idx, s_idx, reduce_bu] * bias[reduce_bu],
            axis=reduce_bu,
        ),
        name="bias_delta_u",
    )
    bias_delta_l = te.compute(
        (domain, spec),
        lambda d_idx, s_idx: te.sum(
            tvm.tir.if_then_else(
                coeff_l[d_idx, s_idx, reduce_bl] >= zero,
                coeff_l[d_idx, s_idx, reduce_bl] * beta_l[d_idx, reduce_bl],
                coeff_l[d_idx, s_idx, reduce_bl] * beta_u[d_idx, reduce_bl],
            )
            + scaled_l[d_idx, s_idx, reduce_bl] * bias[reduce_bl],
            axis=reduce_bl,
        ),
        name="bias_delta_l",
    )
    return te.create_prim_func(
        [
            coeff_u,
            coeff_l,
            alpha_u,
            beta_u,
            alpha_l,
            beta_l,
            weight,
            bias,
            scaled_u,
            scaled_l,
            previous_u,
            previous_l,
            bias_delta_u,
            bias_delta_l,
        ]
    ).with_attr("boundflow.schema_version", UNFUSED_CROWN_SCHEMA_VERSION)


def schedule_unfused_crown_linear(key: FusedCrownLinearKey):
    """Schedule all explicit Linear stages without inlining scaled coefficients."""

    import tvm  # pylint: disable=import-outside-toplevel

    module = tvm.IRModule({"main": build_unfused_crown_linear_primfunc(key)})
    return _schedule_blocks(
        module,
        {
            "scaled_u": 3,
            "scaled_l": 3,
            "previous_u": 3,
            "previous_l": 3,
            "bias_delta_u": 2,
            "bias_delta_l": 2,
        },
    )


def build_unfused_crown_conv2d_primfunc(  # pylint: disable=too-many-locals
    signature: FusedCrownConv2dSignature,
):
    """Build explicit scale, transpose-conv gather and bias stages for Conv2d."""

    signature.validate()
    import tvm  # pylint: disable=import-outside-toplevel
    from tvm import te  # pylint: disable=import-outside-toplevel

    domain, spec = signature.domain_batch, signature.spec_batch
    input_channels = signature.input_channels
    input_h, input_w = signature.input_height, signature.input_width
    output_channels = signature.output_channels
    output_h, output_w = signature.output_height, signature.output_width
    kernel_h, kernel_w = signature.kernel_height, signature.kernel_width
    stride_h, stride_w = signature.stride
    pad_h, pad_w = signature.padding
    dilation_h, dilation_w = signature.dilation
    dtype = signature.dtype
    coefficient_shape = (domain, spec, output_channels, output_h, output_w)
    relaxation_shape = (domain, output_channels, output_h, output_w)
    coeff_u = te.placeholder(coefficient_shape, dtype=dtype, name="coeff_u")
    coeff_l = te.placeholder(coefficient_shape, dtype=dtype, name="coeff_l")
    alpha_u = te.placeholder(relaxation_shape, dtype=dtype, name="alpha_u")
    beta_u = te.placeholder(relaxation_shape, dtype=dtype, name="beta_u")
    alpha_l = te.placeholder(relaxation_shape, dtype=dtype, name="alpha_l")
    beta_l = te.placeholder(relaxation_shape, dtype=dtype, name="beta_l")
    weight = te.placeholder(
        (output_channels, input_channels, kernel_h, kernel_w),
        dtype=dtype,
        name="weight",
    )
    bias = (
        te.placeholder((output_channels,), dtype=dtype, name="bias")
        if signature.bias_present
        else None
    )
    zero = tvm.tir.const(0.0, dtype)
    scaled_u = te.compute(
        coefficient_shape,
        lambda d_idx, s_idx, co_idx, ho_idx, wo_idx: tvm.tir.if_then_else(
            coeff_u[d_idx, s_idx, co_idx, ho_idx, wo_idx] >= zero,
            coeff_u[d_idx, s_idx, co_idx, ho_idx, wo_idx]
            * alpha_u[d_idx, co_idx, ho_idx, wo_idx],
            coeff_u[d_idx, s_idx, co_idx, ho_idx, wo_idx]
            * alpha_l[d_idx, co_idx, ho_idx, wo_idx],
        ),
        name="scaled_u",
    )
    scaled_l = te.compute(
        coefficient_shape,
        lambda d_idx, s_idx, co_idx, ho_idx, wo_idx: tvm.tir.if_then_else(
            coeff_l[d_idx, s_idx, co_idx, ho_idx, wo_idx] >= zero,
            coeff_l[d_idx, s_idx, co_idx, ho_idx, wo_idx]
            * alpha_l[d_idx, co_idx, ho_idx, wo_idx],
            coeff_l[d_idx, s_idx, co_idx, ho_idx, wo_idx]
            * alpha_u[d_idx, co_idx, ho_idx, wo_idx],
        ),
        name="scaled_l",
    )

    def gather(
        scaled,
        d_idx,
        s_idx,
        ci_idx,
        hi_idx,
        wi_idx,
        co_idx,
        kh_idx,
        kw_idx,
    ):
        numerator_h = hi_idx + pad_h - kh_idx * dilation_h
        numerator_w = wi_idx + pad_w - kw_idx * dilation_w
        ho_idx = tvm.tir.floordiv(numerator_h, stride_h)
        wo_idx = tvm.tir.floordiv(numerator_w, stride_w)
        valid = tvm.tir.all(
            numerator_h >= 0,
            numerator_w >= 0,
            tvm.tir.floormod(numerator_h, stride_h) == 0,
            tvm.tir.floormod(numerator_w, stride_w) == 0,
            ho_idx < output_h,
            wo_idx < output_w,
        )
        return tvm.tir.if_then_else(
            valid,
            scaled[d_idx, s_idx, co_idx, ho_idx, wo_idx]
            * weight[co_idx, ci_idx, kh_idx, kw_idx],
            zero,
        )

    def previous(name: str, scaled):
        co_axis = te.reduce_axis((0, output_channels), f"{name}_co")
        kh_axis = te.reduce_axis((0, kernel_h), f"{name}_kh")
        kw_axis = te.reduce_axis((0, kernel_w), f"{name}_kw")
        return te.compute(
            (domain, spec, input_channels, input_h, input_w),
            lambda d_idx, s_idx, ci_idx, hi_idx, wi_idx: te.sum(
                gather(
                    scaled,
                    d_idx,
                    s_idx,
                    ci_idx,
                    hi_idx,
                    wi_idx,
                    co_axis,
                    kh_axis,
                    kw_axis,
                ),
                axis=(co_axis, kh_axis, kw_axis),
            ),
            name=name,
        )

    previous_u = previous("previous_u", scaled_u)
    previous_l = previous("previous_l", scaled_l)

    def bias_delta(name: str, coeff, scaled, positive_beta, negative_beta):
        co_axis = te.reduce_axis((0, output_channels), f"{name}_co")
        h_axis = te.reduce_axis((0, output_h), f"{name}_h")
        w_axis = te.reduce_axis((0, output_w), f"{name}_w")

        def term(d_idx, s_idx):
            coefficient = coeff[d_idx, s_idx, co_axis, h_axis, w_axis]
            relaxation = tvm.tir.if_then_else(
                coefficient >= zero,
                coefficient * positive_beta[d_idx, co_axis, h_axis, w_axis],
                coefficient * negative_beta[d_idx, co_axis, h_axis, w_axis],
            )
            affine = (
                scaled[d_idx, s_idx, co_axis, h_axis, w_axis] * bias[co_axis]
                if bias is not None
                else zero
            )
            return te.sum(relaxation + affine, axis=(co_axis, h_axis, w_axis))

        return te.compute((domain, spec), term, name=name)

    bias_delta_u = bias_delta("bias_delta_u", coeff_u, scaled_u, beta_u, beta_l)
    bias_delta_l = bias_delta("bias_delta_l", coeff_l, scaled_l, beta_l, beta_u)
    parameters = [
        coeff_u,
        coeff_l,
        alpha_u,
        beta_u,
        alpha_l,
        beta_l,
        weight,
    ]
    if bias is not None:
        parameters.append(bias)
    parameters.extend(
        [scaled_u, scaled_l, previous_u, previous_l, bias_delta_u, bias_delta_l]
    )
    return te.create_prim_func(parameters).with_attr(
        "boundflow.schema_version", UNFUSED_CROWN_SCHEMA_VERSION
    )


def schedule_unfused_crown_conv2d(signature: FusedCrownConv2dSignature):
    """Schedule explicit Conv2d stages without inlining scaled coefficients."""

    import tvm  # pylint: disable=import-outside-toplevel

    module = tvm.IRModule({"main": build_unfused_crown_conv2d_primfunc(signature)})
    return _schedule_blocks(
        module,
        {
            "scaled_u": 5,
            "scaled_l": 5,
            "previous_u": 5,
            "previous_l": 5,
            "bias_delta_u": 2,
            "bias_delta_l": 2,
        },
    )


@lru_cache(maxsize=128)
def build_unfused_crown_linear_module(key: FusedCrownLinearKey):
    """Compile/cache the explicit-workspace Linear baseline."""

    import tvm  # pylint: disable=import-outside-toplevel

    return tvm.compile(schedule_unfused_crown_linear(key), target=key.target_string)[
        "main"
    ]


@lru_cache(maxsize=128)
def build_unfused_crown_conv2d_module(signature: FusedCrownConv2dSignature):
    """Compile/cache the explicit-workspace Conv2d baseline."""

    import tvm  # pylint: disable=import-outside-toplevel

    return tvm.compile(
        schedule_unfused_crown_conv2d(signature), target=signature.target_string
    )["main"]


def explicit_workspace_bytes(
    coefficient_shape: Tuple[int, ...], *, itemsize: int = 4
) -> int:
    """Return the two scaled coefficient buffers exposed by the baseline."""

    elements = 1
    for extent in coefficient_shape:
        elements *= int(extent)
    return 2 * elements * itemsize


__all__ = [
    "UNFUSED_CROWN_SCHEMA_VERSION",
    "build_unfused_crown_conv2d_module",
    "build_unfused_crown_conv2d_primfunc",
    "build_unfused_crown_linear_module",
    "build_unfused_crown_linear_primfunc",
    "explicit_workspace_bytes",
    "schedule_unfused_crown_conv2d",
    "schedule_unfused_crown_linear",
]
