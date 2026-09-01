"""TVM/TIR for a root CROWN two-Conv residual block and full VJP."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-locals
# pylint: disable=missing-function-docstring,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,too-many-statements
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)

ROOT_RESIDUAL_FORWARD_SYMBOL = "boundflow_root_crown_residual_forward_v1"
ROOT_RESIDUAL_BACKWARD_SYMBOL = "boundflow_root_crown_residual_backward_v1"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RootCrownResidualTemplateV1:
    """Static residual geometry with independent spec/domain axes."""

    spec_count: int
    domain_count: int
    channels: int
    height: int
    width: int
    entry_alpha_coordinates: tuple[tuple[int, int, int], ...]
    inner_alpha_coordinates: tuple[tuple[int, int, int], ...]
    compute_capability: str
    thread_extent: int = 128
    kernel: tuple[int, int] = (3, 3)
    padding: tuple[int, int] = (1, 1)
    target: str = "cuda"
    forward_symbol: str = ROOT_RESIDUAL_FORWARD_SYMBOL
    backward_symbol: str = ROOT_RESIDUAL_BACKWARD_SYMBOL

    @property
    def coefficient_shape(self) -> tuple[int, ...]:
        return (
            self.spec_count,
            self.domain_count,
            self.channels,
            self.height,
            self.width,
        )

    @property
    def bound_shape(self) -> tuple[int, ...]:
        return (self.domain_count, self.channels, self.height, self.width)

    @property
    def weight_shape(self) -> tuple[int, ...]:
        return (self.channels, self.channels, *self.kernel)

    @property
    def entry_alpha_count(self) -> int:
        return len(self.entry_alpha_coordinates)

    @property
    def inner_alpha_count(self) -> int:
        return len(self.inner_alpha_coordinates)

    def validate(self) -> None:
        coordinates = (
            self.entry_alpha_coordinates,
            self.inner_alpha_coordinates,
        )
        if (
            self.spec_count < 1
            or self.domain_count < 1
            or self.channels < 1
            or self.height < 1
            or self.width < 1
            or any(not values for values in coordinates)
            or any(len(set(values)) != len(values) for values in coordinates)
            or any(
                not (
                    0 <= channel < self.channels
                    and 0 <= height < self.height
                    and 0 <= width < self.width
                )
                for values in coordinates
                for channel, height, width in values
            )
            or not self.compute_capability.startswith("sm_")
            or self.thread_extent not in {32, 64, 128, 256, 512, 1024}
            or self.kernel != (3, 3)
            or self.padding != (1, 1)
            or self.target != "cuda"
            or self.forward_symbol != ROOT_RESIDUAL_FORWARD_SYMBOL
            or self.backward_symbol != ROOT_RESIDUAL_BACKWARD_SYMBOL
        ):
            raise ValueError("root CROWN residual template differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": "boundflow.root-crown-residual-template/v1",
            "spec_count": self.spec_count,
            "domain_count": self.domain_count,
            "channels": self.channels,
            "height": self.height,
            "width": self.width,
            "entry_alpha_coordinates": [
                list(value) for value in self.entry_alpha_coordinates
            ],
            "inner_alpha_coordinates": [
                list(value) for value in self.inner_alpha_coordinates
            ],
            "compute_capability": self.compute_capability,
            "thread_extent": self.thread_extent,
            "kernel": list(self.kernel),
            "padding": list(self.padding),
            "target": self.target,
            "forward_symbol": self.forward_symbol,
            "backward_symbol": self.backward_symbol,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class CompiledRootCrownResidualTIRV1:
    """Compiled residual forward/backward module and compiler identities."""

    executable: DifferentiableLowerTIRExecutable
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    workspace_inventory: tuple[tuple[str, tuple[int, ...]], ...]
    forward_parallel_reduction_count: int
    forward_bias_reduction_lanes: int
    tvm_version: str


def _alpha_value(
    tvm,
    raw_alpha,
    feature_to_ordinal,
    spec,
    domain,
    channel,
    height,
    width,
):
    ordinal = feature_to_ordinal[channel, height, width]
    zero = tvm.tir.const(0.0, "float32")
    return tvm.tir.if_then_else(
        ordinal >= 0,
        raw_alpha[0, spec, domain, tvm.tir.max(ordinal, 0)],
        zero,
    )


def _relu_terms(tvm, incoming, lower, upper, alpha):
    zero = tvm.tir.const(0.0, "float32")
    one = tvm.tir.const(1.0, "float32")
    epsilon = tvm.tir.const(1.1920928955078125e-07, "float32")
    ambiguous = tvm.tir.all(lower < zero, upper > zero)
    denominator = tvm.tir.max(upper - lower, epsilon)
    upper_slope = tvm.tir.if_then_else(
        lower >= zero,
        one,
        tvm.tir.if_then_else(upper <= zero, zero, upper / denominator),
    )
    lower_slope = tvm.tir.if_then_else(
        ambiguous,
        tvm.tir.min(tvm.tir.max(alpha, zero), one),
        tvm.tir.if_then_else(lower >= zero, one, zero),
    )
    slope = tvm.tir.if_then_else(incoming >= zero, lower_slope, upper_slope)
    intercept = tvm.tir.if_then_else(
        tvm.tir.all(incoming < zero, ambiguous), -lower * upper_slope, zero
    )
    return slope, intercept


def _forward_primfunc(template: RootCrownResidualTemplateV1):
    import tvm  # type: ignore[import-not-found]
    from tvm import te

    spec = template.spec_count
    domain = template.domain_count
    channels = template.channels
    height = template.height
    width = template.width
    coefficient_shape = template.coefficient_shape
    bound_shape = template.bound_shape
    incoming = te.placeholder(coefficient_shape, "float32", name="incoming_lower_a")
    entry_lower = te.placeholder(bound_shape, "float32", name="entry_lower")
    entry_upper = te.placeholder(bound_shape, "float32", name="entry_upper")
    entry_alpha = te.placeholder(
        (2, spec, domain, template.entry_alpha_count),
        "float32",
        name="entry_raw_alpha",
    )
    entry_map = te.placeholder(bound_shape[1:], "int32", name="entry_alpha_map")
    main_weight = te.placeholder(template.weight_shape, "float32", name="main_weight")
    main_bias = te.placeholder((channels,), "float32", name="main_bias")
    inner_lower = te.placeholder(bound_shape, "float32", name="inner_lower")
    inner_upper = te.placeholder(bound_shape, "float32", name="inner_upper")
    inner_alpha = te.placeholder(
        (2, spec, domain, template.inner_alpha_count),
        "float32",
        name="inner_raw_alpha",
    )
    inner_map = te.placeholder(bound_shape[1:], "int32", name="inner_alpha_map")
    inner_weight = te.placeholder(template.weight_shape, "float32", name="inner_weight")
    inner_bias = te.placeholder((channels,), "float32", name="inner_bias")
    zero = tvm.tir.const(0.0, "float32")

    def entry_element(s_idx, d_idx, c_idx, h_idx, w_idx):
        alpha = _alpha_value(
            tvm,
            entry_alpha,
            entry_map,
            s_idx,
            d_idx,
            c_idx,
            h_idx,
            w_idx,
        )
        slope, _ = _relu_terms(
            tvm,
            incoming[s_idx, d_idx, c_idx, h_idx, w_idx],
            entry_lower[d_idx, c_idx, h_idx, w_idx],
            entry_upper[d_idx, c_idx, h_idx, w_idx],
            alpha,
        )
        return incoming[s_idx, d_idx, c_idx, h_idx, w_idx] * slope

    entry_a = te.compute(coefficient_shape, entry_element, name="entry_relu_a")
    reduce_main_c = te.reduce_axis((0, channels), "reduce_main_c")
    reduce_main_h = te.reduce_axis((0, 3), "reduce_main_h")
    reduce_main_w = te.reduce_axis((0, 3), "reduce_main_w")

    def main_element(s_idx, d_idx, ci_idx, ih_idx, iw_idx):
        oh_idx = ih_idx + 1 - reduce_main_h
        ow_idx = iw_idx + 1 - reduce_main_w
        valid = tvm.tir.all(0 <= oh_idx, oh_idx < height, 0 <= ow_idx, ow_idx < width)
        return te.sum(
            tvm.tir.if_then_else(
                valid,
                entry_a[s_idx, d_idx, reduce_main_c, oh_idx, ow_idx]
                * main_weight[
                    reduce_main_c,
                    ci_idx,
                    reduce_main_h,
                    reduce_main_w,
                ],
                zero,
            ),
            axis=(reduce_main_c, reduce_main_h, reduce_main_w),
        )

    main_a = te.compute(coefficient_shape, main_element, name="main_conv_a")

    def inner_element(s_idx, d_idx, c_idx, h_idx, w_idx):
        alpha = _alpha_value(
            tvm,
            inner_alpha,
            inner_map,
            s_idx,
            d_idx,
            c_idx,
            h_idx,
            w_idx,
        )
        slope, _ = _relu_terms(
            tvm,
            main_a[s_idx, d_idx, c_idx, h_idx, w_idx],
            inner_lower[d_idx, c_idx, h_idx, w_idx],
            inner_upper[d_idx, c_idx, h_idx, w_idx],
            alpha,
        )
        return main_a[s_idx, d_idx, c_idx, h_idx, w_idx] * slope

    inner_a = te.compute(coefficient_shape, inner_element, name="inner_relu_a")
    reduce_inner_c = te.reduce_axis((0, channels), "reduce_inner_c")
    reduce_inner_h = te.reduce_axis((0, 3), "reduce_inner_h")
    reduce_inner_w = te.reduce_axis((0, 3), "reduce_inner_w")

    def residual_element(s_idx, d_idx, ci_idx, ih_idx, iw_idx):
        oh_idx = ih_idx + 1 - reduce_inner_h
        ow_idx = iw_idx + 1 - reduce_inner_w
        valid = tvm.tir.all(0 <= oh_idx, oh_idx < height, 0 <= ow_idx, ow_idx < width)
        return te.sum(
            tvm.tir.if_then_else(
                valid,
                inner_a[s_idx, d_idx, reduce_inner_c, oh_idx, ow_idx]
                * inner_weight[
                    reduce_inner_c,
                    ci_idx,
                    reduce_inner_h,
                    reduce_inner_w,
                ],
                zero,
            ),
            axis=(reduce_inner_c, reduce_inner_h, reduce_inner_w),
        )

    residual_a = te.compute(coefficient_shape, residual_element, name="residual_conv_a")
    output_a = te.compute(
        coefficient_shape,
        lambda s_idx, d_idx, c_idx, h_idx, w_idx: entry_a[
            s_idx, d_idx, c_idx, h_idx, w_idx
        ]
        + residual_a[s_idx, d_idx, c_idx, h_idx, w_idx],
        name="output_lower_a",
    )
    reduce_entry_c = te.reduce_axis((0, channels), "reduce_entry_bias_c")
    reduce_entry_h = te.reduce_axis((0, height), "reduce_entry_bias_h")
    reduce_entry_w = te.reduce_axis((0, width), "reduce_entry_bias_w")

    def entry_bias_element(s_idx, d_idx):
        alpha = _alpha_value(
            tvm,
            entry_alpha,
            entry_map,
            s_idx,
            d_idx,
            reduce_entry_c,
            reduce_entry_h,
            reduce_entry_w,
        )
        _, intercept = _relu_terms(
            tvm,
            incoming[s_idx, d_idx, reduce_entry_c, reduce_entry_h, reduce_entry_w],
            entry_lower[d_idx, reduce_entry_c, reduce_entry_h, reduce_entry_w],
            entry_upper[d_idx, reduce_entry_c, reduce_entry_h, reduce_entry_w],
            alpha,
        )
        return te.sum(
            incoming[s_idx, d_idx, reduce_entry_c, reduce_entry_h, reduce_entry_w]
            * intercept
            + entry_a[s_idx, d_idx, reduce_entry_c, reduce_entry_h, reduce_entry_w]
            * main_bias[reduce_entry_c],
            axis=(reduce_entry_c, reduce_entry_h, reduce_entry_w),
        )

    entry_bias_delta = te.compute(
        (spec, domain), entry_bias_element, name="entry_bias_delta"
    )
    reduce_inner_bc = te.reduce_axis((0, channels), "reduce_inner_bias_c")
    reduce_inner_bh = te.reduce_axis((0, height), "reduce_inner_bias_h")
    reduce_inner_bw = te.reduce_axis((0, width), "reduce_inner_bias_w")

    def inner_bias_element(s_idx, d_idx):
        alpha = _alpha_value(
            tvm,
            inner_alpha,
            inner_map,
            s_idx,
            d_idx,
            reduce_inner_bc,
            reduce_inner_bh,
            reduce_inner_bw,
        )
        _, intercept = _relu_terms(
            tvm,
            main_a[s_idx, d_idx, reduce_inner_bc, reduce_inner_bh, reduce_inner_bw],
            inner_lower[d_idx, reduce_inner_bc, reduce_inner_bh, reduce_inner_bw],
            inner_upper[d_idx, reduce_inner_bc, reduce_inner_bh, reduce_inner_bw],
            alpha,
        )
        return te.sum(
            main_a[s_idx, d_idx, reduce_inner_bc, reduce_inner_bh, reduce_inner_bw]
            * intercept
            + inner_a[s_idx, d_idx, reduce_inner_bc, reduce_inner_bh, reduce_inner_bw]
            * inner_bias[reduce_inner_bc],
            axis=(reduce_inner_bc, reduce_inner_bh, reduce_inner_bw),
        )

    inner_bias_delta = te.compute(
        (spec, domain), inner_bias_element, name="inner_bias_delta"
    )
    output_bias = te.compute(
        (spec, domain),
        lambda s_idx, d_idx: entry_bias_delta[s_idx, d_idx]
        + inner_bias_delta[s_idx, d_idx],
        name="output_bias",
    )
    return (
        te.create_prim_func(
            [
                incoming,
                entry_lower,
                entry_upper,
                entry_alpha,
                entry_map,
                main_weight,
                main_bias,
                inner_lower,
                inner_upper,
                inner_alpha,
                inner_map,
                inner_weight,
                inner_bias,
                main_a,
                output_a,
                output_bias,
            ]
        )
        .with_attr("global_symbol", template.forward_symbol)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "root-crown-residual-forward/v1")
    )


def _relu_bound_derivative(
    tvm, incoming, lower, upper, transformed_adjoint, bias_adjoint, kind
):
    zero = tvm.tir.const(0.0, "float32")
    epsilon = tvm.tir.const(1.1920928955078125e-07, "float32")
    delta = upper - lower
    denominator = tvm.tir.max(delta, epsilon)
    denominator_square = denominator * denominator
    regular = delta >= epsilon
    dq_dl = tvm.tir.if_then_else(regular, upper / denominator_square, zero)
    dq_du = tvm.tir.if_then_else(
        regular, -lower / denominator_square, tvm.tir.const(1.0, "float32") / epsilon
    )
    dr_dl = tvm.tir.if_then_else(
        regular, -upper * upper / denominator_square, -upper / epsilon
    )
    dr_du = tvm.tir.if_then_else(
        regular, lower * lower / denominator_square, -lower / epsilon
    )
    derivative_slope = tvm.tir.if_then_else(kind == 0, dq_dl, dq_du)
    derivative_intercept = tvm.tir.if_then_else(kind == 0, dr_dl, dr_du)
    return tvm.tir.if_then_else(
        tvm.tir.all(incoming < zero, lower < zero, upper > zero),
        transformed_adjoint * incoming * derivative_slope
        + bias_adjoint * incoming * derivative_intercept,
        zero,
    )


def _backward_primfunc(template: RootCrownResidualTemplateV1):
    import tvm  # type: ignore[import-not-found]
    from tvm import te

    spec = template.spec_count
    domain = template.domain_count
    channels = template.channels
    height = template.height
    width = template.width
    coefficient_shape = template.coefficient_shape
    bound_shape = template.bound_shape
    incoming = te.placeholder(coefficient_shape, "float32", name="incoming_lower_a")
    entry_lower = te.placeholder(bound_shape, "float32", name="entry_lower")
    entry_upper = te.placeholder(bound_shape, "float32", name="entry_upper")
    entry_alpha = te.placeholder(
        (2, spec, domain, template.entry_alpha_count),
        "float32",
        name="entry_raw_alpha",
    )
    entry_map = te.placeholder(bound_shape[1:], "int32", name="entry_alpha_map")
    entry_coordinates = te.placeholder(
        (3, template.entry_alpha_count), "int32", name="entry_alpha_coordinates"
    )
    main_weight = te.placeholder(template.weight_shape, "float32", name="main_weight")
    main_bias = te.placeholder((channels,), "float32", name="main_bias")
    inner_lower = te.placeholder(bound_shape, "float32", name="inner_lower")
    inner_upper = te.placeholder(bound_shape, "float32", name="inner_upper")
    inner_alpha = te.placeholder(
        (2, spec, domain, template.inner_alpha_count),
        "float32",
        name="inner_raw_alpha",
    )
    inner_map = te.placeholder(bound_shape[1:], "int32", name="inner_alpha_map")
    inner_coordinates = te.placeholder(
        (3, template.inner_alpha_count), "int32", name="inner_alpha_coordinates"
    )
    inner_weight = te.placeholder(template.weight_shape, "float32", name="inner_weight")
    inner_bias = te.placeholder((channels,), "float32", name="inner_bias")
    output_a_gradient = te.placeholder(
        coefficient_shape, "float32", name="output_a_gradient"
    )
    output_bias_gradient = te.placeholder(
        (spec, domain), "float32", name="output_bias_gradient"
    )
    zero = tvm.tir.const(0.0, "float32")
    one = tvm.tir.const(1.0, "float32")

    def entry_element(s_idx, d_idx, c_idx, h_idx, w_idx):
        alpha = _alpha_value(
            tvm,
            entry_alpha,
            entry_map,
            s_idx,
            d_idx,
            c_idx,
            h_idx,
            w_idx,
        )
        slope, _ = _relu_terms(
            tvm,
            incoming[s_idx, d_idx, c_idx, h_idx, w_idx],
            entry_lower[d_idx, c_idx, h_idx, w_idx],
            entry_upper[d_idx, c_idx, h_idx, w_idx],
            alpha,
        )
        return incoming[s_idx, d_idx, c_idx, h_idx, w_idx] * slope

    entry_a = te.compute(coefficient_shape, entry_element, name="backward_entry_a")
    main_rc = te.reduce_axis((0, channels), "backward_main_rc")
    main_rh = te.reduce_axis((0, 3), "backward_main_rh")
    main_rw = te.reduce_axis((0, 3), "backward_main_rw")

    def main_element(s_idx, d_idx, ci_idx, ih_idx, iw_idx):
        oh_idx = ih_idx + 1 - main_rh
        ow_idx = iw_idx + 1 - main_rw
        valid = tvm.tir.all(0 <= oh_idx, oh_idx < height, 0 <= ow_idx, ow_idx < width)
        return te.sum(
            tvm.tir.if_then_else(
                valid,
                entry_a[s_idx, d_idx, main_rc, oh_idx, ow_idx]
                * main_weight[main_rc, ci_idx, main_rh, main_rw],
                zero,
            ),
            axis=(main_rc, main_rh, main_rw),
        )

    main_a = te.compute(coefficient_shape, main_element, name="backward_main_a")
    inner_adj_ci = te.reduce_axis((0, channels), "inner_adjoint_ci")
    inner_adj_h = te.reduce_axis((0, 3), "inner_adjoint_h")
    inner_adj_w = te.reduce_axis((0, 3), "inner_adjoint_w")

    def inner_conv_adjoint_element(s_idx, d_idx, co_idx, oh_idx, ow_idx):
        ih_idx = oh_idx - 1 + inner_adj_h
        iw_idx = ow_idx - 1 + inner_adj_w
        return te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(0 <= ih_idx, ih_idx < height, 0 <= iw_idx, iw_idx < width),
                output_a_gradient[s_idx, d_idx, inner_adj_ci, ih_idx, iw_idx]
                * inner_weight[co_idx, inner_adj_ci, inner_adj_h, inner_adj_w],
                zero,
            ),
            axis=(inner_adj_ci, inner_adj_h, inner_adj_w),
        )

    inner_conv_adjoint = te.compute(
        coefficient_shape,
        inner_conv_adjoint_element,
        name="inner_conv_adjoint",
    )

    inner_adjoint = te.compute(
        coefficient_shape,
        lambda s_idx, d_idx, c_idx, h_idx, w_idx: inner_conv_adjoint[
            s_idx, d_idx, c_idx, h_idx, w_idx
        ]
        + output_bias_gradient[s_idx, d_idx] * inner_bias[c_idx],
        name="inner_transformed_adjoint",
    )

    def main_adjoint_element(s_idx, d_idx, c_idx, h_idx, w_idx):
        alpha = _alpha_value(
            tvm,
            inner_alpha,
            inner_map,
            s_idx,
            d_idx,
            c_idx,
            h_idx,
            w_idx,
        )
        slope, intercept = _relu_terms(
            tvm,
            main_a[s_idx, d_idx, c_idx, h_idx, w_idx],
            inner_lower[d_idx, c_idx, h_idx, w_idx],
            inner_upper[d_idx, c_idx, h_idx, w_idx],
            alpha,
        )
        return (
            inner_adjoint[s_idx, d_idx, c_idx, h_idx, w_idx] * slope
            + output_bias_gradient[s_idx, d_idx] * intercept
        )

    main_adjoint = te.compute(
        coefficient_shape, main_adjoint_element, name="main_conv_adjoint"
    )
    entry_adj_ci = te.reduce_axis((0, channels), "entry_adjoint_ci")
    entry_adj_h = te.reduce_axis((0, 3), "entry_adjoint_h")
    entry_adj_w = te.reduce_axis((0, 3), "entry_adjoint_w")

    def entry_conv_adjoint_element(s_idx, d_idx, co_idx, oh_idx, ow_idx):
        ih_idx = oh_idx - 1 + entry_adj_h
        iw_idx = ow_idx - 1 + entry_adj_w
        return te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(0 <= ih_idx, ih_idx < height, 0 <= iw_idx, iw_idx < width),
                main_adjoint[s_idx, d_idx, entry_adj_ci, ih_idx, iw_idx]
                * main_weight[co_idx, entry_adj_ci, entry_adj_h, entry_adj_w],
                zero,
            ),
            axis=(entry_adj_ci, entry_adj_h, entry_adj_w),
        )

    entry_conv_adjoint = te.compute(
        coefficient_shape,
        entry_conv_adjoint_element,
        name="entry_conv_adjoint",
    )

    entry_adjoint = te.compute(
        coefficient_shape,
        lambda s_idx, d_idx, c_idx, h_idx, w_idx: output_a_gradient[
            s_idx, d_idx, c_idx, h_idx, w_idx
        ]
        + entry_conv_adjoint[s_idx, d_idx, c_idx, h_idx, w_idx]
        + output_bias_gradient[s_idx, d_idx] * main_bias[c_idx],
        name="entry_transformed_adjoint",
    )

    def incoming_gradient_element(s_idx, d_idx, c_idx, h_idx, w_idx):
        alpha = _alpha_value(
            tvm,
            entry_alpha,
            entry_map,
            s_idx,
            d_idx,
            c_idx,
            h_idx,
            w_idx,
        )
        slope, intercept = _relu_terms(
            tvm,
            incoming[s_idx, d_idx, c_idx, h_idx, w_idx],
            entry_lower[d_idx, c_idx, h_idx, w_idx],
            entry_upper[d_idx, c_idx, h_idx, w_idx],
            alpha,
        )
        return (
            entry_adjoint[s_idx, d_idx, c_idx, h_idx, w_idx] * slope
            + output_bias_gradient[s_idx, d_idx] * intercept
        )

    incoming_gradient = te.compute(
        coefficient_shape, incoming_gradient_element, name="incoming_gradient"
    )

    def alpha_gradient(
        raw_alpha,
        coordinates,
        source,
        lower,
        upper,
        adjoint,
        plane,
        s_idx,
        d_idx,
        ordinal,
    ):
        c_idx = coordinates[0, ordinal]
        h_idx = coordinates[1, ordinal]
        w_idx = coordinates[2, ordinal]
        alpha = raw_alpha[0, s_idx, d_idx, ordinal]
        return tvm.tir.if_then_else(
            tvm.tir.all(
                plane == 0,
                source[s_idx, d_idx, c_idx, h_idx, w_idx] >= zero,
                lower[d_idx, c_idx, h_idx, w_idx] < zero,
                upper[d_idx, c_idx, h_idx, w_idx] > zero,
                alpha >= zero,
                alpha <= one,
            ),
            adjoint[s_idx, d_idx, c_idx, h_idx, w_idx]
            * source[s_idx, d_idx, c_idx, h_idx, w_idx],
            zero,
        )

    entry_alpha_gradient = te.compute(
        (2, spec, domain, template.entry_alpha_count),
        lambda plane, s_idx, d_idx, ordinal: alpha_gradient(
            entry_alpha,
            entry_coordinates,
            incoming,
            entry_lower,
            entry_upper,
            entry_adjoint,
            plane,
            s_idx,
            d_idx,
            ordinal,
        ),
        name="entry_alpha_gradient",
    )
    inner_alpha_gradient = te.compute(
        (2, spec, domain, template.inner_alpha_count),
        lambda plane, s_idx, d_idx, ordinal: alpha_gradient(
            inner_alpha,
            inner_coordinates,
            main_a,
            inner_lower,
            inner_upper,
            inner_adjoint,
            plane,
            s_idx,
            d_idx,
            ordinal,
        ),
        name="inner_alpha_gradient",
    )
    entry_reduce_spec = te.reduce_axis((0, spec), "entry_bound_spec")
    inner_reduce_spec = te.reduce_axis((0, spec), "inner_bound_spec")

    def entry_bound_gradient(kind, d_idx, c_idx, h_idx, w_idx):
        return te.sum(
            _relu_bound_derivative(
                tvm,
                incoming[entry_reduce_spec, d_idx, c_idx, h_idx, w_idx],
                entry_lower[d_idx, c_idx, h_idx, w_idx],
                entry_upper[d_idx, c_idx, h_idx, w_idx],
                entry_adjoint[entry_reduce_spec, d_idx, c_idx, h_idx, w_idx],
                output_bias_gradient[entry_reduce_spec, d_idx],
                kind,
            ),
            axis=entry_reduce_spec,
        )

    def inner_bound_gradient(kind, d_idx, c_idx, h_idx, w_idx):
        return te.sum(
            _relu_bound_derivative(
                tvm,
                main_a[inner_reduce_spec, d_idx, c_idx, h_idx, w_idx],
                inner_lower[d_idx, c_idx, h_idx, w_idx],
                inner_upper[d_idx, c_idx, h_idx, w_idx],
                inner_adjoint[inner_reduce_spec, d_idx, c_idx, h_idx, w_idx],
                output_bias_gradient[inner_reduce_spec, d_idx],
                kind,
            ),
            axis=inner_reduce_spec,
        )

    entry_lower_gradient = te.compute(
        bound_shape,
        lambda d_idx, c_idx, h_idx, w_idx: entry_bound_gradient(
            0, d_idx, c_idx, h_idx, w_idx
        ),
        name="entry_lower_gradient",
    )
    entry_upper_gradient = te.compute(
        bound_shape,
        lambda d_idx, c_idx, h_idx, w_idx: entry_bound_gradient(
            1, d_idx, c_idx, h_idx, w_idx
        ),
        name="entry_upper_gradient",
    )
    inner_lower_gradient = te.compute(
        bound_shape,
        lambda d_idx, c_idx, h_idx, w_idx: inner_bound_gradient(
            0, d_idx, c_idx, h_idx, w_idx
        ),
        name="inner_lower_gradient",
    )
    inner_upper_gradient = te.compute(
        bound_shape,
        lambda d_idx, c_idx, h_idx, w_idx: inner_bound_gradient(
            1, d_idx, c_idx, h_idx, w_idx
        ),
        name="inner_upper_gradient",
    )
    return (
        te.create_prim_func(
            [
                incoming,
                entry_lower,
                entry_upper,
                entry_alpha,
                entry_map,
                entry_coordinates,
                main_weight,
                main_bias,
                inner_lower,
                inner_upper,
                inner_alpha,
                inner_map,
                inner_coordinates,
                inner_weight,
                inner_bias,
                output_a_gradient,
                output_bias_gradient,
                incoming_gradient,
                entry_lower_gradient,
                entry_upper_gradient,
                entry_alpha_gradient,
                inner_lower_gradient,
                inner_upper_gradient,
                inner_alpha_gradient,
            ]
        )
        .with_attr("global_symbol", template.backward_symbol)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "root-crown-residual-backward/v1")
    )


def _workspace_inventory(module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    import tvm  # type: ignore[import-not-found]

    inventory: set[tuple[str, tuple[int, ...]]] = set()

    def visit(node: Any) -> None:
        if isinstance(node, tvm.tir.Block):
            for buffer in node.alloc_buffers:
                inventory.add(
                    (str(buffer.name), tuple(int(value) for value in buffer.shape))
                )

    for function in module.functions.values():
        tvm.tir.stmt_functor.post_order_visit(function.body, visit)
    return tuple(sorted(inventory))


def _schedule_function(
    tvm,
    symbol: str,
    function,
    blocks,
    thread_extent: int,
    *,
    parallel_reduction_blocks: tuple[str, ...] = (),
):
    schedule = tvm.tir.Schedule(tvm.IRModule({symbol: function}))
    parallel = set(parallel_reduction_blocks)
    if len(parallel) != len(parallel_reduction_blocks):
        raise ValueError("root CROWN parallel reduction inventory differs")
    for block_name in parallel_reduction_blocks:
        block = schedule.get_block(block_name, func_name=symbol)
        loops = schedule.get_loops(block)
        if len(loops) != 5:
            raise ValueError("root CROWN bias reduction loop nest differs")
        reduction = schedule.fuse(*loops[2:])
        _reduction_outer, reduction_lane = schedule.split(
            reduction, factors=[None, thread_extent]
        )
        partial = schedule.rfactor(reduction_lane, factor_axis=2)
        partial_loops = schedule.get_loops(partial)
        if len(partial_loops) != 4:
            raise ValueError("root CROWN partial reduction loop nest differs")
        schedule.reorder(
            partial_loops[0],
            partial_loops[1],
            partial_loops[3],
            partial_loops[2],
        )
        partial_spatial = schedule.fuse(partial_loops[0], partial_loops[1])
        schedule.bind(partial_spatial, "blockIdx.x")
        schedule.bind(partial_loops[3], "threadIdx.x")
        final_loops = schedule.get_loops(block)
        if len(final_loops) != 3:
            raise ValueError("root CROWN final reduction loop nest differs")
        final_spatial = schedule.fuse(*final_loops[:-1])
        schedule.bind(final_spatial, "blockIdx.x")
    for block_name, spatial_count in blocks:
        if block_name in parallel:
            continue
        block = schedule.get_block(block_name, func_name=symbol)
        loops = schedule.get_loops(block)
        fused = schedule.fuse(*loops[:spatial_count])
        outer, inner = schedule.split(fused, factors=[None, thread_extent])
        schedule.bind(outer, "blockIdx.x")
        schedule.bind(inner, "threadIdx.x")
    return schedule.mod[symbol]


def build_root_crown_residual_modules_v1(template: RootCrownResidualTemplateV1):
    """Build deterministic unscheduled and first correctness CUDA schedules."""

    template.validate()
    import tvm  # type: ignore[import-not-found]

    forward = _forward_primfunc(template)
    backward = _backward_primfunc(template)
    unscheduled = tvm.IRModule(
        {template.forward_symbol: forward, template.backward_symbol: backward}
    )
    forward_blocks = (
        ("entry_relu_a", 5),
        ("main_conv_a", 5),
        ("inner_relu_a", 5),
        ("residual_conv_a", 5),
        ("output_lower_a", 5),
        ("entry_bias_delta", 2),
        ("inner_bias_delta", 2),
        ("output_bias", 2),
    )
    backward_blocks = (
        ("backward_entry_a", 5),
        ("backward_main_a", 5),
        ("inner_conv_adjoint", 5),
        ("inner_transformed_adjoint", 5),
        ("main_conv_adjoint", 5),
        ("entry_conv_adjoint", 5),
        ("entry_transformed_adjoint", 5),
        ("incoming_gradient", 5),
        ("entry_alpha_gradient", 4),
        ("inner_alpha_gradient", 4),
        ("entry_lower_gradient", 4),
        ("entry_upper_gradient", 4),
        ("inner_lower_gradient", 4),
        ("inner_upper_gradient", 4),
    )
    scheduled = tvm.IRModule(
        {
            template.forward_symbol: _schedule_function(
                tvm,
                template.forward_symbol,
                forward,
                forward_blocks,
                template.thread_extent,
                parallel_reduction_blocks=(
                    "entry_bias_delta",
                    "inner_bias_delta",
                ),
            ),
            template.backward_symbol: _schedule_function(
                tvm,
                template.backward_symbol,
                backward,
                backward_blocks,
                template.thread_extent,
            ),
        }
    )
    return unscheduled, scheduled, _workspace_inventory(scheduled)


def compile_root_crown_residual_tir_v1(
    template: RootCrownResidualTemplateV1,
) -> CompiledRootCrownResidualTIRV1:
    """Compile the residual forward/full-VJP correctness schedule."""

    import tvm  # type: ignore[import-not-found]

    unscheduled, scheduled, inventory = build_root_crown_residual_modules_v1(template)
    executable = tvm.compile(
        scheduled, target=f"{template.target} -arch={template.compute_capability}"
    )
    sources = tuple(module.inspect_source() for module in executable.mod.imports)
    if not sources:
        raise RuntimeError("root CROWN residual compile produced no CUDA source")
    return CompiledRootCrownResidualTIRV1(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        unscheduled_tir_hash=_canonical_hash(tvm.ir.save_json(unscheduled)),
        scheduled_tir_hash=_canonical_hash(tvm.ir.save_json(scheduled)),
        device_source_hash=_canonical_hash("\n".join(sources)),
        workspace_inventory=inventory,
        forward_parallel_reduction_count=2,
        forward_bias_reduction_lanes=template.thread_extent,
        tvm_version=str(tvm.__version__),
    )


__all__ = [
    "CompiledRootCrownResidualTIRV1",
    "RootCrownResidualTemplateV1",
    "build_root_crown_residual_modules_v1",
    "compile_root_crown_residual_tir_v1",
]
