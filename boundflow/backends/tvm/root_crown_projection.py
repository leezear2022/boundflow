"""TVM/TIR for a root CROWN stride-2 projection residual and full VJP."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-locals
# pylint: disable=missing-function-docstring,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,too-many-statements
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)
from boundflow.backends.tvm.root_crown_residual import (
    _alpha_value,
    _canonical_hash,
    _relu_bound_derivative,
    _relu_terms,
    _schedule_function,
    _workspace_inventory,
)

ROOT_PROJECTION_FORWARD_SYMBOL = "boundflow_root_crown_projection_forward_v1"
ROOT_PROJECTION_BACKWARD_SYMBOL = "boundflow_root_crown_projection_backward_v1"


@dataclass(frozen=True)
class RootCrownProjectionTemplateV1:
    """Static geometry for one downsampling projection residual block."""

    spec_count: int
    domain_count: int
    output_channels: int
    output_height: int
    output_width: int
    input_channels: int
    input_height: int
    input_width: int
    entry_alpha_coordinates: tuple[tuple[int, int, int], ...]
    inner_alpha_coordinates: tuple[tuple[int, int, int], ...]
    compute_capability: str
    thread_extent: int = 128
    stride: tuple[int, int] = (2, 2)
    main_kernel: tuple[int, int] = (3, 3)
    main_padding: tuple[int, int] = (1, 1)
    skip_kernel: tuple[int, int] = (1, 1)
    skip_padding: tuple[int, int] = (0, 0)
    target: str = "cuda"
    forward_symbol: str = ROOT_PROJECTION_FORWARD_SYMBOL
    backward_symbol: str = ROOT_PROJECTION_BACKWARD_SYMBOL

    @property
    def incoming_shape(self) -> tuple[int, ...]:
        return (
            self.spec_count,
            self.domain_count,
            self.output_channels,
            self.output_height,
            self.output_width,
        )

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (
            self.spec_count,
            self.domain_count,
            self.input_channels,
            self.input_height,
            self.input_width,
        )

    @property
    def bound_shape(self) -> tuple[int, ...]:
        return (
            self.domain_count,
            self.output_channels,
            self.output_height,
            self.output_width,
        )

    @property
    def outer_weight_shape(self) -> tuple[int, ...]:
        return (
            self.output_channels,
            self.output_channels,
            *self.main_kernel,
        )

    @property
    def inner_weight_shape(self) -> tuple[int, ...]:
        return (
            self.output_channels,
            self.input_channels,
            *self.main_kernel,
        )

    @property
    def skip_weight_shape(self) -> tuple[int, ...]:
        return (
            self.output_channels,
            self.input_channels,
            *self.skip_kernel,
        )

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
            min(
                self.spec_count,
                self.domain_count,
                self.output_channels,
                self.output_height,
                self.output_width,
                self.input_channels,
                self.input_height,
                self.input_width,
            )
            < 1
            or self.input_height != 2 * self.output_height
            or self.input_width != 2 * self.output_width
            or any(not values for values in coordinates)
            or any(len(set(values)) != len(values) for values in coordinates)
            or any(
                not (
                    0 <= channel < self.output_channels
                    and 0 <= height < self.output_height
                    and 0 <= width < self.output_width
                )
                for values in coordinates
                for channel, height, width in values
            )
            or not self.compute_capability.startswith("sm_")
            or self.thread_extent not in {32, 64, 128, 256, 512, 1024}
            or self.stride != (2, 2)
            or self.main_kernel != (3, 3)
            or self.main_padding != (1, 1)
            or self.skip_kernel != (1, 1)
            or self.skip_padding != (0, 0)
            or self.target != "cuda"
            or self.forward_symbol != ROOT_PROJECTION_FORWARD_SYMBOL
            or self.backward_symbol != ROOT_PROJECTION_BACKWARD_SYMBOL
        ):
            raise ValueError("root CROWN projection template differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": "boundflow.root-crown-projection-template/v1",
            "spec_count": self.spec_count,
            "domain_count": self.domain_count,
            "output_channels": self.output_channels,
            "output_height": self.output_height,
            "output_width": self.output_width,
            "input_channels": self.input_channels,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "entry_alpha_coordinates": [
                list(value) for value in self.entry_alpha_coordinates
            ],
            "inner_alpha_coordinates": [
                list(value) for value in self.inner_alpha_coordinates
            ],
            "compute_capability": self.compute_capability,
            "thread_extent": self.thread_extent,
            "stride": list(self.stride),
            "main_kernel": list(self.main_kernel),
            "main_padding": list(self.main_padding),
            "skip_kernel": list(self.skip_kernel),
            "skip_padding": list(self.skip_padding),
            "target": self.target,
            "forward_symbol": self.forward_symbol,
            "backward_symbol": self.backward_symbol,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class CompiledRootCrownProjectionTIRV1:
    """Compiled projection forward/backward module and compiler identities."""

    executable: DifferentiableLowerTIRExecutable
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    workspace_inventory: tuple[tuple[str, tuple[int, ...]], ...]
    forward_parallel_reduction_count: int
    forward_bias_reduction_lanes: int
    tvm_version: str


def _placeholders(template: RootCrownProjectionTemplateV1):
    import tvm  # type: ignore[import-not-found]
    from tvm import te

    spec = template.spec_count
    domain = template.domain_count
    output_channels = template.output_channels
    incoming = te.placeholder(
        template.incoming_shape, "float32", name="incoming_lower_a"
    )
    entry_lower = te.placeholder(template.bound_shape, "float32", name="entry_lower")
    entry_upper = te.placeholder(template.bound_shape, "float32", name="entry_upper")
    entry_alpha = te.placeholder(
        (2, spec, domain, template.entry_alpha_count),
        "float32",
        name="entry_raw_alpha",
    )
    entry_map = te.placeholder(
        template.bound_shape[1:], "int32", name="entry_alpha_map"
    )
    outer_weight = te.placeholder(
        template.outer_weight_shape, "float32", name="outer_weight"
    )
    outer_bias = te.placeholder((output_channels,), "float32", name="outer_bias")
    inner_lower = te.placeholder(template.bound_shape, "float32", name="inner_lower")
    inner_upper = te.placeholder(template.bound_shape, "float32", name="inner_upper")
    inner_alpha = te.placeholder(
        (2, spec, domain, template.inner_alpha_count),
        "float32",
        name="inner_raw_alpha",
    )
    inner_map = te.placeholder(
        template.bound_shape[1:], "int32", name="inner_alpha_map"
    )
    inner_weight = te.placeholder(
        template.inner_weight_shape, "float32", name="inner_weight"
    )
    inner_bias = te.placeholder((output_channels,), "float32", name="inner_bias")
    skip_weight = te.placeholder(
        template.skip_weight_shape, "float32", name="skip_weight"
    )
    skip_bias = te.placeholder((output_channels,), "float32", name="skip_bias")
    return (
        tvm,
        te,
        incoming,
        entry_lower,
        entry_upper,
        entry_alpha,
        entry_map,
        outer_weight,
        outer_bias,
        inner_lower,
        inner_upper,
        inner_alpha,
        inner_map,
        inner_weight,
        inner_bias,
        skip_weight,
        skip_bias,
    )


def _forward_primfunc(template: RootCrownProjectionTemplateV1):
    (
        tvm,
        te,
        incoming,
        entry_lower,
        entry_upper,
        entry_alpha,
        entry_map,
        outer_weight,
        outer_bias,
        inner_lower,
        inner_upper,
        inner_alpha,
        inner_map,
        inner_weight,
        inner_bias,
        skip_weight,
        skip_bias,
    ) = _placeholders(template)
    spec = template.spec_count
    domain = template.domain_count
    co = template.output_channels
    oh = template.output_height
    ow = template.output_width
    zero = tvm.tir.const(0.0, "float32")

    def entry_element(s_idx, d_idx, c_idx, h_idx, w_idx):
        alpha = _alpha_value(
            tvm, entry_alpha, entry_map, s_idx, d_idx, c_idx, h_idx, w_idx
        )
        slope, _ = _relu_terms(
            tvm,
            incoming[s_idx, d_idx, c_idx, h_idx, w_idx],
            entry_lower[d_idx, c_idx, h_idx, w_idx],
            entry_upper[d_idx, c_idx, h_idx, w_idx],
            alpha,
        )
        return incoming[s_idx, d_idx, c_idx, h_idx, w_idx] * slope

    entry_a = te.compute(template.incoming_shape, entry_element, name="entry_relu_a")
    outer_rc = te.reduce_axis((0, co), "outer_rc")
    outer_rh = te.reduce_axis((0, 3), "outer_rh")
    outer_rw = te.reduce_axis((0, 3), "outer_rw")

    def outer_element(s_idx, d_idx, ci_idx, ih_idx, iw_idx):
        source_h = ih_idx + 1 - outer_rh
        source_w = iw_idx + 1 - outer_rw
        valid = tvm.tir.all(0 <= source_h, source_h < oh, 0 <= source_w, source_w < ow)
        return te.sum(
            tvm.tir.if_then_else(
                valid,
                entry_a[s_idx, d_idx, outer_rc, source_h, source_w]
                * outer_weight[outer_rc, ci_idx, outer_rh, outer_rw],
                zero,
            ),
            axis=(outer_rc, outer_rh, outer_rw),
        )

    outer_a = te.compute(template.incoming_shape, outer_element, name="outer_conv_a")

    def inner_element(s_idx, d_idx, c_idx, h_idx, w_idx):
        alpha = _alpha_value(
            tvm, inner_alpha, inner_map, s_idx, d_idx, c_idx, h_idx, w_idx
        )
        slope, _ = _relu_terms(
            tvm,
            outer_a[s_idx, d_idx, c_idx, h_idx, w_idx],
            inner_lower[d_idx, c_idx, h_idx, w_idx],
            inner_upper[d_idx, c_idx, h_idx, w_idx],
            alpha,
        )
        return outer_a[s_idx, d_idx, c_idx, h_idx, w_idx] * slope

    inner_a = te.compute(template.incoming_shape, inner_element, name="inner_relu_a")
    main_rc = te.reduce_axis((0, co), "main_rc")
    main_rh = te.reduce_axis((0, 3), "main_rh")
    main_rw = te.reduce_axis((0, 3), "main_rw")

    def main_output_element(s_idx, d_idx, ci_idx, ih_idx, iw_idx):
        numerator_h = ih_idx + 1 - main_rh
        numerator_w = iw_idx + 1 - main_rw
        source_h = tvm.tir.floordiv(numerator_h, 2)
        source_w = tvm.tir.floordiv(numerator_w, 2)
        valid = tvm.tir.all(
            numerator_h >= 0,
            numerator_w >= 0,
            tvm.tir.floormod(numerator_h, 2) == 0,
            tvm.tir.floormod(numerator_w, 2) == 0,
            source_h < oh,
            source_w < ow,
        )
        return te.sum(
            tvm.tir.if_then_else(
                valid,
                inner_a[s_idx, d_idx, main_rc, source_h, source_w]
                * inner_weight[main_rc, ci_idx, main_rh, main_rw],
                zero,
            ),
            axis=(main_rc, main_rh, main_rw),
        )

    main_output = te.compute(
        template.output_shape, main_output_element, name="main_output_a"
    )
    skip_rc = te.reduce_axis((0, co), "skip_rc")

    def skip_output_element(s_idx, d_idx, ci_idx, ih_idx, iw_idx):
        skip_valid = tvm.tir.all(
            tvm.tir.floormod(ih_idx, 2) == 0,
            tvm.tir.floormod(iw_idx, 2) == 0,
        )
        return te.sum(
            tvm.tir.if_then_else(
                skip_valid,
                entry_a[
                    s_idx,
                    d_idx,
                    skip_rc,
                    tvm.tir.floordiv(ih_idx, 2),
                    tvm.tir.floordiv(iw_idx, 2),
                ]
                * skip_weight[skip_rc, ci_idx, 0, 0],
                zero,
            ),
            axis=skip_rc,
        )

    skip_output = te.compute(
        template.output_shape, skip_output_element, name="skip_output_a"
    )

    output_a = te.compute(
        template.output_shape,
        lambda s_idx, d_idx, ci_idx, ih_idx, iw_idx: main_output[
            s_idx, d_idx, ci_idx, ih_idx, iw_idx
        ]
        + skip_output[s_idx, d_idx, ci_idx, ih_idx, iw_idx],
        name="output_lower_a",
    )
    entry_bc = te.reduce_axis((0, co), "entry_bc")
    entry_bh = te.reduce_axis((0, oh), "entry_bh")
    entry_bw = te.reduce_axis((0, ow), "entry_bw")

    def entry_bias_element(s_idx, d_idx):
        alpha = _alpha_value(
            tvm,
            entry_alpha,
            entry_map,
            s_idx,
            d_idx,
            entry_bc,
            entry_bh,
            entry_bw,
        )
        _, intercept = _relu_terms(
            tvm,
            incoming[s_idx, d_idx, entry_bc, entry_bh, entry_bw],
            entry_lower[d_idx, entry_bc, entry_bh, entry_bw],
            entry_upper[d_idx, entry_bc, entry_bh, entry_bw],
            alpha,
        )
        return te.sum(
            incoming[s_idx, d_idx, entry_bc, entry_bh, entry_bw] * intercept
            + entry_a[s_idx, d_idx, entry_bc, entry_bh, entry_bw]
            * (outer_bias[entry_bc] + skip_bias[entry_bc]),
            axis=(entry_bc, entry_bh, entry_bw),
        )

    entry_bias_delta = te.compute(
        (spec, domain), entry_bias_element, name="entry_bias_delta"
    )
    inner_bc = te.reduce_axis((0, co), "inner_bc")
    inner_bh = te.reduce_axis((0, oh), "inner_bh")
    inner_bw = te.reduce_axis((0, ow), "inner_bw")

    def inner_bias_element(s_idx, d_idx):
        alpha = _alpha_value(
            tvm,
            inner_alpha,
            inner_map,
            s_idx,
            d_idx,
            inner_bc,
            inner_bh,
            inner_bw,
        )
        _, intercept = _relu_terms(
            tvm,
            outer_a[s_idx, d_idx, inner_bc, inner_bh, inner_bw],
            inner_lower[d_idx, inner_bc, inner_bh, inner_bw],
            inner_upper[d_idx, inner_bc, inner_bh, inner_bw],
            alpha,
        )
        return te.sum(
            outer_a[s_idx, d_idx, inner_bc, inner_bh, inner_bw] * intercept
            + inner_a[s_idx, d_idx, inner_bc, inner_bh, inner_bw]
            * inner_bias[inner_bc],
            axis=(inner_bc, inner_bh, inner_bw),
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
                outer_weight,
                outer_bias,
                inner_lower,
                inner_upper,
                inner_alpha,
                inner_map,
                inner_weight,
                inner_bias,
                skip_weight,
                skip_bias,
                outer_a,
                output_a,
                output_bias,
            ]
        )
        .with_attr("global_symbol", template.forward_symbol)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "root-crown-projection-forward/v1")
    )


def _backward_primfunc(template: RootCrownProjectionTemplateV1):
    (
        tvm,
        te,
        incoming,
        entry_lower,
        entry_upper,
        entry_alpha,
        entry_map,
        outer_weight,
        outer_bias,
        inner_lower,
        inner_upper,
        inner_alpha,
        inner_map,
        inner_weight,
        inner_bias,
        skip_weight,
        skip_bias,
    ) = _placeholders(template)
    spec = template.spec_count
    domain = template.domain_count
    co = template.output_channels
    oh = template.output_height
    ow = template.output_width
    ci = template.input_channels
    entry_coordinates = te.placeholder(
        (3, template.entry_alpha_count), "int32", name="entry_alpha_coordinates"
    )
    inner_coordinates = te.placeholder(
        (3, template.inner_alpha_count), "int32", name="inner_alpha_coordinates"
    )
    output_a_gradient = te.placeholder(
        template.output_shape, "float32", name="output_a_gradient"
    )
    output_bias_gradient = te.placeholder(
        (spec, domain), "float32", name="output_bias_gradient"
    )
    zero = tvm.tir.const(0.0, "float32")
    one = tvm.tir.const(1.0, "float32")

    def entry_element(s_idx, d_idx, c_idx, h_idx, w_idx):
        alpha = _alpha_value(
            tvm, entry_alpha, entry_map, s_idx, d_idx, c_idx, h_idx, w_idx
        )
        slope, _ = _relu_terms(
            tvm,
            incoming[s_idx, d_idx, c_idx, h_idx, w_idx],
            entry_lower[d_idx, c_idx, h_idx, w_idx],
            entry_upper[d_idx, c_idx, h_idx, w_idx],
            alpha,
        )
        return incoming[s_idx, d_idx, c_idx, h_idx, w_idx] * slope

    entry_a = te.compute(
        template.incoming_shape, entry_element, name="backward_entry_a"
    )
    outer_rc = te.reduce_axis((0, co), "backward_outer_rc")
    outer_rh = te.reduce_axis((0, 3), "backward_outer_rh")
    outer_rw = te.reduce_axis((0, 3), "backward_outer_rw")

    def outer_element(s_idx, d_idx, ci_idx, ih_idx, iw_idx):
        source_h = ih_idx + 1 - outer_rh
        source_w = iw_idx + 1 - outer_rw
        valid = tvm.tir.all(0 <= source_h, source_h < oh, 0 <= source_w, source_w < ow)
        return te.sum(
            tvm.tir.if_then_else(
                valid,
                entry_a[s_idx, d_idx, outer_rc, source_h, source_w]
                * outer_weight[outer_rc, ci_idx, outer_rh, outer_rw],
                zero,
            ),
            axis=(outer_rc, outer_rh, outer_rw),
        )

    outer_a = te.compute(
        template.incoming_shape, outer_element, name="backward_outer_a"
    )
    main_adj_ci = te.reduce_axis((0, ci), "main_adj_ci")
    main_adj_h = te.reduce_axis((0, 3), "main_adj_h")
    main_adj_w = te.reduce_axis((0, 3), "main_adj_w")

    def main_conv_adjoint_element(s_idx, d_idx, co_idx, oh_idx, ow_idx):
        input_h = oh_idx * 2 - 1 + main_adj_h
        input_w = ow_idx * 2 - 1 + main_adj_w
        valid = tvm.tir.all(
            0 <= input_h,
            input_h < template.input_height,
            0 <= input_w,
            input_w < template.input_width,
        )
        return te.sum(
            tvm.tir.if_then_else(
                valid,
                output_a_gradient[s_idx, d_idx, main_adj_ci, input_h, input_w]
                * inner_weight[co_idx, main_adj_ci, main_adj_h, main_adj_w],
                zero,
            ),
            axis=(main_adj_ci, main_adj_h, main_adj_w),
        )

    main_conv_adjoint = te.compute(
        template.incoming_shape, main_conv_adjoint_element, name="main_conv_adjoint"
    )
    inner_adjoint = te.compute(
        template.incoming_shape,
        lambda s_idx, d_idx, c_idx, h_idx, w_idx: main_conv_adjoint[
            s_idx, d_idx, c_idx, h_idx, w_idx
        ]
        + output_bias_gradient[s_idx, d_idx] * inner_bias[c_idx],
        name="inner_transformed_adjoint",
    )

    def outer_adjoint_element(s_idx, d_idx, c_idx, h_idx, w_idx):
        alpha = _alpha_value(
            tvm, inner_alpha, inner_map, s_idx, d_idx, c_idx, h_idx, w_idx
        )
        slope, intercept = _relu_terms(
            tvm,
            outer_a[s_idx, d_idx, c_idx, h_idx, w_idx],
            inner_lower[d_idx, c_idx, h_idx, w_idx],
            inner_upper[d_idx, c_idx, h_idx, w_idx],
            alpha,
        )
        return (
            inner_adjoint[s_idx, d_idx, c_idx, h_idx, w_idx] * slope
            + output_bias_gradient[s_idx, d_idx] * intercept
        )

    outer_adjoint = te.compute(
        template.incoming_shape, outer_adjoint_element, name="outer_conv_adjoint"
    )
    entry_outer_ci = te.reduce_axis((0, co), "entry_outer_ci")
    entry_outer_h = te.reduce_axis((0, 3), "entry_outer_h")
    entry_outer_w = te.reduce_axis((0, 3), "entry_outer_w")

    def entry_outer_adjoint_element(s_idx, d_idx, co_idx, oh_idx, ow_idx):
        input_h = oh_idx - 1 + entry_outer_h
        input_w = ow_idx - 1 + entry_outer_w
        valid = tvm.tir.all(0 <= input_h, input_h < oh, 0 <= input_w, input_w < ow)
        return te.sum(
            tvm.tir.if_then_else(
                valid,
                outer_adjoint[s_idx, d_idx, entry_outer_ci, input_h, input_w]
                * outer_weight[co_idx, entry_outer_ci, entry_outer_h, entry_outer_w],
                zero,
            ),
            axis=(entry_outer_ci, entry_outer_h, entry_outer_w),
        )

    entry_outer_adjoint = te.compute(
        template.incoming_shape,
        entry_outer_adjoint_element,
        name="entry_outer_adjoint",
    )
    skip_adj_ci = te.reduce_axis((0, ci), "skip_adj_ci")

    def entry_skip_adjoint_element(s_idx, d_idx, co_idx, oh_idx, ow_idx):
        return te.sum(
            output_a_gradient[s_idx, d_idx, skip_adj_ci, oh_idx * 2, ow_idx * 2]
            * skip_weight[co_idx, skip_adj_ci, 0, 0],
            axis=skip_adj_ci,
        )

    entry_skip_adjoint = te.compute(
        template.incoming_shape,
        entry_skip_adjoint_element,
        name="entry_skip_adjoint",
    )
    entry_adjoint = te.compute(
        template.incoming_shape,
        lambda s_idx, d_idx, co_idx, oh_idx, ow_idx: (
            entry_outer_adjoint[s_idx, d_idx, co_idx, oh_idx, ow_idx]
            + entry_skip_adjoint[s_idx, d_idx, co_idx, oh_idx, ow_idx]
            + output_bias_gradient[s_idx, d_idx]
            * (outer_bias[co_idx] + skip_bias[co_idx])
        ),
        name="entry_transformed_adjoint",
    )

    def incoming_gradient_element(s_idx, d_idx, c_idx, h_idx, w_idx):
        alpha = _alpha_value(
            tvm, entry_alpha, entry_map, s_idx, d_idx, c_idx, h_idx, w_idx
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
        template.incoming_shape, incoming_gradient_element, name="incoming_gradient"
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
            outer_a,
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
                outer_a[inner_reduce_spec, d_idx, c_idx, h_idx, w_idx],
                inner_lower[d_idx, c_idx, h_idx, w_idx],
                inner_upper[d_idx, c_idx, h_idx, w_idx],
                inner_adjoint[inner_reduce_spec, d_idx, c_idx, h_idx, w_idx],
                output_bias_gradient[inner_reduce_spec, d_idx],
                kind,
            ),
            axis=inner_reduce_spec,
        )

    entry_lower_gradient = te.compute(
        template.bound_shape,
        lambda d_idx, c_idx, h_idx, w_idx: entry_bound_gradient(
            0, d_idx, c_idx, h_idx, w_idx
        ),
        name="entry_lower_gradient",
    )
    entry_upper_gradient = te.compute(
        template.bound_shape,
        lambda d_idx, c_idx, h_idx, w_idx: entry_bound_gradient(
            1, d_idx, c_idx, h_idx, w_idx
        ),
        name="entry_upper_gradient",
    )
    inner_lower_gradient = te.compute(
        template.bound_shape,
        lambda d_idx, c_idx, h_idx, w_idx: inner_bound_gradient(
            0, d_idx, c_idx, h_idx, w_idx
        ),
        name="inner_lower_gradient",
    )
    inner_upper_gradient = te.compute(
        template.bound_shape,
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
                outer_weight,
                outer_bias,
                inner_lower,
                inner_upper,
                inner_alpha,
                inner_map,
                inner_coordinates,
                inner_weight,
                inner_bias,
                skip_weight,
                skip_bias,
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
        .with_attr("boundflow.schema_version", "root-crown-projection-backward/v1")
    )


def build_root_crown_projection_modules_v1(template: RootCrownProjectionTemplateV1):
    """Build deterministic unscheduled and correctness-first CUDA schedules."""

    template.validate()
    import tvm  # type: ignore[import-not-found]

    forward = _forward_primfunc(template)
    backward = _backward_primfunc(template)
    unscheduled = tvm.IRModule(
        {template.forward_symbol: forward, template.backward_symbol: backward}
    )
    forward_blocks = (
        ("entry_relu_a", 5),
        ("outer_conv_a", 5),
        ("inner_relu_a", 5),
        ("main_output_a", 5),
        ("skip_output_a", 5),
        ("output_lower_a", 5),
        ("entry_bias_delta", 2),
        ("inner_bias_delta", 2),
        ("output_bias", 2),
    )
    backward_blocks = (
        ("backward_entry_a", 5),
        ("backward_outer_a", 5),
        ("main_conv_adjoint", 5),
        ("inner_transformed_adjoint", 5),
        ("outer_conv_adjoint", 5),
        ("entry_outer_adjoint", 5),
        ("entry_skip_adjoint", 5),
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


def compile_root_crown_projection_tir_v1(
    template: RootCrownProjectionTemplateV1,
) -> CompiledRootCrownProjectionTIRV1:
    """Compile the projection forward/full-VJP correctness schedule."""

    import tvm  # type: ignore[import-not-found]

    unscheduled, scheduled, inventory = build_root_crown_projection_modules_v1(template)
    executable = tvm.compile(
        scheduled, target=f"{template.target} -arch={template.compute_capability}"
    )
    sources = tuple(module.inspect_source() for module in executable.mod.imports)
    if not sources:
        raise RuntimeError("root CROWN projection compile produced no CUDA source")
    return CompiledRootCrownProjectionTIRV1(
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
    "CompiledRootCrownProjectionTIRV1",
    "RootCrownProjectionTemplateV1",
    "build_root_crown_projection_modules_v1",
    "compile_root_crown_projection_tir_v1",
]
