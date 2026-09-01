"""TVM/TIR for root CROWN input Conv fused with L-infinity concretization."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-locals
# pylint: disable=missing-function-docstring,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,duplicate-code
# pylint: disable=too-many-nested-blocks,chained-comparison,too-many-statements

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)
from boundflow.backends.tvm.root_crown_residual import (
    _alpha_value,
    _canonical_hash,
    _relu_terms,
    _workspace_inventory,
)

ROOT_INPUT_DOMAIN_FORWARD_SYMBOL = "boundflow_root_input_domain_forward_v1"
ROOT_INPUT_DOMAIN_BACKWARD_SYMBOL = "boundflow_root_input_domain_backward_v1"


@dataclass(frozen=True)
class RootCrownInputDomainTemplateV1:
    """Static geometry and sparse-alpha mapping for the root input transaction."""

    spec_count: int
    domain_count: int
    output_channels: int
    output_height: int
    output_width: int
    input_channels: int
    input_height: int
    input_width: int
    alpha_coordinates: tuple[tuple[int, int, int], ...]
    compute_capability: str
    thread_extent: int = 128
    stride: tuple[int, int] = (2, 2)
    kernel: tuple[int, int] = (3, 3)
    padding: tuple[int, int] = (1, 1)
    target: str = "cuda"
    forward_symbol: str = ROOT_INPUT_DOMAIN_FORWARD_SYMBOL
    backward_symbol: str = ROOT_INPUT_DOMAIN_BACKWARD_SYMBOL

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
    def bound_shape(self) -> tuple[int, ...]:
        return (
            self.domain_count,
            self.output_channels,
            self.output_height,
            self.output_width,
        )

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (
            self.domain_count,
            self.input_channels,
            self.input_height,
            self.input_width,
        )

    @property
    def coefficient_shape(self) -> tuple[int, ...]:
        return (
            self.spec_count,
            self.domain_count,
            self.input_channels,
            self.input_height,
            self.input_width,
        )

    @property
    def weight_shape(self) -> tuple[int, ...]:
        return (self.output_channels, self.input_channels, *self.kernel)

    @property
    def alpha_count(self) -> int:
        return len(self.alpha_coordinates)

    def validate(self) -> None:
        geometry = (
            self.spec_count,
            self.domain_count,
            self.output_channels,
            self.output_height,
            self.output_width,
            self.input_channels,
            self.input_height,
            self.input_width,
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
            or len(self.alpha_coordinates) != 164
            or len(set(self.alpha_coordinates)) != len(self.alpha_coordinates)
            or any(
                not (
                    0 <= channel < self.output_channels
                    and 0 <= height < self.output_height
                    and 0 <= width < self.output_width
                )
                for channel, height, width in self.alpha_coordinates
            )
            or not self.compute_capability.startswith("sm_")
            or self.thread_extent not in {32, 64, 128, 256, 512, 1024}
            or self.stride != (2, 2)
            or self.kernel != (3, 3)
            or self.padding != (1, 1)
            or self.target != "cuda"
            or self.forward_symbol != ROOT_INPUT_DOMAIN_FORWARD_SYMBOL
            or self.backward_symbol != ROOT_INPUT_DOMAIN_BACKWARD_SYMBOL
            or geometry != (3, 1, 8, 16, 16, 3, 32, 32)
        ):
            raise ValueError("root CROWN input-domain template differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": "boundflow.root-crown-input-domain-template/v1",
            "spec_count": self.spec_count,
            "domain_count": self.domain_count,
            "output_channels": self.output_channels,
            "output_height": self.output_height,
            "output_width": self.output_width,
            "input_channels": self.input_channels,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "alpha_coordinates": [list(value) for value in self.alpha_coordinates],
            "compute_capability": self.compute_capability,
            "thread_extent": self.thread_extent,
            "stride": list(self.stride),
            "kernel": list(self.kernel),
            "padding": list(self.padding),
            "target": self.target,
            "forward_symbol": self.forward_symbol,
            "backward_symbol": self.backward_symbol,
            "dense_input_coefficient_externalized": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class CompiledRootCrownInputDomainTIRV1:
    """Compiled input-domain module and compiler identities."""

    executable: DifferentiableLowerTIRExecutable
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    workspace_inventory: tuple[tuple[str, tuple[int, ...]], ...]
    tvm_version: str


def _placeholders(template: RootCrownInputDomainTemplateV1):
    import tvm
    from tvm import te

    incoming = te.placeholder(
        template.incoming_shape, "float32", name="incoming_lower_a"
    )
    lower = te.placeholder(template.bound_shape, "float32", name="preactivation_lower")
    upper = te.placeholder(template.bound_shape, "float32", name="preactivation_upper")
    raw_alpha = te.placeholder(
        (2, template.spec_count, template.domain_count, template.alpha_count),
        "float32",
        name="raw_alpha",
    )
    alpha_map = te.placeholder(template.bound_shape[1:], "int32", name="alpha_map")
    weight = te.placeholder(template.weight_shape, "float32", name="weight")
    operator_bias = te.placeholder(
        (template.output_channels,), "float32", name="operator_bias"
    )
    input_center = te.placeholder(template.input_shape, "float32", name="input_center")
    input_radius = te.placeholder(template.input_shape, "float32", name="input_radius")
    return (
        tvm,
        te,
        incoming,
        lower,
        upper,
        raw_alpha,
        alpha_map,
        weight,
        operator_bias,
        input_center,
        input_radius,
    )


def _forward_primfunc(template: RootCrownInputDomainTemplateV1):
    (
        tvm,
        te,
        incoming,
        lower,
        upper,
        raw_alpha,
        alpha_map,
        weight,
        operator_bias,
        input_center,
        input_radius,
    ) = _placeholders(template)
    zero = tvm.tir.const(0.0, "float32")

    def transformed_element(s_idx, d_idx, c_idx, h_idx, w_idx):
        alpha = _alpha_value(
            tvm, raw_alpha, alpha_map, s_idx, d_idx, c_idx, h_idx, w_idx
        )
        slope, _intercept = _relu_terms(
            tvm,
            incoming[s_idx, d_idx, c_idx, h_idx, w_idx],
            lower[d_idx, c_idx, h_idx, w_idx],
            upper[d_idx, c_idx, h_idx, w_idx],
            alpha,
        )
        return incoming[s_idx, d_idx, c_idx, h_idx, w_idx] * slope

    transformed = te.compute(
        template.incoming_shape, transformed_element, name="transformed_a"
    )
    reduce_co = te.reduce_axis((0, template.output_channels), "reduce_co")
    reduce_kh = te.reduce_axis((0, 3), "reduce_kh")
    reduce_kw = te.reduce_axis((0, 3), "reduce_kw")

    def coefficient_element(s_idx, d_idx, ci_idx, ih_idx, iw_idx):
        numerator_h = ih_idx + 1 - reduce_kh
        numerator_w = iw_idx + 1 - reduce_kw
        source_h = tvm.tir.floordiv(numerator_h, 2)
        source_w = tvm.tir.floordiv(numerator_w, 2)
        valid = tvm.tir.all(
            numerator_h >= 0,
            numerator_w >= 0,
            tvm.tir.floormod(numerator_h, 2) == 0,
            tvm.tir.floormod(numerator_w, 2) == 0,
            source_h < template.output_height,
            source_w < template.output_width,
        )
        return te.sum(
            tvm.tir.if_then_else(
                valid,
                transformed[s_idx, d_idx, reduce_co, source_h, source_w]
                * weight[reduce_co, ci_idx, reduce_kh, reduce_kw],
                zero,
            ),
            axis=(reduce_co, reduce_kh, reduce_kw),
        )

    coefficient = te.compute(
        template.coefficient_shape,
        coefficient_element,
        name="input_coefficient_scratch",
    )
    reduce_ci = te.reduce_axis((0, template.input_channels), "concrete_ci")
    reduce_ih = te.reduce_axis((0, template.input_height), "concrete_ih")
    reduce_iw = te.reduce_axis((0, template.input_width), "concrete_iw")
    concrete_lower = te.compute(
        (template.domain_count, template.spec_count),
        lambda d_idx, s_idx: te.sum(
            coefficient[s_idx, d_idx, reduce_ci, reduce_ih, reduce_iw]
            * input_center[d_idx, reduce_ci, reduce_ih, reduce_iw]
            - tvm.tir.abs(coefficient[s_idx, d_idx, reduce_ci, reduce_ih, reduce_iw])
            * input_radius[d_idx, reduce_ci, reduce_ih, reduce_iw],
            axis=(reduce_ci, reduce_ih, reduce_iw),
        ),
        name="concrete_lower",
    )
    reduce_bc = te.reduce_axis((0, template.output_channels), "bias_c")
    reduce_bh = te.reduce_axis((0, template.output_height), "bias_h")
    reduce_bw = te.reduce_axis((0, template.output_width), "bias_w")

    def bias_element(s_idx, d_idx):
        alpha = _alpha_value(
            tvm,
            raw_alpha,
            alpha_map,
            s_idx,
            d_idx,
            reduce_bc,
            reduce_bh,
            reduce_bw,
        )
        _slope, intercept = _relu_terms(
            tvm,
            incoming[s_idx, d_idx, reduce_bc, reduce_bh, reduce_bw],
            lower[d_idx, reduce_bc, reduce_bh, reduce_bw],
            upper[d_idx, reduce_bc, reduce_bh, reduce_bw],
            alpha,
        )
        return te.sum(
            incoming[s_idx, d_idx, reduce_bc, reduce_bh, reduce_bw] * intercept
            + transformed[s_idx, d_idx, reduce_bc, reduce_bh, reduce_bw]
            * operator_bias[reduce_bc],
            axis=(reduce_bc, reduce_bh, reduce_bw),
        )

    output_bias = te.compute(
        (template.spec_count, template.domain_count),
        bias_element,
        name="output_bias",
    )
    return (
        te.create_prim_func(
            [
                incoming,
                lower,
                upper,
                raw_alpha,
                alpha_map,
                weight,
                operator_bias,
                input_center,
                input_radius,
                concrete_lower,
                output_bias,
            ]
        )
        .with_attr("global_symbol", template.forward_symbol)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "root-crown-input-domain-forward/v1")
    )


def _streaming_forward_primfunc():  # pylint: disable=too-many-statements
    """One-kernel forward: generate each input coefficient and reduce immediately."""

    from tvm.script import tir as T

    @T.prim_func
    def forward(
        incoming: T.Buffer((3, 1, 8, 16, 16), "float32"),
        lower: T.Buffer((1, 8, 16, 16), "float32"),
        upper: T.Buffer((1, 8, 16, 16), "float32"),
        raw_alpha: T.Buffer((2, 3, 1, 164), "float32"),
        alpha_map: T.Buffer((8, 16, 16), "int32"),
        weight: T.Buffer((8, 3, 3, 3), "float32"),
        operator_bias: T.Buffer((8,), "float32"),
        input_center: T.Buffer((1, 3, 32, 32), "float32"),
        input_radius: T.Buffer((1, 3, 32, 32), "float32"),
        concrete_lower: T.Buffer((1, 3), "float32"),
        output_bias: T.Buffer((3, 1), "float32"),
    ):
        T.func_attr(
            {
                "global_symbol": ROOT_INPUT_DOMAIN_FORWARD_SYMBOL,
                "tir.noalias": True,
                "boundflow.schema_version": "root-crown-input-domain-streaming-forward/v2",
            }
        )
        coefficient = T.alloc_buffer((1,), "float32", scope="local")
        concrete_sum = T.alloc_buffer((1,), "float32", scope="local")
        bias_sum = T.alloc_buffer((1,), "float32", scope="local")
        partial = T.alloc_buffer((2, 128), "float32", scope="shared")
        reduction = T.alloc_buffer((2,), "float32", scope="local")
        for block_x in T.thread_binding(3, thread="blockIdx.x"):
            for thread_x in T.thread_binding(128, thread="threadIdx.x"):
                concrete_sum[0] = T.float32(0)
                for flat in T.serial(thread_x, 3072, step=128):
                    input_w = flat % 32
                    input_h = flat // 32 % 32
                    input_channel = flat // 1024
                    coefficient[0] = T.float32(0)
                    for output_channel, kernel_h, kernel_w in T.grid(8, 3, 3):
                        numerator_h = input_h + 1 - kernel_h
                        numerator_w = input_w + 1 - kernel_w
                        if (
                            numerator_h >= 0
                            and numerator_w >= 0
                            and numerator_h % 2 == 0
                            and numerator_w % 2 == 0
                        ):
                            output_h = numerator_h // 2
                            output_w = numerator_w // 2
                            if output_h < 16 and output_w < 16:
                                source = incoming[
                                    block_x,
                                    0,
                                    output_channel,
                                    output_h,
                                    output_w,
                                ]
                                lower_value = lower[
                                    0, output_channel, output_h, output_w
                                ]
                                upper_value = upper[
                                    0, output_channel, output_h, output_w
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
                                lookup = alpha_map[output_channel, output_h, output_w]
                                alpha_value = T.if_then_else(
                                    lookup >= 0,
                                    raw_alpha[0, block_x, 0, T.max(lookup, 0)],
                                    T.float32(0),
                                )
                                ambiguous = lower_value < T.float32(
                                    0
                                ) and upper_value > T.float32(0)
                                lower_slope = T.if_then_else(
                                    ambiguous,
                                    T.min(
                                        T.max(alpha_value, T.float32(0)),
                                        T.float32(1),
                                    ),
                                    T.if_then_else(
                                        lower_value >= T.float32(0),
                                        T.float32(1),
                                        T.float32(0),
                                    ),
                                )
                                slope = T.if_then_else(
                                    source >= T.float32(0),
                                    lower_slope,
                                    upper_slope,
                                )
                                coefficient[0] = (
                                    coefficient[0]
                                    + source
                                    * slope
                                    * weight[
                                        output_channel,
                                        input_channel,
                                        kernel_h,
                                        kernel_w,
                                    ]
                                )
                    concrete_sum[0] = (
                        concrete_sum[0]
                        + coefficient[0]
                        * input_center[0, input_channel, input_h, input_w]
                        - T.abs(coefficient[0])
                        * input_radius[0, input_channel, input_h, input_w]
                    )
                bias_sum[0] = T.float32(0)
                for flat in T.serial(thread_x, 2048, step=128):
                    output_w = flat % 16
                    output_h = flat // 16 % 16
                    output_channel = flat // 256
                    source = incoming[block_x, 0, output_channel, output_h, output_w]
                    lower_value = lower[0, output_channel, output_h, output_w]
                    upper_value = upper[0, output_channel, output_h, output_w]
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
                    lookup = alpha_map[output_channel, output_h, output_w]
                    alpha_value = T.if_then_else(
                        lookup >= 0,
                        raw_alpha[0, block_x, 0, T.max(lookup, 0)],
                        T.float32(0),
                    )
                    ambiguous = lower_value < T.float32(0) and upper_value > T.float32(
                        0
                    )
                    lower_slope = T.if_then_else(
                        ambiguous,
                        T.min(T.max(alpha_value, T.float32(0)), T.float32(1)),
                        T.if_then_else(
                            lower_value >= T.float32(0),
                            T.float32(1),
                            T.float32(0),
                        ),
                    )
                    slope = T.if_then_else(
                        source >= T.float32(0), lower_slope, upper_slope
                    )
                    intercept = T.if_then_else(
                        source < T.float32(0) and ambiguous,
                        -lower_value * upper_slope,
                        T.float32(0),
                    )
                    bias_sum[0] = (
                        bias_sum[0]
                        + source * intercept
                        + source * slope * operator_bias[output_channel]
                    )
                partial[0, thread_x] = concrete_sum[0]
                partial[1, thread_x] = bias_sum[0]
                T.tvm_storage_sync("shared")
                if thread_x == 0:
                    reduction[0] = T.float32(0)
                    reduction[1] = T.float32(0)
                    for lane in range(128):
                        reduction[0] = reduction[0] + partial[0, lane]
                        reduction[1] = reduction[1] + partial[1, lane]
                    concrete_lower[0, block_x] = reduction[0]
                    output_bias[block_x, 0] = reduction[1]

    return forward


def _streaming_backward_primfunc():  # pylint: disable=too-many-statements
    """One-kernel VJP with coefficient recomputation and compressed-alpha output."""

    from tvm.script import tir as T

    @T.prim_func
    def backward(
        incoming: T.Buffer((3, 1, 8, 16, 16), "float32"),
        lower: T.Buffer((1, 8, 16, 16), "float32"),
        upper: T.Buffer((1, 8, 16, 16), "float32"),
        raw_alpha: T.Buffer((2, 3, 1, 164), "float32"),
        alpha_map: T.Buffer((8, 16, 16), "int32"),
        weight: T.Buffer((8, 3, 3, 3), "float32"),
        operator_bias: T.Buffer((8,), "float32"),
        input_center: T.Buffer((1, 3, 32, 32), "float32"),
        input_radius: T.Buffer((1, 3, 32, 32), "float32"),
        concrete_gradient: T.Buffer((1, 3), "float32"),
        bias_gradient: T.Buffer((3, 1), "float32"),
        incoming_gradient: T.Buffer((3, 1, 8, 16, 16), "float32"),
        alpha_gradient: T.Buffer((2, 3, 1, 164), "float32"),
    ):
        T.func_attr(
            {
                "global_symbol": ROOT_INPUT_DOMAIN_BACKWARD_SYMBOL,
                "tir.noalias": True,
                "boundflow.schema_version": "root-crown-input-domain-streaming-backward/v2",
            }
        )
        coefficient = T.alloc_buffer((1,), "float32", scope="local")
        adjoint = T.alloc_buffer((1,), "float32", scope="local")
        for block_x in T.thread_binding(48, thread="blockIdx.x"):
            for thread_x in T.thread_binding(128, thread="threadIdx.x"):
                flat = block_x * 128 + thread_x
                if flat < 6144:
                    output_w = flat % 16
                    output_h = flat // 16 % 16
                    output_channel = flat // 256 % 8
                    spec = flat // 2048
                    adjoint[0] = T.float32(0)
                    for input_channel, kernel_h, kernel_w in T.grid(3, 3, 3):
                        input_h = output_h * 2 - 1 + kernel_h
                        input_w = output_w * 2 - 1 + kernel_w
                        if (
                            0 <= input_h
                            and input_h < 32
                            and 0 <= input_w
                            and input_w < 32
                        ):
                            coefficient[0] = T.float32(0)
                            for source_channel, source_kh, source_kw in T.grid(8, 3, 3):
                                numerator_h = input_h + 1 - source_kh
                                numerator_w = input_w + 1 - source_kw
                                if (
                                    numerator_h >= 0
                                    and numerator_w >= 0
                                    and numerator_h % 2 == 0
                                    and numerator_w % 2 == 0
                                ):
                                    source_h = numerator_h // 2
                                    source_w = numerator_w // 2
                                    if source_h < 16 and source_w < 16:
                                        source = incoming[
                                            spec,
                                            0,
                                            source_channel,
                                            source_h,
                                            source_w,
                                        ]
                                        source_lower = lower[
                                            0, source_channel, source_h, source_w
                                        ]
                                        source_upper = upper[
                                            0, source_channel, source_h, source_w
                                        ]
                                        denominator = T.max(
                                            source_upper - source_lower,
                                            T.float32(1.1920928955078125e-7),
                                        )
                                        upper_slope = T.if_then_else(
                                            source_lower >= T.float32(0),
                                            T.float32(1),
                                            T.if_then_else(
                                                source_upper <= T.float32(0),
                                                T.float32(0),
                                                source_upper / denominator,
                                            ),
                                        )
                                        source_lookup = alpha_map[
                                            source_channel, source_h, source_w
                                        ]
                                        alpha_value = T.if_then_else(
                                            source_lookup >= 0,
                                            raw_alpha[
                                                0,
                                                spec,
                                                0,
                                                T.max(source_lookup, 0),
                                            ],
                                            T.float32(0),
                                        )
                                        ambiguous = source_lower < T.float32(
                                            0
                                        ) and source_upper > T.float32(0)
                                        lower_slope = T.if_then_else(
                                            ambiguous,
                                            T.min(
                                                T.max(alpha_value, T.float32(0)),
                                                T.float32(1),
                                            ),
                                            T.if_then_else(
                                                source_lower >= T.float32(0),
                                                T.float32(1),
                                                T.float32(0),
                                            ),
                                        )
                                        slope = T.if_then_else(
                                            source >= T.float32(0),
                                            lower_slope,
                                            upper_slope,
                                        )
                                        coefficient[0] = (
                                            coefficient[0]
                                            + source
                                            * slope
                                            * weight[
                                                source_channel,
                                                input_channel,
                                                source_kh,
                                                source_kw,
                                            ]
                                        )
                            coefficient_adjoint = concrete_gradient[0, spec] * (
                                input_center[0, input_channel, input_h, input_w]
                                + T.if_then_else(
                                    coefficient[0] < T.float32(0),
                                    input_radius[0, input_channel, input_h, input_w],
                                    T.if_then_else(
                                        coefficient[0] > T.float32(0),
                                        -input_radius[
                                            0, input_channel, input_h, input_w
                                        ],
                                        T.float32(0),
                                    ),
                                )
                            )
                            adjoint[0] = (
                                adjoint[0]
                                + coefficient_adjoint
                                * weight[
                                    output_channel,
                                    input_channel,
                                    kernel_h,
                                    kernel_w,
                                ]
                            )
                    adjoint[0] = (
                        adjoint[0]
                        + bias_gradient[spec, 0] * operator_bias[output_channel]
                    )
                    source = incoming[spec, 0, output_channel, output_h, output_w]
                    lower_value = lower[0, output_channel, output_h, output_w]
                    upper_value = upper[0, output_channel, output_h, output_w]
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
                    lookup = alpha_map[output_channel, output_h, output_w]
                    alpha_value = T.if_then_else(
                        lookup >= 0,
                        raw_alpha[0, spec, 0, T.max(lookup, 0)],
                        T.float32(0),
                    )
                    ambiguous = lower_value < T.float32(0) and upper_value > T.float32(
                        0
                    )
                    lower_slope = T.if_then_else(
                        ambiguous,
                        T.min(T.max(alpha_value, T.float32(0)), T.float32(1)),
                        T.if_then_else(
                            lower_value >= T.float32(0),
                            T.float32(1),
                            T.float32(0),
                        ),
                    )
                    slope = T.if_then_else(
                        source >= T.float32(0), lower_slope, upper_slope
                    )
                    intercept = T.if_then_else(
                        source < T.float32(0) and ambiguous,
                        -lower_value * upper_slope,
                        T.float32(0),
                    )
                    incoming_gradient[spec, 0, output_channel, output_h, output_w] = (
                        adjoint[0] * slope + bias_gradient[spec, 0] * intercept
                    )
                    if lookup >= 0:
                        alpha_gradient[0, spec, 0, lookup] = T.if_then_else(
                            source >= T.float32(0)
                            and ambiguous
                            and alpha_value >= T.float32(0)
                            and alpha_value <= T.float32(1),
                            adjoint[0] * source,
                            T.float32(0),
                        )
                        alpha_gradient[1, spec, 0, lookup] = T.float32(0)

    return backward


def build_root_crown_input_domain_modules_v1(
    template: RootCrownInputDomainTemplateV1,
):
    """Build deterministic correctness-first CUDA schedules."""

    template.validate()
    import tvm

    forward = _forward_primfunc(template)
    backward = _streaming_backward_primfunc()
    unscheduled = tvm.IRModule(
        {template.forward_symbol: forward, template.backward_symbol: backward}
    )
    scheduled = tvm.IRModule(
        {
            template.forward_symbol: _streaming_forward_primfunc(),
            template.backward_symbol: _streaming_backward_primfunc(),
        }
    )
    return unscheduled, scheduled, _workspace_inventory(scheduled)


def compile_root_crown_input_domain_tir_v1(
    template: RootCrownInputDomainTemplateV1,
) -> CompiledRootCrownInputDomainTIRV1:
    """Compile the fused input Conv/concretization correctness schedule."""

    import tvm

    unscheduled, scheduled, inventory = build_root_crown_input_domain_modules_v1(
        template
    )
    executable = tvm.compile(
        scheduled, target=f"{template.target} -arch={template.compute_capability}"
    )
    sources = tuple(module.inspect_source() for module in executable.mod.imports)
    if not sources:
        raise RuntimeError("root CROWN input-domain compile produced no CUDA source")
    return CompiledRootCrownInputDomainTIRV1(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        unscheduled_tir_hash=_canonical_hash(tvm.ir.save_json(unscheduled)),
        scheduled_tir_hash=_canonical_hash(tvm.ir.save_json(scheduled)),
        device_source_hash=_canonical_hash("\n".join(sources)),
        workspace_inventory=inventory,
        tvm_version=str(tvm.__version__),
    )


__all__ = [
    "CompiledRootCrownInputDomainTIRV1",
    "RootCrownInputDomainTemplateV1",
    "build_root_crown_input_domain_modules_v1",
    "compile_root_crown_input_domain_tir_v1",
]
