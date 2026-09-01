"""TVM/TIR for activation-BaB input Conv fused with L-infinity reduction.

This is a versioned production-shape specialization.  It deliberately does
not modify the root-CROWN implementation whose source identity is already
bound into historical formal artifacts.
"""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-locals
# pylint: disable=missing-function-docstring,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,duplicate-code
# pylint: disable=too-many-nested-blocks,chained-comparison,too-many-statements

from __future__ import annotations

from typing import cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)
from boundflow.backends.tvm.root_crown_input_domain import (
    _forward_primfunc,
    CompiledRootCrownInputDomainTIRV1,
    RootCrownInputDomainTemplateV1,
)
from boundflow.backends.tvm.root_crown_residual import (
    _canonical_hash,
    _workspace_inventory,
)

BAB_INPUT_DOMAIN_FORWARD_SYMBOL = "boundflow_bab_input_domain_forward_v1"
BAB_INPUT_DOMAIN_BACKWARD_SYMBOL = "boundflow_bab_input_domain_backward_v1"


class BabInputDomainTemplateV1(RootCrownInputDomainTemplateV1):
    """Static ABI for the captured spec=1/domain=6 input transaction."""

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
            geometry != (1, 6, 8, 16, 16, 3, 32, 32)
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
            or self.thread_extent != 128
            or self.stride != (2, 2)
            or self.kernel != (3, 3)
            or self.padding != (1, 1)
            or self.target != "cuda"
            or self.forward_symbol != BAB_INPUT_DOMAIN_FORWARD_SYMBOL
            or self.backward_symbol != BAB_INPUT_DOMAIN_BACKWARD_SYMBOL
        ):
            raise ValueError("activation-BaB input-domain template differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": "boundflow.activation-bab-input-domain-template/v1",
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
            "coefficient_lifetime": "register-local-before-concretization",
        }


def _streaming_forward_primfunc():  # pylint: disable=too-many-statements
    """Generate one coefficient locally and immediately consume it."""

    from tvm.script import tir as T

    @T.prim_func
    def forward(
        incoming: T.Buffer((1, 6, 8, 16, 16), "float32"),
        lower: T.Buffer((6, 8, 16, 16), "float32"),
        upper: T.Buffer((6, 8, 16, 16), "float32"),
        raw_alpha: T.Buffer((2, 1, 6, 164), "float32"),
        alpha_map: T.Buffer((8, 16, 16), "int32"),
        weight: T.Buffer((8, 3, 3, 3), "float32"),
        operator_bias: T.Buffer((8,), "float32"),
        input_center: T.Buffer((6, 3, 32, 32), "float32"),
        input_radius: T.Buffer((6, 3, 32, 32), "float32"),
        concrete_lower: T.Buffer((6, 1), "float32"),
        output_bias: T.Buffer((1, 6), "float32"),
    ):
        T.func_attr(
            {
                "global_symbol": BAB_INPUT_DOMAIN_FORWARD_SYMBOL,
                "tir.noalias": True,
                "boundflow.schema_version": (
                    "activation-bab-input-domain-streaming-forward/v1"
                ),
            }
        )
        coefficient = T.alloc_buffer((1,), "float32", scope="local")
        concrete_sum = T.alloc_buffer((1,), "float32", scope="local")
        bias_sum = T.alloc_buffer((1,), "float32", scope="local")
        partial = T.alloc_buffer((2, 128), "float32", scope="shared")
        reduction = T.alloc_buffer((2,), "float32", scope="local")
        for domain in T.thread_binding(6, thread="blockIdx.x"):
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
                                    0,
                                    domain,
                                    output_channel,
                                    output_h,
                                    output_w,
                                ]
                                lower_value = lower[
                                    domain, output_channel, output_h, output_w
                                ]
                                upper_value = upper[
                                    domain, output_channel, output_h, output_w
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
                                    raw_alpha[0, 0, domain, T.max(lookup, 0)],
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
                        * input_center[domain, input_channel, input_h, input_w]
                        - T.abs(coefficient[0])
                        * input_radius[domain, input_channel, input_h, input_w]
                    )
                bias_sum[0] = T.float32(0)
                for flat in T.serial(thread_x, 2048, step=128):
                    output_w = flat % 16
                    output_h = flat // 16 % 16
                    output_channel = flat // 256
                    source = incoming[0, domain, output_channel, output_h, output_w]
                    lower_value = lower[domain, output_channel, output_h, output_w]
                    upper_value = upper[domain, output_channel, output_h, output_w]
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
                        raw_alpha[0, 0, domain, T.max(lookup, 0)],
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
                    concrete_lower[domain, 0] = reduction[0]
                    output_bias[0, domain] = reduction[1]

    return forward


def _streaming_backward_primfunc():  # pylint: disable=too-many-statements
    """Recompute coefficients and emit only incoming/compressed-alpha VJPs."""

    from tvm.script import tir as T

    @T.prim_func
    def backward(
        incoming: T.Buffer((1, 6, 8, 16, 16), "float32"),
        lower: T.Buffer((6, 8, 16, 16), "float32"),
        upper: T.Buffer((6, 8, 16, 16), "float32"),
        raw_alpha: T.Buffer((2, 1, 6, 164), "float32"),
        alpha_map: T.Buffer((8, 16, 16), "int32"),
        weight: T.Buffer((8, 3, 3, 3), "float32"),
        operator_bias: T.Buffer((8,), "float32"),
        input_center: T.Buffer((6, 3, 32, 32), "float32"),
        input_radius: T.Buffer((6, 3, 32, 32), "float32"),
        concrete_gradient: T.Buffer((6, 1), "float32"),
        bias_gradient: T.Buffer((1, 6), "float32"),
        incoming_gradient: T.Buffer((1, 6, 8, 16, 16), "float32"),
        alpha_gradient: T.Buffer((2, 1, 6, 164), "float32"),
    ):
        T.func_attr(
            {
                "global_symbol": BAB_INPUT_DOMAIN_BACKWARD_SYMBOL,
                "tir.noalias": True,
                "boundflow.schema_version": (
                    "activation-bab-input-domain-streaming-backward/v1"
                ),
            }
        )
        coefficient = T.alloc_buffer((1,), "float32", scope="local")
        adjoint = T.alloc_buffer((1,), "float32", scope="local")
        for block_x in T.thread_binding(96, thread="blockIdx.x"):
            for thread_x in T.thread_binding(128, thread="threadIdx.x"):
                flat = block_x * 128 + thread_x
                if flat < 12288:
                    output_w = flat % 16
                    output_h = flat // 16 % 16
                    output_channel = flat // 256 % 8
                    domain = flat // 2048 % 6
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
                                            0,
                                            domain,
                                            source_channel,
                                            source_h,
                                            source_w,
                                        ]
                                        source_lower = lower[
                                            domain,
                                            source_channel,
                                            source_h,
                                            source_w,
                                        ]
                                        source_upper = upper[
                                            domain,
                                            source_channel,
                                            source_h,
                                            source_w,
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
                                                0,
                                                domain,
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
                            coefficient_adjoint = concrete_gradient[domain, 0] * (
                                input_center[domain, input_channel, input_h, input_w]
                                + T.if_then_else(
                                    coefficient[0] < T.float32(0),
                                    input_radius[
                                        domain, input_channel, input_h, input_w
                                    ],
                                    T.if_then_else(
                                        coefficient[0] > T.float32(0),
                                        -input_radius[
                                            domain, input_channel, input_h, input_w
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
                        + bias_gradient[0, domain] * operator_bias[output_channel]
                    )
                    source = incoming[0, domain, output_channel, output_h, output_w]
                    lower_value = lower[domain, output_channel, output_h, output_w]
                    upper_value = upper[domain, output_channel, output_h, output_w]
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
                        raw_alpha[0, 0, domain, T.max(lookup, 0)],
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
                    incoming_gradient[0, domain, output_channel, output_h, output_w] = (
                        adjoint[0] * slope + bias_gradient[0, domain] * intercept
                    )
                    if lookup >= 0:
                        alpha_gradient[0, 0, domain, lookup] = T.if_then_else(
                            source >= T.float32(0)
                            and ambiguous
                            and alpha_value >= T.float32(0)
                            and alpha_value <= T.float32(1),
                            adjoint[0] * source,
                            T.float32(0),
                        )
                        alpha_gradient[1, 0, domain, lookup] = T.float32(0)

    return backward


def build_bab_input_domain_modules_v1(template: BabInputDomainTemplateV1):
    """Build unscheduled semantics plus the streamed production schedule."""

    template.validate()
    import tvm

    unscheduled = tvm.IRModule(
        {
            template.forward_symbol: _forward_primfunc(template),
            template.backward_symbol: _streaming_backward_primfunc(),
        }
    )
    scheduled = tvm.IRModule(
        {
            template.forward_symbol: _streaming_forward_primfunc(),
            template.backward_symbol: _streaming_backward_primfunc(),
        }
    )
    return unscheduled, scheduled, _workspace_inventory(scheduled)


def compile_bab_input_domain_tir_v1(
    template: BabInputDomainTemplateV1,
) -> CompiledRootCrownInputDomainTIRV1:
    """Compile the activation-BaB input streaming schedule."""

    import tvm

    unscheduled, scheduled, inventory = build_bab_input_domain_modules_v1(template)
    executable = tvm.compile(
        scheduled, target=f"{template.target} -arch={template.compute_capability}"
    )
    sources = tuple(module.inspect_source() for module in executable.mod.imports)
    if not sources:
        raise RuntimeError(
            "activation-BaB input-domain compile produced no CUDA source"
        )
    return CompiledRootCrownInputDomainTIRV1(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        unscheduled_tir_hash=_canonical_hash(tvm.ir.save_json(unscheduled)),
        scheduled_tir_hash=_canonical_hash(tvm.ir.save_json(scheduled)),
        device_source_hash=_canonical_hash("\n".join(sources)),
        workspace_inventory=inventory,
        tvm_version=str(tvm.__version__),
    )


__all__ = [
    "BAB_INPUT_DOMAIN_BACKWARD_SYMBOL",
    "BAB_INPUT_DOMAIN_FORWARD_SYMBOL",
    "BabInputDomainTemplateV1",
    "build_bab_input_domain_modules_v1",
    "compile_bab_input_domain_tir_v1",
]
