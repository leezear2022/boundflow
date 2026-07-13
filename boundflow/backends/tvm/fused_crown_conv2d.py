"""Shape and compile-key contract for fused ReLU-plus-Conv2d CROWN tasks."""

# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-statements

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

# mypy: disable-error-code=import-untyped

FUSED_CROWN_CONV2D_SCHEMA_VERSION = "boundflow.fused_crown_conv2d/v1"
COEFFICIENT_LAYOUT = "DSCOHW"
WEIGHT_LAYOUT = "OIHW"
OUTPUT_LAYOUT = "DSCIHW"


def _pair(value: Tuple[int, int], *, name: str) -> Tuple[int, int]:
    if len(value) != 2 or any(int(item) <= 0 for item in value):
        raise ValueError(f"{name} must contain two positive integers")
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class FusedCrownConv2dSignature:  # pylint: disable=too-many-instance-attributes
    """Static Conv signature; weight values remain runtime inputs."""

    domain_batch: int
    spec_batch: int
    input_channels: int
    input_height: int
    input_width: int
    output_channels: int
    output_height: int
    output_width: int
    kernel_height: int
    kernel_width: int
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int] = (1, 1)
    groups: int = 1
    bias_present: bool = True
    coefficient_layout: str = COEFFICIENT_LAYOUT
    weight_layout: str = WEIGHT_LAYOUT
    output_layout: str = OUTPUT_LAYOUT
    dtype: str = "float32"
    target: str = "cuda"
    compute_capability: str = "sm_89"
    schedule_id: str = "output_gather_128t_v1"

    def validate(self) -> None:  # pylint: disable=too-many-branches
        """Reject every Conv attribute outside the frozen PR-12 v0/v1 subset."""

        for name in (
            "domain_batch",
            "spec_batch",
            "input_channels",
            "input_height",
            "input_width",
            "output_channels",
            "output_height",
            "output_width",
            "kernel_height",
            "kernel_width",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        stride_h, stride_w = _pair(self.stride, name="stride")
        dilation_h, dilation_w = _pair(self.dilation, name="dilation")
        if len(self.padding) != 2 or any(int(item) < 0 for item in self.padding):
            raise ValueError("padding must contain two non-negative integers")
        if self.groups != 1:
            raise NotImplementedError("fused Conv2d v1 only supports groups=1")
        if (dilation_h, dilation_w) != (1, 1):
            raise NotImplementedError("fused Conv2d v1 only supports dilation=1")
        if (stride_h, stride_w) not in {(1, 1), (2, 2)}:
            raise NotImplementedError("fused Conv2d v1 only supports stride=1 or 2")
        if (self.kernel_height, self.kernel_width) not in {(1, 1), (3, 3)}:
            raise NotImplementedError("fused Conv2d v1 only supports 1x1 or 3x3")
        if tuple(int(item) for item in self.padding) not in {(0, 0), (1, 1)}:
            raise NotImplementedError("fused Conv2d v1 only supports padding=0 or 1")
        if (
            self.coefficient_layout != COEFFICIENT_LAYOUT
            or self.weight_layout != WEIGHT_LAYOUT
            or self.output_layout != OUTPUT_LAYOUT
        ):
            raise NotImplementedError("unsupported fused Conv2d layout")
        if self.dtype != "float32":
            raise NotImplementedError("fused Conv2d v1 only supports float32")
        if not self.target.startswith("cuda"):
            raise NotImplementedError("fused Conv2d v1 only supports CUDA")
        if not self.compute_capability.startswith("sm_"):
            raise ValueError("compute_capability must use the sm_NN form")
        if self.schedule_id != "output_gather_128t_v1":
            raise NotImplementedError(f"unsupported schedule_id: {self.schedule_id}")
        expected_h, expected_w = self.expected_output_spatial()
        if (self.output_height, self.output_width) != (expected_h, expected_w):
            raise ValueError(
                "Conv2d output shape does not match the explicit input shape: "
                f"expected {(expected_h, expected_w)}, got "
                f"{(self.output_height, self.output_width)}"
            )
        self.output_padding()

    def expected_output_spatial(self) -> Tuple[int, int]:
        """Return the forward Conv2d output shape from the explicit input shape."""

        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation
        output_h = (
            self.input_height + 2 * pad_h - dilation_h * (self.kernel_height - 1) - 1
        ) // stride_h + 1
        output_w = (
            self.input_width + 2 * pad_w - dilation_w * (self.kernel_width - 1) - 1
        ) // stride_w + 1
        return int(output_h), int(output_w)

    def output_padding(self) -> Tuple[int, int]:
        """Derive ConvTranspose output padding from the original Primal input shape."""

        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation
        base_h = (
            (self.output_height - 1) * stride_h
            - 2 * pad_h
            + dilation_h * (self.kernel_height - 1)
            + 1
        )
        base_w = (
            (self.output_width - 1) * stride_w
            - 2 * pad_w
            + dilation_w * (self.kernel_width - 1)
            + 1
        )
        output_padding = self.input_height - base_h, self.input_width - base_w
        if not (
            0 <= output_padding[0] < stride_h and 0 <= output_padding[1] < stride_w
        ):
            raise ValueError(
                "explicit input shape cannot be recovered by ConvTranspose2d: "
                f"output_padding={output_padding}, stride={self.stride}"
            )
        return output_padding

    @property
    def target_string(self) -> str:
        """Return the CUDA target with the cache-keyed compute capability."""

        return f"{self.target} -arch={self.compute_capability}"


def _require_stride_one_v0(signature: FusedCrownConv2dSignature) -> None:
    signature.validate()
    if signature.stride != (1, 1):
        raise NotImplementedError("fused Conv2d lowering v0 only supports stride=1")


def build_fused_crown_conv2d_primfunc(  # pylint: disable=too-many-locals
    signature: FusedCrownConv2dSignature,
):
    """Build output-gather reductions with sign/slope selection inline."""

    _require_stride_one_v0(signature)
    import tvm  # pylint: disable=import-outside-toplevel
    from tvm import te  # pylint: disable=import-outside-toplevel

    domain = signature.domain_batch
    spec = signature.spec_batch
    input_channels = signature.input_channels
    input_h, input_w = signature.input_height, signature.input_width
    output_channels = signature.output_channels
    output_h, output_w = signature.output_height, signature.output_width
    kernel_h, kernel_w = signature.kernel_height, signature.kernel_width
    stride_h, stride_w = signature.stride
    pad_h, pad_w = signature.padding
    dilation_h, dilation_w = signature.dilation
    dtype = signature.dtype
    zero = tvm.tir.const(0.0, dtype)

    coeff_shape = (domain, spec, output_channels, output_h, output_w)
    relaxation_shape = (domain, output_channels, output_h, output_w)
    coeff_u = te.placeholder(coeff_shape, dtype=dtype, name="coeff_u")
    coeff_l = te.placeholder(coeff_shape, dtype=dtype, name="coeff_l")
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

    def scaled_upper(d_idx, s_idx, co_idx, ho_idx, wo_idx):
        coeff = coeff_u[d_idx, s_idx, co_idx, ho_idx, wo_idx]
        return tvm.tir.if_then_else(
            coeff >= zero,
            coeff * alpha_u[d_idx, co_idx, ho_idx, wo_idx],
            coeff * alpha_l[d_idx, co_idx, ho_idx, wo_idx],
        )

    def scaled_lower(d_idx, s_idx, co_idx, ho_idx, wo_idx):
        coeff = coeff_l[d_idx, s_idx, co_idx, ho_idx, wo_idx]
        return tvm.tir.if_then_else(
            coeff >= zero,
            coeff * alpha_l[d_idx, co_idx, ho_idx, wo_idx],
            coeff * alpha_u[d_idx, co_idx, ho_idx, wo_idx],
        )

    def gather_term(
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
            scaled(d_idx, s_idx, co_idx, ho_idx, wo_idx)
            * weight[co_idx, ci_idx, kh_idx, kw_idx],
            zero,
        )

    reduce_u_co = te.reduce_axis((0, output_channels), "reduce_u_co")
    reduce_u_kh = te.reduce_axis((0, kernel_h), "reduce_u_kh")
    reduce_u_kw = te.reduce_axis((0, kernel_w), "reduce_u_kw")
    reduce_l_co = te.reduce_axis((0, output_channels), "reduce_l_co")
    reduce_l_kh = te.reduce_axis((0, kernel_h), "reduce_l_kh")
    reduce_l_kw = te.reduce_axis((0, kernel_w), "reduce_l_kw")
    previous_shape = (domain, spec, input_channels, input_h, input_w)
    previous_u = te.compute(
        previous_shape,
        lambda d_idx, s_idx, ci_idx, hi_idx, wi_idx: te.sum(
            gather_term(
                scaled_upper,
                d_idx,
                s_idx,
                ci_idx,
                hi_idx,
                wi_idx,
                reduce_u_co,
                reduce_u_kh,
                reduce_u_kw,
            ),
            axis=(reduce_u_co, reduce_u_kh, reduce_u_kw),
        ),
        name="previous_u",
    )
    previous_l = te.compute(
        previous_shape,
        lambda d_idx, s_idx, ci_idx, hi_idx, wi_idx: te.sum(
            gather_term(
                scaled_lower,
                d_idx,
                s_idx,
                ci_idx,
                hi_idx,
                wi_idx,
                reduce_l_co,
                reduce_l_kh,
                reduce_l_kw,
            ),
            axis=(reduce_l_co, reduce_l_kh, reduce_l_kw),
        ),
        name="previous_l",
    )

    reduce_bu_co = te.reduce_axis((0, output_channels), "reduce_bu_co")
    reduce_bu_h = te.reduce_axis((0, output_h), "reduce_bu_h")
    reduce_bu_w = te.reduce_axis((0, output_w), "reduce_bu_w")
    reduce_bl_co = te.reduce_axis((0, output_channels), "reduce_bl_co")
    reduce_bl_h = te.reduce_axis((0, output_h), "reduce_bl_h")
    reduce_bl_w = te.reduce_axis((0, output_w), "reduce_bl_w")

    def upper_bias_term(d_idx, s_idx, co_idx, ho_idx, wo_idx):
        coeff = coeff_u[d_idx, s_idx, co_idx, ho_idx, wo_idx]
        slope = tvm.tir.if_then_else(
            coeff >= zero,
            alpha_u[d_idx, co_idx, ho_idx, wo_idx],
            alpha_l[d_idx, co_idx, ho_idx, wo_idx],
        )
        intercept = tvm.tir.if_then_else(
            coeff >= zero,
            beta_u[d_idx, co_idx, ho_idx, wo_idx],
            beta_l[d_idx, co_idx, ho_idx, wo_idx],
        )
        conv_bias = bias[co_idx] if bias is not None else zero
        return coeff * (intercept + slope * conv_bias)

    def lower_bias_term(d_idx, s_idx, co_idx, ho_idx, wo_idx):
        coeff = coeff_l[d_idx, s_idx, co_idx, ho_idx, wo_idx]
        slope = tvm.tir.if_then_else(
            coeff >= zero,
            alpha_l[d_idx, co_idx, ho_idx, wo_idx],
            alpha_u[d_idx, co_idx, ho_idx, wo_idx],
        )
        intercept = tvm.tir.if_then_else(
            coeff >= zero,
            beta_l[d_idx, co_idx, ho_idx, wo_idx],
            beta_u[d_idx, co_idx, ho_idx, wo_idx],
        )
        conv_bias = bias[co_idx] if bias is not None else zero
        return coeff * (intercept + slope * conv_bias)

    bias_delta_u = te.compute(
        (domain, spec),
        lambda d_idx, s_idx: te.sum(
            upper_bias_term(d_idx, s_idx, reduce_bu_co, reduce_bu_h, reduce_bu_w),
            axis=(reduce_bu_co, reduce_bu_h, reduce_bu_w),
        ),
        name="bias_delta_u",
    )
    bias_delta_l = te.compute(
        (domain, spec),
        lambda d_idx, s_idx: te.sum(
            lower_bias_term(d_idx, s_idx, reduce_bl_co, reduce_bl_h, reduce_bl_w),
            axis=(reduce_bl_co, reduce_bl_h, reduce_bl_w),
        ),
        name="bias_delta_l",
    )
    inputs = [coeff_u, coeff_l, alpha_u, beta_u, alpha_l, beta_l, weight]
    if bias is not None:
        inputs.append(bias)
    return te.create_prim_func(
        [*inputs, previous_u, previous_l, bias_delta_u, bias_delta_l]
    ).with_attr("boundflow.schema_version", FUSED_CROWN_CONV2D_SCHEMA_VERSION)


def schedule_fused_crown_conv2d(signature: FusedCrownConv2dSignature):
    """Apply the deterministic output-gather CUDA schedule."""

    _require_stride_one_v0(signature)
    import tvm  # pylint: disable=import-outside-toplevel

    module = tvm.IRModule({"main": build_fused_crown_conv2d_primfunc(signature)})
    schedule = tvm.tir.Schedule(module)
    for block_name in ("previous_u", "previous_l", "bias_delta_u", "bias_delta_l"):
        block = schedule.get_block(block_name, func_name="main")
        loops = schedule.get_loops(block)
        spatial_count = 5 if block_name.startswith("previous") else 2
        fused = schedule.fuse(*loops[:spatial_count])
        block_loop, thread_loop = schedule.split(fused, factors=[None, 128])
        schedule.bind(block_loop, "blockIdx.x")
        schedule.bind(thread_loop, "threadIdx.x")
    return schedule.mod


def allocated_intermediate_buffers(
    signature: FusedCrownConv2dSignature, *, scheduled: bool
) -> Tuple[str, ...]:
    """Return local allocations before or after scheduling."""

    import tvm  # pylint: disable=import-outside-toplevel

    names: list[str] = []

    def visit(node) -> None:
        if isinstance(node, tvm.tir.Block):
            names.extend(buffer.name for buffer in node.alloc_buffers)

    function = (
        schedule_fused_crown_conv2d(signature)["main"]
        if scheduled
        else build_fused_crown_conv2d_primfunc(signature)
    )
    tvm.tir.stmt_functor.post_order_visit(function.body, visit)
    return tuple(names)


def build_fused_crown_conv2d_relax_ir_module(
    signature: FusedCrownConv2dSignature, *, function_name: str = "main"
):
    """Wrap one specialized Conv task in a thin Relax ``call_tir`` function."""

    _require_stride_one_v0(signature)
    from tvm import relax  # pylint: disable=import-outside-toplevel

    domain, spec = signature.domain_batch, signature.spec_batch
    output_shape = (
        domain,
        spec,
        signature.output_channels,
        signature.output_height,
        signature.output_width,
    )
    relaxation_shape = (
        domain,
        signature.output_channels,
        signature.output_height,
        signature.output_width,
    )
    parameters = [
        relax.Var("coeff_u", relax.TensorStructInfo(output_shape, signature.dtype)),
        relax.Var("coeff_l", relax.TensorStructInfo(output_shape, signature.dtype)),
        relax.Var("alpha_u", relax.TensorStructInfo(relaxation_shape, signature.dtype)),
        relax.Var("beta_u", relax.TensorStructInfo(relaxation_shape, signature.dtype)),
        relax.Var("alpha_l", relax.TensorStructInfo(relaxation_shape, signature.dtype)),
        relax.Var("beta_l", relax.TensorStructInfo(relaxation_shape, signature.dtype)),
        relax.Var(
            "weight",
            relax.TensorStructInfo(
                (
                    signature.output_channels,
                    signature.input_channels,
                    signature.kernel_height,
                    signature.kernel_width,
                ),
                signature.dtype,
            ),
        ),
    ]
    if signature.bias_present:
        parameters.append(
            relax.Var(
                "bias",
                relax.TensorStructInfo((signature.output_channels,), signature.dtype),
            )
        )
    builder = relax.BlockBuilder()
    tir_name = f"{function_name}_fused_crown_conv2d_tir"
    primfunc = schedule_fused_crown_conv2d(signature)["main"].with_attr(
        "global_symbol", tir_name
    )
    tir_global = builder.add_func(primfunc, tir_name)
    previous_shape = (
        domain,
        spec,
        signature.input_channels,
        signature.input_height,
        signature.input_width,
    )
    with builder.function(function_name, parameters):
        with builder.dataflow():
            output = builder.emit(
                relax.call_tir(
                    tir_global,
                    relax.Tuple(parameters),
                    out_sinfo=[
                        relax.TensorStructInfo(previous_shape, signature.dtype),
                        relax.TensorStructInfo(previous_shape, signature.dtype),
                        relax.TensorStructInfo((domain, spec), signature.dtype),
                        relax.TensorStructInfo((domain, spec), signature.dtype),
                    ],
                )
            )
            output = builder.emit_output(output)
        builder.emit_func_output(output)
    return builder.get()


@lru_cache(maxsize=128)
def build_fused_crown_conv2d_module(signature: FusedCrownConv2dSignature):
    """Compile and cache one deterministic CUDA task."""

    _require_stride_one_v0(signature)
    import tvm  # pylint: disable=import-outside-toplevel

    return tvm.compile(
        schedule_fused_crown_conv2d(signature), target=signature.target_string
    )["main"]


__all__ = [
    "COEFFICIENT_LAYOUT",
    "FUSED_CROWN_CONV2D_SCHEMA_VERSION",
    "FusedCrownConv2dSignature",
    "OUTPUT_LAYOUT",
    "WEIGHT_LAYOUT",
    "allocated_intermediate_buffers",
    "build_fused_crown_conv2d_module",
    "build_fused_crown_conv2d_primfunc",
    "build_fused_crown_conv2d_relax_ir_module",
    "schedule_fused_crown_conv2d",
]
