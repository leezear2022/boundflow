"""CIBC-parity horizontally fused sparse Conv oracle implemented in Triton."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,too-many-arguments,too-many-locals
# pylint: disable=too-many-positional-arguments,too-few-public-methods
# pylint: disable=not-callable,protected-access,missing-function-docstring
# pylint: disable=invalid-name,too-many-statements,unused-argument
# pylint: disable=abstract-method,arguments-differ,too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
import triton
import triton.language as tl

from boundflow.ir.differentiable_lower_sparse_conv_tir import (
    DifferentiableLowerSparseConvTIRTemplateV1,
)

from .fsg4_b4b2_sparse_conv_tir import (
    SparseConvTIRTensorsV1,
    _validate_sparse_conv_tensors,
    build_b4b2_sparse_conv_template_v1,
    build_b4b2_sparse_conv_tensors_v1,
)
from .fsg4_b4b1_pytorch_reference import build_b4b1_differentiable_lower_ir_v1
from .fsg4_b4b1_reference_capture import ProductionDifferentiableReferenceCaptureV1
from .fsg4_b4b2_sparse_conv_timing import (
    SparseConvExecutionV1,
    execute_sparse_conv_pytorch_baseline_v1,
)


@dataclass(frozen=True)
class CIBCTritonConfigV2:
    """One preregistered horizontal-fusion schedule."""

    ordinal: int
    block_m: int
    block_k: int
    num_warps: int

    def validate(self) -> None:
        if (self.block_m, self.block_k, self.num_warps) not in {
            (32, 16, 4),
            (32, 32, 4),
            (32, 64, 4),
            (64, 16, 4),
            (64, 32, 4),
            (64, 64, 4),
            (64, 32, 8),
            (64, 64, 8),
            (128, 16, 4),
            (128, 32, 4),
            (128, 64, 4),
            (128, 64, 8),
        } or self.ordinal < 0:
            raise ValueError("CIBC Triton schedule differs")


CIBC_TRITON_CONFIGS_V2 = tuple(
    CIBCTritonConfigV2(ordinal, block_m, block_k, num_warps)
    for ordinal, (block_m, block_k, num_warps) in enumerate(
        (
            (32, 16, 4),
            (32, 32, 4),
            (32, 64, 4),
            (64, 16, 4),
            (64, 32, 4),
            (64, 64, 4),
            (64, 32, 8),
            (64, 64, 8),
            (128, 16, 4),
            (128, 32, 4),
            (128, 64, 4),
            (128, 64, 8),
        )
    )
)


@triton.jit
def _cibc_forward_kernel_v2(
    incoming_ptr,
    lower_ptr,
    upper_ptr,
    alpha_ptr,
    incoming_bias_ptr,
    weight_ptr,
    operator_bias_ptr,
    alpha_map_ptr,
    combined_output_ptr,
    N_OUTPUT_A: tl.constexpr,
    N_OUTPUT_BIAS: tl.constexpr,
    DOMAIN: tl.constexpr,
    SPEC: tl.constexpr,
    CHANNEL: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    ALPHA_COUNT: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    PADDING_H: tl.constexpr,
    PADDING_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    output_programs: tl.constexpr = tl.cdiv(N_OUTPUT_A, BLOCK_M)
    if pid < output_programs:
        output_offset = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        output_mask = output_offset < N_OUTPUT_A
        output_w = output_offset % WIDTH
        output_h = (output_offset // WIDTH) % HEIGHT
        output_channel = (output_offset // (WIDTH * HEIGHT)) % CHANNEL
        output_spec = (output_offset // (WIDTH * HEIGHT * CHANNEL)) % SPEC
        output_domain = output_offset // (WIDTH * HEIGHT * CHANNEL * SPEC)
        accumulator = tl.zeros((BLOCK_M,), dtype=tl.float32)
        reduction_extent: tl.constexpr = CHANNEL * KERNEL_H * KERNEL_W
        for reduction_start in range(0, reduction_extent, BLOCK_K):
            reduction = reduction_start + tl.arange(0, BLOCK_K)
            input_channel = reduction // (KERNEL_H * KERNEL_W)
            kernel_remainder = reduction % (KERNEL_H * KERNEL_W)
            kernel_h = kernel_remainder // KERNEL_W
            kernel_w = kernel_remainder % KERNEL_W
            input_h = output_h[:, None] + PADDING_H - kernel_h[None, :]
            input_w = output_w[:, None] + PADDING_W - kernel_w[None, :]
            valid = (
                output_mask[:, None]
                & (reduction[None, :] < reduction_extent)
                & (input_h >= 0)
                & (input_h < HEIGHT)
                & (input_w >= 0)
                & (input_w < WIDTH)
            )
            input_offset = (
                output_domain[:, None] * SPEC * CHANNEL * HEIGHT * WIDTH
                + output_spec[:, None] * CHANNEL * HEIGHT * WIDTH
                + input_channel[None, :] * HEIGHT * WIDTH
                + input_h * WIDTH
                + input_w
            )
            bound_offset = (
                output_domain[:, None] * CHANNEL * HEIGHT * WIDTH
                + input_channel[None, :] * HEIGHT * WIDTH
                + input_h * WIDTH
                + input_w
            )
            incoming = tl.load(incoming_ptr + input_offset, mask=valid, other=0.0)
            lower = tl.load(lower_ptr + bound_offset, mask=valid, other=0.0)
            upper = tl.load(upper_ptr + bound_offset, mask=valid, other=0.0)
            positive = lower >= 0.0
            negative = upper <= 0.0
            ambiguous = (~positive) & (~negative)
            denominator = tl.maximum(upper - lower, 1.1920928955078125e-7)
            upper_slope = tl.where(
                positive, 1.0, tl.where(negative, 0.0, upper / denominator)
            )
            feature_offset = (
                input_channel[None, :] * HEIGHT * WIDTH + input_h * WIDTH + input_w
            )
            alpha_ordinal = tl.load(
                alpha_map_ptr + feature_offset, mask=valid, other=-1
            )
            alpha_valid = valid & (alpha_ordinal >= 0)
            alpha = tl.load(
                alpha_ptr + output_domain[:, None] * ALPHA_COUNT + alpha_ordinal,
                mask=alpha_valid,
                other=0.0,
            )
            alpha = tl.minimum(tl.maximum(alpha, 0.0), 1.0)
            lower_slope = tl.where(
                ambiguous,
                alpha,
                tl.where(positive, 1.0, 0.0),
            )
            selected_slope = tl.where(incoming >= 0.0, lower_slope, upper_slope)
            weight_offset = (
                input_channel[None, :] * CHANNEL * KERNEL_H * KERNEL_W
                + output_channel[:, None] * KERNEL_H * KERNEL_W
                + kernel_h[None, :] * KERNEL_W
                + kernel_w[None, :]
            )
            weight = tl.load(weight_ptr + weight_offset, mask=valid, other=0.0)
            accumulator += tl.sum(incoming * selected_slope * weight, axis=1)
        tl.store(combined_output_ptr + output_offset, accumulator, mask=output_mask)
    else:
        bias_ordinal = pid - output_programs
        if bias_ordinal < N_OUTPUT_BIAS:
            bias_domain = bias_ordinal // SPEC
            bias_spec = bias_ordinal % SPEC
            accumulator = tl.zeros((BLOCK_K,), dtype=tl.float32)
            bias_reduction_extent: tl.constexpr = CHANNEL * HEIGHT * WIDTH
            for reduction_start in range(0, bias_reduction_extent, BLOCK_K):
                reduction = reduction_start + tl.arange(0, BLOCK_K)
                reduction_mask = reduction < bias_reduction_extent
                input_channel = reduction // (HEIGHT * WIDTH)
                spatial_remainder = reduction % (HEIGHT * WIDTH)
                input_h = spatial_remainder // WIDTH
                input_w = spatial_remainder % WIDTH
                input_offset = (
                    bias_domain * SPEC * CHANNEL * HEIGHT * WIDTH
                    + bias_spec * CHANNEL * HEIGHT * WIDTH
                    + input_channel * HEIGHT * WIDTH
                    + input_h * WIDTH
                    + input_w
                )
                bound_offset = (
                    bias_domain * CHANNEL * HEIGHT * WIDTH
                    + input_channel * HEIGHT * WIDTH
                    + input_h * WIDTH
                    + input_w
                )
                incoming = tl.load(
                    incoming_ptr + input_offset, mask=reduction_mask, other=0.0
                )
                lower = tl.load(
                    lower_ptr + bound_offset, mask=reduction_mask, other=0.0
                )
                upper = tl.load(
                    upper_ptr + bound_offset, mask=reduction_mask, other=0.0
                )
                positive = lower >= 0.0
                negative = upper <= 0.0
                ambiguous = (~positive) & (~negative)
                denominator = tl.maximum(upper - lower, 1.1920928955078125e-7)
                upper_slope = tl.where(
                    positive, 1.0, tl.where(negative, 0.0, upper / denominator)
                )
                feature_offset = (
                    input_channel * HEIGHT * WIDTH + input_h * WIDTH + input_w
                )
                alpha_ordinal = tl.load(
                    alpha_map_ptr + feature_offset, mask=reduction_mask, other=-1
                )
                alpha_valid = reduction_mask & (alpha_ordinal >= 0)
                alpha = tl.load(
                    alpha_ptr + bias_domain * ALPHA_COUNT + alpha_ordinal,
                    mask=alpha_valid,
                    other=0.0,
                )
                alpha = tl.minimum(tl.maximum(alpha, 0.0), 1.0)
                lower_slope = tl.where(ambiguous, alpha, tl.where(positive, 1.0, 0.0))
                selected_slope = tl.where(incoming >= 0.0, lower_slope, upper_slope)
                upper_intercept = tl.where(ambiguous, -lower * upper_slope, 0.0)
                selected_intercept = tl.where(incoming >= 0.0, 0.0, upper_intercept)
                operator_bias = tl.load(
                    operator_bias_ptr + input_channel,
                    mask=reduction_mask,
                    other=0.0,
                )
                accumulator += incoming * (
                    selected_intercept + selected_slope * operator_bias
                )
            incoming_bias = tl.load(incoming_bias_ptr + bias_ordinal)
            tl.store(
                combined_output_ptr + N_OUTPUT_A + bias_ordinal,
                incoming_bias + tl.sum(accumulator, axis=0),
            )


@triton.jit
def _cibc_backward_kernel_v2(
    incoming_ptr,
    lower_ptr,
    upper_ptr,
    alpha_ptr,
    weight_ptr,
    operator_bias_ptr,
    output_a_gradient_ptr,
    output_bias_gradient_ptr,
    alpha_map_ptr,
    alpha_channel_ptr,
    alpha_height_ptr,
    alpha_width_ptr,
    combined_gradient_ptr,
    N_INCOMING: tl.constexpr,
    N_ALPHA: tl.constexpr,
    DOMAIN: tl.constexpr,
    SPEC: tl.constexpr,
    CHANNEL: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    ALPHA_COUNT: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    PADDING_H: tl.constexpr,
    PADDING_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    incoming_programs: tl.constexpr = tl.cdiv(N_INCOMING, BLOCK_M)
    if pid < incoming_programs:
        incoming_offset = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        incoming_mask = incoming_offset < N_INCOMING
        input_w = incoming_offset % WIDTH
        input_h = (incoming_offset // WIDTH) % HEIGHT
        input_channel = (incoming_offset // (WIDTH * HEIGHT)) % CHANNEL
        input_spec = (incoming_offset // (WIDTH * HEIGHT * CHANNEL)) % SPEC
        input_domain = incoming_offset // (WIDTH * HEIGHT * CHANNEL * SPEC)
        adjoint = tl.zeros((BLOCK_M,), dtype=tl.float32)
        reduction_extent: tl.constexpr = CHANNEL * KERNEL_H * KERNEL_W
        for reduction_start in range(0, reduction_extent, BLOCK_K):
            reduction = reduction_start + tl.arange(0, BLOCK_K)
            output_channel = reduction // (KERNEL_H * KERNEL_W)
            kernel_remainder = reduction % (KERNEL_H * KERNEL_W)
            kernel_h = kernel_remainder // KERNEL_W
            kernel_w = kernel_remainder % KERNEL_W
            output_h = input_h[:, None] - PADDING_H + kernel_h[None, :]
            output_w = input_w[:, None] - PADDING_W + kernel_w[None, :]
            valid = (
                incoming_mask[:, None]
                & (reduction[None, :] < reduction_extent)
                & (output_h >= 0)
                & (output_h < HEIGHT)
                & (output_w >= 0)
                & (output_w < WIDTH)
            )
            output_offset = (
                input_domain[:, None] * SPEC * CHANNEL * HEIGHT * WIDTH
                + input_spec[:, None] * CHANNEL * HEIGHT * WIDTH
                + output_channel[None, :] * HEIGHT * WIDTH
                + output_h * WIDTH
                + output_w
            )
            weight_offset = (
                input_channel[:, None] * CHANNEL * KERNEL_H * KERNEL_W
                + output_channel[None, :] * KERNEL_H * KERNEL_W
                + kernel_h[None, :] * KERNEL_W
                + kernel_w[None, :]
            )
            output_gradient = tl.load(
                output_a_gradient_ptr + output_offset, mask=valid, other=0.0
            )
            weight = tl.load(weight_ptr + weight_offset, mask=valid, other=0.0)
            adjoint += tl.sum(output_gradient * weight, axis=1)
        bias_gradient_offset = input_domain * SPEC + input_spec
        bias_gradient = tl.load(
            output_bias_gradient_ptr + bias_gradient_offset,
            mask=incoming_mask,
            other=0.0,
        )
        operator_bias = tl.load(
            operator_bias_ptr + input_channel, mask=incoming_mask, other=0.0
        )
        adjoint += bias_gradient * operator_bias
        bound_offset = (
            input_domain * CHANNEL * HEIGHT * WIDTH
            + input_channel * HEIGHT * WIDTH
            + input_h * WIDTH
            + input_w
        )
        incoming = tl.load(
            incoming_ptr + incoming_offset, mask=incoming_mask, other=0.0
        )
        lower = tl.load(lower_ptr + bound_offset, mask=incoming_mask, other=0.0)
        upper = tl.load(upper_ptr + bound_offset, mask=incoming_mask, other=0.0)
        positive = lower >= 0.0
        negative = upper <= 0.0
        ambiguous = (~positive) & (~negative)
        denominator = tl.maximum(upper - lower, 1.1920928955078125e-7)
        upper_slope = tl.where(
            positive, 1.0, tl.where(negative, 0.0, upper / denominator)
        )
        alpha_map_offset = input_channel * HEIGHT * WIDTH + input_h * WIDTH + input_w
        alpha_ordinal = tl.load(
            alpha_map_ptr + alpha_map_offset,
            mask=incoming_mask,
            other=-1,
        )
        alpha_valid = incoming_mask & (alpha_ordinal >= 0)
        alpha = tl.load(
            alpha_ptr + input_domain * ALPHA_COUNT + alpha_ordinal,
            mask=alpha_valid,
            other=0.0,
        )
        alpha = tl.minimum(tl.maximum(alpha, 0.0), 1.0)
        lower_slope = tl.where(ambiguous, alpha, tl.where(positive, 1.0, 0.0))
        selected_slope = tl.where(incoming >= 0.0, lower_slope, upper_slope)
        upper_intercept = tl.where(ambiguous, -lower * upper_slope, 0.0)
        selected_intercept = tl.where(incoming >= 0.0, 0.0, upper_intercept)
        incoming_gradient = (
            adjoint * selected_slope + bias_gradient * selected_intercept
        )
        tl.store(
            combined_gradient_ptr + incoming_offset,
            incoming_gradient,
            mask=incoming_mask,
        )
    else:
        alpha_program = pid - incoming_programs
        alpha_offset = alpha_program * BLOCK_M + tl.arange(0, BLOCK_M)
        alpha_mask = alpha_offset < N_ALPHA
        alpha_ordinal = alpha_offset % ALPHA_COUNT
        alpha_domain = alpha_offset // ALPHA_COUNT
        input_channel = tl.load(
            alpha_channel_ptr + alpha_ordinal, mask=alpha_mask, other=0
        )
        input_h = tl.load(alpha_height_ptr + alpha_ordinal, mask=alpha_mask, other=0)
        input_w = tl.load(alpha_width_ptr + alpha_ordinal, mask=alpha_mask, other=0)
        adjoint = tl.zeros((BLOCK_M,), dtype=tl.float32)
        alpha_reduction_extent: tl.constexpr = SPEC * CHANNEL * KERNEL_H * KERNEL_W
        for reduction_start in range(0, alpha_reduction_extent, BLOCK_K):
            reduction = reduction_start + tl.arange(0, BLOCK_K)
            kernel_reduction = reduction % (CHANNEL * KERNEL_H * KERNEL_W)
            output_spec = reduction // (CHANNEL * KERNEL_H * KERNEL_W)
            output_channel = kernel_reduction // (KERNEL_H * KERNEL_W)
            kernel_remainder = kernel_reduction % (KERNEL_H * KERNEL_W)
            kernel_h = kernel_remainder // KERNEL_W
            kernel_w = kernel_remainder % KERNEL_W
            output_h = input_h[:, None] - PADDING_H + kernel_h[None, :]
            output_w = input_w[:, None] - PADDING_W + kernel_w[None, :]
            valid = (
                alpha_mask[:, None]
                & (reduction[None, :] < alpha_reduction_extent)
                & (output_h >= 0)
                & (output_h < HEIGHT)
                & (output_w >= 0)
                & (output_w < WIDTH)
            )
            output_offset = (
                alpha_domain[:, None] * SPEC * CHANNEL * HEIGHT * WIDTH
                + output_spec[None, :] * CHANNEL * HEIGHT * WIDTH
                + output_channel[None, :] * HEIGHT * WIDTH
                + output_h * WIDTH
                + output_w
            )
            weight_offset = (
                input_channel[:, None] * CHANNEL * KERNEL_H * KERNEL_W
                + output_channel[None, :] * KERNEL_H * KERNEL_W
                + kernel_h[None, :] * KERNEL_W
                + kernel_w[None, :]
            )
            output_gradient = tl.load(
                output_a_gradient_ptr + output_offset, mask=valid, other=0.0
            )
            weight = tl.load(weight_ptr + weight_offset, mask=valid, other=0.0)
            adjoint += tl.sum(output_gradient * weight, axis=1)
        bias_gradient = tl.load(
            output_bias_gradient_ptr + alpha_domain * SPEC, mask=alpha_mask, other=0.0
        )
        operator_bias = tl.load(
            operator_bias_ptr + input_channel, mask=alpha_mask, other=0.0
        )
        adjoint += bias_gradient * operator_bias
        input_offset = (
            alpha_domain * SPEC * CHANNEL * HEIGHT * WIDTH
            + input_channel * HEIGHT * WIDTH
            + input_h * WIDTH
            + input_w
        )
        bound_offset = (
            alpha_domain * CHANNEL * HEIGHT * WIDTH
            + input_channel * HEIGHT * WIDTH
            + input_h * WIDTH
            + input_w
        )
        incoming = tl.load(incoming_ptr + input_offset, mask=alpha_mask, other=0.0)
        lower = tl.load(lower_ptr + bound_offset, mask=alpha_mask, other=0.0)
        upper = tl.load(upper_ptr + bound_offset, mask=alpha_mask, other=0.0)
        alpha_value = tl.load(alpha_ptr + alpha_offset, mask=alpha_mask, other=0.0)
        ambiguous = (lower < 0.0) & (upper > 0.0)
        alpha_owned = (
            ambiguous & (incoming >= 0.0) & (alpha_value >= 0.0) & (alpha_value <= 1.0)
        )
        alpha_gradient = tl.where(alpha_owned, adjoint * incoming, 0.0)
        tl.store(
            combined_gradient_ptr + N_INCOMING + alpha_offset,
            alpha_gradient,
            mask=alpha_mask,
        )


class _CIBCTritonFunctionV2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        incoming,
        lower,
        upper,
        alpha,
        incoming_bias,
        weight,
        operator_bias,
        executor,
    ):
        output_a, output_bias = executor.forward(
            incoming, lower, upper, alpha, incoming_bias, weight, operator_bias
        )
        ctx.executor = executor
        ctx.save_for_backward(incoming, lower, upper, alpha, weight, operator_bias)
        return output_a, output_bias

    @staticmethod
    def backward(ctx, output_a_gradient, output_bias_gradient):
        if torch.is_grad_enabled():
            ctx.executor.fallback_count += 1
            raise RuntimeError("CIBC Triton higher-order gradient is not admitted")
        incoming, lower, upper, alpha, weight, operator_bias = ctx.saved_tensors
        incoming_gradient, alpha_gradient = ctx.executor.backward(
            incoming,
            lower,
            upper,
            alpha,
            weight,
            operator_bias,
            output_a_gradient,
            output_bias_gradient,
        )
        return incoming_gradient, None, None, alpha_gradient, None, None, None, None


class CIBCTritonExecutorV2:
    """Exactly-one-forward/exactly-one-backward horizontal-fusion executor."""

    def __init__(
        self,
        template: DifferentiableLowerSparseConvTIRTemplateV1,
        config: CIBCTritonConfigV2,
        *,
        device: torch.device,
    ) -> None:
        template.validate()
        config.validate()
        if device.type != "cuda":
            raise ValueError("CIBC Triton device differs")
        self.template = template
        self.config = config
        self.device = device
        self.forward_launch_count = 0
        self.backward_launch_count = 0
        self.fallback_count = 0
        self.eager_count = 0
        alpha_map = torch.full(
            (template.channels * template.height * template.width,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        coordinates = template.alpha_coordinates
        flattened = [
            channel * template.height * template.width + height * template.width + width
            for channel, height, width in coordinates
        ]
        index = torch.tensor(flattened, dtype=torch.int64, device=device)
        alpha_map[index] = torch.arange(
            len(coordinates), dtype=torch.int32, device=device
        )
        coordinate_tensors = tuple(
            torch.tensor(values, dtype=torch.int32, device=device).contiguous()
            for values in (
                template.alpha_channels,
                template.alpha_heights,
                template.alpha_widths,
            )
        )
        self.alpha_map = alpha_map.contiguous()
        self.alpha_channel, self.alpha_height, self.alpha_width = coordinate_tensors

    def _validate_call(self, tensors: tuple[torch.Tensor, ...]) -> None:
        if any(
            tensor.device != self.device
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            for tensor in tensors
        ):
            self.fallback_count += 1
            raise ValueError("CIBC Triton call tensor differs")
        pointers = tuple(tensor.data_ptr() for tensor in tensors)
        if len(set(pointers)) != len(pointers):
            self.fallback_count += 1
            raise ValueError("CIBC Triton call aliases differ")

    def forward(
        self, incoming, lower, upper, alpha, incoming_bias, weight, operator_bias
    ):
        tensors = (incoming, lower, upper, alpha, incoming_bias, weight, operator_bias)
        self._validate_call(tensors)
        template = self.template
        output_a_count = incoming.numel()
        output_bias_count = template.domain_count * template.spec_count
        combined = torch.empty(
            output_a_count + output_bias_count,
            dtype=incoming.dtype,
            device=incoming.device,
        )
        grid = (triton.cdiv(output_a_count, self.config.block_m) + output_bias_count,)
        _cibc_forward_kernel_v2[grid](
            incoming,
            lower,
            upper,
            alpha,
            incoming_bias,
            weight,
            operator_bias,
            self.alpha_map,
            combined,
            output_a_count,
            output_bias_count,
            template.domain_count,
            template.spec_count,
            template.channels,
            template.height,
            template.width,
            len(template.alpha_coordinates),
            template.kernel_height,
            template.kernel_width,
            template.padding[0],
            template.padding[1],
            BLOCK_M=self.config.block_m,
            BLOCK_K=self.config.block_k,
            num_warps=self.config.num_warps,
        )
        self.forward_launch_count += 1
        return (
            combined[:output_a_count].view_as(incoming),
            combined[output_a_count:].view(template.domain_count, template.spec_count),
        )

    def backward(
        self,
        incoming,
        lower,
        upper,
        alpha,
        weight,
        operator_bias,
        output_a_gradient,
        output_bias_gradient,
    ):
        tensors = (
            incoming,
            lower,
            upper,
            alpha,
            weight,
            operator_bias,
            output_a_gradient,
            output_bias_gradient,
        )
        self._validate_call(tensors)
        template = self.template
        incoming_count = incoming.numel()
        alpha_count = alpha.numel()
        combined = torch.empty(
            incoming_count + alpha_count,
            dtype=incoming.dtype,
            device=incoming.device,
        )
        grid = (
            triton.cdiv(incoming_count, self.config.block_m)
            + triton.cdiv(alpha_count, self.config.block_m),
        )
        _cibc_backward_kernel_v2[grid](
            incoming,
            lower,
            upper,
            alpha,
            weight,
            operator_bias,
            output_a_gradient,
            output_bias_gradient,
            self.alpha_map,
            self.alpha_channel,
            self.alpha_height,
            self.alpha_width,
            combined,
            incoming_count,
            alpha_count,
            template.domain_count,
            template.spec_count,
            template.channels,
            template.height,
            template.width,
            len(template.alpha_coordinates),
            template.kernel_height,
            template.kernel_width,
            template.padding[0],
            template.padding[1],
            BLOCK_M=self.config.block_m,
            BLOCK_K=self.config.block_k,
            num_warps=self.config.num_warps,
        )
        self.backward_launch_count += 1
        return (
            combined[:incoming_count].view_as(incoming),
            combined[incoming_count:].view_as(alpha),
        )


class PreparedCIBCTritonTimingV2:
    """One admitted production P-anchor with compile/tune outside timed calls."""

    def __init__(
        self,
        capture: ProductionDifferentiableReferenceCaptureV1,
        *,
        config_ordinal: int,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CIBC Triton timing requires CUDA")
        if config_ordinal not in range(len(CIBC_TRITON_CONFIGS_V2)):
            raise ValueError("CIBC Triton config ordinal differs")
        lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
        major, minor = torch.cuda.get_device_capability()
        self.template = build_b4b2_sparse_conv_template_v1(
            lower_ir, capture, compute_capability=f"sm_{major}{minor}"
        )
        self.tensors = build_b4b2_sparse_conv_tensors_v1(
            capture, self.template, device=torch.device("cuda:0")
        )
        self.config = CIBC_TRITON_CONFIGS_V2[config_ordinal]
        self.executor = CIBCTritonExecutorV2(
            self.template, self.config, device=self.tensors.incoming_lower_a.device
        )
        # Force both specializations to compile before any timing starts.
        self.candidate_once()
        torch.cuda.synchronize()

    def baseline_once(self) -> SparseConvExecutionV1:
        return execute_sparse_conv_pytorch_baseline_v1(self.tensors, self.template)

    def candidate_once(self) -> SparseConvExecutionV1:
        tensors = self.tensors
        output_a, output_bias = _CIBCTritonFunctionV2.apply(
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.compressed_alpha,
            tensors.incoming_lower_bias,
            tensors.operator_weight,
            tensors.operator_bias,
            self.executor,
        )
        incoming_gradient, alpha_gradient = torch.autograd.grad(
            (output_a, output_bias),
            (tensors.incoming_lower_a, tensors.compressed_alpha),
            grad_outputs=(
                tensors.output_lower_a_gradient,
                tensors.output_bias_gradient,
            ),
        )
        return SparseConvExecutionV1(
            output_lower_a=output_a,
            output_bias=output_bias,
            compressed_alpha_gradient=alpha_gradient,
            incoming_lower_a_gradient=incoming_gradient,
        )


def triton_compilation_receipt_v2() -> dict[str, object]:
    """Freeze generated IR/assembly for the one specialization in this process."""

    receipt: dict[str, object] = {}
    for direction, jit_function in (
        ("forward", _cibc_forward_kernel_v2),
        ("backward", _cibc_backward_kernel_v2),
    ):
        device_cache = jit_function.device_caches.get(torch.cuda.current_device())
        if device_cache is None:
            raise RuntimeError("CIBC Triton compiled kernel is absent")
        compiled_entries = list(device_cache[0].values())
        if len(compiled_entries) != 1:
            raise RuntimeError("CIBC Triton specialization inventory differs")
        compiled = compiled_entries[0]
        assembly = compiled.asm
        hashes = {}
        for name in ("ttir", "ttgir", "llir", "ptx", "cubin"):
            content = assembly[name]
            if isinstance(content, str):
                content = content.encode("utf-8")
            hashes[name] = hashlib.sha256(content).hexdigest()
        receipt[direction] = {
            "kernel_name": compiled.name,
            "compiled_hash": compiled.hash,
            "source_hash": hashlib.sha256(jit_function.src.encode("utf-8")).hexdigest(),
            "assembly_hashes": hashes,
            "register_count": compiled.n_regs,
            "spill_count": compiled.n_spills,
        }
    return receipt


def execute_cibc_triton_v2(
    tensors: SparseConvTIRTensorsV1,
    template: DifferentiableLowerSparseConvTIRTemplateV1,
    *,
    config_ordinal: int,
) -> tuple[SparseConvExecutionV1, CIBCTritonExecutorV2]:
    """Execute one fused first-order call and expose its structural counters."""

    _validate_sparse_conv_tensors(tensors, template)
    if config_ordinal < 0 or config_ordinal >= len(CIBC_TRITON_CONFIGS_V2):
        raise ValueError("CIBC Triton config ordinal differs")
    executor = CIBCTritonExecutorV2(
        template,
        CIBC_TRITON_CONFIGS_V2[config_ordinal],
        device=tensors.incoming_lower_a.device,
    )
    output_a, output_bias = _CIBCTritonFunctionV2.apply(
        tensors.incoming_lower_a,
        tensors.preactivation_lower,
        tensors.preactivation_upper,
        tensors.compressed_alpha,
        tensors.incoming_lower_bias,
        tensors.operator_weight,
        tensors.operator_bias,
        executor,
    )
    incoming_gradient, alpha_gradient = torch.autograd.grad(
        (output_a, output_bias),
        (tensors.incoming_lower_a, tensors.compressed_alpha),
        grad_outputs=(
            tensors.output_lower_a_gradient,
            tensors.output_bias_gradient,
        ),
    )
    if (
        executor.forward_launch_count != 1
        or executor.backward_launch_count != 1
        or executor.fallback_count != 0
        or executor.eager_count != 0
    ):
        raise RuntimeError("CIBC Triton execution receipt differs")
    return (
        SparseConvExecutionV1(
            output_lower_a=output_a,
            output_bias=output_bias,
            compressed_alpha_gradient=alpha_gradient,
            incoming_lower_a_gradient=incoming_gradient,
        ),
        executor,
    )


__all__ = [
    "CIBC_TRITON_CONFIGS_V2",
    "CIBCTritonConfigV2",
    "CIBCTritonExecutorV2",
    "PreparedCIBCTritonTimingV2",
    "execute_cibc_triton_v2",
    "triton_compilation_receipt_v2",
]
