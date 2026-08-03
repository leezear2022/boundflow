"""Frozen dense semantics and shape contracts for fused CROWN Conv2d."""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=missing-function-docstring,not-callable
# pylint: disable=use-implicit-booleaness-not-comparison

import pytest
import torch
from torch.nn import functional as torch_functional

from boundflow.backends.tvm.fused_crown_conv2d import FusedCrownConv2dSignature
from boundflow.planner.execution_candidate import (
    ExecutionContext,
    OperatorFamily,
    capability_rejections,
    fused_tir_conv_v1_capability,
)
from boundflow.planner.materialization import BoundMethod, OptimizationStage


def dense_conv2d_reference(
    signature: FusedCrownConv2dSignature,
    coeff_u: torch.Tensor,
    coeff_l: torch.Tensor,
    alpha_u: torch.Tensor,
    beta_u: torch.Tensor,
    alpha_l: torch.Tensor,
    beta_l: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent PyTorch oracle using deterministic ConvTranspose2d."""

    signature.validate()
    scaled_u = torch.where(
        coeff_u >= 0, coeff_u * alpha_u[:, None], coeff_u * alpha_l[:, None]
    )
    scaled_l = torch.where(
        coeff_l >= 0, coeff_l * alpha_l[:, None], coeff_l * alpha_u[:, None]
    )
    flat_shape = (
        signature.domain_batch * signature.spec_batch,
        signature.output_channels,
        signature.output_height,
        signature.output_width,
    )
    previous_u = torch_functional.conv_transpose2d(
        scaled_u.reshape(flat_shape),
        weight,
        stride=signature.stride,
        padding=signature.padding,
        output_padding=signature.output_padding(),
        dilation=signature.dilation,
        groups=signature.groups,
    ).reshape(
        signature.domain_batch,
        signature.spec_batch,
        signature.input_channels,
        signature.input_height,
        signature.input_width,
    )
    previous_l = torch_functional.conv_transpose2d(
        scaled_l.reshape(flat_shape),
        weight,
        stride=signature.stride,
        padding=signature.padding,
        output_padding=signature.output_padding(),
        dilation=signature.dilation,
        groups=signature.groups,
    ).reshape_as(previous_u)
    relu_bias_u = torch.where(
        coeff_u >= 0, coeff_u * beta_u[:, None], coeff_u * beta_l[:, None]
    )
    relu_bias_l = torch.where(
        coeff_l >= 0, coeff_l * beta_l[:, None], coeff_l * beta_u[:, None]
    )
    if bias is not None:
        bias_view = bias.view(1, 1, -1, 1, 1)
        relu_bias_u = relu_bias_u + scaled_u * bias_view
        relu_bias_l = relu_bias_l + scaled_l * bias_view
    return (
        previous_u,
        previous_l,
        relu_bias_u.sum(dim=(2, 3, 4)),
        relu_bias_l.sum(dim=(2, 3, 4)),
    )


@pytest.mark.parametrize(
    ("input_size", "kernel", "stride", "padding", "output_size", "output_pad"),
    [
        (7, 3, 1, 1, 7, 0),
        (8, 1, 1, 0, 8, 0),
        (7, 3, 2, 1, 4, 0),
        (8, 3, 2, 1, 4, 1),
        (7, 1, 2, 0, 4, 0),
        (8, 1, 2, 0, 4, 1),
    ],
)
def test_conv_signature_recovers_explicit_input_shape(
    input_size: int,
    kernel: int,
    stride: int,
    padding: int,
    output_size: int,
    output_pad: int,
) -> None:
    signature = FusedCrownConv2dSignature(
        1,
        1,
        3,
        input_size,
        input_size,
        5,
        output_size,
        output_size,
        kernel,
        kernel,
        (stride, stride),
        (padding, padding),
    )

    signature.validate()
    assert signature.output_padding() == (output_pad, output_pad)


def test_conv_signature_requires_original_input_shape_to_match() -> None:
    signature = FusedCrownConv2dSignature(1, 1, 3, 8, 8, 5, 5, 5, 3, 3, (2, 2), (1, 1))

    with pytest.raises(ValueError, match="output shape"):
        signature.validate()


@pytest.mark.parametrize("bias_present", [False, True])
def test_dense_conv_reference_freezes_four_output_semantics(
    bias_present: bool,
) -> None:
    torch.manual_seed(1202)
    signature = FusedCrownConv2dSignature(
        1,
        3,
        2,
        4,
        4,
        3,
        4,
        4,
        3,
        3,
        (1, 1),
        (1, 1),
        bias_present=bias_present,
    )
    coeff_u = torch.randn(1, 3, 3, 4, 4)
    coeff_l = torch.randn(1, 3, 3, 4, 4)
    coeff_u[0, 0, 0, 0, :3] = torch.tensor([0.0, 1.0, -1.0])
    coeff_l[0, 0, 0, 0, :3] = torch.tensor([0.0, -1.0, 1.0])
    alpha_u, alpha_l = torch.rand(1, 3, 4, 4), torch.rand(1, 3, 4, 4)
    beta_u, beta_l = torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)
    weight = torch.randn(3, 2, 3, 3)
    bias = torch.randn(3) if bias_present else None

    outputs = dense_conv2d_reference(
        signature,
        coeff_u,
        coeff_l,
        alpha_u,
        beta_u,
        alpha_l,
        beta_l,
        weight,
        bias,
    )

    assert outputs[0].shape == outputs[1].shape == (1, 3, 2, 4, 4)
    assert outputs[2].shape == outputs[3].shape == (1, 3)
    assert all(torch.isfinite(output).all() for output in outputs)


@pytest.mark.parametrize(
    "updates",
    [
        {"groups": 2},
        {"dilation": (2, 2)},
        {"dtype": "float16"},
        {"coefficient_layout": "SDCOHW"},
        {"target": "llvm"},
    ],
)
def test_conv_signature_rejects_unsupported_attributes(
    updates: dict[str, object],
) -> None:
    values = {
        "domain_batch": 1,
        "spec_batch": 1,
        "input_channels": 4,
        "input_height": 7,
        "input_width": 7,
        "output_channels": 5,
        "output_height": 7,
        "output_width": 7,
        "kernel_height": 3,
        "kernel_width": 3,
        "stride": (1, 1),
        "padding": (1, 1),
        **updates,
    }
    with pytest.raises(NotImplementedError):
        FusedCrownConv2dSignature(**values).validate()  # type: ignore[arg-type]


def test_conv_capability_is_plain_crown_only() -> None:
    context = ExecutionContext(
        bound_method=BoundMethod.CROWN,
        requires_grad=False,
        optimization_stage=OptimizationStage.FINAL_BOUND,
        alpha_enabled=False,
        beta_enabled=False,
        split_state_present=False,
        operator_family=OperatorFamily.CONV2D,
        device="cuda",
        dtype="float32",
        layout="nchw",
        static_shape=True,
    )

    assert capability_rejections(context, fused_tir_conv_v1_capability()) == ()
