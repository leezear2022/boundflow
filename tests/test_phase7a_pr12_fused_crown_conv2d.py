"""CUDA correctness and mechanism tests for fused stride-one CROWN Conv2d."""

# mypy: disable-error-code=import-untyped
# pylint: disable=too-many-locals,not-callable,missing-function-docstring
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=use-implicit-booleaness-not-comparison,import-outside-toplevel

import pytest
import torch
from torch.nn import functional as torch_functional

from boundflow.backends.tvm.fused_crown_conv2d import (
    FusedCrownConv2dSignature,
    allocated_intermediate_buffers,
    build_fused_crown_conv2d_module,
    build_fused_crown_conv2d_primfunc,
    build_fused_crown_conv2d_relax_ir_module,
)
from boundflow.backends.tvm.relax_analysis import collect_relax_ir_stats


def _reference(
    signature: FusedCrownConv2dSignature, tensors: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coeff_u, coeff_l, alpha_u, beta_u, alpha_l, beta_l, weight = tensors[:7]
    bias = tensors[7] if signature.bias_present else None
    scaled_u = torch.where(
        coeff_u >= 0, coeff_u * alpha_u[:, None], coeff_u * alpha_l[:, None]
    )
    scaled_l = torch.where(
        coeff_l >= 0, coeff_l * alpha_l[:, None], coeff_l * alpha_u[:, None]
    )
    flat = (
        signature.domain_batch * signature.spec_batch,
        signature.output_channels,
        signature.output_height,
        signature.output_width,
    )
    output_padding = signature.output_padding()
    previous_u = torch_functional.conv_transpose2d(
        scaled_u.reshape(flat),
        weight,
        stride=signature.stride,
        padding=signature.padding,
        output_padding=output_padding,
    ).reshape(
        signature.domain_batch,
        signature.spec_batch,
        signature.input_channels,
        signature.input_height,
        signature.input_width,
    )
    previous_l = torch_functional.conv_transpose2d(
        scaled_l.reshape(flat),
        weight,
        stride=signature.stride,
        padding=signature.padding,
        output_padding=output_padding,
    ).reshape_as(previous_u)
    delta_u = torch.where(
        coeff_u >= 0, coeff_u * beta_u[:, None], coeff_u * beta_l[:, None]
    )
    delta_l = torch.where(
        coeff_l >= 0, coeff_l * beta_l[:, None], coeff_l * beta_u[:, None]
    )
    if bias is not None:
        bias_view = bias.view(1, 1, -1, 1, 1)
        delta_u = delta_u + scaled_u * bias_view
        delta_l = delta_l + scaled_l * bias_view
    return (
        previous_u,
        previous_l,
        delta_u.sum(dim=(2, 3, 4)),
        delta_l.sum(dim=(2, 3, 4)),
    )


def _inputs(signature: FusedCrownConv2dSignature, mode: str) -> list[torch.Tensor]:
    torch.manual_seed(1203 + signature.spec_batch)
    coeff_shape = (
        signature.domain_batch,
        signature.spec_batch,
        signature.output_channels,
        signature.output_height,
        signature.output_width,
    )
    relaxation_shape = (
        signature.domain_batch,
        signature.output_channels,
        signature.output_height,
        signature.output_width,
    )
    coeff_u, coeff_l = torch.randn(coeff_shape), torch.randn(coeff_shape)
    if mode == "positive":
        coeff_u, coeff_l = coeff_u.abs(), coeff_l.abs()
    elif mode == "negative":
        coeff_u, coeff_l = -coeff_u.abs(), -coeff_l.abs()
    elif mode == "zero":
        coeff_u.zero_()
        coeff_l.zero_()
    else:
        coeff_u.flatten()[:3] = torch.tensor([0.0, 1.0, -1.0])
        coeff_l.flatten()[:3] = torch.tensor([0.0, -1.0, 1.0])
    tensors = [
        coeff_u,
        coeff_l,
        torch.rand(relaxation_shape),
        torch.randn(relaxation_shape),
        torch.rand(relaxation_shape),
        torch.randn(relaxation_shape),
        torch.randn(
            signature.output_channels,
            signature.input_channels,
            signature.kernel_height,
            signature.kernel_width,
        ),
    ]
    if signature.bias_present:
        tensors.append(torch.randn(signature.output_channels))
    return tensors


def _signature(
    domain: int,
    spec: int,
    channels: tuple[int, int],
    spatial: int,
    kernel: int,
    padding: int,
    bias_present: bool,
) -> FusedCrownConv2dSignature:
    output = spatial + 2 * padding - kernel + 1
    return FusedCrownConv2dSignature(
        domain,
        spec,
        channels[0],
        spatial,
        spatial,
        channels[1],
        output,
        output,
        kernel,
        kernel,
        (1, 1),
        (padding, padding),
        bias_present=bias_present,
    )


def _stride_two_signature(
    domain: int,
    spec: int,
    channels: tuple[int, int],
    input_spatial: int,
    kernel: int,
    padding: int,
    bias_present: bool,
) -> FusedCrownConv2dSignature:
    output = (input_spatial + 2 * padding - kernel) // 2 + 1
    return FusedCrownConv2dSignature(
        domain,
        spec,
        channels[0],
        input_spatial,
        input_spatial,
        channels[1],
        output,
        output,
        kernel,
        kernel,
        (2, 2),
        (padding, padding),
        bias_present=bias_present,
    )


def test_conv_primfunc_has_no_scaled_a_or_im2col_allocation() -> None:
    signature = _signature(2, 3, (5, 4), 7, 3, 1, True)

    assert allocated_intermediate_buffers(signature, scheduled=False) == ()
    assert allocated_intermediate_buffers(signature, scheduled=True) == ()
    text = str(build_fused_crown_conv2d_primfunc(signature))
    assert "A_scaled" not in text
    assert "im2col" not in text


def test_conv_has_one_thin_relax_call_tir_wrapper() -> None:
    module = build_fused_crown_conv2d_relax_ir_module(
        _signature(1, 1, (3, 4), 4, 1, 0, False)
    )

    stats = collect_relax_ir_stats(module)
    assert stats["relax_funcs"] == 1
    assert stats["tir_funcs"] == 1
    assert stats["call_tir"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("signature", "mode"),
    [
        (_signature(1, 1, (3, 4), 4, 1, 0, False), "positive"),
        (_signature(2, 3, (5, 4), 7, 3, 1, True), "negative"),
        (_signature(8, 9, (4, 8), 8, 3, 0, True), "zero"),
        (_signature(1, 32, (3, 5), 14, 1, 0, True), "mixed"),
    ],
)
def test_stride_one_fused_conv_cuda_matches_dense_reference(
    signature: FusedCrownConv2dSignature, mode: str
) -> None:
    import tvm

    tensors = _inputs(signature, mode)
    expected = _reference(signature, tensors)
    device = tvm.cuda(0)
    inputs = [tvm.runtime.tensor(tensor.numpy(), device=device) for tensor in tensors]
    previous_shape = expected[0].shape
    outputs = [
        tvm.runtime.empty(previous_shape, "float32", device),
        tvm.runtime.empty(previous_shape, "float32", device),
        tvm.runtime.empty(
            (signature.domain_batch, signature.spec_batch), "float32", device
        ),
        tvm.runtime.empty(
            (signature.domain_batch, signature.spec_batch), "float32", device
        ),
    ]
    compiled = build_fused_crown_conv2d_module(signature)
    assert compiled is build_fused_crown_conv2d_module(signature)

    compiled(*inputs, *outputs)
    device.sync()

    for actual, reference in zip(outputs, expected):
        actual_tensor = torch.from_numpy(actual.numpy())
        assert torch.isfinite(actual_tensor).all()
        torch.testing.assert_close(actual_tensor, reference, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("signature", "mode", "output_padding"),
    [
        (_stride_two_signature(1, 1, (3, 4), 7, 3, 1, True), "mixed", 0),
        (_stride_two_signature(2, 3, (5, 4), 8, 3, 1, False), "positive", 1),
        (_stride_two_signature(1, 9, (3, 5), 7, 1, 0, True), "negative", 0),
        (_stride_two_signature(1, 32, (4, 8), 8, 1, 0, True), "zero", 1),
    ],
)
def test_stride_two_residual_fused_conv_matches_dense_reference(
    signature: FusedCrownConv2dSignature, mode: str, output_padding: int
) -> None:
    import tvm

    assert signature.output_padding() == (output_padding, output_padding)
    tensors = _inputs(signature, mode)
    expected = _reference(signature, tensors)
    device = tvm.cuda(0)
    inputs = [tvm.runtime.tensor(tensor.numpy(), device=device) for tensor in tensors]
    outputs = [
        tvm.runtime.empty(expected[0].shape, "float32", device),
        tvm.runtime.empty(expected[1].shape, "float32", device),
        tvm.runtime.empty(
            (signature.domain_batch, signature.spec_batch), "float32", device
        ),
        tvm.runtime.empty(
            (signature.domain_batch, signature.spec_batch), "float32", device
        ),
    ]

    build_fused_crown_conv2d_module(signature)(*inputs, *outputs)
    device.sync()

    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(
            torch.from_numpy(actual.numpy()), reference, rtol=2e-4, atol=2e-4
        )
