"""Tests for CIBC horizontal IBP Conv2d TIR."""

# pylint: disable=missing-function-docstring,too-many-locals,not-callable

import pytest
import torch
import torch.nn.functional as F

from boundflow.backends.tvm.cibc_ibp_conv import (
    CIBCIBPConvSignatureV1,
    compile_cibc_ibp_conv_tir_v1,
)
from boundflow.runtime.cibc_ibp_conv import CIBCIBPConvExecutorV1


def test_cibc_ibp_conv_signature_derives_output_shape() -> None:
    signature = CIBCIBPConvSignatureV1(
        input_shape=(6, 16, 8, 8),
        weight_shape=(16, 16, 3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
    )
    signature.validate()
    assert signature.output_shape == (6, 16, 8, 8)


def test_cibc_ibp_conv_signature_rejects_groups() -> None:
    signature = CIBCIBPConvSignatureV1(
        input_shape=(6, 16, 8, 8),
        weight_shape=(16, 8, 3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=2,
    )
    with pytest.raises(ValueError, match="signature differs"):
        signature.validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cibc_ibp_conv_horizontal_matches_four_conv_reference() -> None:
    signature = CIBCIBPConvSignatureV1(
        input_shape=(2, 2, 4, 4),
        weight_shape=(3, 2, 3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
    )
    generator = torch.Generator(device="cuda").manual_seed(20260824)
    lower = torch.randn(signature.input_shape, device="cuda", generator=generator)
    upper = lower + torch.rand(
        signature.input_shape, device="cuda", generator=generator
    )
    weight = torch.randn(signature.weight_shape, device="cuda", generator=generator)
    bias = torch.randn(3, device="cuda", generator=generator)
    major, minor = torch.cuda.get_device_capability()
    compiled = compile_cibc_ibp_conv_tir_v1(
        signature,
        threads_per_block=128,
        compute_capability=f"sm_{major}{minor}",
    )
    executor = CIBCIBPConvExecutorV1(lower, upper, weight, bias, compiled=compiled)
    observed_lower, observed_upper = executor.run()
    positive = weight.clamp_min(0)
    negative = weight.clamp_max(0)
    expected_lower = (
        F.conv2d(lower, positive, padding=1)
        + F.conv2d(upper, negative, padding=1)
        + bias.view(1, -1, 1, 1)
    )
    expected_upper = (
        F.conv2d(upper, positive, padding=1)
        + F.conv2d(lower, negative, padding=1)
        + bias.view(1, -1, 1, 1)
    )
    assert torch.allclose(observed_lower, expected_lower, atol=2e-4, rtol=2e-4)
    assert torch.allclose(observed_upper, expected_upper, atol=2e-4, rtol=2e-4)
    assert executor.launch_count == 1
