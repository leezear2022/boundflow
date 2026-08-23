"""Correctness and fail-closed tests for B4-B3 dense-alpha exact call."""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=not-callable,missing-function-docstring

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as torch_functional

from boundflow.backends.tvm.cibc_dense_exact_conv import (
    CIBC_DENSE_BACKWARD_SYMBOL,
    CIBC_DENSE_FORWARD_SYMBOL,
    compile_cibc_dense_exact_conv_tir_v3,
)
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.fsg4_b4b3_cibc_dense_tir import (
    execute_cibc_dense_exact_tir_v3,
)
from boundflow.runtime.fsg4_b4b3_cibc_exact_call import (
    B4B3CIBCExactCallReceiptV1,
)

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1/run_00.pt"


def _reference_capture():
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    return production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][1]
    )


def _dense_reference(
    incoming,
    lower,
    upper,
    alpha,
    incoming_bias,
    weight,
    operator_bias,
):
    zero = torch.zeros((), dtype=lower.dtype, device=lower.device)
    positive = lower >= zero
    negative = upper <= zero
    ambiguous = (~positive) & (~negative)
    upper_slope = torch.where(
        positive,
        torch.ones_like(lower),
        torch.where(
            negative,
            torch.zeros_like(lower),
            upper / (upper - lower).clamp_min(torch.finfo(lower.dtype).eps),
        ),
    )
    lower_slope = torch.where(
        ambiguous,
        alpha.clamp(0.0, 1.0),
        torch.where(positive, torch.ones_like(lower), torch.zeros_like(lower)),
    )
    selected_slope = torch.where(
        incoming >= zero, lower_slope.unsqueeze(1), upper_slope.unsqueeze(1)
    )
    upper_intercept = torch.where(
        ambiguous, -lower * upper_slope, torch.zeros_like(lower)
    )
    selected_intercept = torch.where(
        incoming >= zero, torch.zeros_like(incoming), upper_intercept.unsqueeze(1)
    )
    relu_lower_a = incoming * selected_slope
    output_a = torch_functional.conv_transpose2d(
        relu_lower_a.reshape(6, 16, 8, 8),
        weight,
        bias=None,
        stride=(1, 1),
        padding=(1, 1),
    ).reshape_as(incoming)
    output_bias = incoming_bias + (
        incoming * selected_intercept
        + relu_lower_a * operator_bias.reshape(1, 1, -1, 1, 1)
    ).flatten(2).sum(2)
    return output_a, output_bias


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dense_exact_tir_matches_public_pytorch_forward_and_vjp() -> None:
    capture = _reference_capture()
    values = capture.base.value_map
    device = torch.device("cuda:0")

    def tensor(name: str, *, requires_grad: bool = False) -> torch.Tensor:
        value = values[name].value.to(device).contiguous().clone()
        value.requires_grad_(requires_grad)
        return value

    incoming = tensor("incoming_lower_a", requires_grad=True)
    lower = tensor("preactivation_lower")
    upper = tensor("preactivation_upper")
    alpha = tensor("native_alpha", requires_grad=True)
    incoming_bias = (
        capture.incoming_lower_bias.value.to(device).contiguous().requires_grad_(True)
    )
    weight = tensor("operator_weight")
    assert capture.operator_bias is not None
    operator_bias = capture.operator_bias.value.to(device).contiguous()
    output_a_gradient = capture.output_lower_a_gradient.value.to(device).contiguous()
    scalar_bias_gradient = torch.tensor(-1.0, device=device)
    output_bias_gradient = scalar_bias_gradient.expand(6, 1)
    major, minor = torch.cuda.get_device_capability()
    compiled = compile_cibc_dense_exact_conv_tir_v3(
        compute_capability=f"sm_{major}{minor}"
    )
    candidate_a, candidate_bias, executor = execute_cibc_dense_exact_tir_v3(
        incoming_lower_a=incoming,
        preactivation_lower=lower,
        preactivation_upper=upper,
        native_alpha=alpha,
        incoming_lower_bias=incoming_bias,
        operator_weight=weight,
        operator_bias=operator_bias,
        compiled=compiled,
    )
    candidate_gradients = torch.autograd.grad(
        (candidate_a, candidate_bias),
        (incoming, alpha, incoming_bias),
        grad_outputs=(output_a_gradient, output_bias_gradient),
    )

    reference_incoming = incoming.detach().clone().requires_grad_(True)
    reference_alpha = alpha.detach().clone().requires_grad_(True)
    reference_bias = incoming_bias.detach().clone().requires_grad_(True)
    reference_a, reference_output_bias = _dense_reference(
        reference_incoming,
        lower,
        upper,
        reference_alpha,
        reference_bias,
        weight,
        operator_bias,
    )
    reference_gradients = torch.autograd.grad(
        (reference_a, reference_output_bias),
        (reference_incoming, reference_alpha, reference_bias),
        grad_outputs=(output_a_gradient, output_bias_gradient),
    )
    for reference, candidate in zip(
        (reference_a, reference_output_bias, *reference_gradients),
        (candidate_a, candidate_bias, *candidate_gradients),
    ):
        assert torch.allclose(reference, candidate, atol=2.0e-4, rtol=2.0e-4)
        assert torch.equal(torch.sign(reference), torch.sign(candidate))
    assert executor.forward_launch_count == 1
    assert executor.backward_launch_count == 1
    assert executor.fallback_count == 0
    assert executor.eager_count == 0
    assert executor.adjoint_materialization_count == 0
    assert compiled.exported_symbols == (
        CIBC_DENSE_FORWARD_SYMBOL,
        CIBC_DENSE_BACKWARD_SYMBOL,
    )


def test_exact_call_receipt_rejects_missing_native_value_bridge() -> None:
    receipt = B4B3CIBCExactCallReceiptV1(
        evaluation_count=10,
        update_count=9,
        provider_activation_count=10,
        forward_launch_count=10,
        backward_launch_count=9,
        unsupported_semantic_anchor_count=1,
        correctness_capture_enabled=True,
        native_value_bridge_count=9,
        adjoint_materialization_count=0,
        fallback_count=0,
        eager_count=0,
        module_hash="0" * 64,
        exact_call=True,
    )
    with pytest.raises(ValueError, match="receipt differs"):
        receipt.validate()
