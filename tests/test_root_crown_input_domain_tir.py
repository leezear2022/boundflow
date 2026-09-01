"""Contracts for streaming root Conv/concretization TIR and its VJP."""

# pylint: disable=missing-function-docstring,too-many-locals
# pylint: disable=consider-using-from-import,not-callable

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as functional

from boundflow.backends.tvm.root_crown_input_domain import (
    RootCrownInputDomainTemplateV1,
)
from boundflow.runtime.root_crown_input_domain_tir import (
    RootCrownInputDomainTensorsV1,
    RootCrownInputDomainTIRExecutorV1,
    validate_root_crown_input_domain_tensors_v1,
)


def _coordinates() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (ordinal // 256, ordinal // 16 % 16, ordinal % 16) for ordinal in range(164)
    )


def _template() -> RootCrownInputDomainTemplateV1:
    return RootCrownInputDomainTemplateV1(
        spec_count=3,
        domain_count=1,
        output_channels=8,
        output_height=16,
        output_width=16,
        input_channels=3,
        input_height=32,
        input_width=32,
        alpha_coordinates=_coordinates(),
        compute_capability="sm_89",
    )


def test_root_crown_input_domain_template_is_static_and_deterministic() -> None:
    template = _template()
    template.validate()
    assert template.stable_hash() == template.stable_hash()
    assert template.incoming_shape == (3, 1, 8, 16, 16)
    assert template.coefficient_shape == (3, 1, 3, 32, 32)
    assert template.to_dict()["dense_input_coefficient_externalized"] is False


@pytest.mark.parametrize(
    "changed",
    (
        {"spec_count": 4},
        {"domain_count": 2},
        {"output_channels": 16},
        {"input_height": 31},
        {"alpha_coordinates": _coordinates()[:-1]},
        {"alpha_coordinates": ((0, 0, 0),) * 164},
        {"compute_capability": "cuda"},
        {"thread_extent": 48},
        {"stride": (1, 1)},
        {"target": "llvm"},
    ),
)
def test_root_crown_input_domain_template_rejects_other_abi(
    changed: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="template differs"):
        replace(_template(), **changed).validate()  # type: ignore[arg-type]


def test_root_crown_input_domain_runtime_rejects_cpu_tensors() -> None:
    template = _template()
    tensors = RootCrownInputDomainTensorsV1(
        torch.zeros(template.incoming_shape),
        torch.full(template.bound_shape, -1.0),
        torch.full(template.bound_shape, 1.0),
        torch.full((2, 3, 1, 164), 0.5),
        torch.zeros(template.weight_shape),
        torch.zeros((8,)),
        torch.zeros(template.input_shape),
        torch.ones(template.input_shape),
    )
    with pytest.raises(ValueError, match="runtime tensor differs"):
        validate_root_crown_input_domain_tensors_v1(tensors, template)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_root_crown_input_domain_streaming_tir_matches_pytorch_vjp() -> None:
    torch.manual_seed(31)
    template = _template()
    incoming = torch.randn(template.incoming_shape, device="cuda", requires_grad=True)
    lower = -torch.rand(template.bound_shape, device="cuda")
    upper = torch.rand(template.bound_shape, device="cuda")
    raw_alpha = torch.rand((2, 3, 1, 164), device="cuda", requires_grad=True)
    weight = torch.randn(template.weight_shape, device="cuda") * 0.1
    bias = torch.randn((8,), device="cuda") * 0.1
    center = torch.randn(template.input_shape, device="cuda") * 0.1
    radius = torch.rand(template.input_shape, device="cuda") * 0.01
    tensors = RootCrownInputDomainTensorsV1(
        incoming,
        lower,
        upper,
        raw_alpha,
        weight,
        bias,
        center,
        radius,
    )
    executor = RootCrownInputDomainTIRExecutorV1(template)
    executor.prepare()
    candidate_concrete, candidate_bias = executor.forward(tensors)

    alpha = torch.zeros(template.incoming_shape, device="cuda")
    coordinate = torch.tensor(_coordinates(), device="cuda").transpose(0, 1)
    alpha[:, 0, coordinate[0], coordinate[1], coordinate[2]] = raw_alpha[0, :, 0]
    lower_broadcast = lower.unsqueeze(0)
    upper_broadcast = upper.unsqueeze(0)
    upper_slope = torch.where(
        lower_broadcast >= 0,
        torch.ones_like(lower_broadcast),
        torch.where(
            upper_broadcast <= 0,
            torch.zeros_like(lower_broadcast),
            upper_broadcast
            / (upper_broadcast - lower_broadcast).clamp_min(
                torch.finfo(torch.float32).eps
            ),
        ),
    )
    ambiguous = (lower_broadcast < 0) & (upper_broadcast > 0)
    lower_slope = torch.where(
        ambiguous, alpha.clamp(0, 1), (lower_broadcast >= 0).to(lower.dtype)
    )
    slope = torch.where(incoming >= 0, lower_slope, upper_slope)
    intercept = torch.where(
        (incoming < 0) & ambiguous,
        -lower_broadcast * upper_slope,
        torch.zeros_like(incoming),
    )
    transformed = incoming * slope
    coefficient = functional.conv_transpose2d(
        transformed.reshape(3, 8, 16, 16),
        weight,
        stride=2,
        padding=1,
        output_padding=1,
    ).reshape(template.coefficient_shape)
    reference_concrete = (
        (coefficient * center.unsqueeze(0) - coefficient.abs() * radius.unsqueeze(0))
        .sum(dim=(2, 3, 4))
        .transpose(0, 1)
    )
    reference_bias = (
        incoming * intercept + transformed * bias.view(1, 1, 8, 1, 1)
    ).sum(dim=(2, 3, 4))
    concrete_gradient = torch.randn_like(reference_concrete)
    bias_gradient = torch.randn_like(reference_bias)
    reference_gradients = torch.autograd.grad(
        (reference_concrete, reference_bias),
        (incoming, raw_alpha),
        grad_outputs=(concrete_gradient, bias_gradient),
    )
    candidate_gradients = executor.backward(tensors, concrete_gradient, bias_gradient)

    torch.testing.assert_close(
        candidate_concrete, reference_concrete, atol=3e-5, rtol=1e-5
    )
    torch.testing.assert_close(candidate_bias, reference_bias, atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(
        candidate_gradients[0], reference_gradients[0], atol=3e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        candidate_gradients[1], reference_gradients[1], atol=3e-5, rtol=1e-5
    )
    assert (
        template.coefficient_shape
        not in dict(executor.compiled.workspace_inventory).values()
    )
    assert executor.forward_launch_count == 1
    assert executor.backward_launch_count == 1
    assert executor.pointer_count == executor.pointer_exact_count
