"""MR5 generalized C0/C1/C2 Conv TIR correctness and rejection tests."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

import pytest
import torch

from boundflow.backends.tvm.mr5_generalized_crown_conv import (
    MR5GeneralizedConvSignatureV1,
    build_mr5_generalized_conv_modules,
)
from boundflow.runtime.mr5_generalized_crown_conv import (
    MR5GeneralizedConvModuleCacheV1,
    MR5GeneralizedConvTensorsV1,
    execute_mr5_generalized_conv_v1,
    run_mr5_pytorch_oracle_v1,
    validate_mr5_generalized_conv_tensors,
)


def _signature(site: str) -> MR5GeneralizedConvSignatureV1:
    if site == "C0":
        return MR5GeneralizedConvSignatureV1(
            site_id=site,
            input_channels=3,
            output_channels=8,
            input_height=32,
            input_width=32,
            output_height=16,
            output_width=16,
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1),
        )
    if site == "C1":
        return MR5GeneralizedConvSignatureV1(
            site_id=site,
            input_channels=8,
            output_channels=16,
            input_height=16,
            input_width=16,
            output_height=8,
            output_width=8,
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1),
        )
    if site == "C2":
        return MR5GeneralizedConvSignatureV1(
            site_id=site,
            input_channels=16,
            output_channels=16,
            input_height=8,
            input_width=8,
            output_height=8,
            output_width=8,
            stride=(1, 1),
            padding=(1, 1),
            output_padding=(0, 0),
        )
    raise ValueError(f"unknown MR5 site: {site}")


def _tensors(
    signature: MR5GeneralizedConvSignatureV1,
    *,
    seed: int,
) -> MR5GeneralizedConvTensorsV1:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    center = torch.randn(signature.relaxation_shape, device="cuda", generator=generator)
    radius = (
        torch.rand(signature.relaxation_shape, device="cuda", generator=generator)
        + 0.25
    )
    incoming = torch.randn(
        signature.incoming_shape, device="cuda", generator=generator
    ).requires_grad_(True)
    alpha = torch.rand(
        signature.relaxation_shape, device="cuda", generator=generator
    ).requires_grad_(True)
    return MR5GeneralizedConvTensorsV1(
        incoming=incoming,
        lower=(center - radius).contiguous(),
        upper=(center + radius).contiguous(),
        alpha=alpha,
        incoming_bias=torch.randn(
            (signature.domain_count, signature.spec_count),
            device="cuda",
            generator=generator,
        ),
        weight=torch.randn(signature.weight_shape, device="cuda", generator=generator),
        operator_bias=torch.randn(
            (signature.output_channels,), device="cuda", generator=generator
        ),
    )


@pytest.mark.parametrize("site", ("C0", "C1", "C2"))
def test_mr5_generalized_modules_have_distinct_shape_keyed_identity(site: str) -> None:
    signature = _signature(site)
    unscheduled, scheduled, inventory = build_mr5_generalized_conv_modules(signature)
    assert len(unscheduled.functions) == 2
    assert len(scheduled.functions) == 2
    assert inventory
    assert len(signature.stable_hash()) == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MR5 requires CUDA")
@pytest.mark.parametrize("site", ("C0", "C1", "C2"))
def test_mr5_generalized_tir_matches_independent_pytorch_forward_and_vjp(
    site: str,
) -> None:
    signature = _signature(site)
    cache = MR5GeneralizedConvModuleCacheV1()
    candidate_tensors = _tensors(signature, seed=800 + ord(site[-1]))
    reference_tensors = MR5GeneralizedConvTensorsV1(
        incoming=candidate_tensors.incoming.detach().clone().requires_grad_(True),
        lower=candidate_tensors.lower.detach().clone(),
        upper=candidate_tensors.upper.detach().clone(),
        alpha=candidate_tensors.alpha.detach().clone().requires_grad_(True),
        incoming_bias=candidate_tensors.incoming_bias.detach().clone(),
        weight=candidate_tensors.weight.detach().clone(),
        operator_bias=candidate_tensors.operator_bias.detach().clone(),
    )
    candidate_a, candidate_bias, executor = execute_mr5_generalized_conv_v1(
        signature, candidate_tensors, cache
    )
    reference_a, reference_bias = run_mr5_pytorch_oracle_v1(
        signature, reference_tensors
    )
    assert tuple(candidate_a.shape) == signature.result_shape
    assert torch.allclose(candidate_a, reference_a, atol=2e-4, rtol=2e-4)
    assert torch.allclose(candidate_bias, reference_bias, atol=2e-4, rtol=2e-4)
    generator = torch.Generator(device="cuda").manual_seed(900 + ord(site[-1]))
    a_adjoint = torch.randn(signature.result_shape, device="cuda", generator=generator)
    bias_adjoint = torch.randn(
        (signature.domain_count, signature.spec_count),
        device="cuda",
        generator=generator,
    )
    torch.autograd.backward((candidate_a, candidate_bias), (a_adjoint, bias_adjoint))
    torch.autograd.backward((reference_a, reference_bias), (a_adjoint, bias_adjoint))
    assert candidate_tensors.incoming.grad is not None
    assert candidate_tensors.alpha.grad is not None
    assert reference_tensors.incoming.grad is not None
    assert reference_tensors.alpha.grad is not None
    assert torch.allclose(
        candidate_tensors.incoming.grad,
        reference_tensors.incoming.grad,
        atol=2e-4,
        rtol=2e-4,
    )
    assert torch.allclose(
        candidate_tensors.alpha.grad,
        reference_tensors.alpha.grad,
        atol=2e-4,
        rtol=2e-4,
    )
    assert executor.forward_launch_count == 1
    assert executor.backward_launch_count == 1
    assert executor.fallback_count == 0
    assert executor.eager_count == 0
    assert executor.forward_observation is not None
    assert executor.backward_observation is not None
    assert (
        executor.forward_observation.pointer_exact_count
        == executor.forward_observation.pointer_count
    )
    assert executor.module_receipt.signature_hash == signature.stable_hash()


def test_mr5_generalized_signature_rejects_stride_alias() -> None:
    signature = MR5GeneralizedConvSignatureV1(
        site_id="C0",
        input_channels=3,
        output_channels=8,
        input_height=32,
        input_width=32,
        output_height=16,
        output_width=16,
        stride=(1, 1),
        padding=(1, 1),
        output_padding=(1, 1),
    )
    with pytest.raises(ValueError, match="signature differs"):
        signature.validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MR5 requires CUDA")
def test_mr5_generalized_tensor_validation_rejects_nonfinite() -> None:
    signature = _signature("C1")
    tensors = _tensors(signature, seed=811)
    tensors.lower.reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="tensor differs: lower"):
        validate_mr5_generalized_conv_tensors(signature, tensors)
