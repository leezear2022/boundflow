"""B4-B2 B2-0 first-class IR and differentiable identity TIR gates."""

# pylint: disable=missing-function-docstring,protected-access,import-error,duplicate-code

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.ir.differentiable_lower_tir import (
    DifferentiableLowerTIRInstanceV1,
    DifferentiableLowerTIRLaunchReceiptV1,
    DifferentiableLowerTIRModuleReceiptV1,
    DifferentiableLowerTIRScheduleV1,
    DifferentiableLowerTIRTemplateV1,
)
from boundflow.runtime.fsg4_b4b1_pytorch_reference import (
    build_b4b1_differentiable_lower_instance_v1,
    build_b4b1_differentiable_lower_ir_v1,
)
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime import fsg4_b4b2_identity_tir as identity

CAPTURE = Path("artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1/run_00.pt")


def _lower_identity(numel: int = 257):
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    capture = production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][0]
    )
    lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
    lower_instance = build_b4b1_differentiable_lower_instance_v1(capture, lower_ir)
    template = identity.build_b4b2_identity_template_v1(
        lower_ir, tensor_numel=numel, compute_capability="sm_89"
    )
    schedule = identity.build_b4b2_identity_schedule_v1(template, thread_extent=256)
    return lower_ir, lower_instance, template, schedule


def test_b4b2_identity_first_class_ir_round_trips_and_hashes_stably() -> None:
    _lower_ir, _lower_instance, template, schedule = _lower_identity()
    parsed_template = DifferentiableLowerTIRTemplateV1.from_dict(template.to_dict())
    parsed_schedule = DifferentiableLowerTIRScheduleV1.from_dict(
        schedule.to_dict(template), template
    )

    assert parsed_template == template
    assert parsed_template.stable_hash() == template.stable_hash()
    assert parsed_schedule == schedule
    assert parsed_schedule.stable_hash(template) == schedule.stable_hash(template)
    assert template.enabled_by_default is False
    assert template.performance_admitted is False
    assert template.gradient_targets == ("input",)


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("template", {"abi": "dense-semantic-v1"}),
        ("template", {"enabled_by_default": True}),
        ("template", {"performance_admitted": True}),
        ("schedule", {"thread_extent": 32}),
        ("schedule", {"workspace_bytes": 4}),
        ("schedule", {"candidate_ordinal": 1}),
    ],
)
def test_b4b2_identity_ir_rejects_scope_and_schedule_mutations(
    target: str, value: dict[str, object]
) -> None:
    _lower_ir, _lower_instance, template, schedule = _lower_identity()
    if target == "template":
        with pytest.raises(ValueError, match="template differs"):
            replace(template, **value).validate()
    else:
        with pytest.raises(ValueError, match="schedule differs"):
            replace(schedule, **value).validate_against(template)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_identity_tir_default_stream_zero_copy_autograd_and_cache() -> None:
    lower_ir, lower_instance, template, schedule = _lower_identity()
    device = torch.device("cuda:0")
    source = torch.linspace(
        -1.0, 1.0, template.tensor_numel, device=device
    ).requires_grad_(True)
    upstream = torch.linspace(0.5, 1.5, template.tensor_numel, device=device)
    instance = identity.build_b4b2_identity_instance_v1(
        template, lower_ir, lower_instance, source, upstream
    )
    cache = identity.DifferentiableLowerIdentityModuleCache()

    cold = identity.run_b4b2_identity_tir_probe_v1(
        template, instance, schedule, source, upstream, cache=cache
    )
    warm = identity.run_b4b2_identity_tir_probe_v1(
        template, instance, schedule, source, upstream, cache=cache
    )

    torch.testing.assert_close(cold.output, source, rtol=0, atol=0)
    torch.testing.assert_close(cold.input_gradient, upstream, rtol=0, atol=0)
    assert cold.launch_receipt.cache_event == "miss"
    assert warm.launch_receipt.cache_event == "hit"
    assert cold.launch_receipt.forward_launch_count == 1
    assert cold.launch_receipt.backward_launch_count == 1
    assert cold.launch_receipt.fallback_count == 0
    assert cold.launch_receipt.eager_backward_count == 0
    assert cold.launch_receipt.output_aliases_input is False
    assert cold.launch_receipt.input_gradient_aliases_upstream is False
    assert cold.launch_receipt.performance_claimed is False

    parsed_instance = DifferentiableLowerTIRInstanceV1.from_dict(
        instance.to_dict(template), template
    )
    parsed_module = DifferentiableLowerTIRModuleReceiptV1.from_dict(
        cold.module_receipt.to_dict(template, schedule), template, schedule
    )
    parsed_launch = DifferentiableLowerTIRLaunchReceiptV1.from_dict(
        cold.launch_receipt.to_dict(template, instance, schedule, cold.module_receipt),
        template,
        instance,
        schedule,
        cold.module_receipt,
    )
    assert parsed_instance == instance
    assert parsed_module == cold.module_receipt
    assert parsed_launch == cold.launch_receipt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_identity_tir_preserves_explicit_current_stream() -> None:
    lower_ir, lower_instance, template, schedule = _lower_identity(128)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.randn(128, device="cuda", requires_grad=True)
        upstream = torch.randn(128, device="cuda")
        instance = identity.build_b4b2_identity_instance_v1(
            template, lower_ir, lower_instance, source, upstream
        )
        result = identity.run_b4b2_identity_tir_probe_v1(
            template, instance, schedule, source, upstream
        )
    stream.synchronize()

    assert result.launch_receipt.stream_id == stream.cuda_stream
    assert result.launch_receipt.tvm_ffi_stream_id == stream.cuda_stream
    torch.testing.assert_close(result.output, source, rtol=0, atol=0)
    torch.testing.assert_close(result.input_gradient, upstream, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_identity_tir_rejects_higher_order_gradient() -> None:
    _lower_ir, _lower_instance, template, schedule = _lower_identity(64)
    executor = identity._IdentityTIRExecutor(
        template, schedule, identity.DifferentiableLowerIdentityModuleCache()
    )
    source = torch.randn(64, device="cuda", requires_grad=True)
    output = identity._DifferentiableLowerIdentityTIRFunction.apply(source, executor)
    with pytest.raises(RuntimeError, match="higher-order gradients"):
        torch.autograd.grad(output.sum(), source, create_graph=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_identity_tir_true_fallback_counters_fail_closed() -> None:
    _lower_ir, _lower_instance, template, schedule = _lower_identity(64)
    executor = identity._IdentityTIRExecutor(
        template, schedule, identity.DifferentiableLowerIdentityModuleCache()
    )
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        executor.reject_fallback(eager_backward=True)
    assert executor.fallback_count == 1
    assert executor.eager_backward_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_identity_tir_rejects_invalid_tensor_and_resigned_instance() -> None:
    lower_ir, lower_instance, template, schedule = _lower_identity(64)
    source = torch.randn(64, device="cuda", requires_grad=True)
    upstream = torch.randn(64, device="cuda")
    instance = identity.build_b4b2_identity_instance_v1(
        template, lower_ir, lower_instance, source, upstream
    )
    with pytest.raises(ValueError, match="probe tensor differs"):
        identity.run_b4b2_identity_tir_probe_v1(
            template, instance, schedule, source.to(torch.float64), upstream
        )
    with pytest.raises(ValueError, match="input differs from instance"):
        identity.run_b4b2_identity_tir_probe_v1(
            template, instance, schedule, source + 0.25, upstream
        )
    with pytest.raises(ValueError, match="instance differs"):
        replace(instance, template_hash="0" * 64).validate_against(template)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_identity_receipts_reject_resigned_launch_and_module_claims() -> None:
    lower_ir, lower_instance, template, schedule = _lower_identity(64)
    source = torch.randn(64, device="cuda", requires_grad=True)
    upstream = torch.randn(64, device="cuda")
    instance = identity.build_b4b2_identity_instance_v1(
        template, lower_ir, lower_instance, source, upstream
    )
    result = identity.run_b4b2_identity_tir_probe_v1(
        template, instance, schedule, source, upstream
    )

    with pytest.raises(ValueError, match="module receipt differs"):
        replace(result.module_receipt, performance_claimed=True).validate_against(
            template, schedule
        )
    with pytest.raises(ValueError, match="module receipt differs"):
        replace(result.module_receipt, tvm_commit="a" * 40).validate_against(
            template, schedule
        )
    with pytest.raises(ValueError, match="launch receipt differs"):
        replace(result.launch_receipt, fallback_count=1).validate_against(
            template, instance, schedule, result.module_receipt
        )
    with pytest.raises(ValueError, match="launch receipt differs"):
        replace(result.launch_receipt, output_aliases_input=True).validate_against(
            template, instance, schedule, result.module_receipt
        )
