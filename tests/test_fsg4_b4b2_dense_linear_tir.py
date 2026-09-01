"""B4-B2 B2-1 dense S-anchor Linear TIR correctness and failure gates."""

# pylint: disable=missing-function-docstring,protected-access,import-error
# pylint: disable=duplicate-code,too-many-locals

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.ir.differentiable_lower_dense_linear_tir import (
    DENSE_LINEAR_OUTPUT_NAMES,
    DifferentiableLowerDenseLinearTIRInstanceV1,
    DifferentiableLowerDenseLinearTIRLaunchReceiptV1,
    DifferentiableLowerDenseLinearTIRModuleReceiptV1,
    DifferentiableLowerDenseLinearTIRScheduleV1,
    DifferentiableLowerDenseLinearTIRTemplateV1,
)
from boundflow.runtime.fsg4_b4b1_pytorch_reference import (
    build_b4b1_differentiable_lower_instance_v1,
    build_b4b1_differentiable_lower_ir_v1,
    run_b4b1_pytorch_reference_v1,
)
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime import fsg4_b4b2_dense_linear_tir as dense_linear

ARTIFACT = Path("artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1")


def _capture(run_ordinal: int):
    payload = torch.load(
        ARTIFACT / f"run_{run_ordinal:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    capture = production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][0]
    )
    assert capture.base.anchor.anchor_id == "semantic-active-beta-gemm-14"
    return capture


def _lower(run_ordinal: int = 0):
    capture = _capture(run_ordinal)
    lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
    lower_instance = build_b4b1_differentiable_lower_instance_v1(capture, lower_ir)
    template = dense_linear.build_b4b2_dense_linear_template_v1(
        lower_ir, compute_capability="sm_89"
    )
    schedule = dense_linear.build_b4b2_dense_linear_schedule_v1(template)
    return capture, lower_ir, lower_instance, template, schedule


def test_b4b2_dense_linear_first_class_ir_round_trips_fail_closed() -> None:
    _capture_value, _lower_ir, _lower_instance, template, schedule = _lower()
    assert (
        DifferentiableLowerDenseLinearTIRTemplateV1.from_dict(template.to_dict())
        == template
    )
    assert (
        DifferentiableLowerDenseLinearTIRScheduleV1.from_dict(
            schedule.to_dict(template), template
        )
        == schedule
    )
    assert template.enabled_by_default is False
    assert template.performance_admitted is False
    with pytest.raises(ValueError, match="template differs"):
        replace(template, abi="identity-probe-v1").validate()
    with pytest.raises(ValueError, match="schedule differs"):
        replace(schedule, candidate_ordinal=1).validate_against(template)
    with pytest.raises(ValueError, match="schedule differs"):
        replace(schedule, performance_admitted=True).validate_against(template)


def test_b4b2_dense_linear_rejects_p_anchor_and_scope_broadening() -> None:
    payload = torch.load(ARTIFACT / "run_00.pt", map_location="cpu", weights_only=False)
    performance = production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][1]
    )
    performance_ir = build_b4b1_differentiable_lower_ir_v1(performance)
    with pytest.raises(ValueError, match="S-anchor differs"):
        dense_linear.build_b4b2_dense_linear_template_v1(
            performance_ir, compute_capability="sm_89"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_dense_linear_five_fresh_direct_reference_parity() -> None:
    cache = dense_linear.DifferentiableLowerDenseLinearModuleCache()
    module_hashes = set()
    maximum_difference = 0.0
    for run_ordinal in range(5):
        result = dense_linear.run_b4b2_dense_linear_tir_v1(
            _capture(run_ordinal), fresh_run_ordinal=run_ordinal, cache=cache
        )
        assert (
            tuple(metric.name for metric in result.metrics) == DENSE_LINEAR_OUTPUT_NAMES
        )
        assert all(metric.allclose and metric.sign_exact for metric in result.metrics)
        maximum_difference = max(
            maximum_difference,
            *(metric.maximum_absolute_difference for metric in result.metrics),
        )
        assert result.launch_receipt.cache_event == (
            "miss" if run_ordinal == 0 else "hit"
        )
        assert result.launch_receipt.forward_launch_count == 1
        assert result.launch_receipt.backward_launch_count == 1
        assert result.launch_receipt.fallback_count == 0
        assert result.launch_receipt.eager_backward_count == 0
        assert result.launch_receipt.semantic_passed is True
        assert result.launch_receipt.performance_claimed is False
        assert int(torch.count_nonzero(result.native_beta_gradient).item()) == 6
        module_hashes.add(
            result.module_receipt.stable_hash(
                _lower(run_ordinal)[3], _lower(run_ordinal)[4]
            )
        )
    assert maximum_difference <= 2.0e-4
    assert len(module_hashes) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_dense_linear_receipts_round_trip_and_bind_all_tensors() -> None:
    capture, lower_ir, lower_instance, template, schedule = _lower()
    tensors = dense_linear.build_b4b2_dense_linear_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    instance = dense_linear.build_b4b2_dense_linear_instance_v1(
        template,
        lower_ir,
        lower_instance,
        capture,
        tensors,
        fresh_run_ordinal=0,
    )
    result = dense_linear.run_b4b2_dense_linear_tir_v1(capture, fresh_run_ordinal=0)
    parsed_instance = DifferentiableLowerDenseLinearTIRInstanceV1.from_dict(
        instance.to_dict(template), template
    )
    parsed_module = DifferentiableLowerDenseLinearTIRModuleReceiptV1.from_dict(
        result.module_receipt.to_dict(template, schedule), template, schedule
    )
    parsed_launch = DifferentiableLowerDenseLinearTIRLaunchReceiptV1.from_dict(
        result.launch_receipt.to_dict(
            template, instance, schedule, result.module_receipt
        ),
        template,
        instance,
        schedule,
        result.module_receipt,
    )
    assert parsed_instance == instance
    assert parsed_module == result.module_receipt
    assert parsed_launch == result.launch_receipt
    assert result.launch_receipt.dlpack_pointer_count == 23
    assert result.launch_receipt.dlpack_pointer_exact_count == 23
    assert int(torch.count_nonzero(tensors.dense_split_sign).item()) == 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_dense_linear_clamp_endpoint_gradients_match_reference() -> None:
    capture, lower_ir, lower_instance, template, _schedule = _lower()
    tensors = dense_linear.build_b4b2_dense_linear_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    reference = run_b4b1_pytorch_reference_v1(capture, lower_ir, lower_instance)
    result = dense_linear.run_b4b2_dense_linear_tir_v1(capture, fresh_run_ordinal=0)
    ambiguous = (tensors.preactivation_lower < 0) & (tensors.preactivation_upper > 0)
    positive_a = tensors.incoming_lower_a[:, 0] >= 0
    endpoint = (tensors.native_alpha == 0) | (tensors.native_alpha == 1)
    mask = ambiguous & positive_a & endpoint
    assert int(mask.sum().item()) > 0
    torch.testing.assert_close(
        result.native_alpha_gradient[mask].cpu(),
        reference.native_alpha_gradient[mask.cpu()],
        atol=2.0e-4,
        rtol=2.0e-4,
    )
    assert torch.equal(
        torch.sign(result.native_alpha_gradient[mask].cpu()),
        torch.sign(reference.native_alpha_gradient[mask.cpu()]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_dense_linear_dtype_device_and_nonfinite_rejected() -> None:
    capture, _lower_ir, _lower_instance, template, _schedule = _lower()
    tensors = dense_linear.build_b4b2_dense_linear_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    with pytest.raises(ValueError, match="tensor differs: native_alpha"):
        dense_linear._validate_dense_linear_tensors(
            replace(tensors, native_alpha=tensors.native_alpha.double()), template
        )
    bad = tensors.native_alpha.detach().clone()
    bad[0, 0] = float("nan")
    bad.requires_grad_(True)
    with pytest.raises(ValueError, match="tensor differs: native_alpha"):
        dense_linear._validate_dense_linear_tensors(
            replace(tensors, native_alpha=bad), template
        )
    cpu = tensors.native_beta.detach().cpu().requires_grad_(True)
    with pytest.raises(ValueError, match="tensor differs: native_beta"):
        dense_linear._validate_dense_linear_tensors(
            replace(tensors, native_beta=cpu), template
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_dense_linear_a_equal_zero_owns_lower_branch_with_zero_alpha_vjp() -> None:
    capture, _lower_ir, _lower_instance, template, schedule = _lower()
    tensors = dense_linear.build_b4b2_dense_linear_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    ambiguous = (tensors.preactivation_lower < 0) & (tensors.preactivation_upper > 0)
    domain, feature = (int(value) for value in ambiguous.nonzero()[0].tolist())
    incoming = tensors.incoming_lower_a.clone()
    incoming[domain, 0, feature] = 0.0
    alpha = tensors.native_alpha.detach().clone()
    alpha[domain, feature] = 0.0
    alpha.requires_grad_(True)
    modified = replace(tensors, incoming_lower_a=incoming, native_alpha=alpha)
    executor = dense_linear._DenseLinearTIRExecutor(
        template,
        schedule,
        dense_linear.DifferentiableLowerDenseLinearModuleCache(),
    )
    executor.prime(modified)
    output_a, output_bias = dense_linear._DenseLinearTIRFunction.apply(
        modified.incoming_lower_a,
        modified.preactivation_lower,
        modified.preactivation_upper,
        modified.native_alpha,
        modified.native_beta,
        modified.dense_split_sign,
        modified.incoming_lower_bias,
        modified.operator_weight,
        modified.operator_bias,
        executor,
    )
    alpha_gradient = torch.autograd.grad(
        (output_a, output_bias),
        modified.native_alpha,
        grad_outputs=(
            modified.output_lower_a_gradient,
            modified.output_bias_gradient,
        ),
    )[0]
    assert alpha_gradient[domain, feature].item() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_dense_linear_custom_stream_and_exception_restore_global_state() -> None:
    capture, _lower_ir, _lower_instance, template, schedule = _lower()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tensors = dense_linear.build_b4b2_dense_linear_tensors_v1(
            capture, template, device=torch.device("cuda:0")
        )
        executor = dense_linear._DenseLinearTIRExecutor(
            template,
            schedule,
            dense_linear.DifferentiableLowerDenseLinearModuleCache(),
        )
        entry_device = torch.cuda.current_device()
        entry_stream = torch.cuda.current_stream().cuda_stream
        entry_policy = (
            torch.are_deterministic_algorithms_enabled(),
            torch.get_deterministic_debug_mode(),
        )
        sources = (
            tensors.incoming_lower_a,
            tensors.preactivation_lower,
            tensors.preactivation_upper,
            tensors.native_alpha,
            tensors.native_beta,
            tensors.dense_split_sign,
            tensors.incoming_lower_bias,
            tensors.operator_weight,
            tensors.operator_bias,
        )
        outputs = (
            torch.empty((6, 1, 1024), device="cuda"),
            torch.empty((6, 1), device="cuda"),
        )
        with pytest.raises(Exception):  # TVM runtime type is toolchain-specific.
            executor._launch("missing_dense_linear_symbol", sources, outputs)
        assert torch.cuda.current_device() == entry_device
        assert torch.cuda.current_stream().cuda_stream == entry_stream
        assert (
            torch.are_deterministic_algorithms_enabled(),
            torch.get_deterministic_debug_mode(),
        ) == entry_policy
    stream.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_dense_linear_true_fallback_counters_fail_closed() -> None:
    _capture_value, _lower_ir, _lower_instance, template, schedule = _lower()
    executor = dense_linear._DenseLinearTIRExecutor(
        template,
        schedule,
        dense_linear.DifferentiableLowerDenseLinearModuleCache(),
    )
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        executor.reject_fallback(eager_backward=True)
    assert executor.fallback_count == 1
    assert executor.eager_backward_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_dense_linear_rejects_higher_order_gradient() -> None:
    capture, _lower_ir, _lower_instance, template, schedule = _lower()
    tensors = dense_linear.build_b4b2_dense_linear_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    executor = dense_linear._DenseLinearTIRExecutor(
        template,
        schedule,
        dense_linear.DifferentiableLowerDenseLinearModuleCache(),
    )
    executor.prime(tensors)
    outputs = dense_linear._DenseLinearTIRFunction.apply(
        tensors.incoming_lower_a,
        tensors.preactivation_lower,
        tensors.preactivation_upper,
        tensors.native_alpha,
        tensors.native_beta,
        tensors.dense_split_sign,
        tensors.incoming_lower_bias,
        tensors.operator_weight,
        tensors.operator_bias,
        executor,
    )
    with pytest.raises(RuntimeError, match="higher-order gradients"):
        torch.autograd.grad(
            outputs,
            (tensors.native_alpha, tensors.native_beta),
            grad_outputs=(
                tensors.output_lower_a_gradient,
                tensors.output_bias_gradient,
            ),
            create_graph=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_dense_linear_rejects_resigned_instance_and_receipt_claims() -> None:
    capture, lower_ir, lower_instance, template, schedule = _lower()
    tensors = dense_linear.build_b4b2_dense_linear_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    instance = dense_linear.build_b4b2_dense_linear_instance_v1(
        template,
        lower_ir,
        lower_instance,
        capture,
        tensors,
        fresh_run_ordinal=0,
    )
    result = dense_linear.run_b4b2_dense_linear_tir_v1(capture, fresh_run_ordinal=0)
    with pytest.raises(ValueError, match="instance differs"):
        replace(instance, fresh_run_ordinal=5).validate_against(template)
    with pytest.raises(ValueError, match="module receipt differs"):
        replace(result.module_receipt, tvm_commit="a" * 40).validate_against(
            template, schedule
        )
    with pytest.raises(ValueError, match="launch receipt differs"):
        replace(result.launch_receipt, fallback_count=1).validate_against(
            template, instance, schedule, result.module_receipt
        )
    with pytest.raises(ValueError, match="launch receipt differs"):
        replace(result.launch_receipt, performance_claimed=True).validate_against(
            template, instance, schedule, result.module_receipt
        )
