"""B4-B2 B2-3 P-anchor dense Conv TIR correctness and failure gates."""

# pylint: disable=missing-function-docstring,protected-access,import-error
# pylint: disable=duplicate-code,too-many-locals

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.backends.tvm.differentiable_lower_dense_conv import (
    build_dense_conv_tir_modules,
)
from boundflow.ir.differentiable_lower_dense_conv_tir import (
    DENSE_CONV_OUTPUT_NAMES,
    DENSE_CONV_WORKSPACE_INVENTORY,
    DifferentiableLowerDenseConvTIRInstanceV1,
    DifferentiableLowerDenseConvTIRLaunchReceiptV1,
    DifferentiableLowerDenseConvTIRModuleReceiptV1,
    DifferentiableLowerDenseConvTIRScheduleV1,
    DifferentiableLowerDenseConvTIRTemplateV1,
)
from boundflow.runtime import fsg4_b4b2_dense_conv_tir as dense_conv
from boundflow.runtime.fsg4_b4b1_pytorch_reference import (
    build_b4b1_differentiable_lower_instance_v1,
    build_b4b1_differentiable_lower_ir_v1,
)
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)

ARTIFACT = Path("artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1")


def _capture(run_ordinal: int = 0, anchor_ordinal: int = 1):
    payload = torch.load(
        ARTIFACT / f"run_{run_ordinal:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    return production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][anchor_ordinal]
    )


def _lower(run_ordinal: int = 0):
    capture = _capture(run_ordinal)
    lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
    lower_instance = build_b4b1_differentiable_lower_instance_v1(capture, lower_ir)
    template = dense_conv.build_b4b2_dense_conv_template_v1(
        lower_ir, compute_capability="sm_89"
    )
    schedule = dense_conv.build_b4b2_dense_conv_schedule_v1(template)
    return capture, lower_ir, lower_instance, template, schedule


def test_b4b2_dense_conv_first_class_ir_round_trips_fail_closed() -> None:
    _capture_value, _lower_ir, _lower_instance, template, schedule = _lower()
    assert (
        DifferentiableLowerDenseConvTIRTemplateV1.from_dict(template.to_dict())
        == template
    )
    assert (
        DifferentiableLowerDenseConvTIRScheduleV1.from_dict(
            schedule.to_dict(template), template
        )
        == schedule
    )
    assert template.compressed_beta_shape == (6, 0)
    assert template.beta_gradient_present is False
    assert template.performance_admitted is False
    assert schedule.workspace_inventory == DENSE_CONV_WORKSPACE_INVENTORY
    with pytest.raises(ValueError, match="template differs"):
        replace(template, beta_gradient_present=True).validate()
    with pytest.raises(ValueError, match="template differs"):
        replace(template, abi="dense-linear-semantic-v1").validate()
    with pytest.raises(ValueError, match="schedule differs"):
        replace(schedule, performance_admitted=True).validate_against(template)


def test_b4b2_dense_conv_rejects_s_anchor_and_scope_broadening() -> None:
    semantic = _capture(anchor_ordinal=0)
    semantic_ir = build_b4b1_differentiable_lower_ir_v1(semantic)
    with pytest.raises(ValueError, match="P-anchor differs"):
        dense_conv.build_b4b2_dense_conv_template_v1(
            semantic_ir, compute_capability="sm_89"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_dense_conv_five_fresh_direct_reference_parity() -> None:
    cache = dense_conv.DifferentiableLowerDenseConvModuleCache()
    module_hashes = set()
    maximum_difference = 0.0
    for run_ordinal in range(5):
        result = dense_conv.run_b4b2_dense_conv_tir_v1(
            _capture(run_ordinal), fresh_run_ordinal=run_ordinal, cache=cache
        )
        assert (
            tuple(metric.name for metric in result.metrics) == DENSE_CONV_OUTPUT_NAMES
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
        assert result.launch_receipt.beta_gradient_present is False
        assert result.launch_receipt.performance_claimed is False
        assert result.module_receipt.structural_workspace_check is True
        module_hashes.add(
            result.module_receipt.stable_hash(
                _lower(run_ordinal)[3], _lower(run_ordinal)[4]
            )
        )
    assert maximum_difference <= 2.0e-4
    assert len(module_hashes) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_dense_conv_receipts_round_trip_and_bind_all_tensors() -> None:
    capture, lower_ir, lower_instance, template, schedule = _lower()
    tensors = dense_conv.build_b4b2_dense_conv_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    instance = dense_conv.build_b4b2_dense_conv_instance_v1(
        template,
        lower_ir,
        lower_instance,
        capture,
        tensors,
        fresh_run_ordinal=0,
    )
    result = dense_conv.run_b4b2_dense_conv_tir_v1(capture, fresh_run_ordinal=0)
    parsed_instance = DifferentiableLowerDenseConvTIRInstanceV1.from_dict(
        instance.to_dict(template), template
    )
    parsed_module = DifferentiableLowerDenseConvTIRModuleReceiptV1.from_dict(
        result.module_receipt.to_dict(template, schedule), template, schedule
    )
    parsed_launch = DifferentiableLowerDenseConvTIRLaunchReceiptV1.from_dict(
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
    assert result.launch_receipt.dlpack_pointer_count == 19
    assert result.launch_receipt.dlpack_pointer_exact_count == 19
    assert result.launch_receipt.incoming_a_gradient_present is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_dense_conv_scheduled_tir_workspace_is_structural_and_exact() -> None:
    _capture_value, _lower_ir, _lower_instance, template, schedule = _lower()
    _unscheduled, _scheduled, observed = build_dense_conv_tir_modules(
        template, schedule
    )
    assert observed == DENSE_CONV_WORKSPACE_INVENTORY


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_dense_conv_dtype_device_nonfinite_interval_and_range_rejected() -> None:
    capture, _lower_ir, _lower_instance, template, _schedule = _lower()
    tensors = dense_conv.build_b4b2_dense_conv_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    with pytest.raises(ValueError, match="tensor differs: native_alpha"):
        dense_conv._validate_dense_conv_tensors(
            replace(tensors, native_alpha=tensors.native_alpha[:, :, :, :-1]),
            template,
        )
    with pytest.raises(ValueError, match="tensor differs: native_alpha"):
        dense_conv._validate_dense_conv_tensors(
            replace(tensors, native_alpha=tensors.native_alpha.double()), template
        )
    bad = tensors.native_alpha.detach().clone()
    bad[0, 0, 0, 0] = float("nan")
    bad.requires_grad_(True)
    with pytest.raises(ValueError, match="tensor differs: native_alpha"):
        dense_conv._validate_dense_conv_tensors(
            replace(tensors, native_alpha=bad), template
        )
    out_of_range = tensors.native_alpha.detach().clone()
    out_of_range[0, 0, 0, 0] = 1.5
    out_of_range.requires_grad_(True)
    with pytest.raises(ValueError, match="alpha range differs"):
        dense_conv._validate_dense_conv_tensors(
            replace(tensors, native_alpha=out_of_range), template
        )
    cpu = tensors.incoming_lower_a.detach().cpu().requires_grad_(True)
    with pytest.raises(ValueError, match="tensor differs: incoming_lower_a"):
        dense_conv._validate_dense_conv_tensors(
            replace(tensors, incoming_lower_a=cpu), template
        )
    invalid_lower = tensors.preactivation_upper + 1.0
    with pytest.raises(ValueError, match="interval differs"):
        dense_conv._validate_dense_conv_tensors(
            replace(tensors, preactivation_lower=invalid_lower), template
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_dense_conv_custom_stream_exception_restores_global_state() -> None:
    capture, _lower_ir, _lower_instance, template, schedule = _lower()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tensors = dense_conv.build_b4b2_dense_conv_tensors_v1(
            capture, template, device=torch.device("cuda:0")
        )
        executor = dense_conv._DenseConvTIRExecutor(
            template, schedule, dense_conv.DifferentiableLowerDenseConvModuleCache()
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
            tensors.incoming_lower_bias,
            tensors.operator_weight,
            tensors.operator_bias,
        )
        outputs = (
            torch.empty_like(tensors.incoming_lower_a),
            torch.empty((6, 1), device="cuda"),
        )
        with pytest.raises(Exception):
            executor._launch("missing_dense_conv_symbol", sources, outputs)
        assert torch.cuda.current_device() == entry_device
        assert torch.cuda.current_stream().cuda_stream == entry_stream
        assert (
            torch.are_deterministic_algorithms_enabled(),
            torch.get_deterministic_debug_mode(),
        ) == entry_policy
    stream.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_dense_conv_fallback_higher_order_and_claims_fail_closed() -> None:
    capture, lower_ir, lower_instance, template, schedule = _lower()
    executor = dense_conv._DenseConvTIRExecutor(
        template, schedule, dense_conv.DifferentiableLowerDenseConvModuleCache()
    )
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        executor.reject_fallback(eager_backward=True)
    assert (executor.fallback_count, executor.eager_backward_count) == (1, 1)

    tensors = dense_conv.build_b4b2_dense_conv_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    instance = dense_conv.build_b4b2_dense_conv_instance_v1(
        template,
        lower_ir,
        lower_instance,
        capture,
        tensors,
        fresh_run_ordinal=0,
    )
    higher_order = dense_conv._DenseConvTIRExecutor(
        template, schedule, dense_conv.DifferentiableLowerDenseConvModuleCache()
    )
    higher_order.prime(tensors)
    outputs = dense_conv._DenseConvTIRFunction.apply(
        tensors.incoming_lower_a,
        tensors.preactivation_lower,
        tensors.preactivation_upper,
        tensors.native_alpha,
        tensors.incoming_lower_bias,
        tensors.operator_weight,
        tensors.operator_bias,
        higher_order,
    )
    with pytest.raises(RuntimeError, match="higher-order gradients"):
        torch.autograd.grad(
            outputs,
            (tensors.incoming_lower_a, tensors.native_alpha),
            grad_outputs=(
                tensors.output_lower_a_gradient,
                tensors.output_bias_gradient,
            ),
            create_graph=True,
        )

    result = dense_conv.run_b4b2_dense_conv_tir_v1(capture, fresh_run_ordinal=0)
    with pytest.raises(ValueError, match="instance differs"):
        replace(instance, fresh_run_ordinal=5).validate_against(template)
    with pytest.raises(ValueError, match="module receipt differs"):
        replace(
            result.module_receipt, structural_workspace_check=False
        ).validate_against(template, schedule)
    with pytest.raises(ValueError, match="launch receipt differs"):
        replace(result.launch_receipt, performance_claimed=True).validate_against(
            template, instance, schedule, result.module_receipt
        )
