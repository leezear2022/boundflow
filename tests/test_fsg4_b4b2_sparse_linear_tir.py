"""B4-B2 B2-2 sparse-source S-anchor Linear correctness gates."""

# pylint: disable=protected-access,too-many-locals,missing-function-docstring
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.backends.tvm.differentiable_lower_sparse_linear import (
    build_sparse_linear_tir_modules,
)
from boundflow.ir.differentiable_lower_sparse_linear_tir import (
    SPARSE_LINEAR_OUTPUT_NAMES,
    DifferentiableLowerSparseLinearGradientProjectionReceiptV1,
    DifferentiableLowerSparseLinearTIRInstanceV1,
    DifferentiableLowerSparseLinearTIRLaunchReceiptV1,
    DifferentiableLowerSparseLinearTIRModuleReceiptV1,
    DifferentiableLowerSparseLinearTIRScheduleV1,
    DifferentiableLowerSparseLinearTIRTemplateV1,
)
from boundflow.ir.differentiable_lower_tir import canonical_tir_hash
from boundflow.runtime import fsg4_b4b2_sparse_linear_tir as sparse_linear
from boundflow.runtime.fsg4_b4b1_pytorch_reference import (
    build_b4b1_differentiable_lower_instance_v1,
    build_b4b1_differentiable_lower_ir_v1,
)
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)

ARTIFACT = Path("artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1")


def _capture(run_ordinal: int = 0, anchor_ordinal: int = 0):
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
    template = sparse_linear.build_b4b2_sparse_linear_template_v1(
        lower_ir, capture, compute_capability="sm_89"
    )
    schedule = sparse_linear.build_b4b2_sparse_linear_schedule_v1(template)
    return capture, lower_ir, lower_instance, template, schedule


def test_b4b2_sparse_linear_first_class_ir_round_trips_fail_closed() -> None:
    _capture_value, _lower_ir, _lower_instance, template, schedule = _lower()
    assert (
        DifferentiableLowerSparseLinearTIRTemplateV1.from_dict(template.to_dict())
        == template
    )
    assert (
        DifferentiableLowerSparseLinearTIRScheduleV1.from_dict(
            schedule.to_dict(template), template
        )
        == schedule
    )
    assert template.sparse_source_admitted is True
    assert template.performance_admitted is False
    assert schedule.workspace_names == ("adjoint_matmul", "output_bias_delta")
    assert "native_alpha" in schedule.forbidden_global_workspaces
    with pytest.raises(ValueError, match="template differs"):
        replace(template, abi="dense-linear-semantic-v1").validate()
    with pytest.raises(ValueError, match="template differs"):
        replace(template, sparse_source_admitted=False).validate()
    with pytest.raises(ValueError, match="schedule differs"):
        replace(schedule, performance_admitted=True).validate_against(template)


def test_b4b2_sparse_linear_mapping_constants_and_p_anchor_fail_closed() -> None:
    _capture_value, _lower_ir, _lower_instance, template, _schedule = _lower()
    assert len(template.alpha_feature_indices) == 27
    assert len(template.beta_locations) == 6
    assert len(template.beta_signs) == 6
    duplicated = template.alpha_feature_indices[:-1] + (
        template.alpha_feature_indices[-2],
    )
    with pytest.raises(ValueError, match="template differs"):
        replace(
            template,
            alpha_feature_indices=duplicated,
            alpha_feature_index_hash=canonical_tir_hash(list(duplicated)),
        ).validate()
    with pytest.raises(ValueError, match="template differs"):
        replace(
            template,
            beta_locations=(100,) + template.beta_locations[1:],
            beta_location_hash=canonical_tir_hash([100, *template.beta_locations[1:]]),
        ).validate()
    with pytest.raises(ValueError, match="template differs"):
        replace(
            template,
            beta_signs=(0,) + template.beta_signs[1:],
            beta_sign_hash=canonical_tir_hash([0, *template.beta_signs[1:]]),
        ).validate()
    performance = _capture(anchor_ordinal=1)
    performance_ir = build_b4b1_differentiable_lower_ir_v1(performance)
    with pytest.raises(ValueError, match="S-anchor differs"):
        sparse_linear.build_b4b2_sparse_linear_template_v1(
            performance_ir, performance, compute_capability="sm_89"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_sparse_linear_five_fresh_direct_oracle_parity() -> None:
    cache = sparse_linear.DifferentiableLowerSparseLinearModuleCache()
    maximum_difference = 0.0
    module_hashes = set()
    for run_ordinal in range(5):
        result = sparse_linear.run_b4b2_sparse_linear_tir_v1(
            _capture(run_ordinal), fresh_run_ordinal=run_ordinal, cache=cache
        )
        assert (
            tuple(metric.name for metric in result.metrics)
            == SPARSE_LINEAR_OUTPUT_NAMES
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
        assert result.launch_receipt.sparse_source_admitted is True
        assert result.launch_receipt.performance_claimed is False
        assert result.module_receipt.forbidden_workspace_count == 0
        assert int(torch.count_nonzero(result.compressed_beta_gradient).item()) == 6
        template = _lower(run_ordinal)[3]
        schedule = _lower(run_ordinal)[4]
        module_hashes.add(result.module_receipt.stable_hash(template, schedule))
    assert maximum_difference <= 2.0e-4
    assert len(module_hashes) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_sparse_linear_receipts_round_trip_and_projection_bindings() -> None:
    capture, lower_ir, lower_instance, template, schedule = _lower()
    tensors = sparse_linear.build_b4b2_sparse_linear_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    instance = sparse_linear.build_b4b2_sparse_linear_instance_v1(
        template,
        lower_ir,
        lower_instance,
        capture,
        tensors,
        fresh_run_ordinal=0,
    )
    result = sparse_linear.run_b4b2_sparse_linear_tir_v1(capture, fresh_run_ordinal=0)
    parsed_instance = DifferentiableLowerSparseLinearTIRInstanceV1.from_dict(
        instance.to_dict(template), template
    )
    parsed_module = DifferentiableLowerSparseLinearTIRModuleReceiptV1.from_dict(
        result.module_receipt.to_dict(template, schedule), template, schedule
    )
    parsed_projection = (
        DifferentiableLowerSparseLinearGradientProjectionReceiptV1.from_dict(
            result.projection_receipt.to_dict(template, instance), template, instance
        )
    )
    parsed_launch = DifferentiableLowerSparseLinearTIRLaunchReceiptV1.from_dict(
        result.launch_receipt.to_dict(
            template,
            instance,
            schedule,
            result.module_receipt,
            result.projection_receipt,
        ),
        template,
        instance,
        schedule,
        result.module_receipt,
        result.projection_receipt,
    )
    assert parsed_instance == instance
    assert parsed_module == result.module_receipt
    assert parsed_projection == result.projection_receipt
    assert parsed_launch == result.launch_receipt
    assert result.launch_receipt.dlpack_pointer_count == 21
    assert result.launch_receipt.dlpack_pointer_exact_count == 21
    assert result.projection_receipt.alpha_owned_element_count == 162
    assert result.projection_receipt.beta_owned_element_count == 6
    assert result.projection_receipt.unowned_native_zero_exact is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_sparse_linear_scheduled_tir_has_no_dense_state_workspace() -> None:
    _capture_value, _lower_ir, _lower_instance, template, schedule = _lower()
    _unscheduled, scheduled = build_sparse_linear_tir_modules(template, schedule)
    script = scheduled.script(show_meta=False)
    for forbidden in schedule.forbidden_global_workspaces:
        assert forbidden not in script
    for workspace in schedule.workspace_names:
        assert workspace in script
    assert "compressed_alpha" in script
    assert "compressed_beta" in script


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_sparse_linear_dtype_device_nonfinite_and_range_rejected() -> None:
    capture, _lower_ir, _lower_instance, template, _schedule = _lower()
    tensors = sparse_linear.build_b4b2_sparse_linear_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    with pytest.raises(ValueError, match="tensor differs: compressed_alpha"):
        sparse_linear._validate_sparse_linear_tensors(
            replace(tensors, compressed_alpha=tensors.compressed_alpha.double()),
            template,
        )
    bad = tensors.compressed_alpha.detach().clone()
    bad[0, 0] = float("nan")
    bad.requires_grad_(True)
    with pytest.raises(ValueError, match="tensor differs: compressed_alpha"):
        sparse_linear._validate_sparse_linear_tensors(
            replace(tensors, compressed_alpha=bad), template
        )
    out_of_range = tensors.compressed_alpha.detach().clone()
    out_of_range[0, 0] = 1.5
    out_of_range.requires_grad_(True)
    with pytest.raises(ValueError, match="alpha range differs"):
        sparse_linear._validate_sparse_linear_tensors(
            replace(tensors, compressed_alpha=out_of_range), template
        )
    cpu = tensors.compressed_beta.detach().cpu().requires_grad_(True)
    with pytest.raises(ValueError, match="tensor differs: compressed_beta"):
        sparse_linear._validate_sparse_linear_tensors(
            replace(tensors, compressed_beta=cpu), template
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_sparse_linear_clamp_endpoints_and_a_zero_gradient_ownership() -> None:
    capture, _lower_ir, _lower_instance, template, schedule = _lower()
    tensors = sparse_linear.build_b4b2_sparse_linear_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    indices = torch.tensor(template.alpha_feature_indices, device="cuda")
    lower = tensors.preactivation_lower[:, indices]
    upper = tensors.preactivation_upper[:, indices]
    incoming = tensors.incoming_lower_a[:, 0, indices]
    endpoint = (tensors.compressed_alpha == 0) | (tensors.compressed_alpha == 1)
    mask = (lower < 0) & (upper > 0) & (incoming >= 0) & endpoint
    assert int(mask.sum().item()) > 0
    domain, ordinal = (int(value) for value in mask.nonzero()[0].tolist())
    feature = template.alpha_feature_indices[ordinal]
    modified_incoming = tensors.incoming_lower_a.clone()
    modified_incoming[domain, 0, feature] = 0.0
    modified_alpha = tensors.compressed_alpha.detach().clone().requires_grad_(True)
    modified = replace(
        tensors,
        incoming_lower_a=modified_incoming,
        compressed_alpha=modified_alpha,
    )
    executor = sparse_linear._SparseLinearTIRExecutor(
        template,
        schedule,
        sparse_linear.DifferentiableLowerSparseLinearModuleCache(),
    )
    executor.prime(modified)
    outputs = sparse_linear._SparseLinearTIRFunction.apply(
        modified.incoming_lower_a,
        modified.preactivation_lower,
        modified.preactivation_upper,
        modified.compressed_alpha,
        modified.compressed_beta,
        modified.incoming_lower_bias,
        modified.operator_weight,
        modified.operator_bias,
        executor,
    )
    alpha_gradient = torch.autograd.grad(
        outputs,
        modified.compressed_alpha,
        grad_outputs=(
            modified.output_lower_a_gradient,
            modified.output_bias_gradient,
        ),
    )[0]
    assert alpha_gradient[domain, ordinal].item() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_sparse_linear_custom_stream_exception_restores_global_state() -> None:
    capture, _lower_ir, _lower_instance, template, schedule = _lower()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tensors = sparse_linear.build_b4b2_sparse_linear_tensors_v1(
            capture, template, device=torch.device("cuda:0")
        )
        executor = sparse_linear._SparseLinearTIRExecutor(
            template,
            schedule,
            sparse_linear.DifferentiableLowerSparseLinearModuleCache(),
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
            tensors.compressed_alpha,
            tensors.compressed_beta,
            tensors.incoming_lower_bias,
            tensors.operator_weight,
            tensors.operator_bias,
        )
        outputs = (
            torch.empty((6, 1, 1024), device="cuda"),
            torch.empty((6, 1), device="cuda"),
        )
        with pytest.raises(Exception):
            executor._launch("missing_sparse_linear_symbol", sources, outputs)
        assert torch.cuda.current_device() == entry_device
        assert torch.cuda.current_stream().cuda_stream == entry_stream
        assert (
            torch.are_deterministic_algorithms_enabled(),
            torch.get_deterministic_debug_mode(),
        ) == entry_policy
    stream.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_sparse_linear_true_fallback_counters_fail_closed() -> None:
    _capture_value, _lower_ir, _lower_instance, template, schedule = _lower()
    executor = sparse_linear._SparseLinearTIRExecutor(
        template,
        schedule,
        sparse_linear.DifferentiableLowerSparseLinearModuleCache(),
    )
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        executor.reject_fallback(eager_backward=True)
    assert executor.fallback_count == 1
    assert executor.eager_backward_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_sparse_linear_rejects_higher_order_gradient() -> None:
    capture, _lower_ir, _lower_instance, template, schedule = _lower()
    tensors = sparse_linear.build_b4b2_sparse_linear_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    executor = sparse_linear._SparseLinearTIRExecutor(
        template,
        schedule,
        sparse_linear.DifferentiableLowerSparseLinearModuleCache(),
    )
    executor.prime(tensors)
    outputs = sparse_linear._SparseLinearTIRFunction.apply(
        tensors.incoming_lower_a,
        tensors.preactivation_lower,
        tensors.preactivation_upper,
        tensors.compressed_alpha,
        tensors.compressed_beta,
        tensors.incoming_lower_bias,
        tensors.operator_weight,
        tensors.operator_bias,
        executor,
    )
    with pytest.raises(RuntimeError, match="higher-order gradients"):
        torch.autograd.grad(
            outputs,
            (tensors.compressed_alpha, tensors.compressed_beta),
            grad_outputs=(
                tensors.output_lower_a_gradient,
                tensors.output_bias_gradient,
            ),
            create_graph=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_sparse_linear_resigned_receipts_and_claims_rejected() -> None:
    capture, lower_ir, lower_instance, template, schedule = _lower()
    tensors = sparse_linear.build_b4b2_sparse_linear_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    instance = sparse_linear.build_b4b2_sparse_linear_instance_v1(
        template,
        lower_ir,
        lower_instance,
        capture,
        tensors,
        fresh_run_ordinal=0,
    )
    result = sparse_linear.run_b4b2_sparse_linear_tir_v1(capture, fresh_run_ordinal=0)
    with pytest.raises(ValueError, match="instance differs"):
        replace(instance, fresh_run_ordinal=5).validate_against(template)
    with pytest.raises(ValueError, match="module receipt differs"):
        replace(result.module_receipt, forbidden_workspace_count=1).validate_against(
            template, schedule
        )
    with pytest.raises(ValueError, match="projection differs"):
        replace(
            result.projection_receipt, alpha_numerical_passed=False
        ).validate_against(template, instance)
    with pytest.raises(ValueError, match="launch receipt differs"):
        replace(result.launch_receipt, fallback_count=1).validate_against(
            template,
            instance,
            schedule,
            result.module_receipt,
            result.projection_receipt,
        )
    with pytest.raises(ValueError, match="launch receipt differs"):
        replace(result.launch_receipt, performance_claimed=True).validate_against(
            template,
            instance,
            schedule,
            result.module_receipt,
            result.projection_receipt,
        )
