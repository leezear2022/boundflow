"""B4-B2 B2-4 sparse-source P-anchor Conv correctness and ledger gates."""

# pylint: disable=missing-function-docstring,protected-access,import-error
# pylint: disable=duplicate-code,too-many-locals

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.backends.tvm.differentiable_lower_sparse_conv import (
    build_sparse_conv_tir_modules,
)
from boundflow.ir.differentiable_lower_dense_conv_tir import (
    DENSE_CONV_WORKSPACE_INVENTORY,
)
from boundflow.ir.differentiable_lower_sparse_conv_tir import (
    SPARSE_CONV_CANDIDATE_KNOBS,
    SPARSE_CONV_INPUT_NAMES,
    SPARSE_CONV_OUTPUT_NAMES,
    DifferentiableLowerSparseConvCandidateLedgerV1,
    DifferentiableLowerSparseConvGradientProjectionReceiptV1,
    DifferentiableLowerSparseConvTIRInstanceV1,
    DifferentiableLowerSparseConvTIRLaunchReceiptV1,
    DifferentiableLowerSparseConvTIRModuleReceiptV1,
    DifferentiableLowerSparseConvTIRScheduleV1,
    DifferentiableLowerSparseConvTIRTemplateV1,
)
from boundflow.ir.differentiable_lower_tir import canonical_tir_hash
from boundflow.runtime import fsg4_b4b2_sparse_conv_tir as sparse_conv
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
        ARTIFACT / f"run_{run_ordinal:02d}.pt", map_location="cpu", weights_only=False
    )
    return production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][anchor_ordinal]
    )


def _lower(run_ordinal: int = 0):
    capture = _capture(run_ordinal)
    lower_ir = build_b4b1_differentiable_lower_ir_v1(capture)
    lower_instance = build_b4b1_differentiable_lower_instance_v1(capture, lower_ir)
    template = sparse_conv.build_b4b2_sparse_conv_template_v1(
        lower_ir, capture, compute_capability="sm_89"
    )
    schedules = sparse_conv.build_b4b2_sparse_conv_schedules_v1(template)
    ledger = sparse_conv.build_b4b2_sparse_conv_ledger_v1(template, schedules)
    return capture, lower_ir, lower_instance, template, schedules, ledger


def test_b4b2_sparse_conv_template_schedule_and_ledger_round_trip() -> None:
    _capture_value, _lower_ir, _instance, template, schedules, ledger = _lower()
    assert (
        DifferentiableLowerSparseConvTIRTemplateV1.from_dict(template.to_dict())
        == template
    )
    assert len(template.alpha_coordinates) == 86
    assert len(set(template.alpha_coordinates)) == 86
    assert template.compressed_beta_entries == 0
    assert "compressed_beta" not in SPARSE_CONV_INPUT_NAMES
    assert len(schedules) == 12
    assert (
        tuple(schedule.knob_tuple for schedule in schedules)
        == SPARSE_CONV_CANDIDATE_KNOBS
    )
    assert len(set(ledger.schedule_hashes)) == 12
    for schedule in schedules:
        assert (
            DifferentiableLowerSparseConvTIRScheduleV1.from_dict(
                schedule.to_dict(template), template
            )
            == schedule
        )
    assert (
        DifferentiableLowerSparseConvCandidateLedgerV1.from_dict(
            ledger.to_dict(template, schedules), template, schedules
        )
        == ledger
    )


def test_b4b2_sparse_conv_mapping_and_ledger_tamper_fail_closed() -> None:
    _capture_value, _lower_ir, _instance, template, schedules, ledger = _lower()
    duplicated_channels = template.alpha_channels[:-1] + (template.alpha_channels[-2],)
    duplicated_heights = template.alpha_heights[:-1] + (template.alpha_heights[-2],)
    duplicated_widths = template.alpha_widths[:-1] + (template.alpha_widths[-2],)
    coordinates = list(zip(duplicated_channels, duplicated_heights, duplicated_widths))
    with pytest.raises(ValueError, match="template differs"):
        replace(
            template,
            alpha_channels=duplicated_channels,
            alpha_heights=duplicated_heights,
            alpha_widths=duplicated_widths,
            alpha_coordinate_hash=canonical_tir_hash(
                [list(item) for item in coordinates]
            ),
        ).validate()
    with pytest.raises(ValueError, match="schedule differs"):
        replace(schedules[0], thread_extent=256).validate_against(template)
    with pytest.raises(ValueError, match="candidate ledger differs"):
        replace(ledger, timing_raw_present=True).validate_against(template, schedules)
    with pytest.raises(ValueError, match="candidate ledger differs"):
        replace(ledger, winner_selected=True).validate_against(template, schedules)
    with pytest.raises(ValueError, match="candidate ledger differs"):
        replace(ledger, schedule_hashes=ledger.schedule_hashes[:11]).validate_against(
            template, schedules
        )


def test_b4b2_sparse_conv_all_schedules_have_unique_tir_and_exact_workspace() -> None:
    _capture_value, _lower_ir, _instance, template, schedules, _ledger = _lower()
    scripts = set()
    for schedule in schedules:
        _unscheduled, scheduled, observed = build_sparse_conv_tir_modules(
            template, schedule
        )
        assert observed == DENSE_CONV_WORKSPACE_INVENTORY
        scripts.add(scheduled.script(show_meta=False))
    assert len(scripts) == 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_sparse_conv_p0_five_raw_direct_oracle_parity() -> None:
    cache = sparse_conv.DifferentiableLowerSparseConvModuleCache()
    maximum_difference = 0.0
    module_hashes = set()
    for run_ordinal in range(5):
        result = sparse_conv.run_b4b2_sparse_conv_tir_v1(
            _capture(run_ordinal), fresh_run_ordinal=run_ordinal, cache=cache
        )
        assert (
            tuple(metric.name for metric in result.metrics) == SPARSE_CONV_OUTPUT_NAMES
        )
        assert all(metric.allclose and metric.sign_exact for metric in result.metrics)
        maximum_difference = max(
            maximum_difference,
            *(metric.maximum_absolute_difference for metric in result.metrics),
        )
        assert result.launch_receipt.cache_event == (
            "miss" if run_ordinal == 0 else "hit"
        )
        assert result.launch_receipt.beta_gradient_present is False
        assert result.launch_receipt.dlpack_pointer_count == 19
        assert result.projection_receipt.alpha_owned_element_count == 516
        assert result.projection_receipt.unowned_native_zero_exact is True
        _capture_value, _ir, _instance, template, schedules, _ledger = _lower(
            run_ordinal
        )
        module_hashes.add(result.module_receipt.stable_hash(template, schedules[0]))
    assert maximum_difference <= 2.0e-4
    assert len(module_hashes) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_sparse_conv_all_twelve_candidates_compile_and_match_oracle() -> None:
    cache = sparse_conv.DifferentiableLowerSparseConvModuleCache()
    module_hashes = set()
    for candidate_ordinal in range(12):
        result = sparse_conv.run_b4b2_sparse_conv_tir_v1(
            _capture(),
            fresh_run_ordinal=0,
            candidate_ordinal=candidate_ordinal,
            cache=cache,
        )
        assert all(metric.allclose and metric.sign_exact for metric in result.metrics)
        assert result.launch_receipt.cache_event == "miss"
        assert result.launch_receipt.performance_claimed is False
        template, schedule = _lower()[3], _lower()[4][candidate_ordinal]
        module_hashes.add(result.module_receipt.stable_hash(template, schedule))
    assert len(module_hashes) == 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_sparse_conv_receipts_round_trip_and_claim_tamper_rejected() -> None:
    capture, lower_ir, lower_instance, template, schedules, _ledger = _lower()
    tensors = sparse_conv.build_b4b2_sparse_conv_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    instance = sparse_conv.build_b4b2_sparse_conv_instance_v1(
        template,
        lower_ir,
        lower_instance,
        capture,
        tensors,
        fresh_run_ordinal=0,
    )
    result = sparse_conv.run_b4b2_sparse_conv_tir_v1(capture, fresh_run_ordinal=0)
    schedule = schedules[0]
    parsed_instance = DifferentiableLowerSparseConvTIRInstanceV1.from_dict(
        instance.to_dict(template), template
    )
    parsed_module = DifferentiableLowerSparseConvTIRModuleReceiptV1.from_dict(
        result.module_receipt.to_dict(template, schedule), template, schedule
    )
    parsed_projection = (
        DifferentiableLowerSparseConvGradientProjectionReceiptV1.from_dict(
            result.projection_receipt.to_dict(template, instance), template, instance
        )
    )
    parsed_launch = DifferentiableLowerSparseConvTIRLaunchReceiptV1.from_dict(
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
    with pytest.raises(ValueError, match="module receipt differs"):
        replace(result.module_receipt, performance_claimed=True).validate_against(
            template, schedule
        )
    with pytest.raises(ValueError, match="gradient projection differs"):
        replace(result.projection_receipt, beta_gradient_absent=False).validate_against(
            template, instance
        )
    with pytest.raises(ValueError, match="launch receipt differs"):
        replace(result.launch_receipt, performance_claimed=True).validate_against(
            template,
            instance,
            schedule,
            result.module_receipt,
            result.projection_receipt,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_sparse_conv_shape_dtype_device_nonfinite_and_range_rejected() -> None:
    capture, _ir, _instance, template, _schedules, _ledger = _lower()
    tensors = sparse_conv.build_b4b2_sparse_conv_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    with pytest.raises(ValueError, match="tensor differs: compressed_alpha"):
        sparse_conv._validate_sparse_conv_tensors(
            replace(tensors, compressed_alpha=tensors.compressed_alpha[:, :-1]),
            template,
        )
    with pytest.raises(ValueError, match="tensor differs: compressed_alpha"):
        sparse_conv._validate_sparse_conv_tensors(
            replace(tensors, compressed_alpha=tensors.compressed_alpha.double()),
            template,
        )
    bad = tensors.compressed_alpha.detach().clone()
    bad[0, 0] = float("nan")
    bad.requires_grad_(True)
    with pytest.raises(ValueError, match="tensor differs: compressed_alpha"):
        sparse_conv._validate_sparse_conv_tensors(
            replace(tensors, compressed_alpha=bad), template
        )
    out_of_range = tensors.compressed_alpha.detach().clone()
    out_of_range[0, 0] = 1.5
    out_of_range.requires_grad_(True)
    with pytest.raises(ValueError, match="alpha range differs"):
        sparse_conv._validate_sparse_conv_tensors(
            replace(tensors, compressed_alpha=out_of_range), template
        )
    cpu = tensors.incoming_lower_a.detach().cpu().requires_grad_(True)
    with pytest.raises(ValueError, match="tensor differs: incoming_lower_a"):
        sparse_conv._validate_sparse_conv_tensors(
            replace(tensors, incoming_lower_a=cpu), template
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_b4b2_sparse_conv_fallback_and_higher_order_fail_closed() -> None:
    capture, _ir, _instance, template, schedules, _ledger = _lower()
    executor = sparse_conv._SparseConvTIRExecutor(
        template, schedules[0], sparse_conv.DifferentiableLowerSparseConvModuleCache()
    )
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        executor.reject_fallback(eager_backward=True)
    assert (executor.fallback_count, executor.eager_backward_count) == (1, 1)
    tensors = sparse_conv.build_b4b2_sparse_conv_tensors_v1(
        capture, template, device=torch.device("cuda:0")
    )
    higher_order = sparse_conv._SparseConvTIRExecutor(
        template, schedules[0], sparse_conv.DifferentiableLowerSparseConvModuleCache()
    )
    higher_order.prime(tensors)
    outputs = sparse_conv._SparseConvTIRFunction.apply(
        tensors.incoming_lower_a,
        tensors.preactivation_lower,
        tensors.preactivation_upper,
        tensors.compressed_alpha,
        tensors.incoming_lower_bias,
        tensors.operator_weight,
        tensors.operator_bias,
        higher_order,
    )
    with pytest.raises(RuntimeError, match="higher-order gradients"):
        torch.autograd.grad(
            outputs,
            (tensors.incoming_lower_a, tensors.compressed_alpha),
            grad_outputs=(
                tensors.output_lower_a_gradient,
                tensors.output_bias_gradient,
            ),
            create_graph=True,
        )
