"""Typed IR and pure-PyTorch semantic gates for FSG4/B4-B1."""

# pylint: disable=protected-access,missing-function-docstring,too-many-locals
# pylint: disable=duplicate-code,import-error

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.ir.differentiable_lower_region import (
    DIFFERENTIABLE_LOWER_REGION_STAGE_ORDER,
)
from boundflow.runtime.fsg4_b4b1_pytorch_reference import (
    build_b4b1_differentiable_lower_instance_v1,
    build_b4b1_differentiable_lower_ir_v1,
    build_b4b1_reference_receipt_v1,
    run_b4b1_pytorch_reference_v1,
)
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    ProductionDifferentiableReferenceCaptureV1,
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from scripts import run_fsg4_b4b1_pytorch_reference_artifact as reference_artifact
from scripts import probe_fsg4_b4b1_pytorch_reference_integrity as reference_integrity

ARTIFACT = Path("artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1/run_00.pt")
CAPTURE_ARTIFACT = ARTIFACT.parent
REFERENCE_ARTIFACT_V1 = Path("artifacts/fsg4-b4b1-pytorch-reference/resnet2b-prop0-v1")
REFERENCE_ARTIFACT_V2 = Path("artifacts/fsg4-b4b1-pytorch-reference/resnet2b-prop0-v2")
REFERENCE_INTEGRITY_REPORT = REFERENCE_ARTIFACT_V1.parent / (
    "resnet2b-prop0-v2-integrity-report.json"
)


def _captures() -> tuple[ProductionDifferentiableReferenceCaptureV1, ...]:
    payload = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    return tuple(
        production_differentiable_reference_capture_from_payload_v1(item)
        for item in payload["captures"]
    )


def _compile(capture: ProductionDifferentiableReferenceCaptureV1):
    ir = build_b4b1_differentiable_lower_ir_v1(capture)
    instance = build_b4b1_differentiable_lower_instance_v1(capture, ir)
    return ir, instance


def _replace_snapshot(snapshot, value: torch.Tensor):
    value = value.detach().contiguous()
    return replace(
        snapshot,
        value=value,
        content_sha256=production_tensor_sha256(value),
    )


def test_b4b1_reference_replays_both_formal_anchors() -> None:
    receipts = []
    for capture in _captures():
        ir, instance = _compile(capture)
        result = run_b4b1_pytorch_reference_v1(capture, ir, instance)
        receipt = build_b4b1_reference_receipt_v1(capture, ir, instance, result)
        assert receipt.semantic_passed is True
        assert receipt.performance_claimed is False
        assert receipt.tir_admitted is False
        assert all(metric.allclose and metric.sign_exact for metric in receipt.metrics)
        assert len(receipt.stable_hash(ir, instance)) == 64
        receipts.append(receipt)
    by_anchor = {receipt.anchor_id: receipt for receipt in receipts}
    assert set(by_anchor) == {
        "semantic-active-beta-gemm-14",
        "performance-conv-8-candidate",
    }
    assert by_anchor["semantic-active-beta-gemm-14"].beta_gradient_present is True
    assert (
        by_anchor["semantic-active-beta-gemm-14"].incoming_lower_a_gradient_present
        is False
    )
    assert by_anchor["performance-conv-8-candidate"].beta_gradient_present is False
    assert (
        by_anchor["performance-conv-8-candidate"].incoming_lower_a_gradient_present
        is True
    )


def test_b4b1_ir_freezes_order_and_signed_beta_semantics() -> None:
    semantic, performance = _captures()
    semantic_ir, _instance = _compile(semantic)
    performance_ir, _instance = _compile(performance)
    assert semantic_ir.stage_order == DIFFERENTIABLE_LOWER_REGION_STAGE_ORDER
    assert semantic_ir.beta_pre_add_formula == "negative-value-times-split-sign-v1"
    assert semantic_ir.relu_lower_relaxation == "ambiguous-alpha-sign-select-v1"
    assert semantic_ir.lower_only is True
    assert semantic_ir.coefficient_representation == "dense"
    assert semantic_ir.fanout == "single-consumer"
    assert semantic_ir.stream_ownership == "current-default"
    assert semantic_ir.alias_policy == "none"
    assert semantic_ir.beta_active is True
    assert performance_ir.beta_active is False
    assert semantic_ir.operator_kind == "linear"
    assert performance_ir.operator_kind == "conv2d"
    assert performance_ir.operator_attribute_map["output_padding"] == (0, 0)


def test_b4b1_static_ir_hash_is_stable_across_five_fresh_runs() -> None:
    hashes: dict[str, set[str]] = {}
    for run_index in range(5):
        payload = torch.load(
            ARTIFACT.parent / f"run_{run_index:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for raw in payload["captures"]:
            capture = production_differentiable_reference_capture_from_payload_v1(raw)
            ir = build_b4b1_differentiable_lower_ir_v1(capture)
            hashes.setdefault(ir.anchor_id, set()).add(ir.stable_hash())
    assert {anchor: len(values) for anchor, values in hashes.items()} == {
        "semantic-active-beta-gemm-14": 1,
        "performance-conv-8-candidate": 1,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider_start_node", "/wrong"),
        ("lower_only", False),
        ("coefficient_representation", "patches"),
        ("fanout", "multi-consumer"),
        ("stream_ownership", "non-default"),
        ("alias_policy", "unknown"),
        ("beta_pre_add_formula", "value-times-sign"),
    ],
)
def test_b4b1_ir_rejects_identity_and_semantic_policy_mutations(
    field: str, value: object
) -> None:
    ir, _instance = _compile(_captures()[0])
    with pytest.raises(ValueError, match="IR differs"):
        replace(ir, **{field: value}).validate()


def test_b4b1_ir_rejects_tensor_and_conv_attribute_mutations() -> None:
    ir, _instance = _compile(_captures()[1])
    contracts = list(ir.tensor_contracts)
    contracts[0] = replace(contracts[0], dtype="torch.float64")
    with pytest.raises(ValueError, match="tensor contract differs"):
        replace(ir, tensor_contracts=tuple(contracts)).validate()
    attributes = dict(ir.operator_attributes)
    attributes["output_padding"] = (1, 0)
    mutated = replace(ir, operator_attributes=tuple(sorted(attributes.items())))
    mutated.validate()
    capture = _captures()[1]
    instance = build_b4b1_differentiable_lower_instance_v1(capture, ir)
    with pytest.raises(ValueError, match="differs from capture"):
        run_b4b1_pytorch_reference_v1(
            capture, mutated, replace(instance, ir_hash=mutated.stable_hash())
        )


def test_b4b1_instance_rejects_resigned_input_hash() -> None:
    capture = _captures()[0]
    ir, instance = _compile(capture)
    hashes = list(instance.input_tensor_hashes)
    hashes[0] = (hashes[0][0], "0" * 64)
    mutated = replace(instance, input_tensor_hashes=tuple(hashes))
    mutated.validate_against(ir)
    with pytest.raises(ValueError, match="differs from capture"):
        run_b4b1_pytorch_reference_v1(capture, ir, mutated)


def test_b4b1_sparse_mapping_outer_resign_is_identity_rejected() -> None:
    capture = _captures()[0]
    mappings = list(capture.mapping_tensors)
    name, snapshot = mappings[0]
    changed = snapshot.value.clone()
    changed[1] = changed[0]
    mappings[0] = (name, _replace_snapshot(snapshot, changed))
    mutated = replace(capture, mapping_tensors=tuple(mappings))
    with pytest.raises(ValueError, match="sparse mapping identity differs"):
        mutated.validate()


@pytest.mark.parametrize("mutation", ["incoming-bias", "output-adjoint"])
def test_b4b1_coordinated_dynamic_rewrite_fails_numerical_semantics(
    mutation: str,
) -> None:
    capture = _captures()[0]
    if mutation == "incoming-bias":
        changed = capture.incoming_lower_bias.value + 0.125
        capture = replace(
            capture,
            incoming_lower_bias=_replace_snapshot(capture.incoming_lower_bias, changed),
        )
    else:
        changed = capture.output_lower_a_gradient.value.clone()
        changed.reshape(-1)[2] += 0.25
        capture = replace(
            capture,
            output_lower_a_gradient=_replace_snapshot(
                capture.output_lower_a_gradient, changed
            ),
        )
    capture.validate()
    ir, instance = _compile(capture)
    result = run_b4b1_pytorch_reference_v1(capture, ir, instance)
    receipt = build_b4b1_reference_receipt_v1(capture, ir, instance, result)
    assert receipt.semantic_passed is False
    failed = {metric.name for metric in receipt.metrics if not metric.allclose}
    if mutation == "incoming-bias":
        assert "output_bias" in failed
    else:
        assert {"native_alpha_gradient", "native_beta_gradient"} & failed


def _independent_s_anchor_incoming_gradient(
    capture: ProductionDifferentiableReferenceCaptureV1,
) -> torch.Tensor:
    values = capture.base.value_map
    incoming = values["incoming_lower_a"].value.detach().clone().requires_grad_(True)
    lower = values["preactivation_lower"].value
    upper = values["preactivation_upper"].value
    native_alpha = values["native_alpha"].value
    positive = lower >= 0
    negative = upper <= 0
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
    upper_intercept = torch.where(
        ambiguous, -lower * upper_slope, torch.zeros_like(lower)
    )
    lower_slope = torch.where(
        ambiguous,
        native_alpha.clamp(0.0, 1.0),
        torch.where(positive, torch.ones_like(lower), torch.zeros_like(lower)),
    )
    lower_a = incoming * torch.where(
        incoming >= 0, lower_slope.unsqueeze(1), upper_slope.unsqueeze(1)
    )
    lower_a = lower_a + values["relu_pre_add_coeff_l"].value.unsqueeze(1)
    lower_bias = capture.incoming_lower_bias.value + (
        incoming
        * torch.where(
            incoming >= 0,
            torch.zeros_like(upper_intercept).unsqueeze(1),
            upper_intercept.unsqueeze(1),
        )
    ).sum(2)
    output_a = lower_a @ values["operator_weight"].value
    output_bias = lower_bias + (
        lower_a * capture.operator_bias.value.reshape(1, 1, -1)
    ).sum(2)
    local_vjp = (output_a * capture.output_lower_a_gradient.value).sum() + (
        output_bias * capture.output_bias_gradient.value
    ).sum()
    return torch.autograd.grad(local_vjp, incoming)[0]


def test_b4b1_s_anchor_forced_incoming_gradient_micro_parity() -> None:
    capture = _captures()[0]
    ir, instance = _compile(capture)
    assert capture.base.gradient_map.get("incoming_lower_a") is None
    result = run_b4b1_pytorch_reference_v1(
        capture, ir, instance, force_incoming_a_gradient=True
    )
    assert result.incoming_lower_a_gradient is not None
    expected = _independent_s_anchor_incoming_gradient(capture)
    assert torch.allclose(
        result.incoming_lower_a_gradient, expected, atol=2e-4, rtol=2e-4
    )
    assert torch.equal(
        torch.sign(result.incoming_lower_a_gradient), torch.sign(expected)
    )


def test_b4b1_p_anchor_keeps_empty_beta_gradient_absent() -> None:
    capture = _captures()[1]
    ir, instance = _compile(capture)
    result = run_b4b1_pytorch_reference_v1(capture, ir, instance)
    assert capture.base.value_map["production_beta"].value.shape == (6, 0)
    assert result.native_beta_gradient is None
    assert result.incoming_lower_a_gradient is not None


def test_b4b1_reference_artifact_candidate_recomputes_all_five_fresh() -> None:
    protocol = reference_artifact._protocol(CAPTURE_ARTIFACT)
    reference_artifact._validate_protocol(protocol, CAPTURE_ARTIFACT)
    records = reference_artifact._records_from_source(CAPTURE_ARTIFACT, protocol)
    summary = reference_artifact._summary(records, protocol)
    assert len(records) == summary["capture_count"] == 10
    assert summary["metric_comparison_count"] == 60
    assert summary["element_comparison_count"] == 196380
    assert summary["maximum_absolute_difference"] == 6.109476089477539e-07
    assert summary["all_metrics_allclose"] is True
    assert summary["all_metrics_sign_exact"] is True
    assert summary["s_native_beta_gradient_count"] == 5
    assert summary["p_incoming_a_gradient_count"] == 5
    assert summary["performance_claimed"] is False
    assert summary["tir_admitted"] is False


def test_b4b1_v1_reference_artifact_is_rejected_after_policy_freeze() -> None:
    with pytest.raises(ValueError, match="reference protocol differs"):
        reference_artifact._verify_static_artifact(
            REFERENCE_ARTIFACT_V1, CAPTURE_ARTIFACT
        )


def test_b4b1_v2_deterministic_reference_artifact_root_replays() -> None:
    records, summary, result = reference_artifact._verify_static_artifact(
        REFERENCE_ARTIFACT_V2, CAPTURE_ARTIFACT
    )
    assert len(records) == summary["capture_count"] == result["capture_count"] == 10
    assert summary["summary_hash"] == (
        "becd8ae57536bc678392748bee5568d8b18922526df02da1238720b44045d744"
    )
    assert result["status"] == "replay-passed"
    assert result["maximum_absolute_difference"] == 6.109476089477539e-07
    assert result["all_metrics_sign_exact"] is True
    assert result["performance_claimed"] is False
    assert result["tir_admitted"] is False


def test_b4b1_reference_runner_restores_and_normalizes_thread_policy() -> None:
    previous = torch.get_num_threads()
    try:
        rows = []
        for threads in (1, 4):
            torch.set_num_threads(threads)
            protocol = reference_artifact._protocol(CAPTURE_ARTIFACT)
            records = reference_artifact._records_from_source(
                CAPTURE_ARTIFACT, protocol
            )
            assert torch.get_num_threads() == threads
            rows.append(records)
        assert rows[0] == rows[1]
    finally:
        torch.set_num_threads(previous)


def test_b4b1_coordinated_all_run_rewrites_are_numerically_rejected() -> None:
    report = reference_integrity._probe(CAPTURE_ARTIFACT.resolve())
    assert report["case_count"] == report["rejected_count"] == 2
    assert all(
        row["all_runs_rewritten"] is True
        and row["inner_capture_hashes_resigned"] is True
        and row["source_summary_resigned"] is True
        and row["source_manifest_resigned"] is True
        and row["derived_protocol_resigned"] is True
        and row["rejected_by_numerical_reference"] is True
        for row in report["rows"]
    )


def test_b4b1_formal_integrity_report_is_hash_bound() -> None:
    report = reference_artifact._load_json(REFERENCE_INTEGRITY_REPORT)
    assert report["case_count"] == report["rejected_count"] == 2
    assert report["report_hash"] == (
        "6a3192f6a6ab2e14ab012bfedd3cc4251de416739ec6850290c98cd3aa399313"
    )
    assert report["probe_code_sha256"] == (
        "154c90153d9fd2ac461957a1401c0d5184a59278dd239e956889fd95f731648b"
    )
    assert report["source_git_head"] == ("255d5fb2211faf5983bb9006ce3d2ef75c4f1c0b")
    assert set(report["reference_code_revision"]) == set(reference_artifact.CODE_PATHS)
    assert report["performance_claimed"] is False
    assert report["tir_admitted"] is False
