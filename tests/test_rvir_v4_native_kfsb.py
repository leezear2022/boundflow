"""Contracts for RVIR-v4 V4-3C native KFSB evaluation."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from boundflow.runtime.rvir_v4_native_kfsb import (
    compare_rvir_v4_native_kfsb,
    NativeKfsbEvaluationV4,
)
from scripts import run_rvir_v4_native_kfsb_artifact as artifact_runner

ATOMIC_CAPTURE = Path(
    "artifacts/rvir-v4-atomic-copy-out/resnet2b-core-copy-out-v1/source_capture.pt"
)
WHOLE_TRUTH = Path("artifacts/rvir-v4-whole-core-truth/resnet2b-core-v1/truth.pt")
MODEL = Path("../vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx")
ARTIFACT = Path("artifacts/rvir-v4-native-kfsb/resnet2b-core-v1")
TAMPER_REPORT = ARTIFACT.parent / "resnet2b-core-v1-tamper-report.json"


@pytest.fixture(scope="module")
def formal_evaluation() -> tuple[dict[str, object], dict[str, object]]:
    return artifact_runner._build_evidence(
        artifact_runner._load_torch(ATOMIC_CAPTURE),
        artifact_runner._load_torch(WHOLE_TRUTH),
        MODEL,
    )


def test_native_kfsb_reproduces_masks_candidates_children_and_final_decision(
    formal_evaluation: tuple[dict[str, object], dict[str, object]],
) -> None:
    evaluation, summary = formal_evaluation
    typed = artifact_runner._evaluation_from_payload(evaluation)

    assert isinstance(typed, NativeKfsbEvaluationV4)
    assert summary["unstable_masks_exact"] is True
    assert summary["unstable_neuron_count"] == 4200
    assert summary["candidate_splits_exact"] is True
    assert summary["candidate_decision_count"] == 36
    assert summary["child_domain_evaluation_count"] == 72
    assert summary["child_lower_sign_exact"] is True
    assert summary["child_lower_maximum_absolute_difference"] < 1e-5
    assert summary["final_decision_exact"] is True
    assert typed.final_decision == (
        (5, 27),
        (5, 32),
        (5, 90),
        (5, 90),
        (5, 32),
        (5, 90),
    )
    assert summary["provider_core_callback_count"] == 0
    assert summary["provider_compute_bounds_callback_count"] == 0
    assert summary["provider_update_bounds_callback_count"] == 0
    assert summary["fallback_dispatch_count"] == 0
    assert summary["native_kfsb_admitted"] is True
    assert summary["whole_core_replacement_admitted"] is False
    assert summary["b2_same_solver_timing_admitted"] is False
    assert summary["performance_claimed"] is False


def test_native_kfsb_rejects_provider_callback_and_candidate_width(
    formal_evaluation: tuple[dict[str, object], dict[str, object]],
) -> None:
    evaluation, _summary = formal_evaluation
    typed = artifact_runner._evaluation_from_payload(evaluation)

    with pytest.raises(ValueError, match="contract differs"):
        replace(typed, provider_update_bounds_callback_count=1).validate()
    with pytest.raises(ValueError, match="candidate width differs"):
        replace(
            typed,
            candidate_splits=(typed.candidate_splits[0][:-1],)
            + typed.candidate_splits[1:],
        ).validate()


def test_native_kfsb_comparator_rejects_resigned_semantic_tamper(
    formal_evaluation: tuple[dict[str, object], dict[str, object]],
) -> None:
    evaluation, _summary = formal_evaluation
    typed = artifact_runner._evaluation_from_payload(evaluation)
    changed = typed.candidate_child_lowers[0].clone()
    changed.flatten()[0] += 1.0

    with pytest.raises(ValueError, match="child numeric parity differs"):
        compare_rvir_v4_native_kfsb(
            replace(
                typed,
                candidate_child_lowers=(changed,) + typed.candidate_child_lowers[1:],
            ),
            expected_candidate_splits=typed.candidate_splits,
            expected_candidate_child_lowers=typed.candidate_child_lowers,
            expected_final_decision=typed.final_decision,
            expected_unstable_masks=typed.unstable_masks,
        )

    changed_mask = typed.unstable_masks["/input"].clone()
    changed_mask.flatten()[0] = torch.logical_not(changed_mask.flatten()[0])
    with pytest.raises(ValueError, match="parity differs"):
        compare_rvir_v4_native_kfsb(
            replace(
                typed,
                unstable_mask_by_provider_preactivation=tuple(
                    (name, changed_mask if name == "/input" else value)
                    for name, value in typed.unstable_mask_by_provider_preactivation
                ),
                unstable_counts=tuple(
                    int((changed_mask if name == "/input" else value).sum().item())
                    for name, value in typed.unstable_mask_by_provider_preactivation
                ),
            ),
            expected_candidate_splits=typed.candidate_splits,
            expected_candidate_child_lowers=typed.candidate_child_lowers,
            expected_final_decision=typed.final_decision,
            expected_unstable_masks=typed.unstable_masks,
        )


def test_formal_artifact_replays_and_rejects_all_tamper_probes() -> None:
    result = artifact_runner._replay(ARTIFACT, MODEL)
    report = json.loads(TAMPER_REPORT.read_text(encoding="utf-8"))

    assert result["status"] == "replay-passed"
    assert report["artifact_source_git_head"].startswith("a2097c0")
    assert report["attack_count"] == 8
    assert report["fully_resigned_evaluation_attack_count"] == 6
    assert report["all_rejected"] is True
    assert report["performance_claimed"] is False
