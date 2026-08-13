"""Formal artifact contracts for RVIR-v4 V4-3E five-fresh correctness."""

# pylint: disable=missing-function-docstring,protected-access

import json
from pathlib import Path

from scripts import run_rvir_v4_five_fresh_artifact as artifact_runner

ARTIFACT = Path("artifacts/rvir-v4-five-fresh/resnet2b-prop0-v1")
TAMPER_REPORT = ARTIFACT.parent / "resnet2b-prop0-v1-tamper-report.json"


def test_formal_five_fresh_artifact_admits_all_counterbalanced_pairs() -> None:
    runs, summary, result = artifact_runner._verify_static_artifact(ARTIFACT)

    assert result["status"] == "replay-passed"
    assert summary["status"] == "validated-five-fresh-correctness"
    assert summary["sequence"] == list(artifact_runner.SEQUENCE)
    assert len(runs) == summary["run_count"] == 10
    assert summary["pair_count"] == 5
    assert summary["original_run_count"] == 5
    assert summary["candidate_run_count"] == 5
    assert summary["all_pairs_admitted"] is True
    assert all(pair["pair_admitted"] is True for pair in summary["pairs"])
    assert all(pair["solver_result_exact"] is True for pair in summary["pairs"])
    assert all(pair["queue_accounting_exact"] is True for pair in summary["pairs"])
    assert all(pair["branching_decision_exact"] is True for pair in summary["pairs"])
    assert summary["maximum_absolute_difference"] < 2e-4
    assert summary["tensor_comparison_count"] == 2255
    assert summary["sign_element_comparison_count"] == 1065300
    assert summary["all_sign_exact"] is True
    assert summary["accepted_domain_count_per_run"] == 6
    assert summary["pruned_domain_count_per_run"] == 0
    assert summary["candidate_provider_core_callback_count"] == 0
    assert summary["candidate_provider_compute_bounds_callback_count"] == 0
    assert summary["candidate_provider_update_bounds_callback_count"] == 0
    assert summary["candidate_fallback_dispatch_count"] == 0
    assert summary["five_fresh_correctness_admitted"] is True
    assert summary["whole_core_replacement_admitted"] is True
    assert summary["b2_same_solver_timing_admitted"] is True
    assert summary["performance_claimed"] is False


def test_formal_five_fresh_tamper_report_rejects_all_attacks() -> None:
    report = json.loads(TAMPER_REPORT.read_text(encoding="utf-8"))

    assert report["attack_count"] == 6
    assert report["inner_truth_resigned_attack_count"] == 3
    assert report["outer_artifact_resigned_attack_count"] == 6
    assert report["all_rejected"] is True
    assert all(row["outer_artifact_resigned"] is True for row in report["attacks"])
    assert all(row["rejected"] is True for row in report["attacks"])
    assert report["performance_claimed"] is False
