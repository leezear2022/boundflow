"""Formal artifact contracts for RVIR-v4 V4-3D live return."""

# pylint: disable=missing-function-docstring,protected-access

import json
from pathlib import Path

from scripts import run_rvir_v4_live_return_artifact as artifact_runner

ARTIFACT = Path("artifacts/rvir-v4-live-return/resnet2b-core-v1")
TAMPER_REPORT = ARTIFACT.parent / "resnet2b-core-v1-tamper-report.json"


def test_formal_live_return_artifact_replays_complete_cuda_core_and_post() -> None:
    live, _truth, summary, result = artifact_runner._verify_static_artifact(ARTIFACT)

    assert result["status"] == "replay-passed"
    assert summary["status"] == "validated-live-return-result"
    assert summary["native_devices"] == ["cuda:0"]
    assert summary["committed_path_count"] == 12
    assert summary["changed_path_count"] == 7
    assert summary["atomic_live_and_host_commit"] is True
    assert summary["official_post_queue_consumed"] is True
    assert summary["branching_decision"] == [
        [5, 27],
        [5, 32],
        [5, 90],
        [5, 90],
        [5, 32],
        [5, 90],
    ]
    assert summary["provider_core_callback_count"] == 0
    assert summary["provider_compute_bounds_callback_count"] == 0
    assert summary["provider_update_bounds_callback_count"] == 0
    assert summary["fallback_dispatch_count"] == 0
    parity = summary["semantic_parity"]
    assert parity["tensor_count"] == 451
    assert parity["sign_element_count"] == 213060
    assert parity["sign_exact"] is True
    assert parity["max_abs_diff"] < 2e-4
    assert parity["whole_core_replacement_admitted"] is True
    assert summary["five_fresh_correctness_admitted"] is False
    assert summary["b2_same_solver_timing_admitted"] is False
    assert summary["performance_claimed"] is False
    assert live["solver_result"] == {
        "status": "verified",
        "success": True,
        "visited_domains": [6],
    }


def test_formal_live_return_tamper_report_rejects_all_fully_resigned_attacks() -> None:
    report = json.loads(TAMPER_REPORT.read_text(encoding="utf-8"))

    assert report["attack_count"] == 8
    assert report["fully_resigned_attack_count"] == 8
    assert report["all_rejected"] is True
    assert all(row["fully_resigned"] is True for row in report["attacks"])
    assert all(row["rejected"] is True for row in report["attacks"])
    assert report["clean_fresh_semantic_parity"]["tensor_count"] == 451
    assert report["clean_fresh_semantic_parity"]["sign_exact"] is True
    assert report["performance_claimed"] is False
