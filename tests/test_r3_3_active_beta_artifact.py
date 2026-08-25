"""Formal replay gates for R3-3 S-anchor active-beta correctness."""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path

import pytest

from scripts.run_r3_3_active_beta_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-3-active-beta-correctness-v1"


def test_r3_3_active_beta_artifact_replays_without_timing() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-3 active-beta artifact has not been generated")
    summary = replay(ARTIFACT)
    assert summary["worker_count"] == 5
    assert summary["metric_count"] == 20
    assert summary["beta_nonzero_count"] == 30
    assert summary["active_beta_correctness_admitted"] is True
    assert summary["ownership_admitted"] is True
    assert summary["timing_recorded"] is False
    assert summary["performance_claimed"] is False


def test_r3_3_opens_only_isolated_timing() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-3 active-beta artifact has not been generated")
    summary = replay(ARTIFACT)
    assert summary["isolated_timing_open"] is True
    assert summary["r3_4_open"] is False
    assert summary["same_solver_open"] is False


def test_r3_3_tamper_report_rejects_all_cases() -> None:
    path = ARTIFACT / "tamper_report.json"
    if not path.exists():
        pytest.skip("R3-3 active-beta tamper report has not been generated")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["case_count"] == 12
    assert report["rejected_count"] == 12
    assert all(row["rejected"] is True for row in report["cases"])
