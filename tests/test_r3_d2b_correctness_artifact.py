"""Formal replay gates for D2-B stepwise correctness."""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path

import pytest

from scripts.run_r3_d2b_correctness_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d2b-correctness-v1"


def test_r3d2b_formal_artifact_replays_without_timing() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-D2B correctness artifact has not been generated")
    summary = replay(ARTIFACT)
    assert summary["trajectory_correctness_admitted"] is True
    assert summary["ownership_admitted"] is True
    assert summary["timing_recorded"] is False
    assert summary["performance_claimed"] is False


def test_r3d2b_formal_opens_only_wrapper_timing() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-D2B correctness artifact has not been generated")
    summary = replay(ARTIFACT)
    assert summary["d2b_timing_open"] is True
    assert summary["r3_3_open"] is False
    assert summary["same_solver_open"] is False


def test_r3d2b_tamper_report_rejects_all_cases() -> None:
    path = ARTIFACT / "tamper_report.json"
    if not path.exists():
        pytest.skip("R3-D2B tamper report has not been generated")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["case_count"] == 12
    assert report["rejected_count"] == 12
    assert all(row["rejected"] is True for row in report["cases"])
