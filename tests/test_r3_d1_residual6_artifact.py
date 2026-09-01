"""Formal replay gate for D1-A residual6 correctness."""

# pylint: disable=missing-function-docstring

from pathlib import Path
import json

import pytest

from scripts.run_r3_d1_residual6_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d1a-residual6-staged-v1"


def test_r3d1_residual6_formal_artifact_replays_without_timing() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-D1-A residual6 formal artifact is not generated yet")
    summary = replay(ARTIFACT)
    assert summary["d1a_residual6_correctness"] is True
    assert summary["timing_recorded"] is False
    assert summary["performance_claimed"] is False


def test_r3d1_residual6_formal_opens_only_schedule_qualification() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-D1-A residual6 formal artifact is not generated yet")
    summary = replay(ARTIFACT)
    assert summary["d1b_schedule_qualification_open"] is True
    assert summary["d1c_wrapper_open"] is False


def test_r3d1_residual6_tamper_report_rejects_all_cases() -> None:
    path = ARTIFACT / "tamper_report.json"
    if not path.exists():
        pytest.skip("R3-D1-A residual6 tamper report is not generated yet")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["case_count"] == 10
    assert report["rejected_count"] == 10
    assert all(row["rejected"] is True for row in report["cases"])
