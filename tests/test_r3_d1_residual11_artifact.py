"""Formal replay gate for D1-A residual11 correctness."""

from pathlib import Path
import json

import pytest

from scripts.run_r3_d1_residual11_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d1a-residual11-staged-v1"


def test_r3d1_residual11_formal_artifact_replays_without_timing() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-D1-A formal artifact is not generated yet")
    summary = replay(ARTIFACT)
    assert summary["d1a_residual11_correctness"] is True
    assert summary["timing_recorded"] is False
    assert summary["performance_claimed"] is False


def test_r3d1_residual11_formal_opens_only_residual6_correctness() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-D1-A formal artifact is not generated yet")
    summary = replay(ARTIFACT)
    assert summary["residual6_open"] is True
    assert summary["d1b_timing_open"] is False


def test_r3d1_residual11_tamper_report_rejects_all_cases() -> None:
    path = ARTIFACT / "tamper_report.json"
    if not path.exists():
        pytest.skip("R3-D1-A tamper report is not generated yet")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["case_count"] == 10
    assert report["rejected_count"] == 10
    assert all(row["rejected"] is True for row in report["cases"])
