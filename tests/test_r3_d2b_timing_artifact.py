"""Formal replay gates for D2-B wrapper timing."""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path

import pytest

from scripts.run_r3_d2b_timing_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d2b-wrapper-timing-v1"


def test_r3d2b_timing_replays_as_pending_until_closure() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-D2B timing artifact has not been generated")
    summary = replay(ARTIFACT)
    assert summary["research_gate"] is True
    assert summary["timing_closure_pending_tamper"] is True
    assert summary["performance_claimed"] is False
    assert summary["r3_3_open"] is False


def test_r3d2b_timing_tamper_rejects_all_cases() -> None:
    path = ARTIFACT / "tamper_report.json"
    if not path.exists():
        pytest.skip("R3-D2B timing tamper has not been generated")
    report = json.loads(path.read_text())
    assert report["case_count"] == report["rejected_count"] == 12
    assert all(row["rejected"] is True for row in report["cases"])
