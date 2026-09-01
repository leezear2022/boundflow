"""Formal replay tests for the D1-B fixed schedule qualification."""

# pylint: disable=missing-function-docstring

from pathlib import Path
import json

import pytest

from scripts.run_r3_d1b_schedule_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d1b-schedule-formal-v1"


def test_r3d1b_formal_artifact_replays_isolated_claim_only() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-D1B formal artifact has not been generated")
    summary = replay(ARTIFACT)
    assert summary["d1b_schedule_qualification"] is True
    assert summary["isolated_gate_pass"] is True
    assert summary["isolated_performance_claimed"] is True
    assert summary["wrapper_performance_claimed"] is False


def test_r3d1b_formal_opens_wrapper_only_after_worst_gate() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-D1B formal artifact has not been generated")
    summary = replay(ARTIFACT)
    assert summary["worst_speedup"] >= summary["isolated_opportunity_gate"]
    assert summary["d1c_wrapper_open"] is True


def test_r3d1b_formal_tamper_report_rejects_all_cases() -> None:
    path = ARTIFACT / "tamper_report.json"
    if not path.exists():
        pytest.skip("R3-D1B tamper report has not been generated")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["case_count"] == 10
    assert report["rejected_count"] == 10
    assert all(row["rejected"] is True for row in report["cases"])
