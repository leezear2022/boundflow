"""Formal replay gates for R3-3 active-beta isolated timing."""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path

import pytest

from scripts.run_r3_3_active_beta_timing_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-3-active-beta-timing-v1"


def test_r3_3_active_beta_timing_artifact_replays() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-3 active-beta timing artifact has not been generated")
    summary = replay(ARTIFACT)
    assert summary["worker_count"] == 6
    assert summary["pair_count"] == 180
    assert summary["timing_closure_pending_tamper"] is True
    assert summary["performance_claimed"] is False


def test_r3_3_active_beta_timing_does_not_open_later_scopes() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-3 active-beta timing artifact has not been generated")
    summary = replay(ARTIFACT)
    assert summary["r3_4_open"] is False
    assert summary["same_solver_open"] is False


def test_r3_3_active_beta_timing_tamper_rejects_all_cases() -> None:
    path = ARTIFACT / "tamper_report.json"
    if not path.exists():
        pytest.skip("R3-3 active-beta timing tamper has not been generated")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["case_count"] == 12
    assert report["rejected_count"] == 12
