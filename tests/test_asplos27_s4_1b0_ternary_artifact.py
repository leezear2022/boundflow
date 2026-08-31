"""Frozen S4-1B0 artifact replay and claim gates."""

# pylint: disable=missing-function-docstring,import-error,duplicate-code

import json
from pathlib import Path

import pytest

from scripts.probe_asplos27_s4_1b0_ternary_tamper import run as run_tamper
from scripts.replay_asplos27_s4_1b0_ternary_stdlib import validate

ARTIFACT = Path("artifacts/asplos27-s4-1b0-ternary/resnet2b-prop0-v1")


@pytest.mark.skipif(not ARTIFACT.exists(), reason="formal artifact not generated")
def test_s4_1b0_formal_artifact_replays_from_raw() -> None:
    receipt = validate(ARTIFACT)
    assert receipt["status"] == "PASS"
    assert receipt["worker_count"] == 11
    assert receipt["performance_claimed"] is False


@pytest.mark.skipif(not ARTIFACT.exists(), reason="formal artifact not generated")
def test_s4_1b0_formal_summary_keeps_claims_closed() -> None:
    summary = json.loads((ARTIFACT / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FORMAL-CANDIDATE-PASS-PENDING-EXTERNAL-AUDIT-S4-1B0"
    assert summary["timing_recorded"] is False
    assert summary["performance_claimed"] is False


@pytest.mark.skipif(not ARTIFACT.exists(), reason="formal artifact not generated")
def test_s4_1b0_ten_outer_resigned_tampers_are_rejected() -> None:
    result = run_tamper(ARTIFACT)
    assert result["case_count"] == result["rejected"] == 10
