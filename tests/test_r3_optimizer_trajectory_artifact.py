"""R3-2A formal artifact replay gate."""

# pylint: disable=missing-function-docstring,duplicate-code

import json
from pathlib import Path

import pytest

from scripts.run_r3_optimizer_trajectory_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-2a-optimizer-trajectory-v1"


def test_r32a_formal_artifact_replays() -> None:
    if not ARTIFACT.is_dir():
        pytest.skip("R3-2A formal artifact is not generated yet")
    summary = replay(ARTIFACT)
    assert summary["trajectory_correctness_admitted"] is True
    assert summary["r3_2b_open"] is True
    assert summary["performance_claimed"] is False


def test_r32a_formal_tamper_report_rejects_all_cases() -> None:
    report_path = ARTIFACT / "tamper_report.json"
    if not report_path.is_file():
        pytest.skip("R3-2A tamper report is not generated yet")
    report = json.loads(report_path.read_text())
    assert report["case_count"] == report["rejected_count"] == 12
    assert report["all_rejected"] is True
    assert report["performance_claimed"] is False
