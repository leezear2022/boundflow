"""Formal replay gates for the D1-C cumulative wrapper artifact."""

# pylint: disable=missing-function-docstring

from pathlib import Path
import json

import pytest

from scripts.run_r3_d1c_wrapper_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d1c-wrapper-formal-v1"


def test_r3d1c_formal_artifact_replays_without_claim_drift() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-D1C formal artifact has not been generated")
    summary = replay(ARTIFACT)
    assert summary["performance_claimed"] is summary["d1c_go"]
    assert summary["r3_3_open"] is summary["d1c_go"]


def test_r3d1c_formal_no_go_opens_only_backward_attribution() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-D1C formal artifact has not been generated")
    summary = replay(ARTIFACT)
    if summary["d1c_go"] is False:
        assert summary["status"] == "VALIDATED-NO-GO-R3-D1C-CUMULATIVE-WRAPPER"
        assert summary["backward_attribution_open"] is True
        assert summary["r3_3_open"] is False


def test_r3d1c_formal_tamper_report_rejects_all_cases() -> None:
    path = ARTIFACT / "tamper_report.json"
    if not path.exists():
        pytest.skip("R3-D1C tamper report has not been generated")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["case_count"] >= 10
    assert report["rejected_count"] == report["case_count"]
    assert all(row["rejected"] is True for row in report["cases"])
