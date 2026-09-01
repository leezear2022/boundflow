"""Formal replay gates for D2-A backward attribution."""

# pylint: disable=missing-function-docstring

from pathlib import Path
import json

import pytest

from scripts.run_r3_d2a_backward_attribution_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d2a-backward-attribution-v1"


def test_r3d2a_formal_artifact_replays_as_diagnostic_only() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-D2A formal artifact has not been generated")
    summary = replay(ARTIFACT)
    assert summary["diagnostic_only"] is True
    assert summary["performance_claimed"] is False
    assert summary["r3_3_open"] is False
    assert summary["same_solver_open"] is False


def test_r3d2a_formal_opens_only_a_quantified_d2b_route() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-D2A formal artifact has not been generated")
    summary = replay(ARTIFACT)
    if summary["d2b_open"]:
        assert summary["selected_route"] == "coefficient-sign-staged-residual-reuse"
        assert summary["d1b_residual_signature_mapping"] is True
        assert summary["route_table"]["coefficient_sign"]["admitted"] is True


def test_r3d2a_formal_tamper_report_rejects_all_cases() -> None:
    path = ARTIFACT / "tamper_report.json"
    if not path.exists():
        pytest.skip("R3-D2A tamper report has not been generated")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["case_count"] == 14
    assert report["rejected_count"] == 14
    assert all(row["rejected"] is True for row in report["cases"])
