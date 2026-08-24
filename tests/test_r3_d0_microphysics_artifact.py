"""Formal replay gates for the R3-D0 microphysics artifact."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from scripts.run_r3_d0_microphysics_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d0-microphysics-v1"


def test_r3d0_formal_artifact_replays() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-D0 formal artifact is not generated yet")
    summary = replay(ARTIFACT)
    assert summary["performance_claimed"] is False
    assert summary["r3_3_open"] is False


def test_r3d0_formal_artifact_never_opens_graph_without_capture_gate() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-D0 formal artifact is not generated yet")
    summary = replay(ARTIFACT)
    assert summary["graph_route_open"] is False


def test_r3d0_fully_resigned_tamper_report_rejects_all_cases() -> None:
    path = ARTIFACT / "tamper_report.json"
    if not path.exists():
        pytest.skip("R3-D0 tamper report is not generated yet")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["case_count"] == 12
    assert report["rejected_count"] == 12
    assert all(row["rejected"] is True for row in report["cases"])
