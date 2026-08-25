"""Formal replay gates for R3-3 isolated attribution."""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path

import pytest

from scripts.run_r3_3_isolated_attribution_artifact import _portable_log, replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-3-isolated-attribution-v1"


def test_r3_3_isolated_attribution_log_portability() -> None:
    value = _portable_log(f"{ROOT}/worker.py")
    assert str(ROOT) not in value
    assert value == "<repo>/worker.py"


def test_r3_3_isolated_attribution_artifact_replays() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-3 isolated attribution artifact has not been generated")
    summary = replay(ARTIFACT)
    assert summary["worker_count"] == 5
    assert summary["sample_count"] == 150
    assert summary["attribution_closure_pending_tamper"] is True
    assert summary["performance_claimed"] is False


def test_r3_3_isolated_attribution_never_opens_later_scopes() -> None:
    if not ARTIFACT.exists():
        pytest.skip("R3-3 isolated attribution artifact has not been generated")
    summary = replay(ARTIFACT)
    assert summary["r3_4_open"] is False
    assert summary["same_solver_open"] is False


def test_r3_3_isolated_attribution_tamper_rejects_all_cases() -> None:
    path = ARTIFACT / "tamper_report.json"
    if not path.exists():
        pytest.skip("R3-3 isolated attribution tamper has not been generated")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["case_count"] == 12
    assert report["rejected_count"] == 12
