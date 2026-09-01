"""Formal replay gates for MR0 explicit-event budget."""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path

import pytest

from scripts.run_mr0_explicit_event_budget_artifact import _portable_log, replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/measurement-recovery/mr0-explicit-event-budget-resnet2b-v1"


def test_mr0_log_portability() -> None:
    assert _portable_log(f"{ROOT}/worker.py") == "<repo>/worker.py"


def test_mr0_artifact_replays() -> None:
    if not ARTIFACT.exists():
        pytest.skip("MR0 artifact has not been generated")
    summary = replay(ARTIFACT)
    assert summary["worker_count"] == 5
    assert summary["decision_budget"] == 17
    assert summary["performance_claimed"] is False


def test_mr0_never_opens_same_solver_or_r2() -> None:
    if not ARTIFACT.exists():
        pytest.skip("MR0 artifact has not been generated")
    summary = replay(ARTIFACT)
    assert summary["same_solver_open"] is False
    assert summary["r2_open"] is False


def test_mr0_tamper_rejects_all_cases() -> None:
    path = ARTIFACT / "tamper_report.json"
    if not path.exists():
        pytest.skip("MR0 tamper report has not been generated")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["case_count"] == 12
    assert report["rejected_count"] == 12
