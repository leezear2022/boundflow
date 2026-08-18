"""Formal artifact gates for B4-B0 five-fresh production capture."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

import json
from pathlib import Path

from scripts import run_fsg4_b4b_five_fresh_artifact as artifact

ARTIFACT = Path("artifacts/fsg4-b4b-five-fresh/resnet2b-prop0-v1")
TAMPER = ARTIFACT.parent / "resnet2b-prop0-v1-tamper-report.json"


def test_b4b0_five_fresh_artifact_replays_all_raw_captures() -> None:
    runs, summary, result = artifact._verify_static_artifact(ARTIFACT)

    assert result["status"] == "replay-passed"
    assert len(runs) == summary["run_count"] == 5
    assert summary["capture_count"] == 10
    assert summary["semantic_anchor_count"] == 5
    assert summary["performance_anchor_count"] == 5
    assert summary["all_discrete_structure_exact"] is True
    assert summary["all_numeric_within_tolerance"] is True
    assert summary["all_sign_exact"] is True
    assert summary["root_raw_replay_passed"] is True
    assert summary["maximum_absolute_difference"] <= artifact.ATOL
    assert summary["performance_claimed"] is False
    assert summary["tir_admitted"] is False


def test_b4b0_five_fresh_tamper_report_rejects_all_attacks() -> None:
    report = json.loads(TAMPER.read_text(encoding="utf-8"))

    assert report["attack_count"] == 9
    assert report["rejected_count"] == 9
    assert all(row["outer_resigned"] is True for row in report["rows"])
    assert all(row["rejected"] is True for row in report["rows"])
    assert report["performance_claimed"] is False
    assert report["tir_admitted"] is False
