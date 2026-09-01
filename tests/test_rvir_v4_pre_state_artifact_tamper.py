"""Formal artifact replay/tamper tests for RVIR-v4 V4-2C."""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path

from scripts.probe_rvir_v4_pre_state_artifact_tamper import run_probe_suite

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPOSITORY_ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1"
REPORT = ARTIFACT.parent / "resnet2b-core-pre-state-v1-tamper-report.json"


def test_formal_pre_state_artifact_rejects_fully_resigned_tamper() -> None:
    report = run_probe_suite(ARTIFACT)

    assert report == json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["probe_count"] == 6
    assert report["all_outer_provenance_gates_rejected"] is True
    assert report["all_semantic_identity_gates_rejected"] is True
    assert report["performance_claimed"] is False
    original = report["original_replay"]
    assert isinstance(original, dict)
    assert original["status"] == "replay-passed"
    assert original["mapping_hash"] == (
        "cfcebf92fc58c269899d98cd65cc9454d7caa6051e2c9da46d415eda1fecf8df"
    )
