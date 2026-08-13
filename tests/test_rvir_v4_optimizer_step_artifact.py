"""Formal artifact replay/tamper tests for RVIR-v4 V4-2B."""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path

from scripts.probe_rvir_v4_optimizer_step_artifact_tamper import run_probe_suite

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    REPOSITORY_ROOT / "artifacts/rvir-v4-optimizer-step/resnet2b-core-step-trace-v1"
)
REPORT = ARTIFACT.parent / "resnet2b-core-step-trace-v1-tamper-report.json"


def test_formal_optimizer_step_artifact_replays_and_rejects_resigned_tamper() -> None:
    report = run_probe_suite(ARTIFACT)

    assert report == json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["probe_count"] == 5
    assert report["all_probes_rejected"] is True
    assert report["performance_claimed"] is False
    original = report["original_replay"]
    assert isinstance(original, dict)
    assert original["status"] == "replay-passed"
    assert original["evaluation_count"] == 10
    assert original["update_count"] == 9
