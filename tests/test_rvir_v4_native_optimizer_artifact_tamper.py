"""Formal replay/tamper test for the RVIR-v4 V4-2D artifact."""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path

from scripts.probe_rvir_v4_native_optimizer_artifact_tamper import run_probe_suite

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/rvir-v4-native-optimizer/resnet2b-core-step-parity-v1"
REPORT = ARTIFACT.parent / "resnet2b-core-step-parity-v1-tamper-report.json"


def test_native_optimizer_artifact_rejects_fully_resigned_tamper() -> None:
    report = run_probe_suite(ARTIFACT)

    assert report == json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["probe_count"] == 6
    assert report["all_outer_provenance_gates_rejected"] is True
    assert report["all_semantic_mutation_gates_rejected"] is True
    assert report["performance_claimed"] is False
    original = report["original_replay"]
    assert isinstance(original, dict)
    assert original["status"] == "replay-passed"
    assert original["native_trace_hash"] == (
        "d53cc7fcb7d23958c99c0bc188df11e50b70a7cae97edf0bd64e345a4fee7c8c"
    )
