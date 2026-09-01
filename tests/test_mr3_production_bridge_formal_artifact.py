"""Repository replay tests for the frozen MR3 production bridge artifact."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
from pathlib import Path

from boundflow.runtime.mr3_production_bridge_formal import STATUS
from scripts.run_mr3_production_bridge_formal import replay_artifact

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "artifacts/measurement-recovery/mr3-p-production-bridge-correctness-v1"
)


def test_frozen_mr3_production_bridge_artifact_replays() -> None:
    summary = replay_artifact(ARTIFACT)
    assert summary["status"] == STATUS
    assert summary["pair_count"] == 5
    assert summary["candidate_forward_count"] == 50


def test_frozen_mr3_tamper_report_is_manifest_bound() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((ARTIFACT / "tamper_report.json").read_text(encoding="utf-8"))
    assert report["fully_resigned"] is True
    assert report["rejected_count"] == report["case_count"] == 18
    assert "tamper_report.json" in manifest["files"]
