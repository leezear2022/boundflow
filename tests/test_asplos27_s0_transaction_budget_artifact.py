"""Replay and tamper tests for the S0 explicit-transaction 10x budget."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from boundflow.runtime.gpu_attribution import canonical_hash
from scripts import run_asplos27_s0_transaction_budget_artifact as runner

ARTIFACT = Path("artifacts/asplos27-s0-transaction-budget") / (
    "official-b0-five-pair-v1"
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(runner.canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _resign_manifest(artifact: Path, changed_file: str) -> None:
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][changed_file] = runner.file_sha256(artifact / changed_file)
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact / "manifest.json", manifest)


def test_formal_transaction_budget_artifact_replays() -> None:
    result = runner.replay_artifact(Path.cwd(), ARTIFACT)

    assert result["status"] == "replay-passed"
    assert result["evidence_status"] == "s0-transaction-budget-research-route-open"
    assert result["s1_implementation_open"] is True
    assert result["s1_performance_gate_open"] is False
    assert result["performance_claimed"] is False
    summary = json.loads((ARTIFACT / "summary.json").read_text(encoding="utf-8"))
    assert summary["integration_overhead_share"] == 0.0


def test_fully_resigned_projection_tamper_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, artifact)
    report = json.loads((artifact / "budget_report.json").read_text(encoding="utf-8"))
    report["workloads"][0]["projected_speedup_hypothesis"] = 100.0
    report.pop("report_hash")
    report["report_hash"] = canonical_hash(report)
    _write_json(artifact / "budget_report.json", report)
    _resign_manifest(artifact, "budget_report.json")

    with pytest.raises(ValueError, match="semantic replay differs"):
        runner.replay_artifact(Path.cwd(), artifact)


def test_fully_resigned_axis_target_tamper_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, artifact)
    protocol = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    protocol["axis_policies"][0]["target_speedup"] = 160.0
    protocol.pop("protocol_hash")
    protocol["protocol_hash"] = canonical_hash(protocol)
    _write_json(artifact / "protocol.json", protocol)
    _resign_manifest(artifact, "protocol.json")

    with pytest.raises(ValueError, match="semantic replay differs"):
        runner.replay_artifact(Path.cwd(), artifact)
