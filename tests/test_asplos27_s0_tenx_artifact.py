"""Artifact tests for ASPLOS'27 S0 attribution and 10x budgeting."""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path

import pytest

from boundflow.runtime.gpu_attribution import canonical_hash
from scripts.run_asplos27_s0_tenx_budget_artifact import (
    file_sha256,
    generate_artifact,
    replay_artifact,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_s0_artifact_generates_and_replays(tmp_path: Path) -> None:
    artifact = tmp_path / "s0"

    generated = generate_artifact(ROOT, artifact)
    replayed = replay_artifact(ROOT, artifact)

    assert generated == replayed
    assert replayed["status"] == "replay-passed"
    summary = json.loads((artifact / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "s0-attribution-not-admitted"
    assert summary["run_count"] == 10
    assert summary["admitted_run_count"] == 5
    assert summary["tenx_feasible_run_count"] == 0
    assert summary["s1_performance_gate_open"] is False
    assert summary["direct_ratios_aggregated"] is False
    assert summary["performance_claimed"] is False


def test_s0_artifact_rejects_resigned_semantic_tamper(tmp_path: Path) -> None:
    artifact = tmp_path / "s0"
    generate_artifact(ROOT, artifact)
    report_path = artifact / "budget_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "s0-attribution-admitted"
    report["report_hash"] = canonical_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    _write_json(report_path, report)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["budget_report.json"] = file_sha256(report_path)
    manifest["manifest_hash"] = canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="semantic replay differs"):
        replay_artifact(ROOT, artifact)


def test_s0_artifact_refuses_overwrite(tmp_path: Path) -> None:
    artifact = tmp_path / "s0"
    generate_artifact(ROOT, artifact)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_artifact(ROOT, artifact)
