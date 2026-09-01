"""Replay and tamper tests for the frozen ASPLOS'27 S0 transaction artifact."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from boundflow.runtime.gpu_attribution import canonical_hash
from scripts import run_asplos27_s0_transaction_markers as runner

ARTIFACT = Path("artifacts/asplos27-s0-transactions/official-b0-five-pair-v1")


def _write_json(path: Path, value: object) -> None:
    path.write_text(runner.canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _resign_manifest(artifact: Path, changed_file: str) -> None:
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][changed_file] = runner.file_sha256(artifact / changed_file)
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact / "manifest.json", manifest)


def test_formal_transaction_artifact_replays() -> None:
    result = runner.replay_artifact(ARTIFACT)

    assert result["status"] == "replay-passed"
    assert result["evidence_status"] == "s0-explicit-transactions-admitted"
    assert result["pair_count"] == 10
    assert result["budget_recompute_open"] is True
    assert result["performance_claimed"] is False
    summary = json.loads((ARTIFACT / "summary.json").read_text(encoding="utf-8"))
    assert summary["workloads"]["cifar10_resnet:000"][
        "maximum_perturbation_ratio"
    ] == pytest.approx(1.0416399732653616)
    assert summary["workloads"]["mnistfc:2"][
        "maximum_perturbation_ratio"
    ] == pytest.approx(1.0653544672265136)


def test_fully_resigned_worker_summary_tamper_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, artifact)
    rows = runner._read_jsonl(artifact / "worker_runs.jsonl")
    profile = next(row for row in rows if row["mode"] == "profile")
    profile["transaction_summary"]["mechanism_coverage_share"] = 1.0
    profile.pop("worker_hash")
    profile["worker_hash"] = canonical_hash(profile)
    runner._write_jsonl(artifact / "worker_runs.jsonl", rows)
    _resign_manifest(artifact, "worker_runs.jsonl")

    with pytest.raises(ValueError, match="semantic summary differs"):
        runner.replay_artifact(artifact)


def test_fully_resigned_protocol_target_tamper_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, artifact)
    protocol = json.loads((artifact / "protocol.json").read_text(encoding="utf-8"))
    protocol["target_inventory"][0]["resolution"] = "coarse_scope"
    protocol.pop("protocol_hash")
    protocol["protocol_hash"] = canonical_hash(protocol)
    _write_json(artifact / "protocol.json", protocol)
    _resign_manifest(artifact, "protocol.json")

    with pytest.raises(ValueError, match="protocol semantics differ"):
        runner.replay_artifact(artifact)
