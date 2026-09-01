"""Replay and semantic-tamper tests for the frozen RVIR-v4 capture."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/rvir-v4-production-state/resnet2b-core-capture-v2"
RUNNER = ROOT / "scripts/run_rvir_v4_production_state_capture.py"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replay(artifact: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, str(RUNNER), "replay", "--artifact-dir", str(artifact)),
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def test_frozen_rvir_v4_production_capture_replays() -> None:
    completed = _replay(ARTIFACT)

    assert completed.returncode == 0, completed.stdout
    result = json.loads(completed.stdout)
    assert result == {
        "call_count": 24,
        "core_count": 1,
        "evidence_status": "validated_corrected_capture",
        "performance_claimed": False,
        "status": "replay-passed",
        "summary_hash": (
            "9d1c71b02c42852d0e1a03ffec831b0a43a4c8d61003c125ad05371993d1dbdb"
        ),
    }


def test_frozen_rvir_v4_tensor_tamper_rejected_after_outer_digest_resigning(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, artifact)
    capture_path = artifact / "capture.pt"
    payload = torch.load(capture_path, map_location="cpu", weights_only=True)
    tensor = payload["cores"][0]["pre_snapshot"]["tensors"][0]["value"]
    tensor.view(-1)[0] += 1.0
    torch.save(payload, capture_path)

    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["capture.pt"] = _sha256(capture_path)
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = hashlib.sha256(
        _canonical(semantic).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    completed = _replay(artifact)

    assert completed.returncode != 0
    assert "RVIR-v4 production tensor content differs" in completed.stdout


def test_frozen_rvir_v4_alpha_index_semantic_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, artifact)
    capture_path = artifact / "capture.pt"
    payload = torch.load(capture_path, map_location="cpu", weights_only=True)
    snapshot = payload["cores"][0]["pre_snapshot"]
    item = next(
        tensor
        for tensor in snapshot["tensors"]
        if tensor["role"] == "alpha_feature_index"
    )
    item["value"].view(-1)[0] = 10**6
    item["content_sha256"] = production_tensor_sha256(item["value"])
    metadata = {
        "schema_version": snapshot["schema_version"],
        "snapshot_id": snapshot["snapshot_id"],
        "tensors": [
            {key: value for key, value in tensor.items() if key != "value"}
            for tensor in snapshot["tensors"]
        ],
        "history": snapshot["history"],
        "optimizer_policy": snapshot["optimizer_policy"],
    }
    snapshot["snapshot_hash"] = hashlib.sha256(
        _canonical(metadata).encode("utf-8")
    ).hexdigest()
    torch.save(payload, capture_path)

    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["capture.pt"] = _sha256(capture_path)
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = hashlib.sha256(
        _canonical(semantic).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    completed = _replay(artifact)

    assert completed.returncode != 0
    assert "RVIR-v4 alpha feature index content differs" in completed.stdout
