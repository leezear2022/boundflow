"""Replay and fail-closed tamper tests for the RVIR-v4 V4-1 artifact."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/rvir-v4-frozen-state/resnet2b-core-v1"
MODEL = ROOT.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
RUNNER = ROOT / "scripts/run_rvir_v4_frozen_state_artifact.py"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replay(artifact: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            sys.executable,
            str(RUNNER),
            "replay",
            "--model",
            str(MODEL),
            "--artifact-dir",
            str(artifact),
        ),
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _resign_manifest(artifact: Path, *names: str) -> None:
    path = artifact / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name in names:
        manifest["files"][name] = _sha256(artifact / name)
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = hashlib.sha256(
        _canonical(semantic).encode("utf-8")
    ).hexdigest()
    path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_rvir_v4_frozen_state_artifact_replays() -> None:
    completed = _replay(ARTIFACT)

    assert completed.returncode == 0, completed.stdout
    result = json.loads(completed.stdout)
    assert result["status"] == "replay-passed"
    assert result["summary_hash"] == (
        "3541318b226ffd28cad0862e1b43055cc701d0973144cb58f4e17122a49f60e9"
    )
    assert result["lower_max_abs_diff"] == 2.0265579223632812e-06
    assert result["performance_claimed"] is False


def test_rvir_v4_frozen_topology_tamper_rejected_after_resigning(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, artifact)
    path = artifact / "topology.json"
    topology = json.loads(path.read_text(encoding="utf-8"))
    topology["rows"][0]["provider_activation"] = "/tampered"
    topology["topology_hash"] = hashlib.sha256(
        _canonical(topology["rows"]).encode("utf-8")
    ).hexdigest()
    path.write_text(
        json.dumps(topology, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _resign_manifest(artifact, "topology.json")

    completed = _replay(artifact)

    assert completed.returncode != 0
    assert "RVIR-v4 frozen topology semantic mapping differs" in completed.stdout


def test_rvir_v4_frozen_state_tamper_rejected_after_resigning(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, artifact)
    capture_path = artifact / "capture.pt"
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    capture["cores"][0]["lower"].view(-1)[0] += 1.0
    torch.save(capture, capture_path)
    source_path = artifact / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["capture_sha256"] = _sha256(capture_path)
    source_path.write_text(
        json.dumps(source, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _resign_manifest(artifact, "capture.pt", "source.json")

    completed = _replay(artifact)

    assert completed.returncode != 0
    assert "RVIR-v4 frozen source identity differs" in completed.stdout


def test_rvir_v4_frozen_numeric_replay_accepts_preregistered_drift(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, artifact)
    path = artifact / "execution.json"
    execution = json.loads(path.read_text(encoding="utf-8"))
    execution["native_lower"][0] += 1e-6
    path.write_text(
        json.dumps(execution, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _resign_manifest(artifact, "execution.json")

    completed = _replay(artifact)

    assert completed.returncode == 0, completed.stdout


def test_rvir_v4_frozen_numeric_replay_rejects_out_of_tolerance_drift(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, artifact)
    path = artifact / "execution.json"
    execution = json.loads(path.read_text(encoding="utf-8"))
    execution["native_lower"][0] += 1e-2
    path.write_text(
        json.dumps(execution, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _resign_manifest(artifact, "execution.json")

    completed = _replay(artifact)

    assert completed.returncode != 0
    assert "RVIR-v4 frozen execution numeric lower differs" in completed.stdout
