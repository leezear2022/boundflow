#!/usr/bin/env python3
"""Run synchronized-rehash tamper probes against an RVIR-v4 pre-state artifact."""

# pylint: disable=protected-access,wrong-import-position,duplicate-code
# pylint: disable=too-many-locals,import-outside-toplevel

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
    ProductionReluTopologyV4,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
    production_tensor_sha256,
)
from scripts import run_rvir_v4_pre_state_artifact as artifact_runner

REPORT_SCHEMA_VERSION = "boundflow.rvir-v4-pre-state-tamper-report/v1"
MODEL = (
    REPOSITORY_ROOT.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)
Mutator = Callable[[dict[str, Any]], None]


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root differs: {path}")
    return value


def _load_capture(path: Path) -> dict[str, Any]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("RVIR-v4 pre-state tamper capture root differs")
    return value


def _pre_snapshot(capture: Mapping[str, Any]) -> dict[str, Any]:
    cores = capture.get("cores")
    if not isinstance(cores, list) or len(cores) != 1:
        raise ValueError("RVIR-v4 pre-state tamper core inventory differs")
    core = cores[0]
    if not isinstance(core, dict) or not isinstance(core.get("pre_snapshot"), dict):
        raise TypeError("RVIR-v4 pre-state tamper snapshot differs")
    return cast(dict[str, Any], core["pre_snapshot"])


def _tensor(snapshot: Mapping[str, Any], path: str) -> dict[str, Any]:
    tensors = snapshot.get("tensors")
    if not isinstance(tensors, list):
        raise TypeError("RVIR-v4 pre-state tamper tensor inventory differs")
    matches = [row for row in tensors if row.get("semantic_path") == path]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ValueError(f"RVIR-v4 pre-state tamper tensor path differs: {path}")
    return cast(dict[str, Any], matches[0])


def _resign_tensor(row: dict[str, Any]) -> None:
    row["content_sha256"] = production_tensor_sha256(row["value"])


def _resign_snapshot(snapshot: dict[str, Any]) -> None:
    semantic = {
        "schema_version": snapshot["schema_version"],
        "snapshot_id": snapshot["snapshot_id"],
        "tensors": [
            {key: value for key, value in row.items() if key != "value"}
            for row in snapshot["tensors"]
        ],
        "history": snapshot["history"],
        "optimizer_policy": snapshot["optimizer_policy"],
    }
    snapshot["snapshot_hash"] = _canonical_hash(semantic)


def _mutate_alpha_index(capture: dict[str, Any]) -> None:
    snapshot = _pre_snapshot(capture)
    prefix = "alpha_layout/%2Finput-4/feature_index/"
    rows = [
        row
        for row in snapshot["tensors"]
        if str(row.get("semantic_path", "")).startswith(prefix)
    ]
    if len(rows) != 3:
        raise ValueError("RVIR-v4 pre-state alpha index probe inventory differs")
    for row in rows:
        value = row["value"].clone()
        first = value[0].clone()
        value[0] = value[1]
        value[1] = first
        row["value"] = value
        _resign_tensor(row)
    _resign_snapshot(snapshot)


def _mutate_history_score(capture: dict[str, Any]) -> None:
    snapshot = _pre_snapshot(capture)
    entry = next(row for row in snapshot["history"] if row["locations"])
    entry["scores"] = [0.25] * len(entry["locations"])
    _resign_snapshot(snapshot)


def _mutate_intermediate(capture: dict[str, Any]) -> None:
    snapshot = _pre_snapshot(capture)
    row = _tensor(snapshot, "intermediate/%2F39/lower")
    value = row["value"].clone()
    value.reshape(-1)[0] += 0.125
    row["value"] = value
    _resign_tensor(row)
    _resign_snapshot(snapshot)


def _mutate_upper_alpha(capture: dict[str, Any]) -> None:
    snapshot = _pre_snapshot(capture)
    row = _tensor(snapshot, "alpha/%2Finput-4/%2F49")
    value = row["value"].clone()
    value[1, 0, 0, 0] += 0.125
    row["value"] = value
    _resign_tensor(row)
    _resign_snapshot(snapshot)


def _mutate_beta_location(capture: dict[str, Any]) -> None:
    snapshot = _pre_snapshot(capture)
    row = _tensor(snapshot, "beta/%2Finput-28/0/location")
    value = row["value"].clone()
    value[0, 0] += 1
    row["value"] = value
    _resign_tensor(row)
    entry = next(
        item
        for item in snapshot["history"]
        if item["domain_ordinal"] == 0 and item["layer_name"] == "/input-28"
    )
    entry["locations"][0] += 1
    _resign_snapshot(snapshot)


CAPTURE_PROBES: tuple[tuple[str, str, Mutator], ...] = (
    (
        "alpha-index-internal-rehash",
        "RVIR-v4 pre-state frozen identity differs",
        _mutate_alpha_index,
    ),
    (
        "history-score-internal-rehash",
        "RVIR-v4 pre-state frozen identity differs",
        _mutate_history_score,
    ),
    (
        "intermediate-bound-internal-rehash",
        "RVIR-v4 pre-state frozen identity differs",
        _mutate_intermediate,
    ),
    (
        "upper-alpha-internal-rehash",
        "RVIR-v4 pre-state snapshot/step-zero binding differs",
        _mutate_upper_alpha,
    ),
    (
        "beta-location-history-internal-rehash",
        "RVIR-v4 pre-state frozen identity differs",
        _mutate_beta_location,
    ),
)


def _resign_json_manifest(path: Path) -> None:
    manifest = _load_json(path)
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = _canonical_hash(semantic)
    _write_json(path, manifest)


def _resign_capture_artifact(artifact: Path, capture: Mapping[str, Any]) -> None:
    import torch

    capture_path = artifact / artifact_runner.SOURCE_CAPTURE_FILE
    torch.save(dict(capture), capture_path)
    source_manifest_path = artifact / artifact_runner.SOURCE_MANIFEST_FILE
    source_manifest = _load_json(source_manifest_path)
    source_files = source_manifest.get("files")
    if not isinstance(source_files, dict):
        raise TypeError("RVIR-v4 pre-state source manifest files differ")
    source_files["production_capture.pt"] = _file_sha256(capture_path)
    _write_json(source_manifest_path, source_manifest)
    _resign_json_manifest(source_manifest_path)
    manifest_path = artifact / "manifest.json"
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("RVIR-v4 pre-state manifest files differ")
    files[artifact_runner.SOURCE_CAPTURE_FILE] = _file_sha256(capture_path)
    files[artifact_runner.SOURCE_MANIFEST_FILE] = _file_sha256(source_manifest_path)
    _write_json(manifest_path, manifest)
    _resign_json_manifest(manifest_path)


def _resign_topology_artifact(artifact: Path) -> None:
    topology_path = artifact / "topology.json"
    topology = _load_json(topology_path)
    rows = topology["rows"]
    rows[0]["native_preactivation"] = "18"
    topology["topology_hash"] = _canonical_hash(rows)
    _write_json(topology_path, topology)
    manifest_path = artifact / "manifest.json"
    manifest = _load_json(manifest_path)
    manifest["files"]["topology.json"] = _file_sha256(topology_path)
    _write_json(manifest_path, manifest)
    _resign_json_manifest(manifest_path)


def _run_replay(artifact: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/run_rvir_v4_pre_state_artifact.py"),
            "replay",
            "--model",
            str(MODEL),
            "--artifact-dir",
            str(artifact),
        ),
        cwd=REPOSITORY_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )


def _semantic_rejection(capture: Mapping[str, Any], expected_error: str) -> str:
    try:
        artifact_runner._build_evidence(capture, MODEL)
    except ValueError as error:
        message = str(error)
        if expected_error not in message:
            raise AssertionError(f"unexpected semantic rejection: {message}") from error
        return message
    raise AssertionError("tampered pre-state was admitted by semantic replay")


def _topology_semantic_rejection(capture: Mapping[str, Any]) -> str:
    snapshot = production_snapshot_from_payload_v4(_pre_snapshot(capture))
    topology = list(artifact_runner.TOPOLOGY)
    first = topology[0]
    topology[0] = ProductionReluTopologyV4(
        provider_activation=first.provider_activation,
        provider_preactivation=first.provider_preactivation,
        native_preactivation="18",
        provider_start_node=first.provider_start_node,
    )
    try:
        initialize_rvir_v4_native_pre_state(
            snapshot,
            tuple(topology),
            expected_identity=artifact_runner.EXPECTED_IDENTITY,
        )
    except ValueError as error:
        message = str(error)
        if "RVIR-v4 pre-state frozen identity differs" not in message:
            raise AssertionError(f"unexpected topology rejection: {message}") from error
        return message
    raise AssertionError("tampered topology was admitted by semantic replay")


def run_probe_suite(artifact: Path) -> dict[str, object]:
    """Replay original and reject six internally/externally re-signed attacks."""

    artifact = artifact.resolve()
    original = _run_replay(artifact)
    if original.returncode != 0:
        raise RuntimeError(
            f"original RVIR-v4 pre-state replay failed: {original.stdout}"
        )
    original_result = json.loads(original.stdout.strip().splitlines()[-1])
    original_capture = _load_capture(artifact / artifact_runner.SOURCE_CAPTURE_FILE)
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(
        prefix="boundflow-rvir-v4-pre-state-tamper-"
    ) as temp:
        root = Path(temp)
        topology_probe = root / "topology-internal-rehash"
        shutil.copytree(artifact, topology_probe)
        _resign_topology_artifact(topology_probe)
        completed = _run_replay(topology_probe)
        expected_outer = "RVIR-v4 pre-state topology differs"
        if completed.returncode == 0 or expected_outer not in completed.stdout:
            raise AssertionError(
                f"topology probe did not fail closed: {completed.stdout}"
            )
        results.append(
            {
                "name": "topology-internal-rehash",
                "status": "rejected-as-expected",
                "outer_error": expected_outer,
                "semantic_error": _topology_semantic_rejection(original_capture),
                "internal_hashes_recomputed": True,
                "outer_manifest_resigned": True,
            }
        )
        for name, expected_semantic, mutate in CAPTURE_PROBES:
            probe = root / name
            shutil.copytree(artifact, probe)
            capture = deepcopy(original_capture)
            mutate(capture)
            semantic_error = _semantic_rejection(capture, expected_semantic)
            _resign_capture_artifact(probe, capture)
            completed = _run_replay(probe)
            expected_outer = "RVIR-v4 pre-state frozen source differs"
            if completed.returncode == 0 or expected_outer not in completed.stdout:
                raise AssertionError(
                    f"tamper probe {name} did not fail closed: {completed.stdout}"
                )
            results.append(
                {
                    "name": name,
                    "status": "rejected-as-expected",
                    "outer_error": expected_outer,
                    "semantic_error": semantic_error,
                    "internal_hashes_recomputed": True,
                    "source_manifest_resigned": True,
                    "outer_manifest_resigned": True,
                }
            )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "probe_code_sha256": _file_sha256(Path(__file__).resolve()),
        "artifact_manifest_sha256": _file_sha256(artifact / "manifest.json"),
        "artifact_source_git_head": _load_json(artifact / "manifest.json")[
            "source_git_head"
        ],
        "original_replay": original_result,
        "probe_count": len(results),
        "probes": results,
        "all_outer_provenance_gates_rejected": True,
        "all_semantic_identity_gates_rejected": True,
        "performance_claimed": False,
    }
    report["report_hash"] = _canonical_hash(report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the formal probe suite and write its canonical report."""

    args = _parse_args()
    report = run_probe_suite(args.artifact_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.report, report)
    print(_canonical_json(report))


if __name__ == "__main__":
    main()
