#!/usr/bin/env python3
"""Probe synchronized tampering of the RVIR-v4 V4-3B artifact."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_rvir_v4_native_backward_export_artifact as artifact_runner

REPORT_SCHEMA_VERSION = "boundflow.rvir-v4-native-backward-export-tamper-report/v1"


def _resign_manifest(artifact: Path, *changed_files: str) -> None:
    manifest_path = artifact / "manifest.json"
    manifest = artifact_runner._load_json(manifest_path)
    files = manifest["files"]
    for name in changed_files:
        files[name] = artifact_runner._file_sha256(artifact / name)
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = artifact_runner._canonical_hash(semantic)
    artifact_runner._write_json(manifest_path, manifest)


def _resign_export(artifact: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = artifact / artifact_runner.EXPORT_FILE
    payload = artifact_runner._load_torch(path)
    mutate(payload)
    typed = artifact_runner._export_from_payload(payload, validate_metadata=False)
    payload["metadata"] = typed.metadata()
    torch.save(payload, path)
    _resign_manifest(artifact, artifact_runner.EXPORT_FILE)


def _mutate_l_a(payload: dict[str, Any]) -> None:
    value = payload["lAs"]["/48"].clone()
    value.flatten()[0] += 1.0
    payload["lAs"]["/48"] = value


def _mutate_intermediate(payload: dict[str, Any]) -> None:
    value = payload["intermediates"]["/input-28"]["lower"].clone()
    value.flatten()[0] += 1.0
    payload["intermediates"]["/input-28"]["lower"] = value


def _mutate_lower(payload: dict[str, Any]) -> None:
    value = payload["lower"].clone()
    value.flatten()[0] += 1.0
    payload["lower"] = value


def _mutate_topology(artifact: Path) -> None:
    path = artifact / "topology.json"
    topology = artifact_runner._load_json(path)
    topology["rows"][0]["provider_activation"] = "/tampered"
    topology["topology_hash"] = artifact_runner._canonical_hash(topology["rows"])
    artifact_runner._write_json(path, topology)
    _resign_manifest(artifact, "topology.json")


def _mutate_truth_source(artifact: Path) -> None:
    path = artifact / artifact_runner.SOURCE_TRUTH_FILE
    truth = artifact_runner._load_torch(path)
    core = truth["whole_core_truths"][0]
    value = core["branch_trace"]["input"]["lAs"]["_data"]["/48"]["value"].clone()
    value.flatten()[0] += 1.0
    core["branch_trace"]["input"]["lAs"]["_data"]["/48"]["value"] = value
    torch.save(truth, path)
    _resign_manifest(artifact, artifact_runner.SOURCE_TRUTH_FILE)


def run_probe_suite(*, artifact: Path, model: Path) -> dict[str, object]:
    """Run three full export resigns and two source/topology outer resigns."""

    artifact_runner._replay(artifact, model)
    rows: list[dict[str, object]] = []
    attacks: tuple[tuple[str, Callable[[Path], None]], ...] = (
        (
            "lA-export-full-resign",
            lambda path: _resign_export(path, _mutate_l_a),
        ),
        (
            "intermediate-export-full-resign",
            lambda path: _resign_export(path, _mutate_intermediate),
        ),
        (
            "lower-export-full-resign",
            lambda path: _resign_export(path, _mutate_lower),
        ),
        ("topology-outer-resign", _mutate_topology),
        ("truth-source-outer-resign", _mutate_truth_source),
    )
    with tempfile.TemporaryDirectory(
        prefix="boundflow-rvir-v4-backward-tamper-"
    ) as raw:
        workspace = Path(raw)
        for name, mutate in attacks:
            probe = workspace / name
            shutil.copytree(artifact, probe)
            mutate(probe)
            try:
                artifact_runner._replay(probe, model)
            except (TypeError, ValueError) as error:
                rows.append(
                    {
                        "name": name,
                        "outer_resigned": True,
                        "rejected": True,
                        "rejection": str(error),
                    }
                )
            else:
                raise AssertionError(
                    f"tampered native backward artifact admitted: {name}"
                )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_source_git_head": artifact_runner._load_json(
            artifact / "manifest.json"
        )["source_git_head"],
        "attack_count": len(rows),
        "fully_resigned_export_attack_count": 3,
        "all_rejected": all(row["rejected"] is True for row in rows),
        "attacks": rows,
        "performance_claimed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run probes and write the formal report."""

    args = _parse_args()
    report = run_probe_suite(
        artifact=args.artifact_dir.resolve(), model=args.model.resolve()
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
