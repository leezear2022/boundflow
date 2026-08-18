#!/usr/bin/env python3
"""Probe outer-resigned attacks on the B4-B0 five-fresh artifact."""

# pylint: disable=protected-access,missing-function-docstring,too-many-locals

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable, cast

import torch

from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from scripts import run_fsg4_b4b_five_fresh_artifact as artifact

REPORT_SCHEMA = "boundflow.fsg4-b4b0-five-fresh-tamper-report/v2"


def _load_run(root: Path) -> dict[str, object]:
    value = torch.load(
        root / artifact.RUN_FILES[0], map_location="cpu", weights_only=True
    )
    if not isinstance(value, dict):
        raise TypeError("FSG4/B4-B0 tamper run root differs")
    return value


def _save_run(root: Path, value: dict[str, object]) -> None:
    torch.save(value, root / artifact.RUN_FILES[0])


def _capture(run: dict[str, object], index: int = 0) -> dict[str, object]:
    captures = run["captures"]
    assert isinstance(captures, list) and isinstance(captures[index], dict)
    return cast(dict[str, object], captures[index])


def _metadata(capture: dict[str, object]) -> dict[str, object]:
    value = capture["metadata"]
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _resign_capture(metadata: dict[str, object]) -> None:
    payload = copy.deepcopy(metadata)
    payload.pop("capture_hash", None)
    metadata["capture_hash"] = artifact._canonical_hash(payload)


def _resign_root(root: Path) -> None:
    manifest = artifact._load_json(root / "manifest.json")
    manifest["files"] = artifact._all_files(root)
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    manifest["manifest_hash"] = artifact._canonical_hash(payload)
    artifact._write_json(root / "manifest.json", manifest)


def _state(root: Path) -> None:
    run = _load_run(root)
    metadata = _metadata(_capture(run))
    metadata["source_state_hash"] = "b" * 64
    _resign_capture(metadata)
    _save_run(root, run)


def _start_node(root: Path) -> None:
    run = _load_run(root)
    metadata = _metadata(_capture(run))
    lineage = metadata["production_lineage"]
    assert isinstance(lineage, dict)
    lineage["provider_start_node"] = "/50"
    lineage_payload = dict(lineage)
    lineage_payload.pop("lineage_hash", None)
    lineage["lineage_hash"] = artifact._canonical_hash(lineage_payload)
    _resign_capture(metadata)
    _save_run(root, run)


def _topology(root: Path) -> None:
    run = _load_run(root)
    metadata = _metadata(_capture(run))
    metadata["topology_hash"] = "b" * 64
    _resign_capture(metadata)
    _save_run(root, run)


def _shape(root: Path) -> None:
    run = _load_run(root)
    metadata = _metadata(_capture(run))
    values = metadata["values"]
    assert isinstance(values, dict) and isinstance(values["incoming_lower_a"], dict)
    values["incoming_lower_a"]["shape"] = [6, 100]
    _resign_capture(metadata)
    _save_run(root, run)


def _lineage_hash(root: Path, pattern: str) -> None:
    run = _load_run(root)
    metadata = _metadata(_capture(run))
    lineage = metadata["production_lineage"]
    assert isinstance(lineage, dict)
    hashes = lineage["source_tensor_hashes"]
    assert isinstance(hashes, dict)
    path = next(name for name in hashes if pattern in name)
    hashes[path] = "b" * 64
    lineage_payload = dict(lineage)
    lineage_payload.pop("lineage_hash", None)
    lineage["lineage_hash"] = artifact._canonical_hash(lineage_payload)
    _resign_capture(metadata)
    _save_run(root, run)


def _alpha_index(root: Path) -> None:
    _lineage_hash(root, "/feature_index/0")


def _beta_location(root: Path) -> None:
    _lineage_hash(root, "/location")


def _gradient(root: Path) -> None:
    run = _load_run(root)
    capture = _capture(run)
    metadata = _metadata(capture)
    gradients = capture["gradients"]
    gradient_metadata = metadata["gradients"]
    assert isinstance(gradients, dict) and isinstance(gradient_metadata, dict)
    value = gradients["native_alpha"]
    record = gradient_metadata["native_alpha"]
    assert torch.is_tensor(value) and isinstance(record, dict)
    changed = value.clone()
    changed.reshape(-1)[0] += 1.0
    gradients["native_alpha"] = changed
    record["content_sha256"] = production_tensor_sha256(changed)
    _resign_capture(metadata)
    _save_run(root, run)


def _alias(root: Path) -> None:
    run = _load_run(root)
    metadata = _metadata(_capture(run))
    metadata["source_alias_pairs"] = [["value:native_alpha", "value:native_beta"]]
    _resign_capture(metadata)
    _save_run(root, run)


def _stream(root: Path) -> None:
    run = _load_run(root)
    metadata = _metadata(_capture(run))
    metadata["source_cuda_stream_is_default"] = False
    _resign_capture(metadata)
    _save_run(root, run)


def _coordinated_topology(root: Path) -> None:
    for name in artifact.RUN_FILES:
        run = torch.load(root / name, map_location="cpu", weights_only=True)
        assert isinstance(run, dict)
        captures = run["captures"]
        assert isinstance(captures, list)
        for capture in captures:
            assert isinstance(capture, dict)
            metadata = _metadata(cast(dict[str, object], capture))
            metadata["topology_hash"] = "b" * 64
            _resign_capture(metadata)
        torch.save(run, root / name)


def _coordinated_lineage(root: Path) -> None:
    for name in artifact.RUN_FILES:
        run = torch.load(root / name, map_location="cpu", weights_only=True)
        assert isinstance(run, dict)
        captures = run["captures"]
        assert isinstance(captures, list)
        for capture in captures:
            assert isinstance(capture, dict)
            metadata = _metadata(cast(dict[str, object], capture))
            lineage = metadata["production_lineage"]
            assert isinstance(lineage, dict)
            hashes = lineage["source_tensor_hashes"]
            assert isinstance(hashes, dict)
            first = sorted(hashes)[0]
            hashes[first] = "b" * 64
            lineage_payload = dict(lineage)
            lineage_payload.pop("lineage_hash", None)
            lineage["lineage_hash"] = artifact._canonical_hash(lineage_payload)
            _resign_capture(metadata)
        torch.save(run, root / name)


ATTACKS: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("state-outer-resigned", _state),
    ("start-node-outer-resigned", _start_node),
    ("topology-outer-resigned", _topology),
    ("shape-outer-resigned", _shape),
    ("alpha-index-outer-resigned", _alpha_index),
    ("beta-location-outer-resigned", _beta_location),
    ("gradient-outer-resigned", _gradient),
    ("alias-outer-resigned", _alias),
    ("stream-outer-resigned", _stream),
    ("coordinated-all-runs-topology-outer-resigned", _coordinated_topology),
    ("coordinated-all-runs-lineage-outer-resigned", _coordinated_lineage),
)


def _probe(source: Path) -> dict[str, object]:
    artifact._verify_static_artifact(source)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="fsg4-b4b0-tamper-") as raw:
        parent = Path(raw)
        for ordinal, (name, attack) in enumerate(ATTACKS):
            target = parent / f"attack-{ordinal:02d}"
            shutil.copytree(source, target)
            attack(target)
            _resign_root(target)
            try:
                artifact._verify_static_artifact(target)
            except Exception as error:  # pylint: disable=broad-exception-caught
                rows.append(
                    {
                        "attack": name,
                        "outer_resigned": True,
                        "rejected": True,
                        "exception_type": type(error).__name__,
                        "message": str(error),
                    }
                )
            else:
                rows.append({"attack": name, "outer_resigned": True, "rejected": False})
    if len(rows) != len(ATTACKS) or not all(row["rejected"] is True for row in rows):
        raise ValueError("FSG4/B4-B0 tamper probe accepted an attack")
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "artifact_manifest_sha256": artifact._file_sha256(source / "manifest.json"),
        "attack_count": len(rows),
        "rejected_count": len(rows),
        "rows": rows,
        "performance_claimed": False,
        "tir_admitted": False,
    }
    report["report_hash"] = artifact._canonical_hash(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = _probe(args.artifact_dir.resolve())
    artifact._write_json(args.report.resolve(), report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
