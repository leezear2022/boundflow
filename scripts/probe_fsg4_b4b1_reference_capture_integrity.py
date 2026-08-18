#!/usr/bin/env python3
"""Probe outer-resigned integrity failures in the B4-B1a artifact."""

# pylint: disable=protected-access,too-many-locals,duplicate-code
# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
from typing import Callable, cast

import torch

from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from scripts import run_fsg4_b4b1_reference_five_fresh_artifact as artifact

REPORT_SCHEMA = "boundflow.fsg4-b4b1-reference-integrity-report/v1"


def _run(root: Path) -> dict[str, object]:
    value = torch.load(
        root / artifact.RUN_FILES[0], map_location="cpu", weights_only=True
    )
    if not isinstance(value, dict):
        raise TypeError("FSG4/B4-B1 integrity run differs")
    return cast(dict[str, object], value)


def _capture(run: dict[str, object], ordinal: int = 0) -> dict[str, object]:
    captures = run["captures"]
    if not isinstance(captures, list) or not isinstance(captures[ordinal], dict):
        raise TypeError("FSG4/B4-B1 integrity capture differs")
    return cast(dict[str, object], captures[ordinal])


def _metadata(capture: dict[str, object]) -> dict[str, object]:
    value = capture["metadata"]
    if not isinstance(value, dict):
        raise TypeError("FSG4/B4-B1 integrity metadata differs")
    return cast(dict[str, object], value)


def _resign_snapshot(metadata: dict[str, object], raw: torch.Tensor) -> None:
    metadata["content_sha256"] = production_tensor_sha256(raw)


def _resign_reference(capture: dict[str, object]) -> None:
    metadata = _metadata(capture)
    semantic = dict(metadata)
    semantic.pop("reference_capture_hash", None)
    metadata["reference_capture_hash"] = artifact._canonical_hash(semantic)


def _save_run(root: Path, run: dict[str, object]) -> None:
    torch.save(run, root / artifact.RUN_FILES[0])


def _incoming_bias(root: Path) -> None:
    run = _run(root)
    capture = _capture(run)
    raw = cast(torch.Tensor, capture["incoming_lower_bias"]).clone()
    raw.reshape(-1)[0] += 1.0
    capture["incoming_lower_bias"] = raw
    metadata = _metadata(capture)
    _resign_snapshot(cast(dict[str, object], metadata["incoming_lower_bias"]), raw)
    _resign_reference(capture)
    _save_run(root, run)


def _operator_bias_value(root: Path) -> None:
    run = _run(root)
    capture = _capture(run)
    raw = cast(torch.Tensor, capture["operator_bias"]).clone()
    raw.reshape(-1)[0] += 1.0
    capture["operator_bias"] = raw
    metadata = _metadata(capture)
    _resign_snapshot(cast(dict[str, object], metadata["operator_bias"]), raw)
    _resign_reference(capture)
    _save_run(root, run)


def _operator_bias_presence(root: Path) -> None:
    run = _run(root)
    capture = _capture(run)
    capture["operator_bias"] = None
    metadata = _metadata(capture)
    metadata["operator_bias"] = None
    metadata["operator_bias_present"] = False
    attributes = cast(dict[str, object], metadata["reference_attributes"])
    attributes["operator_bias_present"] = False
    _resign_reference(capture)
    _save_run(root, run)


def _output_gradient(root: Path, name: str) -> None:
    run = _run(root)
    capture = _capture(run)
    raw_map = cast(dict[str, torch.Tensor], capture["output_gradients"])
    raw = raw_map[name].clone()
    raw.reshape(-1)[0] += 1.0
    raw_map[name] = raw
    metadata = _metadata(capture)
    gradient_metadata = cast(dict[str, dict[str, object]], metadata["output_gradients"])
    _resign_snapshot(gradient_metadata[name], raw)
    _resign_reference(capture)
    _save_run(root, run)


def _output_lower_a_gradient(root: Path) -> None:
    _output_gradient(root, "output_lower_a")


def _output_bias_gradient(root: Path) -> None:
    _output_gradient(root, "output_bias")


def _mapping_index(root: Path) -> None:
    run = _run(root)
    capture = _capture(run)
    raw_map = cast(dict[str, torch.Tensor], capture["mapping_tensors"])
    name = next(path for path in sorted(raw_map) if "/feature_index/" in path)
    raw = raw_map[name].clone()
    raw.reshape(-1)[0] += 1
    raw_map[name] = raw
    metadata = _metadata(capture)
    mapping_metadata = cast(dict[str, dict[str, object]], metadata["mapping_tensors"])
    _resign_snapshot(mapping_metadata[name], raw)
    _resign_reference(capture)
    _save_run(root, run)


def _reference_attribute(root: Path) -> None:
    run = _run(root)
    capture = _capture(run, 1)
    metadata = _metadata(capture)
    attributes = cast(dict[str, object], metadata["reference_attributes"])
    attributes["output_padding"] = [1, 0]
    _resign_reference(capture)
    _save_run(root, run)


def _base_topology(root: Path) -> None:
    run = _run(root)
    capture = _capture(run)
    base = cast(dict[str, object], capture["base"])
    base_metadata = cast(dict[str, object], base["metadata"])
    base_metadata["topology_hash"] = "b" * 64
    semantic = dict(base_metadata)
    semantic.pop("capture_hash", None)
    base_metadata["capture_hash"] = artifact._canonical_hash(semantic)
    metadata = _metadata(capture)
    metadata["base_capture_hash"] = base_metadata["capture_hash"]
    _resign_reference(capture)
    _save_run(root, run)


CASES: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("incoming-lower-bias-outer-resigned", _incoming_bias),
    ("operator-bias-value-outer-resigned", _operator_bias_value),
    ("operator-bias-presence-outer-resigned", _operator_bias_presence),
    ("output-lower-a-gradient-outer-resigned", _output_lower_a_gradient),
    ("output-bias-gradient-outer-resigned", _output_bias_gradient),
    ("sparse-mapping-index-outer-resigned", _mapping_index),
    ("reference-attribute-outer-resigned", _reference_attribute),
    ("base-topology-outer-resigned", _base_topology),
)


def _resign_outer(root: Path) -> None:
    manifest = artifact._load_json(root / "manifest.json")
    files = cast(dict[str, str], manifest["files"])
    files[artifact.RUN_FILES[0]] = artifact._file_sha256(root / artifact.RUN_FILES[0])
    semantic = dict(manifest)
    semantic.pop("manifest_hash", None)
    manifest["manifest_hash"] = artifact._canonical_hash(semantic)
    artifact._write_json(root / "manifest.json", manifest)


def _probe(args: argparse.Namespace) -> dict[str, object]:
    source = args.artifact_dir.resolve()
    rows = []
    for name, mutation in CASES:
        with tempfile.TemporaryDirectory(prefix="b4b1-integrity-") as temporary:
            root = Path(temporary) / "artifact"
            shutil.copytree(source, root)
            mutation(root)
            _resign_outer(root)
            rejected = False
            exception_type = ""
            message = ""
            try:
                artifact._verify_static_artifact(root)
            except (TypeError, ValueError) as error:
                rejected = True
                exception_type = type(error).__name__
                message = str(error)
            rows.append(
                {
                    "case": name,
                    "outer_resigned": True,
                    "rejected": rejected,
                    "exception_type": exception_type,
                    "message": message,
                }
            )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "artifact_manifest_sha256": artifact._file_sha256(source / "manifest.json"),
        "case_count": len(rows),
        "rejected_count": sum(row["rejected"] is True for row in rows),
        "rows": rows,
        "known_limit": (
            "coordinated dynamic bias/adjoint rewrites are deferred to B4-B1 "
            "numerical reference semantic replay"
        ),
        "performance_claimed": False,
        "tir_admitted": False,
    }
    report["report_hash"] = artifact._canonical_hash(report)
    artifact._write_json(args.report.resolve(), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = _probe(args)
    print(artifact._canonical_json(report), flush=True)
    if report["rejected_count"] != report["case_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
