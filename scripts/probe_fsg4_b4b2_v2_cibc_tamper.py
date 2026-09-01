#!/usr/bin/env python3
"""Outer-resigned semantic tamper probes for the CIBC formal artifact."""

# pylint: disable=wrong-import-position,missing-function-docstring

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

from scripts import run_fsg4_b4b2_v2_cibc_artifact as artifact
from scripts import run_fsg4_b4b2_v2_cibc_worker as worker


def _write(path: Path, value: object) -> None:
    path.write_text(artifact.canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _resign_worker(path: Path, mutate: Callable[[dict], None]) -> None:
    envelope = artifact.load_json(path)
    result = deepcopy(envelope["result"])
    mutate(result)
    result.pop("worker_hash", None)
    result["worker_hash"] = worker.canonical_hash(result)
    envelope["result"] = result
    envelope.pop("envelope_hash", None)
    envelope["envelope_hash"] = worker.canonical_hash(envelope)
    _write(path, envelope)


def _resign_summary(path: Path, mutate: Callable[[dict], None]) -> None:
    summary = artifact.load_json(path)
    mutate(summary)
    summary.pop("summary_hash", None)
    summary["summary_hash"] = artifact.canonical_hash(summary)
    _write(path, summary)


def _resign_protocol(path: Path, mutate: Callable[[dict], None]) -> None:
    protocol = artifact.load_json(path)
    mutate(protocol)
    protocol.pop("protocol_hash", None)
    protocol["protocol_hash"] = artifact.canonical_hash(protocol)
    _write(path, protocol)


def _resign_manifest(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = artifact.load_json(manifest_path)
    manifest["protocol_hash"] = artifact.load_json(root / "protocol.json")[
        "protocol_hash"
    ]
    manifest["summary_hash"] = artifact.load_json(root / "summary.json")["summary_hash"]
    manifest["files"] = {
        str(path.relative_to(root)): artifact.file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = artifact.canonical_hash(manifest)
    _write(manifest_path, manifest)


def _set_nested(result: dict, *path_and_value) -> None:
    *path, value = path_and_value
    target = result
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value


def _double_all_baseline_samples(result: dict) -> None:
    for pair in result["pairs"]:
        pair["baseline_ms"] *= 2.0
        pair["speedup"] *= 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.artifact.resolve()
    cases: tuple[tuple[str, str, Callable[[Path], None]], ...] = (
        (
            "protocol-gate",
            "protocol.json",
            lambda path: _resign_protocol(
                path, lambda value: value.__setitem__("speedup_geomean_gate", 1.0)
            ),
        ),
        (
            "calibration-sample",
            "raw/calibration_00.json",
            lambda path: _resign_worker(
                path,
                lambda value: value["samples_ms"].__setitem__(
                    0, value["samples_ms"][0] * 0.5
                ),
            ),
        ),
        (
            "calibration-median",
            "raw/calibration_01.json",
            lambda path: _resign_worker(
                path,
                lambda value: value.__setitem__("median_ms", value["median_ms"] * 0.5),
            ),
        ),
        (
            "timing-pair",
            "raw/timing_00_ab.json",
            lambda path: _resign_worker(
                path,
                _double_all_baseline_samples,
            ),
        ),
        (
            "timing-derived-speedup",
            "raw/timing_01_ba.json",
            lambda path: _resign_worker(
                path,
                lambda value: value.__setitem__(
                    "paired_speedup", value["paired_speedup"] * 2.0
                ),
            ),
        ),
        (
            "kernel-inventory",
            "raw/timing_02_ab.json",
            lambda path: _resign_worker(
                path,
                lambda value: value["kernel_inventory"]["kernel_names"].append(
                    "forged_kernel"
                ),
            ),
        ),
        (
            "correctness-tolerance",
            "raw/correctness_00.json",
            lambda path: _resign_worker(
                path,
                lambda value: value["parity"].__setitem__(
                    "maximum_absolute_difference", 1.0
                ),
            ),
        ),
        (
            "compiled-ptx",
            "raw/timing_03_ba.json",
            lambda path: _resign_worker(
                path,
                lambda value: _set_nested(
                    value,
                    "compilation",
                    "forward",
                    "assembly_hashes",
                    "ptx",
                    "0" * 64,
                ),
            ),
        ),
        (
            "summary-geomean",
            "summary.json",
            lambda path: _resign_summary(
                path,
                lambda value: value.__setitem__(
                    "speedup_geomean", value["speedup_geomean"] * 2.0
                ),
            ),
        ),
        (
            "frozen-config",
            "raw/calibration_04.json",
            lambda path: _resign_worker(
                path,
                lambda value: value["config"].__setitem__("block_k", 128),
            ),
        ),
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="boundflow-cibc-tamper-") as directory:
        temporary_root = Path(directory)
        for name, relative_path, mutate in cases:
            root = temporary_root / name
            shutil.copytree(source, root)
            mutate(root / relative_path)
            _resign_manifest(root)
            rejected = False
            error = ""
            try:
                artifact.replay(root)
            except (ValueError, TypeError, KeyError) as exception:
                rejected = True
                error = str(exception)
            results.append({"name": name, "rejected": rejected, "error": error})
    report = {
        "schema_version": "boundflow.fsg4-b4b2-v2-cibc-tamper/v1",
        "case_count": len(results),
        "rejected_count": sum(bool(row["rejected"]) for row in results),
        "all_rejected": all(bool(row["rejected"]) for row in results),
        "results": results,
    }
    report["report_hash"] = artifact.canonical_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, report)
    print(json.dumps(report, sort_keys=True))
    if not report["all_rejected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
