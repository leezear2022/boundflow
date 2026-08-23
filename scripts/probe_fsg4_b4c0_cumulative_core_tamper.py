#!/usr/bin/env python3
"""Outer-resigned semantic tamper probes for B4-C0 cumulative core evidence."""

# pylint: disable=wrong-import-position,missing-function-docstring,too-many-locals

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

from scripts import run_fsg4_b4b3_cibc_exact_worker as hash_worker
from scripts import run_fsg4_b4c0_cumulative_core_artifact as artifact


def _write(path: Path, value: object) -> None:
    path.write_text(artifact.canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _resign_worker(path: Path, mutate: Callable[[dict], None]) -> None:
    value = deepcopy(artifact.load_json(path))
    mutate(value)
    value.pop("worker_hash", None)
    value["worker_hash"] = hash_worker.canonical_hash(value)
    _write(path, value)


def _resign_summary(path: Path, mutate: Callable[[dict], None]) -> None:
    value = artifact.load_json(path)
    mutate(value)
    value.pop("summary_hash", None)
    value["summary_hash"] = artifact.canonical_hash(value)
    _write(path, value)


def _resign_protocol(path: Path, mutate: Callable[[dict], None]) -> None:
    value = artifact.load_json(path)
    mutate(value)
    value.pop("protocol_hash", None)
    value["protocol_hash"] = artifact.canonical_hash(value)
    _write(path, value)


def _resign_manifest(root: Path) -> None:
    path = root / "manifest.json"
    value = artifact.load_json(path)
    value["protocol_hash"] = artifact.load_json(root / "protocol.json")["protocol_hash"]
    value["summary_hash"] = artifact.load_json(root / "summary.json")["summary_hash"]
    value["files"] = {
        str(item.relative_to(root)): artifact.file_sha256(item)
        for item in sorted(root.rglob("*"))
        if item.is_file() and item.name != "manifest.json"
    }
    value.pop("manifest_hash", None)
    value["manifest_hash"] = artifact.canonical_hash(value)
    _write(path, value)


def _double_baseline_groups(value: dict) -> None:
    for group in value["groups"]:
        group["baseline_ms"] *= 2.0
        group["speedup"] *= 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate_mode = artifact.load_json(args.artifact / "protocol.json").get(
        "candidate_mode", "native-value-bridge"
    )
    receipt_count_field = (
        "provider_owned_lower_count"
        if candidate_mode == "provider-owned-lower"
        else "native_value_bridge_count"
    )
    cases: tuple[tuple[str, str, Callable[[Path], None]], ...] = (
        (
            "protocol-gate",
            "protocol.json",
            lambda path: _resign_protocol(
                path,
                lambda value: value.__setitem__("no_regression_geomean_gate", 0.5),
            ),
        ),
        (
            "raw-timing-groups",
            "raw/run_00_bc.json",
            lambda path: _resign_worker(path, _double_baseline_groups),
        ),
        (
            "derived-median",
            "raw/run_01_cb.json",
            lambda path: _resign_worker(
                path,
                lambda value: value.__setitem__(
                    "candidate_median_ms", value["candidate_median_ms"] * 0.5
                ),
            ),
        ),
        (
            "receipt-provider-ownership",
            "raw/run_02_bc.json",
            lambda path: _resign_worker(
                path,
                lambda value: value["receipt"].__setitem__(receipt_count_field, 0),
            ),
        ),
        (
            "semantic-maximum",
            "raw/run_03_cb.json",
            lambda path: _resign_worker(
                path,
                lambda value: value.__setitem__("maximum_absolute_difference", 1.0),
            ),
        ),
        (
            "memory-peak",
            "raw/run_04_bc.json",
            lambda path: _resign_worker(
                path,
                lambda value: value.__setitem__(
                    "candidate_peak_allocated_bytes",
                    value["candidate_peak_allocated_bytes"] * 2,
                ),
            ),
        ),
        (
            "summary-geomean",
            "summary.json",
            lambda path: _resign_summary(
                path,
                lambda value: value.__setitem__("speedup_geomean", 2.0),
            ),
        ),
        (
            "worker-order",
            "raw/run_05_cb.json",
            lambda path: _resign_worker(
                path, lambda value: value.__setitem__("order", "BC")
            ),
        ),
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="boundflow-b4c0-tamper-") as directory:
        temporary_root = Path(directory)
        for name, relative_path, mutate in cases:
            root = temporary_root / name
            shutil.copytree(args.artifact.resolve(), root)
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
        "schema_version": "boundflow.fsg4-b4c0-cumulative-core-tamper/v1",
        "case_count": len(results),
        "rejected_count": sum(bool(item["rejected"]) for item in results),
        "all_rejected": all(bool(item["rejected"]) for item in results),
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
