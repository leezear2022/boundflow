#!/usr/bin/env python3
"""Outer-resigned semantic tamper probes for the manual-TIR artifact."""

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

from scripts import run_fsg4_b4b2_v2_cibc_tir_artifact as artifact
from scripts import run_fsg4_b4b2_v2_cibc_tir_worker as worker


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


def _double_all_baseline_samples(result: dict) -> None:
    for group in result["groups"]:
        group["baseline_ms"] *= 2.0
        group["baseline_over_tir"] *= 2.0


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
                path, lambda value: value.__setitem__("baseline_speedup_gate", 1.0)
            ),
        ),
        (
            "correctness-parity",
            "raw/correctness_00.json",
            lambda path: _resign_worker(
                path,
                lambda value: value["parity"]["baseline_tir"].__setitem__(
                    "maximum_absolute_difference", 1.0
                ),
            ),
        ),
        (
            "tir-module-receipt",
            "raw/correctness_01.json",
            lambda path: _resign_worker(
                path,
                lambda value: value["tir_receipt"].__setitem__("module_hash", "0" * 64),
            ),
        ),
        (
            "triton-config-receipt",
            "raw/correctness_02.json",
            lambda path: _resign_worker(
                path,
                lambda value: value["triton_receipt"].__setitem__("config_ordinal", 0),
            ),
        ),
        (
            "timing-raw-groups",
            "raw/timing_00_btr.json",
            lambda path: _resign_worker(path, _double_all_baseline_samples),
        ),
        (
            "timing-derived-baseline-speedup",
            "raw/timing_01_brt.json",
            lambda path: _resign_worker(
                path,
                lambda value: value.__setitem__(
                    "baseline_over_tir", value["baseline_over_tir"] * 2.0
                ),
            ),
        ),
        (
            "timing-derived-tir-median",
            "raw/timing_02_tbr.json",
            lambda path: _resign_worker(
                path,
                lambda value: value.__setitem__(
                    "tir_median_ms", value["tir_median_ms"] * 0.5
                ),
            ),
        ),
        (
            "kernel-inventory",
            "raw/timing_03_trb.json",
            lambda path: _resign_worker(
                path,
                lambda value: value["tir_receipt"]["profiler_kernel_names"].append(
                    "forged_kernel"
                ),
            ),
        ),
        (
            "summary-geomean",
            "summary.json",
            lambda path: _resign_summary(
                path,
                lambda value: value.__setitem__(
                    "baseline_over_tir_geomean",
                    value["baseline_over_tir_geomean"] * 2.0,
                ),
            ),
        ),
        (
            "timing-order",
            "raw/timing_05_rtb.json",
            lambda path: _resign_worker(
                path, lambda value: value.__setitem__("order", "BTR")
            ),
        ),
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="boundflow-cibc-tir-tamper-") as directory:
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
        "schema_version": "boundflow.fsg4-b4b2-v2-cibc-tir-tamper/v1",
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
