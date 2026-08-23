#!/usr/bin/env python3
"""Outer-resigned semantic tamper probes for B4-B3 five-fresh evidence."""

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

from scripts import run_fsg4_b4b3_cibc_exact_worker as worker
from scripts import run_fsg4_b4b3_cibc_five_fresh_artifact as artifact


def _write(path: Path, value: object) -> None:
    path.write_text(artifact.canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _resign_worker(path: Path, mutate: Callable[[dict], None]) -> None:
    value = artifact.load_json(path)
    payload = deepcopy(value)
    mutate(payload)
    payload.pop("worker_hash", None)
    payload["worker_hash"] = worker.canonical_hash(payload)
    _write(path, payload)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases: tuple[tuple[str, str, Callable[[Path], None]], ...] = (
        (
            "protocol-tolerance",
            "protocol.json",
            lambda path: _resign_protocol(
                path, lambda value: value.__setitem__("terminal_atol", 1.0)
            ),
        ),
        (
            "terminal-maximum",
            "raw/run_00_bc.json",
            lambda path: _resign_worker(
                path,
                lambda value: value.__setitem__("maximum_absolute_difference", 1.0),
            ),
        ),
        (
            "metric-allclose",
            "raw/run_01_cb.json",
            lambda path: _resign_worker(
                path,
                lambda value: value["metrics"]["terminal_lower"].__setitem__(
                    "allclose", False
                ),
            ),
        ),
        (
            "native-value-bridge",
            "raw/run_02_bc.json",
            lambda path: _resign_worker(
                path,
                lambda value: value["receipt"].__setitem__(
                    "native_value_bridge_count", 0
                ),
            ),
        ),
        (
            "module-identity",
            "raw/run_03_cb.json",
            lambda path: _resign_worker(
                path,
                lambda value: value["receipt"].__setitem__("module_hash", "0" * 64),
            ),
        ),
        (
            "local-parity",
            "raw/run_04_bc.json",
            lambda path: _resign_worker(
                path,
                lambda value: value["local_parity"][0].__setitem__(
                    "output_a_max_abs_diff", 1.0
                ),
            ),
        ),
        (
            "summary-maximum",
            "summary.json",
            lambda path: _resign_summary(
                path,
                lambda value: value.__setitem__("maximum_absolute_difference", 0.0),
            ),
        ),
        (
            "worker-order",
            "raw/run_00_bc.json",
            lambda path: _resign_worker(
                path, lambda value: value.__setitem__("order", "CB")
            ),
        ),
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="boundflow-b4b3-tamper-") as directory:
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
        "schema_version": "boundflow.fsg4-b4b3-cibc-five-fresh-tamper/v1",
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
