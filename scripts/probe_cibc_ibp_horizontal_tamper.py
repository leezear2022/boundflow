#!/usr/bin/env python3
"""Probe fully re-signed tampering against the CIBC IBP formal replay."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
from typing import Any, Callable

from scripts import run_cibc_ibp_horizontal_artifact as artifact


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: object) -> None:
    artifact.write_json(path, value)


def _resign(root: Path) -> None:
    for path in sorted((root / "raw").glob("*.json")):
        value = _load(path)
        value.pop("worker_hash", None)
        value["worker_hash"] = artifact.canonical_hash(value)
        _write(path, value)
    protocol = _load(root / "protocol.json")
    protocol.pop("protocol_hash", None)
    protocol["protocol_hash"] = artifact.canonical_hash(protocol)
    _write(root / "protocol.json", protocol)
    summary = _load(root / "summary.json")
    summary.pop("summary_hash", None)
    summary["summary_hash"] = artifact.canonical_hash(summary)
    _write(root / "summary.json", summary)
    _write(root / "manifest.json", artifact.manifest(root, protocol, summary))


def _protocol_claim(root: Path) -> None:
    value = _load(root / "protocol.json")
    value["performance_claimed"] = True
    _write(root / "protocol.json", value)


def _timing_derivation(root: Path) -> None:
    value = _load(root / "raw" / "model_00_bc.json")
    value["groups"][0]["speedup"] = 999.0
    _write(root / "raw" / "model_00_bc.json", value)


def _semantic_error(root: Path) -> None:
    value = _load(root / "raw" / "model_01_cb.json")
    value["maximum_absolute_difference"] = 1.0
    _write(root / "raw" / "model_01_cb.json", value)


def _coverage(root: Path) -> None:
    value = _load(root / "raw" / "model_02_bc.json")
    value["conv_coverage"] = 5
    _write(root / "raw" / "model_02_bc.json", value)


def _worker_identity(root: Path) -> None:
    value = _load(root / "raw" / "model_03_cb.json")
    value["run_ordinal"] = 4
    _write(root / "raw" / "model_03_cb.json", value)


def _operator_inventory(root: Path) -> None:
    value = _load(root / "raw" / "operator_128.json")
    value["operators"][0]["op_ordinal"] = 99
    _write(root / "raw" / "operator_128.json", value)


def _summary_speedup(root: Path) -> None:
    value = _load(root / "summary.json")
    value["model_speedup_geomean"] = 99.0
    _write(root / "summary.json", value)


def _selected_schedule(root: Path) -> None:
    value = _load(root / "summary.json")
    value["selected_threads_per_block"] = 256
    _write(root / "summary.json", value)


def _hardware(root: Path) -> None:
    value = _load(root / "raw" / "model_04_bc.json")
    value["environment"]["device"] = "unbound-device"
    _write(root / "raw" / "model_04_bc.json", value)


def _graph_receipt(root: Path) -> None:
    value = _load(root / "raw" / "model_05_cb.json")
    value["input_copy_included"] = False
    _write(root / "raw" / "model_05_cb.json", value)


PROBES: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("protocol-performance-claim", _protocol_claim),
    ("timing-derivation", _timing_derivation),
    ("semantic-error", _semantic_error),
    ("conv-coverage", _coverage),
    ("worker-identity", _worker_identity),
    ("operator-inventory", _operator_inventory),
    ("summary-speedup", _summary_speedup),
    ("selected-schedule", _selected_schedule),
    ("hardware-identity", _hardware),
    ("graph-receipt", _graph_receipt),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.artifact.resolve()
    results = []
    with tempfile.TemporaryDirectory(prefix="cibc-ibp-tamper-") as temporary:
        temporary_root = Path(temporary)
        pristine = {
            str(path.relative_to(source)): path.read_bytes()
            for path in source.rglob("*")
            if path.is_file()
        }
        for name, mutate in PROBES:
            root = temporary_root / name
            for relative, file_payload in pristine.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(copy.copy(file_payload))
            mutate(root)
            _resign(root)
            try:
                artifact.replay(root)
            except (ValueError, TypeError, KeyError) as error:
                results.append({"probe": name, "rejected": True, "error": str(error)})
            else:
                results.append({"probe": name, "rejected": False, "error": None})
    payload: dict[str, object] = {
        "schema_version": "boundflow.cibc-ibp-horizontal-tamper/v1",
        "probe_count": len(results),
        "rejected_count": sum(bool(item["rejected"]) for item in results),
        "fully_resigned": True,
        "results": results,
    }
    payload["report_hash"] = artifact.canonical_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, payload)
    print(artifact.canonical_json(payload))
    if payload["rejected_count"] != payload["probe_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
