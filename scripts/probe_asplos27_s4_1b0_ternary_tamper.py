#!/usr/bin/env python3
"""Run ten coherent outer-resigned S4-1B0 semantic tamper probes."""

# pylint: disable=missing-function-docstring,wrong-import-position,duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_asplos27_s4_1b0_ternary_stdlib import (  # noqa: E402
    ReplayError,
    canonical,
    canonical_hash,
    file_sha256,
    validate,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: object) -> None:
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def _workers(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "raw/workers.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _save_workers(root: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row.pop("worker_payload_hash", None)
        row["worker_payload_hash"] = canonical_hash(row)
    (root / "raw/workers.jsonl").write_text(
        "".join(canonical(row) + "\n" for row in rows), encoding="utf-8"
    )


def _resign_manifest(root: Path) -> None:
    files = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    value: dict[str, Any] = {
        "schema_version": "boundflow.asplos27-s4-1b0-ternary-manifest/v1",
        "artifact_schema": "boundflow.asplos27-s4-1b0-ternary-summary/v1",
        "files": files,
        "performance_claimed": False,
    }
    value["manifest_hash"] = canonical_hash(value)
    _write(root / "manifest.json", value)


def _binary_flip(root: Path, record_name: str) -> None:
    rows = _workers(root)
    row = rows[0]
    path = root / "raw/binary/positive-00.bin"
    payload = bytearray(path.read_bytes())
    record = next(
        item for item in row["binary"]["index"] if item["name"] == record_name
    )
    byte_offset = record["offset"] + (3 if record_name == "coefficient" else 0)
    payload[byte_offset] ^= 0x80 if record_name == "coefficient" else 1
    path.write_bytes(payload)
    chunk = payload[record["offset"] : record["offset"] + record["byte_count"]]
    record["sha256"] = hashlib.sha256(chunk).hexdigest()
    row["binary"]["sha256"] = hashlib.sha256(payload).hexdigest()
    _save_workers(root, rows)


def _protocol(root: Path) -> None:
    value = _load(root / "protocol.json")
    value["construction_model_hash"] = "0" * 64
    _write(root / "protocol.json", value)


def _module(root: Path) -> None:
    path = root / "module/device_source.cu"
    path.write_text(
        path.read_text(encoding="utf-8") + "\n// coherent tamper\n", encoding="utf-8"
    )
    rows = _workers(root)
    digest = file_sha256(path)
    for row in rows[:5]:
        row["module_files"]["device_source.cu"] = digest
    _save_workers(root, rows)


def _cache(root: Path) -> None:
    rows = _workers(root)
    rows[5]["compile_count"] = 2
    _save_workers(root, rows)


def _descriptor(root: Path) -> None:
    rows = _workers(root)
    rows[0]["descriptor_hashes"][0] = "bad"
    _save_workers(root, rows)


def _ordinal(root: Path) -> None:
    rows = _workers(root)
    rows[4]["run_ordinal"] = 9
    _save_workers(root, rows)


def _fault(root: Path) -> None:
    rows = _workers(root)
    rows[6]["result"]["reason"] = "TERNARY_ENDPOINT_POLICY_MISMATCH"
    _save_workers(root, rows)


def _claim(root: Path) -> None:
    summary = _load(root / "summary.json")
    summary["performance_claimed"] = True
    summary.pop("summary_hash", None)
    summary["summary_hash"] = canonical_hash(summary)
    _write(root / "summary.json", summary)


CASES: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("coefficient_binary_and_index", lambda root: _binary_flip(root, "coefficient")),
    ("selector_binary_and_class_counts", lambda root: _binary_flip(root, "selector")),
    ("selected_binary_and_expected_bits", lambda root: _binary_flip(root, "selected")),
    ("midpoint_policy_and_counterexample", _protocol),
    ("module_ir_or_device_source_and_receipt", _module),
    ("cache_event_or_compile_count", _cache),
    ("descriptor_pointer_storage_inventory", _descriptor),
    ("worker_ordinal_or_sequence", _ordinal),
    ("fault_reason_or_cleanup", _fault),
    ("claim_status_or_performance_flag", _claim),
)


def run(artifact: Path) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for name, mutate in CASES:
        with tempfile.TemporaryDirectory(prefix="s4-1b0-tamper-") as temporary:
            root = Path(temporary) / "artifact"
            shutil.copytree(artifact, root)
            mutate(root)
            _resign_manifest(root)
            try:
                validate(root)
            except ReplayError as error:
                results.append({"name": name, "rejected": True, "reason": str(error)})
            else:
                results.append({"name": name, "rejected": False, "reason": "accepted"})
    rejected = sum(1 for row in results if row["rejected"] is True)
    return {
        "schema_version": "boundflow.asplos27-s4-1b0-ternary-tamper/v1",
        "case_count": len(results),
        "rejected": rejected,
        "results": results,
        "coherent_full_resign_e0_boundary_disclosed": True,
        "performance_claimed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.artifact.resolve())
    text = canonical(result) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["rejected"] == result["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
