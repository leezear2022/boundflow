#!/usr/bin/env python3
"""Apply ten fully outer-resigned semantic attacks to the S4-1A artifact."""

# pylint: disable=protected-access,too-many-locals,wrong-import-position
# pylint: disable=missing-function-docstring

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

from scripts import replay_asplos27_s4_1a_buffer_stdlib as replay_tool
from scripts import run_asplos27_s4_1a_buffer_artifact as artifact_tool


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(replay_tool.canonical(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resign_receipt(receipt: dict[str, Any]) -> None:
    receipt["receipt_hash"] = replay_tool._hash_without(receipt, "receipt_hash")


def _resign_row(row: dict[str, Any]) -> None:
    payload = row["admission"]
    payload["worker_payload_hash"] = replay_tool._hash_without(
        payload, "worker_payload_hash"
    )
    row["raw_hash"] = replay_tool._hash_without(row, "raw_hash")


def _resign(root: Path) -> None:
    rows_path = root / "raw/workers.jsonl"
    protocol = replay_tool.load_json(root / "protocol.json")
    protocol["workers_jsonl_sha256"] = replay_tool.file_sha256(rows_path)
    protocol["negative_registry_sha256"] = replay_tool.file_sha256(
        root / "negative_registry.json"
    )
    protocol["protocol_hash"] = replay_tool._hash_without(protocol, "protocol_hash")
    artifact_tool._write_json(root / "protocol.json", protocol)
    rows = _load_rows(rows_path)
    registry = replay_tool.load_json(root / "negative_registry.json")
    try:
        summary = replay_tool._derive_summary(root, rows, protocol, registry)
    except (ValueError, TypeError, KeyError, OverflowError):
        stored = replay_tool.load_json(root / "summary.json")
        stored["workers_jsonl_sha256"] = protocol["workers_jsonl_sha256"]
        stored["negative_registry_sha256"] = protocol["negative_registry_sha256"]
        stored["summary_hash"] = replay_tool._hash_without(stored, "summary_hash")
        summary = stored
    artifact_tool._write_json(root / "summary.json", summary)
    artifact_tool._write_manifest(root)


def _mutate_row(
    root: Path, ordinal: int, mutation: Callable[[dict[str, Any]], None]
) -> None:
    path = root / "raw/workers.jsonl"
    rows = _load_rows(path)
    mutation(rows[ordinal])
    _resign_row(rows[ordinal])
    _write_rows(path, rows)


def _binary_candidate(root: Path) -> None:
    path = root / "raw/workers.jsonl"
    rows = _load_rows(path)
    payload = rows[0]["admission"]
    index = payload["binary_index"][0]
    sidecar = root / payload["binary_sidecar"]["relative_path"]
    data = bytearray(sidecar.read_bytes())
    offset = int(index["candidate_offset"])
    data[offset] ^= 1
    sidecar.write_bytes(bytes(data))
    index["candidate_sha256"] = hashlib.sha256(
        bytes(data[offset : offset + int(index["byte_count"])])
    ).hexdigest()
    payload["binary_sidecar"]["sha256"] = replay_tool.file_sha256(sidecar)
    _resign_row(rows[0])
    _write_rows(path, rows)


def _descriptor_count(row: dict[str, Any]) -> None:
    row["admission"]["physical"]["candidate_storage_count"] = 15


def _view_count(row: dict[str, Any]) -> None:
    row["admission"]["physical"]["base_dlpack_view_count"] = 15


def _empty_beta(row: dict[str, Any]) -> None:
    receipt = row["admission"]["buffer_receipt"]
    receipt["empty_beta_tokens"][0]["physical_buffer_present"] = True
    _resign_receipt(receipt)


def _accounting(row: dict[str, Any]) -> None:
    receipt = row["admission"]["buffer_receipt"]
    receipt["s4_1a_d2h_bytes"] = 0
    _resign_receipt(receipt)


def _fault_detail(row: dict[str, Any]) -> None:
    row["admission"]["error"]["detail_code"] = "BASE_DLPACK_VIEW_KEY_MISMATCH"


def _fault_cleanup(row: dict[str, Any]) -> None:
    row["admission"]["allocator"]["allocated_delta"] = 1024


def _claim(row: dict[str, Any]) -> None:
    row["admission"]["performance_claimed"] = True


def _ordinal(row: dict[str, Any]) -> None:
    row["admission"]["run_ordinal"] = 99


def _negative_registry(root: Path) -> None:
    path = root / "negative_registry.json"
    value = replay_tool.load_json(path)
    value["case_count"] = 1
    artifact_tool._write_json(path, value)


ATTACKS: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("candidate-binary", _binary_candidate),
    ("candidate-storage-count", lambda root: _mutate_row(root, 0, _descriptor_count)),
    ("base-view-count", lambda root: _mutate_row(root, 0, _view_count)),
    ("empty-beta-physical", lambda root: _mutate_row(root, 0, _empty_beta)),
    ("validation-accounting", lambda root: _mutate_row(root, 0, _accounting)),
    ("fault-detail", lambda root: _mutate_row(root, 5, _fault_detail)),
    ("fault-cleanup", lambda root: _mutate_row(root, 5, _fault_cleanup)),
    ("claim-flag", lambda root: _mutate_row(root, 0, _claim)),
    ("worker-ordinal", lambda root: _mutate_row(root, 0, _ordinal)),
    ("negative-registry", _negative_registry),
)


def probe(source: Path) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for name, attack in ATTACKS:
        with tempfile.TemporaryDirectory(prefix=f"s4-1a-tamper-{name}-") as tmp:
            target = Path(tmp) / "artifact"
            shutil.copytree(source, target)
            attack(target)
            _resign(target)
            rejected = False
            replay_error = ""
            try:
                replay_tool.replay(target)
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                rejected = True
                replay_error = str(error)
            cases.append(
                {
                    "name": name,
                    "outer_resign_completed": True,
                    "semantic_recompute_rejected": rejected,
                    "replay_error": replay_error,
                }
            )
    report: dict[str, object] = {
        "schema_version": "boundflow.asplos27-s4-1a-buffer-tamper/v1",
        "case_count": len(cases),
        "fully_outer_resigned_attack_count": len(cases),
        "rejected_count": sum(
            1 for item in cases if item["semantic_recompute_rejected"] is True
        ),
        "cases": cases,
        "performance_claimed": False,
    }
    if report["case_count"] != 10 or report["rejected_count"] != 10:
        raise RuntimeError("S4-1A tamper probe accepted a semantic attack")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = probe(args.artifact.resolve())
    payload = replay_tool.canonical(report) + "\n"
    if args.output is not None:
        args.output.resolve().write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
