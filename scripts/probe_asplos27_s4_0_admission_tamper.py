#!/usr/bin/env python3
"""Apply ten fully outer-resigned attacks to the S4-0 artifact."""

# pylint: disable=protected-access,too-many-locals
# pylint: disable=unnecessary-lambda,missing-function-docstring

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

from scripts import replay_asplos27_s4_0_admission_stdlib as replay_tool
from scripts import run_asplos27_s4_0_admission_artifact as artifact_tool


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(replay_tool.canonical(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resign_worker(row: dict[str, Any]) -> None:
    admission = row["admission"]
    receipt = admission["receipt"]
    receipt["admission_hash"] = replay_tool._hash_payload(receipt, "admission_hash")
    admission["worker_payload_hash"] = replay_tool._hash_payload(
        admission, "worker_payload_hash"
    )
    row["raw_hash"] = replay_tool._hash_payload(row, "raw_hash")


def _resign(root: Path) -> None:
    rows_path = root / "raw/workers.jsonl"
    rows = _load_rows(rows_path)
    _write_rows(rows_path, rows)
    protocol = replay_tool.load_json(root / "protocol.json")
    protocol["workers_jsonl_sha256"] = replay_tool.file_sha256(rows_path)
    protocol["negative_registry_sha256"] = replay_tool.file_sha256(
        root / "negative_registry.json"
    )
    protocol["protocol_hash"] = replay_tool._hash_payload(protocol, "protocol_hash")
    artifact_tool._write_json(root / "protocol.json", protocol)
    summary = replay_tool.load_json(root / "summary.json")
    summary["workers_jsonl_sha256"] = protocol["workers_jsonl_sha256"]
    summary["negative_registry_sha256"] = protocol["negative_registry_sha256"]
    summary["summary_hash"] = replay_tool._hash_payload(summary, "summary_hash")
    artifact_tool._write_json(root / "summary.json", summary)
    artifact_tool._write_manifest(root)


def _mutate_receipt(field: str, value: object) -> Callable[[Path], None]:
    def mutation(root: Path) -> None:
        path = root / "raw/workers.jsonl"
        rows = _load_rows(path)
        rows[0]["admission"]["receipt"][field] = value
        _resign_worker(rows[0])
        _write_rows(path, rows)

    return mutation


def _mutate_projection(root: Path) -> None:
    path = root / "raw/workers.jsonl"
    rows = _load_rows(path)
    rows[0]["admission"]["provider"]["live_projection"][0]["content_hash"] = "1" * 64
    _resign_worker(rows[0])
    _write_rows(path, rows)


def _mutate_structure(root: Path) -> None:
    path = root / "raw/workers.jsonl"
    rows = _load_rows(path)
    rows[0]["admission"]["provider"]["structure"]["alpha_data"] = "builtins.dict"
    _resign_worker(rows[0])
    _write_rows(path, rows)


def _mutate_lease(root: Path) -> None:
    path = root / "raw/workers.jsonl"
    rows = _load_rows(path)
    rows[0]["admission"]["lease"]["retained_tensor_count_after_close"] = 1
    _resign_worker(rows[0])
    _write_rows(path, rows)


def _mutate_counter(root: Path) -> None:
    path = root / "raw/workers.jsonl"
    rows = _load_rows(path)
    rows[0]["admission"]["counters"]["provider_core_execute_count"] = 1
    _resign_worker(rows[0])
    _write_rows(path, rows)


def _mutate_negative(root: Path) -> None:
    path = root / "negative_registry.json"
    value = replay_tool.load_json(path)
    value["cases"][0]["exact_detail_and_reason_asserted"] = False
    artifact_tool._write_json(path, value)


def _mutate_worker_ordinal(root: Path) -> None:
    path = root / "raw/workers.jsonl"
    rows = _load_rows(path)
    rows[0]["admission"]["run_ordinal"] = 9
    _resign_worker(rows[0])
    _write_rows(path, rows)


ATTACKS: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("exact-call-hash", _mutate_receipt("exact_call_identity_hash", "1" * 64)),
    ("live-copy-count", _mutate_receipt("live_tensor_count", 13)),
    ("claim-flag", _mutate_receipt("performance_claimed", True)),
    ("slot-order", lambda root: _swap_slots(root)),
    ("provider-content", _mutate_projection),
    ("provider-structure", _mutate_structure),
    ("lease-close", _mutate_lease),
    ("provider-execute", _mutate_counter),
    ("negative-registry", _mutate_negative),
    ("worker-ordinal", _mutate_worker_ordinal),
)


def _swap_slots(root: Path) -> None:
    path = root / "raw/workers.jsonl"
    rows = _load_rows(path)
    slots = rows[0]["admission"]["receipt"]["slots"]
    slots[0], slots[1] = slots[1], slots[0]
    _resign_worker(rows[0])
    _write_rows(path, rows)


def probe(source: Path) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for name, attack in ATTACKS:
        with tempfile.TemporaryDirectory(prefix=f"s4-admission-tamper-{name}-") as tmp:
            target = Path(tmp) / "artifact"
            shutil.copytree(source, target)
            attack(target)
            _resign(target)
            rejected = False
            replay_error = ""
            try:
                replay_tool.replay(target)
            except (ValueError, TypeError, KeyError, OverflowError) as exception:
                rejected = True
                replay_error = str(exception)
            cases.append(
                {
                    "name": name,
                    "outer_resign_attempted": True,
                    "outer_resign_completed": True,
                    "outer_resign_error": "",
                    "rejected": rejected,
                    "replay_error": replay_error,
                }
            )
    report: dict[str, object] = {
        "schema_version": "boundflow.asplos27-s4-admission-tamper/v1",
        "case_count": len(cases),
        "rejected_count": sum(1 for row in cases if row["rejected"] is True),
        "fully_outer_resigned_attack_count": len(cases),
        "cases": cases,
        "performance_claimed": False,
    }
    if report["case_count"] != 10 or report["rejected_count"] != 10:
        raise RuntimeError("S4 admission tamper probe accepted a case")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = probe(args.artifact.resolve())
    payload = replay_tool.canonical(report) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
