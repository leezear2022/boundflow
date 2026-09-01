#!/usr/bin/env python3
"""Probe outer-resigned attacks on five-pair FSG4/B3 correctness evidence."""

# pylint: disable=protected-access,missing-function-docstring,too-many-locals

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from boundflow.runtime.fsg3_same_solver_timing import canonical_hash
from scripts import probe_fsg4_b3_counter_artifact_tamper as counter_probe
from scripts import run_fsg4_b3_correctness_pairs as pairs

REPORT_SCHEMA = "boundflow.fsg4-b3-five-fresh-tamper-report/v1"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("FSG4/B3 five-fresh tamper JSON root differs")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resign_report(report: dict[str, object]) -> None:
    payload = dict(report)
    payload.pop("report_hash", None)
    report["report_hash"] = canonical_hash(payload)


def _resign_root(root: Path) -> None:
    protocol = _load(root / "protocol.json")
    report = _load(root / "report.json")
    manifest = _load(root / "manifest.json")
    manifest["protocol_hash"] = protocol["protocol_hash"]
    manifest["report_hash"] = report["report_hash"]
    manifest["files"] = pairs._all_files(root)
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_hash(payload)
    _write(root / "manifest.json", manifest)


def _report_projection(root: Path) -> None:
    report = _load(root / "report.json")
    report["all_counter_gates_passed"] = False
    _resign_report(report)
    _write(root / "report.json", report)
    _resign_root(root)


def _protocol_schedule(root: Path) -> None:
    protocol = _load(root / "protocol.json")
    schedule = protocol["schedule"]
    assert isinstance(schedule, list) and isinstance(schedule[0], dict)
    schedule[0]["positions"] = ["B3-C", "B2"]
    protocol_payload = dict(protocol)
    protocol_payload.pop("protocol_hash", None)
    protocol["protocol_hash"] = canonical_hash(protocol_payload)
    _write(root / "protocol.json", protocol)
    report = _load(root / "report.json")
    report["protocol_hash"] = protocol["protocol_hash"]
    _resign_report(report)
    _write(root / "report.json", report)
    _resign_root(root)


def _nested_counter(root: Path) -> None:
    target = pairs._run_directory(root, 0, 1, "B3-C")
    counter_probe._counter_and_journal(target)
    _resign_root(root)


def _nested_worker_semantic(root: Path) -> None:
    target = pairs._run_directory(root, 0, 1, "B3-C")
    counter_probe._worker_semantic(target)
    _resign_root(root)


def _nested_audit_receipt(root: Path) -> None:
    target = pairs._run_directory(root, 0, 1, "B3-C")
    worker_path = target / "worker.json"
    worker = _load(worker_path)
    diagnostics = worker["diagnostics"]
    assert isinstance(diagnostics, dict)
    receipts = diagnostics["commit_receipts"]
    assert isinstance(receipts, list) and isinstance(receipts[0], dict)
    receipt = receipts[0]
    receipt["candidate_d2h_copy_count"] = 1
    payload = dict(receipt)
    payload.pop("commit_hash", None)
    receipt["commit_hash"] = canonical_hash(payload)
    _write(worker_path, worker)
    counter_probe._resign(target)
    _resign_root(root)


def _swap_pair_positions(root: Path) -> None:
    pair = root / "runs/pair-00"
    left = pair / "position-0-b2"
    right = pair / "position-1-b3-c"
    temporary = pair / "position-swap-temporary"
    left.rename(temporary)
    right.rename(left)
    temporary.rename(right)
    _resign_root(root)


def _delete_raw_run(root: Path) -> None:
    shutil.rmtree(pairs._run_directory(root, 4, 1, "B3-C"))
    _resign_root(root)


ATTACKS: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("report-projection-outer-resign", _report_projection),
    ("protocol-schedule-outer-resign", _protocol_schedule),
    ("nested-counter-journal-outer-resign", _nested_counter),
    ("nested-worker-semantic-outer-resign", _nested_worker_semantic),
    ("nested-audit-receipt-outer-resign", _nested_audit_receipt),
    ("swap-pair-positions-outer-resign", _swap_pair_positions),
    ("delete-raw-run-outer-resign", _delete_raw_run),
)


def _probe(artifact: Path) -> dict[str, object]:
    pairs._replay(artifact)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="fsg4-b3-five-fresh-tamper-") as raw:
        parent = Path(raw)
        for ordinal, (name, attack) in enumerate(ATTACKS):
            target = parent / f"attack-{ordinal}"
            shutil.copytree(artifact, target)
            attack(target)
            try:
                pairs._replay(target)
            except Exception as error:  # pylint: disable=broad-exception-caught
                rows.append(
                    {
                        "attack": name,
                        "rejected": True,
                        "exception_type": type(error).__name__,
                        "message": str(error),
                    }
                )
            else:
                rows.append({"attack": name, "rejected": False})
    if len(rows) != len(ATTACKS) or not all(row["rejected"] is True for row in rows):
        raise ValueError("FSG4/B3 five-fresh tamper probe accepted an attack")
    result: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "artifact_manifest_sha256": pairs._file_sha256(artifact / "manifest.json"),
        "attack_count": len(rows),
        "rejected_count": len(rows),
        "rows": rows,
        "timing_admitted": False,
        "performance_claimed": False,
    }
    result["report_hash"] = canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = _probe(args.artifact_dir.resolve())
    _write(args.report.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
