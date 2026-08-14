#!/usr/bin/env python3
"""Probe outer-resigned tampering of one FSG4/B3 counter artifact."""

# pylint: disable=protected-access,too-many-locals,missing-function-docstring

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from boundflow.runtime.fsg3_same_solver_timing import canonical_hash
from boundflow.runtime.fsg4_b3_explicit_counters import (
    events_from_rows,
    Fsg4B3CounterRecorder,
)
from scripts import run_fsg4_b3_counter_diagnostic as diagnostic

REPORT_SCHEMA = "boundflow.fsg4-b3-counter-tamper-report/v1"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("FSG4/B3 tamper JSON root differs")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_events(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _resign(root: Path, *, derive_counts: bool = True) -> None:
    worker_path = root / "worker.json"
    events_path = root / "events.jsonl"
    report_path = root / "report.json"
    manifest_path = root / "manifest.json"
    worker = _load(worker_path)
    report = _load(report_path)
    snapshot = report["snapshot"]
    if not isinstance(snapshot, dict):
        raise TypeError("FSG4/B3 tamper snapshot differs")
    run = worker["run"]
    if not isinstance(run, dict) or not isinstance(run.get("semantics"), dict):
        raise TypeError("FSG4/B3 tamper worker semantics differ")
    rows = _events(events_path)
    if derive_counts:
        parsed = events_from_rows(rows)
        snapshot["counts"] = Fsg4B3CounterRecorder(events=list(parsed)).counts()
    snapshot["semantic_hash"] = canonical_hash(run["semantics"])
    snapshot["worker_result_sha256"] = _sha(worker_path)
    snapshot_payload = dict(snapshot)
    snapshot_payload.pop("snapshot_hash", None)
    snapshot["snapshot_hash"] = canonical_hash(snapshot_payload)
    report["event_count"] = len(rows)
    report["event_journal_sha256"] = _sha(events_path)
    report_payload = dict(report)
    report_payload.pop("report_hash", None)
    report["report_hash"] = canonical_hash(report_payload)
    _write(report_path, report)
    manifest = _load(manifest_path)
    manifest["files"] = {
        name: _sha(root / name)
        for name in ("events.jsonl", "report.json", "worker.json")
    }
    manifest["report_hash"] = report["report_hash"]
    manifest_payload = dict(manifest)
    manifest_payload.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_hash(manifest_payload)
    _write(manifest_path, manifest)


def _counter_report_only(root: Path) -> None:
    report = _load(root / "report.json")
    snapshot = report["snapshot"]
    assert isinstance(snapshot, dict) and isinstance(snapshot["counts"], dict)
    snapshot["counts"]["forward_trace_build_count"] = 4
    _write(root / "report.json", report)
    _resign(root, derive_counts=False)


def _counter_and_journal(root: Path) -> None:
    rows = _events(root / "events.jsonl")
    row = next(item for item in rows if item["counter"] == "forward_trace_build_count")
    row["counter"] = "stable_hash_call_count"
    _write_events(root / "events.jsonl", rows)
    _resign(root)


def _delete_journal_event(root: Path) -> None:
    rows = _events(root / "events.jsonl")
    index = next(
        index
        for index, row in enumerate(rows)
        if row["counter"] == "optimizer_bound_evaluation_call_count"
    )
    rows.pop(index)
    for ordinal, row in enumerate(rows):
        row["ordinal"] = ordinal
    _write_events(root / "events.jsonl", rows)
    _resign(root)


def _worker_semantic(root: Path) -> None:
    worker = _load(root / "worker.json")
    run = worker["run"]
    assert isinstance(run, dict) and isinstance(run["semantics"], dict)
    lower = run["semantics"]["lower_values"]
    assert isinstance(lower, list)
    lower[0] = float(lower[0]) + 0.1
    _write(root / "worker.json", worker)
    _resign(root)


def _provider_count(root: Path) -> None:
    worker = _load(root / "worker.json")
    run = worker["run"]
    assert isinstance(run, dict) and isinstance(run["execution"], dict)
    run["execution"]["provider_core_call_count"] = 1
    _write(root / "worker.json", worker)
    report = _load(root / "report.json")
    snapshot = report["snapshot"]
    assert isinstance(snapshot, dict)
    snapshot["provider_core_call_count"] = 1
    _write(root / "report.json", report)
    _resign(root)


def _code_revision(root: Path) -> None:
    manifest = _load(root / "manifest.json")
    revision = manifest["code_revision"]
    assert isinstance(revision, dict)
    revision["scripts/run_fsg4_b3_counter_diagnostic.py"] = "0" * 64
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_hash(payload)
    _write(root / "manifest.json", manifest)


ATTACKS: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("counter-report-only-outer-resign", _counter_report_only),
    ("counter-and-journal-outer-resign", _counter_and_journal),
    ("delete-journal-event-outer-resign", _delete_journal_event),
    ("worker-semantic-outer-resign", _worker_semantic),
    ("provider-count-outer-resign", _provider_count),
    ("code-revision-outer-resign", _code_revision),
)


def _probe(artifact: Path) -> dict[str, object]:
    diagnostic._replay(artifact)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="fsg4-b3-counter-tamper-") as raw:
        parent = Path(raw)
        for ordinal, (name, attack) in enumerate(ATTACKS):
            target = parent / f"attack-{ordinal}"
            shutil.copytree(artifact, target)
            attack(target)
            try:
                diagnostic._replay(target)
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
        raise ValueError("FSG4/B3 tamper probe did not reject every attack")
    result: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "artifact_manifest_sha256": _sha(artifact / "manifest.json"),
        "attack_count": len(rows),
        "rejected_count": sum(row["rejected"] is True for row in rows),
        "rows": rows,
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
