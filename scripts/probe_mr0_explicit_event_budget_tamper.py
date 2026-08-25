#!/usr/bin/env python3
"""Probe fully re-signed MR0 explicit-event artifact tampering."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

from boundflow.runtime.mr0_explicit_event_budget import canonical_hash
from scripts.run_mr0_explicit_event_budget_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/measurement-recovery/mr0-explicit-event-budget-resnet2b-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("MR0 tamper JSON differs")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _mutate_raw(root: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    path = sorted((root / "raw").glob("*.json"))[0]
    raw = _json(path)
    mutation(raw)
    for row in raw.get("budget_rows", []):
        row.pop("row_hash", None)
        row["row_hash"] = canonical_hash(row)
    raw.pop("worker_hash", None)
    raw["worker_hash"] = canonical_hash(raw)
    _write(path, raw)


def _mutate_protocol(root: Path, field: str, value: object) -> None:
    path = root / "protocol.json"
    protocol = _json(path)
    protocol[field] = value
    protocol.pop("protocol_hash", None)
    protocol["protocol_hash"] = canonical_hash(protocol)
    _write(path, protocol)
    manifest = _json(root / "manifest.json")
    manifest["protocol_hash"] = protocol["protocol_hash"]
    _write(root / "manifest.json", manifest)


def _mutate_summary(root: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    path = root / "summary.json"
    summary = _json(path)
    mutation(summary)
    summary.pop("summary_hash", None)
    summary["summary_hash"] = canonical_hash(summary)
    _write(path, summary)
    manifest = _json(root / "manifest.json")
    manifest["summary_hash"] = summary["summary_hash"]
    _write(root / "manifest.json", manifest)


def _resign(root: Path) -> None:
    path = root / "manifest.json"
    manifest = _json(path)
    manifest["files"] = {
        name: _file_hash(root / name) for name in sorted(manifest["files"])
    }
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write(path, manifest)


def run(artifact: Path) -> dict[str, object]:
    cases: tuple[tuple[str, Callable[[Path], None]], ...] = (
        (
            "control-sample",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["budget_rows"][-1]["control_ms"].__setitem__(0, 100.0),
            ),
        ),
        (
            "instrumented-sample",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["budget_rows"][-1]["instrumented_ms"].__setitem__(
                    0, 0.0001
                ),
            ),
        ),
        (
            "row-ratio",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["budget_rows"][-1].__setitem__("overhead_ratio", 1.0),
            ),
        ),
        (
            "budget",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["budget_rows"][-1].__setitem__("budget", 8),
            ),
        ),
        (
            "event-count",
            lambda root: _mutate_raw(
                root, lambda raw: raw.__setitem__("event_object_count", 2)
            ),
        ),
        (
            "semantic",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["semantic_receipt"].__setitem__("exact", False),
            ),
        ),
        (
            "source-digest",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw.__setitem__("source_capture_sha256", "0" * 64),
            ),
        ),
        (
            "order",
            lambda root: _mutate_raw(root, lambda raw: raw.__setitem__("order", "IC")),
        ),
        (
            "performance-claim",
            lambda root: _mutate_raw(
                root, lambda raw: raw.__setitem__("performance_claimed", True)
            ),
        ),
        (
            "protocol-gate",
            lambda root: _mutate_protocol(root, "geomean_gate", 2.0),
        ),
        (
            "summary-verdict",
            lambda root: _mutate_summary(
                root, lambda summary: summary.__setitem__("verdict", "PASS")
            ),
        ),
        (
            "summary-scope-open",
            lambda root: _mutate_summary(
                root, lambda summary: summary.__setitem__("same_solver_open", True)
            ),
        ),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="boundflow-mr0-tamper-") as tmp:
        for name, mutation in cases:
            root = Path(tmp) / name
            shutil.copytree(artifact, root)
            mutation(root)
            _resign(root)
            try:
                replay(root)
            except (KeyError, TypeError, ValueError) as caught:
                rows.append({"case": name, "rejected": True, "error": str(caught)})
                continue
            raise RuntimeError(f"MR0 tamper accepted: {name}")
    report: dict[str, object] = {
        "schema_version": "boundflow.mr0-explicit-event-budget-tamper/v1",
        "case_count": len(rows),
        "rejected_count": len(rows),
        "cases": rows,
        "performance_claimed": False,
    }
    report["report_hash"] = canonical_hash(report)
    return report


def main() -> None:
    report = run(ARTIFACT)
    path = ARTIFACT / "tamper_report.json"
    _write(path, report)
    manifest_path = ARTIFACT / "manifest.json"
    manifest = _json(manifest_path)
    manifest["files"]["tamper_report.json"] = _file_hash(path)
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write(manifest_path, manifest)
    replay(ARTIFACT)
    print(f"MR0 tamper PASS: {report['rejected_count']}/{report['case_count']}")


if __name__ == "__main__":
    main()
