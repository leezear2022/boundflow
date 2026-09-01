#!/usr/bin/env python3
"""Probe outer-re-signed R3-3 active-beta timing artifact tampering."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

from scripts.run_r3_3_active_beta_timing_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-3-active-beta-timing-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("R3-3 timing tamper JSON differs")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _mutate_raw(root: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    path = root / "raw/run_00.json"
    raw = _json(path)
    mutation(raw)
    raw.pop("worker_hash", None)
    raw["worker_hash"] = _hash(raw)
    _write(path, raw)


def _mutate_protocol(root: Path, field: str, value: object) -> None:
    path = root / "protocol.json"
    protocol = _json(path)
    protocol[field] = value
    protocol.pop("protocol_hash", None)
    protocol["protocol_hash"] = _hash(protocol)
    _write(path, protocol)
    manifest = _json(root / "manifest.json")
    manifest["protocol_hash"] = protocol["protocol_hash"]
    _write(root / "manifest.json", manifest)


def _mutate_summary(root: Path, field: str, value: object) -> None:
    path = root / "summary.json"
    summary = _json(path)
    summary[field] = value
    summary.pop("summary_hash", None)
    summary["summary_hash"] = _hash(summary)
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
    manifest["manifest_hash"] = _hash(manifest)
    _write(path, manifest)


def run(artifact: Path) -> dict[str, object]:
    cases: tuple[tuple[str, Callable[[Path], None]], ...] = (
        (
            "baseline-latency",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["pairs"][0].__setitem__("baseline_ms", 100.0),
            ),
        ),
        (
            "pair-speedup",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["pairs"][0].__setitem__("speedup", 100.0),
            ),
        ),
        (
            "worker-speedup",
            lambda root: _mutate_raw(
                root, lambda raw: raw.__setitem__("paired_speedup", 100.0)
            ),
        ),
        (
            "parity",
            lambda root: _mutate_raw(
                root, lambda raw: raw["parity"].__setitem__("allclose", False)
            ),
        ),
        (
            "module-call",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["module_call_count"].__setitem__("backward", 2),
            ),
        ),
        (
            "fallback",
            lambda root: _mutate_raw(
                root, lambda raw: raw.__setitem__("fallback_count", 1)
            ),
        ),
        (
            "memory",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["candidate_memory"].__setitem__(
                    "peak_allocated_bytes", 1
                ),
            ),
        ),
        (
            "order",
            lambda root: _mutate_raw(root, lambda raw: raw.__setitem__("order", "BA")),
        ),
        (
            "protocol-gate",
            lambda root: _mutate_protocol(root, "speedup_geomean_gate", 0.1),
        ),
        (
            "protocol-count",
            lambda root: _mutate_protocol(root, "pairs_per_worker", 1),
        ),
        (
            "summary-verdict",
            lambda root: _mutate_summary(root, "provisional_verdict", "PASS"),
        ),
        (
            "summary-open",
            lambda root: _mutate_summary(root, "r3_4_open", True),
        ),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="boundflow-r3-3-timing-tamper-") as tmp:
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
            raise RuntimeError(f"R3-3 timing tamper accepted: {name}")
    report: dict[str, object] = {
        "schema_version": "boundflow.r3-3-active-beta-timing-tamper/v1",
        "case_count": len(rows),
        "rejected_count": len(rows),
        "cases": rows,
        "performance_claimed": False,
    }
    report["report_hash"] = _hash(report)
    return report


def main() -> None:
    report = run(ARTIFACT)
    path = ARTIFACT / "tamper_report.json"
    _write(path, report)
    manifest_path = ARTIFACT / "manifest.json"
    manifest = _json(manifest_path)
    manifest["files"]["tamper_report.json"] = _file_hash(path)
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = _hash(manifest)
    _write(manifest_path, manifest)
    replay(ARTIFACT)
    print(f"R3-3 timing tamper PASS: {report['rejected_count']}/{report['case_count']}")


if __name__ == "__main__":
    main()
