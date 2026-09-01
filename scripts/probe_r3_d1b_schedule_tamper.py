#!/usr/bin/env python3
"""Probe fully re-signed D1-B fixed schedule artifact tampering."""

# pylint: disable=missing-function-docstring,duplicate-code,too-many-locals

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

from scripts.run_r3_d1b_schedule_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d1b-schedule-formal-v1"


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
        raise TypeError("R3-D1B tamper JSON differs")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _resign(root: Path, name: str, hash_name: str) -> None:
    path = root / name
    value = _json(path)
    value.pop(hash_name, None)
    value[hash_name] = _hash(value)
    _write(path, value)


def _resign_manifest(root: Path) -> None:
    path = root / "manifest.json"
    value = _json(path)
    value["files"] = {name: _file_hash(root / name) for name in sorted(value["files"])}
    value.pop("manifest_hash", None)
    value["manifest_hash"] = _hash(value)
    _write(path, value)


def _mutate_raw(root: Path, path: tuple[str, ...], value: object) -> None:
    target = root / "raw/run-00.json"
    raw = _json(target)
    cursor: Any = raw
    for name in path[:-1]:
        cursor = cursor[int(name)] if isinstance(cursor, list) else cursor[name]
    if isinstance(cursor, list):
        cursor[int(path[-1])] = value
    else:
        cursor[path[-1]] = value
    _write(target, raw)


def _mutate_json(
    root: Path, name: str, hash_name: str, field: str, value: object
) -> None:
    path = root / name
    payload = _json(path)
    payload[field] = value
    _write(path, payload)
    _resign(root, name, hash_name)


def run(artifact: Path) -> dict[str, object]:
    cases: tuple[tuple[str, Callable[[Path], None]], ...] = (
        (
            "duration",
            lambda root: _mutate_raw(
                root, ("measurement", "candidate_ms"), [10.0] * 10
            ),
        ),
        (
            "derived-speedup",
            lambda root: _mutate_raw(root, ("measurement", "speedup"), 1000.0),
        ),
        (
            "threads",
            lambda root: _mutate_raw(root, ("measurement", "threads_per_block"), 128),
        ),
        (
            "schedule-hash",
            lambda root: _mutate_raw(
                root, ("measurement", "candidate_scheduled_tir_hash"), "0" * 64
            ),
        ),
        (
            "scratch",
            lambda root: _mutate_raw(root, ("measurement", "scratch_count"), 3),
        ),
        (
            "sign",
            lambda root: _mutate_raw(root, ("measurement", "sign_exact"), False),
        ),
        (
            "protocol-gate",
            lambda root: _mutate_json(
                root, "protocol.json", "protocol_hash", "isolated_opportunity_gate", 1.0
            ),
        ),
        (
            "summary-worst",
            lambda root: _mutate_json(
                root, "summary.json", "summary_hash", "worst_speedup", 1000.0
            ),
        ),
        (
            "d1c-open",
            lambda root: _mutate_json(
                root, "summary.json", "summary_hash", "d1c_wrapper_open", False
            ),
        ),
        (
            "wrapper-claim",
            lambda root: _mutate_json(
                root,
                "summary.json",
                "summary_hash",
                "wrapper_performance_claimed",
                True,
            ),
        ),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="boundflow-r3d1b-tamper-") as temporary:
        for name, mutation in cases:
            root = Path(temporary) / name
            shutil.copytree(artifact, root)
            mutation(root)
            if name.startswith("summary") or name in ("d1c-open", "wrapper-claim"):
                manifest = _json(root / "manifest.json")
                manifest["summary_hash"] = _json(root / "summary.json")["summary_hash"]
                _write(root / "manifest.json", manifest)
            if name == "protocol-gate":
                manifest = _json(root / "manifest.json")
                manifest["protocol_hash"] = _json(root / "protocol.json")[
                    "protocol_hash"
                ]
                _write(root / "manifest.json", manifest)
            _resign_manifest(root)
            try:
                replay(root)
            except (TypeError, ValueError) as caught:
                rows.append({"case": name, "rejected": True, "error": str(caught)})
                continue
            raise RuntimeError(f"R3-D1B tamper was accepted: {name}")
    report: dict[str, object] = {
        "schema_version": "boundflow.r3-d1b-schedule-tamper/v1",
        "case_count": len(rows),
        "rejected_count": len(rows),
        "cases": rows,
        "isolated_performance_claimed": True,
        "wrapper_performance_claimed": False,
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
    print(
        f"R3-D1B tamper PASS: rejected={report['rejected_count']}/{report['case_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
