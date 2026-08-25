#!/usr/bin/env python3
"""Probe fully re-signed D1-C cumulative wrapper artifact tampering."""

# pylint: disable=missing-function-docstring,duplicate-code,too-many-locals
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

import torch

from scripts.run_r3_d1c_wrapper_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d1c-wrapper-formal-v1"


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
        raise TypeError("R3-D1C tamper JSON differs")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _resign_json(root: Path, name: str, hash_name: str) -> None:
    path = root / name
    value = _json(path)
    value.pop(hash_name, None)
    value[hash_name] = _hash(value)
    _write(path, value)


def _resign_manifest(root: Path) -> None:
    path = root / "manifest.json"
    manifest = _json(path)
    manifest["files"] = {
        name: _file_hash(root / name) for name in sorted(manifest["files"])
    }
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = _hash(manifest)
    _write(path, manifest)


def _raw_path(root: Path) -> Path:
    matches = sorted((root / "raw").glob("run-00-*-d1c.pt"))
    if len(matches) != 1:
        raise ValueError("R3-D1C tamper raw inventory differs")
    return matches[0]


def _mutate_raw(root: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    path = _raw_path(root)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    mutation(raw)
    torch.save(raw, path)


def _mutate_protocol(root: Path, field: str, value: object) -> None:
    path = root / "protocol.json"
    protocol = _json(path)
    protocol[field] = value
    _write(path, protocol)
    _resign_json(root, "protocol.json", "protocol_hash")
    manifest = _json(root / "manifest.json")
    manifest["protocol_hash"] = _json(path)["protocol_hash"]
    _write(root / "manifest.json", manifest)


def _mutate_summary(root: Path, field: str, value: object) -> None:
    path = root / "summary.json"
    summary = _json(path)
    summary[field] = value
    _write(path, summary)
    _resign_json(root, "summary.json", "summary_hash")
    manifest = _json(root / "manifest.json")
    manifest["summary_hash"] = _json(path)["summary_hash"]
    _write(root / "manifest.json", manifest)


def run(artifact: Path) -> dict[str, object]:
    def duration(raw: dict[str, Any]) -> None:
        raw["latency_ns"] = [10_000_000_000] * 30

    def terminal(raw: dict[str, Any]) -> None:
        raw["terminal_lower"][0, 0] += 1.0
        from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

        raw["terminal_lower_sha256"] = production_tensor_sha256(raw["terminal_lower"])

    def alpha(raw: dict[str, Any]) -> None:
        raw["terminal_alpha"][0, 0, 0, 0] += 1.0
        from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

        raw["terminal_alpha_sha256"] = production_tensor_sha256(raw["terminal_alpha"])

    cases: tuple[tuple[str, Callable[[Path], None]], ...] = (
        ("duration", lambda root: _mutate_raw(root, duration)),
        ("terminal-lower", lambda root: _mutate_raw(root, terminal)),
        ("terminal-alpha", lambda root: _mutate_raw(root, alpha)),
        (
            "optimizer-count",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["execution"].__setitem__("optimizer_mutation_count", 8),
            ),
        ),
        (
            "schedule-hash",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["d1c_receipt"].__setitem__(
                    "scheduled_tir_hash", "0" * 64
                ),
            ),
        ),
        (
            "scratch-pointer",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["d1c_receipt"].__setitem__(
                    "scratch_region_pointers", (1, 2)
                ),
            ),
        ),
        (
            "raw-claim",
            lambda root: _mutate_raw(
                root, lambda raw: raw.__setitem__("formal_performance_claimed", True)
            ),
        ),
        (
            "protocol-wrapper-gate",
            lambda root: _mutate_protocol(root, "wrapper_geomean_gate", 0.1),
        ),
        (
            "protocol-cumulative-gate",
            lambda root: _mutate_protocol(root, "cumulative_gate", 1.0),
        ),
        (
            "summary-wrapper",
            lambda root: _mutate_summary(root, "wrapper_geomean_speedup", 10.0),
        ),
        (
            "summary-status",
            lambda root: _mutate_summary(
                root, "status", "VALIDATED-R3-D1-P-LOCAL-WRAPPER"
            ),
        ),
        ("summary-r3-open", lambda root: _mutate_summary(root, "r3_3_open", True)),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="boundflow-r3d1c-tamper-") as temporary:
        for name, mutation in cases:
            root = Path(temporary) / name
            shutil.copytree(artifact, root)
            mutation(root)
            _resign_manifest(root)
            try:
                replay(root)
            except (TypeError, ValueError) as caught:
                rows.append({"case": name, "rejected": True, "error": str(caught)})
                continue
            raise RuntimeError(f"R3-D1C tamper accepted: {name}")
    report: dict[str, object] = {
        "schema_version": "boundflow.r3-d1c-wrapper-tamper/v1",
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
    print(f"R3-D1C tamper PASS: {report['rejected_count']}/{report['case_count']}")


if __name__ == "__main__":
    main()
