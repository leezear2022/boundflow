#!/usr/bin/env python3
"""Probe fully re-signed D2-A backward attribution artifact tampering."""

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

from scripts.run_r3_d2a_backward_attribution_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d2a-backward-attribution-v1"


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
        raise TypeError("R3-D2A tamper JSON differs")
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


def _mutate_raw(root: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    path = root / "raw/run-00.pt"
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
    def terminal(raw: dict[str, Any]) -> None:
        raw["terminal_lower"][0, 0] += 1.0
        from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

        raw["terminal_lower_sha256"] = production_tensor_sha256(raw["terminal_lower"])

    cases: tuple[tuple[str, Callable[[Path], None]], ...] = (
        (
            "host",
            lambda root: _mutate_raw(
                root, lambda raw: raw.__setitem__("host_wrapper_ns", 1)
            ),
        ),
        (
            "readiness",
            lambda root: _mutate_raw(
                root, lambda raw: raw.__setitem__("readiness_pass", False)
            ),
        ),
        (
            "anchor",
            lambda root: _mutate_raw(
                root, lambda raw: raw.__setitem__("anchor_host_ns", 1)
            ),
        ),
        (
            "phase-duration",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["phase_ms"]["coefficient_sign"].__setitem__(0, 1000.0),
            ),
        ),
        (
            "phase-total",
            lambda root: _mutate_raw(
                root, lambda raw: raw["phase_totals_ms"].__setitem__("backward", 1.0)
            ),
        ),
        ("terminal", lambda root: _mutate_raw(root, terminal)),
        (
            "symbol",
            lambda root: _mutate_raw(
                root, lambda raw: raw["symbol_ms"].pop("b1:boundflow_r31b1_residual6")
            ),
        ),
        (
            "headline-flag",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw.__setitem__("symbol_profile_headline_forbidden", False),
            ),
        ),
        (
            "counter",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["execution"].__setitem__("optimizer_mutation_count", 8),
            ),
        ),
        (
            "protocol-generic-cap",
            lambda root: _mutate_protocol(root, "generic_required_cap", 100.0),
        ),
        (
            "protocol-residual-cap",
            lambda root: _mutate_protocol(
                root, "verified_residual_required_cap", 100.0
            ),
        ),
        (
            "summary-route",
            lambda root: _mutate_summary(root, "selected_route", "anything"),
        ),
        ("summary-open", lambda root: _mutate_summary(root, "d2b_open", False)),
        (
            "summary-claim",
            lambda root: _mutate_summary(root, "performance_claimed", True),
        ),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="boundflow-r3d2a-tamper-") as temporary:
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
            raise RuntimeError(f"R3-D2A tamper accepted: {name}")
    report: dict[str, object] = {
        "schema_version": "boundflow.r3-d2a-backward-attribution-tamper/v1",
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
    print(f"R3-D2A tamper PASS: {report['rejected_count']}/{report['case_count']}")


if __name__ == "__main__":
    main()
