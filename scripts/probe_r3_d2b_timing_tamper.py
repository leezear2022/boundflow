#!/usr/bin/env python3
"""Probe fully re-signed D2-B timing artifact tampering."""

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

from scripts.run_r3_d2b_timing_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d2b-wrapper-timing-v1"


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
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("R3-D2B timing tamper JSON differs")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n")


def _mutate_raw(
    root: Path, name: str, mutation: Callable[[dict[str, Any]], None]
) -> None:
    path = root / "raw" / name
    raw = torch.load(path, map_location="cpu", weights_only=True)
    mutation(raw)
    torch.save(raw, path)


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


def _resign_manifest(root: Path) -> None:
    path = root / "manifest.json"
    manifest = _json(path)
    manifest["files"] = {
        name: _file_hash(root / name) for name in sorted(manifest["files"])
    }
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = _hash(manifest)
    _write(path, manifest)


def run(artifact: Path) -> dict[str, object]:
    candidate = "run-00-2-d2b.pt"
    control = "run-00-1-d1c.pt"

    def latency(raw: dict[str, Any]) -> None:
        raw["latency_ns"] = [value * 2 for value in raw["latency_ns"]]
        raw["median_latency_ns"] *= 2

    def terminal(raw: dict[str, Any]) -> None:
        raw["terminal_lower"][0, 0] += 1.0
        from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

        raw["terminal_lower_sha256"] = production_tensor_sha256(raw["terminal_lower"])

    cases: tuple[tuple[str, Callable[[Path], None]], ...] = (
        ("latency", lambda root: _mutate_raw(root, candidate, latency)),
        (
            "region",
            lambda root: _mutate_raw(
                root,
                candidate,
                lambda raw: raw.__setitem__("coefficient_sign_region_ms", 1000.0),
            ),
        ),
        ("terminal", lambda root: _mutate_raw(root, candidate, terminal)),
        (
            "execution",
            lambda root: _mutate_raw(
                root,
                candidate,
                lambda raw: raw["execution"].__setitem__("optimizer_mutation_count", 8),
            ),
        ),
        (
            "receipt-claim",
            lambda root: _mutate_raw(
                root,
                candidate,
                lambda raw: raw["d2b_receipt"].__setitem__("performance_claimed", True),
            ),
        ),
        (
            "memory",
            lambda root: _mutate_raw(
                root,
                candidate,
                lambda raw: raw["memory"].__setitem__("peak_allocated", 10**12),
            ),
        ),
        (
            "control-region",
            lambda root: _mutate_raw(
                root,
                control,
                lambda raw: raw.__setitem__("coefficient_sign_region_ms", 1.0),
            ),
        ),
        ("protocol-region", lambda root: _mutate_protocol(root, "region_gate", 1.0)),
        (
            "protocol-research",
            lambda root: _mutate_protocol(root, "research_gate", 1.0),
        ),
        (
            "summary-research",
            lambda root: _mutate_summary(root, "research_gate", False),
        ),
        ("summary-open", lambda root: _mutate_summary(root, "r3_3_open", True)),
        (
            "summary-claim",
            lambda root: _mutate_summary(root, "performance_claimed", True),
        ),
    )
    rows = []
    with tempfile.TemporaryDirectory(
        prefix="boundflow-r3d2b-timing-tamper-"
    ) as temporary:
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
            raise RuntimeError(f"R3-D2B timing tamper accepted: {name}")
    report: dict[str, object] = {
        "schema_version": "boundflow.r3-d2b-wrapper-timing-tamper/v1",
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
    print(
        f"R3-D2B timing tamper PASS: {report['rejected_count']}/{report['case_count']}"
    )


if __name__ == "__main__":
    main()
