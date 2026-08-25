#!/usr/bin/env python3
"""Probe fully outer-re-signed R3-3 correctness artifact tampering."""

# pylint: disable=missing-function-docstring,duplicate-code,too-many-locals

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

import torch

from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from scripts.run_r3_3_active_beta_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-3-active-beta-correctness-v1"


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
        raise TypeError("R3-3 tamper JSON differs")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _mutate_raw(root: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    path = root / "raw/run_00.pt"
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


def _tamper_beta_tensor(raw: dict[str, Any]) -> None:
    tensor = raw["outputs"]["compressed_beta_gradient"]
    tensor.add_(1.0)
    digest = production_tensor_sha256(tensor)
    raw["output_hashes"]["compressed_beta_gradient"] = digest
    for metric in raw["metrics"]:
        if metric["name"] == "compressed_beta_gradient":
            metric["candidate_hash"] = digest


def _tamper_reference_and_candidate(raw: dict[str, Any]) -> None:
    for owner, hashes in (
        ("outputs", "output_hashes"),
        ("references", "reference_hashes"),
    ):
        tensor = raw[owner]["output_bias"]
        tensor.add_(1.0)
        raw[hashes]["output_bias"] = production_tensor_sha256(tensor)
    for metric in raw["metrics"]:
        if metric["name"] == "output_bias":
            metric["candidate_hash"] = raw["output_hashes"]["output_bias"]
            metric["reference_hash"] = raw["reference_hashes"]["output_bias"]


def _tamper_cache(root: Path) -> None:
    path = root / "cache_probe.json"
    payload = _json(path)
    payload["rows"][0]["cache_event"] = "hit"
    _write(path, payload)


def run(artifact: Path) -> dict[str, object]:
    cases: tuple[tuple[str, Callable[[Path], None]], ...] = (
        ("beta-tensor", lambda root: _mutate_raw(root, _tamper_beta_tensor)),
        (
            "oracle-and-candidate",
            lambda root: _mutate_raw(root, _tamper_reference_and_candidate),
        ),
        (
            "beta-location",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["beta_locations"].__setitem__(0, 99),
            ),
        ),
        (
            "beta-sign",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["beta_signs"].__setitem__(0, 0),
            ),
        ),
        (
            "projection-unowned",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["projection_receipt"].__setitem__(
                    "unowned_native_zero_exact", False
                ),
            ),
        ),
        (
            "projection-beta",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["projection_receipt"].__setitem__(
                    "beta_numerical_passed", False
                ),
            ),
        ),
        (
            "launch",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["launch_receipt"].__setitem__(
                    "backward_launch_count", 2
                ),
            ),
        ),
        (
            "empty-specialization",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw.__setitem__(
                    "empty_beta_specialization_rejected", False
                ),
            ),
        ),
        ("cache-sequence", _tamper_cache),
        (
            "protocol-tolerance",
            lambda root: _mutate_protocol(
                root, "tolerance", {"atol": 1.0, "rtol": 1.0, "sign_exact": True}
            ),
        ),
        (
            "summary-correctness",
            lambda root: _mutate_summary(
                root, "active_beta_correctness_admitted", False
            ),
        ),
        (
            "summary-scope",
            lambda root: _mutate_summary(root, "r3_4_open", True),
        ),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="boundflow-r3-3-tamper-") as temporary:
        for name, mutation in cases:
            root = Path(temporary) / name
            shutil.copytree(artifact, root)
            mutation(root)
            _resign_manifest(root)
            try:
                replay(root)
            except (KeyError, TypeError, ValueError) as caught:
                rows.append({"case": name, "rejected": True, "error": str(caught)})
                continue
            raise RuntimeError(f"R3-3 tamper accepted: {name}")
    report: dict[str, object] = {
        "schema_version": "boundflow.r3-3-active-beta-tamper/v1",
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
    print(f"R3-3 tamper PASS: {report['rejected_count']}/{report['case_count']}")


if __name__ == "__main__":
    main()
