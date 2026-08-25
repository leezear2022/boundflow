#!/usr/bin/env python3
"""Probe fully re-signed D1-A residual11 artifact tampering."""

# pylint: disable=too-many-locals,missing-function-docstring,import-outside-toplevel
# pylint: disable=duplicate-code

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

import torch

from scripts.run_r3_d1_residual11_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d1a-residual11-staged-v1"
REPORT_SCHEMA = "boundflow.r3-d1a-residual11-tamper/v1"


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
        raise TypeError("R3-D1 tamper JSON differs")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _tensor_hash(value: torch.Tensor) -> str:
    from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

    return production_tensor_sha256(value)


def _resign_manifest(root: Path) -> None:
    path = root / "manifest.json"
    manifest = _json(path)
    files = manifest["files"]
    manifest["files"] = {name: _file_hash(root / name) for name in sorted(files)}
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = _hash(manifest)
    _write(path, manifest)


def _resign_summary(root: Path) -> None:
    path = root / "summary.json"
    summary = _json(path)
    summary.pop("summary_hash", None)
    summary["summary_hash"] = _hash(summary)
    _write(path, summary)


def _resign_protocol(root: Path) -> None:
    path = root / "protocol.json"
    protocol = _json(path)
    protocol.pop("protocol_hash", None)
    protocol["protocol_hash"] = _hash(protocol)
    _write(path, protocol)


def _raw(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root / "raw/run-00.pt"
    return path, torch.load(path, map_location="cpu", weights_only=True)


def _mutate_input(root: Path) -> None:
    path, raw = _raw(root)
    raw["inputs"]["incoming"][0] += 1.0
    torch.save(raw, path)


def _mutate_output(root: Path, name: str) -> None:
    path, raw = _raw(root)
    raw[name][0] += 1.0
    raw["tensor_hashes"][name] = _tensor_hash(raw[name])
    torch.save(raw, path)


def _mutate_receipt(root: Path, name: str, value: object) -> None:
    path, raw = _raw(root)
    raw["receipt"][name] = value
    torch.save(raw, path)


def _mutate_summary(root: Path, name: str, value: object) -> None:
    path = root / "summary.json"
    summary = _json(path)
    summary[name] = value
    _write(path, summary)
    _resign_summary(root)


def _mutate_protocol(root: Path, name: str, value: object) -> None:
    path = root / "protocol.json"
    protocol = _json(path)
    protocol[name] = value
    _write(path, protocol)
    _resign_protocol(root)


def run(artifact: Path) -> dict[str, object]:
    cases: tuple[tuple[str, Callable[[Path], None]], ...] = (
        ("input", _mutate_input),
        ("candidate-output", lambda root: _mutate_output(root, "candidate_output")),
        ("reference-output", lambda root: _mutate_output(root, "reference_output")),
        (
            "schedule-hash",
            lambda root: _mutate_receipt(root, "scheduled_tir_hash", "0" * 64),
        ),
        ("launch-count", lambda root: _mutate_receipt(root, "launch_count", 1)),
        ("scratch-count", lambda root: _mutate_receipt(root, "scratch_count", 2)),
        (
            "timing-recorded",
            lambda root: _mutate_receipt(root, "timing_recorded", True),
        ),
        ("atol", lambda root: _mutate_protocol(root, "atol", 1.0)),
        ("residual6-open", lambda root: _mutate_summary(root, "residual6_open", False)),
        (
            "performance-claim",
            lambda root: _mutate_summary(root, "performance_claimed", True),
        ),
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="boundflow-r3d1a-tamper-") as temporary:
        for name, mutation in cases:
            root = Path(temporary) / name
            shutil.copytree(artifact, root)
            mutation(root)
            _resign_manifest(root)
            rejected = False
            error = ""
            try:
                replay(root)
            except (TypeError, ValueError) as caught:
                rejected = True
                error = str(caught)
            if not rejected:
                raise RuntimeError(f"R3-D1 tamper was accepted: {name}")
            results.append({"case": name, "rejected": True, "error": error})
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "case_count": len(results),
        "rejected_count": len(results),
        "cases": results,
        "timing_recorded": False,
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
        f"R3-D1-A tamper PASS: rejected={report['rejected_count']}/{report['case_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
