#!/usr/bin/env python3
"""Probe fully re-signed D2-B correctness artifact tampering."""

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

from scripts.run_r3_d2b_correctness_artifact import replay, TENSOR_NAMES

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d2b-correctness-v1"


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
        raise TypeError("R3-D2B tamper JSON differs")
    return value


def _write(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _tensor_hash(value: torch.Tensor) -> str:
    from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256

    return production_tensor_sha256(value)


def _resign_worker(raw: dict[str, Any]) -> None:
    metadata = raw["metadata"]
    metadata["step_hashes"] = []
    for step in raw["steps"]:
        step["tensor_hashes"] = {
            name: _tensor_hash(step[name]) for name in TENSOR_NAMES
        }
        projection = {
            name: value for name, value in step.items() if name not in TENSOR_NAMES
        }
        metadata["step_hashes"].append(_hash(projection))
    metadata["initial_alpha_sha256"] = _tensor_hash(raw["initial_alpha"])
    metadata["terminal_alpha_sha256"] = _tensor_hash(raw["terminal_alpha"])
    metadata.pop("trajectory_hash", None)
    metadata["trajectory_hash"] = _hash(metadata)


def _mutate_raw(
    root: Path, name: str, mutation: Callable[[dict[str, Any]], None]
) -> None:
    path = root / "raw" / name
    raw = torch.load(path, map_location="cpu", weights_only=True)
    mutation(raw)
    _resign_worker(raw)
    torch.save(raw, path)


def _resign_json(root: Path, name: str, hash_name: str) -> None:
    path = root / name
    value = _json(path)
    value.pop(hash_name, None)
    value[hash_name] = _hash(value)
    _write(path, value)


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
    d2b = "run-00-d2b.pt"
    d1c = "run-00-d1c.pt"
    cases: tuple[tuple[str, Callable[[Path], None]], ...] = (
        (
            "lower",
            lambda root: _mutate_raw(
                root, d2b, lambda raw: raw["steps"][4]["lower"].add_(1.0)
            ),
        ),
        (
            "gradient",
            lambda root: _mutate_raw(
                root, d2b, lambda raw: raw["steps"][4]["gradient"].add_(1.0)
            ),
        ),
        (
            "alpha-lineage",
            lambda root: _mutate_raw(
                root, d2b, lambda raw: raw["steps"][4]["alpha_after"].add_(1.0)
            ),
        ),
        (
            "adam-moment",
            lambda root: _mutate_raw(
                root,
                d2b,
                lambda raw: raw["steps"][4]["optimizer_exp_avg"].add_(1.0),
            ),
        ),
        (
            "receipt-claim",
            lambda root: _mutate_raw(
                root,
                d2b,
                lambda raw: raw["steps"][4]["d2b_receipt"].__setitem__(
                    "performance_claimed", True
                ),
            ),
        ),
        (
            "receipt-pointer",
            lambda root: _mutate_raw(
                root,
                d2b,
                lambda raw: raw["steps"][4]["d2b_receipt"].__setitem__(
                    "scratch_region_pointers", (1, 1)
                ),
            ),
        ),
        (
            "control-receipt",
            lambda root: _mutate_raw(
                root,
                d1c,
                lambda raw: raw["steps"][4].__setitem__(
                    "d2b_receipt", raw["steps"][4]["d1c_receipt"]
                ),
            ),
        ),
        (
            "protocol-tolerance",
            lambda root: _mutate_protocol(
                root,
                "state_tolerance",
                {"atol": 1.0, "rtol": 1.0},
            ),
        ),
        (
            "protocol-order",
            lambda root: _mutate_protocol(root, "order", [["d2b", "d1c"]] * 5),
        ),
        (
            "summary-correctness",
            lambda root: _mutate_summary(
                root, "trajectory_correctness_admitted", False
            ),
        ),
        (
            "summary-open",
            lambda root: _mutate_summary(root, "d2b_timing_open", False),
        ),
        (
            "summary-claim",
            lambda root: _mutate_summary(root, "performance_claimed", True),
        ),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="boundflow-r3d2b-tamper-") as temporary:
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
            raise RuntimeError(f"R3-D2B tamper accepted: {name}")
    report: dict[str, object] = {
        "schema_version": "boundflow.r3-d2b-correctness-tamper/v1",
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
    print(f"R3-D2B tamper PASS: {report['rejected_count']}/{report['case_count']}")


if __name__ == "__main__":
    main()
