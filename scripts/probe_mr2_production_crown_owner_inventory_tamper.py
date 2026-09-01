#!/usr/bin/env python3
"""Probe fully re-signed MR2 owner inventory artifact tampering."""

# pylint: disable=duplicate-code,missing-function-docstring

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

from boundflow.runtime.mr2_production_crown_owner_inventory import canonical_hash
from scripts.run_mr2_production_crown_owner_inventory_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "artifacts/measurement-recovery/mr2-production-crown-subgraph-owner-inventory-v1"
)


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
        raise TypeError("MR2 tamper JSON differs")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")


def _mutate_source(
    root: Path, name: str, mutation: Callable[[dict[str, Any]], None]
) -> None:
    path = root / "source" / name
    value = _json(path)
    mutation(value)
    _write(path, value)
    protocol_path = root / "protocol.json"
    protocol = _json(protocol_path)
    protocol["input_sha256"][name] = _file_hash(path)
    protocol.pop("protocol_hash", None)
    protocol["protocol_hash"] = canonical_hash(protocol)
    _write(protocol_path, protocol)
    manifest = _json(root / "manifest.json")
    manifest["protocol_hash"] = protocol["protocol_hash"]
    _write(root / "manifest.json", manifest)


def _mutate_ledger(
    root: Path, mutation: Callable[[list[dict[str, Any]]], None]
) -> None:
    path = root / "site_ledger.jsonl"
    rows = _jsonl(path)
    mutation(rows)
    for row in rows:
        row.pop("site_hash", None)
        row["site_hash"] = canonical_hash(row)
    _write_jsonl(path, rows)


def _mutate_derived(
    root: Path, name: str, hash_field: str, mutation: Callable[[dict[str, Any]], None]
) -> None:
    path = root / name
    value = _json(path)
    mutation(value)
    value.pop(hash_field, None)
    value[hash_field] = canonical_hash(value)
    _write(path, value)
    manifest = _json(root / "manifest.json")
    manifest[hash_field] = value[hash_field]
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
            "p-start-node",
            lambda root: _mutate_source(
                root,
                "p_bundle.json",
                lambda value: value["template"].__setitem__(
                    "start_node_id", "31/Gemm_14"
                ),
            ),
        ),
        (
            "p-beta-shape",
            lambda root: _mutate_source(
                root,
                "p_bundle.json",
                lambda value: next(
                    item
                    for item in value["instance"]["bindings"]
                    if item["name"] == "beta"
                ).__setitem__("shape", [6, 1]),
            ),
        ),
        (
            "p-production-connected",
            lambda root: _mutate_source(
                root,
                "p_bundle.json",
                lambda value: value["receipt"].__setitem__(
                    "production_connected", True
                ),
            ),
        ),
        (
            "p-trajectory",
            lambda root: _mutate_source(
                root,
                "p_trajectory_summary.json",
                lambda value: value.__setitem__(
                    "trajectory_correctness_admitted", False
                ),
            ),
        ),
        (
            "p-correctness",
            lambda root: _mutate_source(
                root,
                "p_cibc_summary.json",
                lambda value: value.__setitem__("maximum_absolute_difference", 1.0),
            ),
        ),
        (
            "s-active-beta",
            lambda root: _mutate_source(
                root,
                "s_correctness_summary.json",
                lambda value: value.__setitem__("beta_nonzero_count", 0),
            ),
        ),
        (
            "mr1-eligibility",
            lambda root: _mutate_source(
                root,
                "mr1_summary.json",
                lambda value: value.__setitem__("eligible_target_model_call_count", 1),
            ),
        ),
        (
            "ledger-site",
            lambda root: _mutate_ledger(
                root, lambda rows: rows[0].__setitem__("site_id", "P:wrong")
            ),
        ),
        (
            "ledger-gate",
            lambda root: _mutate_ledger(
                root,
                lambda rows: rows[0]["gates"][
                    "production_exact_call_connection"
                ].__setitem__("status", "proven"),
            ),
        ),
        (
            "ledger-ready",
            lambda root: _mutate_ledger(
                root,
                lambda rows: rows[1].__setitem__("ready_for_bridge_correctness", True),
            ),
        ),
        (
            "matrix",
            lambda root: _mutate_derived(
                root,
                "gap_matrix.json",
                "matrix_hash",
                lambda value: value["P:25/Conv_8"].__setitem__(
                    "production_exact_call_connection", "proven"
                ),
            ),
        ),
        (
            "summary-route",
            lambda root: _mutate_derived(
                root,
                "summary.json",
                "summary_hash",
                lambda value: value.__setitem__("timing_open", True),
            ),
        ),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="boundflow-mr2-tamper-") as tmp:
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
            raise RuntimeError(f"MR2 tamper accepted: {name}")
    report: dict[str, object] = {
        "schema_version": "boundflow.mr2-production-crown-owner-inventory-tamper/v1",
        "case_count": len(rows),
        "rejected_count": len(rows),
        "cases": rows,
        "performance_claimed": False,
    }
    report["report_hash"] = canonical_hash(report)
    return report


def main() -> None:
    report = run(ARTIFACT)
    path = ARTIFACT / "tamper_results.json"
    _write(path, report)
    manifest_path = ARTIFACT / "manifest.json"
    manifest = _json(manifest_path)
    manifest["files"]["tamper_results.json"] = _file_hash(path)
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write(manifest_path, manifest)
    replay(ARTIFACT)
    print(f"MR2 tamper PASS: {report['rejected_count']}/{report['case_count']}")


if __name__ == "__main__":
    main()
