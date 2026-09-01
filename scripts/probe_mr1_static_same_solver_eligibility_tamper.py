#!/usr/bin/env python3
"""Probe fully re-signed MR1 static eligibility artifact tampering."""

# pylint: disable=duplicate-code,missing-function-docstring

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

from boundflow.runtime.mr1_static_same_solver_eligibility import canonical_hash
from scripts.run_mr1_static_same_solver_eligibility_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/measurement-recovery/mr1-static-same-solver-eligibility-v1"


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
        raise TypeError("MR1 tamper JSON differs")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")


def _mutate_source(
    root: Path, mutation: Callable[[list[dict[str, Any]]], None]
) -> None:
    path = root / "source/activation_calls.jsonl"
    rows = _jsonl(path)
    mutation(rows)
    _write_jsonl(path, rows)
    protocol_path = root / "protocol.json"
    protocol = _json(protocol_path)
    protocol["input_sha256"]["activation_calls.jsonl"] = _file_hash(path)
    protocol.pop("protocol_hash", None)
    protocol["protocol_hash"] = canonical_hash(protocol)
    _write(protocol_path, protocol)
    manifest = _json(root / "manifest.json")
    manifest["protocol_hash"] = protocol["protocol_hash"]
    _write(root / "manifest.json", manifest)


def _mutate_ledger(
    root: Path, mutation: Callable[[list[dict[str, Any]]], None]
) -> None:
    path = root / "ledger.jsonl"
    rows = _jsonl(path)
    mutation(rows)
    for row in rows:
        row.pop("ledger_hash", None)
        row["ledger_hash"] = canonical_hash(row)
    _write_jsonl(path, rows)


def _mutate_json(
    root: Path, name: str, mutation: Callable[[dict[str, Any]], None]
) -> None:
    path = root / name
    value = _json(path)
    mutation(value)
    hash_field = "summary_hash" if name == "summary.json" else "coverage_hash"
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


def _target(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next(
        row for row in rows if row["source_workload"] == "vnncomp21-resnet2b-prop0"
    )


def _delete_last(rows: list[dict[str, Any]]) -> None:
    rows.pop()


def _duplicate_last(rows: list[dict[str, Any]]) -> None:
    rows.append(rows[-1])


def run(artifact: Path) -> dict[str, object]:
    cases: tuple[tuple[str, Callable[[Path], None]], ...] = (
        ("delete-call", lambda root: _mutate_source(root, _delete_last)),
        (
            "duplicate-call",
            lambda root: _mutate_source(root, _duplicate_last),
        ),
        (
            "model-hash",
            lambda root: _mutate_source(
                root,
                lambda rows: _target(rows)["query"].__setitem__(
                    "model_structure_hash", "onnx:" + "0" * 64
                ),
            ),
        ),
        (
            "phase",
            lambda root: _mutate_source(
                root,
                lambda rows: _target(rows)["query"]["execution_options"].__setitem__(
                    "solver_phase", "initial_ibp"
                ),
            ),
        ),
        (
            "method",
            lambda root: _mutate_source(
                root,
                lambda rows: _target(rows)["query"].__setitem__("bound_method", "IBP"),
            ),
        ),
        (
            "grad",
            lambda root: _mutate_source(
                root,
                lambda rows: _target(rows)["query"].__setitem__("requires_grad", False),
            ),
        ),
        (
            "split",
            lambda root: _mutate_source(
                root,
                lambda rows: _target(rows)["query"]["execution_options"].__setitem__(
                    "split_state_present", False
                ),
            ),
        ),
        (
            "semantics-owner",
            lambda root: _mutate_source(
                root,
                lambda rows: _target(rows).__setitem__(
                    "semantics_owner", "boundflow_cibc_full_graph"
                ),
            ),
        ),
        (
            "ledger-eligibility",
            lambda root: _mutate_ledger(
                root, lambda rows: rows[0].__setitem__("eligible", True)
            ),
        ),
        (
            "ledger-reason",
            lambda root: _mutate_ledger(
                root,
                lambda rows: rows[0].__setitem__(
                    "primary_rejection_reason",
                    "compile_key_or_topology_receipt_missing",
                ),
            ),
        ),
        (
            "coverage-count",
            lambda root: _mutate_json(
                root,
                "coverage.json",
                lambda value: value.__setitem__("activation_call_count", 393),
            ),
        ),
        (
            "summary-count",
            lambda root: _mutate_json(
                root,
                "summary.json",
                lambda value: value.__setitem__("eligible_target_model_call_count", 1),
            ),
        ),
        (
            "summary-route",
            lambda root: _mutate_json(
                root,
                "summary.json",
                lambda value: value.__setitem__(
                    "direct_end_to_end_ab_preregistration_open", True
                ),
            ),
        ),
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="boundflow-mr1-tamper-") as tmp:
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
            raise RuntimeError(f"MR1 tamper accepted: {name}")
    report: dict[str, object] = {
        "schema_version": "boundflow.mr1-static-same-solver-eligibility-tamper/v1",
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
    print(f"MR1 tamper PASS: {report['rejected_count']}/{report['case_count']}")


if __name__ == "__main__":
    main()
