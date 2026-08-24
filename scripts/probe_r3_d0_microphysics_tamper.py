#!/usr/bin/env python3
"""Probe fully re-signed R3-D0 artifact tampering."""

# pylint: disable=too-many-locals,missing-function-docstring

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

import torch

from scripts.run_r3_d0_microphysics_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-d0-microphysics-v1"
REPORT_SCHEMA = "boundflow.r3-d0-microphysics-tamper-report/v1"


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
        raise TypeError("R3-D0 tamper JSON differs")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _resign_manifest(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("R3-D0 tamper manifest files differ")
    manifest["files"] = {
        path: _file_hash(root / path) for path in sorted(str(item) for item in files)
    }
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = _hash(manifest)
    _write_json(manifest_path, manifest)


def _resign_summary(root: Path) -> None:
    path = root / "summary.json"
    value = _json(path)
    value.pop("summary_hash", None)
    value["summary_hash"] = _hash(value)
    _write_json(path, value)


def _resign_protocol(root: Path) -> None:
    path = root / "protocol.json"
    value = _json(path)
    value.pop("protocol_hash", None)
    value["protocol_hash"] = _hash(value)
    _write_json(path, value)


def _candidate_raw(root: Path) -> Path:
    return sorted((root / "raw").glob("*candidate.pt"))[0]


def _mutate_raw(root: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    path = _candidate_raw(root)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    mutation(raw)
    torch.save(raw, path)


def _event_mutation(
    field: str, replacement: Callable[[Any], Any]
) -> Callable[[dict[str, Any]], None]:
    def mutate(raw: dict[str, Any]) -> None:
        events = raw["events"]
        event = next(row for row in events if row["kind"] == "cuda_kernel")
        event[field] = replacement(event[field])
        if field in {"start_ns", "end_ns"}:
            event["duration_ns"] = event["end_ns"] - event["start_ns"]
        raw["event_hash"] = _hash(events)

    return mutate


def run(artifact: Path) -> dict[str, object]:
    cases: tuple[tuple[str, Callable[[Path], None]], ...] = (
        (
            "kernel-duration",
            lambda root: _mutate_raw(
                root, _event_mutation("end_ns", lambda value: value + 1)
            ),
        ),
        (
            "correlation-id",
            lambda root: _mutate_raw(
                root, _event_mutation("correlation_id", lambda value: value + 1)
            ),
        ),
        (
            "marker-ordinal",
            lambda root: _mutate_raw(
                root,
                _event_mutation(
                    "marker_ordinal", lambda value: 0 if value is None else value + 1
                ),
            ),
        ),
        (
            "fallback-method",
            lambda root: _mutate_raw(
                root,
                _event_mutation(
                    "attribution_method", lambda _value: "marker_containment"
                ),
            ),
        ),
        (
            "region-family",
            lambda root: _mutate_raw(
                root, _event_mutation("family", lambda value: value + "-tampered")
            ),
        ),
        (
            "host-calibration",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw.__setitem__(
                    "profiled_host_wall_ns", raw["profiled_host_wall_ns"] + 1
                ),
            ),
        ),
        (
            "launch-count",
            lambda root: _mutate_raw(
                root,
                lambda raw: raw["ledger"].__setitem__(
                    "compiled_launch_marker_count",
                    raw["ledger"]["compiled_launch_marker_count"] + 1,
                ),
            ),
        ),
        (
            "target-budget",
            lambda root: _mutate_summary(root, "target_candidate_ns", 1.0),
        ),
        (
            "required-speedup",
            lambda root: _mutate_summary(root, "required_speedup", 1.0),
        ),
        ("route", lambda root: _mutate_summary(root, "route", "graph-opportunity")),
        (
            "summary-verdict",
            lambda root: _mutate_top_summary(root, "verdict", "VALIDATED-SPEEDUP"),
        ),
        ("performance-claim", _mutate_protocol_claim),
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="boundflow-r3d0-tamper-") as temporary:
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
                raise RuntimeError(f"R3-D0 tamper was accepted: {name}")
            results.append({"case": name, "rejected": True, "error": error})
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "case_count": len(results),
        "rejected_count": sum(1 for row in results if row["rejected"] is True),
        "cases": results,
        "performance_claimed": False,
    }
    report["report_hash"] = _hash(report)
    return report


def _mutate_summary(root: Path, field: str, value: object) -> None:
    path = root / "summary.json"
    summary = _json(path)
    summary["pair_metrics"][0]["route"]["compiled_region_route"][field] = value
    _write_json(path, summary)
    _resign_summary(root)


def _mutate_top_summary(root: Path, field: str, value: object) -> None:
    path = root / "summary.json"
    summary = _json(path)
    summary[field] = value
    _write_json(path, summary)
    _resign_summary(root)


def _mutate_protocol_claim(root: Path) -> None:
    path = root / "protocol.json"
    protocol = _json(path)
    protocol["performance_claimed"] = True
    _write_json(path, protocol)
    _resign_protocol(root)


def main() -> None:
    report = run(DEFAULT_ARTIFACT)
    path = DEFAULT_ARTIFACT / "tamper_report.json"
    _write_json(path, report)
    manifest = _json(DEFAULT_ARTIFACT / "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("R3-D0 formal manifest files differ")
    files["tamper_report.json"] = _file_hash(path)
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = _hash(manifest)
    _write_json(DEFAULT_ARTIFACT / "manifest.json", manifest)
    replay(DEFAULT_ARTIFACT)
    print(
        f"R3-D0 tamper PASS: rejected={report['rejected_count']}/{report['case_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
