#!/usr/bin/env python3
"""Probe synchronized outer-rehash attacks against an FSG3 artifact."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.fsg3_same_solver_timing import canonical_hash
from scripts import run_fsg3_same_solver_experiment as artifact_runner

REPORT_SCHEMA_VERSION = "boundflow.fsg3-same-solver-tamper-report/v1"


def _resign_manifest(artifact: Path) -> None:
    manifest = artifact_runner._load_json(artifact / "manifest.json")
    files = cast(Mapping[str, str], manifest["files"])
    manifest["files"] = {
        name: artifact_runner._file_sha256(artifact / name) for name in files
    }
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = canonical_hash(semantic)
    artifact_runner._write_json(artifact / "manifest.json", manifest)


def _worker_rows(artifact: Path) -> list[dict[str, Any]]:
    return artifact_runner._load_jsonl(artifact / "worker_runs.jsonl")


def _write_worker_rows(artifact: Path, rows: list[dict[str, Any]]) -> None:
    artifact_runner._write_jsonl(artifact / "worker_runs.jsonl", rows)


def _mutate_control_latency(artifact: Path) -> None:
    rows = _worker_rows(artifact)
    metrics = cast(dict[str, Any], rows[0]["metrics"])
    metrics["query_wall_ns"] = int(metrics["query_wall_ns"]) + 10_000_000
    metrics["cold_total_ns"] = int(metrics["cold_total_ns"]) + 10_000_000
    _write_worker_rows(artifact, rows)


def _delete_run(artifact: Path) -> None:
    rows = _worker_rows(artifact)
    _write_worker_rows(artifact, rows[:-1])


def _swap_configuration_mode_order(artifact: Path) -> None:
    rows = _worker_rows(artifact)
    rows[0], rows[1] = rows[1], rows[0]
    _write_worker_rows(artifact, rows)


def _mutate_b1_provider_count(artifact: Path) -> None:
    rows = _worker_rows(artifact)
    row = next(item for item in rows if item["configuration"] == "B1")
    execution = cast(dict[str, Any], row["execution"])
    execution["provider_compute_bounds_call_count"] = (
        int(execution["provider_compute_bounds_call_count"]) + 1
    )
    _write_worker_rows(artifact, rows)


def _mutate_b2_fallback_count(artifact: Path) -> None:
    rows = _worker_rows(artifact)
    row = next(item for item in rows if item["configuration"] == "B2")
    cast(dict[str, Any], row["execution"])["fallback_dispatch_count"] = 1
    _write_worker_rows(artifact, rows)


def _mutate_semantic_tensor(artifact: Path) -> None:
    rows = _worker_rows(artifact)
    semantics = cast(dict[str, Any], rows[0]["semantics"])
    lower = list(cast(list[float], semantics["lower_values"]))
    lower[0] += 0.25
    semantics["lower_values"] = lower
    _write_worker_rows(artifact, rows)


def _mutate_temperature_gate(artifact: Path) -> None:
    rows = artifact_runner._load_jsonl(artifact / "run_metadata.jsonl")
    preflight = cast(dict[str, Any], rows[0]["formal_preflight"])
    sample = cast(list[dict[str, Any]], preflight["samples"])[-1]
    sample["temperature_celsius"] = artifact_runner.PREFLIGHT_TEMPERATURE_LIMIT_C + 1
    artifact_runner._write_jsonl(artifact / "run_metadata.jsonl", rows)


def _mutate_summary_ratio(artifact: Path) -> None:
    summary = artifact_runner._load_json(artifact / "summary.json")
    speedups = cast(dict[str, Any], summary["speedups_b0_over_candidate"])
    query = cast(dict[str, Any], cast(dict[str, Any], speedups["B2"])["query_wall_ns"])
    query["geometric_mean"] = 9.0
    summary.pop("summary_hash", None)
    summary["summary_hash"] = canonical_hash(summary)
    artifact_runner._write_json(artifact / "summary.json", summary)
    manifest = artifact_runner._load_json(artifact / "manifest.json")
    manifest["summary_hash"] = summary["summary_hash"]
    artifact_runner._write_json(artifact / "manifest.json", manifest)


ATTACKS: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("control-latency-outer-resign", _mutate_control_latency),
    ("delete-run-outer-resign", _delete_run),
    ("configuration-mode-order-outer-resign", _swap_configuration_mode_order),
    ("b1-provider-count-outer-resign", _mutate_b1_provider_count),
    ("b2-fallback-count-outer-resign", _mutate_b2_fallback_count),
    ("semantic-tensor-outer-resign", _mutate_semantic_tensor),
    ("temperature-gate-outer-resign", _mutate_temperature_gate),
    ("summary-ratio-outer-resign", _mutate_summary_ratio),
)


def run_probe_suite(*, artifact: Path) -> dict[str, object]:
    """Run the preregistered attacks after independently replaying the clean input."""

    _runs, clean_summary, _result = artifact_runner._verify_static_artifact(artifact)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-fsg3-tamper-") as raw:
        workspace = Path(raw)
        for name, mutate in ATTACKS:
            probe = workspace / name
            shutil.copytree(artifact, probe)
            mutate(probe)
            _resign_manifest(probe)
            rejection = ""
            try:
                artifact_runner._verify_static_artifact(probe)
            except (TypeError, ValueError) as error:
                rejection = str(error)
            else:
                raise AssertionError(f"tampered FSG3 artifact admitted: {name}")
            rows.append(
                {
                    "name": name,
                    "payload_modified": True,
                    "manifest_file_digest_resigned": True,
                    "manifest_hash_resigned": True,
                    "rejected": True,
                    "rejection": rejection,
                }
            )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_source_git_head": artifact_runner._load_json(
            artifact / "manifest.json"
        )["source_git_head"],
        "clean_summary_hash": clean_summary["summary_hash"],
        "attack_count": len(rows),
        "outer_resigned_attack_count": len(rows),
        "all_rejected": all(row["rejected"] is True for row in rows),
        "attacks": rows,
        "performance_claimed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the tamper suite and write its canonical report."""

    args = _parse_args()
    report = run_probe_suite(artifact=args.artifact_dir.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
