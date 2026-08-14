#!/usr/bin/env python3
"""Probe outer-resigned attacks against an FSG4 B3 timing artifact."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.fsg3_same_solver_timing import canonical_hash
from scripts import run_fsg4_b3_same_solver_experiment as artifact_runner

REPORT_SCHEMA = "boundflow.fsg4-b3-same-solver-tamper-report/v1"


def _resign_manifest(artifact: Path) -> None:
    manifest = artifact_runner._load_json(artifact / "manifest.json")
    files = sorted(
        str(path.relative_to(artifact))
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
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


def _envelope(artifact: Path, index: int) -> dict[str, Any]:
    return artifact_runner._load_json(artifact / f"workers/run_{index:02d}.json")


def _write_envelope(artifact: Path, index: int, value: dict[str, Any]) -> None:
    artifact_runner._write_json(artifact / f"workers/run_{index:02d}.json", value)


def _first_index(artifact: Path, *, configuration: str, mode: str | None = None) -> int:
    rows = _worker_rows(artifact)
    return next(
        index
        for index, row in enumerate(rows)
        if row["configuration"] == configuration
        and (mode is None or row["mode"] == mode)
    )


def _sync_run_mutation(
    artifact: Path,
    *,
    index: int,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    envelope = _envelope(artifact, index)
    run = cast(dict[str, Any], envelope["run"])
    mutate(run)
    _write_envelope(artifact, index, envelope)
    rows = _worker_rows(artifact)
    rows[index] = run
    _write_worker_rows(artifact, rows)


def _mutate_control_latency(artifact: Path) -> None:
    index = _first_index(artifact, configuration="B3", mode="control")

    def mutate(run: dict[str, Any]) -> None:
        metrics = cast(dict[str, Any], run["metrics"])
        metrics["query_wall_ns"] = int(metrics["query_wall_ns"]) + 10_000_000
        metrics["cold_total_ns"] = int(metrics["cold_total_ns"]) + 10_000_000

    _sync_run_mutation(artifact, index=index, mutate=mutate)


def _delete_worker(artifact: Path) -> None:
    (artifact / "workers/run_35.json").unlink()


def _swap_aggregate_order(artifact: Path) -> None:
    rows = _worker_rows(artifact)
    rows[0], rows[1] = rows[1], rows[0]
    _write_worker_rows(artifact, rows)


def _mutate_b3_activation(artifact: Path) -> None:
    index = _first_index(artifact, configuration="B3", mode="control")

    def mutate(run: dict[str, Any]) -> None:
        activation = cast(dict[str, Any], run["activation"])
        activation["prepared_core_template_count"] = 0

    _sync_run_mutation(artifact, index=index, mutate=mutate)


def _mutate_b3_profile_counter(artifact: Path) -> None:
    index = _first_index(artifact, configuration="B3", mode="profile")

    def mutate(run: dict[str, Any]) -> None:
        activation = cast(dict[str, Any], run["activation"])
        counts = cast(dict[str, Any], activation["detailed_counts"])
        counts["forward_trace_build_count"] = 5

    _sync_run_mutation(artifact, index=index, mutate=mutate)


def _mutate_b3_fallback(artifact: Path) -> None:
    index = _first_index(artifact, configuration="B3", mode="control")

    def mutate(run: dict[str, Any]) -> None:
        execution = cast(dict[str, Any], run["execution"])
        execution["fallback_dispatch_count"] = 1

    _sync_run_mutation(artifact, index=index, mutate=mutate)


def _mutate_b3_semantic(artifact: Path) -> None:
    index = _first_index(artifact, configuration="B3", mode="control")

    def mutate(run: dict[str, Any]) -> None:
        semantics = cast(dict[str, Any], run["semantics"])
        lower = list(cast(list[float], semantics["lower_values"]))
        lower[0] += 0.25
        semantics["lower_values"] = lower

    _sync_run_mutation(artifact, index=index, mutate=mutate)


def _mutate_formal_preflight(artifact: Path) -> None:
    metadata_path = artifact / "metadata/run_00.json"
    metadata = artifact_runner._load_json(metadata_path)
    preflight = cast(dict[str, Any], metadata["formal_preflight"])
    sample = cast(list[dict[str, Any]], preflight["samples"])[-1]
    sample["temperature_celsius"] = (
        artifact_runner.base_experiment.PREFLIGHT_TEMPERATURE_LIMIT_C + 1
    )
    artifact_runner._write_json(metadata_path, metadata)
    rows = artifact_runner._load_jsonl(artifact / "run_metadata.jsonl")
    rows[0] = metadata
    artifact_runner._write_jsonl(artifact / "run_metadata.jsonl", rows)


def _mutate_protocol_sequence(artifact: Path) -> None:
    protocol = artifact_runner._load_json(artifact / "protocol.json")
    sequence = cast(list[list[Any]], protocol["expected_sequence"])
    sequence[0][2] = "B3"
    payload = dict(protocol)
    payload.pop("protocol_hash", None)
    protocol["protocol_hash"] = canonical_hash(payload)
    artifact_runner._write_json(artifact / "protocol.json", protocol)
    manifest = artifact_runner._load_json(artifact / "manifest.json")
    manifest["protocol_hash"] = protocol["protocol_hash"]
    artifact_runner._write_json(artifact / "manifest.json", manifest)


def _mutate_summary_ratio(artifact: Path) -> None:
    summary = artifact_runner._load_json(artifact / "summary.json")
    ratios = cast(dict[str, Any], summary["speedups_b2_over_b3"])
    core = cast(dict[str, Any], ratios["core_wall_ns"])
    core["geometric_mean"] = 9.0
    summary.pop("summary_hash", None)
    summary["summary_hash"] = canonical_hash(summary)
    artifact_runner._write_json(artifact / "summary.json", summary)
    manifest = artifact_runner._load_json(artifact / "manifest.json")
    manifest["summary_hash"] = summary["summary_hash"]
    artifact_runner._write_json(artifact / "manifest.json", manifest)


ATTACKS: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("control-latency-outer-resign", _mutate_control_latency),
    ("delete-worker-outer-resign", _delete_worker),
    ("aggregate-order-outer-resign", _swap_aggregate_order),
    ("b3-activation-outer-resign", _mutate_b3_activation),
    ("b3-profile-counter-outer-resign", _mutate_b3_profile_counter),
    ("b3-fallback-outer-resign", _mutate_b3_fallback),
    ("b3-semantic-outer-resign", _mutate_b3_semantic),
    ("formal-preflight-outer-resign", _mutate_formal_preflight),
    ("protocol-sequence-outer-resign", _mutate_protocol_sequence),
    ("summary-ratio-outer-resign", _mutate_summary_ratio),
)


def run_probe_suite(*, artifact: Path) -> dict[str, object]:
    """Replay the clean input, then require every outer-resigned attack to fail."""

    _runs, clean_summary, _result = artifact_runner._verify_static_artifact(artifact)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-fsg4-b3-tamper-") as raw:
        workspace = Path(raw)
        for name, mutate in ATTACKS:
            probe = workspace / name
            shutil.copytree(artifact, probe)
            mutate(probe)
            _resign_manifest(probe)
            rejection = ""
            try:
                artifact_runner._verify_static_artifact(probe)
            except (FileNotFoundError, TypeError, ValueError) as error:
                rejection = str(error)
            else:
                raise AssertionError(f"tampered FSG4/B3 artifact admitted: {name}")
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
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
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
    report["report_hash"] = canonical_hash(report)
    return report


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
