#!/usr/bin/env python3
"""Probe outer-resigned attacks against an FSG4/B4-A formal artifact."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=missing-function-docstring

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.fsg3_same_solver_timing import canonical_hash
from scripts import run_fsg4_b4a_formal_timing as artifact_runner

REPORT_SCHEMA = "boundflow.fsg4-b4a-formal-timing-tamper-report/v1"


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


def _envelope(artifact: Path, index: int) -> dict[str, Any]:
    return artifact_runner._load_json(artifact / f"workers/run_{index:02d}.json")


def _write_envelope(artifact: Path, index: int, value: dict[str, Any]) -> None:
    artifact_runner._write_json(artifact / f"workers/run_{index:02d}.json", value)


def _first_index(artifact: Path, *, configuration: str, mode: str) -> int:
    for index in range(24):
        envelope = _envelope(artifact, index)
        if (
            envelope.get("configuration") == configuration
            and envelope.get("mode") == mode
        ):
            return index
    raise ValueError("FSG4/B4-A tamper target differs")


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
    rows = artifact_runner._load_jsonl(artifact / "worker_runs.jsonl")
    rows[index] = run
    artifact_runner._write_jsonl(artifact / "worker_runs.jsonl", rows)


def _mutate_control_latency(artifact: Path) -> None:
    index = _first_index(artifact, configuration="B3", mode="control")

    def mutate(run: dict[str, Any]) -> None:
        metrics = cast(dict[str, Any], run["metrics"])
        metrics["query_wall_ns"] = int(metrics["query_wall_ns"]) + 10_000_000
        metrics["cold_total_ns"] = int(metrics["cold_total_ns"]) + 10_000_000

    _sync_run_mutation(artifact, index=index, mutate=mutate)


def _delete_worker(artifact: Path) -> None:
    (artifact / "workers/run_23.json").unlink()


def _swap_aggregate_order(artifact: Path) -> None:
    rows = artifact_runner._load_jsonl(artifact / "worker_runs.jsonl")
    rows[0], rows[1] = rows[1], rows[0]
    artifact_runner._write_jsonl(artifact / "worker_runs.jsonl", rows)


def _mutate_candidate_activation(artifact: Path) -> None:
    index = _first_index(artifact, configuration="B4-A", mode="control")
    envelope = _envelope(artifact, index)
    activation = cast(dict[str, Any], envelope["activation"])
    activation["terminal_lower_adjoint_handoff_count"] = 0
    _write_envelope(artifact, index, envelope)


def _mutate_candidate_profile_counter(artifact: Path) -> None:
    index = _first_index(artifact, configuration="B4-A", mode="profile")
    envelope = _envelope(artifact, index)
    activation = cast(dict[str, Any], envelope["activation"])
    counts = cast(dict[str, Any], activation["profile_counter_counts"])
    counts["optimizer_evaluation_count"] = 9
    activation["profile_counter_counts_hash"] = canonical_hash(
        dict(sorted(counts.items()))
    )
    activation.pop("activation_hash", None)
    activation["activation_hash"] = canonical_hash(activation)
    _write_envelope(artifact, index, envelope)


def _mutate_export_payload(artifact: Path) -> None:
    index = _first_index(artifact, configuration="B4-A", mode="control")
    envelope = _envelope(artifact, index)
    diagnostics = cast(dict[str, Any], envelope["diagnostics"])
    payload = cast(
        list[dict[str, Any]], diagnostics["native_backward_export_payloads"]
    )[0]
    lower = cast(dict[str, Any], payload["lower"])
    raw = base64.b64decode(str(lower["content_base64"]))
    lower["content_base64"] = base64.b64encode(bytes(len(raw))).decode("ascii")
    _write_envelope(artifact, index, envelope)


def _mutate_runtime_environment(artifact: Path) -> None:
    index = _first_index(artifact, configuration="B3", mode="control")
    envelope = _envelope(artifact, index)
    diagnostics = cast(dict[str, Any], envelope["diagnostics"])
    runtime = cast(dict[str, Any], diagnostics["runtime_environment"])
    runtime["torch_version"] = "tampered"
    _write_envelope(artifact, index, envelope)


def _mutate_environment_counter_delta(artifact: Path) -> None:
    index = _first_index(artifact, configuration="B3", mode="control")
    envelope = _envelope(artifact, index)
    diagnostics = cast(dict[str, Any], envelope["diagnostics"])
    after = cast(dict[str, Any], diagnostics["environment_after"])
    after["sw_thermal_slowdown_counter_us"] = (
        int(after["sw_thermal_slowdown_counter_us"]) + 1
    )
    _write_envelope(artifact, index, envelope)


def _mutate_worker_protocol(artifact: Path) -> None:
    index = _first_index(artifact, configuration="B4-A", mode="profile")
    envelope = _envelope(artifact, index)
    protocol = cast(dict[str, Any], envelope["protocol"])
    protocol["source_git_head"] = "0" * 40
    payload = dict(protocol)
    payload.pop("protocol_hash", None)
    protocol["protocol_hash"] = canonical_hash(payload)
    _write_envelope(artifact, index, envelope)


def _mutate_formal_preflight(artifact: Path) -> None:
    metadata_path = artifact / "metadata/run_00.json"
    metadata = artifact_runner._load_json(metadata_path)
    preflight = cast(dict[str, Any], metadata["formal_preflight"])
    sample = cast(list[dict[str, Any]], preflight["samples"])[-1]
    sample["temperature_celsius"] = (
        artifact_runner.B4A_PREFLIGHT_TEMPERATURE_LIMIT_C + 1
    )
    artifact_runner._write_json(metadata_path, metadata)
    rows = artifact_runner._load_jsonl(artifact / "run_metadata.jsonl")
    rows[0] = metadata
    artifact_runner._write_jsonl(artifact / "run_metadata.jsonl", rows)


def _mutate_power_policy(artifact: Path) -> None:
    metadata_path = artifact / "metadata/run_00.json"
    metadata = artifact_runner._load_json(metadata_path)
    preflight = cast(dict[str, Any], metadata["formal_preflight"])
    preflight["nvidia_powerd_state"] = "active"
    artifact_runner._write_json(metadata_path, metadata)
    rows = artifact_runner._load_jsonl(artifact / "run_metadata.jsonl")
    rows[0] = metadata
    artifact_runner._write_jsonl(artifact / "run_metadata.jsonl", rows)


def _mutate_protocol_sequence(artifact: Path) -> None:
    protocol = artifact_runner._load_json(artifact / "protocol.json")
    sequence = cast(list[dict[str, Any]], protocol["sequence"])
    positions = cast(list[dict[str, str]], sequence[0]["positions"])
    positions[0]["configuration"] = "B4-A"
    payload = dict(protocol)
    payload.pop("protocol_hash", None)
    protocol["protocol_hash"] = canonical_hash(payload)
    artifact_runner._write_json(artifact / "protocol.json", protocol)
    manifest = artifact_runner._load_json(artifact / "manifest.json")
    manifest["protocol_hash"] = protocol["protocol_hash"]
    artifact_runner._write_json(artifact / "manifest.json", manifest)


def _mutate_paired_ratio(artifact: Path) -> None:
    rows = artifact_runner._load_jsonl(artifact / "paired_runs.jsonl")
    ratios = cast(dict[str, Any], rows[0]["ratios"])
    ratios["core_wall_ns"] = 9.0
    rows[0].pop("pair_hash", None)
    rows[0]["pair_hash"] = canonical_hash(rows[0])
    artifact_runner._write_jsonl(artifact / "paired_runs.jsonl", rows)


def _mutate_summary(artifact: Path) -> None:
    summary = artifact_runner._load_json(artifact / "summary.json")
    geomeans = cast(dict[str, Any], summary["metric_geomeans"])
    geomeans["core_wall_ns"] = 9.0
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
    ("candidate-activation-outer-resign", _mutate_candidate_activation),
    ("candidate-profile-counter-outer-resign", _mutate_candidate_profile_counter),
    ("export-payload-outer-resign", _mutate_export_payload),
    ("runtime-environment-outer-resign", _mutate_runtime_environment),
    ("environment-counter-delta-outer-resign", _mutate_environment_counter_delta),
    ("worker-protocol-outer-resign", _mutate_worker_protocol),
    ("formal-preflight-outer-resign", _mutate_formal_preflight),
    ("power-policy-outer-resign", _mutate_power_policy),
    ("protocol-sequence-outer-resign", _mutate_protocol_sequence),
    ("paired-ratio-outer-resign", _mutate_paired_ratio),
    ("summary-outer-resign", _mutate_summary),
)


def run_probe_suite(*, artifact: Path) -> dict[str, object]:
    """Replay the clean input, then require every outer-resigned attack to fail."""

    clean_summary, _result = artifact_runner._verify(artifact)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="boundflow-fsg4-b4a-tamper-") as raw:
        workspace = Path(raw)
        for name, mutate in ATTACKS:
            probe = workspace / name
            shutil.copytree(artifact, probe)
            mutate(probe)
            _resign_manifest(probe)
            rejection = ""
            try:
                artifact_runner._verify(probe)
            except (
                FileNotFoundError,
                TypeError,
                ValueError,
                subprocess.SubprocessError,
            ) as error:
                rejection = str(error)
            else:
                raise AssertionError(f"tampered FSG4/B4-A artifact admitted: {name}")
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
