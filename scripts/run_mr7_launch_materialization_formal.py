#!/usr/bin/env python3
"""Generate or replay the MR7 six-fresh launch/materialization artifact."""

# pylint: disable=missing-function-docstring,protected-access,too-many-locals
# pylint: disable=too-many-statements,wrong-import-position,duplicate-code
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.mr3_provider_hook_feasibility import canonical_hash  # noqa: E402
from boundflow.runtime.mr7_launch_materialization_attribution import (  # noqa: E402
    RAW_SCHEMA,
    derive_summary,
)
from scripts import run_mr3_production_bridge_timing_formal as helpers  # noqa: E402
from scripts import run_mr6_guard_attribution_formal as mr6_formal  # noqa: E402

ARTIFACT_SCHEMA = "boundflow.mr7-launch-materialization-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.mr7-launch-materialization-protocol/v1"
SOURCE_COMMIT = "527618241861dab9d03dfd4b3c8229ef53b79b55"
WORKER = "scripts/run_mr7_launch_materialization_worker.py"
EXPECTED_RUNS = (
    (0, 0, "control"),
    (0, 1, "profile"),
    (1, 0, "profile"),
    (1, 1, "control"),
    (2, 0, "control"),
    (2, 1, "profile"),
)
MR6_ARTIFACT = ROOT / "artifacts/measurement-recovery/mr6-hot-path-guard-attribution-v1"
CODE_PATHS = (
    "boundflow/runtime/mr7_launch_materialization_attribution.py",
    WORKER,
    "scripts/run_mr7_launch_materialization_formal.py",
    "scripts/probe_mr7_launch_materialization_tamper.py",
    "tests/test_mr7_launch_materialization_attribution.py",
    "tests/test_mr7_launch_materialization_artifact.py",
)


def _validate_raw_extras(raw: dict[str, Any]) -> None:
    runs = raw.get("runs")
    if not isinstance(runs, list):
        raise ValueError("MR7 raw runs absent")
    expected_counts = {
        "forward.C0": 10,
        "forward.C1": 10,
        "forward.C2": 10,
        "backward.C0": 9,
        "backward.C1": 9,
        "backward.C2": 9,
    }
    for wrapper in runs:
        if not isinstance(wrapper, dict) or not isinstance(wrapper.get("worker"), dict):
            raise ValueError("MR7 worker extra envelope absent")
        worker = cast(dict[str, Any], wrapper["worker"])
        events = worker.get("device_events")
        marker_totals = worker.get("device_marker_totals")
        kind = wrapper.get("kind")
        if (
            not isinstance(events, list)
            or not isinstance(marker_totals, dict)
            or worker.get("device_event_hash") != canonical_hash(events)
            or worker.get("launch_marker_counts") != expected_counts
            or worker.get("timing_recorded") is not True
            or worker.get("production_admitted") is not False
            or worker.get("performance_claimed") is not False
            or (kind == "control" and (events or marker_totals))
            or (kind == "profile" and (not events or len(marker_totals) != 57))
        ):
            raise ValueError("MR7 worker extra envelope differs")


def _mr6_identity() -> dict[str, object]:
    summary = mr6_formal.replay_artifact(MR6_ARTIFACT)
    manifest = helpers._load_json(MR6_ARTIFACT / "manifest.json")
    return {
        "status": summary["status"],
        "summary_hash": summary["summary_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "manifest_file_sha256": helpers._sha256(MR6_ARTIFACT / "manifest.json"),
    }


def _protocol(args: argparse.Namespace) -> dict[str, object]:
    head = helpers._git("rev-parse", "HEAD")
    if helpers._git("merge-base", "--is-ancestor", SOURCE_COMMIT, head) != "":
        raise AssertionError("git merge-base emitted unexpected output")
    for path in (
        "boundflow/runtime/mr7_launch_materialization_attribution.py",
        WORKER,
    ):
        if helpers._sha256(ROOT / path) != helpers._historical_sha(SOURCE_COMMIT, path):
            raise ValueError(f"MR7 source changed after freeze: {path}")
    value: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "generator_commit": head,
        "code_revision": {path: helpers._sha256(ROOT / path) for path in CODE_PATHS},
        "mr6_identity": _mr6_identity(),
        "run_order": [list(item) for item in EXPECTED_RUNS],
        "worker_count": 6,
        "pair_count": 3,
        "headline_host_clock": "unprofiled-host-perf-counter-ns",
        "device_clock": "profiled-cupti-kernel-and-record-function-device-total",
        "clock_domain_policy": "host-and-device-ledgers-never-added",
        "calibration_policy": "explicit-cpu-parent-correlation-no-temporal-share",
        "profile_control_cuda_event_ratio_gate": 1.10,
        "host_closure_error_gate": 0.02,
        "device_envelope_error_gate": 0.02,
        "boundary_share_gate": 0.15,
        "boundary_absolute_ns_gate": 15_000_000,
        "kernel_share_gate": 0.50,
        "parity_candidate_speedup": 1.107412,
        "research_candidate_speedup": 1.273523,
        "maximum_required_region_speedup": 10.0,
        "resume_policy": "reject-any-existing-artifact",
        "diagnostic_production_admitted": False,
        "performance_claimed": False,
        "model_name": args.model.name,
        "property_name": args.property.name,
        "python_name": args.abcrown_python.name,
    }
    value["protocol_hash"] = canonical_hash(value)
    return value


def _run_worker(
    args: argparse.Namespace,
    *,
    pair_index: int,
    position: int,
    kind: str,
    workspace: Path,
) -> dict[str, object]:
    result_path = workspace / f"run_{pair_index}_{position}_{kind}.json"
    command = (
        str(args.abcrown_python),
        str(ROOT / WORKER),
        "--benchmark-root",
        str(args.benchmark_root),
        "--abcrown-root",
        str(args.abcrown_root),
        "--model",
        str(args.model),
        "--property",
        str(args.property),
        "--kind",
        kind,
        "--result-json",
        str(result_path),
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=helpers._worker_environment(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=240,
    )
    if completed.returncode or not result_path.is_file():
        raise RuntimeError(
            f"MR7 worker failed pair={pair_index} position={position} "
            f"kind={kind}: rc={completed.returncode}\n{completed.stderr[-4000:]}"
        )
    return {
        "pair_index": pair_index,
        "position": position,
        "kind": kind,
        "worker": helpers._load_json(result_path),
    }


def _files(artifact: Path) -> dict[str, str]:
    return {
        str(path.relative_to(artifact)): helpers._sha256(path)
        for path in sorted(artifact.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def refresh_manifest(artifact: Path) -> dict[str, object]:
    protocol = helpers._load_json(artifact / "protocol.json")
    raw = helpers._load_json(artifact / "raw.json")
    summary = helpers._load_json(artifact / "summary.json")
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "generator_commit": protocol["generator_commit"],
        "protocol_hash": protocol["protocol_hash"],
        "raw_hash": raw["raw_hash"],
        "summary_hash": summary["summary_hash"],
        "status": summary["status"],
        "performance_claimed": summary["performance_claimed"],
        "files": _files(artifact),
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    helpers._write_json(artifact / "manifest.json", manifest)
    return manifest


def generate_artifact(args: argparse.Namespace) -> dict[str, object]:
    if args.artifact.exists():
        raise FileExistsError("MR7 formal artifact already exists; resume forbidden")
    protocol = _protocol(args)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="boundflow-mr7-formal-", dir=args.artifact.parent
    ) as temporary:
        temporary_path = Path(temporary)
        workspace = temporary_path / "workers"
        workspace.mkdir()
        runs = [
            _run_worker(
                args,
                pair_index=pair,
                position=position,
                kind=kind,
                workspace=workspace,
            )
            for pair, position, kind in EXPECTED_RUNS
        ]
        raw: dict[str, object] = {
            "schema_version": RAW_SCHEMA,
            "source_commit": SOURCE_COMMIT,
            "run_order": [list(item) for item in EXPECTED_RUNS],
            "runs": runs,
        }
        raw["raw_hash"] = canonical_hash(raw)
        _validate_raw_extras(cast(dict[str, Any], raw))
        summary = derive_summary(raw)
        helpers._write_json(temporary_path / "protocol.json", protocol)
        helpers._write_json(temporary_path / "raw.json", raw)
        helpers._write_json(temporary_path / "summary.json", summary)
        helpers._write_jsonl(
            temporary_path / "pair_metrics.jsonl",
            cast(Sequence[object], summary["pair_metrics"]),
        )
        (temporary_path / "replay_stdout.txt").write_text(
            helpers._canonical_json(
                {"status": summary["status"], "summary_hash": summary["summary_hash"]}
            )
            + "\n",
            encoding="utf-8",
        )
        for worker_file in workspace.iterdir():
            worker_file.unlink()
        workspace.rmdir()
        refresh_manifest(temporary_path)
        os.replace(temporary_path, args.artifact)
    return replay_artifact(args.artifact)


def replay_artifact(artifact: Path) -> dict[str, object]:
    manifest = helpers._load_json(artifact / "manifest.json")
    unsigned_manifest = dict(manifest)
    manifest_hash = unsigned_manifest.pop("manifest_hash", None)
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest.get("source_commit") != SOURCE_COMMIT
        or manifest_hash != canonical_hash(unsigned_manifest)
        or manifest.get("files") != _files(artifact)
    ):
        raise ValueError("MR7 artifact manifest differs")
    protocol = helpers._load_json(artifact / "protocol.json")
    unsigned_protocol = dict(protocol)
    protocol_hash = unsigned_protocol.pop("protocol_hash", None)
    generator_commit = protocol.get("generator_commit")
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("source_commit") != SOURCE_COMMIT
        or protocol_hash != canonical_hash(unsigned_protocol)
        or manifest.get("protocol_hash") != protocol_hash
        or not isinstance(generator_commit, str)
        or protocol.get("code_revision")
        != {
            path: helpers._historical_sha(generator_commit, path) for path in CODE_PATHS
        }
        or protocol.get("mr6_identity") != _mr6_identity()
    ):
        raise ValueError("MR7 artifact protocol differs")
    raw = helpers._load_json(artifact / "raw.json")
    _validate_raw_extras(raw)
    summary = derive_summary(raw)
    if (
        summary != helpers._load_json(artifact / "summary.json")
        or manifest.get("raw_hash") != raw.get("raw_hash")
        or manifest.get("summary_hash") != summary.get("summary_hash")
        or manifest.get("status") != summary.get("status")
        or manifest.get("performance_claimed") is not False
        or helpers._load_jsonl(artifact / "pair_metrics.jsonl")
        != summary["pair_metrics"]
    ):
        raise ValueError("MR7 artifact derived payload differs")
    for path in artifact.rglob("*"):
        if path.is_file() and "/home/" in path.read_text(encoding="utf-8"):
            raise ValueError("MR7 artifact leaks a local path")
    return summary


def _load_json(path: Path) -> dict[str, Any]:
    return helpers._load_json(path)


def _write_json(path: Path, value: object) -> None:
    helpers._write_json(path, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--abcrown-root", type=Path)
    parser.add_argument("--abcrown-python", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--property", type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        summary = replay_artifact(args.artifact)
    else:
        required = (
            args.benchmark_root,
            args.abcrown_root,
            args.abcrown_python,
            args.model,
            args.property,
        )
        if any(item is None for item in required):
            parser.error(
                "generation requires all repositories, Python, model and property"
            )
        summary = generate_artifact(args)
    print(
        helpers._canonical_json(
            {"status": summary["status"], "summary_hash": summary["summary_hash"]}
        )
    )


if __name__ == "__main__":
    main()
