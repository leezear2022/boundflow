#!/usr/bin/env python3
"""Generate or replay the MR6 three-triplet guard attribution artifact."""

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
from boundflow.runtime.mr6_guard_attribution import (  # noqa: E402
    EXPECTED_RUNS,
    RAW_SCHEMA,
    SOURCE_COMMIT,
    derive_summary,
)
from scripts import run_mr3_production_bridge_timing_formal as helpers  # noqa: E402
from scripts import run_mr5_multi_conv_timing_formal as mr5_formal  # noqa: E402

ARTIFACT_SCHEMA = "boundflow.mr6-hot-path-guard-attribution-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.mr6-hot-path-guard-attribution-protocol/v1"
WORKER = "scripts/run_mr6_guard_attribution_worker.py"
MR5_TIMING_ARTIFACT = ROOT / "artifacts/measurement-recovery/mr5-multi-conv-timing-v1"
CODE_PATHS = (
    "boundflow/runtime/mr6_guard_attribution.py",
    WORKER,
    "scripts/run_mr6_guard_attribution_formal.py",
    "scripts/probe_mr6_guard_attribution_tamper.py",
    "tests/test_mr6_guard_attribution.py",
    "tests/test_mr6_guard_attribution_worker.py",
)


def _mr5_identity() -> dict[str, object]:
    summary = mr5_formal.replay_artifact(MR5_TIMING_ARTIFACT)
    manifest = helpers._load_json(MR5_TIMING_ARTIFACT / "manifest.json")
    return {
        "status": summary["status"],
        "summary_hash": summary["summary_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "manifest_file_sha256": helpers._sha256(MR5_TIMING_ARTIFACT / "manifest.json"),
    }


def _protocol(args: argparse.Namespace) -> dict[str, object]:
    head = helpers._git("rev-parse", "HEAD")
    if helpers._git("merge-base", "--is-ancestor", SOURCE_COMMIT, head) != "":
        raise AssertionError("git merge-base emitted unexpected output")
    if helpers._sha256(ROOT / WORKER) != helpers._historical_sha(SOURCE_COMMIT, WORKER):
        raise ValueError("MR6 attribution worker changed after source freeze")
    value: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "generator_commit": head,
        "code_revision": {path: helpers._sha256(ROOT / path) for path in CODE_PATHS},
        "mr5_timing_identity": _mr5_identity(),
        "run_order": [list(item) for item in EXPECTED_RUNS],
        "worker_count": 9,
        "triplet_count": 3,
        "headline_clock": "host-perf-counter-ns",
        "diagnostic_clock": "same-current-stream-cuda-event",
        "full_diagnostic_geomean_gate": 1.10,
        "provider_diagnostic_geomean_gate": 0.98,
        "provider_diagnostic_worst_gate": 0.95,
        "full_guard_count": 360,
        "diagnostic_guard_count": 60,
        "resume_policy": "reject-any-existing-artifact",
        "diagnostic_production_admitted": False,
        "performance_claimed": False,
        "model_name": args.model.name,
        "property_name": args.property.name,
        "python_name": args.abcrown_python.name,
    }
    value["protocol_hash"] = canonical_hash(value)
    return value


def _worker_environment() -> dict[str, str]:
    return helpers._worker_environment()


def _run_worker(
    args: argparse.Namespace,
    *,
    triplet_index: int,
    position: int,
    mode: str,
    workspace: Path,
) -> dict[str, object]:
    result_path = workspace / f"run_{triplet_index}_{position}_{mode}.json"
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
        "--mode",
        mode,
        "--result-json",
        str(result_path),
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_worker_environment(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    if completed.returncode or not result_path.is_file():
        raise RuntimeError(
            f"MR6 worker failed triplet={triplet_index} position={position} "
            f"mode={mode}: rc={completed.returncode}\n{completed.stderr[-4000:]}"
        )
    return {
        "triplet_index": triplet_index,
        "position": position,
        "mode": mode,
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
        raise FileExistsError("MR6 formal artifact already exists; resume forbidden")
    protocol = _protocol(args)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="boundflow-mr6-guard-formal-", dir=args.artifact.parent
    ) as temporary:
        temporary_path = Path(temporary)
        workspace = temporary_path / "workers"
        workspace.mkdir()
        runs = [
            _run_worker(
                args,
                triplet_index=triplet,
                position=position,
                mode=mode,
                workspace=workspace,
            )
            for triplet, position, mode in EXPECTED_RUNS
        ]
        raw: dict[str, object] = {
            "schema_version": RAW_SCHEMA,
            "source_commit": SOURCE_COMMIT,
            "run_order": [list(item) for item in EXPECTED_RUNS],
            "runs": runs,
        }
        raw["raw_hash"] = canonical_hash(raw)
        summary = derive_summary(raw)
        helpers._write_json(temporary_path / "protocol.json", protocol)
        helpers._write_json(temporary_path / "raw.json", raw)
        helpers._write_json(temporary_path / "summary.json", summary)
        helpers._write_jsonl(
            temporary_path / "triplet_metrics.jsonl",
            cast(Sequence[object], summary["triplet_metrics"]),
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
        raise ValueError("MR6 artifact manifest differs")
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
        or protocol.get("mr5_timing_identity") != _mr5_identity()
    ):
        raise ValueError("MR6 artifact protocol differs")
    raw = helpers._load_json(artifact / "raw.json")
    summary = derive_summary(raw)
    if (
        summary != helpers._load_json(artifact / "summary.json")
        or manifest.get("raw_hash") != raw.get("raw_hash")
        or manifest.get("summary_hash") != summary.get("summary_hash")
        or manifest.get("status") != summary.get("status")
        or manifest.get("performance_claimed") is not False
        or helpers._load_jsonl(artifact / "triplet_metrics.jsonl")
        != summary["triplet_metrics"]
    ):
        raise ValueError("MR6 artifact derived payload differs")
    for path in artifact.rglob("*"):
        if path.is_file() and "/home/" in path.read_text(encoding="utf-8"):
            raise ValueError("MR6 artifact leaks a local path")
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
