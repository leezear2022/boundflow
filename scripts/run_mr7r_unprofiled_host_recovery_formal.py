#!/usr/bin/env python3
"""Generate or replay the MR7-R ten-fresh unprofiled recovery artifact."""

# pylint: disable=missing-function-docstring,protected-access,too-many-locals
# pylint: disable=too-many-statements,wrong-import-position,duplicate-code
# pylint: disable=too-many-boolean-expressions,too-many-branches

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

from boundflow.runtime.mr3_provider_hook_feasibility import (  # noqa: E402
    ABCROWN_COMMIT,
    AUTO_LIRPA_COMMIT,
    VNNCOMP_COMMIT,
    canonical_hash,
)
from boundflow.runtime.mr7r_unprofiled_host_recovery import (  # noqa: E402
    EXPECTED_RUNS,
    RAW_SCHEMA,
    derive_summary,
)
from scripts import run_mr3_production_bridge_timing_formal as helpers  # noqa: E402
from scripts import run_mr6_guard_attribution_formal as mr6_formal  # noqa: E402
from scripts import run_mr7_launch_materialization_formal as mr7_formal  # noqa: E402

ARTIFACT_SCHEMA = "boundflow.mr7r-unprofiled-host-recovery-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.mr7r-unprofiled-host-recovery-protocol/v1"
SOURCE_COMMIT = "0a1e79553e216ed5c34604a235d537288fcf8e19"
BASELINE_WORKER = "scripts/run_mr6_guard_attribution_worker.py"
LEDGER_WORKER = "scripts/run_mr7_launch_materialization_worker.py"
MR6_ARTIFACT = ROOT / "artifacts/measurement-recovery/mr6-hot-path-guard-attribution-v1"
MR7_ARTIFACT = (
    ROOT / "artifacts/measurement-recovery/mr7-launch-materialization-attribution-v1"
)
FROZEN_SOURCE_PATHS = (
    "boundflow/runtime/mr7r_unprofiled_host_recovery.py",
    BASELINE_WORKER,
    LEDGER_WORKER,
)
CODE_PATHS = (
    *FROZEN_SOURCE_PATHS,
    "scripts/run_mr7r_unprofiled_host_recovery_formal.py",
    "scripts/probe_mr7r_unprofiled_host_recovery_tamper.py",
    "tests/test_mr7r_unprofiled_host_recovery.py",
    "tests/test_mr7r_unprofiled_host_recovery_artifact.py",
)


def _predecessor_identity(artifact: Path, replay: Any) -> dict[str, object]:
    summary = replay(artifact)
    manifest = helpers._load_json(artifact / "manifest.json")
    return {
        "status": summary["status"],
        "summary_hash": summary["summary_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "manifest_file_sha256": helpers._sha256(artifact / "manifest.json"),
    }


def _static_protocol_fields() -> dict[str, object]:
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "run_order": [list(item) for item in EXPECTED_RUNS],
        "worker_count": 10,
        "pair_count": 5,
        "pair_order": "BL/LB/BL/LB/BL",
        "baseline_worker": BASELINE_WORKER,
        "baseline_mode": "diagnostic",
        "ledger_worker": LEDGER_WORKER,
        "ledger_kind": "control",
        "headline_clock": "unprofiled-host-perf-counter-ns",
        "diagnostic_clock": "same-current-stream-cuda-event",
        "profile_enabled": False,
        "host_closure_error_gate": 0.02,
        "ledger_baseline_host_ratio_median_gate": [0.95, 1.05],
        "ledger_baseline_host_ratio_run_gate": [0.90, 1.10],
        "host_event_direction_required_count": 5,
        "boundary_categories": [
            "ffi_dlpack_stream",
            "layout_materialization",
            "post_output_guard",
        ],
        "boundary_share_gate": 0.15,
        "boundary_absolute_ns_gate": 15_000_000,
        "boundary_qualifying_run_gate": 4,
        "parity_target": 1.107412,
        "maximum_required_region_speedup": 10.0,
        "resume_policy": "reject-any-existing-artifact",
        "diagnostic_production_admitted": False,
        "timing_open": False,
        "performance_claimed": False,
        "model_name": "resnet_2b.onnx",
        "property_name": "prop_0_eps_0.008.vnnlib",
        "python_name": "python",
        "abcrown_commit": ABCROWN_COMMIT,
        "auto_lirpa_commit": AUTO_LIRPA_COMMIT,
        "vnncomp_commit": VNNCOMP_COMMIT,
    }


def _protocol(args: argparse.Namespace) -> dict[str, object]:
    head = helpers._git("rev-parse", "HEAD")
    if helpers._git("merge-base", "--is-ancestor", SOURCE_COMMIT, head) != "":
        raise AssertionError("git merge-base emitted unexpected output")
    for path in FROZEN_SOURCE_PATHS:
        if helpers._sha256(ROOT / path) != helpers._historical_sha(SOURCE_COMMIT, path):
            raise ValueError(f"MR7-R source changed after freeze: {path}")
    static = _static_protocol_fields()
    if (
        args.model.name != static["model_name"]
        or args.property.name != static["property_name"]
        or args.abcrown_python.name != static["python_name"]
    ):
        raise ValueError("MR7-R frozen path basename differs")
    value: dict[str, object] = {
        **static,
        "generator_commit": head,
        "code_revision": {path: helpers._sha256(ROOT / path) for path in CODE_PATHS},
        "mr6_identity": _predecessor_identity(MR6_ARTIFACT, mr6_formal.replay_artifact),
        "mr7_identity": _predecessor_identity(MR7_ARTIFACT, mr7_formal.replay_artifact),
    }
    value["protocol_hash"] = canonical_hash(value)
    return value


def _run_worker(
    args: argparse.Namespace,
    *,
    pair_index: int,
    position: int,
    role: str,
    workspace: Path,
) -> dict[str, object]:
    result_path = workspace / f"run_{pair_index}_{position}_{role}.json"
    if role == "baseline":
        command = (
            str(args.abcrown_python),
            str(ROOT / BASELINE_WORKER),
            "--benchmark-root",
            str(args.benchmark_root),
            "--abcrown-root",
            str(args.abcrown_root),
            "--model",
            str(args.model),
            "--property",
            str(args.property),
            "--mode",
            "diagnostic",
            "--result-json",
            str(result_path),
        )
    elif role == "ledger":
        command = (
            str(args.abcrown_python),
            str(ROOT / LEDGER_WORKER),
            "--benchmark-root",
            str(args.benchmark_root),
            "--abcrown-root",
            str(args.abcrown_root),
            "--model",
            str(args.model),
            "--property",
            str(args.property),
            "--kind",
            "control",
            "--result-json",
            str(result_path),
        )
    else:
        raise ValueError("MR7-R worker role differs")
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
            f"MR7-R worker failed pair={pair_index} position={position} "
            f"role={role}: rc={completed.returncode}\n{completed.stderr[-4000:]}"
        )
    return {
        "pair_index": pair_index,
        "position": position,
        "role": role,
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
        raise FileExistsError("MR7-R formal artifact already exists; resume forbidden")
    protocol = _protocol(args)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="boundflow-mr7r-formal-", dir=args.artifact.parent
    ) as temporary:
        temporary_path = Path(temporary)
        workspace = temporary_path / "workers"
        workspace.mkdir()
        runs = [
            _run_worker(
                args,
                pair_index=pair,
                position=position,
                role=role,
                workspace=workspace,
            )
            for pair, position, role in EXPECTED_RUNS
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


def _validate_protocol(protocol: dict[str, Any], manifest: dict[str, Any]) -> None:
    unsigned = dict(protocol)
    protocol_hash = unsigned.pop("protocol_hash", None)
    generator_commit = protocol.get("generator_commit")
    if (
        protocol_hash != canonical_hash(unsigned)
        or manifest.get("protocol_hash") != protocol_hash
        or not isinstance(generator_commit, str)
        or any(
            protocol.get(key) != value
            for key, value in _static_protocol_fields().items()
        )
        or protocol.get("code_revision")
        != {
            path: helpers._historical_sha(generator_commit, path) for path in CODE_PATHS
        }
        or protocol.get("mr6_identity")
        != _predecessor_identity(MR6_ARTIFACT, mr6_formal.replay_artifact)
        or protocol.get("mr7_identity")
        != _predecessor_identity(MR7_ARTIFACT, mr7_formal.replay_artifact)
    ):
        raise ValueError("MR7-R artifact protocol differs")


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
        raise ValueError("MR7-R artifact manifest differs")
    protocol = helpers._load_json(artifact / "protocol.json")
    _validate_protocol(protocol, manifest)
    raw = helpers._load_json(artifact / "raw.json")
    summary = derive_summary(raw)
    if (
        raw.get("source_commit") != SOURCE_COMMIT
        or summary != helpers._load_json(artifact / "summary.json")
        or manifest.get("raw_hash") != raw.get("raw_hash")
        or manifest.get("summary_hash") != summary.get("summary_hash")
        or manifest.get("status") != summary.get("status")
        or manifest.get("performance_claimed") is not False
        or helpers._load_jsonl(artifact / "pair_metrics.jsonl")
        != summary["pair_metrics"]
    ):
        raise ValueError("MR7-R artifact derived payload differs")
    for path in artifact.rglob("*"):
        if path.is_file() and "/home/" in path.read_text(encoding="utf-8"):
            raise ValueError("MR7-R artifact leaks a local path")
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
