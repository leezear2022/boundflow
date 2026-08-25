#!/usr/bin/env python3
"""Generate or replay the MR3 single-site production bridge timing artifact."""

# pylint: disable=missing-function-docstring,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-arguments,protected-access
# pylint: disable=line-too-long,wrong-import-position,too-many-boolean-expressions

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.mr3_production_bridge_timing import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    EXPECTED_RUNS,
    HOST_GEOMEAN_GATE,
    MEMORY_RATIO_GATE,
    SOURCE_COMMIT,
    TIMING_SCHEMA,
    WORST_PAIR_GATE,
    canonical_hash,
    derive_summary,
)

ARTIFACT_SCHEMA = "boundflow.mr3-production-bridge-timing-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.mr3-production-bridge-timing-protocol/v1"
WORKER = "scripts/run_mr3_production_bridge_timing_worker.py"
CORRECTNESS_ARTIFACT = (
    ROOT / "artifacts/measurement-recovery/mr3-p-production-bridge-correctness-v1"
)
CODE_PATHS = (
    "boundflow/runtime/mr3_production_bridge_timing.py",
    WORKER,
    "scripts/run_mr3_production_bridge_timing_formal.py",
    "scripts/probe_mr3_production_bridge_timing_tamper.py",
)


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"MR3 timing JSON root differs: {path.name}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"MR3 timing JSONL row differs: {path.name}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _historical_sha(commit: str, path: str) -> str:
    value = subprocess.run(
        ("git", "show", f"{commit}:{path}"),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    return hashlib.sha256(value).hexdigest()


def _current_code_revision() -> dict[str, str]:
    return {path: _sha256(ROOT / path) for path in CODE_PATHS}


def _historical_code_revision(commit: str) -> dict[str, str]:
    return {path: _historical_sha(commit, path) for path in CODE_PATHS}


def _correctness_identity() -> dict[str, object]:
    replay = subprocess.run(
        (
            sys.executable,
            str(ROOT / "scripts/run_mr3_production_bridge_formal.py"),
            "--artifact",
            str(CORRECTNESS_ARTIFACT),
            "--replay",
        ),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    manifest = _load_json(CORRECTNESS_ARTIFACT / "manifest.json")
    if "VALIDATED-MR3-P-PRODUCTION-BRIDGE-CORRECTNESS" not in replay.stdout:
        raise ValueError("MR3 correctness prerequisite did not replay")
    return {
        "manifest_file_sha256": _sha256(CORRECTNESS_ARTIFACT / "manifest.json"),
        "manifest_hash": manifest["manifest_hash"],
        "summary_hash": manifest["summary_hash"],
        "replay_status": "VALIDATED-MR3-P-PRODUCTION-BRIDGE-CORRECTNESS",
    }


def _protocol(args: argparse.Namespace) -> dict[str, object]:
    head = _git("rev-parse", "HEAD")
    if _git("merge-base", "--is-ancestor", SOURCE_COMMIT, head) != "":
        raise AssertionError("git merge-base emitted unexpected output")
    if _sha256(ROOT / WORKER) != _historical_sha(SOURCE_COMMIT, WORKER):
        raise ValueError("MR3 timing worker changed after source freeze")
    value: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "generator_commit": head,
        "code_revision": _current_code_revision(),
        "correctness_identity": _correctness_identity(),
        "run_order": [list(run) for run in EXPECTED_RUNS],
        "worker_count": 12,
        "pair_count": 6,
        "headline_clock": "host-perf-counter-ns",
        "diagnostic_clock": "same-current-stream-cuda-event",
        "host_geomean_gate": HOST_GEOMEAN_GATE,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_lower_gate": 1.0,
        "worst_pair_gate": WORST_PAIR_GATE,
        "absolute_peak_memory_ratio_gate": MEMORY_RATIO_GATE,
        "resume_policy": "reject-any-existing-artifact",
        "performance_claimed": False,
        "model_name": args.model.name,
        "property_name": args.property.name,
        "python_name": args.abcrown_python.name,
    }
    value["protocol_hash"] = canonical_hash(value)
    return value


def _worker_command(
    args: argparse.Namespace, *, mode: str, output: Path
) -> tuple[str, ...]:
    return (
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
        str(output),
    )


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    tvm_build = ROOT / "boundflow/3rdparty/tvm/build-boundflow"
    python_path = [str(ROOT / "boundflow/3rdparty/tvm/python"), str(ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    library_path = [str(tvm_build), str(tvm_build / "lib")]
    if environment.get("LD_LIBRARY_PATH"):
        library_path.append(environment["LD_LIBRARY_PATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_path),
            "TVM_LIBRARY_PATH": str(tvm_build),
            "LD_LIBRARY_PATH": os.pathsep.join(library_path),
            "TVM_FFI_DISABLE_TORCH_C_DLPACK": "1",
        }
    )
    return environment


def _run_worker(
    args: argparse.Namespace,
    *,
    pair_index: int,
    position: int,
    mode: str,
    workspace: Path,
) -> dict[str, object]:
    result_path = workspace / f"run_{pair_index}_{position}_{mode}.json"
    completed = subprocess.run(
        _worker_command(args, mode=mode, output=result_path),
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
            f"MR3 timing worker failed pair={pair_index} position={position} "
            f"mode={mode}: rc={completed.returncode}\n{completed.stderr[-4000:]}"
        )
    return {
        "pair_index": pair_index,
        "position": position,
        "mode": mode,
        "worker": _load_json(result_path),
    }


def _files(artifact: Path) -> dict[str, str]:
    return {
        str(path.relative_to(artifact)): _sha256(path)
        for path in sorted(artifact.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def refresh_manifest(artifact: Path) -> dict[str, object]:
    protocol = _load_json(artifact / "protocol.json")
    raw = _load_json(artifact / "raw.json")
    summary = _load_json(artifact / "summary.json")
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
    _write_json(artifact / "manifest.json", manifest)
    return manifest


def generate_artifact(args: argparse.Namespace) -> dict[str, object]:
    artifact = args.artifact
    if artifact.exists():
        raise FileExistsError(
            "MR3 timing formal artifact already exists; resume forbidden"
        )
    protocol = _protocol(args)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="boundflow-mr3-timing-formal-", dir=artifact.parent
    ) as temporary:
        temporary_path = Path(temporary)
        worker_workspace = temporary_path / "workers"
        worker_workspace.mkdir()
        runs = [
            _run_worker(
                args,
                pair_index=pair,
                position=position,
                mode=mode,
                workspace=worker_workspace,
            )
            for pair, position, mode in EXPECTED_RUNS
        ]
        raw: dict[str, object] = {
            "schema_version": TIMING_SCHEMA,
            "source_commit": SOURCE_COMMIT,
            "run_order": [list(run) for run in EXPECTED_RUNS],
            "runs": runs,
        }
        raw["raw_hash"] = canonical_hash(raw)
        _write_json(temporary_path / "raw.json", raw)
        summary = derive_summary(raw)
        _write_json(temporary_path / "protocol.json", protocol)
        _write_json(temporary_path / "summary.json", summary)
        _write_jsonl(temporary_path / "pair_metrics.jsonl", summary["pair_metrics"])
        (temporary_path / "replay_stdout.txt").write_text(
            _canonical_json(
                {
                    "status": summary["status"],
                    "summary_hash": summary["summary_hash"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        refresh_manifest(temporary_path)
        os.replace(temporary_path, artifact)
    return replay_artifact(artifact)


def replay_artifact(artifact: Path) -> dict[str, object]:
    manifest = _load_json(artifact / "manifest.json")
    unsigned_manifest = dict(manifest)
    manifest_hash = unsigned_manifest.pop("manifest_hash", None)
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest.get("source_commit") != SOURCE_COMMIT
        or manifest_hash != canonical_hash(unsigned_manifest)
        or manifest.get("files") != _files(artifact)
    ):
        raise ValueError("MR3 timing artifact manifest differs")
    protocol = _load_json(artifact / "protocol.json")
    unsigned_protocol = dict(protocol)
    protocol_hash = unsigned_protocol.pop("protocol_hash", None)
    generator_commit = protocol.get("generator_commit")
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("source_commit") != SOURCE_COMMIT
        or protocol_hash != canonical_hash(unsigned_protocol)
        or manifest.get("protocol_hash") != protocol_hash
        or not isinstance(generator_commit, str)
        or protocol.get("code_revision") != _historical_code_revision(generator_commit)
    ):
        raise ValueError("MR3 timing artifact protocol differs")
    raw = _load_json(artifact / "raw.json")
    summary = derive_summary(raw)
    frozen_summary = _load_json(artifact / "summary.json")
    if (
        summary != frozen_summary
        or manifest.get("raw_hash") != raw.get("raw_hash")
        or manifest.get("summary_hash") != summary.get("summary_hash")
        or manifest.get("status") != summary.get("status")
        or manifest.get("performance_claimed") != summary.get("performance_claimed")
        or _load_jsonl(artifact / "pair_metrics.jsonl") != summary["pair_metrics"]
    ):
        raise ValueError("MR3 timing artifact derived payload differs")
    for path in artifact.rglob("*"):
        if path.is_file() and "/home/" in path.read_text(encoding="utf-8"):
            raise ValueError("MR3 timing artifact leaks a local path")
    return summary


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
                "generation requires all repository, Python, model, and property paths"
            )
        summary = generate_artifact(args)
    print(
        _canonical_json(
            {"status": summary["status"], "summary_hash": summary["summary_hash"]}
        )
    )


if __name__ == "__main__":
    main()
