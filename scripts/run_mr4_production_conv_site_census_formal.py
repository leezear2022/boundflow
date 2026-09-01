#!/usr/bin/env python3
"""Generate or replay the MR4 production Conv-site census artifact."""

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

from boundflow.runtime.mr4_production_conv_site_census import (  # noqa: E402
    EXPECTED_RUNS,
    FORMAL_SCHEMA,
    SOURCE_COMMIT,
    canonical_hash,
    derive_summary,
)

ARTIFACT_SCHEMA = "boundflow.mr4-production-conv-site-census-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.mr4-production-conv-site-census-protocol/v1"
WORKER = "scripts/run_mr4_production_conv_site_census_worker.py"
MR3_TIMING_ARTIFACT = (
    ROOT / "artifacts/measurement-recovery/mr3-p-production-bridge-timing-v1"
)
CODE_PATHS = (
    "boundflow/runtime/mr4_production_conv_site_census.py",
    WORKER,
    "scripts/run_mr4_production_conv_site_census_formal.py",
    "scripts/probe_mr4_production_conv_site_census_tamper.py",
    "tests/test_mr4_production_conv_site_census.py",
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
        raise TypeError(f"MR4 census JSON root differs: {path.name}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"MR4 census JSONL row differs: {path.name}")
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


def _mr3_identity() -> dict[str, object]:
    replay = subprocess.run(
        (
            sys.executable,
            str(ROOT / "scripts/run_mr3_production_bridge_timing_formal.py"),
            "--artifact",
            str(MR3_TIMING_ARTIFACT),
            "--replay",
        ),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    manifest = _load_json(MR3_TIMING_ARTIFACT / "manifest.json")
    if "VALIDATED-NO-GO-MR3-P-PRODUCTION-BRIDGE-PHYSICS" not in replay.stdout:
        raise ValueError("MR4 census MR3 prerequisite did not replay")
    return {
        "manifest_file_sha256": _sha256(MR3_TIMING_ARTIFACT / "manifest.json"),
        "manifest_hash": manifest["manifest_hash"],
        "summary_hash": manifest["summary_hash"],
        "status": "VALIDATED-NO-GO-MR3-P-PRODUCTION-BRIDGE-PHYSICS",
    }


def _protocol(args: argparse.Namespace) -> dict[str, object]:
    head = _git("rev-parse", "HEAD")
    _git("merge-base", "--is-ancestor", SOURCE_COMMIT, head)
    if _sha256(ROOT / WORKER) != _historical_sha(SOURCE_COMMIT, WORKER):
        raise ValueError("MR4 census worker changed after source freeze")
    value: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "generator_commit": head,
        "code_revision": _current_code_revision(),
        "mr3_prerequisite": _mr3_identity(),
        "run_order": list(EXPECTED_RUNS),
        "worker_count": 5,
        "candidate_executed": False,
        "timing_recorded": False,
        "eligible_total_mac_ratio_gate": 1.75,
        "new_site_mac_ratio_gate": 0.75,
        "resume_policy": "reject-any-existing-artifact",
        "performance_claimed": False,
        "model_name": args.model.name,
        "property_name": args.property.name,
        "python_name": args.abcrown_python.name,
    }
    value["protocol_hash"] = canonical_hash(value)
    return value


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
    args: argparse.Namespace, *, run_index: int, workspace: Path
) -> dict[str, object]:
    result_path = workspace / f"run_{run_index}.json"
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
        "--run-index",
        str(run_index),
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
            f"MR4 census worker failed run={run_index}: rc={completed.returncode}\n"
            f"{completed.stderr[-4000:]}"
        )
    return {"run_index": run_index, "worker": _load_json(result_path)}


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
        "performance_claimed": False,
        "files": _files(artifact),
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact / "manifest.json", manifest)
    return manifest


def generate_artifact(args: argparse.Namespace) -> dict[str, object]:
    artifact = args.artifact
    if artifact.exists():
        raise FileExistsError("MR4 census artifact already exists; resume forbidden")
    protocol = _protocol(args)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="boundflow-mr4-census-formal-", dir=artifact.parent
    ) as temporary:
        root = Path(temporary)
        staging = root / "artifact"
        workers = root / "workers"
        staging.mkdir()
        workers.mkdir()
        runs = [
            _run_worker(args, run_index=run_index, workspace=workers)
            for run_index in EXPECTED_RUNS
        ]
        raw: dict[str, object] = {
            "schema_version": FORMAL_SCHEMA,
            "source_commit": SOURCE_COMMIT,
            "run_order": list(EXPECTED_RUNS),
            "runs": runs,
        }
        raw["raw_hash"] = canonical_hash(raw)
        _write_json(staging / "raw.json", raw)
        summary = derive_summary(raw)
        _write_json(staging / "protocol.json", protocol)
        _write_json(staging / "summary.json", summary)
        _write_jsonl(staging / "semantic_metrics.jsonl", summary["semantic_metrics"])
        (staging / "replay_stdout.txt").write_text(
            _canonical_json(
                {"status": summary["status"], "summary_hash": summary["summary_hash"]}
            )
            + "\n",
            encoding="utf-8",
        )
        refresh_manifest(staging)
        os.replace(staging, artifact)
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
        raise ValueError("MR4 census artifact manifest differs")
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
        raise ValueError("MR4 census artifact protocol differs")
    raw = _load_json(artifact / "raw.json")
    summary = derive_summary(raw)
    if (
        summary != _load_json(artifact / "summary.json")
        or manifest.get("raw_hash") != raw.get("raw_hash")
        or manifest.get("summary_hash") != summary.get("summary_hash")
        or manifest.get("status") != summary.get("status")
        or _load_jsonl(artifact / "semantic_metrics.jsonl")
        != summary["semantic_metrics"]
    ):
        raise ValueError("MR4 census derived payload differs")
    for path in artifact.rglob("*"):
        if path.is_file() and "/home/" in path.read_text(encoding="utf-8"):
            raise ValueError("MR4 census artifact leaks a local path")
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
