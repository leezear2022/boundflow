#!/usr/bin/env python3
"""Generate or replay one full-stack GPU attribution contract artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from boundflow.runtime.gpu_attribution import (
    canonical_hash,
    full_stack_run_from_dict,
    summarize_run,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.full-stack-gpu-attribution-artifact/v1"
MANIFEST_FILE = "manifest.json"
ARTIFACT_FILES = (
    "raw_run.json",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    """Hash one artifact or code file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=_repo_root(),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _code_revision() -> dict[str, str]:
    root = _repo_root()
    paths = (
        "boundflow/runtime/gpu_attribution.py",
        "scripts/run_full_stack_gpu_baseline_attribution.py",
    )
    return {path: file_sha256(root / path) for path in paths}


def _replay_payload(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "run_id": summary["run_id"],
        "configuration_id": summary["configuration_id"],
        "closure_passed": summary["closure_passed"],
        "residual_passed": summary["residual_passed"],
        "attribution_passed": summary["attribution_passed"],
        "run_hash": summary["run_hash"],
        "summary_hash": summary["summary_hash"],
    }


def _stdout(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def generate_artifact(raw_run_path: Path, artifact_dir: Path) -> dict[str, object]:
    """Validate one producer trace and freeze a replayable FSG0 artifact."""

    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run = full_stack_run_from_dict(_load_json(raw_run_path))
    raw_payload = run.to_dict()
    summary = summarize_run(run)
    if summary["attribution_passed"] is not True:
        raise ValueError("full-stack attribution closure or residual gate failed")
    _write_json(artifact_dir / "raw_run.json", raw_payload)
    _write_json(artifact_dir / "summary.json", summary)
    replay_stdout = _stdout(_replay_payload(summary)) + "\n"
    (artifact_dir / "replay_stdout.txt").write_text(replay_stdout, encoding="utf-8")
    (artifact_dir / "README.md").write_text(
        "\n".join(
            (
                "# Full-Stack GPU Attribution Contract Artifact",
                "",
                "This FSG0 artifact validates schema, closure, and feature activation.",
                "It is not a GPU performance result.",
                "",
            )
        ),
        encoding="utf-8",
    )
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "contract-only",
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_dirty_paths": (_git_value("status", "--porcelain=v1") or "").splitlines(),
        "code_revision": _code_revision(),
        "files": {name: file_sha256(artifact_dir / name) for name in ARTIFACT_FILES},
        "run_hash": summary["run_hash"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact_dir / MANIFEST_FILE, manifest)
    return _replay_payload(summary)


def replay_artifact(artifact_dir: Path) -> dict[str, object]:
    """Recompute closure and feature evidence from the frozen raw run."""

    manifest = _load_json(artifact_dir / MANIFEST_FILE)
    semantic_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("status") != "contract-only"
        or manifest.get("performance_claimed") is not False
        or manifest.get("manifest_hash") != canonical_hash(semantic_manifest)
        or manifest.get("code_revision") != _code_revision()
    ):
        raise ValueError("full-stack artifact manifest envelope differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("full-stack artifact file inventory differs")
    if any(
        file_sha256(artifact_dir / name) != digest for name, digest in files.items()
    ):
        raise ValueError("full-stack artifact file digest differs")
    run = full_stack_run_from_dict(_load_json(artifact_dir / "raw_run.json"))
    rebuilt = summarize_run(run)
    stored = _load_json(artifact_dir / "summary.json")
    if (
        rebuilt != stored
        or manifest.get("run_hash") != rebuilt["run_hash"]
        or manifest.get("summary_hash") != rebuilt["summary_hash"]
    ):
        raise ValueError("full-stack artifact semantic replay differs")
    result = _replay_payload(rebuilt)
    if (artifact_dir / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _stdout(result) + "\n"
    ):
        raise ValueError("full-stack artifact replay stdout differs")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--raw-run", type=Path, required=True)
    generate.add_argument("--artifact-dir", type=Path, required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the contract generator or deterministic replay."""

    args = _parse_args()
    if args.command == "generate":
        result = generate_artifact(args.raw_run.resolve(), args.artifact_dir.resolve())
    else:
        result = replay_artifact(args.artifact_dir.resolve())
    print(_stdout(result))


if __name__ == "__main__":
    main()
