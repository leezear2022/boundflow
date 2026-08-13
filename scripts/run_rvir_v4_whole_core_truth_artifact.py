#!/usr/bin/env python3
"""Generate or replay the RVIR-v4 V4-3A whole-core truth artifact."""

# pylint: disable=wrong-import-position,protected-access,duplicate-code
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, cast

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.rvir_v4_whole_core_truth import (
    compare_rvir_v4_whole_core_truth,
    validate_rvir_v4_whole_core_truth,
)
from scripts import run_rvir_v4_production_state_capture as capture_runner

ARTIFACT_SCHEMA_VERSION = "boundflow.rvir-v4-whole-core-truth-artifact/v1"
REPLAY_CONTRACT = {
    "mode": "fresh-pinned-provider-rerun",
    "atol": 2e-4,
    "rtol": 2e-4,
    "shape_dtype_device_exact": True,
    "sign_exact": True,
    "discrete_structure_exact": True,
}
TRUTH_FILE = "truth.pt"
ARTIFACT_FILES = (
    TRUTH_FILE,
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)
CODE_PATHS = (
    "boundflow/runtime/rvir_v4_production_state.py",
    "boundflow/runtime/rvir_v4_whole_core_truth.py",
    "scripts/run_rvir_v4_production_state_capture.py",
    "scripts/run_rvir_v4_whole_core_truth_artifact.py",
)


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"RVIR-v4 whole-core JSON root differs: {path}")
    return value


def _load_truth(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("RVIR-v4 whole-core truth root differs")
    return value


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _code_revision() -> dict[str, str]:
    return {path: _file_sha256(REPOSITORY_ROOT / path) for path in CODE_PATHS}


def _code_paths_clean() -> bool:
    return not _git_value("status", "--porcelain=v1", "--", *CODE_PATHS)


def _verify_code_provenance(manifest: Mapping[str, Any]) -> None:
    source_head = manifest.get("source_git_head")
    revision = manifest.get("code_revision")
    if not isinstance(source_head, str) or not isinstance(revision, Mapping):
        raise ValueError("RVIR-v4 whole-core code provenance differs")
    if _git_value("rev-parse", "HEAD") == source_head:
        observed = _code_revision()
    else:
        observed = {
            path: hashlib.sha256(
                subprocess.run(
                    ("git", "show", f"{source_head}:{path}"),
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ).stdout
            ).hexdigest()
            for path in CODE_PATHS
        }
    if dict(revision) != observed:
        raise ValueError("RVIR-v4 whole-core code revision differs")


def _summary(truth: Mapping[str, Any]) -> dict[str, object]:
    if (
        truth.get("schema_version") != capture_runner.WHOLE_CORE_WORKER_SCHEMA_VERSION
        or truth.get("performance_claimed") is not False
    ):
        raise ValueError("RVIR-v4 whole-core worker schema differs")
    cores = truth.get("whole_core_truths")
    posts = truth.get("whole_post_truths")
    calls = truth.get("calls")
    solver = truth.get("solver_result")
    if (
        not isinstance(cores, list)
        or len(cores) != 1
        or not isinstance(cores[0], Mapping)
        or not isinstance(posts, list)
        or len(posts) != 1
        or not isinstance(posts[0], Mapping)
        or not isinstance(calls, list)
        or len(calls) != 24
        or not isinstance(solver, Mapping)
    ):
        raise ValueError("RVIR-v4 whole-core worker inventory differs")
    summary = validate_rvir_v4_whole_core_truth(
        cast(Mapping[str, Any], cores[0]), cast(Mapping[str, Any], posts[0])
    )
    if (
        solver.get("status") != "verified"
        or solver.get("success") is not True
        or solver.get("visited_domains") != [6]
    ):
        raise ValueError("RVIR-v4 whole-core solver accounting differs")
    summary["call_count"] = len(calls)
    summary["solver_status"] = solver["status"]
    summary["solver_success"] = solver["success"]
    summary["visited_domains"] = solver["visited_domains"]
    summary.pop("summary_hash")
    summary["summary_hash"] = _canonical_hash(summary)
    return summary


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "core_truth_hash": summary["core_truth_hash"],
        "post_truth_hash": summary["post_truth_hash"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# RVIR-v4 V4-3A Whole-Core Truth\n\n"
        "This raw-first artifact freezes the complete production core return, the "
        "pre-consumption KFSB lA inputs, three child-bound calls, final decision, "
        "post packet, and solver accounting. It is original truth, not replacement "
        "or performance evidence. Formal replay reruns the pinned external "
        "provider and compares every truth tensor at atol=rtol=2e-4 with exact "
        "shape, dtype, device, sign, discrete structure and inventory.\n"
    )


def _run_worker(
    *,
    benchmark: Path,
    abcrown: Path,
    python: Path,
    result: Path,
) -> dict[str, Any]:
    capture_runner._validate_inputs(benchmark, abcrown, python)
    command = (
        str(python),
        str(REPOSITORY_ROOT / "scripts/run_rvir_v4_production_state_capture.py"),
        "worker",
        "--benchmark-root",
        str(benchmark),
        "--abcrown-root",
        str(abcrown),
        "--model",
        str(benchmark / capture_runner.MODEL_RELATIVE_PATH),
        "--property",
        str(benchmark / capture_runner.PROPERTY_RELATIVE_PATH),
        "--result",
        str(result),
        "--whole-core-truth",
    )
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"RVIR-v4 whole-core worker failed:\n{completed.stdout}")
    return _load_truth(result)


def _truth_pair(
    truth: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    cores = truth.get("whole_core_truths")
    posts = truth.get("whole_post_truths")
    if (
        not isinstance(cores, list)
        or len(cores) != 1
        or not isinstance(cores[0], Mapping)
        or not isinstance(posts, list)
        or len(posts) != 1
        or not isinstance(posts[0], Mapping)
    ):
        raise ValueError("RVIR-v4 whole-core truth pair inventory differs")
    return cast(Mapping[str, Any], cores[0]), cast(Mapping[str, Any], posts[0])


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if not _code_paths_clean():
        raise ValueError("RVIR-v4 whole-core code paths must be clean")
    output = args.artifact_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    benchmark = args.benchmark_root.resolve()
    abcrown = args.abcrown_root.resolve()
    python = args.abcrown_python.resolve()
    truth = _run_worker(
        benchmark=benchmark,
        abcrown=abcrown,
        python=python,
        result=output / TRUTH_FILE,
    )
    summary = _summary(truth)
    _write_json(output / "summary.json", summary)
    result = _replay_result(summary)
    (output / "replay_stdout.txt").write_text(
        _canonical_json(result) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(_readme(), encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_git_head": _git_value("rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "files": {name: _file_sha256(output / name) for name in ARTIFACT_FILES},
        "abcrown_commit": capture_runner.ABCROWN_COMMIT,
        "auto_lirpa_commit": capture_runner.AUTO_LIRPA_COMMIT,
        "vnncomp_commit": capture_runner.VNNCOMP_COMMIT,
        "model_sha256": capture_runner.MODEL_SHA256,
        "property_sha256": capture_runner.PROPERTY_SHA256,
        "summary_hash": summary["summary_hash"],
        "status": summary["status"],
        "replay_contract": REPLAY_CONTRACT,
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = _canonical_hash(manifest)
    _write_json(output / "manifest.json", manifest)
    return result


def _replay(args: argparse.Namespace) -> dict[str, object]:
    artifact = args.artifact_dir.resolve()
    manifest = _load_json(artifact / "manifest.json")
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("manifest_hash") != _canonical_hash(semantic)
        or manifest.get("replay_contract") != REPLAY_CONTRACT
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("RVIR-v4 whole-core manifest differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("RVIR-v4 whole-core artifact inventory differs")
    for name in ARTIFACT_FILES:
        if files[name] != _file_sha256(artifact / name):
            raise ValueError(f"RVIR-v4 whole-core digest differs: {name}")
    summary = _summary(_load_truth(artifact / TRUTH_FILE))
    if _load_json(artifact / "summary.json") != summary:
        raise ValueError("RVIR-v4 whole-core semantic replay differs")
    if manifest.get("summary_hash") != summary["summary_hash"]:
        raise ValueError("RVIR-v4 whole-core summary identity differs")
    result = _replay_result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(result) + "\n"
    ):
        raise ValueError("RVIR-v4 whole-core replay stdout differs")
    if (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("RVIR-v4 whole-core README differs")
    frozen = _load_truth(artifact / TRUTH_FILE)
    with tempfile.TemporaryDirectory(prefix="boundflow-rvir-v4-whole-replay-") as raw:
        fresh = _run_worker(
            benchmark=args.benchmark_root.resolve(),
            abcrown=args.abcrown_root.resolve(),
            python=args.abcrown_python.resolve(),
            result=Path(raw) / "fresh-truth.pt",
        )
    fresh_summary = _summary(fresh)
    if (
        fresh_summary["call_count"] != summary["call_count"]
        or fresh_summary["solver_status"] != summary["solver_status"]
        or fresh_summary["solver_success"] != summary["solver_success"]
        or fresh_summary["visited_domains"] != summary["visited_domains"]
    ):
        raise ValueError("RVIR-v4 whole-core fresh solver accounting differs")
    frozen_core, frozen_post = _truth_pair(frozen)
    fresh_core, fresh_post = _truth_pair(fresh)
    parity = compare_rvir_v4_whole_core_truth(
        frozen_core, frozen_post, fresh_core, fresh_post
    )
    return {**result, "live_semantic_replay": parity}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--benchmark-root", type=Path, required=True)
    generate.add_argument("--abcrown-root", type=Path, required=True)
    generate.add_argument("--abcrown-python", type=Path, required=True)
    generate.add_argument("--artifact-dir", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    replay.add_argument("--benchmark-root", type=Path, required=True)
    replay.add_argument("--abcrown-root", type=Path, required=True)
    replay.add_argument("--abcrown-python", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Generate production truth or replay the frozen artifact."""

    args = _parse_args()
    result = _generate(args) if args.command == "generate" else _replay(args)
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
