#!/usr/bin/env python3
"""Generate or replay the MR3 five-pair production bridge artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=missing-function-docstring,line-too-long
# pylint: disable=wrong-import-position,too-many-arguments

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.mr3_production_bridge_formal import (  # noqa: E402
    EXPECTED_RUNS,
    FORMAL_SCHEMA,
    SOURCE_COMMIT,
    canonical_hash,
    derive_summary,
)

MODEL_RELATIVE_PATH = "benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
PROPERTY_RELATIVE_PATH = (
    "benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/"
    "resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"
)
CODE_PATHS = (
    "boundflow/runtime/fsg4_b4b3_cibc_dense_tir.py",
    "boundflow/runtime/mr3_production_p_anchor_bridge.py",
    "boundflow/runtime/mr3_production_bridge_formal.py",
    "scripts/run_mr3_production_p_anchor_bridge_worker.py",
    "scripts/run_mr3_production_bridge_formal.py",
    "scripts/probe_mr3_production_bridge_tamper.py",
    "tests/test_mr3_production_p_anchor_bridge.py",
    "tests/test_mr3_production_bridge_formal.py",
)


def _json_text(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(_json_text(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


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


def _code_revision() -> dict[str, str]:
    return {path: _sha256(ROOT / path) for path in CODE_PATHS}


def _validate_code_revision(value: object) -> None:
    if value != _code_revision():
        raise ValueError("MR3 formal code revision differs")


def _worker_command(
    *,
    python: Path,
    benchmark_root: Path,
    abcrown_root: Path,
    mode: str,
    result_json: Path,
    inject_failure_evaluation: int | None = None,
) -> list[str]:
    command = [
        str(python),
        str(ROOT / "scripts/run_mr3_production_p_anchor_bridge_worker.py"),
        "--benchmark-root",
        str(benchmark_root),
        "--abcrown-root",
        str(abcrown_root),
        "--model",
        str(benchmark_root / MODEL_RELATIVE_PATH),
        "--property",
        str(benchmark_root / PROPERTY_RELATIVE_PATH),
        "--mode",
        mode,
        "--result-json",
        str(result_json),
    ]
    if inject_failure_evaluation is not None:
        command.extend(["--inject-failure-evaluation", str(inject_failure_evaluation)])
    return command


def _worker_env() -> dict[str, str]:
    environment = dict(os.environ)
    tvm_python = ROOT / "boundflow/3rdparty/tvm/python"
    tvm_build = ROOT / "boundflow/3rdparty/tvm/build-boundflow"
    old_python = environment.get("PYTHONPATH")
    old_library = environment.get("LD_LIBRARY_PATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(tvm_python),
            str(ROOT),
            *(old_python.split(os.pathsep) if old_python else []),
        ]
    )
    environment["TVM_LIBRARY_PATH"] = str(tvm_build)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [
            str(tvm_build),
            str(tvm_build / "lib"),
            *(old_library.split(os.pathsep) if old_library else []),
        ]
    )
    environment["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
    return environment


def _run_worker(command: list[str], *, environment: dict[str, str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MR3 formal worker failed with {result.returncode}:\n{result.stdout[-4000:]}"
        )
    result_path = Path(command[command.index("--result-json") + 1])
    return _load_json(result_path)


def replay_artifact(artifact: Path) -> dict[str, object]:
    manifest = _load_json(artifact / "manifest.json")
    if manifest.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("MR3 formal manifest source differs")
    _validate_code_revision(manifest.get("code_revision"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("MR3 formal manifest files are absent")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TypeError("MR3 formal manifest file entry differs")
        if _sha256(artifact / relative) != expected:
            raise ValueError(f"MR3 formal artifact digest differs: {relative}")
    raw = _load_json(artifact / "raw.json")
    summary = derive_summary(raw)
    if summary != _load_json(artifact / "summary.json"):
        raise ValueError("MR3 formal summary replay differs")
    return summary


def _write_manifest(artifact: Path, code_revision: dict[str, str]) -> None:
    files = {
        path.relative_to(artifact).as_posix(): _sha256(path)
        for path in sorted(artifact.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest: dict[str, object] = {
        "schema_version": FORMAL_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "code_revision": code_revision,
        "files": files,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact / "manifest.json", manifest)


def generate_artifact(args: argparse.Namespace) -> dict[str, object]:
    if args.artifact.exists():
        raise FileExistsError(f"artifact already exists: {args.artifact}")
    subprocess.run(
        ("git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if _git("status", "--porcelain=v1", "--", *CODE_PATHS):
        raise ValueError("MR3 formal code paths are dirty")
    code_revision = _code_revision()
    environment = _worker_env()
    with tempfile.TemporaryDirectory(prefix="boundflow-mr3-formal-") as temporary:
        temporary_root = Path(temporary)
        runs: list[dict[str, object]] = []
        for ordinal, (pair_index, position, mode) in enumerate(EXPECTED_RUNS):
            result_path = temporary_root / f"run-{ordinal:02d}.json"
            worker = _run_worker(
                _worker_command(
                    python=args.python,
                    benchmark_root=args.benchmark_root,
                    abcrown_root=args.abcrown_root,
                    mode=mode,
                    result_json=result_path,
                ),
                environment=environment,
            )
            runs.append(
                {
                    "pair_index": pair_index,
                    "position": position,
                    "mode": mode,
                    "worker": worker,
                }
            )
        rollback_path = temporary_root / "rollback.json"
        rollback = _run_worker(
            _worker_command(
                python=args.python,
                benchmark_root=args.benchmark_root,
                abcrown_root=args.abcrown_root,
                mode="bridge",
                result_json=rollback_path,
                inject_failure_evaluation=5,
            ),
            environment=environment,
        )
        raw: dict[str, object] = {
            "schema_version": FORMAL_SCHEMA,
            "source_commit": SOURCE_COMMIT,
            "run_order": [list(run) for run in EXPECTED_RUNS],
            "runs": runs,
            "rollback_probe": rollback,
            "timing_recorded": False,
            "performance_claimed": False,
        }
        raw["raw_hash"] = canonical_hash(raw)
        summary = derive_summary(raw)
        staging = args.artifact.parent / f".{args.artifact.name}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        _write_json(staging / "raw.json", raw)
        _write_json(staging / "summary.json", summary)
        (staging / "README.md").write_text(
            "# MR3 P-anchor Production Bridge Formal Artifact\n\n"
            "Five fresh provider/bridge pairs plus one mutation-after-failure atomic rollback probe.\n"
            "No timing is recorded and no performance claim is made.\n",
            encoding="utf-8",
        )
        _write_manifest(staging, code_revision)
        replayed = replay_artifact(staging)
        (staging / "replay_stdout.txt").write_text(
            f"status={replayed['status']}\nsummary_hash={replayed['summary_hash']}\n",
            encoding="utf-8",
        )
        _write_manifest(staging, code_revision)
        staging.rename(args.artifact)
    return replay_artifact(args.artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--abcrown-root", type=Path)
    args = parser.parse_args()
    if args.replay:
        summary = replay_artifact(args.artifact)
    else:
        if (
            args.python is None
            or args.benchmark_root is None
            or args.abcrown_root is None
        ):
            parser.error(
                "generation requires --python, --benchmark-root, and --abcrown-root"
            )
        summary = generate_artifact(args)
    print(f"status={summary['status']}")
    print(f"summary_hash={summary['summary_hash']}")


if __name__ == "__main__":
    main()
