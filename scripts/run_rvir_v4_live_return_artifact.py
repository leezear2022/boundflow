#!/usr/bin/env python3
"""Generate or replay the RVIR-v4 V4-3D live-return artifact."""

# pylint: disable=wrong-import-position,protected-access,duplicate-code
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions

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
from typing import Any, Mapping, cast

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.rvir_v4_whole_core_truth import (
    compare_rvir_v4_live_return_truth,
    validate_rvir_v4_live_return_result,
    validate_rvir_v4_whole_core_truth,
)
from scripts import run_rvir_v4_live_return_capture as live_runner
from scripts import run_rvir_v4_production_state_capture as capture_runner
from scripts.run_rvir_v4_pre_state_artifact import EXPECTED_IDENTITY

ARTIFACT_SCHEMA_VERSION = "boundflow.rvir-v4-live-return-artifact/v1"
REPLAY_CONTRACT = {
    "mode": "fresh-pinned-live-candidate-rerun",
    "atol": 2e-4,
    "rtol": 2e-4,
    "shape_dtype_device_exact": True,
    "sign_exact": True,
    "discrete_structure_exact": True,
    "native_device": "cuda:0",
    "official_post_queue_unmodified": True,
}
LIVE_RESULT_FILE = "live_result.pt"
SOURCE_TRUTH_FILE = "source_whole_core_truth.pt"
SOURCE_TRUTH_MANIFEST_FILE = "source_whole_core_manifest.json"
SOURCE_TRUTH_DIR = Path("artifacts/rvir-v4-whole-core-truth/resnet2b-core-v1")
SOURCE_TRUTH_SHA256 = "d0126427dcdc868d33c7a7ec6326bdb86c8fb6e624a16c650d46c401ecabd0e9"
SOURCE_TRUTH_MANIFEST_SHA256 = (
    "0e6ed721dbf796cf8923dd57e09636f05895a1a065595ea1154b170a4a0c9818"
)
ARTIFACT_FILES = (
    LIVE_RESULT_FILE,
    SOURCE_TRUTH_FILE,
    SOURCE_TRUTH_MANIFEST_FILE,
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)
CODE_PATHS = (
    "boundflow/runtime/rvir_v4_atomic_copy_out.py",
    "boundflow/runtime/rvir_v4_pre_state_initializer.py",
    "boundflow/runtime/rvir_v4_live_return.py",
    "boundflow/runtime/rvir_v4_whole_core_truth.py",
    "scripts/run_rvir_v4_live_return_capture.py",
    "scripts/run_rvir_v4_live_return_artifact.py",
    "scripts/probe_rvir_v4_live_return_artifact_tamper.py",
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
        raise TypeError(f"RVIR-v4 live-return JSON root differs: {path}")
    return value


def _load_torch(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"RVIR-v4 live-return torch root differs: {path}")
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
        raise ValueError("RVIR-v4 live-return code provenance differs")
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
        raise ValueError("RVIR-v4 live-return code revision differs")


def _result_pair(
    live: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    cores = live.get("whole_core_results")
    posts = live.get("whole_post_results")
    if (
        not isinstance(cores, list)
        or len(cores) != 1
        or not isinstance(cores[0], Mapping)
        or not isinstance(posts, list)
        or len(posts) != 1
        or not isinstance(posts[0], Mapping)
    ):
        raise ValueError("RVIR-v4 live-return result pair inventory differs")
    return cast(Mapping[str, Any], cores[0]), cast(Mapping[str, Any], posts[0])


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
        raise ValueError("RVIR-v4 live-return source truth pair differs")
    return cast(Mapping[str, Any], cores[0]), cast(Mapping[str, Any], posts[0])


def _validate_source_truth(truth: Mapping[str, Any]) -> None:
    if (
        truth.get("schema_version") != capture_runner.WHOLE_CORE_WORKER_SCHEMA_VERSION
        or truth.get("performance_claimed") is not False
    ):
        raise ValueError("RVIR-v4 live-return source truth schema differs")
    core, post = _truth_pair(truth)
    validate_rvir_v4_whole_core_truth(core, post)


def _tensor_source_device(record: object, *, label: str) -> str:
    if not isinstance(record, Mapping) or not isinstance(
        record.get("source_device"), str
    ):
        raise TypeError(f"RVIR-v4 live-return device record differs: {label}")
    return cast(str, record["source_device"])


def _validate_receipts(
    live: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    assemblies = live.get("assembly_metadata")
    receipts = live.get("commit_receipts")
    identities = live.get("pre_state_identities")
    if (
        not isinstance(assemblies, list)
        or len(assemblies) != 1
        or not isinstance(assemblies[0], Mapping)
        or not isinstance(receipts, list)
        or len(receipts) != 1
        or not isinstance(receipts[0], Mapping)
        or not isinstance(identities, list)
        or len(identities) != 1
        or not isinstance(identities[0], Mapping)
    ):
        raise ValueError("RVIR-v4 live-return receipt inventory differs")
    assembly = cast(Mapping[str, Any], assemblies[0])
    receipt = cast(Mapping[str, Any], receipts[0])
    identity = cast(Mapping[str, Any], identities[0])
    assembly_semantic = {
        key: value for key, value in assembly.items() if key != "assembly_hash"
    }
    receipt_semantic = {
        key: value for key, value in receipt.items() if key != "live_return_commit_hash"
    }
    commit_keys = {
        "live_copy_out_hash",
        "committed_path_count",
        "changed_path_count",
        "committed_paths",
        "host_packet_candidate_hash",
        "atomic_live_and_host_commit",
        "provider_callback_count",
        "fallback_dispatch_count",
        "performance_claimed",
    }
    commit_semantic = {key: receipt[key] for key in commit_keys if key in receipt}
    expected_decision = [[5, 27], [5, 32], [5, 90], [5, 90], [5, 32], [5, 90]]
    if (
        assembly.get("assembly_hash") != _canonical_hash(assembly_semantic)
        or receipt.get("live_return_commit_hash") != _canonical_hash(receipt_semantic)
        or set(commit_semantic) != commit_keys
        or receipt.get("commit_hash") != _canonical_hash(commit_semantic)
        or assembly.get("live_copy_out_hash") != receipt.get("live_copy_out_hash")
        or assembly.get("assembly_hash") != receipt.get("assembly_hash")
        or assembly.get("final_decision") != expected_decision
        or assembly.get("live_return_assembly_admitted") is not True
        or assembly.get("five_fresh_correctness_admitted") is not False
        or assembly.get("b2_same_solver_timing_admitted") is not False
        or receipt.get("atomic_live_and_host_commit") is not True
        or receipt.get("committed_path_count") != 12
        or receipt.get("changed_path_count") != 7
        or len(cast(list[Any], receipt.get("committed_paths", []))) != 12
        or identity.get("topology_hash") != EXPECTED_IDENTITY.topology_hash
        or any(
            live.get(name) != 0
            for name in (
                "provider_core_callback_count",
                "provider_compute_bounds_callback_count",
                "provider_update_bounds_callback_count",
                "fallback_dispatch_count",
            )
        )
        or any(
            assembly.get(name) != 0
            for name in (
                "provider_core_callback_count",
                "provider_compute_bounds_callback_count",
                "provider_update_bounds_callback_count",
                "fallback_dispatch_count",
            )
        )
        or any(
            receipt.get(name) != 0
            for name in (
                "provider_core_callback_count",
                "provider_compute_bounds_callback_count",
                "provider_update_bounds_callback_count",
                "fallback_dispatch_count",
            )
        )
        or live.get("performance_claimed") is not False
        or assembly.get("performance_claimed") is not False
        or receipt.get("performance_claimed") is not False
    ):
        raise ValueError("RVIR-v4 live-return atomic receipt differs")
    return assembly, receipt


def _structural_summary(live: Mapping[str, Any]) -> dict[str, object]:
    if live.get("schema_version") != live_runner.WORKER_SCHEMA:
        raise ValueError("RVIR-v4 live-return worker schema differs")
    source = live.get("source")
    protocol = live.get("protocol")
    solver = live.get("solver_result")
    if (
        not isinstance(source, Mapping)
        or dict(source)
        != {
            "abcrown_commit": capture_runner.ABCROWN_COMMIT,
            "auto_lirpa_commit": capture_runner.AUTO_LIRPA_COMMIT,
            "vnncomp_commit": capture_runner.VNNCOMP_COMMIT,
            "model_sha256": capture_runner.MODEL_SHA256,
            "property_sha256": capture_runner.PROPERTY_SHA256,
        }
        or not isinstance(protocol, Mapping)
        or dict(protocol)
        != {
            "device": "cuda",
            "seed": 100,
            "max_iterations": 1,
            "batch_size": 64,
            "alpha_steps": 5,
            "beta_steps": 10,
            "performance_claimed": False,
        }
        or not isinstance(solver, Mapping)
        or solver.get("status") != "verified"
        or solver.get("success") is not True
        or solver.get("visited_domains") != [6]
    ):
        raise ValueError("RVIR-v4 live-return source/protocol/accounting differs")
    core, post = _result_pair(live)
    result_summary = validate_rvir_v4_live_return_result(core, post)
    assembly, receipt = _validate_receipts(live)
    branch = cast(Mapping[str, Any], core["branch_trace"])
    inputs = cast(Mapping[str, Any], branch["input"])
    child_lowers = cast(list[Any], branch["candidate_child_lowers"])
    l_a_data = cast(Mapping[str, Any], cast(Mapping[str, Any], inputs["lAs"])["_data"])
    devices = {
        _tensor_source_device(
            cast(Mapping[str, Any], core["fields"])["lb"], label="lb"
        ),
        *(
            _tensor_source_device(value, label="candidate child lower")
            for value in child_lowers
        ),
        *(
            _tensor_source_device(value, label=f"lA {name}")
            for name, value in l_a_data.items()
        ),
    }
    if devices != {"cuda:0"}:
        raise ValueError("RVIR-v4 live-return native CUDA ownership differs")
    summary: dict[str, object] = {
        **result_summary,
        "solver_status": solver["status"],
        "solver_success": solver["success"],
        "visited_domains": solver["visited_domains"],
        "native_devices": sorted(devices),
        "committed_path_count": receipt["committed_path_count"],
        "changed_path_count": receipt["changed_path_count"],
        "atomic_live_and_host_commit": receipt["atomic_live_and_host_commit"],
        "assembly_hash": assembly["assembly_hash"],
        "commit_hash": receipt["live_return_commit_hash"],
        "provider_core_callback_count": 0,
        "provider_compute_bounds_callback_count": 0,
        "provider_update_bounds_callback_count": 0,
        "fallback_dispatch_count": 0,
        "official_post_queue_consumed": True,
        "five_fresh_correctness_admitted": False,
        "b2_same_solver_timing_admitted": False,
        "performance_claimed": False,
    }
    summary.pop("summary_hash", None)
    return summary


def _summary(live: Mapping[str, Any], truth: Mapping[str, Any]) -> dict[str, object]:
    summary = _structural_summary(live)
    truth_core, truth_post = _truth_pair(truth)
    live_core, live_post = _result_pair(live)
    parity = compare_rvir_v4_live_return_truth(
        truth_core, truth_post, live_core, live_post
    )
    summary["semantic_parity"] = parity
    summary["summary_hash"] = _canonical_hash(summary)
    return summary


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    parity = cast(Mapping[str, Any], summary["semantic_parity"])
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "summary_hash": summary["summary_hash"],
        "max_abs_diff": parity["max_abs_diff"],
        "tensor_count": parity["tensor_count"],
        "sign_element_count": parity["sign_element_count"],
        "provider_callbacks": [0, 0, 0],
        "fallback_dispatch_count": 0,
        "official_post_queue_consumed": True,
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# RVIR-v4 V4-3D Live Return\n\n"
        "This raw-first artifact records one BoundFlow-owned CUDA whole-core "
        "replacement consumed by the unmodified alpha-beta-CROWN post/queue path. "
        "It binds twelve live alpha/beta paths, the host packet, callback counters, "
        "complete core/post trees, solver accounting, source truth and code revision. "
        "It is correctness evidence only; five-fresh and B2 timing remain closed.\n"
    )


def _run_worker(
    *, benchmark: Path, abcrown: Path, python: Path, result: Path
) -> dict[str, Any]:
    capture_runner._validate_inputs(benchmark, abcrown, python)
    command = (
        str(python),
        str(REPOSITORY_ROOT / "scripts/run_rvir_v4_live_return_capture.py"),
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
        raise RuntimeError(f"RVIR-v4 live-return worker failed:\n{completed.stdout}")
    return _load_torch(result)


def _copy_source_truth(output: Path) -> dict[str, Any]:
    source = REPOSITORY_ROOT / SOURCE_TRUTH_DIR
    truth_path = source / "truth.pt"
    manifest_path = source / "manifest.json"
    if (
        _file_sha256(truth_path) != SOURCE_TRUTH_SHA256
        or _file_sha256(manifest_path) != SOURCE_TRUTH_MANIFEST_SHA256
    ):
        raise ValueError("RVIR-v4 live-return frozen source truth differs")
    shutil.copy2(truth_path, output / SOURCE_TRUTH_FILE)
    shutil.copy2(manifest_path, output / SOURCE_TRUTH_MANIFEST_FILE)
    truth = _load_torch(output / SOURCE_TRUTH_FILE)
    _validate_source_truth(truth)
    return truth


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if not _code_paths_clean():
        raise ValueError("RVIR-v4 live-return code paths must be clean")
    output = args.artifact_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    truth = _copy_source_truth(output)
    live = _run_worker(
        benchmark=args.benchmark_root.resolve(),
        abcrown=args.abcrown_root.resolve(),
        python=args.abcrown_python.expanduser().absolute(),
        result=output / LIVE_RESULT_FILE,
    )
    summary = _summary(live, truth)
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
        "source_truth_manifest_sha256": SOURCE_TRUTH_MANIFEST_SHA256,
        "summary_hash": summary["summary_hash"],
        "status": summary["status"],
        "replay_contract": REPLAY_CONTRACT,
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = _canonical_hash(manifest)
    _write_json(output / "manifest.json", manifest)
    return result


def _verify_static_artifact(
    artifact: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, object], dict[str, object]]:
    manifest = _load_json(artifact / "manifest.json")
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("manifest_hash") != _canonical_hash(semantic)
        or manifest.get("replay_contract") != REPLAY_CONTRACT
        or manifest.get("source_truth_manifest_sha256") != SOURCE_TRUTH_MANIFEST_SHA256
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("RVIR-v4 live-return manifest differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("RVIR-v4 live-return artifact inventory differs")
    for name in ARTIFACT_FILES:
        if files[name] != _file_sha256(artifact / name):
            raise ValueError(f"RVIR-v4 live-return digest differs: {name}")
    if (
        _file_sha256(artifact / SOURCE_TRUTH_FILE) != SOURCE_TRUTH_SHA256
        or _file_sha256(artifact / SOURCE_TRUTH_MANIFEST_FILE)
        != SOURCE_TRUTH_MANIFEST_SHA256
    ):
        raise ValueError("RVIR-v4 live-return source truth identity differs")
    truth = _load_torch(artifact / SOURCE_TRUTH_FILE)
    live = _load_torch(artifact / LIVE_RESULT_FILE)
    _validate_source_truth(truth)
    summary = _summary(live, truth)
    if _load_json(artifact / "summary.json") != summary:
        raise ValueError("RVIR-v4 live-return semantic replay differs")
    if (
        manifest.get("summary_hash") != summary["summary_hash"]
        or manifest.get("status") != summary["status"]
    ):
        raise ValueError("RVIR-v4 live-return summary identity differs")
    result = _replay_result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(result) + "\n"
    ):
        raise ValueError("RVIR-v4 live-return replay stdout differs")
    if (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("RVIR-v4 live-return README differs")
    return live, truth, summary, result


def _replay(args: argparse.Namespace) -> dict[str, object]:
    artifact = args.artifact_dir.resolve()
    frozen, truth, summary, result = _verify_static_artifact(artifact)
    with tempfile.TemporaryDirectory(prefix="boundflow-rvir-v4-live-replay-") as raw:
        fresh = _run_worker(
            benchmark=args.benchmark_root.resolve(),
            abcrown=args.abcrown_root.resolve(),
            python=args.abcrown_python.expanduser().absolute(),
            result=Path(raw) / "fresh-live.pt",
        )
    fresh_summary = _summary(fresh, truth)
    stable = (
        "solver_status",
        "solver_success",
        "visited_domains",
        "native_devices",
        "committed_path_count",
        "changed_path_count",
        "provider_core_callback_count",
        "provider_compute_bounds_callback_count",
        "provider_update_bounds_callback_count",
        "fallback_dispatch_count",
        "branching_decision",
    )
    if any(fresh_summary[name] != summary[name] for name in stable):
        raise ValueError("RVIR-v4 live-return fresh discrete parity differs")
    frozen_core, frozen_post = _result_pair(frozen)
    fresh_core, fresh_post = _result_pair(fresh)
    truth_core, truth_post = _truth_pair(truth)
    # Both runs must independently match the frozen provider truth. Comparing the
    # two candidate trees directly would make harmless fresh float drift the oracle.
    frozen_parity = compare_rvir_v4_live_return_truth(
        truth_core, truth_post, frozen_core, frozen_post
    )
    fresh_parity = compare_rvir_v4_live_return_truth(
        truth_core, truth_post, fresh_core, fresh_post
    )
    return {
        **result,
        "frozen_semantic_parity": frozen_parity,
        "fresh_semantic_parity": fresh_parity,
        "fresh_solver_status": fresh_summary["solver_status"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "replay"):
        command = commands.add_parser(name)
        command.add_argument("--artifact-dir", type=Path, required=True)
        command.add_argument("--benchmark-root", type=Path, required=True)
        command.add_argument("--abcrown-root", type=Path, required=True)
        command.add_argument("--abcrown-python", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Generate or replay the formal live-return artifact."""

    args = _parse_args()
    result = _generate(args) if args.command == "generate" else _replay(args)
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
