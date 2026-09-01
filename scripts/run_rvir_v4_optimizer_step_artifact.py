#!/usr/bin/env python3
"""Generate or replay the RVIR-v4 V4-2B production optimizer-step artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=import-outside-toplevel,line-too-long,import-error
# pylint: disable=too-many-boolean-expressions,protected-access
# pylint: disable=duplicate-code
# pylint: disable=wrong-import-position

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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_rvir_v4_production_state_capture as production_capture

ARTIFACT_SCHEMA_VERSION = "boundflow.rvir-v4-optimizer-step-artifact/v1"
WORKER_CAPTURE_FILE = "production_capture.pt"
TRACE_FILE = "optimizer_step_trace.pt"
ARTIFACT_FILES = (
    WORKER_CAPTURE_FILE,
    TRACE_FILE,
    "trace.json",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)
CODE_PATHS = (
    "boundflow/runtime/rvir_v4_production_state.py",
    "boundflow/runtime/rvir_v4_optimizer_mutation.py",
    "scripts/run_rvir_v4_production_state_capture.py",
    "scripts/run_rvir_v4_optimizer_step_artifact.py",
)


def _repo_root() -> Path:
    return REPOSITORY_ROOT


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


def _git_value(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(_canonical_json(payload, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"RVIR-v4 JSON root differs: {path}")
    return payload


def _load_torch(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"RVIR-v4 torch payload root differs: {path}")
    return payload


def _code_revision() -> dict[str, str]:
    root = _repo_root()
    return {path: _file_sha256(root / path) for path in CODE_PATHS}


def _code_paths_clean() -> bool:
    return not _git_value(_repo_root(), "status", "--porcelain=v1", "--", *CODE_PATHS)


def _verify_code_provenance(manifest: Mapping[str, Any]) -> None:
    root = _repo_root()
    source_head = manifest.get("source_git_head")
    expected = manifest.get("code_revision")
    if not isinstance(source_head, str) or not isinstance(expected, Mapping):
        raise ValueError("RVIR-v4 optimizer artifact code provenance differs")
    if _git_value(root, "rev-parse", "HEAD") == source_head:
        observed = _code_revision()
    else:
        observed = {}
        for path in CODE_PATHS:
            blob = subprocess.run(
                ("git", "show", f"{source_head}:{path}"),
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
            observed[path] = hashlib.sha256(blob).hexdigest()
    if dict(expected) != observed:
        raise ValueError("RVIR-v4 optimizer artifact code revision differs")


def _is_cuda_device(value: str) -> bool:
    if value == "cuda":
        return True
    prefix, separator, ordinal = value.partition(":")
    return prefix == "cuda" and separator == ":" and ordinal.isdigit()


def validate_worker_capture(
    capture: Mapping[str, Any],
) -> tuple[dict[str, object], dict[str, object]]:
    """Replay the raw production worker capture and return trace/summary views."""

    from boundflow.runtime.rvir_v4_optimizer_mutation import (
        production_optimizer_step_trace_from_payload_v4,
    )

    expected_fields = {
        "schema_version",
        "source",
        "protocol",
        "solver_result",
        "calls",
        "cores",
        "optimizer_step_traces",
        "performance_claimed",
    }
    if (
        set(capture) != expected_fields
        or capture.get("schema_version")
        != production_capture.OPTIMIZER_WORKER_SCHEMA_VERSION
        or capture.get("performance_claimed") is not False
    ):
        raise ValueError("RVIR-v4 optimizer worker envelope differs")
    source = capture["source"]
    expected_source = {
        "abcrown_commit": production_capture.ABCROWN_COMMIT,
        "auto_lirpa_commit": production_capture.AUTO_LIRPA_COMMIT,
        "vnncomp_commit": production_capture.VNNCOMP_COMMIT,
        "model_relative_path": production_capture.MODEL_RELATIVE_PATH,
        "property_relative_path": production_capture.PROPERTY_RELATIVE_PATH,
        "model_sha256": production_capture.MODEL_SHA256,
        "property_sha256": production_capture.PROPERTY_SHA256,
    }
    if source != expected_source:
        raise ValueError("RVIR-v4 optimizer worker source identity differs")
    protocol = capture["protocol"]
    if not isinstance(protocol, Mapping) or protocol != {
        "device": "cuda",
        "seed": 100,
        "max_iterations": 1,
        "batch_size": 64,
        "alpha_steps": 5,
        "beta_steps": 10,
        "property_cache": "cold_isolated_copy",
        "performance_claimed": False,
    }:
        raise ValueError("RVIR-v4 optimizer worker protocol differs")
    calls_raw = capture["calls"]
    cores_raw = capture["cores"]
    traces_raw = capture["optimizer_step_traces"]
    if (
        not isinstance(calls_raw, list)
        or not isinstance(cores_raw, list)
        or not isinstance(traces_raw, list)
        or len(calls_raw) != 24
        or len(cores_raw) != 1
        or len(traces_raw) != 1
        or not isinstance(traces_raw[0], Mapping)
    ):
        raise ValueError("RVIR-v4 optimizer worker inventory differs")
    calls = cast(list[dict[str, Any]], calls_raw)
    cores = cast(list[dict[str, Any]], cores_raw)
    if [call.get("call_id") for call in calls] != list(range(24)):
        raise ValueError("RVIR-v4 optimizer worker call identities differ")
    phase_counts = {
        phase: sum(call.get("phase") == phase for call in calls)
        for phase in ("initial_crown", "alpha_optimize", "beta_split", "unclassified")
    }
    if phase_counts != {
        "initial_crown": 12,
        "alpha_optimize": 1,
        "beta_split": 11,
        "unclassified": 0,
    }:
        raise ValueError("RVIR-v4 optimizer worker call phases differ")

    trace = production_optimizer_step_trace_from_payload_v4(traces_raw[0])
    if any(
        not _is_cuda_device(tensor.source_device)
        for step in trace.steps
        for tensor in step.state_tensors
    ):
        raise ValueError("RVIR-v4 optimizer trace state is not CUDA production state")
    nested_calls = [
        call
        for call in calls
        if call.get("phase") == "beta_split"
        and call.get("depth") == 1
        and call.get("core_id") == 0
    ]
    if [step.call_id for step in trace.steps] != [
        call.get("call_id") for call in nested_calls
    ] or any(
        call.get("parent_call_id") != step.parent_call_id
        or call.get("bound_lower") is not True
        or call.get("bound_upper") is not False
        for call, step in zip(nested_calls, trace.steps)
    ):
        raise ValueError("RVIR-v4 optimizer trace/call lineage differs")
    for call, step in zip(nested_calls, trace.steps):
        if call.get("pre_state") != [
            tensor.metadata() for tensor in step.state_tensors
        ]:
            raise ValueError("RVIR-v4 optimizer trace/call state binding differs")
        result_rows = call.get("result_tensors")
        if not isinstance(result_rows, list):
            raise TypeError("RVIR-v4 optimizer call result rows differ")
        lower_rows = [
            row
            for row in result_rows
            if isinstance(row, Mapping) and row.get("path") == "result[0]"
        ]
        if len(lower_rows) != 1:
            raise ValueError("RVIR-v4 optimizer call lower inventory differs")
        lower_row = lower_rows[0]
        device = lower_row.get("device")
        if (
            lower_row.get("shape") != list(step.lower.shape)
            or lower_row.get("dtype") != str(step.lower.dtype)
            or lower_row.get("content_sha256") != step.lower_sha256
            or not isinstance(device, str)
            or not _is_cuda_device(device)
        ):
            raise ValueError("RVIR-v4 optimizer trace/call lower binding differs")
    core = cores[0]
    pre_snapshot = core.get("pre_snapshot")
    if (
        core.get("core_id") != 0
        or not isinstance(pre_snapshot, Mapping)
        or pre_snapshot.get("optimizer_policy")
        != trace.mutation_policy.production.to_dict()
    ):
        raise ValueError("RVIR-v4 optimizer trace/core policy binding differs")
    solver_result = capture["solver_result"]
    if not isinstance(solver_result, Mapping):
        raise TypeError("RVIR-v4 optimizer solver result differs")
    status = solver_result.get("status")
    success = solver_result.get("success")
    visited = solver_result.get("visited_domains")
    if (
        not isinstance(status, str)
        or not isinstance(success, bool)
        or not isinstance(visited, list)
        or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in visited
        )
    ):
        raise TypeError("RVIR-v4 optimizer solver result fields differ")

    trace_metadata = trace.metadata()
    mutable_changes = []
    for left, right in zip(trace.steps, trace.steps[1:]):
        mutable_changes.append(
            sum(
                left.tensor_map[path].content_sha256
                != right.tensor_map[path].content_sha256
                for path in left.tensor_map
                if left.tensor_map[path].ownership.value == "mutable_copy_out"
            )
        )
    summary: dict[str, object] = {
        "status": "validated-step-capture-schema",
        "workload_id": "cifar10_resnet:000",
        "solver_status": status,
        "call_count": len(calls),
        "phase_call_counts": phase_counts,
        "core_count": len(cores),
        "optimizer_step_trace_count": 1,
        "evaluation_count": len(trace.steps),
        "update_count": sum(step.update_after for step in trace.steps),
        "state_tensor_counts": [len(step.state_tensors) for step in trace.steps],
        "all_state_sources_cuda": True,
        "adjacent_mutable_change_counts": mutable_changes,
        "mutation_policy_hash": trace.mutation_policy.stable_hash(),
        "trace_hash": trace_metadata["trace_hash"],
        "optimizer_replacement_admitted": False,
        "b2_same_solver_timing_admitted": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = _canonical_hash(summary)
    return trace_metadata, summary


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "evaluation_count": summary["evaluation_count"],
        "update_count": summary["update_count"],
        "trace_hash": summary["trace_hash"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# RVIR-v4 V4-2B Production Optimizer Step Trace\n\n"
        "This artifact binds one production solver core to ten nested bound "
        "evaluations, nine observed Adam steps, the two learning-rate schedules, "
        "and all 24 alpha/SparseBeta tensors before every evaluation. Replay "
        "rebuilds the typed trace from raw tensors and fails closed on semantic "
        "tampering. It does not claim optimizer replacement or performance.\n"
    )


def _generate(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if not _code_paths_clean():
        raise ValueError("RVIR-v4 optimizer artifact code paths must be clean")
    artifact = args.artifact_dir.resolve()
    if artifact.exists() and any(artifact.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    benchmark_root = args.benchmark_root.resolve()
    abcrown_root = args.abcrown_root.resolve()
    abcrown_python = Path(os.path.abspath(args.abcrown_python))
    production_capture._validate_inputs(benchmark_root, abcrown_root, abcrown_python)
    with tempfile.TemporaryDirectory(prefix="boundflow-rvir-v4-step-") as temporary:
        worker_result = Path(temporary) / WORKER_CAPTURE_FILE
        completed = subprocess.run(
            (
                str(abcrown_python),
                str(_repo_root() / "scripts/run_rvir_v4_production_state_capture.py"),
                "worker",
                "--benchmark-root",
                str(benchmark_root),
                "--abcrown-root",
                str(abcrown_root),
                "--model",
                str(benchmark_root / production_capture.MODEL_RELATIVE_PATH),
                "--property",
                str(benchmark_root / production_capture.PROPERTY_RELATIVE_PATH),
                "--result",
                str(worker_result),
                "--optimizer-step-trace",
            ),
            cwd=_repo_root(),
            env=production_capture._external_env(),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0 or not worker_result.is_file():
            raise RuntimeError(
                f"RVIR-v4 optimizer capture worker failed: {completed.stdout[-12000:]}"
            )
        shutil.copy2(worker_result, artifact / WORKER_CAPTURE_FILE)
        print(completed.stdout.strip()[-3000:], flush=True)
    capture = _load_torch(artifact / WORKER_CAPTURE_FILE)
    trace_metadata, summary = validate_worker_capture(capture)
    traces = cast(list[dict[str, object]], capture["optimizer_step_traces"])
    torch.save(traces[0], artifact / TRACE_FILE)
    _write_json(artifact / "trace.json", trace_metadata)
    _write_json(artifact / "summary.json", summary)
    result = _replay_result(summary)
    (artifact / "replay_stdout.txt").write_text(
        _canonical_json(result) + "\n", encoding="utf-8"
    )
    (artifact / "README.md").write_text(_readme(), encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_git_head": _git_value(_repo_root(), "rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "files": {name: _file_sha256(artifact / name) for name in ARTIFACT_FILES},
        "trace_hash": trace_metadata["trace_hash"],
        "summary_hash": summary["summary_hash"],
        "status": summary["status"],
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = _canonical_hash(manifest)
    _write_json(artifact / "manifest.json", manifest)
    return result


def _replay(artifact: Path) -> dict[str, object]:
    from boundflow.runtime.rvir_v4_optimizer_mutation import (
        production_optimizer_step_trace_from_payload_v4,
    )

    manifest = _load_json(artifact / "manifest.json")
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("manifest_hash") != _canonical_hash(semantic)
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("RVIR-v4 optimizer artifact manifest differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("RVIR-v4 optimizer artifact file inventory differs")
    for name in ARTIFACT_FILES:
        if files[name] != _file_sha256(artifact / name):
            raise ValueError(f"RVIR-v4 optimizer artifact digest differs: {name}")
    trace_metadata, summary = validate_worker_capture(
        _load_torch(artifact / WORKER_CAPTURE_FILE)
    )
    trace = production_optimizer_step_trace_from_payload_v4(
        _load_torch(artifact / TRACE_FILE)
    )
    if (
        trace.metadata() != trace_metadata
        or _load_json(artifact / "trace.json") != trace_metadata
        or _load_json(artifact / "summary.json") != summary
        or manifest.get("trace_hash") != trace_metadata["trace_hash"]
        or manifest.get("summary_hash") != summary["summary_hash"]
    ):
        raise ValueError("RVIR-v4 optimizer artifact semantic replay differs")
    result = _replay_result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(result) + "\n"
    ):
        raise ValueError("RVIR-v4 optimizer artifact replay stdout differs")
    return result


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
    return parser.parse_args()


def main() -> None:
    """Run formal generation or deterministic artifact replay."""

    args = _parse_args()
    result = (
        _generate(args) if args.command == "generate" else _replay(args.artifact_dir)
    )
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
