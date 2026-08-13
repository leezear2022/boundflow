#!/usr/bin/env python3
"""Generate or replay the RVIR-v4 V4-2E atomic copy-out artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions
# pylint: disable=protected-access,duplicate-code,wrong-import-position

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, cast

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.native_alpha_beta_optimization_state import (
    build_native_alpha_beta_scope,
)
from boundflow.runtime.rvir_v4_atomic_copy_out import (
    commit_rvir_v4_atomic_copy_out,
    stage_rvir_v4_atomic_copy_out,
)
from boundflow.runtime.rvir_v4_native_optimizer import (
    execute_rvir_v4_native_optimizer_trace,
)
from boundflow.runtime.rvir_v4_optimizer_mutation import (
    production_optimizer_step_trace_from_payload_v4,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
    ProductionTensorOwnership,
    ProductionTensorRole,
)
from boundflow.runtime.task_executor import InputSpec
from scripts import run_rvir_v4_native_optimizer_artifact as source_artifact_runner

ARTIFACT_SCHEMA_VERSION = "boundflow.rvir-v4-atomic-copy-out-artifact/v1"
SOURCE_CAPTURE_FILE = "source_capture.pt"
SOURCE_MANIFEST_FILE = "source_manifest.json"
ARTIFACT_FILES = (
    SOURCE_CAPTURE_FILE,
    SOURCE_MANIFEST_FILE,
    "topology.json",
    "copy_out.json",
    "commit.json",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)
SOURCE_ARTIFACT_MANIFEST_SHA256 = (
    "0b4ae1a88294f48a2c4fcf0b085f8f3374047d856d24624683d0f73160388493"
)
SOURCE_CAPTURE_SHA256 = (
    "f42229dd06be8521865d14af4cc51450ec8bec4e792934b457c276aad50126dc"
)
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
CODE_PATHS = (
    "boundflow/runtime/rvir_v4_atomic_copy_out.py",
    "boundflow/runtime/rvir_v4_native_optimizer.py",
    "scripts/run_rvir_v4_atomic_copy_out_artifact.py",
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
        raise TypeError(f"RVIR-v4 atomic copy-out JSON root differs: {path}")
    return value


def _load_torch(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("RVIR-v4 atomic copy-out capture root differs")
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
        raise ValueError("RVIR-v4 atomic copy-out code provenance differs")
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
        raise ValueError("RVIR-v4 atomic copy-out code revision differs")


def _one_role(snapshot: Any, role: ProductionTensorRole) -> torch.Tensor:
    values = [tensor.value for tensor in snapshot.tensors if tensor.role == role]
    if len(values) != 1:
        raise ValueError(f"RVIR-v4 atomic copy-out requires one {role.value}")
    return values[0]


def _topology_payload() -> dict[str, object]:
    topology = source_artifact_runner.source_artifact_runner.TOPOLOGY
    rows = [item.to_dict() for item in topology]
    return {"rows": rows, "topology_hash": _canonical_hash(rows)}


def _build_evidence_single_thread(
    capture: Mapping[str, Any], model_path: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    source_artifact_runner.source_artifact_runner.source_artifact_runner.validate_worker_capture(
        capture
    )
    cores = capture.get("cores")
    traces = capture.get("optimizer_step_traces")
    if not isinstance(cores, list) or len(cores) != 1:
        raise ValueError("RVIR-v4 atomic copy-out core inventory differs")
    if not isinstance(traces, list) or len(traces) != 1:
        raise ValueError("RVIR-v4 atomic copy-out trace inventory differs")
    if not isinstance(cores[0], Mapping) or not isinstance(traces[0], Mapping):
        raise TypeError("RVIR-v4 atomic copy-out source rows differ")
    core = cast(Mapping[str, Any], cores[0])
    pre_raw = core.get("pre_snapshot")
    post_raw = core.get("post_snapshot")
    if not isinstance(pre_raw, Mapping) or not isinstance(post_raw, Mapping):
        raise TypeError("RVIR-v4 atomic copy-out snapshots differ")
    pre = production_snapshot_from_payload_v4(pre_raw)
    post = production_snapshot_from_payload_v4(post_raw)
    production = production_optimizer_step_trace_from_payload_v4(
        cast(Mapping[str, object], traces[0])
    )
    topology = source_artifact_runner.source_artifact_runner.TOPOLOGY
    mapping = initialize_rvir_v4_native_pre_state(
        pre,
        topology,
        expected_identity=source_artifact_runner.source_artifact_runner.EXPECTED_IDENTITY,
    )
    if _file_sha256(model_path) != MODEL_SHA256:
        raise ValueError("RVIR-v4 atomic copy-out model digest differs")
    program = import_onnx(str(model_path), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=_one_role(pre, ProductionTensorRole.INPUT_LOWER),
        upper=_one_role(pre, ProductionTensorRole.INPUT_UPPER),
    )
    objective = _one_role(pre, ProductionTensorRole.LINEAR_SPEC)
    policy = production.mutation_policy.to_native_policy()
    scope = build_native_alpha_beta_scope(
        module,
        input_spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        relu_split_state=mapping.splits,
        policy=policy,
    )
    initial = mapping.to_native_state(scope)
    native = execute_rvir_v4_native_optimizer_trace(
        module,
        input_spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        initial_state=initial,
        mutation_policy=production.mutation_policy,
    )
    terminal = type(initial)(
        scope=initial.scope,
        split_by_relu_input=initial.split_by_relu_input,
        alpha_by_relu_input=native.steps[-1].alpha_by_relu_input,
        beta_by_relu_input=native.steps[-1].beta_by_relu_input,
    )
    staged = stage_rvir_v4_atomic_copy_out(
        pre=pre,
        terminal_state=terminal,
        topology=topology,
        expected_post=post,
        terminal_lower=native.steps[-1].lower,
        expected_lower=production.steps[-1].lower,
        candidate_snapshot_id="core:000000:native-candidate",
    )
    live = {
        path: tensor.value.clone()
        for path, tensor in pre.tensor_map().items()
        if tensor.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
    }
    commit = commit_rvir_v4_atomic_copy_out(staged, pre=pre, live_targets=live)
    copy_out = staged.metadata()
    changed_path_count = sum(
        receipt.before_sha256 != receipt.candidate_sha256
        for receipt in staged.path_receipts
    )
    expected_changed_path_count = sum(
        receipt.before_sha256 != receipt.expected_sha256
        for receipt in staged.path_receipts
    )
    summary: dict[str, object] = {
        "status": "validated-atomic-copy-out",
        "workload_id": "cifar10_resnet:000",
        "core_count": len(cores),
        "domain_count": int(objective.shape[0]),
        "topology_count": len(topology),
        "evaluation_count": len(native.steps),
        "update_count": sum(step.update_after for step in native.steps),
        "staged_path_count": len(staged.path_receipts),
        "committed_path_count": commit["committed_path_count"],
        "changed_path_count": changed_path_count,
        "expected_changed_path_count": expected_changed_path_count,
        "copy_out_hash": copy_out["copy_out_hash"],
        "commit_hash": commit["commit_hash"],
        "alpha_maximum_absolute_difference": max(
            receipt.maximum_absolute_difference
            for receipt in staged.path_receipts
            if receipt.role == ProductionTensorRole.ALPHA
        ),
        "beta_maximum_absolute_difference": max(
            receipt.maximum_absolute_difference
            for receipt in staged.path_receipts
            if receipt.role == ProductionTensorRole.BETA_VALUE
        ),
        "lower_maximum_absolute_difference": staged.lower_maximum_absolute_difference,
        "all_sign_exact": all(receipt.sign_exact for receipt in staged.path_receipts)
        and staged.lower_sign_exact,
        "atomic_commit": commit["atomic_commit"],
        "provider_callback_count": commit["provider_callback_count"],
        "fallback_dispatch_count": commit["fallback_dispatch_count"],
        "optimizer_replacement_admitted": True,
        "b2_same_solver_timing_admitted": False,
        "performance_claimed": False,
    }
    if (
        summary["core_count"] != 1
        or summary["domain_count"] != 6
        or summary["topology_count"] != 6
        or summary["evaluation_count"] != 10
        or summary["update_count"] != 9
        or summary["staged_path_count"] != 12
        or summary["committed_path_count"] != 12
        or summary["changed_path_count"] != 7
        or summary["expected_changed_path_count"] != 7
        or summary["atomic_commit"] is not True
        or summary["all_sign_exact"] is not True
        or summary["provider_callback_count"] != 0
        or summary["fallback_dispatch_count"] != 0
    ):
        raise ValueError("RVIR-v4 atomic copy-out formal gate failed")
    summary["summary_hash"] = _canonical_hash(summary)
    return copy_out, commit, summary


def _build_evidence(
    capture: Mapping[str, Any], model_path: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _build_evidence_single_thread(capture, model_path)
    finally:
        torch.set_num_threads(previous_threads)


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "copy_out_hash": summary["copy_out_hash"],
        "commit_hash": summary["commit_hash"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# RVIR-v4 V4-2E Atomic Copy-Out\n\n"
        "This artifact re-executes the native optimizer, stages all twelve mutable "
        "production paths privately, validates terminal state and lower parity, and "
        "commits to isolated live clones atomically. B2 timing remains disabled.\n"
    )


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if not _code_paths_clean():
        raise ValueError("RVIR-v4 atomic copy-out code paths must be clean")
    source = args.source_artifact.resolve()
    model = args.model.resolve()
    output = args.artifact_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if (
        _file_sha256(source / "manifest.json") != SOURCE_ARTIFACT_MANIFEST_SHA256
        or _file_sha256(source / source_artifact_runner.SOURCE_CAPTURE_FILE)
        != SOURCE_CAPTURE_SHA256
    ):
        raise ValueError("RVIR-v4 atomic copy-out source artifact digest differs")
    source_artifact_runner._replay(source, model)
    shutil.copy2(
        source / source_artifact_runner.SOURCE_CAPTURE_FILE,
        output / SOURCE_CAPTURE_FILE,
    )
    shutil.copy2(source / "manifest.json", output / SOURCE_MANIFEST_FILE)
    copy_out, commit, summary = _build_evidence(
        _load_torch(output / SOURCE_CAPTURE_FILE), model
    )
    _write_json(output / "topology.json", _topology_payload())
    _write_json(output / "copy_out.json", copy_out)
    _write_json(output / "commit.json", commit)
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
        "copy_out_hash": summary["copy_out_hash"],
        "commit_hash": summary["commit_hash"],
        "summary_hash": summary["summary_hash"],
        "status": summary["status"],
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = _canonical_hash(manifest)
    _write_json(output / "manifest.json", manifest)
    return result


def _replay(artifact: Path, model: Path) -> dict[str, object]:
    manifest = _load_json(artifact / "manifest.json")
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("manifest_hash") != _canonical_hash(semantic)
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("RVIR-v4 atomic copy-out manifest differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("RVIR-v4 atomic copy-out artifact inventory differs")
    for name in ARTIFACT_FILES:
        if files[name] != _file_sha256(artifact / name):
            raise ValueError(f"RVIR-v4 atomic copy-out digest differs: {name}")
    if (
        _file_sha256(artifact / SOURCE_MANIFEST_FILE) != SOURCE_ARTIFACT_MANIFEST_SHA256
        or _file_sha256(artifact / SOURCE_CAPTURE_FILE) != SOURCE_CAPTURE_SHA256
    ):
        raise ValueError("RVIR-v4 atomic copy-out frozen source differs")
    if _load_json(artifact / "topology.json") != _topology_payload():
        raise ValueError("RVIR-v4 atomic copy-out topology differs")
    copy_out, commit, summary = _build_evidence(
        _load_torch(artifact / SOURCE_CAPTURE_FILE), model
    )
    if (
        _load_json(artifact / "copy_out.json") != copy_out
        or _load_json(artifact / "commit.json") != commit
        or _load_json(artifact / "summary.json") != summary
    ):
        raise ValueError("RVIR-v4 atomic copy-out semantic replay differs")
    if (
        manifest.get("copy_out_hash") != summary["copy_out_hash"]
        or manifest.get("commit_hash") != summary["commit_hash"]
        or manifest.get("summary_hash") != summary["summary_hash"]
    ):
        raise ValueError("RVIR-v4 atomic copy-out semantic identity differs")
    result = _replay_result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(result) + "\n"
    ):
        raise ValueError("RVIR-v4 atomic copy-out replay stdout differs")
    if (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("RVIR-v4 atomic copy-out README differs")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--source-artifact", type=Path, required=True)
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--artifact-dir", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run formal V4-2E generation or deterministic semantic replay."""

    args = _parse_args()
    result = (
        _generate(args)
        if args.command == "generate"
        else _replay(args.artifact_dir.resolve(), args.model.resolve())
    )
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
