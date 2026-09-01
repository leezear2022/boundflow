#!/usr/bin/env python3
"""Generate or replay the RVIR-v4 V4-2D native optimizer artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
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
from boundflow.runtime.rvir_v4_native_optimizer import (
    compare_rvir_v4_native_optimizer_trace,
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
    ProductionTensorRole,
)
from boundflow.runtime.task_executor import InputSpec
from scripts import run_rvir_v4_pre_state_artifact as source_artifact_runner

ARTIFACT_SCHEMA_VERSION = "boundflow.rvir-v4-native-optimizer-artifact/v1"
SOURCE_CAPTURE_FILE = "source_capture.pt"
SOURCE_MANIFEST_FILE = "source_manifest.json"
ARTIFACT_FILES = (
    SOURCE_CAPTURE_FILE,
    SOURCE_MANIFEST_FILE,
    "topology.json",
    "native_trace.json",
    "parity.json",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)
SOURCE_ARTIFACT_MANIFEST_SHA256 = (
    "daee2fa0d150cab40262b76fc45fba970e35b7f8092b61d5b8bfe4ca6d660218"
)
SOURCE_CAPTURE_SHA256 = (
    "f42229dd06be8521865d14af4cc51450ec8bec4e792934b457c276aad50126dc"
)
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
CODE_PATHS = (
    "boundflow/runtime/rvir_v4_native_optimizer.py",
    "boundflow/runtime/rvir_v4_optimizer_mutation.py",
    "boundflow/runtime/rvir_v4_pre_state_initializer.py",
    "scripts/run_rvir_v4_native_optimizer_artifact.py",
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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(_canonical_json(payload, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"RVIR-v4 native optimizer JSON root differs: {path}")
    return payload


def _load_torch(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("RVIR-v4 native optimizer capture root differs")
    return payload


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
        raise ValueError("RVIR-v4 native optimizer code provenance differs")
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
        raise ValueError("RVIR-v4 native optimizer code revision differs")


def _one_role(snapshot: Any, role: ProductionTensorRole) -> torch.Tensor:
    values = [tensor.value for tensor in snapshot.tensors if tensor.role == role]
    if len(values) != 1:
        raise ValueError(f"RVIR-v4 native optimizer requires one {role.value}")
    return values[0]


def _topology_payload() -> dict[str, object]:
    rows = [item.to_dict() for item in source_artifact_runner.TOPOLOGY]
    return {"rows": rows, "topology_hash": _canonical_hash(rows)}


def _validate_source_manifest(manifest: Mapping[str, Any]) -> None:
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != source_artifact_runner.ARTIFACT_SCHEMA_VERSION
        or manifest.get("manifest_hash") != _canonical_hash(semantic)
        or manifest.get("performance_claimed") is not False
        or not isinstance(files, Mapping)
        or files.get(source_artifact_runner.SOURCE_CAPTURE_FILE)
        != SOURCE_CAPTURE_SHA256
    ):
        raise ValueError("RVIR-v4 native optimizer source manifest differs")


def _build_evidence_single_thread(
    capture: Mapping[str, Any], model_path: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    source_artifact_runner.source_artifact_runner.validate_worker_capture(capture)
    cores = capture.get("cores")
    traces = capture.get("optimizer_step_traces")
    if not isinstance(cores, list) or len(cores) != 1:
        raise ValueError("RVIR-v4 native optimizer source core inventory differs")
    if not isinstance(traces, list) or len(traces) != 1:
        raise ValueError("RVIR-v4 native optimizer trace inventory differs")
    if not isinstance(cores[0], Mapping) or not isinstance(traces[0], Mapping):
        raise TypeError("RVIR-v4 native optimizer source rows differ")
    pre_raw = cast(Mapping[str, Any], cores[0]).get("pre_snapshot")
    if not isinstance(pre_raw, Mapping):
        raise TypeError("RVIR-v4 native optimizer pre-snapshot differs")
    base = production_snapshot_from_payload_v4(pre_raw)
    production = production_optimizer_step_trace_from_payload_v4(
        cast(Mapping[str, object], traces[0])
    )
    mapping = initialize_rvir_v4_native_pre_state(
        base,
        source_artifact_runner.TOPOLOGY,
        expected_identity=source_artifact_runner.EXPECTED_IDENTITY,
    )
    if mapping.stable_hash() != source_artifact_runner.EXPECTED_MAPPING_HASH:
        raise ValueError("RVIR-v4 native optimizer source mapping differs")
    if _file_sha256(model_path) != MODEL_SHA256:
        raise ValueError("RVIR-v4 native optimizer model digest differs")
    program = import_onnx(str(model_path), do_shape_infer=True, normalize=True)
    if len(program.graph.inputs) != 1:
        raise ValueError("RVIR-v4 native optimizer model input count differs")
    module = plan_interval_ibp_v0(program)
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=_one_role(base, ProductionTensorRole.INPUT_LOWER),
        upper=_one_role(base, ProductionTensorRole.INPUT_UPPER),
    )
    objective = _one_role(base, ProductionTensorRole.LINEAR_SPEC)
    native_policy = production.mutation_policy.to_native_policy()
    scope = build_native_alpha_beta_scope(
        module,
        input_spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        relu_split_state=mapping.splits,
        policy=native_policy,
    )
    state = mapping.to_native_state(scope)
    native = execute_rvir_v4_native_optimizer_trace(
        module,
        input_spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        initial_state=state,
        mutation_policy=production.mutation_policy,
    )
    parity = compare_rvir_v4_native_optimizer_trace(
        native,
        production,
        base_snapshot=base,
        topology=source_artifact_runner.TOPOLOGY,
    )
    native_payload = native.metadata()
    parity_payload = parity.metadata()
    summary: dict[str, object] = {
        "status": "validated-native-step-parity",
        "workload_id": "cifar10_resnet:000",
        "evaluation_count": native_payload["evaluation_count"],
        "update_count": native_payload["update_count"],
        "provider_callback_count": native_payload["provider_callback_count"],
        "source_state_hash": native.source_state_hash,
        "native_trace_hash": native_payload["trace_hash"],
        "production_trace_hash": parity.production_trace_hash,
        "parity_hash": parity_payload["parity_hash"],
        "lower_maximum_absolute_difference": parity.lower_maximum_absolute_difference,
        "alpha_maximum_absolute_difference": parity.alpha_maximum_absolute_difference,
        "beta_maximum_absolute_difference": parity.beta_maximum_absolute_difference,
        "all_steps_allclose": all(row["allclose"] is True for row in parity.step_rows),
        "all_steps_sign_exact": all(
            row["sign_exact"] is True for row in parity.step_rows
        ),
        "atomic_copy_out_executed": False,
        "optimizer_replacement_admitted": False,
        "b2_same_solver_timing_admitted": False,
        "performance_claimed": False,
    }
    if (
        summary["evaluation_count"] != 10
        or summary["update_count"] != 9
        or summary["provider_callback_count"] != 0
        or summary["all_steps_allclose"] is not True
        or summary["all_steps_sign_exact"] is not True
    ):
        raise ValueError("RVIR-v4 native optimizer formal gate failed")
    summary["summary_hash"] = _canonical_hash(summary)
    return native_payload, parity_payload, summary


def _build_evidence(
    capture: Mapping[str, Any], model_path: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Build deterministic evidence in a fixed single-thread CPU reduction domain."""

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
        "native_trace_hash": summary["native_trace_hash"],
        "parity_hash": summary["parity_hash"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# RVIR-v4 V4-2D Native Optimizer Step Parity\n\n"
        "This artifact executes ten native lower evaluations and nine Adam updates "
        "from the V4-2C state with zero provider callbacks, then compares every lower, "
        "alpha, and beta step to the frozen production trace. It does not perform "
        "atomic copy-out or claim performance.\n"
    )


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if not _code_paths_clean():
        raise ValueError("RVIR-v4 native optimizer code paths must be clean")
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
        raise ValueError("RVIR-v4 native optimizer source artifact digest differs")
    source_artifact_runner._replay(source, model)
    shutil.copy2(
        source / source_artifact_runner.SOURCE_CAPTURE_FILE,
        output / SOURCE_CAPTURE_FILE,
    )
    shutil.copy2(source / "manifest.json", output / SOURCE_MANIFEST_FILE)
    native, parity, summary = _build_evidence(
        _load_torch(output / SOURCE_CAPTURE_FILE), model
    )
    _write_json(output / "topology.json", _topology_payload())
    _write_json(output / "native_trace.json", native)
    _write_json(output / "parity.json", parity)
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
        "native_trace_hash": summary["native_trace_hash"],
        "parity_hash": summary["parity_hash"],
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
        raise ValueError("RVIR-v4 native optimizer manifest differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("RVIR-v4 native optimizer artifact inventory differs")
    for name in ARTIFACT_FILES:
        if files[name] != _file_sha256(artifact / name):
            raise ValueError(
                f"RVIR-v4 native optimizer artifact digest differs: {name}"
            )
    if (
        _file_sha256(artifact / SOURCE_MANIFEST_FILE) != SOURCE_ARTIFACT_MANIFEST_SHA256
        or _file_sha256(artifact / SOURCE_CAPTURE_FILE) != SOURCE_CAPTURE_SHA256
    ):
        raise ValueError("RVIR-v4 native optimizer frozen source differs")
    _validate_source_manifest(_load_json(artifact / SOURCE_MANIFEST_FILE))
    if _load_json(artifact / "topology.json") != _topology_payload():
        raise ValueError("RVIR-v4 native optimizer topology differs")
    native, parity, summary = _build_evidence(
        _load_torch(artifact / SOURCE_CAPTURE_FILE), model
    )
    if (
        _load_json(artifact / "native_trace.json") != native
        or _load_json(artifact / "parity.json") != parity
        or _load_json(artifact / "summary.json") != summary
    ):
        raise ValueError("RVIR-v4 native optimizer semantic replay differs")
    if (
        manifest.get("native_trace_hash") != summary["native_trace_hash"]
        or manifest.get("parity_hash") != summary["parity_hash"]
        or manifest.get("summary_hash") != summary["summary_hash"]
    ):
        raise ValueError("RVIR-v4 native optimizer semantic identity differs")
    result = _replay_result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(result) + "\n"
    ):
        raise ValueError("RVIR-v4 native optimizer replay stdout differs")
    if (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("RVIR-v4 native optimizer README differs")
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
    """Run formal V4-2D generation or deterministic semantic replay."""

    args = _parse_args()
    result = (
        _generate(args)
        if args.command == "generate"
        else _replay(args.artifact_dir.resolve(), args.model.resolve())
    )
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
