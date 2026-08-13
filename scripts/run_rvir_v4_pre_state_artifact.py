#!/usr/bin/env python3
"""Generate or replay the RVIR-v4 V4-2C native pre-state artifact."""

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
from boundflow.runtime.rvir_v4_optimizer_mutation import (
    production_optimizer_step_trace_from_payload_v4,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
    ProductionPreStateIdentityV4,
    ProductionReluTopologyV4,
)
from boundflow.runtime.rvir_v4_production_state import (
    ProductionTensorRole,
    production_snapshot_from_payload_v4,
)
from boundflow.runtime.task_executor import InputSpec
from scripts import run_rvir_v4_optimizer_step_artifact as source_artifact_runner

ARTIFACT_SCHEMA_VERSION = "boundflow.rvir-v4-pre-state-artifact/v1"
SOURCE_CAPTURE_FILE = "source_capture.pt"
SOURCE_MANIFEST_FILE = "source_manifest.json"
ARTIFACT_FILES = (
    SOURCE_CAPTURE_FILE,
    SOURCE_MANIFEST_FILE,
    "topology.json",
    "mapping.json",
    "native_state.json",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)
SOURCE_MANIFEST_SHA256 = (
    "7d7745e40a901c4f3a420188c42efa2f487248d2da992ca0c08730beb612fbe6"
)
SOURCE_CAPTURE_SHA256 = (
    "f42229dd06be8521865d14af4cc51450ec8bec4e792934b457c276aad50126dc"
)
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
EXPECTED_IDENTITY = ProductionPreStateIdentityV4(
    snapshot_hash="2a775b66559c20ddfc0bec97ec026898ba5eccfc984e02b217fcb7472d03a256",
    topology_hash="9be361625e492b1401a402fd19ad5d80ac06a977c74f137c7563e96de06bca35",
    history_hash="8921a052baa3a1444c468851f9a8be6429b23830982a61ee285b2cb2b115a08a",
    intermediate_bounds_hash=(
        "f82523fb83031f5d0699dc5ff15078a7b6be1c0ca03511f2d53093721288cf06"
    ),
)
EXPECTED_MAPPING_HASH = (
    "cfcebf92fc58c269899d98cd65cc9454d7caa6051e2c9da46d415eda1fecf8df"
)
TOPOLOGY = tuple(
    ProductionReluTopologyV4(*values, provider_start_node="/49")
    for values in (
        ("/input-4", "/input", "17"),
        ("/input-12", "/input-8", "19"),
        ("/input-16", "/39", "23"),
        ("/input-24", "/input-20", "25"),
        ("/45", "/44", "28"),
        ("/48", "/input-28", "31"),
    )
)
CODE_PATHS = (
    "boundflow/runtime/rvir_v4_pre_state_initializer.py",
    "boundflow/runtime/rvir_v4_optimizer_mutation.py",
    "scripts/run_rvir_v4_optimizer_step_artifact.py",
    "scripts/run_rvir_v4_pre_state_artifact.py",
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
        raise TypeError(f"RVIR-v4 pre-state JSON root differs: {path}")
    return payload


def _load_torch(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("RVIR-v4 pre-state source capture root differs")
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
        raise ValueError("RVIR-v4 pre-state code provenance differs")
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
        raise ValueError("RVIR-v4 pre-state code revision differs")


def _one_role(snapshot: Any, role: ProductionTensorRole) -> torch.Tensor:
    values = [tensor.value for tensor in snapshot.tensors if tensor.role == role]
    if len(values) != 1:
        raise ValueError(f"RVIR-v4 pre-state artifact requires one {role.value}")
    return values[0]


def _topology_payload() -> dict[str, object]:
    rows = [item.to_dict() for item in TOPOLOGY]
    return {"rows": rows, "topology_hash": _canonical_hash(rows)}


def _validate_source_manifest(manifest: Mapping[str, Any]) -> None:
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != source_artifact_runner.ARTIFACT_SCHEMA_VERSION
        or manifest.get("manifest_hash") != _canonical_hash(semantic)
        or manifest.get("performance_claimed") is not False
        or not isinstance(files, Mapping)
        or files.get(source_artifact_runner.WORKER_CAPTURE_FILE)
        != SOURCE_CAPTURE_SHA256
    ):
        raise ValueError("RVIR-v4 pre-state source manifest differs")


def _build_evidence(
    capture: Mapping[str, Any], model_path: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    source_artifact_runner.validate_worker_capture(capture)
    cores = capture.get("cores")
    traces = capture.get("optimizer_step_traces")
    if not isinstance(cores, list) or len(cores) != 1:
        raise ValueError("RVIR-v4 pre-state source inventory differs")
    if not isinstance(traces, list) or len(traces) != 1:
        raise ValueError("RVIR-v4 pre-state source inventory differs")
    if not isinstance(cores[0], Mapping) or not isinstance(traces[0], Mapping):
        raise ValueError("RVIR-v4 pre-state source inventory differs")
    core = cast(Mapping[str, Any], cores[0])
    pre_raw = core.get("pre_snapshot")
    if not isinstance(pre_raw, Mapping):
        raise TypeError("RVIR-v4 pre-state snapshot payload differs")
    snapshot = production_snapshot_from_payload_v4(pre_raw)
    trace = production_optimizer_step_trace_from_payload_v4(
        cast(Mapping[str, object], traces[0])
    )
    step_zero = {
        tensor.semantic_path: tensor.content_sha256
        for tensor in trace.steps[0].state_tensors
    }
    mutable = {
        tensor.semantic_path: tensor.content_sha256
        for tensor in snapshot.tensors
        if tensor.role in {ProductionTensorRole.ALPHA, ProductionTensorRole.BETA_VALUE}
    }
    if len(mutable) != 12 or mutable != {path: step_zero[path] for path in mutable}:
        raise ValueError("RVIR-v4 pre-state snapshot/step-zero binding differs")
    mapping = initialize_rvir_v4_native_pre_state(
        snapshot, TOPOLOGY, expected_identity=EXPECTED_IDENTITY
    )
    if mapping.stable_hash() != EXPECTED_MAPPING_HASH:
        raise ValueError("RVIR-v4 pre-state mapping identity differs")
    if _file_sha256(model_path) != MODEL_SHA256:
        raise ValueError("RVIR-v4 pre-state model digest differs")
    program = import_onnx(str(model_path), do_shape_infer=True, normalize=True)
    if len(program.graph.inputs) != 1:
        raise ValueError("RVIR-v4 pre-state model input count differs")
    module = plan_interval_ibp_v0(program)
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=_one_role(snapshot, ProductionTensorRole.INPUT_LOWER),
        upper=_one_role(snapshot, ProductionTensorRole.INPUT_UPPER),
    )
    native_policy = trace.mutation_policy.to_native_policy()
    scope = build_native_alpha_beta_scope(
        module,
        input_spec,
        linear_spec_C=_one_role(snapshot, ProductionTensorRole.LINEAR_SPEC),
        relu_pre=mapping.relu_pre,
        relu_split_state=mapping.splits,
        policy=native_policy,
    )
    native_state = mapping.to_native_state(scope)
    mapping_payload = mapping.metadata()
    native_payload = native_state.to_dict()
    alpha_receipts = [
        receipt
        for receipt in mapping.round_trip_receipts
        if receipt.role == ProductionTensorRole.ALPHA
    ]
    summary: dict[str, object] = {
        "status": "validated-pre-state-initializer",
        "workload_id": "cifar10_resnet:000",
        "source_snapshot_hash": mapping.identity.snapshot_hash,
        "topology_hash": mapping.identity.topology_hash,
        "history_hash": mapping.identity.history_hash,
        "intermediate_bounds_hash": mapping.identity.intermediate_bounds_hash,
        "mapping_hash": mapping.stable_hash(),
        "native_scope_hash": scope.stable_hash(),
        "native_state_hash": native_state.stable_hash(),
        "relu_state_count": len(mapping.alphas),
        "round_trip_receipt_count": len(mapping.round_trip_receipts),
        "round_trip_exact_count": sum(
            receipt.mapped_source_hash == receipt.mapped_round_trip_hash
            and receipt.full_source_hash == receipt.full_round_trip_hash
            for receipt in mapping.round_trip_receipts
        ),
        "alpha_copy_through_receipt_count": sum(
            receipt.copy_through_element_count > 0 for receipt in alpha_receipts
        ),
        "step_zero_mutable_binding_count": len(mutable),
        "native_optimizer_update_count": native_policy.steps,
        "native_alpha_learning_rate": native_policy.lr,
        "native_beta_learning_rate": native_policy.effective_beta_lr,
        "optimizer_mutation_executed": False,
        "optimizer_replacement_admitted": False,
        "b2_same_solver_timing_admitted": False,
        "performance_claimed": False,
    }
    passed = (
        summary["relu_state_count"] == 6
        and summary["round_trip_receipt_count"] == 12
        and summary["round_trip_exact_count"] == 12
        and summary["alpha_copy_through_receipt_count"] == 6
        and summary["step_zero_mutable_binding_count"] == 12
        and native_policy.steps == 9
        and native_policy.lr == 0.01
        and native_policy.effective_beta_lr == 0.05
    )
    if not passed:
        raise ValueError("RVIR-v4 pre-state formal gate failed")
    summary["summary_hash"] = _canonical_hash(summary)
    return mapping_payload, native_payload, summary


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "mapping_hash": summary["mapping_hash"],
        "native_state_hash": summary["native_state_hash"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# RVIR-v4 V4-2C Native Pre-State Initializer\n\n"
        "This artifact maps the formal production pre-snapshot into six dense "
        "native alpha/beta/split states, binds external intermediate bounds and "
        "the real native scope, and closes twelve compressed/full round trips. "
        "It does not execute optimizer mutation or claim performance.\n"
    )


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if not _code_paths_clean():
        raise ValueError("RVIR-v4 pre-state code paths must be clean")
    source = args.source_artifact.resolve()
    model = args.model.resolve()
    output = args.artifact_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if (
        _file_sha256(source / "manifest.json") != SOURCE_MANIFEST_SHA256
        or _file_sha256(source / source_artifact_runner.WORKER_CAPTURE_FILE)
        != SOURCE_CAPTURE_SHA256
    ):
        raise ValueError("RVIR-v4 pre-state source artifact digest differs")
    source_artifact_runner._replay(source)
    shutil.copy2(
        source / source_artifact_runner.WORKER_CAPTURE_FILE,
        output / SOURCE_CAPTURE_FILE,
    )
    shutil.copy2(source / "manifest.json", output / SOURCE_MANIFEST_FILE)
    capture = _load_torch(output / SOURCE_CAPTURE_FILE)
    mapping, native_state, summary = _build_evidence(capture, model)
    _write_json(output / "topology.json", _topology_payload())
    _write_json(output / "mapping.json", mapping)
    _write_json(output / "native_state.json", native_state)
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
        "mapping_hash": summary["mapping_hash"],
        "native_state_hash": summary["native_state_hash"],
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
        raise ValueError("RVIR-v4 pre-state manifest differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("RVIR-v4 pre-state artifact inventory differs")
    for name in ARTIFACT_FILES:
        if files[name] != _file_sha256(artifact / name):
            raise ValueError(f"RVIR-v4 pre-state artifact digest differs: {name}")
    if (
        _file_sha256(artifact / SOURCE_MANIFEST_FILE) != SOURCE_MANIFEST_SHA256
        or _file_sha256(artifact / SOURCE_CAPTURE_FILE) != SOURCE_CAPTURE_SHA256
    ):
        raise ValueError("RVIR-v4 pre-state frozen source differs")
    _validate_source_manifest(_load_json(artifact / SOURCE_MANIFEST_FILE))
    if _load_json(artifact / "topology.json") != _topology_payload():
        raise ValueError("RVIR-v4 pre-state topology differs")
    mapping, native_state, summary = _build_evidence(
        _load_torch(artifact / SOURCE_CAPTURE_FILE), model
    )
    if (
        _load_json(artifact / "mapping.json") != mapping
        or _load_json(artifact / "native_state.json") != native_state
        or _load_json(artifact / "summary.json") != summary
    ):
        raise ValueError("RVIR-v4 pre-state semantic replay differs")
    if (
        manifest.get("mapping_hash") != summary["mapping_hash"]
        or manifest.get("native_state_hash") != summary["native_state_hash"]
        or manifest.get("summary_hash") != summary["summary_hash"]
    ):
        raise ValueError("RVIR-v4 pre-state semantic replay differs")
    result = _replay_result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(result) + "\n"
    ):
        raise ValueError("RVIR-v4 pre-state replay stdout differs")
    if (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("RVIR-v4 pre-state README differs")
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
    """Run formal V4-2C generation or deterministic replay."""

    args = _parse_args()
    result = (
        _generate(args)
        if args.command == "generate"
        else _replay(args.artifact_dir.resolve(), args.model.resolve())
    )
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
