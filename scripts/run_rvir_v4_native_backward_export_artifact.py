#!/usr/bin/env python3
"""Generate or replay the RVIR-v4 V4-3B native backward export artifact."""

# pylint: disable=wrong-import-position,protected-access,duplicate-code
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions
# pylint: disable=no-member

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

from boundflow.domains.interval import IntervalState
from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.native_alpha_beta_optimization_state import (
    build_native_alpha_beta_scope,
)
from boundflow.runtime.rvir_v4_native_backward_export import (
    compare_rvir_v4_native_backward_export,
    export_rvir_v4_native_backward,
    NativeBackwardExportV4,
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
    ProductionTensorRole,
)
from boundflow.runtime.task_executor import InputSpec
from scripts import run_rvir_v4_atomic_copy_out_artifact as atomic_runner

ARTIFACT_SCHEMA_VERSION = "boundflow.rvir-v4-native-backward-export-artifact/v1"
SOURCE_CAPTURE_FILE = "source_capture.pt"
SOURCE_ATOMIC_MANIFEST_FILE = "source_atomic_manifest.json"
SOURCE_TRUTH_FILE = "source_whole_core_truth.pt"
SOURCE_TRUTH_MANIFEST_FILE = "source_whole_core_manifest.json"
EXPORT_FILE = "export.pt"
ARTIFACT_FILES = (
    SOURCE_CAPTURE_FILE,
    SOURCE_ATOMIC_MANIFEST_FILE,
    SOURCE_TRUTH_FILE,
    SOURCE_TRUTH_MANIFEST_FILE,
    EXPORT_FILE,
    "topology.json",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)
ATOMIC_MANIFEST_SHA256 = (
    "b76ee57348f2311996e6b40f013b46acdf39171a3ddc12ae2be9fa0119800136"
)
SOURCE_CAPTURE_SHA256 = (
    "f42229dd06be8521865d14af4cc51450ec8bec4e792934b457c276aad50126dc"
)
TRUTH_MANIFEST_SHA256 = (
    "0e6ed721dbf796cf8923dd57e09636f05895a1a065595ea1154b170a4a0c9818"
)
SOURCE_TRUTH_SHA256 = "d0126427dcdc868d33c7a7ec6326bdb86c8fb6e624a16c650d46c401ecabd0e9"
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
CODE_PATHS = (
    "boundflow/runtime/crown_ibp.py",
    "boundflow/runtime/rvir_v4_native_backward_export.py",
    "scripts/run_rvir_v4_native_backward_export_artifact.py",
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


def _load_torch(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"RVIR-v4 native backward torch root differs: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"RVIR-v4 native backward JSON root differs: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


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
        raise ValueError("RVIR-v4 native backward code provenance differs")
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
        raise ValueError("RVIR-v4 native backward code revision differs")


def _topology() -> tuple[Any, ...]:
    source_runner = cast(
        Any, atomic_runner.source_artifact_runner.source_artifact_runner
    )
    return cast(tuple[Any, ...], source_runner.TOPOLOGY)


def _topology_payload() -> dict[str, object]:
    rows = [item.to_dict() for item in _topology()]
    return {"rows": rows, "topology_hash": _canonical_hash(rows)}


def _one_role(snapshot: Any, role: ProductionTensorRole) -> torch.Tensor:
    values = [tensor.value for tensor in snapshot.tensors if tensor.role == role]
    if len(values) != 1:
        raise ValueError(f"RVIR-v4 native backward requires one {role.value}")
    return values[0]


def _truth_inputs(
    truth: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, IntervalState]]:
    cores = truth.get("whole_core_truths")
    if (
        not isinstance(cores, list)
        or len(cores) != 1
        or not isinstance(cores[0], Mapping)
    ):
        raise ValueError("RVIR-v4 native backward truth core inventory differs")
    core = cast(Mapping[str, Any], cores[0])
    expected_lower = core["fields"]["lb"]["value"]
    branch_l_as = core["branch_trace"]["input"]["lAs"]["_data"]
    intermediate = core["working_intermediate_bounds"]
    if (
        not torch.is_tensor(expected_lower)
        or not isinstance(branch_l_as, Mapping)
        or not isinstance(intermediate, Mapping)
    ):
        raise TypeError("RVIR-v4 native backward truth tensors differ")
    expected_l_as = {
        str(name): cast(Mapping[str, Any], record)["value"]
        for name, record in branch_l_as.items()
    }
    expected_intermediates = {
        str(name): IntervalState(
            lower=cast(Mapping[str, Any], record)["lower"]["value"],
            upper=cast(Mapping[str, Any], record)["upper"]["value"],
        )
        for name, record in intermediate.items()
    }
    if not all(torch.is_tensor(value) for value in expected_l_as.values()):
        raise TypeError("RVIR-v4 native backward truth lA differs")
    return (
        expected_lower,
        cast(dict[str, torch.Tensor], expected_l_as),
        expected_intermediates,
    )


def _export_payload(export: NativeBackwardExportV4) -> dict[str, object]:
    export.validate()
    return {
        "schema_version": export.schema_version,
        "lower": export.lower,
        "lAs": export.l_as,
        "intermediates": {
            name: {"lower": value.lower, "upper": value.upper}
            for name, value in export.intermediates.items()
        },
        "metadata": export.metadata(),
    }


def _export_from_payload(
    payload: Mapping[str, Any], *, validate_metadata: bool = True
) -> NativeBackwardExportV4:
    l_as = payload.get("lAs")
    intermediate = payload.get("intermediates")
    lower = payload.get("lower")
    metadata = payload.get("metadata")
    if (
        payload.get("schema_version") != "boundflow.rvir-v4-native-backward-export/v1"
        or not torch.is_tensor(lower)
        or not isinstance(l_as, Mapping)
        or not isinstance(intermediate, Mapping)
        or not isinstance(metadata, Mapping)
    ):
        raise ValueError("RVIR-v4 native backward recorded export differs")
    export = NativeBackwardExportV4(
        lower=lower,
        l_a_by_provider_activation=tuple(
            sorted((str(name), value) for name, value in l_as.items())
        ),
        intermediate_by_provider_preactivation=tuple(
            sorted(
                (
                    str(name),
                    IntervalState(
                        lower=cast(Mapping[str, Any], value)["lower"],
                        upper=cast(Mapping[str, Any], value)["upper"],
                    ),
                )
                for name, value in intermediate.items()
            )
        ),
        intermediate_source="shared-pre-result-external-bounds",
    )
    export.validate()
    if validate_metadata and export.metadata() != dict(metadata):
        raise ValueError("RVIR-v4 native backward recorded metadata differs")
    return export


def _build_evidence_single_thread(
    capture: Mapping[str, Any], truth: Mapping[str, Any], model_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    source_runner = cast(
        Any, atomic_runner.source_artifact_runner.source_artifact_runner
    )
    capture_runner = cast(Any, source_runner.source_artifact_runner)
    capture_runner.validate_worker_capture(capture)
    cores = capture.get("cores")
    traces = capture.get("optimizer_step_traces")
    if (
        not isinstance(cores, list)
        or len(cores) != 1
        or not isinstance(cores[0], Mapping)
        or not isinstance(traces, list)
        or len(traces) != 1
        or not isinstance(traces[0], Mapping)
    ):
        raise ValueError("RVIR-v4 native backward source inventory differs")
    core = cast(Mapping[str, Any], cores[0])
    pre_raw = core.get("pre_snapshot")
    if not isinstance(pre_raw, Mapping):
        raise TypeError("RVIR-v4 native backward pre snapshot differs")
    pre = production_snapshot_from_payload_v4(pre_raw)
    production = production_optimizer_step_trace_from_payload_v4(
        cast(Mapping[str, object], traces[0])
    )
    topology = _topology()
    mapping = initialize_rvir_v4_native_pre_state(
        pre,
        topology,
        expected_identity=source_runner.EXPECTED_IDENTITY,
    )
    if _file_sha256(model_path) != MODEL_SHA256:
        raise ValueError("RVIR-v4 native backward model digest differs")
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
    export = export_rvir_v4_native_backward(
        module=module,
        input_spec=input_spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        terminal_state=terminal,
        topology=topology,
    )
    expected_lower, expected_l_as, expected_intermediates = _truth_inputs(truth)
    parity = compare_rvir_v4_native_backward_export(
        export,
        expected_lower=expected_lower,
        expected_l_as=expected_l_as,
        expected_intermediates=expected_intermediates,
    )
    export_metadata = export.metadata()
    parity_metadata = parity.metadata()
    summary: dict[str, object] = {
        "status": "validated-native-backward-export",
        "workload_id": "cifar10_resnet:000",
        "domain_count": 6,
        "lA_count": 6,
        "intermediate_tensor_count": 12,
        "provider_core_callback_count": 0,
        "provider_compute_bounds_callback_count": 0,
        "provider_update_bounds_callback_count": 0,
        "fallback_dispatch_count": 0,
        "export_hash": export_metadata["export_hash"],
        **parity_metadata,
    }
    summary["summary_hash"] = _canonical_hash(summary)
    return _export_payload(export), summary


def _build_evidence(
    capture: Mapping[str, Any], truth: Mapping[str, Any], model_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _build_evidence_single_thread(capture, truth, model_path)
    finally:
        torch.set_num_threads(previous_threads)


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "export_hash": summary["export_hash"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# RVIR-v4 V4-3B Native Backward Export\n\n"
        "This artifact independently exports six native lower adjoints and twelve "
        "shared-input intermediate-bound tensors, then compares them with the V4-3A "
        "truth. It is not KFSB, live whole-core replacement, or performance evidence.\n"
    )


def _validate_sources(artifact: Path) -> None:
    expected = {
        SOURCE_CAPTURE_FILE: SOURCE_CAPTURE_SHA256,
        SOURCE_ATOMIC_MANIFEST_FILE: ATOMIC_MANIFEST_SHA256,
        SOURCE_TRUTH_FILE: SOURCE_TRUTH_SHA256,
        SOURCE_TRUTH_MANIFEST_FILE: TRUTH_MANIFEST_SHA256,
    }
    for name, digest in expected.items():
        if _file_sha256(artifact / name) != digest:
            raise ValueError(f"RVIR-v4 native backward frozen source differs: {name}")


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if not _code_paths_clean():
        raise ValueError("RVIR-v4 native backward code paths must be clean")
    atomic = args.atomic_artifact.resolve()
    truth = args.truth_artifact.resolve()
    model = args.model.resolve()
    output = args.artifact_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if (
        _file_sha256(atomic / "manifest.json") != ATOMIC_MANIFEST_SHA256
        or _file_sha256(atomic / SOURCE_CAPTURE_FILE) != SOURCE_CAPTURE_SHA256
        or _file_sha256(truth / "manifest.json") != TRUTH_MANIFEST_SHA256
        or _file_sha256(truth / "truth.pt") != SOURCE_TRUTH_SHA256
    ):
        raise ValueError("RVIR-v4 native backward source artifact identity differs")
    shutil.copy2(atomic / SOURCE_CAPTURE_FILE, output / SOURCE_CAPTURE_FILE)
    shutil.copy2(atomic / "manifest.json", output / SOURCE_ATOMIC_MANIFEST_FILE)
    shutil.copy2(truth / "truth.pt", output / SOURCE_TRUTH_FILE)
    shutil.copy2(truth / "manifest.json", output / SOURCE_TRUTH_MANIFEST_FILE)
    export, summary = _build_evidence(
        _load_torch(output / SOURCE_CAPTURE_FILE),
        _load_torch(output / SOURCE_TRUTH_FILE),
        model,
    )
    torch.save(export, output / EXPORT_FILE)
    _write_json(output / "topology.json", _topology_payload())
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
        "export_hash": summary["export_hash"],
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
        raise ValueError("RVIR-v4 native backward manifest differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("RVIR-v4 native backward artifact inventory differs")
    for name in ARTIFACT_FILES:
        if files[name] != _file_sha256(artifact / name):
            raise ValueError(f"RVIR-v4 native backward digest differs: {name}")
    _validate_sources(artifact)
    if _load_json(artifact / "topology.json") != _topology_payload():
        raise ValueError("RVIR-v4 native backward topology differs")
    export, summary = _build_evidence(
        _load_torch(artifact / SOURCE_CAPTURE_FILE),
        _load_torch(artifact / SOURCE_TRUTH_FILE),
        model,
    )
    recorded = _load_torch(artifact / EXPORT_FILE)
    recorded_export = _export_from_payload(recorded)
    observed_export = _export_from_payload(export)
    compare_rvir_v4_native_backward_export(
        recorded_export,
        expected_lower=observed_export.lower,
        expected_l_as=observed_export.l_as,
        expected_intermediates=observed_export.intermediates,
    )
    if _load_json(artifact / "summary.json") != summary:
        raise ValueError("RVIR-v4 native backward semantic replay differs")
    if (
        manifest.get("export_hash") != summary["export_hash"]
        or manifest.get("summary_hash") != summary["summary_hash"]
    ):
        raise ValueError("RVIR-v4 native backward semantic identity differs")
    result = _replay_result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(result) + "\n"
    ):
        raise ValueError("RVIR-v4 native backward replay stdout differs")
    if (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("RVIR-v4 native backward README differs")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--atomic-artifact", type=Path, required=True)
    generate.add_argument("--truth-artifact", type=Path, required=True)
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--artifact-dir", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Generate or replay V4-3B formal evidence."""

    args = _parse_args()
    result = (
        _generate(args)
        if args.command == "generate"
        else _replay(args.artifact_dir.resolve(), args.model.resolve())
    )
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
