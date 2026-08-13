#!/usr/bin/env python3
"""Generate or replay the RVIR-v4 V4-3C native KFSB artifact."""

# pylint: disable=wrong-import-position,protected-access,duplicate-code
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-boolean-expressions,no-member

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence, cast

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.native_alpha_beta_optimization_state import (
    build_native_alpha_beta_scope,
)
from boundflow.runtime.rvir_v4_native_backward_export import (
    export_rvir_v4_native_backward,
)
from boundflow.runtime.rvir_v4_native_kfsb import (
    compare_rvir_v4_native_kfsb,
    evaluate_rvir_v4_native_kfsb,
    NativeKfsbEvaluationV4,
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
from scripts import run_rvir_v4_native_backward_export_artifact as backward_runner

ARTIFACT_SCHEMA_VERSION = "boundflow.rvir-v4-native-kfsb-artifact/v1"
SOURCE_CAPTURE_FILE = backward_runner.SOURCE_CAPTURE_FILE
SOURCE_ATOMIC_MANIFEST_FILE = backward_runner.SOURCE_ATOMIC_MANIFEST_FILE
SOURCE_TRUTH_FILE = backward_runner.SOURCE_TRUTH_FILE
SOURCE_TRUTH_MANIFEST_FILE = backward_runner.SOURCE_TRUTH_MANIFEST_FILE
EVALUATION_FILE = "evaluation.pt"
ARTIFACT_FILES = (
    SOURCE_CAPTURE_FILE,
    SOURCE_ATOMIC_MANIFEST_FILE,
    SOURCE_TRUTH_FILE,
    SOURCE_TRUTH_MANIFEST_FILE,
    EVALUATION_FILE,
    "topology.json",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)
ATOMIC_MANIFEST_SHA256 = backward_runner.ATOMIC_MANIFEST_SHA256
SOURCE_CAPTURE_SHA256 = backward_runner.SOURCE_CAPTURE_SHA256
TRUTH_MANIFEST_SHA256 = backward_runner.TRUTH_MANIFEST_SHA256
SOURCE_TRUTH_SHA256 = backward_runner.SOURCE_TRUTH_SHA256
MODEL_SHA256 = backward_runner.MODEL_SHA256
CODE_PATHS = (
    "boundflow/runtime/crown_ibp.py",
    "boundflow/runtime/rvir_v4_native_backward_export.py",
    "boundflow/runtime/rvir_v4_native_kfsb.py",
    "scripts/run_rvir_v4_native_kfsb_artifact.py",
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
        raise TypeError(f"RVIR-v4 native KFSB torch root differs: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"RVIR-v4 native KFSB JSON root differs: {path}")
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
        raise ValueError("RVIR-v4 native KFSB code provenance differs")
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
        raise ValueError("RVIR-v4 native KFSB code revision differs")


def _topology() -> tuple[Any, ...]:
    return cast(tuple[Any, ...], backward_runner._topology())


def _topology_payload() -> dict[str, object]:
    rows = [item.to_dict() for item in _topology()]
    return {"rows": rows, "topology_hash": _canonical_hash(rows)}


def _one_role(snapshot: Any, role: ProductionTensorRole) -> torch.Tensor:
    values = [tensor.value for tensor in snapshot.tensors if tensor.role == role]
    if len(values) != 1:
        raise ValueError(f"RVIR-v4 native KFSB requires one {role.value}")
    return values[0]


def _truth_inputs(
    truth: Mapping[str, Any],
) -> tuple[
    tuple[tuple[tuple[int, int], ...], ...],
    tuple[torch.Tensor, ...],
    tuple[tuple[int, int], ...],
    dict[str, torch.Tensor],
]:
    cores = truth.get("whole_core_truths")
    if (
        not isinstance(cores, list)
        or len(cores) != 1
        or not isinstance(cores[0], Mapping)
    ):
        raise ValueError("RVIR-v4 native KFSB truth core inventory differs")
    branch = cast(Mapping[str, Any], cores[0]).get("branch_trace")
    if not isinstance(branch, Mapping):
        raise TypeError("RVIR-v4 native KFSB truth branch differs")
    candidates = branch.get("candidate_splits")
    child_lowers = branch.get("candidate_child_lowers")
    final = branch.get("final_decision")
    inputs = branch.get("input")
    if (
        not isinstance(candidates, list)
        or not isinstance(child_lowers, list)
        or not isinstance(final, Mapping)
        or not isinstance(inputs, Mapping)
        or not isinstance(inputs.get("mask"), Mapping)
    ):
        raise TypeError("RVIR-v4 native KFSB truth payload differs")
    normalized_candidates = tuple(
        tuple(
            (int(decision[0]), int(decision[1]))
            for decision in cast(Mapping[str, Any], candidate)["decision"]
        )
        for candidate in candidates
    )
    normalized_lowers = tuple(
        cast(Mapping[str, Any], record)["value"] for record in child_lowers
    )
    normalized_final = tuple(
        (int(decision[0]), int(decision[1]))
        for decision in cast(Sequence[Any], final["decision"])
    )
    masks = {
        str(name): cast(Mapping[str, Any], record)["value"]
        for name, record in cast(Mapping[str, Any], inputs["mask"]).items()
    }
    if not all(
        torch.is_tensor(value) for value in (*normalized_lowers, *masks.values())
    ):
        raise TypeError("RVIR-v4 native KFSB truth tensor differs")
    return (
        normalized_candidates,
        cast(tuple[torch.Tensor, ...], normalized_lowers),
        normalized_final,
        cast(dict[str, torch.Tensor], masks),
    )


def _evaluation_payload(evaluation: NativeKfsbEvaluationV4) -> dict[str, object]:
    evaluation.validate()
    return {
        "schema_version": evaluation.schema_version,
        "candidate_splits": [
            [[layer, neuron] for layer, neuron in decisions]
            for decisions in evaluation.candidate_splits
        ],
        "candidate_child_lowers": list(evaluation.candidate_child_lowers),
        "final_decision": [list(decision) for decision in evaluation.final_decision],
        "alpha_score_topk_values": evaluation.alpha_score_topk_values,
        "intercept_score_topk_values": evaluation.intercept_score_topk_values,
        "reduced_candidate_values": evaluation.reduced_candidate_values,
        "unstable_masks": evaluation.unstable_masks,
        "layer_sizes": list(evaluation.layer_sizes),
        "unstable_counts": list(evaluation.unstable_counts),
        "metadata": evaluation.metadata(),
    }


def _evaluation_from_payload(
    payload: Mapping[str, Any], *, validate_metadata: bool = True
) -> NativeKfsbEvaluationV4:
    candidates = payload.get("candidate_splits")
    child_lowers = payload.get("candidate_child_lowers")
    final = payload.get("final_decision")
    masks = payload.get("unstable_masks")
    metadata = payload.get("metadata")
    if (
        payload.get("schema_version") != "boundflow.rvir-v4-native-kfsb/v1"
        or not isinstance(candidates, list)
        or not isinstance(child_lowers, list)
        or not isinstance(final, list)
        or not isinstance(masks, Mapping)
        or not isinstance(metadata, Mapping)
    ):
        raise ValueError("RVIR-v4 native KFSB recorded evaluation differs")
    evaluation = NativeKfsbEvaluationV4(
        candidate_splits=tuple(
            tuple((int(value[0]), int(value[1])) for value in decisions)
            for decisions in candidates
        ),
        candidate_child_lowers=tuple(child_lowers),
        final_decision=tuple((int(value[0]), int(value[1])) for value in final),
        alpha_score_topk_values=payload["alpha_score_topk_values"],
        intercept_score_topk_values=payload["intercept_score_topk_values"],
        reduced_candidate_values=payload["reduced_candidate_values"],
        unstable_mask_by_provider_preactivation=tuple(
            (str(name), value) for name, value in masks.items()
        ),
        layer_sizes=tuple(int(value) for value in payload["layer_sizes"]),
        unstable_counts=tuple(int(value) for value in payload["unstable_counts"]),
    )
    evaluation.validate()
    if validate_metadata and evaluation.metadata() != dict(metadata):
        raise ValueError("RVIR-v4 native KFSB recorded metadata differs")
    return evaluation


def _build_evidence_single_thread(
    capture: Mapping[str, Any], truth: Mapping[str, Any], model_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    source_runner = cast(
        Any, backward_runner.atomic_runner.source_artifact_runner.source_artifact_runner
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
        raise ValueError("RVIR-v4 native KFSB source inventory differs")
    pre_raw = cast(Mapping[str, Any], cores[0]).get("pre_snapshot")
    if not isinstance(pre_raw, Mapping):
        raise TypeError("RVIR-v4 native KFSB pre snapshot differs")
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
        raise ValueError("RVIR-v4 native KFSB model digest differs")
    program = import_onnx(str(model_path), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=_one_role(pre, ProductionTensorRole.INPUT_LOWER),
        upper=_one_role(pre, ProductionTensorRole.INPUT_UPPER),
    )
    objective = _one_role(pre, ProductionTensorRole.LINEAR_SPEC)
    thresholds = _one_role(pre, ProductionTensorRole.DECISION_THRESHOLD)
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
    evaluation = evaluate_rvir_v4_native_kfsb(
        module=module,
        input_spec=input_spec,
        linear_spec_C=objective,
        thresholds=thresholds,
        terminal_state=terminal,
        topology=topology,
        backward_export=export,
    )
    expected_candidates, expected_lowers, expected_final, expected_masks = (
        _truth_inputs(truth)
    )
    parity = compare_rvir_v4_native_kfsb(
        evaluation,
        expected_candidate_splits=expected_candidates,
        expected_candidate_child_lowers=expected_lowers,
        expected_final_decision=expected_final,
        expected_unstable_masks=expected_masks,
    )
    evaluation_metadata = evaluation.metadata()
    summary: dict[str, object] = {
        "status": "validated-native-kfsb",
        "workload_id": "cifar10_resnet:000",
        "domain_count": 6,
        "candidate_count": 3,
        "candidate_decision_count": 36,
        "child_domain_evaluation_count": 72,
        "final_decision_count": 6,
        "unstable_neuron_count": sum(evaluation.unstable_counts),
        "provider_core_callback_count": 0,
        "provider_compute_bounds_callback_count": 0,
        "provider_update_bounds_callback_count": 0,
        "fallback_dispatch_count": 0,
        "evaluation_hash": evaluation_metadata["evaluation_hash"],
        **parity.metadata(),
    }
    summary["summary_hash"] = _canonical_hash(summary)
    return _evaluation_payload(evaluation), summary


def _build_evidence(
    capture: Mapping[str, Any], truth: Mapping[str, Any], model_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _build_evidence_single_thread(capture, truth, model_path)
    finally:
        torch.set_num_threads(previous_threads)


def _compare_recorded(
    recorded: NativeKfsbEvaluationV4, observed: NativeKfsbEvaluationV4
) -> None:
    compare_rvir_v4_native_kfsb(
        recorded,
        expected_candidate_splits=observed.candidate_splits,
        expected_candidate_child_lowers=observed.candidate_child_lowers,
        expected_final_decision=observed.final_decision,
        expected_unstable_masks=observed.unstable_masks,
    )
    if (
        recorded.layer_sizes != observed.layer_sizes
        or recorded.unstable_counts != observed.unstable_counts
    ):
        raise ValueError("RVIR-v4 native KFSB recorded structure differs")
    for actual, expected in (
        (recorded.alpha_score_topk_values, observed.alpha_score_topk_values),
        (
            recorded.intercept_score_topk_values,
            observed.intercept_score_topk_values,
        ),
        (recorded.reduced_candidate_values, observed.reduced_candidate_values),
    ):
        if not torch.allclose(
            actual,
            expected,
            atol=2e-4,
            rtol=2e-4,
            equal_nan=False,
        ) or not torch.equal(torch.sign(actual), torch.sign(expected)):
            raise ValueError("RVIR-v4 native KFSB recorded score differs")


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "evaluation_hash": summary["evaluation_hash"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# RVIR-v4 V4-3C Native KFSB\n\n"
        "This artifact derives six unstable masks, reproduces three top-3 KFSB "
        "candidate sets, evaluates 72 child domains with BoundFlow, and reproduces "
        "the final six-domain decision. It performs no provider bound callback. "
        "It is not live whole-core replacement or performance evidence.\n"
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
            raise ValueError(f"RVIR-v4 native KFSB frozen source differs: {name}")


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if not _code_paths_clean():
        raise ValueError("RVIR-v4 native KFSB code paths must be clean")
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
        raise ValueError("RVIR-v4 native KFSB source artifact identity differs")
    shutil.copy2(atomic / SOURCE_CAPTURE_FILE, output / SOURCE_CAPTURE_FILE)
    shutil.copy2(atomic / "manifest.json", output / SOURCE_ATOMIC_MANIFEST_FILE)
    shutil.copy2(truth / "truth.pt", output / SOURCE_TRUTH_FILE)
    shutil.copy2(truth / "manifest.json", output / SOURCE_TRUTH_MANIFEST_FILE)
    evaluation, summary = _build_evidence(
        _load_torch(output / SOURCE_CAPTURE_FILE),
        _load_torch(output / SOURCE_TRUTH_FILE),
        model,
    )
    torch.save(evaluation, output / EVALUATION_FILE)
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
        "evaluation_hash": summary["evaluation_hash"],
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
        raise ValueError("RVIR-v4 native KFSB manifest differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("RVIR-v4 native KFSB artifact inventory differs")
    for name in ARTIFACT_FILES:
        if files[name] != _file_sha256(artifact / name):
            raise ValueError(f"RVIR-v4 native KFSB digest differs: {name}")
    _validate_sources(artifact)
    if _load_json(artifact / "topology.json") != _topology_payload():
        raise ValueError("RVIR-v4 native KFSB topology differs")
    evaluation, summary = _build_evidence(
        _load_torch(artifact / SOURCE_CAPTURE_FILE),
        _load_torch(artifact / SOURCE_TRUTH_FILE),
        model,
    )
    recorded = _evaluation_from_payload(_load_torch(artifact / EVALUATION_FILE))
    observed = _evaluation_from_payload(evaluation)
    _compare_recorded(recorded, observed)
    if _load_json(artifact / "summary.json") != summary:
        raise ValueError("RVIR-v4 native KFSB semantic replay differs")
    if (
        manifest.get("evaluation_hash") != summary["evaluation_hash"]
        or manifest.get("summary_hash") != summary["summary_hash"]
    ):
        raise ValueError("RVIR-v4 native KFSB semantic identity differs")
    result = _replay_result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(result) + "\n"
    ):
        raise ValueError("RVIR-v4 native KFSB replay stdout differs")
    if (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("RVIR-v4 native KFSB README differs")
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
    """Generate or replay V4-3C formal evidence."""

    args = _parse_args()
    result = (
        _generate(args)
        if args.command == "generate"
        else _replay(args.artifact_dir.resolve(), args.model.resolve())
    )
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
