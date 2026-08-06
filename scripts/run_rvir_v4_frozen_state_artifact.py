#!/usr/bin/env python3
"""Generate or replay RVIR-v4 V4-1 native frozen-state evidence."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.rvir_v4_frozen_state import (
    ProductionReluTopologyV4,
    evaluate_rvir_v4_frozen_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
)

SCHEMA_VERSION = "boundflow.rvir-v4-frozen-state-artifact/v1"
CAPTURE_FILE = "capture.pt"
ARTIFACT_FILES = (
    CAPTURE_FILE,
    "source.json",
    "topology.json",
    "execution.json",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)
CAPTURE_SHA256 = "99c2c0766621bc0c7db77e1aa4f9f262baa07ff2f9d64d742984a03de000df1e"
CAPTURE_MANIFEST_SHA256 = (
    "8706e1176a9d29a232fcc8d455a88c7889920f34ad70c8fb75fd0c711142d255"
)
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
CODE_PATHS = (
    "boundflow/runtime/rvir_v4_production_state.py",
    "boundflow/runtime/rvir_v4_frozen_state.py",
    "scripts/run_rvir_v4_frozen_state_artifact.py",
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


def canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"RVIR-v4 frozen JSON root differs: {path}")
    return value


def _code_revision() -> dict[str, str]:
    root = _repo_root()
    return {path: file_sha256(root / path) for path in CODE_PATHS}


def _code_paths_clean() -> bool:
    return not _git_value("status", "--porcelain=v1", "--", *CODE_PATHS)


def _verify_code_provenance(manifest: Mapping[str, Any]) -> None:
    root = _repo_root()
    source_head = manifest.get("source_git_head")
    revision = manifest.get("code_revision")
    if not isinstance(source_head, str) or not isinstance(revision, Mapping):
        raise ValueError("RVIR-v4 frozen source provenance differs")
    if _git_value("rev-parse", "HEAD") == source_head:
        observed = _code_revision()
    else:
        observed = {
            path: hashlib.sha256(
                subprocess.run(
                    ("git", "show", f"{source_head}:{path}"),
                    cwd=root,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ).stdout
            ).hexdigest()
            for path in CODE_PATHS
        }
    if dict(revision) != observed:
        raise ValueError("RVIR-v4 frozen source code revision differs")


def _topology_payload() -> dict[str, object]:
    rows = [item.__dict__ for item in TOPOLOGY]
    return {"rows": rows, "topology_hash": canonical_hash(rows)}


def _validate_topology(path: Path) -> None:
    if _load_json(path) != _topology_payload():
        raise ValueError("RVIR-v4 frozen topology semantic mapping differs")


def _execute(
    capture_path: Path, model_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    if file_sha256(capture_path) != CAPTURE_SHA256:
        raise ValueError("RVIR-v4 frozen source state digest differs")
    if file_sha256(model_path) != MODEL_SHA256:
        raise ValueError("RVIR-v4 frozen model digest differs")
    payload = torch.load(capture_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("RVIR-v4 frozen capture root differs")
    cores = payload.get("cores")
    if not isinstance(cores, list) or len(cores) != 1:
        raise ValueError("RVIR-v4 frozen capture core count differs")
    core = cores[0]
    if not isinstance(core, Mapping) or not torch.is_tensor(core.get("lower")):
        raise TypeError("RVIR-v4 frozen core payload differs")
    pre_raw = core.get("pre_snapshot")
    post_raw = core.get("post_snapshot")
    if not isinstance(pre_raw, Mapping) or not isinstance(post_raw, Mapping):
        raise TypeError("RVIR-v4 frozen snapshot payload differs")
    pre = production_snapshot_from_payload_v4(pre_raw)
    post = production_snapshot_from_payload_v4(post_raw)
    expected_lower = core["lower"]
    program = import_onnx(str(model_path), do_shape_infer=True, normalize=True)
    if len(program.graph.inputs) != 1:
        raise ValueError("RVIR-v4 frozen model input count differs")
    module = plan_interval_ibp_v0(program)
    result = evaluate_rvir_v4_frozen_state(
        module=module,
        input_value_name=program.graph.inputs[0],
        pre=pre,
        post=post,
        topology=TOPOLOGY,
        query_id="rvir-v4-v4-1-resnet2b-core-000000",
        expected_lower=expected_lower,
    )
    difference = (result.lower - expected_lower).abs()
    signs = (result.lower >= 0) == (expected_lower >= 0)
    max_abs_diff = float(difference.max().item())
    execution: dict[str, object] = {
        "result_shape": list(result.lower.shape),
        "expected_shape": list(expected_lower.shape),
        "native_lower": result.lower.flatten().tolist(),
        "production_lower": expected_lower.flatten().tolist(),
        "lower_max_abs_diff": max_abs_diff,
        "lower_allclose": bool(
            torch.allclose(result.lower, expected_lower, atol=2e-4, rtol=2e-4)
        ),
        "sign_agreement": int(signs.sum().item()),
        "sign_total": int(signs.numel()),
        "state_hash": result.state_hash,
        "ir_hashes": dict(result.ir_hashes),
        "replacement_dispatch_count": 1,
        "original_callback_count": 0,
        "fallback_dispatch_count": 0,
        "performance_claimed": False,
    }
    passed = (
        execution["result_shape"] == [6, 1]
        and execution["expected_shape"] == [6, 1]
        and execution["lower_allclose"] is True
        and execution["sign_agreement"] == execution["sign_total"] == 6
        and max_abs_diff <= 2e-4
        and len(result.ir_hashes) == 10
        and execution["original_callback_count"] == 0
        and execution["fallback_dispatch_count"] == 0
    )
    summary: dict[str, object] = {
        "status": "validated_reduced_frozen_state_evaluation" if passed else "no_go",
        "workload_id": "cifar10_resnet:000",
        "core_count": 1,
        "domain_count": 6,
        "lower_max_abs_diff": execution["lower_max_abs_diff"],
        "sign_agreement": execution["sign_agreement"],
        "sign_total": execution["sign_total"],
        "original_callback_count": 0,
        "fallback_dispatch_count": 0,
        "optimizer_mutation_replacement_admitted": False,
        "b2_same_solver_timing_admitted": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    if not passed:
        raise ValueError("RVIR-v4 frozen-state evaluation gate failed")
    return execution, summary


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "lower_max_abs_diff": summary["lower_max_abs_diff"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# RVIR-v4 V4-1 Frozen-State Evaluation\n\n"
        "This artifact maps frozen production alpha/beta/split state into the native "
        "Bound/Plan/Task/Schedule evaluator with zero provider callbacks. It does not "
        "replace optimizer mutation and makes no performance claim.\n"
    )


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if not _code_paths_clean():
        raise ValueError("RVIR-v4 frozen code paths must be clean")
    artifact = args.artifact_dir.resolve()
    if artifact.exists() and any(artifact.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    capture = args.source_capture.resolve()
    capture_manifest = args.source_manifest.resolve()
    model = args.model.resolve()
    if (
        file_sha256(capture) != CAPTURE_SHA256
        or file_sha256(capture_manifest) != CAPTURE_MANIFEST_SHA256
        or file_sha256(model) != MODEL_SHA256
    ):
        raise ValueError("RVIR-v4 frozen source inputs differ")
    shutil.copy2(capture, artifact / CAPTURE_FILE)
    source = {
        "capture_sha256": CAPTURE_SHA256,
        "capture_manifest_sha256": CAPTURE_MANIFEST_SHA256,
        "model_sha256": MODEL_SHA256,
        "performance_claimed": False,
    }
    _write_json(artifact / "source.json", source)
    _write_json(artifact / "topology.json", _topology_payload())
    execution, summary = _execute(artifact / CAPTURE_FILE, model)
    _write_json(artifact / "execution.json", execution)
    _write_json(artifact / "summary.json", summary)
    replay = _replay_result(summary)
    (artifact / "replay_stdout.txt").write_text(
        canonical_json(replay) + "\n", encoding="utf-8"
    )
    (artifact / "README.md").write_text(_readme(), encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source_git_head": _git_value("rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "files": {name: file_sha256(artifact / name) for name in ARTIFACT_FILES},
        "summary_hash": summary["summary_hash"],
        "status": summary["status"],
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact / "manifest.json", manifest)
    return replay


def _replay(artifact: Path, model: Path) -> dict[str, object]:
    manifest = _load_json(artifact / "manifest.json")
    semantic = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("manifest_hash") != canonical_hash(semantic)
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("RVIR-v4 frozen manifest envelope differs")
    _verify_code_provenance(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("RVIR-v4 frozen artifact inventory differs")
    for name, digest in files.items():
        if file_sha256(artifact / name) != digest:
            raise ValueError("RVIR-v4 frozen artifact file digest differs")
    source = _load_json(artifact / "source.json")
    if source != {
        "capture_sha256": CAPTURE_SHA256,
        "capture_manifest_sha256": CAPTURE_MANIFEST_SHA256,
        "model_sha256": MODEL_SHA256,
        "performance_claimed": False,
    }:
        raise ValueError("RVIR-v4 frozen source identity differs")
    _validate_topology(artifact / "topology.json")
    execution, summary = _execute(artifact / CAPTURE_FILE, model)
    if _load_json(artifact / "execution.json") != execution:
        raise ValueError("RVIR-v4 frozen execution replay differs")
    if _load_json(artifact / "summary.json") != summary:
        raise ValueError("RVIR-v4 frozen summary replay differs")
    if manifest.get("summary_hash") != summary["summary_hash"]:
        raise ValueError("RVIR-v4 frozen manifest summary differs")
    result = _replay_result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        canonical_json(result) + "\n"
    ):
        raise ValueError("RVIR-v4 frozen replay stdout differs")
    if (artifact / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("RVIR-v4 frozen README differs")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--source-capture", type=Path, required=True)
    generate.add_argument("--source-manifest", type=Path, required=True)
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--artifact-dir", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = (
        _generate(args)
        if args.command == "generate"
        else _replay(args.artifact_dir.resolve(), args.model.resolve())
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
