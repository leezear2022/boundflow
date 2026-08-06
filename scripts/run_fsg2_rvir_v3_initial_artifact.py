#!/usr/bin/env python3
"""Generate or replay the real ResNet RVIR-v3 native initial-CROWN artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches,duplicate-code
# pylint: disable=missing-function-docstring

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
from boundflow.runtime.abcrown_adapter import deserialize_intermediate_bounds
from boundflow.runtime.rvir_v3_native_crown import (
    NativePlainCrownRVIRV3Backend,
    build_native_initial_crown_payload,
)
from boundflow.runtime.rvir_v3_replacement import execute_rvir_v3_replacement

SCHEMA_VERSION = "boundflow.fsg2-rvir-v3-initial-artifact/v1"
PAYLOAD_FILE = "payload.pt"
ARTIFACT_FILES = (
    PAYLOAD_FILE,
    "source.json",
    "execution.json",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)
CODE_PATHS = (
    "boundflow/runtime/rvir_v3_replacement.py",
    "boundflow/runtime/rvir_v3_native_crown.py",
    "scripts/run_fsg2_rvir_v3_initial_artifact.py",
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


def _code_revision() -> dict[str, str]:
    root = _repo_root()
    return {name: file_sha256(root / name) for name in CODE_PATHS}


def _code_paths_clean() -> bool:
    return not _git_value("status", "--porcelain=v1", "--", *CODE_PATHS)


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--source-payload", type=Path, required=True)
    generate.add_argument("--source-manifest", type=Path, required=True)
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--artifact-dir", type=Path, required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def _plain_tensor(value: object, label: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"FSG2 source {label} must be a tensor")
    tensor = value.detach().cpu().contiguous()
    if type(tensor) is not torch.Tensor:  # pylint: disable=unidiomatic-typecheck
        tensor = tensor.as_subclass(torch.Tensor)
    return tensor


def _execute(
    payload_path: Path, model: Path
) -> tuple[dict[str, object], dict[str, object]]:
    raw = torch.load(payload_path, weights_only=True, map_location="cpu")
    if not isinstance(raw, Mapping):
        raise TypeError("FSG2 source payload must be a mapping")
    intermediate_raw = raw.get("external_intermediate_bounds")
    if not isinstance(intermediate_raw, Mapping):
        raise ValueError("FSG2 source omits executable intermediate bounds")
    intermediate = deserialize_intermediate_bounds(intermediate_raw)
    input_lower = _plain_tensor(raw.get("input_lower"), "input lower")
    input_upper = _plain_tensor(raw.get("input_upper"), "input upper")
    linear_spec = _plain_tensor(raw.get("linear_spec_c"), "linear spec")
    expected_lower = _plain_tensor(raw.get("external_lower"), "external lower")
    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    if len(program.graph.inputs) != 1:
        raise ValueError("FSG2 native replacement requires one model input")
    module = plan_interval_ibp_v0(program)
    payload = build_native_initial_crown_payload(
        query_id="fsg2-rvir-v3-resnet-initial-00000000",
        sequence_number=0,
        parent_query_id=None,
        module=module,
        input_lower=input_lower,
        input_upper=input_upper,
        linear_spec_c=linear_spec,
        intermediate_lowers=[item.lower for item in intermediate],
        intermediate_uppers=[item.upper for item in intermediate],
        requested_polarities=("lower",),
    )
    backend = NativePlainCrownRVIRV3Backend(module, program.graph.inputs[0])
    first = execute_rvir_v3_replacement(payload, backend)
    second = execute_rvir_v3_replacement(payload, backend)
    if first.lower is None or second.lower is None or backend.last_ir_hashes is None:
        raise ValueError("FSG2 native replacement result is incomplete")
    difference = (first.lower - expected_lower).abs()
    sign = (first.lower >= 0) == (expected_lower >= 0)
    execution = {
        "query_id": first.query_id,
        "sequence_number": first.sequence_number,
        "backend_id": first.backend_id,
        "payload_hash": first.payload_hash,
        "result_hash": first.result_hash,
        "repeat_result_hash": second.result_hash,
        "ir_hashes": dict(backend.last_ir_hashes),
        "result_shape": list(first.lower.shape),
        "external_shape": list(expected_lower.shape),
        "lower_max_abs_diff": float(difference.max().item()),
        "lower_allclose": bool(
            torch.allclose(first.lower, expected_lower, atol=2e-4, rtol=2e-4)
        ),
        "sign_agreement": int(sign.sum().item()),
        "sign_total": int(sign.numel()),
        "intermediate_bound_count": len(intermediate),
        "replacement_dispatch_count": first.replacement_dispatch_count,
        "original_callback_count": first.original_callback_count,
        "fallback_dispatch_count": first.fallback_dispatch_count,
        "mutation_receipt_count": len(first.mutations),
        "performance_claimed": False,
    }
    passed = (
        execution["result_hash"] == execution["repeat_result_hash"]
        and execution["lower_allclose"] is True
        and execution["sign_agreement"] == execution["sign_total"]
        and execution["intermediate_bound_count"] == 6
        and execution["replacement_dispatch_count"] == 1
        and execution["original_callback_count"] == 0
        and execution["fallback_dispatch_count"] == 0
        and len(backend.last_ir_hashes) == 5
    )
    summary = {
        "status": "validated_reduced_initial_replacement" if passed else "no_go",
        "workload_id": "cifar10_resnet:000",
        "phase": "initial_crown",
        "replacement_coverage": "initial_crown_only",
        "lower_max_abs_diff": execution["lower_max_abs_diff"],
        "sign_agreement": execution["sign_agreement"],
        "sign_total": execution["sign_total"],
        "original_callback_count": 0,
        "fallback_dispatch_count": 0,
        "alpha_beta_split_replacement": "not_admitted",
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    if not passed:
        raise ValueError("FSG2 real initial replacement gate failed")
    return execution, summary


def _replay_result(summary: Mapping[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "replacement_coverage": summary["replacement_coverage"],
        "summary_hash": summary["summary_hash"],
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# FSG2 RVIR-v3 Real Initial-CROWN Replacement\n\n"
        "This artifact proves no-provider-callback native replacement for one frozen "
        "ResNet initial-CROWN call. Alpha/beta/split replacement remains unadmitted. "
        "It makes no performance claim.\n"
    )


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if not _code_paths_clean():
        raise ValueError("FSG2 code paths must be clean before formal generation")
    artifact_dir = args.artifact_dir.resolve()
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source_payload = args.source_payload.resolve()
    source_manifest_path = args.source_manifest.resolve()
    model = args.model.resolve()
    if (
        not source_payload.is_file()
        or not source_manifest_path.is_file()
        or not model.is_file()
    ):
        raise FileNotFoundError("FSG2 source payload, manifest, or model is missing")
    source_manifest = _load_json(source_manifest_path)
    if source_manifest.get("model_sha256") != file_sha256(model):
        raise ValueError("FSG2 source model digest differs")
    shutil.copy2(source_payload, artifact_dir / PAYLOAD_FILE)
    source = {
        "source_schema_version": source_manifest.get("schema_version"),
        "source_payload_sha256": file_sha256(source_payload),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "model_sha256": file_sha256(model),
        "vnnlib_sha256": source_manifest.get("vnnlib_sha256"),
        "abcrown_commit": source_manifest.get("abcrown_commit"),
        "performance_claimed": False,
    }
    _write_json(artifact_dir / "source.json", source)
    execution, summary = _execute(artifact_dir / PAYLOAD_FILE, model)
    _write_json(artifact_dir / "execution.json", execution)
    _write_json(artifact_dir / "summary.json", summary)
    replay_result = _replay_result(summary)
    (artifact_dir / "replay_stdout.txt").write_text(
        canonical_json(replay_result) + "\n", encoding="utf-8"
    )
    (artifact_dir / "README.md").write_text(_readme(), encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source_git_head": _git_value("rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "files": {name: file_sha256(artifact_dir / name) for name in ARTIFACT_FILES},
        "summary_hash": summary["summary_hash"],
        "status": summary["status"],
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact_dir / "manifest.json", manifest)
    return replay_result


def _replay(args: argparse.Namespace) -> dict[str, object]:
    artifact_dir = args.artifact_dir.resolve()
    model = args.model.resolve()
    manifest = _load_json(artifact_dir / "manifest.json")
    semantic_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("manifest_hash") != canonical_hash(semantic_manifest)
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("FSG2 manifest envelope differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(ARTIFACT_FILES):
        raise ValueError("FSG2 artifact file inventory differs")
    for name, digest in files.items():
        if file_sha256(artifact_dir / name) != digest:
            raise ValueError("FSG2 artifact file digest differs")
    source = _load_json(artifact_dir / "source.json")
    if source.get("model_sha256") != file_sha256(model):
        raise ValueError("FSG2 replay model digest differs")
    execution, summary = _execute(artifact_dir / PAYLOAD_FILE, model)
    if _load_json(artifact_dir / "execution.json") != execution:
        raise ValueError("FSG2 execution semantic replay differs")
    if _load_json(artifact_dir / "summary.json") != summary:
        raise ValueError("FSG2 summary semantic replay differs")
    if manifest.get("summary_hash") != summary["summary_hash"]:
        raise ValueError("FSG2 manifest summary projection differs")
    result = _replay_result(summary)
    if (artifact_dir / "replay_stdout.txt").read_text(
        encoding="utf-8"
    ) != canonical_json(result) + "\n":
        raise ValueError("FSG2 replay stdout differs")
    if (artifact_dir / "README.md").read_text(encoding="utf-8") != _readme():
        raise ValueError("FSG2 README differs")
    return result


def main() -> None:
    args = _parse_args()
    result = _generate(args) if args.command == "generate" else _replay(args)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
