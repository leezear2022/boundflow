#!/usr/bin/env python3
"""Generate or replay the first native real-network compiler-IR artifact."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.frontends.plain_crown_bound_ir import tensor_content_hash
from boundflow.ir.bound import BoundOpKind
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.abcrown_adapter import (
    bind_intermediate_bounds,
    deserialize_intermediate_bounds,
    file_sha256,
    intermediate_bounds_sha256,
)
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_verifier_ir_integration import (
    NATIVE_PLAIN_CROWN_COMPILER_VERSION,
    compile_native_plain_crown_query,
    execute_native_plain_crown_query,
)
from boundflow.runtime.task_executor import InputSpec

ARTIFACT_SCHEMA_VERSION = "boundflow.native-real-network-ir-artifact/v1"
ARTIFACT_PAYLOAD_SCHEMA_VERSION = "boundflow.native-real-network-ir-payload/v1"
SOURCE_MANIFEST_SCHEMA_VERSION = "boundflow.pr14-initial-crown-replay/v1"
SOURCE_PAYLOAD_SCHEMA_VERSION = "boundflow.pr14-initial-crown-payload/v2"
QUERY_ID = "vnncomp21-resnet2b-prop0-native-ir1"
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
VNNLIB_SHA256 = "89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff"
VNNCOMP_COMMIT = "90419aadcf06cf543ce5c1706cae1059dc9fa6cf"
ABCROWN_COMMIT = "e5c7e17bf0488843acb77b7519f59876717a49f4"
INTERMEDIATE_BOUNDS_SHA256 = (
    "d51615b04dfb205afd67d2c21680ece4ca92f693157da1e32c7f8202a8e08cf1"
)
MEMORY_BUDGET_BYTES = 1 << 30
ATOL = 2e-4
RTOL = 2e-4
EXPECTED_PRIMAL_OPS = (
    "conv2d",
    "relu",
    "conv2d",
    "relu",
    "conv2d",
    "conv2d",
    "add",
    "relu",
    "conv2d",
    "relu",
    "conv2d",
    "add",
    "relu",
    "flatten",
    "linear",
    "relu",
    "linear",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--source-manifest", type=Path, required=True)
    generate.add_argument("--source-payload", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


# pylint: disable-next=too-many-branches
def _validate_source_capture(
    manifest: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("source capture manifest schema differs")
    if manifest.get("payload_schema_version") != SOURCE_PAYLOAD_SCHEMA_VERSION:
        raise ValueError("source capture payload schema declaration differs")
    if manifest.get("status") != "ok":
        raise ValueError("source capture did not pass its semantic gate")
    if manifest.get("abcrown_commit") != ABCROWN_COMMIT:
        raise ValueError("source capture alpha-beta-CROWN commit differs")
    if manifest.get("model_sha256") != MODEL_SHA256:
        raise ValueError("source capture model digest differs")
    if manifest.get("vnnlib_sha256") != VNNLIB_SHA256:
        raise ValueError("source capture VNNLIB digest differs")
    if payload.get("schema_version") != SOURCE_PAYLOAD_SCHEMA_VERSION:
        raise ValueError("source capture payload schema differs")
    capture = _mapping(manifest.get("capture"), "source capture")
    if str(capture.get("method", "")).lower() != "crown":
        raise ValueError("source capture method is not plain CROWN")
    if capture.get("solver_phase") != "alpha_crown_initialization":
        raise ValueError("source capture solver phase differs")
    if capture.get("intermediate_bound_count") != 6:
        raise ValueError("source capture intermediate-bound count differs")
    if capture.get("intermediate_bounds_hash") != INTERMEDIATE_BOUNDS_SHA256:
        raise ValueError("source capture intermediate-bound digest differs")
    if capture.get("intermediate_bound_source") != "external_verifier":
        raise ValueError("source capture intermediate-bound owner differs")
    if capture.get("relu_lower_slope_policy") != "adaptive":
        raise ValueError("source capture ReLU lower-slope policy differs")
    eager = _mapping(
        _mapping(manifest.get("boundflow"), "source BoundFlow results").get(
            "pytorch_eager"
        ),
        "source BoundFlow eager result",
    )
    comparison = _mapping(
        eager.get("lower_vs_external"), "source eager lower comparison"
    )
    if comparison.get("allclose") is not True or comparison.get("sign_agreement") != 9:
        raise ValueError("source capture eager semantics did not pass")
    bounds = deserialize_intermediate_bounds(
        _mapping(payload.get("external_intermediate_bounds"), "intermediate bounds")
    )
    if (
        len(bounds) != 6
        or intermediate_bounds_sha256(bounds) != INTERMEDIATE_BOUNDS_SHA256
    ):
        raise ValueError("source capture external intervals differ")
    tensors = _payload_tensors(payload)
    expected_shapes = {
        "input_lower": (1, 3, 32, 32),
        "input_upper": (1, 3, 32, 32),
        "linear_spec_c": (1, 9, 10),
        "external_lower": (1, 9),
    }
    for name, shape in expected_shapes.items():
        if tuple(tensors[name].shape) != shape or tensors[name].dtype != torch.float32:
            raise ValueError(f"source capture tensor contract differs: {name}")


def _payload_tensors(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    found: dict[str, torch.Tensor] = {}
    for name in ("input_lower", "input_upper", "linear_spec_c", "external_lower"):
        value = payload.get(name)
        if not torch.is_tensor(value):
            raise TypeError(f"artifact payload {name} must be a tensor")
        found[name] = value.detach().cpu().contiguous()
    if not bool((found["input_lower"] <= found["input_upper"]).all()):
        raise ValueError("artifact input lower exceeds upper")
    if not all(bool(torch.isfinite(value).all()) for value in found.values()):
        raise ValueError("artifact payload tensors must be finite")
    return found


def _curate_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    tensors = _payload_tensors(source)
    external_bounds = _mapping(
        source.get("external_intermediate_bounds"), "external intermediate bounds"
    )
    deserialize_intermediate_bounds(external_bounds)
    return {
        "schema_version": ARTIFACT_PAYLOAD_SCHEMA_VERSION,
        **{name: value.clone() for name, value in tensors.items()},
        "external_intermediate_bounds": dict(external_bounds),
    }


def _run_native(  # pylint: disable=too-many-locals
    model: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("native artifact model digest differs")
    if payload.get("schema_version") != ARTIFACT_PAYLOAD_SCHEMA_VERSION:
        raise ValueError("native artifact payload schema differs")
    tensors = _payload_tensors(payload)
    external_bounds = deserialize_intermediate_bounds(
        _mapping(payload.get("external_intermediate_bounds"), "intermediate bounds")
    )
    if (
        len(external_bounds) != 6
        or intermediate_bounds_sha256(external_bounds) != INTERMEDIATE_BOUNDS_SHA256
    ):
        raise ValueError("native artifact requires six external ReLU intervals")

    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    legacy_module = plan_interval_ibp_v0(program)
    primal_ops = tuple(op.op_type for op in legacy_module.get_entry_task().ops)
    if primal_ops != EXPECTED_PRIMAL_OPS:
        raise ValueError("native artifact primal topology differs")
    if len(program.graph.inputs) != 1:
        raise ValueError("native artifact model must have exactly one input")
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors["input_lower"],
        upper=tensors["input_upper"],
    )
    interval_env, local_relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    relu_pre = bind_intermediate_bounds(external_bounds, local_relu_pre)
    compilation = compile_native_plain_crown_query(
        legacy_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=tensors["linear_spec_c"],
        intermediate_bounds_hash=INTERMEDIATE_BOUNDS_SHA256,
        query_id=QUERY_ID,
        available_memory_bytes=MEMORY_BUDGET_BYTES,
    )
    result, trace = execute_native_plain_crown_query(
        compilation,
        legacy_task_module=legacy_module,
        input_spec=input_spec,
        relu_pre=relu_pre,
        linear_spec_C=tensors["linear_spec_c"],
    )
    expected = tensors["external_lower"].to(result.lower)
    difference = (result.lower - expected).abs()
    sign_match = (result.lower >= 0) == (expected >= 0)
    comparison = {
        "allclose": bool(torch.allclose(result.lower, expected, atol=ATOL, rtol=RTOL)),
        "max_abs_diff": float(difference.max().item()),
        "sign_agreement": int(sign_match.sum().item()),
        "sign_total": int(sign_match.numel()),
        "atol": ATOL,
        "rtol": RTOL,
        "native_lower_sha256": tensor_content_hash(result.lower),
        "external_lower_sha256": tensor_content_hash(expected),
    }
    if not comparison["allclose"] or comparison["sign_agreement"] != 9:
        raise ValueError(
            "native real-network semantics differ from the external oracle"
        )
    if not bool(
        torch.isfinite(result.lower).all()
        and torch.isfinite(result.upper).all()
        and (result.lower <= result.upper).all()
    ):
        raise ValueError("native real-network result is malformed")

    bound_kinds = Counter(op.kind.value for op in compilation.bound_module.graph.ops)
    task_kinds = Counter(task.kind.value for task in compilation.task_module.tasks)
    action_kinds = Counter(action.kind.value for action in compilation.schedule.actions)
    arena_bytes = max(
        buffer.offset_bytes + buffer.size_bytes
        for buffer in compilation.schedule.buffers
    )
    result_payload: dict[str, Any] = {
        "compiler_version": NATIVE_PLAIN_CROWN_COMPILER_VERSION,
        "query_id": QUERY_ID,
        "ir_hashes": compilation.hashes(),
        "primal": {
            "op_count": len(primal_ops),
            "op_kinds": list(primal_ops),
        },
        "bound_ir": {
            "op_count": len(compilation.bound_module.graph.ops),
            "op_kinds": dict(sorted(bound_kinds.items())),
            "external_verifier_call_count": bound_kinds.get(
                BoundOpKind.EXTERNAL_VERIFIER_CALL.value, 0
            ),
        },
        "plan_ir": {
            "region_candidate_count": len(compilation.template.region_candidates),
            "selected_region_count": len(compilation.instance.region_decisions),
            "backend_candidate_count": len(compilation.template.backend_candidates),
            "batch_candidate_count": len(compilation.template.batch_candidates),
            "storage_candidate_count": len(compilation.template.storage_candidates),
            "materialization_candidate_count": len(
                compilation.template.materialization_candidates
            ),
        },
        "task_ir": {
            "task_count": len(compilation.task_module.tasks),
            "task_kinds": dict(sorted(task_kinds.items())),
            "external_verifier_call_count": task_kinds.get("external_verifier_call", 0),
        },
        "schedule_ir": {
            "action_count": len(compilation.schedule.actions),
            "action_kinds": dict(sorted(action_kinds.items())),
            "buffer_count": len(compilation.schedule.buffers),
            "arena_bytes": arena_bytes,
        },
        "execution": {
            "event_count": len(trace.events),
            "trace_hash": trace.stable_hash(),
            "comparison": comparison,
            "finite": True,
            "ordered": True,
        },
    }
    if (
        result_payload["bound_ir"]["external_verifier_call_count"] != 0
        or result_payload["task_ir"]["external_verifier_call_count"] != 0
        or action_kinds.get("launch", 0) != len(compilation.bound_module.graph.ops)
    ):
        raise ValueError("native compiler ownership gate failed")
    return result_payload


def _tensor_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: tensor_content_hash(value)
        for name, value in sorted(_payload_tensors(payload).items())
    }


def _generate(args: argparse.Namespace) -> None:
    model = _require_file(args.model, "ONNX model")
    source_manifest_path = _require_file(args.source_manifest, "source manifest")
    source_payload_path = _require_file(args.source_payload, "source payload")
    source_manifest = _load_json(source_manifest_path)
    source_payload = torch.load(
        source_payload_path, map_location="cpu", weights_only=True
    )
    if not isinstance(source_payload, Mapping):
        raise TypeError("source payload root must be a mapping")
    _validate_source_capture(source_manifest, source_payload)

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _curate_payload(source_payload)
    payload_path = output_dir / "payload.pt"
    torch.save(payload, payload_path)
    result = _run_native(model, payload)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "inputs": {
            "model_sha256": MODEL_SHA256,
            "vnnlib_sha256": VNNLIB_SHA256,
            "vnncomp_commit": VNNCOMP_COMMIT,
            "abcrown_commit": ABCROWN_COMMIT,
            "source_manifest_sha256": file_sha256(source_manifest_path),
            "source_payload_sha256": file_sha256(source_payload_path),
            "intermediate_bounds_sha256": INTERMEDIATE_BOUNDS_SHA256,
        },
        "payload": {
            "schema_version": ARTIFACT_PAYLOAD_SCHEMA_VERSION,
            "sha256": file_sha256(payload_path),
            "tensor_sha256": _tensor_hashes(payload),
        },
        "result": result,
        "limitations": [
            "CPU correctness and compiler-ownership evidence only",
            "external intermediate bounds remain the calibrated semantic oracle",
            "one dense storage candidate and one full-query batch candidate",
            "no materialization alternative, GPU backend, or performance claim",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": "generate",
                "output_dir": str(output_dir),
                "ir_hashes": result["ir_hashes"],
            },
            sort_keys=True,
        )
    )


def _replay(args: argparse.Namespace) -> None:
    model = _require_file(args.model, "ONNX model")
    artifact_dir = args.artifact_dir.expanduser().resolve()
    manifest_path = _require_file(artifact_dir / "manifest.json", "manifest")
    payload_path = _require_file(artifact_dir / "payload.pt", "payload")
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("native artifact manifest schema differs")
    if (
        manifest.get("status") != "ok"
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("native artifact status/claim contract differs")
    inputs = _mapping(manifest.get("inputs"), "artifact inputs")
    expected_inputs = {
        "model_sha256": MODEL_SHA256,
        "vnnlib_sha256": VNNLIB_SHA256,
        "vnncomp_commit": VNNCOMP_COMMIT,
        "abcrown_commit": ABCROWN_COMMIT,
        "intermediate_bounds_sha256": INTERMEDIATE_BOUNDS_SHA256,
    }
    for name, expected in expected_inputs.items():
        if inputs.get(name) != expected:
            raise ValueError(f"native artifact input identity differs: {name}")
    payload_manifest = _mapping(manifest.get("payload"), "artifact payload")
    if payload_manifest.get("sha256") != file_sha256(payload_path):
        raise ValueError("native artifact payload file digest differs")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("native artifact payload root must be a mapping")
    if payload_manifest.get("tensor_sha256") != _tensor_hashes(payload):
        raise ValueError("native artifact payload tensor digests differ")
    actual = _run_native(model, payload)
    if actual != manifest.get("result"):
        raise ValueError("native artifact semantic/IR replay differs")
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": "replay",
                "artifact_dir": str(artifact_dir),
                "ir_hashes": actual["ir_hashes"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    """Dispatch artifact generation or deterministic semantic replay."""

    args = _parse_args()
    if args.command == "generate":
        _generate(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
