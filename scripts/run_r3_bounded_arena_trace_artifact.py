#!/usr/bin/env python3
"""Generate or replay the R3-1b0 exact trace/liveness artifact."""

# pylint: disable=wrong-import-position,too-many-locals,missing-function-docstring
# pylint: disable=duplicate-code,too-many-boolean-expressions

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_structured_owner_custom_backward import (
    compile_r31_full_region_plan_v1,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
)
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

ARTIFACT_SCHEMA = "boundflow.r3-1b0-trace-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.r3-1b0-trace-protocol/v1"
SUMMARY_SCHEMA = "boundflow.r3-1b0-trace-summary/v1"
EXPECTED_TRACE_HASH = "a5279f8e76b722dbebd8df23a417f9de7b5d65c4dce5067035627be9137e20bc"
EXPECTED_SOURCE_HASH = (
    "f510204e8f82a4f2defdcdca694e1ae133eb73978f9c4a7f8b26a6cf0406743e"
)
EXPECTED_TOPOLOGY_HASH = (
    "8ebd62ca506db4a2e59b752c5db78635ac125b65100a72dafc9b24dd137cce0b"
)
EXPECTED_PRODUCTION_PLAN_HASH = (
    "39d61775caac6d64a5a2d697073d0caa434d34bb2f054351f474700e9d61910f"
)
CODE_PATHS = (
    "boundflow/ir/r3_bounded_arena.py",
    "boundflow/runtime/r3_bounded_arena_trace_compiler.py",
    "boundflow/runtime/r3_structured_owner_custom_backward.py",
    "scripts/run_r3_bounded_arena_trace_artifact.py",
    "tests/test_r3_bounded_arena_trace_compiler.py",
)
ARTIFACT_FILES = (
    "protocol.json",
    "trace.json",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"R3-1b0 JSON root differs: {path.name}")
    return value


def _git(*args: str) -> str:
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
    return {name: _file_sha256(REPOSITORY_ROOT / name) for name in CODE_PATHS}


def _trace_payload(source_capture: Path, model: Path) -> dict[str, object]:
    raw = torch.load(source_capture, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("R3-1b0 source capture root differs")
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    production_plan = compile_r31_full_region_plan_v1(
        module, snapshot, mapping, TOPOLOGY
    )
    trace = compile_r31b_bounded_arena_trace_v1(program, module, production_plan)
    payload = trace.identity_payload()
    payload["trace_hash"] = trace.stable_hash()
    _validate_trace_payload(payload)
    return payload


def _validate_trace_payload(payload: dict[str, Any]) -> None:
    expected_fields = {
        "schema_version",
        "source_hash",
        "topology_hash",
        "production_plan_hash",
        "steps",
        "scratch_slot_count",
        "scratch_capacity_elements",
        "start_node_id",
        "domain_count",
        "spec_count",
        "dtype",
        "compiled_region",
        "timing_recorded",
        "performance_claimed",
        "trace_hash",
    }
    semantic = {name: payload[name] for name in payload if name != "trace_hash"}
    steps = payload.get("steps")
    if (
        set(payload) != expected_fields
        or payload.get("trace_hash") != _canonical_hash(semantic)
        or payload.get("trace_hash") != EXPECTED_TRACE_HASH
        or payload.get("source_hash") != EXPECTED_SOURCE_HASH
        or payload.get("topology_hash") != EXPECTED_TOPOLOGY_HASH
        or payload.get("production_plan_hash") != EXPECTED_PRODUCTION_PLAN_HASH
        or not isinstance(steps, list)
        or len(steps) != 12
        or payload.get("scratch_slot_count") != 2
        or payload.get("scratch_capacity_elements") != 18432
        or payload.get("compiled_region") is not False
        or payload.get("timing_recorded") is not False
        or payload.get("performance_claimed") is not False
    ):
        raise ValueError("R3-1b0 frozen trace differs")


def _summary(trace: dict[str, Any]) -> dict[str, object]:
    _validate_trace_payload(trace)
    steps = trace["steps"]
    residuals = [step for step in steps if step["kind"] == "residual_region"]
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "validated-r3-1b0-trace-liveness",
        "trace_hash": trace["trace_hash"],
        "source_hash": trace["source_hash"],
        "topology_hash": trace["topology_hash"],
        "production_plan_hash": trace["production_plan_hash"],
        "step_count": len(steps),
        "residual_region_count": len(residuals),
        "scratch_slot_count": trace["scratch_slot_count"],
        "scratch_capacity_elements": trace["scratch_capacity_elements"],
        "scratch_capacity_bytes_per_slot": trace["scratch_capacity_elements"] * 4,
        "maximum_coefficient_shape": [6, 1, 3, 32, 32],
        "b1_open": True,
        "compiled_region": False,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = _canonical_hash(summary)
    return summary


def _result(summary: dict[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "trace_hash": summary["trace_hash"],
        "step_count": summary["step_count"],
        "residual_region_count": summary["residual_region_count"],
        "scratch_slot_count": summary["scratch_slot_count"],
        "scratch_capacity_bytes_per_slot": summary["scratch_capacity_bytes_per_slot"],
        "b1_open": summary["b1_open"],
        "compiled_region": False,
        "summary_hash": summary["summary_hash"],
        "timing_recorded": False,
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# R3-1b0 Exact Trace/Liveness\n\n"
        "This contract-only artifact freezes the 12-step ResNet2B reverse recurrence, "
        "two fused residual regions, and a two-slot/73,728-byte-per-slot liveness "
        "schedule. It contains no compiled module, CUDA execution, timing, or "
        "performance claim.\n"
    )


def _protocol(source_capture: Path, model: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": _git("rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "source_capture_sha256": _file_sha256(source_capture),
        "model_sha256": _file_sha256(model),
        "expected_trace_hash": EXPECTED_TRACE_HASH,
        "expected_source_hash": EXPECTED_SOURCE_HASH,
        "expected_topology_hash": EXPECTED_TOPOLOGY_HASH,
        "expected_production_plan_hash": EXPECTED_PRODUCTION_PLAN_HASH,
        "run_kind": "contract-only",
        "compiled_region": False,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    payload["protocol_hash"] = _canonical_hash(payload)
    return payload


def _all_files(root: Path) -> dict[str, str]:
    return {name: _file_sha256(root / name) for name in ARTIFACT_FILES}


def _generate(args: argparse.Namespace) -> dict[str, object]:
    if _git("status", "--porcelain=v1", "--", *CODE_PATHS):
        raise ValueError("R3-1b0 formal code paths must be committed")
    output = args.artifact.resolve()
    if output.exists():
        raise FileExistsError(f"R3-1b0 artifact exists: {output}")
    source_capture = args.source_capture.resolve()
    model = args.model.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.incomplete-", dir=output.parent
    ) as temporary:
        root = Path(temporary)
        protocol = _protocol(source_capture, model)
        trace = _trace_payload(source_capture, model)
        summary = _summary(trace)
        result = _result(summary)
        _write_json(root / "protocol.json", protocol)
        _write_json(root / "trace.json", trace)
        _write_json(root / "summary.json", summary)
        (root / "replay_stdout.txt").write_text(
            _canonical_json(result) + "\n", encoding="utf-8"
        )
        (root / "README.md").write_text(_readme(), encoding="utf-8")
        manifest: dict[str, object] = {
            "schema_version": ARTIFACT_SCHEMA,
            "source_git_head": _git("rev-parse", "HEAD"),
            "code_revision": _code_revision(),
            "protocol_hash": protocol["protocol_hash"],
            "summary_hash": summary["summary_hash"],
            "files": _all_files(root),
            "compiled_region": False,
            "timing_recorded": False,
            "performance_claimed": False,
        }
        manifest["manifest_hash"] = _canonical_hash(manifest)
        _write_json(root / "manifest.json", manifest)
        shutil.move(root, output)
    _verify_static(output)
    return result


def _verify_static(artifact: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(artifact / "manifest.json")
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_hash", None)
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or claimed != _canonical_hash(unsigned)
        or not isinstance(files, dict)
        or set(files) != set(ARTIFACT_FILES)
        or any(files[name] != _file_sha256(artifact / name) for name in files)
        or manifest.get("compiled_region") is not False
        or manifest.get("timing_recorded") is not False
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("R3-1b0 artifact manifest differs")
    protocol = _load_json(artifact / "protocol.json")
    unsigned_protocol = dict(protocol)
    protocol_hash = unsigned_protocol.pop("protocol_hash", None)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol_hash != _canonical_hash(unsigned_protocol)
        or manifest.get("protocol_hash") != protocol_hash
        or protocol.get("expected_trace_hash") != EXPECTED_TRACE_HASH
    ):
        raise ValueError("R3-1b0 artifact protocol differs")
    return manifest, protocol


def _replay(artifact: Path) -> dict[str, object]:
    manifest, _protocol_value = _verify_static(artifact)
    trace = _load_json(artifact / "trace.json")
    expected = _summary(trace)
    observed = _load_json(artifact / "summary.json")
    if observed != expected or manifest.get("summary_hash") != expected["summary_hash"]:
        raise ValueError("R3-1b0 artifact summary differs")
    result = _result(expected)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != (
        _canonical_json(result) + "\n"
    ):
        raise ValueError("R3-1b0 replay stdout differs")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--source-capture", type=Path)
    parser.add_argument("--model", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.replay:
        result = _replay(args.artifact.resolve())
    else:
        if args.source_capture is None or args.model is None:
            raise ValueError("R3-1b0 generation requires source capture and model")
        result = _generate(args)
    print(_canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
