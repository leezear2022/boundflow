#!/usr/bin/env python3
"""Generate or replay the formal R3-1b1 compiled full-lower artifact."""

# pylint: disable=wrong-import-position,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring,duplicate-code
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.backends.tvm.r3_full_lower_forward import R31B1_EXPORTED_SYMBOLS
from scripts.run_r3_full_lower_forward_worker import WORKER_SCHEMA

ARTIFACT_SCHEMA = "boundflow.r3-1b1-full-lower-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.r3-1b1-full-lower-protocol/v1"
SUMMARY_SCHEMA = "boundflow.r3-1b1-full-lower-summary/v1"
EXPECTED_TRACE_HASH = "a5279f8e76b722dbebd8df23a417f9de7b5d65c4dce5067035627be9137e20bc"
EXPECTED_PLAN_HASH = "39d61775caac6d64a5a2d697073d0caa434d34bb2f054351f474700e9d61910f"
EXPECTED_MODULE_HASH = (
    "003f38c0cccee27cd210014fadda9c7fa8f9b2fae2e93b853be0a3c3101649ba"
)
EXPECTED_DEVICE_SOURCE_HASH = (
    "c4112b4055636259cde16514be7f58145bcfa316256121bdae4f1c60778e1ddf"
)
EXPECTED_RECEIPT_HASH = (
    "b412872d0ab5713c6f20288b65d5a4848bb8d4672eec7af5b8a44aca63159bd2"
)
EXPECTED_CAPTURE_HASH = (
    "f42229dd06be8521865d14af4cc51450ec8bec4e792934b457c276aad50126dc"
)
EXPECTED_MODEL_HASH = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
EXPECTED_CANDIDATE = (
    -0.3708958625793457,
    -0.4221745729446411,
    -0.4737401008605957,
    -0.36605799198150635,
    -0.4408513307571411,
    -0.4903639554977417,
)
EXPECTED_NATIVE = (
    -0.3708932399749756,
    -0.422172486782074,
    -0.4737361669540405,
    -0.3660602569580078,
    -0.44085168838500977,
    -0.49036335945129395,
)
CODE_PATHS = (
    "boundflow/backends/tvm/r3_full_lower_forward.py",
    "boundflow/runtime/r3_full_lower_forward_tir.py",
    "scripts/run_r3_full_lower_forward_worker.py",
    "scripts/run_r3_full_lower_forward_artifact.py",
    "scripts/probe_r3_full_lower_forward_tamper.py",
    "tests/test_r3_full_lower_forward_tir.py",
    "tests/test_r3_full_lower_forward_artifact.py",
)
SIGNED_FILES = (
    "protocol.json",
    "raw.jsonl",
    "summary.json",
    "replay_stdout.txt",
    "README.md",
)


def _canonical(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"R3-1b1 JSON root differs: {path.name}")
    return value


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _code_revision() -> dict[str, str]:
    return {path: _file_hash(REPOSITORY_ROOT / path) for path in CODE_PATHS}


def _validate_worker(record: dict[str, Any]) -> None:
    metric = record.get("metric")
    compilation = record.get("compilation_receipt")
    launch = record.get("launch_receipt")
    environment = record.get("environment")
    if not all(
        isinstance(value, dict) for value in (metric, compilation, launch, environment)
    ):
        raise ValueError("R3-1b1 worker receipt root differs")
    assert isinstance(metric, dict)
    assert isinstance(compilation, dict)
    assert isinstance(launch, dict)
    assert isinstance(environment, dict)
    candidate = tuple(float(value) for value in metric.get("candidate_lower", ()))
    native = tuple(float(value) for value in metric.get("native_lower", ()))
    if (
        record.get("schema_version") != WORKER_SCHEMA
        or record.get("source_capture_sha256") != EXPECTED_CAPTURE_HASH
        or record.get("model_sha256") != EXPECTED_MODEL_HASH
        or record.get("trace_hash") != EXPECTED_TRACE_HASH
        or record.get("production_plan_hash") != EXPECTED_PLAN_HASH
        or record.get("compiled_full_lower") is not True
        or record.get("custom_vjp") is not False
        or record.get("timing_recorded") is not False
        or record.get("performance_claimed") is not False
        or candidate != EXPECTED_CANDIDATE
        or len(native) != 6
        or max(abs(left - right) for left, right in zip(native, EXPECTED_NATIVE)) > 5e-6
        or metric.get("allclose") is not True
        or metric.get("finite") is not True
        or metric.get("sign_exact") is not True
        or float(metric.get("max_abs_diff", 1.0)) > 2e-4
        or float(metric.get("atol", 0.0)) != 2e-4
        or float(metric.get("rtol", 0.0)) != 2e-4
        or compilation.get("module_hash") != EXPECTED_MODULE_HASH
        or compilation.get("device_source_hash") != EXPECTED_DEVICE_SOURCE_HASH
        or compilation.get("trace_hash") != EXPECTED_TRACE_HASH
        or compilation.get("production_plan_hash") != EXPECTED_PLAN_HASH
        or tuple(compilation.get("exported_symbols", ())) != R31B1_EXPORTED_SYMBOLS
        or compilation.get("global_workspace_bytes") != 0
        or compilation.get("tensor_free_module_cache") is not True
        or record.get("compilation_receipt_hash") != EXPECTED_RECEIPT_HASH
        or launch.get("compiled_region") is not True
        or launch.get("coefficient_scratch_count") != 2
        or tuple(launch.get("scratch_capacity_elements", ())) != (18432, 18432)
        or tuple(launch.get("scratch_high_water_elements", ())) != (18432, 18432)
        or len(set(launch.get("scratch_pointers", ()))) != 2
        or launch.get("launch_count") != 15
        or launch.get("dlpack_pointer_count") != 70
        or launch.get("dlpack_pointer_exact_count") != 70
        or launch.get("warm_dynamic_allocated_bytes") != 0
        or launch.get("python_visible_intermediate_coefficient_count") != 0
        or launch.get("fallback_count") != 0
        or launch.get("eager_count") != 0
        or launch.get("native_shadow_count") != 0
        or launch.get("stream_id") != launch.get("tvm_ffi_stream_id")
        or int(launch.get("stream_id", 0)) <= 0
        or launch.get("timing_recorded") is not False
        or launch.get("performance_claimed") is not False
        or environment.get("compute_capability") != "sm_89"
    ):
        raise ValueError("R3-1b1 worker semantic receipt differs")


def _summary(record: dict[str, Any]) -> dict[str, object]:
    _validate_worker(record)
    metric = record["metric"]
    compilation = record["compilation_receipt"]
    launch = record["launch_receipt"]
    result: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "validated-r3-1b1-compiled-full-lower",
        "trace_hash": record["trace_hash"],
        "production_plan_hash": record["production_plan_hash"],
        "module_hash": compilation["module_hash"],
        "device_source_hash": compilation["device_source_hash"],
        "compilation_receipt_hash": record["compilation_receipt_hash"],
        "native_lower": metric["native_lower"],
        "candidate_lower": metric["candidate_lower"],
        "max_abs_diff": metric["max_abs_diff"],
        "sign_exact": metric["sign_exact"],
        "launch_count": launch["launch_count"],
        "coefficient_scratch_count": launch["coefficient_scratch_count"],
        "scratch_capacity_elements": launch["scratch_capacity_elements"],
        "dlpack_pointer_count": launch["dlpack_pointer_count"],
        "dlpack_pointer_exact_count": launch["dlpack_pointer_exact_count"],
        "warm_dynamic_allocated_bytes": launch["warm_dynamic_allocated_bytes"],
        "python_visible_intermediate_coefficient_count": launch[
            "python_visible_intermediate_coefficient_count"
        ],
        "compiled_region": launch["compiled_region"],
        "b1_closed": True,
        "b2_open": True,
        "custom_vjp": False,
        "r3_1_admitted": False,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    result["summary_hash"] = _hash(result)
    return result


def _result(summary: dict[str, Any]) -> dict[str, object]:
    return {
        "status": "replay-passed",
        "evidence_status": summary["status"],
        "max_abs_diff": summary["max_abs_diff"],
        "sign_exact": summary["sign_exact"],
        "launch_count": summary["launch_count"],
        "coefficient_scratch_count": summary["coefficient_scratch_count"],
        "warm_dynamic_allocated_bytes": summary["warm_dynamic_allocated_bytes"],
        "compiled_region": summary["compiled_region"],
        "b2_open": summary["b2_open"],
        "summary_hash": summary["summary_hash"],
        "timing_recorded": False,
        "performance_claimed": False,
    }


def _protocol(source_capture: Path, model: Path) -> dict[str, object]:
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "artifact_schema": ARTIFACT_SCHEMA,
        "source_commit": _git("rev-parse", "HEAD"),
        "code_revision": _code_revision(),
        "source_capture_sha256": _file_hash(source_capture),
        "model_sha256": _file_hash(model),
        "fresh_process_count": 1,
        "comparison": "independent-native-vs-compiled-full-lower",
        "tolerance": {"atol": 2e-4, "rtol": 2e-4, "sign_exact": True},
        "hardware": "NVIDIA GeForce RTX 4060 Laptop GPU / sm_89",
        "compiled_scope": "full-lower-forward-from-objective-seed-through-input-concretize",
        "custom_vjp": False,
        "timing_recorded": False,
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# R3-1b1 compiled full-lower forward\n\n"
        "One fresh subprocess compares the independent eager native oracle against the "
        "15-symbol CUDA TIR full-lower recurrence. Replay hash-checks code and every signed "
        "file, then independently recomputes all semantic gates. This closes only b1; custom "
        "VJP, five-fresh and timing remain closed.\n"
    )


def generate(output: Path, source_capture: Path, model: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"R3-1b1 artifact output already exists: {output}")
    output.mkdir(parents=True)
    protocol = _protocol(source_capture, model)
    completed = subprocess.run(
        (
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/run_r3_full_lower_forward_worker.py"),
            "--source-capture",
            str(source_capture),
            "--model",
            str(model),
        ),
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("R3-1b1 worker stdout differs")
    record = json.loads(lines[0])
    if not isinstance(record, dict):
        raise TypeError("R3-1b1 worker record differs")
    summary = _summary(record)
    result = _result(summary)
    _write_json(output / "protocol.json", protocol)
    (output / "raw.jsonl").write_text(_canonical(record) + "\n", encoding="utf-8")
    _write_json(output / "summary.json", summary)
    (output / "replay_stdout.txt").write_text(
        _canonical(result) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(_readme(), encoding="utf-8")
    manifest = {
        "schema_version": ARTIFACT_SCHEMA,
        "files": {name: _file_hash(output / name) for name in SIGNED_FILES},
    }
    manifest["manifest_hash"] = _hash(manifest)
    _write_json(output / "manifest.json", manifest)
    replayed = replay(output)
    if replayed != result:
        raise RuntimeError("R3-1b1 generated replay differs")
    return replayed


def replay(artifact: Path) -> dict[str, object]:
    manifest = _load_json(artifact / "manifest.json")
    manifest_hash = manifest.get("manifest_hash")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest_hash != _hash(unsigned)
        or not isinstance(files, dict)
        or set(files) != set(SIGNED_FILES)
        or any(_file_hash(artifact / name) != digest for name, digest in files.items())
    ):
        raise ValueError("R3-1b1 artifact manifest differs")
    protocol = _load_json(artifact / "protocol.json")
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("artifact_schema") != ARTIFACT_SCHEMA
        or protocol.get("code_revision") != _code_revision()
        or protocol.get("source_capture_sha256") != EXPECTED_CAPTURE_HASH
        or protocol.get("model_sha256") != EXPECTED_MODEL_HASH
        or protocol.get("fresh_process_count") != 1
        or protocol.get("custom_vjp") is not False
        or protocol.get("timing_recorded") is not False
        or protocol.get("performance_claimed") is not False
    ):
        raise ValueError("R3-1b1 artifact protocol differs")
    raw_lines = (artifact / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    if len(raw_lines) != 1:
        raise ValueError("R3-1b1 artifact raw cardinality differs")
    record = json.loads(raw_lines[0])
    if not isinstance(record, dict):
        raise TypeError("R3-1b1 artifact raw record differs")
    expected_summary = _summary(record)
    summary = _load_json(artifact / "summary.json")
    if summary != expected_summary:
        raise ValueError("R3-1b1 artifact semantic recomputation differs")
    result = _result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != _canonical(
        result
    ) + "\n":
        raise ValueError("R3-1b1 artifact replay stdout differs")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--source-capture", type=Path, required=True)
    generate_parser.add_argument("--model", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        result = generate(args.output, args.source_capture, args.model)
    else:
        result = replay(args.artifact)
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
