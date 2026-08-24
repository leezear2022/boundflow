#!/usr/bin/env python3
"""Generate or replay the formal R3-1b2 compiled P-alpha VJP artifact."""

# pylint: disable=wrong-import-position,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring,duplicate-code,protected-access
# pylint: disable=too-many-boolean-expressions,too-many-branches

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.backends.tvm.r3_p_alpha_vjp import R31B2_EXPORTED_SYMBOLS
from scripts.run_r3_compiled_p_alpha_vjp_worker import WORKER_SCHEMA

ARTIFACT_SCHEMA = "boundflow.r3-1b2-compiled-p-alpha-vjp-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.r3-1b2-compiled-p-alpha-vjp-protocol/v1"
SUMMARY_SCHEMA = "boundflow.r3-1b2-compiled-p-alpha-vjp-summary/v1"
EXPECTED_TRACE_HASH = "a5279f8e76b722dbebd8df23a417f9de7b5d65c4dce5067035627be9137e20bc"
EXPECTED_PLAN_HASH = "39d61775caac6d64a5a2d697073d0caa434d34bb2f054351f474700e9d61910f"
EXPECTED_B1_MODULE_HASH = (
    "003f38c0cccee27cd210014fadda9c7fa8f9b2fae2e93b853be0a3c3101649ba"
)
EXPECTED_B2_MODULE_HASH = (
    "3871bf0e42ec9ce129d32bb408a5e9320d51026da6998aa81ebf0415822be575"
)
EXPECTED_B2_DEVICE_HASH = (
    "842cb3f28c66ec013a9a78aded3741ed63f36935f0183454e967fdf606413fd8"
)
EXPECTED_CAPTURE_HASH = (
    "f42229dd06be8521865d14af4cc51450ec8bec4e792934b457c276aad50126dc"
)
EXPECTED_MODEL_HASH = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
EXPECTED_CANDIDATE_LOWER_HASH = (
    "caa90002a44116444938340a139f4cb944e5d56322add9cecd835dc96de57cbb"
)
EXPECTED_CANDIDATE_GRADIENT_HASH = (
    "59a35857595de0ceb4beec4ce13d91505c9b50dfd4a927fbf1379729706b2813"
)
EXPECTED_NATIVE_GRADIENT_HASH = (
    "6eca0131b85f0132dbd32a1dad89efb9bae8a8fe69c81f53b32b0dc36bff19a9"
)
EXPECTED_NATIVE_LOWER = (
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
    "boundflow/backends/tvm/r3_p_alpha_vjp.py",
    "boundflow/runtime/r3_compiled_p_alpha_vjp.py",
    "boundflow/runtime/r3_bounded_arena_trace_compiler.py",
    "boundflow/ir/r3_bounded_arena.py",
    "boundflow/runtime/r3_structured_owner_custom_backward.py",
    "scripts/run_r3_compiled_p_alpha_vjp_worker.py",
    "scripts/run_r3_compiled_p_alpha_vjp_artifact.py",
    "scripts/probe_r3_compiled_p_alpha_vjp_tamper.py",
    "tests/test_r3_compiled_p_alpha_vjp.py",
    "tests/test_r3_compiled_p_alpha_vjp_artifact.py",
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


def _float32_hash(values: tuple[float, ...]) -> str:
    return hashlib.sha256(
        b"".join(struct.pack("<f", value) for value in values)
    ).hexdigest()


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
        raise TypeError(f"R3-1b2 JSON root differs: {path.name}")
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


def _validate_metric(
    metric: dict[str, Any],
    *,
    count: int,
    candidate_hash: str,
    native_hash: str | None,
    expected_nonzero: int,
) -> None:
    candidate = tuple(float(value) for value in metric.get("candidate", ()))
    native = tuple(float(value) for value in metric.get("native", ()))
    if len(candidate) != count or len(native) != count:
        raise ValueError("R3-1b2 metric cardinality differs")
    if not all(math.isfinite(value) for value in (*candidate, *native)):
        raise ValueError("R3-1b2 metric finiteness differs")
    recomputed_max = max(abs(left - right) for left, right in zip(candidate, native))
    recomputed_sign = all(
        (left > 0) - (left < 0) == (right > 0) - (right < 0)
        for left, right in zip(candidate, native)
    )
    candidate_nonzero = sum(value != 0 for value in candidate)
    native_nonzero = sum(value != 0 for value in native)
    if (
        _float32_hash(candidate) != candidate_hash
        or metric.get("candidate_sha256") != candidate_hash
        or metric.get("native_sha256") != _float32_hash(native)
        or (native_hash is not None and metric.get("native_sha256") != native_hash)
        or int(metric.get("element_count", -1)) != count
        or abs(float(metric.get("max_abs_diff", -1.0)) - recomputed_max) > 1e-12
        or recomputed_max > 2e-4
        or metric.get("allclose") is not True
        or metric.get("finite") is not True
        or metric.get("sign_exact") is not recomputed_sign
        or recomputed_sign is not True
        or int(metric.get("candidate_nonzero", -1)) != candidate_nonzero
        or int(metric.get("native_nonzero", -1)) != native_nonzero
        or candidate_nonzero != expected_nonzero
        or native_nonzero != expected_nonzero
        or float(metric.get("atol", 0.0)) != 2e-4
        or float(metric.get("rtol", 0.0)) != 2e-4
    ):
        raise ValueError("R3-1b2 metric semantic receipt differs")


def _validate_worker(record: dict[str, Any]) -> None:
    lower = record.get("lower_metric")
    gradient = record.get("gradient_metric")
    receipt = record.get("execution_receipt")
    environment = record.get("environment")
    if not all(
        isinstance(value, dict) for value in (lower, gradient, receipt, environment)
    ):
        raise ValueError("R3-1b2 worker receipt root differs")
    assert isinstance(lower, dict) and isinstance(gradient, dict)
    assert isinstance(receipt, dict) and isinstance(environment, dict)
    _validate_metric(
        lower,
        count=6,
        candidate_hash=EXPECTED_CANDIDATE_LOWER_HASH,
        native_hash=None,
        expected_nonzero=6,
    )
    native_lower = tuple(float(value) for value in lower["native"])
    if (
        max(
            abs(left - right)
            for left, right in zip(native_lower, EXPECTED_NATIVE_LOWER)
        )
        > 5e-6
    ):
        raise ValueError("R3-1b2 native lower drifted")
    _validate_metric(
        gradient,
        count=1032,
        candidate_hash=EXPECTED_CANDIDATE_GRADIENT_HASH,
        native_hash=EXPECTED_NATIVE_GRADIENT_HASH,
        expected_nonzero=281,
    )
    if (
        record.get("schema_version") != WORKER_SCHEMA
        or record.get("source_capture_sha256") != EXPECTED_CAPTURE_HASH
        or record.get("model_sha256") != EXPECTED_MODEL_HASH
        or record.get("trace_hash") != EXPECTED_TRACE_HASH
        or record.get("production_plan_hash") != EXPECTED_PLAN_HASH
        or receipt.get("production_plan_hash") != EXPECTED_PLAN_HASH
        or receipt.get("trace_hash") != EXPECTED_TRACE_HASH
        or receipt.get("b1_module_hash") != EXPECTED_B1_MODULE_HASH
        or receipt.get("b2_module_hash") != EXPECTED_B2_MODULE_HASH
        or receipt.get("b2_device_source_hash") != EXPECTED_B2_DEVICE_HASH
        or tuple(receipt.get("b2_exported_symbols", ())) != R31B2_EXPORTED_SYMBOLS
        or receipt.get("custom_forward_count") != 1
        or receipt.get("custom_backward_count") != 1
        or receipt.get("b1_forward_launch_count") != 15
        or receipt.get("b1_backward_launch_count") != 15
        or receipt.get("b2_launch_count") != 10
        or receipt.get("coefficient_scratch_count") != 2
        or receipt.get("sign_bitmap_count") != 4
        or receipt.get("sign_bitmap_bytes") != 43008
        or receipt.get("saved_dense_a_count") != 0
        or receipt.get("python_visible_intermediate_coefficient_count") != 0
        or receipt.get("warm_dynamic_allocated_bytes") != 0
        or receipt.get("fallback_count") != 0
        or receipt.get("eager_candidate_count") != 0
        or receipt.get("native_shadow_count") != 0
        or receipt.get("dlpack_pointer_count") != 79
        or receipt.get("dlpack_pointer_exact_count") != 79
        or receipt.get("runtime_dlpack_pointer_count") != 1
        or receipt.get("runtime_dlpack_pointer_exact_count") != 1
        or receipt.get("compiled_vjp") is not True
        or receipt.get("custom_vjp") is not True
        or receipt.get("timing_recorded") is not False
        or receipt.get("performance_claimed") is not False
        or environment.get("compute_capability") != "sm_89"
        or record.get("compiled_full_lower") is not True
        or record.get("compiled_vjp") is not True
        or record.get("custom_vjp") is not True
        or record.get("r3_1_admitted") is not False
        or record.get("timing_recorded") is not False
        or record.get("performance_claimed") is not False
    ):
        raise ValueError("R3-1b2 worker semantic receipt differs")


def _summary(record: dict[str, Any]) -> dict[str, object]:
    _validate_worker(record)
    lower = record["lower_metric"]
    gradient = record["gradient_metric"]
    receipt = record["execution_receipt"]
    result: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "validated-r3-1b2-compiled-p-alpha-vjp",
        "trace_hash": record["trace_hash"],
        "production_plan_hash": record["production_plan_hash"],
        "b1_module_hash": receipt["b1_module_hash"],
        "b2_module_hash": receipt["b2_module_hash"],
        "b2_device_source_hash": receipt["b2_device_source_hash"],
        "candidate_lower_sha256": lower["candidate_sha256"],
        "candidate_gradient_sha256": gradient["candidate_sha256"],
        "native_gradient_sha256": gradient["native_sha256"],
        "lower_max_abs_diff": lower["max_abs_diff"],
        "gradient_max_abs_diff": gradient["max_abs_diff"],
        "lower_sign_exact": lower["sign_exact"],
        "gradient_sign_exact": gradient["sign_exact"],
        "gradient_nonzero": gradient["candidate_nonzero"],
        "b1_forward_launch_count": receipt["b1_forward_launch_count"],
        "b1_backward_launch_count": receipt["b1_backward_launch_count"],
        "b2_launch_count": receipt["b2_launch_count"],
        "coefficient_scratch_count": receipt["coefficient_scratch_count"],
        "sign_bitmap_count": receipt["sign_bitmap_count"],
        "sign_bitmap_bytes": receipt["sign_bitmap_bytes"],
        "saved_dense_a_count": receipt["saved_dense_a_count"],
        "warm_dynamic_allocated_bytes": receipt["warm_dynamic_allocated_bytes"],
        "dlpack_pointer_count": receipt["dlpack_pointer_count"],
        "dlpack_pointer_exact_count": receipt["dlpack_pointer_exact_count"],
        "compiled_vjp": True,
        "custom_vjp": True,
        "b2_closed": True,
        "b3_open": True,
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
        "lower_max_abs_diff": summary["lower_max_abs_diff"],
        "gradient_max_abs_diff": summary["gradient_max_abs_diff"],
        "gradient_sign_exact": summary["gradient_sign_exact"],
        "gradient_nonzero": summary["gradient_nonzero"],
        "coefficient_scratch_count": summary["coefficient_scratch_count"],
        "saved_dense_a_count": summary["saved_dense_a_count"],
        "warm_dynamic_allocated_bytes": summary["warm_dynamic_allocated_bytes"],
        "compiled_vjp": summary["compiled_vjp"],
        "custom_vjp": summary["custom_vjp"],
        "b3_open": summary["b3_open"],
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
        "comparison": "independent-native-autograd-vs-compiled-custom-vjp",
        "tolerance": {"atol": 2e-4, "rtol": 2e-4, "sign_exact": True},
        "hardware": "NVIDIA GeForce RTX 4060 Laptop GPU / sm_89",
        "scope": "one-evaluation-P-anchor-25-Conv_8-lower-and-compressed-dalpha",
        "timing_recorded": False,
        "performance_claimed": False,
    }


def _readme() -> str:
    return (
        "# R3-1b2 compiled P-alpha VJP\n\n"
        "One fresh subprocess compares independent native autograd with the compiled "
        "full-lower custom Function and compressed P-alpha VJP. Replay hash-checks code "
        "and every signed file, then independently recomputes tensor hashes, numerical "
        "parity and ownership receipts. This closes only b2; b3 five-fresh memory gating, "
        "optimizer integration and timing remain separate.\n"
    )


def generate(output: Path, source_capture: Path, model: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"R3-1b2 artifact output already exists: {output}")
    output.mkdir(parents=True)
    protocol = _protocol(source_capture, model)
    completed = subprocess.run(
        (
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/run_r3_compiled_p_alpha_vjp_worker.py"),
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
        raise RuntimeError("R3-1b2 worker stdout differs")
    record = json.loads(lines[0])
    if not isinstance(record, dict):
        raise TypeError("R3-1b2 worker record differs")
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
        raise RuntimeError("R3-1b2 generated replay differs")
    return replayed


def replay(artifact: Path) -> dict[str, object]:
    manifest = _load_json(artifact / "manifest.json")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest.get("manifest_hash") != _hash(unsigned)
        or not isinstance(files, dict)
        or set(files) != set(SIGNED_FILES)
        or any(_file_hash(artifact / name) != digest for name, digest in files.items())
    ):
        raise ValueError("R3-1b2 artifact manifest differs")
    protocol = _load_json(artifact / "protocol.json")
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("artifact_schema") != ARTIFACT_SCHEMA
        or protocol.get("code_revision") != _code_revision()
        or protocol.get("source_capture_sha256") != EXPECTED_CAPTURE_HASH
        or protocol.get("model_sha256") != EXPECTED_MODEL_HASH
        or protocol.get("fresh_process_count") != 1
        or protocol.get("timing_recorded") is not False
        or protocol.get("performance_claimed") is not False
    ):
        raise ValueError("R3-1b2 artifact protocol differs")
    raw_lines = (artifact / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    if len(raw_lines) != 1:
        raise ValueError("R3-1b2 artifact raw cardinality differs")
    record = json.loads(raw_lines[0])
    if not isinstance(record, dict):
        raise TypeError("R3-1b2 artifact raw record differs")
    expected_summary = _summary(record)
    summary = _load_json(artifact / "summary.json")
    if summary != expected_summary:
        raise ValueError("R3-1b2 artifact semantic recomputation differs")
    result = _result(summary)
    if (artifact / "replay_stdout.txt").read_text(encoding="utf-8") != _canonical(
        result
    ) + "\n":
        raise ValueError("R3-1b2 artifact replay stdout differs")
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
