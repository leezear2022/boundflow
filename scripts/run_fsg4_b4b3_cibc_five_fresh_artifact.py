#!/usr/bin/env python3
"""Generate or replay five fresh B4-B3 CIBC exact-call semantic pairs."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=missing-function-docstring,subprocess-run-check
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCHEMA = "boundflow.fsg4-b4b3-cibc-five-fresh-protocol/v1"
SUMMARY_SCHEMA = "boundflow.fsg4-b4b3-cibc-five-fresh-summary/v1"
MANIFEST_SCHEMA = "boundflow.fsg4-b4b3-cibc-five-fresh-manifest/v1"
WORKER_SCHEMA = "boundflow.fsg4-b4b3-cibc-exact-worker/v1"
ORDERS = ("BC", "CB", "BC", "CB", "BC")
ATOL = 2.0e-4
CODE_PATHS = (
    "boundflow/backends/tvm/cibc_dense_exact_conv.py",
    "boundflow/runtime/fsg4_b4b3_cibc_dense_tir.py",
    "boundflow/runtime/fsg4_b4b3_cibc_exact_call.py",
    "boundflow/runtime/crown_ibp.py",
    "boundflow/runtime/fsg4_b3_terminal_optimizer_schedule.py",
    "scripts/run_fsg4_b4b3_cibc_exact_worker.py",
    "scripts/run_fsg4_b4b3_cibc_five_fresh_artifact.py",
    "tests/test_fsg4_b4b3_cibc_exact_call.py",
)
REFERENCE_MANIFEST = REPOSITORY_ROOT / (
    "artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1/manifest.json"
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


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"B4-B3 artifact JSON root differs: {path.name}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, binary: bool = False):
    completed = subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    return completed.stdout if binary else completed.stdout.strip()


def historical_sha256(source: str, path: str) -> str:
    if git("rev-parse", "HEAD") == source:
        return file_sha256(REPOSITORY_ROOT / path)
    return hashlib.sha256(git("show", f"{source}:{path}", binary=True)).hexdigest()


def code_revision(source: str | None = None) -> dict[str, str]:
    if source is None:
        return {path: file_sha256(REPOSITORY_ROOT / path) for path in CODE_PATHS}
    return {path: historical_sha256(source, path) for path in CODE_PATHS}


def protocol(source_capture: Path, model: Path) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": git("rev-parse", "HEAD"),
        "code_revision": code_revision(),
        "source_capture_sha256": file_sha256(source_capture),
        "model_sha256": file_sha256(model),
        "reference_manifest_sha256": file_sha256(REFERENCE_MANIFEST),
        "run_count": 5,
        "orders": list(ORDERS),
        "terminal_atol": ATOL,
        "terminal_rtol": ATOL,
        "terminal_sign_exact_required": True,
        "local_parity_atol": ATOL,
        "exact_evaluation_count": 10,
        "exact_update_count": 9,
        "native_value_bridge_required": True,
        "timing_diagnostic_only": True,
        "performance_claimed": False,
    }
    value["protocol_hash"] = canonical_hash(value)
    return value


def validate_protocol(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    claimed = payload.pop("protocol_hash", None)
    source = value.get("source_git_head")
    if (
        claimed != canonical_hash(payload)
        or value.get("schema_version") != PROTOCOL_SCHEMA
        or not isinstance(source, str)
        or value.get("code_revision") != code_revision(source)
        or value.get("reference_manifest_sha256") != file_sha256(REFERENCE_MANIFEST)
        or value.get("run_count") != 5
        or value.get("orders") != list(ORDERS)
        or value.get("terminal_atol") != ATOL
        or value.get("terminal_rtol") != ATOL
        or value.get("native_value_bridge_required") is not True
        or value.get("timing_diagnostic_only") is not True
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("B4-B3 CIBC protocol differs")


def validate_worker(value: Mapping[str, Any], ordinal: int) -> None:
    payload = dict(value)
    claimed = payload.pop("worker_hash", None)
    metrics = value.get("metrics")
    receipt = value.get("receipt")
    local = value.get("local_parity")
    if (
        claimed != canonical_hash(payload)
        or value.get("schema_version") != WORKER_SCHEMA
        or value.get("run_ordinal") != ordinal
        or value.get("order") != ORDERS[ordinal]
        or value.get("allclose") is not True
        or value.get("sign_exact") is not True
        or float(value.get("maximum_absolute_difference", float("inf"))) > ATOL
        or not isinstance(metrics, Mapping)
        or len(metrics) != 13
        or not isinstance(receipt, Mapping)
        or receipt.get("evaluation_count") != 10
        or receipt.get("update_count") != 9
        or receipt.get("provider_activation_count") != 10
        or receipt.get("forward_launch_count") != 10
        or receipt.get("backward_launch_count") != 9
        or receipt.get("unsupported_semantic_anchor_count") != 1
        or receipt.get("native_value_bridge_count") != 10
        or receipt.get("adjoint_materialization_count") != 0
        or receipt.get("fallback_count") != 0
        or receipt.get("eager_count") != 0
        or receipt.get("exact_call") is not True
        or receipt.get("performance_claimed") is not False
        or not isinstance(local, list)
        or len(local) != 1
        or float(local[0].get("output_a_max_abs_diff", float("inf"))) > ATOL
        or float(local[0].get("output_bias_max_abs_diff", float("inf"))) > ATOL
        or local[0].get("output_a_sign_exact") is not True
        or local[0].get("output_bias_sign_exact") is not True
        or value.get("performance_claimed") is not False
    ):
        raise ValueError(f"B4-B3 CIBC worker differs: {ordinal}")
    for name, metric in metrics.items():
        if (
            not isinstance(name, str)
            or not isinstance(metric, Mapping)
            or metric.get("allclose") is not True
            or metric.get("sign_exact") is not True
            or int(metric.get("element_count", 0)) < 1
            or float(metric.get("maximum_absolute_difference", float("inf"))) > ATOL
        ):
            raise ValueError(f"B4-B3 CIBC metric differs: {ordinal}:{name}")


def derive_summary(workers: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    if len(workers) != 5:
        raise ValueError("B4-B3 CIBC worker count differs")
    for ordinal, worker in enumerate(workers):
        validate_worker(worker, ordinal)
    module_hashes = {str(worker["receipt"]["module_hash"]) for worker in workers}
    if len(module_hashes) != 1:
        raise ValueError("B4-B3 CIBC module identity differs")
    maximum = max(float(worker["maximum_absolute_difference"]) for worker in workers)
    local_a = max(
        float(worker["local_parity"][0]["output_a_max_abs_diff"]) for worker in workers
    )
    local_bias = max(
        float(worker["local_parity"][0]["output_bias_max_abs_diff"])
        for worker in workers
    )
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "validated-b4-b3-cibc-exact-call",
        "run_count": len(workers),
        "allclose": True,
        "sign_exact": True,
        "maximum_absolute_difference": maximum,
        "local_output_a_maximum_absolute_difference": local_a,
        "local_output_bias_maximum_absolute_difference": local_bias,
        "module_hash": next(iter(module_hashes)),
        "provider_activation_count": 50,
        "forward_launch_count": 50,
        "backward_launch_count": 45,
        "native_value_bridge_count": 50,
        "fallback_count": 0,
        "eager_count": 0,
        "adjoint_materialization_count": 0,
        "diagnostic_wall_speedups": [
            float(worker["paired_speedup"]) for worker in workers
        ],
        "timing_diagnostic_only": True,
        "performance_claimed": False,
        "cumulative_core_timing_admitted": True,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


def run_worker(
    *,
    source_capture: Path,
    model: Path,
    ordinal: int,
    order: str,
    output: Path,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPOSITORY_ROOT), inherited_pythonpath) if part
    )
    completed = subprocess.run(
        (
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/run_fsg4_b4b3_cibc_exact_worker.py"),
            "--source-capture",
            str(source_capture),
            "--model",
            str(model),
            "--run-ordinal",
            str(ordinal),
            "--order",
            order,
            "--output",
            str(output),
        ),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"B4-B3 CIBC worker failed ordinal={ordinal}:\n{completed.stdout}"
        )


def manifest(
    root: Path, protocol_value: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "source_git_head": protocol_value["source_git_head"],
        "protocol_hash": protocol_value["protocol_hash"],
        "summary_hash": summary["summary_hash"],
        "files": {
            str(path.relative_to(root)): file_sha256(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
    }
    value["manifest_hash"] = canonical_hash(value)
    return value


def generate(root: Path, *, source_capture: Path, model: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    protocol_value = protocol(source_capture, model)
    write_json(root / "protocol.json", protocol_value)
    workers = []
    for ordinal, order in enumerate(ORDERS):
        output = root / "raw" / f"run_{ordinal:02d}_{order.lower()}.json"
        run_worker(
            source_capture=source_capture,
            model=model,
            ordinal=ordinal,
            order=order,
            output=output,
        )
        workers.append(load_json(output))
    summary = derive_summary(workers)
    write_json(root / "summary.json", summary)
    write_json(root / "manifest.json", manifest(root, protocol_value, summary))
    return summary


def replay(root: Path) -> dict[str, object]:
    protocol_value = load_json(root / "protocol.json")
    summary = load_json(root / "summary.json")
    manifest_value = load_json(root / "manifest.json")
    validate_protocol(protocol_value)
    payload = dict(manifest_value)
    claimed = payload.pop("manifest_hash", None)
    files = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    if (
        claimed != canonical_hash(payload)
        or manifest_value.get("schema_version") != MANIFEST_SCHEMA
        or manifest_value.get("protocol_hash") != protocol_value["protocol_hash"]
        or manifest_value.get("summary_hash") != summary.get("summary_hash")
        or manifest_value.get("files") != files
    ):
        raise ValueError("B4-B3 CIBC manifest differs")
    workers = [
        load_json(root / "raw" / f"run_{ordinal:02d}_{order.lower()}.json")
        for ordinal, order in enumerate(ORDERS)
    ]
    if derive_summary(workers) != summary:
        raise ValueError("B4-B3 CIBC semantic replay differs")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-capture", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        result = replay(args.artifact)
    else:
        if args.source_capture is None or args.model is None:
            parser.error("generation requires --source-capture and --model")
        result = generate(
            args.artifact,
            source_capture=args.source_capture,
            model=args.model,
        )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
