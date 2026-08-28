#!/usr/bin/env python3
"""Generate, replay, and tamper-probe the ASPLOS'27 S1 CIBC artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=missing-function-docstring
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCHEMA = "boundflow.asplos27-s1-cibc-protocol/v1"
SUMMARY_SCHEMA = "boundflow.asplos27-s1-cibc-summary/v1"
MANIFEST_SCHEMA = "boundflow.asplos27-s1-cibc-manifest/v1"
WORKER_SCHEMA = "boundflow.asplos27-s1-cibc-worker/v1"
ORDERS = ("BDP", "BPD", "DBP", "DPB", "PBD", "PDB")
SOURCE_CAPTURE_SHA256 = (
    "f42229dd06be8521865d14af4cc51450ec8bec4e792934b457c276aad50126dc"
)
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
ATOL = 3.0e-4
PIPELINE_GEOMEAN_GATE = 2.20
PIPELINE_WORST_GATE = 2.00
PROPAGATION_GEOMEAN_GATE = 0.90
CODE_PATHS = (
    "boundflow/backends/tvm/cibc_ibp_conv.py",
    "boundflow/backends/tvm/relax_interval_task_ops.py",
    "boundflow/runtime/asplos27_s1_cibc_pipeline.py",
    "scripts/install_dev.sh",
    "scripts/run_asplos27_s1_cibc_worker.py",
    "scripts/run_asplos27_s1_cibc_artifact.py",
    "tests/test_asplos27_s1_cibc_pipeline.py",
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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("S1 CIBC JSON root differs")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value, indent=2) + "\n", encoding="utf-8")


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


def geomean(values: Sequence[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("S1 CIBC geomean inputs differ")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def protocol(source_capture: Path, model: Path) -> dict[str, object]:
    if file_sha256(source_capture) != SOURCE_CAPTURE_SHA256:
        raise ValueError("S1 CIBC source capture digest differs")
    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("S1 CIBC model digest differs")
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": git("rev-parse", "HEAD"),
        "code_revision": code_revision(),
        "source_capture_sha256": SOURCE_CAPTURE_SHA256,
        "model_sha256": MODEL_SHA256,
        "orders": list(ORDERS),
        "run_count": 6,
        "group_count": 30,
        "repeats": 200,
        "semantic_atol": ATOL,
        "semantic_rtol": ATOL,
        "pipeline_geomean_gate": PIPELINE_GEOMEAN_GATE,
        "pipeline_worst_gate": PIPELINE_WORST_GATE,
        "propagation_geomean_gate": PROPAGATION_GEOMEAN_GATE,
        "required_conv_coverage": 6,
        "required_cublas_partitions": 2,
        "input_copy_included": True,
        "warm_dlpack_view_count": 0,
        "fallback_count": 0,
        "performance_claimed": False,
    }
    payload["protocol_hash"] = canonical_hash(payload)
    return payload


def validate_protocol(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    claimed = payload.pop("protocol_hash", None)
    source = value.get("source_git_head")
    if (
        claimed != canonical_hash(payload)
        or value.get("schema_version") != PROTOCOL_SCHEMA
        or not isinstance(source, str)
        or value.get("code_revision") != code_revision(source)
        or value.get("orders") != list(ORDERS)
        or value.get("semantic_atol") != ATOL
        or value.get("pipeline_geomean_gate") != PIPELINE_GEOMEAN_GATE
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("S1 CIBC protocol differs")


def validate_worker(value: Mapping[str, Any], *, ordinal: int, order: str) -> None:
    payload = dict(value)
    claimed = payload.pop("worker_hash", None)
    groups = value.get("groups")
    metrics = value.get("metrics")
    compile_receipt = value.get("compile_receipt")
    execution_receipt = value.get("execution_receipt")
    if (
        claimed != canonical_hash(payload)
        or value.get("schema_version") != WORKER_SCHEMA
        or value.get("run_ordinal") != ordinal
        or value.get("order") != order
        or not isinstance(groups, list)
        or len(groups) != 30
        or value.get("group_count") != 30
        or value.get("repeats") != 200
        or not isinstance(metrics, dict)
        or len(metrics) != 6
        or not isinstance(compile_receipt, dict)
        or not isinstance(execution_receipt, dict)
        or value.get("allclose") is not True
        or value.get("sign_exact") is not True
        or float(value.get("maximum_absolute_difference", math.inf)) > ATOL
        or value.get("input_copy_included") is not True
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("S1 CIBC worker identity/semantics differs")
    for index, row in enumerate(groups):
        if row.get("group") != index or any(
            float(row[name]) <= 0.0
            for name in ("baseline_ms", "direct_ms", "pipeline_ms")
        ):
            raise ValueError("S1 CIBC timing group differs")
    medians = {
        name: sorted(float(row[name]) for row in groups)[14:16]
        for name in ("baseline_ms", "direct_ms", "pipeline_ms")
    }
    derived = {name: sum(pair) / 2.0 for name, pair in medians.items()}
    if any(value.get(name) != derived[name] for name in derived):
        raise ValueError("S1 CIBC timing median differs")
    if (
        value.get("direct_speedup") != derived["baseline_ms"] / derived["direct_ms"]
        or value.get("pipeline_speedup")
        != derived["baseline_ms"] / derived["pipeline_ms"]
        or value.get("pipeline_direct_propagation")
        != derived["direct_ms"] / derived["pipeline_ms"]
    ):
        raise ValueError("S1 CIBC timing derivation differs")
    if any(
        not bool(metric.get("allclose"))
        or not bool(metric.get("sign_exact"))
        or float(metric.get("maximum_absolute_difference", math.inf)) > ATOL
        for metric in metrics.values()
    ):
        raise ValueError("S1 CIBC metric differs")
    if (
        compile_receipt.get("performance_claimed") is not False
        or compile_receipt.get("fallback_admitted") is not False
        or compile_receipt.get("op_count") != 17
        or len(compile_receipt.get("cibc_conv_ops", [])) != 6
        or compile_receipt.get("cublas_partition_count") != 2
        or execution_receipt.get("cibc_conv_call_tir_count") != 6
        or execution_receipt.get("cuda_graph_replay_count") != 1
        or execution_receipt.get("warm_dlpack_view_count") != 0
        or execution_receipt.get("fallback_count") != 0
        or execution_receipt.get("eager_shadow_count") != 0
        or execution_receipt.get("output_materialization_copy_included") is not False
        or execution_receipt.get("performance_claimed") is not False
    ):
        raise ValueError("S1 CIBC compiler/runtime receipt differs")
    environment = value.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("device") != "NVIDIA GeForce RTX 4060 Laptop GPU"
        or environment.get("compute_capability") != [8, 9]
    ):
        raise ValueError("S1 CIBC worker environment differs")


def derive_summary(workers: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    if len(workers) != len(ORDERS):
        raise ValueError("S1 CIBC worker inventory differs")
    for ordinal, (worker, order) in enumerate(zip(workers, ORDERS)):
        validate_worker(worker, ordinal=ordinal, order=order)
    direct = [float(worker["direct_speedup"]) for worker in workers]
    pipeline = [float(worker["pipeline_speedup"]) for worker in workers]
    propagation = [float(worker["pipeline_direct_propagation"]) for worker in workers]
    admitted = (
        geomean(pipeline) >= PIPELINE_GEOMEAN_GATE
        and min(pipeline) >= PIPELINE_WORST_GATE
        and geomean(propagation) >= PROPAGATION_GEOMEAN_GATE
    )
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": (
            "validated-s1-cibc-pipeline" if admitted else "validated-no-go-s1-cibc"
        ),
        "run_count": len(workers),
        "orders": list(ORDERS),
        "direct_speedups": direct,
        "direct_speedup_geomean": geomean(direct),
        "direct_speedup_worst": min(direct),
        "pipeline_speedups": pipeline,
        "pipeline_speedup_geomean": geomean(pipeline),
        "pipeline_speedup_worst": min(pipeline),
        "pipeline_direct_propagations": propagation,
        "pipeline_direct_propagation_geomean": geomean(propagation),
        "pipeline_direct_propagation_worst": min(propagation),
        "maximum_absolute_difference": max(
            float(worker["maximum_absolute_difference"]) for worker in workers
        ),
        "allclose": True,
        "sign_exact": True,
        "op_count": 17,
        "cibc_conv_coverage": 6,
        "cublas_partition_count": 2,
        "fallback_count": 0,
        "eager_shadow_count": 0,
        "warm_dlpack_view_count": 0,
        "input_copy_included": True,
        "s1_performance_admitted": admitted,
        "same_solver_claimed": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


def run_worker(
    *, source_capture: Path, model: Path, ordinal: int, order: str, output: Path
) -> None:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (str(REPOSITORY_ROOT), environment.get("PYTHONPATH", ""))
        if item
    )
    subprocess.run(
        (
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/run_asplos27_s1_cibc_worker.py"),
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
        check=True,
        stdout=subprocess.DEVNULL,
    )


def tamper_report(worker: Mapping[str, Any]) -> dict[str, object]:
    cases: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
        (
            "semantic-diff",
            lambda value: value.__setitem__("maximum_absolute_difference", 1.0),
        ),
        ("sign", lambda value: value.__setitem__("sign_exact", False)),
        (
            "fallback",
            lambda value: value["execution_receipt"].__setitem__("fallback_count", 1),
        ),
        (
            "cublas",
            lambda value: value["compile_receipt"].__setitem__(
                "cublas_partition_count", 0
            ),
        ),
        (
            "conv-coverage",
            lambda value: value["execution_receipt"].__setitem__(
                "cibc_conv_call_tir_count", 5
            ),
        ),
        (
            "dlpack",
            lambda value: value["execution_receipt"].__setitem__(
                "warm_dlpack_view_count", 1
            ),
        ),
        ("claim", lambda value: value.__setitem__("performance_claimed", True)),
        ("order", lambda value: value.__setitem__("order", "PDB")),
    )
    rows = []
    for name, mutation in cases:
        changed = copy.deepcopy(dict(worker))
        changed.pop("worker_hash", None)
        mutation(changed)
        changed["worker_hash"] = canonical_hash(changed)
        rejected = False
        error = ""
        try:
            validate_worker(changed, ordinal=0, order=ORDERS[0])
        except (TypeError, ValueError) as exception:
            rejected = True
            error = str(exception)
        rows.append(
            {
                "case": name,
                "outer_resigned": True,
                "rejected": rejected,
                "error": error,
            }
        )
    report: dict[str, object] = {
        "schema_version": "boundflow.asplos27-s1-cibc-tamper/v1",
        "case_count": len(rows),
        "rejected_count": sum(bool(row["rejected"]) for row in rows),
        "rows": rows,
        "performance_claimed": False,
    }
    if report["rejected_count"] != report["case_count"]:
        raise ValueError("S1 CIBC tamper probe did not reject every case")
    report["report_hash"] = canonical_hash(report)
    return report


def manifest(artifact: Path, protocol_value: Mapping[str, Any]) -> dict[str, object]:
    files = {
        str(path.relative_to(artifact)): file_sha256(path)
        for path in sorted(artifact.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    value: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_hash": protocol_value["protocol_hash"],
        "source_git_head": protocol_value["source_git_head"],
        "files": files,
        "performance_claimed": False,
    }
    value["manifest_hash"] = canonical_hash(value)
    return value


def replay(artifact: Path) -> dict[str, object]:
    protocol_value = load_json(artifact / "protocol.json")
    validate_protocol(protocol_value)
    manifest_value = load_json(artifact / "manifest.json")
    manifest_payload = dict(manifest_value)
    manifest_hash = manifest_payload.pop("manifest_hash", None)
    if manifest_hash != canonical_hash(manifest_payload):
        raise ValueError("S1 CIBC manifest hash differs")
    for name, digest in manifest_value["files"].items():
        if file_sha256(artifact / name) != digest:
            raise ValueError(f"S1 CIBC manifest file differs: {name}")
    workers = [
        load_json(artifact / "raw" / f"run_{index:02d}.json") for index in range(6)
    ]
    expected = derive_summary(workers)
    if load_json(artifact / "summary.json") != expected:
        raise ValueError("S1 CIBC summary replay differs")
    report = load_json(artifact / "tamper_report.json")
    if (
        report.get("rejected_count") != report.get("case_count")
        or report.get("performance_claimed") is not False
    ):
        raise ValueError("S1 CIBC tamper report differs")
    return {
        "status": "replay-passed",
        "summary_hash": expected["summary_hash"],
        "manifest_hash": manifest_value["manifest_hash"],
        "run_count": len(workers),
        "performance_claimed": False,
    }


def generate(artifact: Path, source_capture: Path, model: Path) -> dict[str, object]:
    artifact.mkdir(parents=True, exist_ok=False)
    protocol_value = protocol(source_capture, model)
    write_json(artifact / "protocol.json", protocol_value)
    workers = []
    for ordinal, order in enumerate(ORDERS):
        output = artifact / "raw" / f"run_{ordinal:02d}.json"
        run_worker(
            source_capture=source_capture,
            model=model,
            ordinal=ordinal,
            order=order,
            output=output,
        )
        workers.append(load_json(output))
    summary = derive_summary(workers)
    write_json(artifact / "summary.json", summary)
    write_json(artifact / "tamper_report.json", tamper_report(workers[0]))
    write_json(artifact / "manifest.json", manifest(artifact, protocol_value))
    return replay(artifact)


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
        result = generate(args.artifact, args.source_capture, args.model)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
