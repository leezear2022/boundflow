#!/usr/bin/env python3
"""Generate or replay CIBC IBP horizontal-fusion formal evidence."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-arguments
# pylint: disable=too-many-boolean-expressions,missing-function-docstring

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCHEMA = "boundflow.cibc-ibp-horizontal-protocol/v1"
SUMMARY_SCHEMA = "boundflow.cibc-ibp-horizontal-summary/v1"
MANIFEST_SCHEMA = "boundflow.cibc-ibp-horizontal-manifest/v1"
WORKER_SCHEMA = "boundflow.cibc-ibp-horizontal-worker/v1"
SOURCE_CAPTURE_SHA256 = (
    "f42229dd06be8521865d14af4cc51450ec8bec4e792934b457c276aad50126dc"
)
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
SCHEDULES = (64, 128, 256)
CONV_ORDINALS = (0, 2, 4, 5, 8, 10)
OPERATOR_ORDERS = ("BC", "CB", "BC")
MODEL_ORDERS = ("BC", "CB", "BC", "CB", "BC", "CB")
ATOL = 3.0e-4
OPERATOR_GEOMEAN_GATE = 2.0
OPERATOR_WORST_GATE = 1.2
MODEL_GEOMEAN_GATE = 1.5
MODEL_WORST_GATE = 1.2
BOOTSTRAP_SEED = 20260824
BOOTSTRAP_SAMPLES = 10_000
CODE_PATHS = (
    "boundflow/backends/tvm/cibc_ibp_conv.py",
    "boundflow/runtime/cibc_ibp_conv.py",
    "boundflow/runtime/cibc_ibp_graph.py",
    "boundflow/domains/interval.py",
    "scripts/run_cibc_ibp_horizontal_worker.py",
    "scripts/run_cibc_ibp_horizontal_artifact.py",
    "tests/test_cibc_ibp_conv.py",
    "tests/test_cibc_ibp_graph.py",
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


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("CIBC IBP artifact JSON root differs")
    return value


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
        raise ValueError("CIBC IBP geomean inputs differ")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def bootstrap_lower(values: Sequence[float]) -> float:
    generator = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[generator.randrange(len(values))] for _ in values]
        samples.append(geomean(sample))
    return sorted(samples)[int(0.025 * len(samples))]


def protocol(source_capture: Path, model: Path) -> dict[str, object]:
    if file_sha256(source_capture) != SOURCE_CAPTURE_SHA256:
        raise ValueError("CIBC IBP source capture digest differs")
    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("CIBC IBP model digest differs")
    value: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": git("rev-parse", "HEAD"),
        "code_revision": code_revision(),
        "source_capture_sha256": SOURCE_CAPTURE_SHA256,
        "model_sha256": MODEL_SHA256,
        "schedules": list(SCHEDULES),
        "operator_orders": list(OPERATOR_ORDERS),
        "model_orders": list(MODEL_ORDERS),
        "operator_shape_count": 6,
        "group_count": 30,
        "operator_repeats": 500,
        "model_repeats": 100,
        "semantic_atol": ATOL,
        "semantic_rtol": ATOL,
        "operator_geomean_gate": OPERATOR_GEOMEAN_GATE,
        "operator_worst_gate": OPERATOR_WORST_GATE,
        "model_geomean_gate": MODEL_GEOMEAN_GATE,
        "model_worst_gate": MODEL_WORST_GATE,
        "input_copy_included": True,
        "baseline_cuda_graph": True,
        "candidate_cuda_graph": True,
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
        or value.get("schedules") != list(SCHEDULES)
        or value.get("operator_orders") != list(OPERATOR_ORDERS)
        or value.get("model_orders") != list(MODEL_ORDERS)
        or value.get("semantic_atol") != ATOL
        or value.get("source_capture_sha256") != SOURCE_CAPTURE_SHA256
        or value.get("model_sha256") != MODEL_SHA256
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("CIBC IBP protocol differs")


def validate_groups(groups: object) -> tuple[float, float]:
    if not isinstance(groups, list) or len(groups) != 30:
        raise ValueError("CIBC IBP timing groups differ")
    for ordinal, group in enumerate(groups):
        baseline = float(group["baseline_ms"])
        candidate = float(group["candidate_ms"])
        if (
            group.get("group") != ordinal
            or baseline <= 0.0
            or candidate <= 0.0
            or group.get("speedup") != baseline / candidate
        ):
            raise ValueError("CIBC IBP timing group derivation differs")
    return (
        statistics.median(float(item["baseline_ms"]) for item in groups),
        statistics.median(float(item["candidate_ms"]) for item in groups),
    )


def validate_worker(
    value: Mapping[str, Any], *, mode: str, ordinal: int, order: str, threads: int
) -> None:
    payload = dict(value)
    claimed = payload.pop("worker_hash", None)
    if (
        claimed != canonical_hash(payload)
        or value.get("schema_version") != WORKER_SCHEMA
        or value.get("mode") != mode
        or value.get("run_ordinal") != ordinal
        or value.get("order") != order
        or value.get("threads_per_block") != threads
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("CIBC IBP worker identity differs")
    rows = value.get("operators") if mode == "operator" else [value]
    if not isinstance(rows, list) or len(rows) != (6 if mode == "operator" else 1):
        raise ValueError("CIBC IBP worker row count differs")
    environment = value.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("device") != "NVIDIA GeForce RTX 4060 Laptop GPU"
        or environment.get("compute_capability") != [8, 9]
    ):
        raise ValueError("CIBC IBP worker hardware differs")
    if mode == "operator" and (
        value.get("operator_count") != 6
        or value.get("group_count") != 30
        or value.get("operator_repeats") != 500
        or value.get("plan_owned") is not True
        or tuple(int(row["op_ordinal"]) for row in rows) != CONV_ORDINALS
    ):
        raise ValueError("CIBC IBP operator inventory differs")
    for row in rows:
        baseline, candidate = validate_groups(row.get("groups"))
        if (
            row.get("baseline_median_ms") != baseline
            or row.get("candidate_median_ms") != candidate
            or row.get("speedup") != baseline / candidate
            or float(row.get("maximum_absolute_difference", math.inf)) > ATOL
            or row.get("sign_exact") is not True
        ):
            raise ValueError("CIBC IBP worker semantic derivation differs")
    if mode == "model" and (
        value.get("conv_coverage") != 6
        or float(value.get("final_maximum_absolute_difference", math.inf)) > ATOL
        or value.get("group_count") != 30
        or value.get("model_repeats") != 100
        or value.get("input_copy_included") is not True
        or value.get("baseline_cuda_graph") is not True
        or value.get("candidate_cuda_graph") is not True
    ):
        raise ValueError("CIBC IBP model coverage differs")


def derive_summary(
    operator_workers: Sequence[Mapping[str, Any]],
    model_workers: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    if len(operator_workers) != 3 or len(model_workers) != 6:
        raise ValueError("CIBC IBP worker inventory differs")
    schedule_geomeans: dict[str, float] = {}
    for ordinal, worker in enumerate(operator_workers):
        validate_worker(
            worker,
            mode="operator",
            ordinal=ordinal,
            order=OPERATOR_ORDERS[ordinal],
            threads=SCHEDULES[ordinal],
        )
        schedule_geomeans[str(SCHEDULES[ordinal])] = geomean(
            [float(row["speedup"]) for row in worker["operators"]]
        )
    selected = max(SCHEDULES, key=lambda item: schedule_geomeans[str(item)])
    selected_worker = operator_workers[SCHEDULES.index(selected)]
    operator_speedups = [float(row["speedup"]) for row in selected_worker["operators"]]
    model_speedups = []
    for ordinal, worker in enumerate(model_workers):
        validate_worker(
            worker,
            mode="model",
            ordinal=ordinal,
            order=MODEL_ORDERS[ordinal],
            threads=selected,
        )
        model_speedups.append(float(worker["speedup"]))
    operator_geometric_mean = geomean(operator_speedups)
    operator_worst = min(operator_speedups)
    model_geometric_mean = geomean(model_speedups)
    model_worst = min(model_speedups)
    admitted = (
        operator_geometric_mean >= OPERATOR_GEOMEAN_GATE
        and operator_worst >= OPERATOR_WORST_GATE
        and model_geometric_mean >= MODEL_GEOMEAN_GATE
        and model_worst >= MODEL_WORST_GATE
    )
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": (
            "validated-cibc-ibp-horizontal" if admitted else "validated-no-go-cibc-ibp"
        ),
        "schedule_geomeans": schedule_geomeans,
        "selected_threads_per_block": selected,
        "operator_speedups": operator_speedups,
        "operator_speedup_geomean": operator_geometric_mean,
        "operator_speedup_worst": operator_worst,
        "model_speedups": model_speedups,
        "model_speedup_geomean": model_geometric_mean,
        "model_speedup_bootstrap_lower_95": bootstrap_lower(model_speedups),
        "model_speedup_worst": model_worst,
        "maximum_absolute_difference": max(
            float(worker["maximum_absolute_difference"]) for worker in model_workers
        ),
        "final_maximum_absolute_difference": max(
            float(worker["final_maximum_absolute_difference"])
            for worker in model_workers
        ),
        "sign_exact": True,
        "conv_coverage": 6,
        "input_copy_included": True,
        "baseline_cuda_graph": True,
        "candidate_cuda_graph": True,
        "performance_admitted": admitted,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


def run_worker(
    *,
    mode: str,
    source_capture: Path,
    model: Path,
    ordinal: int,
    order: str,
    threads: int,
    output: Path,
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
            str(REPOSITORY_ROOT / "scripts/run_cibc_ibp_horizontal_worker.py"),
            "--mode",
            mode,
            "--source-capture",
            str(source_capture),
            "--model",
            str(model),
            "--run-ordinal",
            str(ordinal),
            "--order",
            order,
            "--threads",
            str(threads),
            "--output",
            str(output),
        ),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        stdout=subprocess.DEVNULL,
        timeout=300,
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
    operator_workers = []
    for ordinal, (threads, order) in enumerate(zip(SCHEDULES, OPERATOR_ORDERS)):
        output = root / "raw" / f"operator_{threads}.json"
        run_worker(
            mode="operator",
            source_capture=source_capture,
            model=model,
            ordinal=ordinal,
            order=order,
            threads=threads,
            output=output,
        )
        operator_workers.append(load_json(output))
    schedule_geomeans = [
        geomean([float(row["speedup"]) for row in worker["operators"]])
        for worker in operator_workers
    ]
    selected = SCHEDULES[max(range(3), key=schedule_geomeans.__getitem__)]
    model_workers = []
    for ordinal, order in enumerate(MODEL_ORDERS):
        output = root / "raw" / f"model_{ordinal:02d}_{order.lower()}.json"
        run_worker(
            mode="model",
            source_capture=source_capture,
            model=model,
            ordinal=ordinal,
            order=order,
            threads=selected,
            output=output,
        )
        model_workers.append(load_json(output))
    summary = derive_summary(operator_workers, model_workers)
    write_json(root / "summary.json", summary)
    write_json(root / "manifest.json", manifest(root, protocol_value, summary))
    return summary


def replay(root: Path) -> dict[str, object]:
    protocol_value = load_json(root / "protocol.json")
    summary = load_json(root / "summary.json")
    manifest_value = load_json(root / "manifest.json")
    validate_protocol(protocol_value)
    manifest_payload = dict(manifest_value)
    claimed = manifest_payload.pop("manifest_hash", None)
    files = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    if (
        claimed != canonical_hash(manifest_payload)
        or manifest_value.get("schema_version") != MANIFEST_SCHEMA
        or manifest_value.get("files") != files
        or manifest_value.get("protocol_hash") != protocol_value.get("protocol_hash")
        or manifest_value.get("summary_hash") != summary.get("summary_hash")
    ):
        raise ValueError("CIBC IBP manifest differs")
    operator_workers = [
        load_json(root / "raw" / f"operator_{threads}.json") for threads in SCHEDULES
    ]
    selected = int(summary["selected_threads_per_block"])
    model_workers = [
        load_json(root / "raw" / f"model_{ordinal:02d}_{order.lower()}.json")
        for ordinal, order in enumerate(MODEL_ORDERS)
    ]
    if (
        selected not in SCHEDULES
        or derive_summary(operator_workers, model_workers) != summary
    ):
        raise ValueError("CIBC IBP semantic replay differs")
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
            parser.error("generation requires source capture and model")
        result = generate(
            args.artifact, source_capture=args.source_capture, model=args.model
        )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
