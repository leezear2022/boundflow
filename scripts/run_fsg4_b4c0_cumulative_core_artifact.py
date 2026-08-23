#!/usr/bin/env python3
"""Generate or replay six-fresh warmed B4-C0 cumulative core timing."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
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
PROTOCOL_SCHEMA = "boundflow.fsg4-b4c0-cumulative-core-protocol/v1"
SUMMARY_SCHEMA = "boundflow.fsg4-b4c0-cumulative-core-summary/v1"
MANIFEST_SCHEMA = "boundflow.fsg4-b4c0-cumulative-core-manifest/v1"
WORKER_SCHEMA = "boundflow.fsg4-b4c0-cumulative-core-worker/v1"
ORDERS = ("BC", "CB", "BC", "CB", "BC", "CB")
NO_REGRESSION_GATE = 1.00
RESEARCH_GATE = 1.05
WORST_WORKER_GATE = 0.98
MEMORY_RATIO_GATE = 1.05
ATOL = 2.0e-4
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260824
CODE_PATHS = (
    "boundflow/backends/tvm/cibc_dense_exact_conv.py",
    "boundflow/runtime/fsg4_b4b3_cibc_dense_tir.py",
    "boundflow/runtime/fsg4_b4b3_cibc_exact_call.py",
    "boundflow/runtime/crown_ibp.py",
    "boundflow/runtime/fsg4_b3_terminal_optimizer_schedule.py",
    "scripts/run_fsg4_b4c0_cumulative_core_worker.py",
    "scripts/run_fsg4_b4c0_cumulative_core_artifact.py",
    "tests/test_fsg4_b4b3_cibc_exact_call.py",
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
        raise TypeError(f"B4-C0 artifact JSON root differs: {path.name}")
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


def geomean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("B4-C0 geomean inputs differ")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def bootstrap_lower(values: Sequence[float]) -> float:
    generator = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[generator.randrange(len(values))] for _ in values]
        samples.append(geomean(sample))
    samples.sort()
    return samples[int(0.025 * len(samples))]


def protocol(source_capture: Path, model: Path) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": git("rev-parse", "HEAD"),
        "code_revision": code_revision(),
        "source_capture_sha256": file_sha256(source_capture),
        "model_sha256": file_sha256(model),
        "run_count": 6,
        "orders": list(ORDERS),
        "warmups_per_side": 3,
        "groups_per_worker": 30,
        "compile_excluded": True,
        "full_optimizer_runtime_included": True,
        "native_value_bridge_included": True,
        "correctness_capture_excluded": True,
        "semantic_atol": ATOL,
        "semantic_rtol": ATOL,
        "no_regression_geomean_gate": NO_REGRESSION_GATE,
        "research_geomean_gate": RESEARCH_GATE,
        "worst_worker_gate": WORST_WORKER_GATE,
        "memory_ratio_gate": MEMORY_RATIO_GATE,
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
        or value.get("run_count") != 6
        or value.get("orders") != list(ORDERS)
        or value.get("warmups_per_side") != 3
        or value.get("groups_per_worker") != 30
        or value.get("native_value_bridge_included") is not True
        or value.get("correctness_capture_excluded") is not True
        or value.get("no_regression_geomean_gate") != NO_REGRESSION_GATE
        or value.get("research_geomean_gate") != RESEARCH_GATE
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("B4-C0 cumulative core protocol differs")


def validate_worker(value: Mapping[str, Any], ordinal: int) -> None:
    payload = dict(value)
    claimed = payload.pop("worker_hash", None)
    groups = value.get("groups")
    receipt = value.get("receipt")
    if (
        claimed != canonical_hash(payload)
        or value.get("schema_version") != WORKER_SCHEMA
        or value.get("run_ordinal") != ordinal
        or value.get("order") != ORDERS[ordinal]
        or value.get("warmups_per_side") != 3
        or value.get("group_count") != 30
        or not isinstance(groups, list)
        or len(groups) != 30
        or value.get("allclose") is not True
        or value.get("sign_exact") is not True
        or float(value.get("maximum_absolute_difference", math.inf)) > ATOL
        or not isinstance(receipt, Mapping)
        or receipt.get("evaluation_count") != 10
        or receipt.get("update_count") != 9
        or receipt.get("provider_activation_count") != 10
        or receipt.get("forward_launch_count") != 10
        or receipt.get("backward_launch_count") != 9
        or receipt.get("correctness_capture_enabled") is not False
        or receipt.get("unsupported_semantic_anchor_count") != 0
        or receipt.get("native_value_bridge_count") != 10
        or receipt.get("fallback_count") != 0
        or receipt.get("eager_count") != 0
        or receipt.get("adjoint_materialization_count") != 0
        or value.get("performance_claimed") is not False
    ):
        raise ValueError(f"B4-C0 cumulative core worker differs: {ordinal}")
    for group_ordinal, group in enumerate(groups):
        baseline = float(group["baseline_ms"])
        candidate = float(group["candidate_ms"])
        if (
            group.get("group_ordinal") != group_ordinal
            or group.get("order") != ORDERS[ordinal]
            or any(
                not math.isfinite(item) or item <= 0.0 for item in (baseline, candidate)
            )
            or group.get("speedup") != baseline / candidate
        ):
            raise ValueError(f"B4-C0 cumulative core group differs: {ordinal}")
    baseline_median = statistics.median(float(row["baseline_ms"]) for row in groups)
    candidate_median = statistics.median(float(row["candidate_ms"]) for row in groups)
    if (
        value.get("baseline_median_ms") != baseline_median
        or value.get("candidate_median_ms") != candidate_median
        or value.get("paired_speedup") != baseline_median / candidate_median
    ):
        raise ValueError(f"B4-C0 cumulative core derivation differs: {ordinal}")


def derive_summary(workers: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    if len(workers) != 6:
        raise ValueError("B4-C0 cumulative core worker count differs")
    for ordinal, worker in enumerate(workers):
        validate_worker(worker, ordinal)
    speedups = [float(worker["paired_speedup"]) for worker in workers]
    allocated_ratios = [
        float(worker["candidate_peak_allocated_bytes"])
        / float(worker["baseline_peak_allocated_bytes"])
        for worker in workers
    ]
    reserved_ratios = [
        float(worker["candidate_peak_reserved_bytes"])
        / float(worker["baseline_peak_reserved_bytes"])
        for worker in workers
    ]
    geometric_mean = geomean(speedups)
    lower = bootstrap_lower(speedups)
    worst = min(speedups)
    no_regression = (
        geometric_mean >= NO_REGRESSION_GATE
        and lower > 1.0
        and worst >= WORST_WORKER_GATE
        and max(allocated_ratios) <= MEMORY_RATIO_GATE
        and max(reserved_ratios) <= MEMORY_RATIO_GATE
    )
    research = no_regression and geometric_mean >= RESEARCH_GATE
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": (
            "validated-b4-c0-cumulative-core"
            if no_regression
            else "validated-no-go-b4-c0-native-value-bridge"
        ),
        "run_count": len(workers),
        "worker_speedups": speedups,
        "speedup_geomean": geometric_mean,
        "speedup_bootstrap_lower_95": lower,
        "speedup_worst_worker": worst,
        "baseline_worker_medians_ms": [
            float(worker["baseline_median_ms"]) for worker in workers
        ],
        "candidate_worker_medians_ms": [
            float(worker["candidate_median_ms"]) for worker in workers
        ],
        "maximum_absolute_difference": max(
            float(worker["maximum_absolute_difference"]) for worker in workers
        ),
        "sign_exact": True,
        "maximum_allocated_ratio": max(allocated_ratios),
        "maximum_reserved_ratio": max(reserved_ratios),
        "no_regression_admitted": no_regression,
        "research_speedup_admitted": research,
        "provider_ownership_rewrite_admitted": not no_regression,
        "native_value_bridge_included": True,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


def run_worker(
    *, source_capture: Path, model: Path, ordinal: int, order: str, output: Path
) -> None:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    inherited = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(REPOSITORY_ROOT), inherited) if item
    )
    subprocess.run(
        (
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/run_fsg4_b4c0_cumulative_core_worker.py"),
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
        raise ValueError("B4-C0 cumulative core manifest differs")
    workers = [
        load_json(root / "raw" / f"run_{ordinal:02d}_{order.lower()}.json")
        for ordinal, order in enumerate(ORDERS)
    ]
    if derive_summary(workers) != summary:
        raise ValueError("B4-C0 cumulative core semantic replay differs")
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
