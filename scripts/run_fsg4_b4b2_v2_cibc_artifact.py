#!/usr/bin/env python3
"""Generate or replay the CIBC-parity horizontal-fusion formal artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-lines,missing-function-docstring,subprocess-run-check
# pylint: disable=too-many-boolean-expressions,duplicate-code
# pylint: disable=wrong-import-position,redefined-outer-name

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
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import run_fsg4_b4b2_v2_cibc_worker as worker

ARTIFACT_SCHEMA = "boundflow.fsg4-b4b2-v2-cibc-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.fsg4-b4b2-v2-cibc-protocol/v1"
SUMMARY_SCHEMA = "boundflow.fsg4-b4b2-v2-cibc-summary/v1"
MANIFEST_SCHEMA = "boundflow.fsg4-b4b2-v2-cibc-manifest/v1"
TIMING_ORDERS = ("AB", "BA", "AB", "BA", "AB", "BA")
SPEEDUP_GATE = 1.20
BOOTSTRAP_LOWER_GATE = 1.0
WORST_WORKER_GATE = 0.98
MEMORY_RATIO_GATE = 1.05
RESEARCH_TARGET = 2.0
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260824
CODE_PATHS = (
    "boundflow/runtime/fsg4_b4b2_cibc_triton.py",
    "scripts/run_fsg4_b4b2_v2_cibc_worker.py",
    "scripts/run_fsg4_b4b2_v2_cibc_artifact.py",
    "tests/test_fsg4_b4b2_cibc_triton.py",
    "gemini_doc/BOUNDFLOW_FSG4_B4B2_V2_CIBC_PARITY_FUSION_PLAN_2026_08_24.md",
)
CAPTURE_ARTIFACT = REPOSITORY_ROOT / (
    "artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1"
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
        raise TypeError(f"CIBC artifact JSON root differs: {path}")
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


def capture_identity() -> dict[str, object]:
    manifest = CAPTURE_ARTIFACT / "manifest.json"
    return {
        "relative_path": str(CAPTURE_ARTIFACT.relative_to(REPOSITORY_ROOT)),
        "manifest_sha256": file_sha256(manifest),
        "manifest_hash": load_json(manifest)["manifest_hash"],
        "run_payload_sha256": [
            file_sha256(CAPTURE_ARTIFACT / f"run_{ordinal:02d}.pt")
            for ordinal in range(5)
        ],
    }


def protocol() -> dict[str, object]:
    source = git("rev-parse", "HEAD")
    value: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": source,
        "code_revision": code_revision(),
        "capture_identity": capture_identity(),
        "config_count": 12,
        "correctness_worker_count": 5,
        "timing_worker_count": 6,
        "timing_orders": list(TIMING_ORDERS),
        "calibration_warmups": worker.CALIBRATION_WARMUPS,
        "calibration_repeats": worker.CALIBRATION_REPEATS,
        "timing_warmups": worker.TIMING_WARMUPS,
        "timing_pairs": worker.TIMING_PAIRS,
        "speedup_geomean_gate": SPEEDUP_GATE,
        "bootstrap_lower_gate": BOOTSTRAP_LOWER_GATE,
        "worst_worker_gate": WORST_WORKER_GATE,
        "memory_ratio_gate": MEMORY_RATIO_GATE,
        "research_target": RESEARCH_TARGET,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "compile_and_calibration_excluded": True,
        "baseline": "public-pytorch-sparse-reconstruction-and-autograd-v1",
        "candidate": "triton-horizontal-fused-forward-backward-v2",
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
        or value.get("capture_identity") != capture_identity()
        or value.get("config_count") != 12
        or value.get("correctness_worker_count") != 5
        or value.get("timing_worker_count") != 6
        or value.get("timing_orders") != list(TIMING_ORDERS)
        or value.get("speedup_geomean_gate") != SPEEDUP_GATE
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("CIBC artifact protocol differs")


def validate_envelope(value: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    payload = dict(value)
    claimed = payload.pop("envelope_hash", None)
    result = value.get("result")
    if (
        claimed != worker.canonical_hash(payload)
        or value.get("schema_version") != worker.WORKER_SCHEMA
        or value.get("mode") != mode
        or value.get("performance_claimed") is not False
        or not isinstance(result, Mapping)
    ):
        raise ValueError("CIBC artifact worker envelope differs")
    result_payload = dict(result)
    result_hash = result_payload.pop("worker_hash", None)
    if result_hash != worker.canonical_hash(result_payload):
        raise ValueError("CIBC artifact worker result hash differs")
    return result


def geomean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("CIBC artifact geomean inputs differ")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def bootstrap_lower(values: Sequence[float]) -> float:
    generator = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[generator.randrange(len(values))] for _ in values]
        samples.append(geomean(sample))
    samples.sort()
    return samples[int(0.025 * len(samples))]


def _validate_receipt(result: Mapping[str, Any], winner: int) -> None:
    inventory = result.get("kernel_inventory")
    compilation = result.get("compilation")
    if (
        result.get("config_ordinal") != winner
        or not isinstance(inventory, Mapping)
        or inventory.get("kernel_names")
        != ["_cibc_forward_kernel_v2", "_cibc_backward_kernel_v2"]
        or inventory.get("forward_kernel_count") != 1
        or inventory.get("backward_kernel_count") != 1
        or inventory.get("total_kernel_count") != 2
        or inventory.get("global_intermediate_workspace_bytes") != 0
        or not isinstance(compilation, Mapping)
        or sorted(compilation) != ["backward", "forward"]
        or result.get("fallback_count") != 0
        or result.get("eager_count") != 0
    ):
        raise ValueError("CIBC artifact structural receipt differs")


def derive_summary(
    calibrations: Sequence[Mapping[str, Any]],
    correctness: Sequence[Mapping[str, Any]],
    timings: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    if len(calibrations) != 12 or [
        row.get("config_ordinal") for row in calibrations
    ] != list(range(12)):
        raise ValueError("CIBC artifact calibration inventory differs")
    for row in calibrations:
        parity = row.get("parity")
        if (
            not isinstance(parity, Mapping)
            or parity.get("allclose") is not True
            or parity.get("sign_exact") is not True
            or row.get("calibration_warmups") != worker.CALIBRATION_WARMUPS
            or row.get("calibration_repeats") != worker.CALIBRATION_REPEATS
            or len(row.get("samples_ms", [])) != worker.CALIBRATION_REPEATS
        ):
            raise ValueError("CIBC artifact calibration row differs")
    winner_row = min(calibrations, key=lambda row: float(row["median_ms"]))
    winner = int(winner_row["config_ordinal"])
    winner_compilation = winner_row["compilation"]
    maximum_difference = 0.0
    if len(correctness) != 5:
        raise ValueError("CIBC artifact correctness count differs")
    for ordinal, result in enumerate(correctness):
        parity = result.get("parity")
        _validate_receipt(result, winner)
        if (
            result.get("run_ordinal") != ordinal
            or result.get("semantic_passed") is not True
            or not isinstance(parity, Mapping)
            or parity.get("allclose") is not True
            or parity.get("sign_exact") is not True
            or result.get("compilation") != winner_compilation
        ):
            raise ValueError("CIBC artifact correctness row differs")
        maximum_difference = max(
            maximum_difference, float(parity["maximum_absolute_difference"])
        )
    speedups: list[float] = []
    allocated_ratios: list[float] = []
    reserved_ratios: list[float] = []
    if len(timings) != 6:
        raise ValueError("CIBC artifact timing count differs")
    for ordinal, result in enumerate(timings):
        pairs = result.get("pairs")
        parity = result.get("parity")
        _validate_receipt(result, winner)
        if (
            result.get("run_ordinal") != ordinal
            or result.get("order") != TIMING_ORDERS[ordinal]
            or result.get("warmups_per_side") != worker.TIMING_WARMUPS
            or result.get("pair_count") != worker.TIMING_PAIRS
            or not isinstance(pairs, list)
            or len(pairs) != worker.TIMING_PAIRS
            or not isinstance(parity, Mapping)
            or parity.get("allclose") is not True
            or parity.get("sign_exact") is not True
            or result.get("compilation") != winner_compilation
        ):
            raise ValueError("CIBC artifact timing row differs")
        baseline = statistics.median(float(row["baseline_ms"]) for row in pairs)
        candidate = statistics.median(float(row["candidate_ms"]) for row in pairs)
        speedup = baseline / candidate
        if (
            result.get("baseline_median_ms") != baseline
            or result.get("candidate_median_ms") != candidate
            or result.get("paired_speedup") != speedup
        ):
            raise ValueError("CIBC artifact timing derivation differs")
        speedups.append(speedup)
        allocated_ratios.append(float(result["allocated_ratio"]))
        reserved_ratios.append(float(result["reserved_ratio"]))
    aggregate = geomean(speedups)
    lower = bootstrap_lower(speedups)
    worst = min(speedups)
    allocated_max = max(allocated_ratios)
    reserved_max = max(reserved_ratios)
    admitted = (
        aggregate >= SPEEDUP_GATE
        and lower > BOOTSTRAP_LOWER_GATE
        and worst >= WORST_WORKER_GATE
        and allocated_max <= MEMORY_RATIO_GATE
        and reserved_max <= MEMORY_RATIO_GATE
    )
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": (
            "validated-b4-b2-v2-triton-physics"
            if admitted
            else "validated-no-go-b4-b2-v2-triton-physics"
        ),
        "winner_config_ordinal": winner,
        "winner_config": winner_row["config"],
        "winner_calibration_median_ms": winner_row["median_ms"],
        "winner_compilation": winner_compilation,
        "calibration_medians_ms": [row["median_ms"] for row in calibrations],
        "correctness_worker_count": len(correctness),
        "maximum_absolute_difference": maximum_difference,
        "sign_exact": True,
        "kernel_inventory": winner_row["kernel_inventory"],
        "speedups": speedups,
        "speedup_geomean": aggregate,
        "bootstrap_lower_95": lower,
        "worst_worker_speedup": worst,
        "maximum_allocated_ratio": allocated_max,
        "maximum_reserved_ratio": reserved_max,
        "minimum_go_admitted": admitted,
        "research_target_2x_met": aggregate >= RESEARCH_TARGET,
        "tvm_tir_port_admitted": admitted,
        "b4_b3_admitted": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


def run_worker(arguments: Sequence[str], output: Path) -> None:
    command = (
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/run_fsg4_b4b2_v2_cibc_worker.py"),
        *arguments,
        "--output",
        str(output),
    )
    subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        env=os.environ.copy(),
    )


def manifest(
    root: Path, protocol_value: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, object]:
    files = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    value: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "artifact_schema": ARTIFACT_SCHEMA,
        "source_git_head": protocol_value["source_git_head"],
        "protocol_hash": protocol_value["protocol_hash"],
        "summary_hash": summary["summary_hash"],
        "files": files,
    }
    value["manifest_hash"] = canonical_hash(value)
    return value


def generate(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    protocol_value = protocol()
    write_json(root / "protocol.json", protocol_value)
    calibration_paths = []
    for ordinal in range(12):
        path = root / "raw" / f"calibration_{ordinal:02d}.json"
        run_worker(("--mode", "calibration", "--config-ordinal", str(ordinal)), path)
        calibration_paths.append(path)
    calibrations = [
        validate_envelope(load_json(path), "calibration") for path in calibration_paths
    ]
    winner = int(
        min(calibrations, key=lambda row: float(row["median_ms"]))["config_ordinal"]
    )
    correctness_paths = []
    for ordinal in range(5):
        path = root / "raw" / f"correctness_{ordinal:02d}.json"
        run_worker(
            (
                "--mode",
                "correctness",
                "--run-ordinal",
                str(ordinal),
                "--config-ordinal",
                str(winner),
            ),
            path,
        )
        correctness_paths.append(path)
    timing_paths = []
    for ordinal, order in enumerate(TIMING_ORDERS):
        path = root / "raw" / f"timing_{ordinal:02d}_{order.lower()}.json"
        run_worker(
            (
                "--mode",
                "timing",
                "--run-ordinal",
                str(ordinal),
                "--config-ordinal",
                str(winner),
                "--order",
                order,
            ),
            path,
        )
        timing_paths.append(path)
    correctness = [
        validate_envelope(load_json(path), "correctness") for path in correctness_paths
    ]
    timings = [validate_envelope(load_json(path), "timing") for path in timing_paths]
    summary = derive_summary(calibrations, correctness, timings)
    write_json(root / "summary.json", summary)
    manifest_value = manifest(root, protocol_value, summary)
    write_json(root / "manifest.json", manifest_value)
    return summary


def replay(root: Path) -> dict[str, object]:
    protocol_value = load_json(root / "protocol.json")
    summary = load_json(root / "summary.json")
    manifest_value = load_json(root / "manifest.json")
    validate_protocol(protocol_value)
    manifest_payload = dict(manifest_value)
    claimed_manifest_hash = manifest_payload.pop("manifest_hash", None)
    expected_files = manifest_value.get("files")
    observed_files = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    if (
        claimed_manifest_hash != canonical_hash(manifest_payload)
        or manifest_value.get("schema_version") != MANIFEST_SCHEMA
        or manifest_value.get("protocol_hash") != protocol_value["protocol_hash"]
        or manifest_value.get("summary_hash") != summary.get("summary_hash")
        or expected_files != observed_files
    ):
        raise ValueError("CIBC artifact manifest differs")
    calibrations = [
        validate_envelope(
            load_json(root / "raw" / f"calibration_{ordinal:02d}.json"), "calibration"
        )
        for ordinal in range(12)
    ]
    correctness = [
        validate_envelope(
            load_json(root / "raw" / f"correctness_{ordinal:02d}.json"), "correctness"
        )
        for ordinal in range(5)
    ]
    timing_paths = [
        root / "raw" / f"timing_{ordinal:02d}_{order.lower()}.json"
        for ordinal, order in enumerate(TIMING_ORDERS)
    ]
    timings = [validate_envelope(load_json(path), "timing") for path in timing_paths]
    derived = derive_summary(calibrations, correctness, timings)
    if derived != summary:
        raise ValueError("CIBC artifact semantic replay differs")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    root = args.artifact.resolve()
    summary = replay(root) if args.replay else generate(root)
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
