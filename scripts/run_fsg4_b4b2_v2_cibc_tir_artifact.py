#!/usr/bin/env python3
"""Generate or replay the three-way manual-TIR formal artifact."""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-lines,missing-function-docstring,subprocess-run-check
# pylint: disable=too-many-boolean-expressions,wrong-import-position,duplicate-code
# pylint: disable=redefined-outer-name

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

from scripts import run_fsg4_b4b2_v2_cibc_tir_worker as worker

PROTOCOL_SCHEMA = "boundflow.fsg4-b4b2-v2-cibc-tir-protocol/v1"
SUMMARY_SCHEMA = "boundflow.fsg4-b4b2-v2-cibc-tir-summary/v1"
MANIFEST_SCHEMA = "boundflow.fsg4-b4b2-v2-cibc-tir-manifest/v1"
TIMING_ORDERS = ("BTR", "BRT", "TBR", "TRB", "RBT", "RTB")
BASELINE_SPEEDUP_GATE = 1.20
TRITON_RETENTION_GATE = 0.90
MEMORY_RATIO_GATE = 1.05
PARITY_TOLERANCE = 2.0e-4
PARITY_ELEMENT_COUNT = 12_810
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260824
CODE_PATHS = (
    "boundflow/backends/tvm/cibc_horizontal_fused_conv.py",
    "boundflow/runtime/fsg4_b4b2_cibc_tir.py",
    "scripts/run_fsg4_b4b2_v2_cibc_tir_worker.py",
    "scripts/run_fsg4_b4b2_v2_cibc_tir_artifact.py",
    "tests/test_fsg4_b4b2_cibc_tir.py",
    "gemini_doc/BOUNDFLOW_FSG4_B4B2_V2_CIBC_PARITY_FUSION_PLAN_2026_08_24.md",
)
CAPTURE_ARTIFACT = REPOSITORY_ROOT / (
    "artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1"
)
TRITON_ARTIFACT = REPOSITORY_ROOT / (
    "artifacts/fsg4-b4b2-v2-cibc-formal/resnet2b-prop0-v1"
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
        raise TypeError(f"CIBC TIR artifact JSON root differs: {path}")
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
    return {
        "manifest_sha256": file_sha256(CAPTURE_ARTIFACT / "manifest.json"),
        "run_payload_sha256": [
            file_sha256(CAPTURE_ARTIFACT / f"run_{ordinal:02d}.pt")
            for ordinal in range(5)
        ],
    }


def triton_identity() -> dict[str, object]:
    protocol = load_json(TRITON_ARTIFACT / "protocol.json")
    summary = load_json(TRITON_ARTIFACT / "summary.json")
    manifest = load_json(TRITON_ARTIFACT / "manifest.json")
    return {
        "source_git_head": protocol["source_git_head"],
        "winner_config_ordinal": summary["winner_config_ordinal"],
        "winner_config": summary["winner_config"],
        "summary_hash": summary["summary_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "summary_sha256": file_sha256(TRITON_ARTIFACT / "summary.json"),
    }


def protocol() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "source_git_head": git("rev-parse", "HEAD"),
        "code_revision": code_revision(),
        "capture_identity": capture_identity(),
        "triton_identity": triton_identity(),
        "correctness_worker_count": 5,
        "timing_worker_count": 6,
        "timing_orders": list(TIMING_ORDERS),
        "warmups_per_side": worker.TIMING_WARMUPS,
        "groups_per_worker": worker.TIMING_GROUPS,
        "baseline_speedup_gate": BASELINE_SPEEDUP_GATE,
        "triton_retention_gate": TRITON_RETENTION_GATE,
        "memory_ratio_gate": MEMORY_RATIO_GATE,
        "parity_tolerance": PARITY_TOLERANCE,
        "parity_element_count": PARITY_ELEMENT_COUNT,
        "compile_excluded": True,
        "plan_instance_reuse_included": True,
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
        or value.get("triton_identity") != triton_identity()
        or value.get("timing_orders") != list(TIMING_ORDERS)
        or value.get("baseline_speedup_gate") != BASELINE_SPEEDUP_GATE
        or value.get("triton_retention_gate") != TRITON_RETENTION_GATE
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("CIBC TIR protocol differs")


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
        raise ValueError("CIBC TIR worker envelope differs")
    result_payload = dict(result)
    result_hash = result_payload.pop("worker_hash", None)
    if result_hash != worker.canonical_hash(result_payload):
        raise ValueError("CIBC TIR worker hash differs")
    return result


def geomean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("CIBC TIR geomean inputs differ")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def bootstrap_lower(values: Sequence[float]) -> float:
    generator = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[generator.randrange(len(values))] for _ in values]
        samples.append(geomean(sample))
    samples.sort()
    return samples[int(0.025 * len(samples))]


def _validate_parity(parity: object) -> float:
    if not isinstance(parity, Mapping) or sorted(parity) != [
        "baseline_tir",
        "baseline_triton",
        "triton_tir",
    ]:
        raise ValueError("CIBC TIR parity inventory differs")
    maximum = 0.0
    for metric in parity.values():
        if (
            not isinstance(metric, Mapping)
            or metric.get("allclose") is not True
            or metric.get("sign_exact") is not True
            or metric.get("element_count") != PARITY_ELEMENT_COUNT
            or float(metric.get("maximum_absolute_difference", math.inf))
            > PARITY_TOLERANCE
        ):
            raise ValueError("CIBC TIR parity metric differs")
        maximum = max(maximum, float(metric["maximum_absolute_difference"]))
    return maximum


def _validate_receipts(
    result: Mapping[str, Any], expected_tir, expected_triton
) -> None:
    tir = result.get("tir_receipt")
    triton = result.get("triton_receipt")
    if (
        not isinstance(tir, Mapping)
        or tir.get("profiler_kernel_names")
        != [
            "boundflow_cibc_horizontal_forward_v2_kernel",
            "boundflow_cibc_horizontal_backward_v2_kernel",
        ]
        or tir.get("forward_kernel_count") != 1
        or tir.get("backward_kernel_count") != 1
        or tir.get("global_workspace_bytes") != 0
        or tir.get("plan_instance_reuses_dlpack_and_output_buffers") is not True
        or not isinstance(triton, Mapping)
        or triton.get("config_ordinal") != 1
        or triton.get("forward_kernel_count") != 1
        or triton.get("backward_kernel_count") != 1
        or triton.get("global_workspace_bytes") != 0
        or result.get("tir_fallback_count") != 0
        or result.get("triton_fallback_count") != 0
        or (expected_tir is not None and tir != expected_tir)
        or (expected_triton is not None and triton != expected_triton)
    ):
        raise ValueError("CIBC TIR structural receipt differs")


def derive_summary(
    correctness: Sequence[Mapping[str, Any]],
    timings: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    if len(correctness) != 5 or len(timings) != 6:
        raise ValueError("CIBC TIR worker count differs")
    expected_tir = correctness[0].get("tir_receipt")
    expected_triton = correctness[0].get("triton_receipt")
    maximum = 0.0
    for ordinal, result in enumerate(correctness):
        if (
            result.get("run_ordinal") != ordinal
            or result.get("semantic_passed") is not True
        ):
            raise ValueError("CIBC TIR correctness row differs")
        maximum = max(maximum, _validate_parity(result.get("parity")))
        _validate_receipts(result, expected_tir, expected_triton)
    baseline_speedups = []
    triton_retentions = []
    allocated_ratios = []
    reserved_ratios = []
    medians = []
    for ordinal, result in enumerate(timings):
        groups = result.get("groups")
        if (
            result.get("run_ordinal") != ordinal
            or result.get("order") != TIMING_ORDERS[ordinal]
            or result.get("warmups_per_side") != worker.TIMING_WARMUPS
            or result.get("group_count") != worker.TIMING_GROUPS
            or not isinstance(groups, list)
            or len(groups) != worker.TIMING_GROUPS
        ):
            raise ValueError("CIBC TIR timing row differs")
        maximum = max(maximum, _validate_parity(result.get("parity")))
        _validate_receipts(result, expected_tir, expected_triton)
        for group_ordinal, group in enumerate(groups):
            baseline = float(group["baseline_ms"])
            triton = float(group["triton_ms"])
            tir = float(group["tir_ms"])
            if (
                group.get("group_ordinal") != group_ordinal
                or group.get("order") != TIMING_ORDERS[ordinal]
                or any(
                    not math.isfinite(value) or value <= 0.0
                    for value in (baseline, triton, tir)
                )
                or group.get("baseline_over_tir") != baseline / tir
                or group.get("triton_over_tir") != triton / tir
            ):
                raise ValueError("CIBC TIR timing group differs")
        baseline_median = statistics.median(float(row["baseline_ms"]) for row in groups)
        triton_median = statistics.median(float(row["triton_ms"]) for row in groups)
        tir_median = statistics.median(float(row["tir_ms"]) for row in groups)
        baseline_speedup = baseline_median / tir_median
        triton_retention = triton_median / tir_median
        if (
            result.get("baseline_median_ms") != baseline_median
            or result.get("triton_median_ms") != triton_median
            or result.get("tir_median_ms") != tir_median
            or result.get("baseline_over_tir") != baseline_speedup
            or result.get("triton_over_tir") != triton_retention
        ):
            raise ValueError("CIBC TIR timing derivation differs")
        baseline_speedups.append(baseline_speedup)
        triton_retentions.append(triton_retention)
        allocated_ratios.append(
            float(result["tir_peak_allocated_bytes"])
            / float(result["baseline_peak_allocated_bytes"])
        )
        reserved_ratios.append(
            float(result["tir_peak_reserved_bytes"])
            / float(result["baseline_peak_reserved_bytes"])
        )
        medians.append(
            {
                "baseline_ms": baseline_median,
                "triton_ms": triton_median,
                "tir_ms": tir_median,
            }
        )
    baseline_geomean = geomean(baseline_speedups)
    triton_geomean = geomean(triton_retentions)
    admitted = (
        baseline_geomean >= BASELINE_SPEEDUP_GATE
        and bootstrap_lower(baseline_speedups) > 1.0
        and min(baseline_speedups) >= 0.98
        and triton_geomean >= TRITON_RETENTION_GATE
        and max(allocated_ratios) <= MEMORY_RATIO_GATE
        and max(reserved_ratios) <= MEMORY_RATIO_GATE
    )
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": (
            "validated-b4-b2-v2-manual-tir"
            if admitted
            else "validated-no-go-b4-b2-v2-manual-tir"
        ),
        "correctness_worker_count": len(correctness),
        "maximum_absolute_difference": maximum,
        "sign_exact": True,
        "tir_receipt": expected_tir,
        "triton_receipt": expected_triton,
        "worker_medians_ms": medians,
        "baseline_over_tir_speedups": baseline_speedups,
        "baseline_over_tir_geomean": baseline_geomean,
        "baseline_bootstrap_lower_95": bootstrap_lower(baseline_speedups),
        "baseline_worst_worker": min(baseline_speedups),
        "triton_over_tir_ratios": triton_retentions,
        "triton_over_tir_geomean": triton_geomean,
        "triton_retention_bootstrap_lower_95": bootstrap_lower(triton_retentions),
        "triton_retention_worst_worker": min(triton_retentions),
        "maximum_allocated_ratio": max(allocated_ratios),
        "maximum_reserved_ratio": max(reserved_ratios),
        "tir_port_admitted": admitted,
        "b4_b3_admitted": admitted,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


def run_worker(arguments: Sequence[str], output: Path) -> None:
    subprocess.run(
        (
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/run_fsg4_b4b2_v2_cibc_tir_worker.py"),
            *arguments,
            "--output",
            str(output),
        ),
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        env=os.environ.copy(),
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


def generate(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    protocol_value = protocol()
    write_json(root / "protocol.json", protocol_value)
    correctness_paths = []
    for ordinal in range(5):
        path = root / "raw" / f"correctness_{ordinal:02d}.json"
        run_worker(("--mode", "correctness", "--run-ordinal", str(ordinal)), path)
        correctness_paths.append(path)
    timing_paths = []
    for ordinal, order in enumerate(TIMING_ORDERS):
        path = root / "raw" / f"timing_{ordinal:02d}_{order.lower()}.json"
        run_worker(
            ("--mode", "timing", "--run-ordinal", str(ordinal), "--order", order),
            path,
        )
        timing_paths.append(path)
    correctness = [
        validate_envelope(load_json(path), "correctness") for path in correctness_paths
    ]
    timings = [validate_envelope(load_json(path), "timing") for path in timing_paths]
    summary = derive_summary(correctness, timings)
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
        raise ValueError("CIBC TIR manifest differs")
    correctness = [
        validate_envelope(
            load_json(root / "raw" / f"correctness_{ordinal:02d}.json"), "correctness"
        )
        for ordinal in range(5)
    ]
    timings = [
        validate_envelope(
            load_json(root / "raw" / f"timing_{ordinal:02d}_{order.lower()}.json"),
            "timing",
        )
        for ordinal, order in enumerate(TIMING_ORDERS)
    ]
    if derive_summary(correctness, timings) != summary:
        raise ValueError("CIBC TIR semantic replay differs")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    summary = (
        replay(args.artifact.resolve())
        if args.replay
        else generate(args.artifact.resolve())
    )
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
