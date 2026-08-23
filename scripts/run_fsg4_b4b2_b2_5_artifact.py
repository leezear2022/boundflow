#!/usr/bin/env python3
"""Generate or replay the B4-B2 B2-5 formal micro-physics artifact."""

# pylint: disable=wrong-import-position,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-lines,missing-function-docstring
# pylint: disable=too-many-boolean-expressions,subprocess-run-check,duplicate-code

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
from typing import Any, Mapping, Sequence, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.fsg4_b4b2_sparse_conv_timing import (
    B2_5_BOOTSTRAP_SAMPLE_COUNT,
    B2_5_BOOTSTRAP_SEED,
    B2_5_MEMORY_RATIO_GATE,
    B2_5_PAIR_COUNT,
    B2_5_SPEEDUP_GATE,
    B2_5_WARMUP_COUNT,
    B2_5_WORST_WORKER_GATE,
    PreparedSparseConvTimingV1,
)
from scripts import run_fsg4_b4b2_b2_5_worker as worker

ARTIFACT_SCHEMA = "boundflow.fsg4-b4b2-b2-5-artifact/v1"
PROTOCOL_SCHEMA = "boundflow.fsg4-b4b2-b2-5-protocol/v1"
SUMMARY_SCHEMA = "boundflow.fsg4-b4b2-b2-5-summary/v1"
TIMING_ORDERS = ("AB", "BA", "AB", "BA", "AB", "BA")
CODE_PATHS = (
    "boundflow/ir/differentiable_lower_sparse_conv_tir.py",
    "boundflow/backends/tvm/differentiable_lower_sparse_conv.py",
    "boundflow/runtime/fsg4_b4b2_sparse_conv_tir.py",
    "boundflow/runtime/fsg4_b4b2_sparse_conv_timing.py",
    "boundflow/runtime/fsg4_b4b2_sparse_linear_tir.py",
    "scripts/run_fsg4_b4b2_b2_5_worker.py",
    "scripts/run_fsg4_b4b2_b2_5_artifact.py",
    "scripts/probe_fsg4_b4b2_b2_5_artifact_tamper.py",
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
        raise TypeError(f"B4-B2 B2-5 JSON root differs: {path.name}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def historical_file_sha256(source: str, path: str) -> str:
    if git("rev-parse", "HEAD") == source:
        return file_sha256(REPOSITORY_ROOT / path)
    content = subprocess.run(
        ("git", "show", f"{source}:{path}"),
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    return hashlib.sha256(content).hexdigest()


def code_revision(source: str | None = None) -> dict[str, str]:
    if source is None:
        return {path: file_sha256(REPOSITORY_ROOT / path) for path in CODE_PATHS}
    return {path: historical_file_sha256(source, path) for path in CODE_PATHS}


def capture_identity() -> dict[str, object]:
    manifest_path = CAPTURE_ARTIFACT / "manifest.json"
    return {
        "relative_path": str(CAPTURE_ARTIFACT.relative_to(REPOSITORY_ROOT)),
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_hash": load_json(manifest_path)["manifest_hash"],
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
        "candidate_count": 12,
        "correctness_workers": {"S": 5, "P": 5},
        "timing_worker_count": 6,
        "timing_orders": list(TIMING_ORDERS),
        "warmups_per_side": B2_5_WARMUP_COUNT,
        "pairs_per_worker": B2_5_PAIR_COUNT,
        "speedup_geomean_gate": B2_5_SPEEDUP_GATE,
        "bootstrap_lower_gate": 1.0,
        "worst_worker_gate": B2_5_WORST_WORKER_GATE,
        "memory_ratio_gate": B2_5_MEMORY_RATIO_GATE,
        "bootstrap_sample_count": B2_5_BOOTSTRAP_SAMPLE_COUNT,
        "bootstrap_seed": B2_5_BOOTSTRAP_SEED,
        "candidate_compile_cache_excluded": True,
        "wrapper_output_gradient_allocation_included": True,
        "module_call_is_not_kernel_count": True,
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
        or value.get("candidate_count") != 12
        or value.get("correctness_workers") != {"S": 5, "P": 5}
        or value.get("timing_worker_count") != 6
        or value.get("timing_orders") != list(TIMING_ORDERS)
        or value.get("warmups_per_side") != B2_5_WARMUP_COUNT
        or value.get("pairs_per_worker") != B2_5_PAIR_COUNT
        or value.get("speedup_geomean_gate") != B2_5_SPEEDUP_GATE
        or value.get("worst_worker_gate") != B2_5_WORST_WORKER_GATE
        or value.get("memory_ratio_gate") != B2_5_MEMORY_RATIO_GATE
        or value.get("module_call_is_not_kernel_count") is not True
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("B4-B2 B2-5 protocol differs")


def validate_worker_envelope(value: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    payload = dict(value)
    claimed = payload.pop("envelope_hash", None)
    result = value.get("result")
    if (
        claimed != canonical_hash(payload)
        or value.get("schema_version") != worker.WORKER_SCHEMA
        or value.get("mode") != mode
        or value.get("performance_claimed") is not False
        or not isinstance(result, Mapping)
    ):
        raise ValueError("B4-B2 B2-5 worker envelope differs")
    return result


def geomean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("B4-B2 B2-5 geomean inputs differ")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def bootstrap_lower(values: Sequence[float]) -> float:
    generator = random.Random(B2_5_BOOTSTRAP_SEED)
    samples = []
    for _ in range(B2_5_BOOTSTRAP_SAMPLE_COUNT):
        sample = [values[generator.randrange(len(values))] for _ in values]
        samples.append(geomean(sample))
    samples.sort()
    return samples[int(0.025 * len(samples))]


def derive_summary(
    calibration: Mapping[str, Any],
    correctness: Sequence[Mapping[str, Any]],
    timings: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    rows = calibration.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("B4-B2 B2-5 calibration ledger differs")
    ordinals = [
        row.get("candidate_ordinal") for row in rows if isinstance(row, Mapping)
    ]
    if ordinals != list(range(12)):
        raise ValueError("B4-B2 B2-5 calibration ordinals differ")
    winner_row = min(rows, key=lambda row: float(row["median_ms"]))
    winner = int(winner_row["candidate_ordinal"])
    if (
        calibration.get("candidate_count") != 12
        or calibration.get("winner_ordinal") != winner
        or calibration.get("winner_schedule_hash") != winner_row.get("schedule_hash")
        or calibration.get("winner_module_receipt_hash")
        != winner_row.get("module_receipt_hash")
        or calibration.get("winner_selected_from_raw") is not True
        or calibration.get("performance_claimed") is not False
    ):
        raise ValueError("B4-B2 B2-5 calibration winner differs")
    anchor_counts = {"S": 0, "P": 0}
    maximum_difference = 0.0
    for result in correctness:
        anchor = result.get("anchor")
        if (
            anchor not in anchor_counts
            or result.get("run_ordinal") != anchor_counts[anchor]
            or result.get("semantic_passed") is not True
            or result.get("fallback_count") != 0
            or result.get("eager_backward_count") != 0
            or result.get("module_call_count") != {"forward": 1, "backward": 1}
        ):
            raise ValueError("B4-B2 B2-5 correctness worker differs")
        anchor_counts[anchor] += 1
        maximum_difference = max(
            maximum_difference, float(result["maximum_absolute_difference"])
        )
    if anchor_counts != {"S": 5, "P": 5}:
        raise ValueError("B4-B2 B2-5 correctness count differs")
    speedups: list[float] = []
    allocated: list[float] = []
    reserved: list[float] = []
    kernel_inventory = winner_row.get("kernel_inventory")
    for ordinal, result in enumerate(timings):
        pairs = result.get("pairs")
        parity = result.get("parity")
        if (
            result.get("run_ordinal") != ordinal
            or result.get("order") != TIMING_ORDERS[ordinal]
            or result.get("candidate_ordinal") != winner
            or result.get("warmups_per_side") != B2_5_WARMUP_COUNT
            or result.get("pair_count") != B2_5_PAIR_COUNT
            or not isinstance(pairs, list)
            or len(pairs) != B2_5_PAIR_COUNT
            or not isinstance(parity, Mapping)
            or parity.get("allclose") is not True
            or parity.get("sign_exact") is not True
            or result.get("kernel_inventory") != kernel_inventory
            or result.get("module_call_count") != {"forward": 1, "backward": 1}
            or result.get("fallback_count") != 0
            or result.get("eager_backward_count") != 0
            or result.get("performance_claimed") is not False
        ):
            raise ValueError("B4-B2 B2-5 timing worker differs")
        baseline_median = statistics.median(float(row["baseline_ms"]) for row in pairs)
        candidate_median = statistics.median(
            float(row["candidate_ms"]) for row in pairs
        )
        speedup = baseline_median / candidate_median
        if (
            result.get("baseline_median_ms") != baseline_median
            or result.get("candidate_median_ms") != candidate_median
            or result.get("paired_speedup") != speedup
        ):
            raise ValueError("B4-B2 B2-5 timing derivation differs")
        speedups.append(speedup)
        allocated.append(float(result["allocated_ratio"]))
        reserved.append(float(result["reserved_ratio"]))
    if len(timings) != 6:
        raise ValueError("B4-B2 B2-5 timing worker count differs")
    speedup_geomean = geomean(speedups)
    confidence_lower = bootstrap_lower(speedups)
    worst = min(speedups)
    allocated_max = max(allocated)
    reserved_max = max(reserved)
    timing_admitted = (
        speedup_geomean >= B2_5_SPEEDUP_GATE
        and confidence_lower > 1.0
        and worst >= B2_5_WORST_WORKER_GATE
        and allocated_max <= B2_5_MEMORY_RATIO_GATE
        and reserved_max <= B2_5_MEMORY_RATIO_GATE
    )
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": (
            "validated-b4-b2-typed-cuda-tir-candidate"
            if timing_admitted
            else "validated-no-go-b4-b2-v1-physics"
        ),
        "winner_ordinal": winner,
        "winner_schedule_hash": winner_row["schedule_hash"],
        "winner_module_receipt_hash": winner_row["module_receipt_hash"],
        "correctness_worker_count": len(correctness),
        "timing_worker_count": len(timings),
        "maximum_absolute_difference": maximum_difference,
        "worker_speedups": speedups,
        "paired_speedup_geomean": speedup_geomean,
        "bootstrap_95_lower": confidence_lower,
        "worst_worker_speedup": worst,
        "maximum_allocated_ratio": allocated_max,
        "maximum_reserved_ratio": reserved_max,
        "speedup_geomean_passed": speedup_geomean >= B2_5_SPEEDUP_GATE,
        "bootstrap_lower_passed": confidence_lower > 1.0,
        "worst_worker_passed": worst >= B2_5_WORST_WORKER_GATE,
        "allocated_passed": allocated_max <= B2_5_MEMORY_RATIO_GATE,
        "reserved_passed": reserved_max <= B2_5_MEMORY_RATIO_GATE,
        "kernel_inventory": kernel_inventory,
        "timing_admitted": timing_admitted,
        "b4b3_open": timing_admitted,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


def run_worker(
    command: Sequence[str], output: Path, stdout: Path, stderr: Path
) -> None:
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        tuple(command),
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stdout.write_text(completed.stdout, encoding="utf-8")
    stderr.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"B4-B2 B2-5 worker failed: exit={completed.returncode}: {completed.stderr[-1000:]}"
        )


def worker_command(*args: str) -> tuple[str, ...]:
    return (
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/run_fsg4_b4b2_b2_5_worker.py"),
        *args,
    )


def collect_files(artifact: Path) -> dict[str, str]:
    return {
        str(path.relative_to(artifact)): file_sha256(path)
        for path in sorted(artifact.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def generate(artifact: Path) -> dict[str, object]:
    if artifact.exists():
        raise FileExistsError(f"B4-B2 B2-5 artifact already exists: {artifact}")
    if subprocess.run(("git", "diff", "--quiet"), cwd=REPOSITORY_ROOT).returncode != 0:
        raise RuntimeError("B4-B2 B2-5 generate requires a clean tracked worktree")
    if (
        subprocess.run(
            ("git", "diff", "--cached", "--quiet"), cwd=REPOSITORY_ROOT
        ).returncode
        != 0
    ):
        raise RuntimeError("B4-B2 B2-5 generate requires a clean index")
    artifact.mkdir(parents=True)
    protocol_value = protocol()
    write_json(artifact / "protocol.json", protocol_value)
    calibration_path = artifact / "calibration.json"
    run_worker(
        worker_command("--mode", "calibrate", "--output", str(calibration_path)),
        calibration_path,
        artifact / "logs/calibration.stdout.txt",
        artifact / "logs/calibration.stderr.txt",
    )
    calibration = validate_worker_envelope(load_json(calibration_path), "calibrate")
    winner = int(calibration["winner_ordinal"])
    correctness_results: list[Mapping[str, Any]] = []
    for anchor in ("S", "P"):
        for ordinal in range(5):
            index = len(correctness_results)
            output = artifact / f"correctness/{anchor}_{ordinal:02d}.json"
            run_worker(
                worker_command(
                    "--mode",
                    "correctness",
                    "--anchor",
                    anchor,
                    "--run-ordinal",
                    str(ordinal),
                    "--output",
                    str(output),
                ),
                output,
                artifact / f"logs/correctness_{index:02d}.stdout.txt",
                artifact / f"logs/correctness_{index:02d}.stderr.txt",
            )
            correctness_results.append(
                validate_worker_envelope(load_json(output), "correctness")
            )
    timing_results: list[Mapping[str, Any]] = []
    for ordinal, order in enumerate(TIMING_ORDERS):
        output = artifact / f"timing/run_{ordinal:02d}.json"
        run_worker(
            worker_command(
                "--mode",
                "timing",
                "--run-ordinal",
                str(ordinal),
                "--candidate-ordinal",
                str(winner),
                "--order",
                order,
                "--output",
                str(output),
            ),
            output,
            artifact / f"logs/timing_{ordinal:02d}.stdout.txt",
            artifact / f"logs/timing_{ordinal:02d}.stderr.txt",
        )
        timing_results.append(validate_worker_envelope(load_json(output), "timing"))
    summary = derive_summary(calibration, correctness_results, timing_results)
    write_json(artifact / "summary.json", summary)
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA,
        "source_git_head": protocol_value["source_git_head"],
        "protocol_hash": protocol_value["protocol_hash"],
        "summary_hash": summary["summary_hash"],
        "files": collect_files(artifact),
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    write_json(artifact / "manifest.json", manifest)
    return manifest


def _load_results(artifact: Path):
    calibration = validate_worker_envelope(
        load_json(artifact / "calibration.json"), "calibrate"
    )
    correctness = [
        validate_worker_envelope(load_json(path), "correctness")
        for path in sorted((artifact / "correctness").glob("*.json"))
    ]
    timings = [
        validate_worker_envelope(load_json(path), "timing")
        for path in sorted((artifact / "timing").glob("*.json"))
    ]
    return calibration, correctness, timings


def recompile_receipt(summary: Mapping[str, Any]) -> dict[str, object]:
    payload = torch.load(
        CAPTURE_ARTIFACT / "run_00.pt", map_location="cpu", weights_only=False
    )
    capture = production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][1]
    )
    prepared = PreparedSparseConvTimingV1(
        capture, candidate_ordinal=int(summary["winner_ordinal"])
    )
    return {
        "template_hash": prepared.template.stable_hash(),
        "ledger_hash": prepared.ledger.stable_hash(
            prepared.template, prepared.schedules
        ),
        "schedule_hash": prepared.schedule.stable_hash(prepared.template),
        "module_receipt_hash": prepared.module_receipt.stable_hash(
            prepared.template, prepared.schedule
        ),
        "unscheduled_tir_hash": prepared.module_receipt.unscheduled_tir_hash,
        "scheduled_tir_hash": prepared.module_receipt.scheduled_tir_hash,
        "device_source_hash": prepared.module_receipt.device_source_hash,
        "kernel_inventory": prepared.kernel_inventory.to_dict(),
    }


def replay(artifact: Path, *, recompile: bool = True) -> dict[str, object]:
    manifest = load_json(artifact / "manifest.json")
    manifest_payload = dict(manifest)
    claimed_manifest_hash = manifest_payload.pop("manifest_hash", None)
    files = manifest.get("files")
    if (
        claimed_manifest_hash != canonical_hash(manifest_payload)
        or manifest.get("schema_version") != ARTIFACT_SCHEMA
        or manifest.get("performance_claimed") is not False
        or not isinstance(files, Mapping)
        or files != collect_files(artifact)
    ):
        raise ValueError("B4-B2 B2-5 manifest differs")
    protocol_value = load_json(artifact / "protocol.json")
    validate_protocol(protocol_value)
    if manifest.get("source_git_head") != protocol_value.get(
        "source_git_head"
    ) or manifest.get("protocol_hash") != protocol_value.get("protocol_hash"):
        raise ValueError("B4-B2 B2-5 manifest protocol binding differs")
    calibration, correctness, timings = _load_results(artifact)
    derived = derive_summary(calibration, correctness, timings)
    frozen = load_json(artifact / "summary.json")
    if derived != frozen or manifest.get("summary_hash") != derived.get("summary_hash"):
        raise ValueError("B4-B2 B2-5 frozen summary differs")
    receipt: dict[str, object] | None = None
    if recompile:
        receipt = recompile_receipt(derived)
        timing_receipt = timings[0]
        calibration_rows = calibration["rows"]
        winner_row = calibration_rows[cast(int, derived["winner_ordinal"])]
        if (
            receipt["schedule_hash"] != derived["winner_schedule_hash"]
            or receipt["module_receipt_hash"] != derived["winner_module_receipt_hash"]
            or receipt["module_receipt_hash"] != timing_receipt["module_receipt_hash"]
            or receipt["kernel_inventory"] != derived["kernel_inventory"]
            or receipt["kernel_inventory"] != winner_row["kernel_inventory"]
        ):
            raise ValueError("B4-B2 B2-5 independent recompile differs")
    return {
        "status": "replay-pass",
        "manifest_hash": manifest["manifest_hash"],
        "summary_hash": derived["summary_hash"],
        "timing_admitted": derived["timing_admitted"],
        "result_status": derived["status"],
        "recompile_receipt": receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "replay"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--artifact-dir", type=Path, required=True)
    replay_parser = subparsers.choices["replay"]
    replay_parser.add_argument("--no-recompile", action="store_true")
    args = parser.parse_args()
    if args.command == "generate":
        result = generate(args.artifact_dir)
    else:
        result = replay(args.artifact_dir, recompile=not args.no_recompile)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
