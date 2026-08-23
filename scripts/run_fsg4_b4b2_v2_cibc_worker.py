#!/usr/bin/env python3
"""Fresh-process CIBC-parity correctness, calibration, and timing worker."""

# pylint: disable=wrong-import-position,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring,import-error

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from torch.profiler import ProfilerActivity, profile

from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.fsg4_b4b2_cibc_triton import (
    CIBC_TRITON_CONFIGS_V2,
    PreparedCIBCTritonTimingV2,
    triton_compilation_receipt_v2,
)
from boundflow.runtime.fsg4_b4b2_sparse_conv_timing import (
    compare_sparse_conv_executions_v1,
    cuda_event_call_ms_v1,
    measure_peak_memory_v1,
)

WORKER_SCHEMA = "boundflow.fsg4-b4b2-v2-cibc-worker/v1"
CAPTURE_ARTIFACT = REPOSITORY_ROOT / (
    "artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1"
)
CALIBRATION_WARMUPS = 3
CALIBRATION_REPEATS = 10
TIMING_WARMUPS = 10
TIMING_PAIRS = 30


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _capture(run_ordinal: int):
    payload = torch.load(
        CAPTURE_ARTIFACT / f"run_{run_ordinal:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    return production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][1]
    )


def _environment() -> dict[str, object]:
    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
    }


def _gpu_snapshot() -> dict[str, object]:
    query = (
        "name,temperature.gpu,power.draw,clocks.current.graphics,"
        "clocks.current.memory,enforced.power.limit,driver_version"
    )
    completed = subprocess.run(
        ("nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    values = [part.strip() for part in completed.stdout.strip().split(",")]
    if len(values) != 7:
        raise RuntimeError("CIBC Triton GPU snapshot differs")
    return {
        "name": values[0],
        "temperature_celsius": int(values[1]),
        "power_draw_watts": float(values[2]),
        "graphics_clock_mhz": int(values[3]),
        "memory_clock_mhz": int(values[4]),
        "enforced_power_limit_watts": float(values[5]),
        "driver_version": values[6],
    }


def _warm(call: Callable[[], object], count: int) -> None:
    for _ in range(count):
        call()
    torch.cuda.synchronize()


def _kernel_inventory(prepared: PreparedCIBCTritonTimingV2) -> dict[str, object]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        prepared.candidate_once()
        torch.cuda.synchronize()
    names = [
        event.name
        for event in profiler.events()
        if str(event.device_type).endswith("CUDA")
    ]
    expected = ["_cibc_forward_kernel_v2", "_cibc_backward_kernel_v2"]
    if names != expected:
        raise RuntimeError(f"CIBC Triton kernel inventory differs: {names}")
    return {
        "kernel_names": names,
        "forward_kernel_count": names.count(expected[0]),
        "backward_kernel_count": names.count(expected[1]),
        "total_kernel_count": len(names),
        "global_intermediate_workspace_bytes": 0,
    }


def _base_receipt(prepared: PreparedCIBCTritonTimingV2) -> dict[str, object]:
    return {
        "config_ordinal": prepared.config.ordinal,
        "config": {
            "block_m": prepared.config.block_m,
            "block_k": prepared.config.block_k,
            "num_warps": prepared.config.num_warps,
        },
        "template_hash": prepared.template.stable_hash(),
        "kernel_inventory": _kernel_inventory(prepared),
        "compilation": triton_compilation_receipt_v2(),
        "fallback_count": prepared.executor.fallback_count,
        "eager_count": prepared.executor.eager_count,
    }


def _correctness(run_ordinal: int, config_ordinal: int) -> dict[str, object]:
    prepared = PreparedCIBCTritonTimingV2(
        _capture(run_ordinal), config_ordinal=config_ordinal
    )
    parity = compare_sparse_conv_executions_v1(
        prepared.baseline_once(), prepared.candidate_once()
    )
    if not parity.allclose or not parity.sign_exact:
        raise RuntimeError("CIBC Triton correctness differs")
    value = {
        "run_ordinal": run_ordinal,
        "parity": parity.to_dict(),
        "semantic_passed": True,
        **_base_receipt(prepared),
    }
    value["worker_hash"] = canonical_hash(value)
    return value


def _calibration(config_ordinal: int) -> dict[str, object]:
    prepared = PreparedCIBCTritonTimingV2(_capture(0), config_ordinal=config_ordinal)
    parity = compare_sparse_conv_executions_v1(
        prepared.baseline_once(), prepared.candidate_once()
    )
    if not parity.allclose or not parity.sign_exact:
        raise RuntimeError("CIBC Triton calibration parity differs")
    _warm(prepared.candidate_once, CALIBRATION_WARMUPS)
    samples = [
        cuda_event_call_ms_v1(prepared.candidate_once)
        for _ in range(CALIBRATION_REPEATS)
    ]
    value = {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "calibration_warmups": CALIBRATION_WARMUPS,
        "calibration_repeats": CALIBRATION_REPEATS,
        "parity": parity.to_dict(),
        **_base_receipt(prepared),
    }
    value["worker_hash"] = canonical_hash(value)
    return value


def _timing(run_ordinal: int, config_ordinal: int, order: str) -> dict[str, object]:
    prepared = PreparedCIBCTritonTimingV2(
        _capture(run_ordinal % 5), config_ordinal=config_ordinal
    )
    parity = compare_sparse_conv_executions_v1(
        prepared.baseline_once(), prepared.candidate_once()
    )
    if not parity.allclose or not parity.sign_exact:
        raise RuntimeError("CIBC Triton timing parity differs")
    _warm(prepared.baseline_once, TIMING_WARMUPS)
    _warm(prepared.candidate_once, TIMING_WARMUPS)
    before = _gpu_snapshot()
    pairs: list[dict[str, float | int | str]] = []
    for pair_ordinal in range(TIMING_PAIRS):
        if order == "AB":
            baseline_ms = cuda_event_call_ms_v1(prepared.baseline_once)
            candidate_ms = cuda_event_call_ms_v1(prepared.candidate_once)
        else:
            candidate_ms = cuda_event_call_ms_v1(prepared.candidate_once)
            baseline_ms = cuda_event_call_ms_v1(prepared.baseline_once)
        pairs.append(
            {
                "pair_ordinal": pair_ordinal,
                "order": order,
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "speedup": baseline_ms / candidate_ms,
            }
        )
    baseline_peak = measure_peak_memory_v1(prepared.baseline_once)
    candidate_peak = measure_peak_memory_v1(prepared.candidate_once)
    baseline_median = statistics.median(float(row["baseline_ms"]) for row in pairs)
    candidate_median = statistics.median(float(row["candidate_ms"]) for row in pairs)
    value = {
        "run_ordinal": run_ordinal,
        "order": order,
        "warmups_per_side": TIMING_WARMUPS,
        "pair_count": TIMING_PAIRS,
        "pairs": pairs,
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "paired_speedup": baseline_median / candidate_median,
        "baseline_peak_allocated_bytes": baseline_peak[0],
        "baseline_peak_reserved_bytes": baseline_peak[1],
        "candidate_peak_allocated_bytes": candidate_peak[0],
        "candidate_peak_reserved_bytes": candidate_peak[1],
        "allocated_ratio": candidate_peak[0] / baseline_peak[0],
        "reserved_ratio": candidate_peak[1] / baseline_peak[1],
        "parity": parity.to_dict(),
        "gpu_before": before,
        "gpu_after": _gpu_snapshot(),
        **_base_receipt(prepared),
    }
    value["worker_hash"] = canonical_hash(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("correctness", "calibration", "timing"), required=True
    )
    parser.add_argument("--run-ordinal", type=int, default=0)
    parser.add_argument("--config-ordinal", type=int, required=True)
    parser.add_argument("--order", choices=("AB", "BA"), default="AB")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CIBC Triton worker requires CUDA")
    if args.config_ordinal not in range(len(CIBC_TRITON_CONFIGS_V2)):
        parser.error("config ordinal differs")
    if args.mode == "correctness":
        if args.run_ordinal not in range(5):
            parser.error("correctness run ordinal differs")
        result = _correctness(args.run_ordinal, args.config_ordinal)
    elif args.mode == "calibration":
        result = _calibration(args.config_ordinal)
    else:
        if args.run_ordinal not in range(6):
            parser.error("timing run ordinal differs")
        result = _timing(args.run_ordinal, args.config_ordinal, args.order)
    envelope = {
        "schema_version": WORKER_SCHEMA,
        "mode": args.mode,
        "environment": _environment(),
        "result": result,
        "performance_claimed": False,
    }
    envelope["envelope_hash"] = canonical_hash(envelope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(envelope) + "\n", encoding="utf-8")
    print(canonical_json(envelope))


if __name__ == "__main__":
    main()
