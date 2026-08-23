#!/usr/bin/env python3
"""Fresh-process three-way PyTorch/Triton/manual-TIR formal worker."""

# mypy: disable-error-code=import-untyped
# pylint: disable=wrong-import-position,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring,import-error,duplicate-code
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from torch.profiler import ProfilerActivity, profile

from boundflow.backends.tvm.cibc_horizontal_fused_conv import (
    CIBC_TIR_BACKWARD_SYMBOL,
    CIBC_TIR_FORWARD_SYMBOL,
)
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.fsg4_b4b2_cibc_tir import PreparedCIBCHorizontalTIRV2
from boundflow.runtime.fsg4_b4b2_cibc_triton import (
    PreparedCIBCTritonTimingV2,
    triton_compilation_receipt_v2,
)
from boundflow.runtime.fsg4_b4b2_sparse_conv_timing import (
    compare_sparse_conv_executions_v1,
    cuda_event_call_ms_v1,
    measure_peak_memory_v1,
)

WORKER_SCHEMA = "boundflow.fsg4-b4b2-v2-cibc-tir-worker/v1"
CAPTURE_ARTIFACT = REPOSITORY_ROOT / (
    "artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1"
)
TRITON_WINNER_ORDINAL = 1
TIMING_WARMUPS = 10
TIMING_GROUPS = 30


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
    import triton
    import tvm
    import tvm_ffi

    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "tvm": tvm.__version__,
        "tvm_ffi": getattr(tvm_ffi, "__version__", "unknown"),
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
        raise RuntimeError("CIBC TIR GPU snapshot differs")
    return {
        "name": values[0],
        "temperature_celsius": int(values[1]),
        "power_draw_watts": float(values[2]),
        "graphics_clock_mhz": int(values[3]),
        "memory_clock_mhz": int(values[4]),
        "enforced_power_limit_watts": float(values[5]),
        "driver_version": values[6],
    }


def _warm(call: Callable[[], object]) -> None:
    for _ in range(TIMING_WARMUPS):
        call()
    torch.cuda.synchronize()


def _profiler_names(call: Callable[[], object]) -> list[str]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        call()
        torch.cuda.synchronize()
    return [
        event.name
        for event in profiler.events()
        if str(event.device_type).endswith("CUDA")
    ]


def _tir_receipt(prepared: PreparedCIBCHorizontalTIRV2) -> dict[str, object]:
    names = _profiler_names(prepared.candidate_once)
    expected = [
        CIBC_TIR_FORWARD_SYMBOL + "_kernel",
        CIBC_TIR_BACKWARD_SYMBOL + "_kernel",
    ]
    if names != expected:
        raise RuntimeError(f"CIBC TIR profiler inventory differs: {names}")
    source_names = re.findall(
        r'extern "C" __global__ void(?: __launch_bounds__\([^)]*\))? ([A-Za-z0-9_]+)\(',
        prepared.compiled.device_source,
    )
    if set(source_names) != set(expected):
        raise RuntimeError("CIBC TIR source kernel inventory differs")
    receipt = prepared.executor.compilation_receipt
    receipt.validate()
    return {
        "module_hash": receipt.module_hash,
        "device_source_hash": receipt.device_source_hash,
        "exported_symbols": list(receipt.exported_symbols),
        "source_kernel_names": source_names,
        "profiler_kernel_names": names,
        "forward_kernel_count": names.count(expected[0]),
        "backward_kernel_count": names.count(expected[1]),
        "global_workspace_bytes": receipt.global_workspace_bytes,
        "plan_instance_reuses_dlpack_and_output_buffers": True,
    }


def _triton_receipt(prepared: PreparedCIBCTritonTimingV2) -> dict[str, object]:
    names = _profiler_names(prepared.candidate_once)
    expected = ["_cibc_forward_kernel_v2", "_cibc_backward_kernel_v2"]
    if names != expected:
        raise RuntimeError("CIBC Triton comparison inventory differs")
    return {
        "config_ordinal": prepared.config.ordinal,
        "config": {
            "block_m": prepared.config.block_m,
            "block_k": prepared.config.block_k,
            "num_warps": prepared.config.num_warps,
        },
        "compilation": triton_compilation_receipt_v2(),
        "profiler_kernel_names": names,
        "forward_kernel_count": names.count(expected[0]),
        "backward_kernel_count": names.count(expected[1]),
        "global_workspace_bytes": 0,
    }


def _prepare(run_ordinal: int):
    capture = _capture(run_ordinal % 5)
    tir = PreparedCIBCHorizontalTIRV2(capture)
    triton = PreparedCIBCTritonTimingV2(capture, config_ordinal=TRITON_WINNER_ORDINAL)
    return tir, triton


def _parity(tir, triton) -> dict[str, object]:
    baseline = tir.baseline_once()
    tir_result = tir.candidate_once()
    triton_result = triton.candidate_once()
    baseline_tir = compare_sparse_conv_executions_v1(baseline, tir_result)
    baseline_triton = compare_sparse_conv_executions_v1(baseline, triton_result)
    triton_tir = compare_sparse_conv_executions_v1(triton_result, tir_result)
    for metric in (baseline_tir, baseline_triton, triton_tir):
        if not metric.allclose or not metric.sign_exact:
            raise RuntimeError("CIBC TIR three-way parity differs")
    return {
        "baseline_tir": baseline_tir.to_dict(),
        "baseline_triton": baseline_triton.to_dict(),
        "triton_tir": triton_tir.to_dict(),
    }


def _correctness(run_ordinal: int) -> dict[str, object]:
    tir, triton = _prepare(run_ordinal)
    value = {
        "run_ordinal": run_ordinal,
        "parity": _parity(tir, triton),
        "tir_receipt": _tir_receipt(tir),
        "triton_receipt": _triton_receipt(triton),
        "tir_fallback_count": tir.executor.fallback_count,
        "triton_fallback_count": triton.executor.fallback_count,
        "semantic_passed": True,
    }
    value["worker_hash"] = canonical_hash(value)
    return value


def _timing(run_ordinal: int, order: str) -> dict[str, object]:
    tir, triton = _prepare(run_ordinal)
    parity = _parity(tir, triton)
    calls = {
        "B": tir.baseline_once,
        "T": triton.candidate_once,
        "R": tir.candidate_once,
    }
    for call in calls.values():
        _warm(call)
    before = _gpu_snapshot()
    groups: list[dict[str, float | int | str]] = []
    for group_ordinal in range(TIMING_GROUPS):
        durations = {name: cuda_event_call_ms_v1(calls[name]) for name in order}
        groups.append(
            {
                "group_ordinal": group_ordinal,
                "order": order,
                "baseline_ms": durations["B"],
                "triton_ms": durations["T"],
                "tir_ms": durations["R"],
                "baseline_over_tir": durations["B"] / durations["R"],
                "triton_over_tir": durations["T"] / durations["R"],
            }
        )
    baseline_peak = measure_peak_memory_v1(tir.baseline_once)
    triton_peak = measure_peak_memory_v1(triton.candidate_once)
    tir_peak = measure_peak_memory_v1(tir.candidate_once)
    baseline_median = statistics.median(float(row["baseline_ms"]) for row in groups)
    triton_median = statistics.median(float(row["triton_ms"]) for row in groups)
    tir_median = statistics.median(float(row["tir_ms"]) for row in groups)
    value = {
        "run_ordinal": run_ordinal,
        "order": order,
        "warmups_per_side": TIMING_WARMUPS,
        "group_count": TIMING_GROUPS,
        "groups": groups,
        "baseline_median_ms": baseline_median,
        "triton_median_ms": triton_median,
        "tir_median_ms": tir_median,
        "baseline_over_tir": baseline_median / tir_median,
        "triton_over_tir": triton_median / tir_median,
        "baseline_peak_allocated_bytes": baseline_peak[0],
        "baseline_peak_reserved_bytes": baseline_peak[1],
        "triton_peak_allocated_bytes": triton_peak[0],
        "triton_peak_reserved_bytes": triton_peak[1],
        "tir_peak_allocated_bytes": tir_peak[0],
        "tir_peak_reserved_bytes": tir_peak[1],
        "parity": parity,
        "tir_receipt": _tir_receipt(tir),
        "triton_receipt": _triton_receipt(triton),
        "tir_fallback_count": tir.executor.fallback_count,
        "triton_fallback_count": triton.executor.fallback_count,
        "gpu_before": before,
        "gpu_after": _gpu_snapshot(),
    }
    value["worker_hash"] = canonical_hash(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("correctness", "timing"), required=True)
    parser.add_argument("--run-ordinal", type=int, required=True)
    parser.add_argument("--order", default="BTR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CIBC TIR worker requires CUDA")
    if args.mode == "correctness":
        if args.run_ordinal not in range(5):
            parser.error("correctness run ordinal differs")
        result = _correctness(args.run_ordinal)
    else:
        if args.run_ordinal not in range(6) or sorted(args.order) != ["B", "R", "T"]:
            parser.error("timing protocol differs")
        result = _timing(args.run_ordinal, args.order)
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
