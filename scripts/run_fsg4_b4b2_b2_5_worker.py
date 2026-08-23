#!/usr/bin/env python3
"""Independent B4-B2 B2-5 correctness, calibration, and timing worker."""

# pylint: disable=wrong-import-position,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring,import-outside-toplevel,import-error

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Callable, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.fsg4_b4b2_sparse_conv_timing import (
    B2_5_PAIR_COUNT,
    B2_5_WARMUP_COUNT,
    PreparedSparseConvTimingV1,
    compare_sparse_conv_executions_v1,
    cuda_event_call_ms_v1,
    measure_peak_memory_v1,
)
from boundflow.runtime.fsg4_b4b2_sparse_conv_tir import (
    DifferentiableLowerSparseConvModuleCache,
    run_b4b2_sparse_conv_tir_v1,
)
from boundflow.runtime.fsg4_b4b2_sparse_linear_tir import (
    DifferentiableLowerSparseLinearModuleCache,
    run_b4b2_sparse_linear_tir_v1,
)

WORKER_SCHEMA = "boundflow.fsg4-b4b2-b2-5-worker/v1"
CAPTURE_ARTIFACT = REPOSITORY_ROOT / (
    "artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1"
)
CALIBRATION_WARMUPS = 3
CALIBRATION_REPEATS = 10


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _capture(run_ordinal: int, anchor_ordinal: int):
    payload = torch.load(
        CAPTURE_ARTIFACT / f"run_{run_ordinal:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    return production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][anchor_ordinal]
    )


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
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("B4-B2 B2-5 GPU inventory differs")
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != 7:
        raise RuntimeError("B4-B2 B2-5 GPU snapshot differs")
    return {
        "name": values[0],
        "temperature_celsius": int(values[1]),
        "power_draw_watts": float(values[2]),
        "graphics_clock_mhz": int(values[3]),
        "memory_clock_mhz": int(values[4]),
        "enforced_power_limit_watts": float(values[5]),
        "driver_version": values[6],
    }


def _environment() -> dict[str, object]:
    import tvm  # type: ignore[import-untyped]
    import tvm_ffi

    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tvm": tvm.__version__,
        "tvm_ffi": getattr(tvm_ffi, "__version__", "unknown"),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
    }


def _metric_rows(result) -> list[dict[str, object]]:
    return [metric.to_dict() for metric in result.metrics]


def _correctness(anchor: str, run_ordinal: int) -> dict[str, object]:
    if anchor == "S":
        capture = _capture(run_ordinal, 0)
        linear_result = run_b4b2_sparse_linear_tir_v1(
            capture,
            fresh_run_ordinal=run_ordinal,
            cache=DifferentiableLowerSparseLinearModuleCache(),
        )
        linear_projection = linear_result.projection_receipt
        metrics = _metric_rows(linear_result)
        value: dict[str, object] = {
            "anchor": "S",
            "run_ordinal": run_ordinal,
            "metrics": metrics,
            "semantic_passed": all(
                metric.allclose and metric.sign_exact
                for metric in linear_result.metrics
            ),
            "module_call_count": {
                "forward": linear_result.launch_receipt.forward_launch_count,
                "backward": linear_result.launch_receipt.backward_launch_count,
            },
            "fallback_count": linear_result.launch_receipt.fallback_count,
            "eager_backward_count": linear_result.launch_receipt.eager_backward_count,
            "projection": {
                "alpha_mapping_exact": linear_projection.alpha_mapping_exact,
                "beta_mapping_exact": linear_projection.beta_mapping_exact,
                "alpha_numerical_passed": linear_projection.alpha_numerical_passed,
                "beta_numerical_passed": linear_projection.beta_numerical_passed,
                "nonzero_sign_exact": linear_projection.nonzero_sign_exact,
                "unowned_native_zero_exact": (
                    linear_projection.unowned_native_zero_exact
                ),
            },
        }
    else:
        capture = _capture(run_ordinal, 1)
        conv_result = run_b4b2_sparse_conv_tir_v1(
            capture,
            fresh_run_ordinal=run_ordinal,
            candidate_ordinal=0,
            cache=DifferentiableLowerSparseConvModuleCache(),
        )
        conv_projection = conv_result.projection_receipt
        metrics = _metric_rows(conv_result)
        value = {
            "anchor": "P",
            "run_ordinal": run_ordinal,
            "metrics": metrics,
            "semantic_passed": all(
                metric.allclose and metric.sign_exact for metric in conv_result.metrics
            ),
            "module_call_count": {
                "forward": conv_result.launch_receipt.forward_launch_count,
                "backward": conv_result.launch_receipt.backward_launch_count,
            },
            "fallback_count": conv_result.launch_receipt.fallback_count,
            "eager_backward_count": conv_result.launch_receipt.eager_backward_count,
            "projection": {
                "coordinate_mapping_exact": conv_projection.coordinate_mapping_exact,
                "alpha_numerical_passed": conv_projection.alpha_numerical_passed,
                "nonzero_sign_exact": conv_projection.nonzero_sign_exact,
                "unowned_native_zero_exact": conv_projection.unowned_native_zero_exact,
                "beta_gradient_absent": conv_projection.beta_gradient_absent,
            },
        }
    value["maximum_absolute_difference"] = max(
        cast(float, row["maximum_absolute_difference"]) for row in metrics
    )
    value["worker_hash"] = _canonical_hash(value)
    return value


def _warm(call: Callable[[], object], count: int) -> None:
    for _ in range(count):
        call()
    torch.cuda.synchronize()


def _calibration() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for ordinal in range(12):
        prepared = PreparedSparseConvTimingV1(_capture(0, 1), candidate_ordinal=ordinal)
        parity = compare_sparse_conv_executions_v1(
            prepared.baseline_once(), prepared.candidate_once()
        )
        if not parity.allclose or not parity.sign_exact:
            raise RuntimeError("B4-B2 B2-5 calibration parity failed")
        _warm(prepared.candidate_once, CALIBRATION_WARMUPS)
        samples = [
            cuda_event_call_ms_v1(prepared.candidate_once)
            for _ in range(CALIBRATION_REPEATS)
        ]
        rows.append(
            {
                "candidate_ordinal": ordinal,
                "knobs": list(prepared.schedule.knob_tuple),
                "schedule_hash": prepared.schedule.stable_hash(prepared.template),
                "module_receipt_hash": prepared.module_receipt.stable_hash(
                    prepared.template, prepared.schedule
                ),
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "parity": parity.to_dict(),
                "kernel_inventory": prepared.kernel_inventory.to_dict(),
            }
        )
    winner = min(rows, key=lambda row: cast(float, row["median_ms"]))
    return {
        "calibration_warmups": CALIBRATION_WARMUPS,
        "calibration_repeats": CALIBRATION_REPEATS,
        "candidate_count": len(rows),
        "rows": rows,
        "winner_ordinal": winner["candidate_ordinal"],
        "winner_schedule_hash": winner["schedule_hash"],
        "winner_module_receipt_hash": winner["module_receipt_hash"],
        "winner_selected_from_raw": True,
        "performance_claimed": False,
    }


def _timing(run_ordinal: int, candidate_ordinal: int, order: str) -> dict[str, object]:
    prepared = PreparedSparseConvTimingV1(
        _capture(run_ordinal % 5, 1), candidate_ordinal=candidate_ordinal
    )
    baseline = prepared.baseline_once()
    candidate = prepared.candidate_once()
    parity = compare_sparse_conv_executions_v1(baseline, candidate)
    if not parity.allclose or not parity.sign_exact:
        raise RuntimeError("B4-B2 B2-5 timing parity failed")
    _warm(prepared.baseline_once, B2_5_WARMUP_COUNT)
    _warm(prepared.candidate_once, B2_5_WARMUP_COUNT)
    before = _gpu_snapshot()
    pairs: list[dict[str, object]] = []
    for pair in range(B2_5_PAIR_COUNT):
        if order == "AB":
            baseline_ms = cuda_event_call_ms_v1(prepared.baseline_once)
            candidate_ms = cuda_event_call_ms_v1(prepared.candidate_once)
        else:
            candidate_ms = cuda_event_call_ms_v1(prepared.candidate_once)
            baseline_ms = cuda_event_call_ms_v1(prepared.baseline_once)
        pairs.append(
            {
                "pair_ordinal": pair,
                "order": order,
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "speedup": baseline_ms / candidate_ms,
            }
        )
    baseline_peak = measure_peak_memory_v1(prepared.baseline_once)
    candidate_peak = measure_peak_memory_v1(prepared.candidate_once)
    after = _gpu_snapshot()
    baseline_median = statistics.median(
        cast(float, row["baseline_ms"]) for row in pairs
    )
    candidate_median = statistics.median(
        cast(float, row["candidate_ms"]) for row in pairs
    )
    value = {
        "run_ordinal": run_ordinal,
        "order": order,
        "candidate_ordinal": candidate_ordinal,
        "warmups_per_side": B2_5_WARMUP_COUNT,
        "pair_count": B2_5_PAIR_COUNT,
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
        "kernel_inventory": prepared.kernel_inventory.to_dict(),
        "template_hash": prepared.template.stable_hash(),
        "ledger_hash": prepared.ledger.stable_hash(
            prepared.template, prepared.schedules
        ),
        "schedule_hash": prepared.schedule.stable_hash(prepared.template),
        "module_receipt_hash": prepared.module_receipt.stable_hash(
            prepared.template, prepared.schedule
        ),
        "module_call_count": {"forward": 1, "backward": 1},
        "fallback_count": 0,
        "eager_backward_count": 0,
        "gpu_before": before,
        "gpu_after": after,
        "performance_claimed": False,
    }
    value["worker_hash"] = _canonical_hash(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("correctness", "calibrate", "timing"), required=True
    )
    parser.add_argument("--anchor", choices=("S", "P"))
    parser.add_argument("--run-ordinal", type=int, default=0)
    parser.add_argument("--candidate-ordinal", type=int, default=0)
    parser.add_argument("--order", choices=("AB", "BA"), default="AB")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("B4-B2 B2-5 worker requires CUDA")
    if args.mode == "correctness":
        if args.anchor is None or args.run_ordinal not in range(5):
            parser.error("correctness requires anchor and run ordinal 0..4")
        result = _correctness(args.anchor, args.run_ordinal)
    elif args.mode == "calibrate":
        result = _calibration()
    else:
        if args.run_ordinal not in range(6) or args.candidate_ordinal not in range(12):
            parser.error("timing ordinal differs")
        result = _timing(args.run_ordinal, args.candidate_ordinal, args.order)
    envelope = {
        "schema_version": WORKER_SCHEMA,
        "mode": args.mode,
        "environment": _environment(),
        "result": result,
        "performance_claimed": False,
    }
    envelope["envelope_hash"] = _canonical_hash(envelope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_canonical_json(envelope) + "\n", encoding="utf-8")
    print(_canonical_json(envelope))


if __name__ == "__main__":
    main()
