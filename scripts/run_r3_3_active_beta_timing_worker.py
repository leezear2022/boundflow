#!/usr/bin/env python3
"""Run one independent R3-3 active-beta isolated timing worker."""

# pylint: disable=wrong-import-position,too-many-locals,missing-function-docstring

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.r3_3_active_beta_timing import (
    R3_3_TIMING_PAIR_COUNT,
    R3_3_TIMING_WARMUP_COUNT,
    PreparedR3ActiveBetaTimingV1,
    compare_r3_active_beta_executions_v1,
    cuda_event_wrapper_ms_v1,
    measure_r3_active_beta_memory_v1,
)

CAPTURE = ROOT / "artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1"
SCHEMA = "boundflow.r3-3-active-beta-timing-worker/v1"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(run_ordinal: int):  # type: ignore[no-untyped-def]
    capture_ordinal = run_ordinal % 5
    path = CAPTURE / f"run_{capture_ordinal:02d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return path, production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][0]
    )


def _gpu_snapshot() -> dict[str, object]:
    query = (
        "name,temperature.gpu,power.draw,clocks.current.graphics,"
        "clocks.current.memory,enforced.power.limit,driver_version"
    )
    completed = subprocess.run(
        ("nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"),
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("R3-3 timing GPU inventory differs")
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != 7:
        raise RuntimeError("R3-3 timing GPU snapshot differs")
    return {
        "name": values[0],
        "temperature_celsius": int(values[1]),
        "power_draw_watts": float(values[2]),
        "graphics_clock_mhz": int(values[3]),
        "memory_clock_mhz": int(values[4]),
        "enforced_power_limit_watts": float(values[5]),
        "driver_version": values[6],
    }


def _warm(call) -> None:  # type: ignore[no-untyped-def]
    for _ in range(R3_3_TIMING_WARMUP_COUNT):
        call()
    torch.cuda.synchronize()


def _run(run_ordinal: int, order: str) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("R3-3 timing worker requires CUDA")
    capture_path, capture = _capture(run_ordinal)
    prepared = PreparedR3ActiveBetaTimingV1(capture)
    parity = compare_r3_active_beta_executions_v1(
        prepared.baseline_once(), prepared.candidate_once()
    )
    if not parity.allclose or not parity.sign_exact:
        raise RuntimeError("R3-3 timing parity failed")
    _warm(prepared.baseline_once)
    _warm(prepared.candidate_once)
    gpu_before = _gpu_snapshot()
    pairs: list[dict[str, object]] = []
    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    for ordinal in range(R3_3_TIMING_PAIR_COUNT):
        if order == "AB":
            baseline_ms = cuda_event_wrapper_ms_v1(prepared.baseline_once)
            candidate_ms = cuda_event_wrapper_ms_v1(prepared.candidate_once)
        else:
            candidate_ms = cuda_event_wrapper_ms_v1(prepared.candidate_once)
            baseline_ms = cuda_event_wrapper_ms_v1(prepared.baseline_once)
        pairs.append(
            {
                "pair_ordinal": ordinal,
                "order": order,
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "speedup": baseline_ms / candidate_ms,
            }
        )
        baseline_samples.append(baseline_ms)
        candidate_samples.append(candidate_ms)
    baseline_memory = measure_r3_active_beta_memory_v1(prepared.baseline_once)
    candidate_memory = measure_r3_active_beta_memory_v1(prepared.candidate_once)
    gpu_after = _gpu_snapshot()
    baseline_median = statistics.median(baseline_samples)
    candidate_median = statistics.median(candidate_samples)
    value: dict[str, object] = {
        "schema_version": SCHEMA,
        "run_ordinal": run_ordinal,
        "capture_ordinal": run_ordinal % 5,
        "capture_sha256": _file_hash(capture_path),
        "order": order,
        "warmup_count": R3_3_TIMING_WARMUP_COUNT,
        "pair_count": R3_3_TIMING_PAIR_COUNT,
        "pairs": pairs,
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "paired_speedup": baseline_median / candidate_median,
        "parity": parity.to_dict(),
        "baseline_memory": baseline_memory.to_dict(),
        "candidate_memory": candidate_memory.to_dict(),
        "template_hash": prepared.template.stable_hash(),
        "schedule_hash": prepared.schedule.stable_hash(prepared.template),
        "module_receipt_hash": prepared.module_receipt.stable_hash(
            prepared.template, prepared.schedule
        ),
        "forbidden_workspace_count": (
            prepared.module_receipt.forbidden_workspace_count
        ),
        "module_call_count": {"forward": 1, "backward": 1},
        "fallback_count": 0,
        "eager_backward_count": 0,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "compile_excluded": True,
        "performance_claimed": False,
    }
    value["worker_hash"] = _hash(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ordinal", type=int, required=True)
    parser.add_argument("--order", choices=("AB", "BA"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.run_ordinal not in range(6):
        raise ValueError("R3-3 timing run ordinal differs")
    value = _run(args.run_ordinal, args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_canonical(value) + "\n", encoding="utf-8")
    speedup = cast(float, value["paired_speedup"])
    print(
        f"R3-3 timing run={args.run_ordinal} order={args.order} "
        f"speedup={speedup:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
