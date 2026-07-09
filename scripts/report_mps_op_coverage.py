from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import traceback
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import torch

from boundflow.runtime.bound_planner import FinalConcretizationRequest
from scripts.bench_phase7a_shared_crown_path_attribution import (
    _TimerMode,
    _collect_row,
    _device_meta,
    _device_name,
    _git_sha,
    _make_device,
    _parse_workloads,
)
from scripts.bench_phase7b_crossover_matrix import _SCALES, _parse_csv


def _error_payload(exc: BaseException) -> Dict[str, Any]:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "error_type": type(exc).__name__,
        "error_msg": str(exc),
        "traceback_hash": hashlib.sha256(tb.encode("utf-8")).hexdigest()[:16],
        "traceback_tail": "\n".join(tb.splitlines()[-8:]),
    }


def _run_one(
    *,
    workload: str,
    scale_id: str,
    policy: FinalConcretizationRequest,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    warmup: int,
    iters: int,
    timer: _TimerMode,
    torch_benchmark_min_run_time_s: float,
) -> Dict[str, Any]:
    try:
        row = _collect_row(
            workload,
            device=device,
            dtype=dtype,
            profile=scale_id,
            seed=seed,
            warmup=warmup,
            iters=iters,
            timer=timer,
            torch_benchmark_min_run_time_s=torch_benchmark_min_run_time_s,
            final_policy_request=policy,
        )
        row_dict = asdict(row)
        return {
            "workload": workload,
            "scale_id": scale_id,
            "policy_request": policy,
            "status": "ok",
            "compare_target": row_dict["compare_target"],
            "planner_decision": row_dict["planner_decision"],
            "structured_ms_p50": row_dict["structured_ms_p50"],
            "baseline_ms_p50": row_dict["baseline_ms_p50"],
            "speedup": row_dict["speedup"],
            "error": None,
        }
    except Exception as exc:
        return {
            "workload": workload,
            "scale_id": scale_id,
            "policy_request": policy,
            "status": "fail",
            "error": _error_payload(exc),
        }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Report MPS op coverage for Phase 7B shared CROWN workloads.")
    parser.add_argument("--device", type=str, default="mps", choices=["mps"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32"])
    parser.add_argument("--workloads", type=str, default="all")
    parser.add_argument("--scales", type=str, default="smoke")
    parser.add_argument("--policy", type=str, default="auto", choices=["structured", "dense_barrier", "auto"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timer", type=str, default="perf_counter", choices=["perf_counter", "torch_benchmark"])
    parser.add_argument("--torch-benchmark-min-run-time-s", type=float, default=0.2)
    parser.add_argument("--allow-mps-fallback", action="store_true")
    args = parser.parse_args(argv)

    dtype = torch.float32
    device = _make_device(str(args.device), dtype_name=str(args.dtype), allow_mps_fallback=bool(args.allow_mps_fallback))
    workloads = _parse_workloads(args.workloads)
    scales = _parse_csv(args.scales, allowed=_SCALES, label="scales")
    timer: _TimerMode = args.timer

    rows: List[Dict[str, Any]] = []
    case_idx = 0
    for scale_id in scales:
        for workload in workloads:
            rows.append(
                _run_one(
                    workload=workload,
                    scale_id=scale_id,
                    policy=args.policy,  # type: ignore[arg-type]
                    device=device,
                    dtype=dtype,
                    seed=int(args.seed) + case_idx,
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                    timer=timer,
                    torch_benchmark_min_run_time_s=float(args.torch_benchmark_min_run_time_s),
                )
            )
            case_idx += 1

    payload = {
        "meta": {
            "schema_version": "mps_op_coverage.v1",
            "script": "report_mps_op_coverage",
            "git_sha": _git_sha(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "device_name": _device_name(device),
            "device_meta": _device_meta(device, allow_mps_fallback=bool(args.allow_mps_fallback)),
            "dtype": str(dtype).replace("torch.", ""),
            "workloads": workloads,
            "scales": scales,
            "policy": str(args.policy),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "timer": timer,
            "torch_benchmark_min_run_time_s": float(args.torch_benchmark_min_run_time_s),
            "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        },
        "rows": rows,
        "summary": {
            "ok": sum(1 for row in rows if row["status"] == "ok"),
            "fail": sum(1 for row in rows if row["status"] == "fail"),
        },
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
