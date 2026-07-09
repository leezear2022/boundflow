from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bench_phase7a_shared_crown_path_attribution import (
    _TimerMode,
    _collect_row,
    _device_meta,
    _device_name,
    _git_sha,
    _make_device,
    _parse_workloads,
)
from scripts.bench_phase7b_crossover_matrix import _SCALES, _extract_matrix_metrics, _parse_csv


def _counter_value(table: Dict[str, Any], key: str, field: str) -> int:
    return int(table.get(key, {}).get(field, 0))


def _dispatch_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    attr = row["counts_structured"]["operator_attribution"]
    materialization = attr["materialization"]
    relu_pullback = attr.get("relu_pullback", {})
    fallback = attr.get("fallback", {})
    cache = attr.get("cache", {})
    by_op = materialization.get("by_op", {})
    by_reason = materialization.get("by_reason", {})
    by_phase = materialization.get("by_phase", {})
    return {
        "materialization_total_calls": int(materialization.get("total_calls", 0)),
        "materialization_total_bytes": int(materialization.get("total_bytes", 0)),
        "materialization_by_op": by_op,
        "materialization_by_reason": by_reason,
        "materialization_by_phase": by_phase,
        "unknown_materialization_calls": _counter_value(by_reason, "unknown_materialization", "calls"),
        "right_matmul_exact_sign_split_calls": _counter_value(
            by_reason,
            "right_matmul_exact_sign_split_required",
            "calls",
        ),
        "final_bound_concretization_calls": _counter_value(by_reason, "final_bound_concretization", "calls"),
        "final_bound_dense_barrier_calls": _counter_value(by_reason, "final_bound_dense_barrier", "calls"),
        "relu_pullback_by_op": relu_pullback.get("by_op", {}),
        "fallback_by_reason": fallback.get("by_reason", {}),
        "cache": cache,
    }


def _torch_profiler_top_ops(fn, *, limit: int) -> List[Dict[str, Any]]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        fn()
    rows: List[Dict[str, Any]] = []
    for event in prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=limit).splitlines():
        if event.strip():
            rows.append({"raw": event})
    return rows


@contextlib.contextmanager
def _maybe_mps_signpost(enabled: bool):
    if not bool(enabled) or not hasattr(torch, "mps") or not hasattr(torch.mps, "profiler"):
        yield
        return
    with torch.mps.profiler.profile(mode="interval", wait_until_completed=False):
        yield


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MPS dispatch/profile report for Phase 7B workloads.")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32"])
    parser.add_argument("--workloads", type=str, default="all")
    parser.add_argument("--scales", type=str, default="smoke")
    parser.add_argument("--policy", type=str, default="auto", choices=["structured", "dense_barrier", "auto"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timer", type=str, default="perf_counter", choices=["perf_counter", "torch_benchmark"])
    parser.add_argument("--torch-benchmark-min-run-time-s", type=float, default=0.2)
    parser.add_argument("--with-mps-signposts", action="store_true")
    parser.add_argument("--with-torch-profiler", action="store_true")
    parser.add_argument("--torch-profiler-top", type=int, default=10)
    parser.add_argument("--allow-mps-fallback", action="store_true")
    args = parser.parse_args(argv)

    dtype = torch.float32
    device = _make_device(str(args.device), dtype_name=str(args.dtype), allow_mps_fallback=bool(args.allow_mps_fallback))
    workloads = _parse_workloads(str(args.workloads))
    scales = _parse_csv(str(args.scales), allowed=_SCALES, label="scales")
    timer: _TimerMode = args.timer

    rows: List[Dict[str, Any]] = []
    case_idx = 0
    for scale in scales:
        for workload in workloads:
            seed = int(args.seed) + case_idx

            def run_row() -> Dict[str, Any]:
                row = _collect_row(
                    workload,
                    device=device,
                    dtype=dtype,
                    profile=scale,
                    seed=seed,
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                    timer=timer,
                    torch_benchmark_min_run_time_s=float(args.torch_benchmark_min_run_time_s),
                    final_policy_request=args.policy,  # type: ignore[arg-type]
                )
                return asdict(row)

            profiler_top_ops: List[Dict[str, Any]] = []
            if bool(args.with_torch_profiler):
                captured: Dict[str, Any] = {}

                def profiled_run() -> None:
                    captured["row"] = run_row()

                profiler_top_ops = _torch_profiler_top_ops(profiled_run, limit=int(args.torch_profiler_top))
                row_dict = captured["row"]
            else:
                with _maybe_mps_signpost(bool(args.with_mps_signposts) and device.type == "mps"):
                    row_dict = run_row()

            rows.append(
                {
                    "workload": workload,
                    "scale_id": scale,
                    "policy_request": str(args.policy),
                    "compare_target": row_dict["compare_target"],
                    "planner_decision": row_dict["planner_decision"],
                    "metrics": _extract_matrix_metrics(row_dict),
                    "dispatch": _dispatch_metrics(row_dict),
                    "torch_profiler_top_ops": profiler_top_ops,
                }
            )
            case_idx += 1

    payload = {
        "meta": {
            "schema_version": "mps_dispatch_profile.v1",
            "script": "report_mps_dispatch_profile",
            "git_sha": _git_sha(),
            "torch_version": torch.__version__,
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
            "with_mps_signposts": bool(args.with_mps_signposts),
            "with_torch_profiler": bool(args.with_torch_profiler),
            "pytorch_mps_prefer_metal": os.environ.get("PYTORCH_MPS_PREFER_METAL"),
            "pytorch_mps_fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH"),
        },
        "rows": rows,
        "summary": {
            "rows": len(rows),
            "unknown_materialization_total": sum(
                int(row["dispatch"]["unknown_materialization_calls"]) for row in rows
            ),
            "materialization_bytes_total": sum(int(row["dispatch"]["materialization_total_bytes"]) for row in rows),
            "cache_hits_total": sum(int(row["dispatch"]["cache"].get("hits", 0)) for row in rows),
            "cache_misses_total": sum(int(row["dispatch"]["cache"].get("misses", 0)) for row in rows),
        },
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
