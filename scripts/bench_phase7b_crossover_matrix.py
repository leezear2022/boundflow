from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

import torch

from boundflow.runtime.bound_planner import FinalConcretizationRequest, phase7a_capability_table_jsonable
from scripts.bench_phase7a_shared_crown_path_attribution import (
    _TimerMode,
    _collect_row,
    _device_meta,
    _device_name,
    _git_sha,
    _make_device,
    _parse_workloads,
)

_POLICIES = ("structured", "dense_barrier", "auto")
_SCALES = ("smoke", "small", "bench")


def _parse_csv(raw: str, *, allowed: Iterable[str], label: str) -> List[str]:
    allowed_set = set(allowed)
    if raw.strip() == "all":
        return list(allowed)
    out: List[str] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if item not in allowed_set:
            raise ValueError(f"unknown {label}: {item}")
        out.append(item)
    if not out:
        raise ValueError(f"empty --{label}")
    return out


def _counter_value(table: Dict[str, Any], key: str, field: str) -> int:
    return int(table.get(key, {}).get(field, 0))


def _extract_matrix_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    attr = row["counts_structured"]["operator_attribution"]
    materialization = attr["materialization"]
    by_reason = materialization["by_reason"]
    cache = attr.get("cache", {})
    return {
        "structured_ms_p50": float(row["structured_ms_p50"]),
        "baseline_ms_p50": float(row["baseline_ms_p50"]),
        "speedup": float(row["speedup"]),
        "materialized_bytes": int(materialization["total_bytes"]),
        "materialized_numel": int(materialization["total_numel"]),
        "unknown_materialization_calls": _counter_value(by_reason, "unknown_materialization", "calls"),
        "right_matmul_exact_bytes": _counter_value(
            by_reason,
            "right_matmul_exact_sign_split_required",
            "total_bytes",
        ),
        "final_bound_concretization_bytes": _counter_value(
            by_reason,
            "final_bound_concretization",
            "total_bytes",
        ),
        "final_bound_dense_barrier_bytes": _counter_value(
            by_reason,
            "final_bound_dense_barrier",
            "total_bytes",
        ),
        "cache_hits": int(cache.get("hits", 0)),
        "cache_misses": int(cache.get("misses", 0)),
        "split_pos_neg_dense_total": int(row["counts_structured"]["split_pos_neg_dense_total"]),
        "planner_final_concretization_policy": row["planner_decision"]["final_concretization_policy"],
        "planner_reason": row["planner_decision"]["reason"],
    }


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["workload"], row["scale_id"]), []).append(row)

    summaries: List[Dict[str, Any]] = []
    for (workload, scale_id), items in sorted(grouped.items()):
        best_latency = min(items, key=lambda item: float(item["metrics"]["structured_ms_p50"]))
        best_speedup = max(items, key=lambda item: float(item["metrics"]["speedup"]))
        structured = next((item for item in items if item["policy_request"] == "structured"), None)
        auto = next((item for item in items if item["policy_request"] == "auto"), None)
        dense = next((item for item in items if item["policy_request"] == "dense_barrier"), None)
        summary: Dict[str, Any] = {
            "workload": workload,
            "scale_id": scale_id,
            "policies_observed": [item["policy_request"] for item in items],
            "best_policy_by_structured_ms": best_latency["policy_request"],
            "best_policy_by_speedup": best_speedup["policy_request"],
            "auto_final_concretization_policy": (
                auto["metrics"]["planner_final_concretization_policy"] if auto is not None else None
            ),
        }
        if structured is not None and dense is not None:
            base_ms = float(structured["metrics"]["structured_ms_p50"])
            dense_ms = float(dense["metrics"]["structured_ms_p50"])
            summary["dense_barrier_vs_structured_ms_ratio"] = None if base_ms == 0.0 else dense_ms / base_ms
        summaries.append(summary)
    return summaries


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 7B PR-19: shared CROWN policy crossover matrix.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--workloads", type=str, default="all")
    parser.add_argument("--scales", type=str, default="smoke,small")
    parser.add_argument("--policies", type=str, default="structured,dense_barrier,auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timer", type=str, default="perf_counter", choices=["perf_counter", "torch_benchmark"])
    parser.add_argument("--torch-benchmark-min-run-time-s", type=float, default=0.2)
    parser.add_argument("--allow-mps-fallback", action="store_true")
    args = parser.parse_args(argv)

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = _make_device(str(args.device), dtype_name=str(args.dtype), allow_mps_fallback=bool(args.allow_mps_fallback))
    workloads = _parse_workloads(args.workloads)
    scales = _parse_csv(args.scales, allowed=_SCALES, label="scales")
    policies = _parse_csv(args.policies, allowed=_POLICIES, label="policies")
    timer: _TimerMode = args.timer

    rows: List[Dict[str, Any]] = []
    case_idx = 0
    for scale_id in scales:
        for workload in workloads:
            case_seed = int(args.seed) + case_idx
            for policy in policies:
                row = _collect_row(
                    workload,
                    device=device,
                    dtype=dtype,
                    profile=scale_id,
                    seed=case_seed,
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                    timer=timer,
                    torch_benchmark_min_run_time_s=float(args.torch_benchmark_min_run_time_s),
                    final_policy_request=policy,  # type: ignore[arg-type]
                )
                row_dict = asdict(row)
                rows.append(
                    {
                        "workload": str(workload),
                        "scale_id": str(scale_id),
                        "policy_request": str(policy),
                        "compare_target": row_dict["compare_target"],
                        "planner_decision": row_dict["planner_decision"],
                        "metrics": _extract_matrix_metrics(row_dict),
                        "raw_row": row_dict,
                    }
                )
            case_idx += 1

    payload = {
        "meta": {
            "schema_version": "phase7b_crossover_matrix.v1",
            "script": "bench_phase7b_crossover_matrix",
            "git_sha": _git_sha(),
            "torch_version": torch.__version__,
            "device": str(device),
            "device_name": _device_name(device),
            "device_meta": _device_meta(device, allow_mps_fallback=bool(args.allow_mps_fallback)),
            "dtype": str(dtype).replace("torch.", ""),
            "workloads": workloads,
            "scales": scales,
            "policies": policies,
            "timer": timer,
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "seed": int(args.seed),
            "torch_benchmark_min_run_time_s": float(args.torch_benchmark_min_run_time_s),
            "capability_table": phase7a_capability_table_jsonable(),
        },
        "rows": rows,
        "summary": _summarize(rows),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
