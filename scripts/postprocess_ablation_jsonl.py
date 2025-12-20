#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _read_jsonl(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        text = p.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"invalid json at {p}:{i}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"jsonl row is not an object at {p}:{i}")
            obj["_source_file"] = str(p)
            obj["_source_line"] = int(i)
            rows.append(obj)
    return rows


def _get(d: Any, *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


_ENUM_RE = re.compile(r"MemoryPlanMode\.(?P<name>[A-Z_]+)")
_ENUM_VALUE_RE = re.compile(r":\\s*'(?P<value>[^']+)'")


def _normalize_enum_repr(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        # Example: "<MemoryPlanMode.DEFAULT: 'default'>"
        m = _ENUM_VALUE_RE.search(x)
        if m:
            return m.group("value")
        m2 = _ENUM_RE.search(x)
        if m2:
            return m2.group("name").lower()
        return x
    return str(x)


@dataclass(frozen=True)
class FlatRow:
    data: Dict[str, Any]


def _flatten_row(row: Dict[str, Any]) -> FlatRow:
    planner_cfg = _get(row, "config", "planner_config_dump", default={}) or {}
    tvm_opts = _get(row, "config", "tvm_options", default={}) or {}

    partition_policy = _get(planner_cfg, "partition", "policy")
    reuse_on = _get(planner_cfg, "storage_reuse", "enabled", default=_get(planner_cfg, "enable_storage_reuse"))
    memory_plan_mode = _normalize_enum_repr(_get(tvm_opts, "memory_plan_mode"))
    fusion_on = _get(tvm_opts, "enable_task_fusion_pipeline")

    baseline_auto = _get(row, "baseline", "auto_lirpa")
    baseline_available = bool(isinstance(baseline_auto, dict) and baseline_auto and baseline_auto.get("available", True))

    out: Dict[str, Any] = {
        "schema_version": row.get("schema_version"),
        "git_commit": _get(row, "meta", "git_commit"),
        "time_utc": _get(row, "meta", "time_utc"),
        "source_file": row.get("_source_file"),
        "source_line": row.get("_source_line"),
        "workload": _get(row, "workload", "model"),
        "input_shape": json.dumps(_get(row, "workload", "input_shape"), ensure_ascii=False),
        "eps": _get(row, "workload", "eps"),
        "partition_policy": partition_policy,
        "reuse_on": bool(reuse_on) if reuse_on is not None else None,
        "memory_plan_mode": memory_plan_mode,
        "fusion_on": bool(fusion_on) if fusion_on is not None else None,
        "plan_ms_total": _get(row, "planner", "plan_ms_total"),
        "num_tasks": _get(row, "planner", "num_tasks"),
        "num_edges": _get(row, "planner", "num_edges"),
        "logical_buffers": _get(row, "planner", "storage", "logical_buffers"),
        "physical_buffers": _get(row, "planner", "storage", "physical_buffers"),
        "physical_bytes_est": _get(row, "planner", "storage", "physical_bytes_est"),
        "compile_first_run_ms": _get(row, "runtime", "compile_first_run_ms"),
        "run_ms_avg": _get(row, "runtime", "run_ms_avg"),
        "run_ms_p50": _get(row, "runtime", "run_ms_p50"),
        "run_ms_p95": _get(row, "runtime", "run_ms_p95"),
        "tvm_compile_ms_total": _get(row, "tvm", "compile_stats_agg", "compile_ms_total"),
        "call_tir_after_legalize_sum": _get(row, "tvm", "compile_stats_agg", "call_tir_after_legalize_sum"),
        "call_tir_after_fuse_tir_sum": _get(row, "tvm", "compile_stats_agg", "call_tir_after_fuse_tir_sum"),
        "tvm_memory_by_scan_alloc_storage_total_bytes_sum": _get(
            row, "tvm", "compile_stats_agg", "memory_by_scan_alloc_storage_total_bytes_sum"
        ),
        "tvm_memory_by_scan_alloc_storage_max_bytes_max": _get(
            row, "tvm", "compile_stats_agg", "memory_by_scan_alloc_storage_max_bytes_max"
        ),
        "tvm_estimator_stage": _get(row, "tvm", "compile_stats_agg", "memory_by_tvm_estimator_stage"),
        "compile_cache_miss_delta_first_run": _get(
            row, "tvm", "compile_cache_stats_delta_compile_first_run", "task_compile_cache_miss"
        ),
        "compile_cache_hit_delta_first_run": _get(
            row, "tvm", "compile_cache_stats_delta_compile_first_run", "task_compile_cache_hit"
        ),
        "compile_cache_fail_delta_first_run": _get(
            row, "tvm", "compile_cache_stats_delta_compile_first_run", "task_compile_fail"
        ),
        "python_vs_tvm_max_abs_diff_lb": _get(row, "correctness", "python_vs_tvm_max_abs_diff_lb"),
        "python_vs_tvm_max_abs_diff_ub": _get(row, "correctness", "python_vs_tvm_max_abs_diff_ub"),
        "python_vs_tvm_max_rel_diff_lb": _get(row, "correctness", "python_vs_tvm_max_rel_diff_lb"),
        "python_vs_tvm_max_rel_diff_ub": _get(row, "correctness", "python_vs_tvm_max_rel_diff_ub"),
        "auto_lirpa_available": baseline_available,
        "auto_lirpa_setup_ms": _get(row, "baseline", "auto_lirpa", "setup_ms"),
        "auto_lirpa_ms_avg": _get(row, "baseline", "auto_lirpa", "compute_bounds_ms_avg"),
        "auto_lirpa_ms_p50": _get(row, "baseline", "auto_lirpa", "compute_bounds_ms_p50"),
        "auto_lirpa_ms_p95": _get(row, "baseline", "auto_lirpa", "compute_bounds_ms_p95"),
        "python_vs_auto_lirpa_max_rel_diff_lb": _get(row, "correctness", "python_vs_auto_lirpa_max_rel_diff_lb"),
        "python_vs_auto_lirpa_max_rel_diff_ub": _get(row, "correctness", "python_vs_auto_lirpa_max_rel_diff_ub"),
    }
    return FlatRow(data=out)


def _write_csv(path: Path, rows: List[Dict[str, Any]], *, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _group_key(fr: FlatRow) -> Tuple[Any, ...]:
    d = fr.data
    return (
        d.get("workload"),
        d.get("partition_policy"),
        d.get("reuse_on"),
        d.get("memory_plan_mode"),
        d.get("fusion_on"),
    )


def _mean(xs: List[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def _summarize(rows: List[FlatRow]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[FlatRow]] = {}
    for r in rows:
        groups.setdefault(_group_key(r), []).append(r)

    out: List[Dict[str, Any]] = []
    for k, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        workload, partition_policy, reuse_on, memory_plan_mode, fusion_on = k
        run_p50 = [float(x.data["run_ms_p50"]) for x in rs if x.data.get("run_ms_p50") is not None]
        compile_first = [float(x.data["compile_first_run_ms"]) for x in rs if x.data.get("compile_first_run_ms") is not None]
        miss_delta = [
            float(x.data["compile_cache_miss_delta_first_run"])
            for x in rs
            if x.data.get("compile_cache_miss_delta_first_run") is not None
        ]
        rel_diff = [
            max(float(x.data.get("python_vs_tvm_max_rel_diff_lb") or 0.0), float(x.data.get("python_vs_tvm_max_rel_diff_ub") or 0.0))
            for x in rs
        ]
        out.append(
            {
                "workload": workload,
                "partition_policy": partition_policy,
                "reuse_on": reuse_on,
                "memory_plan_mode": memory_plan_mode,
                "fusion_on": fusion_on,
                "n": int(len(rs)),
                "run_ms_p50_mean": _mean(run_p50),
                "compile_first_run_ms_mean": _mean(compile_first),
                "compile_cache_miss_delta_first_run_mean": _mean(miss_delta),
                "python_vs_tvm_max_rel_diff_max": float(max(rel_diff) if rel_diff else 0.0),
            }
        )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", nargs="+", help="Input JSONL files (bench outputs).")
    p.add_argument("--out-dir", type=str, default="out/phase5d")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    paths = [Path(x) for x in args.jsonl]
    rows = _read_jsonl(paths)
    flat = [_flatten_row(r) for r in rows]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({k for fr in flat for k in fr.data.keys()})
    _write_csv(out_dir / "ablation.csv", [fr.data for fr in flat], fieldnames=fieldnames)
    summary = _summarize(flat)
    if summary:
        _write_csv(out_dir / "tables" / "ablation_summary.csv", summary, fieldnames=list(summary[0].keys()))

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            xs = [float(fr.data.get("compile_cache_miss_delta_first_run") or 0.0) for fr in flat]
            ys = [float(fr.data.get("compile_first_run_ms") or 0.0) for fr in flat]
            plt.figure(figsize=(5, 3))
            plt.scatter(xs, ys, s=12)
            plt.xlabel("compile_cache_miss_delta_first_run")
            plt.ylabel("compile_first_run_ms")
            fig_dir = out_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(fig_dir / "cache_miss_vs_compile_first_run.png", dpi=200)
            plt.close()
        except Exception:
            # Optional dependency; keep the pipeline usable in minimal environments.
            pass

    # Write a small manifest for human inspection.
    (out_dir / "MANIFEST.txt").write_text(
        "\\n".join(
            [
                f"inputs: {', '.join(str(p) for p in paths)}",
                f"rows: {len(rows)}",
                f"csv: {out_dir / 'ablation.csv'}",
                f"table: {out_dir / 'tables' / 'ablation_summary.csv'}" if summary else "table: (none)",
            ]
        )
        + "\\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

