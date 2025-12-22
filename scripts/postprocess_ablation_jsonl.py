#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _read_jsonl(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
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
_ENUM_VALUE_RE = re.compile(r":\s*'(?P<value>[^']+)'")


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
    baseline_key = _get(row, "baseline", "auto_lirpa", "baseline_key")

    status = row.get("status", "ok")
    err_type = _get(row, "error", "error_type")
    err_msg = _get(row, "error", "error_msg")

    out: Dict[str, Any] = {
        "schema_version": row.get("schema_version"),
        "status": status,
        "error_type": err_type,
        "error_msg": err_msg,
        "git_commit": _get(row, "meta", "git_commit"),
        "time_utc": _get(row, "meta", "time_utc"),
        "source_file": row.get("_source_file"),
        "source_line": row.get("_source_line"),
        "workload": _get(row, "workload", "model"),
        "input_shape": json.dumps(_get(row, "workload", "input_shape"), ensure_ascii=False),
        "eps": _get(row, "workload", "eps"),
        "domain": _get(row, "workload", "domain"),
        "spec": _get(row, "workload", "spec"),
        "baseline_key": baseline_key,
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
        "run_ms_cold": _get(row, "runtime", "run_ms_cold"),
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
        "compile_cache_tag": _get(row, "tvm", "compile_cache_tag"),
        "compile_keyset_size": _get(row, "tvm", "compile_keyset_size"),
        "compile_keyset_digest": _get(row, "tvm", "compile_keyset_digest"),
        "python_vs_tvm_max_abs_diff_lb": _get(row, "correctness", "python_vs_tvm_max_abs_diff_lb"),
        "python_vs_tvm_max_abs_diff_ub": _get(row, "correctness", "python_vs_tvm_max_abs_diff_ub"),
        "python_vs_tvm_max_rel_diff_lb": _get(row, "correctness", "python_vs_tvm_max_rel_diff_lb"),
        "python_vs_tvm_max_rel_diff_ub": _get(row, "correctness", "python_vs_tvm_max_rel_diff_ub"),
        "python_vs_tvm_ok": _get(row, "correctness", "python_vs_tvm_gate", "ok"),
        "auto_lirpa_available": baseline_available,
        "auto_lirpa_reason": _get(row, "baseline", "auto_lirpa", "reason"),
        "auto_lirpa_version": _get(row, "baseline", "auto_lirpa", "version"),
        "auto_lirpa_method": _get(row, "baseline", "auto_lirpa", "method"),
        "auto_lirpa_cache_hit": _get(row, "baseline", "auto_lirpa", "cache_hit"),
        "auto_lirpa_baseline_key": baseline_key,
        "auto_lirpa_spec_hash": _get(row, "baseline", "auto_lirpa", "spec_hash"),
        "auto_lirpa_init_ms": _get(
            row,
            "baseline",
            "auto_lirpa",
            "init_ms",
            default=_get(row, "baseline", "auto_lirpa", "setup_ms"),
        ),
        "auto_lirpa_run_ms_cold": _get(row, "baseline", "auto_lirpa", "run_ms_cold"),
        "auto_lirpa_run_ms_avg": _get(
            row,
            "baseline",
            "auto_lirpa",
            "run_ms_avg",
            default=_get(row, "baseline", "auto_lirpa", "compute_bounds_ms_avg"),
        ),
        "auto_lirpa_run_ms_p50": _get(
            row,
            "baseline",
            "auto_lirpa",
            "run_ms_p50",
            default=_get(row, "baseline", "auto_lirpa", "compute_bounds_ms_p50"),
        ),
        "auto_lirpa_run_ms_p95": _get(
            row,
            "baseline",
            "auto_lirpa",
            "run_ms_p95",
            default=_get(row, "baseline", "auto_lirpa", "compute_bounds_ms_p95"),
        ),
        "python_vs_auto_lirpa_max_rel_diff_lb": _get(row, "correctness", "python_vs_auto_lirpa_max_rel_diff_lb"),
        "python_vs_auto_lirpa_max_rel_diff_ub": _get(row, "correctness", "python_vs_auto_lirpa_max_rel_diff_ub"),
        "python_vs_auto_lirpa_ok": _get(row, "correctness", "python_vs_auto_lirpa_gate", "ok"),
        "python_vs_auto_lirpa_max_abs_diff_lb": _get(row, "correctness", "python_vs_auto_lirpa_max_abs_diff_lb"),
        "python_vs_auto_lirpa_max_abs_diff_ub": _get(row, "correctness", "python_vs_auto_lirpa_max_abs_diff_ub"),
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
        d.get("input_shape"),
        d.get("eps"),
        d.get("domain"),
        d.get("spec"),
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
        workload, input_shape, eps, domain, spec, partition_policy, reuse_on, memory_plan_mode, fusion_on = k
        ok_rs = [x for x in rs if str(x.data.get("status", "ok")) == "ok"]
        n_ok = int(len(ok_rs))
        n_fail = int(len(rs) - len(ok_rs))

        run_p50 = [float(x.data["run_ms_p50"]) for x in ok_rs if x.data.get("run_ms_p50") is not None]
        compile_first = [float(x.data["compile_first_run_ms"]) for x in ok_rs if x.data.get("compile_first_run_ms") is not None]
        miss_delta = [
            float(x.data["compile_cache_miss_delta_first_run"])
            for x in ok_rs
            if x.data.get("compile_cache_miss_delta_first_run") is not None
        ]

        def _max_rel_diff(fr: FlatRow) -> Optional[float]:
            lb = fr.data.get("python_vs_tvm_max_rel_diff_lb")
            ub = fr.data.get("python_vs_tvm_max_rel_diff_ub")
            if lb is None and ub is None:
                return None
            return max(float(lb or 0.0), float(ub or 0.0))

        rel_vals = [_max_rel_diff(x) for x in ok_rs]
        rel_present = [v for v in rel_vals if v is not None]
        rel_missing = int(sum(v is None for v in rel_vals))

        out.append(
            {
                "workload": workload,
                "input_shape": input_shape,
                "eps": eps,
                "domain": domain,
                "spec": spec,
                "partition_policy": partition_policy,
                "reuse_on": reuse_on,
                "memory_plan_mode": memory_plan_mode,
                "fusion_on": fusion_on,
                "n": int(len(rs)),
                "n_ok": n_ok,
                "n_fail": n_fail,
                "run_ms_p50_mean": _mean(run_p50),
                "compile_first_run_ms_mean": _mean(compile_first),
                "compile_cache_miss_delta_first_run_mean": _mean(miss_delta),
                "python_vs_tvm_max_rel_diff_max": float(max(rel_present)) if rel_present else None,
                "python_vs_tvm_rel_diff_missing": rel_missing,
            }
        )
    return out


def _summarize_main(flat: List[FlatRow]) -> List[Dict[str, Any]]:
    """
    Paper-facing main table (minimal): keep core group keys and main metrics.
    """
    # Baseline summary keyed by baseline_key (deduped across matrix points).
    baseline_map: Dict[str, Dict[str, Any]] = {}
    for fr in flat:
        d = fr.data
        if str(d.get("status", "ok")) != "ok":
            continue
        if not bool(d.get("auto_lirpa_available")):
            continue
        bk = d.get("auto_lirpa_baseline_key") or d.get("baseline_key")
        if not isinstance(bk, str) or not bk:
            continue
        # Prefer the cache-miss row as the baseline source of truth.
        prefer = not bool(d.get("auto_lirpa_cache_hit"))
        if bk in baseline_map and (not prefer):
            continue
        baseline_map[bk] = {
            "auto_lirpa_available": bool(d.get("auto_lirpa_available")),
            "auto_lirpa_reason": d.get("auto_lirpa_reason"),
            "auto_lirpa_version": d.get("auto_lirpa_version"),
            "auto_lirpa_method": d.get("auto_lirpa_method"),
            "auto_lirpa_spec_hash": d.get("auto_lirpa_spec_hash"),
            "auto_lirpa_init_ms": d.get("auto_lirpa_init_ms"),
            "auto_lirpa_run_ms_cold": d.get("auto_lirpa_run_ms_cold"),
            "auto_lirpa_run_ms_p50": d.get("auto_lirpa_run_ms_p50"),
            "auto_lirpa_run_ms_p95": d.get("auto_lirpa_run_ms_p95"),
            "auto_lirpa_cache_hit": bool(d.get("auto_lirpa_cache_hit")),
        }

    groups: Dict[Tuple[Any, ...], List[FlatRow]] = {}
    for r in flat:
        groups.setdefault(_group_key(r), []).append(r)

    out: List[Dict[str, Any]] = []
    for k, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        workload, input_shape, eps, domain, spec, partition_policy, reuse_on, memory_plan_mode, fusion_on = k
        ok_rs = [x for x in rs if str(x.data.get("status", "ok")) == "ok"]
        n_ok = int(len(ok_rs))
        n_fail = int(len(rs) - len(ok_rs))

        run_p50 = [float(x.data["run_ms_p50"]) for x in ok_rs if x.data.get("run_ms_p50") is not None]
        compile_first = [float(x.data["compile_first_run_ms"]) for x in ok_rs if x.data.get("compile_first_run_ms") is not None]
        plan_ms_total = [float(x.data["plan_ms_total"]) for x in ok_rs if x.data.get("plan_ms_total") is not None]
        phys_bytes = [float(x.data["physical_bytes_est"]) for x in ok_rs if x.data.get("physical_bytes_est") is not None]
        call_tir_legalize = [float(x.data["call_tir_after_legalize_sum"]) for x in ok_rs if x.data.get("call_tir_after_legalize_sum") is not None]
        call_tir_fuse_tir = [float(x.data["call_tir_after_fuse_tir_sum"]) for x in ok_rs if x.data.get("call_tir_after_fuse_tir_sum") is not None]
        miss_delta = [
            float(x.data["compile_cache_miss_delta_first_run"])
            for x in ok_rs
            if x.data.get("compile_cache_miss_delta_first_run") is not None
        ]
        hit_delta = [
            float(x.data["compile_cache_hit_delta_first_run"])
            for x in ok_rs
            if x.data.get("compile_cache_hit_delta_first_run") is not None
        ]
        keyset_size = [float(x.data["compile_keyset_size"]) for x in ok_rs if x.data.get("compile_keyset_size") is not None]

        # Baseline: de-duplicate by baseline_key to avoid matrix-size dependent weighting.
        baseline_keys = sorted(
            {str(x.data.get("auto_lirpa_baseline_key") or "") for x in ok_rs if x.data.get("auto_lirpa_baseline_key")}
        )
        baseline_key = baseline_keys[0] if len(baseline_keys) == 1 else None
        baseline = baseline_map.get(str(baseline_key)) if baseline_key else None

        # Correctness gate against auto_LiRPA: summarize ok-rate + max abs diff.
        gate_vals = [x.data.get("python_vs_auto_lirpa_ok") for x in ok_rs if x.data.get("python_vs_auto_lirpa_ok") is not None]
        gate_ok = [bool(v) for v in gate_vals]
        gate_ok_rate = (sum(1 for v in gate_ok if v) / len(gate_ok)) if gate_ok else None
        abs_max_vals: List[float] = []
        for x in ok_rs:
            lb = x.data.get("python_vs_auto_lirpa_max_abs_diff_lb")
            ub = x.data.get("python_vs_auto_lirpa_max_abs_diff_ub")
            if lb is not None:
                abs_max_vals.append(float(lb))
            if ub is not None:
                abs_max_vals.append(float(ub))
        gate_max_abs = float(max(abs_max_vals)) if abs_max_vals else None

        def _max_rel_diff(fr: FlatRow) -> Optional[float]:
            lb = fr.data.get("python_vs_tvm_max_rel_diff_lb")
            ub = fr.data.get("python_vs_tvm_max_rel_diff_ub")
            if lb is None and ub is None:
                return None
            return max(float(lb or 0.0), float(ub or 0.0))

        rel_vals = [_max_rel_diff(x) for x in ok_rs]
        rel_present = [v for v in rel_vals if v is not None]
        rel_missing = int(sum(v is None for v in rel_vals))

        # Speedup against baseline (hot): auto_lirpa_run_ms_p50 / boundflow_run_ms_p50_mean.
        baseline_hot = None
        if isinstance(baseline, dict):
            try:
                baseline_hot = float(baseline.get("auto_lirpa_run_ms_p50")) if baseline.get("auto_lirpa_run_ms_p50") is not None else None
            except Exception:
                baseline_hot = None
        hot_mean = _mean(run_p50)
        speedup_hot = (baseline_hot / hot_mean) if (baseline_hot is not None and hot_mean > 0) else None

        out.append(
            {
                "workload": workload,
                "input_shape": input_shape,
                "eps": eps,
                "domain": domain,
                "spec": spec,
                "auto_lirpa_baseline_key": baseline_key,
                "partition_policy": partition_policy,
                "reuse_on": reuse_on,
                "memory_plan_mode": memory_plan_mode,
                "fusion_on": fusion_on,
                "n": int(len(rs)),
                "n_ok": n_ok,
                "n_fail": n_fail,
                "plan_ms_total_mean": _mean(plan_ms_total),
                "compile_first_run_ms_mean": _mean(compile_first),
                "run_ms_p50_mean": _mean(run_p50),
                "physical_bytes_est_mean": _mean(phys_bytes),
                "call_tir_after_legalize_sum_mean": _mean(call_tir_legalize),
                "call_tir_after_fuse_tir_sum_mean": _mean(call_tir_fuse_tir),
                "compile_cache_miss_delta_first_run_mean": _mean(miss_delta),
                "compile_cache_hit_delta_first_run_mean": _mean(hit_delta),
                "compile_keyset_size_mean": _mean(keyset_size),
                "python_vs_tvm_max_rel_diff_max": float(max(rel_present)) if rel_present else None,
                "python_vs_tvm_rel_diff_missing": rel_missing,
                "auto_lirpa_available": bool(baseline.get("auto_lirpa_available")) if isinstance(baseline, dict) else None,
                "auto_lirpa_reason": baseline.get("auto_lirpa_reason") if isinstance(baseline, dict) else None,
                "auto_lirpa_method": baseline.get("auto_lirpa_method") if isinstance(baseline, dict) else None,
                "auto_lirpa_version": baseline.get("auto_lirpa_version") if isinstance(baseline, dict) else None,
                "auto_lirpa_spec_hash": baseline.get("auto_lirpa_spec_hash") if isinstance(baseline, dict) else None,
                "auto_lirpa_init_ms": baseline.get("auto_lirpa_init_ms") if isinstance(baseline, dict) else None,
                "auto_lirpa_run_ms_cold": baseline.get("auto_lirpa_run_ms_cold") if isinstance(baseline, dict) else None,
                "auto_lirpa_run_ms_p50": baseline.get("auto_lirpa_run_ms_p50") if isinstance(baseline, dict) else None,
                "auto_lirpa_run_ms_p95": baseline.get("auto_lirpa_run_ms_p95") if isinstance(baseline, dict) else None,
                "speedup_hot_vs_auto_lirpa": speedup_hot,
                "python_vs_auto_lirpa_ok_rate": gate_ok_rate,
                "python_vs_auto_lirpa_max_abs_diff_max": gate_max_abs,
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

    # Paper-facing tables with stable filenames.
    main_table = _summarize_main(flat)
    if main_table:
        _write_csv(out_dir / "tables" / "table_main.csv", main_table, fieldnames=list(main_table[0].keys()))

    if not args.no_plots:
        try:
            # Prefer a headless backend for artifact runs / servers without GUI.
            os.environ.setdefault("MPLBACKEND", "Agg")
            import matplotlib.pyplot as plt  # type: ignore

            flat_ok = [fr for fr in flat if str(fr.data.get("status", "ok")) == "ok"]

            xs = [float(fr.data.get("compile_cache_miss_delta_first_run") or 0.0) for fr in flat_ok]
            ys = [float(fr.data.get("compile_first_run_ms") or 0.0) for fr in flat_ok]
            plt.figure(figsize=(5, 3))
            plt.scatter(xs, ys, s=12)
            plt.xlabel("compile_cache_miss_delta_first_run")
            plt.ylabel("compile_first_run_ms")
            fig_dir = out_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(fig_dir / "cache_miss_vs_compile_first_run.png", dpi=200)
            plt.close()

            # Figure: call_tir counts vs fusion knob (best-effort; zeros if missing).
            xs2 = [str(fr.data.get("fusion_on")) for fr in flat_ok]
            y_legalize = [float(fr.data.get("call_tir_after_legalize_sum") or 0.0) for fr in flat_ok]
            y_fuse_tir = [float(fr.data.get("call_tir_after_fuse_tir_sum") or 0.0) for fr in flat_ok]
            # Aggregate by fusion_on for readability.
            buckets: Dict[str, Dict[str, List[float]]] = {}
            for x, a, b in zip(xs2, y_legalize, y_fuse_tir):
                buckets.setdefault(x, {"legalize": [], "fuse_tir": []})
                buckets[x]["legalize"].append(a)
                buckets[x]["fuse_tir"].append(b)
            labels = sorted(buckets.keys(), key=str)
            legalize_mean = [statistics.mean(buckets[k]["legalize"]) if buckets[k]["legalize"] else 0.0 for k in labels]
            fuse_tir_mean = [statistics.mean(buckets[k]["fuse_tir"]) if buckets[k]["fuse_tir"] else 0.0 for k in labels]

            plt.figure(figsize=(5, 3))
            x_idx = list(range(len(labels)))
            w = 0.35
            plt.bar([i - w / 2 for i in x_idx], legalize_mean, width=w, label="after_legalize")
            plt.bar([i + w / 2 for i in x_idx], fuse_tir_mean, width=w, label="after_fuse_tir")
            plt.xticks(x_idx, labels, rotation=0)
            plt.ylabel("call_tir (mean)")
            plt.xlabel("fusion_on")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(fig_dir / "call_tir_vs_fusion.png", dpi=200)
            plt.close()

            # Figure: memory(bytes) vs runtime(p50) scatter (planner estimate).
            mx = [float(fr.data.get("physical_bytes_est") or 0.0) for fr in flat_ok]
            my = [float(fr.data.get("run_ms_p50") or 0.0) for fr in flat_ok]
            plt.figure(figsize=(5, 3))
            plt.scatter(mx, my, s=12)
            plt.xlabel("physical_bytes_est (planner)")
            plt.ylabel("run_ms_p50 (steady-state)")
            plt.tight_layout()
            plt.savefig(fig_dir / "mem_bytes_vs_runtime.png", dpi=200)
            plt.close()

            # Figure: memory(bytes) vs runtime(p50) colored by workload (AE-friendly when multiple workloads are merged).
            workloads = [str(fr.data.get("workload") or "") for fr in flat_ok]
            uniq_w = [w for w in sorted(set(workloads), key=str) if w]
            if uniq_w:
                colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
                plt.figure(figsize=(5, 3))
                for i, wname in enumerate(uniq_w):
                    idx = [j for j, ww in enumerate(workloads) if ww == wname]
                    if not idx:
                        continue
                    xx = [mx[j] for j in idx]
                    yy = [my[j] for j in idx]
                    plt.scatter(xx, yy, s=14, color=colors[i % len(colors)], label=wname)
                plt.xlabel("physical_bytes_est (planner)")
                plt.ylabel("run_ms_p50 (steady-state)")
                plt.legend(fontsize=7, loc="best")
                plt.tight_layout()
                plt.savefig(fig_dir / "mem_bytes_vs_runtime_by_workload.png", dpi=200)
                plt.close()

            # Figure: cold vs hot runtime relationship.
            cx = [float(fr.data.get("compile_first_run_ms") or 0.0) for fr in flat_ok]
            cy = [float(fr.data.get("run_ms_p50") or 0.0) for fr in flat_ok]
            plt.figure(figsize=(5, 3))
            plt.scatter(cx, cy, s=12)
            plt.xlabel("compile_first_run_ms (cold)")
            plt.ylabel("run_ms_p50 (hot)")
            plt.tight_layout()
            plt.savefig(fig_dir / "runtime_cold_vs_hot.png", dpi=200)
            plt.close()

            # Figure: "memory quadrants" view (reuse_on x memory_plan_mode categories).
            # x: physical_bytes_est (planner), y: run_ms_p50 (steady-state).
            cats = []
            for fr in flat_ok:
                reuse = str(fr.data.get("reuse_on"))
                mpm = str(fr.data.get("memory_plan_mode"))
                cats.append(f"reuse={reuse},mpm={mpm}")
            xq = [float(fr.data.get("physical_bytes_est") or 0.0) for fr in flat_ok]
            yq = [float(fr.data.get("run_ms_p50") or 0.0) for fr in flat_ok]
            uniq = sorted(set(cats), key=str)
            markers = ["o", "s", "^", "D", "v", "P", "X"]
            plt.figure(figsize=(5, 3))
            for i, c in enumerate(uniq):
                idx = [j for j, cc in enumerate(cats) if cc == c]
                if not idx:
                    continue
                xx = [xq[j] for j in idx]
                yy = [yq[j] for j in idx]
                plt.scatter(xx, yy, s=14, marker=markers[i % len(markers)], label=c)
            plt.xlabel("physical_bytes_est (planner)")
            plt.ylabel("run_ms_p50 (steady-state)")
            plt.legend(fontsize=6, loc="best")
            plt.tight_layout()
            plt.savefig(fig_dir / "mem_quadrants.png", dpi=200)
            plt.close()

            # Figure: runtime breakdown (mean per config group; bars for plan/cold/hot).
            if main_table:
                labels = []
                plan_v = []
                cold_v = []
                hot_v = []
                for r in main_table:
                    labels.append(
                        f"{r.get('partition_policy')}\nreuse={r.get('reuse_on')}\nmpm={r.get('memory_plan_mode')}\nfuse={r.get('fusion_on')}"
                    )
                    plan_v.append(float(r.get("plan_ms_total_mean") or 0.0))
                    cold_v.append(float(r.get("compile_first_run_ms_mean") or 0.0))
                    hot_v.append(float(r.get("run_ms_p50_mean") or 0.0))
                x_idx = list(range(len(labels)))
                plt.figure(figsize=(max(6, 0.7 * len(labels)), 3))
                plt.bar(x_idx, plan_v, label="planner.plan_ms_total", alpha=0.8)
                plt.bar(x_idx, cold_v, bottom=plan_v, label="runtime.compile_first_run_ms", alpha=0.8)
                plt.bar(
                    x_idx,
                    hot_v,
                    bottom=[a + b for a, b in zip(plan_v, cold_v)],
                    label="runtime.run_ms_p50",
                    alpha=0.8,
                )
                plt.xticks(x_idx, labels, rotation=0, fontsize=7)
                plt.ylabel("ms (stacked)")
                plt.tight_layout()
                plt.legend(fontsize=7)
                plt.savefig(fig_dir / "runtime_breakdown.png", dpi=200)
                plt.close()

            # Figure: speedup (hot) vs auto_LiRPA baseline (if available).
            if main_table:
                pts = [r for r in main_table if r.get("speedup_hot_vs_auto_lirpa") is not None]
                if pts:
                    workloads = [str(r.get("workload") or "") for r in pts]
                    uniq_w = [w for w in sorted(set(workloads), key=str) if w]
                    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
                    plt.figure(figsize=(5, 3))
                    for i, wname in enumerate(uniq_w):
                        idx = [j for j, ww in enumerate(workloads) if ww == wname]
                        if not idx:
                            continue
                        xx = [float(pts[j].get("physical_bytes_est_mean") or 0.0) for j in idx]
                        yy = [float(pts[j].get("speedup_hot_vs_auto_lirpa") or 0.0) for j in idx]
                        plt.scatter(xx, yy, s=18, color=colors[i % len(colors)], label=wname)
                    plt.axhline(1.0, color="gray", linewidth=1, linestyle="--")
                    plt.xlabel("physical_bytes_est_mean (planner)")
                    plt.ylabel("speedup_hot_vs_auto_lirpa (p50)")
                    plt.legend(fontsize=7, loc="best")
                    plt.tight_layout()
                    plt.savefig(fig_dir / "speedup_hot_vs_auto_lirpa_by_workload.png", dpi=200)
                    plt.close()
        except Exception:
            # Optional dependency; keep the pipeline usable in minimal environments.
            print(
                "WARNING: plots are skipped (matplotlib missing or backend error). "
                "Install matplotlib (e.g. `conda install -c conda-forge matplotlib`).",
                file=sys.stderr,
            )
            pass

    # Write a small manifest for human inspection.
    (out_dir / "MANIFEST.txt").write_text(
        "\n".join(
            [
                f"inputs: {', '.join(str(p) for p in paths)}",
                f"rows: {len(rows)}",
                f"csv: {out_dir / 'ablation.csv'}",
                f"table: {out_dir / 'tables' / 'ablation_summary.csv'}" if summary else "table: (none)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
