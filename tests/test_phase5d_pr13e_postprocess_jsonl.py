from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.postprocess_ablation_jsonl import main as post_main


def test_postprocess_jsonl_to_csv(tmp_path: Path) -> None:
    jsonl = tmp_path / "ablation.jsonl"
    out_dir = tmp_path / "out"

    row = {
        "schema_version": "0.1",
        "meta": {"git_commit": "deadbeef", "time_utc": "2025-12-20T00:00:00+00:00"},
        "workload": {"model": "mlp", "input_shape": [4, 16], "eps": 0.1, "domain": "interval_ibp", "spec": "none"},
        "config": {
            "planner_config_dump": {"partition": {"policy": "v2_baseline"}, "storage_reuse": {"enabled": False}},
            "tvm_options": {"memory_plan_mode": "<MemoryPlanMode.DEFAULT: 'default'>", "enable_task_fusion_pipeline": True},
        },
        "planner": {"plan_ms_total": 1.0, "num_tasks": 2, "num_edges": 1, "storage": {"logical_buffers": 8, "physical_buffers": 8}},
        "tvm": {
            "compile_stats_agg": {"compile_ms_total": 2.0, "call_tir_after_legalize_sum": 10, "call_tir_after_fuse_tir_sum": 5},
            "compile_cache_stats_delta_compile_first_run": {"task_compile_cache_miss": 2, "task_compile_cache_hit": 0, "task_compile_fail": 0},
        },
        "runtime": {"compile_first_run_ms": 10.0, "run_ms_avg": 1.0, "run_ms_p50": 1.0, "run_ms_p95": 1.0},
        "correctness": {
            "python_vs_tvm_max_abs_diff_lb": 0.0,
            "python_vs_tvm_max_abs_diff_ub": 0.0,
            "python_vs_tvm_max_rel_diff_lb": 0.0,
            "python_vs_tvm_max_rel_diff_ub": 0.0,
        },
        "baseline": {"auto_lirpa": None},
    }

    jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

    rc = post_main([str(jsonl), "--out-dir", str(out_dir), "--no-plots"])
    assert rc == 0

    csv_path = out_dir / "ablation.csv"
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    assert "schema_version" in text
    assert "compile_first_run_ms" in text
    assert "auto_lirpa_run_ms_p50" in text

    summary_path = out_dir / "tables" / "ablation_summary.csv"
    assert summary_path.exists()
    assert (out_dir / "tables" / "table_main.csv").exists()
    manifest = (out_dir / "MANIFEST.txt").read_text(encoding="utf-8")
    assert "\\n" not in manifest
    assert "inputs:" in manifest and "rows:" in manifest


def test_postprocess_does_not_treat_missing_correctness_as_zero(tmp_path: Path) -> None:
    jsonl = tmp_path / "ablation.jsonl"
    out_dir = tmp_path / "out"

    row = {
        "schema_version": "0.1",
        "meta": {"git_commit": "deadbeef", "time_utc": "2025-12-20T00:00:00+00:00"},
        "workload": {"model": "mlp", "input_shape": [4, 16], "eps": 0.1, "domain": "interval_ibp", "spec": "none"},
        "config": {
            "planner_config_dump": {"partition": {"policy": "v2_baseline"}, "storage_reuse": {"enabled": False}},
            "tvm_options": {"memory_plan_mode": "<MemoryPlanMode.DEFAULT: 'default'>", "enable_task_fusion_pipeline": True},
        },
        "planner": {"plan_ms_total": 1.0, "num_tasks": 2, "num_edges": 1, "storage": {"logical_buffers": 8, "physical_buffers": 8}},
        "tvm": {
            "compile_cache_stats_delta_compile_first_run": {"task_compile_cache_miss": 1, "task_compile_cache_hit": 0, "task_compile_fail": 0},
        },
        "runtime": {"compile_first_run_ms": 10.0, "run_ms_avg": 1.0, "run_ms_p50": 1.0, "run_ms_p95": 1.0},
        # correctness intentionally missing / null.
        "correctness": {},
        "baseline": {"auto_lirpa": None},
    }
    jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

    rc = post_main([str(jsonl), "--out-dir", str(out_dir), "--no-plots"])
    assert rc == 0

    summary_path = out_dir / "tables" / "ablation_summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    r0 = rows[0]
    assert r0["python_vs_tvm_max_rel_diff_max"] in ("", "None")
    assert r0["python_vs_tvm_rel_diff_missing"] == "1"


def test_postprocess_group_key_includes_eps_and_shape(tmp_path: Path) -> None:
    jsonl = tmp_path / "ablation.jsonl"
    out_dir = tmp_path / "out"

    base = {
        "schema_version": "0.1",
        "meta": {"git_commit": "deadbeef", "time_utc": "2025-12-20T00:00:00+00:00"},
        "workload": {"model": "mlp", "input_shape": [4, 16], "eps": 0.1, "domain": "interval_ibp", "spec": "none"},
        "config": {
            "planner_config_dump": {"partition": {"policy": "v2_baseline"}, "storage_reuse": {"enabled": False}},
            "tvm_options": {"memory_plan_mode": "<MemoryPlanMode.DEFAULT: 'default'>", "enable_task_fusion_pipeline": True},
        },
        "planner": {"plan_ms_total": 1.0, "num_tasks": 2, "num_edges": 1, "storage": {"logical_buffers": 8, "physical_buffers": 8}},
        "tvm": {"compile_cache_stats_delta_compile_first_run": {"task_compile_cache_miss": 1, "task_compile_cache_hit": 0, "task_compile_fail": 0}},
        "runtime": {"compile_first_run_ms": 10.0, "run_ms_avg": 1.0, "run_ms_p50": 1.0, "run_ms_p95": 1.0},
        "correctness": {
            "python_vs_tvm_max_abs_diff_lb": 0.0,
            "python_vs_tvm_max_abs_diff_ub": 0.0,
            "python_vs_tvm_max_rel_diff_lb": 0.0,
            "python_vs_tvm_max_rel_diff_ub": 0.0,
        },
        "baseline": {"auto_lirpa": None},
    }

    row1 = dict(base)
    row2 = json.loads(json.dumps(base))
    row2["workload"]["eps"] = 0.2
    row2["workload"]["input_shape"] = [2, 16]

    jsonl.write_text(json.dumps(row1) + "\n" + json.dumps(row2) + "\n", encoding="utf-8")
    rc = post_main([str(jsonl), "--out-dir", str(out_dir), "--no-plots"])
    assert rc == 0

    summary_path = out_dir / "tables" / "ablation_summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
