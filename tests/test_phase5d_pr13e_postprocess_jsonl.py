from __future__ import annotations

import json
from pathlib import Path

from scripts.postprocess_ablation_jsonl import main as post_main


def test_postprocess_jsonl_to_csv(tmp_path: Path) -> None:
    jsonl = tmp_path / "ablation.jsonl"
    out_dir = tmp_path / "out"

    row = {
        "schema_version": "0.1",
        "meta": {"git_commit": "deadbeef", "time_utc": "2025-12-20T00:00:00+00:00"},
        "workload": {"model": "mlp", "input_shape": [4, 16], "eps": 0.1},
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

    summary_path = out_dir / "tables" / "ablation_summary.csv"
    assert summary_path.exists()

