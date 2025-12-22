from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.postprocess_ablation_jsonl import main as post_main


def test_pr14d_postprocess_uses_deduped_baseline_by_key(tmp_path: Path) -> None:
    jsonl = tmp_path / "ablation.jsonl"
    out_dir = tmp_path / "out"

    # Two matrix points, same baseline_key but different per-row baseline fields.
    # Postprocess should de-duplicate by baseline_key and prefer cache_hit=False row.
    base = {
        "schema_version": "0.1",
        "status": "ok",
        "error": None,
        "meta": {"git_commit": "deadbeef", "time_utc": "2025-12-21T00:00:00+00:00"},
        "workload": {"model": "mlp", "input_shape": [4, 16], "eps": 0.1, "domain": "interval_ibp", "spec": "none"},
        "config": {
            "planner_config_dump": {"partition": {"policy": "v2_baseline"}, "storage_reuse": {"enabled": False}},
            "tvm_options": {"memory_plan_mode": "<MemoryPlanMode.DEFAULT: 'default'>", "enable_task_fusion_pipeline": True},
        },
        "planner": {"plan_ms_total": 1.0, "num_tasks": 2, "num_edges": 1, "storage": {"logical_buffers": 8, "physical_buffers": 8}},
        "tvm": {"compile_cache_stats_delta_compile_first_run": {"task_compile_cache_miss": 1, "task_compile_cache_hit": 0, "task_compile_fail": 0}},
        "runtime": {"compile_first_run_ms": 10.0, "run_ms_cold": 2.0, "run_ms_avg": 1.0, "run_ms_p50": 1.0, "run_ms_p95": 1.0},
        "correctness": {"python_vs_auto_lirpa_gate": {"ok": True}},
    }

    row0 = json.loads(json.dumps(base))
    row0["baseline"] = {
        "auto_lirpa": {
            "available": True,
            "baseline_key": "k0",
            "cache_hit": False,
            "init_ms": 100.0,
            "run_ms_cold": 200.0,
            "run_ms_p50": 50.0,
            "run_ms_p95": 60.0,
            "method": "IBP",
            "version": "x",
            "reason": "",
            "spec_hash": "s0",
        }
    }

    row1 = json.loads(json.dumps(base))
    row1["config"]["tvm_options"]["enable_task_fusion_pipeline"] = False
    row1["baseline"] = {
        "auto_lirpa": {
            "available": True,
            "baseline_key": "k0",
            "cache_hit": True,
            "init_ms": 999.0,
            "run_ms_cold": 999.0,
            "run_ms_p50": 999.0,
            "run_ms_p95": 999.0,
            "method": "IBP",
            "version": "x",
            "reason": "",
            "spec_hash": "s0",
        }
    }

    jsonl.write_text(json.dumps(row0) + "\n" + json.dumps(row1) + "\n", encoding="utf-8")
    rc = post_main([str(jsonl), "--out-dir", str(out_dir), "--no-plots"])
    assert rc == 0

    main_table = out_dir / "tables" / "table_main.csv"
    with main_table.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    for r in rows:
        assert r.get("auto_lirpa_baseline_key") == "k0"
        assert r.get("auto_lirpa_run_ms_p50") in ("50.0", "50")
