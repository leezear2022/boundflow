import csv
import json
from pathlib import Path


def test_phase6h_report_csv_schema(tmp_path: Path) -> None:
    from scripts.report_phase6h_e2e import main as report_main

    jsonl = tmp_path / "in.jsonl"
    out_json = tmp_path / "out.json"
    out_csv = tmp_path / "out.csv"
    out_md = tmp_path / "summary.md"

    # Produce one JSONL line using the bench (8 switch combos) via subprocess to capture stdout JSON.
    import subprocess
    import sys

    out = subprocess.check_output(
        [
            sys.executable,
            "scripts/bench_phase6h_bab_e2e_time_to_verify.py",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--workload",
            "1d_relu",
            "--oracle",
            "alpha_beta",
            "--steps",
            "0",
            "--max-nodes",
            "64",
            "--node-batch-size",
            "8",
            "--warmup",
            "1",
            "--iters",
            "1",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
    ).decode("utf-8")
    payload = json.loads(out)
    out_json.write_text(json.dumps(payload), encoding="utf-8")
    jsonl.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    rc2 = report_main(["--in-jsonl", str(jsonl), "--out-csv", str(out_csv), "--out-summary-md", str(out_md)])
    assert rc2 == 0
    assert out_csv.exists() and out_csv.stat().st_size > 0
    assert out_md.exists() and out_md.stat().st_size > 0

    rows = list(csv.DictReader(out_csv.read_text(encoding="utf-8").splitlines()))
    # 8 combos * 2 paths (batch/serial)
    assert len(rows) == 16

    required_cols = [
        "path",
        "workload",
        "enable_node_eval_cache",
        "use_branch_hint",
        "enable_batch_infeasible_prune",
        "comparable",
        "speedup",
        "speedup_p90",
        "verdict",
        "time_ms_p50",
        "time_ms_p90",
        "runs_count",
        "valid_runs_count",
        "timeouts_count",
        "counts_oracle_calls",
        "counts_forward_trace_calls",
        "counts_cache_hit_rate",
        "stats_avg_batch_fill_rate",
    ]
    for c in required_cols:
        assert c in rows[0], c
