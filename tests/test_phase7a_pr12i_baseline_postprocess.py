"""Derived-evidence checks for the PR-12I baseline report."""

import csv
import json
from pathlib import Path

from scripts.postprocess_phase7a_pr12i_baselines import main


def _row(
    *, backend: str, latency: float, peak: int, status: str = "ok"
) -> dict[str, object]:
    row: dict[str, object] = {
        "benchmark_contract": {"level": "end_to_end_final_bound"},
        "workload": {"case_id": "case-a", "family": "linear"},
        "candidate": {"backend": backend, "stream": "default"},
        "status": status,
    }
    if status == "ok":
        row.update(
            {
                "runtime": {
                    "host_group_per_query": {"median_ms": latency},
                    "estimated_compile_overhead_ms": 3.0,
                },
                "memory": {"peak_allocated_delta_bytes": peak},
                "correctness": {"max_abs_diff": 0.0, "max_rel_diff": 0.0},
            }
        )
    return row


def test_postprocess_keeps_failures_and_computes_pareto(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    rows = [
        _row(backend="pytorch_eager", latency=2.0, peak=100),
        _row(backend="tvm_fused_tir", latency=1.0, peak=50),
        _row(backend="torch_compile", latency=0.0, peak=0, status="not_applicable"),
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    out_dir = tmp_path / "report"

    assert main(["--raw", str(raw), "--out-dir", str(out_dir), "--no-plots"]) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    with (out_dir / "baseline.csv").open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert summary["rows"] == 3
    assert summary["status_counts"] == {"not_applicable": 1, "ok": 2}
    assert len(table) == 2
    fused = next(row for row in table if row["backend"] == "tvm_fused_tir")
    assert float(fused["speedup_vs_eager"]) == 2.0
    assert float(fused["peak_ratio_vs_eager"]) == 0.5
    assert fused["pareto"] == "True"
