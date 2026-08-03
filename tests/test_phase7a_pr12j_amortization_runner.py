"""Artifact smoke for PR-12J compile amortization and postprocessing."""

import csv
import json
from pathlib import Path

import pytest
import torch

from scripts.benchmark_phase7a_pr12j_compile_amortization import main as run_main
from scripts.postprocess_phase7a_pr12j_compile_amortization import (
    main as postprocess_main,
)
from tests.pr12_split_fixtures import write_pr12_v2_split


def _baseline_row(backend: str, latency_ms: float) -> dict[str, object]:
    return {
        "status": "ok",
        "benchmark_contract": {
            "level": "end_to_end_final_bound",
            "compliant": True,
        },
        "workload": {"case_id": "linear-unseen-shape-a"},
        "candidate": {"backend": backend, "stream": "default"},
        "runtime": {"host_group_per_query": {"median_ms": latency_ms}},
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pr12j_runner_records_restart_disk_hit_and_report(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(
        json.dumps(_baseline_row("pytorch_eager", 1.0))
        + "\n"
        + json.dumps(_baseline_row("pytorch_chunked", 1.2))
        + "\n",
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw"
    split = write_pr12_v2_split(tmp_path)

    assert (
        run_main(
            [
                "--split-file",
                str(split),
                "--baseline-raw",
                str(baseline),
                "--out-dir",
                str(raw_dir),
                "--case-ids",
                "linear-unseen-shape-a",
                "--warmup",
                "1",
                "--groups",
                "1",
                "--repeats",
                "1",
            ]
        )
        == 0
    )
    row = json.loads((raw_dir / "raw.jsonl").read_text(encoding="utf-8"))
    assert row["status"] == "ok"
    assert row["unique_compiled_modules"] == 1
    assert any(event["event"] == "miss" for event in row["runtime"]["cache_events"])
    restart = row["process_restart_disk_cache"]
    assert restart["all_cache_events_disk_or_memory"]
    assert any(event["event"] == "disk_hit" for event in restart["cache_events"])
    assert len(row["amortization"]["fresh_query_totals"]) == 11

    report_dir = tmp_path / "report"
    assert (
        postprocess_main(
            [
                "--raw",
                str(raw_dir / "raw.jsonl"),
                "--out-dir",
                str(report_dir),
                "--no-plots",
            ]
        )
        == 0
    )
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    with (report_dir / "amortization.csv").open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert summary["correct_rows"] == 1
    assert len(table) == 66
