"""JSONL-to-report contract for the PR-12G three-candidate evaluation."""

import csv
import json
from pathlib import Path

from scripts.postprocess_phase7a_pr12g_multibackend import main


def _candidate(backend: str, latency: float, peak: int) -> dict:
    return {
        "status": "ok",
        "split": {"role": "heldout"},
        "workload": {
            "case_id": "held",
            "family": "linear",
            "boundary_bytes": 4096,
            "budget_bytes": 8192,
        },
        "candidate": {"backend": backend, "stream": "default", "eligible": True},
        "runtime": {
            "host_group_per_query": {"median_ms": latency},
            "estimated_compile_overhead_ms": 0.0,
        },
        "memory": {"peak_allocated_delta_bytes": peak},
        "correctness": {"max_abs_diff": 0.0, "max_rel_diff": 0.0},
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_multibackend_report_tracks_selected_speedup_and_hashes(tmp_path: Path) -> None:
    calibration = tmp_path / "cal.jsonl"
    heldout = tmp_path / "held.jsonl"
    planner = tmp_path / "planner.jsonl"
    _write(
        calibration,
        [
            {**_candidate(backend, latency, peak), "split": {"role": "calibration"}}
            for backend, latency, peak in (
                ("pytorch_eager", 2.0, 4000),
                ("pytorch_chunked", 1.5, 3000),
                ("tvm_fused_tir", 3.0, 2000),
            )
        ],
    )
    _write(
        heldout,
        [
            _candidate(backend, latency, peak)
            for backend, latency, peak in (
                ("pytorch_eager", 2.0, 4000),
                ("pytorch_chunked", 1.0, 3000),
                ("tvm_fused_tir", 3.0, 2000),
            )
        ],
    )
    _write(
        planner,
        [
            {
                "role": "heldout",
                "case_id": "held",
                "family": "linear",
                "decision": {"backend": "pytorch_chunked"},
                "oracle_backend": "pytorch_chunked",
                "latency_regret": 1.0,
                "selected_peak_allocated_bytes": 3000,
                "budget_bytes": 8192,
                "selected_budget_feasible": True,
                "unsafe_fusion": False,
                "planner_overhead_ms": 0.01,
            }
        ],
    )
    out_dir = tmp_path / "report"

    assert (
        main(
            [
                "--calibration",
                str(calibration),
                "--heldout",
                str(heldout),
                "--planner",
                str(planner),
                "--out-dir",
                str(out_dir),
                "--no-plots",
            ]
        )
        == 0
    )

    planner_rows = list(csv.DictReader((out_dir / "planner.csv").open()))
    assert float(planner_rows[0]["selected_speedup_vs_eager"]) == 2.0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_hits"] == 1
    assert summary["geomean_selected_speedup_vs_eager"] == 2.0
    assert summary["selected_backend_counts"] == {"pytorch_chunked": 1}
    assert summary["fallback_controls"] == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["candidate_rows"] == 6
    assert manifest["outputs"]["planner.csv"]
