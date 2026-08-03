"""Contract tests for PR-12 JSONL-to-CSV/Pareto postprocessing."""

import csv
import json
from pathlib import Path

from scripts.postprocess_phase7a_pr12_runtime_pareto import main


def _candidate(case: str, backend: str, latency: float, peak: int) -> dict:
    return {
        "status": "ok",
        "split": {"role": "heldout"},
        "workload": {
            "case_id": case,
            "family": "linear",
            "boundary_bytes": 4096,
            "budget_bytes": 8192,
        },
        "candidate": {"backend": backend, "stream": "default", "eligible": True},
        "runtime": {
            "host_group_per_query": {"median_ms": latency},
            "estimated_compile_overhead_ms": (
                10.0 if backend == "tvm_fused_tir" else 0.0
            ),
        },
        "memory": {"peak_allocated_delta_bytes": peak},
        "correctness": {"max_abs_diff": 0.0, "max_rel_diff": 0.0},
    }


def test_pareto_postprocess_writes_traceable_csv_and_manifest(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    planner = tmp_path / "planner.jsonl"
    calibration.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _candidate("cal", "pytorch_eager", 2.0, 4000),
                _candidate("cal", "tvm_fused_tir", 1.0, 2000),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    heldout.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _candidate("held", "pytorch_eager", 3.0, 6000),
                _candidate("held", "tvm_fused_tir", 2.0, 3000),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    planner.write_text(
        json.dumps(
            {
                "role": "heldout",
                "latency_regret": 1.0,
                "selected_budget_feasible": True,
                "unsafe_fusion": False,
                "decision": {"reason": "calibration_predicts_fused_faster"},
            }
        )
        + "\n",
        encoding="utf-8",
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

    rows = list(csv.DictReader((out_dir / "candidate_summary.csv").open()))
    assert len(rows) == 2
    assert float(rows[1]["warm_speedup"]) == 1.5
    assert float(rows[1]["fused_over_eager_peak"]) == 0.5
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["candidate_rows"] == 2
    assert manifest["planner"]["unsafe_fusion_count"] == 0
    assert manifest["outputs"]["candidate_summary.csv"]
