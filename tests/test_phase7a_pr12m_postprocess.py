"""Artifact contract for PR-12M normalized report generation."""

import csv
import json
from pathlib import Path

from scripts.postprocess_phase7a_pr12m_compile_aware import main


def test_pr12m_postprocess_emits_csv_and_manifest(tmp_path: Path) -> None:
    planner = tmp_path / "planner.jsonl"
    rows = [
        {
            "case_id": "held",
            "family": "linear",
            "policy_id": policy,
            "budget_mib": budget,
            "decision": {"backend": backend, "selected_risk_tier": 0},
            "oracle_backend": backend,
            "amortized_latency_regret": 1.0,
            "selected_peak_allocated_bytes": 100,
            "any_measured_budget_feasible": True,
            "selected_measured_budget_feasible": True,
        }
        for policy, budget, backend in (
            ("cold", 16, "pytorch_eager"),
            ("warm", None, "tvm_fused_tir"),
        )
    ]
    planner.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"rows": 2}), encoding="utf-8")
    output = tmp_path / "report"

    assert (
        main(
            [
                "--planner",
                str(planner),
                "--summary",
                str(summary),
                "--out-dir",
                str(output),
                "--no-plots",
            ]
        )
        == 0
    )
    with (output / "decisions.csv").open(encoding="utf-8", newline="") as handle:
        normalized = list(csv.DictReader(handle))
    manifest = json.loads((output / "manifest.json").read_text())

    assert normalized[1]["budget_mib"] == "unbounded"
    assert manifest["summary"]["rows"] == 2
    assert manifest["outputs"]["decisions.csv"]
