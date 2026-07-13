#!/usr/bin/env python
"""Replay a frozen PR-12 planner from calibration and candidate JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from boundflow.planner.fused_crown_backend import FusedCrownMultiBackendPlanner
from scripts.benchmark_phase7a_pr12_runtime_pareto import (
    _fallback_control_workload,
    _observations_from_rows,
    _planner_evaluation,
    _read_jsonl,
    _workload,
    _write_jsonl,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Fit from calibration only and replay decisions over immutable raw rows."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    split_id = str(split["split_id"])
    planner = FusedCrownMultiBackendPlanner.fit(
        _observations_from_rows(_read_jsonl(args.calibration))
    )
    workloads = [
        _workload(record, split_role="heldout") for record in split["final_heldout"]
    ]
    workloads.append(_fallback_control_workload())
    candidate_rows = _read_jsonl(args.candidates)
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        by_case.setdefault(str(row["workload"]["case_id"]), []).append(row)
    evaluations = [
        _planner_evaluation(
            workload,
            by_case[workload.case_id],
            planner,
            split_id=split_id,
        )
        for workload in workloads
    ]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    planner_path = args.out_dir / "planner.jsonl"
    model_path = args.out_dir / "planner_model.json"
    _write_jsonl(planner_path, evaluations)
    model_path.write_text(
        json.dumps(planner.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    heldout = [row for row in evaluations if row["role"] == "heldout"]
    summary = {
        "schema_version": "boundflow.pr12-planner-replay/v2",
        "split_id": split_id,
        "heldout_cases": len(heldout),
        "oracle_hits": sum(row["latency_regret"] <= 1.0 + 1e-9 for row in heldout),
        "budget_feasible": sum(row["selected_budget_feasible"] for row in heldout),
        "unsafe_fusion_count": sum(row["unsafe_fusion"] for row in evaluations),
        "planner_overhead_ms": [row["planner_overhead_ms"] for row in evaluations],
        "inputs": {
            "split": _sha256(args.split_file),
            "calibration": _sha256(args.calibration),
            "candidates": _sha256(args.candidates),
        },
        "outputs": {
            "planner.jsonl": _sha256(planner_path),
            "planner_model.json": _sha256(model_path),
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
