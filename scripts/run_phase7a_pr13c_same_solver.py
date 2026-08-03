"""Generate PR-13C original-vs-query-runtime same-solver evidence."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import torch
from run_phase7a_pr13a_fixed_replay import make_workload

from boundflow.runtime.bab import solve_bab_mlp
from boundflow.runtime.bab_query import (
    FixedBabQueryRecorder,
    compare_query_results,
)
from boundflow.runtime.bab_query_runtime import (
    SameSolverQueryRuntime,
    SameSolverRuntimeConfig,
)

ARTIFACT_SCHEMA_VERSION = "boundflow.pr13c-same-solver-artifact/v1"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, encoding="utf-8"
    ).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(out_dir: Path) -> None:  # pylint: disable=too-many-locals
    """Run one solver twice while replacing only bound-query execution."""

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty artifact: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    module, spec, base_config = make_workload()
    config = replace(base_config, node_batch_size=4, alpha_steps=3)
    baseline_recorder = FixedBabQueryRecorder()
    runtime_recorder = FixedBabQueryRecorder()

    baseline_start = time.perf_counter_ns()
    baseline = solve_bab_mlp(
        module,
        spec,
        config=config,
        query_recorder=baseline_recorder,
    )
    baseline_wall_us = (time.perf_counter_ns() - baseline_start) / 1_000.0

    runtime = SameSolverQueryRuntime(
        SameSolverRuntimeConfig(
            max_batch_size=4,
            memory_budget_bytes=1 << 30,
            allow_legacy_alpha_beta=True,
        )
    )
    adapted_start = time.perf_counter_ns()
    adapted = solve_bab_mlp(
        module,
        spec,
        config=config,
        query_recorder=runtime_recorder,
        query_runtime=runtime,
    )
    adapted_wall_us = (time.perf_counter_ns() - adapted_start) / 1_000.0
    baseline_recorder.validate_complete()
    runtime_recorder.validate_complete()

    comparisons = []
    for expected, actual in zip(
        baseline_recorder.entries, runtime_recorder.entries
    ):
        if expected.result is None or actual.result is None:
            raise AssertionError("same-solver trace contains incomplete results")
        comparisons.append(
            compare_query_results(
                expected.query.query_id,
                expected.result,
                actual.result,
            )
        )
    query_ids_match = [
        entry.query.query_id for entry in baseline_recorder.entries
    ] == [entry.query.query_id for entry in runtime_recorder.entries]
    solver_match = {
        "status": adapted.status == baseline.status,
        "nodes_visited": adapted.nodes_visited == baseline.nodes_visited,
        "nodes_evaluated": adapted.nodes_evaluated == baseline.nodes_evaluated,
        "nodes_expanded": adapted.nodes_expanded == baseline.nodes_expanded,
        "max_queue": adapted.max_queue == baseline.max_queue,
        "batch_rounds": adapted.batch_rounds == baseline.batch_rounds,
        "best_lower": abs(adapted.best_lower - baseline.best_lower) <= 2e-4,
        "best_upper": abs(adapted.best_upper - baseline.best_upper) <= 2e-4,
    }
    all_passed = (
        query_ids_match
        and all(solver_match.values())
        and len(comparisons) == len(baseline_recorder.entries)
        and all(item.passed for item in comparisons)
    )
    raw_path = out_dir / "raw.jsonl"
    raw_path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": ARTIFACT_SCHEMA_VERSION,
                    "query_id": comparison.query_id,
                    "passed": comparison.passed,
                    "status_match": comparison.status_match,
                    "branch_match": comparison.branch_match,
                    "state_version_match": comparison.state_version_match,
                    "max_abs_diff": comparison.max_abs_diff,
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for comparison in comparisons
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ok" if all_passed else "fail",
        "scope": "PR-13C same-solver correctness smoke; wall time is non-authoritative",
        "query_count_baseline": len(baseline_recorder.entries),
        "query_count_runtime": len(runtime_recorder.entries),
        "query_ids_match": query_ids_match,
        "query_replay_passed": sum(item.passed for item in comparisons),
        "query_replay_failed": sum(not item.passed for item in comparisons),
        "max_abs_diff": max(item.max_abs_diff for item in comparisons),
        "solver_match": solver_match,
        "baseline": {
            "status": baseline.status,
            "nodes_visited": baseline.nodes_visited,
            "nodes_evaluated": baseline.nodes_evaluated,
            "nodes_expanded": baseline.nodes_expanded,
            "wall_us_non_authoritative": baseline_wall_us,
        },
        "runtime": {
            "status": adapted.status,
            "nodes_visited": adapted.nodes_visited,
            "nodes_evaluated": adapted.nodes_evaluated,
            "nodes_expanded": adapted.nodes_expanded,
            "wall_us_non_authoritative": adapted_wall_us,
            "batch_audit": runtime.audit(),
        },
    }
    summary_path = out_dir / "summary.json"
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "torch_version": torch.__version__,
        "seed": 0,
        "command": (
            "python scripts/run_phase7a_pr13c_same_solver.py "
            f"--out-dir {out_dir}"
        ),
        "files": {
            raw_path.name: _sha256(raw_path),
            summary_path.name: _sha256(summary_path),
        },
    }
    _write_json(out_dir / "manifest.json", manifest)
    if not all_passed:
        raise RuntimeError("PR-13C same-solver mismatch")


def main() -> None:
    """Parse the immutable artifact path and run PR-13C smoke."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.out_dir)


if __name__ == "__main__":
    main()
