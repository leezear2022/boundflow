"""Generate PR-13B dynamic batching and OOM-split replay evidence."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

import torch
from run_phase7a_pr13a_fixed_replay import make_workload

from boundflow.ir.task import BFTaskModule
from boundflow.runtime.bab import BabResult, solve_bab_mlp
from boundflow.runtime.bab_query import (
    BoundQueryRequest,
    BoundQueryResult,
    FixedBabQueryRecorder,
    QueryBatch,
    compare_query_results,
)
from boundflow.runtime.query_executor import execute_alpha_beta_query_batch
from boundflow.runtime.query_batcher import BatchPolicy, DynamicBatchManager

ARTIFACT_SCHEMA_VERSION = "boundflow.pr13b-dynamic-batch-artifact/v1"


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


def _record_real_stream() -> tuple[
    BFTaskModule, FixedBabQueryRecorder, BabResult
]:
    module, spec, config = make_workload()
    recorder = FixedBabQueryRecorder()
    result = solve_bab_mlp(
        module,
        spec,
        config=config,
        query_recorder=recorder,
    )
    recorder.validate_complete()
    return module, recorder, result


def run(out_dir: Path) -> None:  # pylint: disable=too-many-locals
    """Run dynamic batching plus explicit deterministic OOM fault injection."""

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty artifact: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    module, recorder, solver_result = _record_real_stream()
    expected_by_id = {
        entry.query.query_id: entry.result for entry in recorder.entries
    }

    now = [0]
    manager = DynamicBatchManager(
        BatchPolicy(
            max_batch_size=4,
            memory_budget_bytes=1 << 30,
            max_wait_us=100,
            minimum_fill_ratio=0.75,
        ),
        clock_us=lambda: now[0],
    )
    dynamic_results = []
    final_deadline_us = 0
    for entry in recorder.entries:
        now[0] = int(entry.query.sequence_number) * 25
        final_deadline_us = now[0] + 100
        manager.submit(
            BoundQueryRequest(entry.query, entry.payload),
            deadline_us=final_deadline_us,
            now_us=now[0],
        )
        for batch in manager.pop_ready(now_us=now[0]):
            dynamic_results.extend(
                manager.execute_batch_with_oom_retry(
                    batch,
                    lambda candidate: execute_alpha_beta_query_batch(
                        module, candidate
                    ),
                )
            )
    assert manager.next_wakeup_us is not None
    now[0] = manager.next_wakeup_us
    for batch in manager.pop_ready(now_us=now[0]):
        dynamic_results.extend(
            manager.execute_batch_with_oom_retry(
                batch,
                lambda candidate: execute_alpha_beta_query_batch(module, candidate),
            )
        )

    comparisons = []
    for query_id, actual in dynamic_results:
        expected = expected_by_id[query_id]
        assert expected is not None
        comparisons.append(compare_query_results(query_id, expected, actual))

    fault_manager = DynamicBatchManager(
        BatchPolicy(
            max_batch_size=len(recorder.entries),
            memory_budget_bytes=1 << 30,
            max_wait_us=1_000,
        )
    )
    for entry in recorder.entries:
        fault_manager.submit(
            BoundQueryRequest(entry.query, entry.payload), now_us=0
        )
    fault_batch = fault_manager.pop_ready(now_us=0, force=True)[0]
    physical_sizes: list[int] = []

    def fault_injected_executor(
        candidate: QueryBatch,
    ) -> Sequence[tuple[str, BoundQueryResult]]:
        physical_sizes.append(len(candidate.requests))
        if len(candidate.requests) > 2:
            raise RuntimeError("CUDA out of memory: PR-13B deterministic fault injection")
        return execute_alpha_beta_query_batch(module, candidate)

    fault_results = fault_manager.execute_batch_with_oom_retry(
        fault_batch, fault_injected_executor
    )
    fault_comparisons = []
    for query_id, actual in fault_results:
        expected = expected_by_id[query_id]
        assert expected is not None
        fault_comparisons.append(compare_query_results(query_id, expected, actual))

    raw_path = out_dir / "raw.jsonl"
    raw_rows = [
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "mode": "dynamic_batch",
            "query_id": comparison.query_id,
            "passed": comparison.passed,
            "max_abs_diff": comparison.max_abs_diff,
        }
        for comparison in comparisons
    ] + [
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "mode": "oom_fault_injection",
            "query_id": comparison.query_id,
            "passed": comparison.passed,
            "max_abs_diff": comparison.max_abs_diff,
        }
        for comparison in fault_comparisons
    ]
    raw_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in raw_rows
        ),
        encoding="utf-8",
    )
    all_passed = all(item.passed for item in comparisons + fault_comparisons)
    summary = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ok" if all_passed else "fail",
        "scope": "PR-13B dynamic-batch correctness smoke; not performance",
        "solver_status": solver_result.status,
        "query_count": len(recorder.entries),
        "dynamic_replay_passed": sum(item.passed for item in comparisons),
        "dynamic_replay_failed": sum(not item.passed for item in comparisons),
        "dynamic_max_abs_diff": max(item.max_abs_diff for item in comparisons),
        "dynamic_metrics": manager.audit(),
        "fault_replay_passed": sum(item.passed for item in fault_comparisons),
        "fault_replay_failed": sum(not item.passed for item in fault_comparisons),
        "fault_max_abs_diff": max(item.max_abs_diff for item in fault_comparisons),
        "fault_physical_batch_sizes": physical_sizes,
        "fault_metrics": fault_manager.audit(),
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
            "python scripts/run_phase7a_pr13b_dynamic_batch_replay.py "
            f"--out-dir {out_dir}"
        ),
        "files": {
            raw_path.name: _sha256(raw_path),
            summary_path.name: _sha256(summary_path),
        },
    }
    _write_json(out_dir / "manifest.json", manifest)
    if not all_passed:
        raise RuntimeError("PR-13B dynamic replay mismatch")


def main() -> None:
    """Parse the immutable artifact path and run PR-13B smoke."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.out_dir)


if __name__ == "__main__":
    main()
