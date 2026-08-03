"""Contract tests for independent PR-11 profile aggregation."""

import json

from scripts.aggregate_phase7a_pr11_barrier_profiles import main
from tests.test_phase7a_pr11_barrier_eval_runner import _write


def test_profile_aggregation_uses_cross_replicate_medians(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    third = tmp_path / "third.jsonl"
    out = tmp_path / "aggregate"
    _write(first, workload="net", scale=1.0)
    _write(second, workload="net", scale=2.0)
    _write(third, workload="net", scale=3.0)

    result = main(
        [
            "--profile",
            str(first),
            "--profile",
            str(second),
            "--profile",
            str(third),
            "--out-dir",
            str(out),
        ]
    )

    assert result == 0
    rows = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
    dense = next(row for row in rows if row["query_id"].endswith(":DD"))
    assert dense["timing_trace_off"]["latency_ms_median"] == 2.0
    assert dense["timing_trace_off"]["peak_cuda_allocated_bytes"] == 3_000_000
    assert dense["replicate_aggregation"]["replicate_count"] == 3
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["replicate_count"] == 3
