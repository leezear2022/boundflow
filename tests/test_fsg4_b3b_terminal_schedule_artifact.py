"""Frozen evidence checks for the FSG4/B3-B terminal Schedule artifact."""

# pylint: disable=missing-function-docstring,protected-access

import json
from pathlib import Path

from scripts import run_fsg4_b3_counter_diagnostic as diagnostic

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3b-v1"
TAMPER_REPORT = ARTIFACT.with_name("resnet2b-prop0-b3b-v1-tamper-report.json")


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_b3b_artifact_replays_from_source_commit() -> None:
    result = diagnostic._replay(ARTIFACT)
    assert result["status"] == "replay-passed"
    assert result["source_git_head"] == ("42df2dcae2d5c5a10f27ab707d8d7aff7686d15e")
    assert result["manifest_hash"] == (
        "2960c85c9b6dfe1382bef39804a9a88b618b438b9b2cb55d629aa24a99c18644"
    )
    assert result["report_hash"] == (
        "f7c24e9080a51fba990bf67502ee91519b8d67047a37d29a64654cbe4ea77061"
    )
    assert result["event_count"] == 5157


def test_frozen_b3b_terminal_schedule_physical_counts() -> None:
    report = _load(ARTIFACT / "report.json")
    snapshot = report["snapshot"]
    assert isinstance(snapshot, dict)
    assert report["configuration"] == "B3-B"
    assert snapshot["configuration"] == "B3-B"
    counts = snapshot["counts"]
    assert isinstance(counts, dict)
    assert counts["template_compile_count"] == 1
    assert counts["template_hit_in_core_count"] == 1
    assert counts["module_binding_move_in_core_count"] == 0
    assert counts["scope_construction_count"] == 1
    assert counts["optimizer_evaluation_count"] == 10
    assert counts["optimizer_update_count"] == 9
    assert counts["full_optimizer_step_snapshot_count"] == 0
    assert counts["forward_trace_build_count"] == 4
    assert counts["kfsb_candidate_count"] == 3
    assert counts["kfsb_child_batch_count"] == 3
    assert counts["timed_candidate_d2h_copy_count"] == 12
    assert counts["committed_mutable_path_count"] == 12
    assert report["fsg3_reference_semantic_failures"] == []
    assert report["correctness_passed"] is True
    assert report["diagnostic_timing_claimed"] is False
    assert report["performance_claimed"] is False


def test_frozen_b3b_provider_fallback_and_outer_resigned_attacks() -> None:
    report = _load(ARTIFACT / "report.json")
    snapshot = report["snapshot"]
    assert isinstance(snapshot, dict)
    for field in (
        "provider_core_call_count",
        "provider_compute_bounds_call_count",
        "provider_update_bounds_call_count",
        "fallback_dispatch_count",
    ):
        assert snapshot[field] == 0
    tamper = _load(TAMPER_REPORT)
    assert tamper["attack_count"] == 6
    assert tamper["rejected_count"] == 6
    assert tamper["report_hash"] == (
        "6c1dde930b250d62a9eb00026729888363ea02bae42eb3331daa384ece73dbcf"
    )
    rows = tamper["rows"]
    assert isinstance(rows, list)
    assert all(isinstance(row, dict) and row["rejected"] is True for row in rows)
