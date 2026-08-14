"""Frozen evidence checks for the FSG4/B3-0 real B2 counter artifact."""

# pylint: disable=missing-function-docstring,protected-access

import json
from pathlib import Path

from scripts import run_fsg4_b3_counter_diagnostic as diagnostic

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b2-v1"
TAMPER_REPORT = ARTIFACT.with_name("resnet2b-prop0-b2-v1-tamper-report.json")


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_b3_0_artifact_replays_from_source_commit() -> None:
    result = diagnostic._replay(ARTIFACT)
    assert result["status"] == "replay-passed"
    assert result["source_git_head"] == ("419536126504e2666a5db14681668b7d1add166a")
    assert result["manifest_hash"] == (
        "ccf15ee17cb1ee74b95984a203cb4893e52d70becbc3ba2d3db70618490bb376"
    )
    assert result["report_hash"] == (
        "4304ffe87ce09c6e14ff633ae72f469b6b1fb7c60d297179e74176a3a41ad68e"
    )
    assert result["event_count"] == 4625


def test_frozen_b3_0_physical_counters_and_claim_boundary() -> None:
    report = _load(ARTIFACT / "report.json")
    snapshot = report["snapshot"]
    assert isinstance(snapshot, dict)
    counts = snapshot["counts"]
    assert isinstance(counts, dict)
    assert counts["scope_construction_count"] == 2
    assert counts["optimizer_evaluation_count"] == 10
    assert counts["optimizer_update_count"] == 9
    assert counts["full_optimizer_step_snapshot_count"] == 10
    assert counts["forward_trace_build_count"] == 5
    assert counts["kfsb_child_batch_count"] == 3
    assert counts["timed_candidate_d2h_copy_count"] == 12
    assert counts["committed_mutable_path_count"] == 12
    assert counts["tensor_content_hash_count"] == 4417
    assert counts["typed_validate_call_count"] == 84
    assert counts["stable_hash_call_count"] == 10
    assert report["fsg3_reference_semantic_failures"] == []
    assert report["diagnostic_timing_claimed"] is False
    assert report["performance_claimed"] is False


def test_frozen_b3_0_outer_resigned_attacks_are_all_rejected() -> None:
    report = _load(TAMPER_REPORT)
    assert report["attack_count"] == 6
    assert report["rejected_count"] == 6
    assert report["report_hash"] == (
        "f6392fa609c02d043b2397e36e54e52124630aa93fe51679892058efff644d1d"
    )
    rows = report["rows"]
    assert isinstance(rows, list)
    assert all(isinstance(row, dict) and row["rejected"] is True for row in rows)
