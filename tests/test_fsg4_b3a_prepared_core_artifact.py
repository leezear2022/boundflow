"""Frozen evidence checks for the FSG4/B3-A prepared-core artifact."""

# pylint: disable=missing-function-docstring,protected-access

import json
from pathlib import Path

from scripts import run_fsg4_b3_counter_diagnostic as diagnostic

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3a-v1"
TAMPER_REPORT = ARTIFACT.with_name("resnet2b-prop0-b3a-v1-tamper-report.json")


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_b3a_artifact_replays_from_source_commit() -> None:
    result = diagnostic._replay(ARTIFACT)
    assert result["status"] == "replay-passed"
    assert result["source_git_head"] == ("c7851c8bae1bc943aa9e3d458e5105deafc553f1")
    assert result["manifest_hash"] == (
        "205978cb69238598dfcb860922e3202677d5b1775f0bd6062218f0369e982c95"
    )
    assert result["report_hash"] == (
        "89a3584dddb47d2a835bca689bdb0ba6b936d26fa5aff20a968c2323dc6cd05b"
    )
    assert result["event_count"] == 5157


def test_frozen_b3a_changes_only_prepared_core_physical_counts() -> None:
    report = _load(ARTIFACT / "report.json")
    snapshot = report["snapshot"]
    assert isinstance(snapshot, dict)
    assert report["configuration"] == "B3-A"
    assert snapshot["configuration"] == "B3-A"
    counts = snapshot["counts"]
    assert isinstance(counts, dict)
    assert counts["template_compile_count"] == 1
    assert counts["template_hit_in_core_count"] == 1
    assert counts["module_binding_move_in_core_count"] == 0
    assert counts["scope_construction_count"] == 1
    assert counts["optimizer_evaluation_count"] == 10
    assert counts["optimizer_update_count"] == 9
    assert counts["full_optimizer_step_snapshot_count"] == 10
    assert counts["forward_trace_build_count"] == 5
    assert counts["kfsb_candidate_count"] == 3
    assert counts["kfsb_child_batch_count"] == 3
    assert counts["timed_candidate_d2h_copy_count"] == 12
    assert counts["committed_mutable_path_count"] == 12
    assert report["fsg3_reference_semantic_failures"] == []
    assert report["correctness_passed"] is True
    assert report["diagnostic_timing_claimed"] is False
    assert report["performance_claimed"] is False


def test_frozen_b3a_provider_fallback_and_outer_resigned_attacks() -> None:
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
        "92a1900a8cdba5f42833dbd02efd2aa510d6027d58d43a1152d9a20f280d9997"
    )
    rows = tamper["rows"]
    assert isinstance(rows, list)
    assert all(isinstance(row, dict) and row["rejected"] is True for row in rows)
