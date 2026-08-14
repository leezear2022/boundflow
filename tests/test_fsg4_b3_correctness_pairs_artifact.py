"""Frozen evidence checks for five fresh FSG4/B3 correctness pairs."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

import json
from pathlib import Path

from scripts import run_fsg4_b3_correctness_pairs as pairs

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/fsg4-b3-correctness-pairs/resnet2b-prop0-v1"
TAMPER_REPORT = ARTIFACT.with_name("resnet2b-prop0-v1-tamper-report.json")
SOURCE_GIT_HEAD = "75dfd8103e8e3dfe824a63e15c2222f8742e28c1"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _raw_reports() -> list[dict[str, object]]:
    return [_load(path) for path in sorted((ARTIFACT / "runs").glob("**/report.json"))]


def test_frozen_five_pair_artifact_replays_from_source_commit() -> None:
    result = pairs._replay(ARTIFACT)

    assert result == {
        "manifest_hash": (
            "457ab1adc8488c5353ec66294583e7a2bedf2e92fca5901a72a41e8321df1573"
        ),
        "pair_count": 5,
        "performance_claimed": False,
        "report_hash": (
            "0d649200f423875db23ee23660447f7f0a8b91ce91510f254d7b3bb8f8a2827d"
        ),
        "source_git_head": SOURCE_GIT_HEAD,
        "status": "replay-passed",
        "timing_admitted": False,
        "worker_count": 10,
    }


def test_frozen_five_pair_root_gates_and_schedule() -> None:
    protocol = _load(ARTIFACT / "protocol.json")
    report = _load(ARTIFACT / "report.json")

    assert protocol["source_git_head"] == SOURCE_GIT_HEAD
    assert protocol["pair_count"] == 5
    assert protocol["worker_count"] == 10
    assert protocol["schedule"] == [
        {"pair_index": 0, "positions": ["B2", "B3-C"]},
        {"pair_index": 1, "positions": ["B3-C", "B2"]},
        {"pair_index": 2, "positions": ["B2", "B3-C"]},
        {"pair_index": 3, "positions": ["B3-C", "B2"]},
        {"pair_index": 4, "positions": ["B2", "B3-C"]},
    ]
    for field in (
        "all_counter_gates_passed",
        "all_direct_semantic_pairs_passed",
        "all_environments_admitted",
        "all_post_query_audits_passed",
        "all_provider_fallback_zero",
    ):
        assert report[field] is True
    assert protocol["timing_admitted"] is False
    assert report["timing_admitted"] is False
    assert protocol["performance_claimed"] is False
    assert report["performance_claimed"] is False


def test_frozen_five_pairs_are_direct_semantic_matches() -> None:
    report = _load(ARTIFACT / "report.json")
    rows = report["pairs"]
    assert isinstance(rows, list) and len(rows) == 5

    assert [row["pair_index"] for row in rows] == list(range(5))
    assert [row["schedule"] for row in rows] == [
        ["B2", "B3-C"],
        ["B3-C", "B2"],
        ["B2", "B3-C"],
        ["B3-C", "B2"],
        ["B2", "B3-C"],
    ]
    assert len({row["protocol_identity"] for row in rows}) == 1
    assert len({row["runtime_identity"] for row in rows}) == 1
    assert len({row["source_identity"] for row in rows}) == 1
    assert len({row["gpu_uuid"] for row in rows}) == 1
    for row in rows:
        assert row["semantic_failures"] == []
        assert row["environment_admitted"] is True
        assert row["provider_fallback_zero"] is True
        assert row["b2_counter_gate_passed"] is True
        assert row["b3c_counter_gate_passed"] is True
        assert row["timing_admitted"] is False
        assert row["performance_claimed"] is False


def test_frozen_ten_workers_preserve_physical_counter_distinction() -> None:
    reports = _raw_reports()
    assert len(reports) == 10
    assert sum(report["configuration"] == "B2" for report in reports) == 5
    assert sum(report["configuration"] == "B3-C" for report in reports) == 5

    expected = {
        "B2": {
            "event_count": 4625,
            "template_hit_in_core_count": 0,
            "module_binding_move_in_core_count": 1,
            "scope_construction_count": 2,
            "full_optimizer_step_snapshot_count": 10,
            "forward_trace_build_count": 5,
            "timed_candidate_d2h_copy_count": 12,
        },
        "B3-C": {
            "event_count": 1484,
            "template_hit_in_core_count": 1,
            "module_binding_move_in_core_count": 0,
            "scope_construction_count": 1,
            "full_optimizer_step_snapshot_count": 0,
            "forward_trace_build_count": 4,
            "timed_candidate_d2h_copy_count": 0,
        },
    }
    for report in reports:
        configuration = report["configuration"]
        assert isinstance(configuration, str)
        snapshot = report["snapshot"]
        assert isinstance(snapshot, dict)
        counts = snapshot["counts"]
        assert isinstance(counts, dict)
        assert report["source_git_head"] == SOURCE_GIT_HEAD
        assert report["correctness_passed"] is True
        assert report["environment_passed"] is True
        assert report["fixed_counter_expectations_passed"] is True
        assert report["fsg3_reference_semantic_failures"] == []
        assert report["diagnostic_timing_claimed"] is False
        assert report["performance_claimed"] is False
        assert report["event_count"] == expected[configuration]["event_count"]
        for field, value in expected[configuration].items():
            if field != "event_count":
                assert counts[field] == value
        assert counts["optimizer_evaluation_count"] == 10
        assert counts["optimizer_update_count"] == 9
        assert counts["kfsb_candidate_count"] == 3
        assert counts["kfsb_child_batch_count"] == 3
        assert counts["candidate_snapshot_materialization_count"] == 12
        assert snapshot["provider_core_call_count"] == 0
        assert snapshot["provider_compute_bounds_call_count"] == 0
        assert snapshot["provider_update_bounds_call_count"] == 0
        assert snapshot["fallback_dispatch_count"] == 0


def test_frozen_b3c_audits_and_outer_resigned_attacks() -> None:
    root_report = _load(ARTIFACT / "report.json")
    rows = root_report["pairs"]
    assert isinstance(rows, list)
    for row in rows:
        audit = row["b3c_audit"]
        assert isinstance(audit, dict)
        assert audit["headline_content_digest_count"] == 0
        assert int(audit["post_query_audit_ns"]) > 0
        assert len(str(audit["assembly_hash"])) == 64
        assert len(str(audit["commit_hash"])) == 64
        assert len(str(audit["audit_hash"])) == 64

    tamper = _load(TAMPER_REPORT)
    assert tamper["artifact_manifest_sha256"] == (
        "bf8b3ecccea992cce9dca56c963518510af8dc8d410c0d02b94513160189cb98"
    )
    assert tamper["attack_count"] == 7
    assert tamper["rejected_count"] == 7
    assert tamper["report_hash"] == (
        "52dd43fdbb4de8411c52e31e34006e191e5d3e3cbc57727a0a0f964a0cf32798"
    )
    attack_rows = tamper["rows"]
    assert isinstance(attack_rows, list)
    assert all(isinstance(row, dict) and row["rejected"] is True for row in attack_rows)
    assert tamper["timing_admitted"] is False
    assert tamper["performance_claimed"] is False
