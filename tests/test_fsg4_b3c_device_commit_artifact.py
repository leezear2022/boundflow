"""Frozen evidence checks for the FSG4/B3-C device commit artifact."""

# pylint: disable=missing-function-docstring,protected-access

import json
from pathlib import Path

from scripts import run_fsg4_b3_counter_diagnostic as diagnostic

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3c-v1"
TAMPER_REPORT = ARTIFACT.with_name("resnet2b-prop0-b3c-v1-tamper-report.json")


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_b3c_artifact_replays_from_source_commit() -> None:
    result = diagnostic._replay(ARTIFACT)
    assert result["status"] == "replay-passed"
    assert result["source_git_head"] == ("72bec5ee1bdabfdefbf51201ac49395489eeef65")
    assert result["manifest_hash"] == (
        "091f6ac8f57ba3de642e5ae0f390bc8b005904b64d6eea02526dd93bb166e1c2"
    )
    assert result["report_hash"] == (
        "72812a3565e33e0103ce734090c728b535ca32fefa8aa4b928b0b09d93e5cb44"
    )
    assert result["event_count"] == 1484


def test_frozen_b3c_device_commit_physical_counts() -> None:
    report = _load(ARTIFACT / "report.json")
    snapshot = report["snapshot"]
    assert isinstance(snapshot, dict)
    assert report["configuration"] == "B3-C"
    assert snapshot["configuration"] == "B3-C"
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
    assert counts["candidate_snapshot_materialization_count"] == 12
    assert counts["timed_candidate_d2h_copy_count"] == 0
    assert counts["committed_mutable_path_count"] == 12
    assert counts["device_rollback_backup_count"] == 12
    assert counts["commit_copy_call_count"] == 12
    assert counts["gpu_tensor_content_hash_count"] == 24
    assert report["fsg3_reference_semantic_failures"] == []
    assert report["correctness_passed"] is True
    assert report["diagnostic_timing_claimed"] is False
    assert report["performance_claimed"] is False


def test_frozen_b3c_post_query_audit_is_outside_headline() -> None:
    worker = _load(ARTIFACT / "worker.json")
    diagnostics = worker["diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["post_query_audit_excluded_from_timing"] is True
    assert int(diagnostics["post_query_audit_ns"]) > 0
    assemblies = diagnostics["assembly_metadata"]
    receipts = diagnostics["commit_receipts"]
    audits = diagnostics["device_commit_audits"]
    assert isinstance(assemblies, list) and len(assemblies) == 1
    assert isinstance(receipts, list) and len(receipts) == 1
    assert isinstance(audits, list) and len(audits) == 1
    assembly = assemblies[0]
    receipt = receipts[0]
    audit = audits[0]
    assert isinstance(assembly, dict)
    assert isinstance(receipt, dict)
    assert isinstance(audit, dict)
    assert assembly["headline_content_digest_count"] == 0
    assert assembly["candidate_device_resident"] is True
    assert receipt["candidate_d2h_copy_count"] == 0
    assert receipt["committed_path_count"] == 12
    assert receipt["device_rollback_backup_count"] == 12
    assert audit["commit_hash"] == receipt["commit_hash"]
    assert audit["headline_timing_excluded"] is True
    assert audit["content_audit_complete"] is True
    assert len(audit["path_digests"]) == 12
    assert audit["audit_hash"] == (
        "b0a978ae16fc386e17795bbcc79d3ff97a1e9fdf44c5742a6070d792acdd0ac8"
    )


def test_frozen_b3c_provider_fallback_and_outer_resigned_attacks() -> None:
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
        "af772e09d570e504ea54177805473e5d901d57fe3c72fb372623b85519045ce1"
    )
    rows = tamper["rows"]
    assert isinstance(rows, list)
    assert all(isinstance(row, dict) and row["rejected"] is True for row in rows)
