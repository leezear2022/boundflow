"""Frozen B4-A five-fresh correctness artifact gates."""

# pylint: disable=missing-function-docstring,protected-access

import json
from pathlib import Path

from scripts import run_fsg4_b4a_correctness_pairs as pairs

ARTIFACT = Path("artifacts/fsg4-b4a-five-fresh/resnet2b-prop0-v1")


def test_b4a_five_fresh_artifact_replays_from_raw() -> None:
    result = pairs._replay(ARTIFACT)
    assert result["status"] == "replay-passed"
    assert result["pair_count"] == 5
    assert result["worker_count"] == 10
    assert result["maximum_export_absolute_difference"] == 6.109476089477539e-06
    assert result["timing_admitted"] is True
    assert result["performance_claimed"] is False


def test_b4a_five_fresh_artifact_freezes_all_tensor_and_activation_gates() -> None:
    report = json.loads((ARTIFACT / "report.json").read_text(encoding="utf-8"))
    assert report["all_direct_semantic_pairs_passed"] is True
    assert report["all_terminal_export_pairs_passed"] is True
    assert report["all_lineage_and_counter_gates_passed"] is True
    assert [row["export_pair"]["tensor_count"] for row in report["pairs"]] == [19] * 5
    assert all(
        row["b4a_activation"]["terminal_lower_adjoint_handoff_count"] == 1
        and row["b4a_activation"]["terminal_export_crown_rerun_count"] == 0
        and row["b4a_activation"]["lineage_count"] == 6
        and row["b4a_activation"]["provider_callback_count"] == 0
        and row["b4a_activation"]["fallback_dispatch_count"] == 0
        for row in report["pairs"]
    )
