"""Repository replay gates for the frozen MR4 Conv-site census artifact."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

import json
from pathlib import Path

from boundflow.runtime.mr4_production_conv_site_census import OPEN_STATUS
from scripts.run_mr4_production_conv_site_census_formal import replay_artifact

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/measurement-recovery/mr4-production-conv-site-census-v1"


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_mr4_census_artifact_replays() -> None:
    summary = replay_artifact(ARTIFACT)
    assert summary["status"] == OPEN_STATUS
    assert summary["mr5_correctness_preregistration_open"] is True
    assert summary["timing_open"] is False
    assert summary["performance_claimed"] is False


def test_frozen_mr4_census_counts_and_static_ledger_are_exact() -> None:
    summary = _load("summary.json")
    assert summary["run_count"] == 5
    assert summary["row_count"] == 150
    assert summary["site_evaluation_counts"] == {"C0": 50, "C1": 50, "C2": 50}
    assert summary["site_grad_enabled_counts"] == {"C0": 45, "C1": 45, "C2": 45}
    assert summary["site_beta_numel"] == {"C0": 0, "C1": 0, "C2": 0}
    assert summary["site_handoff_content_match_counts"] == {
        "C0": 50,
        "C1": 50,
        "C2": 50,
    }
    assert summary["site_forward_mac_units"] == {
        "C0": 1_327_104,
        "C1": 1_769_472,
        "C2": 884_736,
    }
    assert summary["eligible_total_mac_ratio_to_p"] == 4.5
    assert summary["new_site_mac_ratio_to_p"] == 3.5
    assert (
        summary["candidate_minimum_materialization_bytes_per_outer_call"] == 3_441_360
    )
    assert summary["projected_candidate_forward_launch_count"] == 30
    assert summary["projected_candidate_backward_launch_count"] == 27


def test_frozen_mr4_census_semantics_and_gates_are_exact() -> None:
    summary = _load("summary.json")
    assert (
        summary["global_semantic_maximum_absolute_difference"] == 3.516674041748047e-06
    )
    metrics = summary["semantic_metrics"]
    assert isinstance(metrics, list) and len(metrics) == 4
    assert all(metric["element_count"] == 9540 for metric in metrics)
    assert all(
        metric["allclose"] is True and metric["sign_exact"] is True
        for metric in metrics
    )
    gates = summary["gates"]
    assert isinstance(gates, dict)
    assert all(gates.values())


def test_frozen_mr4_census_tamper_and_path_boundaries_are_exact() -> None:
    tamper = _load("tamper_report.json")
    assert tamper["attack_count"] == 16
    assert tamper["rejected_count"] == 16
    assert tamper["all_rejected"] is True
    for path in ARTIFACT.rglob("*"):
        if path.is_file():
            assert "/home/" not in path.read_text(encoding="utf-8")
