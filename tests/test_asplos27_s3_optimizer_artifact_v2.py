"""Replay and tamper gates for the robust ASPLOS'27 S3 v2 artifact."""

# pylint: disable=missing-function-docstring,import-outside-toplevel

from pathlib import Path

import pytest

ARTIFACT = Path("artifacts/asplos27-s3-optimizer/resnet2b-p-anchor-v2")


def test_s3_v2_artifact_replays_and_passes_robust_3x_gate() -> None:
    from scripts import run_asplos27_s3_optimizer_artifact_v2 as artifact

    if not ARTIFACT.is_dir():
        pytest.skip("S3 v2 formal artifact unavailable")
    result = artifact.replay(ARTIFACT)
    summary = __import__("json").loads(
        (ARTIFACT / "summary.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "replay-passed"
    assert summary["status"] == "validated-s3-3x-local-optimizer-v2"
    assert summary["run_count"] == 18
    assert summary["replicate_count_per_order"] == 3
    assert summary["p_over_n_geomean"] >= 3.00
    assert summary["p_over_n_worst"] >= 2.50
    assert summary["p_over_d_geomean"] >= 1.50
    assert set(summary["order_median_p_over_n"]) == set(artifact.ORDERS)
    assert all(len(values) == 3 for values in summary["raw_p_over_n_by_order"].values())
    assert summary["max_step_abs_diff"]["lower"] <= 2e-4
    assert summary["max_step_abs_diff"]["gradient"] <= 2e-5
    assert summary["lower_sign_exact"] is True
    assert summary["gradient_sign_exact"] is True
    assert summary["v1_no_go_preserved"] is True
    assert summary["performance_claimed"] is False


def test_s3_v2_artifact_rejects_ten_outer_resigned_attacks() -> None:
    if not ARTIFACT.is_dir():
        pytest.skip("S3 v2 formal artifact unavailable")
    report = __import__("json").loads(
        (ARTIFACT / "tamper_report.json").read_text(encoding="utf-8")
    )
    assert report["case_count"] == report["rejected_count"] == 10
    assert all(row["outer_resigned"] is True for row in report["rows"])
    assert all(row["rejected"] is True for row in report["rows"])
    assert report["performance_claimed"] is False
