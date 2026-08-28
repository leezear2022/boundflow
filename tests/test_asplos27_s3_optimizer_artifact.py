"""Replay and tamper gates for the ASPLOS'27 S3 artifact."""

# pylint: disable=missing-function-docstring,import-outside-toplevel

from pathlib import Path

import pytest

ARTIFACT = Path("artifacts/asplos27-s3-optimizer/resnet2b-p-anchor-v1")


def test_s3_v1_artifact_replays_as_preserved_environmental_no_go() -> None:
    from scripts import run_asplos27_s3_optimizer_artifact as artifact

    if not ARTIFACT.is_dir():
        pytest.skip("S3 formal artifact unavailable")
    result = artifact.replay(ARTIFACT, require_validated_3x=False)
    summary = artifact.load_json(ARTIFACT / "summary.json")
    assert result["status"] == "replay-passed"
    assert summary["status"] == "validated-no-go-s3-local-optimizer"
    assert summary["run_count"] == 6
    assert summary["orders"] == ["NDP", "NPD", "DNP", "DPN", "PND", "PDN"]
    assert summary["p_over_n_geomean"] == pytest.approx(2.569574644743942)
    assert summary["p_over_n_worst"] == pytest.approx(0.7595404807250521)
    assert summary["p_over_d_geomean"] >= 1.50
    assert summary["max_step_abs_diff"]["lower"] <= 2e-4
    for field in (
        "alpha_before",
        "gradient",
        "alpha_after",
        "optimizer_exp_avg",
        "optimizer_exp_avg_sq",
    ):
        assert summary["max_step_abs_diff"][field] <= 2e-5
    assert summary["lower_sign_exact"] is True
    assert summary["gradient_sign_exact"] is True
    assert summary["validated_s3_3x"] is False
    assert summary["same_solver_gate_open"] is False
    assert summary["same_solver_claimed"] is False
    assert summary["complete_query_claimed"] is False
    assert summary["tenx_claimed"] is False
    assert summary["performance_claimed"] is False


def test_s3_formal_artifact_rejects_ten_outer_resigned_attacks() -> None:
    if not ARTIFACT.is_dir():
        pytest.skip("S3 formal artifact unavailable")
    report = __import__("json").loads(
        (ARTIFACT / "tamper_report.json").read_text(encoding="utf-8")
    )
    assert report["case_count"] == report["rejected_count"] == 10
    assert all(row["outer_resigned"] is True for row in report["rows"])
    assert all(row["rejected"] is True for row in report["rows"])
    assert report["performance_claimed"] is False
