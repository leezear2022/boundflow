"""Contract tests for the PR-12 calibration-only sanity benchmark."""

from scripts.benchmark_phase7a_pr12_fused_sanity import _percentile, _summary


def test_sanity_summary_uses_median_and_nearest_rank_p90() -> None:
    samples = [1.0, 2.0, 3.0, 4.0, 20.0]

    assert _percentile(samples, 0.9) == 20.0
    assert _summary(samples) == {
        "median_ms": 3.0,
        "p90_ms": 20.0,
        "min_ms": 1.0,
        "max_ms": 20.0,
    }
