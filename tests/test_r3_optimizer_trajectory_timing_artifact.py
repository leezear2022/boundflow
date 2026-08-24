"""R3-2B formal wrapper timing replay gate."""

# pylint: disable=missing-function-docstring,duplicate-code

from pathlib import Path

import pytest

from scripts.run_r3_optimizer_trajectory_timing_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-2b-wrapper-timing-v1"


def test_r32b_formal_artifact_replays() -> None:
    if not ARTIFACT.is_dir():
        pytest.skip("R3-2B formal artifact is not generated yet")
    summary = replay(ARTIFACT)
    assert summary["pair_count"] == 5
    assert summary["worker_count"] == 10
