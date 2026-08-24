"""R3-2A formal artifact replay gate."""

# pylint: disable=missing-function-docstring,duplicate-code

from pathlib import Path

import pytest

from scripts.run_r3_optimizer_trajectory_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-2a-optimizer-trajectory-v1"


def test_r32a_formal_artifact_replays() -> None:
    if not ARTIFACT.is_dir():
        pytest.skip("R3-2A formal artifact is not generated yet")
    summary = replay(ARTIFACT)
    assert summary["trajectory_correctness_admitted"] is True
    assert summary["r3_2b_open"] is True
    assert summary["performance_claimed"] is False
