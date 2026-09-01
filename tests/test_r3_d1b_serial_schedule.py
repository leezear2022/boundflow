"""D1-B fixed serial schedule qualification tests."""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path

import pytest

from boundflow.backends.tvm.r3_d1b_serial_schedule import (
    R3D1B_SERIAL_THREADS,
    compile_r3d1b_serial_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "artifacts/r3-structured-owner/r3-d1b-serial-calibration-v1.json"


def test_r3d1b_serial_schedule_space_is_frozen() -> None:
    assert R3D1B_SERIAL_THREADS == (64, 128, 256)
    with pytest.raises(ValueError, match="threads per block differs"):
        compile_r3d1b_serial_candidate(32)


def test_r3d1b_calibration_selects_a_correct_gate_passing_winner() -> None:
    if not CALIBRATION.exists():
        pytest.skip("R3-D1B calibration has not been generated")
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    assert payload["calibration_only"] is True
    assert payload["formal_performance_claimed"] is False
    assert payload["winner_threads_per_block"] in R3D1B_SERIAL_THREADS
    assert payload["winner_gate_pass"] is True
    assert all(row["maximum_diff"] <= 2.0e-4 for row in payload["candidates"])
    assert all(row["sign_exact"] is True for row in payload["candidates"])
