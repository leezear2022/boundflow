"""R3-1b1 artifact replay and fully re-signed tamper tests."""

# pylint: disable=missing-function-docstring

from pathlib import Path

import pytest

from scripts.probe_r3_full_lower_forward_tamper import probe
from scripts.run_r3_full_lower_forward_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-1b1-full-lower-v1"


def test_r31b1_formal_artifact_replays() -> None:
    if not ARTIFACT.is_dir():
        pytest.skip("R3-1b1 formal artifact is not generated yet")
    result = replay(ARTIFACT)

    assert result["status"] == "replay-passed"
    assert result["evidence_status"] == "validated-r3-1b1-compiled-full-lower"
    assert result["coefficient_scratch_count"] == 2
    assert result["warm_dynamic_allocated_bytes"] == 0
    assert result["compiled_region"] is True
    assert result["b2_open"] is True
    assert result["performance_claimed"] is False


def test_r31b1_formal_artifact_rejects_fully_resigned_tampering() -> None:
    if not ARTIFACT.is_dir():
        pytest.skip("R3-1b1 formal artifact is not generated yet")
    result = probe(ARTIFACT)

    assert result["attack_count"] == 10
    assert result["rejected_count"] == 10
