"""R3-1b2 artifact replay and fully re-signed tamper tests."""

# pylint: disable=missing-function-docstring

from pathlib import Path

import pytest

from scripts.probe_r3_compiled_p_alpha_vjp_tamper import probe
from scripts.run_r3_compiled_p_alpha_vjp_artifact import replay

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/r3-structured-owner/r3-1b2-compiled-p-alpha-vjp-v1"


def test_r31b2_formal_artifact_replays() -> None:
    if not ARTIFACT.is_dir():
        pytest.skip("R3-1b2 formal artifact is not generated yet")
    result = replay(ARTIFACT)

    assert result["status"] == "replay-passed"
    assert result["evidence_status"] == "validated-r3-1b2-compiled-p-alpha-vjp"
    assert result["gradient_sign_exact"] is True
    assert result["gradient_nonzero"] == 281
    assert result["coefficient_scratch_count"] == 2
    assert result["saved_dense_a_count"] == 0
    assert result["warm_dynamic_allocated_bytes"] == 0
    assert result["compiled_vjp"] is True
    assert result["custom_vjp"] is True
    assert result["b3_open"] is True
    assert result["performance_claimed"] is False


def test_r31b2_formal_artifact_rejects_fully_resigned_tampering() -> None:
    if not ARTIFACT.is_dir():
        pytest.skip("R3-1b2 formal artifact is not generated yet")
    result = probe(ARTIFACT)

    assert result["attack_count"] == 12
    assert result["rejected_count"] == 12
