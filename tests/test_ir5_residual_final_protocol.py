"""Frozen IR-5 residual final protocol contracts."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_ir5_family_fair_artifact import (
    RESIDUAL_FINAL_ARTIFACT_SCHEMA,
    RESIDUAL_FINAL_V2,
    generate_artifact,
)


def test_residual_final_v2_freezes_disjoint_families_and_fresh_identities() -> None:
    assert RESIDUAL_FINAL_V2.schema_version == RESIDUAL_FINAL_ARTIFACT_SCHEMA
    assert RESIDUAL_FINAL_V2.calibration_family == "chain_cnn"
    assert RESIDUAL_FINAL_V2.heldout_family == "residual_cnn"
    assert RESIDUAL_FINAL_V2.from_forward_trace
    assert RESIDUAL_FINAL_V2.baseline_plan_id == "batched-original-from-forward-trace"
    calibration = tuple(item.to_dict() for item in RESIDUAL_FINAL_V2.calibration)
    heldout = tuple(item.to_dict() for item in RESIDUAL_FINAL_V2.heldout)
    assert all(item["family"] == "chain_cnn" for item in calibration)
    assert all(item["split"] == "calibration" for item in calibration)
    assert all(item["family"] == "residual_cnn" for item in heldout)
    assert all(item["split"] == "heldout" for item in heldout)
    identities = {
        (str(item["workload_id"]), int(item["seed"]))
        for item in (*calibration, *heldout)
    }
    assert len(identities) == len(calibration) + len(heldout)


def test_residual_final_v2_rejects_cpu_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    with pytest.raises(ValueError, match="CUDA-only"):
        generate_artifact(
            output,
            device="cpu",
            warm_samples=1,
            suite=RESIDUAL_FINAL_V2,
        )
    assert not output.exists()
