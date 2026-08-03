"""IR-5 measured evaluation contracts."""

# pylint: disable=duplicate-code,missing-function-docstring

from __future__ import annotations

from pathlib import Path

import pytest

from boundflow.ir.plan import BackendKind
from boundflow.planner.measured_adaptive_benchmark import (
    TypedCNNWorkloadSpec,
    TypedWorkloadSpec,
    build_heldout_observations,
    fit_backend_calibrations,
    measure_workload,
)


def test_measured_candidates_are_typed_correct_and_calibration_is_frozen(
    tmp_path: Path,
) -> None:
    calibration_spec = TypedWorkloadSpec(
        "calibration-contract", "calibration", 2, 4, 5, 3, 91
    )
    calibration = measure_workload(
        calibration_spec,
        (BackendKind.REFERENCE, BackendKind.PYTORCH_DENSE),
        device="cpu",
        warm_samples=2,
        cache_root=tmp_path / "calibration",
    )
    assert all(item.semantic_allclose for item in calibration)
    assert all(item.measured_peak_bytes > 0 for item in calibration)
    assert calibration[0].plan_instance_hash != calibration[1].plan_instance_hash
    models = fit_backend_calibrations(calibration)

    heldout_spec = TypedWorkloadSpec("heldout-contract", "heldout", 2, 5, 6, 4, 92)
    heldout = measure_workload(
        heldout_spec,
        (BackendKind.REFERENCE, BackendKind.PYTORCH_DENSE),
        device="cpu",
        warm_samples=2,
        cache_root=tmp_path / "heldout",
    )
    observations = build_heldout_observations(heldout, models)
    assert {item.plan_id for item in observations} == {
        "reference",
        "pytorch_dense",
    }
    assert all(item.predicted_latency_ms > 0 for item in observations)

    with pytest.raises(ValueError, match="leaked"):
        fit_backend_calibrations(heldout)


def test_cnn_heldout_uses_mlp_calibration_without_family_leakage(
    tmp_path: Path,
) -> None:
    calibration = measure_workload(
        TypedWorkloadSpec("mlp-family", "calibration", 1, 6, 8, 3, 93),
        (BackendKind.REFERENCE, BackendKind.PYTORCH_DENSE),
        device="cpu",
        warm_samples=1,
        cache_root=tmp_path / "mlp",
    )
    models = fit_backend_calibrations(calibration)
    cnn_spec = TypedCNNWorkloadSpec(
        "cnn-family",
        "heldout",
        1,
        1,
        4,
        2,
        3,
        2,
        94,
    )
    heldout = measure_workload(
        cnn_spec,
        (BackendKind.REFERENCE, BackendKind.PYTORCH_DENSE),
        device="cpu",
        warm_samples=1,
        cache_root=tmp_path / "cnn",
    )
    observations = build_heldout_observations(heldout, models)

    assert cnn_spec.to_dict()["family"] == "chain_cnn"
    assert cnn_spec.work_units > 0
    assert all(item.semantic_allclose for item in heldout)
    assert all(item.predicted_latency_ms > 0 for item in observations)
