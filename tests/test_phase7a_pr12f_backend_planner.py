"""Calibration/held-out contract tests for the PR-12F backend planner."""

import pytest

from boundflow.planner.execution_candidate import BackendVariant, OperatorFamily
from boundflow.planner.fused_crown_backend import (
    FusedCrownBackendObservation,
    FusedCrownBackendPlanner,
)


def _observation(
    case_id: str,
    family: OperatorFamily,
    backend: BackendVariant,
    *,
    latency: float,
    peak: int,
    boundary: int = 1_000,
    correct: bool = True,
    eligible: bool = True,
) -> FusedCrownBackendObservation:
    return FusedCrownBackendObservation(
        case_id=case_id,
        family=family,
        backend=backend,
        boundary_bytes=boundary,
        region_count=1,
        warm_latency_ms=latency,
        peak_allocated_bytes=peak,
        eligible=eligible,
        correct=correct,
    )


def _planner() -> FusedCrownBackendPlanner:
    return FusedCrownBackendPlanner.fit(
        [
            _observation(
                "linear-small",
                OperatorFamily.LINEAR,
                BackendVariant.PYTORCH_EAGER,
                latency=2.0,
                peak=1_000,
            ),
            _observation(
                "linear-small",
                OperatorFamily.LINEAR,
                BackendVariant.TVM_FUSED_TIR,
                latency=1.0,
                peak=600,
            ),
            _observation(
                "conv-small",
                OperatorFamily.CONV2D,
                BackendVariant.PYTORCH_EAGER,
                latency=1.0,
                peak=800,
            ),
            _observation(
                "conv-small",
                OperatorFamily.CONV2D,
                BackendVariant.TVM_FUSED_TIR,
                latency=1.5,
                peak=500,
            ),
        ]
    )


def test_backend_planner_selects_distinct_family_decisions() -> None:
    planner = _planner()

    linear = planner.decide(
        family=OperatorFamily.LINEAR,
        boundary_bytes=1_200,
        region_count=1,
        budget_bytes=10_000,
        eligible=True,
    )
    conv = planner.decide(
        family=OperatorFamily.CONV2D,
        boundary_bytes=1_200,
        region_count=1,
        budget_bytes=10_000,
        eligible=True,
    )

    assert linear.backend == BackendVariant.TVM_FUSED_TIR
    assert linear.reason == "calibration_predicts_fused_faster"
    assert conv.backend == BackendVariant.PYTORCH_EAGER
    assert conv.reason == "calibration_predicts_eager_faster"


def test_backend_planner_uses_memory_budget_and_capability_fallback() -> None:
    planner = _planner()

    memory_rescue = planner.decide(
        family=OperatorFamily.CONV2D,
        boundary_bytes=1_000,
        region_count=1,
        budget_bytes=700,
        eligible=True,
    )
    fallback = planner.decide(
        family=OperatorFamily.LINEAR,
        boundary_bytes=1_000,
        region_count=1,
        budget_bytes=10_000,
        eligible=False,
    )

    assert memory_rescue.backend == BackendVariant.TVM_FUSED_TIR
    assert memory_rescue.reason == "fused_only_budget_feasible"
    assert fallback.backend == BackendVariant.PYTORCH_EAGER
    assert fallback.reason == "capability_or_graph_ineligible_fallback"
    assert fallback.calibration_case_id is None


def test_backend_planner_rejects_unpaired_or_mismatched_calibration() -> None:
    with pytest.raises(ValueError, match="at least one paired"):
        FusedCrownBackendPlanner.fit(
            [
                _observation(
                    "only-eager",
                    OperatorFamily.LINEAR,
                    BackendVariant.PYTORCH_EAGER,
                    latency=1.0,
                    peak=1,
                )
            ]
        )

    eager = _observation(
        "mismatch",
        OperatorFamily.LINEAR,
        BackendVariant.PYTORCH_EAGER,
        latency=1.0,
        peak=1,
        boundary=1_000,
    )
    fused = _observation(
        "mismatch",
        OperatorFamily.LINEAR,
        BackendVariant.TVM_FUSED_TIR,
        latency=1.0,
        peak=1,
        boundary=2_000,
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        FusedCrownBackendPlanner.fit([eager, fused])


def test_backend_decision_dump_is_json_compatible() -> None:
    decision = _planner().decide(
        family=OperatorFamily.LINEAR,
        boundary_bytes=1_000,
        region_count=1,
        budget_bytes=10_000,
        eligible=True,
    )

    payload = decision.to_dict()
    assert payload["backend"] == "tvm_fused_tir"
    assert payload["use_fused"] is True
    assert _planner().to_dict()["model"] == (
        "same_family_nearest_log_bytes_per_region_v1"
    )
