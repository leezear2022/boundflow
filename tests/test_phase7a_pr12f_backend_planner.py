"""Calibration/held-out contract tests for the PR-12F backend planner."""

import pytest

from boundflow.planner.execution_candidate import BackendVariant, OperatorFamily
from boundflow.planner.fused_crown_backend import (
    FusedCrownBackendObservation,
    FusedCrownBackendPlanner,
    FusedCrownMultiBackendPlanner,
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


def test_multibackend_planner_selects_chunked_memory_rescue_and_eager_conv() -> None:
    observations: list[FusedCrownBackendObservation] = []
    for case_id, family, rows in (
        (
            "linear-large",
            OperatorFamily.LINEAR,
            (
                (BackendVariant.PYTORCH_EAGER, 1.0, 1000),
                (BackendVariant.PYTORCH_CHUNKED, 1.2, 600),
                (BackendVariant.TVM_FUSED_TIR, 4.0, 500),
            ),
        ),
        (
            "conv",
            OperatorFamily.CONV2D,
            (
                (BackendVariant.PYTORCH_EAGER, 1.0, 1000),
                (BackendVariant.PYTORCH_CHUNKED, 1.3, 800),
                (BackendVariant.TVM_FUSED_TIR, 1.5, 500),
            ),
        ),
    ):
        observations.extend(
            _observation(
                case_id,
                family,
                backend,
                latency=latency,
                peak=peak,
            )
            for backend, latency, peak in rows
        )
    planner = FusedCrownMultiBackendPlanner.fit(observations)

    linear = planner.decide(
        family=OperatorFamily.LINEAR,
        boundary_bytes=1_000,
        region_count=1,
        budget_bytes=700,
        eligible_backends=tuple(BackendVariant),
    )
    conv = planner.decide(
        family=OperatorFamily.CONV2D,
        boundary_bytes=1_000,
        region_count=1,
        budget_bytes=2_000,
        eligible_backends=tuple(BackendVariant),
    )

    assert linear.backend == BackendVariant.PYTORCH_CHUNKED
    assert linear.predicted_peak_bytes == 600
    assert conv.backend == BackendVariant.PYTORCH_EAGER
    assert str(planner.to_dict()["model"]).endswith("multibackend_v2")


def test_multibackend_planner_never_selects_ineligible_backend() -> None:
    planner = FusedCrownMultiBackendPlanner.fit(
        [
            _observation(
                "linear",
                OperatorFamily.LINEAR,
                BackendVariant.PYTORCH_EAGER,
                latency=1.0,
                peak=1000,
            ),
            _observation(
                "linear",
                OperatorFamily.LINEAR,
                BackendVariant.PYTORCH_CHUNKED,
                latency=0.5,
                peak=500,
            ),
        ]
    )

    decision = planner.decide(
        family=OperatorFamily.LINEAR,
        boundary_bytes=1_000,
        region_count=1,
        budget_bytes=2_000,
        eligible_backends=(BackendVariant.PYTORCH_EAGER,),
    )

    assert decision.backend == BackendVariant.PYTORCH_EAGER
    assert set(decision.predictions) == {"pytorch_eager"}
