"""Contract tests for B4-C2 materialization-frontier ownership."""

# pylint: disable=missing-function-docstring

import pytest

from boundflow.runtime.fsg4_b4c2_materialization_frontier import (
    B4C2MaterializationFrontierObserverV1,
    B4C2MaterializationFrontierReceiptV1,
    B4C2_RELU_FRONTIERS,
)


def test_materialization_frontier_receipt_accepts_complete_optimizer() -> None:
    receipt = B4C2MaterializationFrontierReceiptV1(
        evaluation_count=10,
        materialization_count=60,
        per_frontier_count={name: 10 for name in B4C2_RELU_FRONTIERS},
        fallback_count=0,
        eager_recompute_count=0,
        exact_call=True,
    )
    receipt.validate()


def test_materialization_frontier_receipt_rejects_missing_site() -> None:
    receipt = B4C2MaterializationFrontierReceiptV1(
        evaluation_count=10,
        materialization_count=59,
        per_frontier_count={
            name: 9 if name == "25" else 10 for name in B4C2_RELU_FRONTIERS
        },
        fallback_count=0,
        eager_recompute_count=0,
        exact_call=True,
    )
    with pytest.raises(ValueError, match="receipt differs"):
        receipt.validate()


def test_materialization_frontier_observer_rejects_order_drift() -> None:
    observer = B4C2MaterializationFrontierObserverV1()
    observer.begin_evaluation(
        0, native_alphas={}, native_betas={}, relu_pre_add_coeff_l={}
    )
    with pytest.raises(ValueError, match="order differs"):
        observer.record_relu_lower_materialization("25")
