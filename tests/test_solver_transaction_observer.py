"""Tests for low-perturbation external solver transaction markers."""

# pylint: disable=missing-function-docstring

from types import SimpleNamespace
from typing import cast

import pytest

from boundflow.runtime.solver_transaction_observer import (
    SolverTransactionObserver,
    TransactionCategory,
    TransactionResolution,
    TransactionTarget,
    host_transaction_span_from_dict,
    summarize_solver_transactions,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000

    def __call__(self) -> int:
        return self.now

    def advance(self, duration: int) -> None:
        self.now += duration


def test_observer_records_nested_spans_and_restores_targets() -> None:
    clock = _Clock()
    owner = SimpleNamespace()

    def exact() -> str:
        clock.advance(50)
        return "ok"

    def coarse() -> str:
        clock.advance(10)
        value = owner.exact()
        clock.advance(40)
        return value

    owner.exact = exact
    owner.coarse = coarse
    original_exact = owner.exact
    original_coarse = owner.coarse
    observer = SolverTransactionObserver(scope_started_ns=1000, clock_ns=clock)
    targets = (
        TransactionTarget(
            owner,
            "coarse",
            "fake.coarse",
            TransactionCategory.VERIFY_SCOPE,
            TransactionResolution.COARSE_SCOPE,
        ),
        TransactionTarget(
            owner,
            "exact",
            "fake.exact",
            TransactionCategory.BOUND_CORE,
            TransactionResolution.EXACT_TRANSACTION,
        ),
    )

    with observer.instrument(targets):
        assert owner.coarse() == "ok"

    assert owner.exact is original_exact
    assert owner.coarse is original_coarse
    spans = observer.finish(scope_ns=100)
    assert len(spans) == 2
    coarse_span = next(span for span in spans if span.target_id == "fake.coarse")
    exact_span = next(span for span in spans if span.target_id == "fake.exact")
    assert coarse_span.host_start_ns == 0
    assert coarse_span.host_end_ns == 100
    assert exact_span.parent_transaction_id == coarse_span.transaction_id
    assert exact_span.host_start_ns == 10
    assert exact_span.host_end_ns == 60


def test_exclusive_summary_places_bound_call_below_exact_marker() -> None:
    clock = _Clock()
    owner = SimpleNamespace()

    def exact() -> None:
        clock.advance(50)

    def coarse() -> None:
        clock.advance(10)
        owner.exact()
        clock.advance(40)

    owner.exact = exact
    owner.coarse = coarse
    observer = SolverTransactionObserver(scope_started_ns=1000, clock_ns=clock)
    with observer.instrument(
        (
            TransactionTarget(
                owner,
                "coarse",
                "fake.coarse",
                TransactionCategory.VERIFY_SCOPE,
                TransactionResolution.COARSE_SCOPE,
            ),
            TransactionTarget(
                owner,
                "exact",
                "fake.exact",
                TransactionCategory.BOUND_CORE,
                TransactionResolution.EXACT_TRANSACTION,
            ),
        )
    ):
        owner.coarse()
    spans = observer.finish(scope_ns=100)
    summary = summarize_solver_transactions(
        spans,
        scope_ns=100,
        compute_calls=(
            {
                "call_id": 0,
                "depth": 0,
                "host_start_ns": 20,
                "host_end_ns": 40,
                "phase": "beta_split",
                "external_phase": "activation_bab_bound",
            },
        ),
    )

    category_ns = cast(dict[str, int], summary["category_ns"])
    assert category_ns["bound_core"] == 30
    assert category_ns["bound_compute:beta_split:activation_bab_bound"] == 20
    assert category_ns["mechanism_unresolved:verify_scope"] == 50
    assert summary["mechanism_coverage_share"] == pytest.approx(0.5)
    assert summary["mechanism_admitted"] is False


def test_exact_outer_transaction_can_close_entire_scope() -> None:
    clock = _Clock()
    owner = SimpleNamespace()

    def setup() -> None:
        clock.advance(100)

    owner.setup = setup
    observer = SolverTransactionObserver(scope_started_ns=1000, clock_ns=clock)
    with observer.instrument(
        (
            TransactionTarget(
                owner,
                "setup",
                "fake.setup",
                TransactionCategory.FRONTEND_SETUP,
                TransactionResolution.EXACT_TRANSACTION,
            ),
        )
    ):
        owner.setup()
    summary = summarize_solver_transactions(
        observer.finish(scope_ns=100), compute_calls=(), scope_ns=100
    )

    assert summary["mechanism_coverage_share"] == 1.0
    assert summary["mechanism_admitted"] is True


def test_observer_records_raised_outcome_and_restores() -> None:
    clock = _Clock()
    owner = SimpleNamespace()

    def fail() -> None:
        clock.advance(5)
        raise RuntimeError("expected")

    owner.fail = fail
    original = owner.fail
    observer = SolverTransactionObserver(scope_started_ns=1000, clock_ns=clock)
    with pytest.raises(RuntimeError, match="expected"):
        with observer.instrument(
            (
                TransactionTarget(
                    owner,
                    "fail",
                    "fake.fail",
                    TransactionCategory.BOUND_CORE,
                    TransactionResolution.EXACT_TRANSACTION,
                ),
            )
        ):
            owner.fail()

    assert owner.fail is original
    spans = observer.finish(scope_ns=5)
    assert spans[0].outcome == "raised"


def test_observer_rejects_duplicate_patch_location() -> None:
    owner = SimpleNamespace(call=lambda: None)
    observer = SolverTransactionObserver(scope_started_ns=0, clock_ns=lambda: 0)
    target = TransactionTarget(
        owner,
        "call",
        "fake.call",
        TransactionCategory.BOUND_CORE,
        TransactionResolution.EXACT_TRANSACTION,
    )

    with pytest.raises(ValueError, match="targets duplicate"):
        with observer.instrument((target, target)):
            pass


def test_transaction_span_round_trip_rejects_extra_field() -> None:
    clock = _Clock()
    owner = SimpleNamespace(call=lambda: clock.advance(1))
    observer = SolverTransactionObserver(scope_started_ns=1000, clock_ns=clock)
    with observer.instrument(
        (
            TransactionTarget(
                owner,
                "call",
                "fake.call",
                TransactionCategory.BOUND_CORE,
                TransactionResolution.EXACT_TRANSACTION,
            ),
        )
    ):
        owner.call()
    row = observer.finish(scope_ns=1)[0].to_dict()

    assert host_transaction_span_from_dict(row).to_dict() == row
    row["extra"] = True
    with pytest.raises(ValueError, match="fields differ"):
        host_transaction_span_from_dict(row)
