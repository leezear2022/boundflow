"""Mechanical unit gates for MR6 formal guard attribution."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import pytest

from boundflow.runtime.mr3_provider_hook_feasibility import canonical_hash
from boundflow.runtime.mr6_guard_attribution import (
    RAW_SCHEMA,
    SOURCE_COMMIT,
    derive_summary,
    validate_guard_receipt,
)
from scripts.run_mr6_guard_attribution_worker import _GuardLedger


@pytest.mark.parametrize(
    ("mode", "site_evaluations", "expected"),
    (("provider", 0, 0), ("full", 30, 360), ("diagnostic", 30, 60)),
)
def test_mr6_formal_guard_receipts_accept_only_frozen_counts(
    mode: str, site_evaluations: int, expected: int
) -> None:
    ledger = _GuardLedger(mode)
    if mode != "provider":
        ledger.validation_calls = 30
        ledger.content_calls = 30
    receipt = ledger.to_dict(site_evaluations=site_evaluations)
    assert receipt["synchronizing_guards_executed"] == expected
    validate_guard_receipt(receipt, mode=mode)


def test_mr6_formal_guard_receipt_rejects_fully_resigned_count() -> None:
    ledger = _GuardLedger("diagnostic")
    ledger.validation_calls = 30
    ledger.content_calls = 30
    receipt = ledger.to_dict(site_evaluations=30)
    receipt["synchronizing_guards_executed"] = 0
    receipt.pop("receipt_hash")
    receipt["receipt_hash"] = canonical_hash(receipt)
    with pytest.raises(ValueError, match="guard receipt differs"):
        validate_guard_receipt(receipt, mode="diagnostic")


def test_mr6_formal_rejects_empty_fully_resigned_raw() -> None:
    raw: dict[str, object] = {
        "schema_version": RAW_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "run_order": [],
        "runs": [],
    }
    raw["raw_hash"] = canonical_hash(raw)
    with pytest.raises(ValueError, match="raw provenance differs"):
        derive_summary(raw)
