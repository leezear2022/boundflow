"""Tests for the MR2 production CROWN owner inventory."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boundflow.runtime.mr2_production_crown_owner_inventory import (
    canonical_hash,
    derive_site_ledger,
    derive_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _inputs() -> dict[str, dict[str, object]]:
    return {
        "inventory": _json(
            "artifacts/fsg2-rvir-v3/resnet2b-production-state-inventory-v2/inventory.json"
        ),
        "p_bundle": _json("artifacts/r3-structured-owner/r3-0-contract-v2/bundle.json"),
        "p_trajectory": _json(
            "artifacts/r3-structured-owner/r3-d2b-correctness-v1/summary.json"
        ),
        "s_correctness": _json(
            "artifacts/r3-structured-owner/r3-3-active-beta-correctness-v1/summary.json"
        ),
        "p_cibc": _json(
            "artifacts/fsg4-b4b2-v2-cibc-formal/resnet2b-prop0-v1/summary.json"
        ),
        "p_v1": _json(
            "artifacts/fsg4-b4b2-b2-5-formal-microphysics/resnet2b-prop0-v1/summary.json"
        ),
        "mr1": _json(
            "artifacts/measurement-recovery/mr1-static-same-solver-eligibility-v1/summary.json"
        ),
    }


def test_canonical_hash_is_order_independent() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_p_anchor_is_the_only_bridge_ready_site() -> None:
    ledger = derive_site_ledger(_inputs())
    assert [row["site_id"] for row in ledger] == ["P:25/Conv_8", "S:31/Gemm_14"]
    assert [row["ready_for_bridge_correctness"] for row in ledger] == [True, False]
    assert ledger[0]["missing_gates"] == ["production_exact_call_connection"]


def test_s_anchor_preserves_three_missing_boundaries() -> None:
    row = derive_site_ledger(_inputs())[1]
    assert row["missing_gates"] == [
        "production_site_identity",
        "optimizer_trajectory_correctness",
        "multi_site_consumer_closure",
        "production_exact_call_connection",
    ]


def test_summary_opens_only_p_anchor_bridge_preregistration() -> None:
    summary = derive_summary(derive_site_ledger(_inputs()))
    assert summary["selected_site"] == "P:25/Conv_8"
    assert summary["bridge_correctness_preregistration_open"] is True
    assert summary["bridge_implemented"] is False
    assert summary["timing_open"] is False


def test_frozen_beta_shape_tamper_fails_closed() -> None:
    inputs = _inputs()
    beta = next(
        item
        for item in inputs["p_bundle"]["instance"]["bindings"]
        if item["name"] == "beta"
    )
    beta["shape"] = [6, 1]
    with pytest.raises(ValueError, match="frozen evidence"):
        derive_site_ledger(inputs)


def test_site_hash_tamper_fails_closed() -> None:
    ledger = derive_site_ledger(_inputs())
    ledger[0]["site_hash"] = "0" * 64
    with pytest.raises(ValueError, match="site hash"):
        derive_summary(ledger)
