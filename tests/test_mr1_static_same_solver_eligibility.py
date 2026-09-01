"""Tests for the MR1 static same-solver eligibility derivation."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boundflow.runtime.mr1_static_same_solver_eligibility import (
    EXPECTED_CALL_COUNT,
    TARGET_MODEL_HASH,
    canonical_hash,
    classify_call,
    derive_coverage,
    derive_summary,
)

ROOT = Path(__file__).resolve().parents[1]
ACTIVATION = (
    ROOT / "artifacts/rvir/rvir-cpu-correctness-v2-20260803/activation_calls.jsonl"
)


def _rows() -> list[dict[str, object]]:
    return [
        json.loads(line) for line in ACTIVATION.read_text(encoding="utf-8").splitlines()
    ]


def _eligible_row() -> dict[str, object]:
    return {
        "backend": "cibc_ibp_full_graph/v1",
        "identity_limitations": [],
        "query": {
            "bound_method": "IBP",
            "compatibility_key": {
                "backend_capability_class": "cibc_ibp_full_graph",
                "input_shape": [1, 3, 32, 32],
                "spec_shape": None,
            },
            "cuts_version": None,
            "device": "cuda:0",
            "dtype": "torch.float32",
            "execution_options": {
                "cibc_compile_key": "key",
                "cibc_topology_receipt": "receipt",
                "solver_phase": "initial_ibp",
                "split_state_present": False,
            },
            "model_structure_hash": TARGET_MODEL_HASH,
            "optimization_stage": "initial_bound",
            "parent_query_id": None,
            "query_id": "eligible-0",
            "requested_outputs": ["interval_graph"],
            "requires_grad": False,
            "sequence_number": 0,
            "split_signature": None,
        },
        "semantics_owner": "boundflow_cibc_full_graph",
        "source_workload": "vnncomp21-resnet2b-prop0",
    }


def test_canonical_hash_is_order_independent() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_hypothetical_full_graph_call_is_eligible() -> None:
    result = classify_call(_eligible_row())
    assert result["eligible"] is True
    assert result["rejection_reasons"] == []


def test_frozen_resnet_calls_are_all_rejected() -> None:
    ledger = [
        classify_call(row)
        for row in _rows()
        if row["query"]["model_structure_hash"] == TARGET_MODEL_HASH
    ]
    assert len(ledger) == 51
    assert all(row["eligible"] is False for row in ledger)
    assert {row["primary_rejection_reason"] for row in ledger} == {
        "solver_phase_not_initial_ibp"
    }


def test_coverage_is_lossless() -> None:
    coverage = derive_coverage(_rows())
    assert coverage["activation_call_count"] == EXPECTED_CALL_COUNT
    assert coverage["workload_counts"] == {
        "official-simple-mlp-cuda-bab": 343,
        "vnncomp21-resnet2b-prop0": 51,
    }
    assert coverage["split_state_present_counts"] == {"True": 394}


def test_summary_mechanically_closes_full_graph_route() -> None:
    rows = _rows()
    coverage = derive_coverage(rows)
    ledger = [
        classify_call(row)
        for row in rows
        if row["query"]["model_structure_hash"] == TARGET_MODEL_HASH
    ]
    summary = derive_summary(coverage=coverage, target_ledger=ledger)
    assert summary["eligible_target_model_call_count"] == 0
    assert summary["verdict"] == "VALIDATED-NO-GO-MR1-CIBC-FULL-GRAPH-SAME-SOLVER"
    assert summary["direct_end_to_end_ab_preregistration_open"] is False


def test_call_count_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="call count"):
        derive_coverage(_rows()[:-1])
