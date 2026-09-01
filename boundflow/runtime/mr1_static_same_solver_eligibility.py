"""Deterministic MR1 static same-solver eligibility derivation."""

# pylint: disable=missing-function-docstring,too-many-locals

from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Iterable, Mapping

MR1_SCHEMA = "boundflow.mr1-static-same-solver-eligibility/v1"
TARGET_MODEL_HASH = (
    "onnx:791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
)
TARGET_TOPOLOGY = {"add": 2, "conv2d": 6, "flatten": 1, "linear": 2, "relu": 6}
EXPECTED_CALL_COUNT = 394
REASON_ORDER = (
    "non_target_model",
    "solver_phase_not_initial_ibp",
    "bound_method_not_ibp",
    "requires_grad_owner_present",
    "split_state_present_or_unresolved",
    "requested_output_not_full_interval_graph",
    "provider_owned_external_exact_call",
    "cibc_runtime_contract_unproven",
    "dynamic_state_or_lineage_unresolved",
    "compile_key_or_topology_receipt_missing",
)


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _query(row: Mapping[str, Any]) -> Mapping[str, Any]:
    query = row.get("query")
    if not isinstance(query, Mapping):
        raise ValueError("MR1 query envelope differs")
    return query


def _compatibility(query: Mapping[str, Any]) -> Mapping[str, Any]:
    value = query.get("compatibility_key")
    if not isinstance(value, Mapping):
        raise ValueError("MR1 compatibility key differs")
    return value


def _execution(query: Mapping[str, Any]) -> Mapping[str, Any]:
    value = query.get("execution_options")
    if not isinstance(value, Mapping):
        raise ValueError("MR1 execution options differ")
    return value


def classify_call(row: Mapping[str, Any]) -> dict[str, object]:
    """Classify one frozen RVIR activation call against the CIBC full-graph ABI."""

    query = _query(row)
    compatibility = _compatibility(query)
    execution = _execution(query)
    limitations = row.get("identity_limitations")
    requested = query.get("requested_outputs")
    if (
        not isinstance(limitations, list)
        or not all(isinstance(item, str) for item in limitations)
        or not isinstance(requested, list)
        or not all(isinstance(item, str) for item in requested)
    ):
        raise ValueError("MR1 typed call fields differ")

    model_hash = query.get("model_structure_hash")
    phase = execution.get("solver_phase")
    stage = query.get("optimization_stage")
    method = query.get("bound_method")
    split_present = execution.get("split_state_present")
    requires_grad = query.get("requires_grad")
    if not isinstance(requires_grad, bool) or not isinstance(split_present, bool):
        raise ValueError("MR1 boolean ownership fields differ")

    reasons: list[str] = []
    if model_hash != TARGET_MODEL_HASH:
        reasons.append("non_target_model")
    if phase not in {"initial_ibp", "ibp_graph_evaluation"} or stage not in {
        "initial_bound",
        "ibp_graph_evaluation",
    }:
        reasons.append("solver_phase_not_initial_ibp")
    if method != "IBP":
        reasons.append("bound_method_not_ibp")
    if requires_grad:
        reasons.append("requires_grad_owner_present")
    split_signature = query.get("split_signature")
    if (
        split_present
        or split_signature not in {None, "none"}
        or any("split" in item for item in limitations)
    ):
        reasons.append("split_state_present_or_unresolved")
    if requested != ["interval_graph"]:
        reasons.append("requested_output_not_full_interval_graph")
    if (
        row.get("semantics_owner") != "boundflow_cibc_full_graph"
        or row.get("backend") == "external_abcrown_exact_call/v1"
    ):
        reasons.append("provider_owned_external_exact_call")
    if (
        query.get("device") != "cuda:0"
        or query.get("dtype") != "torch.float32"
        or compatibility.get("input_shape") is None
        or compatibility.get("spec_shape") is not None
    ):
        reasons.append("cibc_runtime_contract_unproven")
    if (
        query.get("cuts_version") is not None
        or query.get("parent_query_id") is not None
        or limitations
    ):
        reasons.append("dynamic_state_or_lineage_unresolved")
    if (
        compatibility.get("backend_capability_class") != "cibc_ibp_full_graph"
        or execution.get("cibc_topology_receipt") is None
        or execution.get("cibc_compile_key") is None
    ):
        reasons.append("compile_key_or_topology_receipt_missing")

    ordered = [reason for reason in REASON_ORDER if reason in reasons]
    eligible = not ordered
    result: dict[str, object] = {
        "query_id": query.get("query_id"),
        "sequence_number": query.get("sequence_number"),
        "source_workload": row.get("source_workload"),
        "model_structure_hash": model_hash,
        "solver_phase": phase,
        "optimization_stage": stage,
        "bound_method": method,
        "requires_grad": requires_grad,
        "split_state_present": split_present,
        "semantics_owner": row.get("semantics_owner"),
        "backend": row.get("backend"),
        "eligible": eligible,
        "primary_rejection_reason": None if eligible else ordered[0],
        "rejection_reasons": ordered,
    }
    result["ledger_hash"] = canonical_hash(result)
    return result


def derive_coverage(rows: Iterable[Mapping[str, Any]]) -> dict[str, object]:
    """Derive lossless distributions over all frozen RVIR activation calls."""

    materialized = list(rows)
    if len(materialized) != EXPECTED_CALL_COUNT:
        raise ValueError("MR1 activation call count differs")

    def counts(values: Iterable[object]) -> dict[str, int]:
        return dict(sorted(Counter(str(value) for value in values).items()))

    coverage: dict[str, object] = {
        "schema_version": "boundflow.mr1-static-coverage/v1",
        "activation_call_count": len(materialized),
        "workload_counts": counts(row.get("source_workload") for row in materialized),
        "model_counts": counts(
            _query(row).get("model_structure_hash") for row in materialized
        ),
        "method_counts": counts(
            _query(row).get("bound_method") for row in materialized
        ),
        "phase_counts": counts(
            _execution(_query(row)).get("solver_phase") for row in materialized
        ),
        "requires_grad_counts": counts(
            _query(row).get("requires_grad") for row in materialized
        ),
        "split_state_present_counts": counts(
            _execution(_query(row)).get("split_state_present") for row in materialized
        ),
        "performance_claimed": False,
    }
    coverage["coverage_hash"] = canonical_hash(coverage)
    return coverage


def derive_summary(
    *, coverage: Mapping[str, Any], target_ledger: Iterable[Mapping[str, Any]]
) -> dict[str, object]:
    """Derive the mechanical MR1 route from the target-model ledger."""

    ledger = list(target_ledger)
    if coverage.get("activation_call_count") != EXPECTED_CALL_COUNT or not ledger:
        raise ValueError("MR1 coverage or target ledger differs")
    for row in ledger:
        unsigned = dict(row)
        ledger_hash = unsigned.pop("ledger_hash", None)
        if ledger_hash != canonical_hash(unsigned):
            raise ValueError("MR1 ledger hash differs")
    eligible = sum(row.get("eligible") is True for row in ledger)
    primary = Counter(
        str(row["primary_rejection_reason"])
        for row in ledger
        if row.get("eligible") is not True
    )
    all_reasons = Counter(
        str(reason) for row in ledger for reason in row.get("rejection_reasons", [])
    )
    admitted = eligible > 0
    result: dict[str, object] = {
        "schema_version": MR1_SCHEMA,
        "activation_call_count": EXPECTED_CALL_COUNT,
        "target_model_call_count": len(ledger),
        "eligible_target_model_call_count": eligible,
        "rejected_target_model_call_count": len(ledger) - eligible,
        "primary_rejection_counts": dict(sorted(primary.items())),
        "all_rejection_counts": dict(sorted(all_reasons.items())),
        "mr1_static_eligibility_admitted": admitted,
        "verdict": (
            "VALIDATED-MR1-STATIC-ELIGIBILITY"
            if admitted
            else "VALIDATED-NO-GO-MR1-CIBC-FULL-GRAPH-SAME-SOLVER"
        ),
        "direct_end_to_end_ab_preregistration_open": admitted,
        "same_solver_timing_open": False,
        "r2_open": False,
        "performance_claimed": False,
    }
    result["summary_hash"] = canonical_hash(result)
    return result


__all__ = [
    "EXPECTED_CALL_COUNT",
    "MR1_SCHEMA",
    "REASON_ORDER",
    "TARGET_MODEL_HASH",
    "TARGET_TOPOLOGY",
    "canonical_hash",
    "classify_call",
    "derive_coverage",
    "derive_summary",
]
