"""Deterministic MR2 production CROWN subgraph owner inventory."""

# pylint: disable=missing-function-docstring,too-many-locals,too-many-boolean-expressions

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

MR2_SCHEMA = "boundflow.mr2-production-crown-owner-inventory/v1"
GATE_ORDER = (
    "production_site_identity",
    "typed_input_output_abi",
    "state_ownership",
    "forward_backward_correctness",
    "optimizer_trajectory_correctness",
    "multi_site_consumer_closure",
    "production_exact_call_connection",
)


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _gate(status: str, evidence: Sequence[str]) -> dict[str, object]:
    if status not in {"proven", "bounded_single_site", "missing", "rejected"}:
        raise ValueError("MR2 gate status differs")
    return {"status": status, "evidence": list(evidence)}


def _binding(bundle: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    bindings = bundle.get("instance", {}).get("bindings", [])
    matches = [item for item in bindings if item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"MR2 binding differs: {name}")
    return matches[0]


def derive_site_ledger(
    inputs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    bundle = inputs["p_bundle"]
    p_trajectory = inputs["p_trajectory"]
    s_correctness = inputs["s_correctness"]
    p_cibc = inputs["p_cibc"]
    p_v1 = inputs["p_v1"]
    mr1 = inputs["mr1"]
    inventory = inputs["inventory"]

    alpha = _binding(bundle, "alpha")
    beta = _binding(bundle, "beta")
    lower = _binding(bundle, "lower")
    weight = _binding(bundle, "weight")
    receipt = bundle.get("receipt", {})
    template = bundle.get("template", {})
    if (
        template.get("start_node_id") != "25/Conv_8"
        or alpha.get("shape") != [2, 1, 6, 86]
        or beta.get("shape") != [6, 0]
        or lower.get("shape") != [6, 16, 8, 8]
        or weight.get("shape") != [16, 16, 3, 3]
        or receipt.get("production_connected") is not False
        or receipt.get("dense_escape_count") != 0
        or receipt.get("context_tensor_count") != 0
        or p_trajectory.get("trajectory_correctness_admitted") is not True
        or p_trajectory.get("ownership_admitted") is not True
        or p_trajectory.get("timing_recorded") is not False
        or p_cibc.get("maximum_absolute_difference", 1.0) > 0.0002
        or p_cibc.get("sign_exact") is not True
        or p_cibc.get("performance_claimed") is not False
        or p_v1.get("status") != "validated-no-go-b4-b2-v1-physics"
        or s_correctness.get("active_beta_correctness_admitted") is not True
        or s_correctness.get("ownership_admitted") is not True
        or s_correctness.get("beta_nonzero_count") != 30
        or s_correctness.get("timing_recorded") is not False
        or mr1.get("eligible_target_model_call_count") != 0
        or inventory.get("source", {}).get("model_sha256")
        != "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
    ):
        raise ValueError("MR2 frozen evidence differs")

    p_gates = {
        "production_site_identity": _gate(
            "proven", ["p_bundle.template.start_node_id", "inventory.source"]
        ),
        "typed_input_output_abi": _gate(
            "proven", ["p_bundle.instance.bindings", "p_bundle.saved_tensor_ledger"]
        ),
        "state_ownership": _gate(
            "proven", ["p_bundle.receipt", "p_trajectory.ownership_admitted"]
        ),
        "forward_backward_correctness": _gate(
            "proven", ["p_cibc.maximum_absolute_difference", "p_cibc.sign_exact"]
        ),
        "optimizer_trajectory_correctness": _gate(
            "proven", ["p_trajectory.trajectory_correctness_admitted"]
        ),
        "multi_site_consumer_closure": _gate(
            "bounded_single_site",
            [
                "p_bundle.receipt.external_consumer_boundary",
                "p_bundle.receipt.dense_escape_count",
            ],
        ),
        "production_exact_call_connection": _gate(
            "missing", ["p_bundle.receipt.production_connected=false", "mr1.eligible=0"]
        ),
    }
    s_gates = {
        "production_site_identity": _gate(
            "missing",
            ["s_correctness anchor ordinal lacks first-class start-node receipt"],
        ),
        "typed_input_output_abi": _gate(
            "proven",
            ["s_correctness.template_hash", "s_correctness.module_receipt_hash"],
        ),
        "state_ownership": _gate(
            "proven",
            ["s_correctness.ownership_admitted", "s_correctness.beta_nonzero_count"],
        ),
        "forward_backward_correctness": _gate(
            "proven", ["s_correctness.active_beta_correctness_admitted"]
        ),
        "optimizer_trajectory_correctness": _gate(
            "missing", ["no S-anchor 10/9 mutation artifact"]
        ),
        "multi_site_consumer_closure": _gate(
            "missing", ["single S-anchor only; adjacent consumers unbound"]
        ),
        "production_exact_call_connection": _gate(
            "missing", ["s_correctness.same_solver_open=false", "mr1.eligible=0"]
        ),
    }
    if tuple(p_gates) != GATE_ORDER or tuple(s_gates) != GATE_ORDER:
        raise ValueError("MR2 gate order differs")

    rows: list[dict[str, object]] = []
    for site_id, op_kind, beta_mode, gates in (
        ("P:25/Conv_8", "conv2d_right", "absent", p_gates),
        ("S:31/Gemm_14", "linear_right", "active", s_gates),
    ):
        first_five = [gates[name]["status"] == "proven" for name in GATE_ORDER[:5]]
        bounded = (
            gates["multi_site_consumer_closure"]["status"] == "bounded_single_site"
        )
        connection_missing = (
            gates["production_exact_call_connection"]["status"] == "missing"
        )
        ready = all(first_five) and bounded and connection_missing
        row: dict[str, object] = {
            "site_id": site_id,
            "op_kind": op_kind,
            "beta_mode": beta_mode,
            "gates": gates,
            "missing_gates": [
                name for name in GATE_ORDER if gates[name]["status"] == "missing"
            ],
            "ready_for_bridge_correctness": ready,
            "performance_claimed": False,
        }
        row["site_hash"] = canonical_hash(row)
        rows.append(row)
    return rows


def derive_summary(site_ledger: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    if [row.get("site_id") for row in site_ledger] != ["P:25/Conv_8", "S:31/Gemm_14"]:
        raise ValueError("MR2 site inventory differs")
    for row in site_ledger:
        unsigned = dict(row)
        site_hash = unsigned.pop("site_hash", None)
        if site_hash != canonical_hash(unsigned):
            raise ValueError("MR2 site hash differs")
    ready = [
        str(row["site_id"])
        for row in site_ledger
        if row.get("ready_for_bridge_correctness")
    ]
    conflict = any(
        gate.get("status") == "rejected"
        for row in site_ledger
        for gate in row.get("gates", {}).values()
    )
    if conflict:
        route = "BLOCKED-CONTRACT-CONFLICT"
        selected = None
    elif ready:
        selected = ready[0]
        route = "OPEN-P-ANCHOR-PRODUCTION-EXACT-CALL-BRIDGE-CORRECTNESS-PREREGISTRATION"
    else:
        selected = None
        route = "NO-GO-MR2-CURRENT-SITES"
    result: dict[str, object] = {
        "schema_version": MR2_SCHEMA,
        "site_count": len(site_ledger),
        "ready_site_count": len(ready),
        "ready_sites": ready,
        "selected_site": selected,
        "route": route,
        "bridge_correctness_preregistration_open": selected is not None,
        "bridge_implemented": False,
        "timing_open": False,
        "same_solver_open": False,
        "r2_open": False,
        "performance_claimed": False,
    }
    result["summary_hash"] = canonical_hash(result)
    return result


__all__ = [
    "GATE_ORDER",
    "MR2_SCHEMA",
    "canonical_hash",
    "derive_site_ledger",
    "derive_summary",
]
