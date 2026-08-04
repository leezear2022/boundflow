"""Replay and synchronized-tamper gates for the NRIR-38 pilot artifact."""

# pylint: disable=missing-function-docstring

from copy import deepcopy

import pytest

from scripts.run_full_frontier_tightness_attribution_pilot import (
    ARTIFACT_DIR,
    _canonical_hash,
    _load_json,
    _semantic_pilot,
    validate_pilot,
)


def _pilot():
    return _load_json(ARTIFACT_DIR / "pilot.json")


def _rehash(pilot) -> None:
    pilot["pilot_hash"] = _canonical_hash(_semantic_pilot(pilot))


def test_full_frontier_pilot_closes_optimizer_steps_as_no_go() -> None:
    pilot = _pilot()

    validate_pilot(pilot)

    assert pilot["status"] == "no_go"
    assert pilot["all_candidate_gates_passed"] is False
    assert [row["decision"]["go"] for row in pilot["clauses"]] == [False, False]
    assert [row["decision"]["improved_node_count"] for row in pilot["clauses"]] == [
        16,
        16,
    ]


@pytest.mark.parametrize(
    "tamper",
    (
        "active_omission",
        "policy_steps",
        "candidate_delta",
        "decision_flip",
        "task_hash",
        "cache_outcome",
        "sibling_batch",
        "execution_hash",
    ),
)
def test_full_frontier_pilot_rejects_semantic_tamper(tamper: str) -> None:
    pilot = deepcopy(_pilot())
    clause = pilot["clauses"][0]
    if tamper == "active_omission":
        clause["node_rows"][-1]["active"] = False
    elif tamper == "policy_steps":
        clause["plan"]["candidate_optimizer_steps"] = 16
    elif tamper == "candidate_delta":
        row = clause["candidate_rows"][0]
        row["candidate_lower"] += 1.0
        row["candidate_lower_delta"] += 1.0
    elif tamper == "decision_flip":
        clause["decision"]["go"] = True
    elif tamper == "task_hash":
        clause["task_ir_hash"] = "0" * 64
    elif tamper == "cache_outcome":
        clause["candidate_cache_outcomes"][1] = "miss_compiled"
    elif tamper == "sibling_batch":
        clause["candidate_rows"][1]["sibling_batch_index"] = 1
    else:
        clause["execution_hash"] = "0" * 64
    _rehash(pilot)

    with pytest.raises(ValueError):
        validate_pilot(pilot)
