"""Frozen NRIR-36 formal artifact and synchronized tamper tests."""

# pylint: disable=missing-function-docstring

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_multi_clause_anytime_priority_formal import (
    ARTIFACT_DIR,
    _canonical_hash,
    _semantic_worker,
    validate_formal,
)

ROOT = Path(__file__).resolve().parents[1]


def _formal() -> dict[str, object]:
    value = json.loads((ROOT / ARTIFACT_DIR / "formal.json").read_text("utf-8"))
    assert isinstance(value, dict)
    return value


def _tamper_worker(formal: dict[str, object]) -> dict[str, object]:
    payload = formal["formal_payload"]
    assert isinstance(payload, dict)
    results = payload["repeat_results"]
    assert isinstance(results, list)
    result = results[0]
    assert isinstance(result, dict)
    return result


def _rehash_worker(result: dict[str, object]) -> None:
    result["result_hash"] = _canonical_hash(_semantic_worker(result))


def test_multi_clause_anytime_formal_covers_two_ranked_clauses() -> None:
    formal = _formal()

    validate_formal(formal)

    payload = formal["formal_payload"]
    assert isinstance(payload, dict)
    assert payload["all_worker_gates_passed"] is False
    assert payload["all_original_ordinals_preserved"] is True
    assert payload["selected_original_clause_indices"] == [[2, 3]] * 3
    assert payload["packed_accepted_nodes"] == [[3, 3], [3, 3], [3, 1]]
    assert formal["status"] == "no_go"
    assert formal["performance_claimed"] is False


@pytest.mark.parametrize(
    "tamper",
    (
        "wrong_rank",
        "wrong_selection",
        "wrong_source",
        "slice_inflation",
        "deadline_reset",
        "ordinal_omission",
    ),
)
def test_multi_clause_anytime_formal_rejects_control_tamper(tamper: str) -> None:
    formal = deepcopy(_formal())
    result = _tamper_worker(formal)
    decision = result["decision"]
    floor = result["floor"]
    slices = result["slices"]
    assert isinstance(decision, dict)
    assert isinstance(floor, dict)
    assert isinstance(slices, list)
    first = slices[0]
    assert isinstance(first, dict)
    slice_ir = first["slice"]
    assert isinstance(slice_ir, dict)
    if tamper == "wrong_rank":
        decision["ranked_original_clause_indices"][0:2] = [3, 2]
    elif tamper == "wrong_selection":
        decision["selected_original_clause_indices"] = [2, 4]
    elif tamper == "wrong_source":
        slice_ir["source_refinement_plan_hash"] = "0" * 64
    elif tamper == "slice_inflation":
        slice_ir["allocated_slice_ns"] += 1
    elif tamper == "deadline_reset":
        first["source_elapsed_ns"] = 0
    else:
        floor["completed_original_clause_indices"] = list(range(8))
    _rehash_worker(result)

    with pytest.raises(ValueError, match="worker result differs"):
        validate_formal(formal)


@pytest.mark.parametrize(
    "tamper", ("non_monotone_aggregate", "partial_group", "trace_binding")
)
def test_multi_clause_anytime_formal_rejects_result_tamper(tamper: str) -> None:
    formal = deepcopy(_formal())
    result = _tamper_worker(formal)
    aggregate = result["aggregate"]
    slices = result["slices"]
    runtime_trace = result["runtime_trace"]
    assert isinstance(aggregate, dict)
    assert isinstance(slices, list)
    assert isinstance(runtime_trace, dict)
    if tamper == "non_monotone_aggregate":
        aggregate["final_status"] = "verified"
        aggregate["final_verified_clause_indices"] = list(range(9))
        aggregate["final_unresolved_clause_indices"] = []
    elif tamper == "partial_group":
        slices[0]["slice"]["sibling_group_count"] = 0
    else:
        runtime_trace["actions"] = runtime_trace["actions"][:-1]
    _rehash_worker(result)

    with pytest.raises(ValueError, match="worker result differs"):
        validate_formal(formal)
