"""Frozen NRIR-35 cross-clause anytime artifact and tamper tests."""

# pylint: disable=missing-function-docstring

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_cross_clause_anytime_objective_formal import (
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


def test_cross_clause_anytime_formal_preserves_all_original_ordinals() -> None:
    formal = _formal()

    validate_formal(formal)

    payload = formal["formal_payload"]
    assert isinstance(payload, dict)
    assert payload["all_worker_gates_passed"] is True
    assert payload["all_original_ordinals_preserved"] is True
    assert len(payload["repeat_results"]) == 3
    assert formal["status"] == "validated-reduced"
    assert formal["performance_claimed"] is False


@pytest.mark.parametrize(
    "tamper",
    ("wrong_ordinal", "wrong_source", "deadline_reset", "baseline_omission"),
)
def test_cross_clause_anytime_formal_rejects_admission_tamper(tamper: str) -> None:
    formal = deepcopy(_formal())
    result = _tamper_worker(formal)
    decision = result["decision"]
    floor = result["floor"]
    packed = result["packed"]
    assert isinstance(decision, dict)
    assert isinstance(floor, dict)
    assert isinstance(packed, dict)
    if tamper == "wrong_ordinal":
        decision["admitted_original_clause_index"] = 1
    elif tamper == "wrong_source":
        decision["root_refinement_plan_hash"] = "0" * 64
    elif tamper == "deadline_reset":
        packed["source_elapsed_ns"] = 0
    else:
        floor["completed_original_clause_indices"] = list(range(8))
    _rehash_worker(result)

    with pytest.raises(ValueError, match="worker result differs"):
        validate_formal(formal)


@pytest.mark.parametrize("tamper", ("non_monotone_aggregate", "partial_group"))
def test_cross_clause_anytime_formal_rejects_result_tamper(tamper: str) -> None:
    formal = deepcopy(_formal())
    result = _tamper_worker(formal)
    aggregate = result["aggregate"]
    packed = result["packed"]
    assert isinstance(aggregate, dict)
    assert isinstance(packed, dict)
    if tamper == "non_monotone_aggregate":
        aggregate["final_status"] = "verified"
        aggregate["final_verified_clause_indices"] = list(range(9))
        aggregate["final_unresolved_clause_indices"] = []
    else:
        packed["sibling_group_count"] = packed["sibling_group_count"] - 1
    _rehash_worker(result)

    with pytest.raises(ValueError, match="worker result differs"):
        validate_formal(formal)
