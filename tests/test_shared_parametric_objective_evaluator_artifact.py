"""Frozen NRIR-37 formal artifact and synchronized tamper tests."""

# pylint: disable=missing-function-docstring

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_shared_parametric_objective_evaluator_formal import (
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


def test_shared_parametric_formal_closes_three_repeats() -> None:
    formal = _formal()

    validate_formal(formal)

    payload = formal["formal_payload"]
    assert isinstance(payload, dict)
    assert payload["all_worker_gates_passed"] is True
    assert payload["selected_original_clause_indices"] == [[2, 3]] * 3
    assert payload["packed_accepted_nodes"] == [[31, 31]] * 3
    assert payload["cache_miss_counts"] == [1, 1, 1]
    assert formal["status"] == "validated-reduced"
    assert formal["performance_claimed"] is False


@pytest.mark.parametrize(
    "tamper",
    (
        "wrong_rank",
        "wrong_selection",
        "wrong_source",
        "slice_inflation",
        "partial_group",
        "ordinal_omission",
    ),
)
def test_shared_parametric_formal_rejects_control_tamper(tamper: str) -> None:
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
        decision["candidates"][0]["root_refinement_plan_hash"] = "0" * 64
    elif tamper == "slice_inflation":
        slice_ir["allocated_slice_ns"] += 1
    elif tamper == "partial_group":
        slice_ir["sibling_group_count"] -= 1
    else:
        floor["completed_original_clause_indices"] = list(range(8))
    _rehash_worker(result)

    with pytest.raises(ValueError, match="worker result differs"):
        validate_formal(formal)


@pytest.mark.parametrize(
    "tamper",
    (
        "second_miss",
        "template_count",
        "event_ordinal",
        "native_reexecution",
        "compiler_coverage",
    ),
)
def test_shared_parametric_formal_rejects_compiler_tamper(tamper: str) -> None:
    formal = deepcopy(_formal())
    result = _tamper_worker(formal)
    cache = result["cache"]
    slices = result["slices"]
    assert isinstance(cache, dict)
    assert isinstance(slices, list)
    events = cache["events"]
    assert isinstance(events, list)
    if tamper == "second_miss":
        events[1]["outcome"] = "miss_compiled"
    elif tamper == "template_count":
        cache["template_count"] = 2
    elif tamper == "event_ordinal":
        events[1]["event_index"] = 9
    elif tamper == "native_reexecution":
        slices[0]["compiler"]["selected_native_reexecution"] = True
    else:
        slices[0]["compiler_batch_count"] -= 1
    _rehash_worker(result)

    with pytest.raises(ValueError, match="worker result differs"):
        validate_formal(formal)
