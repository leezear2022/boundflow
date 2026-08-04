"""Frozen NRIR-40 whole-query formal artifact and tamper gates."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_objective_branch_whole_query_formal import (
    _canonical_hash,
    _semantic_worker,
    validate_formal,
)

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts/objective-branch-whole-query"
    / "vnncomp21-resnet2b-property0-three-repeat-cpu-formal-v1"
)


def _formal() -> dict[str, object]:
    value = json.loads((ARTIFACT_DIR / "formal.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_formal_records_three_correct_no_go_repeats() -> None:
    """The frozen result closes correctness but not production admission."""

    formal = _formal()
    validate_formal(formal)
    payload = formal["formal_payload"]
    assert isinstance(payload, dict)
    assert formal["status"] == "no_go"
    assert payload["all_correctness_gates_passed"] is True
    assert payload["all_production_gates_passed"] is False
    assert payload["accepted_nodes"] == [[29, 23], [29, 21], [29, 21]]
    assert payload["branch_execution_counts"] == payload["accepted_nodes"]


def test_frozen_formal_rejects_synchronized_branch_coverage_tamper() -> None:
    """Rehashing every envelope cannot hide duplicate branch coverage."""

    formal = deepcopy(_formal())
    payload = formal["formal_payload"]
    assert isinstance(payload, dict)
    results = payload["repeat_results"]
    assert isinstance(results, list)
    result = results[0]
    bindings = result["slices"][0]["branch_bindings"]
    bindings[1]["node_id"] = bindings[0]["node_id"]
    result["result_hash"] = _canonical_hash(_semantic_worker(result))
    formal["formal_payload_hash"] = _canonical_hash(payload)
    with pytest.raises(ValueError, match="NRIR-40 worker result differs"):
        validate_formal(formal)
