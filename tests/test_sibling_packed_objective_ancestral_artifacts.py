"""Frozen NRIR-34 profile, formal, and full-query artifact tests."""

# pylint: disable=missing-function-docstring,duplicate-code

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_sibling_packed_objective_ancestral_formal import (
    ARTIFACT_DIR as FORMAL_ARTIFACT_DIR,
    validate_formal,
)
from scripts.run_sibling_packed_objective_ancestral_full_query import (
    ARTIFACT_DIR as FULL_QUERY_ARTIFACT_DIR,
    validate_evidence,
)
from scripts.run_sibling_packed_objective_ancestral_profile import (
    ARTIFACT_DIR as PROFILE_ARTIFACT_DIR,
    validate_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_sibling_pair_profile_passes_mechanism_gate() -> None:
    profile = _load(ROOT / PROFILE_ARTIFACT_DIR / "profile.json")

    validate_profile(profile)

    comparison = profile["comparison"]
    assert isinstance(comparison, dict)
    assert comparison["mechanism_gate_passed"] is True
    assert comparison["serial_optimizer_groups"] == 2
    assert comparison["packed_optimizer_groups"] == 1


def test_frozen_three_repeat_formal_claims_strict_node_gain() -> None:
    formal = _load(ROOT / FORMAL_ARTIFACT_DIR / "formal.json")

    validate_formal(formal)

    payload = formal["formal_payload"]
    assert isinstance(payload, dict)
    assert payload["serial_accepted_nodes"] == [7, 7, 7]
    assert payload["packed_accepted_nodes"] == [15, 15, 15]
    assert payload["minimum_node_gain"] == 8
    assert formal["performance_claimed"] is True


def test_frozen_formal_rejects_coverage_tamper() -> None:
    formal = deepcopy(_load(ROOT / FORMAL_ARTIFACT_DIR / "formal.json"))
    payload = formal["formal_payload"]
    assert isinstance(payload, dict)
    results = payload["repeat_results"]
    assert isinstance(results, list)
    packed = results[0]["packed"]
    assert isinstance(packed, dict)
    packed["accepted_nodes"] = 7

    with pytest.raises(ValueError, match="worker result differs"):
        validate_formal(formal)


def test_frozen_full_query_preserves_original_ordinal_accounting() -> None:
    evidence = _load(ROOT / FULL_QUERY_ARTIFACT_DIR / "evidence.json")

    validate_evidence(evidence)

    trace = evidence["query_trace"]
    assert isinstance(trace, dict)
    assert trace["status"] == "unknown"
    assert trace["unresolved_clause_indices"] == [0]
    assert trace["pending_clause_indices"] == list(range(1, 9))
    assert evidence["performance_claimed"] is False


def test_frozen_full_query_rejects_pending_ordinal_tamper() -> None:
    evidence = deepcopy(_load(ROOT / FULL_QUERY_ARTIFACT_DIR / "evidence.json"))
    trace = evidence["query_trace"]
    assert isinstance(trace, dict)
    trace["pending_clause_indices"] = list(range(2, 9))

    with pytest.raises(ValueError, match="evidence envelope differs"):
        validate_evidence(evidence)
