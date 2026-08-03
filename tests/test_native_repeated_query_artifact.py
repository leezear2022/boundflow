"""Frozen NRIR-7 repeated-query artifact and tamper rejection."""

# pylint: disable=missing-function-docstring,implicit-str-concat

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from boundflow.runtime.abcrown_adapter import file_sha256
from scripts.run_native_real_network_repeated_query_artifact import (
    REPEATED_QUERY_ARTIFACT_SCHEMA_VERSION,
    validate_repeated_query_evidence,
)

ARTIFACT_DIR = Path(
    "artifacts/native-real-network-repeated-query/" "vnncomp21-resnet2b-prop0-cpu-v1"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_repeated_query_artifact_digest_and_contract() -> None:
    manifest = _load("manifest.json")
    evidence = _load("repeated_query.json")
    assert manifest["schema_version"] == REPEATED_QUERY_ARTIFACT_SCHEMA_VERSION
    assert manifest["status"] == "ok"
    assert manifest["performance_claimed"] is False
    assert manifest["files"] == {
        "repeated_query.json": file_sha256(ARTIFACT_DIR / "repeated_query.json")
    }
    validate_repeated_query_evidence(evidence)


@pytest.mark.parametrize(
    "gate",
    [
        "nine_distinct_property_queries_are_explicit",
        "packed_three_children_replace_nine_serial_children",
        "first_compile_misses_second_exact_compile_hits",
        "cache_key_tracks_objective_order_and_state",
        "nine_results_restore_exact_query_lineage",
        "packed_cached_serial_and_external_semantics_match",
        "mechanism_counts_do_not_claim_performance",
    ],
)
def test_repeated_query_artifact_rejects_failed_gate(gate: str) -> None:
    evidence = deepcopy(_load("repeated_query.json"))
    gates = evidence["gates"]
    assert isinstance(gates, dict)
    gates[gate] = False
    with pytest.raises(ValueError, match="gates"):
        validate_repeated_query_evidence(evidence)


def test_repeated_query_artifact_rejects_query_range_relink() -> None:
    evidence = deepcopy(_load("repeated_query.json"))
    layout = evidence["layout_trace"]
    assert isinstance(layout, dict)
    ranges = layout["query_ranges"]
    assert isinstance(ranges, list)
    first = ranges[0]
    assert isinstance(first, dict)
    first["stop_index"] = 2
    with pytest.raises(ValueError, match="range lineage"):
        validate_repeated_query_evidence(evidence)


def test_repeated_query_artifact_rejects_cache_key_alias() -> None:
    evidence = deepcopy(_load("repeated_query.json"))
    cache = evidence["cache"]
    assert isinstance(cache, dict)
    probe = cache["key_probe"]
    assert isinstance(probe, dict)
    probe["state_tamper_cache_key"] = probe["primary_cache_key"]
    with pytest.raises(ValueError, match="cache identity"):
        validate_repeated_query_evidence(evidence)


def test_repeated_query_artifact_rejects_result_lineage_tamper() -> None:
    evidence = deepcopy(_load("repeated_query.json"))
    packed = evidence["packed"]
    assert isinstance(packed, dict)
    results = packed["results"]
    assert isinstance(results, list)
    second = results[1]
    assert isinstance(second, dict)
    second["query_id"] = "property-00"
    with pytest.raises(ValueError, match="restored query result"):
        validate_repeated_query_evidence(evidence)


def test_repeated_query_artifact_rejects_semantic_gate_tamper() -> None:
    evidence = deepcopy(_load("repeated_query.json"))
    semantics = evidence["semantics"]
    assert isinstance(semantics, dict)
    comparison = semantics["packed_vs_serial_lower"]
    assert isinstance(comparison, dict)
    comparison["allclose"] = False
    with pytest.raises(ValueError, match="semantic comparison"):
        validate_repeated_query_evidence(evidence)


def test_repeated_query_artifact_rejects_claim_inflation() -> None:
    evidence = deepcopy(_load("repeated_query.json"))
    evidence["performance_claimed"] = True
    evidence["claim_boundary"] = "repeated-query cache speedup"
    with pytest.raises(ValueError, match="contract"):
        validate_repeated_query_evidence(evidence)
