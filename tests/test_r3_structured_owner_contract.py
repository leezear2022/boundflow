"""R3-0 frozen bundle and semantic replay tests."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import cast

import pytest

from boundflow.runtime.r3_structured_owner_contract import (
    build_r30_bundle,
    validate_r30_bundle,
)


def _resign(payload: dict[str, object]) -> None:
    unsigned = {name: payload[name] for name in payload if name != "bundle_hash"}
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    payload["bundle_hash"] = hashlib.sha256(canonical.encode()).hexdigest()


def test_frozen_bundle_replays_to_claim_bounded_summary() -> None:
    bundle = build_r30_bundle()
    summary = validate_r30_bundle(bundle)
    assert summary["status"] == "validated-r3-0-contract"
    assert summary["node_count"] == 8
    assert summary["edge_count"] == 8
    assert summary["scratch_slot_count"] == 2
    assert summary["saved_coefficient_bytes"] == 0
    assert summary["dense_escape_count"] == 0
    assert summary["context_tensor_count"] == 0
    assert summary["production_connected"] is False
    assert summary["timing_recorded"] is False
    assert summary["performance_claimed"] is False
    assert summary["r3_1_open"] is True


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("template", "start_node_id", "31/Gemm_14"),
        ("template", "root_node_id", "add"),
        ("template", "topology_hash", "0" * 64),
        ("instance", "mutation_ordinal", 1),
        ("instance", "current_stream", 8),
        ("instance", "split_history_hash", "0" * 64),
        ("saved_tensor_ledger", "coefficient_bytes", 4),
        ("receipt", "saved_coefficient_bytes", 4),
        ("receipt", "dense_escape_count", 1),
        ("receipt", "context_tensor_count", 1),
        ("receipt", "production_connected", True),
        ("receipt", "performance_claimed", True),
    ),
)
def test_full_resign_tamper_is_rejected(
    section: str, field: str, value: object
) -> None:
    bundle = deepcopy(build_r30_bundle())
    cast(dict[str, object], bundle[section])[field] = value
    _resign(bundle)
    with pytest.raises(ValueError):
        validate_r30_bundle(bundle)


def test_bundle_rejects_outer_hash_and_extra_field() -> None:
    bundle = build_r30_bundle()
    bundle["bundle_hash"] = "0" * 64
    with pytest.raises(ValueError, match="bundle hash"):
        validate_r30_bundle(bundle)

    bundle = build_r30_bundle()
    bundle["latency_ms"] = 1.0
    with pytest.raises(ValueError, match="fields differ"):
        validate_r30_bundle(bundle)
