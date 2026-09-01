"""Synthetic gates for MR5 multi-site production timing."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

from copy import deepcopy
import json
import lzma
from pathlib import Path

import pytest

from boundflow.runtime.mr3_provider_hook_feasibility import canonical_hash
from boundflow.runtime.mr5_multi_conv_timing import (
    SOURCE_COMMIT,
    TIMING_SCHEMA,
    _validate_bridge_receipt,
    _validate_candidate_receipt,
    derive_summary,
)

ROOT = Path(__file__).resolve().parents[1]
CORRECTNESS_RAW = (
    ROOT
    / "artifacts/measurement-recovery/mr5-multi-conv-production-bridge-v1/raw.json.xz"
)


def _receipts() -> tuple[dict[str, object], dict[str, object]]:
    raw = json.loads(lzma.decompress(CORRECTNESS_RAW.read_bytes()).decode("utf-8"))
    correctness = next(
        wrapper["worker"]["bridge_receipt"]
        for wrapper in raw["runs"]
        if wrapper["mode"] == "bridge"
    )
    modules = deepcopy(correctness["module_receipts"])
    candidate: dict[str, object] = {
        "site_order": ["C2", "C1", "C0"],
        "module_receipts": modules,
        "dummy_forward_launch_count": 3,
        "dummy_backward_launch_count": 3,
        "dummy_fallback_count": 0,
        "dummy_eager_count": 0,
    }
    candidate["receipt_hash"] = canonical_hash(candidate)
    bridge = deepcopy(correctness)
    bridge["cache_miss_count"] = {"C0": 0, "C1": 0, "C2": 0}
    bridge["cache_hit_count"] = {"C0": 10, "C1": 10, "C2": 10}
    bridge["prewarmed_before_outer"] = True
    bridge["timing_recorded"] = True
    return candidate, bridge


def test_mr5_timing_accepts_prewarmed_three_module_receipts() -> None:
    candidate, bridge = _receipts()
    _validate_candidate_receipt(candidate)
    _validate_bridge_receipt(bridge, candidate)


def test_mr5_timing_rejects_cache_miss_inside_outer() -> None:
    candidate, bridge = _receipts()
    bridge["cache_miss_count"]["C1"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="bridge receipt differs"):
        _validate_bridge_receipt(bridge, candidate)


def test_mr5_timing_rejects_candidate_bridge_module_mismatch() -> None:
    candidate, bridge = _receipts()
    bridge["module_receipts"]["C0"]["scheduled_tir_hash"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="bridge receipt differs"):
        _validate_bridge_receipt(bridge, candidate)


def test_mr5_timing_raw_rejects_empty_fully_resigned_payload() -> None:
    raw: dict[str, object] = {
        "schema_version": TIMING_SCHEMA,
        "source_commit": SOURCE_COMMIT,
        "run_order": [],
        "runs": [],
    }
    raw["raw_hash"] = canonical_hash(raw)
    with pytest.raises(ValueError, match="raw provenance differs"):
        derive_summary(raw)
