"""CPU gates for MR7-R unprofiled host recovery."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, cast

import pytest

from boundflow.runtime.mr3_provider_hook_feasibility import canonical_hash
from boundflow.runtime.mr7r_unprofiled_host_recovery import (
    EXPECTED_RUNS,
    RAW_SCHEMA,
    derive_summary,
)

ROOT = Path(__file__).resolve().parents[1]
MR6_RAW = (
    ROOT / "artifacts/measurement-recovery/mr6-hot-path-guard-attribution-v1/raw.json"
)
MR7_RAW = (
    ROOT
    / "artifacts/measurement-recovery/mr7-launch-materialization-attribution-v1/raw.json"
)


def _workers() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    mr6 = json.loads(MR6_RAW.read_text(encoding="utf-8"))
    mr7 = json.loads(MR7_RAW.read_text(encoding="utf-8"))
    baselines = [
        deepcopy(row["worker"]) for row in mr6["runs"] if row["mode"] == "diagnostic"
    ]
    ledgers = [
        deepcopy(row["worker"]) for row in mr7["runs"] if row["kind"] == "control"
    ]
    assert len(baselines) == len(ledgers) == 3
    return baselines, ledgers


def _raw() -> dict[str, Any]:
    baselines, ledgers = _workers()
    runs = []
    for pair, position, role in EXPECTED_RUNS:
        source = baselines if role == "baseline" else ledgers
        runs.append(
            {
                "pair_index": pair,
                "position": position,
                "role": role,
                "worker": deepcopy(source[pair % len(source)]),
            }
        )
    raw: dict[str, Any] = {
        "schema_version": RAW_SCHEMA,
        "source_commit": "a" * 40,
        "run_order": [list(item) for item in EXPECTED_RUNS],
        "runs": runs,
    }
    raw["raw_hash"] = canonical_hash(raw)
    return raw


def test_mr7r_replays_frozen_workers_without_claim_drift() -> None:
    summary = derive_summary(_raw())
    assert summary["run_count"] == 10
    assert summary["pair_count"] == 5
    assert len(cast(list[object], summary["pair_metrics"])) == 5
    assert summary["performance_claimed"] is False
    assert summary["timing_open"] is False


def test_mr7r_rejects_fully_resigned_host_call_drift() -> None:
    raw = _raw()
    worker = raw["runs"][1]["worker"]
    receipt = worker["host_receipt"]
    receipt["category_calls"]["ffi_dlpack_stream"] = 56
    receipt.pop("receipt_hash")
    receipt["receipt_hash"] = canonical_hash(receipt)
    worker.pop("worker_hash")
    worker["worker_hash"] = canonical_hash(worker)
    raw.pop("raw_hash")
    raw["raw_hash"] = canonical_hash(raw)
    with pytest.raises(ValueError, match="ledger worker differs"):
        derive_summary(raw)


def test_mr7r_rejects_empty_fully_resigned_raw() -> None:
    raw: dict[str, object] = {
        "schema_version": RAW_SCHEMA,
        "source_commit": "a" * 40,
        "run_order": [],
        "runs": [],
    }
    raw["raw_hash"] = canonical_hash(raw)
    with pytest.raises(ValueError, match="raw provenance differs"):
        derive_summary(raw)
