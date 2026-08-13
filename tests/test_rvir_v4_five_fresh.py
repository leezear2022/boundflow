"""Contracts for RVIR-v4 V4-3E five-fresh correctness."""

# pylint: disable=missing-function-docstring,protected-access

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, cast

import pytest
import torch

from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from scripts import run_rvir_v4_five_fresh_artifact as artifact_runner
from scripts import run_rvir_v4_five_fresh_worker as worker

ORIGINAL = Path("artifacts/rvir-v4-whole-core-truth/resnet2b-core-v1/truth.pt")
CANDIDATE = Path("artifacts/rvir-v4-live-return/resnet2b-core-v1/live_result.pt")


def _one_tensor(mapping: Mapping[str, Any]) -> torch.Tensor:
    record = next(iter(mapping.values()))
    assert isinstance(record, Mapping) and torch.is_tensor(record["value"])
    return cast(torch.Tensor, record["value"])


def _augment(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    value = deepcopy(payload)
    core_key = "whole_core_truths" if mode == "original" else "whole_core_results"
    post_key = "whole_post_truths" if mode == "original" else "whole_post_results"
    core = value[core_key][0]
    post = value[post_key][0]
    lower = _one_tensor(post["lower_bounds"])
    upper = _one_tensor(post["upper_bounds"])
    thresholds = core["fields"]["thresholds"]["value"]
    value["five_fresh_mode"] = mode
    value["five_fresh_worker_schema"] = worker.WORKER_SCHEMA
    value["queue_events"] = [
        {
            "schema_version": worker.WORKER_SCHEMA,
            "before_domain_count": 0,
            "input_domain_count": 6,
            "accepted_domain_count": 6,
            "pruned_domain_count": 0,
            "after_domain_count": 6,
            "final_name": "/49",
            "lower_sha256": production_tensor_sha256(lower),
            "upper_sha256": production_tensor_sha256(upper),
            "thresholds_sha256": production_tensor_sha256(thresholds),
            "history_count": 6,
            "depths": [1, 1, 1, 1, 1, 1],
            "performance_claimed": False,
        }
    ]
    return value


def _fixtures() -> tuple[dict[str, Any], dict[str, Any]]:
    original = torch.load(ORIGINAL, map_location="cpu", weights_only=True)
    candidate = torch.load(CANDIDATE, map_location="cpu", weights_only=True)
    assert isinstance(original, dict) and isinstance(candidate, dict)
    return _augment(original, "original"), _augment(candidate, "candidate")


def test_five_fresh_contract_admits_only_all_five_complete_pairs() -> None:
    original, candidate = _fixtures()
    summary = artifact_runner._summary(
        [
            original if mode == "original" else candidate
            for mode in artifact_runner.SEQUENCE
        ]
    )

    assert summary["sequence"] == list(artifact_runner.SEQUENCE)
    assert summary["run_count"] == 10
    assert summary["pair_count"] == 5
    assert summary["all_pairs_admitted"] is True
    assert summary["tensor_comparison_count"] == 2255
    assert summary["sign_element_comparison_count"] == 1065300
    assert summary["maximum_absolute_difference"] < 2e-4
    assert summary["all_sign_exact"] is True
    assert summary["accepted_domain_count_per_run"] == 6
    assert summary["pruned_domain_count_per_run"] == 0
    assert summary["five_fresh_correctness_admitted"] is True
    assert summary["b2_same_solver_timing_admitted"] is True
    assert summary["performance_claimed"] is False


def test_five_fresh_contract_rejects_queue_and_callback_drift() -> None:
    original, candidate = _fixtures()
    changed_queue = deepcopy(candidate)
    changed_queue["queue_events"][0]["accepted_domain_count"] = 5
    with pytest.raises(ValueError, match="queue accounting differs"):
        artifact_runner._summary(
            [
                original if mode == "original" else changed_queue
                for mode in artifact_runner.SEQUENCE
            ]
        )

    changed_callback = deepcopy(candidate)
    changed_callback["provider_update_bounds_callback_count"] = 1
    with pytest.raises(ValueError, match="atomic receipt differs"):
        artifact_runner._summary(
            [
                original if mode == "original" else changed_callback
                for mode in artifact_runner.SEQUENCE
            ]
        )
