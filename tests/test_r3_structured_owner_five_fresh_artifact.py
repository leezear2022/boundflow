"""R3-1 M0 five-fresh semantic and negative memory-gate tests."""

# pylint: disable=missing-function-docstring,protected-access

import copy
from typing import cast

import pytest
import torch

from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from scripts import run_r3_structured_owner_five_fresh_artifact as artifact


def _protocol() -> dict[str, object]:
    return {
        "source_capture_sha256": "a" * 64,
        "model_sha256": "b" * 64,
    }


def _worker(run_index: int, mode: str) -> dict[str, object]:
    lower = torch.linspace(-0.6, -0.1, 6, dtype=torch.float32).reshape(6, 1)
    gradient = torch.linspace(-0.02, 0.02, 2 * 1 * 6 * 86, dtype=torch.float32).reshape(
        2, 1, 6, 86
    )
    memory = {
        "allocated_before": 100,
        "reserved_before": 200,
        "peak_allocated": 1118 if mode == "candidate" else 1000,
        "peak_reserved": 2000,
        "peak_allocated_increment": 1018 if mode == "candidate" else 900,
        "peak_reserved_increment": 1800,
    }
    if mode == "candidate":
        receipt: dict[str, object] = {
            "execution_kind": "r3-custom-backward",
            "forward_count": 1,
            "backward_count": 1,
            "custom_backward_count": 1,
            "saved_tensor_count": 43,
            "saved_logical_bytes": 1,
            "saved_unique_storage_bytes": 1,
            "saved_dense_a_count": 0,
            "scratch_slot_count": 2,
            "alpha_version_unchanged": True,
            "beta_version_unchanged": True,
            "fallback_count": 0,
            "eager_escape_count": 0,
            "native_shadow_count": 0,
            "optimizer_mutation_count": 0,
            "production_connected": True,
            "timing_recorded": False,
            "performance_claimed": False,
            "schema_version": "boundflow.r3-1-custom-backward-receipt/v1",
            "compiled_region": False,
            "python_eager_rematerialization": True,
        }
    else:
        receipt = {
            "execution_kind": "independent-native-oracle",
            "forward_count": 1,
            "backward_count": 1,
            "custom_backward_count": 0,
            "optimizer_mutation_count": 0,
            "compiled_region": False,
            "python_eager_rematerialization": True,
            "performance_claimed": False,
        }
    return {
        "schema_version": artifact.WORKER_SCHEMA,
        "run_index": run_index,
        "mode": mode,
        "source_capture_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "source_state_hash": "c" * 64,
        "plan_hash": "d" * 64,
        "final_lower": lower,
        "compressed_alpha_gradient": gradient,
        "final_lower_sha256": production_tensor_sha256(lower),
        "compressed_alpha_gradient_sha256": production_tensor_sha256(gradient),
        "execution_receipt": receipt,
        "memory": memory,
        "alpha_versions_before": [0] * 6,
        "alpha_versions_after": [0] * 6,
        "beta_versions_before": [0] * 6,
        "beta_versions_after": [0] * 6,
        "environment": {},
        "timing_recorded": False,
        "performance_claimed": False,
    }


def _runs():  # type: ignore[no-untyped-def]
    return {
        (run_index, mode): _worker(run_index, mode)
        for run_index in range(artifact.RUN_COUNT)
        for mode in ("native", "candidate")
    }


def test_five_fresh_semantics_pass_but_memory_and_compiled_gates_close_r31() -> None:
    summary = artifact._summary(_runs(), _protocol())

    assert summary["all_semantic_passed"] is True
    assert summary["all_structure_passed"] is True
    assert summary["all_peak_allocated_passed"] is False
    assert summary["all_peak_reserved_passed"] is True
    assert summary["all_compiled_region_passed"] is False
    assert summary["worst_peak_allocated_ratio"] == pytest.approx(1.118)
    assert summary["status"] == ("validated-no-go-r3-1-m0-python-rematerialization")
    assert summary["r3_1_admitted"] is False
    assert summary["r3_2a_open"] is False
    assert summary["timing_recorded"] is False
    assert summary["performance_claimed"] is False


def test_worker_digest_and_resigned_semantic_tamper_fail_closed() -> None:
    runs = _runs()
    row = copy.deepcopy(runs[(0, "candidate")])
    row["compressed_alpha_gradient"][0, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="raw worker differs"):
        artifact._validate_worker(row, _protocol(), run_index=0, mode="candidate")

    row["compressed_alpha_gradient_sha256"] = production_tensor_sha256(
        row["compressed_alpha_gradient"]
    )
    runs[(0, "candidate")] = row
    summary = artifact._summary(runs, _protocol())
    assert summary["all_semantic_passed"] is False
    assert summary["r3_1_admitted"] is False


def test_claim_and_compiled_receipt_tamper_fail_closed() -> None:
    row = _worker(0, "candidate")
    row["performance_claimed"] = True
    with pytest.raises(ValueError, match="raw worker differs"):
        artifact._validate_worker(row, _protocol(), run_index=0, mode="candidate")

    row = _worker(0, "candidate")
    cast(dict[str, object], row["execution_receipt"])["compiled_region"] = True
    with pytest.raises(ValueError, match="candidate execution receipt differs"):
        artifact._validate_worker(row, _protocol(), run_index=0, mode="candidate")
