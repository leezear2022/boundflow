"""R3-1b3 synthetic gate, replay and tamper tests."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

from pathlib import Path

import pytest
import torch

from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256
from scripts import run_r3_compiled_five_fresh_artifact as artifact
from scripts.probe_r3_compiled_five_fresh_tamper import probe

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "artifacts/r3-structured-owner/r3-1b3-compiled-five-fresh-v1"


def _protocol() -> dict[str, object]:
    return {"source_capture_sha256": "a" * 64, "model_sha256": "b" * 64}


def _worker(run_index: int, mode: str) -> dict[str, object]:
    lower = torch.linspace(-0.6, -0.1, 6).reshape(6, 1)
    gradient = torch.zeros(2, 1, 6, 86)
    gradient.flatten()[:281] = torch.linspace(-0.02, -0.001, 281)
    candidate = mode == "candidate"
    receipt = (
        {
            "execution_kind": "r3-1b-compiled-custom-vjp",
            "custom_forward_count": 1,
            "custom_backward_count": 1,
            "b1_forward_launch_count": 15,
            "b1_backward_launch_count": 15,
            "b2_launch_count": 10,
            "coefficient_scratch_count": 2,
            "sign_bitmap_count": 4,
            "sign_bitmap_bytes": 43008,
            "saved_dense_a_count": 0,
            "python_visible_intermediate_coefficient_count": 0,
            "warm_dynamic_allocated_bytes": 0,
            "fallback_count": 0,
            "eager_candidate_count": 0,
            "native_shadow_count": 0,
            "compiled_vjp": True,
            "custom_vjp": True,
            "compiled_region": True,
            "timing_recorded": False,
            "performance_claimed": False,
        }
        if candidate
        else {
            "execution_kind": "independent-native-autograd",
            "forward_count": 1,
            "backward_count": 1,
            "optimizer_mutation_count": 0,
            "compiled_region": False,
        }
    )
    peak_allocated = 1000 if not candidate else 100
    peak_reserved = 2000 if not candidate else 500
    return {
        "schema_version": artifact.WORKER_SCHEMA,
        "run_index": run_index,
        "mode": mode,
        "source_capture_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "source_state_hash": "c" * 64,
        "plan_hash": artifact.EXPECTED_PLAN_HASH,
        "trace_hash": artifact.EXPECTED_TRACE_HASH,
        "final_lower": lower,
        "compressed_alpha_gradient": gradient,
        "final_lower_sha256": production_tensor_sha256(lower),
        "compressed_alpha_gradient_sha256": production_tensor_sha256(gradient),
        "execution_receipt": receipt,
        "memory": {
            "allocated_before": 50,
            "reserved_before": 100,
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
            "peak_allocated_increment": peak_allocated - 50,
            "peak_reserved_increment": peak_reserved - 100,
        },
        "alpha_versions_before": [0] * 6,
        "alpha_versions_after": [0] * 6,
        "beta_versions_before": [0] * 6,
        "beta_versions_after": [0] * 6,
        "environment": {
            "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "compute_capability": [8, 9],
        },
        "timing_recorded": False,
        "performance_claimed": False,
    }


def _runs():
    return {
        (run_index, mode): _worker(run_index, mode)
        for run_index in range(5)
        for mode in ("native", "candidate")
    }


def test_r31b3_synthetic_gate_admits_only_all_green() -> None:
    summary = artifact._summary(_runs(), _protocol())
    assert summary["status"] == "validated-r3-1b3-compiled-five-fresh"
    assert summary["worst_peak_allocated_ratio"] == pytest.approx(0.1)
    assert summary["worst_peak_reserved_ratio"] == pytest.approx(0.25)
    assert summary["r3_1_admitted"] is True
    assert summary["r3_2a_open"] is True
    assert summary["performance_claimed"] is False


def test_r31b3_memory_over_one_is_no_go() -> None:
    runs = _runs()
    candidate = runs[(0, "candidate")]
    candidate["memory"]["peak_allocated"] = 1001
    candidate["memory"]["peak_allocated_increment"] = 951
    summary = artifact._summary(runs, _protocol())
    assert summary["status"] == "validated-no-go-r3-1b3-correctness-memory"
    assert summary["r3_1_admitted"] is False
    assert summary["r3_2a_open"] is False


def test_r31b3_formal_replay_and_tamper() -> None:
    if not FORMAL.is_dir():
        pytest.skip("R3-1b3 formal artifact is not generated yet")
    result = artifact.replay(FORMAL)
    assert result["status"] == "replay-passed"
    tamper = probe(FORMAL)
    assert tamper["probe_count"] == tamper["rejected_count"] == 9
