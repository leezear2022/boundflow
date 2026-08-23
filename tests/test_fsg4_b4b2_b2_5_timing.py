"""Tests for the B4-B2 B2-5 formal micro-physics layer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.fsg4_b4b2_sparse_conv_timing import (
    B2_5_PAIR_COUNT,
    B2_5_WARMUP_COUNT,
    PreparedSparseConvTimingV1,
    compare_sparse_conv_executions_v1,
)
from scripts.run_fsg4_b4b2_b2_5_artifact import (
    TIMING_ORDERS,
    derive_summary,
)

CAPTURE = Path("artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1")


def _capture():
    payload = torch.load(CAPTURE / "run_00.pt", map_location="cpu", weights_only=False)
    return production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][1]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="B2-5 requires CUDA")
def test_b2_5_wrapper_baseline_matches_tir_and_counts_real_kernels() -> None:
    prepared = PreparedSparseConvTimingV1(_capture(), candidate_ordinal=0)
    parity = compare_sparse_conv_executions_v1(
        prepared.baseline_once(), prepared.candidate_once()
    )
    assert parity.allclose is True
    assert parity.sign_exact is True
    assert parity.maximum_absolute_difference <= 2.0e-4
    assert prepared.kernel_inventory.forward_kernel_count == 3
    assert prepared.kernel_inventory.backward_kernel_count == 3
    assert prepared.kernel_inventory.total_kernel_count == 6
    assert prepared.kernel_inventory.shared_memory_token_count == 0


def _fake_inputs(speedups: list[float]):
    kernel = {
        "kernel_names": ["forward", "backward"],
        "forward_kernel_count": 1,
        "backward_kernel_count": 1,
        "total_kernel_count": 2,
        "shared_memory_token_count": 0,
        "vector_token_count": 0,
        "half_token_count": 0,
    }
    rows = [
        {
            "candidate_ordinal": ordinal,
            "median_ms": float(ordinal + 2),
            "schedule_hash": f"{ordinal:064x}",
            "module_receipt_hash": f"{ordinal + 20:064x}",
            "kernel_inventory": kernel,
        }
        for ordinal in range(12)
    ]
    calibration = {
        "candidate_count": 12,
        "rows": rows,
        "winner_ordinal": 0,
        "winner_schedule_hash": rows[0]["schedule_hash"],
        "winner_module_receipt_hash": rows[0]["module_receipt_hash"],
        "winner_selected_from_raw": True,
        "performance_claimed": False,
    }
    correctness = []
    for anchor in ("S", "P"):
        for ordinal in range(5):
            correctness.append(
                {
                    "anchor": anchor,
                    "run_ordinal": ordinal,
                    "semantic_passed": True,
                    "fallback_count": 0,
                    "eager_backward_count": 0,
                    "module_call_count": {"forward": 1, "backward": 1},
                    "maximum_absolute_difference": 1.0e-6,
                }
            )
    timings = []
    for ordinal, speedup in enumerate(speedups):
        pairs = [
            {
                "baseline_ms": speedup,
                "candidate_ms": 1.0,
                "speedup": speedup,
            }
            for _ in range(B2_5_PAIR_COUNT)
        ]
        timings.append(
            {
                "run_ordinal": ordinal,
                "order": TIMING_ORDERS[ordinal],
                "candidate_ordinal": 0,
                "warmups_per_side": B2_5_WARMUP_COUNT,
                "pair_count": B2_5_PAIR_COUNT,
                "pairs": pairs,
                "baseline_median_ms": speedup,
                "candidate_median_ms": 1.0,
                "paired_speedup": speedup,
                "allocated_ratio": 1.0,
                "reserved_ratio": 1.0,
                "parity": {"allclose": True, "sign_exact": True},
                "kernel_inventory": kernel,
                "module_call_count": {"forward": 1, "backward": 1},
                "fallback_count": 0,
                "eager_backward_count": 0,
                "performance_claimed": False,
            }
        )
    return calibration, correctness, timings


def test_b2_5_summary_admits_only_all_physical_gates() -> None:
    admitted = derive_summary(*_fake_inputs([1.10] * 6))
    assert admitted["timing_admitted"] is True
    assert admitted["b4b3_open"] is True
    assert admitted["status"] == "validated-b4-b2-typed-cuda-tir-candidate"

    rejected = derive_summary(*_fake_inputs([0.80] * 6))
    assert rejected["timing_admitted"] is False
    assert rejected["b4b3_open"] is False
    assert rejected["status"] == "validated-no-go-b4-b2-v1-physics"
