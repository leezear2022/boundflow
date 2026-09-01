"""R3-3 active-beta isolated wrapper timing primitives."""

# pylint: disable=missing-function-docstring

from pathlib import Path

import pytest
import torch

from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.r3_3_active_beta_timing import (
    PreparedR3ActiveBetaTimingV1,
    compare_r3_active_beta_executions_v1,
    cuda_event_wrapper_ms_v1,
    measure_r3_active_beta_memory_v1,
)

ARTIFACT = Path("artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1")


def _capture():  # type: ignore[no-untyped-def]
    payload = torch.load(ARTIFACT / "run_00.pt", map_location="cpu", weights_only=False)
    return production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][0]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_r3_3_active_beta_baseline_candidate_wrapper_parity() -> None:
    prepared = PreparedR3ActiveBetaTimingV1(_capture())
    parity = compare_r3_active_beta_executions_v1(
        prepared.baseline_once(), prepared.candidate_once()
    )
    assert parity.allclose is True
    assert parity.sign_exact is True
    assert parity.maximum_absolute_difference <= 2.0e-4
    assert parity.element_count == 6 * 27 + 6 + 6 + 6 * 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_r3_3_active_beta_timing_and_memory_observations_are_physical() -> None:
    prepared = PreparedR3ActiveBetaTimingV1(_capture())
    baseline_ms = cuda_event_wrapper_ms_v1(prepared.baseline_once)
    candidate_ms = cuda_event_wrapper_ms_v1(prepared.candidate_once)
    baseline_memory = measure_r3_active_beta_memory_v1(prepared.baseline_once)
    candidate_memory = measure_r3_active_beta_memory_v1(prepared.candidate_once)
    assert baseline_ms > 0.0
    assert candidate_ms > 0.0
    assert baseline_memory.peak_allocated_bytes >= baseline_memory.base_allocated_bytes
    assert (
        candidate_memory.peak_allocated_bytes >= candidate_memory.base_allocated_bytes
    )
    assert baseline_memory.incremental_allocated_bytes > 0
    assert candidate_memory.incremental_allocated_bytes > 0
