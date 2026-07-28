"""IR-5C3 physical batching baseline contracts."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from pathlib import Path

from boundflow.ir.plan import BackendKind
from boundflow.planner.fair_batching_measurement import (
    batched_original_observation,
    compiler_candidate_observation,
    fixed_single_observation,
    measure_batched_original,
    measure_batched_original_from_forward_trace,
    ordinary_batching_observation,
    verify_single_query_matches_batch,
)
from boundflow.planner.measured_adaptive_benchmark import (
    TypedCNNWorkloadSpec,
    TypedWorkloadSpec,
    fit_backend_calibrations,
    measure_workload,
)
from boundflow.planner.typed_benchmark_workloads import build_cnn_candidate


def test_fair_batching_measurements_share_semantics_and_normalize_per_query(
    tmp_path: Path,
) -> None:
    calibration = measure_workload(
        TypedWorkloadSpec("mlp-cal", "calibration", 1, 6, 8, 3, 95),
        (BackendKind.REFERENCE, BackendKind.PYTORCH_DENSE),
        device="cpu",
        warm_samples=1,
        cache_root=tmp_path / "calibration",
    )
    models = fit_backend_calibrations(calibration)
    workload = TypedCNNWorkloadSpec(
        "cnn-heldout",
        "heldout",
        2,
        1,
        4,
        2,
        3,
        2,
        96,
    )
    measured = measure_workload(
        workload,
        (BackendKind.REFERENCE, BackendKind.PYTORCH_DENSE),
        device="cpu",
        warm_samples=2,
        cache_root=tmp_path / "heldout",
    )
    prepared_reference = build_cnn_candidate(
        workload_id=workload.workload_id,
        backend=BackendKind.REFERENCE,
        device="cpu",
        batch=workload.batch,
        input_channels=workload.input_channels,
        image_size=workload.image_size,
        conv1_channels=workload.conv1_channels,
        conv2_channels=workload.conv2_channels,
        output_dim=workload.output_dim,
        seed=workload.seed,
    )
    original = measure_batched_original(
        prepared_reference,
        workload,
        device="cpu",
        warm_samples=2,
    )
    original_from_trace = measure_batched_original_from_forward_trace(
        prepared_reference,
        workload,
        device="cpu",
        warm_samples=2,
    )
    single = measure_workload(
        TypedCNNWorkloadSpec(
            "cnn-heldout-single",
            "heldout",
            1,
            workload.input_channels,
            workload.image_size,
            workload.conv1_channels,
            workload.conv2_channels,
            workload.output_dim,
            workload.seed,
        ),
        (BackendKind.REFERENCE,),
        device="cpu",
        warm_samples=2,
        cache_root=tmp_path / "single",
    )[0]
    prepared_single = build_cnn_candidate(
        workload_id="cnn-heldout-single",
        backend=BackendKind.REFERENCE,
        device="cpu",
        batch=1,
        input_channels=workload.input_channels,
        image_size=workload.image_size,
        conv1_channels=workload.conv1_channels,
        conv2_channels=workload.conv2_channels,
        output_dim=workload.output_dim,
        seed=workload.seed,
    )

    ordinary = ordinary_batching_observation(measured[0])
    batched = batched_original_observation(original)
    batched_from_trace = batched_original_observation(original_from_trace)
    fixed = fixed_single_observation(single)
    compiler = compiler_candidate_observation(
        measured[1], models[BackendKind.PYTORCH_DENSE]
    )

    assert original.semantic_allclose
    assert len(original.warm_per_query_latency_ms) == 2
    assert ordinary.plan_id == "ordinary-batching"
    assert batched.plan_id == "batched-original"
    assert batched_from_trace.plan_id == "batched-original-from-forward-trace"
    assert original_from_trace.semantic_allclose
    assert fixed.plan_id == "fixed-single"
    assert compiler.plan_id == "compiler:pytorch_dense"
    assert ordinary.measured_peak_bytes == measured[0].measured_peak_bytes
    assert verify_single_query_matches_batch(prepared_reference, prepared_single)[
        "semantic_allclose"
    ]
