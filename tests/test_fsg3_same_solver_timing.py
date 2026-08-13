"""Contracts and replay tests for FSG3 B0/B1/B2 same-solver timing."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from boundflow.runtime.fsg3_same_solver_timing import (
    derive_fsg3_timing_evidence,
    expected_fsg3_sequence,
    FSG3Configuration,
    FSG3EnvironmentGate,
    FSG3ExecutionCounters,
    FSG3Mode,
    FSG3SemanticResult,
    FSG3TimingMetrics,
    FSG3TimingRun,
    fsg3_timing_run_from_dict,
)


def _semantics() -> FSG3SemanticResult:
    return FSG3SemanticResult(
        status="verified",
        success=True,
        visited_domains=(6,),
        queue_before=0,
        queue_input=6,
        queue_accepted=6,
        queue_pruned=0,
        queue_after=6,
        depths=(1, 1, 1, 1, 1, 1),
        history_count=6,
        lower_shape=(2,),
        lower_values=(-0.4, -0.2),
        upper_shape=(2,),
        upper_values=(0.3, 0.5),
        final_decision=((5, 27), (5, 32), (5, 90), (5, 90), (5, 32), (5, 90)),
        split_depth=1,
        batch_size=6,
        n_verified=0,
        n_splits=6,
    )


def _metrics(configuration: FSG3Configuration, *, profile: bool) -> FSG3TimingMetrics:
    query = 1_020_000 if profile else 1_000_000
    if configuration == FSG3Configuration.B1:
        query = 1_071_000 if profile else 1_050_000
    if configuration == FSG3Configuration.B2:
        query = 816_000 if profile else 800_000
    compile_ns = 200_000 if configuration == FSG3Configuration.B2 else 0
    cold = query + compile_ns if configuration == FSG3Configuration.B2 else query
    return FSG3TimingMetrics(
        cold_total_ns=cold,
        boundflow_compile_ns=compile_ns,
        query_wall_ns=query,
        query_gpu_ns=int(query * 0.7),
        core_wall_ns=int(query * 0.5),
        core_gpu_ns=int(query * 0.4),
        post_validation_ns=10_000,
        peak_allocated_bytes=400,
        peak_reserved_bytes=500,
    )


def _execution(configuration: FSG3Configuration) -> FSG3ExecutionCounters:
    if configuration == FSG3Configuration.B0:
        return FSG3ExecutionCounters(0, 1, 24, 3, 0, "auto-lirpa", "original_provider")
    if configuration == FSG3Configuration.B1:
        return FSG3ExecutionCounters(1, 1, 24, 3, 0, "auto-lirpa", "rvir_passthrough")
    return FSG3ExecutionCounters(1, 0, 0, 0, 0, "torch-eager-reference", "whole_call")


def _runs() -> list[FSG3TimingRun]:
    rows: list[FSG3TimingRun] = []
    for index, (block, position, configuration, mode) in enumerate(
        expected_fsg3_sequence()
    ):
        rows.append(
            FSG3TimingRun(
                run_id=f"run-{index:02d}-{configuration.value}-{mode.value}",
                block_index=block,
                sequence_position=position,
                configuration=configuration,
                mode=mode,
                source_identity="source",
                protocol_identity="protocol",
                metrics=_metrics(configuration, profile=mode == FSG3Mode.PROFILE),
                semantics=_semantics(),
                execution=_execution(configuration),
                environment=FSG3EnvironmentGate(
                    gpu_uuid="GPU-1",
                    gpu_name="RTX 4060",
                    external_compute_processes=(),
                    thermal_slowdown=False,
                    worker_overlap=False,
                    device_identity_stable=True,
                    ac_powered=True,
                ),
                profile_closure_error=(0.005 if mode == FSG3Mode.PROFILE else None),
                profile_residual_share=(0.02 if mode == FSG3Mode.PROFILE else None),
            )
        )
    return rows


def test_sequence_contains_six_permutations_and_36_workers() -> None:
    sequence = expected_fsg3_sequence()
    assert len(sequence) == 36
    for configuration in FSG3Configuration:
        positions = [
            row[1] // 2
            for row in sequence
            if row[2] == configuration and row[3] == FSG3Mode.CONTROL
        ]
        assert sorted(positions) == [0, 0, 1, 1, 2, 2]


def test_raw_run_round_trips_only_canonical_payload() -> None:
    run = _runs()[0]
    payload = run.to_dict()
    assert fsg3_timing_run_from_dict(payload) == run
    cast(dict[str, object], payload["environment"])["admitted"] = False
    with pytest.raises(ValueError, match="admission projection"):
        fsg3_timing_run_from_dict(payload)


def test_configuration_specific_execution_counters_fail_closed() -> None:
    b2 = next(run for run in _runs() if run.configuration == FSG3Configuration.B2)
    with pytest.raises(ValueError, match="provider-free"):
        replace(
            b2,
            execution=replace(b2.execution, provider_compute_bounds_call_count=1),
        ).validate()


def test_passthrough_provider_counts_must_equal_original() -> None:
    runs = _runs()
    index = next(
        i
        for i, run in enumerate(runs)
        if run.block_index == 0
        and run.configuration == FSG3Configuration.B1
        and run.mode == FSG3Mode.CONTROL
    )
    runs[index] = replace(
        runs[index],
        execution=replace(
            runs[index].execution,
            provider_compute_bounds_call_count=23,
        ),
    )
    summary = derive_fsg3_timing_evidence(runs)
    assert summary["status"] == "not-auditable"
    failure_rows = cast(list[str], summary["failure_rows"])
    assert any("provider-count-differs" in item for item in failure_rows)


def test_b2_cold_scope_must_include_compile_and_query() -> None:
    b2 = next(run for run in _runs() if run.configuration == FSG3Configuration.B2)
    with pytest.raises(ValueError, match="omits compile"):
        replace(
            b2,
            metrics=replace(b2.metrics, cold_total_ns=b2.metrics.query_wall_ns),
        ).validate()


def test_control_cannot_smuggle_profile_projection() -> None:
    control = next(run for run in _runs() if run.mode == FSG3Mode.CONTROL)
    with pytest.raises(ValueError, match="control cannot"):
        replace(control, profile_closure_error=0.0).validate()


def test_replay_reports_paired_speedups_and_break_even() -> None:
    summary = derive_fsg3_timing_evidence(_runs())
    assert summary["status"] == "validated-fsg3-b0-b1-b2-baseline"
    speedups = cast(
        dict[str, dict[str, dict[str, object]]], summary["speedups_b0_over_candidate"]
    )
    assert speedups["B1"]["query_wall_ns"]["median"] == pytest.approx(1.0 / 1.05)
    assert speedups["B2"]["query_wall_ns"]["median"] == pytest.approx(1.25)
    assert summary["b2_compile_break_even_queries"] == 1
    assert summary["performance_claimed"] is False


def test_replay_rejects_order_tampering() -> None:
    runs = _runs()
    runs[0], runs[1] = runs[1], runs[0]
    with pytest.raises(ValueError, match="sequence differs"):
        derive_fsg3_timing_evidence(runs)


def test_replay_rejects_deleted_run() -> None:
    with pytest.raises(ValueError, match="run count differs"):
        derive_fsg3_timing_evidence(_runs()[:-1])


def test_semantic_numeric_tamper_makes_result_not_auditable() -> None:
    runs = _runs()
    index = next(
        i
        for i, run in enumerate(runs)
        if run.block_index == 0
        and run.configuration == FSG3Configuration.B2
        and run.mode == FSG3Mode.CONTROL
    )
    runs[index] = replace(
        runs[index],
        semantics=replace(runs[index].semantics, lower_values=(-0.3, -0.2)),
    )
    summary = derive_fsg3_timing_evidence(runs)
    assert summary["status"] == "not-auditable"
    assert summary["correctness_passed"] is False


def test_profile_semantics_are_checked_against_same_configuration_control() -> None:
    runs = _runs()
    index = next(
        i
        for i, run in enumerate(runs)
        if run.block_index == 1
        and run.configuration == FSG3Configuration.B0
        and run.mode == FSG3Mode.PROFILE
    )
    runs[index] = replace(
        runs[index],
        semantics=replace(runs[index].semantics, status="unknown"),
    )
    summary = derive_fsg3_timing_evidence(runs)
    assert summary["status"] == "not-auditable"
    failure_rows = cast(list[str], summary["failure_rows"])
    assert any("profile-control:status" in item for item in failure_rows)


def test_environment_and_profile_gates_are_recomputed() -> None:
    runs = _runs()
    runs[0] = replace(
        runs[0],
        environment=replace(runs[0].environment, external_compute_processes=("rogue",)),
    )
    profile_index = next(
        i for i, run in enumerate(runs) if run.mode == FSG3Mode.PROFILE
    )
    runs[profile_index] = replace(runs[profile_index], profile_residual_share=0.04)
    summary = derive_fsg3_timing_evidence(runs)
    assert summary["status"] == "not-auditable"
    assert summary["environment_passed"] is False
    failure_rows = cast(list[str], summary["failure_rows"])
    assert any("residual-failed" in item for item in failure_rows)


def test_profile_perturbation_is_not_used_as_headline_latency() -> None:
    runs = _runs()
    for index, run in enumerate(runs):
        if run.configuration == FSG3Configuration.B2 and run.mode == FSG3Mode.PROFILE:
            runs[index] = replace(
                run,
                metrics=replace(
                    run.metrics,
                    query_wall_ns=1_200_000,
                    cold_total_ns=1_400_000,
                    query_gpu_ns=800_000,
                    core_wall_ns=600_000,
                    core_gpu_ns=500_000,
                ),
            )
    summary = derive_fsg3_timing_evidence(runs)
    assert summary["status"] == "not-auditable"
    speedups = cast(
        dict[str, dict[str, dict[str, object]]], summary["speedups_b0_over_candidate"]
    )
    assert speedups["B2"]["query_wall_ns"]["median"] == pytest.approx(1.25)
    failure_rows = cast(list[str], summary["failure_rows"])
    assert any("profile-perturbation" in item for item in failure_rows)
