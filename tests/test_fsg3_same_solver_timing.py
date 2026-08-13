"""Contracts and replay tests for FSG3 B0/B1/B2 same-solver timing."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
from typing import cast

import pytest

from boundflow.runtime.fsg3_same_solver_timing import (
    derive_fsg3_timing_evidence,
    expected_fsg3_sequence,
    FSG3Configuration,
    FSG3EnvironmentGate,
    FSG3ExecutionCounters,
    FSG3Mode,
    FSG3ProfileSpan,
    FSG3SemanticResult,
    FSG3TimingMetrics,
    FSG3TimingRun,
    fsg3_timing_run_from_dict,
)
from scripts import run_fsg3_same_solver_experiment as experiment_runner


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
        upper_positive_infinity_mask=(False, False),
        final_decision=((5, 27), (5, 32), (5, 90), (5, 90), (5, 32), (5, 90)),
        split_depth=1,
        batch_size=6,
        n_verified=0,
        n_splits=6,
    )


def _idle_gpu_snapshot() -> dict[str, object]:
    return {
        "temperature": "49 C",
        "sw_thermal_slowdown": "Not Active",
        "sw_power_cap": "Not Active",
        "hw_thermal_slowdown": "Not Active",
        "sw_thermal_slowdown_counter_us": 0,
        "sw_power_cap_counter_us": 0,
        "hw_thermal_slowdown_counter_us": 0,
    }


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


def _profile_spans(
    configuration: FSG3Configuration, metrics: FSG3TimingMetrics
) -> tuple[FSG3ProfileSpan, ...]:
    layouts = {
        FSG3Configuration.B0: (("core", "provider_core"),),
        FSG3Configuration.B1: (
            ("core", "typed_pre_state"),
            ("core", "provider_core"),
        ),
        FSG3Configuration.B2: (
            ("compile", "compile"),
            ("core", "typed_pre_state"),
            ("core", "optimizer"),
            ("core", "backward"),
            ("core", "kfsb"),
            ("core", "atomic_commit"),
        ),
    }
    core_names = [name for scope, name in layouts[configuration] if scope == "core"]
    base, remainder = divmod(metrics.core_wall_ns, len(core_names))
    core_durations = {
        name: base + (1 if index < remainder else 0)
        for index, name in enumerate(core_names)
    }
    rows: list[FSG3ProfileSpan] = []
    offset = 0
    for scope, name in layouts[configuration] + (("post", "official_post_queue"),):
        duration = (
            metrics.boundflow_compile_ns
            if scope == "compile"
            else core_durations[name] if scope == "core" else 10_000
        )
        rows.append(
            FSG3ProfileSpan(
                scope=scope,
                name=name,
                stack_layer="test/layer",
                solver_phase="test_phase",
                resource="host" if scope == "compile" else "host+cuda",
                cache_state="cold" if scope == "compile" else "process-hit",
                start_offset_ns=offset,
                end_offset_ns=offset + duration,
                wall_ns=duration,
                gpu_ns=0 if scope == "compile" else max(1, duration // 2),
            )
        )
        offset += duration + 1_000
    return tuple(rows)


def _runs() -> list[FSG3TimingRun]:
    rows: list[FSG3TimingRun] = []
    for index, (block, position, configuration, mode) in enumerate(
        expected_fsg3_sequence()
    ):
        metrics = _metrics(configuration, profile=mode == FSG3Mode.PROFILE)
        rows.append(
            FSG3TimingRun(
                run_id=f"run-{index:02d}-{configuration.value}-{mode.value}",
                block_index=block,
                sequence_position=position,
                configuration=configuration,
                mode=mode,
                source_identity="source",
                protocol_identity="protocol",
                metrics=metrics,
                semantics=_semantics(),
                execution=_execution(configuration),
                environment=FSG3EnvironmentGate(
                    gpu_uuid="GPU-1",
                    gpu_name="RTX 4060",
                    runtime_identity="runtime",
                    external_compute_processes=(),
                    software_thermal_signal=False,
                    software_power_cap_signal=False,
                    software_thermal_power_counters_coupled=False,
                    hardware_thermal_slowdown=False,
                    worker_overlap=False,
                    device_identity_stable=True,
                    ac_powered=True,
                ),
                profile_spans=(
                    _profile_spans(configuration, metrics)
                    if mode == FSG3Mode.PROFILE
                    else ()
                ),
                profile_closure_error=(0.0 if mode == FSG3Mode.PROFILE else None),
                profile_residual_share=(0.0 if mode == FSG3Mode.PROFILE else None),
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


def test_lower_only_positive_infinity_upper_uses_canonical_mask() -> None:
    semantic = replace(
        _semantics(),
        upper_values=(0.0, 0.0),
        upper_positive_infinity_mask=(True, True),
    )
    semantic.validate()
    assert semantic.to_dict()["upper_positive_infinity_mask"] == [True, True]
    with pytest.raises(ValueError, match="placeholder differs"):
        replace(semantic, upper_values=(1.0, 0.0)).validate()


def test_control_cannot_smuggle_profile_projection() -> None:
    control = next(run for run in _runs() if run.mode == FSG3Mode.CONTROL)
    with pytest.raises(ValueError, match="control cannot"):
        replace(control, profile_closure_error=0.0).validate()


def test_profile_spans_are_bound_and_closure_is_recomputed() -> None:
    profile = next(run for run in _runs() if run.mode == FSG3Mode.PROFILE)
    with pytest.raises(ValueError, match="closure projection"):
        replace(profile, profile_closure_error=0.001).validate()
    with pytest.raises(ValueError, match="span layout"):
        replace(profile, profile_spans=profile.profile_spans[:-1]).validate()
    first = profile.profile_spans[0]
    with pytest.raises(ValueError, match="wall projection"):
        replace(
            profile,
            profile_spans=(replace(first, wall_ns=first.wall_ns + 1),)
            + profile.profile_spans[1:],
        ).validate()


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


def test_experiment_derivations_preserve_all_pairs_spans_and_closures() -> None:
    runs = _runs()
    paired = experiment_runner._paired_rows(runs)
    spans = experiment_runner._profile_rows(runs)
    closure = experiment_runner._closure(runs)

    assert len(paired) == 12
    assert len(spans) == 72
    assert len(cast(list[object], closure["rows"])) == 18
    assert closure["all_closed"] is True
    assert all(row["performance_claimed"] is False for row in paired)
    b2 = next(
        row for row in paired if row["block_index"] == 0 and row["candidate"] == "B2"
    )
    assert cast(dict[str, float], b2["speedup_b0_over_candidate"])[
        "query_wall_ns"
    ] == pytest.approx(1.25)


def test_formal_preflight_admission_is_recomputed() -> None:
    sample = {
        "temperature_limit_celsius": 50,
        "poll_seconds": 5,
        "timeout_seconds": 900,
        "sample_count": 1,
        "wait_ns": 1,
        "samples": [
            {
                "elapsed_ns": 0,
                "temperature_celsius": 49,
                "independent_thermal_active": False,
                "gpu_snapshot": _idle_gpu_snapshot(),
                "compute_processes": [
                    {
                        "pid": 1,
                        "name": "/usr/bin/kwin_wayland",
                        "used_memory_mib": 7,
                    }
                ],
                "ac_powered": True,
            }
        ],
        "admitted": True,
    }
    experiment_runner._validate_formal_preflight(sample)
    cast(dict[str, object], cast(list[object], sample["samples"])[0])[
        "temperature_celsius"
    ] = 51
    with pytest.raises(ValueError, match="admission differs"):
        experiment_runner._validate_formal_preflight(sample)


def test_formal_preflight_accepts_only_exact_coupled_software_alias() -> None:
    snapshot = _idle_gpu_snapshot()
    snapshot.update(
        {
            "sw_thermal_slowdown": "Active",
            "sw_power_cap": "Active",
            "sw_thermal_slowdown_counter_us": 42,
            "sw_power_cap_counter_us": 42,
        }
    )
    sample = {
        "temperature_limit_celsius": 50,
        "poll_seconds": 5,
        "timeout_seconds": 900,
        "sample_count": 1,
        "wait_ns": 1,
        "samples": [
            {
                "elapsed_ns": 0,
                "temperature_celsius": 49,
                "independent_thermal_active": False,
                "gpu_snapshot": snapshot,
                "compute_processes": [],
                "ac_powered": True,
            }
        ],
        "admitted": True,
    }
    experiment_runner._validate_formal_preflight(sample)
    snapshot["sw_power_cap_counter_us"] = 41
    with pytest.raises(ValueError, match="thermal projection differs"):
        experiment_runner._validate_formal_preflight(sample)


def test_parent_timeout_outlives_worker_post_init_preflight() -> None:
    assert experiment_runner.WORKER_SUBPROCESS_TIMEOUT_SECONDS >= (
        experiment_runner.worker.WORKER_PREFLIGHT_TIMEOUT_SECONDS + 180
    )


def test_worker_timeout_preserves_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(
            cmd=("python", "worker.py"),
            timeout=experiment_runner.WORKER_SUBPROCESS_TIMEOUT_SECONDS,
            output=b"partial stdout\n",
            stderr=b"partial stderr\n",
        )

    def host_snapshot() -> dict[str, int]:
        return {"monotonic_ns": 1}

    monkeypatch.setattr(experiment_runner.subprocess, "run", timeout)
    monkeypatch.setattr(experiment_runner, "_host_snapshot", host_snapshot)
    with pytest.raises(RuntimeError, match="timed out"):
        experiment_runner._run_worker(
            artifact=tmp_path,
            index=3,
            command=("python", "worker.py"),
        )
    failure = json.loads((tmp_path / "failed_worker.json").read_text())
    assert failure["index"] == 3
    assert failure["timed_out"] is True
    assert failure["performance_claimed"] is False
    assert (tmp_path / "logs/run_03.stdout.txt").read_text() == "partial stdout\n"
    assert (tmp_path / "logs/run_03.stderr.txt").read_text() == "partial stderr\n"


def test_worker_nonzero_exit_preserves_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=("python", "worker.py"),
            returncode=2,
            stdout="worker stdout\n",
            stderr="worker stderr\n",
        )

    def host_snapshot() -> dict[str, int]:
        return {"monotonic_ns": 1}

    monkeypatch.setattr(experiment_runner.subprocess, "run", failed)
    monkeypatch.setattr(experiment_runner, "_host_snapshot", host_snapshot)
    with pytest.raises(RuntimeError, match="failed with 2"):
        experiment_runner._run_worker(
            artifact=tmp_path,
            index=4,
            command=("python", "worker.py"),
        )
    failure = json.loads((tmp_path / "failed_worker.json").read_text())
    assert failure["index"] == 4
    assert failure["returncode"] == 2
    assert failure["timed_out"] is False
    assert failure["performance_claimed"] is False
    assert (tmp_path / "logs/run_04.stdout.txt").read_text() == "worker stdout\n"
    assert (tmp_path / "logs/run_04.stderr.txt").read_text() == "worker stderr\n"


def test_artifact_environment_gate_is_recomputed_from_raw_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def ac_powered() -> bool:
        return True

    monkeypatch.setattr(experiment_runner.worker, "_ac_powered", ac_powered)
    runtime_environment: dict[str, object] = {"runtime": "test"}
    runtime_identity = experiment_runner.canonical_hash(runtime_environment)
    run = replace(
        _runs()[0],
        environment=replace(
            _runs()[0].environment,
            runtime_identity=runtime_identity,
        ),
    )
    snapshot = {
        **_idle_gpu_snapshot(),
        "uuid": run.environment.gpu_uuid,
        "name": run.environment.gpu_name,
    }
    worker_pid = 1234
    processes = [{"pid": worker_pid, "name": "/python", "used_memory_mib": 100}]
    worker_preflight = {
        "worker_pid": worker_pid,
        "temperature_limit_celsius": 50,
        "poll_seconds": 5,
        "timeout_seconds": 900,
        "sample_count": 1,
        "wait_ns": 1,
        "samples": [
            {
                "elapsed_ns": 0,
                "temperature_celsius": 50,
                "independent_thermal_active": False,
                "gpu_snapshot": snapshot,
                "compute_processes": processes,
                "ac_powered": True,
            }
        ],
        "admitted": True,
    }
    envelope: dict[str, object] = {
        "run": run.to_dict(),
        "diagnostics": {
            "runtime_environment": runtime_environment,
            "environment_before": snapshot,
            "environment_after": snapshot,
            "compute_processes_before": processes,
            "compute_processes_after": processes,
            "worker_preflight": worker_preflight,
        },
    }
    outer = {"host_before": {}, "host_after": {}}
    environment = experiment_runner._environment([envelope], [outer])
    assert environment["all_workers_environment_admitted"] is True

    gate = cast(
        dict[str, object], cast(dict[str, object], envelope["run"])["environment"]
    )
    gate["software_thermal_signal"] = True
    gate["software_power_cap_signal"] = True
    gate["software_thermal_power_counters_coupled"] = True
    with pytest.raises(ValueError, match="raw environment gate projection differs"):
        experiment_runner._environment([envelope], [outer])


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
    profile = runs[profile_index]
    core_index = next(
        index
        for index, span in enumerate(profile.profile_spans)
        if span.scope == "core"
    )
    core_span = profile.profile_spans[core_index]
    shortened = replace(
        core_span,
        end_offset_ns=core_span.end_offset_ns - 40_000,
        wall_ns=core_span.wall_ns - 40_000,
    )
    spans = list(profile.profile_spans)
    spans[core_index] = shortened
    residual = 40_000 / profile.metrics.core_wall_ns
    runs[profile_index] = replace(
        profile,
        profile_spans=tuple(spans),
        profile_closure_error=residual,
        profile_residual_share=residual,
    )
    summary = derive_fsg3_timing_evidence(runs)
    assert summary["status"] == "not-auditable"
    assert summary["environment_passed"] is False
    failure_rows = cast(list[str], summary["failure_rows"])
    assert any("residual-failed" in item for item in failure_rows)


def test_exact_coupled_power_thermal_signal_is_not_independent_thermal() -> None:
    environment = _runs()[0].environment
    coupled = replace(
        environment,
        software_thermal_signal=True,
        software_power_cap_signal=True,
        software_thermal_power_counters_coupled=True,
    )
    coupled.validate()
    assert coupled.independent_thermal_slowdown is False
    assert coupled.admitted is True
    independent = replace(coupled, software_thermal_power_counters_coupled=False)
    assert independent.independent_thermal_slowdown is True
    assert independent.admitted is False


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
