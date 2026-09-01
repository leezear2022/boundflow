"""Contracts for the FSG4 B0/B2/B3 formal timing schema."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

from argparse import Namespace
from dataclasses import replace
import json
from pathlib import Path
from typing import cast

import pytest

from boundflow.runtime.fsg3_same_solver_timing import (
    FSG3EnvironmentGate,
    FSG3Mode,
    FSG3ProfileSpan,
    FSG3SemanticResult,
    FSG3TimingMetrics,
)
from boundflow.runtime.fsg4_b3_explicit_counters import (
    COUNTER_NAMES,
    EXPECTED_B2_FIXED_COUNTERS,
    EXPECTED_B3C_FIXED_COUNTERS,
)
from boundflow.runtime.fsg4_b3_same_solver_timing import (
    derive_fsg4_b3_timing_evidence,
    expected_fsg4_b3_sequence,
    FSG4_B3_PROFILE_SPAN_LAYOUT,
    FSG4B3ActivationReceipt,
    FSG4B3ExecutionCounters,
    FSG4B3TimingConfiguration,
    FSG4B3TimingRun,
    fsg4_b3_timing_run_from_dict,
)
from scripts import run_fsg4_b3_same_solver_experiment as experiment
from scripts import probe_fsg4_b3_same_solver_artifact_tamper as tamper


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


def _metrics(
    configuration: FSG4B3TimingConfiguration, *, profile: bool
) -> FSG3TimingMetrics:
    query = {
        FSG4B3TimingConfiguration.B0: 1_000_000,
        FSG4B3TimingConfiguration.B2: 1_200_000,
        FSG4B3TimingConfiguration.B3: 800_000,
    }[configuration]
    if profile:
        query = int(query * 1.02)
    compile_ns = 0 if configuration == FSG4B3TimingConfiguration.B0 else 200_000
    core = int(query * 0.5)
    return FSG3TimingMetrics(
        cold_total_ns=query + compile_ns,
        boundflow_compile_ns=compile_ns,
        query_wall_ns=query,
        query_gpu_ns=int(query * 0.7),
        core_wall_ns=core,
        core_gpu_ns=int(query * 0.4),
        post_validation_ns=10_000,
        peak_allocated_bytes=400,
        peak_reserved_bytes=500,
    )


def _detailed_counts(configuration: FSG4B3TimingConfiguration) -> dict[str, int]:
    counts = {name: 0 for name in COUNTER_NAMES}
    counts.update(
        EXPECTED_B2_FIXED_COUNTERS
        if configuration == FSG4B3TimingConfiguration.B2
        else EXPECTED_B3C_FIXED_COUNTERS
    )
    for name in (
        "tensor_content_hash_count",
        "gpu_tensor_content_hash_count",
        "typed_validate_call_count",
        "stable_hash_call_count",
    ):
        counts[name] = max(counts[name], 1)
    return counts


def _activation(
    configuration: FSG4B3TimingConfiguration, mode: FSG3Mode
) -> FSG4B3ActivationReceipt:
    detailed = (
        _detailed_counts(configuration)
        if mode == FSG3Mode.PROFILE and configuration != FSG4B3TimingConfiguration.B0
        else None
    )
    if configuration == FSG4B3TimingConfiguration.B0:
        receipt = (0, 0, 0, 0, 0, 0)
    elif configuration == FSG4B3TimingConfiguration.B2:
        receipt = (0, 0, 0, 1, 1, 0)
    else:
        receipt = (1, 1, 1, 1, 1, 1)
    return FSG4B3ActivationReceipt(
        prepared_core_template_count=receipt[0],
        prepared_core_instance_count=receipt[1],
        terminal_optimizer_schedule_count=receipt[2],
        assembly_count=receipt[3],
        commit_receipt_count=receipt[4],
        device_commit_audit_count=receipt[5],
        post_query_audit_ns=(1 if configuration == FSG4B3TimingConfiguration.B3 else 0),
        post_query_audit_excluded_from_timing=True,
        headline_content_digest_count=(
            0 if configuration == FSG4B3TimingConfiguration.B3 else None
        ),
        candidate_d2h_copy_count=(
            0 if configuration == FSG4B3TimingConfiguration.B3 else None
        ),
        detailed_counts_by_name=(
            None if detailed is None else tuple(sorted(detailed.items()))
        ),
    )


def _execution(
    configuration: FSG4B3TimingConfiguration,
) -> FSG4B3ExecutionCounters:
    if configuration == FSG4B3TimingConfiguration.B0:
        return FSG4B3ExecutionCounters(
            0, 1, 24, 3, 0, "auto-lirpa", "original_provider"
        )
    return FSG4B3ExecutionCounters(
        1,
        0,
        0,
        0,
        0,
        "torch-eager-reference",
        (
            "whole_call_reference"
            if configuration == FSG4B3TimingConfiguration.B2
            else "b3_ir_graph_plan_schedule"
        ),
    )


def _spans(
    configuration: FSG4B3TimingConfiguration, metrics: FSG3TimingMetrics
) -> tuple[FSG3ProfileSpan, ...]:
    layout = FSG4_B3_PROFILE_SPAN_LAYOUT[configuration]
    core_names = [name for scope, name in layout if scope == "core"]
    base, remainder = divmod(metrics.core_wall_ns, len(core_names))
    core_durations = {
        name: base + (index < remainder) for index, name in enumerate(core_names)
    }
    rows: list[FSG3ProfileSpan] = []
    offset = 0
    for scope, name in layout:
        duration = (
            metrics.boundflow_compile_ns
            if scope == "compile"
            else int(core_durations[name]) if scope == "core" else 10_000
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


def _runs() -> list[FSG4B3TimingRun]:
    rows: list[FSG4B3TimingRun] = []
    environment = FSG3EnvironmentGate(
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
    )
    for index, (block, position, configuration, mode) in enumerate(
        expected_fsg4_b3_sequence()
    ):
        metrics = _metrics(configuration, profile=mode == FSG3Mode.PROFILE)
        spans = _spans(configuration, metrics) if mode == FSG3Mode.PROFILE else ()
        covered = sum(span.wall_ns for span in spans if span.scope == "core")
        rows.append(
            FSG4B3TimingRun(
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
                environment=environment,
                activation=_activation(configuration, mode),
                profile_spans=spans,
                profile_closure_error=(
                    abs(metrics.core_wall_ns - covered) / metrics.core_wall_ns
                    if spans
                    else None
                ),
                profile_residual_share=(
                    max(metrics.core_wall_ns - covered, 0) / metrics.core_wall_ns
                    if spans
                    else None
                ),
            )
        )
    return rows


def test_sequence_has_six_permutations_and_36_workers() -> None:
    sequence = expected_fsg4_b3_sequence()
    assert len(sequence) == 36
    for configuration in FSG4B3TimingConfiguration:
        positions = [
            position // 2
            for _block, position, observed, mode in sequence
            if observed == configuration and mode == FSG3Mode.CONTROL
        ]
        assert sorted(positions) == [0, 0, 1, 1, 2, 2]


def test_raw_run_round_trip_recomputes_activation() -> None:
    run = next(
        item
        for item in _runs()
        if item.configuration == FSG4B3TimingConfiguration.B3
        and item.mode == FSG3Mode.PROFILE
    )
    payload = run.to_dict()
    assert fsg4_b3_timing_run_from_dict(payload) == run
    activation = cast(dict[str, object], payload["activation"])
    activation["prepared_core_template_count"] = 0
    with pytest.raises(ValueError, match="activation receipt"):
        fsg4_b3_timing_run_from_dict(payload)


def test_control_cannot_smuggle_profile_counters() -> None:
    run = next(
        item
        for item in _runs()
        if item.configuration == FSG4B3TimingConfiguration.B3
        and item.mode == FSG3Mode.CONTROL
    )
    with pytest.raises(ValueError, match="profile counter admission"):
        replace(
            run,
            activation=replace(
                run.activation,
                detailed_counts_by_name=tuple(
                    sorted(_detailed_counts(run.configuration).items())
                ),
            ),
        ).validate()


def test_profile_physical_counter_mismatch_fails_closed() -> None:
    run = next(
        item
        for item in _runs()
        if item.configuration == FSG4B3TimingConfiguration.B3
        and item.mode == FSG3Mode.PROFILE
    )
    counts = cast(dict[str, int], run.activation.detailed_counts)
    counts["forward_trace_build_count"] = 5
    with pytest.raises(ValueError, match="profile counter differs"):
        replace(
            run,
            activation=replace(
                run.activation, detailed_counts_by_name=tuple(sorted(counts.items()))
            ),
        ).validate()


def test_replay_applies_preregistered_b3_decision() -> None:
    summary = derive_fsg4_b3_timing_evidence(_runs())
    assert summary["status"] == "validated-b3"
    decision = cast(dict[str, float], summary["decision_inputs"])
    assert decision["b2_over_b3_core_geomean"] == pytest.approx(1.5)
    assert decision["b0_over_b3_query_geomean"] == pytest.approx(1.25)
    assert summary["performance_claimed"] is False


def test_replay_no_go_is_not_relabelled_as_failure() -> None:
    runs = _runs()
    for index, run in enumerate(runs):
        if run.configuration == FSG4B3TimingConfiguration.B3:
            metrics = replace(
                run.metrics,
                query_wall_ns=1_250_000,
                query_gpu_ns=875_000,
                core_wall_ns=625_000,
                core_gpu_ns=500_000,
                cold_total_ns=1_450_000,
            )
            spans = _spans(run.configuration, metrics) if run.profile_spans else ()
            runs[index] = replace(
                run,
                metrics=metrics,
                profile_spans=spans,
                profile_closure_error=(0.0 if spans else None),
                profile_residual_share=(0.0 if spans else None),
            )
    summary = derive_fsg4_b3_timing_evidence(runs)
    assert summary["status"] == "validated-no-go-b3"
    assert summary["measurement_auditable"] is True


def test_experiment_derivations_bind_incremental_pairs_and_activation() -> None:
    runs = _runs()
    paired = experiment._paired_rows(runs)
    activations = experiment._activation_rows(runs)
    closure = experiment._closure(runs)

    assert len(paired) == 18
    assert len(activations) == 36
    assert len(cast(list[object], closure["rows"])) == 18
    assert closure["all_closed"] is True
    incremental = next(
        row
        for row in paired
        if row["block_index"] == 0
        and row["numerator"] == "B2"
        and row["denominator"] == "B3"
    )
    assert cast(dict[str, float], incremental["ratio"])[
        "core_wall_ns"
    ] == pytest.approx(1.5)


def test_protocol_contains_no_host_local_absolute_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    args = Namespace(
        benchmark_root=root.parent / "vnncomp2021",
        abcrown_root=root.parent / "alpha-beta-CROWN",
        abcrown_python=Path("/users/local/miniconda3/envs/alpha-beta-crown/bin/python"),
        model=(
            root.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
        ),
        property=(
            root.parent
            / "vnncomp2021/benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered"
            / "resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"
        ),
    )
    protocol = experiment._protocol(args)
    experiment._validate_protocol(protocol)
    encoded = json.dumps(protocol, sort_keys=True)
    assert "/users/local/" not in encoded
    assert protocol["expected_sequence"] == experiment._expected_sequence_payload()


def test_protocol_outer_resign_cannot_change_sequence() -> None:
    root = Path(__file__).resolve().parents[1]
    args = Namespace(
        benchmark_root=root.parent / "vnncomp2021",
        abcrown_root=root.parent / "alpha-beta-CROWN",
        abcrown_python=Path("python"),
        model=(
            root.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
        ),
        property=(
            root.parent
            / "vnncomp2021/benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered"
            / "resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"
        ),
    )
    protocol = experiment._protocol(args)
    cast(list[list[object]], protocol["expected_sequence"])[0][2] = "B3"
    payload = dict(protocol)
    payload.pop("protocol_hash")
    protocol["protocol_hash"] = experiment.canonical_hash(payload)
    with pytest.raises(ValueError, match="protocol differs"):
        experiment._validate_protocol(protocol)


def test_resume_rejects_partial_worker(tmp_path: Path) -> None:
    path = tmp_path / "workers/run_00.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="partial worker"):
        experiment._load_complete_worker(
            artifact=tmp_path,
            index=0,
            block=0,
            position=0,
            configuration=FSG4B3TimingConfiguration.B0,
            mode=FSG3Mode.CONTROL,
        )


def test_outer_resigned_attack_inventory_covers_all_evidence_layers() -> None:
    names = tuple(name for name, _attack in tamper.ATTACKS)
    assert len(names) == 10
    assert any("latency" in name for name in names)
    assert any("delete" in name for name in names)
    assert any("order" in name for name in names)
    assert any("activation" in name for name in names)
    assert any("counter" in name for name in names)
    assert any("fallback" in name for name in names)
    assert any("semantic" in name for name in names)
    assert any("preflight" in name for name in names)
    assert any("protocol" in name for name in names)
    assert any("summary" in name for name in names)
