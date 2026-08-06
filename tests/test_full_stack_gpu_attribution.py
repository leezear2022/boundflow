"""Contracts for full-stack GPU attribution and cumulative ablation."""

# pylint: disable=missing-function-docstring

from dataclasses import replace
import json
from pathlib import Path
from typing import cast

import pytest

from boundflow.runtime.gpu_attribution import (
    CacheState,
    CriticalPathSegment,
    FeatureActivationLedger,
    FeatureKind,
    FullStackAttributionRun,
    FullStackSpan,
    ReplacementMode,
    ResourceKind,
    SolverPhase,
    StackLayer,
    canonical_hash,
    deletion_only_ceiling,
    full_stack_run_from_dict,
    interval_union_ns,
    joint_amdahl_speedup,
    summarize_cumulative_ablation,
    summarize_run,
)
from scripts.run_full_stack_gpu_baseline_attribution import (
    file_sha256,
    generate_artifact,
    replay_artifact,
)


def _features(
    *, replacement_mode: ReplacementMode = ReplacementMode.ORIGINAL_PROVIDER
) -> FeatureActivationLedger:
    return FeatureActivationLedger(
        bound_graph_ir_drives_execution=False,
        plan_compiled_before_execution=False,
        task_schedule_drives_execution=False,
        backend_kind="pytorch_reference",
        physical_backend_dispatches=0,
        fallback_dispatches=0,
        jit_cache_state=CacheState.NOT_APPLICABLE,
        stream_count=2,
        event_count=2,
        wait_count=1,
        storage_plan_enforced=False,
        replacement_mode=replacement_mode,
    )


def _run() -> FullStackAttributionRun:
    spans = (
        FullStackSpan(
            span_id="host-root",
            parent_span_id=None,
            layer=StackLayer.SOLVER_CONTROL,
            phase=SolverPhase.SETUP,
            resource=ResourceKind.HOST_THREAD,
            cache_state=CacheState.WARM_EXECUTE,
            start_ns=0,
            end_ns=100,
        ),
        FullStackSpan(
            span_id="cuda-0",
            parent_span_id="host-root",
            layer=StackLayer.OPERATOR_EXECUTION,
            phase=SolverPhase.INITIAL_CROWN,
            resource=ResourceKind.CUDA_STREAM,
            cache_state=CacheState.WARM_EXECUTE,
            start_ns=10,
            end_ns=50,
            stream_id="stream-0",
        ),
        FullStackSpan(
            span_id="cuda-1",
            parent_span_id="host-root",
            layer=StackLayer.OPERATOR_EXECUTION,
            phase=SolverPhase.SELECTED_CROWN,
            resource=ResourceKind.CUDA_STREAM,
            cache_state=CacheState.WARM_EXECUTE,
            start_ns=30,
            end_ns=70,
            stream_id="stream-1",
        ),
    )
    critical = (
        CriticalPathSegment(
            "cp-0", StackLayer.SOLVER_CONTROL, SolverPhase.SETUP, 0, 10, ("host-root",)
        ),
        CriticalPathSegment(
            "cp-1",
            StackLayer.OPERATOR_EXECUTION,
            SolverPhase.INITIAL_CROWN,
            10,
            50,
            ("cuda-0",),
        ),
        CriticalPathSegment(
            "cp-2",
            StackLayer.RUNTIME_SCHEDULE,
            SolverPhase.SELECTED_CROWN,
            50,
            70,
            ("cuda-1",),
        ),
        CriticalPathSegment(
            "cp-3",
            StackLayer.SOLVER_CONTROL,
            SolverPhase.TERMINATION,
            70,
            100,
            ("host-root",),
        ),
    )
    return FullStackAttributionRun(
        run_id="run-0",
        configuration_id="B0",
        scope_start_ns=0,
        scope_end_ns=100,
        spans=spans,
        critical_path=critical,
        features=_features(),
    )


def test_full_stack_summary_separates_gpu_sum_union_and_critical_path() -> None:
    summary = summarize_run(_run())

    assert summary["closure_passed"] is True
    assert summary["residual_passed"] is True
    assert summary["attribution_passed"] is True
    assert summary["closure_error"] == 0.0
    assert summary["critical_path_ns"] == 100
    assert summary["gpu_sum_ns"] == 80
    assert summary["gpu_union_ns"] == 60
    assert summary["gpu_overlap_ns"] == 20
    layer_ns = cast(dict[str, int], summary["layer_ns"])
    assert layer_ns[StackLayer.OPERATOR_EXECUTION.value] == 40
    assert layer_ns[StackLayer.RUNTIME_SCHEDULE.value] == 20


def test_full_stack_summary_marks_incomplete_critical_path_not_auditable() -> None:
    run = _run()
    incomplete = replace(run, critical_path=run.critical_path[:-1])

    summary = summarize_run(incomplete)

    assert summary["closure_passed"] is False
    assert summary["attribution_passed"] is False
    assert summary["closure_error"] == pytest.approx(0.30)


def test_full_stack_rejects_overlapping_exclusive_critical_segments() -> None:
    run = _run()
    overlap = replace(run.critical_path[1], start_ns=5)

    with pytest.raises(ValueError, match="segments overlap"):
        summarize_run(replace(run, critical_path=(run.critical_path[0], overlap)))


def test_full_stack_summary_rejects_large_unclassified_residual() -> None:
    run = _run()
    residual = replace(run.critical_path[3], layer=StackLayer.UNCLASSIFIED_RESIDUAL)

    summary = summarize_run(
        replace(run, critical_path=(*run.critical_path[:3], residual))
    )

    assert summary["closure_passed"] is True
    assert summary["residual_share"] == pytest.approx(0.30)
    assert summary["residual_passed"] is False
    assert summary["attribution_passed"] is False


def test_full_stack_rejects_parent_or_dependency_outside_contract() -> None:
    run = _run()
    escaped = replace(run.spans[1], parent_span_id="missing")
    with pytest.raises(ValueError, match="parent is missing"):
        replace(run, spans=(run.spans[0], escaped)).validate()

    late_dependency = replace(run.spans[2], dependency_span_ids=("cuda-0",))
    with pytest.raises(ValueError, match="ends after dependent starts"):
        replace(run, spans=(run.spans[0], run.spans[1], late_dependency)).validate()


def test_full_stack_rejects_parent_and_dependency_cycles() -> None:
    run = _run()
    first = replace(
        run.spans[0],
        span_id="first",
        parent_span_id="second",
        start_ns=0,
        end_ns=100,
    )
    second = replace(
        run.spans[1],
        span_id="second",
        parent_span_id="first",
        start_ns=0,
        end_ns=100,
    )
    with pytest.raises(ValueError, match="parent graph contains a cycle"):
        replace(run, spans=(first, second), critical_path=()).validate()

    first = replace(
        first,
        parent_span_id=None,
        dependency_span_ids=("second",),
        end_ns=0,
    )
    second = replace(
        second,
        parent_span_id=None,
        dependency_span_ids=("first",),
        end_ns=0,
    )
    with pytest.raises(ValueError, match="dependency graph contains a cycle"):
        replace(run, spans=(first, second), critical_path=()).validate()


def test_interval_union_does_not_double_count_stream_overlap() -> None:
    assert interval_union_ns(((10, 50), (30, 70), (80, 90))) == 70
    with pytest.raises(ValueError, match="interval is invalid"):
        interval_union_ns(((5, 4),))


def test_feature_ledger_distinguishes_objects_from_physical_activation() -> None:
    inactive = _features(replacement_mode=ReplacementMode.RVIR_PASSTHROUGH)
    assert FeatureKind.BOUNDFLOW_REPLACEMENT not in inactive.activated_features()
    assert inactive.missing(
        (FeatureKind.BOUND_GRAPH_IR, FeatureKind.PHYSICAL_BACKEND)
    ) == (
        FeatureKind.BOUND_GRAPH_IR,
        FeatureKind.PHYSICAL_BACKEND,
    )

    active = FeatureActivationLedger(
        bound_graph_ir_drives_execution=True,
        plan_compiled_before_execution=True,
        task_schedule_drives_execution=True,
        backend_kind="tvm_cuda",
        physical_backend_dispatches=4,
        fallback_dispatches=0,
        jit_cache_state=CacheState.PROCESS_HIT,
        stream_count=2,
        event_count=3,
        wait_count=1,
        storage_plan_enforced=True,
        replacement_mode=ReplacementMode.NESTED_REGION,
    )
    assert set(active.activated_features()) == set(FeatureKind)


def test_feature_ledger_rejects_impossible_stream_events() -> None:
    with pytest.raises(ValueError, match="at least one physical stream"):
        replace(_features(), stream_count=0).validate()


def test_selected_crown_deletion_ceiling_is_not_full_stack_ceiling() -> None:
    assert deletion_only_ceiling(0.07098631834282758) == pytest.approx(1.0764104115)
    assert deletion_only_ceiling(1.0) is None
    with pytest.raises(ValueError, match="share must be"):
        deletion_only_ceiling(1.1)


def test_joint_amdahl_uses_multiple_baseline_regions() -> None:
    result = joint_amdahl_speedup(
        {"operator": 0.2, "runtime": 0.3},
        {"operator": 2.0, "runtime": None},
    )
    assert result == pytest.approx(1.0 / 0.6)

    with pytest.raises(ValueError, match="unknown baseline region"):
        joint_amdahl_speedup({"operator": 0.2}, {"jit": 2.0})
    with pytest.raises(ValueError, match="exceed one"):
        joint_amdahl_speedup({"a": 0.7, "b": 0.5}, {})


def test_cumulative_ablation_reports_interactions_instead_of_adding_layers() -> None:
    summary = summarize_cumulative_ablation(
        {"B0": 100, "B1": 80, "B2": 50},
        ("B0", "B1", "B2"),
        leave_one_out_wall_ns={"ir": 70, "runtime": 60},
    )

    cumulative = cast(dict[str, float], summary["cumulative_speedup"])
    incremental = cast(dict[str, float], summary["incremental_speedup"])
    assert cumulative["B2"] == 2.0
    assert incremental["B2"] == 1.6
    assert summary["leave_one_out_penalty_ns"] == {"ir": 20, "runtime": 10}
    assert summary["interaction_residual_ns"] == 20


def test_run_hash_binds_feature_activation_and_raw_timing() -> None:
    run = _run()
    assert (
        run.stable_hash()
        != replace(
            run,
            features=replace(
                run.features,
                replacement_mode=ReplacementMode.SHADOW_ONLY,
            ),
        ).stable_hash()
    )
    assert (
        run.stable_hash()
        != replace(
            run,
            spans=(replace(run.spans[0], end_ns=99), *run.spans[1:]),
        ).stable_hash()
    )


def test_raw_run_cannot_self_declare_performance() -> None:
    with pytest.raises(ValueError, match="cannot claim performance"):
        replace(_run(), performance_claimed=True).validate()


def test_raw_run_parser_round_trips_only_canonical_payload() -> None:
    payload = _run().to_dict()

    assert full_stack_run_from_dict(payload) == _run()

    payload["unexpected"] = True
    with pytest.raises(ValueError, match="fields differ"):
        full_stack_run_from_dict(payload)


def test_raw_run_parser_recomputes_feature_activation_projection() -> None:
    payload = _run().to_dict()
    features = cast(dict[str, object], payload["features"])
    features["activated_features"] = ["boundflow_replacement"]

    with pytest.raises(ValueError, match="activation projection differs"):
        full_stack_run_from_dict(payload)


def test_contract_artifact_generates_and_replays(tmp_path: Path) -> None:
    raw_run = tmp_path / "input.json"
    raw_run.write_text(json.dumps(_run().to_dict()), encoding="utf-8")
    artifact_dir = tmp_path / "artifact"

    generated = generate_artifact(raw_run, artifact_dir)

    assert generated == replay_artifact(artifact_dir)
    assert generated["status"] == "replay-passed"
    assert generated["closure_passed"] is True
    assert generated["residual_passed"] is True
    assert generated["attribution_passed"] is True
    manifest = json.loads((artifact_dir / "manifest.json").read_text("utf-8"))
    assert manifest["status"] == "contract-only"
    assert manifest["performance_claimed"] is False


def test_contract_artifact_requires_closed_attribution(tmp_path: Path) -> None:
    run = _run()
    residual = replace(run.critical_path[3], layer=StackLayer.UNCLASSIFIED_RESIDUAL)
    raw_run = tmp_path / "input.json"
    raw_run.write_text(
        json.dumps(
            replace(run, critical_path=(*run.critical_path[:3], residual)).to_dict()
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="closure or residual gate failed"):
        generate_artifact(raw_run, tmp_path / "artifact")


def test_contract_replay_rejects_digest_synchronized_summary_tamper(
    tmp_path: Path,
) -> None:
    raw_run = tmp_path / "input.json"
    raw_run.write_text(json.dumps(_run().to_dict()), encoding="utf-8")
    artifact_dir = tmp_path / "artifact"
    generate_artifact(raw_run, artifact_dir)

    summary_path = artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text("utf-8"))
    summary["gpu_sum_ns"] += 1
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["files"]["summary.json"] = file_sha256(summary_path)
    semantic_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    manifest["manifest_hash"] = canonical_hash(semantic_manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="semantic replay differs"):
        replay_artifact(artifact_dir)


def test_contract_replay_rejects_manifest_synchronized_git_head_tamper(
    tmp_path: Path,
) -> None:
    raw_run = tmp_path / "input.json"
    raw_run.write_text(json.dumps(_run().to_dict()), encoding="utf-8")
    artifact_dir = tmp_path / "artifact"
    generate_artifact(raw_run, artifact_dir)

    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["git_head"] = "0" * 40
    semantic_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    manifest["manifest_hash"] = canonical_hash(semantic_manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="manifest envelope differs"):
        replay_artifact(artifact_dir)
