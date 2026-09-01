"""Contracts for the FSG1 official αβ-CROWN B0 control attribution."""

# pylint: disable=missing-function-docstring

from copy import deepcopy
from typing import Any, cast

import pytest

from boundflow.runtime.official_control_attribution import (
    OFFICIAL_CONTROL_WORKER_SCHEMA_VERSION,
    build_official_control_run,
    derive_official_control_evidence,
    validate_control_profile_pair,
    validate_official_control_worker,
)
from boundflow.runtime.gpu_attribution import summarize_run
from scripts.run_fsg1_official_control_baseline import (
    _derived_payloads,
    _payload_text,
    _selected_workloads,
)


def _call(  # pylint: disable=too-many-arguments
    call_id: int,
    *,
    parent_call_id: int | None,
    depth: int,
    phase: str,
    host_start_ns: int,
    host_end_ns: int,
    cuda_start_ns: int,
    cuda_end_ns: int,
) -> dict[str, object]:
    return {
        "call_id": call_id,
        "parent_call_id": parent_call_id,
        "depth": depth,
        "method": "CROWN-Optimized",
        "phase": phase,
        "external_phase": "incomplete_verification",
        "host_start_ns": host_start_ns,
        "host_end_ns": host_end_ns,
        "cuda_start_ns": cuda_start_ns,
        "cuda_end_ns": cuda_end_ns,
        "stream_id": "stream-0",
        "memory_allocated_before_bytes": 10,
        "memory_allocated_after_bytes": 20,
        "memory_reserved_before_bytes": 30,
        "memory_reserved_after_bytes": 40,
        "bound_lower": True,
        "bound_upper": False,
        "kwargs_keys": ["bound_lower", "bound_upper", "method"],
    }


def _record(
    mode: str,
    *,
    repeat_index: int = 0,
    scope_ns: int | None = None,
    workload_id: str = "mnistfc:2",
) -> dict[str, Any]:
    profile = mode == "profile"
    return {
        "schema_version": OFFICIAL_CONTROL_WORKER_SCHEMA_VERSION,
        "run_id": f"{workload_id}-{repeat_index}-{mode}",
        "configuration_id": "B0",
        "workload_id": workload_id,
        "mode": mode,
        "repeat_index": repeat_index,
        "pair_order": "control-profile" if repeat_index % 2 == 0 else "profile-control",
        "source": {
            "abcrown_commit": "a" * 40,
            "auto_lirpa_commit": "b" * 40,
            "vnncomp_commit": "c" * 40,
            "model_relative_path": "model.onnx",
            "property_relative_path": "property.vnnlib",
            "model_sha256": "d" * 64,
            "property_sha256": "e" * 64,
        },
        "protocol": {
            "device": "cuda",
            "seed": 100,
            "reset_seed_after_precompile": True,
            "timeout_seconds": 60,
            "max_iterations": 16,
            "alpha_steps": 5,
            "beta_steps": 10,
            "batch_size": 256,
            "auto_enlarge_batch_size": False,
        },
        "environment": {
            "python": "3.11.15",
            "torch": "2.11.0+cu130",
            "torch_cuda": "13.0",
            "gpu_name": "RTX 4060 Laptop GPU",
            "gpu_total_memory": 8_000_000_000,
        },
        "result": {"status": "verified", "success": True, "visited_domains": [1]},
        "scope_ns": scope_ns if scope_ns is not None else (1000 if profile else 990),
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
        "calls": (
            [
                _call(
                    0,
                    parent_call_id=None,
                    depth=0,
                    phase="initial_crown",
                    host_start_ns=100,
                    host_end_ns=800,
                    cuda_start_ns=120,
                    cuda_end_ns=700,
                ),
                _call(
                    1,
                    parent_call_id=0,
                    depth=1,
                    phase="alpha_optimize",
                    host_start_ns=200,
                    host_end_ns=500,
                    cuda_start_ns=220,
                    cuda_end_ns=450,
                ),
            ]
            if profile
            else []
        ),
        "performance_claimed": False,
    }


def test_profile_reconstruction_closes_nested_host_and_gpu_intervals() -> None:
    run = build_official_control_run(_record("profile"))
    summary = run.to_dict()

    assert run.scope_end_ns == 1000
    assert len(run.spans) == 5
    assert [segment.phase.value for segment in run.critical_path] == [
        "setup",
        "initial_crown",
        "alpha_optimize",
        "initial_crown",
        "termination",
    ]
    assert sum(segment.duration_ns for segment in run.critical_path) == 1000
    assert summary["performance_claimed"] is False


def test_profile_reconstruction_separates_gpu_sum_and_union() -> None:
    summary = summarize_run(build_official_control_run(_record("profile")))

    assert summary["gpu_sum_ns"] == 810
    assert summary["gpu_union_ns"] == 580
    assert summary["gpu_overlap_ns"] == 230
    assert summary["attribution_passed"] is True


def test_control_profile_pair_requires_exact_official_semantics() -> None:
    control = _record("control")
    profile = _record("profile")

    assert validate_control_profile_pair(control, profile) == pytest.approx(1000 / 990)

    changed = deepcopy(profile)
    changed["result"]["visited_domains"] = [2]
    with pytest.raises(ValueError, match="semantics differ"):
        validate_control_profile_pair(control, changed)


def test_worker_schema_rejects_unprofiled_calls_and_missing_profile_calls() -> None:
    control = _record("control")
    control["calls"] = [
        _call(
            0,
            parent_call_id=None,
            depth=0,
            phase="initial_crown",
            host_start_ns=1,
            host_end_ns=2,
            cuda_start_ns=1,
            cuda_end_ns=2,
        )
    ]
    with pytest.raises(ValueError, match="cannot contain call spans"):
        validate_official_control_worker(control)

    profile = _record("profile")
    profile["calls"] = []
    with pytest.raises(ValueError, match="requires call spans"):
        validate_official_control_worker(profile)


def test_worker_schema_rejects_invalid_nesting_and_unknown_fields() -> None:
    profile = _record("profile")
    calls = cast(list[dict[str, object]], profile["calls"])
    calls[1]["depth"] = 3
    with pytest.raises(ValueError, match="nesting differs"):
        validate_official_control_worker(profile)

    profile = _record("profile")
    profile["unexpected"] = True
    with pytest.raises(ValueError, match="fields differ"):
        validate_official_control_worker(profile)


def test_five_fresh_pairs_derive_b0_control_evidence() -> None:
    rows: list[dict[str, Any]] = []
    for repeat_index in range(5):
        rows.extend(
            (
                _record("control", repeat_index=repeat_index),
                _record("profile", repeat_index=repeat_index),
            )
        )

    evidence = derive_official_control_evidence(rows)

    assert evidence["status"] == "validated_b0_control"
    assert evidence["pair_count"] == 5
    workload = cast(dict[str, dict[str, object]], evidence["workloads"])["mnistfc:2"]
    assert workload["repeat_count"] == 5
    assert workload["perturbation_passed"] is True
    assert len(cast(list[object], evidence["runs"])) == 5
    assert evidence["performance_claimed"] is False


def test_perturbation_gate_marks_trace_not_auditable() -> None:
    rows: list[dict[str, Any]] = []
    for repeat_index in range(5):
        rows.extend(
            (
                _record("control", repeat_index=repeat_index, scope_ns=900),
                _record("profile", repeat_index=repeat_index, scope_ns=1000),
            )
        )

    evidence = derive_official_control_evidence(rows)

    assert evidence["status"] == "not_auditable"
    workload = cast(dict[str, dict[str, object]], evidence["workloads"])["mnistfc:2"]
    assert workload["perturbation_passed"] is False


def test_pair_coverage_rejects_missing_or_duplicate_modes() -> None:
    with pytest.raises(ValueError, match="pair is incomplete"):
        derive_official_control_evidence([_record("control")])

    duplicate = [_record("control"), _record("control")]
    duplicate[1]["run_id"] = "duplicate"
    with pytest.raises(ValueError, match="mode duplicates"):
        derive_official_control_evidence(duplicate)


def test_runner_derives_raw_first_artifact_payloads() -> None:
    rows: list[dict[str, Any]] = []
    for repeat_index in range(5):
        rows.extend(
            (
                _record("control", repeat_index=repeat_index),
                _record("profile", repeat_index=repeat_index),
            )
        )

    payloads = _derived_payloads(rows)

    summary = cast(dict[str, object], payloads["summary.json"])
    assert summary["status"] == "validated_b0_control"
    assert summary["control_run_count"] == 5
    assert summary["profile_run_count"] == 5
    assert len(cast(list[object], payloads["raw_events.jsonl"])) == 25
    assert len(cast(list[object], payloads["normalized_spans.jsonl"])) == 25
    assert not cast(list[object], payloads["failure_rows.jsonl"])
    assert _payload_text("summary.json", summary).endswith("\n")
    assert (
        _payload_text("paired_runs.jsonl", payloads["paired_runs.jsonl"]).count("\n")
        == 5
    )


def test_runner_workload_selection_is_fail_closed() -> None:
    assert [row["workload_id"] for row in _selected_workloads(["mnistfc:2"])] == [
        "mnistfc:2"
    ]
    with pytest.raises(ValueError, match="selected workload differs"):
        _selected_workloads(["unknown"])
