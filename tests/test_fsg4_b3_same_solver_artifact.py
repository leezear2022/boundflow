"""Frozen evidence checks for the FSG4 B0/B2/B3 timing artifact."""

# pylint: disable=missing-function-docstring,protected-access,duplicate-code

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from boundflow.runtime.fsg4_b3_same_solver_timing import (
    expected_fsg4_b3_sequence,
)
from scripts import run_fsg4_b3_same_solver_experiment as experiment

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1"
TAMPER_REPORT = ARTIFACT.with_name("resnet2b-prop0-v1-tamper-report.json")
SOURCE_GIT_HEAD = "36e9069ca4f21183c9b36d74024de0ca8b20f59c"
SUMMARY_HASH = "4c19afd43c18e0409932b86506efdaf6bfc3e07baabcc222dbe79c8149f99bac"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rows() -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(line))
        for line in (ARTIFACT / "worker_runs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_b3_timing_artifact_replays_from_source_commit() -> None:
    runs, summary, result = experiment._verify_static_artifact(ARTIFACT)

    assert len(runs) == 36
    assert summary["summary_hash"] == SUMMARY_HASH
    assert result == {
        "correctness_passed": True,
        "decision_inputs": {
            "b0_over_b3_query_geomean": 0.9100012637918488,
            "b2_over_b3_core_geomean": 1.0716174805930418,
            "b2_over_b3_query_geomean": 1.0066228954759742,
            "cumulative_query_gate": 1.0,
            "pair_degradation_floor": 0.9523809523809523,
            "reduced_core_gate": 1.05,
            "validated_core_gate": 1.15,
            "worst_pair_b2_over_b3_core": 1.0635877032562384,
        },
        "environment_passed": True,
        "measurement_auditable": True,
        "performance_claimed": False,
        "run_count": 36,
        "status": "validated-reduced-b3",
        "summary_hash": SUMMARY_HASH,
    }


def test_frozen_b3_timing_protocol_and_sequence_are_exact() -> None:
    protocol = _load(ARTIFACT / "protocol.json")
    manifest = _load(ARTIFACT / "manifest.json")
    rows = _rows()

    assert protocol["source_git_head"] == SOURCE_GIT_HEAD
    assert manifest["source_git_head"] == SOURCE_GIT_HEAD
    assert protocol["worker_count"] == 36
    assert manifest["worker_count"] == 36
    assert protocol["expected_sequence"] == [
        [block_index, sequence_position, configuration.value, mode.value]
        for block_index, sequence_position, configuration, mode in expected_fsg4_b3_sequence()
    ]
    assert [
        (
            row["block_index"],
            row["sequence_position"],
            row["configuration"],
            row["mode"],
        )
        for row in rows
    ] == [
        (block_index, sequence_position, configuration.value, mode.value)
        for block_index, sequence_position, configuration, mode in expected_fsg4_b3_sequence()
    ]
    assert len({row["source_identity"] for row in rows}) == 1
    assert all(row["environment"]["admitted"] is True for row in rows)
    assert all(row["performance_claimed"] is False for row in rows)


def test_frozen_b3_timing_activation_is_physical_and_measurement_safe() -> None:
    rows = _rows()
    b0_rows = [row for row in rows if row["configuration"] == "B0"]
    b2_rows = [row for row in rows if row["configuration"] == "B2"]
    b3_rows = [row for row in rows if row["configuration"] == "B3"]

    assert len(b0_rows) == len(b2_rows) == len(b3_rows) == 12
    assert all(
        row["execution"]["replacement_mode"] == "original_provider" for row in b0_rows
    )
    assert all(
        row["execution"]["replacement_mode"] == "whole_call_reference"
        for row in b2_rows
    )
    assert all(
        row["execution"]["replacement_mode"] == "b3_ir_graph_plan_schedule"
        for row in b3_rows
    )
    assert all(row["execution"]["provider_core_call_count"] == 1 for row in b0_rows)
    for row in b2_rows + b3_rows:
        assert row["execution"]["provider_core_call_count"] == 0
        assert row["execution"]["provider_compute_bounds_call_count"] == 0
        assert row["execution"]["provider_update_bounds_call_count"] == 0
        assert row["execution"]["fallback_dispatch_count"] == 0
    for row in b3_rows:
        activation = row["activation"]
        assert activation["prepared_core_template_count"] == 1
        assert activation["prepared_core_instance_count"] == 1
        assert activation["terminal_optimizer_schedule_count"] == 1
        assert activation["assembly_count"] == 1
        assert activation["commit_receipt_count"] == 1
        assert activation["device_commit_audit_count"] == 1
        assert activation["headline_content_digest_count"] == 0
        assert activation["candidate_d2h_copy_count"] == 0
        assert activation["post_query_audit_ns"] > 0
        assert activation["post_query_audit_excluded_from_timing"] is True
    control_rows = [row for row in rows if row["mode"] == "control"]
    assert all(row["activation"]["detailed_counts"] is None for row in control_rows)
    for row in rows:
        detailed = row["activation"]["detailed_counts"]
        if row["mode"] != "profile" or row["configuration"] == "B0":
            assert detailed is None
            continue
        assert isinstance(detailed, dict)
        assert detailed["optimizer_evaluation_count"] == 10
        assert detailed["optimizer_update_count"] == 9
        if row["configuration"] == "B2":
            assert detailed["full_optimizer_step_snapshot_count"] == 10
            assert detailed["forward_trace_build_count"] == 5
            assert detailed["timed_candidate_d2h_copy_count"] == 12
        else:
            assert detailed["template_hit_in_core_count"] == 1
            assert detailed["full_optimizer_step_snapshot_count"] == 0
            assert detailed["forward_trace_build_count"] == 4
            assert detailed["timed_candidate_d2h_copy_count"] == 0


def test_frozen_b3_timing_classification_and_measurement_gates() -> None:
    summary = _load(ARTIFACT / "summary.json")
    closure = _load(ARTIFACT / "closure.json")
    environment = _load(ARTIFACT / "environment.json")

    assert summary["status"] == "validated-reduced-b3"
    assert summary["correctness_passed"] is True
    assert summary["environment_passed"] is True
    assert summary["measurement_auditable"] is True
    assert summary["performance_claimed"] is False
    assert summary["control_count_by_configuration"] == {"B0": 6, "B2": 6, "B3": 6}
    assert summary["profile_count_by_configuration"] == {"B0": 6, "B2": 6, "B3": 6}
    assert all(value["passed"] is True for value in summary["perturbation"].values())
    assert closure["all_closed"] is True
    assert len(closure["rows"]) == 18
    assert max(row["closure_error"] for row in closure["rows"]) == pytest.approx(
        0.002510499028552414
    )
    assert max(row["residual_share"] for row in closure["rows"]) == pytest.approx(
        0.002510499028552414
    )
    assert environment["all_workers_environment_admitted"] is True
    assert environment["run_count"] == 36
    assert environment["runtime_identity_count"] == 1


def test_frozen_b3_timing_artifact_is_path_free_and_digest_bound() -> None:
    for path in ARTIFACT.rglob("*"):
        if path.is_file():
            assert b"/home/" not in path.read_bytes()

    assert _sha256(ARTIFACT / "protocol.json") == (
        "cecec7c262040a2953d1e0ec7e1479942d648b39f12f6edaa8787efbca6e79e2"
    )
    assert _sha256(ARTIFACT / "manifest.json") == (
        "d88eeecafcd6a7a9394cdf9654962a36497b1c7afd15d7862048b1c3ccd7db4a"
    )
    assert _sha256(ARTIFACT / "summary.json") == (
        "c8666213479328d8406172dd56d70f015e96e6fa9b2f937c90a5ec07fedb2ff0"
    )
    manifest = _load(ARTIFACT / "manifest.json")
    assert manifest["protocol_hash"] == (
        "8193010ee200d11ad12ea68c01462c5c1a5854078fa58d895ed745e12c3d642b"
    )
    assert manifest["manifest_hash"] == (
        "d553a72ea5cae99c367f7c0120966bb36058966bc081241befba24455be2c1c9"
    )


def test_frozen_b3_timing_outer_resigned_attacks_are_rejected() -> None:
    report = _load(TAMPER_REPORT)

    assert _sha256(TAMPER_REPORT) == (
        "bd392e5ca49912e376c4090e7e929077f804b6bbd5d11a250d0790b83847e21b"
    )
    assert report["artifact_source_git_head"] == SOURCE_GIT_HEAD
    assert report["clean_summary_hash"] == SUMMARY_HASH
    assert report["attack_count"] == 10
    assert report["outer_resigned_attack_count"] == 10
    assert report["all_rejected"] is True
    assert all(row["rejected"] is True for row in report["attacks"])
    assert report["report_hash"] == (
        "b89ada48f40c5766bc0b93c1542d0a5aa7cc741fe250f9cb9efc5c38a2cae799"
    )
    assert report["performance_claimed"] is False
