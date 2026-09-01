"""Frozen FSG3 same-solver baseline replay and tamper contracts."""

# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

from pathlib import Path
from typing import cast

from scripts import probe_fsg3_same_solver_artifact_tamper as tamper_runner
from scripts import run_fsg3_same_solver_experiment as artifact_runner

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPOSITORY_ROOT / "artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5"
TAMPER_REPORT = (
    REPOSITORY_ROOT
    / "artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5-tamper-report.json"
)


def test_formal_artifact_static_replay_is_auditable() -> None:
    runs, summary, result = artifact_runner._verify_static_artifact(ARTIFACT)
    assert len(runs) == 36
    assert result["status"] == "replay-passed"
    assert summary["status"] == "validated-fsg3-b0-b1-b2-baseline"
    assert summary["correctness_passed"] is True
    assert summary["environment_passed"] is True
    assert summary["measurement_auditable"] is True
    assert not summary["failure_rows"]
    assert summary["performance_claimed"] is False
    assert summary["summary_hash"] == (
        "df852590d99be09962c1287e7166b421edb260416403a3c91545dca6e2e1318e"
    )


def test_formal_artifact_execution_paths_and_speedup_direction_are_frozen() -> None:
    runs, summary, _result = artifact_runner._verify_static_artifact(ARTIFACT)
    for run in runs:
        assert run.environment.admitted is True
        if run.configuration.value == "B0":
            assert run.execution.typed_validation_count == 0
            assert (
                run.execution.provider_core_call_count,
                run.execution.provider_compute_bounds_call_count,
                run.execution.provider_update_bounds_call_count,
            ) == (1, 14, 3)
        elif run.configuration.value == "B1":
            assert run.execution.typed_validation_count == 1
            assert (
                run.execution.provider_core_call_count,
                run.execution.provider_compute_bounds_call_count,
                run.execution.provider_update_bounds_call_count,
            ) == (1, 14, 3)
        else:
            assert run.execution.typed_validation_count == 1
            assert (
                run.execution.provider_core_call_count,
                run.execution.provider_compute_bounds_call_count,
                run.execution.provider_update_bounds_call_count,
                run.execution.fallback_dispatch_count,
            ) == (0, 0, 0, 0)
    speedups = cast(
        dict[str, dict[str, dict[str, float]]], summary["speedups_b0_over_candidate"]
    )
    assert speedups["B1"]["query_wall_ns"]["geometric_mean"] == 0.9956565794571265
    assert speedups["B2"]["query_wall_ns"]["geometric_mean"] == 0.9083995539523697
    assert speedups["B2"]["core_wall_ns"]["geometric_mean"] == 0.5167670145223869
    assert summary["b2_compile_break_even_queries"] == "not_reachable"


def test_formal_tamper_report_rejects_all_preregistered_attacks() -> None:
    report = artifact_runner._load_json(TAMPER_REPORT)
    assert report["schema_version"] == tamper_runner.REPORT_SCHEMA_VERSION
    assert report["clean_summary_hash"] == (
        "df852590d99be09962c1287e7166b421edb260416403a3c91545dca6e2e1318e"
    )
    assert report["attack_count"] == 8
    assert report["outer_resigned_attack_count"] == 8
    assert report["all_rejected"] is True
    assert report["performance_claimed"] is False
    attacks = cast(list[dict[str, object]], report["attacks"])
    assert {row["name"] for row in attacks} == {
        "control-latency-outer-resign",
        "delete-run-outer-resign",
        "configuration-mode-order-outer-resign",
        "b1-provider-count-outer-resign",
        "b2-fallback-count-outer-resign",
        "semantic-tensor-outer-resign",
        "temperature-gate-outer-resign",
        "summary-ratio-outer-resign",
    }
    assert all(row["rejected"] is True for row in attacks)
