"""Smoke and split-contract tests for the PR-11 held-out evaluator."""

import csv
import json

from scripts.evaluate_phase7a_pr11_planner import (
    EVAL_SCHEMA_VERSION,
    POLICIES,
    load_profile_cases,
    main,
)
from scripts.summarize_phase7a_pr11_planner import main as summarize_main

FIELDS = [
    "status",
    "run_id",
    "workload",
    "tier",
    "method",
    "relu_backward_mode",
    "spec_batch",
    "domain_batch",
    "latency_ms_median_trace_off",
    "peak_cuda_allocated_bytes_trace_off",
]


def _row(
    workload: str,
    mode: str,
    *,
    peak: int | None,
    latency: float | None,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "status": status,
        "run_id": "test-profile",
        "workload": workload,
        "tier": "test",
        "method": "CROWN",
        "relu_backward_mode": mode,
        "spec_batch": 16,
        "domain_batch": 8,
        "latency_ms_median_trace_off": "" if latency is None else latency,
        "peak_cuda_allocated_bytes_trace_off": "" if peak is None else peak,
    }


def _write_profile(path) -> None:
    rows = [
        _row("mlp_chain", "dense", peak=500_000, latency=2.0),
        _row("mlp_chain", "structured", peak=400_000, latency=5.0),
        _row("add_concat_dag", "dense", peak=650_000, latency=2.5),
        _row("add_concat_dag", "structured", peak=500_000, latency=6.0),
        _row("mini_resnet", "dense", peak=800_000, latency=3.0),
        _row("mini_resnet", "structured", peak=600_000, latency=8.0),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_eval_runner_keeps_architecture_families_disjoint(tmp_path) -> None:
    profile = tmp_path / "profile.csv"
    output = tmp_path / "eval.jsonl"
    model_output = tmp_path / "model.json"
    manifest_output = tmp_path / "manifest.json"
    summary_output = tmp_path / "summary.csv"
    _write_profile(profile)

    result = main(
        [
            "--input",
            str(profile),
            "--output",
            str(output),
            "--model-output",
            str(model_output),
            "--manifest-output",
            str(manifest_output),
            "--calibration-workloads",
            "mlp_chain",
            "--heldout-workloads",
            "mini_resnet",
            "--validation-workloads",
            "add_concat_dag",
            "--budgets-mib",
            "1",
        ]
    )

    assert result == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == len(POLICIES)
    assert {row["schema_version"] for row in rows} == {EVAL_SCHEMA_VERSION}
    assert {row["workload"]["name"] for row in rows} == {"mini_resnet"}
    assert {row["split"]["workload_role"] for row in rows} == {"final_heldout"}
    assert {tuple(row["split"]["calibration_workloads"]) for row in rows} == {
        ("mlp_chain",)
    }
    assert {row["decision"]["plan"]["policy"] for row in rows} == {
        policy.value for policy in POLICIES
    }
    model = json.loads(model_output.read_text())
    assert model["split"]["calibration_workloads"] == ["mlp_chain"]
    assert model["split"]["validation_workloads"] == ["add_concat_dag"]
    assert model["split"]["heldout_workloads"] == ["mini_resnet"]
    manifest = json.loads(manifest_output.read_text())
    assert manifest["row_count"] == len(POLICIES)
    assert manifest["outputs"]["jsonl"]["sha256"]
    assert manifest["outputs"]["model"]["sha256"]

    assert (
        summarize_main(["--input", str(output), "--output", str(summary_output)]) == 0
    )
    with summary_output.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == len(POLICIES) * 2
    global_all = next(
        row for row in summary if row["scope"] == "all" and row["policy"] == "global"
    )
    assert int(global_all["rows"]) == 1
    assert float(global_all["feasible_coverage"]) == 1.0


def test_profile_loader_preserves_oom_as_oracle_observation(tmp_path) -> None:
    profile = tmp_path / "profile.csv"
    rows = [
        _row("mini_resnet", "dense", peak=800_000, latency=3.0),
        _row(
            "mini_resnet",
            "structured",
            peak=None,
            latency=None,
            status="oom",
        ),
    ]
    with profile.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    cases = load_profile_cases(profile)

    assert len(cases) == 1
    assert {observation.status for observation in cases[0].observations} == {
        "ok",
        "oom",
    }
