"""Non-GPU contracts for the PR-12E runtime Pareto runner."""

import json
from pathlib import Path

import torch

from boundflow.planner.execution_candidate import BackendVariant, OperatorFamily
from boundflow.planner.fused_crown_backend import (
    FusedCrownBackendObservation,
    FusedCrownBackendPlanner,
)
from boundflow.runtime.fused_crown import plan_fused_crown_regions
from scripts.benchmark_phase7a_pr12_runtime_pareto import (
    HELDOUT_SPLIT_ID,
    _build_query,
    _fallback_control_workload,
    _observations_from_rows,
    _planner_evaluation,
    _workload,
)


def _split() -> dict:
    path = Path("artifacts/phase7a-pr12/baseline/heldout_split.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_runtime_workloads_preserve_frozen_split_and_region_counts() -> None:
    split = _split()
    assert split["split_id"] == HELDOUT_SPLIT_ID
    workloads = [
        _workload(record, split_role="calibration") for record in split["calibration"]
    ] + [_workload(record, split_role="heldout") for record in split["final_heldout"]]

    for workload in workloads:
        module, _input_spec = _build_query(workload, torch.device("cpu"))
        assert len(plan_fused_crown_regions(module.get_entry_task().ops)) == (
            workload.expected_regions
        )
        assert workload.boundary_bytes > 0
        assert workload.budget_bytes > 0

    control = _fallback_control_workload()
    module, _input_spec = _build_query(control, torch.device("cpu"))
    assert plan_fused_crown_regions(module.get_entry_task().ops) == ()
    assert control.split_role == "fallback_control"


def _row(backend: str, stream: str, *, status: str = "ok") -> dict:
    return {
        "status": status,
        "workload": {
            "case_id": "case",
            "planner_family": "linear",
            "boundary_bytes": 4096,
            "expected_fused_regions": 1,
        },
        "candidate": {
            "backend": backend,
            "stream": stream,
            "eligible": True,
        },
        "runtime": {"host_group_per_query": {"median_ms": 1.25}},
        "memory": {"peak_allocated_delta_bytes": 2048},
    }


def test_calibration_observations_use_default_stream_and_success_only() -> None:
    observations = _observations_from_rows(
        [
            _row("pytorch_eager", "default"),
            _row("tvm_fused_tir", "default"),
            _row("pytorch_eager", "custom"),
            _row("tvm_fused_tir", "default", status="fail"),
        ]
    )

    assert len(observations) == 2
    assert {observation.backend for observation in observations} == {
        BackendVariant.PYTORCH_EAGER,
        BackendVariant.TVM_FUSED_TIR,
    }
    assert all(observation.case_id == "case" for observation in observations)


def test_fallback_control_excludes_ineligible_fused_label_from_oracle() -> None:
    planner = FusedCrownBackendPlanner.fit(
        [
            FusedCrownBackendObservation(
                "cal",
                OperatorFamily.LINEAR,
                backend,
                4096,
                1,
                latency,
                2048,
                True,
                True,
            )
            for backend, latency in (
                (BackendVariant.PYTORCH_EAGER, 2.0),
                (BackendVariant.TVM_FUSED_TIR, 1.0),
            )
        ]
    )
    eager = _row("pytorch_eager", "default")
    eager["runtime"]["host_group_per_query"]["median_ms"] = 1.1
    fused_label = _row("tvm_fused_tir", "default")
    fused_label["candidate"]["eligible"] = False
    fused_label["runtime"]["host_group_per_query"]["median_ms"] = 0.9

    evaluation = _planner_evaluation(
        _fallback_control_workload(),
        [eager, fused_label],
        planner,
        split_id=HELDOUT_SPLIT_ID,
    )

    assert evaluation["decision"]["backend"] == "pytorch_eager"
    assert evaluation["oracle_backend"] == "pytorch_eager"
    assert evaluation["latency_regret"] == 1.0
