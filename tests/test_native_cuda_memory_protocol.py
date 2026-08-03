"""Frozen NRIR-3 CUDA memory protocol contracts without requiring a GPU."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scripts.bench_native_real_network_cuda_memory import (
    ENVIRONMENT_SCHEMA_VERSION,
    PLAN_IDS,
    WORKER_ROW_SCHEMA_VERSION,
    ProtocolConfig,
    aggregate_rows,
    probe_cuda_environment,
    replay_environment_probe_artifact,
    validate_environment_probe,
    validate_summary,
    validate_worker_row,
    write_environment_probe_artifact,
)
from scripts.run_native_real_network_ir_artifact import (
    INTERMEDIATE_BOUNDS_SHA256,
    MODEL_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_PROBE = (
    ROOT
    / "artifacts"
    / "native-real-network-cuda-memory-protocol"
    / "environment-unavailable-20260804"
)


def _ready_environment() -> dict[str, object]:
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "status": "ready",
        "performance_claimed": False,
        "torch_version": "test",
        "torch_cuda_build": "test",
        "cuda_available": True,
        "device_count": 1,
        "devices": [
            {
                "index": 0,
                "name": "test-gpu",
                "capability": [9, 9],
                "total_memory_bytes": 1 << 30,
            }
        ],
        "nvidia_smi": {"returncode": 0, "stdout": "test", "stderr": ""},
    }


def _row(
    plan: str,
    repeat: int,
    *,
    allocated: int,
    latency: float,
    config: ProtocolConfig,
) -> dict[str, Any]:
    hashes = {
        "bound_module_hash": "a" * 64,
        "plan_template_hash": "b" * 64,
        "plan_instance_hash": ("c" if plan == "retain" else "d") * 64,
        "task_module_hash": ("e" if plan == "retain" else "f") * 64,
        "schedule_hash": ("1" if plan == "retain" else "2") * 64,
    }
    samples = [latency] * config.measured_iterations
    return {
        "schema_version": WORKER_ROW_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "worker_pid": 1000 + repeat * 2 + (0 if plan == "retain" else 1),
        "plan": plan,
        "repeat": repeat,
        "protocol": config.to_dict(),
        "environment": _ready_environment(),
        "model_sha256": MODEL_SHA256,
        "intermediate_bounds_sha256": INTERMEDIATE_BOUNDS_SHA256,
        "ir_hashes": hashes,
        "storage_candidate_id": PLAN_IDS[plan],
        "planned_peak_bytes": 4096 if plan == "retain" else 2048,
        "observed_logical_peak_bytes": 4096 if plan == "retain" else 2048,
        "baseline_allocated_bytes": 100,
        "peak_allocated_bytes": 100 + allocated,
        "peak_allocated_delta_bytes": allocated,
        "baseline_reserved_bytes": 200,
        "peak_reserved_bytes": 200 + allocated,
        "peak_reserved_delta_bytes": allocated,
        "latency_samples_ms": samples,
        "latency_median_ms": latency,
        "lower_sha256": "5" * 64,
        "upper_sha256": "6" * 64,
        "task_trace_hash": "7" * 64,
        "storage_trace_hash": "8" * 64,
        "comparison": {
            "allclose": True,
            "max_abs_diff": 0.0,
            "sign_agreement": 9,
            "sign_total": 9,
        },
    }


def _matrix(
    *, reuse_allocated: int, reuse_latency: float
) -> tuple[ProtocolConfig, list[dict[str, Any]]]:
    config = ProtocolConfig(repeats=3, warmup_iterations=1, measured_iterations=2)
    rows: list[dict[str, Any]] = []
    for repeat in range(config.repeats):
        order = ("retain", "reuse") if repeat % 2 == 0 else ("reuse", "retain")
        rows.extend(
            _row(
                plan,
                repeat,
                allocated=1000 if plan == "retain" else reuse_allocated,
                latency=1.0 if plan == "retain" else reuse_latency,
                config=config,
            )
            for plan in order
        )
    return config, rows


def test_protocol_requires_pre_registered_repeat_and_threshold_contract() -> None:
    ProtocolConfig().validate()
    with pytest.raises(ValueError, match=">=3 repeats"):
        ProtocolConfig(repeats=2).validate()
    with pytest.raises(ValueError, match="reduction threshold"):
        ProtocolConfig(minimum_allocated_reduction=0.0).validate()


def test_environment_probe_cannot_invent_cuda_readiness() -> None:
    probe = probe_cuda_environment()
    validate_environment_probe(probe)
    if not probe["cuda_available"]:
        assert probe["status"] == "environment_unavailable"
        assert probe["device_count"] == 0
        assert not probe["devices"]
    tampered = deepcopy(probe)
    tampered["status"] = "ready"
    if not probe["cuda_available"]:
        with pytest.raises(ValueError, match="contradicts"):
            validate_environment_probe(tampered)


def test_environment_probe_artifact_replays_and_rejects_digest_tamper(
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "cuda-probe"
    generated = write_environment_probe_artifact(artifact_dir)
    replayed = replay_environment_probe_artifact(artifact_dir)
    assert replayed == generated
    environment_path = artifact_dir / "environment.json"
    environment_path.write_text(
        environment_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="digest differs"):
        replay_environment_probe_artifact(artifact_dir)


def test_committed_environment_probe_replays_without_a_performance_claim() -> None:
    probe = replay_environment_probe_artifact(COMMITTED_PROBE)
    assert probe["status"] == "environment_unavailable"
    assert probe["performance_claimed"] is False


def test_aggregate_accepts_only_pre_registered_cuda_pareto() -> None:
    config, rows = _matrix(reuse_allocated=700, reuse_latency=1.1)
    summary = aggregate_rows(rows, config=config)
    assert summary["status"] == "validated"
    assert summary["performance_claimed"] is True
    assert summary["allocated_reduction"] == pytest.approx(0.30)
    assert summary["latency_ratio"] == pytest.approx(1.10)

    tampered = deepcopy(summary)
    tampered["allocated_reduction"] = 0.99
    with pytest.raises(ValueError, match="derived field differs"):
        validate_summary(tampered, rows=rows, config=config)


@pytest.mark.parametrize(
    ("reuse_allocated", "reuse_latency", "failed_gate"),
    (
        (850, 1.1, "allocated_reduction_gte_threshold"),
        (700, 1.21, "latency_ratio_lte_threshold"),
    ),
)
def test_aggregate_keeps_failed_thresholds_as_no_go(
    reuse_allocated: int, reuse_latency: float, failed_gate: str
) -> None:
    config, rows = _matrix(reuse_allocated=reuse_allocated, reuse_latency=reuse_latency)
    summary = aggregate_rows(rows, config=config)
    assert summary["status"] == "no_go"
    assert summary["performance_claimed"] is False
    gates = summary["gates"]
    assert isinstance(gates, dict)
    assert gates[failed_gate] is False


def test_aggregate_rejects_worker_reuse_and_order_drift() -> None:
    config, rows = _matrix(reuse_allocated=700, reuse_latency=1.1)
    duplicate_process = deepcopy(rows)
    duplicate_process[1]["worker_pid"] = duplicate_process[0]["worker_pid"]
    with pytest.raises(ValueError, match="fresh worker"):
        aggregate_rows(duplicate_process, config=config)

    wrong_order = deepcopy(rows)
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
    with pytest.raises(ValueError, match="alternating order"):
        aggregate_rows(wrong_order, config=config)


def test_aggregate_does_not_claim_across_plan_template_drift() -> None:
    config, rows = _matrix(reuse_allocated=700, reuse_latency=1.1)
    rows[1]["ir_hashes"]["plan_template_hash"] = "9" * 64
    summary = aggregate_rows(rows, config=config)
    assert summary["status"] == "no_go"
    assert summary["performance_claimed"] is False
    gates = summary["gates"]
    assert isinstance(gates, dict)
    assert gates["cross_plan_semantic_identity"] is False


def test_worker_row_rejects_unavailable_environment() -> None:
    config, rows = _matrix(reuse_allocated=700, reuse_latency=1.1)
    row = deepcopy(rows[0])
    row["environment"] = {
        **_ready_environment(),
        "status": "environment_unavailable",
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }
    with pytest.raises(ValueError, match="unavailable environment"):
        validate_worker_row(row, config=config)
