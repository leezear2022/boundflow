"""Synthetic and fully re-signed gates for MR3 production bridge timing."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Callable

import pytest

from boundflow.runtime import mr3_production_bridge_timing as timing


def _tensor(value: float) -> dict[str, object]:
    return {
        "shape": [1],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "requires_grad": False,
        "stride": [1],
        "values": [value],
        "content_sha256": "0" * 64,
    }


def _module_receipt() -> dict[str, object]:
    value: dict[str, object] = {
        "module_hash": "1" * 64,
        "device_source_hash": "2" * 64,
        "tvm_version": "0.23.dev0",
        "exported_symbols": [
            "boundflow_cibc_dense_exact_forward_v3",
            "boundflow_cibc_dense_exact_backward_v3",
        ],
        "global_workspace_bytes": 0,
        "dummy_forward_launch_count": 1,
        "dummy_backward_launch_count": 1,
        "dummy_fallback_count": 0,
        "dummy_eager_count": 0,
    }
    value["receipt_hash"] = timing.canonical_hash(value)
    return value


def _bridge_receipt() -> dict[str, object]:
    return {
        "evaluation_count": 10,
        "forward_launch_count": 10,
        "backward_launch_count": 9,
        "empty_beta_tensor_count": 10,
        "empty_beta_numel": 0,
        "relu_conv_content_match_count": 10,
        "relu_conv_pointer_match_count": 0,
        "persistent_dense_a_count": 0,
        "fallback_count": 0,
        "eager_count": 0,
        "native_shadow_count": 0,
        "timing_recorded": False,
        "performance_claimed": False,
    }


def _worker(mode: str, *, host_ns: int, event_ms: float) -> dict[str, Any]:
    gpu = {
        "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "driver_version": "590.01",
        "temperature.gpu": "40",
        "power.draw": "20.0",
        "clocks.current.graphics": "2100",
        "clocks.current.memory": "7001",
        "power.limit": "[N/A]",
    }
    measurement = {
        "host_ns": host_ns,
        "cuda_event_ms": event_ms,
        "device_before": 0,
        "device_after": 0,
        "stream_before": 0,
        "stream_after": 0,
        "base_allocated_bytes": 100,
        "base_reserved_bytes": 200,
        "peak_allocated_bytes": 120 if mode == "provider" else 124,
        "peak_reserved_bytes": 220 if mode == "provider" else 224,
    }
    worker: dict[str, Any] = {
        "schema_version": timing.WORKER_SCHEMA,
        "mode": mode,
        "source": {
            "abcrown_commit": timing.ABCROWN_COMMIT,
            "auto_lirpa_commit": timing.AUTO_LIRPA_COMMIT,
            "vnncomp_commit": timing.VNNCOMP_COMMIT,
            "model_sha256": timing.MODEL_SHA256,
            "property_sha256": timing.PROPERTY_SHA256,
        },
        "protocol": {
            "device": "cuda",
            "seed": 100,
            "max_iterations": 1,
            "batch_size": 64,
            "alpha_steps": 5,
            "beta_steps": 10,
            "formal_observation_enabled": False,
            "compile_timed": False,
            "dummy_module_warm_timed": False,
        },
        "gpu_before": gpu,
        "gpu_after": dict(gpu),
        "device_before": 0,
        "device_after": 0,
        "stream_before": 0,
        "stream_after": 0,
        "solver_result": {
            "status": "verified",
            "success": True,
            "visited_domains": [6],
        },
        "outer_result_state": [_tensor(1.0)],
        "final_target_alpha_state": _tensor(0.5),
        "final_module_state": [_tensor(-0.25)],
        "measurement": measurement,
        "candidate_module_receipt": _module_receipt() if mode == "bridge" else None,
        "bridge_receipt": _bridge_receipt() if mode == "bridge" else None,
        "timing_recorded": True,
        "performance_claimed": False,
    }
    worker["worker_hash"] = timing.canonical_hash(worker)
    return worker


def _raw(*, speedup: float = 1.06) -> dict[str, Any]:
    runs = []
    for pair, position, mode in timing.EXPECTED_RUNS:
        provider_ns = 106_000_000
        bridge_ns = round(provider_ns / speedup)
        provider_event = 106.0
        bridge_event = provider_event / speedup
        runs.append(
            {
                "pair_index": pair,
                "position": position,
                "mode": mode,
                "worker": _worker(
                    mode,
                    host_ns=provider_ns if mode == "provider" else bridge_ns,
                    event_ms=provider_event if mode == "provider" else bridge_event,
                ),
            }
        )
    raw: dict[str, Any] = {
        "schema_version": timing.TIMING_SCHEMA,
        "source_commit": timing.SOURCE_COMMIT,
        "run_order": [list(run) for run in timing.EXPECTED_RUNS],
        "runs": runs,
    }
    raw["raw_hash"] = timing.canonical_hash(raw)
    return raw


def _resign(raw: dict[str, Any]) -> None:
    for wrapper in raw.get("runs", []):
        worker = wrapper.get("worker")
        if isinstance(worker, dict):
            worker.pop("worker_hash", None)
            worker["worker_hash"] = timing.canonical_hash(worker)
    raw.pop("raw_hash", None)
    raw["raw_hash"] = timing.canonical_hash(raw)


def test_mr3_timing_go_and_no_go_are_mechanical() -> None:
    go = timing.derive_summary(_raw(speedup=1.06))
    assert go["status"] == timing.GO_STATUS
    assert go["performance_claimed"] is True
    assert go["same_solver_complete_query_timing_open"] is True
    assert go["host_event_direction_consistent_count"] == 6
    no_go = timing.derive_summary(_raw(speedup=0.99))
    assert no_go["status"] == timing.NO_GO_STATUS
    assert no_go["performance_claimed"] is False
    assert no_go["same_solver_complete_query_timing_open"] is False


def _mutate_host_zero(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["measurement"]["host_ns"] = 0


def _mutate_event_zero(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["measurement"]["cuda_event_ms"] = 0.0


def _mutate_peak_below_base(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["measurement"]["peak_allocated_bytes"] = 99


def _mutate_source(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["source"]["abcrown_commit"] = "0" * 40


def _mutate_protocol(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["protocol"]["beta_steps"] = 9


def _mutate_solver(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["solver_result"]["visited_domains"] = [5]


def _mutate_semantic(raw: dict[str, Any]) -> None:
    raw["runs"][1]["worker"]["outer_result_state"][0]["values"][0] = 2.0


def _mutate_module_hash(raw: dict[str, Any]) -> None:
    receipt = raw["runs"][1]["worker"]["candidate_module_receipt"]
    receipt["module_hash"] = "3" * 64
    receipt.pop("receipt_hash")
    receipt["receipt_hash"] = timing.canonical_hash(receipt)


def _mutate_module_receipt_hash(raw: dict[str, Any]) -> None:
    raw["runs"][1]["worker"]["candidate_module_receipt"]["receipt_hash"] = "4" * 64


def _mutate_bridge_count(raw: dict[str, Any]) -> None:
    raw["runs"][1]["worker"]["bridge_receipt"]["forward_launch_count"] = 9


def _mutate_stream(raw: dict[str, Any]) -> None:
    worker = raw["runs"][0]["worker"]
    worker["stream_after"] = 1
    worker["measurement"]["stream_after"] = 1


def _mutate_device_projection(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["device_after"] = 1


def _mutate_order(raw: dict[str, Any]) -> None:
    raw["runs"][0], raw["runs"][1] = raw["runs"][1], raw["runs"][0]


def _mutate_mode(raw: dict[str, Any]) -> None:
    raw["runs"][0]["mode"] = "bridge"


def _mutate_delete_run(raw: dict[str, Any]) -> None:
    raw["runs"].pop()


def _mutate_performance_claim(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["performance_claimed"] = True


ATTACKS: tuple[Callable[[dict[str, Any]], None], ...] = (
    _mutate_host_zero,
    _mutate_event_zero,
    _mutate_peak_below_base,
    _mutate_source,
    _mutate_protocol,
    _mutate_solver,
    _mutate_semantic,
    _mutate_module_hash,
    _mutate_module_receipt_hash,
    _mutate_bridge_count,
    _mutate_stream,
    _mutate_device_projection,
    _mutate_order,
    _mutate_mode,
    _mutate_delete_run,
    _mutate_performance_claim,
)


def test_timing_worker_is_unchanged_from_frozen_source_commit() -> None:
    root = Path(__file__).resolve().parents[1]
    path = "scripts/run_mr3_production_bridge_timing_worker.py"
    historical = subprocess.run(
        ("git", "show", f"{timing.SOURCE_COMMIT}:{path}"),
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    assert (
        hashlib.sha256(historical).hexdigest()
        == hashlib.sha256((root / path).read_bytes()).hexdigest()
    )
    assert len(timing.EXPECTED_RUNS) == 12
    assert [mode for _, _, mode in timing.EXPECTED_RUNS] == [
        "provider",
        "bridge",
        "bridge",
        "provider",
        "provider",
        "bridge",
        "bridge",
        "provider",
        "provider",
        "bridge",
        "bridge",
        "provider",
    ]


@pytest.mark.parametrize("attack", ATTACKS, ids=lambda item: item.__name__)
def test_fully_resigned_timing_tamper_is_rejected(
    attack: Callable[[dict[str, Any]], None],
) -> None:
    raw = deepcopy(_raw())
    attack(raw)
    _resign(raw)
    with pytest.raises(ValueError):
        timing.derive_summary(raw)
