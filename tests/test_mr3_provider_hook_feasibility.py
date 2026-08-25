"""Tests for the MR3-0 real-provider hook semantic gates."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from copy import deepcopy

import pytest

from boundflow.runtime.mr3_provider_hook_feasibility import (
    ABCROWN_COMMIT,
    AUTO_LIRPA_COMMIT,
    EXPECTED_RUNS,
    MR3_HOOK_WORKER_SCHEMA,
    VNNCOMP_COMMIT,
    canonical_hash,
    derive_summary,
)


def _tensor(shape: tuple[int, ...], pointer: int, *, grad: bool = False):
    stride: list[int] = []
    running = 1
    for dimension in reversed(shape):
        stride.append(running)
        running *= dimension
    return {
        "shape": list(shape),
        "stride": list(reversed(stride)),
        "dtype": "torch.float32",
        "device": "cuda:0",
        "requires_grad": grad,
        "numel": running,
        "data_ptr": pointer,
        "version": 0,
        "content_sha256": f"{pointer:064x}",
    }


def _state_tensor(shape: tuple[int, ...], value: float, *, grad: bool = False):
    receipt = _tensor(shape, max(1, int(abs(value) * 1000) + 1), grad=grad)
    receipt.pop("data_ptr")
    receipt.pop("version")
    receipt["values"] = [value] * receipt["numel"]
    return receipt


def _evaluation(ordinal: int):
    relu_output = _tensor((1, 6, 16, 8, 8), 1000 + ordinal)
    return {
        "evaluation_ordinal": ordinal,
        "start_node": "/49",
        "relu_name": "/input-24",
        "conv_name": "/input-20",
        "relu_incoming_lower_a": _tensor((1, 6, 16, 8, 8), 2000 + ordinal),
        "preactivation_lower": _tensor((6, 16, 8, 8), 3000 + ordinal),
        "preactivation_upper": _tensor((6, 16, 8, 8), 4000 + ordinal),
        "compressed_alpha": _tensor((2, 1, 6, 86), 5000 + ordinal, grad=True),
        "alpha_feature_index_shapes": [[86], [86], [86]],
        "alpha_feature_index_unique_count": 86,
        "target_beta_tensor_count": 1,
        "target_beta_numel": 0,
        "relu_output_lower_a": relu_output,
        "relu_lower_bias": _tensor((1, 6), 6000 + ordinal),
        "conv_input_lower_a": dict(relu_output),
        "conv_weight": _tensor((16, 16, 3, 3), 7000 + ordinal),
        "conv_bias": _tensor((16,), 8000 + ordinal),
        "conv_output_lower_a": _tensor((1, 6, 16, 8, 8), 9000 + ordinal),
        "conv_lower_bias": _tensor((1, 6), 10000 + ordinal),
    }


def _hook():
    return {
        "topology": {
            "provider_start_node": "/49",
            "relu_name": "/input-24",
            "relu_class": "BoundRelu",
            "conv_name": "/input-20",
            "conv_class": "BoundConv",
            "relu_input_is_conv": True,
        },
        "evaluations": [_evaluation(ordinal) for ordinal in range(10)],
        "counters": {
            "outer_exact_call_count": 1,
            "inner_evaluation_count": 10,
            "relu_original_call_count": 10,
            "conv_original_call_count": 10,
            "replacement_count": 0,
            "fallback_count": 0,
            "eager_count": 0,
            "native_shadow_count": 0,
        },
        "device_before": 0,
        "device_after": 0,
        "stream_before": 1234,
        "stream_after": 1234,
    }


def _run(pair: int, position: int, mode: str):
    outer_state = [_state_tensor((6, 1), -0.25)]
    inner_states = [
        [_state_tensor((6, 1), -0.5 + ordinal * 0.01)] for ordinal in range(10)
    ]
    alpha_state = _state_tensor((2, 1, 6, 86), 0.5, grad=False)
    module_state = [
        {
            "node": "/input-24",
            "attribute": "alpha",
            "ordinal": 0,
            "tensor": alpha_state,
        }
    ]
    value = {
        "schema_version": MR3_HOOK_WORKER_SCHEMA,
        "pair_index": pair,
        "position": position,
        "mode": mode,
        "source": {
            "abcrown_commit": ABCROWN_COMMIT,
            "auto_lirpa_commit": AUTO_LIRPA_COMMIT,
            "vnncomp_commit": VNNCOMP_COMMIT,
        },
        "protocol": {
            "device": "cuda",
            "seed": 100,
            "max_iterations": 1,
            "batch_size": 64,
            "alpha_steps": 5,
            "beta_steps": 10,
        },
        "solver_result": {
            "status": "unknown",
            "success": False,
            "visited_domains": [6],
        },
        "outer_beta_exact_call_count": 1,
        "inner_beta_evaluation_count": 10,
        "outer_result_hash": canonical_hash(outer_state),
        "outer_result_state": outer_state,
        "inner_result_hashes": [canonical_hash(state) for state in inner_states],
        "inner_result_states": inner_states,
        "final_target_alpha_hash": canonical_hash(alpha_state),
        "final_target_alpha_state": alpha_state,
        "final_module_state_hash": canonical_hash(module_state),
        "final_module_state": module_state,
        "hook": _hook() if mode == "probe" else None,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    value["worker_hash"] = canonical_hash(value)
    return value


def _raw():
    return {
        "runs": [_run(pair, position, mode) for pair, position, mode in EXPECTED_RUNS]
    }


def _resign(run):
    unsigned = dict(run)
    unsigned.pop("worker_hash", None)
    run["worker_hash"] = canonical_hash(unsigned)


def test_hook_feasibility_opens_only_bridge_implementation() -> None:
    summary = derive_summary(_raw())
    assert summary["status"] == "VALIDATED-MR3-0-PROVIDER-HOOK-FEASIBILITY"
    assert summary["candidate_bridge_implementation_open"] is True
    assert summary["timing_open"] is False
    assert summary["probe_relu_call_count"] == 20


def test_resigned_adjacency_content_tamper_fails_closed() -> None:
    raw = _raw()
    probe = raw["runs"][1]
    probe["hook"]["evaluations"][3]["conv_input_lower_a"]["content_sha256"] = "f" * 64
    _resign(probe)
    with pytest.raises(ValueError, match="adjacency"):
        derive_summary(raw)


def test_resigned_alpha_shape_tamper_fails_closed() -> None:
    raw = _raw()
    probe = raw["runs"][1]
    probe["hook"]["evaluations"][0]["compressed_alpha"]["shape"][-1] = 85
    _resign(probe)
    with pytest.raises(ValueError, match="tensor ABI"):
        derive_summary(raw)


def test_resigned_stream_drift_fails_closed() -> None:
    raw = _raw()
    probe = raw["runs"][1]
    probe["hook"]["stream_after"] += 1
    _resign(probe)
    with pytest.raises(ValueError, match="context drifted"):
        derive_summary(raw)


def test_resigned_provider_numeric_tamper_fails_closed() -> None:
    raw = _raw()
    control = raw["runs"][0]
    control["outer_result_state"][0]["values"][0] += 0.01
    control["outer_result_hash"] = canonical_hash(control["outer_result_state"])
    _resign(control)
    with pytest.raises(ValueError, match="provider numeric state differs"):
        derive_summary(raw)


def test_unsigned_worker_tamper_fails_closed() -> None:
    raw = deepcopy(_raw())
    raw["runs"][0]["inner_beta_evaluation_count"] = 9
    with pytest.raises(ValueError, match="provenance"):
        derive_summary(raw)
