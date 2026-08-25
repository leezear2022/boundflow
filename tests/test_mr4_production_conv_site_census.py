"""Synthetic and fully re-signed gates for MR4 production Conv census."""

# pylint: disable=missing-function-docstring,duplicate-code

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Callable

import pytest

from boundflow.runtime import mr4_production_conv_site_census as census


def _meta(shape: list[int], *, requires_grad: bool = False) -> dict[str, object]:
    numel = math.prod(shape)
    return {
        "shape": shape,
        "stride": [1] * len(shape),
        "dtype": "torch.float32",
        "device": "cuda:0",
        "requires_grad": requires_grad,
        "numel": numel,
        "element_size": 4,
        "contiguous": True,
    }


def _state(value: float) -> dict[str, object]:
    return {
        "shape": [1],
        "stride": [1],
        "dtype": "torch.float32",
        "device": "cuda:0",
        "requires_grad": False,
        "numel": 1,
        "content_sha256": "0" * 64,
        "values": [value],
    }


def _row(evaluation: int, site: str) -> dict[str, object]:
    contract = census.SITE_CONTRACTS[site]
    return {
        "site": site,
        "evaluation_ordinal": evaluation,
        "grad_enabled": evaluation < 9,
        "start_node": "/49",
        "relu_name": contract["relu_name"],
        "conv_name": contract["conv_name"],
        "lower_only": True,
        "incoming_lower_a": _meta(
            contract["incoming_shape"], requires_grad=evaluation < 9
        ),
        "preactivation_lower": _meta(contract["bounds_shape"]),
        "preactivation_upper": _meta(contract["bounds_shape"]),
        "bounds_finite": True,
        "lower_le_upper": True,
        "compressed_alpha": _meta(contract["alpha_shape"], requires_grad=True),
        "alpha_feature_index_shapes": [[1], [1], [1]],
        "reconstructed_full_alpha": _meta(contract["full_alpha_shape"]),
        "beta_tensor_count": 1,
        "beta_shapes": [[6, 0]],
        "beta_numel": 0,
        "relu_output_lower_a": _meta(
            contract["incoming_shape"], requires_grad=evaluation < 9
        ),
        "relu_lower_bias": _meta([1, 6]),
        "conv_input_lower_a": _meta(
            contract["incoming_shape"], requires_grad=evaluation < 9
        ),
        "relu_conv_handoff_content_exact": True,
        "relu_conv_handoff_pointer_exact": False,
        "conv_weight": _meta(contract["weight_shape"]),
        "conv_bias": _meta(contract["bias_shape"]),
        "conv_output_lower_a": _meta(
            contract["output_shape"], requires_grad=evaluation < 9
        ),
        "conv_lower_bias": _meta([1, 6], requires_grad=evaluation < 9),
        "forward_mac_units": contract["mac_units"],
        "candidate_minimum_materialization_bytes": contract["materialization_bytes"],
    }


def _worker(run_index: int) -> dict[str, Any]:
    rows = [
        _row(evaluation, site) for evaluation in range(10) for site in census.SITE_ORDER
    ]
    worker: dict[str, Any] = {
        "schema_version": census.WORKER_SCHEMA,
        "run_index": run_index,
        "source": {
            "abcrown_commit": census.ABCROWN_COMMIT,
            "auto_lirpa_commit": census.AUTO_LIRPA_COMMIT,
            "vnncomp_commit": census.VNNCOMP_COMMIT,
            "model_sha256": census.MODEL_SHA256,
            "property_sha256": census.PROPERTY_SHA256,
        },
        "protocol": {
            "device": "cuda",
            "seed": 100,
            "max_iterations": 1,
            "batch_size": 64,
            "alpha_steps": 5,
            "beta_steps": 10,
            "property_cache": "cold_isolated_copy",
            "candidate_executed": False,
        },
        "solver_result": {
            "status": "verified",
            "success": True,
            "visited_domains": [6],
        },
        "outer_exact_call_count": 1,
        "inner_evaluation_count": 10,
        "census": {
            "target_start": "/49",
            "topology": deepcopy(census.TOPOLOGY),
            "rows": rows,
            "counters": {
                "row_count": 30,
                "relu_calls": {"C0": 10, "C1": 10, "C2": 10},
                "conv_calls": {"C0": 10, "C1": 10, "C2": 10},
                "unexpected_target_start_calls": 0,
                "replacement_count": 0,
                "timing_observation_count": 0,
            },
            "device_before": 0,
            "device_after": 0,
            "stream_before": 0,
            "stream_after": 0,
        },
        "outer_result_state": [_state(1.0)],
        "final_target_alpha_state": _state(0.5),
        "final_module_state": [_state(-0.25)],
        "timing_recorded": False,
        "performance_claimed": False,
    }
    worker["worker_hash"] = census.canonical_hash(worker)
    return worker


def _raw() -> dict[str, Any]:
    raw: dict[str, Any] = {
        "schema_version": census.FORMAL_SCHEMA,
        "source_commit": census.SOURCE_COMMIT,
        "run_order": list(census.EXPECTED_RUNS),
        "runs": [
            {"run_index": run_index, "worker": _worker(run_index)}
            for run_index in census.EXPECTED_RUNS
        ],
    }
    raw["raw_hash"] = census.canonical_hash(raw)
    return raw


def _resign(raw: dict[str, Any]) -> None:
    for wrapper in raw.get("runs", []):
        worker = wrapper.get("worker")
        if isinstance(worker, dict):
            worker.pop("worker_hash", None)
            worker["worker_hash"] = census.canonical_hash(worker)
    raw.pop("raw_hash", None)
    raw["raw_hash"] = census.canonical_hash(raw)


def test_mr4_census_opens_only_mr5_correctness_preregistration() -> None:
    summary = census.derive_summary(_raw())
    assert summary["status"] == census.OPEN_STATUS
    assert summary["run_count"] == 5
    assert summary["row_count"] == 150
    assert summary["eligible_total_mac_ratio_to_p"] == 4.5
    assert summary["new_site_mac_ratio_to_p"] == 3.5
    assert summary["candidate_minimum_materialization_bytes_per_evaluation"] == 344_136
    assert summary["projected_candidate_forward_launch_count"] == 30
    assert summary["projected_candidate_backward_launch_count"] == 27
    assert summary["mr5_correctness_preregistration_open"] is True
    assert summary["timing_open"] is False
    assert summary["performance_claimed"] is False


def _source(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["source"]["abcrown_commit"] = "0" * 40


def _protocol(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["protocol"]["candidate_executed"] = True


def _run_index(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["run_index"] = 1


def _solver(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["solver_result"]["visited_domains"] = [5]


def _topology(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["topology"].pop()


def _delete_row(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"].pop()


def _row_order(raw: dict[str, Any]) -> None:
    rows = raw["runs"][0]["worker"]["census"]["rows"]
    rows[0], rows[1] = rows[1], rows[0]


def _evaluation(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"][0]["evaluation_ordinal"] = 1


def _grad(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"][0]["grad_enabled"] = False


def _beta(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"][0]["beta_shapes"] = [[6, 1]]


def _handoff(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"][0][
        "relu_conv_handoff_content_exact"
    ] = False


def _shape(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"][0]["conv_weight"]["shape"] = [1]


def _mac(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"][0]["forward_mac_units"] = 1


def _materialization(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["census"]["rows"][0][
        "candidate_minimum_materialization_bytes"
    ] = 1


def _timing(raw: dict[str, Any]) -> None:
    raw["runs"][0]["worker"]["timing_recorded"] = True


def _semantic(raw: dict[str, Any]) -> None:
    raw["runs"][1]["worker"]["outer_result_state"][0]["values"][0] = 2.0


ATTACKS: tuple[Callable[[dict[str, Any]], None], ...] = (
    _source,
    _protocol,
    _run_index,
    _solver,
    _topology,
    _delete_row,
    _row_order,
    _evaluation,
    _grad,
    _beta,
    _handoff,
    _shape,
    _mac,
    _materialization,
    _timing,
    _semantic,
)


@pytest.mark.parametrize("attack", ATTACKS, ids=lambda item: item.__name__)
def test_mr4_fully_resigned_tamper_is_rejected(
    attack: Callable[[dict[str, Any]], None],
) -> None:
    raw = deepcopy(_raw())
    attack(raw)
    _resign(raw)
    with pytest.raises((ValueError, TypeError)):
        census.derive_summary(raw)
