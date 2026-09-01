"""Mechanical gates for the MR4 production Conv-site census."""

# pylint: disable=missing-function-docstring,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence, TypedDict, cast

from .mr3_production_bridge_formal import _compare_payload

FORMAL_SCHEMA = "boundflow.mr4-production-conv-site-census-formal/v1"
WORKER_SCHEMA = "boundflow.mr4-production-conv-site-census-worker/v1"
SOURCE_COMMIT = "1fa4f0f952bae344a24b78aab8b3ca72e6bcd244"
ABCROWN_COMMIT = "e5c7e17bf0488843acb77b7519f59876717a49f4"
AUTO_LIRPA_COMMIT = "5a098e8f9fb5786a428a024981d833d303921f2d"
VNNCOMP_COMMIT = "90419aadcf06cf543ce5c1706cae1059dc9fa6cf"
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
PROPERTY_SHA256 = "89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff"
OPEN_STATUS = "OPEN-MR5-MULTI-CONV-PRODUCTION-BRIDGE-CORRECTNESS-PREREGISTRATION"
NO_GO_STATUS = "VALIDATED-NO-GO-MR4-PRODUCTION-CONV-SITE-CENSUS"
EXPECTED_RUNS = tuple(range(5))
SITE_ORDER = ("C2", "C1", "C0")
TOPOLOGY = [
    {
        "relu_name": "/input-4",
        "relu_class": "BoundRelu",
        "conv_name": "/input",
        "conv_class": "BoundConv",
    },
    {
        "relu_name": "/input-12",
        "relu_class": "BoundRelu",
        "conv_name": "/input-8",
        "conv_class": "BoundConv",
    },
    {
        "relu_name": "/input-24",
        "relu_class": "BoundRelu",
        "conv_name": "/input-20",
        "conv_class": "BoundConv",
    },
]


class SiteContract(TypedDict):
    """Frozen shape and static-cost contract for one direct Conv site."""

    relu_name: str
    conv_name: str
    incoming_shape: list[int]
    bounds_shape: list[int]
    alpha_shape: list[int]
    full_alpha_shape: list[int]
    weight_shape: list[int]
    bias_shape: list[int]
    output_shape: list[int]
    mac_units: int
    materialization_bytes: int


SITE_CONTRACTS: dict[str, SiteContract] = {
    "C0": {
        "relu_name": "/input-4",
        "conv_name": "/input",
        "incoming_shape": [1, 6, 8, 16, 16],
        "bounds_shape": [6, 8, 16, 16],
        "alpha_shape": [2, 1, 6, 164],
        "full_alpha_shape": [6, 8, 16, 16],
        "weight_shape": [8, 3, 3, 3],
        "bias_shape": [8],
        "output_shape": [1, 6, 3, 32, 32],
        "mac_units": 1_327_104,
        "materialization_bytes": 172_056,
    },
    "C1": {
        "relu_name": "/input-12",
        "conv_name": "/input-8",
        "incoming_shape": [1, 6, 16, 8, 8],
        "bounds_shape": [6, 16, 8, 8],
        "alpha_shape": [2, 1, 6, 132],
        "full_alpha_shape": [6, 16, 8, 8],
        "weight_shape": [16, 8, 3, 3],
        "bias_shape": [16],
        "output_shape": [1, 6, 8, 16, 16],
        "mac_units": 1_769_472,
        "materialization_bytes": 98_328,
    },
    "C2": {
        "relu_name": "/input-24",
        "conv_name": "/input-20",
        "incoming_shape": [1, 6, 16, 8, 8],
        "bounds_shape": [6, 16, 8, 8],
        "alpha_shape": [2, 1, 6, 86],
        "full_alpha_shape": [6, 16, 8, 8],
        "weight_shape": [16, 16, 3, 3],
        "bias_shape": [16],
        "output_shape": [1, 6, 16, 8, 8],
        "mac_units": 884_736,
        "materialization_bytes": 73_752,
    },
}


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_source(source: object) -> None:
    if source != {
        "abcrown_commit": ABCROWN_COMMIT,
        "auto_lirpa_commit": AUTO_LIRPA_COMMIT,
        "vnncomp_commit": VNNCOMP_COMMIT,
        "model_sha256": MODEL_SHA256,
        "property_sha256": PROPERTY_SHA256,
    }:
        raise ValueError("MR4 census source differs")


def _validate_protocol(protocol: object) -> None:
    if protocol != {
        "device": "cuda",
        "seed": 100,
        "max_iterations": 1,
        "batch_size": 64,
        "alpha_steps": 5,
        "beta_steps": 10,
        "property_cache": "cold_isolated_copy",
        "candidate_executed": False,
    }:
        raise ValueError("MR4 census protocol differs")


def _shape(meta: Mapping[str, Any], expected: Sequence[int], *, path: str) -> None:
    if (
        meta.get("shape") != list(expected)
        or meta.get("dtype") != "torch.float32"
        or meta.get("device") != "cuda:0"
        or not isinstance(meta.get("numel"), int)
        or meta.get("element_size") != 4
        or not isinstance(meta.get("contiguous"), bool)
    ):
        raise ValueError(f"MR4 census tensor metadata differs at {path}")


def _validate_row(row: Mapping[str, Any], *, evaluation: int, site: str) -> None:
    contract = SITE_CONTRACTS[site]
    if (
        row.get("site") != site
        or row.get("evaluation_ordinal") != evaluation
        or row.get("grad_enabled") is not (evaluation < 9)
        or row.get("start_node") != "/49"
        or row.get("relu_name") != contract["relu_name"]
        or row.get("conv_name") != contract["conv_name"]
        or row.get("lower_only") is not True
        or row.get("bounds_finite") is not True
        or row.get("lower_le_upper") is not True
        or row.get("beta_tensor_count") != 1
        or row.get("beta_shapes") != [[6, 0]]
        or row.get("beta_numel") != 0
        or row.get("relu_conv_handoff_content_exact") is not True
        or row.get("relu_conv_handoff_pointer_exact") is not False
        or row.get("forward_mac_units") != contract["mac_units"]
        or row.get("candidate_minimum_materialization_bytes")
        != contract["materialization_bytes"]
    ):
        raise ValueError(f"MR4 census row invariant differs at {evaluation}.{site}")
    _shape(row["incoming_lower_a"], contract["incoming_shape"], path=f"{site}.incoming")
    _shape(row["preactivation_lower"], contract["bounds_shape"], path=f"{site}.lower")
    _shape(row["preactivation_upper"], contract["bounds_shape"], path=f"{site}.upper")
    _shape(row["compressed_alpha"], contract["alpha_shape"], path=f"{site}.alpha")
    _shape(
        row["reconstructed_full_alpha"],
        contract["full_alpha_shape"],
        path=f"{site}.full_alpha",
    )
    _shape(
        row["conv_input_lower_a"], contract["incoming_shape"], path=f"{site}.conv_input"
    )
    _shape(row["conv_weight"], contract["weight_shape"], path=f"{site}.weight")
    _shape(row["conv_bias"], contract["bias_shape"], path=f"{site}.bias")
    _shape(row["conv_output_lower_a"], contract["output_shape"], path=f"{site}.output")


def validate_worker(worker: Mapping[str, Any], *, run_index: int) -> None:
    unsigned = dict(worker)
    worker_hash = unsigned.pop("worker_hash", None)
    if (
        worker.get("schema_version") != WORKER_SCHEMA
        or worker.get("run_index") != run_index
        or worker_hash != canonical_hash(unsigned)
        or worker.get("solver_result")
        != {"status": "verified", "success": True, "visited_domains": [6]}
        or worker.get("outer_exact_call_count") != 1
        or worker.get("inner_evaluation_count") != 10
        or worker.get("timing_recorded") is not False
        or worker.get("performance_claimed") is not False
    ):
        raise ValueError("MR4 census worker envelope differs")
    _validate_source(worker.get("source"))
    _validate_protocol(worker.get("protocol"))
    census = worker.get("census")
    if not isinstance(census, Mapping):
        raise ValueError("MR4 census receipt is absent")
    counters = census.get("counters")
    rows = census.get("rows")
    if (
        census.get("target_start") != "/49"
        or census.get("topology") != TOPOLOGY
        or census.get("device_before") != census.get("device_after")
        or census.get("stream_before") != census.get("stream_after")
        or counters
        != {
            "row_count": 30,
            "relu_calls": {"C0": 10, "C1": 10, "C2": 10},
            "conv_calls": {"C0": 10, "C1": 10, "C2": 10},
            "unexpected_target_start_calls": 0,
            "replacement_count": 0,
            "timing_observation_count": 0,
        }
        or not isinstance(rows, list)
        or len(rows) != 30
    ):
        raise ValueError("MR4 census receipt differs")
    expected = [(evaluation, site) for evaluation in range(10) for site in SITE_ORDER]
    for row, (evaluation, site) in zip(rows, expected):
        if not isinstance(row, Mapping):
            raise TypeError("MR4 census row is not a mapping")
        _validate_row(row, evaluation=evaluation, site=site)


def _semantic_metric(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, object]:
    maximum = 0.0
    count = 0
    for field in (
        "solver_result",
        "outer_result_state",
        "final_target_alpha_state",
        "final_module_state",
    ):
        field_maximum, field_count = _compare_payload(
            reference[field], candidate[field], atol=2.0e-4, rtol=2.0e-4, path=field
        )
        maximum = max(maximum, field_maximum)
        count += field_count
    metric: dict[str, object] = {
        "reference_run": reference["run_index"],
        "candidate_run": candidate["run_index"],
        "maximum_absolute_difference": maximum,
        "element_count": count,
        "sign_exact": True,
        "allclose": True,
    }
    metric["metric_hash"] = canonical_hash(metric)
    return metric


def derive_summary(raw: Mapping[str, Any]) -> dict[str, object]:
    unsigned = dict(raw)
    raw_hash = unsigned.pop("raw_hash", None)
    runs_value = raw.get("runs")
    if (
        raw.get("schema_version") != FORMAL_SCHEMA
        or raw.get("source_commit") != SOURCE_COMMIT
        or raw.get("run_order") != list(EXPECTED_RUNS)
        or raw_hash != canonical_hash(unsigned)
        or not isinstance(runs_value, list)
        or len(runs_value) != 5
    ):
        raise ValueError("MR4 census raw provenance differs")
    runs: list[Mapping[str, Any]] = []
    for expected, wrapper in zip(EXPECTED_RUNS, runs_value):
        if (
            not isinstance(wrapper, Mapping)
            or wrapper.get("run_index") != expected
            or not isinstance(wrapper.get("worker"), Mapping)
        ):
            raise ValueError("MR4 census run order differs")
        worker = wrapper["worker"]
        validate_worker(worker, run_index=expected)
        runs.append(worker)
    semantic_metrics = [_semantic_metric(runs[0], run) for run in runs[1:]]
    mac_units = {
        site: int(SITE_CONTRACTS[site]["mac_units"]) for site in ("C0", "C1", "C2")
    }
    p_units = mac_units["C2"]
    total_ratio = sum(mac_units.values()) / p_units
    new_ratio = (mac_units["C0"] + mac_units["C1"]) / p_units
    materialization = {
        site: int(SITE_CONTRACTS[site]["materialization_bytes"])
        for site in ("C0", "C1", "C2")
    }
    gates = {
        "fresh_runs": len(runs) == 5,
        "solver_semantics": all(
            bool(metric["allclose"]) for metric in semantic_metrics
        ),
        "topology": True,
        "site_count": len(SITE_CONTRACTS) == 3,
        "trajectory": True,
        "absent_beta": True,
        "handoff": True,
        "metadata_stability": True,
        "eligible_total_mac_ratio": total_ratio >= 1.75,
        "new_site_mac_ratio": new_ratio >= 0.75,
    }
    opened = all(gates.values())
    summary: dict[str, object] = {
        "schema_version": FORMAL_SCHEMA,
        "status": OPEN_STATUS if opened else NO_GO_STATUS,
        "source_commit": SOURCE_COMMIT,
        "run_count": len(runs),
        "site_count": 3,
        "row_count": sum(len(run["census"]["rows"]) for run in runs),
        "site_evaluation_counts": {"C0": 50, "C1": 50, "C2": 50},
        "site_grad_enabled_counts": {"C0": 45, "C1": 45, "C2": 45},
        "site_beta_numel": {"C0": 0, "C1": 0, "C2": 0},
        "site_handoff_content_match_counts": {"C0": 50, "C1": 50, "C2": 50},
        "semantic_metrics": semantic_metrics,
        "global_semantic_maximum_absolute_difference": max(
            cast(float, metric["maximum_absolute_difference"])
            for metric in semantic_metrics
        ),
        "site_forward_mac_units": mac_units,
        "eligible_total_mac_ratio_to_p": total_ratio,
        "new_site_mac_ratio_to_p": new_ratio,
        "site_candidate_minimum_materialization_bytes_per_evaluation": materialization,
        "candidate_minimum_materialization_bytes_per_evaluation": sum(
            materialization.values()
        ),
        "candidate_minimum_materialization_bytes_per_outer_call": 10
        * sum(materialization.values()),
        "projected_candidate_forward_launch_count": 30,
        "projected_candidate_backward_launch_count": 27,
        "gates": gates,
        "mr5_correctness_preregistration_open": opened,
        "timing_open": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


__all__ = [
    "EXPECTED_RUNS",
    "FORMAL_SCHEMA",
    "NO_GO_STATUS",
    "OPEN_STATUS",
    "SITE_CONTRACTS",
    "SOURCE_COMMIT",
    "TOPOLOGY",
    "WORKER_SCHEMA",
    "canonical_hash",
    "derive_summary",
    "validate_worker",
]
