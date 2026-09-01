"""Mechanical semantic gates for the MR3 production bridge formal artifact."""

# pylint: disable=missing-function-docstring,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence, cast

FORMAL_SCHEMA = "boundflow.mr3-production-bridge-formal/v1"
WORKER_SCHEMA = "boundflow.mr3-production-p-anchor-bridge-worker/v1"
STATUS = "VALIDATED-MR3-P-PRODUCTION-BRIDGE-CORRECTNESS"
SOURCE_COMMIT = "baddf7c9fb7d6881a4429de484847bcfe2b52368"
ABCROWN_COMMIT = "e5c7e17bf0488843acb77b7519f59876717a49f4"
AUTO_LIRPA_COMMIT = "5a098e8f9fb5786a428a024981d833d303921f2d"
VNNCOMP_COMMIT = "90419aadcf06cf543ce5c1706cae1059dc9fa6cf"
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
PROPERTY_SHA256 = "89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff"
EXPECTED_RUNS = (
    (0, 0, "provider"),
    (0, 1, "bridge"),
    (1, 0, "bridge"),
    (1, 1, "provider"),
    (2, 0, "provider"),
    (2, 1, "bridge"),
    (3, 0, "bridge"),
    (3, 1, "provider"),
    (4, 0, "provider"),
    (4, 1, "bridge"),
)


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_worker(worker: Mapping[str, Any], *, mode: str) -> None:
    unsigned = dict(worker)
    worker_hash = unsigned.pop("worker_hash", None)
    source = worker.get("source")
    protocol = worker.get("protocol")
    expected_atomic = {
        "exact_call_launch_count": 1,
        "staged_emit_count": 1,
        "atomic_commit_count": 1,
        "rollback_count": 0,
    }
    if (
        worker.get("schema_version") != WORKER_SCHEMA
        or worker.get("mode") != mode
        or worker_hash != canonical_hash(unsigned)
        or not isinstance(source, Mapping)
        or source.get("abcrown_commit") != ABCROWN_COMMIT
        or source.get("auto_lirpa_commit") != AUTO_LIRPA_COMMIT
        or source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or source.get("model_sha256") != MODEL_SHA256
        or source.get("property_sha256") != PROPERTY_SHA256
        or protocol
        != {
            "device": "cuda",
            "seed": 100,
            "max_iterations": 1,
            "batch_size": 64,
            "alpha_steps": 5,
            "beta_steps": 10,
        }
        or worker.get("timing_recorded") is not False
        or worker.get("performance_claimed") is not False
        or worker.get("atomic_receipt") != expected_atomic
        or len(worker.get("region_states", [])) != 10
        or len(worker.get("evaluation_trajectory", [])) != 10
        or len(worker.get("mutation_trajectory", [])) != 9
        or not isinstance(worker.get("final_clip_state"), Mapping)
    ):
        raise ValueError("MR3 formal worker provenance/count differs")
    if worker.get("solver_result") != {
        "status": "verified",
        "success": True,
        "visited_domains": [6],
    }:
        raise ValueError("MR3 formal solver result differs")
    bridge_receipt = worker.get("bridge_receipt")
    if mode == "provider":
        if bridge_receipt is not None:
            raise ValueError("MR3 provider unexpectedly has bridge receipt")
    elif bridge_receipt != {
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
    }:
        raise ValueError("MR3 bridge receipt differs")
    for ordinal, row in enumerate(worker["evaluation_trajectory"]):
        if row.get("evaluation_ordinal") != ordinal:
            raise ValueError("MR3 evaluation order differs")
    for ordinal, row in enumerate(worker["mutation_trajectory"]):
        if (
            row.get("mutation_ordinal") != ordinal
            or row.get("optimizer_step") != float(ordinal + 1)
            or not isinstance(row.get("clamp_mask"), Mapping)
        ):
            raise ValueError("MR3 mutation order differs")


def _compare_tensor(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
    path: str,
) -> tuple[float, int]:
    left_meta = {
        key: value
        for key, value in left.items()
        if key not in {"values", "content_sha256", "stride"}
    }
    right_meta = {
        key: value
        for key, value in right.items()
        if key not in {"values", "content_sha256", "stride"}
    }
    left_values = left.get("values")
    right_values = right.get("values")
    if (
        left_meta != right_meta
        or not isinstance(left_values, list)
        or not isinstance(right_values, list)
        or len(left_values) != len(right_values)
    ):
        raise ValueError(f"MR3 tensor identity differs at {path}")
    maximum = 0.0
    for ordinal, (left_value, right_value) in enumerate(zip(left_values, right_values)):
        if (
            not isinstance(left_value, (int, float))
            or not isinstance(right_value, (int, float))
            or not math.isfinite(float(left_value))
            or not math.isfinite(float(right_value))
        ):
            raise ValueError(f"MR3 tensor nonfinite at {path}")
        difference = abs(float(left_value) - float(right_value))
        if difference > atol + rtol * abs(float(left_value)) or (
            (left_value > 0) - (left_value < 0) != (right_value > 0) - (right_value < 0)
        ):
            raise ValueError(f"MR3 tensor numeric drift at {path}[{ordinal}]")
        maximum = max(maximum, difference)
    return maximum, len(left_values)


def _compare_payload(
    left: object,
    right: object,
    *,
    atol: float,
    rtol: float,
    path: str,
) -> tuple[float, int]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ValueError(f"MR3 structure differs at {path}")
        if "values" in left:
            return _compare_tensor(left, right, atol=atol, rtol=rtol, path=path)
        maximum = 0.0
        count = 0
        for key in sorted(left):
            child_maximum, child_count = _compare_payload(
                left[key],
                right[key],
                atol=atol,
                rtol=rtol,
                path=f"{path}.{key}",
            )
            maximum = max(maximum, child_maximum)
            count += child_count
        return maximum, count
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(f"MR3 list length differs at {path}")
        maximum = 0.0
        count = 0
        for ordinal, (left_item, right_item) in enumerate(zip(left, right)):
            child_maximum, child_count = _compare_payload(
                left_item,
                right_item,
                atol=atol,
                rtol=rtol,
                path=f"{path}[{ordinal}]",
            )
            maximum = max(maximum, child_maximum)
            count += child_count
        return maximum, count
    if isinstance(left, float) and isinstance(right, float):
        difference = abs(left - right)
        if (
            not math.isfinite(left)
            or not math.isfinite(right)
            or difference > (atol + rtol * abs(left))
        ):
            raise ValueError(f"MR3 scalar numeric drift at {path}")
        return difference, 1
    if left != right:
        raise ValueError(f"MR3 discrete drift at {path}")
    return 0.0, 0


def _validate_rollback(rollback: Mapping[str, Any]) -> None:
    unsigned = dict(rollback)
    worker_hash = unsigned.pop("worker_hash", None)
    receipt = rollback.get("atomic_receipt")
    if (
        rollback.get("schema_version") != WORKER_SCHEMA
        or rollback.get("mode") != "bridge"
        or rollback.get("injected_failure_evaluation") != 5
        or rollback.get("caught_failure") != "MR3 injected candidate failure"
        or rollback.get("timing_recorded") is not False
        or rollback.get("performance_claimed") is not False
        or worker_hash != canonical_hash(unsigned)
        or not isinstance(receipt, Mapping)
        or receipt.get("exact_call_launch_count") != 1
        or receipt.get("staged_emit_count") != 0
        or receipt.get("atomic_commit_count") != 0
        or receipt.get("rollback_count") != 1
        or receipt.get("owner_tensor_count") != 12
        or receipt.get("owner_content_hash_before")
        != receipt.get("owner_content_hash_after")
        or receipt.get("owner_pointer_hash_before")
        != receipt.get("owner_pointer_hash_after")
        or not _is_sha256(receipt.get("owner_content_hash_before"))
        or not _is_sha256(receipt.get("owner_pointer_hash_before"))
        or not isinstance(receipt.get("version_delta_min"), int)
        or receipt.get("version_delta_min", 0) < 1
        or receipt.get("version_delta_max", 0) < receipt.get("version_delta_min", 0)
    ):
        raise ValueError("MR3 atomic rollback receipt differs")


def _pair_metric(
    pair_index: int, provider: Mapping[str, Any], bridge: Mapping[str, Any]
) -> dict[str, object]:
    general_fields: Sequence[str] = (
        "solver_result",
        "region_states",
        "evaluation_trajectory",
        "inner_result_states",
        "outer_result_state",
        "final_target_alpha_state",
        "final_module_state",
        "final_clip_state",
    )
    general_maximum = 0.0
    general_count = 0
    for field in general_fields:
        maximum, count = _compare_payload(
            provider[field],
            bridge[field],
            atol=2.0e-4,
            rtol=2.0e-4,
            path=field,
        )
        general_maximum = max(general_maximum, maximum)
        general_count += count
    optimizer_maximum = 0.0
    optimizer_count = 0
    for ordinal, (provider_row, bridge_row) in enumerate(
        zip(provider["mutation_trajectory"], bridge["mutation_trajectory"])
    ):
        for field in (
            "gradient",
            "alpha_pre_clamp",
            "exp_avg",
            "exp_avg_sq",
            "alpha_post_clamp",
        ):
            maximum, count = _compare_payload(
                provider_row[field],
                bridge_row[field],
                atol=2.0e-5,
                rtol=2.0e-5,
                path=f"mutation[{ordinal}].{field}",
            )
            optimizer_maximum = max(optimizer_maximum, maximum)
            optimizer_count += count
        for field in ("mutation_ordinal", "optimizer_step", "lr_used", "clamp_mask"):
            if provider_row[field] != bridge_row[field]:
                raise ValueError(
                    f"MR3 optimizer discrete drift at mutation[{ordinal}].{field}"
                )
    metric: dict[str, object] = {
        "pair_index": pair_index,
        "general_maximum_absolute_difference": general_maximum,
        "general_element_count": general_count,
        "optimizer_maximum_absolute_difference": optimizer_maximum,
        "optimizer_element_count": optimizer_count,
        "sign_exact": True,
        "allclose": True,
    }
    metric["metric_hash"] = canonical_hash(metric)
    return metric


def derive_summary(raw: Mapping[str, Any]) -> dict[str, object]:
    unsigned = dict(raw)
    raw_hash = unsigned.pop("raw_hash", None)
    runs_value = raw.get("runs")
    rollback = raw.get("rollback_probe")
    if (
        raw.get("schema_version") != FORMAL_SCHEMA
        or raw.get("source_commit") != SOURCE_COMMIT
        or raw.get("run_order") != [list(run) for run in EXPECTED_RUNS]
        or raw_hash != canonical_hash(unsigned)
        or not isinstance(runs_value, list)
        or len(runs_value) != 10
        or not isinstance(rollback, Mapping)
    ):
        raise ValueError("MR3 formal raw provenance differs")
    _validate_rollback(rollback)
    runs: list[Mapping[str, Any]] = []
    for expected, wrapper in zip(EXPECTED_RUNS, runs_value):
        if (
            not isinstance(wrapper, Mapping)
            or (wrapper.get("pair_index"), wrapper.get("position"), wrapper.get("mode"))
            != expected
        ):
            raise ValueError("MR3 formal run order differs")
        worker = wrapper.get("worker")
        if not isinstance(worker, Mapping):
            raise ValueError("MR3 formal worker is absent")
        _validate_worker(worker, mode=expected[2])
        runs.append(wrapper)
    metrics: list[dict[str, object]] = []
    for pair_index in range(5):
        pair = [run for run in runs if run["pair_index"] == pair_index]
        provider = next(run["worker"] for run in pair if run["mode"] == "provider")
        bridge = next(run["worker"] for run in pair if run["mode"] == "bridge")
        metrics.append(_pair_metric(pair_index, provider, bridge))
    summary: dict[str, object] = {
        "schema_version": FORMAL_SCHEMA,
        "status": STATUS,
        "source_commit": SOURCE_COMMIT,
        "pair_count": 5,
        "fresh_process_count": 10,
        "run_order": [list(run) for run in EXPECTED_RUNS],
        "candidate_forward_count": 50,
        "candidate_backward_count": 45,
        "atomic_rollback_probe_count": 1,
        "pair_metrics": metrics,
        "general_maximum_absolute_difference": max(
            cast(float, metric["general_maximum_absolute_difference"])
            for metric in metrics
        ),
        "optimizer_maximum_absolute_difference": max(
            cast(float, metric["optimizer_maximum_absolute_difference"])
            for metric in metrics
        ),
        "timing_open": False,
        "multi_site_open": False,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


__all__ = [
    "EXPECTED_RUNS",
    "FORMAL_SCHEMA",
    "SOURCE_COMMIT",
    "STATUS",
    "canonical_hash",
    "derive_summary",
]
