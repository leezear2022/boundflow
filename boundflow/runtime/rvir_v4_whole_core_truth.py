"""Typed semantic validation for RVIR-v4 whole-core production truth."""

# pylint: disable=too-many-branches,too-many-statements,too-many-locals
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, cast

import torch

from .rvir_v4_production_state import production_tensor_sha256

WHOLE_CORE_TRUTH_SCHEMA = "boundflow.rvir-v4-whole-core-truth/v1"
WHOLE_POST_TRUTH_SCHEMA = "boundflow.rvir-v4-whole-post-truth/v1"
PROVIDER_ACTIVATIONS = (
    "/input-4",
    "/input-12",
    "/input-16",
    "/input-24",
    "/45",
    "/48",
)
PROVIDER_PREACTIVATIONS = (
    "/input",
    "/input-8",
    "/39",
    "/input-20",
    "/44",
    "/input-28",
)
CORE_FIELD_NAMES = {
    "lb",
    "ub",
    "lb_last",
    "ub_last",
    "new_x_Ls",
    "new_x_Us",
    "c",
    "decision_thresh",
    "depths",
    "thresholds",
    "input_split_idx",
    "primal_x",
    "x_Ls",
    "x_Us",
}
TENSOR_RECORD_KEYS = {
    "shape",
    "dtype",
    "source_device",
    "content_sha256",
    "value",
}
PARITY_ATOL = 2e-4
PARITY_RTOL = 2e-4


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def whole_core_truth_metadata(value: object) -> object:
    """Strip raw tensors while retaining schema, device and content identities."""

    if isinstance(value, Mapping):
        if set(value) == TENSOR_RECORD_KEYS:
            return {key: item for key, item in sorted(value.items()) if key != "value"}
        return {
            str(key): whole_core_truth_metadata(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [whole_core_truth_metadata(item) for item in value]
    return value


def _validate_tensor_record(value: object, *, label: str) -> torch.Tensor:
    if not isinstance(value, Mapping) or set(value) != TENSOR_RECORD_KEYS:
        raise TypeError(f"RVIR-v4 whole-core tensor record differs: {label}")
    tensor = value.get("value")
    shape = value.get("shape")
    digest = value.get("content_sha256")
    if (
        not torch.is_tensor(tensor)
        or not isinstance(shape, list)
        or shape != list(cast(torch.Tensor, tensor).shape)
        or value.get("dtype") != str(cast(torch.Tensor, tensor).dtype)
        or not isinstance(value.get("source_device"), str)
        or not isinstance(digest, str)
        or digest != production_tensor_sha256(cast(torch.Tensor, tensor))
    ):
        raise ValueError(f"RVIR-v4 whole-core tensor identity differs: {label}")
    return cast(torch.Tensor, tensor)


def _validate_tree(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        if set(value) == TENSOR_RECORD_KEYS:
            _validate_tensor_record(value, label=label)
            return
        for key, item in value.items():
            _validate_tree(item, label=f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_tree(item, label=f"{label}[{index}]")
        return
    if value is not None and not isinstance(value, (bool, int, float, str)):
        raise TypeError(f"RVIR-v4 whole-core tree leaf differs: {label}")


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"RVIR-v4 whole-core mapping differs: {label}")
    return cast(Mapping[str, Any], value)


def _compare_truth_tree(
    reference: object,
    observed: object,
    *,
    label: str,
    metrics: dict[str, float | int],
) -> None:
    """Compare independently captured truth while ignoring content identities."""

    if isinstance(reference, Mapping):
        if set(reference) == TENSOR_RECORD_KEYS:
            lhs = _validate_tensor_record(reference, label=f"reference.{label}")
            rhs = _validate_tensor_record(observed, label=f"observed.{label}")
            if (
                reference.get("shape") != cast(Mapping[str, Any], observed).get("shape")
                or reference.get("dtype")
                != cast(Mapping[str, Any], observed).get("dtype")
                or reference.get("source_device")
                != cast(Mapping[str, Any], observed).get("source_device")
            ):
                raise ValueError(
                    f"RVIR-v4 whole-core tensor schema parity differs: {label}"
                )
            if lhs.is_floating_point():
                if not torch.allclose(
                    lhs,
                    rhs,
                    atol=PARITY_ATOL,
                    rtol=PARITY_RTOL,
                    equal_nan=False,
                ):
                    raise ValueError(
                        f"RVIR-v4 whole-core numeric semantic parity differs: {label}"
                    )
                if not torch.equal(torch.sign(lhs), torch.sign(rhs)):
                    raise ValueError(
                        f"RVIR-v4 whole-core sign semantic parity differs: {label}"
                    )
                finite = torch.isfinite(lhs) & torch.isfinite(rhs)
                if bool(finite.any()):
                    difference = float(
                        torch.max(torch.abs(lhs[finite] - rhs[finite])).item()
                    )
                    metrics["max_abs_diff"] = max(
                        float(metrics["max_abs_diff"]), difference
                    )
                metrics["sign_element_count"] = int(
                    metrics["sign_element_count"]
                ) + int(lhs.numel())
            elif not torch.equal(lhs, rhs):
                raise ValueError(
                    f"RVIR-v4 whole-core discrete tensor parity differs: {label}"
                )
            metrics["tensor_count"] = int(metrics["tensor_count"]) + 1
            if reference.get("content_sha256") != cast(Mapping[str, Any], observed).get(
                "content_sha256"
            ):
                metrics["numeric_digest_difference_count"] = (
                    int(metrics["numeric_digest_difference_count"]) + 1
                )
            return
        if not isinstance(observed, Mapping) or set(reference) != set(observed):
            raise ValueError(f"RVIR-v4 whole-core mapping parity differs: {label}")
        for key in sorted(reference, key=str):
            if key == "truth_hash":
                if reference[key] != observed[key]:
                    metrics["truth_hash_difference_count"] = (
                        int(metrics["truth_hash_difference_count"]) + 1
                    )
                continue
            _compare_truth_tree(
                reference[key], observed[key], label=f"{label}.{key}", metrics=metrics
            )
        return
    if isinstance(reference, list):
        if not isinstance(observed, list) or len(reference) != len(observed):
            raise ValueError(f"RVIR-v4 whole-core list parity differs: {label}")
        for index, item in enumerate(reference):
            _compare_truth_tree(
                item, observed[index], label=f"{label}[{index}]", metrics=metrics
            )
        return
    if isinstance(reference, float):
        if not isinstance(observed, (int, float)) or not torch.isclose(
            torch.tensor(reference, dtype=torch.float64),
            torch.tensor(float(observed), dtype=torch.float64),
            atol=PARITY_ATOL,
            rtol=PARITY_RTOL,
            equal_nan=False,
        ):
            raise ValueError(f"RVIR-v4 whole-core scalar parity differs: {label}")
        return
    if reference != observed:
        raise ValueError(f"RVIR-v4 whole-core discrete parity differs: {label}")


def compare_rvir_v4_whole_core_truth(
    reference_core: Mapping[str, Any],
    reference_post: Mapping[str, Any],
    observed_core: Mapping[str, Any],
    observed_post: Mapping[str, Any],
) -> dict[str, object]:
    """Compare a frozen truth pair against an independent provider rerun."""

    validate_rvir_v4_whole_core_truth(reference_core, reference_post)
    validate_rvir_v4_whole_core_truth(observed_core, observed_post)
    metrics: dict[str, float | int] = {
        "tensor_count": 0,
        "sign_element_count": 0,
        "max_abs_diff": 0.0,
        "numeric_digest_difference_count": 0,
        "truth_hash_difference_count": 0,
    }
    _compare_truth_tree(reference_core, observed_core, label="core", metrics=metrics)
    _compare_truth_tree(reference_post, observed_post, label="post", metrics=metrics)
    result: dict[str, object] = {
        "status": "live-provider-semantic-parity-passed",
        **metrics,
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "sign_exact": True,
        "performance_claimed": False,
    }
    return result


def _validate_rvir_v4_whole_core_result(
    core: Mapping[str, Any],
    post: Mapping[str, Any],
    *,
    provider_update_bounds_call_count: int,
    status: str,
    whole_core_replacement_admitted: bool,
) -> dict[str, object]:
    """Validate one fixed ResNet production whole-core and post result pair."""

    _validate_tree(core, label="core")
    _validate_tree(post, label="post")
    if (
        core.get("schema_version") != WHOLE_CORE_TRUTH_SCHEMA
        or post.get("schema_version") != WHOLE_POST_TRUTH_SCHEMA
        or core.get("performance_claimed") is not False
        or post.get("performance_claimed") is not False
    ):
        raise ValueError("RVIR-v4 whole-core truth schema differs")
    core_semantic = {key: value for key, value in core.items() if key != "truth_hash"}
    post_semantic = {key: value for key, value in post.items() if key != "truth_hash"}
    if core.get("truth_hash") != _canonical_hash(
        whole_core_truth_metadata(core_semantic)
    ) or post.get("truth_hash") != _canonical_hash(
        whole_core_truth_metadata(post_semantic)
    ):
        raise ValueError("RVIR-v4 whole-core truth hash differs")

    fields = _mapping(core.get("fields"), label="fields")
    if set(fields) != CORE_FIELD_NAMES:
        raise ValueError("RVIR-v4 whole-core field inventory differs")
    for name in ("new_x_Ls", "new_x_Us", "input_split_idx", "primal_x", "x_Ls", "x_Us"):
        if fields[name] is not None:
            raise ValueError(f"RVIR-v4 whole-core unused field differs: {name}")
    lb = _validate_tensor_record(fields["lb"], label="fields.lb")
    ub = _validate_tensor_record(fields["ub"], label="fields.ub")
    c = _validate_tensor_record(fields["c"], label="fields.c")
    threshold = _validate_tensor_record(
        fields["decision_thresh"], label="fields.decision_thresh"
    )
    if (
        tuple(lb.shape) != (6, 1)
        or tuple(ub.shape) != (6, 1)
        or tuple(c.shape) != (6, 1, 10)
        or tuple(threshold.shape) != (6, 1)
        or not bool(torch.isfinite(lb).all())
        or not bool(torch.isinf(ub).all())
    ):
        raise ValueError("RVIR-v4 whole-core bound result differs")

    working = _mapping(
        core.get("working_intermediate_bounds"), label="working intermediate"
    )
    if set(working) != set(PROVIDER_PREACTIVATIONS):
        raise ValueError("RVIR-v4 whole-core intermediate inventory differs")
    for name, bounds_raw in working.items():
        bounds = _mapping(bounds_raw, label=f"working.{name}")
        if set(bounds) != {"lower", "upper"}:
            raise ValueError("RVIR-v4 whole-core intermediate polarity differs")
        lower = _validate_tensor_record(bounds["lower"], label=f"working.{name}.lower")
        upper = _validate_tensor_record(bounds["upper"], label=f"working.{name}.upper")
        if lower.shape != upper.shape or not bool((lower <= upper).all()):
            raise ValueError("RVIR-v4 whole-core intermediate bound differs")

    branch = _mapping(core.get("branch_trace"), label="branch trace")
    branch_input = _mapping(branch.get("input"), label="branch input")
    l_as = _mapping(branch_input.get("lAs"), label="branch lAs")
    l_a_data = _mapping(l_as.get("_data"), label="branch lA data")
    if set(l_a_data) != set(PROVIDER_ACTIVATIONS):
        raise ValueError("RVIR-v4 whole-core lA inventory differs")
    for name, record in l_a_data.items():
        tensor = _validate_tensor_record(record, label=f"branch lA.{name}")
        if int(tensor.shape[0]) != 6 or int(tensor.shape[1]) != 1:
            raise ValueError("RVIR-v4 whole-core lA axes differ")
    candidates = branch.get("candidate_splits")
    child_lowers = branch.get("candidate_child_lowers")
    if (
        branch.get("provider_update_bounds_call_count")
        != provider_update_bounds_call_count
        or not isinstance(candidates, list)
        or not isinstance(child_lowers, list)
        or len(candidates) != 3
        or len(child_lowers) != 3
    ):
        raise ValueError("RVIR-v4 whole-core KFSB candidate inventory differs")
    for ordinal, record in enumerate(child_lowers):
        tensor = _validate_tensor_record(record, label=f"candidate lower {ordinal}")
        if tuple(tensor.shape) != (24, 1) or not bool(torch.isfinite(tensor).all()):
            raise ValueError("RVIR-v4 whole-core KFSB child lower differs")

    decision = _mapping(core.get("branching_decision"), label="core decision")
    final_decision = _mapping(branch.get("final_decision"), label="branch decision")
    if (
        decision != final_decision
        or decision.get("split_depth") != 1
        or decision.get("batch_size") != 6
        or not isinstance(decision.get("decision"), list)
        or len(cast(list[object], decision["decision"])) != 6
        or decision.get("points") is not None
    ):
        raise ValueError("RVIR-v4 whole-core final decision differs")
    if core.get("n_verified") != 0 or core.get("n_splits") != 6:
        raise ValueError("RVIR-v4 whole-core domain accounting differs")

    post_lowers = _mapping(post.get("lower_bounds"), label="post lower bounds")
    post_decision = _mapping(post.get("decision_info"), label="post decision")
    if len(post_lowers) != 1 or post_decision.get("batch_size") != 6:
        raise ValueError("RVIR-v4 whole post result differs")
    post_lower = _validate_tensor_record(
        next(iter(post_lowers.values())), label="post final lower"
    )
    if tuple(post_lower.shape) != (6, 1) or not torch.equal(post_lower, lb.cpu()):
        raise ValueError("RVIR-v4 whole post/core lower binding differs")

    summary: dict[str, object] = {
        "status": status,
        "core_count": 1,
        "domain_count": 6,
        "intermediate_count": len(working),
        "lA_count": len(l_a_data),
        "kfsb_candidate_count": len(candidates),
        "provider_update_bounds_call_count": provider_update_bounds_call_count,
        "branching_decision": decision["decision"],
        "n_verified": 0,
        "n_splits": 6,
        "core_truth_hash": core["truth_hash"],
        "post_truth_hash": post["truth_hash"],
        "whole_core_replacement_admitted": whole_core_replacement_admitted,
        "b2_same_solver_timing_admitted": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = _canonical_hash(summary)
    return summary


def validate_rvir_v4_whole_core_truth(
    core: Mapping[str, Any], post: Mapping[str, Any]
) -> dict[str, object]:
    """Validate one fixed ResNet provider whole-core and post truth pair."""

    return _validate_rvir_v4_whole_core_result(
        core,
        post,
        provider_update_bounds_call_count=3,
        status="validated-whole-core-truth",
        whole_core_replacement_admitted=False,
    )


def validate_rvir_v4_live_return_result(
    core: Mapping[str, Any], post: Mapping[str, Any]
) -> dict[str, object]:
    """Validate one BoundFlow-owned core result consumed by the provider post path."""

    return _validate_rvir_v4_whole_core_result(
        core,
        post,
        provider_update_bounds_call_count=0,
        status="validated-live-return-result",
        whole_core_replacement_admitted=True,
    )


def compare_rvir_v4_live_return_truth(
    reference_core: Mapping[str, Any],
    reference_post: Mapping[str, Any],
    observed_core: Mapping[str, Any],
    observed_post: Mapping[str, Any],
) -> dict[str, object]:
    """Compare a BoundFlow-owned live return against frozen provider truth."""

    validate_rvir_v4_whole_core_truth(reference_core, reference_post)
    validate_rvir_v4_live_return_result(observed_core, observed_post)
    normalized_core = copy.deepcopy(dict(observed_core))
    branch = _mapping(normalized_core.get("branch_trace"), label="live branch trace")
    normalized_branch = dict(branch)
    normalized_branch["provider_update_bounds_call_count"] = 3
    normalized_core["branch_trace"] = normalized_branch
    semantic = {
        key: value for key, value in normalized_core.items() if key != "truth_hash"
    }
    normalized_core["truth_hash"] = _canonical_hash(whole_core_truth_metadata(semantic))
    parity = compare_rvir_v4_whole_core_truth(
        reference_core,
        reference_post,
        normalized_core,
        observed_post,
    )
    parity.update(
        {
            "status": "live-return-semantic-parity-passed",
            "reference_provider_update_bounds_call_count": 3,
            "observed_provider_update_bounds_call_count": 0,
            "whole_core_replacement_admitted": True,
            "b2_same_solver_timing_admitted": False,
            "performance_claimed": False,
        }
    )
    parity["parity_hash"] = _canonical_hash(parity)
    return parity


__all__ = [
    "compare_rvir_v4_whole_core_truth",
    "compare_rvir_v4_live_return_truth",
    "PARITY_ATOL",
    "PARITY_RTOL",
    "PROVIDER_ACTIVATIONS",
    "PROVIDER_PREACTIVATIONS",
    "validate_rvir_v4_whole_core_truth",
    "validate_rvir_v4_live_return_result",
    "whole_core_truth_metadata",
    "WHOLE_CORE_TRUTH_SCHEMA",
    "WHOLE_POST_TRUTH_SCHEMA",
]
