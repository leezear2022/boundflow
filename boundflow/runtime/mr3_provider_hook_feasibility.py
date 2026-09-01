"""Semantic derivation for the MR3-0 real-provider hook preflight."""

# pylint: disable=missing-function-docstring,too-many-boolean-expressions
# pylint: disable=too-many-locals,too-many-statements

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence, cast

MR3_HOOK_SCHEMA = "boundflow.mr3-provider-hook-feasibility/v1"
MR3_HOOK_WORKER_SCHEMA = "boundflow.mr3-provider-hook-worker/v1"
ABCROWN_COMMIT = "e5c7e17bf0488843acb77b7519f59876717a49f4"
AUTO_LIRPA_COMMIT = "5a098e8f9fb5786a428a024981d833d303921f2d"
VNNCOMP_COMMIT = "90419aadcf06cf543ce5c1706cae1059dc9fa6cf"
EXPECTED_RUNS = (
    (0, 0, "control"),
    (0, 1, "probe"),
    (1, 0, "probe"),
    (1, 1, "control"),
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


def _tensor(
    value: Mapping[str, Any],
    shape: Sequence[int],
    *,
    require_grad: bool | None = None,
) -> None:
    if (
        value.get("shape") != list(shape)
        or value.get("dtype") != "torch.float32"
        or value.get("device") != "cuda:0"
        or not _is_sha256(value.get("content_sha256"))
        or not isinstance(value.get("data_ptr"), int)
        or value.get("data_ptr", 0) <= 0
        or not isinstance(value.get("version"), int)
        or (require_grad is not None and value.get("requires_grad") is not require_grad)
    ):
        raise ValueError("MR3-0 tensor ABI differs")


def _validate_source(run: Mapping[str, Any]) -> None:
    source = run.get("source")
    protocol = run.get("protocol")
    unsigned = dict(run)
    worker_hash = unsigned.pop("worker_hash", None)
    inner_states = run.get("inner_result_states")
    inner_hashes = run.get("inner_result_hashes")
    states_bound = (
        isinstance(inner_states, list)
        and isinstance(inner_hashes, list)
        and len(inner_states) == 10
        and inner_hashes == [canonical_hash(state) for state in inner_states]
        and run.get("outer_result_hash")
        == canonical_hash(run.get("outer_result_state"))
        and run.get("final_target_alpha_hash")
        == canonical_hash(run.get("final_target_alpha_state"))
        and run.get("final_module_state_hash")
        == canonical_hash(run.get("final_module_state"))
    )
    if (
        run.get("schema_version") != MR3_HOOK_WORKER_SCHEMA
        or not isinstance(source, Mapping)
        or source.get("abcrown_commit") != ABCROWN_COMMIT
        or source.get("auto_lirpa_commit") != AUTO_LIRPA_COMMIT
        or source.get("vnncomp_commit") != VNNCOMP_COMMIT
        or not isinstance(protocol, Mapping)
        or protocol.get("device") != "cuda"
        or protocol.get("seed") != 100
        or protocol.get("max_iterations") != 1
        or protocol.get("batch_size") != 64
        or protocol.get("alpha_steps") != 5
        or protocol.get("beta_steps") != 10
        or run.get("performance_claimed") is not False
        or run.get("timing_recorded") is not False
        or not states_bound
        or worker_hash != canonical_hash(unsigned)
    ):
        raise ValueError("MR3-0 worker provenance differs")


def _validate_probe(run: Mapping[str, Any]) -> None:
    hook = run.get("hook")
    if not isinstance(hook, Mapping):
        raise ValueError("MR3-0 hook receipt is absent")
    topology = hook.get("topology")
    evaluations = hook.get("evaluations")
    counters = hook.get("counters")
    if (
        topology
        != {
            "provider_start_node": "/49",
            "relu_name": "/input-24",
            "relu_class": "BoundRelu",
            "conv_name": "/input-20",
            "conv_class": "BoundConv",
            "relu_input_is_conv": True,
        }
        or not isinstance(counters, Mapping)
        or counters.get("outer_exact_call_count") != 1
        or counters.get("inner_evaluation_count") != 10
        or counters.get("relu_original_call_count") != 10
        or counters.get("conv_original_call_count") != 10
        or counters.get("replacement_count") != 0
        or counters.get("fallback_count") != 0
        or counters.get("eager_count") != 0
        or counters.get("native_shadow_count") != 0
        or not isinstance(evaluations, list)
        or len(evaluations) != 10
    ):
        raise ValueError("MR3-0 hook topology/count differs")
    if hook.get("device_before") != hook.get("device_after") or hook.get(
        "stream_before"
    ) != hook.get("stream_after"):
        raise ValueError("MR3-0 CUDA context drifted")
    for ordinal, evaluation in enumerate(evaluations):
        if (
            not isinstance(evaluation, Mapping)
            or evaluation.get("evaluation_ordinal") != ordinal
            or evaluation.get("start_node") != "/49"
            or evaluation.get("relu_name") != "/input-24"
            or evaluation.get("conv_name") != "/input-20"
            or evaluation.get("alpha_feature_index_shapes") != [[86], [86], [86]]
            or evaluation.get("alpha_feature_index_unique_count") != 86
            or evaluation.get("target_beta_tensor_count") != 1
            or evaluation.get("target_beta_numel") != 0
        ):
            raise ValueError("MR3-0 evaluation identity differs")
        incoming = evaluation.get("relu_incoming_lower_a")
        relu_output = evaluation.get("relu_output_lower_a")
        conv_input = evaluation.get("conv_input_lower_a")
        conv_output = evaluation.get("conv_output_lower_a")
        lower = evaluation.get("preactivation_lower")
        upper = evaluation.get("preactivation_upper")
        alpha = evaluation.get("compressed_alpha")
        weight = evaluation.get("conv_weight")
        bias = evaluation.get("conv_bias")
        relu_bias = evaluation.get("relu_lower_bias")
        conv_bias = evaluation.get("conv_lower_bias")
        tensors = (
            incoming,
            relu_output,
            conv_input,
            conv_output,
            lower,
            upper,
            alpha,
            weight,
            bias,
            relu_bias,
            conv_bias,
        )
        if not all(isinstance(item, Mapping) for item in tensors):
            raise ValueError("MR3-0 evaluation tensor receipt is absent")
        incoming = cast(Mapping[str, Any], incoming)
        relu_output = cast(Mapping[str, Any], relu_output)
        conv_input = cast(Mapping[str, Any], conv_input)
        conv_output = cast(Mapping[str, Any], conv_output)
        lower = cast(Mapping[str, Any], lower)
        upper = cast(Mapping[str, Any], upper)
        alpha = cast(Mapping[str, Any], alpha)
        weight = cast(Mapping[str, Any], weight)
        bias = cast(Mapping[str, Any], bias)
        relu_bias = cast(Mapping[str, Any], relu_bias)
        conv_bias = cast(Mapping[str, Any], conv_bias)
        _tensor(incoming, (1, 6, 16, 8, 8))
        _tensor(relu_output, (1, 6, 16, 8, 8))
        _tensor(conv_input, (1, 6, 16, 8, 8))
        _tensor(conv_output, (1, 6, 16, 8, 8))
        _tensor(lower, (6, 16, 8, 8), require_grad=False)
        _tensor(upper, (6, 16, 8, 8), require_grad=False)
        _tensor(alpha, (2, 1, 6, 86), require_grad=True)
        _tensor(weight, (16, 16, 3, 3), require_grad=False)
        _tensor(bias, (16,), require_grad=False)
        _tensor(relu_bias, (1, 6))
        _tensor(conv_bias, (1, 6))
        if any(
            relu_output.get(key) != conv_input.get(key)
            for key in ("shape", "stride", "version", "content_sha256")
        ):
            raise ValueError("MR3-0 ReLU-to-Conv adjacency differs")


def _compare_numeric_payload(
    control: object,
    probe: object,
    *,
    path: str = "root",
) -> tuple[float, int]:
    if isinstance(control, Mapping) and isinstance(probe, Mapping):
        if set(control) != set(probe):
            raise ValueError(f"MR3-0 state structure differs at {path}")
        if "values" in control:
            control_meta = {
                key: value
                for key, value in control.items()
                if key not in {"values", "content_sha256"}
            }
            probe_meta = {
                key: value
                for key, value in probe.items()
                if key not in {"values", "content_sha256"}
            }
            control_values = control.get("values")
            probe_values = probe.get("values")
            if (
                control_meta != probe_meta
                or not isinstance(control_values, list)
                or not isinstance(probe_values, list)
                or len(control_values) != len(probe_values)
            ):
                raise ValueError(f"MR3-0 tensor state identity differs at {path}")
            maximum = 0.0
            for ordinal, (left, right) in enumerate(zip(control_values, probe_values)):
                if (
                    not isinstance(left, (float, int))
                    or not isinstance(right, (float, int))
                    or not math.isfinite(float(left))
                    or not math.isfinite(float(right))
                ):
                    raise ValueError("MR3-0 tensor state is nonfinite")
                difference = abs(float(left) - float(right))
                tolerance = 2.0e-4 + 2.0e-4 * abs(float(left))
                if difference > tolerance or (left > 0) - (left < 0) != (
                    (right > 0) - (right < 0)
                ):
                    raise ValueError(
                        f"MR3-0 provider numeric state differs at {path}[{ordinal}]"
                    )
                maximum = max(maximum, difference)
            return maximum, len(control_values)
        maximum = 0.0
        count = 0
        for key in sorted(control):
            child_maximum, child_count = _compare_numeric_payload(
                control[key], probe[key], path=f"{path}.{key}"
            )
            maximum = max(maximum, child_maximum)
            count += child_count
        return maximum, count
    if isinstance(control, list) and isinstance(probe, list):
        if len(control) != len(probe):
            raise ValueError(f"MR3-0 state list differs at {path}")
        maximum = 0.0
        count = 0
        for ordinal, (left, right) in enumerate(zip(control, probe)):
            child_maximum, child_count = _compare_numeric_payload(
                left, right, path=f"{path}[{ordinal}]"
            )
            maximum = max(maximum, child_maximum)
            count += child_count
        return maximum, count
    if control != probe:
        raise ValueError(f"MR3-0 discrete state differs at {path}")
    return 0.0, 0


def derive_summary(raw: Mapping[str, Any]) -> dict[str, object]:
    runs_value = raw.get("runs")
    if not isinstance(runs_value, list) or len(runs_value) != len(EXPECTED_RUNS):
        raise ValueError("MR3-0 run inventory differs")
    runs: list[Mapping[str, Any]] = []
    for expected, value in zip(EXPECTED_RUNS, runs_value):
        if not isinstance(value, Mapping):
            raise ValueError("MR3-0 run payload differs")
        _validate_source(value)
        identity = (value.get("pair_index"), value.get("position"), value.get("mode"))
        if identity != expected:
            raise ValueError("MR3-0 run order differs")
        if (
            value.get("outer_beta_exact_call_count") != 1
            or value.get("inner_beta_evaluation_count") != 10
        ):
            raise ValueError("MR3-0 provider exact-call count differs")
        if value.get("mode") == "probe":
            _validate_probe(value)
        elif value.get("hook") is not None:
            raise ValueError("MR3-0 control unexpectedly installed a node hook")
        runs.append(value)

    pair_hashes: list[str] = []
    pair_metrics: list[dict[str, object]] = []
    for pair_index in range(2):
        pair = [run for run in runs if run.get("pair_index") == pair_index]
        control = next(run for run in pair if run.get("mode") == "control")
        probe = next(run for run in pair if run.get("mode") == "probe")
        if control.get("solver_result") != probe.get("solver_result"):
            raise ValueError("MR3-0 pass-through probe changed provider semantics")
        maximum = 0.0
        element_count = 0
        for field in (
            "outer_result_state",
            "inner_result_states",
            "final_target_alpha_state",
            "final_module_state",
        ):
            field_maximum, field_count = _compare_numeric_payload(
                control.get(field), probe.get(field), path=field
            )
            maximum = max(maximum, field_maximum)
            element_count += field_count
        metric: dict[str, object] = {
            "pair_index": pair_index,
            "maximum_absolute_difference": maximum,
            "element_count": element_count,
            "sign_exact": True,
            "allclose": True,
        }
        metric["metric_hash"] = canonical_hash(metric)
        pair_metrics.append(metric)
        pair_hashes.append(str(metric["metric_hash"]))

    summary: dict[str, object] = {
        "schema_version": MR3_HOOK_SCHEMA,
        "status": "VALIDATED-MR3-0-PROVIDER-HOOK-FEASIBILITY",
        "pair_count": 2,
        "fresh_process_count": 4,
        "outer_exact_call_count_per_run": 1,
        "inner_evaluation_count_per_run": 10,
        "probe_relu_call_count": 20,
        "probe_conv_call_count": 20,
        "provider_start_node": "/49",
        "provider_relu": "/input-24",
        "provider_conv": "/input-20",
        "pair_semantic_hashes": pair_hashes,
        "pair_metrics": pair_metrics,
        "maximum_absolute_difference": max(
            cast(float, metric["maximum_absolute_difference"])
            for metric in pair_metrics
        ),
        "sign_exact": True,
        "candidate_bridge_implementation_open": True,
        "timing_open": False,
        "same_solver_performance_open": False,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


__all__ = [
    "ABCROWN_COMMIT",
    "AUTO_LIRPA_COMMIT",
    "EXPECTED_RUNS",
    "MR3_HOOK_SCHEMA",
    "MR3_HOOK_WORKER_SCHEMA",
    "VNNCOMP_COMMIT",
    "canonical_hash",
    "derive_summary",
]
