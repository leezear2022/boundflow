#!/usr/bin/env python3
"""Validate and time fused root input Conv/L-infinity TVM/TIR."""

# pylint: disable=import-error,too-many-locals,wrong-import-position
# pylint: disable=duplicate-code,import-outside-toplevel
# pylint: disable=consider-using-from-import,not-callable

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.backends.tvm.root_crown_input_domain import (  # noqa: E402
    RootCrownInputDomainTemplateV1,
)
from boundflow.runtime.root_crown_input_domain_tir import (  # noqa: E402
    RootCrownInputDomainTensorsV1,
    RootCrownInputDomainTIRExecutorV1,
    validate_root_crown_input_domain_tensors_v1,
)
from scripts.probe_root_crown_input_oracle import (  # noqa: E402
    _alpha_full,
    _relu_terms,
)
from scripts.probe_root_crown_residual_tir import _coordinates  # noqa: E402


def _template(item: dict[str, Any]) -> RootCrownInputDomainTemplateV1:
    import torch

    incoming = item["incoming_lower_a"]
    center = item["input_center"]
    major, minor = torch.cuda.get_device_capability()
    return RootCrownInputDomainTemplateV1(
        spec_count=int(incoming.shape[0]),
        domain_count=int(incoming.shape[1]),
        output_channels=int(incoming.shape[2]),
        output_height=int(incoming.shape[3]),
        output_width=int(incoming.shape[4]),
        input_channels=int(center.shape[1]),
        input_height=int(center.shape[2]),
        input_width=int(center.shape[3]),
        alpha_coordinates=_coordinates(item["alpha_feature_indices"]),
        compute_capability=f"sm_{major}{minor}",
        thread_extent=128,
    )


def _tensors(item: dict[str, Any]) -> RootCrownInputDomainTensorsV1:
    return RootCrownInputDomainTensorsV1(
        item["incoming_lower_a"].cuda(),
        item["preactivation_lower"].cuda(),
        item["preactivation_upper"].cuda(),
        item["raw_alpha"].cuda(),
        item["operator_weight"].cuda(),
        item["operator_bias"].cuda(),
        item["input_center"].cuda(),
        ((item["input_upper"] - item["input_lower"]) * 0.5).cuda(),
    )


def _oracle(
    item: dict[str, Any], *, backward: bool
) -> tuple[Any, Any, Any | None, Any | None]:
    import torch
    import torch.nn.functional as functional

    incoming = item["incoming_lower_a"].cuda().requires_grad_(backward)
    raw_alpha = item["raw_alpha"].cuda().requires_grad_(backward)
    lower = item["preactivation_lower"].cuda()
    upper = item["preactivation_upper"].cuda()
    weight = item["operator_weight"].cuda()
    operator_bias = item["operator_bias"].cuda()
    center = item["input_center"].cuda()
    radius = ((item["input_upper"] - item["input_lower"]) * 0.5).cuda()
    coordinates = tuple(value.cuda().long() for value in item["alpha_feature_indices"])
    alpha = _alpha_full(raw_alpha, coordinates, tuple(incoming.shape))
    slope, intercept = _relu_terms(
        incoming, lower.unsqueeze(0), upper.unsqueeze(0), alpha
    )
    transformed = incoming * slope
    spec, domain, channels, height, width = transformed.shape
    coefficient = functional.conv_transpose2d(
        transformed.reshape(spec * domain, channels, height, width),
        weight,
        stride=2,
        padding=1,
        output_padding=1,
    ).reshape(spec, domain, 3, 32, 32)
    concrete = (
        (coefficient * center.unsqueeze(0) - coefficient.abs() * radius.unsqueeze(0))
        .sum(dim=(2, 3, 4))
        .transpose(0, 1)
    )
    bias = (
        incoming * intercept + transformed * operator_bias.view(1, 1, -1, 1, 1)
    ).sum(dim=(2, 3, 4))
    if not backward:
        return concrete, bias, None, None
    gradients = torch.autograd.grad(
        (concrete, bias),
        (incoming, raw_alpha),
        grad_outputs=(
            item["concrete_lower_gradient"].cuda(),
            item["output_bias_gradient"].cuda(),
        ),
    )
    return concrete, bias, gradients[0], gradients[1]


def _events(operation: Any, repeats: int) -> list[float]:
    import torch

    values: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def _run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    payload = torch.load(args.capture, map_location="cpu", weights_only=True)
    evaluations = cast(list[dict[str, Any]], payload["evaluations"])
    if (
        payload.get("schema_version") != "boundflow.root-crown-input-tensors/v1"
        or len(evaluations) != 5
    ):
        raise ValueError("root CROWN input-domain probe capture differs")
    template = _template(evaluations[0])
    started = time.perf_counter_ns()
    executor = RootCrownInputDomainTIRExecutorV1(template)
    executor.prepare()
    prepare_ns = time.perf_counter_ns() - started
    metrics: list[dict[str, object]] = []
    maximum = 0.0
    all_sign_exact = True
    for ordinal, item in enumerate(evaluations):
        tensors = _tensors(item)
        validate_root_crown_input_domain_tensors_v1(tensors, template)
        concrete, bias = executor.forward(tensors)
        incoming_gradient, alpha_gradient = executor.backward(
            tensors,
            (
                item["concrete_lower_gradient"].cuda()
                if ordinal < 4
                else torch.zeros_like(concrete)
            ),
            (
                item["output_bias_gradient"].cuda()
                if ordinal < 4
                else torch.zeros_like(bias)
            ),
        )
        oracle = _oracle(item, backward=ordinal < 4)
        candidates = (concrete, bias, incoming_gradient, alpha_gradient)
        names = ("concrete", "bias", "incoming_gradient", "alpha_gradient")
        differences: dict[str, float] = {}
        signs: dict[str, bool] = {}
        for name, candidate, reference in zip(names, candidates, oracle):
            if reference is None:
                continue
            difference = float(
                (candidate.detach() - reference.detach()).abs().max().item()
            )
            sign_exact = bool(
                torch.equal(
                    torch.sign(candidate.detach()), torch.sign(reference.detach())
                )
            )
            differences[name] = difference
            signs[name] = sign_exact
            maximum = max(maximum, difference)
            all_sign_exact = all_sign_exact and sign_exact
        metrics.append(
            {
                "ordinal": ordinal,
                "maximum_absolute_differences": differences,
                "sign_exact": signs,
            }
        )
    timing_item = evaluations[0]
    timing_tensors = _tensors(timing_item)
    concrete_gradient = timing_item["concrete_lower_gradient"].cuda()
    bias_gradient = timing_item["output_bias_gradient"].cuda()
    for _ in range(args.warmup):
        executor.forward(timing_tensors)
        executor.backward(timing_tensors, concrete_gradient, bias_gradient)
        _oracle(timing_item, backward=True)
    torch.cuda.synchronize()
    candidate_ms = _events(
        lambda: (
            executor.forward(timing_tensors),
            executor.backward(timing_tensors, concrete_gradient, bias_gradient),
        ),
        args.repeats,
    )
    oracle_ms = _events(lambda: _oracle(timing_item, backward=True), args.repeats)
    candidate_median = statistics.median(candidate_ms)
    oracle_median = statistics.median(oracle_ms)
    return {
        "schema_version": "boundflow.root-crown-input-domain-tir-probe/v1",
        "evaluation_count": len(metrics),
        "metrics": metrics,
        "maximum_absolute_difference": maximum,
        "all_sign_exact": all_sign_exact,
        "template_hash": template.stable_hash(),
        "unscheduled_tir_hash": executor.compiled.unscheduled_tir_hash,
        "scheduled_tir_hash": executor.compiled.scheduled_tir_hash,
        "device_source_hash": executor.compiled.device_source_hash,
        "workspace_inventory": [
            [name, list(shape)] for name, shape in executor.compiled.workspace_inventory
        ],
        "dense_input_coefficient_externalized": False,
        "dense_input_coefficient_internal_scratch_present": any(
            "coefficient" in name and shape == template.coefficient_shape
            for name, shape in executor.compiled.workspace_inventory
        ),
        "prepare_ns": prepare_ns,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "candidate_median_ms": candidate_median,
        "native_oracle_median_ms": oracle_median,
        "native_over_candidate_speedup": oracle_median / candidate_median,
        "forward_launch_count": executor.forward_launch_count,
        "backward_launch_count": executor.backward_launch_count,
        "fallback_count": executor.fallback_count,
        "dlpack_pointer_count": executor.pointer_count,
        "dlpack_pointer_exact_count": executor.pointer_exact_count,
        "performance_claimed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    """Run correctness and local performance probes."""

    args = _parse_args()
    result = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
