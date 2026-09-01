#!/usr/bin/env python3
"""Independently close the root input Conv/L-infinity transaction in PyTorch."""

# pylint: disable=import-error,too-many-locals,wrong-import-position
# pylint: disable=import-outside-toplevel,consider-using-from-import,not-callable

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _alpha_full(
    raw_alpha: Any,
    coordinates: tuple[Any, ...],
    shape: tuple[int, int, int, int, int],
) -> Any:
    import torch

    full = torch.zeros(shape, dtype=raw_alpha.dtype, device=raw_alpha.device)
    channels, heights, widths = coordinates
    full[:, :, channels, heights, widths] = raw_alpha[0]
    return full


def _relu_terms(incoming: Any, lower: Any, upper: Any, alpha: Any) -> tuple[Any, Any]:
    import torch

    epsilon = torch.finfo(torch.float32).eps
    upper_slope = torch.where(
        lower >= 0,
        torch.ones_like(lower),
        torch.where(
            upper <= 0,
            torch.zeros_like(lower),
            upper / (upper - lower).clamp_min(epsilon),
        ),
    )
    lower_slope = torch.where(
        (lower < 0) & (upper > 0),
        alpha.clamp(0, 1),
        (lower >= 0).to(lower.dtype),
    )
    slope = torch.where(incoming >= 0, lower_slope, upper_slope)
    intercept = torch.where(
        (incoming < 0) & (lower < 0) & (upper > 0),
        -lower * upper_slope,
        torch.zeros_like(incoming),
    )
    return slope, intercept


def _evaluate(payload: dict[str, object]) -> dict[str, object]:
    import torch
    import torch.nn.functional as functional

    evaluations = cast(list[dict[str, Any]], payload["evaluations"])
    metrics: list[dict[str, object]] = []
    for ordinal, item in enumerate(evaluations):
        incoming = item["incoming_lower_a"].cuda().requires_grad_(ordinal < 4)
        lower = item["preactivation_lower"].cuda()
        upper = item["preactivation_upper"].cuda()
        raw_alpha = item["raw_alpha"].cuda().requires_grad_(ordinal < 4)
        weight = item["operator_weight"].cuda()
        operator_bias = item["operator_bias"].cuda()
        center = item["input_center"].cuda()
        input_lower = item["input_lower"].cuda()
        input_upper = item["input_upper"].cuda()
        coordinates = tuple(
            value.cuda().long() for value in item["alpha_feature_indices"]
        )
        alpha = _alpha_full(raw_alpha, coordinates, tuple(incoming.shape))
        slope, intercept = _relu_terms(
            incoming, lower.unsqueeze(0), upper.unsqueeze(0), alpha
        )
        transformed = incoming * slope
        spec, domain, channels, height, width = transformed.shape
        output_a = functional.conv_transpose2d(
            transformed.reshape(spec * domain, channels, height, width),
            weight,
            stride=2,
            padding=1,
            output_padding=1,
        ).reshape(spec, domain, 3, 32, 32)
        if output_a.requires_grad:
            output_a.retain_grad()
        relu_bias = (incoming * intercept).sum(dim=(2, 3, 4))
        conv_bias = (transformed * operator_bias.view(1, 1, -1, 1, 1)).sum(
            dim=(2, 3, 4)
        )
        output_bias = relu_bias + conv_bias
        input_midpoint = (input_lower + input_upper) * 0.5
        input_radius = (input_upper - input_lower) * 0.5
        concrete_sd = (
            output_a * input_midpoint.unsqueeze(0)
            - output_a.abs() * input_radius.unsqueeze(0)
        ).sum(dim=(2, 3, 4))
        concrete_lower = concrete_sd.transpose(0, 1)
        differences = {
            "output_a": float(
                (output_a.detach().cpu() - item["output_lower_a"]).abs().max()
            ),
            "output_bias": float(
                (output_bias.detach().cpu() - item["output_bias"]).abs().max()
            ),
            "input_center": float((center - input_midpoint).abs().max()),
            "concrete_lower": float(
                (concrete_lower.detach().cpu() - item["concrete_lower"]).abs().max()
            ),
        }
        gradient_differences: dict[str, float] = {}
        gradient_sign_exact: dict[str, bool] = {}
        if ordinal < 4:
            torch.autograd.backward(
                (concrete_lower, output_bias),
                (
                    item["concrete_lower_gradient"].cuda(),
                    item["output_bias_gradient"].cuda(),
                ),
            )
            gradients = {
                "output_a": output_a.grad,
                "incoming": incoming.grad,
                "raw_alpha": raw_alpha.grad,
            }
            references = {
                "output_a": item["output_lower_a_gradient"],
                "incoming": item["incoming_lower_a_gradient"],
                "raw_alpha": item["raw_alpha_gradient"],
            }
            for name, value in gradients.items():
                if value is None:
                    raise ValueError(f"root input oracle gradient absent: {name}")
                reference = references[name]
                gradient_differences[name] = float(
                    (value.detach().cpu() - reference).abs().max()
                )
                gradient_sign_exact[name] = bool(
                    torch.equal(torch.sign(value.detach().cpu()), torch.sign(reference))
                )
        metrics.append(
            {
                "ordinal": ordinal,
                "forward_max_abs_diff": differences,
                "gradient_max_abs_diff": gradient_differences,
                "gradient_sign_exact": gradient_sign_exact,
            }
        )
    maxima = [
        cast(float, value)
        for metric in metrics
        for section in ("forward_max_abs_diff", "gradient_max_abs_diff")
        for value in cast(dict[str, object], metric[section]).values()
    ]
    return {
        "schema_version": "boundflow.root-crown-input-oracle/v1",
        "evaluation_count": len(metrics),
        "metrics": metrics,
        "maximum_absolute_difference": max(maxima),
        "all_gradient_sign_exact": all(
            cast(bool, value)
            for metric in metrics
            for value in cast(dict[str, object], metric["gradient_sign_exact"]).values()
        ),
        "dense_input_coefficient_materialized": True,
        "oracle_only": True,
        "performance_claimed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the independent expression on all captured evaluations."""

    import torch

    args = _parse_args()
    payload = torch.load(args.capture, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "boundflow.root-crown-input-tensors/v1":
        raise ValueError("root CROWN input oracle capture differs")
    result = _evaluate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
