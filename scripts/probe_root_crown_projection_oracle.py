#!/usr/bin/env python3
"""Close the captured root projection residual with an independent Torch oracle."""

# pylint: disable=import-error,wrong-import-position,too-many-locals
# pylint: disable=too-many-statements,too-many-arguments,not-callable,duplicate-code

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402
import torch.nn.functional as torch_functional  # noqa: E402

from scripts.probe_root_crown_residual_tir import (  # noqa: E402
    _coordinates,
    _difference,
    _relu_backward_oracle,
    _sign_exact,
)

GRADIENT_FIELDS = (
    "incoming_lower_a_gradient",
    "entry_lower_gradient",
    "entry_upper_gradient",
    "entry_raw_alpha_gradient",
    "inner_lower_gradient",
    "inner_upper_gradient",
    "inner_raw_alpha_gradient",
)


def _cuda(value: torch.Tensor, *, requires_grad: bool = False) -> torch.Tensor:
    result = value.detach().to("cuda").contiguous().clone()
    result.requires_grad_(requires_grad)
    return result


def _inputs(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "incoming": _cuda(evaluation["incoming_lower_a"], requires_grad=True),
        "entry_lower": _cuda(evaluation["entry_lower"], requires_grad=True),
        "entry_upper": _cuda(evaluation["entry_upper"], requires_grad=True),
        "entry_alpha": _cuda(evaluation["entry_raw_alpha"], requires_grad=True),
        "entry_coordinates": _coordinates(evaluation["entry_alpha_feature_indices"]),
        "outer_weight": _cuda(evaluation["main_outer_conv_weight"]),
        "outer_bias": _cuda(evaluation["main_outer_conv_bias"]),
        "inner_lower": _cuda(evaluation["inner_lower"], requires_grad=True),
        "inner_upper": _cuda(evaluation["inner_upper"], requires_grad=True),
        "inner_alpha": _cuda(evaluation["inner_raw_alpha"], requires_grad=True),
        "inner_coordinates": _coordinates(evaluation["inner_alpha_feature_indices"]),
        "inner_weight": _cuda(evaluation["main_inner_conv_weight"]),
        "inner_bias": _cuda(evaluation["main_inner_conv_bias"]),
        "skip_weight": _cuda(evaluation["skip_conv_weight"]),
        "skip_bias": _cuda(evaluation["skip_conv_bias"]),
    }


def _differentiable(values: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    return tuple(
        cast(torch.Tensor, values[name])
        for name in (
            "incoming",
            "entry_lower",
            "entry_upper",
            "entry_alpha",
            "inner_lower",
            "inner_upper",
            "inner_alpha",
        )
    )


def _oracle(values: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    incoming = cast(torch.Tensor, values["incoming"])
    spec, domain, output_channels, output_height, output_width = incoming.shape
    merged = spec * domain
    entry_a, entry_bias = _relu_backward_oracle(
        incoming,
        values["entry_lower"],
        values["entry_upper"],
        values["entry_alpha"],
        values["entry_coordinates"],
    )
    output_shape = (merged, output_channels, output_height, output_width)
    main_a = torch_functional.conv_transpose2d(
        entry_a.reshape(output_shape),
        values["outer_weight"],
        padding=1,
    ).reshape(incoming.shape)
    outer_bias = (
        entry_a * values["outer_bias"].reshape(1, 1, output_channels, 1, 1)
    ).sum(dim=(-3, -2, -1))
    inner_a, inner_relu_bias = _relu_backward_oracle(
        main_a,
        values["inner_lower"],
        values["inner_upper"],
        values["inner_alpha"],
        values["inner_coordinates"],
    )
    input_channels = int(values["inner_weight"].shape[1])
    expanded_shape = (spec, domain, input_channels, output_height * 2, output_width * 2)
    main_output = torch_functional.conv_transpose2d(
        inner_a.reshape(output_shape),
        values["inner_weight"],
        stride=2,
        padding=1,
        output_padding=1,
    ).reshape(expanded_shape)
    skip_output = torch_functional.conv_transpose2d(
        entry_a.reshape(output_shape),
        values["skip_weight"],
        stride=2,
        output_padding=1,
    ).reshape(expanded_shape)
    inner_bias = (
        inner_a * values["inner_bias"].reshape(1, 1, output_channels, 1, 1)
    ).sum(dim=(-3, -2, -1))
    skip_bias = (
        entry_a * values["skip_bias"].reshape(1, 1, output_channels, 1, 1)
    ).sum(dim=(-3, -2, -1))
    return (
        main_output + skip_output,
        entry_bias + outer_bias + inner_relu_bias + inner_bias + skip_bias,
    )


def _run(args: argparse.Namespace) -> dict[str, object]:
    payload = torch.load(args.tensor_input, map_location="cpu", weights_only=True)
    evaluations = payload.get("evaluations")
    if (
        payload.get("schema_version") != "boundflow.root-crown-projection-tensors/v1"
        or not isinstance(evaluations, list)
        or len(evaluations) != 5
    ):
        raise ValueError("root CROWN projection tensor artifact differs")
    metrics: list[dict[str, object]] = []
    for ordinal, raw in enumerate(evaluations):
        evaluation = cast(dict[str, Any], raw)
        values = _inputs(evaluation)
        outputs = _oracle(values)
        reference = (
            _cuda(evaluation["output_lower_a"]),
            _cuda(evaluation["output_bias"]),
        )
        row: dict[str, object] = {
            "ordinal": ordinal,
            "output_a_max_abs_diff": _difference(outputs[0], reference[0]),
            "output_bias_max_abs_diff": _difference(outputs[1], reference[1]),
            "output_sign_exact": _sign_exact(outputs[0], reference[0])
            and _sign_exact(outputs[1], reference[1]),
            "has_backward": evaluation["output_lower_a_gradient"] is not None,
        }
        if row["has_backward"]:
            gradients = torch.autograd.grad(
                outputs,
                _differentiable(values),
                grad_outputs=(
                    _cuda(evaluation["output_lower_a_gradient"]),
                    _cuda(evaluation["output_bias_gradient"]),
                ),
            )
            for name, gradient in zip(GRADIENT_FIELDS, gradients):
                row[f"{name}_max_abs_diff"] = _difference(
                    gradient, _cuda(evaluation[name])
                )
                row[f"{name}_sign_exact"] = _sign_exact(
                    gradient, _cuda(evaluation[name])
                )
        metrics.append(row)
    local_differences = [
        float(value)
        for row in metrics
        for name, value in row.items()
        if isinstance(value, float)
        and ("output" in name or "alpha_gradient" in name or "incoming" in name)
    ]
    return {
        "schema_version": "boundflow.root-crown-projection-oracle/v1",
        "evaluation_count": len(metrics),
        "metrics": metrics,
        "maximum_local_absolute_difference": max(local_differences),
        "all_local_sign_exact": all(
            bool(value)
            for row in metrics
            for name, value in row.items()
            if name.endswith("sign_exact")
            and ("output" in name or "alpha_gradient" in name or "incoming" in name)
        ),
        "captured_bound_gradients_include_outside_region_uses": True,
        "performance_claimed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-input", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the independent closure and persist its numerical receipt."""

    args = _parse_args()
    result = _run(args)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
