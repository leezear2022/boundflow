"""Independent closed-form oracle for the frozen R3-1b2 P-alpha VJP."""

# pylint: disable=missing-function-docstring,too-many-locals,protected-access,not-callable

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as torch_functional

from .r3_structured_owner_custom_backward import (
    R31FullRegionPlanV1,
    _runtime_parts,
)


def _selected_relaxation_value(
    value: torch.Tensor,
    *,
    name: str,
    incoming_coefficient: torch.Tensor,
    relu_pre: Mapping[str, object],
    alphas: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    pre = relu_pre[name]
    lower = getattr(pre, "lower")
    upper = getattr(pre, "upper")
    ambiguous = (lower < 0) & (upper > 0)
    epsilon = torch.finfo(lower.dtype).eps
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
        ambiguous,
        alphas[name].clamp(0, 1),
        torch.where(lower >= 0, torch.ones_like(lower), torch.zeros_like(lower)),
    )
    coefficient = incoming_coefficient[:, 0]
    selected_slope = torch.where(coefficient >= 0, lower_slope, upper_slope)
    selected_intercept = torch.where(
        (coefficient < 0) & ambiguous,
        -lower * upper_slope,
        torch.zeros_like(lower),
    )
    return selected_slope * value + selected_intercept


def evaluate_r31b2_p_alpha_closed_form_v1(
    plan: R31FullRegionPlanV1,
    tensors: tuple[torch.Tensor, ...],
    *,
    input_lower_coefficient: torch.Tensor,
    relu_lower_coefficients: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Evaluate `d(-sum(lower))/d(P-alpha)` without autograd or dense alpha."""

    plan.validate()
    if tuple(input_lower_coefficient.shape) != (6, 1, 3, 32, 32) or set(
        relu_lower_coefficients
    ) != {"17", "19", "23", "25", "28", "31"}:
        raise ValueError("R3-1b2 closed-form coefficient coverage differs")
    module, _spec, _objective, relu_pre, alphas, _betas, _splits = _runtime_parts(
        plan, tensors
    )
    params = module.bindings["params"]
    input_lower = tensors[0]
    input_upper = tensors[1]
    input_coefficient = input_lower_coefficient[:, 0]
    selected_input = torch.where(input_coefficient >= 0, input_lower, input_upper)
    pre17 = torch_functional.conv2d(
        selected_input,
        params["conv1.weight"],
        params["conv1.bias"],
        stride=2,
        padding=1,
    )
    value18 = _selected_relaxation_value(
        pre17,
        name="17",
        incoming_coefficient=relu_lower_coefficients["17"],
        relu_pre=relu_pre,
        alphas=alphas,
    )
    pre19 = torch_functional.conv2d(
        value18,
        params["layer1.0.conv1.weight"],
        params["layer1.0.conv1.bias"],
        stride=2,
        padding=1,
    )
    value20 = _selected_relaxation_value(
        pre19,
        name="19",
        incoming_coefficient=relu_lower_coefficients["19"],
        relu_pre=relu_pre,
        alphas=alphas,
    )
    branch21 = torch_functional.conv2d(
        value20,
        params["layer1.0.conv2.weight"],
        params["layer1.0.conv2.bias"],
        padding=1,
    )
    branch22 = torch_functional.conv2d(
        value18,
        params["layer1.0.shortcut.0.weight"],
        params["layer1.0.shortcut.0.bias"],
        stride=2,
    )
    value24 = _selected_relaxation_value(
        branch21 + branch22,
        name="23",
        incoming_coefficient=relu_lower_coefficients["23"],
        relu_pre=relu_pre,
        alphas=alphas,
    )
    pre25 = torch_functional.conv2d(
        value24,
        params["layer1.1.conv1.weight"],
        params["layer1.1.conv1.bias"],
        padding=1,
    )
    coefficient25 = relu_lower_coefficients["25"][:, 0]
    bounds25 = relu_pre["25"]
    ambiguous25 = (bounds25.lower < 0) & (bounds25.upper > 0)
    dense_gradient = -torch.where(
        (coefficient25 >= 0) & ambiguous25,
        coefficient25 * pre25,
        torch.zeros_like(coefficient25),
    )
    layout25 = next(
        layout for layout in plan.relu_layouts if layout.native_preactivation == "25"
    )
    indices = torch.tensor(
        layout25.alpha_flat_indices, dtype=torch.int64, device=dense_gradient.device
    )
    compressed = torch.zeros(
        (2, 1, 6, 86), dtype=dense_gradient.dtype, device=dense_gradient.device
    )
    compressed[0, 0] = dense_gradient.flatten(1)[:, indices]
    return compressed


__all__ = ["evaluate_r31b2_p_alpha_closed_form_v1"]
