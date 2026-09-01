#!/usr/bin/env python3
"""Check and time the root residual TIR against a direct PyTorch oracle."""

# pylint: disable=import-error,wrong-import-position,too-many-locals
# pylint: disable=too-many-statements,import-outside-toplevel,too-many-arguments
# pylint: disable=missing-function-docstring,not-callable,duplicate-code

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402
import torch.nn.functional as torch_functional  # noqa: E402

from boundflow.backends.tvm.root_crown_residual import (  # noqa: E402
    RootCrownResidualTemplateV1,
)
from boundflow.runtime.root_crown_residual_tir import (  # noqa: E402
    RootCrownResidualTensorsV1,
    RootCrownResidualTIRExecutorV1,
    execute_root_crown_residual_tir_v1,
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


def _coordinates(values: Sequence[torch.Tensor]) -> tuple[tuple[int, int, int], ...]:
    if len(values) != 3 or any(value.ndim != 1 for value in values):
        raise ValueError("root residual alpha coordinate rank differs")
    lengths = {int(value.numel()) for value in values}
    if len(lengths) != 1:
        raise ValueError("root residual alpha coordinate length differs")
    return cast(
        tuple[tuple[int, int, int], ...],
        tuple(
            tuple(int(values[axis][ordinal]) for axis in range(3))
            for ordinal in range(int(values[0].numel()))
        ),
    )


def _tensors(evaluation: dict[str, Any]) -> RootCrownResidualTensorsV1:
    return RootCrownResidualTensorsV1(
        incoming_lower_a=_cuda(evaluation["incoming_lower_a"], requires_grad=True),
        entry_lower=_cuda(evaluation["entry_lower"], requires_grad=True),
        entry_upper=_cuda(evaluation["entry_upper"], requires_grad=True),
        entry_raw_alpha=_cuda(evaluation["entry_raw_alpha"], requires_grad=True),
        main_conv_weight=_cuda(evaluation["main_conv_weight"]),
        main_conv_bias=_cuda(evaluation["main_conv_bias"]),
        inner_lower=_cuda(evaluation["inner_lower"], requires_grad=True),
        inner_upper=_cuda(evaluation["inner_upper"], requires_grad=True),
        inner_raw_alpha=_cuda(evaluation["inner_raw_alpha"], requires_grad=True),
        inner_conv_weight=_cuda(evaluation["inner_conv_weight"]),
        inner_conv_bias=_cuda(evaluation["inner_conv_bias"]),
    )


def _differentiable_inputs(
    tensors: RootCrownResidualTensorsV1,
) -> tuple[torch.Tensor, ...]:
    return (
        tensors.incoming_lower_a,
        tensors.entry_lower,
        tensors.entry_upper,
        tensors.entry_raw_alpha,
        tensors.inner_lower,
        tensors.inner_upper,
        tensors.inner_raw_alpha,
    )


def _dense_alpha(
    raw_alpha: torch.Tensor,
    coordinates: tuple[tuple[int, int, int], ...],
    shape: tuple[int, int, int],
) -> torch.Tensor:
    result = torch.zeros(
        (*raw_alpha.shape[:3], *shape),
        dtype=raw_alpha.dtype,
        device=raw_alpha.device,
    )
    coordinate_tensor = torch.tensor(
        coordinates, dtype=torch.int64, device=raw_alpha.device
    )
    result[
        :,
        :,
        :,
        coordinate_tensor[:, 0],
        coordinate_tensor[:, 1],
        coordinate_tensor[:, 2],
    ] = raw_alpha
    return result


def _relu_backward_oracle(
    incoming: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    raw_alpha: torch.Tensor,
    coordinates: tuple[tuple[int, int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    epsilon = torch.finfo(torch.float32).eps
    denominator = (upper - lower).clamp_min(epsilon)
    upper_slope = torch.where(
        lower >= 0,
        torch.ones_like(lower),
        torch.where(upper <= 0, torch.zeros_like(lower), upper / denominator),
    )
    full_alpha = _dense_alpha(
        raw_alpha,
        coordinates,
        cast(tuple[int, int, int], tuple(lower.shape[1:])),
    )
    lower_slope = torch.where(
        lower >= 0,
        torch.ones_like(lower),
        torch.where(upper <= 0, torch.zeros_like(lower), full_alpha[0].clamp(0, 1)),
    )
    slope = torch.where(incoming >= 0, lower_slope, upper_slope)
    intercept = torch.where(
        (incoming < 0) & (lower < 0) & (upper > 0),
        -lower * upper_slope,
        torch.zeros_like(incoming),
    )
    return incoming * slope, (incoming * intercept).sum(dim=(-3, -2, -1))


def _oracle(
    tensors: RootCrownResidualTensorsV1,
    template: RootCrownResidualTemplateV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    entry_a, entry_bias = _relu_backward_oracle(
        tensors.incoming_lower_a,
        tensors.entry_lower,
        tensors.entry_upper,
        tensors.entry_raw_alpha,
        template.entry_alpha_coordinates,
    )
    merged = template.spec_count * template.domain_count
    shape = (merged, template.channels, template.height, template.width)
    main_a = torch_functional.conv_transpose2d(
        entry_a.reshape(shape), tensors.main_conv_weight, padding=1
    ).reshape(template.coefficient_shape)
    main_bias = (
        entry_a * tensors.main_conv_bias.reshape(1, 1, template.channels, 1, 1)
    ).sum(dim=(-3, -2, -1))
    inner_a, inner_relu_bias = _relu_backward_oracle(
        main_a,
        tensors.inner_lower,
        tensors.inner_upper,
        tensors.inner_raw_alpha,
        template.inner_alpha_coordinates,
    )
    residual_a = torch_functional.conv_transpose2d(
        inner_a.reshape(shape), tensors.inner_conv_weight, padding=1
    ).reshape(template.coefficient_shape)
    inner_conv_bias = (
        inner_a * tensors.inner_conv_bias.reshape(1, 1, template.channels, 1, 1)
    ).sum(dim=(-3, -2, -1))
    return (
        entry_a + residual_a,
        entry_bias + main_bias + inner_relu_bias + inner_conv_bias,
    )


def _difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _sign_exact(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(torch.sign(left.detach()), torch.sign(right.detach())))


def _elapsed_ms(callback: Any, *, repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        callback()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def _gradients(
    outputs: tuple[torch.Tensor, torch.Tensor],
    tensors: RootCrownResidualTensorsV1,
    output_a_gradient: torch.Tensor,
    output_bias_gradient: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return torch.autograd.grad(
        outputs,
        _differentiable_inputs(tensors),
        grad_outputs=(output_a_gradient, output_bias_gradient),
    )


def _run(args: argparse.Namespace) -> dict[str, object]:
    payload = torch.load(args.tensor_input, map_location="cpu", weights_only=True)
    evaluations = payload.get("evaluations")
    if (
        payload.get("schema_version") != "boundflow.root-crown-residual-tensors/v1"
        or not isinstance(evaluations, list)
        or len(evaluations) != 5
    ):
        raise ValueError("root CROWN residual tensor artifact differs")
    first = cast(dict[str, Any], evaluations[0])
    incoming = cast(torch.Tensor, first["incoming_lower_a"])
    entry_coordinates = _coordinates(first["entry_alpha_feature_indices"])
    inner_coordinates = _coordinates(first["inner_alpha_feature_indices"])
    major, minor = torch.cuda.get_device_capability()
    template = RootCrownResidualTemplateV1(
        spec_count=int(incoming.shape[0]),
        domain_count=int(incoming.shape[1]),
        channels=int(incoming.shape[2]),
        height=int(incoming.shape[3]),
        width=int(incoming.shape[4]),
        entry_alpha_coordinates=entry_coordinates,
        inner_alpha_coordinates=inner_coordinates,
        compute_capability=f"sm_{major}{minor}",
        thread_extent=args.thread_extent,
    )
    compile_started = time.perf_counter_ns()
    executor = RootCrownResidualTIRExecutorV1(template)
    compile_ns = time.perf_counter_ns() - compile_started
    metrics: list[dict[str, object]] = []
    for ordinal, raw_evaluation in enumerate(evaluations):
        evaluation = cast(dict[str, Any], raw_evaluation)
        candidate_tensors = _tensors(evaluation)
        oracle_tensors = _tensors(evaluation)
        candidate = execute_root_crown_residual_tir_v1(candidate_tensors, executor)
        oracle = _oracle(oracle_tensors, template)
        reference = (
            _cuda(evaluation["output_lower_a"]),
            _cuda(evaluation["output_bias"]),
        )
        has_backward = evaluation["output_lower_a_gradient"] is not None
        candidate_gradients: tuple[torch.Tensor, ...] | None = None
        oracle_gradients: tuple[torch.Tensor, ...] | None = None
        gradient_metrics: dict[str, object] = {}
        if has_backward:
            output_a_gradient = _cuda(evaluation["output_lower_a_gradient"])
            output_bias_gradient = _cuda(evaluation["output_bias_gradient"])
            candidate_gradients = _gradients(
                candidate,
                candidate_tensors,
                output_a_gradient,
                output_bias_gradient,
            )
            oracle_gradients = _gradients(
                oracle,
                oracle_tensors,
                output_a_gradient,
                output_bias_gradient,
            )
            for index, field in enumerate(GRADIENT_FIELDS):
                candidate_gradient = candidate_gradients[index]
                oracle_gradient = oracle_gradients[index]
                captured_gradient = _cuda(evaluation[field])
                gradient_metrics[f"{field}_oracle_max_abs_diff"] = _difference(
                    candidate_gradient, oracle_gradient
                )
                gradient_metrics[f"{field}_oracle_sign_exact"] = _sign_exact(
                    candidate_gradient, oracle_gradient
                )
                gradient_metrics[f"{field}_captured_max_abs_diff"] = _difference(
                    candidate_gradient, captured_gradient
                )
        metrics.append(
            {
                "ordinal": ordinal,
                "candidate_reference_a_max_abs_diff": _difference(
                    candidate[0], reference[0]
                ),
                "candidate_reference_bias_max_abs_diff": _difference(
                    candidate[1], reference[1]
                ),
                "candidate_oracle_a_max_abs_diff": _difference(candidate[0], oracle[0]),
                "candidate_oracle_bias_max_abs_diff": _difference(
                    candidate[1], oracle[1]
                ),
                "forward_sign_exact": _sign_exact(candidate[0], reference[0])
                and _sign_exact(candidate[1], reference[1]),
                "has_backward": has_backward,
                **gradient_metrics,
            }
        )

    benchmark = cast(dict[str, Any], evaluations[0])
    candidate_tensors = _tensors(benchmark)
    oracle_tensors = _tensors(benchmark)
    output_a_gradient = _cuda(benchmark["output_lower_a_gradient"])
    output_bias_gradient = _cuda(benchmark["output_bias_gradient"])

    def candidate_iteration() -> None:
        outputs = execute_root_crown_residual_tir_v1(candidate_tensors, executor)
        _gradients(
            outputs,
            candidate_tensors,
            output_a_gradient,
            output_bias_gradient,
        )

    def oracle_iteration() -> None:
        outputs = _oracle(oracle_tensors, template)
        _gradients(
            outputs,
            oracle_tensors,
            output_a_gradient,
            output_bias_gradient,
        )

    for _ in range(5):
        candidate_iteration()
        oracle_iteration()
    candidate_ms = _elapsed_ms(candidate_iteration, repeats=args.repeats)
    oracle_ms = _elapsed_ms(oracle_iteration, repeats=args.repeats)
    torch.cuda.synchronize()
    numerical_values = [
        float(value)
        for row in metrics
        for name, value in row.items()
        if isinstance(value, float) and math.isfinite(value) and "captured" not in name
    ]
    candidate_median = statistics.median(candidate_ms)
    oracle_median = statistics.median(oracle_ms)
    return {
        "schema_version": "boundflow.root-crown-residual-tir-probe/v1",
        "template_hash": template.stable_hash(),
        "unscheduled_tir_hash": executor.compiled.unscheduled_tir_hash,
        "scheduled_tir_hash": executor.compiled.scheduled_tir_hash,
        "device_source_hash": executor.compiled.device_source_hash,
        "workspace_inventory": [
            [name, list(shape)] for name, shape in executor.compiled.workspace_inventory
        ],
        "compile_ns": compile_ns,
        "evaluation_count": len(metrics),
        "metrics": metrics,
        "maximum_oracle_absolute_difference": max(numerical_values),
        "all_oracle_sign_exact": all(
            bool(value)
            for row in metrics
            for name, value in row.items()
            if name.endswith("oracle_sign_exact") or name == "forward_sign_exact"
        ),
        "forward_launch_count": executor.forward_launch_count,
        "backward_launch_count": executor.backward_launch_count,
        "fallback_count": executor.fallback_count,
        "dlpack_pointer_count": executor.pointer_count,
        "dlpack_pointer_exact_count": executor.pointer_exact_count,
        "benchmark_repeats": args.repeats,
        "candidate_median_ms": candidate_median,
        "native_oracle_median_ms": oracle_median,
        "native_over_candidate_speedup": oracle_median / candidate_median,
        "performance_scope": "isolated-root-residual-forward-full-vjp",
        "performance_claimed": False,
        "captured_bound_gradients_include_outside-region_uses": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-input", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--thread-extent", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def main() -> None:
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
