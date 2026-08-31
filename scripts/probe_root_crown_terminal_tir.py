#!/usr/bin/env python3
"""Replay captured root CROWN terminal tensors through PyTorch and TVM/TIR."""

# pylint: disable=import-error,wrong-import-position,too-many-locals
# pylint: disable=too-many-statements,import-outside-toplevel

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402

from boundflow.backends.tvm.root_crown_terminal_linear import (  # noqa: E402
    RootCrownTerminalLinearTemplateV1,
)
from boundflow.runtime.root_crown_terminal_tir import (  # noqa: E402
    RootCrownTerminalTIRExecutorV1,
    RootCrownTerminalTensorsV1,
    execute_root_crown_terminal_tir_v1,
)


def _cuda(value: torch.Tensor, *, requires_grad: bool = False) -> torch.Tensor:
    result = value.detach().to("cuda").contiguous().clone()
    result.requires_grad_(requires_grad)
    return result


def _tensors(evaluation: dict[str, Any]) -> RootCrownTerminalTensorsV1:
    return RootCrownTerminalTensorsV1(
        incoming_lower_a=_cuda(evaluation["incoming_lower_a"]),
        preactivation_lower=_cuda(
            evaluation["preactivation_lower"], requires_grad=True
        ),
        preactivation_upper=_cuda(
            evaluation["preactivation_upper"], requires_grad=True
        ),
        raw_alpha=_cuda(evaluation["raw_alpha"], requires_grad=True),
        operator_weight=_cuda(evaluation["operator_weight"]),
        operator_bias=_cuda(evaluation["operator_bias"]),
    )


def _native(
    tensors: RootCrownTerminalTensorsV1, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    incoming = tensors.incoming_lower_a
    lower = tensors.preactivation_lower
    upper = tensors.preactivation_upper
    ambiguous = (lower < 0) & (upper > 0)
    denominator = (upper - lower).clamp_min(torch.finfo(torch.float32).eps)
    upper_slope = torch.where(
        lower >= 0,
        torch.ones_like(lower),
        torch.where(upper <= 0, torch.zeros_like(lower), upper / denominator),
    )
    lower_slope = (
        torch.where(lower >= 0, torch.ones_like(lower), torch.zeros_like(lower))
        .expand(incoming.shape)
        .clone()
    )
    selected_alpha = tensors.raw_alpha[0].clamp(0, 1)
    lower_slope[..., indices] = selected_alpha
    selected_slope = torch.where(incoming >= 0, lower_slope, upper_slope)
    relu_a = incoming * selected_slope
    output_a = relu_a.matmul(tensors.operator_weight)
    intercept = torch.where(
        (incoming < 0) & ambiguous,
        -lower * upper_slope,
        torch.zeros_like(incoming),
    )
    output_bias = (incoming * intercept + relu_a * tensors.operator_bias).sum(dim=-1)
    return output_a, output_bias


def _maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


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


def _run(args: argparse.Namespace) -> dict[str, object]:
    payload = torch.load(args.tensor_input, map_location="cpu", weights_only=True)
    evaluations = payload.get("evaluations")
    if (
        payload.get("schema_version") != "boundflow.root-crown-terminal-tensors/v1"
        or not isinstance(evaluations, list)
        or len(evaluations) != 5
    ):
        raise ValueError("root CROWN terminal tensor artifact differs")
    first = evaluations[0]
    indices_cpu = first["alpha_feature_indices"][0].to(torch.int64).contiguous()
    incoming = first["incoming_lower_a"]
    weight = first["operator_weight"]
    major, minor = torch.cuda.get_device_capability()
    template = RootCrownTerminalLinearTemplateV1(
        spec_count=int(incoming.shape[0]),
        domain_count=int(incoming.shape[1]),
        current_features=int(incoming.shape[2]),
        previous_features=int(weight.shape[1]),
        alpha_feature_indices=tuple(int(item) for item in indices_cpu.tolist()),
        compute_capability=f"sm_{major}{minor}",
        thread_extent=args.thread_extent,
    )
    compile_started = time.perf_counter_ns()
    executor = RootCrownTerminalTIRExecutorV1(template)
    compile_ns = time.perf_counter_ns() - compile_started
    metrics: list[dict[str, object]] = []
    for ordinal, evaluation in enumerate(evaluations):
        tensors = _tensors(evaluation)
        indices = indices_cpu.to(tensors.incoming_lower_a.device)
        candidate_a, candidate_bias = execute_root_crown_terminal_tir_v1(
            tensors, executor
        )
        has_backward = evaluation["output_lower_a_gradient"] is not None
        candidate_gradients = None
        native_gradients = None
        if has_backward:
            output_a_gradient = _cuda(evaluation["output_lower_a_gradient"])
            output_bias_gradient = _cuda(evaluation["output_bias_gradient"])
            candidate_gradients = torch.autograd.grad(
                (candidate_a, candidate_bias),
                (
                    tensors.raw_alpha,
                    tensors.preactivation_lower,
                    tensors.preactivation_upper,
                ),
                grad_outputs=(output_a_gradient, output_bias_gradient),
            )
        candidate_alpha_gradient = (
            None if candidate_gradients is None else candidate_gradients[0]
        )
        reference_a = _cuda(evaluation["output_lower_a"])
        reference_bias = _cuda(evaluation["relu_lower_bias"]) + _cuda(
            evaluation["linear_lower_bias"]
        )
        reference_alpha_gradient = (
            _cuda(evaluation["raw_alpha_gradient"]) if has_backward else None
        )
        native_tensors = _tensors(evaluation)
        native_a, native_bias = _native(native_tensors, indices)
        if has_backward:
            native_gradients = torch.autograd.grad(
                (native_a, native_bias),
                (
                    native_tensors.raw_alpha,
                    native_tensors.preactivation_lower,
                    native_tensors.preactivation_upper,
                ),
                grad_outputs=(output_a_gradient, output_bias_gradient),
            )
        native_alpha_gradient = (
            None if native_gradients is None else native_gradients[0]
        )
        row: dict[str, object] = {
            "ordinal": ordinal,
            "output_a_max_abs_diff": _maximum_difference(candidate_a, reference_a),
            "output_bias_max_abs_diff": _maximum_difference(
                candidate_bias, reference_bias
            ),
            "raw_alpha_gradient_max_abs_diff": (
                _maximum_difference(
                    cast(torch.Tensor, candidate_alpha_gradient),
                    cast(torch.Tensor, reference_alpha_gradient),
                )
                if has_backward
                else None
            ),
            "candidate_native_a_max_abs_diff": _maximum_difference(
                candidate_a, native_a
            ),
            "candidate_native_bias_max_abs_diff": _maximum_difference(
                candidate_bias, native_bias
            ),
            "candidate_native_alpha_gradient_max_abs_diff": (
                _maximum_difference(
                    cast(torch.Tensor, candidate_alpha_gradient),
                    cast(torch.Tensor, native_alpha_gradient),
                )
                if has_backward
                else None
            ),
            "preactivation_lower_gradient_max_abs_diff": (
                _maximum_difference(
                    cast(tuple[torch.Tensor, ...], candidate_gradients)[1],
                    _cuda(evaluation["preactivation_lower_gradient"]),
                )
                if has_backward
                else None
            ),
            "preactivation_upper_gradient_max_abs_diff": (
                _maximum_difference(
                    cast(tuple[torch.Tensor, ...], candidate_gradients)[2],
                    _cuda(evaluation["preactivation_upper_gradient"]),
                )
                if has_backward
                else None
            ),
            "candidate_native_lower_gradient_max_abs_diff": (
                _maximum_difference(
                    cast(tuple[torch.Tensor, ...], candidate_gradients)[1],
                    cast(tuple[torch.Tensor, ...], native_gradients)[1],
                )
                if has_backward
                else None
            ),
            "candidate_native_upper_gradient_max_abs_diff": (
                _maximum_difference(
                    cast(tuple[torch.Tensor, ...], candidate_gradients)[2],
                    cast(tuple[torch.Tensor, ...], native_gradients)[2],
                )
                if has_backward
                else None
            ),
            "sign_exact": bool(
                torch.equal(torch.sign(candidate_a), torch.sign(reference_a))
                and torch.equal(torch.sign(candidate_bias), torch.sign(reference_bias))
                and (
                    not has_backward
                    or torch.equal(
                        torch.sign(cast(torch.Tensor, candidate_alpha_gradient)),
                        torch.sign(cast(torch.Tensor, reference_alpha_gradient)),
                    )
                )
            ),
        }
        metrics.append(row)
    benchmark_evaluation = evaluations[0]
    output_a_gradient = _cuda(benchmark_evaluation["output_lower_a_gradient"])
    output_bias_gradient = _cuda(benchmark_evaluation["output_bias_gradient"])
    indices = indices_cpu.to("cuda")
    candidate_tensors = _tensors(benchmark_evaluation)
    native_tensors = _tensors(benchmark_evaluation)

    def candidate_iteration() -> None:
        output_a, output_bias = execute_root_crown_terminal_tir_v1(
            candidate_tensors, executor
        )
        torch.autograd.grad(
            (output_a, output_bias),
            (
                candidate_tensors.raw_alpha,
                candidate_tensors.preactivation_lower,
                candidate_tensors.preactivation_upper,
            ),
            grad_outputs=(output_a_gradient, output_bias_gradient),
        )

    def native_iteration() -> None:
        output_a, output_bias = _native(native_tensors, indices)
        torch.autograd.grad(
            (output_a, output_bias),
            (
                native_tensors.raw_alpha,
                native_tensors.preactivation_lower,
                native_tensors.preactivation_upper,
            ),
            grad_outputs=(output_a_gradient, output_bias_gradient),
        )

    for _ in range(5):
        candidate_iteration()
        native_iteration()
    candidate_ms = _elapsed_ms(candidate_iteration, repeats=args.repeats)
    native_ms = _elapsed_ms(native_iteration, repeats=args.repeats)
    candidate_forward_ms = _elapsed_ms(
        lambda: executor.forward(candidate_tensors), repeats=args.repeats
    )
    candidate_backward_ms = _elapsed_ms(
        lambda: executor.backward(
            candidate_tensors, output_a_gradient, output_bias_gradient
        ),
        repeats=args.repeats,
    )
    native_forward_ms = _elapsed_ms(
        lambda: _native(native_tensors, indices), repeats=args.repeats
    )
    retained_native_a, retained_native_bias = _native(native_tensors, indices)

    def retained_native_backward() -> None:
        torch.autograd.grad(
            (retained_native_a, retained_native_bias),
            (
                native_tensors.raw_alpha,
                native_tensors.preactivation_lower,
                native_tensors.preactivation_upper,
            ),
            grad_outputs=(output_a_gradient, output_bias_gradient),
            retain_graph=True,
        )

    native_backward_ms = _elapsed_ms(retained_native_backward, repeats=args.repeats)
    torch.cuda.synchronize()
    candidate_median = statistics.median(candidate_ms)
    native_median = statistics.median(native_ms)
    maximum = max(
        float(value)
        for row in metrics
        for value in row.values()
        if isinstance(value, float) and math.isfinite(value)
    )
    result: dict[str, object] = {
        "schema_version": "boundflow.root-crown-terminal-tir-probe/v1",
        "template_hash": template.stable_hash(),
        "unscheduled_tir_hash": executor.compiled.unscheduled_tir_hash,
        "scheduled_tir_hash": executor.compiled.scheduled_tir_hash,
        "device_source_hash": executor.compiled.device_source_hash,
        "compile_ns": compile_ns,
        "evaluation_count": len(metrics),
        "metrics": metrics,
        "maximum_absolute_difference": maximum,
        "all_sign_exact": all(bool(row["sign_exact"]) for row in metrics),
        "forward_launch_count": executor.forward_launch_count,
        "backward_launch_count": executor.backward_launch_count,
        "fallback_count": executor.fallback_count,
        "dlpack_pointer_count": executor.pointer_count,
        "dlpack_pointer_exact_count": executor.pointer_exact_count,
        "benchmark_repeats": args.repeats,
        "candidate_median_ms": candidate_median,
        "native_median_ms": native_median,
        "native_over_candidate_speedup": native_median / candidate_median,
        "candidate_forward_median_ms": statistics.median(candidate_forward_ms),
        "candidate_backward_median_ms": statistics.median(candidate_backward_ms),
        "native_forward_median_ms": statistics.median(native_forward_ms),
        "native_backward_median_ms": statistics.median(native_backward_ms),
        "performance_scope": "isolated-root-terminal-forward-backward",
        "performance_claimed": False,
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-input", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--thread-extent", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    """Run correctness and isolated timing for captured production tensors."""
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
