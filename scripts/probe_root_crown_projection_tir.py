#!/usr/bin/env python3
"""Check and time projection-residual TIR against the independent Torch oracle."""

# pylint: disable=import-error,wrong-import-position,too-many-locals
# pylint: disable=too-many-statements,import-outside-toplevel,not-callable,duplicate-code

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

from boundflow.backends.tvm.root_crown_projection import (  # noqa: E402
    RootCrownProjectionTemplateV1,
)
from boundflow.runtime.root_crown_projection_tir import (  # noqa: E402
    RootCrownProjectionTensorsV1,
    RootCrownProjectionTIRExecutorV1,
    execute_root_crown_projection_tir_v1,
)
from scripts.probe_root_crown_projection_oracle import (  # noqa: E402
    GRADIENT_FIELDS,
    _differentiable,
    _inputs,
    _oracle,
)
from scripts.probe_root_crown_residual_tir import (  # noqa: E402
    _coordinates,
    _difference,
    _sign_exact,
)


def _cuda(value: torch.Tensor, *, requires_grad: bool = False) -> torch.Tensor:
    result = value.detach().to("cuda").contiguous().clone()
    result.requires_grad_(requires_grad)
    return result


def _tensors(evaluation: dict[str, Any]) -> RootCrownProjectionTensorsV1:
    return RootCrownProjectionTensorsV1(
        incoming_lower_a=_cuda(evaluation["incoming_lower_a"], requires_grad=True),
        entry_lower=_cuda(evaluation["entry_lower"], requires_grad=True),
        entry_upper=_cuda(evaluation["entry_upper"], requires_grad=True),
        entry_raw_alpha=_cuda(evaluation["entry_raw_alpha"], requires_grad=True),
        main_outer_conv_weight=_cuda(evaluation["main_outer_conv_weight"]),
        main_outer_conv_bias=_cuda(evaluation["main_outer_conv_bias"]),
        inner_lower=_cuda(evaluation["inner_lower"], requires_grad=True),
        inner_upper=_cuda(evaluation["inner_upper"], requires_grad=True),
        inner_raw_alpha=_cuda(evaluation["inner_raw_alpha"], requires_grad=True),
        main_inner_conv_weight=_cuda(evaluation["main_inner_conv_weight"]),
        main_inner_conv_bias=_cuda(evaluation["main_inner_conv_bias"]),
        skip_conv_weight=_cuda(evaluation["skip_conv_weight"]),
        skip_conv_bias=_cuda(evaluation["skip_conv_bias"]),
    )


def _candidate_inputs(
    tensors: RootCrownProjectionTensorsV1,
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


def _gradients(
    outputs: tuple[torch.Tensor, torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    output_a_gradient: torch.Tensor,
    output_bias_gradient: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=(output_a_gradient, output_bias_gradient),
    )


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
        payload.get("schema_version") != "boundflow.root-crown-projection-tensors/v1"
        or not isinstance(evaluations, list)
        or len(evaluations) != 5
    ):
        raise ValueError("root CROWN projection tensor artifact differs")
    first = cast(dict[str, Any], evaluations[0])
    incoming = cast(torch.Tensor, first["incoming_lower_a"])
    output = cast(torch.Tensor, first["output_lower_a"])
    major, minor = torch.cuda.get_device_capability()
    template = RootCrownProjectionTemplateV1(
        spec_count=int(incoming.shape[0]),
        domain_count=int(incoming.shape[1]),
        output_channels=int(incoming.shape[2]),
        output_height=int(incoming.shape[3]),
        output_width=int(incoming.shape[4]),
        input_channels=int(output.shape[2]),
        input_height=int(output.shape[3]),
        input_width=int(output.shape[4]),
        entry_alpha_coordinates=_coordinates(first["entry_alpha_feature_indices"]),
        inner_alpha_coordinates=_coordinates(first["inner_alpha_feature_indices"]),
        compute_capability=f"sm_{major}{minor}",
        thread_extent=args.thread_extent,
    )
    compile_started = time.perf_counter_ns()
    executor = RootCrownProjectionTIRExecutorV1(template)
    compile_ns = time.perf_counter_ns() - compile_started
    metrics: list[dict[str, object]] = []
    for ordinal, raw in enumerate(evaluations):
        evaluation = cast(dict[str, Any], raw)
        candidate_tensors = _tensors(evaluation)
        oracle_values = _inputs(evaluation)
        candidate = execute_root_crown_projection_tir_v1(candidate_tensors, executor)
        oracle = _oracle(oracle_values)
        reference = (
            _cuda(evaluation["output_lower_a"]),
            _cuda(evaluation["output_bias"]),
        )
        has_backward = evaluation["output_lower_a_gradient"] is not None
        row: dict[str, object] = {
            "ordinal": ordinal,
            "candidate_reference_a_max_abs_diff": _difference(
                candidate[0], reference[0]
            ),
            "candidate_reference_bias_max_abs_diff": _difference(
                candidate[1], reference[1]
            ),
            "candidate_oracle_a_max_abs_diff": _difference(candidate[0], oracle[0]),
            "candidate_oracle_bias_max_abs_diff": _difference(candidate[1], oracle[1]),
            "forward_sign_exact": _sign_exact(candidate[0], reference[0])
            and _sign_exact(candidate[1], reference[1]),
            "has_backward": has_backward,
        }
        if has_backward:
            output_a_gradient = _cuda(evaluation["output_lower_a_gradient"])
            output_bias_gradient = _cuda(evaluation["output_bias_gradient"])
            candidate_gradients = _gradients(
                candidate,
                _candidate_inputs(candidate_tensors),
                output_a_gradient,
                output_bias_gradient,
            )
            oracle_gradients = _gradients(
                oracle,
                _differentiable(oracle_values),
                output_a_gradient,
                output_bias_gradient,
            )
            for name, candidate_gradient, oracle_gradient in zip(
                GRADIENT_FIELDS, candidate_gradients, oracle_gradients
            ):
                row[f"{name}_oracle_max_abs_diff"] = _difference(
                    candidate_gradient, oracle_gradient
                )
                row[f"{name}_oracle_sign_exact"] = _sign_exact(
                    candidate_gradient, oracle_gradient
                )
                row[f"{name}_captured_max_abs_diff"] = _difference(
                    candidate_gradient, _cuda(evaluation[name])
                )
        metrics.append(row)

    benchmark = cast(dict[str, Any], evaluations[0])
    candidate_tensors = _tensors(benchmark)
    oracle_values = _inputs(benchmark)
    output_a_gradient = _cuda(benchmark["output_lower_a_gradient"])
    output_bias_gradient = _cuda(benchmark["output_bias_gradient"])

    def candidate_iteration() -> None:
        outputs = execute_root_crown_projection_tir_v1(candidate_tensors, executor)
        _gradients(
            outputs,
            _candidate_inputs(candidate_tensors),
            output_a_gradient,
            output_bias_gradient,
        )

    def oracle_iteration() -> None:
        outputs = _oracle(oracle_values)
        _gradients(
            outputs,
            _differentiable(oracle_values),
            output_a_gradient,
            output_bias_gradient,
        )

    for _ in range(5):
        candidate_iteration()
        oracle_iteration()
    candidate_ms = _elapsed_ms(candidate_iteration, repeats=args.repeats)
    oracle_ms = _elapsed_ms(oracle_iteration, repeats=args.repeats)
    torch.cuda.synchronize()
    numerical = [
        float(value)
        for row in metrics
        for name, value in row.items()
        if isinstance(value, float) and math.isfinite(value) and "captured" not in name
    ]
    candidate_median = statistics.median(candidate_ms)
    oracle_median = statistics.median(oracle_ms)
    return {
        "schema_version": "boundflow.root-crown-projection-tir-probe/v1",
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
        "maximum_oracle_absolute_difference": max(numerical),
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
        "performance_scope": "isolated-root-projection-forward-full-vjp",
        "captured_bound_gradients_include_outside_region_uses": True,
        "performance_claimed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-input", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--thread-extent", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    """Run correctness and local timing, then persist the receipt."""

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
