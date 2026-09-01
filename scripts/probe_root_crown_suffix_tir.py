#!/usr/bin/env python3
"""Check and time the cumulative terminal/residual CROWN owner."""

# pylint: disable=import-error,wrong-import-position,import-outside-toplevel
# pylint: disable=too-many-locals,too-many-statements,duplicate-code
# pylint: disable=too-many-boolean-expressions
# pylint: disable=protected-access
# mypy: disable-error-code=import-untyped

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

import torch  # noqa: E402

from boundflow.backends.tvm.root_crown_residual import (  # noqa: E402
    RootCrownResidualTemplateV1,
)
from boundflow.backends.tvm.root_crown_terminal_linear import (  # noqa: E402
    RootCrownTerminalLinearTemplateV1,
)
from boundflow.runtime.root_crown_residual_tir import (  # noqa: E402
    RootCrownResidualTensorsV1,
)
from boundflow.runtime.root_crown_suffix_tir import (  # noqa: E402
    RootCrownSuffixTensorsV1,
    RootCrownSuffixTIRExecutorV1,
    execute_root_crown_suffix_tir_v1,
)
from boundflow.runtime.root_crown_terminal_tir import (  # noqa: E402
    RootCrownTerminalTensorsV1,
)
from scripts.probe_root_crown_residual_tir import (  # noqa: E402
    _coordinates,
    _cuda,
    _oracle as residual_oracle,
)
from scripts.probe_root_crown_terminal_tir import (  # noqa: E402
    _native as terminal_oracle,
    _tensors as terminal_tensors,
)


def _residual_tensors(
    evaluation: dict[str, Any], incoming: torch.Tensor
) -> RootCrownResidualTensorsV1:
    return RootCrownResidualTensorsV1(
        incoming_lower_a=incoming,
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


def _gradient_inputs(
    terminal: RootCrownTerminalTensorsV1,
    residual: RootCrownResidualTensorsV1,
) -> tuple[torch.Tensor, ...]:
    return (
        terminal.raw_alpha,
        terminal.preactivation_lower,
        terminal.preactivation_upper,
        residual.entry_lower,
        residual.entry_upper,
        residual.entry_raw_alpha,
        residual.inner_lower,
        residual.inner_upper,
        residual.inner_raw_alpha,
    )


def _replace_residual_incoming(
    tensors: RootCrownResidualTensorsV1, incoming: torch.Tensor
) -> RootCrownResidualTensorsV1:
    return RootCrownResidualTensorsV1(
        incoming,
        tensors.entry_lower,
        tensors.entry_upper,
        tensors.entry_raw_alpha,
        tensors.main_conv_weight,
        tensors.main_conv_bias,
        tensors.inner_lower,
        tensors.inner_upper,
        tensors.inner_raw_alpha,
        tensors.inner_conv_weight,
        tensors.inner_conv_bias,
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


def _templates(
    terminal_evaluation: dict[str, Any],
    residual_evaluation: dict[str, Any],
    thread_extent: int,
) -> tuple[RootCrownTerminalLinearTemplateV1, RootCrownResidualTemplateV1]:
    residual_incoming = cast(torch.Tensor, residual_evaluation["incoming_lower_a"])
    entry_coordinates = _coordinates(residual_evaluation["entry_alpha_feature_indices"])
    inner_coordinates = _coordinates(residual_evaluation["inner_alpha_feature_indices"])
    major, minor = torch.cuda.get_device_capability()
    compute_capability = f"sm_{major}{minor}"
    residual = RootCrownResidualTemplateV1(
        spec_count=int(residual_incoming.shape[0]),
        domain_count=int(residual_incoming.shape[1]),
        channels=int(residual_incoming.shape[2]),
        height=int(residual_incoming.shape[3]),
        width=int(residual_incoming.shape[4]),
        entry_alpha_coordinates=entry_coordinates,
        inner_alpha_coordinates=inner_coordinates,
        compute_capability=compute_capability,
        thread_extent=thread_extent,
    )
    terminal_incoming = cast(torch.Tensor, terminal_evaluation["incoming_lower_a"])
    terminal_weight = cast(torch.Tensor, terminal_evaluation["operator_weight"])
    terminal_indices = terminal_evaluation["alpha_feature_indices"][0]
    terminal = RootCrownTerminalLinearTemplateV1(
        spec_count=int(terminal_incoming.shape[0]),
        domain_count=int(terminal_incoming.shape[1]),
        current_features=int(terminal_incoming.shape[2]),
        previous_features=int(terminal_weight.shape[1]),
        alpha_feature_indices=tuple(int(value) for value in terminal_indices.tolist()),
        compute_capability=compute_capability,
        thread_extent=thread_extent,
    )
    return terminal, residual


def _candidate(
    terminal_evaluation: dict[str, Any],
    residual_evaluation: dict[str, Any],
    executor: RootCrownSuffixTIRExecutorV1,
    output_gradients: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...] | None]:
    terminal = terminal_tensors(terminal_evaluation)
    staged_a, _staged_bias = executor.stage_terminal(terminal)
    residual = _residual_tensors(
        residual_evaluation,
        staged_a.view(executor.residual_template.coefficient_shape),
    )
    outputs = execute_root_crown_suffix_tir_v1(
        RootCrownSuffixTensorsV1(terminal, residual), executor
    )
    gradients = None
    if output_gradients is not None:
        gradients = torch.autograd.grad(
            outputs,
            _gradient_inputs(terminal, residual),
            grad_outputs=output_gradients,
        )
    return outputs, gradients


def _oracle(
    terminal_evaluation: dict[str, Any],
    residual_evaluation: dict[str, Any],
    terminal_template: RootCrownTerminalLinearTemplateV1,
    residual_template: RootCrownResidualTemplateV1,
    output_gradients: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...] | None]:
    terminal = terminal_tensors(terminal_evaluation)
    indices = torch.tensor(
        terminal_template.alpha_feature_indices, dtype=torch.int64, device="cuda"
    )
    terminal_a, terminal_bias = terminal_oracle(terminal, indices)
    residual = _residual_tensors(
        residual_evaluation,
        terminal_a.view(residual_template.coefficient_shape),
    )
    residual_a, residual_bias = residual_oracle(residual, residual_template)
    outputs = (residual_a, terminal_bias + residual_bias)
    gradients = None
    if output_gradients is not None:
        gradients = torch.autograd.grad(
            outputs,
            _gradient_inputs(terminal, residual),
            grad_outputs=output_gradients,
        )
    return outputs, gradients


def _run(args: argparse.Namespace) -> dict[str, object]:
    terminal_payload = torch.load(
        args.terminal_input, map_location="cpu", weights_only=True
    )
    residual_payload = torch.load(
        args.residual_input, map_location="cpu", weights_only=True
    )
    terminal_evaluations = terminal_payload.get("evaluations")
    residual_evaluations = residual_payload.get("evaluations")
    if (
        terminal_payload.get("schema_version")
        != "boundflow.root-crown-terminal-tensors/v1"
        or residual_payload.get("schema_version")
        != "boundflow.root-crown-residual-tensors/v1"
        or not isinstance(terminal_evaluations, list)
        or not isinstance(residual_evaluations, list)
        or len(terminal_evaluations) != 5
        or len(residual_evaluations) != 5
    ):
        raise ValueError("root CROWN suffix tensor artifacts differ")
    terminal_template, residual_template = _templates(
        terminal_evaluations[0], residual_evaluations[0], args.thread_extent
    )
    started = time.perf_counter_ns()
    executor = RootCrownSuffixTIRExecutorV1(terminal_template, residual_template)
    executor.prepare()
    prepare_ns = time.perf_counter_ns() - started
    rows: list[dict[str, object]] = []
    maximum_difference = 0.0
    all_sign_exact = True
    for ordinal, (terminal_evaluation, residual_evaluation) in enumerate(
        zip(terminal_evaluations, residual_evaluations)
    ):
        boundary_reference = _cuda(residual_evaluation["incoming_lower_a"])
        terminal_reference = _cuda(terminal_evaluation["output_lower_a"]).view_as(
            boundary_reference
        )
        boundary_difference = _difference(terminal_reference, boundary_reference)
        has_backward = residual_evaluation["output_lower_a_gradient"] is not None
        output_gradients = (
            (
                _cuda(residual_evaluation["output_lower_a_gradient"]),
                _cuda(residual_evaluation["output_bias_gradient"]),
            )
            if has_backward
            else None
        )
        candidate_outputs, candidate_gradients = _candidate(
            terminal_evaluation, residual_evaluation, executor, output_gradients
        )
        oracle_outputs, oracle_gradients = _oracle(
            terminal_evaluation,
            residual_evaluation,
            terminal_template,
            residual_template,
            output_gradients,
        )
        output_differences = [
            _difference(left, right)
            for left, right in zip(candidate_outputs, oracle_outputs)
        ]
        gradient_differences = (
            [
                _difference(left, right)
                for left, right in zip(
                    cast(tuple[torch.Tensor, ...], candidate_gradients),
                    cast(tuple[torch.Tensor, ...], oracle_gradients),
                )
            ]
            if has_backward
            else []
        )
        sign_exact = all(
            _sign_exact(left, right)
            for left, right in zip(candidate_outputs, oracle_outputs)
        ) and (
            not has_backward
            or all(
                _sign_exact(left, right)
                for left, right in zip(
                    cast(tuple[torch.Tensor, ...], candidate_gradients),
                    cast(tuple[torch.Tensor, ...], oracle_gradients),
                )
            )
        )
        row_maximum = max(
            [boundary_difference, *output_differences, *gradient_differences]
        )
        maximum_difference = max(maximum_difference, row_maximum)
        all_sign_exact = all_sign_exact and sign_exact
        rows.append(
            {
                "ordinal": ordinal,
                "boundary_max_abs_diff": boundary_difference,
                "output_a_max_abs_diff": output_differences[0],
                "output_bias_max_abs_diff": output_differences[1],
                "gradient_max_abs_diffs": gradient_differences,
                "sign_exact": sign_exact,
            }
        )
    benchmark_terminal = terminal_evaluations[0]
    benchmark_residual = residual_evaluations[0]
    benchmark_gradients = (
        _cuda(benchmark_residual["output_lower_a_gradient"]),
        _cuda(benchmark_residual["output_bias_gradient"]),
    )
    candidate_terminal = terminal_tensors(benchmark_terminal)
    staged_a, _staged_bias = executor.stage_terminal(candidate_terminal)
    candidate_residual = _residual_tensors(
        benchmark_residual,
        staged_a.view(residual_template.coefficient_shape),
    )

    def candidate_forward() -> tuple[torch.Tensor, torch.Tensor]:
        executor.stage_terminal(candidate_terminal)
        return execute_root_crown_suffix_tir_v1(
            RootCrownSuffixTensorsV1(candidate_terminal, candidate_residual),
            executor,
        )

    # Consume the setup stage before entering the warmup/timing loop.
    setup_outputs = execute_root_crown_suffix_tir_v1(
        RootCrownSuffixTensorsV1(candidate_terminal, candidate_residual), executor
    )
    torch.autograd.grad(
        setup_outputs,
        _gradient_inputs(candidate_terminal, candidate_residual),
        grad_outputs=benchmark_gradients,
    )
    oracle_terminal = terminal_tensors(benchmark_terminal)
    oracle_indices = torch.tensor(
        terminal_template.alpha_feature_indices, dtype=torch.int64, device="cuda"
    )
    oracle_residual_static = _residual_tensors(
        benchmark_residual,
        _cuda(benchmark_residual["incoming_lower_a"]),
    )

    def candidate_iteration() -> None:
        outputs = candidate_forward()
        torch.autograd.grad(
            outputs,
            _gradient_inputs(candidate_terminal, candidate_residual),
            grad_outputs=benchmark_gradients,
        )

    def oracle_iteration() -> None:
        terminal_a, terminal_bias = terminal_oracle(oracle_terminal, oracle_indices)
        residual = _replace_residual_incoming(
            oracle_residual_static,
            terminal_a.view(residual_template.coefficient_shape),
        )
        residual_a, residual_bias = residual_oracle(residual, residual_template)
        torch.autograd.grad(
            (residual_a, terminal_bias + residual_bias),
            _gradient_inputs(oracle_terminal, residual),
            grad_outputs=benchmark_gradients,
        )

    candidate_iteration()
    oracle_iteration()
    candidate_times = _elapsed_ms(candidate_iteration, repeats=args.repeats)
    oracle_times = _elapsed_ms(oracle_iteration, repeats=args.repeats)
    candidate_median = statistics.median(candidate_times)
    oracle_median = statistics.median(oracle_times)
    result = {
        "schema_version": "boundflow.root-crown-suffix-tir-probe/v1",
        "evaluation_count": len(rows),
        "metrics": rows,
        "maximum_oracle_absolute_difference": maximum_difference,
        "all_oracle_sign_exact": all_sign_exact,
        "benchmark_repeats": args.repeats,
        "candidate_median_ms": candidate_median,
        "native_oracle_median_ms": oracle_median,
        "native_over_candidate_speedup": oracle_median / candidate_median,
        "prepare_ns": prepare_ns,
        "terminal_template_hash": terminal_template.stable_hash(),
        "residual_template_hash": residual_template.stable_hash(),
        "fallback_count": executor.fallback_count,
        "performance_scope": "isolated-cumulative-terminal-residual-full-vjp",
        "performance_claimed": False,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-input", type=Path, required=True)
    parser.add_argument("--residual-input", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--thread-extent", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    """Run the local cumulative correctness/timing probe."""

    _run(_parse_args())


if __name__ == "__main__":
    main()
