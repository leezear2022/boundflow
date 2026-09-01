"""Ten-evaluation/nine-mutation optimizer over the prepared S4 evaluator."""

# pylint: disable=protected-access,too-many-locals,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from boundflow.runtime.asplos27_s4_all_state_evaluator import (
    PreparedS4AllStateEvaluatorV1,
)


@dataclass(frozen=True)
class S4OptimizerRunResultV1:
    """Terminal state and scalar execution counters for one prepared run."""

    terminal_lower: torch.Tensor
    terminal_parameters: tuple[torch.Tensor, ...]
    terminal_las: tuple[torch.Tensor, ...]
    learning_rates: tuple[tuple[float, float], ...]
    evaluation_count: int
    optimizer_mutation_count: int
    scheduler_call_count: int
    value_graph_submission_count: int
    compact_coefficient_launch_count: int
    fallback_count: int
    performance_claimed: bool = False

    def validate(self) -> None:
        """Validate the fixed 10/9 execution inventory and terminal payload."""

        if (
            tuple(self.terminal_lower.shape) != (6,)
            or len(self.terminal_parameters) != 7
            or len(self.terminal_las) != 6
            or len(self.learning_rates) != 10
            or self.evaluation_count != 10
            or self.optimizer_mutation_count != 9
            or self.scheduler_call_count != 10
            or self.value_graph_submission_count != 10
            or self.compact_coefficient_launch_count != 180
            or self.fallback_count
            or self.performance_claimed
            or not all(
                bool(torch.isfinite(value).all().item())
                for value in (
                    self.terminal_lower,
                    *self.terminal_parameters,
                    *self.terminal_las,
                )
            )
        ):
            raise ValueError("S4 optimizer run result differs")


def execute_s4_optimizer_v1(
    evaluator: PreparedS4AllStateEvaluatorV1,
) -> S4OptimizerRunResultV1:
    """Run the frozen 10/9 policy without dense alpha/beta materialization."""

    resources = evaluator.buffers._resources
    if resources is None or len(resources._parameters) != 7:
        raise ValueError("S4 optimizer parameter owner differs")
    alpha = list(resources._parameters[:6])
    beta = resources._parameters[6]
    optimizer = torch.optim.Adam(
        (
            {"params": alpha, "lr": 0.01},
            {"params": [beta], "lr": 0.05},
        ),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    learning_rates: list[tuple[float, float]] = []
    terminal_lower: torch.Tensor | None = None
    terminal_las: tuple[torch.Tensor, ...] | None = None
    compact_launches = 0
    submissions = 0
    scheduler_calls = 0
    with torch.cuda.stream(evaluator.stream):
        for ordinal in range(10):
            expected_alpha_lr = 0.01 * 0.98**ordinal
            expected_beta_lr = 0.05 * 0.98**ordinal
            alpha_lr = float(optimizer.param_groups[0]["lr"])
            beta_lr = float(optimizer.param_groups[1]["lr"])
            if not math.isclose(
                alpha_lr, expected_alpha_lr, rel_tol=0.0, abs_tol=1e-15
            ) or not math.isclose(
                beta_lr, expected_beta_lr, rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError("S4 optimizer learning-rate sequence differs")
            learning_rates.append((alpha_lr, beta_lr))
            row = evaluator.evaluate(ordinal, terminal=ordinal == 9)
            compact_launches += row.compact_coefficient_launch_count
            submissions += row.value_graph_submission_count
            if ordinal < 9:
                optimizer.zero_grad(set_to_none=True)
                for parameter, gradient in zip(resources._parameters, row.gradients):
                    parameter.grad = gradient
                optimizer.step()
                with torch.no_grad():
                    for parameter in alpha:
                        parameter.clamp_(0.0, 1.0)
                    beta.clamp_(min=0.0)
                scheduler.step()
                scheduler_calls += 1
            else:
                terminal_lower = row.lower.detach().clone()
                if row.terminal_lease is None:
                    raise ValueError("S4 optimizer terminal lA lease is absent")
                terminal_las = tuple(
                    value.detach().clone()
                    for value in row.terminal_lease.consume(
                        evaluation_generation=ordinal + 1
                    )
                )
                scheduler.step()
                scheduler_calls += 1
    evaluator.stream.synchronize()
    if terminal_lower is None or terminal_las is None:
        raise ValueError("S4 optimizer terminal result is absent")
    result = S4OptimizerRunResultV1(
        terminal_lower=terminal_lower,
        terminal_parameters=tuple(
            value.detach().clone() for value in resources._parameters
        ),
        terminal_las=terminal_las,
        learning_rates=tuple(learning_rates),
        evaluation_count=10,
        optimizer_mutation_count=9,
        scheduler_call_count=scheduler_calls,
        value_graph_submission_count=submissions,
        compact_coefficient_launch_count=compact_launches,
        fallback_count=0,
    )
    result.validate()
    return result


__all__ = ["S4OptimizerRunResultV1", "execute_s4_optimizer_v1"]
