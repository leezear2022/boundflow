"""Dual-lane compiled CROWN path for one intermediate linear start node."""

# mypy: disable-error-code=import-untyped
# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
# pylint: disable=too-many-positional-arguments,missing-function-docstring
# pylint: disable=abstract-method,arguments-differ

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from boundflow.backends.tvm.root_crown_input_domain import (
    RootCrownInputDomainTemplateV1,
)
from boundflow.backends.tvm.root_crown_projection import RootCrownProjectionTemplateV1
from boundflow.backends.tvm.root_crown_residual import RootCrownResidualTemplateV1
from boundflow.runtime.root_crown_input_domain_tir import (
    RootCrownInputDomainTensorsV1,
    RootCrownInputDomainTIRExecutorV1,
)
from boundflow.runtime.root_crown_projection_tir import (
    RootCrownProjectionTensorsV1,
    RootCrownProjectionTIRExecutorV1,
    execute_root_crown_projection_tir_v1,
)
from boundflow.runtime.root_crown_residual_tir import (
    RootCrownResidualTensorsV1,
    RootCrownResidualTIRExecutorV1,
    execute_root_crown_residual_tir_v1,
)


@dataclass(frozen=True)
class RootCrownIntermediateDualLaneTensorsV1:
    """Dynamic state for lower and upper propagation from a linear start."""

    lower_seed: torch.Tensor
    upper_seed: torch.Tensor
    lower_linear_bias: torch.Tensor
    upper_linear_bias: torch.Tensor
    residual_entry_lower: torch.Tensor
    residual_entry_upper: torch.Tensor
    residual_entry_alpha: torch.Tensor
    residual_main_weight: torch.Tensor
    residual_main_bias: torch.Tensor
    residual_inner_lower: torch.Tensor
    residual_inner_upper: torch.Tensor
    residual_inner_alpha: torch.Tensor
    residual_inner_weight: torch.Tensor
    residual_inner_bias: torch.Tensor
    projection_entry_lower: torch.Tensor
    projection_entry_upper: torch.Tensor
    projection_entry_alpha: torch.Tensor
    projection_outer_weight: torch.Tensor
    projection_outer_bias: torch.Tensor
    projection_inner_lower: torch.Tensor
    projection_inner_upper: torch.Tensor
    projection_inner_alpha: torch.Tensor
    projection_inner_weight: torch.Tensor
    projection_inner_bias: torch.Tensor
    projection_skip_weight: torch.Tensor
    projection_skip_bias: torch.Tensor
    input_lower: torch.Tensor
    input_upper: torch.Tensor
    input_alpha: torch.Tensor
    input_weight: torch.Tensor
    input_bias: torch.Tensor
    input_center: torch.Tensor
    input_radius: torch.Tensor


@dataclass(frozen=True)
class _LaneExecutorsV1:
    residual: RootCrownResidualTIRExecutorV1
    projection: RootCrownProjectionTIRExecutorV1
    input_domain: RootCrownInputDomainTIRExecutorV1

    def prepare(self) -> None:
        self.residual.prepare()
        self.projection.prepare()
        self.input_domain.prepare()


class _InputDomainTIRFunction(torch.autograd.Function):
    """Attach the existing input-domain full VJP to a standalone lane."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        incoming: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        alpha: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        center: torch.Tensor,
        radius: torch.Tensor,
        executor: RootCrownInputDomainTIRExecutorV1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = RootCrownInputDomainTensorsV1(
            incoming, lower, upper, alpha, weight, bias, center, radius
        )
        ctx.tensors = tensors
        ctx.executor = executor
        ctx.set_materialize_grads(False)
        return executor.forward(tensors)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        concrete_gradient: torch.Tensor,
        bias_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("root intermediate higher-order gradient unsupported")
        incoming_gradient, alpha_gradient = ctx.executor.backward(
            ctx.tensors,
            concrete_gradient.contiguous(),
            bias_gradient.contiguous(),
        )
        return (
            incoming_gradient,
            None,
            None,
            alpha_gradient,
            None,
            None,
            None,
            None,
            None,
        )


def _execute_input_domain(
    tensors: RootCrownInputDomainTensorsV1,
    executor: RootCrownInputDomainTIRExecutorV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _InputDomainTIRFunction.apply(
        tensors.incoming_lower_a,
        tensors.preactivation_lower,
        tensors.preactivation_upper,
        tensors.raw_alpha,
        tensors.operator_weight,
        tensors.operator_bias,
        tensors.input_center,
        tensors.input_radius,
        executor,
    )


class RootCrownIntermediateDualLaneTIRExecutorV1:
    """Execute lower and upper CROWN through the same compiled lower primitive.

    The upper lane uses the exact dual identity ``U(A) = -L(-A)`` and swaps
    the two alpha planes before entering the lower primitive.  The composition
    intentionally remains three custom-autograd nodes per lane in this first
    correctness stage; a later owner folds them into one rematerializing VJP.
    """

    def __init__(
        self,
        residual_template: RootCrownResidualTemplateV1,
        projection_template: RootCrownProjectionTemplateV1,
        input_template: RootCrownInputDomainTemplateV1,
    ) -> None:
        if (
            residual_template.coefficient_shape != projection_template.incoming_shape
            or projection_template.output_shape != input_template.incoming_shape
            or len(
                {
                    residual_template.spec_count,
                    projection_template.spec_count,
                    input_template.spec_count,
                }
            )
            != 1
            or len(
                {
                    residual_template.domain_count,
                    projection_template.domain_count,
                    input_template.domain_count,
                }
            )
            != 1
        ):
            raise ValueError("root intermediate dual-lane template boundary differs")
        self.residual_template = residual_template
        self.projection_template = projection_template
        self.input_template = input_template
        self.lower = self._new_lane()
        self.upper = self._new_lane()
        self.prepare_count = 0
        self.call_count = 0
        self.performance_claimed = False

    def _new_lane(self) -> _LaneExecutorsV1:
        return _LaneExecutorsV1(
            RootCrownResidualTIRExecutorV1(self.residual_template),
            RootCrownProjectionTIRExecutorV1(self.projection_template),
            RootCrownInputDomainTIRExecutorV1(self.input_template),
        )

    def prepare(self) -> None:
        if self.prepare_count:
            raise RuntimeError("root intermediate dual-lane executor already prepared")
        self.lower.prepare()
        self.upper.prepare()
        self.prepare_count = 1

    @staticmethod
    def _swap_alpha_lane(alpha: torch.Tensor) -> torch.Tensor:
        """Move the native upper plane to lane zero without detaching its VJP."""

        return alpha.flip(0).contiguous()

    def _execute_lane(
        self,
        tensors: RootCrownIntermediateDualLaneTensorsV1,
        lane: _LaneExecutorsV1,
        *,
        upper: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seed = -tensors.upper_seed if upper else tensors.lower_seed
        residual_entry_alpha = (
            self._swap_alpha_lane(tensors.residual_entry_alpha)
            if upper
            else tensors.residual_entry_alpha
        )
        residual_inner_alpha = (
            self._swap_alpha_lane(tensors.residual_inner_alpha)
            if upper
            else tensors.residual_inner_alpha
        )
        projection_entry_alpha = (
            self._swap_alpha_lane(tensors.projection_entry_alpha)
            if upper
            else tensors.projection_entry_alpha
        )
        projection_inner_alpha = (
            self._swap_alpha_lane(tensors.projection_inner_alpha)
            if upper
            else tensors.projection_inner_alpha
        )
        input_alpha = (
            self._swap_alpha_lane(tensors.input_alpha) if upper else tensors.input_alpha
        )
        residual = RootCrownResidualTensorsV1(
            seed,
            tensors.residual_entry_lower,
            tensors.residual_entry_upper,
            residual_entry_alpha,
            tensors.residual_main_weight,
            tensors.residual_main_bias,
            tensors.residual_inner_lower,
            tensors.residual_inner_upper,
            residual_inner_alpha,
            tensors.residual_inner_weight,
            tensors.residual_inner_bias,
        )
        residual_a, residual_bias = execute_root_crown_residual_tir_v1(
            residual, lane.residual
        )
        projection = RootCrownProjectionTensorsV1(
            residual_a,
            tensors.projection_entry_lower,
            tensors.projection_entry_upper,
            projection_entry_alpha,
            tensors.projection_outer_weight,
            tensors.projection_outer_bias,
            tensors.projection_inner_lower,
            tensors.projection_inner_upper,
            projection_inner_alpha,
            tensors.projection_inner_weight,
            tensors.projection_inner_bias,
            tensors.projection_skip_weight,
            tensors.projection_skip_bias,
        )
        projection_a, projection_bias = execute_root_crown_projection_tir_v1(
            projection, lane.projection
        )
        input_domain = RootCrownInputDomainTensorsV1(
            projection_a,
            tensors.input_lower,
            tensors.input_upper,
            input_alpha,
            tensors.input_weight,
            tensors.input_bias,
            tensors.input_center,
            tensors.input_radius,
        )
        concrete, input_bias = _execute_input_domain(input_domain, lane.input_domain)
        return concrete, residual_bias + projection_bias + input_bias

    def execute(
        self, tensors: RootCrownIntermediateDualLaneTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prepare_count != 1:
            raise RuntimeError("root intermediate dual-lane executor is not prepared")
        lower_concrete, lower_bias = self._execute_lane(
            tensors, self.lower, upper=False
        )
        upper_negative_concrete, upper_negative_bias = self._execute_lane(
            tensors, self.upper, upper=True
        )
        lower = (
            lower_concrete
            + lower_bias.transpose(0, 1)
            + tensors.lower_linear_bias.transpose(0, 1)
        )
        upper = (
            -upper_negative_concrete
            - upper_negative_bias.transpose(0, 1)
            + tensors.upper_linear_bias.transpose(0, 1)
        )
        self.call_count += 1
        return lower, upper

    def receipt(self) -> dict[str, object]:
        return {
            "schema_version": "boundflow.root-intermediate-dual-lane-tir/v1",
            "call_count": self.call_count,
            "lower_forward_launch_count": sum(
                executor.forward_launch_count
                for executor in (
                    self.lower.residual,
                    self.lower.projection,
                    self.lower.input_domain,
                )
            ),
            "upper_forward_launch_count": sum(
                executor.forward_launch_count
                for executor in (
                    self.upper.residual,
                    self.upper.projection,
                    self.upper.input_domain,
                )
            ),
            "alpha_lane_policy": "lower=0;upper=flip-to-0;upper=-lower(-A)",
            "dense_a_crossing_count": 4 * self.call_count,
            "single_rematerializing_owner": False,
            "performance_claimed": False,
        }


__all__ = [
    "RootCrownIntermediateDualLaneTensorsV1",
    "RootCrownIntermediateDualLaneTIRExecutorV1",
]
