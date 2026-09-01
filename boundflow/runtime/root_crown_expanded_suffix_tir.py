"""Single custom-autograd owner for terminal, residual, and projection TIR."""

# mypy: disable-error-code=import-untyped
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=abstract-method,arguments-differ,too-many-locals
# pylint: disable=protected-access,duplicate-code,too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from boundflow.backends.tvm.root_crown_projection import (
    RootCrownProjectionTemplateV1,
)
from boundflow.backends.tvm.root_crown_residual import RootCrownResidualTemplateV1
from boundflow.backends.tvm.root_crown_terminal_linear import (
    RootCrownTerminalLinearTemplateV1,
)
from boundflow.runtime.root_crown_projection_tir import (
    RootCrownProjectionTensorsV1,
    RootCrownProjectionTIRExecutorV1,
    _validate_runtime_structure as _validate_projection_structure,
)
from boundflow.runtime.root_crown_suffix_tir import (
    RootCrownSuffixTensorsV1,
    RootCrownSuffixTIRExecutorV1,
)
from boundflow.runtime.root_crown_terminal_tir import RootCrownTerminalTensorsV1
from boundflow.runtime.root_crown_residual_tir import RootCrownResidualTensorsV1


@dataclass(frozen=True)
class RootCrownExpandedSuffixTensorsV1:
    """All dynamic inputs for one terminal-to-projection CROWN evaluation."""

    suffix: RootCrownSuffixTensorsV1
    projection: RootCrownProjectionTensorsV1


def validate_root_crown_expanded_templates_v1(
    terminal: RootCrownTerminalLinearTemplateV1,
    residual: RootCrownResidualTemplateV1,
    projection: RootCrownProjectionTemplateV1,
) -> None:
    """Validate both zero-copy coefficient boundaries before compilation."""

    terminal.validate()
    residual.validate()
    projection.validate()
    residual_features = residual.channels * residual.height * residual.width
    if (
        terminal.spec_count != residual.spec_count
        or terminal.domain_count != residual.domain_count
        or terminal.previous_features != residual_features
        or residual.coefficient_shape != projection.incoming_shape
        or terminal.compute_capability != residual.compute_capability
        or residual.compute_capability != projection.compute_capability
    ):
        raise ValueError("root CROWN expanded suffix template boundary differs")


class RootCrownExpandedSuffixTIRExecutorV1:
    """Own three prepared modules behind one cumulative autograd boundary."""

    def __init__(
        self,
        terminal_template: RootCrownTerminalLinearTemplateV1,
        residual_template: RootCrownResidualTemplateV1,
        projection_template: RootCrownProjectionTemplateV1,
    ) -> None:
        validate_root_crown_expanded_templates_v1(
            terminal_template, residual_template, projection_template
        )
        self.terminal_template = terminal_template
        self.residual_template = residual_template
        self.projection_template = projection_template
        self.suffix = RootCrownSuffixTIRExecutorV1(terminal_template, residual_template)
        self.projection = RootCrownProjectionTIRExecutorV1(projection_template)
        self.prepare_count = 0
        self.residual_stage_count = 0
        self.consume_count = 0
        self.fallback_count = 0
        self._staged_suffix: RootCrownSuffixTensorsV1 | None = None
        self._staged_a: torch.Tensor | None = None
        self._staged_bias: torch.Tensor | None = None
        self._combined_bias: torch.Tensor | None = None

    def prepare(self) -> None:
        """Warm all modules and allocate the final persistent bias arena."""

        if self.prepare_count:
            raise RuntimeError("root CROWN expanded executor already prepared")
        self.suffix.prepare()
        self.projection.prepare()
        self._combined_bias = torch.empty(
            (self.terminal_template.spec_count, self.terminal_template.domain_count),
            dtype=torch.float32,
            device="cuda",
        )
        self.prepare_count = 1

    def stage_terminal(
        self, tensors: RootCrownTerminalTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Delegate terminal staging without creating an autograd node."""

        if self.prepare_count != 1 or self._staged_suffix is not None:
            raise RuntimeError("root CROWN expanded terminal stage differs")
        return self.suffix.stage_terminal(tensors)

    def stage_residual(
        self, tensors: RootCrownSuffixTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute residual TIR while deferring the only autograd owner."""

        if self._staged_suffix is not None:
            raise RuntimeError("root CROWN expanded residual stage differs")
        output_a, output_bias = self.suffix.consume(tensors)
        self._staged_suffix = tensors
        self._staged_a = output_a
        self._staged_bias = output_bias
        self.residual_stage_count += 1
        return output_a, output_bias

    def consume(
        self, tensors: RootCrownExpandedSuffixTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Finish projection TIR and publish one cumulative result."""

        if (
            self._staged_suffix is None
            or self._staged_a is None
            or self._staged_bias is None
            or self._combined_bias is None
        ):
            raise RuntimeError("root CROWN expanded staged transaction differs")
        terminal_names = (
            "incoming_lower_a",
            "preactivation_lower",
            "preactivation_upper",
            "raw_alpha",
            "operator_weight",
            "operator_bias",
        )
        residual_names = (
            "incoming_lower_a",
            "entry_lower",
            "entry_upper",
            "entry_raw_alpha",
            "main_conv_weight",
            "main_conv_bias",
            "inner_lower",
            "inner_upper",
            "inner_raw_alpha",
            "inner_conv_weight",
            "inner_conv_bias",
        )
        if any(
            getattr(self._staged_suffix.terminal, name).data_ptr()
            != getattr(tensors.suffix.terminal, name).data_ptr()
            for name in terminal_names
        ) or any(
            getattr(self._staged_suffix.residual, name).data_ptr()
            != getattr(tensors.suffix.residual, name).data_ptr()
            for name in residual_names
        ):
            raise ValueError("root CROWN expanded suffix identity differs")
        _validate_projection_structure(tensors.projection, self.projection_template)
        if tensors.projection.incoming_lower_a.data_ptr() != self._staged_a.data_ptr():
            raise ValueError("root CROWN expanded residual/projection boundary differs")
        output_a, projection_bias = self.projection.forward(tensors.projection)
        torch.add(self._staged_bias, projection_bias, out=self._combined_bias)
        output_bias = self._combined_bias
        self._staged_suffix = None
        self._staged_a = None
        self._staged_bias = None
        self.consume_count += 1
        return output_a, output_bias

    def backward(
        self,
        tensors: RootCrownExpandedSuffixTensorsV1,
        output_a_gradient: torch.Tensor,
        output_bias_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Chain projection, residual, and terminal full VJPs in reverse order."""

        projection_gradients = self.projection.backward(
            tensors.projection, output_a_gradient, output_bias_gradient
        )
        suffix_gradients = self.suffix.backward(
            tensors.suffix, projection_gradients[0], output_bias_gradient
        )
        return (*suffix_gradients, *projection_gradients[1:])

    @property
    def last_terminal_a(self) -> torch.Tensor:
        return self.suffix.last_terminal_a

    @property
    def last_residual_main_a(self) -> torch.Tensor:
        return self.suffix.last_main_a

    @property
    def last_projection_outer_a(self) -> torch.Tensor:
        return self.projection.last_outer_a

    @property
    def staged_suffix(self) -> RootCrownSuffixTensorsV1:
        if self._staged_suffix is None:
            raise RuntimeError("root CROWN expanded suffix state is absent")
        return self._staged_suffix


class _RootCrownExpandedSuffixTIRFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        terminal_incoming: torch.Tensor,
        terminal_lower: torch.Tensor,
        terminal_upper: torch.Tensor,
        terminal_alpha: torch.Tensor,
        terminal_weight: torch.Tensor,
        terminal_bias: torch.Tensor,
        residual_incoming: torch.Tensor,
        residual_entry_lower: torch.Tensor,
        residual_entry_upper: torch.Tensor,
        residual_entry_alpha: torch.Tensor,
        residual_outer_weight: torch.Tensor,
        residual_outer_bias: torch.Tensor,
        residual_inner_lower: torch.Tensor,
        residual_inner_upper: torch.Tensor,
        residual_inner_alpha: torch.Tensor,
        residual_inner_weight: torch.Tensor,
        residual_inner_bias: torch.Tensor,
        projection_incoming: torch.Tensor,
        projection_entry_lower: torch.Tensor,
        projection_entry_upper: torch.Tensor,
        projection_entry_alpha: torch.Tensor,
        projection_outer_weight: torch.Tensor,
        projection_outer_bias: torch.Tensor,
        projection_inner_lower: torch.Tensor,
        projection_inner_upper: torch.Tensor,
        projection_inner_alpha: torch.Tensor,
        projection_inner_weight: torch.Tensor,
        projection_inner_bias: torch.Tensor,
        projection_skip_weight: torch.Tensor,
        projection_skip_bias: torch.Tensor,
        executor: RootCrownExpandedSuffixTIRExecutorV1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = RootCrownExpandedSuffixTensorsV1(
            suffix=RootCrownSuffixTensorsV1(
                terminal=RootCrownTerminalTensorsV1(
                    terminal_incoming,
                    terminal_lower,
                    terminal_upper,
                    terminal_alpha,
                    terminal_weight,
                    terminal_bias,
                ),
                residual=RootCrownResidualTensorsV1(
                    residual_incoming,
                    residual_entry_lower,
                    residual_entry_upper,
                    residual_entry_alpha,
                    residual_outer_weight,
                    residual_outer_bias,
                    residual_inner_lower,
                    residual_inner_upper,
                    residual_inner_alpha,
                    residual_inner_weight,
                    residual_inner_bias,
                ),
            ),
            projection=RootCrownProjectionTensorsV1(
                projection_incoming,
                projection_entry_lower,
                projection_entry_upper,
                projection_entry_alpha,
                projection_outer_weight,
                projection_outer_bias,
                projection_inner_lower,
                projection_inner_upper,
                projection_inner_alpha,
                projection_inner_weight,
                projection_inner_bias,
                projection_skip_weight,
                projection_skip_bias,
            ),
        )
        ctx.tensors = tensors
        ctx.executor = executor
        ctx.set_materialize_grads(False)
        return executor.consume(tensors)

    @staticmethod
    def backward(
        ctx: Any,
        output_a_gradient: torch.Tensor,
        output_bias_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("root CROWN expanded higher-order gradient unsupported")
        gradients = ctx.executor.backward(
            ctx.tensors,
            output_a_gradient.contiguous(),
            output_bias_gradient.contiguous(),
        )
        (
            terminal_alpha_gradient,
            terminal_lower_gradient,
            terminal_upper_gradient,
            residual_entry_lower_gradient,
            residual_entry_upper_gradient,
            residual_entry_alpha_gradient,
            residual_inner_lower_gradient,
            residual_inner_upper_gradient,
            residual_inner_alpha_gradient,
            projection_entry_lower_gradient,
            projection_entry_upper_gradient,
            projection_entry_alpha_gradient,
            projection_inner_lower_gradient,
            projection_inner_upper_gradient,
            projection_inner_alpha_gradient,
        ) = gradients
        return (
            None,
            terminal_lower_gradient,
            terminal_upper_gradient,
            terminal_alpha_gradient,
            None,
            None,
            None,
            residual_entry_lower_gradient,
            residual_entry_upper_gradient,
            residual_entry_alpha_gradient,
            None,
            None,
            residual_inner_lower_gradient,
            residual_inner_upper_gradient,
            residual_inner_alpha_gradient,
            None,
            None,
            None,
            projection_entry_lower_gradient,
            projection_entry_upper_gradient,
            projection_entry_alpha_gradient,
            None,
            None,
            projection_inner_lower_gradient,
            projection_inner_upper_gradient,
            projection_inner_alpha_gradient,
            None,
            None,
            None,
            None,
            None,
        )


def execute_root_crown_expanded_suffix_tir_v1(
    tensors: RootCrownExpandedSuffixTensorsV1,
    executor: RootCrownExpandedSuffixTIRExecutorV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attach one custom-autograd owner to all three prepared modules."""

    terminal = tensors.suffix.terminal
    residual = tensors.suffix.residual
    projection = tensors.projection
    return _RootCrownExpandedSuffixTIRFunction.apply(
        terminal.incoming_lower_a,
        terminal.preactivation_lower,
        terminal.preactivation_upper,
        terminal.raw_alpha,
        terminal.operator_weight,
        terminal.operator_bias,
        residual.incoming_lower_a,
        residual.entry_lower,
        residual.entry_upper,
        residual.entry_raw_alpha,
        residual.main_conv_weight,
        residual.main_conv_bias,
        residual.inner_lower,
        residual.inner_upper,
        residual.inner_raw_alpha,
        residual.inner_conv_weight,
        residual.inner_conv_bias,
        projection.incoming_lower_a,
        projection.entry_lower,
        projection.entry_upper,
        projection.entry_raw_alpha,
        projection.main_outer_conv_weight,
        projection.main_outer_conv_bias,
        projection.inner_lower,
        projection.inner_upper,
        projection.inner_raw_alpha,
        projection.main_inner_conv_weight,
        projection.main_inner_conv_bias,
        projection.skip_conv_weight,
        projection.skip_conv_bias,
        executor,
    )


__all__ = [
    "RootCrownExpandedSuffixTensorsV1",
    "RootCrownExpandedSuffixTIRExecutorV1",
    "execute_root_crown_expanded_suffix_tir_v1",
    "validate_root_crown_expanded_templates_v1",
]
