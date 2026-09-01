"""One custom-autograd owner from terminal CROWN through input concretization."""

# mypy: disable-error-code=import-untyped
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=abstract-method,arguments-differ,too-many-locals
# pylint: disable=protected-access,duplicate-code,too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from boundflow.backends.tvm.root_crown_input_domain import (
    RootCrownInputDomainTemplateV1,
)
from boundflow.backends.tvm.root_crown_projection import RootCrownProjectionTemplateV1
from boundflow.backends.tvm.root_crown_residual import RootCrownResidualTemplateV1
from boundflow.backends.tvm.root_crown_terminal_linear import (
    RootCrownTerminalLinearTemplateV1,
)
from boundflow.runtime.root_crown_expanded_suffix_tir import (
    RootCrownExpandedSuffixTensorsV1,
    RootCrownExpandedSuffixTIRExecutorV1,
)
from boundflow.runtime.root_crown_input_domain_tir import (
    RootCrownInputDomainTensorsV1,
    RootCrownInputDomainTIRExecutorV1,
    _validate_runtime_structure as _validate_input_structure,
)
from boundflow.runtime.root_crown_projection_tir import RootCrownProjectionTensorsV1
from boundflow.runtime.root_crown_residual_tir import RootCrownResidualTensorsV1
from boundflow.runtime.root_crown_suffix_tir import RootCrownSuffixTensorsV1
from boundflow.runtime.root_crown_terminal_tir import RootCrownTerminalTensorsV1


@dataclass(frozen=True)
class RootCrownFullPipelineTensorsV1:
    """Dynamic inputs for terminal-to-input-domain execution."""

    expanded: RootCrownExpandedSuffixTensorsV1
    input_domain: RootCrownInputDomainTensorsV1


class RootCrownFullPipelineTIRExecutorV1:
    """Own four prepared modules behind one cumulative autograd boundary."""

    def __init__(
        self,
        terminal_template: RootCrownTerminalLinearTemplateV1,
        residual_template: RootCrownResidualTemplateV1,
        projection_template: RootCrownProjectionTemplateV1,
        input_template: RootCrownInputDomainTemplateV1,
    ) -> None:
        self.expanded = RootCrownExpandedSuffixTIRExecutorV1(
            terminal_template, residual_template, projection_template
        )
        if projection_template.output_shape != input_template.incoming_shape:
            raise ValueError("root CROWN full projection/input boundary differs")
        self.input_domain = RootCrownInputDomainTIRExecutorV1(input_template)
        self.terminal_template = terminal_template
        self.residual_template = residual_template
        self.projection_template = projection_template
        self.input_template = input_template
        self.prepare_count = 0
        self.projection_stage_count = 0
        self.consume_count = 0
        self.fallback_count = 0
        self.exact_warmup_reset_count = 0
        self._staged_expanded: RootCrownExpandedSuffixTensorsV1 | None = None
        self._staged_a: torch.Tensor | None = None
        self._staged_bias: torch.Tensor | None = None
        self._final_bias: torch.Tensor | None = None

    @property
    def suffix(self):
        return self.expanded.suffix

    @property
    def projection(self):
        return self.expanded.projection

    @property
    def staged_suffix(self) -> RootCrownSuffixTensorsV1:
        return self.expanded.staged_suffix

    @property
    def staged_expanded(self) -> RootCrownExpandedSuffixTensorsV1:
        if self._staged_expanded is None:
            raise RuntimeError("root CROWN full expanded state is absent")
        return self._staged_expanded

    @property
    def last_terminal_a(self) -> torch.Tensor:
        return self.expanded.last_terminal_a

    @property
    def last_residual_main_a(self) -> torch.Tensor:
        return self.expanded.last_residual_main_a

    @property
    def last_projection_outer_a(self) -> torch.Tensor:
        return self.expanded.last_projection_outer_a

    @property
    def residual_stage_count(self) -> int:
        """Expose the upstream stage counter for cumulative receipts."""

        return self.expanded.residual_stage_count

    @property
    def staged_projection_a(self) -> torch.Tensor:
        """Return the prepared projection/input boundary without exposing internals."""

        if self._staged_a is None:
            raise RuntimeError("root CROWN full projection state is absent")
        return self._staged_a

    def prepare(self) -> None:
        if self.prepare_count:
            raise RuntimeError("root CROWN full executor already prepared")
        self.expanded.prepare()
        self.input_domain.prepare()
        self._final_bias = torch.empty(
            (self.terminal_template.spec_count, self.terminal_template.domain_count),
            dtype=torch.float32,
            device="cuda",
        )
        self.prepare_count = 1

    def reset_after_exact_warmup_v1(self) -> None:
        """Reuse prepared modules while starting a fresh measured transaction."""

        suffix = self.expanded.suffix
        modules = (
            suffix.terminal,
            suffix.residual,
            self.expanded.projection,
            self.input_domain,
        )
        if (
            self.prepare_count != 1
            or self.exact_warmup_reset_count != 0
            or self.projection_stage_count != 5
            or self.consume_count != 5
            or self.expanded.residual_stage_count != 5
            or self.expanded.consume_count != 5
            or suffix.stage_count != 5
            or suffix.consume_count != 5
            or any(
                module.forward_launch_count != 5
                or module.backward_launch_count != 4
                or module.fallback_count != 0
                for module in modules
            )
            or any(
                value is not None
                for value in (
                    self._staged_expanded,
                    self._staged_a,
                    self._staged_bias,
                    self.expanded._staged_suffix,
                    self.expanded._staged_a,
                    self.expanded._staged_bias,
                    suffix._staged_tensors,
                    suffix._staged_a,
                    suffix._staged_bias,
                )
            )
        ):
            raise ValueError("root CROWN exact warmup reset precondition differs")
        self.projection_stage_count = 0
        self.consume_count = 0
        self.fallback_count = 0
        self.expanded.residual_stage_count = 0
        self.expanded.consume_count = 0
        self.expanded.fallback_count = 0
        suffix.stage_count = 0
        suffix.consume_count = 0
        suffix.fallback_count = 0
        for module in modules:
            module.forward_launch_count = 0
            module.backward_launch_count = 0
            module.fallback_count = 0
            module.pointer_count = 0
            module.pointer_exact_count = 0
        self.exact_warmup_reset_count = 1

    def stage_terminal(
        self, tensors: RootCrownTerminalTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expanded.stage_terminal(tensors)

    def stage_residual(
        self, tensors: RootCrownSuffixTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expanded.stage_residual(tensors)

    def stage_projection(
        self, tensors: RootCrownExpandedSuffixTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._staged_expanded is not None:
            raise RuntimeError("root CROWN full projection stage differs")
        output_a, output_bias = self.expanded.consume(tensors)
        self._staged_expanded = tensors
        self._staged_a = output_a
        self._staged_bias = output_bias
        self.projection_stage_count += 1
        return output_a, output_bias

    def consume(
        self, tensors: RootCrownFullPipelineTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._staged_expanded is None
            or self._staged_a is None
            or self._staged_bias is None
            or self._final_bias is None
        ):
            raise RuntimeError("root CROWN full staged transaction differs")
        if tensors.expanded is not self._staged_expanded:
            raise ValueError("root CROWN full expanded identity differs")
        _validate_input_structure(tensors.input_domain, self.input_template)
        if (
            tensors.input_domain.incoming_lower_a.data_ptr()
            != self._staged_a.data_ptr()
        ):
            raise ValueError("root CROWN full input coefficient boundary differs")
        concrete, local_bias = self.input_domain.forward(tensors.input_domain)
        torch.add(self._staged_bias, local_bias, out=self._final_bias)
        output_bias = self._final_bias
        self._staged_expanded = None
        self._staged_a = None
        self._staged_bias = None
        self.consume_count += 1
        return concrete, output_bias

    def backward(
        self,
        tensors: RootCrownFullPipelineTensorsV1,
        concrete_gradient: torch.Tensor,
        bias_gradient: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        projection_gradient, input_alpha_gradient = self.input_domain.backward(
            tensors.input_domain, concrete_gradient, bias_gradient
        )
        expanded_gradients = self.expanded.backward(
            tensors.expanded, projection_gradient, bias_gradient
        )
        return expanded_gradients, input_alpha_gradient


class _RootCrownFullPipelineFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments
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
        input_lower: torch.Tensor,
        input_upper: torch.Tensor,
        input_alpha: torch.Tensor,
        input_weight: torch.Tensor,
        input_bias: torch.Tensor,
        input_center: torch.Tensor,
        input_radius: torch.Tensor,
        executor: RootCrownFullPipelineTIRExecutorV1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        supplied = RootCrownExpandedSuffixTensorsV1(
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
        expanded = executor.staged_expanded
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
        projection_names = (
            "incoming_lower_a",
            "entry_lower",
            "entry_upper",
            "entry_raw_alpha",
            "main_outer_conv_weight",
            "main_outer_conv_bias",
            "inner_lower",
            "inner_upper",
            "inner_raw_alpha",
            "main_inner_conv_weight",
            "main_inner_conv_bias",
            "skip_conv_weight",
            "skip_conv_bias",
        )
        if (
            any(
                getattr(expanded.suffix.terminal, name).data_ptr()
                != getattr(supplied.suffix.terminal, name).data_ptr()
                for name in terminal_names
            )
            or any(
                getattr(expanded.suffix.residual, name).data_ptr()
                != getattr(supplied.suffix.residual, name).data_ptr()
                for name in residual_names
            )
            or any(
                getattr(expanded.projection, name).data_ptr()
                != getattr(supplied.projection, name).data_ptr()
                for name in projection_names
            )
        ):
            raise ValueError("root CROWN full supplied transaction differs")
        tensors = RootCrownFullPipelineTensorsV1(
            expanded=expanded,
            input_domain=RootCrownInputDomainTensorsV1(
                executor.staged_projection_a,
                input_lower,
                input_upper,
                input_alpha,
                input_weight,
                input_bias,
                input_center,
                input_radius,
            ),
        )
        ctx.tensors = tensors
        ctx.executor = executor
        ctx.set_materialize_grads(False)
        return executor.consume(tensors)

    @staticmethod
    def backward(
        ctx: Any,
        concrete_gradient: torch.Tensor,
        bias_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("root CROWN full higher-order gradient unsupported")
        expanded, input_alpha = ctx.executor.backward(
            ctx.tensors,
            concrete_gradient.contiguous(),
            bias_gradient.contiguous(),
        )
        (
            terminal_alpha,
            terminal_lower,
            terminal_upper,
            residual_entry_lower,
            residual_entry_upper,
            residual_entry_alpha,
            residual_inner_lower,
            residual_inner_upper,
            residual_inner_alpha,
            projection_entry_lower,
            projection_entry_upper,
            projection_entry_alpha,
            projection_inner_lower,
            projection_inner_upper,
            projection_inner_alpha,
        ) = expanded
        return (
            None,
            terminal_lower,
            terminal_upper,
            terminal_alpha,
            None,
            None,
            None,
            residual_entry_lower,
            residual_entry_upper,
            residual_entry_alpha,
            None,
            None,
            residual_inner_lower,
            residual_inner_upper,
            residual_inner_alpha,
            None,
            None,
            None,
            projection_entry_lower,
            projection_entry_upper,
            projection_entry_alpha,
            None,
            None,
            projection_inner_lower,
            projection_inner_upper,
            projection_inner_alpha,
            None,
            None,
            None,
            None,
            None,
            None,
            input_alpha,
            None,
            None,
            None,
            None,
            None,
        )


def execute_root_crown_full_pipeline_tir_v1(
    expanded: RootCrownExpandedSuffixTensorsV1,
    input_lower: torch.Tensor,
    input_upper: torch.Tensor,
    input_alpha: torch.Tensor,
    input_weight: torch.Tensor,
    input_bias: torch.Tensor,
    input_center: torch.Tensor,
    input_radius: torch.Tensor,
    executor: RootCrownFullPipelineTIRExecutorV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attach the only custom-autograd owner after input concretization."""

    terminal = expanded.suffix.terminal
    residual = expanded.suffix.residual
    projection = expanded.projection
    return _RootCrownFullPipelineFunction.apply(
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
        input_lower,
        input_upper,
        input_alpha,
        input_weight,
        input_bias,
        input_center,
        input_radius,
        executor,
    )


__all__ = [
    "RootCrownFullPipelineTIRExecutorV1",
    "RootCrownFullPipelineTensorsV1",
    "execute_root_crown_full_pipeline_tir_v1",
]
