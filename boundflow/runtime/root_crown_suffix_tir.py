"""Cumulative custom-autograd owner for terminal and residual CROWN TIR."""

# mypy: disable-error-code=import-untyped
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=abstract-method,arguments-differ,too-many-locals
# pylint: disable=protected-access

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from boundflow.backends.tvm.root_crown_residual import (
    RootCrownResidualTemplateV1,
)
from boundflow.backends.tvm.root_crown_terminal_linear import (
    RootCrownTerminalLinearTemplateV1,
)
from boundflow.runtime.root_crown_residual_tir import (
    RootCrownResidualTensorsV1,
    RootCrownResidualTIRExecutorV1,
    _validate_runtime_structure as _validate_residual_structure,
)
from boundflow.runtime.root_crown_terminal_tir import (
    RootCrownTerminalTensorsV1,
    RootCrownTerminalTIRExecutorV1,
    _validate_runtime_structure as _validate_terminal_structure,
)


@dataclass(frozen=True)
class RootCrownSuffixTensorsV1:
    """All dynamic tensors owned by one terminal-plus-residual evaluation."""

    terminal: RootCrownTerminalTensorsV1
    residual: RootCrownResidualTensorsV1


def validate_root_crown_suffix_templates_v1(
    terminal_template: RootCrownTerminalLinearTemplateV1,
    residual_template: RootCrownResidualTemplateV1,
) -> None:
    """Validate the zero-copy boundary before either module is compiled."""

    terminal_template.validate()
    residual_template.validate()
    flattened = (
        residual_template.channels * residual_template.height * residual_template.width
    )
    if (
        terminal_template.spec_count != residual_template.spec_count
        or terminal_template.domain_count != residual_template.domain_count
        or terminal_template.previous_features != flattened
        or terminal_template.compute_capability != residual_template.compute_capability
    ):
        raise ValueError("root CROWN suffix template boundary differs")


class RootCrownSuffixTIRExecutorV1:
    """Own two compiled modules behind one cumulative autograd boundary."""

    def __init__(
        self,
        terminal_template: RootCrownTerminalLinearTemplateV1,
        residual_template: RootCrownResidualTemplateV1,
    ) -> None:
        validate_root_crown_suffix_templates_v1(terminal_template, residual_template)
        self.terminal_template = terminal_template
        self.residual_template = residual_template
        self.terminal = RootCrownTerminalTIRExecutorV1(terminal_template)
        self.residual = RootCrownResidualTIRExecutorV1(residual_template)
        self.prepare_count = 0
        self.stage_count = 0
        self.consume_count = 0
        self.fallback_count = 0
        self._staged_tensors: RootCrownTerminalTensorsV1 | None = None
        self._staged_a: torch.Tensor | None = None
        self._staged_bias: torch.Tensor | None = None
        self._combined_bias: torch.Tensor | None = None

    def prepare(self) -> None:
        """Materialize both modules and every persistent output/VJP arena."""

        if self.prepare_count:
            raise RuntimeError("root CROWN suffix executor already prepared")
        self._prepare_terminal()
        self.residual.prepare()
        self._combined_bias = torch.empty(
            (
                self.terminal_template.spec_count,
                self.terminal_template.domain_count,
            ),
            dtype=torch.float32,
            device="cuda",
        )
        self.prepare_count = 1

    def _prepare_terminal(self) -> None:
        """Warm the frozen terminal executor without changing its public ABI."""

        template = self.terminal_template
        device = torch.device("cuda")
        tensors = RootCrownTerminalTensorsV1(
            incoming_lower_a=torch.zeros(
                (template.spec_count, template.domain_count, template.current_features),
                dtype=torch.float32,
                device=device,
            ),
            preactivation_lower=torch.full(
                (template.domain_count, template.current_features),
                -1.0,
                dtype=torch.float32,
                device=device,
            ),
            preactivation_upper=torch.full(
                (template.domain_count, template.current_features),
                1.0,
                dtype=torch.float32,
                device=device,
            ),
            raw_alpha=torch.full(
                (
                    2,
                    template.spec_count,
                    template.domain_count,
                    template.alpha_feature_count,
                ),
                0.5,
                dtype=torch.float32,
                device=device,
            ),
            operator_weight=torch.zeros(
                (template.current_features, template.previous_features),
                dtype=torch.float32,
                device=device,
            ),
            operator_bias=torch.zeros(
                (template.current_features,), dtype=torch.float32, device=device
            ),
        )
        output_a, output_bias = self.terminal.forward(tensors)
        self.terminal.backward(
            tensors,
            torch.zeros_like(output_a),
            torch.zeros_like(output_bias),
        )
        torch.cuda.synchronize(device)
        persistent = {
            value.data_ptr()
            for value in (
                self.terminal._feature_indices,
                self.terminal._feature_to_ordinal,
                self.terminal._output_a,
                self.terminal._output_bias,
                self.terminal._alpha_gradient,
                self.terminal._bound_gradient,
            )
            if value is not None
        }
        self.terminal._view_cache = {
            key: view
            for key, view in self.terminal._view_cache.items()
            if key[0] in persistent
        }
        self.terminal.forward_launch_count = 0
        self.terminal.backward_launch_count = 0
        self.terminal.pointer_count = 0
        self.terminal.pointer_exact_count = 0

    def stage_terminal(
        self, tensors: RootCrownTerminalTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the first module without creating an intermediate autograd node."""

        if self.prepare_count != 1 or self._staged_tensors is not None:
            raise RuntimeError("root CROWN suffix terminal stage differs")
        _validate_terminal_structure(tensors, self.terminal_template)
        output_a, output_bias = self.terminal.forward(tensors)
        self._staged_tensors = tensors
        self._staged_a = output_a
        self._staged_bias = output_bias
        self.stage_count += 1
        return output_a, output_bias

    def consume(
        self, tensors: RootCrownSuffixTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume the staged terminal value and run the residual module."""

        staged_tensors = self._staged_tensors
        if (
            staged_tensors is None
            or self._staged_a is None
            or self._staged_bias is None
            or self._combined_bias is None
        ):
            raise RuntimeError("root CROWN suffix staged transaction differs")
        terminal_names = (
            "incoming_lower_a",
            "preactivation_lower",
            "preactivation_upper",
            "raw_alpha",
            "operator_weight",
            "operator_bias",
        )
        if any(
            getattr(staged_tensors, name).data_ptr()
            != getattr(tensors.terminal, name).data_ptr()
            for name in terminal_names
        ):
            raise ValueError("root CROWN suffix terminal identity differs")
        _validate_residual_structure(tensors.residual, self.residual_template)
        staged = self._staged_a
        incoming = tensors.residual.incoming_lower_a
        if (
            incoming.data_ptr() != staged.data_ptr()
            or incoming.numel() != staged.numel()
            or not incoming.is_contiguous()
        ):
            raise ValueError("root CROWN suffix terminal/residual boundary differs")
        output_a, residual_bias = self.residual.forward(tensors.residual)
        torch.add(self._staged_bias, residual_bias, out=self._combined_bias)
        combined_bias = self._combined_bias
        self._staged_tensors = None
        self._staged_a = None
        self._staged_bias = None
        self.consume_count += 1
        return output_a, combined_bias

    def backward(
        self,
        tensors: RootCrownSuffixTensorsV1,
        output_a_gradient: torch.Tensor,
        output_bias_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Chain residual and terminal full VJPs without exposing dense A."""

        residual_gradients = self.residual.backward(
            tensors.residual,
            output_a_gradient,
            output_bias_gradient,
        )
        terminal_output_gradient = residual_gradients[0].view(
            self.terminal_template.spec_count,
            self.terminal_template.domain_count,
            self.terminal_template.previous_features,
        )
        terminal_gradients = self.terminal.backward(
            tensors.terminal,
            terminal_output_gradient,
            output_bias_gradient,
        )
        return (*terminal_gradients, *residual_gradients[1:])

    @property
    def last_terminal_a(self) -> torch.Tensor:
        """Return the prepared terminal arena used at the residual boundary."""

        output = self.terminal._output_a  # pylint: disable=protected-access
        if output is None:
            raise RuntimeError("root CROWN suffix terminal state is absent")
        return output

    @property
    def last_main_a(self) -> torch.Tensor:
        """Return the residual main-branch state required by the host."""

        return self.residual.last_main_a


class _RootCrownSuffixTIRFunction(torch.autograd.Function):
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
        entry_lower: torch.Tensor,
        entry_upper: torch.Tensor,
        entry_alpha: torch.Tensor,
        main_weight: torch.Tensor,
        main_bias: torch.Tensor,
        inner_lower: torch.Tensor,
        inner_upper: torch.Tensor,
        inner_alpha: torch.Tensor,
        inner_weight: torch.Tensor,
        inner_bias: torch.Tensor,
        executor: RootCrownSuffixTIRExecutorV1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = RootCrownSuffixTensorsV1(
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
                entry_lower,
                entry_upper,
                entry_alpha,
                main_weight,
                main_bias,
                inner_lower,
                inner_upper,
                inner_alpha,
                inner_weight,
                inner_bias,
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
            raise RuntimeError("root CROWN suffix higher-order gradient unsupported")
        gradients = ctx.executor.backward(
            ctx.tensors,
            output_a_gradient.contiguous(),
            output_bias_gradient.contiguous(),
        )
        (
            terminal_alpha_gradient,
            terminal_lower_gradient,
            terminal_upper_gradient,
            entry_lower_gradient,
            entry_upper_gradient,
            entry_alpha_gradient,
            inner_lower_gradient,
            inner_upper_gradient,
            inner_alpha_gradient,
        ) = gradients
        return (
            None,
            terminal_lower_gradient,
            terminal_upper_gradient,
            terminal_alpha_gradient,
            None,
            None,
            None,
            entry_lower_gradient,
            entry_upper_gradient,
            entry_alpha_gradient,
            None,
            None,
            inner_lower_gradient,
            inner_upper_gradient,
            inner_alpha_gradient,
            None,
            None,
            None,
        )


def execute_root_crown_suffix_tir_v1(
    tensors: RootCrownSuffixTensorsV1,
    executor: RootCrownSuffixTIRExecutorV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the cumulative suffix through one custom-autograd owner."""

    terminal = tensors.terminal
    residual = tensors.residual
    return _RootCrownSuffixTIRFunction.apply(
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
        executor,
    )


__all__ = [
    "RootCrownSuffixTensorsV1",
    "RootCrownSuffixTIRExecutorV1",
    "execute_root_crown_suffix_tir_v1",
    "validate_root_crown_suffix_templates_v1",
]
