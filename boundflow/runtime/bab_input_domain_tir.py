"""Current-stream runtime for activation-BaB input streaming TIR."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-locals
# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=abstract-method,arguments-differ,protected-access
# pylint: disable=too-many-arguments,too-many-positional-arguments,duplicate-code
# pylint: disable=super-init-not-called

from __future__ import annotations

from typing import Any

import torch

from boundflow.backends.tvm.bab_input_domain import (
    BabInputDomainTemplateV1,
    compile_bab_input_domain_tir_v1,
)
from boundflow.runtime.root_crown_input_domain_tir import (
    _validate_runtime_structure,
    RootCrownInputDomainTensorsV1,
    RootCrownInputDomainTIRExecutorV1,
)
from boundflow.runtime.root_crown_projection_tir import _ordinal_map

BabInputDomainTensorsV1 = RootCrownInputDomainTensorsV1


def validate_bab_input_domain_tensors_v1(
    tensors: BabInputDomainTensorsV1,
    template: BabInputDomainTemplateV1,
) -> None:
    """Fail closed before launch, including interval and alpha constraints."""

    template.validate()
    _validate_runtime_structure(tensors, template)
    for name in (
        "incoming_lower_a",
        "preactivation_lower",
        "preactivation_upper",
        "raw_alpha",
        "operator_weight",
        "operator_bias",
        "input_center",
        "input_radius",
    ):
        if not bool(torch.isfinite(getattr(tensors, name)).all().item()):
            raise ValueError(f"activation-BaB input-domain nonfinite tensor: {name}")
    if (
        bool((tensors.preactivation_lower > tensors.preactivation_upper).any().item())
        or bool(((tensors.raw_alpha < 0) | (tensors.raw_alpha > 1)).any().item())
        or bool((tensors.input_radius < 0).any().item())
    ):
        raise ValueError("activation-BaB input-domain legality differs")


class BabInputDomainTIRExecutorV1(RootCrownInputDomainTIRExecutorV1):
    """Prepared BaB specialization reusing the proven current-stream runtime."""

    template: BabInputDomainTemplateV1

    def __init__(self, template: BabInputDomainTemplateV1) -> None:
        template.validate()
        self.template = template
        self.compiled = compile_bab_input_domain_tir_v1(template)
        geometry = (
            template.output_channels,
            template.output_height,
            template.output_width,
        )
        self._alpha_map = _ordinal_map(template.alpha_coordinates, geometry)
        self._view_cache: dict[tuple[int, tuple[int, ...], str, str], Any] = {}
        self._concrete_lower: torch.Tensor | None = None
        self._output_bias: torch.Tensor | None = None
        self._incoming_gradient: torch.Tensor | None = None
        self._alpha_gradient: torch.Tensor | None = None
        self.forward_launch_count = 0
        self.backward_launch_count = 0
        self.fallback_count = 0
        self.pointer_count = 0
        self.pointer_exact_count = 0
        self.prepare_count = 0


class _BabInputDomainTIRFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        incoming: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        alpha: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        center: torch.Tensor,
        radius: torch.Tensor,
        executor: BabInputDomainTIRExecutorV1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = BabInputDomainTensorsV1(
            incoming,
            lower,
            upper,
            alpha,
            weight,
            bias,
            center,
            radius,
        )
        _validate_runtime_structure(tensors, executor.template)
        ctx.tensors = tensors
        ctx.executor = executor
        ctx.set_materialize_grads(False)
        return executor.forward(tensors)

    @staticmethod
    def backward(
        ctx: Any,
        concrete_gradient: torch.Tensor,
        bias_gradient: torch.Tensor,
    ) -> tuple[Any, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "activation-BaB input-domain higher-order gradient unsupported"
            )
        incoming, alpha = ctx.executor.backward(
            ctx.tensors,
            concrete_gradient.contiguous(),
            bias_gradient.contiguous(),
        )
        return incoming, None, None, alpha, None, None, None, None, None


def execute_bab_input_domain_tir_v1(
    tensors: BabInputDomainTensorsV1,
    executor: BabInputDomainTIRExecutorV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the streamed input transaction behind custom autograd."""

    return _BabInputDomainTIRFunction.apply(
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


__all__ = [
    "BabInputDomainTensorsV1",
    "BabInputDomainTIRExecutorV1",
    "execute_bab_input_domain_tir_v1",
    "validate_bab_input_domain_tensors_v1",
]
