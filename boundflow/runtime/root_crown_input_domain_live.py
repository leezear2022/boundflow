"""Live auto_LiRPA bridge for fused root input Conv/L-infinity concretization."""

# pylint: disable=protected-access,too-many-locals,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,too-many-statements,duplicate-code

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from types import MethodType
from typing import Any, Iterator, Mapping, Sequence, cast

import torch

from boundflow.backends.tvm.root_crown_input_domain import (
    RootCrownInputDomainTemplateV1,
)
from boundflow.runtime.root_crown_full_pipeline_tir import (
    RootCrownFullPipelineTIRExecutorV1,
    execute_root_crown_full_pipeline_tir_v1,
)
from boundflow.runtime.root_crown_input_capture import (
    ROOT_INPUT_CONV,
    ROOT_INPUT_NODE,
    ROOT_INPUT_RELU,
)
from boundflow.runtime.root_crown_terminal_capture import ROOT_START_NODE


def _coordinates(
    values: Sequence[torch.Tensor],
) -> tuple[tuple[int, int, int], ...]:
    if len(values) != 3 or any(value.ndim != 1 for value in values):
        raise ValueError("root CROWN input-domain alpha coordinates differ")
    lengths = {int(value.numel()) for value in values}
    if len(lengths) != 1:
        raise ValueError("root CROWN input-domain alpha coordinate length differs")
    return cast(
        tuple[tuple[int, int, int], ...],
        tuple(
            tuple(int(values[axis][ordinal]) for axis in range(3))
            for ordinal in range(int(values[0].numel()))
        ),
    )


def _is_tensor_shape(value: Any, shape: tuple[int, ...]) -> bool:
    return torch.is_tensor(value) and tuple(cast(torch.Tensor, value).shape) == shape


@dataclass(frozen=True)
class DeferredRootInputCoefficientV1:
    """Typed root marker whose concrete value is already produced by TIR."""

    concrete_lower: torch.Tensor
    batch_size: int
    output_dim: int
    transaction_ordinal: int


class RootCrownInputDomainLiveBridgeV1:
    """Replace C0 and concretization while preserving host solver control."""

    def __init__(
        self,
        template: RootCrownInputDomainTemplateV1,
        executor: RootCrownFullPipelineTIRExecutorV1,
    ) -> None:
        template.validate()
        if executor.input_template != template:
            raise ValueError("root CROWN input-domain live template differs")
        self.template = template
        self.executor = executor
        self.outer_call_count = 0
        self.relu_replacement_count = 0
        self.conv_replacement_count = 0
        self.concretize_replacement_count = 0
        self.deferred_dense_a_count = 0
        self.fallback_count = 0
        self._active = False
        self._pending: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        self._last_pending: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        self._zero_bias: torch.Tensor | None = None
        self._input_center: torch.Tensor | None = None
        self._input_radius: torch.Tensor | None = None
        self._upper_d: torch.Tensor | None = None
        self._zero_coefficients: bool | None = None
        self._static_admitted = False

    @staticmethod
    def _eligible(instance: Any, kwargs: Mapping[str, Any]) -> bool:
        opts = getattr(instance, "bound_opts", {}).get("optimize_bound_args", {})
        names = {str(getattr(node, "name", "")) for node in instance.nodes()}
        return (
            kwargs.get("bound_side", "lower") == "lower"
            and int(opts.get("iteration", -1)) == 5
            and not bool(getattr(instance, "return_A", False))
            and {ROOT_INPUT_RELU, ROOT_INPUT_CONV, ROOT_INPUT_NODE} <= names
        )

    @contextmanager
    def install(self, bounded_module_type: type[Any]) -> Iterator[None]:
        """Install the typed deferred-root transaction for one optimizer call."""

        original_optimized = bounded_module_type._get_optimized_bounds
        original_concretize = bounded_module_type.concretize_root

        def concretize_wrapped(
            instance: Any,
            root: Any,
            batch_size: int,
            output_dim: int,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            marker = getattr(root, "lA", None)
            if not isinstance(marker, DeferredRootInputCoefficientV1):
                return original_concretize(
                    instance, root, batch_size, output_dim, *args, **kwargs
                )
            mismatches = {
                "root": str(getattr(root, "name", "")) != ROOT_INPUT_NODE,
                "batch": marker.batch_size != batch_size,
                "output": marker.output_dim != output_dim,
                "ordinal": marker.transaction_ordinal
                != self.concretize_replacement_count,
                "shape": tuple(marker.concrete_lower.shape) != (batch_size, output_dim),
                "average": bool(kwargs.get("average_A", False)),
                "save": bool(kwargs.get("save_bounds", False)),
                "upper": getattr(root, "uA", None) is not None,
            }
            failures = sorted(name for name, failed in mismatches.items() if failed)
            if failures:
                self.fallback_count += 1
                raise ValueError(
                    "root CROWN deferred concretization differs: " + ",".join(failures)
                )
            self.concretize_replacement_count += 1
            return marker.concrete_lower, None, False

        def optimized_wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                self._active
                or self.outer_call_count
                or not self._eligible(instance, kwargs)
            ):
                return original_optimized(instance, *args, **kwargs)
            self.outer_call_count += 1
            self._active = True
            try:
                with self._install_nodes(instance):
                    return original_optimized(instance, *args, **kwargs)
            finally:
                self._active = False

        bounded_module_type.concretize_root = concretize_wrapped
        bounded_module_type._get_optimized_bounds = optimized_wrapped
        try:
            yield
        finally:
            bounded_module_type._get_optimized_bounds = original_optimized
            bounded_module_type.concretize_root = original_concretize
            if self._pending is not None:
                raise ValueError(
                    "root CROWN input-domain live transaction remained partial"
                )

    def _set_relu_state(
        self,
        node: Any,
        lower: torch.Tensor,
        upper: torch.Tensor,
        incoming: torch.Tensor,
    ) -> None:
        if self._upper_d is None or self._zero_coefficients is None:
            with torch.no_grad():
                self._upper_d = node._relu_upper_bound(lower, upper, 0.0)[0].unsqueeze(
                    0
                )
                self._zero_coefficients = bool((upper <= 0).all().item())
        node.d = self._upper_d
        node.lA = incoming
        node.init_d = None
        node.zero_backward_coeffs_l = self._zero_coefficients
        node.zero_backward_coeffs_u = self._zero_coefficients
        node.masked_beta_lower = node.masked_beta_upper = None

    def _admit_static(self, relu: Any, input_node: Any) -> None:
        if self._static_admitted:
            return
        indices = getattr(relu, "alpha_indices", None)
        perturbation = getattr(input_node, "perturbation", None)
        center = getattr(input_node, "center", None)
        x_lower = getattr(perturbation, "x_L", None)
        x_upper = getattr(perturbation, "x_U", None)
        if (
            not isinstance(indices, (tuple, list))
            or _coordinates(indices) != self.template.alpha_coordinates
            or bool(getattr(relu, "cut_used", False))
            or float(getattr(relu, "leaky_alpha", 0.0)) != 0.0
            or perturbation is None
            or float(getattr(perturbation, "norm", 0.0)) != math.inf
            or getattr(perturbation, "constraints", None) is not None
            or not _is_tensor_shape(center, self.template.input_shape)
            or not _is_tensor_shape(x_lower, self.template.input_shape)
            or not _is_tensor_shape(x_upper, self.template.input_shape)
        ):
            raise ValueError("root CROWN input-domain static admission differs")
        center = cast(torch.Tensor, center)
        x_lower = cast(torch.Tensor, x_lower)
        x_upper = cast(torch.Tensor, x_upper)
        midpoint = ((x_lower + x_upper) * 0.5).contiguous()
        if not torch.equal(center, midpoint):
            raise ValueError("root CROWN input-domain center differs")
        self._input_center = center.contiguous()
        self._input_radius = ((x_upper - x_lower) * 0.5).contiguous()
        self._static_admitted = True

    @contextmanager
    def _install_nodes(self, instance: Any) -> Iterator[None]:
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        relu = nodes[ROOT_INPUT_RELU]
        conv = nodes[ROOT_INPUT_CONV]
        input_node = nodes[ROOT_INPUT_NODE]
        if (
            not relu.inputs
            or relu.inputs[0] is not conv
            or not conv.inputs
            or conv.inputs[0] is not input_node
        ):
            raise ValueError("root CROWN input-domain live topology differs")
        original_relu = relu.bound_backward
        original_conv = conv.bound_backward

        def relu_replacement(_self: Any, *args: Any, **kwargs: Any) -> Any:
            if str(getattr(kwargs.get("start_node"), "name", "")) != ROOT_START_NODE:
                return original_relu(*args, **kwargs)
            if (
                self._pending is not None
                or len(args) < 3
                or not torch.is_tensor(args[0])
                or args[1] is not None
                or kwargs.get("unstable_idx") is not None
            ):
                raise ValueError("root CROWN input-domain ReLU call differs")
            self._admit_static(relu, input_node)
            incoming = args[0]
            preactivation = args[2]
            raw_alpha = getattr(relu, "alpha", {}).get(ROOT_START_NODE)
            if (
                not torch.is_tensor(raw_alpha)
                or incoming.data_ptr() != self.executor.staged_projection_a.data_ptr()
            ):
                raise ValueError("root CROWN input-domain incoming boundary differs")
            self._pending = (
                incoming,
                preactivation.lower,
                preactivation.upper,
                raw_alpha,
            )
            self._set_relu_state(
                relu, preactivation.lower, preactivation.upper, incoming
            )
            if self._zero_bias is None:
                self._zero_bias = torch.zeros(
                    (self.template.spec_count, self.template.domain_count),
                    dtype=incoming.dtype,
                    device=incoming.device,
                )
            self.relu_replacement_count += 1
            return [(incoming, None)], self._zero_bias, 0

        def conv_replacement(_self: Any, *args: Any, **kwargs: Any) -> Any:
            if self._pending is None:
                return original_conv(*args, **kwargs)
            if (
                str(getattr(kwargs.get("start_node"), "name", "")) != ROOT_START_NODE
                or len(args) < 5
                or args[1] is not None
                or not torch.is_tensor(args[0])
                or args[0].data_ptr() != self._pending[0].data_ptr()
                or self._input_center is None
                or self._input_radius is None
            ):
                raise ValueError("root CROWN input-domain Conv call differs")
            _incoming, lower, upper, raw_alpha = self._pending
            expanded = self.executor.staged_expanded
            concrete, output_bias = execute_root_crown_full_pipeline_tir_v1(
                expanded,
                lower,
                upper,
                raw_alpha,
                args[3].lower,
                args[4].lower,
                self._input_center,
                self._input_radius,
                self.executor,
            )
            marker = DeferredRootInputCoefficientV1(
                concrete_lower=concrete,
                batch_size=self.template.domain_count,
                output_dim=self.template.spec_count,
                transaction_ordinal=self.conv_replacement_count,
            )
            self._last_pending = self._pending
            self._pending = None
            self.conv_replacement_count += 1
            self.deferred_dense_a_count += 1
            return [(marker, None), (None, None), (None, None)], output_bias, 0

        relu.bound_backward = MethodType(relu_replacement, relu)
        conv.bound_backward = MethodType(conv_replacement, conv)
        try:
            yield
            if self._last_pending is not None:
                incoming, lower, upper, _alpha = self._last_pending
                self._set_relu_state(relu, lower, upper, incoming)
        finally:
            relu.bound_backward = original_relu
            conv.bound_backward = original_conv

    def validate(self) -> None:
        """Require exact activation and a single cumulative autograd owner."""

        if (
            self.outer_call_count != 1
            or self.executor.prepare_count != 1
            or self.relu_replacement_count != 5
            or self.conv_replacement_count != 5
            or self.concretize_replacement_count != 5
            or self.deferred_dense_a_count != 5
            or self.executor.projection_stage_count != 5
            or self.executor.consume_count != 5
            or self.executor.input_domain.forward_launch_count != 5
            or self.executor.input_domain.backward_launch_count != 4
            or self.fallback_count != 0
            or self.executor.fallback_count != 0
            or self._pending is not None
        ):
            raise ValueError("root CROWN input-domain live activation count differs")

    def receipt(self) -> dict[str, object]:
        """Return compiler identities and actual production activation counters."""

        self.validate()
        compiled = self.executor.input_domain.compiled
        return {
            "schema_version": "boundflow.root-crown-input-domain-live/v1",
            "template_hash": self.template.stable_hash(),
            "unscheduled_tir_hash": compiled.unscheduled_tir_hash,
            "scheduled_tir_hash": compiled.scheduled_tir_hash,
            "device_source_hash": compiled.device_source_hash,
            "workspace_inventory": [
                [name, list(shape)] for name, shape in compiled.workspace_inventory
            ],
            "outer_call_count": self.outer_call_count,
            "relu_replacement_count": self.relu_replacement_count,
            "conv_replacement_count": self.conv_replacement_count,
            "concretize_replacement_count": self.concretize_replacement_count,
            "deferred_dense_a_count": self.deferred_dense_a_count,
            "dense_input_a_external_materialization_count": 0,
            "forward_launch_count": self.executor.input_domain.forward_launch_count,
            "backward_launch_count": self.executor.input_domain.backward_launch_count,
            "cumulative_autograd_owner_count": 1,
            "custom_autograd_invocation_count": self.executor.consume_count,
            "fallback_count": 0,
            "performance_claimed": False,
        }


__all__ = [
    "DeferredRootInputCoefficientV1",
    "RootCrownInputDomainLiveBridgeV1",
]
