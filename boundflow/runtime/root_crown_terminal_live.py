"""Live auto_LiRPA bridge for the compiled root terminal CROWN transaction."""

# pylint: disable=protected-access,too-many-locals,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,too-many-statements,duplicate-code

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator, Mapping

import torch

from boundflow.backends.tvm.root_crown_terminal_linear import (
    RootCrownTerminalLinearTemplateV1,
)
from boundflow.runtime.root_crown_terminal_capture import (
    ROOT_START_NODE,
    ROOT_TERMINAL_LINEAR,
    ROOT_TERMINAL_RELU,
)
from boundflow.runtime.root_crown_terminal_tir import (
    RootCrownTerminalTIRExecutorV1,
    RootCrownTerminalTensorsV1,
    execute_root_crown_terminal_tir_v1,
)


class RootCrownTerminalLiveBridgeV1:
    """Replace one root ReLU/Linear pair while preserving host ownership."""

    def __init__(
        self,
        template: RootCrownTerminalLinearTemplateV1,
        *,
        capture_debug: bool = False,
    ) -> None:
        template.validate()
        self.template = template
        self.executor = RootCrownTerminalTIRExecutorV1(template)
        self.outer_call_count = 0
        self.relu_replacement_count = 0
        self.linear_replacement_count = 0
        self.fallback_count = 0
        self._active = False
        self._pending: RootCrownTerminalTensorsV1 | None = None
        self._upper_d: torch.Tensor | None = None
        self._zero_bias: torch.Tensor | None = None
        self._capture_debug = capture_debug
        self._debug_evaluations: list[dict[str, torch.Tensor]] = []
        self._debug_gradients: list[torch.Tensor] = []
        self._debug_hook_handles: list[Any] = []
        self._debug_alpha_objects: set[int] = set()

    @staticmethod
    def _eligible(instance: Any, kwargs: Mapping[str, Any]) -> bool:
        opts = getattr(instance, "bound_opts", {}).get("optimize_bound_args", {})
        names = {str(getattr(node, "name", "")) for node in instance.nodes()}
        return (
            kwargs.get("bound_side", "lower") == "lower"
            and int(opts.get("iteration", -1)) == 5
            and ROOT_TERMINAL_RELU in names
            and ROOT_TERMINAL_LINEAR in names
        )

    @contextmanager
    def install(self, bounded_module_type: type[Any]) -> Iterator[None]:
        """Install for one eligible optimizer transaction, then restore exactly."""

        original = bounded_module_type._get_optimized_bounds

        def optimized_wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                self._active
                or self.outer_call_count
                or not self._eligible(instance, kwargs)
            ):
                return original(instance, *args, **kwargs)
            self.outer_call_count += 1
            self._active = True
            try:
                with self._install_nodes(instance):
                    return original(instance, *args, **kwargs)
            finally:
                self._active = False

        bounded_module_type._get_optimized_bounds = optimized_wrapped
        try:
            yield
        finally:
            bounded_module_type._get_optimized_bounds = original
            for handle in self._debug_hook_handles:
                handle.remove()
            self._debug_hook_handles.clear()
            if self._pending is not None:
                raise ValueError("root CROWN live transaction remained partial")

    @contextmanager
    def _install_nodes(self, instance: Any) -> Iterator[None]:
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        relu = nodes[ROOT_TERMINAL_RELU]
        linear = nodes[ROOT_TERMINAL_LINEAR]
        if not getattr(relu, "inputs", ()) or relu.inputs[0] is not linear:
            raise ValueError("root CROWN live topology differs")
        original_relu = relu.bound_backward
        original_linear = linear.bound_backward

        def relu_replacement(_self: Any, *args: Any, **kwargs: Any) -> Any:
            start_node = kwargs.get("start_node")
            if str(getattr(start_node, "name", "")) != ROOT_START_NODE:
                return original_relu(*args, **kwargs)
            if self._pending is not None or len(args) < 3:
                raise ValueError("root CROWN live ReLU call differs")
            last_l_a = args[0]
            last_u_a = args[1]
            preactivation = args[2]
            unstable_idx = kwargs.get("unstable_idx")
            raw_alpha = getattr(relu, "alpha", {}).get(ROOT_START_NODE)
            alpha_indices = getattr(relu, "alpha_indices", None)
            if (
                not torch.is_tensor(last_l_a)
                or last_u_a is not None
                or unstable_idx is not None
                or not torch.is_tensor(raw_alpha)
                or not isinstance(alpha_indices, (tuple, list))
                or len(alpha_indices) != 1
                or tuple(int(value) for value in alpha_indices[0].tolist())
                != self.template.alpha_feature_indices
                or bool(getattr(relu, "cut_used", False))
                or float(getattr(relu, "leaky_alpha", 0.0)) != 0.0
            ):
                raise ValueError("root CROWN live ReLU admission differs")
            lower = preactivation.lower
            upper = preactivation.upper
            tensors = RootCrownTerminalTensorsV1(
                incoming_lower_a=last_l_a,
                preactivation_lower=lower,
                preactivation_upper=upper,
                raw_alpha=raw_alpha,
                operator_weight=linear.inputs[1].lower,
                operator_bias=linear.inputs[2].lower,
            )
            expected = (
                self.template.spec_count,
                self.template.domain_count,
                self.template.current_features,
            )
            if tuple(last_l_a.shape) != expected:
                raise ValueError("root CROWN live incoming shape differs")
            if self._capture_debug:
                self._debug_evaluations.append(
                    {
                        "incoming_lower_a": last_l_a.detach().cpu().contiguous(),
                        "preactivation_lower": lower.detach().cpu().contiguous(),
                        "preactivation_upper": upper.detach().cpu().contiguous(),
                        "raw_alpha": raw_alpha.detach().cpu().contiguous(),
                    }
                )
                identity = id(raw_alpha)
                if identity not in self._debug_alpha_objects:
                    self._debug_alpha_objects.add(identity)

                    def capture_gradient(gradient: torch.Tensor) -> torch.Tensor:
                        self._debug_gradients.append(
                            gradient.detach().cpu().contiguous()
                        )
                        return gradient

                    self._debug_hook_handles.append(
                        raw_alpha.register_hook(capture_gradient)
                    )
            if self._upper_d is None:
                with torch.no_grad():
                    self._upper_d = relu._relu_upper_bound(lower, upper, 0.0)[
                        0
                    ].unsqueeze(0)
                self._zero_bias = torch.zeros(
                    (self.template.spec_count, self.template.domain_count),
                    dtype=last_l_a.dtype,
                    device=last_l_a.device,
                )
            relu.d = self._upper_d
            relu.lA = last_l_a
            relu.init_d = None
            zero_coeffs = bool((upper <= 0).all().item())
            relu.zero_backward_coeffs_l = zero_coeffs
            relu.zero_backward_coeffs_u = zero_coeffs
            relu.masked_beta_lower = relu.masked_beta_upper = None
            self._pending = tensors
            self.relu_replacement_count += 1
            return [(last_l_a, None)], self._zero_bias, 0

        def linear_replacement(_self: Any, *args: Any, **kwargs: Any) -> Any:
            if self._pending is None:
                return original_linear(*args, **kwargs)
            start_node = kwargs.get("start_node")
            if (
                str(getattr(start_node, "name", "")) != ROOT_START_NODE
                or len(args) < 5
                or args[1] is not None
                or args[0].data_ptr() != self._pending.incoming_lower_a.data_ptr()
                or args[3].lower.data_ptr() != self._pending.operator_weight.data_ptr()
                or args[4].lower.data_ptr() != self._pending.operator_bias.data_ptr()
            ):
                raise ValueError("root CROWN live Linear admission differs")
            linear._start = ROOT_START_NODE
            output_a, output_bias = execute_root_crown_terminal_tir_v1(
                self._pending, self.executor
            )
            if self._capture_debug:
                self._debug_evaluations[-1].update(
                    {
                        "output_lower_a": output_a.detach().cpu().contiguous(),
                        "output_bias": output_bias.detach().cpu().contiguous(),
                    }
                )
            self._pending = None
            self.linear_replacement_count += 1
            return (
                [
                    (output_a, None),
                    (None, None),
                    (None, None),
                ],
                output_bias,
                0,
            )

        relu.bound_backward = MethodType(relu_replacement, relu)
        linear.bound_backward = MethodType(linear_replacement, linear)
        try:
            yield
        finally:
            relu.bound_backward = original_relu
            linear.bound_backward = original_linear

    def validate(self) -> None:
        """Require exact 5-forward/4-backward activation without fallback."""

        if (
            self.outer_call_count != 1
            or self.relu_replacement_count != 5
            or self.linear_replacement_count != 5
            or self.executor.forward_launch_count != 5
            or self.executor.backward_launch_count != 4
            or self.fallback_count != 0
            or self.executor.fallback_count != 0
            or self._pending is not None
        ):
            raise ValueError("root CROWN live activation count differs")

    def receipt(self) -> dict[str, object]:
        """Return structural activation and compiler identities."""

        self.validate()
        return {
            "schema_version": "boundflow.root-crown-terminal-live/v1",
            "template_hash": self.template.stable_hash(),
            "unscheduled_tir_hash": self.executor.compiled.unscheduled_tir_hash,
            "scheduled_tir_hash": self.executor.compiled.scheduled_tir_hash,
            "device_source_hash": self.executor.compiled.device_source_hash,
            "outer_call_count": self.outer_call_count,
            "relu_replacement_count": self.relu_replacement_count,
            "linear_replacement_count": self.linear_replacement_count,
            "forward_launch_count": self.executor.forward_launch_count,
            "backward_launch_count": self.executor.backward_launch_count,
            "fallback_count": 0,
            "dlpack_pointer_count": self.executor.pointer_count,
            "dlpack_pointer_exact_count": self.executor.pointer_exact_count,
            "performance_claimed": False,
        }

    def debug_payload(self) -> dict[str, object]:
        """Return optional local-only tensors for trajectory diagnosis."""

        if not self._capture_debug:
            raise RuntimeError("root CROWN live debug capture is disabled")
        self.validate()
        if len(self._debug_evaluations) != 5 or len(self._debug_gradients) != 4:
            raise ValueError("root CROWN live debug trajectory differs")
        return {
            "schema_version": "boundflow.root-crown-terminal-live-debug/v1",
            "evaluations": self._debug_evaluations,
            "raw_alpha_gradients": self._debug_gradients,
            "performance_claimed": False,
        }


__all__ = ["RootCrownTerminalLiveBridgeV1"]
