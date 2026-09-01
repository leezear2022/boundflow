"""Live auto_LiRPA bridge for the cumulative terminal/residual suffix."""

# pylint: disable=protected-access,too-many-locals,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,too-many-statements,duplicate-code

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator, Mapping, Sequence, cast

import torch

from boundflow.backends.tvm.root_crown_residual import (
    RootCrownResidualTemplateV1,
)
from boundflow.backends.tvm.root_crown_terminal_linear import (
    RootCrownTerminalLinearTemplateV1,
)
from boundflow.runtime.root_crown_residual_capture import (
    ROOT_RESIDUAL_ADD,
    ROOT_RESIDUAL_ENTRY_RELU,
    ROOT_RESIDUAL_EXIT_RELU,
    ROOT_RESIDUAL_INNER_CONV,
    ROOT_RESIDUAL_INNER_RELU,
    ROOT_RESIDUAL_MAIN_CONV,
)
from boundflow.runtime.root_crown_residual_tir import (
    RootCrownResidualTensorsV1,
)
from boundflow.runtime.root_crown_suffix_tir import (
    RootCrownSuffixTensorsV1,
    RootCrownSuffixTIRExecutorV1,
    execute_root_crown_suffix_tir_v1,
)
from boundflow.runtime.root_crown_terminal_capture import (
    ROOT_START_NODE,
    ROOT_TERMINAL_LINEAR,
    ROOT_TERMINAL_RELU,
)
from boundflow.runtime.root_crown_terminal_tir import (
    RootCrownTerminalTensorsV1,
)


def _coordinates(
    values: Sequence[torch.Tensor],
) -> tuple[tuple[int, int, int], ...]:
    if len(values) != 3 or any(value.ndim != 1 for value in values):
        raise ValueError("root CROWN suffix alpha coordinates differ")
    lengths = {int(value.numel()) for value in values}
    if len(lengths) != 1:
        raise ValueError("root CROWN suffix alpha coordinate length differs")
    return cast(
        tuple[tuple[int, int, int], ...],
        tuple(
            tuple(int(values[axis][ordinal]) for axis in range(3))
            for ordinal in range(int(values[0].numel()))
        ),
    )


class RootCrownSuffixLiveBridgeV1:
    """Replace terminal and one residual block with one gradient owner."""

    def __init__(
        self,
        terminal_template: RootCrownTerminalLinearTemplateV1,
        residual_template: RootCrownResidualTemplateV1,
        expanded_executor: Any | None = None,
    ) -> None:
        self.terminal_template = terminal_template
        self.residual_template = residual_template
        self.executor = (
            RootCrownSuffixTIRExecutorV1(terminal_template, residual_template)
            if expanded_executor is None
            else expanded_executor.suffix
        )
        self._expanded_executor = expanded_executor
        self.outer_call_count = 0
        self.terminal_relu_count = 0
        self.terminal_linear_count = 0
        self.residual_entry_count = 0
        self.residual_add_count = 0
        self.bypassed_main_call_count = 0
        self.fallback_count = 0
        self._active = False
        self._terminal_pending: RootCrownTerminalTensorsV1 | None = None
        self._terminal_current: RootCrownTerminalTensorsV1 | None = None
        self._residual_pending: RootCrownResidualTensorsV1 | None = None
        self._last_suffix: RootCrownSuffixTensorsV1 | None = None
        self._zero_bias: torch.Tensor | None = None
        self._upper_d: dict[str, torch.Tensor] = {}
        self._zero_coefficients: dict[str, bool] = {}
        self._static_admitted = False

    @staticmethod
    def _eligible(instance: Any, kwargs: Mapping[str, Any]) -> bool:
        opts = getattr(instance, "bound_opts", {}).get("optimize_bound_args", {})
        names = {str(getattr(node, "name", "")) for node in instance.nodes()}
        expected = {
            ROOT_TERMINAL_RELU,
            ROOT_TERMINAL_LINEAR,
            ROOT_RESIDUAL_ENTRY_RELU,
            ROOT_RESIDUAL_ADD,
            ROOT_RESIDUAL_MAIN_CONV,
            ROOT_RESIDUAL_INNER_RELU,
            ROOT_RESIDUAL_INNER_CONV,
            ROOT_RESIDUAL_EXIT_RELU,
        }
        return (
            kwargs.get("bound_side", "lower") == "lower"
            and int(opts.get("iteration", -1)) == 5
            and expected <= names
        )

    @contextmanager
    def install(self, bounded_module_type: type[Any]) -> Iterator[None]:
        """Install for one optimizer transaction, then restore all methods."""

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
            if any(
                value is not None
                for value in (
                    self._terminal_pending,
                    self._terminal_current,
                    self._residual_pending,
                )
            ):
                raise ValueError("root CROWN suffix transaction remained partial")

    def _set_relu_state(
        self,
        node: Any,
        lower: torch.Tensor,
        upper: torch.Tensor,
        incoming: torch.Tensor,
        key: str,
    ) -> None:
        upper_d = self._upper_d.get(key)
        zero_coefficients = self._zero_coefficients.get(key)
        if upper_d is None or zero_coefficients is None:
            with torch.no_grad():
                upper_d = node._relu_upper_bound(lower, upper, 0.0)[0].unsqueeze(0)
                zero_coefficients = bool((upper <= 0).all().item())
            self._upper_d[key] = upper_d
            self._zero_coefficients[key] = zero_coefficients
        node.d = upper_d
        node.lA = incoming
        node.init_d = None
        node.zero_backward_coeffs_l = zero_coefficients
        node.zero_backward_coeffs_u = zero_coefficients
        node.masked_beta_lower = node.masked_beta_upper = None

    def _admit_static(self, terminal_relu: Any, entry: Any, inner: Any) -> None:
        if self._static_admitted:
            return
        terminal_indices = getattr(terminal_relu, "alpha_indices", None)
        entry_indices = getattr(entry, "alpha_indices", None)
        inner_indices = getattr(inner, "alpha_indices", None)
        if (
            not isinstance(terminal_indices, (tuple, list))
            or len(terminal_indices) != 1
            or tuple(int(value) for value in terminal_indices[0].tolist())
            != self.terminal_template.alpha_feature_indices
            or not isinstance(entry_indices, (tuple, list))
            or not isinstance(inner_indices, (tuple, list))
            or _coordinates(entry_indices)
            != self.residual_template.entry_alpha_coordinates
            or _coordinates(inner_indices)
            != self.residual_template.inner_alpha_coordinates
            or any(
                bool(getattr(node, "cut_used", False))
                or float(getattr(node, "leaky_alpha", 0.0)) != 0.0
                for node in (terminal_relu, entry, inner)
            )
        ):
            raise ValueError("root CROWN suffix static admission differs")
        self._static_admitted = True

    @contextmanager
    def _install_nodes(self, instance: Any) -> Iterator[None]:
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        terminal_relu = nodes[ROOT_TERMINAL_RELU]
        terminal_linear = nodes[ROOT_TERMINAL_LINEAR]
        entry = nodes[ROOT_RESIDUAL_ENTRY_RELU]
        add = nodes[ROOT_RESIDUAL_ADD]
        main_conv = nodes[ROOT_RESIDUAL_MAIN_CONV]
        inner = nodes[ROOT_RESIDUAL_INNER_RELU]
        inner_conv = nodes[ROOT_RESIDUAL_INNER_CONV]
        exit_relu = nodes[ROOT_RESIDUAL_EXIT_RELU]
        if (
            not terminal_relu.inputs
            or terminal_relu.inputs[0] is not terminal_linear
            or not entry.inputs
            or entry.inputs[0] is not add
            or len(add.inputs) != 2
            or add.inputs[0] is not main_conv
            or add.inputs[1] is not exit_relu
            or not main_conv.inputs
            or main_conv.inputs[0] is not inner
            or not inner.inputs
            or inner.inputs[0] is not inner_conv
            or not inner_conv.inputs
            or inner_conv.inputs[0] is not exit_relu
        ):
            raise ValueError("root CROWN suffix topology differs")
        selected = (
            terminal_relu,
            terminal_linear,
            entry,
            add,
            main_conv,
            inner,
            inner_conv,
        )
        originals = {node: node.bound_backward for node in selected}

        def terminal_relu_replacement(_self: Any, *args: Any, **kwargs: Any) -> Any:
            if str(getattr(kwargs.get("start_node"), "name", "")) != ROOT_START_NODE:
                return originals[terminal_relu](*args, **kwargs)
            if (
                self._terminal_pending is not None
                or self._terminal_current is not None
                or len(args) < 3
                or not torch.is_tensor(args[0])
                or args[1] is not None
                or kwargs.get("unstable_idx") is not None
            ):
                raise ValueError("root CROWN suffix terminal ReLU call differs")
            self._admit_static(terminal_relu, entry, inner)
            incoming = args[0]
            preactivation = args[2]
            raw_alpha = getattr(terminal_relu, "alpha", {}).get(ROOT_START_NODE)
            if not torch.is_tensor(raw_alpha):
                raise ValueError("root CROWN suffix terminal alpha differs")
            self._terminal_pending = RootCrownTerminalTensorsV1(
                incoming,
                preactivation.lower,
                preactivation.upper,
                raw_alpha,
                terminal_linear.inputs[1].lower,
                terminal_linear.inputs[2].lower,
            )
            self._set_relu_state(
                terminal_relu,
                preactivation.lower,
                preactivation.upper,
                incoming,
                "terminal",
            )
            if self._zero_bias is None:
                self._zero_bias = torch.zeros(
                    (
                        self.terminal_template.spec_count,
                        self.terminal_template.domain_count,
                    ),
                    dtype=incoming.dtype,
                    device=incoming.device,
                )
            self.terminal_relu_count += 1
            return [(incoming, None)], self._zero_bias, 0

        def terminal_linear_replacement(_self: Any, *args: Any, **kwargs: Any) -> Any:
            if self._terminal_pending is None:
                return originals[terminal_linear](*args, **kwargs)
            if (
                str(getattr(kwargs.get("start_node"), "name", "")) != ROOT_START_NODE
                or len(args) < 5
                or args[1] is not None
                or not torch.is_tensor(args[0])
                or self._zero_bias is None
                or args[0].data_ptr()
                != self._terminal_pending.incoming_lower_a.data_ptr()
            ):
                raise ValueError("root CROWN suffix terminal Linear call differs")
            terminal_linear._start = ROOT_START_NODE
            output_a, _output_bias = self.executor.stage_terminal(
                self._terminal_pending
            )
            self._terminal_current = self._terminal_pending
            self._terminal_pending = None
            self.terminal_linear_count += 1
            return (
                [(output_a, None), (None, None), (None, None)],
                self._zero_bias,
                0,
            )

        def entry_replacement(_self: Any, *args: Any, **kwargs: Any) -> Any:
            if str(getattr(kwargs.get("start_node"), "name", "")) != ROOT_START_NODE:
                return originals[entry](*args, **kwargs)
            if (
                self._terminal_current is None
                or self._residual_pending is not None
                or len(args) < 3
                or not torch.is_tensor(args[0])
                or args[1] is not None
                or kwargs.get("unstable_idx") is not None
            ):
                raise ValueError("root CROWN suffix residual entry call differs")
            incoming = args[0]
            if incoming.data_ptr() != self.executor.last_terminal_a.data_ptr():
                raise ValueError("root CROWN suffix host reshape boundary differs")
            preactivation = args[2]
            inner_preactivation = inner.inputs[0]
            entry_alpha = getattr(entry, "alpha", {}).get(ROOT_START_NODE)
            inner_alpha = getattr(inner, "alpha", {}).get(ROOT_START_NODE)
            if not torch.is_tensor(entry_alpha) or not torch.is_tensor(inner_alpha):
                raise ValueError("root CROWN suffix residual alpha differs")
            self._residual_pending = RootCrownResidualTensorsV1(
                incoming,
                preactivation.lower,
                preactivation.upper,
                entry_alpha,
                main_conv.inputs[1].lower,
                main_conv.inputs[2].lower,
                inner_preactivation.lower,
                inner_preactivation.upper,
                inner_alpha,
                inner_conv.inputs[1].lower,
                inner_conv.inputs[2].lower,
            )
            self._set_relu_state(
                entry,
                preactivation.lower,
                preactivation.upper,
                incoming,
                "entry",
            )
            self.residual_entry_count += 1
            return [(incoming, None)], self._zero_bias, 0

        def add_replacement(_self: Any, *args: Any, **kwargs: Any) -> Any:
            del kwargs
            if self._residual_pending is None:
                return originals[add](*args)
            if (
                self._terminal_current is None
                or len(args) < 4
                or args[1] is not None
                or not torch.is_tensor(args[0])
                or args[0].data_ptr()
                != self._residual_pending.incoming_lower_a.data_ptr()
            ):
                raise ValueError("root CROWN suffix Add call differs")
            suffix = RootCrownSuffixTensorsV1(
                self._terminal_current, self._residual_pending
            )
            zero_bias = self._zero_bias
            if zero_bias is None:
                raise RuntimeError("root CROWN suffix zero bias is absent")
            if self._expanded_executor is None:
                output_a, output_bias = execute_root_crown_suffix_tir_v1(
                    suffix, self.executor
                )
            else:
                output_a, _staged_bias = self._expanded_executor.stage_residual(suffix)
                output_bias = zero_bias
            self._last_suffix = suffix
            self._terminal_current = None
            self._residual_pending = None
            self.residual_add_count += 1
            return [(None, None), (output_a, None)], output_bias, 0

        def reject_bypassed(node: Any, original: Any, *args: Any, **kwargs: Any) -> Any:
            if str(getattr(kwargs.get("start_node"), "name", "")) != ROOT_START_NODE:
                return original(*args, **kwargs)
            self.bypassed_main_call_count += 1
            raise RuntimeError(f"root CROWN suffix bypassed node executed: {node.name}")

        replacements = {
            terminal_relu: terminal_relu_replacement,
            terminal_linear: terminal_linear_replacement,
            entry: entry_replacement,
            add: add_replacement,
            main_conv: lambda _self, *args, **kwargs: reject_bypassed(
                main_conv, originals[main_conv], *args, **kwargs
            ),
            inner: lambda _self, *args, **kwargs: reject_bypassed(
                inner, originals[inner], *args, **kwargs
            ),
            inner_conv: lambda _self, *args, **kwargs: reject_bypassed(
                inner_conv, originals[inner_conv], *args, **kwargs
            ),
        }
        for node, replacement in replacements.items():
            node.bound_backward = MethodType(replacement, node)
        try:
            yield
            if self._last_suffix is not None:
                residual = self._last_suffix.residual
                self._set_relu_state(
                    entry,
                    residual.entry_lower,
                    residual.entry_upper,
                    self.executor.last_terminal_a.view(
                        self.residual_template.coefficient_shape
                    ),
                    "entry",
                )
                self._set_relu_state(
                    inner,
                    residual.inner_lower,
                    residual.inner_upper,
                    self.executor.last_main_a,
                    "inner",
                )
        finally:
            for node, original in originals.items():
                node.bound_backward = original

    def validate(self) -> None:
        """Require exact five-forward/four-backward cumulative activation."""

        if (
            self.outer_call_count != 1
            or self.executor.prepare_count != 1
            or self.terminal_relu_count != 5
            or self.terminal_linear_count != 5
            or self.residual_entry_count != 5
            or self.residual_add_count != 5
            or self.executor.stage_count != 5
            or self.executor.consume_count != 5
            or self.executor.terminal.forward_launch_count != 5
            or self.executor.terminal.backward_launch_count != 4
            or self.executor.residual.forward_launch_count != 5
            or self.executor.residual.backward_launch_count != 4
            or self.bypassed_main_call_count != 0
            or self.fallback_count != 0
            or self.executor.fallback_count != 0
        ):
            raise ValueError("root CROWN suffix activation count differs")

    def receipt(self) -> dict[str, object]:
        """Return activation, compiler identity and boundary counters."""

        self.validate()
        return {
            "schema_version": "boundflow.root-crown-suffix-live/v1",
            "terminal_template_hash": self.terminal_template.stable_hash(),
            "residual_template_hash": self.residual_template.stable_hash(),
            "terminal_scheduled_tir_hash": (
                self.executor.terminal.compiled.scheduled_tir_hash
            ),
            "residual_scheduled_tir_hash": (
                self.executor.residual.compiled.scheduled_tir_hash
            ),
            "terminal_device_source_hash": (
                self.executor.terminal.compiled.device_source_hash
            ),
            "residual_device_source_hash": (
                self.executor.residual.compiled.device_source_hash
            ),
            "prepare_count": self.executor.prepare_count,
            "outer_call_count": self.outer_call_count,
            "terminal_relu_count": self.terminal_relu_count,
            "terminal_linear_count": self.terminal_linear_count,
            "residual_entry_count": self.residual_entry_count,
            "residual_add_count": self.residual_add_count,
            "cumulative_autograd_owner_count": (
                self.residual_add_count if self._expanded_executor is None else 0
            ),
            "intermediate_autograd_owner_count": 0,
            "terminal_forward_launch_count": (
                self.executor.terminal.forward_launch_count
            ),
            "terminal_backward_launch_count": (
                self.executor.terminal.backward_launch_count
            ),
            "residual_forward_launch_count": (
                self.executor.residual.forward_launch_count
            ),
            "residual_backward_launch_count": (
                self.executor.residual.backward_launch_count
            ),
            "bypassed_main_call_count": self.bypassed_main_call_count,
            "fallback_count": 0,
            "performance_claimed": False,
        }


__all__ = ["RootCrownSuffixLiveBridgeV1"]
