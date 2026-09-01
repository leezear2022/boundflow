"""Live auto_LiRPA bridge for the compiled root projection residual."""

# pylint: disable=protected-access,too-many-locals,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,too-many-statements,duplicate-code

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator, Mapping, Sequence, cast

import torch

from boundflow.backends.tvm.root_crown_projection import (
    RootCrownProjectionTemplateV1,
)
from boundflow.runtime.root_crown_expanded_suffix_tir import (
    RootCrownExpandedSuffixTensorsV1,
    execute_root_crown_expanded_suffix_tir_v1,
)
from boundflow.runtime.root_crown_projection_capture import (
    ROOT_PROJECTION_ADD,
    ROOT_PROJECTION_ENTRY_RELU,
    ROOT_PROJECTION_EXIT_RELU,
    ROOT_PROJECTION_INNER_RELU,
    ROOT_PROJECTION_MAIN_INNER_CONV,
    ROOT_PROJECTION_MAIN_OUTER_CONV,
    ROOT_PROJECTION_SKIP_CONV,
)
from boundflow.runtime.root_crown_projection_tir import (
    RootCrownProjectionTensorsV1,
    RootCrownProjectionTIRExecutorV1,
    execute_root_crown_projection_tir_v1,
)
from boundflow.runtime.root_crown_terminal_capture import ROOT_START_NODE


def _coordinates(
    values: Sequence[torch.Tensor],
) -> tuple[tuple[int, int, int], ...]:
    if len(values) != 3 or any(value.ndim != 1 for value in values):
        raise ValueError("root CROWN projection live alpha coordinates differ")
    lengths = {int(value.numel()) for value in values}
    if len(lengths) != 1:
        raise ValueError("root CROWN projection live alpha coordinate length differs")
    return cast(
        tuple[tuple[int, int, int], ...],
        tuple(
            tuple(int(values[axis][ordinal]) for axis in range(3))
            for ordinal in range(int(values[0].numel()))
        ),
    )


class RootCrownProjectionLiveBridgeV1:
    """Replace the stride-2 projection region while host control remains native."""

    def __init__(
        self,
        template: RootCrownProjectionTemplateV1,
        expanded_executor: Any | None = None,
    ) -> None:
        template.validate()
        self.template = template
        self.executor = (
            RootCrownProjectionTIRExecutorV1(template)
            if expanded_executor is None
            else expanded_executor.projection
        )
        self._expanded_executor = expanded_executor
        self.outer_call_count = 0
        self.entry_replacement_count = 0
        self.add_replacement_count = 0
        self.skip_carrier_count = 0
        self.bypassed_main_call_count = 0
        self.fallback_count = 0
        self._active = False
        self._pending: RootCrownProjectionTensorsV1 | None = None
        self._last_tensors: RootCrownProjectionTensorsV1 | None = None
        self._zero_bias: torch.Tensor | None = None
        self._entry_upper_d: torch.Tensor | None = None
        self._inner_upper_d: torch.Tensor | None = None
        self._entry_zero_coefficients: bool | None = None
        self._inner_zero_coefficients: bool | None = None
        self._static_admitted = False
        self.exact_warmup_reuse_count = 0

    @staticmethod
    def _eligible(instance: Any, kwargs: Mapping[str, Any]) -> bool:
        opts = getattr(instance, "bound_opts", {}).get("optimize_bound_args", {})
        names = {str(getattr(node, "name", "")) for node in instance.nodes()}
        expected = {
            ROOT_PROJECTION_ENTRY_RELU,
            ROOT_PROJECTION_ADD,
            ROOT_PROJECTION_MAIN_OUTER_CONV,
            ROOT_PROJECTION_INNER_RELU,
            ROOT_PROJECTION_MAIN_INNER_CONV,
            ROOT_PROJECTION_SKIP_CONV,
            ROOT_PROJECTION_EXIT_RELU,
        }
        return (
            kwargs.get("bound_side", "lower") == "lower"
            and int(opts.get("iteration", -1)) == 5
            and expected <= names
        )

    @contextmanager
    def install(self, bounded_module_type: type[Any]) -> Iterator[None]:
        """Install for one eligible optimizer transaction and restore exactly."""

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
            if self._pending is not None:
                raise ValueError(
                    "root CROWN projection live transaction remained partial"
                )

    def _admit_static(self, entry: Any, inner: Any) -> None:
        if self._static_admitted:
            return
        entry_indices = getattr(entry, "alpha_indices", None)
        inner_indices = getattr(inner, "alpha_indices", None)
        if (
            not isinstance(entry_indices, (tuple, list))
            or not isinstance(inner_indices, (tuple, list))
            or _coordinates(entry_indices) != self.template.entry_alpha_coordinates
            or _coordinates(inner_indices) != self.template.inner_alpha_coordinates
            or bool(getattr(entry, "cut_used", False))
            or bool(getattr(inner, "cut_used", False))
            or float(getattr(entry, "leaky_alpha", 0.0)) != 0.0
            or float(getattr(inner, "leaky_alpha", 0.0)) != 0.0
        ):
            raise ValueError("root CROWN projection live static admission differs")
        self._static_admitted = True

    def _set_relu_state(
        self,
        node: Any,
        lower: torch.Tensor,
        upper: torch.Tensor,
        incoming: torch.Tensor,
        *,
        inner: bool,
    ) -> None:
        upper_d = self._inner_upper_d if inner else self._entry_upper_d
        zero_coefficients = (
            self._inner_zero_coefficients if inner else self._entry_zero_coefficients
        )
        if upper_d is None or zero_coefficients is None:
            with torch.no_grad():
                upper_d = node._relu_upper_bound(lower, upper, 0.0)[0].unsqueeze(0)
                zero_coefficients = bool((upper <= 0).all().item())
            if inner:
                self._inner_upper_d = upper_d
                self._inner_zero_coefficients = zero_coefficients
            else:
                self._entry_upper_d = upper_d
                self._entry_zero_coefficients = zero_coefficients
        node.d = upper_d
        node.lA = incoming
        node.init_d = None
        node.zero_backward_coeffs_l = zero_coefficients
        node.zero_backward_coeffs_u = zero_coefficients
        node.masked_beta_lower = node.masked_beta_upper = None

    @contextmanager
    def _install_nodes(self, instance: Any) -> Iterator[None]:
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        entry = nodes[ROOT_PROJECTION_ENTRY_RELU]
        add = nodes[ROOT_PROJECTION_ADD]
        outer_conv = nodes[ROOT_PROJECTION_MAIN_OUTER_CONV]
        inner = nodes[ROOT_PROJECTION_INNER_RELU]
        inner_conv = nodes[ROOT_PROJECTION_MAIN_INNER_CONV]
        skip_conv = nodes[ROOT_PROJECTION_SKIP_CONV]
        exit_relu = nodes[ROOT_PROJECTION_EXIT_RELU]
        if (
            not entry.inputs
            or entry.inputs[0] is not add
            or len(add.inputs) != 2
            or add.inputs[0] is not outer_conv
            or add.inputs[1] is not skip_conv
            or not outer_conv.inputs
            or outer_conv.inputs[0] is not inner
            or not inner.inputs
            or inner.inputs[0] is not inner_conv
            or not inner_conv.inputs
            or inner_conv.inputs[0] is not exit_relu
            or not skip_conv.inputs
            or skip_conv.inputs[0] is not exit_relu
        ):
            raise ValueError("root CROWN projection live topology differs")
        selected = (entry, add, outer_conv, inner, inner_conv, skip_conv)
        originals = {node: node.bound_backward for node in selected}

        def entry_replacement(_self: Any, *args: Any, **kwargs: Any) -> Any:
            if str(getattr(kwargs.get("start_node"), "name", "")) != ROOT_START_NODE:
                return originals[entry](*args, **kwargs)
            if (
                self._pending is not None
                or len(args) < 3
                or not torch.is_tensor(args[0])
                or args[1] is not None
                or kwargs.get("unstable_idx") is not None
            ):
                raise ValueError("root CROWN projection live entry call differs")
            self._admit_static(entry, inner)
            incoming = args[0]
            preactivation = args[2]
            inner_preactivation = inner.inputs[0]
            entry_alpha = getattr(entry, "alpha", {}).get(ROOT_START_NODE)
            inner_alpha = getattr(inner, "alpha", {}).get(ROOT_START_NODE)
            if not torch.is_tensor(entry_alpha) or not torch.is_tensor(inner_alpha):
                raise ValueError("root CROWN projection live alpha differs")
            self._pending = RootCrownProjectionTensorsV1(
                incoming_lower_a=incoming,
                entry_lower=preactivation.lower,
                entry_upper=preactivation.upper,
                entry_raw_alpha=entry_alpha,
                main_outer_conv_weight=outer_conv.inputs[1].lower,
                main_outer_conv_bias=outer_conv.inputs[2].lower,
                inner_lower=inner_preactivation.lower,
                inner_upper=inner_preactivation.upper,
                inner_raw_alpha=inner_alpha,
                main_inner_conv_weight=inner_conv.inputs[1].lower,
                main_inner_conv_bias=inner_conv.inputs[2].lower,
                skip_conv_weight=skip_conv.inputs[1].lower,
                skip_conv_bias=skip_conv.inputs[2].lower,
            )
            self._set_relu_state(
                entry,
                preactivation.lower,
                preactivation.upper,
                incoming,
                inner=False,
            )
            if self._zero_bias is None:
                self._zero_bias = torch.zeros(
                    (self.template.spec_count, self.template.domain_count),
                    dtype=incoming.dtype,
                    device=incoming.device,
                )
            self.entry_replacement_count += 1
            return [(incoming, None)], self._zero_bias, 0

        def add_replacement(_self: Any, *args: Any, **kwargs: Any) -> Any:
            del kwargs
            if self._pending is None:
                return originals[add](*args)
            if (
                len(args) < 4
                or args[1] is not None
                or not torch.is_tensor(args[0])
                or args[0].data_ptr() != self._pending.incoming_lower_a.data_ptr()
            ):
                raise ValueError("root CROWN projection live Add call differs")
            tensors = self._pending
            if self._expanded_executor is None:
                output_a, output_bias = execute_root_crown_projection_tir_v1(
                    tensors, self.executor
                )
            else:
                expanded_tensors = RootCrownExpandedSuffixTensorsV1(
                    self._expanded_executor.staged_suffix, tensors
                )
                if hasattr(self._expanded_executor, "stage_projection"):
                    output_a, _staged_bias = self._expanded_executor.stage_projection(
                        expanded_tensors
                    )
                    zero_bias = self._zero_bias
                    if zero_bias is None:
                        raise RuntimeError("root CROWN projection zero bias is absent")
                    output_bias = zero_bias
                else:
                    output_a, output_bias = execute_root_crown_expanded_suffix_tir_v1(
                        expanded_tensors, self._expanded_executor
                    )
            self._last_tensors = tensors
            self._pending = None
            self.add_replacement_count += 1
            return [(None, None), (output_a, None)], output_bias, 0

        def skip_carrier_replacement(_self: Any, *args: Any, **kwargs: Any) -> Any:
            if str(getattr(kwargs.get("start_node"), "name", "")) != ROOT_START_NODE:
                return originals[skip_conv](*args, **kwargs)
            if (
                len(args) < 2
                or not torch.is_tensor(args[0])
                or args[1] is not None
                or tuple(args[0].shape) != self.template.output_shape
                or self._zero_bias is None
            ):
                raise ValueError("root CROWN projection skip carrier differs")
            self.skip_carrier_count += 1
            return [(args[0], None), (None, None), (None, None)], self._zero_bias, 0

        def reject_bypassed(node: Any, original: Any, *args: Any, **kwargs: Any) -> Any:
            if str(getattr(kwargs.get("start_node"), "name", "")) != ROOT_START_NODE:
                return original(*args, **kwargs)
            self.bypassed_main_call_count += 1
            raise RuntimeError(
                f"root CROWN projection bypassed node executed: {node.name}"
            )

        replacements = {
            entry: entry_replacement,
            add: add_replacement,
            outer_conv: lambda _self, *args, **kwargs: reject_bypassed(
                outer_conv, originals[outer_conv], *args, **kwargs
            ),
            inner: lambda _self, *args, **kwargs: reject_bypassed(
                inner, originals[inner], *args, **kwargs
            ),
            inner_conv: lambda _self, *args, **kwargs: reject_bypassed(
                inner_conv, originals[inner_conv], *args, **kwargs
            ),
            skip_conv: skip_carrier_replacement,
        }
        for node, replacement in replacements.items():
            node.bound_backward = MethodType(replacement, node)
        try:
            yield
            if self._last_tensors is not None:
                tensors = self._last_tensors
                self._set_relu_state(
                    entry,
                    tensors.entry_lower,
                    tensors.entry_upper,
                    tensors.incoming_lower_a,
                    inner=False,
                )
                self._set_relu_state(
                    inner,
                    tensors.inner_lower,
                    tensors.inner_upper,
                    self.executor.last_outer_a,
                    inner=True,
                )
        finally:
            for node, original in originals.items():
                node.bound_backward = original

    def validate(self) -> None:
        """Require exactly five forward and four backward compiled calls."""

        if (
            self.outer_call_count != 1
            or self.executor.prepare_count != 1
            or self.entry_replacement_count != 5
            or self.add_replacement_count != 5
            or self.skip_carrier_count != 5
            or self.bypassed_main_call_count != 0
            or self.executor.forward_launch_count != 5
            or self.executor.backward_launch_count != 4
            or self.fallback_count != 0
            or self.executor.fallback_count != 0
            or self._pending is not None
        ):
            raise ValueError("root CROWN projection live activation count differs")

    def reset_after_exact_warmup_v1(self) -> None:
        """Preserve static admission and clear one completed warm transaction."""

        self.validate()
        if (
            self.exact_warmup_reuse_count != 0
            or not self._static_admitted
            or self._entry_upper_d is None
            or self._inner_upper_d is None
            or self._entry_zero_coefficients is None
            or self._inner_zero_coefficients is None
            or self._active
            or self._pending is not None
        ):
            raise ValueError("root CROWN projection warm reuse state differs")
        self.outer_call_count = 0
        self.entry_replacement_count = 0
        self.add_replacement_count = 0
        self.skip_carrier_count = 0
        self.bypassed_main_call_count = 0
        self.fallback_count = 0
        self._last_tensors = None
        self.exact_warmup_reuse_count = 1

    def receipt(self) -> dict[str, object]:
        """Return actual activation and compiler identities."""

        self.validate()
        return {
            "schema_version": "boundflow.root-crown-projection-live/v1",
            "template_hash": self.template.stable_hash(),
            "unscheduled_tir_hash": self.executor.compiled.unscheduled_tir_hash,
            "scheduled_tir_hash": self.executor.compiled.scheduled_tir_hash,
            "device_source_hash": self.executor.compiled.device_source_hash,
            "workspace_inventory": [
                [name, list(shape)]
                for name, shape in self.executor.compiled.workspace_inventory
            ],
            "outer_call_count": self.outer_call_count,
            "prepare_count": self.executor.prepare_count,
            "entry_replacement_count": self.entry_replacement_count,
            "add_replacement_count": self.add_replacement_count,
            "skip_carrier_count": self.skip_carrier_count,
            "bypassed_main_call_count": self.bypassed_main_call_count,
            "exact_warmup_reuse_count": self.exact_warmup_reuse_count,
            "forward_launch_count": self.executor.forward_launch_count,
            "backward_launch_count": self.executor.backward_launch_count,
            "fallback_count": 0,
            "dlpack_pointer_count": self.executor.pointer_count,
            "dlpack_pointer_exact_count": self.executor.pointer_exact_count,
            "performance_claimed": False,
        }


__all__ = ["RootCrownProjectionLiveBridgeV1"]
