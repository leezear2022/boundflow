"""Capture the native root CROWN residual-block transaction and full VJP."""

# pylint: disable=protected-access,too-many-instance-attributes,too-many-locals
# pylint: disable=too-many-statements,too-many-boolean-expressions

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Iterator, Mapping

import torch

from boundflow.runtime.root_crown_terminal_capture import (
    ROOT_START_NODE,
    _cpu_clone,
    _tensor,
)

ROOT_RESIDUAL_ENTRY_RELU = "/45"
ROOT_RESIDUAL_ADD = "/44"
ROOT_RESIDUAL_MAIN_CONV = "/43"
ROOT_RESIDUAL_INNER_RELU = "/input-24"
ROOT_RESIDUAL_INNER_CONV = "/input-20"
ROOT_RESIDUAL_EXIT_RELU = "/input-16"


def _retain(value: torch.Tensor) -> None:
    if value.requires_grad:
        value.retain_grad()


def _gradient(value: torch.Tensor | None, *, name: str) -> torch.Tensor:
    live = _tensor(value, name=f"live {name}")
    return _tensor(live.grad, name=f"{name} gradient")


@dataclass
class RootCrownResidualEvaluationCaptureV1:
    """One native residual evaluation and the VJP observed after backward."""

    ordinal: int
    incoming_lower_a: torch.Tensor
    entry_lower: torch.Tensor
    entry_upper: torch.Tensor
    entry_raw_alpha: torch.Tensor
    entry_alpha_feature_indices: tuple[torch.Tensor, ...]
    main_conv_weight: torch.Tensor | None = None
    main_conv_bias: torch.Tensor | None = None
    inner_lower: torch.Tensor | None = None
    inner_upper: torch.Tensor | None = None
    inner_raw_alpha: torch.Tensor | None = None
    inner_alpha_feature_indices: tuple[torch.Tensor, ...] = ()
    inner_conv_weight: torch.Tensor | None = None
    inner_conv_bias: torch.Tensor | None = None
    output_lower_a: torch.Tensor | None = None
    output_bias: torch.Tensor | None = None
    output_lower_a_gradient: torch.Tensor | None = None
    output_bias_gradient: torch.Tensor | None = None
    incoming_lower_a_gradient: torch.Tensor | None = None
    entry_lower_gradient: torch.Tensor | None = None
    entry_upper_gradient: torch.Tensor | None = None
    entry_raw_alpha_gradient: torch.Tensor | None = None
    inner_lower_gradient: torch.Tensor | None = None
    inner_upper_gradient: torch.Tensor | None = None
    inner_raw_alpha_gradient: torch.Tensor | None = None
    _stage: int = 1
    _bias_parts: list[torch.Tensor] = field(default_factory=list)
    _live_incoming_lower_a: torch.Tensor | None = None
    _live_entry_lower: torch.Tensor | None = None
    _live_entry_upper: torch.Tensor | None = None
    _live_entry_raw_alpha: torch.Tensor | None = None
    _live_inner_lower: torch.Tensor | None = None
    _live_inner_upper: torch.Tensor | None = None
    _live_inner_raw_alpha: torch.Tensor | None = None
    _live_output_lower_a: torch.Tensor | None = None

    @property
    def backward_captured(self) -> bool:
        """Whether native autograd evidence was copied for this evaluation."""

        return self.output_lower_a_gradient is not None

    def tensor_payload(self) -> dict[str, object]:
        """Return the local-only production tensors used by the next compiler step."""

        return {
            "ordinal": self.ordinal,
            "incoming_lower_a": self.incoming_lower_a,
            "entry_lower": self.entry_lower,
            "entry_upper": self.entry_upper,
            "entry_raw_alpha": self.entry_raw_alpha,
            "entry_alpha_feature_indices": self.entry_alpha_feature_indices,
            "main_conv_weight": self.main_conv_weight,
            "main_conv_bias": self.main_conv_bias,
            "inner_lower": self.inner_lower,
            "inner_upper": self.inner_upper,
            "inner_raw_alpha": self.inner_raw_alpha,
            "inner_alpha_feature_indices": self.inner_alpha_feature_indices,
            "inner_conv_weight": self.inner_conv_weight,
            "inner_conv_bias": self.inner_conv_bias,
            "output_lower_a": self.output_lower_a,
            "output_bias": self.output_bias,
            "output_lower_a_gradient": self.output_lower_a_gradient,
            "output_bias_gradient": self.output_bias_gradient,
            "incoming_lower_a_gradient": self.incoming_lower_a_gradient,
            "entry_lower_gradient": self.entry_lower_gradient,
            "entry_upper_gradient": self.entry_upper_gradient,
            "entry_raw_alpha_gradient": self.entry_raw_alpha_gradient,
            "inner_lower_gradient": self.inner_lower_gradient,
            "inner_upper_gradient": self.inner_upper_gradient,
            "inner_raw_alpha_gradient": self.inner_raw_alpha_gradient,
        }


class RootCrownResidualCaptureV1:
    """Observe exactly one five-forward/four-backward root residual transaction."""

    def __init__(self) -> None:
        self.outer_call_count = 0
        self.backward_call_count = 0
        self.node_call_counts = {
            name: 0
            for name in (
                ROOT_RESIDUAL_ENTRY_RELU,
                ROOT_RESIDUAL_ADD,
                ROOT_RESIDUAL_MAIN_CONV,
                ROOT_RESIDUAL_INNER_RELU,
                ROOT_RESIDUAL_INNER_CONV,
                ROOT_RESIDUAL_EXIT_RELU,
            )
        }
        self.evaluations: list[RootCrownResidualEvaluationCaptureV1] = []
        self._pending: RootCrownResidualEvaluationCaptureV1 | None = None
        self._active = False
        self.device_before: int | None = None
        self.device_after: int | None = None
        self.stream_before: int | None = None
        self.stream_after: int | None = None

    @staticmethod
    def _eligible(instance: Any, kwargs: Mapping[str, Any]) -> bool:
        opts = getattr(instance, "bound_opts", {}).get("optimize_bound_args", {})
        names = {str(getattr(node, "name", "")) for node in instance.nodes()}
        expected = {
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
        """Install the diagnostic hooks, then restore all methods exactly."""

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
            self.device_before = torch.cuda.current_device()
            self.stream_before = int(torch.cuda.current_stream().cuda_stream)
            try:
                with self._install_nodes(instance), self._install_backward():
                    return original(instance, *args, **kwargs)
            finally:
                self.device_after = torch.cuda.current_device()
                self.stream_after = int(torch.cuda.current_stream().cuda_stream)
                self._active = False

        bounded_module_type._get_optimized_bounds = optimized_wrapped
        try:
            yield
        finally:
            bounded_module_type._get_optimized_bounds = original
            if self._pending is not None:
                raise ValueError("root CROWN residual capture remained partial")

    @staticmethod
    def _indices(node: Any) -> tuple[torch.Tensor, ...]:
        value = getattr(node, "alpha_indices", None)
        if (
            not isinstance(value, (tuple, list))
            or not value
            or not all(torch.is_tensor(item) for item in value)
        ):
            raise ValueError("root CROWN residual alpha indices differ")
        return tuple(_cpu_clone(item) for item in value)

    @staticmethod
    def _alpha(node: Any) -> torch.Tensor:
        return _tensor(
            getattr(node, "alpha", {}).get(ROOT_START_NODE), name="residual alpha"
        )

    def _record_bias(self, result: Any) -> None:
        if self._pending is None:
            raise ValueError("root CROWN residual bias arrived without transaction")
        try:
            value = result[1]
        except (IndexError, TypeError) as error:
            raise ValueError("root CROWN residual bias result differs") from error
        if torch.is_tensor(value):
            _retain(value)
            self._pending._bias_parts.append(value)
        elif value not in (0, 0.0, None):
            raise ValueError("root CROWN residual non-tensor bias differs")

    @contextmanager
    def _install_nodes(self, instance: Any) -> Iterator[None]:
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        originals = {name: nodes[name].bound_backward for name in self.node_call_counts}

        entry = nodes[ROOT_RESIDUAL_ENTRY_RELU]
        add = nodes[ROOT_RESIDUAL_ADD]
        main_conv = nodes[ROOT_RESIDUAL_MAIN_CONV]
        inner = nodes[ROOT_RESIDUAL_INNER_RELU]
        inner_conv = nodes[ROOT_RESIDUAL_INNER_CONV]
        exit_relu = nodes[ROOT_RESIDUAL_EXIT_RELU]
        if (
            not entry.inputs
            or entry.inputs[0] is not add
            or len(add.inputs) != 2
            or add.inputs[0] is not main_conv
            or not main_conv.inputs
            or main_conv.inputs[0] is not inner
            or not inner.inputs
            or inner.inputs[0] is not inner_conv
            or not inner_conv.inputs
            or inner_conv.inputs[0] is not exit_relu
            or add.inputs[1] is not exit_relu
        ):
            raise ValueError("root CROWN residual topology differs")

        def entry_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = originals[ROOT_RESIDUAL_ENTRY_RELU](*args, **kwargs)
            if str(getattr(kwargs.get("start_node"), "name", "")) != ROOT_START_NODE:
                return result
            if self._pending is not None or len(args) < 3 or args[1] is not None:
                raise ValueError("root CROWN residual entry order differs")
            incoming = _tensor(args[0], name="residual incoming A")
            preactivation = args[2]
            lower = _tensor(preactivation.lower, name="residual entry lower")
            upper = _tensor(preactivation.upper, name="residual entry upper")
            alpha = self._alpha(entry)
            for value in (incoming, lower, upper, alpha):
                _retain(value)
            self._pending = RootCrownResidualEvaluationCaptureV1(
                ordinal=len(self.evaluations),
                incoming_lower_a=_cpu_clone(incoming),
                entry_lower=_cpu_clone(lower),
                entry_upper=_cpu_clone(upper),
                entry_raw_alpha=_cpu_clone(alpha),
                entry_alpha_feature_indices=self._indices(entry),
                _live_incoming_lower_a=incoming,
                _live_entry_lower=lower,
                _live_entry_upper=upper,
                _live_entry_raw_alpha=alpha,
            )
            self._record_bias(result)
            self.node_call_counts[ROOT_RESIDUAL_ENTRY_RELU] += 1
            return result

        def add_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = originals[ROOT_RESIDUAL_ADD](*args, **kwargs)
            if self._pending is None or self._pending._stage != 1:
                return result
            if len(result[0]) != 2:
                raise ValueError("root CROWN residual Add fanout differs")
            self._pending._stage = 2
            self._record_bias(result)
            self.node_call_counts[ROOT_RESIDUAL_ADD] += 1
            return result

        def main_conv_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = originals[ROOT_RESIDUAL_MAIN_CONV](*args, **kwargs)
            if self._pending is None or self._pending._stage != 2 or len(args) < 5:
                return result
            self._pending.main_conv_weight = _cpu_clone(
                _tensor(args[3].lower, name="residual main weight")
            )
            self._pending.main_conv_bias = _cpu_clone(
                _tensor(args[4].lower, name="residual main bias")
            )
            self._pending._stage = 3
            self._record_bias(result)
            self.node_call_counts[ROOT_RESIDUAL_MAIN_CONV] += 1
            return result

        def inner_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = originals[ROOT_RESIDUAL_INNER_RELU](*args, **kwargs)
            if self._pending is None or self._pending._stage != 3 or len(args) < 3:
                return result
            preactivation = args[2]
            lower = _tensor(preactivation.lower, name="residual inner lower")
            upper = _tensor(preactivation.upper, name="residual inner upper")
            alpha = self._alpha(inner)
            for value in (lower, upper, alpha):
                _retain(value)
            self._pending.inner_lower = _cpu_clone(lower)
            self._pending.inner_upper = _cpu_clone(upper)
            self._pending.inner_raw_alpha = _cpu_clone(alpha)
            self._pending.inner_alpha_feature_indices = self._indices(inner)
            self._pending._live_inner_lower = lower
            self._pending._live_inner_upper = upper
            self._pending._live_inner_raw_alpha = alpha
            self._pending._stage = 4
            self._record_bias(result)
            self.node_call_counts[ROOT_RESIDUAL_INNER_RELU] += 1
            return result

        def inner_conv_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = originals[ROOT_RESIDUAL_INNER_CONV](*args, **kwargs)
            if self._pending is None or self._pending._stage != 4 or len(args) < 5:
                return result
            self._pending.inner_conv_weight = _cpu_clone(
                _tensor(args[3].lower, name="residual inner weight")
            )
            self._pending.inner_conv_bias = _cpu_clone(
                _tensor(args[4].lower, name="residual inner bias")
            )
            self._pending._stage = 5
            self._record_bias(result)
            self.node_call_counts[ROOT_RESIDUAL_INNER_CONV] += 1
            return result

        def exit_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            if self._pending is not None and self._pending._stage == 5:
                if len(args) < 1 or args[1] is not None:
                    raise ValueError("root CROWN residual exit ABI differs")
                output_a = _tensor(args[0], name="residual output A")
                _retain(output_a)
                self._pending.output_lower_a = _cpu_clone(output_a)
                self._pending._live_output_lower_a = output_a
                if not self._pending._bias_parts:
                    raise ValueError("root CROWN residual bias evidence is absent")
                self._pending.output_bias = _cpu_clone(
                    torch.stack(self._pending._bias_parts).sum(dim=0)
                )
                self._pending._stage = 6
                self.evaluations.append(self._pending)
                self._pending = None
                self.node_call_counts[ROOT_RESIDUAL_EXIT_RELU] += 1
            return originals[ROOT_RESIDUAL_EXIT_RELU](*args, **kwargs)

        replacements = {
            ROOT_RESIDUAL_ENTRY_RELU: entry_wrapped,
            ROOT_RESIDUAL_ADD: add_wrapped,
            ROOT_RESIDUAL_MAIN_CONV: main_conv_wrapped,
            ROOT_RESIDUAL_INNER_RELU: inner_wrapped,
            ROOT_RESIDUAL_INNER_CONV: inner_conv_wrapped,
            ROOT_RESIDUAL_EXIT_RELU: exit_wrapped,
        }
        for name, replacement in replacements.items():
            nodes[name].bound_backward = MethodType(replacement, nodes[name])
        try:
            yield
        finally:
            for name, original in originals.items():
                nodes[name].bound_backward = original

    @contextmanager
    def _install_backward(self) -> Iterator[None]:
        original_backward = torch.autograd.backward

        def backward_wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original_backward(*args, **kwargs)
            candidates = [
                item
                for item in self.evaluations
                if item._live_output_lower_a is not None
                and item._live_output_lower_a.requires_grad
                and not item.backward_captured
            ]
            if len(candidates) != 1:
                raise ValueError("root CROWN residual VJP cardinality differs")
            capture = candidates[0]
            capture.output_lower_a_gradient = _cpu_clone(
                _gradient(capture._live_output_lower_a, name="output A")
            )
            bias_gradients = [
                _tensor(value.grad, name="residual bias gradient")
                for value in capture._bias_parts
                if value.requires_grad
            ]
            if not bias_gradients or any(
                not torch.equal(bias_gradients[0], value)
                for value in bias_gradients[1:]
            ):
                raise ValueError("root CROWN residual bias adjoint differs")
            capture.output_bias_gradient = _cpu_clone(bias_gradients[0])
            for destination, live, name in (
                (
                    "incoming_lower_a_gradient",
                    capture._live_incoming_lower_a,
                    "input A",
                ),
                ("entry_lower_gradient", capture._live_entry_lower, "entry lower"),
                ("entry_upper_gradient", capture._live_entry_upper, "entry upper"),
                (
                    "entry_raw_alpha_gradient",
                    capture._live_entry_raw_alpha,
                    "entry alpha",
                ),
                ("inner_lower_gradient", capture._live_inner_lower, "inner lower"),
                ("inner_upper_gradient", capture._live_inner_upper, "inner upper"),
                (
                    "inner_raw_alpha_gradient",
                    capture._live_inner_raw_alpha,
                    "inner alpha",
                ),
            ):
                setattr(capture, destination, _cpu_clone(_gradient(live, name=name)))
            capture._bias_parts.clear()
            self.backward_call_count += 1
            return result

        torch.autograd.backward = backward_wrapped
        try:
            yield
        finally:
            torch.autograd.backward = original_backward

    def validate(self) -> None:
        """Require a complete five-forward/four-backward production trace."""

        if (
            self.outer_call_count != 1
            or self.backward_call_count != 4
            or len(self.evaluations) != 5
            or any(value != 5 for value in self.node_call_counts.values())
            or self.device_before != self.device_after
            or self.stream_before != self.stream_after
        ):
            raise ValueError("root CROWN residual capture count/context differs")
        for ordinal, capture in enumerate(self.evaluations):
            if (
                capture.ordinal != ordinal
                or capture._stage != 6
                or capture.backward_captured is not (ordinal < 4)
            ):
                raise ValueError("root CROWN residual capture order differs")
            for name, value in capture.tensor_payload().items():
                if name == "ordinal" or value is None or isinstance(value, tuple):
                    continue
                tensor = _tensor(value, name=name)
                if tensor.device.type != "cpu" or not tensor.is_contiguous():
                    raise ValueError(f"root CROWN residual tensor differs: {name}")

    def shape_receipt(self) -> dict[str, object]:
        """Return a compact value-free shape receipt."""

        self.validate()
        first = self.evaluations[0]
        shapes: dict[str, object] = {
            name: list(value.shape)
            for name, value in first.tensor_payload().items()
            if torch.is_tensor(value)
        }
        shapes["entry_alpha_feature_indices"] = [
            list(value.shape) for value in first.entry_alpha_feature_indices
        ]
        shapes["inner_alpha_feature_indices"] = [
            list(value.shape) for value in first.inner_alpha_feature_indices
        ]
        return {
            "schema_version": "boundflow.root-crown-residual-capture/v1",
            "start_node": ROOT_START_NODE,
            "topology": [
                ROOT_RESIDUAL_ENTRY_RELU,
                ROOT_RESIDUAL_ADD,
                ROOT_RESIDUAL_MAIN_CONV,
                ROOT_RESIDUAL_INNER_RELU,
                ROOT_RESIDUAL_INNER_CONV,
                ROOT_RESIDUAL_EXIT_RELU,
            ],
            "outer_call_count": self.outer_call_count,
            "forward_count": len(self.evaluations),
            "backward_count": self.backward_call_count,
            "node_call_counts": self.node_call_counts,
            "shapes": shapes,
            "performance_claimed": False,
        }


__all__ = [
    "RootCrownResidualCaptureV1",
    "RootCrownResidualEvaluationCaptureV1",
]
