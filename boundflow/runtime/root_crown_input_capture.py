"""Capture the native root CROWN input Conv and L-infinity concretization."""

# pylint: disable=protected-access,too-many-instance-attributes,too-many-locals
# pylint: disable=too-many-statements,too-many-boolean-expressions

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Iterator, Mapping, cast

import torch

from boundflow.runtime.root_crown_residual_capture import _gradient, _retain
from boundflow.runtime.root_crown_terminal_capture import (
    ROOT_START_NODE,
    _cpu_clone,
    _lower_a,
    _lower_bias,
    _tensor,
)

ROOT_INPUT_RELU = "/input-4"
ROOT_INPUT_CONV = "/input"
ROOT_INPUT_NODE = "/input-1"


@dataclass
class RootCrownInputEvaluationCaptureV1:
    """One input-side ReLU/Conv/concretization evaluation and observed VJP."""

    ordinal: int
    incoming_lower_a: torch.Tensor
    preactivation_lower: torch.Tensor
    preactivation_upper: torch.Tensor
    raw_alpha: torch.Tensor
    alpha_feature_indices: tuple[torch.Tensor, ...]
    operator_weight: torch.Tensor | None = None
    operator_bias: torch.Tensor | None = None
    output_lower_a: torch.Tensor | None = None
    output_bias: torch.Tensor | None = None
    input_center: torch.Tensor | None = None
    input_lower: torch.Tensor | None = None
    input_upper: torch.Tensor | None = None
    concrete_lower: torch.Tensor | None = None
    concrete_sign: float | None = None
    output_lower_a_gradient: torch.Tensor | None = None
    output_bias_gradient: torch.Tensor | None = None
    concrete_lower_gradient: torch.Tensor | None = None
    incoming_lower_a_gradient: torch.Tensor | None = None
    preactivation_lower_gradient: torch.Tensor | None = None
    preactivation_upper_gradient: torch.Tensor | None = None
    raw_alpha_gradient: torch.Tensor | None = None
    _bias_parts: list[torch.Tensor] = field(default_factory=list)
    _live_incoming_lower_a: torch.Tensor | None = None
    _live_preactivation_lower: torch.Tensor | None = None
    _live_preactivation_upper: torch.Tensor | None = None
    _live_raw_alpha: torch.Tensor | None = None
    _live_output_lower_a: torch.Tensor | None = None
    _live_concrete_lower: torch.Tensor | None = None

    @property
    def backward_captured(self) -> bool:
        """Whether native autograd evidence was copied for this evaluation."""

        return self.output_lower_a_gradient is not None

    def tensor_payload(self) -> dict[str, object]:
        """Return CPU tensor evidence without live autograd references."""

        return {
            "ordinal": self.ordinal,
            "incoming_lower_a": self.incoming_lower_a,
            "preactivation_lower": self.preactivation_lower,
            "preactivation_upper": self.preactivation_upper,
            "raw_alpha": self.raw_alpha,
            "alpha_feature_indices": self.alpha_feature_indices,
            "operator_weight": self.operator_weight,
            "operator_bias": self.operator_bias,
            "output_lower_a": self.output_lower_a,
            "output_bias": self.output_bias,
            "input_center": self.input_center,
            "input_lower": self.input_lower,
            "input_upper": self.input_upper,
            "concrete_lower": self.concrete_lower,
            "concrete_sign": self.concrete_sign,
            "output_lower_a_gradient": self.output_lower_a_gradient,
            "output_bias_gradient": self.output_bias_gradient,
            "concrete_lower_gradient": self.concrete_lower_gradient,
            "incoming_lower_a_gradient": self.incoming_lower_a_gradient,
            "preactivation_lower_gradient": self.preactivation_lower_gradient,
            "preactivation_upper_gradient": self.preactivation_upper_gradient,
            "raw_alpha_gradient": self.raw_alpha_gradient,
        }


class RootCrownInputCaptureV1:
    """Observe one five-forward/four-backward root input-domain transaction."""

    def __init__(self) -> None:
        self.outer_call_count = 0
        self.relu_call_count = 0
        self.conv_call_count = 0
        self.concretize_call_count = 0
        self.backward_call_count = 0
        self.evaluations: list[RootCrownInputEvaluationCaptureV1] = []
        self._pending: RootCrownInputEvaluationCaptureV1 | None = None
        self._active = False
        self.device_before: int | None = None
        self.device_after: int | None = None
        self.stream_before: int | None = None
        self.stream_after: int | None = None

    @staticmethod
    def _eligible(instance: Any, kwargs: Mapping[str, Any]) -> bool:
        opts = getattr(instance, "bound_opts", {}).get("optimize_bound_args", {})
        names = {str(getattr(node, "name", "")) for node in instance.nodes()}
        return (
            kwargs.get("bound_side", "lower") == "lower"
            and int(opts.get("iteration", -1)) == 5
            and {ROOT_INPUT_RELU, ROOT_INPUT_CONV, ROOT_INPUT_NODE} <= names
        )

    @contextmanager
    def install(self, bounded_module_type: type[Any]) -> Iterator[None]:
        """Install capture only for the eligible production optimizer call."""

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
                with self._install_transaction(instance), self._install_backward():
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
                raise ValueError("root CROWN input capture remained partial")

    @staticmethod
    def _indices(node: Any) -> tuple[torch.Tensor, ...]:
        value = getattr(node, "alpha_indices", None)
        if (
            not isinstance(value, (tuple, list))
            or not value
            or not all(torch.is_tensor(item) for item in value)
        ):
            raise ValueError("root CROWN input alpha indices differ")
        return tuple(_cpu_clone(item) for item in value)

    @staticmethod
    def _alpha(node: Any) -> torch.Tensor:
        return _tensor(
            getattr(node, "alpha", {}).get(ROOT_START_NODE), name="input alpha"
        )

    @staticmethod
    def _sign(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> float:
        value = kwargs.get("sign", args[2] if len(args) > 2 else -1)
        return float(value)

    @contextmanager
    def _install_transaction(self, instance: Any) -> Iterator[None]:
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        relu = nodes[ROOT_INPUT_RELU]
        conv = nodes[ROOT_INPUT_CONV]
        input_node = nodes[ROOT_INPUT_NODE]
        if (
            not relu.inputs
            or relu.inputs[0] is not conv
            or not conv.inputs
            or conv.inputs[0] is not input_node
            or getattr(input_node, "perturbation", None) is None
        ):
            raise ValueError("root CROWN input topology differs")
        perturbation = input_node.perturbation
        original_relu = relu.bound_backward
        original_conv = conv.bound_backward
        original_concretize = perturbation.concretize

        def ours(kwargs: Mapping[str, Any]) -> bool:
            return str(getattr(kwargs.get("start_node"), "name", "")) == ROOT_START_NODE

        def relu_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_relu(*args, **kwargs)
            if not ours(kwargs):
                return result
            if self._pending is not None or len(args) < 3 or args[1] is not None:
                raise ValueError("root CROWN input ReLU order differs")
            incoming = _tensor(args[0], name="input incoming A")
            preactivation = args[2]
            lower = _tensor(preactivation.lower, name="input preactivation lower")
            upper = _tensor(preactivation.upper, name="input preactivation upper")
            alpha = self._alpha(relu)
            for value in (incoming, lower, upper, alpha):
                _retain(value)
            bias = _lower_bias(result)
            _retain(bias)
            self._pending = RootCrownInputEvaluationCaptureV1(
                ordinal=len(self.evaluations),
                incoming_lower_a=_cpu_clone(incoming),
                preactivation_lower=_cpu_clone(lower),
                preactivation_upper=_cpu_clone(upper),
                raw_alpha=_cpu_clone(alpha),
                alpha_feature_indices=self._indices(relu),
                _bias_parts=[bias],
                _live_incoming_lower_a=incoming,
                _live_preactivation_lower=lower,
                _live_preactivation_upper=upper,
                _live_raw_alpha=alpha,
            )
            self.relu_call_count += 1
            return result

        def conv_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_conv(*args, **kwargs)
            if not ours(kwargs):
                return result
            pending = self._pending
            if pending is None or pending.output_lower_a is not None or len(args) < 5:
                raise ValueError("root CROWN input Conv order differs")
            output_a = _lower_a(result)
            bias = _lower_bias(result)
            for value in (output_a, bias):
                _retain(value)
            pending.operator_weight = _cpu_clone(
                _tensor(args[3].lower, name="input Conv weight")
            )
            pending.operator_bias = _cpu_clone(
                _tensor(args[4].lower, name="input Conv bias")
            )
            pending.output_lower_a = _cpu_clone(output_a)
            pending._live_output_lower_a = output_a
            pending._bias_parts.append(bias)
            pending.output_bias = _cpu_clone(torch.stack(pending._bias_parts).sum(0))
            self.conv_call_count += 1
            return result

        def concretize_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_concretize(*args, **kwargs)
            pending = self._pending
            if pending is None or pending.output_lower_a is None:
                return result
            if len(args) < 2:
                raise ValueError("root CROWN input concretize ABI differs")
            x = _tensor(args[0], name="input center")
            coefficient = _tensor(args[1], name="input concretize coefficient")
            sign = self._sign(args, kwargs)
            expected_numel = int(pending.output_lower_a.numel())
            if (
                sign != -1.0
                or coefficient.numel() != expected_numel
                or pending._live_output_lower_a is None
                or coefficient.data_ptr() != pending._live_output_lower_a.data_ptr()
                or not torch.is_tensor(result)
            ):
                return result
            x_lower, x_upper, _active_x0, _active_eps = perturbation.get_input_bounds(
                x, coefficient
            )
            concrete = _tensor(result, name="input concrete lower")
            _retain(concrete)
            pending.input_center = _cpu_clone(x)
            pending.input_lower = _cpu_clone(_tensor(x_lower, name="input lower"))
            pending.input_upper = _cpu_clone(_tensor(x_upper, name="input upper"))
            pending.concrete_lower = _cpu_clone(concrete)
            pending.concrete_sign = sign
            pending._live_concrete_lower = concrete
            self.evaluations.append(pending)
            self._pending = None
            self.concretize_call_count += 1
            return result

        relu.bound_backward = MethodType(relu_wrapped, relu)
        conv.bound_backward = MethodType(conv_wrapped, conv)
        perturbation.concretize = MethodType(concretize_wrapped, perturbation)
        try:
            yield
        finally:
            relu.bound_backward = original_relu
            conv.bound_backward = original_conv
            perturbation.concretize = original_concretize

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
                raise ValueError("root CROWN input VJP cardinality differs")
            capture = candidates[0]
            bias_gradients = [
                _gradient(value, name="input bias")
                for value in capture._bias_parts
                if value.requires_grad
            ]
            if not bias_gradients or any(
                not torch.equal(bias_gradients[0], value)
                for value in bias_gradients[1:]
            ):
                raise ValueError("root CROWN input bias adjoint differs")
            capture.output_lower_a_gradient = _cpu_clone(
                _gradient(capture._live_output_lower_a, name="input output A")
            )
            capture.output_bias_gradient = _cpu_clone(bias_gradients[0])
            capture.concrete_lower_gradient = _cpu_clone(
                _gradient(capture._live_concrete_lower, name="concrete lower")
            )
            for destination, live, name in (
                (
                    "incoming_lower_a_gradient",
                    capture._live_incoming_lower_a,
                    "incoming A",
                ),
                (
                    "preactivation_lower_gradient",
                    capture._live_preactivation_lower,
                    "preactivation lower",
                ),
                (
                    "preactivation_upper_gradient",
                    capture._live_preactivation_upper,
                    "preactivation upper",
                ),
                ("raw_alpha_gradient", capture._live_raw_alpha, "raw alpha"),
            ):
                tensor = _tensor(live, name=f"live {name}")
                setattr(
                    capture,
                    destination,
                    (
                        _cpu_clone(_gradient(tensor, name=name))
                        if tensor.requires_grad
                        else None
                    ),
                )
            capture._bias_parts.clear()
            self.backward_call_count += 1
            return result

        torch.autograd.backward = backward_wrapped
        try:
            yield
        finally:
            torch.autograd.backward = original_backward

    def validate(self) -> None:
        """Require the complete five-forward/four-backward input transaction."""

        if (
            self.outer_call_count != 1
            or self.relu_call_count != 5
            or self.conv_call_count != 5
            or self.concretize_call_count != 5
            or self.backward_call_count != 4
            or len(self.evaluations) != 5
            or self.device_before != self.device_after
            or self.stream_before != self.stream_after
        ):
            raise ValueError("root CROWN input capture count/context differs")
        for ordinal, capture in enumerate(self.evaluations):
            payload = capture.tensor_payload()
            if (
                capture.ordinal != ordinal
                or capture.concrete_sign != -1.0
                or capture.backward_captured is not (ordinal < 4)
            ):
                raise ValueError("root CROWN input capture order differs")
            required = (
                "operator_weight",
                "operator_bias",
                "output_lower_a",
                "output_bias",
                "input_center",
                "input_lower",
                "input_upper",
                "concrete_lower",
            )
            if any(payload[name] is None for name in required):
                raise ValueError("root CROWN input tensor evidence is incomplete")
            for name, value in payload.items():
                if name in {"ordinal", "concrete_sign"} or value is None:
                    continue
                if isinstance(value, tuple):
                    continue
                tensor = _tensor(value, name=name)
                if tensor.device.type != "cpu" or not tensor.is_contiguous():
                    raise ValueError(f"root CROWN input tensor differs: {name}")

    def shape_receipt(self) -> dict[str, object]:
        """Return compact shape and lifecycle evidence."""

        self.validate()
        first = self.evaluations[0]
        shapes: dict[str, object] = {}
        for name, value in first.tensor_payload().items():
            if torch.is_tensor(value):
                shapes[name] = list(cast(torch.Tensor, value).shape)
        shapes["alpha_feature_indices"] = [
            list(value.shape) for value in first.alpha_feature_indices
        ]
        return {
            "schema_version": "boundflow.root-crown-input-capture/v1",
            "start_node": ROOT_START_NODE,
            "topology": [ROOT_INPUT_RELU, ROOT_INPUT_CONV, ROOT_INPUT_NODE],
            "outer_call_count": self.outer_call_count,
            "forward_count": len(self.evaluations),
            "backward_count": self.backward_call_count,
            "relu_call_count": self.relu_call_count,
            "conv_call_count": self.conv_call_count,
            "concretize_call_count": self.concretize_call_count,
            "shapes": shapes,
            "performance_claimed": False,
        }


__all__ = [
    "ROOT_INPUT_CONV",
    "ROOT_INPUT_NODE",
    "ROOT_INPUT_RELU",
    "RootCrownInputCaptureV1",
    "RootCrownInputEvaluationCaptureV1",
]
