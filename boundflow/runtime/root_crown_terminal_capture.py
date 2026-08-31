"""Capture the native root CROWN terminal ReLU/Linear transaction.

The capture is deliberately diagnostic: it observes the real auto_LiRPA
objects without replacing an operator and records no performance result.
"""

# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=protected-access,too-many-locals,too-many-statements
# pylint: disable=too-many-boolean-expressions,duplicate-code

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Iterator, Mapping

import torch

ROOT_START_NODE = "/49"
ROOT_TERMINAL_RELU = "/48"
ROOT_TERMINAL_LINEAR = "/input-28"


def _tensor(value: Any, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"root CROWN {name} is not a tensor")
    return value


def _lower_a(result: Any) -> torch.Tensor:
    try:
        return _tensor(result[0][0][0], name="lower A")
    except (IndexError, TypeError) as error:
        raise ValueError("root CROWN bound_backward result differs") from error


def _lower_bias(result: Any) -> torch.Tensor:
    try:
        return _tensor(result[1], name="lower bias")
    except (IndexError, TypeError) as error:
        raise ValueError("root CROWN bound_backward bias differs") from error


def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous().clone()


@dataclass
class RootCrownTerminalEvaluationCaptureV1:
    """One native optimizer evaluation and its optional VJP evidence."""

    ordinal: int
    incoming_lower_a: torch.Tensor
    preactivation_lower: torch.Tensor
    preactivation_upper: torch.Tensor
    raw_alpha: torch.Tensor
    selected_lower_alpha: torch.Tensor
    start_spec_indices: torch.Tensor
    alpha_feature_indices: tuple[torch.Tensor, ...]
    relu_output_lower_a: torch.Tensor
    relu_lower_bias: torch.Tensor
    operator_weight: torch.Tensor | None = None
    operator_bias: torch.Tensor | None = None
    output_lower_a: torch.Tensor | None = None
    linear_lower_bias: torch.Tensor | None = None
    output_lower_a_gradient: torch.Tensor | None = None
    output_bias_gradient: torch.Tensor | None = None
    raw_alpha_gradient: torch.Tensor | None = None
    preactivation_lower_gradient: torch.Tensor | None = None
    preactivation_upper_gradient: torch.Tensor | None = None
    _live_output_lower_a: torch.Tensor | None = field(
        default=None, repr=False, compare=False
    )
    _live_relu_lower_bias: torch.Tensor | None = field(
        default=None, repr=False, compare=False
    )
    _live_linear_lower_bias: torch.Tensor | None = field(
        default=None, repr=False, compare=False
    )
    _live_raw_alpha: torch.Tensor | None = field(
        default=None, repr=False, compare=False
    )
    _live_preactivation_lower: torch.Tensor | None = field(
        default=None, repr=False, compare=False
    )
    _live_preactivation_upper: torch.Tensor | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def backward_captured(self) -> bool:
        return self.raw_alpha_gradient is not None

    def tensor_payload(self) -> dict[str, object]:
        """Return CPU tensors suitable for a local torch.save diagnostic."""

        required = {
            "incoming_lower_a": self.incoming_lower_a,
            "preactivation_lower": self.preactivation_lower,
            "preactivation_upper": self.preactivation_upper,
            "raw_alpha": self.raw_alpha,
            "selected_lower_alpha": self.selected_lower_alpha,
            "start_spec_indices": self.start_spec_indices,
            "relu_output_lower_a": self.relu_output_lower_a,
            "relu_lower_bias": self.relu_lower_bias,
            "operator_weight": self.operator_weight,
            "operator_bias": self.operator_bias,
            "output_lower_a": self.output_lower_a,
            "linear_lower_bias": self.linear_lower_bias,
        }
        if any(value is None for value in required.values()):
            raise ValueError("root CROWN terminal capture is incomplete")
        payload: dict[str, object] = {
            "ordinal": self.ordinal,
            **required,
            "alpha_feature_indices": self.alpha_feature_indices,
            "output_lower_a_gradient": self.output_lower_a_gradient,
            "output_bias_gradient": self.output_bias_gradient,
            "raw_alpha_gradient": self.raw_alpha_gradient,
            "preactivation_lower_gradient": self.preactivation_lower_gradient,
            "preactivation_upper_gradient": self.preactivation_upper_gradient,
        }
        return payload


class RootCrownTerminalCaptureV1:
    """Observe exactly one five-evaluation root alpha-CROWN transaction."""

    def __init__(self) -> None:
        self.evaluations: list[RootCrownTerminalEvaluationCaptureV1] = []
        self.outer_call_count = 0
        self.relu_call_count = 0
        self.linear_call_count = 0
        self.backward_call_count = 0
        self._pending: RootCrownTerminalEvaluationCaptureV1 | None = None
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
            and ROOT_TERMINAL_RELU in names
            and ROOT_TERMINAL_LINEAR in names
        )

    @contextmanager
    def install(self, bounded_module_type: type[Any]) -> Iterator[None]:
        """Patch only the eligible outer optimizer call and restore on exit."""

        original_optimized = bounded_module_type._get_optimized_bounds

        def optimized_wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                self._active
                or self.outer_call_count
                or not self._eligible(instance, kwargs)
            ):
                return original_optimized(instance, *args, **kwargs)
            self.outer_call_count += 1
            self._active = True
            self.device_before = int(torch.cuda.current_device())
            self.stream_before = int(torch.cuda.current_stream().cuda_stream)
            try:
                with self._install_nodes(instance), self._install_backward():
                    return original_optimized(instance, *args, **kwargs)
            finally:
                self.device_after = int(torch.cuda.current_device())
                self.stream_after = int(torch.cuda.current_stream().cuda_stream)
                self._active = False

        bounded_module_type._get_optimized_bounds = optimized_wrapped
        try:
            yield
        finally:
            bounded_module_type._get_optimized_bounds = original_optimized

    @contextmanager
    def _install_nodes(self, instance: Any) -> Iterator[None]:
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        relu = nodes[ROOT_TERMINAL_RELU]
        linear = nodes[ROOT_TERMINAL_LINEAR]
        relu_inputs = getattr(relu, "inputs", ())
        if not relu_inputs or relu_inputs[0] is not linear:
            raise ValueError("root CROWN terminal topology differs")
        original_relu = relu.bound_backward
        original_linear = linear.bound_backward

        def relu_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_relu(*args, **kwargs)
            start_node = kwargs.get("start_node")
            if str(getattr(start_node, "name", "")) != ROOT_START_NODE:
                return result
            if self._pending is not None:
                raise ValueError("root CROWN ReLU repeated before Linear")
            if len(args) < 3:
                raise ValueError("root CROWN ReLU arguments differ")
            incoming = _tensor(args[0], name="incoming lower A")
            preactivation = args[2]
            raw_alpha = getattr(relu, "alpha", {}).get(ROOT_START_NODE)
            raw_alpha = _tensor(raw_alpha, name="raw alpha")
            alpha_indices = getattr(relu, "alpha_indices", None)
            if not isinstance(alpha_indices, (tuple, list)) or not alpha_indices:
                raise ValueError("root CROWN alpha feature mapping differs")
            index_tensors = tuple(
                _cpu_clone(_tensor(value, name="alpha feature index"))
                for value in alpha_indices
            )
            unstable_idx = kwargs.get("unstable_idx")
            if unstable_idx is None:
                start_spec_indices = torch.arange(
                    raw_alpha.shape[1], device=raw_alpha.device, dtype=torch.int64
                )
            else:
                start_spec_indices = _tensor(
                    unstable_idx, name="start specification index"
                )
                if start_spec_indices.ndim != 1:
                    raise ValueError("root CROWN start specification mapping differs")
            selected_alpha, _lookup = relu.select_alpha_by_idx(
                args[0], args[1], unstable_idx, start_node
            )
            selected_alpha = _tensor(selected_alpha, name="selected alpha")
            output_a = _lower_a(result)
            output_bias = _lower_bias(result)
            lower = _tensor(preactivation.lower, name="preactivation lower")
            upper = _tensor(preactivation.upper, name="preactivation upper")
            if output_a.requires_grad:
                output_a.retain_grad()
            if output_bias.requires_grad:
                output_bias.retain_grad()
            if lower.requires_grad:
                lower.retain_grad()
            if upper.requires_grad:
                upper.retain_grad()
            capture = RootCrownTerminalEvaluationCaptureV1(
                ordinal=len(self.evaluations),
                incoming_lower_a=_cpu_clone(incoming),
                preactivation_lower=_cpu_clone(lower),
                preactivation_upper=_cpu_clone(upper),
                raw_alpha=_cpu_clone(raw_alpha),
                selected_lower_alpha=_cpu_clone(selected_alpha[0]),
                start_spec_indices=_cpu_clone(start_spec_indices),
                alpha_feature_indices=index_tensors,
                relu_output_lower_a=_cpu_clone(output_a),
                relu_lower_bias=_cpu_clone(output_bias),
                _live_relu_lower_bias=output_bias,
                _live_raw_alpha=raw_alpha,
                _live_preactivation_lower=lower,
                _live_preactivation_upper=upper,
            )
            self._pending = capture
            self.relu_call_count += 1
            return result

        def linear_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            if self._pending is None:
                return original_linear(*args, **kwargs)
            start_node = kwargs.get("start_node")
            if str(getattr(start_node, "name", "")) != ROOT_START_NODE:
                raise ValueError("root CROWN Linear start node differs")
            if len(args) < 5:
                raise ValueError("root CROWN Linear arguments differ")
            incoming = _tensor(args[0], name="Linear incoming lower A")
            if not torch.equal(
                incoming, self._pending.relu_output_lower_a.to(incoming)
            ):
                raise ValueError("root CROWN ReLU-to-Linear value differs")
            result = original_linear(*args, **kwargs)
            output_a = _lower_a(result)
            output_bias = _lower_bias(result)
            if output_a.requires_grad:
                output_a.retain_grad()
            if output_bias.requires_grad:
                output_bias.retain_grad()
            self._pending.operator_weight = _cpu_clone(
                _tensor(args[3].lower, name="operator weight")
            )
            self._pending.operator_bias = _cpu_clone(
                _tensor(args[4].lower, name="operator bias")
            )
            self._pending.output_lower_a = _cpu_clone(output_a)
            self._pending.linear_lower_bias = _cpu_clone(output_bias)
            self._pending._live_output_lower_a = output_a
            self._pending._live_linear_lower_bias = output_bias
            self.evaluations.append(self._pending)
            self._pending = None
            self.linear_call_count += 1
            return result

        relu.bound_backward = MethodType(relu_wrapped, relu)
        linear.bound_backward = MethodType(linear_wrapped, linear)
        try:
            yield
        finally:
            relu.bound_backward = original_relu
            linear.bound_backward = original_linear
            if self._pending is not None:
                raise ValueError("root CROWN terminal capture remained partial")

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
                raise ValueError("root CROWN backward capture cardinality differs")
            capture = candidates[0]
            output_a_gradient = _tensor(
                _tensor(capture._live_output_lower_a, name="live output A").grad,
                name="output A gradient",
            )
            relu_bias_gradient = _tensor(
                _tensor(capture._live_relu_lower_bias, name="live ReLU bias").grad,
                name="ReLU bias gradient",
            )
            linear_bias_gradient = _tensor(
                _tensor(capture._live_linear_lower_bias, name="live Linear bias").grad,
                name="Linear bias gradient",
            )
            raw_alpha_gradient = _tensor(
                _tensor(capture._live_raw_alpha, name="live alpha").grad,
                name="alpha gradient",
            )
            lower_gradient = _tensor(
                _tensor(
                    capture._live_preactivation_lower, name="live lower bound"
                ).grad,
                name="lower-bound gradient",
            )
            upper_gradient = _tensor(
                _tensor(
                    capture._live_preactivation_upper, name="live upper bound"
                ).grad,
                name="upper-bound gradient",
            )
            if not torch.equal(relu_bias_gradient, linear_bias_gradient):
                raise ValueError("root CROWN accumulated bias adjoint differs")
            capture.output_lower_a_gradient = _cpu_clone(output_a_gradient)
            capture.output_bias_gradient = _cpu_clone(relu_bias_gradient)
            capture.raw_alpha_gradient = _cpu_clone(raw_alpha_gradient)
            capture.preactivation_lower_gradient = _cpu_clone(lower_gradient)
            capture.preactivation_upper_gradient = _cpu_clone(upper_gradient)
            self.backward_call_count += 1
            return result

        torch.autograd.backward = backward_wrapped
        try:
            yield
        finally:
            torch.autograd.backward = original_backward

    def validate(self) -> None:
        """Validate the frozen five-forward/four-backward root contract."""

        if (
            self.outer_call_count != 1
            or self.relu_call_count != 5
            or self.linear_call_count != 5
            or self.backward_call_count != 4
            or len(self.evaluations) != 5
            or self.device_before != self.device_after
            or self.stream_before != self.stream_after
        ):
            raise ValueError("root CROWN terminal capture count/context differs")
        for ordinal, capture in enumerate(self.evaluations):
            payload = capture.tensor_payload()
            if capture.ordinal != ordinal:
                raise ValueError("root CROWN evaluation order differs")
            has_backward = capture.backward_captured
            if has_backward is not (ordinal < 4):
                raise ValueError("root CROWN backward order differs")
            for name, value in payload.items():
                if name in {"ordinal", "alpha_feature_indices"} or value is None:
                    continue
                tensor = _tensor(value, name=name)
                if tensor.device.type != "cpu" or not tensor.is_contiguous():
                    raise ValueError(f"root CROWN capture tensor differs: {name}")

    def shape_receipt(self) -> dict[str, object]:
        """Return a compact, value-free receipt for the diagnostic run."""

        self.validate()
        first = self.evaluations[0].tensor_payload()
        shapes: dict[str, object] = {
            name: list(value.shape)
            for name, value in first.items()
            if torch.is_tensor(value)
        }
        shapes["alpha_feature_indices"] = [
            list(value.shape) for value in self.evaluations[0].alpha_feature_indices
        ]
        return {
            "schema_version": "boundflow.root-crown-terminal-capture/v1",
            "start_node": ROOT_START_NODE,
            "relu_node": ROOT_TERMINAL_RELU,
            "linear_node": ROOT_TERMINAL_LINEAR,
            "outer_call_count": self.outer_call_count,
            "forward_count": len(self.evaluations),
            "backward_count": self.backward_call_count,
            "shapes": shapes,
            "performance_claimed": False,
        }


__all__ = [
    "ROOT_START_NODE",
    "ROOT_TERMINAL_LINEAR",
    "ROOT_TERMINAL_RELU",
    "RootCrownTerminalCaptureV1",
    "RootCrownTerminalEvaluationCaptureV1",
]
