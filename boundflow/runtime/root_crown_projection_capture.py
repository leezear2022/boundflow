"""Capture the native root CROWN projection-residual transaction and full VJP."""

# pylint: disable=protected-access,too-many-instance-attributes,too-many-locals
# pylint: disable=too-many-statements,too-many-boolean-expressions

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Iterator, Mapping

import torch

from boundflow.runtime.root_crown_residual_capture import _gradient, _retain
from boundflow.runtime.root_crown_terminal_capture import (
    ROOT_START_NODE,
    _cpu_clone,
    _tensor,
)

ROOT_PROJECTION_ENTRY_RELU = "/input-16"
ROOT_PROJECTION_ADD = "/39"
ROOT_PROJECTION_MAIN_OUTER_CONV = "/37"
ROOT_PROJECTION_INNER_RELU = "/input-12"
ROOT_PROJECTION_MAIN_INNER_CONV = "/input-8"
ROOT_PROJECTION_SKIP_CONV = "/38"
ROOT_PROJECTION_EXIT_RELU = "/input-4"

_TOPOLOGY = (
    ROOT_PROJECTION_ENTRY_RELU,
    ROOT_PROJECTION_ADD,
    ROOT_PROJECTION_MAIN_OUTER_CONV,
    ROOT_PROJECTION_INNER_RELU,
    ROOT_PROJECTION_MAIN_INNER_CONV,
    ROOT_PROJECTION_SKIP_CONV,
    ROOT_PROJECTION_EXIT_RELU,
)


@dataclass
class RootCrownProjectionEvaluationCaptureV1:
    """One native projection-residual evaluation and its observed VJP."""

    ordinal: int
    incoming_lower_a: torch.Tensor
    entry_lower: torch.Tensor
    entry_upper: torch.Tensor
    entry_raw_alpha: torch.Tensor
    entry_alpha_feature_indices: tuple[torch.Tensor, ...]
    main_outer_conv_weight: torch.Tensor | None = None
    main_outer_conv_bias: torch.Tensor | None = None
    inner_lower: torch.Tensor | None = None
    inner_upper: torch.Tensor | None = None
    inner_raw_alpha: torch.Tensor | None = None
    inner_alpha_feature_indices: tuple[torch.Tensor, ...] = ()
    main_inner_conv_weight: torch.Tensor | None = None
    main_inner_conv_bias: torch.Tensor | None = None
    skip_conv_weight: torch.Tensor | None = None
    skip_conv_bias: torch.Tensor | None = None
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
    _visited: set[str] = field(default_factory=set)
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
        """Return local production tensors without live autograd references."""

        return {
            "ordinal": self.ordinal,
            "incoming_lower_a": self.incoming_lower_a,
            "entry_lower": self.entry_lower,
            "entry_upper": self.entry_upper,
            "entry_raw_alpha": self.entry_raw_alpha,
            "entry_alpha_feature_indices": self.entry_alpha_feature_indices,
            "main_outer_conv_weight": self.main_outer_conv_weight,
            "main_outer_conv_bias": self.main_outer_conv_bias,
            "inner_lower": self.inner_lower,
            "inner_upper": self.inner_upper,
            "inner_raw_alpha": self.inner_raw_alpha,
            "inner_alpha_feature_indices": self.inner_alpha_feature_indices,
            "main_inner_conv_weight": self.main_inner_conv_weight,
            "main_inner_conv_bias": self.main_inner_conv_bias,
            "skip_conv_weight": self.skip_conv_weight,
            "skip_conv_bias": self.skip_conv_bias,
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


class RootCrownProjectionCaptureV1:
    """Observe one five-forward/four-backward root projection transaction."""

    def __init__(self) -> None:
        self.outer_call_count = 0
        self.backward_call_count = 0
        self.node_call_counts = {name: 0 for name in _TOPOLOGY}
        self.evaluations: list[RootCrownProjectionEvaluationCaptureV1] = []
        self._pending: RootCrownProjectionEvaluationCaptureV1 | None = None
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
            and set(_TOPOLOGY) <= names
        )

    @contextmanager
    def install(self, bounded_module_type: type[Any]) -> Iterator[None]:
        """Install capture hooks, restoring all global methods exactly."""

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
                raise ValueError("root CROWN projection capture remained partial")

    @staticmethod
    def _indices(node: Any) -> tuple[torch.Tensor, ...]:
        value = getattr(node, "alpha_indices", None)
        if (
            not isinstance(value, (tuple, list))
            or not value
            or not all(torch.is_tensor(item) for item in value)
        ):
            raise ValueError("root CROWN projection alpha indices differ")
        return tuple(_cpu_clone(item) for item in value)

    @staticmethod
    def _alpha(node: Any) -> torch.Tensor:
        return _tensor(
            getattr(node, "alpha", {}).get(ROOT_START_NODE), name="projection alpha"
        )

    def _record_bias(self, result: Any) -> None:
        pending = self._pending
        if pending is None:
            raise ValueError("root CROWN projection bias arrived without transaction")
        try:
            value = result[1]
        except (IndexError, TypeError) as error:
            raise ValueError("root CROWN projection bias result differs") from error
        if torch.is_tensor(value):
            _retain(value)
            pending._bias_parts.append(value)
        elif value not in (0, 0.0, None):
            raise ValueError("root CROWN projection non-tensor bias differs")

    @staticmethod
    def _conv_parameters(
        args: tuple[Any, ...], *, name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(args) < 5:
            raise ValueError(f"root CROWN projection {name} ABI differs")
        return (
            _cpu_clone(_tensor(args[3].lower, name=f"projection {name} weight")),
            _cpu_clone(_tensor(args[4].lower, name=f"projection {name} bias")),
        )

    @contextmanager
    def _install_nodes(self, instance: Any) -> Iterator[None]:
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        originals = {name: nodes[name].bound_backward for name in _TOPOLOGY}

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
            raise ValueError("root CROWN projection topology differs")

        def _ours(kwargs: Mapping[str, Any]) -> bool:
            return str(getattr(kwargs.get("start_node"), "name", "")) == ROOT_START_NODE

        def entry_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = originals[ROOT_PROJECTION_ENTRY_RELU](*args, **kwargs)
            if not _ours(kwargs):
                return result
            if self._pending is not None or len(args) < 3 or args[1] is not None:
                raise ValueError("root CROWN projection entry order differs")
            incoming = _tensor(args[0], name="projection incoming A")
            preactivation = args[2]
            lower = _tensor(preactivation.lower, name="projection entry lower")
            upper = _tensor(preactivation.upper, name="projection entry upper")
            alpha = self._alpha(entry)
            for value in (incoming, lower, upper, alpha):
                _retain(value)
            self._pending = RootCrownProjectionEvaluationCaptureV1(
                ordinal=len(self.evaluations),
                incoming_lower_a=_cpu_clone(incoming),
                entry_lower=_cpu_clone(lower),
                entry_upper=_cpu_clone(upper),
                entry_raw_alpha=_cpu_clone(alpha),
                entry_alpha_feature_indices=self._indices(entry),
                _visited={ROOT_PROJECTION_ENTRY_RELU},
                _live_incoming_lower_a=incoming,
                _live_entry_lower=lower,
                _live_entry_upper=upper,
                _live_entry_raw_alpha=alpha,
            )
            self._record_bias(result)
            self.node_call_counts[ROOT_PROJECTION_ENTRY_RELU] += 1
            return result

        def add_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = originals[ROOT_PROJECTION_ADD](*args, **kwargs)
            if not _ours(kwargs):
                return result
            pending = self._pending
            if pending is None or ROOT_PROJECTION_ADD in pending._visited:
                raise ValueError("root CROWN projection Add order differs")
            if len(result[0]) != 2:
                raise ValueError("root CROWN projection Add fanout differs")
            pending._visited.add(ROOT_PROJECTION_ADD)
            self._record_bias(result)
            self.node_call_counts[ROOT_PROJECTION_ADD] += 1
            return result

        def outer_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = originals[ROOT_PROJECTION_MAIN_OUTER_CONV](*args, **kwargs)
            if not _ours(kwargs):
                return result
            pending = self._pending
            if pending is None or ROOT_PROJECTION_ADD not in pending._visited:
                raise ValueError("root CROWN projection outer Conv order differs")
            pending.main_outer_conv_weight, pending.main_outer_conv_bias = (
                self._conv_parameters(args, name="outer Conv")
            )
            pending._visited.add(ROOT_PROJECTION_MAIN_OUTER_CONV)
            self._record_bias(result)
            self.node_call_counts[ROOT_PROJECTION_MAIN_OUTER_CONV] += 1
            return result

        def inner_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = originals[ROOT_PROJECTION_INNER_RELU](*args, **kwargs)
            if not _ours(kwargs):
                return result
            pending = self._pending
            if (
                pending is None
                or ROOT_PROJECTION_MAIN_OUTER_CONV not in pending._visited
                or len(args) < 3
            ):
                raise ValueError("root CROWN projection inner ReLU order differs")
            preactivation = args[2]
            lower = _tensor(preactivation.lower, name="projection inner lower")
            upper = _tensor(preactivation.upper, name="projection inner upper")
            alpha = self._alpha(inner)
            for value in (lower, upper, alpha):
                _retain(value)
            pending.inner_lower = _cpu_clone(lower)
            pending.inner_upper = _cpu_clone(upper)
            pending.inner_raw_alpha = _cpu_clone(alpha)
            pending.inner_alpha_feature_indices = self._indices(inner)
            pending._live_inner_lower = lower
            pending._live_inner_upper = upper
            pending._live_inner_raw_alpha = alpha
            pending._visited.add(ROOT_PROJECTION_INNER_RELU)
            self._record_bias(result)
            self.node_call_counts[ROOT_PROJECTION_INNER_RELU] += 1
            return result

        def inner_conv_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = originals[ROOT_PROJECTION_MAIN_INNER_CONV](*args, **kwargs)
            if not _ours(kwargs):
                return result
            pending = self._pending
            if pending is None or ROOT_PROJECTION_INNER_RELU not in pending._visited:
                raise ValueError("root CROWN projection inner Conv order differs")
            pending.main_inner_conv_weight, pending.main_inner_conv_bias = (
                self._conv_parameters(args, name="inner Conv")
            )
            pending._visited.add(ROOT_PROJECTION_MAIN_INNER_CONV)
            self._record_bias(result)
            self.node_call_counts[ROOT_PROJECTION_MAIN_INNER_CONV] += 1
            return result

        def skip_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = originals[ROOT_PROJECTION_SKIP_CONV](*args, **kwargs)
            if not _ours(kwargs):
                return result
            pending = self._pending
            if pending is None or ROOT_PROJECTION_ADD not in pending._visited:
                raise ValueError("root CROWN projection skip Conv order differs")
            pending.skip_conv_weight, pending.skip_conv_bias = self._conv_parameters(
                args, name="skip Conv"
            )
            pending._visited.add(ROOT_PROJECTION_SKIP_CONV)
            self._record_bias(result)
            self.node_call_counts[ROOT_PROJECTION_SKIP_CONV] += 1
            return result

        def exit_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            if _ours(kwargs):
                pending = self._pending
                required = set(_TOPOLOGY[:-1])
                if pending is None or pending._visited != required:
                    raise ValueError("root CROWN projection exit order differs")
                if len(args) < 2 or args[1] is not None:
                    raise ValueError("root CROWN projection exit ABI differs")
                output_a = _tensor(args[0], name="projection output A")
                _retain(output_a)
                pending.output_lower_a = _cpu_clone(output_a)
                pending._live_output_lower_a = output_a
                if not pending._bias_parts:
                    raise ValueError("root CROWN projection bias evidence is absent")
                pending.output_bias = _cpu_clone(
                    torch.stack(pending._bias_parts).sum(dim=0)
                )
                pending._visited.add(ROOT_PROJECTION_EXIT_RELU)
                self.evaluations.append(pending)
                self._pending = None
                self.node_call_counts[ROOT_PROJECTION_EXIT_RELU] += 1
            return originals[ROOT_PROJECTION_EXIT_RELU](*args, **kwargs)

        replacements = {
            ROOT_PROJECTION_ENTRY_RELU: entry_wrapped,
            ROOT_PROJECTION_ADD: add_wrapped,
            ROOT_PROJECTION_MAIN_OUTER_CONV: outer_wrapped,
            ROOT_PROJECTION_INNER_RELU: inner_wrapped,
            ROOT_PROJECTION_MAIN_INNER_CONV: inner_conv_wrapped,
            ROOT_PROJECTION_SKIP_CONV: skip_wrapped,
            ROOT_PROJECTION_EXIT_RELU: exit_wrapped,
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
                raise ValueError("root CROWN projection VJP cardinality differs")
            capture = candidates[0]
            capture.output_lower_a_gradient = _cpu_clone(
                _gradient(capture._live_output_lower_a, name="projection output A")
            )
            bias_gradients = [
                _tensor(value.grad, name="projection bias gradient")
                for value in capture._bias_parts
                if value.requires_grad
            ]
            if not bias_gradients or any(
                not torch.equal(bias_gradients[0], value)
                for value in bias_gradients[1:]
            ):
                raise ValueError("root CROWN projection bias adjoint differs")
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
            raise ValueError("root CROWN projection capture count/context differs")
        for ordinal, capture in enumerate(self.evaluations):
            if (
                capture.ordinal != ordinal
                or capture._visited != set(_TOPOLOGY)
                or capture.backward_captured is not (ordinal < 4)
            ):
                raise ValueError("root CROWN projection capture order differs")
            payload = capture.tensor_payload()
            required = set(payload) - {
                "ordinal",
                "output_lower_a_gradient",
                "output_bias_gradient",
                "incoming_lower_a_gradient",
                "entry_lower_gradient",
                "entry_upper_gradient",
                "entry_raw_alpha_gradient",
                "inner_lower_gradient",
                "inner_upper_gradient",
                "inner_raw_alpha_gradient",
            }
            for name, value in payload.items():
                if name == "ordinal" or isinstance(value, tuple):
                    continue
                if value is None:
                    if name in required or ordinal < 4:
                        raise ValueError(f"root CROWN projection tensor absent: {name}")
                    continue
                tensor = _tensor(value, name=name)
                if tensor.device.type != "cpu" or not tensor.is_contiguous():
                    raise ValueError(f"root CROWN projection tensor differs: {name}")

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
            "schema_version": "boundflow.root-crown-projection-capture/v1",
            "start_node": ROOT_START_NODE,
            "topology": list(_TOPOLOGY),
            "outer_call_count": self.outer_call_count,
            "forward_count": len(self.evaluations),
            "backward_count": self.backward_call_count,
            "node_call_counts": self.node_call_counts,
            "shapes": shapes,
            "performance_claimed": False,
        }


__all__ = [
    "RootCrownProjectionCaptureV1",
    "RootCrownProjectionEvaluationCaptureV1",
]
