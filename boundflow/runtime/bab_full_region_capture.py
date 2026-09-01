"""Versioned activation-BaB full-region capture without mutating root artifacts."""

# pylint: disable=protected-access,too-many-instance-attributes,too-many-locals
# pylint: disable=too-many-statements,too-many-boolean-expressions,duplicate-code

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterator, Mapping

import torch

from .root_crown_input_capture import RootCrownInputCaptureV1
from .root_crown_projection_capture import (
    _TOPOLOGY,
    RootCrownProjectionCaptureV1,
)
from .root_crown_residual_capture import (
    _gradient,
    RootCrownResidualCaptureV1,
)
from .root_crown_terminal_capture import (
    _cpu_clone,
    _lower_a,
    _lower_bias,
    _tensor,
    ROOT_START_NODE,
    ROOT_TERMINAL_LINEAR,
    ROOT_TERMINAL_RELU,
    RootCrownTerminalCaptureV1,
    RootCrownTerminalEvaluationCaptureV1,
)

BAB_OPTIMIZER_ITERATIONS = 10


@dataclass
class BabTerminalBetaEvidenceV1:
    """Active-beta injection and VJP evidence for one optimizer evaluation."""

    ordinal: int
    value: torch.Tensor
    location: torch.Tensor
    sign: torch.Tensor
    relu_output_lower_a: torch.Tensor
    linear_incoming_lower_a: torch.Tensor | None = None
    gradient: torch.Tensor | None = None
    _live_value: torch.Tensor | None = None

    def tensor_payload(self) -> dict[str, object]:
        """Return CPU evidence without requiring the final no-backward gradient."""

        if self.linear_incoming_lower_a is None:
            raise ValueError("activation-BaB beta injection capture is incomplete")
        return {
            "ordinal": self.ordinal,
            "value": self.value,
            "location": self.location,
            "sign": self.sign,
            "relu_output_lower_a": self.relu_output_lower_a,
            "linear_incoming_lower_a": self.linear_incoming_lower_a,
            "gradient": self.gradient,
        }


def _sparse_beta(preactivation: Any) -> Any:
    value = getattr(preactivation, "sparse_betas", None)
    if isinstance(value, Mapping):
        result = value.get(ROOT_START_NODE)
    elif isinstance(value, (tuple, list)) and len(value) == 1:
        result = value[0]
    else:
        result = None
    if result is None:
        raise ValueError("activation-BaB active beta is absent")
    return result


class BabTerminalCaptureV1(RootCrownTerminalCaptureV1):
    """Capture the terminal active-beta transaction for ten evaluations."""

    def __init__(self) -> None:
        super().__init__()
        self.beta_evidence: list[BabTerminalBetaEvidenceV1] = []
        self._pending_beta: BabTerminalBetaEvidenceV1 | None = None

    @staticmethod
    def _eligible(instance: Any, kwargs: Mapping[str, Any]) -> bool:
        opts = getattr(instance, "bound_opts", {}).get("optimize_bound_args", {})
        names = {str(getattr(node, "name", "")) for node in instance.nodes()}
        return (
            kwargs.get("bound_side", "lower") == "lower"
            and int(opts.get("iteration", -1)) == BAB_OPTIMIZER_ITERATIONS
            and ROOT_TERMINAL_RELU in names
            and ROOT_TERMINAL_LINEAR in names
        )

    @contextmanager
    def _install_nodes(self, instance: Any) -> Iterator[None]:
        nodes = {str(getattr(node, "name", "")): node for node in instance.nodes()}
        relu = nodes[ROOT_TERMINAL_RELU]
        linear = nodes[ROOT_TERMINAL_LINEAR]
        if not relu.inputs or relu.inputs[0] is not linear:
            raise ValueError("activation-BaB terminal topology differs")
        original_relu = relu.bound_backward
        original_linear = linear.bound_backward

        def relu_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_relu(*args, **kwargs)
            start_node = kwargs.get("start_node")
            if str(getattr(start_node, "name", "")) != ROOT_START_NODE:
                return result
            if (
                self._pending is not None
                or self._pending_beta is not None
                or len(args) < 3
            ):
                raise ValueError("activation-BaB terminal ReLU order differs")
            incoming = _tensor(args[0], name="incoming lower A")
            preactivation = args[2]
            raw_alpha = _tensor(
                getattr(relu, "alpha", {}).get(ROOT_START_NODE), name="raw alpha"
            )
            alpha_indices = getattr(relu, "alpha_indices", None)
            if not isinstance(alpha_indices, (tuple, list)) or not alpha_indices:
                raise ValueError("activation-BaB alpha feature mapping differs")
            index_tensors = tuple(
                _cpu_clone(_tensor(value, name="alpha feature index"))
                for value in alpha_indices
            )
            unstable_idx = kwargs.get("unstable_idx")
            start_spec_indices = (
                torch.arange(
                    raw_alpha.shape[1], device=raw_alpha.device, dtype=torch.int64
                )
                if unstable_idx is None
                else _tensor(unstable_idx, name="start specification index")
            )
            if start_spec_indices.ndim != 1:
                raise ValueError("activation-BaB start specification mapping differs")
            selected_alpha, _lookup = relu.select_alpha_by_idx(
                args[0], args[1], unstable_idx, start_node
            )
            selected_alpha = _tensor(selected_alpha, name="selected alpha")
            output_a = _lower_a(result)
            output_bias = _lower_bias(result)
            lower = _tensor(preactivation.lower, name="preactivation lower")
            upper = _tensor(preactivation.upper, name="preactivation upper")
            beta = _sparse_beta(linear)
            beta_value = _tensor(getattr(beta, "val", None), name="sparse beta")
            beta_location = _tensor(
                getattr(beta, "loc", None), name="sparse beta location"
            )
            beta_sign = _tensor(getattr(beta, "sign", None), name="sparse beta sign")
            if beta_value.numel() == 0 or not beta_value.requires_grad:
                raise ValueError("activation-BaB beta ownership differs")
            for value in (output_a, output_bias, raw_alpha, beta_value):
                if value.requires_grad:
                    value.retain_grad()
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
            beta_capture = BabTerminalBetaEvidenceV1(
                ordinal=capture.ordinal,
                value=_cpu_clone(beta_value),
                location=_cpu_clone(beta_location),
                sign=_cpu_clone(beta_sign),
                relu_output_lower_a=_cpu_clone(output_a),
                _live_value=beta_value,
            )
            self._pending = capture
            self._pending_beta = beta_capture
            self.relu_call_count += 1
            return result

        def linear_wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
            if self._pending is None:
                return original_linear(*args, **kwargs)
            if self._pending_beta is None or len(args) < 5:
                raise ValueError("activation-BaB Linear order differs")
            start_node = kwargs.get("start_node")
            if str(getattr(start_node, "name", "")) != ROOT_START_NODE:
                raise ValueError("activation-BaB Linear start node differs")
            incoming = _tensor(args[0], name="post-beta Linear incoming A")
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
            self._pending_beta.linear_incoming_lower_a = _cpu_clone(incoming)
            self.evaluations.append(self._pending)
            self.beta_evidence.append(self._pending_beta)
            self._pending = None
            self._pending_beta = None
            self.linear_call_count += 1
            return result

        relu.bound_backward = MethodType(relu_wrapped, relu)
        linear.bound_backward = MethodType(linear_wrapped, linear)
        try:
            yield
        finally:
            relu.bound_backward = original_relu
            linear.bound_backward = original_linear
            if self._pending is not None or self._pending_beta is not None:
                raise ValueError("activation-BaB terminal capture remained partial")

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
                raise ValueError("activation-BaB terminal VJP cardinality differs")
            capture = candidates[0]
            beta = self.beta_evidence[capture.ordinal]
            output_gradient = _tensor(
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
            if not torch.equal(relu_bias_gradient, linear_bias_gradient):
                raise ValueError("activation-BaB accumulated bias adjoint differs")
            capture.output_lower_a_gradient = _cpu_clone(output_gradient)
            capture.output_bias_gradient = _cpu_clone(relu_bias_gradient)
            capture.raw_alpha_gradient = _cpu_clone(
                _tensor(
                    _tensor(capture._live_raw_alpha, name="live alpha").grad,
                    name="alpha gradient",
                )
            )
            beta.gradient = _cpu_clone(
                _tensor(
                    _tensor(beta._live_value, name="live beta").grad,
                    name="beta gradient",
                )
            )
            self.backward_call_count += 1
            return result

        torch.autograd.backward = backward_wrapped
        try:
            yield
        finally:
            torch.autograd.backward = original_backward

    def validate(self) -> None:
        if (
            self.outer_call_count != 1
            or self.relu_call_count != BAB_OPTIMIZER_ITERATIONS
            or self.linear_call_count != BAB_OPTIMIZER_ITERATIONS
            or self.backward_call_count != BAB_OPTIMIZER_ITERATIONS - 1
            or len(self.evaluations) != BAB_OPTIMIZER_ITERATIONS
            or len(self.beta_evidence) != BAB_OPTIMIZER_ITERATIONS
            or self.device_before != self.device_after
            or self.stream_before != self.stream_after
        ):
            raise ValueError("activation-BaB terminal lifecycle differs")
        for ordinal, (capture, beta) in enumerate(
            zip(self.evaluations, self.beta_evidence)
        ):
            if (
                capture.ordinal != ordinal
                or beta.ordinal != ordinal
                or capture.backward_captured
                is not (ordinal < BAB_OPTIMIZER_ITERATIONS - 1)
                or (beta.gradient is not None)
                is not (ordinal < BAB_OPTIMIZER_ITERATIONS - 1)
            ):
                raise ValueError("activation-BaB terminal order differs")
            for payload in (capture.tensor_payload(), beta.tensor_payload()):
                for name, value in payload.items():
                    if name == "ordinal" or value is None or isinstance(value, tuple):
                        continue
                    tensor = _tensor(value, name=name)
                    if tensor.device.type != "cpu" or not tensor.is_contiguous():
                        raise ValueError(f"activation-BaB tensor differs: {name}")

    def shape_receipt(self) -> dict[str, object]:
        self.validate()
        receipt = super().shape_receipt()
        beta = self.beta_evidence[0].tensor_payload()
        shapes = receipt["shapes"]
        if not isinstance(shapes, dict):
            raise TypeError("activation-BaB terminal shape receipt differs")
        for name, value in beta.items():
            if torch.is_tensor(value):
                shapes[f"beta_{name}"] = list(value.shape)
        receipt.update(
            {
                "schema_version": "boundflow.activation-bab-terminal-capture/v1",
                "optimizer_iterations": BAB_OPTIMIZER_ITERATIONS,
                "active_beta_captured": True,
                "bound_gradients_required": False,
            }
        )
        return receipt


class _BabSegmentMixin:  # pylint: disable=too-few-public-methods
    """Shared ten-evaluation admission and optional-gradient capture helpers."""

    optimizer_iterations = BAB_OPTIMIZER_ITERATIONS

    @staticmethod
    def _eligible(instance: Any, kwargs: Mapping[str, Any]) -> bool:
        opts = getattr(instance, "bound_opts", {}).get("optimize_bound_args", {})
        return (
            kwargs.get("bound_side", "lower") == "lower"
            and int(opts.get("iteration", -1)) == BAB_OPTIMIZER_ITERATIONS
        )

    @staticmethod
    def _capture_gradient(
        live: object, *, name: str, required: bool
    ) -> torch.Tensor | None:
        tensor = _tensor(live, name=f"live {name}")
        gradient = tensor.grad if tensor.requires_grad else None
        if gradient is None and required:
            raise ValueError(f"activation-BaB {name} gradient is absent")
        return None if gradient is None else _cpu_clone(gradient)


class BabResidualCaptureV1(_BabSegmentMixin, RootCrownResidualCaptureV1):
    """Capture the residual segment with frozen intermediate bounds."""

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
                raise ValueError("activation-BaB residual VJP cardinality differs")
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
                raise ValueError("activation-BaB residual bias adjoint differs")
            capture.output_bias_gradient = _cpu_clone(bias_gradients[0])
            for destination, live, name, required in (
                (
                    "incoming_lower_a_gradient",
                    capture._live_incoming_lower_a,
                    "input A",
                    True,
                ),
                (
                    "entry_lower_gradient",
                    capture._live_entry_lower,
                    "entry lower",
                    False,
                ),
                (
                    "entry_upper_gradient",
                    capture._live_entry_upper,
                    "entry upper",
                    False,
                ),
                (
                    "entry_raw_alpha_gradient",
                    capture._live_entry_raw_alpha,
                    "entry alpha",
                    True,
                ),
                (
                    "inner_lower_gradient",
                    capture._live_inner_lower,
                    "inner lower",
                    False,
                ),
                (
                    "inner_upper_gradient",
                    capture._live_inner_upper,
                    "inner upper",
                    False,
                ),
                (
                    "inner_raw_alpha_gradient",
                    capture._live_inner_raw_alpha,
                    "inner alpha",
                    True,
                ),
            ):
                setattr(
                    capture,
                    destination,
                    self._capture_gradient(live, name=name, required=required),
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
        if (
            self.outer_call_count != 1
            or self.backward_call_count != BAB_OPTIMIZER_ITERATIONS - 1
            or len(self.evaluations) != BAB_OPTIMIZER_ITERATIONS
            or any(
                value != BAB_OPTIMIZER_ITERATIONS
                for value in self.node_call_counts.values()
            )
            or self.device_before != self.device_after
            or self.stream_before != self.stream_after
        ):
            raise ValueError("activation-BaB residual lifecycle differs")
        for ordinal, capture in enumerate(self.evaluations):
            if (
                capture.ordinal != ordinal
                or capture._stage != 6
                or capture.backward_captured
                is not (ordinal < BAB_OPTIMIZER_ITERATIONS - 1)
            ):
                raise ValueError("activation-BaB residual order differs")

    def shape_receipt(self) -> dict[str, object]:
        receipt = super().shape_receipt()
        receipt.update(
            {
                "schema_version": "boundflow.activation-bab-residual-capture/v1",
                "optimizer_iterations": BAB_OPTIMIZER_ITERATIONS,
                "bound_gradients_required": False,
            }
        )
        return receipt


class BabProjectionCaptureV1(_BabSegmentMixin, RootCrownProjectionCaptureV1):
    """Capture the projection segment with frozen intermediate bounds."""

    @staticmethod
    def _eligible(instance: Any, kwargs: Mapping[str, Any]) -> bool:
        names = {str(getattr(node, "name", "")) for node in instance.nodes()}
        return _BabSegmentMixin._eligible(instance, kwargs) and set(_TOPOLOGY) <= names

    _install_backward = BabResidualCaptureV1._install_backward

    def validate(self) -> None:
        if (
            self.outer_call_count != 1
            or self.backward_call_count != BAB_OPTIMIZER_ITERATIONS - 1
            or len(self.evaluations) != BAB_OPTIMIZER_ITERATIONS
            or any(
                value != BAB_OPTIMIZER_ITERATIONS
                for value in self.node_call_counts.values()
            )
            or self.device_before != self.device_after
            or self.stream_before != self.stream_after
        ):
            raise ValueError("activation-BaB projection lifecycle differs")
        for ordinal, capture in enumerate(self.evaluations):
            if (
                capture.ordinal != ordinal
                or capture._visited != set(_TOPOLOGY)
                or capture.backward_captured
                is not (ordinal < BAB_OPTIMIZER_ITERATIONS - 1)
            ):
                raise ValueError("activation-BaB projection order differs")

    def shape_receipt(self) -> dict[str, object]:
        receipt = super().shape_receipt()
        receipt.update(
            {
                "schema_version": "boundflow.activation-bab-projection-capture/v1",
                "optimizer_iterations": BAB_OPTIMIZER_ITERATIONS,
                "bound_gradients_required": False,
            }
        )
        return receipt


class BabInputCaptureV1(_BabSegmentMixin, RootCrownInputCaptureV1):
    """Capture the input-domain segment for the six-domain BaB batch."""

    def validate(self) -> None:
        if (
            self.outer_call_count != 1
            or self.relu_call_count != BAB_OPTIMIZER_ITERATIONS
            or self.conv_call_count != BAB_OPTIMIZER_ITERATIONS
            or self.concretize_call_count != BAB_OPTIMIZER_ITERATIONS
            or self.backward_call_count != BAB_OPTIMIZER_ITERATIONS - 1
            or len(self.evaluations) != BAB_OPTIMIZER_ITERATIONS
            or self.device_before != self.device_after
            or self.stream_before != self.stream_after
        ):
            raise ValueError("activation-BaB input lifecycle differs")
        for ordinal, capture in enumerate(self.evaluations):
            if (
                capture.ordinal != ordinal
                or capture.concrete_sign != -1.0
                or capture.backward_captured
                is not (ordinal < BAB_OPTIMIZER_ITERATIONS - 1)
            ):
                raise ValueError("activation-BaB input order differs")

    def shape_receipt(self) -> dict[str, object]:
        receipt = super().shape_receipt()
        receipt.update(
            {
                "schema_version": "boundflow.activation-bab-input-capture/v1",
                "optimizer_iterations": BAB_OPTIMIZER_ITERATIONS,
                "bound_gradients_required": False,
            }
        )
        return receipt


__all__ = [
    "BAB_OPTIMIZER_ITERATIONS",
    "BabInputCaptureV1",
    "BabProjectionCaptureV1",
    "BabResidualCaptureV1",
    "BabTerminalBetaEvidenceV1",
    "BabTerminalCaptureV1",
]
