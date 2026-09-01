"""Fail-closed P-anchor bridge for the real αβ-CROWN exact call."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,too-many-instance-attributes,too-many-arguments
# pylint: disable=missing-function-docstring,too-many-boolean-expressions
# pylint: disable=abstract-method,arguments-differ
# pylint: disable=too-many-locals

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from boundflow.backends.tvm.cibc_dense_exact_conv import (
    CompiledCIBCDenseExactConvTIRV3,
    compile_cibc_dense_exact_conv_tir_v3,
)

from .fsg4_b4b3_cibc_dense_tir import execute_cibc_dense_exact_tir_v3

TARGET_START = "/49"
TARGET_RELU = "/input-24"
TARGET_CONV = "/input-20"


@dataclass(frozen=True)
class MR3PAnchorBridgeReceiptV1:
    """Closed 10/9 launch and ownership ledger for one provider exact call."""

    evaluation_count: int
    forward_launch_count: int
    backward_launch_count: int
    empty_beta_tensor_count: int
    empty_beta_numel: int
    relu_conv_content_match_count: int
    relu_conv_pointer_match_count: int
    persistent_dense_a_count: int
    fallback_count: int
    eager_count: int
    native_shadow_count: int
    timing_recorded: bool
    performance_claimed: bool

    def validate(self) -> None:
        if (
            self.evaluation_count != 10
            or self.forward_launch_count != 10
            or self.backward_launch_count != 9
            or self.empty_beta_tensor_count != 10
            or self.empty_beta_numel != 0
            or self.relu_conv_content_match_count != 10
            or self.relu_conv_pointer_match_count not in range(11)
            or self.persistent_dense_a_count != 0
            or self.fallback_count != 0
            or self.eager_count != 0
            or self.native_shadow_count != 0
            or self.timing_recorded
            or self.performance_claimed
        ):
            raise ValueError("MR3 P-anchor bridge receipt differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return dict(self.__dict__)


def _content_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return tuple(left.shape) == tuple(right.shape) and torch.equal(left, right)


class MR3ProductionPAnchorBridgeV1:
    """Replace one provider ReLU+Conv lower path without owning the optimizer."""

    def __init__(
        self,
        *,
        compiled: CompiledCIBCDenseExactConvTIRV3 | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("MR3 production bridge requires CUDA")
        major, minor = torch.cuda.get_device_capability()
        self.compiled = compiled or compile_cibc_dense_exact_conv_tir_v3(
            compute_capability=f"sm_{major}{minor}"
        )
        self._evaluation_ordinal: int | None = None
        self._pending: dict[str, torch.Tensor] = {}
        self._combined_output: torch.Tensor | None = None
        self._combined_gradient: torch.Tensor | None = None
        self._evaluation_count = 0
        self._forward_count = 0
        self._backward_count = 0
        self._empty_beta_tensor_count = 0
        self._empty_beta_numel = 0
        self._content_match_count = 0
        self._pointer_match_count = 0
        self._fallback_count = 0
        self._eager_count = 0
        self._native_shadow_count = 0

    def begin_evaluation(self, ordinal: int) -> None:
        if (
            self._pending
            or ordinal != self._evaluation_count
            or ordinal not in range(10)
        ):
            self._fallback_count += 1
            raise ValueError("MR3 production bridge evaluation order differs")
        self._evaluation_ordinal = ordinal

    def _backward_completed(self) -> None:
        self._backward_count += 1

    def route_relu(
        self,
        relu: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        *,
        beta_tensors: tuple[torch.Tensor, ...],
    ) -> Any:
        start_node = kwargs.get("start_node")
        incoming = args[0] if args else None
        preactivation = args[2] if len(args) >= 3 else None
        alpha = getattr(relu, "alpha", {}).get(TARGET_START)
        indices = getattr(relu, "alpha_indices", None)
        if (
            self._evaluation_ordinal is None
            or str(getattr(start_node, "name", "")) != TARGET_START
            or not torch.is_tensor(incoming)
            or tuple(incoming.shape) != (1, 6, 16, 8, 8)
            or args[1] is not None
            or preactivation is None
            or tuple(preactivation.lower.shape) != (6, 16, 8, 8)
            or tuple(preactivation.upper.shape) != (6, 16, 8, 8)
            or not torch.is_tensor(alpha)
            or tuple(alpha.shape) != (2, 1, 6, 86)
            or not alpha.requires_grad
            or not isinstance(indices, list)
            or [tuple(item.shape) for item in indices] != [(86,), (86,), (86,)]
            or len(beta_tensors) != 1
            or tuple(beta_tensors[0].shape) != (6, 0)
            or beta_tensors[0].numel() != 0
            or self._pending
        ):
            self._fallback_count += 1
            raise ValueError("MR3 production bridge ReLU admission differs")
        compressed = alpha[0]
        full_alpha_with_spec = relu.reconstruct_full_alpha(
            compressed, (1, 6, 16, 8, 8), indices
        )
        full_alpha = full_alpha_with_spec[0].contiguous()
        expects_gradient = self._evaluation_ordinal < 9
        if (
            tuple(full_alpha.shape) != (6, 16, 8, 8)
            or torch.is_grad_enabled() != expects_gradient
            or full_alpha.requires_grad != expects_gradient
        ):
            self._fallback_count += 1
            raise ValueError("MR3 production bridge alpha reconstruction differs")
        bridge_handoff = incoming.detach()
        bridge_bias = torch.zeros(
            (1, 6), dtype=torch.float32, device=bridge_handoff.device
        )
        self._pending = {
            "incoming": incoming.permute(1, 0, 2, 3, 4).contiguous(),
            "lower": preactivation.lower.contiguous(),
            "upper": preactivation.upper.contiguous(),
            "full_alpha": full_alpha,
            "bridge_handoff": bridge_handoff,
        }
        self._empty_beta_tensor_count += 1
        self._empty_beta_numel += int(beta_tensors[0].numel())
        return [(bridge_handoff, None)], bridge_bias, 0

    def route_conv(
        self,
        args: tuple[Any, ...],
    ) -> Any:
        if self._evaluation_ordinal is None or len(args) < 5 or not self._pending:
            self._fallback_count += 1
            raise ValueError("MR3 production bridge Conv admission differs")
        provider_input = args[0]
        weight = args[3].lower
        bias = args[4].lower
        if (
            not torch.is_tensor(provider_input)
            or not _content_equal(provider_input, self._pending["bridge_handoff"])
            or tuple(weight.shape) != (16, 16, 3, 3)
            or tuple(bias.shape) != (16,)
        ):
            self._fallback_count += 1
            raise ValueError("MR3 production bridge ReLU-to-Conv handoff differs")
        self._content_match_count += 1
        self._pointer_match_count += int(
            provider_input.data_ptr() == self._pending["bridge_handoff"].data_ptr()
        )
        zero_bias = torch.zeros(
            (6, 1), dtype=torch.float32, device=provider_input.device
        )
        candidate_a, candidate_bias, executor = execute_cibc_dense_exact_tir_v3(
            incoming_lower_a=self._pending["incoming"],
            preactivation_lower=self._pending["lower"],
            preactivation_upper=self._pending["upper"],
            native_alpha=self._pending["full_alpha"],
            incoming_lower_bias=zero_bias,
            operator_weight=weight.contiguous(),
            operator_bias=bias.contiguous(),
            compiled=self.compiled,
            combined_output=self._combined_output,
            combined_gradient=self._combined_gradient,
            backward_observer=self._backward_completed,
        )
        if self._combined_output is None:
            self._combined_output = executor.combined_output
            self._combined_gradient = executor.combined_gradient
        if executor.forward_launch_count != 1:
            self._fallback_count += 1
            raise RuntimeError("MR3 production bridge forward launch differs")
        self._forward_count += 1
        routed_a = candidate_a.permute(1, 0, 2, 3, 4).contiguous()
        routed_bias = candidate_bias.transpose(0, 1).contiguous()
        if not torch.isfinite(routed_a).all() or not torch.isfinite(routed_bias).all():
            self._fallback_count += 1
            raise ValueError("MR3 production bridge candidate output is nonfinite")
        self._pending = {}
        self._evaluation_count += 1
        self._evaluation_ordinal = None
        return (
            [
                (routed_a, None),
                (None, None),
                (None, None),
            ],
            routed_bias,
            0,
        )

    def receipt(self) -> MR3PAnchorBridgeReceiptV1:
        receipt = MR3PAnchorBridgeReceiptV1(
            evaluation_count=self._evaluation_count,
            forward_launch_count=self._forward_count,
            backward_launch_count=self._backward_count,
            empty_beta_tensor_count=self._empty_beta_tensor_count,
            empty_beta_numel=self._empty_beta_numel,
            relu_conv_content_match_count=self._content_match_count,
            relu_conv_pointer_match_count=self._pointer_match_count,
            persistent_dense_a_count=int(bool(self._pending)),
            fallback_count=self._fallback_count,
            eager_count=self._eager_count,
            native_shadow_count=self._native_shadow_count,
            timing_recorded=False,
            performance_claimed=False,
        )
        receipt.validate()
        return receipt


__all__ = [
    "MR3PAnchorBridgeReceiptV1",
    "MR3ProductionPAnchorBridgeV1",
    "TARGET_CONV",
    "TARGET_RELU",
    "TARGET_START",
]
