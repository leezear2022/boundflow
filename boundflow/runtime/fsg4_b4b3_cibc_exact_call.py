"""B4-B3 production exact-call adapter for the CIBC manual TIR executor."""

# mypy: disable-error-code=import-untyped
# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
# pylint: disable=missing-function-docstring,import-error
# pylint: disable=abstract-method,arguments-differ,missing-class-docstring
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from boundflow.backends.tvm.cibc_dense_exact_conv import (
    CompiledCIBCDenseExactConvTIRV3,
    compile_cibc_dense_exact_conv_tir_v3,
)

from .fsg4_b4b1_reference_capture import (
    ProductionDifferentiableReferenceCaptureV1,
)
from .fsg4_b4b3_cibc_dense_tir import (
    CIBCDenseExactTIRExecutorV3,
    execute_cibc_dense_exact_tir_v3,
)
from .fsg4_b4b_production_region_capture import (
    B4B_PERFORMANCE_ANCHOR_V1,
    B4B_SEMANTIC_ANCHOR_V1,
    B4BRegionLiveObserverV1,
)


class _ExactValueCandidateGradient(torch.autograd.Function):
    """Preserve the native float32 trajectory while routing gradient to TIR."""

    @staticmethod
    def forward(ctx, native_value: torch.Tensor, candidate: torch.Tensor):
        del ctx, candidate
        return native_value

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        del ctx
        return None, gradient


@dataclass(frozen=True)
class B4B3CIBCExactCallReceiptV1:
    evaluation_count: int
    update_count: int
    provider_activation_count: int
    forward_launch_count: int
    backward_launch_count: int
    unsupported_semantic_anchor_count: int
    correctness_capture_enabled: bool
    native_value_bridge_count: int
    adjoint_materialization_count: int
    fallback_count: int
    eager_count: int
    module_hash: str
    exact_call: bool
    performance_claimed: bool = False

    def validate(self) -> None:
        if (
            self.evaluation_count != 10
            or self.update_count != 9
            or self.provider_activation_count != 10
            or self.forward_launch_count != 10
            or self.backward_launch_count != 9
            or self.unsupported_semantic_anchor_count
            != int(self.correctness_capture_enabled)
            or self.native_value_bridge_count != 10
            or self.adjoint_materialization_count != 0
            or self.fallback_count != 0
            or self.eager_count != 0
            or len(self.module_hash) != 64
            or not self.exact_call
            or self.performance_claimed
        ):
            raise ValueError("B4-B3 CIBC exact-call receipt differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return dict(self.__dict__)


class B4B3CIBCExactCallObserverV1:
    """Replace the P anchor in all evaluations and capture evaluation zero."""

    def __init__(
        self,
        reference_capture: ProductionDifferentiableReferenceCaptureV1,
        *,
        compiled: CompiledCIBCDenseExactConvTIRV3 | None = None,
        record_local_parity: bool = True,
        capture_evaluation_zero: bool = True,
    ) -> None:
        reference_capture.validate()
        if reference_capture.base.anchor != B4B_PERFORMANCE_ANCHOR_V1:
            raise ValueError("B4-B3 CIBC reference anchor differs")
        if not torch.cuda.is_available():
            raise RuntimeError("B4-B3 CIBC exact call requires CUDA")
        major, minor = torch.cuda.get_device_capability()
        compute_capability = f"sm_{major}{minor}"
        self.compiled = compiled or compile_cibc_dense_exact_conv_tir_v3(
            compute_capability=compute_capability
        )
        self._capture = B4BRegionLiveObserverV1()
        self._evaluation_ordinal: int | None = None
        self._native_alphas: Mapping[str, torch.Tensor] = {}
        self._pending: dict[str, torch.Tensor] = {}
        self._executors: list[CIBCDenseExactTIRExecutorV3] = []
        self._provider_activation_count = 0
        self._unsupported_semantic_anchor_count = 0
        self.local_parity: list[dict[str, float | bool]] = []
        self._record_local_parity = record_local_parity
        self._capture_enabled = capture_evaluation_zero
        self._reference_operator_attributes = dict(
            reference_capture.base.operator_attributes
        )

    @property
    def observations(self):
        return self._capture.observations

    def _validate_previous_evaluation(self) -> None:
        if not self._executors:
            return
        executor = self._executors[-1]
        ordinal = len(self._executors) - 1
        expected_backward = 1 if ordinal < 9 else 0
        if (
            executor.forward_launch_count != 1
            or executor.backward_launch_count != expected_backward
            or executor.fallback_count != 0
            or executor.eager_count != 0
        ):
            raise ValueError("B4-B3 CIBC previous evaluation differs")

    def begin_evaluation(
        self,
        evaluation_ordinal: int,
        *,
        native_alphas: Mapping[str, torch.Tensor],
        native_betas: Mapping[str, torch.Tensor],
        relu_pre_add_coeff_l: Mapping[str, torch.Tensor],
    ) -> None:
        if evaluation_ordinal != len(
            self._executors
        ) or evaluation_ordinal not in range(10):
            raise ValueError("B4-B3 CIBC evaluation order differs")
        self._validate_previous_evaluation()
        self._evaluation_ordinal = evaluation_ordinal
        self._native_alphas = native_alphas
        self._pending = {}
        if self._capture_enabled:
            self._capture.begin_evaluation(
                evaluation_ordinal,
                native_alphas=native_alphas,
                native_betas=native_betas,
                relu_pre_add_coeff_l=relu_pre_add_coeff_l,
            )

    def wants(self, native_preactivation: str) -> bool:
        if self._evaluation_ordinal is None:
            return False
        return (
            native_preactivation == B4B_PERFORMANCE_ANCHOR_V1.native_preactivation
            or (
                self._capture_enabled
                and self._evaluation_ordinal == 0
                and native_preactivation == B4B_SEMANTIC_ANCHOR_V1.native_preactivation
            )
        )

    def observe_relu_input(
        self,
        native_preactivation: str,
        *,
        incoming_lower_a: torch.Tensor,
        preactivation_lower: torch.Tensor,
        preactivation_upper: torch.Tensor,
        incoming_lower_bias: torch.Tensor,
    ) -> None:
        if not self.wants(native_preactivation):
            raise ValueError("B4-B3 CIBC received an ineligible ReLU")
        if self._evaluation_ordinal == 0 and self._capture_enabled:
            self._capture.observe_relu_input(
                native_preactivation,
                incoming_lower_a=incoming_lower_a,
                preactivation_lower=preactivation_lower,
                preactivation_upper=preactivation_upper,
                incoming_lower_bias=incoming_lower_bias,
            )
            incoming = self._capture.observed_incoming_lower_a(native_preactivation)
        else:
            incoming = incoming_lower_a.contiguous()
        if native_preactivation == B4B_PERFORMANCE_ANCHOR_V1.native_preactivation:
            if self._pending:
                raise ValueError("B4-B3 CIBC performance ReLU repeats")
            self._pending = {
                "incoming_lower_a": incoming,
                "preactivation_lower": preactivation_lower.contiguous(),
                "preactivation_upper": preactivation_upper.contiguous(),
                "incoming_lower_bias": incoming_lower_bias.contiguous(),
            }

    def observed_incoming_lower_a(self, native_preactivation: str) -> torch.Tensor:
        if native_preactivation == B4B_PERFORMANCE_ANCHOR_V1.native_preactivation:
            value = self._pending.get("incoming_lower_a")
            if value is None:
                raise ValueError("B4-B3 CIBC incoming lower A is unavailable")
            return value
        if not self._capture_enabled:
            raise ValueError("B4-B3 CIBC semantic capture is disabled")
        return self._capture.observed_incoming_lower_a(native_preactivation)

    def observe_affine_output(
        self,
        native_preactivation: str,
        *,
        operator_weight: torch.Tensor,
        operator_bias: torch.Tensor | None,
        output_lower_a: torch.Tensor,
        output_bias: torch.Tensor,
        operator_attributes: Mapping[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if native_preactivation == B4B_SEMANTIC_ANCHOR_V1.native_preactivation:
            if self._evaluation_ordinal != 0:
                raise ValueError("B4-B3 CIBC semantic anchor escaped evaluation zero")
            self._unsupported_semantic_anchor_count += 1
            self._capture.observe_affine_output(
                native_preactivation,
                operator_weight=operator_weight,
                operator_bias=operator_bias,
                output_lower_a=output_lower_a,
                output_bias=output_bias,
                operator_attributes=operator_attributes,
            )
            return None
        if (
            native_preactivation != B4B_PERFORMANCE_ANCHOR_V1.native_preactivation
            or operator_bias is None
            or dict(operator_attributes) != self._reference_operator_attributes
            or set(self._pending)
            != {
                "incoming_lower_a",
                "preactivation_lower",
                "preactivation_upper",
                "incoming_lower_bias",
            }
        ):
            raise ValueError("B4-B3 CIBC affine admission differs")
        native_alpha = self._native_alphas.get(native_preactivation)
        if (
            native_alpha is None
            or tuple(native_alpha.shape) != (6, 16, 8, 8)
            or not native_alpha.requires_grad
        ):
            raise ValueError("B4-B3 CIBC native alpha differs")
        candidate_a, candidate_bias, executor = execute_cibc_dense_exact_tir_v3(
            incoming_lower_a=self._pending["incoming_lower_a"],
            preactivation_lower=self._pending["preactivation_lower"],
            preactivation_upper=self._pending["preactivation_upper"],
            native_alpha=native_alpha,
            incoming_lower_bias=self._pending["incoming_lower_bias"],
            operator_weight=operator_weight.contiguous(),
            operator_bias=operator_bias.contiguous(),
            compiled=self.compiled,
        )
        self._executors.append(executor)
        self._provider_activation_count += 1
        if self._evaluation_ordinal == 0 and self._record_local_parity:
            self.local_parity.append(
                {
                    "output_a_max_abs_diff": float(
                        (output_lower_a - candidate_a).abs().max().item()
                    ),
                    "output_bias_max_abs_diff": float(
                        (output_bias - candidate_bias).abs().max().item()
                    ),
                    "output_a_sign_exact": bool(
                        torch.equal(torch.sign(output_lower_a), torch.sign(candidate_a))
                    ),
                    "output_bias_sign_exact": bool(
                        torch.equal(torch.sign(output_bias), torch.sign(candidate_bias))
                    ),
                }
            )
        routed_a = _ExactValueCandidateGradient.apply(output_lower_a, candidate_a)
        routed_bias = _ExactValueCandidateGradient.apply(output_bias, candidate_bias)
        if self._evaluation_ordinal == 0 and self._capture_enabled:
            self._capture.observe_affine_output(
                native_preactivation,
                operator_weight=operator_weight,
                operator_bias=operator_bias,
                output_lower_a=routed_a,
                output_bias=routed_bias,
                operator_attributes=operator_attributes,
            )
            return routed_a, routed_bias
        return routed_a, routed_bias

    def complete_evaluation(self, *, loss_seed: torch.Tensor) -> None:
        if self._evaluation_ordinal != 0:
            raise ValueError("B4-B3 CIBC capture closure differs")
        if self._capture_enabled:
            self._capture.complete_evaluation(loss_seed=loss_seed)

    def receipt(self) -> B4B3CIBCExactCallReceiptV1:
        self._validate_previous_evaluation()
        receipt = B4B3CIBCExactCallReceiptV1(
            evaluation_count=len(self._executors),
            update_count=sum(item.backward_launch_count for item in self._executors),
            provider_activation_count=self._provider_activation_count,
            forward_launch_count=sum(
                item.forward_launch_count for item in self._executors
            ),
            backward_launch_count=sum(
                item.backward_launch_count for item in self._executors
            ),
            unsupported_semantic_anchor_count=self._unsupported_semantic_anchor_count,
            correctness_capture_enabled=self._capture_enabled,
            native_value_bridge_count=len(self._executors),
            adjoint_materialization_count=sum(
                item.adjoint_materialization_count for item in self._executors
            ),
            fallback_count=sum(item.fallback_count for item in self._executors),
            eager_count=sum(item.eager_count for item in self._executors),
            module_hash=self.compiled.module_hash,
            exact_call=True,
        )
        receipt.validate()
        return receipt


__all__ = ["B4B3CIBCExactCallObserverV1", "B4B3CIBCExactCallReceiptV1"]
