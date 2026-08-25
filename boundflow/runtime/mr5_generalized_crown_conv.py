"""Autograd and exact CUDA-stream runtime for MR5 generalized Conv sites."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-arguments
# pylint: disable=too-many-locals,too-many-instance-attributes
# pylint: disable=abstract-method,arguments-differ,too-few-public-methods
# pylint: disable=missing-function-docstring,too-many-boolean-expressions
# pylint: disable=too-many-positional-arguments,not-callable,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as torch_functional

from boundflow.backends.tvm.mr5_generalized_crown_conv import (
    CompiledMR5GeneralizedConvV1,
    MR5GeneralizedConvSignatureV1,
    MR5_BACKWARD_SYMBOL,
    MR5_FORWARD_SYMBOL,
    compile_mr5_generalized_conv,
)


@dataclass(frozen=True)
class MR5GeneralizedConvTensorsV1:
    """Complete dense beta-free ABI for one production site evaluation."""

    incoming: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    alpha: torch.Tensor
    incoming_bias: torch.Tensor
    weight: torch.Tensor
    operator_bias: torch.Tensor


@dataclass(frozen=True)
class MR5GeneralizedConvModuleReceiptV1:
    """Compile identity and workspace evidence for one site module."""

    site_id: str
    signature_hash: str
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    tvm_version: str
    torch_version: str
    workspace_inventory: tuple[tuple[str, tuple[int, ...]], ...]

    def validate_against(self, signature: MR5GeneralizedConvSignatureV1) -> None:
        signature.validate()
        hashes = (
            self.signature_hash,
            self.unscheduled_tir_hash,
            self.scheduled_tir_hash,
            self.device_source_hash,
        )
        if (
            self.site_id != signature.site_id
            or self.signature_hash != signature.stable_hash()
            or any(len(value) != 64 for value in hashes)
            or not self.tvm_version
            or not self.torch_version
            or not self.workspace_inventory
        ):
            raise ValueError("MR5 generalized Conv module receipt differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "signature_hash": self.signature_hash,
            "unscheduled_tir_hash": self.unscheduled_tir_hash,
            "scheduled_tir_hash": self.scheduled_tir_hash,
            "device_source_hash": self.device_source_hash,
            "tvm_version": self.tvm_version,
            "torch_version": self.torch_version,
            "workspace_inventory": [
                {"name": name, "shape": list(shape)}
                for name, shape in self.workspace_inventory
            ],
        }


class MR5GeneralizedConvModuleCacheV1:
    """Signature-keyed module cache; dynamic tensors never enter compile keys."""

    def __init__(self) -> None:
        self._entries: dict[
            str,
            tuple[CompiledMR5GeneralizedConvV1, MR5GeneralizedConvModuleReceiptV1],
        ] = {}

    def get(
        self, signature: MR5GeneralizedConvSignatureV1
    ) -> tuple[CompiledMR5GeneralizedConvV1, MR5GeneralizedConvModuleReceiptV1, str]:
        signature.validate()
        key = signature.stable_hash()
        cached = self._entries.get(key)
        if cached is not None:
            return (*cached, "hit")
        compiled = compile_mr5_generalized_conv(signature)
        receipt = MR5GeneralizedConvModuleReceiptV1(
            site_id=signature.site_id,
            signature_hash=compiled.signature_hash,
            unscheduled_tir_hash=compiled.unscheduled_tir_hash,
            scheduled_tir_hash=compiled.scheduled_tir_hash,
            device_source_hash=compiled.device_source_hash,
            tvm_version=compiled.tvm_version,
            torch_version=str(torch.__version__),
            workspace_inventory=compiled.workspace_inventory,
        )
        receipt.validate_against(signature)
        self._entries[key] = (compiled, receipt)
        return compiled, receipt, "miss"


def validate_mr5_generalized_conv_tensors(
    signature: MR5GeneralizedConvSignatureV1,
    tensors: MR5GeneralizedConvTensorsV1,
) -> None:
    signature.validate()
    expected = {
        "incoming": signature.incoming_shape,
        "lower": signature.relaxation_shape,
        "upper": signature.relaxation_shape,
        "alpha": signature.relaxation_shape,
        "incoming_bias": (signature.domain_count, signature.spec_count),
        "weight": signature.weight_shape,
        "operator_bias": (signature.output_channels,),
    }
    for name, shape in expected.items():
        tensor = getattr(tensors, name)
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.float32
            or tensor.device.type != "cuda"
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise ValueError(f"MR5 generalized Conv tensor differs: {name}")
    if not tensors.incoming.requires_grad or not tensors.alpha.requires_grad:
        raise ValueError("MR5 generalized Conv gradient ownership differs")
    if tensors.lower.requires_grad or tensors.upper.requires_grad:
        raise ValueError("MR5 generalized Conv bound ownership differs")
    if bool((tensors.lower > tensors.upper).any().item()):
        raise ValueError("MR5 generalized Conv interval differs")
    if bool(((tensors.alpha < 0) | (tensors.alpha > 1)).any().item()):
        raise ValueError("MR5 generalized Conv alpha range differs")


@dataclass(frozen=True)
class _LaunchObservation:
    stream_id: int
    ffi_stream_id: int
    pointer_count: int
    pointer_exact_count: int


class MR5GeneralizedConvExecutorV1:
    """Exactly one forward and optional one backward launch for a site evaluation."""

    def __init__(
        self,
        signature: MR5GeneralizedConvSignatureV1,
        cache: MR5GeneralizedConvModuleCacheV1,
    ) -> None:
        self.signature = signature
        self.compiled, self.module_receipt, self.cache_event = cache.get(signature)
        self.forward_launch_count = 0
        self.backward_launch_count = 0
        self.fallback_count = 0
        self.eager_count = 0
        self.forward_observation: Optional[_LaunchObservation] = None
        self.backward_observation: Optional[_LaunchObservation] = None
        self._tensors: Optional[MR5GeneralizedConvTensorsV1] = None

    def reject(self, reason: str) -> None:
        self.fallback_count += 1
        raise ValueError(reason)

    def prime(self, tensors: MR5GeneralizedConvTensorsV1) -> None:
        if self._tensors is not None:
            self.reject("MR5 generalized Conv executor already primed")
        validate_mr5_generalized_conv_tensors(self.signature, tensors)
        self._tensors = tensors

    def _launch(
        self,
        symbol: str,
        sources: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
    ) -> _LaunchObservation:
        import tvm
        import tvm_ffi

        device = sources[0].device
        current = torch.cuda.current_stream(device)
        device_before = torch.cuda.current_device()
        stream_before = int(current.cuda_stream)
        deterministic_before = (
            torch.are_deterministic_algorithms_enabled(),
            torch.get_deterministic_debug_mode(),
        )
        ordinal = device.index if device.index is not None else device_before
        exact = 0
        ffi_stream = -1
        try:
            with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
                ffi_stream = int(
                    tvm_ffi.get_raw_stream(tvm_ffi.device(f"cuda:{ordinal}"))
                )
                if ffi_stream != stream_before:
                    raise RuntimeError("MR5 generalized Conv stream differs")
                source_views = []
                output_views = []
                for tensor in sources:
                    view = tvm.runtime.from_dlpack(tensor)
                    exact += int(
                        torch.from_dlpack(view).data_ptr() == tensor.data_ptr()
                    )
                    source_views.append(view)
                for tensor in outputs:
                    view = tvm.runtime.from_dlpack(tensor)
                    exact += int(
                        torch.from_dlpack(view).data_ptr() == tensor.data_ptr()
                    )
                    output_views.append(view)
                if exact != len(sources) + len(outputs):
                    raise RuntimeError("MR5 generalized Conv DLPack pointer differs")
                self.compiled.executable[symbol](*source_views, *output_views)
        finally:
            deterministic_after = (
                torch.are_deterministic_algorithms_enabled(),
                torch.get_deterministic_debug_mode(),
            )
            if (
                torch.cuda.current_device() != device_before
                or int(torch.cuda.current_stream(device).cuda_stream) != stream_before
                or deterministic_after != deterministic_before
            ):
                raise RuntimeError("MR5 generalized Conv global state drifted")
        return _LaunchObservation(
            stream_id=stream_before,
            ffi_stream_id=ffi_stream,
            pointer_count=len(sources) + len(outputs),
            pointer_exact_count=exact,
        )

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = self._tensors
        if tensors is None or self.forward_launch_count:
            self.reject("MR5 generalized Conv forward lifecycle differs")
        assert tensors is not None
        result_a = torch.empty(
            self.signature.result_shape,
            dtype=torch.float32,
            device=tensors.incoming.device,
        )
        result_bias = torch.empty_like(tensors.incoming_bias)
        self.forward_observation = self._launch(
            MR5_FORWARD_SYMBOL,
            (
                tensors.incoming,
                tensors.lower,
                tensors.upper,
                tensors.alpha,
                tensors.incoming_bias,
                tensors.weight,
                tensors.operator_bias,
            ),
            (result_a, result_bias),
        )
        self.forward_launch_count += 1
        return result_a, result_bias

    def backward(
        self, result_a_gradient: torch.Tensor, result_bias_gradient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = self._tensors
        if (
            tensors is None
            or self.forward_launch_count != 1
            or self.backward_launch_count
        ):
            self.reject("MR5 generalized Conv backward lifecycle differs")
        assert tensors is not None
        if (
            tuple(result_a_gradient.shape) != self.signature.result_shape
            or tuple(result_bias_gradient.shape)
            != (self.signature.domain_count, self.signature.spec_count)
            or result_a_gradient.dtype != torch.float32
            or result_bias_gradient.dtype != torch.float32
            or result_a_gradient.device != tensors.incoming.device
            or result_bias_gradient.device != tensors.incoming.device
        ):
            self.reject("MR5 generalized Conv output adjoint differs")
        alpha_gradient = torch.empty_like(tensors.alpha)
        incoming_gradient = torch.empty_like(tensors.incoming)
        self.backward_observation = self._launch(
            MR5_BACKWARD_SYMBOL,
            (
                tensors.incoming,
                tensors.lower,
                tensors.upper,
                tensors.alpha,
                tensors.weight,
                tensors.operator_bias,
                result_a_gradient.contiguous(),
                result_bias_gradient.contiguous(),
            ),
            (alpha_gradient, incoming_gradient),
        )
        self.backward_launch_count += 1
        return incoming_gradient, alpha_gradient


class _MR5GeneralizedConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        incoming: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        alpha: torch.Tensor,
        incoming_bias: torch.Tensor,
        weight: torch.Tensor,
        operator_bias: torch.Tensor,
        executor: MR5GeneralizedConvExecutorV1,
    ):
        tensors = executor._tensors  # pylint: disable=protected-access
        if tensors is None:
            raise RuntimeError("MR5 generalized Conv executor is not primed")
        passed = (incoming, lower, upper, alpha, incoming_bias, weight, operator_bias)
        expected = tuple(
            getattr(tensors, name) for name in tensors.__dataclass_fields__
        )
        if any(
            left.data_ptr() != right.data_ptr() for left, right in zip(passed, expected)
        ):
            executor.reject("MR5 generalized Conv autograd input differs")
        ctx.executor = executor
        ctx.set_materialize_grads(False)
        return executor.forward()

    @staticmethod
    def backward(
        ctx, result_a_gradient: torch.Tensor, result_bias_gradient: torch.Tensor
    ):
        if torch.is_grad_enabled():
            ctx.executor.eager_count += 1
            raise RuntimeError(
                "MR5 generalized Conv higher-order gradients unsupported"
            )
        incoming_gradient, alpha_gradient = ctx.executor.backward(
            result_a_gradient, result_bias_gradient
        )
        return incoming_gradient, None, None, alpha_gradient, None, None, None, None


def execute_mr5_generalized_conv_v1(
    signature: MR5GeneralizedConvSignatureV1,
    tensors: MR5GeneralizedConvTensorsV1,
    cache: MR5GeneralizedConvModuleCacheV1,
) -> tuple[torch.Tensor, torch.Tensor, MR5GeneralizedConvExecutorV1]:
    """Execute one typed site through the custom TIR backward boundary."""

    executor = MR5GeneralizedConvExecutorV1(signature, cache)
    executor.prime(tensors)
    result_a, result_bias = _MR5GeneralizedConvFunction.apply(
        tensors.incoming,
        tensors.lower,
        tensors.upper,
        tensors.alpha,
        tensors.incoming_bias,
        tensors.weight,
        tensors.operator_bias,
        executor,
    )
    return result_a, result_bias, executor


def run_mr5_pytorch_oracle_v1(
    signature: MR5GeneralizedConvSignatureV1,
    tensors: MR5GeneralizedConvTensorsV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent PyTorch expression for generalized forward/VJP parity."""

    signature.validate()
    lower = tensors.lower
    upper = tensors.upper
    epsilon = torch.finfo(torch.float32).eps
    upper_slope = torch.where(
        lower >= 0,
        torch.ones_like(lower),
        torch.where(
            upper <= 0,
            torch.zeros_like(lower),
            upper / (upper - lower).clamp_min(epsilon),
        ),
    )
    lower_slope = torch.where(
        (lower < 0) & (upper > 0),
        tensors.alpha.clamp(0, 1),
        (lower >= 0).to(lower.dtype),
    )
    positive = tensors.incoming >= 0
    slope = torch.where(positive, lower_slope.unsqueeze(1), upper_slope.unsqueeze(1))
    intercept = torch.where(
        positive,
        torch.zeros_like(tensors.incoming),
        torch.where(
            ((lower < 0) & (upper > 0)).unsqueeze(1),
            (-lower * upper_slope).unsqueeze(1),
            torch.zeros_like(tensors.incoming),
        ),
    )
    relu_a = tensors.incoming * slope
    flattened = relu_a.reshape(
        signature.domain_count * signature.spec_count,
        signature.output_channels,
        signature.output_height,
        signature.output_width,
    )
    result_a = torch_functional.conv_transpose2d(
        flattened,
        tensors.weight,
        stride=signature.stride,
        padding=signature.padding,
        output_padding=signature.output_padding,
        dilation=signature.dilation,
    ).reshape(signature.result_shape)
    bias_delta = (
        tensors.incoming * intercept
        + relu_a * tensors.operator_bias.reshape(1, 1, -1, 1, 1)
    ).sum(dim=(2, 3, 4))
    return result_a, tensors.incoming_bias + bias_delta


__all__ = [
    "MR5GeneralizedConvExecutorV1",
    "MR5GeneralizedConvModuleCacheV1",
    "MR5GeneralizedConvModuleReceiptV1",
    "MR5GeneralizedConvTensorsV1",
    "execute_mr5_generalized_conv_v1",
    "run_mr5_pytorch_oracle_v1",
    "validate_mr5_generalized_conv_tensors",
]
