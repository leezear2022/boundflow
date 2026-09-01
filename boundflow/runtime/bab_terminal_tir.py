"""Current-stream runtime for the beta-aware activation-BaB terminal TIR."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-locals
# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=abstract-method,arguments-differ
# pylint: disable=too-many-arguments,too-many-positional-arguments,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from boundflow.backends.tvm.bab_terminal_linear import (
    BabTerminalLinearTemplateV1,
    CompiledBabTerminalLinearTIRV1,
    compile_bab_terminal_tir_v1,
)


@dataclass(frozen=True)
class BabTerminalTensorsV1:
    """Dynamic and frozen inputs for one beta-aware terminal evaluation."""

    incoming_lower_a: torch.Tensor
    preactivation_lower: torch.Tensor
    preactivation_upper: torch.Tensor
    compressed_alpha: torch.Tensor
    sparse_beta: torch.Tensor
    beta_location: torch.Tensor
    beta_sign: torch.Tensor
    linear_weight: torch.Tensor
    linear_bias: torch.Tensor


def _expected_shapes(
    template: BabTerminalLinearTemplateV1,
) -> dict[str, tuple[int, ...]]:
    return {
        "incoming_lower_a": (
            template.spec_count,
            template.domain_count,
            template.current_features,
        ),
        "preactivation_lower": (template.domain_count, template.current_features),
        "preactivation_upper": (template.domain_count, template.current_features),
        "compressed_alpha": (
            2,
            template.spec_count,
            template.domain_count,
            template.alpha_count,
        ),
        "sparse_beta": (template.domain_count, template.beta_count),
        "beta_location": (template.domain_count, template.beta_count),
        "beta_sign": (template.domain_count, template.beta_count),
        "linear_weight": (template.current_features, template.previous_features),
        "linear_bias": (template.current_features,),
    }


def _validate_runtime_structure(
    tensors: BabTerminalTensorsV1, template: BabTerminalLinearTemplateV1
) -> None:
    devices: set[torch.device] = set()
    for name, shape in _expected_shapes(template).items():
        value = getattr(tensors, name)
        devices.add(value.device)
        expected_dtype = torch.int64 if name == "beta_location" else torch.float32
        if (
            tuple(value.shape) != shape
            or value.device.type != "cuda"
            or value.dtype != expected_dtype
            or not value.is_contiguous()
        ):
            raise ValueError(f"activation-BaB terminal runtime tensor differs: {name}")
    if len(devices) != 1:
        raise ValueError("activation-BaB terminal runtime device differs")


def validate_bab_terminal_tensors_v1(
    tensors: BabTerminalTensorsV1, template: BabTerminalLinearTemplateV1
) -> None:
    """Fail closed before launch, including sparse-state value constraints."""

    template.validate()
    _validate_runtime_structure(tensors, template)
    for name in _expected_shapes(template):
        value = getattr(tensors, name)
        if name != "beta_location" and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"activation-BaB terminal nonfinite tensor: {name}")
    if (
        bool((tensors.preactivation_lower > tensors.preactivation_upper).any().item())
        or bool(
            ((tensors.compressed_alpha < 0) | (tensors.compressed_alpha > 1))
            .any()
            .item()
        )
        or bool((tensors.sparse_beta < 0).any().item())
        or bool(
            (
                (tensors.beta_location < 0)
                | (tensors.beta_location >= template.current_features)
            )
            .any()
            .item()
        )
        or bool(
            (~torch.isin(tensors.beta_sign, tensors.beta_sign.new_tensor((-1.0, 1.0))))
            .any()
            .item()
        )
    ):
        raise ValueError("activation-BaB terminal sparse-state legality differs")


class BabTerminalTIRExecutorV1:
    """Prepared current-stream executable with persistent output/VJP arenas."""

    def __init__(
        self,
        template: BabTerminalLinearTemplateV1,
        *,
        compiled: CompiledBabTerminalLinearTIRV1 | None = None,
    ) -> None:
        self.template = template
        self.compiled = compiled or compile_bab_terminal_tir_v1(template)
        self.forward_launch_count = 0
        self.backward_launch_count = 0
        self.fallback_count = 0
        self.pointer_count = 0
        self.pointer_exact_count = 0
        self._view_cache: dict[tuple[int, tuple[int, ...], str, str], Any] = {}
        self._output_a: torch.Tensor | None = None
        self._output_bias: torch.Tensor | None = None
        self._incoming_gradient: torch.Tensor | None = None
        self._alpha_gradient: torch.Tensor | None = None
        self._beta_gradient: torch.Tensor | None = None
        self._alpha_map = torch.full(
            (template.current_features,), -1, dtype=torch.int32, device="cuda"
        )
        self._feature_indices = torch.tensor(
            template.alpha_feature_indices, dtype=torch.int32, device="cuda"
        )
        self._alpha_map[self._feature_indices.to(torch.int64)] = torch.arange(
            template.alpha_count, dtype=torch.int32, device="cuda"
        )

    def _view(self, tensor: torch.Tensor) -> Any:
        import tvm

        key = (
            tensor.data_ptr(),
            tuple(tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
        )
        existing = self._view_cache.get(key)
        if existing is not None:
            return existing
        view = tvm.runtime.from_dlpack(tensor)
        if torch.from_dlpack(view).data_ptr() != tensor.data_ptr():
            raise RuntimeError("activation-BaB terminal DLPack pointer differs")
        self._view_cache[key] = view
        return view

    def _launch(
        self,
        symbol: str,
        sources: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
    ) -> None:
        import tvm_ffi

        device = sources[0].device
        current = torch.cuda.current_stream(device)
        stream_id = int(current.cuda_stream)
        ordinal = device.index
        if ordinal is None:
            ordinal = torch.cuda.current_device()
        with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
            ffi_stream = int(tvm_ffi.get_raw_stream(tvm_ffi.device(f"cuda:{ordinal}")))
            if ffi_stream != stream_id:
                raise RuntimeError("activation-BaB terminal current stream differs")
            views = tuple(self._view(value) for value in (*sources, *outputs))
            self.compiled.executable[symbol](*views)
        count = len(sources) + len(outputs)
        self.pointer_count += count
        self.pointer_exact_count += count

    def forward(
        self, tensors: BabTerminalTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_runtime_structure(tensors, self.template)
        if self._output_a is None:
            self._output_a = torch.empty(
                (
                    self.template.spec_count,
                    self.template.domain_count,
                    self.template.previous_features,
                ),
                dtype=torch.float32,
                device=tensors.incoming_lower_a.device,
            )
            self._output_bias = torch.empty(
                (self.template.spec_count, self.template.domain_count),
                dtype=torch.float32,
                device=tensors.incoming_lower_a.device,
            )
        if self._output_bias is None:
            raise RuntimeError("activation-BaB terminal output arena differs")
        self._launch(
            self.template.forward_symbol,
            (
                tensors.incoming_lower_a,
                tensors.preactivation_lower,
                tensors.preactivation_upper,
                tensors.compressed_alpha,
                self._alpha_map,
                tensors.sparse_beta,
                tensors.beta_location,
                tensors.beta_sign,
                tensors.linear_weight,
                tensors.linear_bias,
            ),
            (self._output_a, self._output_bias),
        )
        self.forward_launch_count += 1
        return self._output_a, self._output_bias

    def backward(
        self,
        tensors: BabTerminalTensorsV1,
        output_a_gradient: torch.Tensor,
        output_bias_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for name, value, shape in (
            (
                "output A gradient",
                output_a_gradient,
                (
                    self.template.spec_count,
                    self.template.domain_count,
                    self.template.previous_features,
                ),
            ),
            (
                "output bias gradient",
                output_bias_gradient,
                (self.template.spec_count, self.template.domain_count),
            ),
        ):
            if (
                tuple(value.shape) != shape
                or value.device != tensors.incoming_lower_a.device
                or value.dtype != torch.float32
                or not value.is_contiguous()
            ):
                raise ValueError(f"activation-BaB terminal {name} differs")
        if self._incoming_gradient is None:
            self._incoming_gradient = torch.empty_like(tensors.incoming_lower_a)
            self._alpha_gradient = torch.empty_like(tensors.compressed_alpha)
            self._beta_gradient = torch.empty_like(tensors.sparse_beta)
        if self._alpha_gradient is None or self._beta_gradient is None:
            raise RuntimeError("activation-BaB terminal gradient arena differs")
        self._launch(
            self.template.backward_symbol,
            (
                tensors.incoming_lower_a,
                tensors.preactivation_lower,
                tensors.preactivation_upper,
                tensors.compressed_alpha,
                self._alpha_map,
                tensors.sparse_beta,
                tensors.beta_location,
                tensors.beta_sign,
                tensors.linear_weight,
                tensors.linear_bias,
                output_a_gradient,
                output_bias_gradient,
                self._feature_indices,
            ),
            (self._incoming_gradient, self._alpha_gradient, self._beta_gradient),
        )
        self.backward_launch_count += 1
        return self._incoming_gradient, self._alpha_gradient, self._beta_gradient


class _BabTerminalTIRFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        incoming: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        beta_location: torch.Tensor,
        beta_sign: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        executor: BabTerminalTIRExecutorV1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = BabTerminalTensorsV1(
            incoming,
            lower,
            upper,
            alpha,
            beta,
            beta_location,
            beta_sign,
            weight,
            bias,
        )
        _validate_runtime_structure(tensors, executor.template)
        ctx.tensors = tensors
        ctx.executor = executor
        ctx.set_materialize_grads(False)
        return executor.forward(tensors)

    @staticmethod
    def backward(
        ctx: Any,
        output_a_gradient: torch.Tensor,
        output_bias_gradient: torch.Tensor,
    ) -> tuple[Any, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "activation-BaB terminal higher-order gradient unsupported"
            )
        incoming, alpha, beta = ctx.executor.backward(
            ctx.tensors,
            output_a_gradient.contiguous(),
            output_bias_gradient.contiguous(),
        )
        return incoming, None, None, alpha, beta, None, None, None, None, None


def execute_bab_terminal_tir_v1(
    tensors: BabTerminalTensorsV1, executor: BabTerminalTIRExecutorV1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute one fused terminal transaction behind custom autograd."""

    return _BabTerminalTIRFunction.apply(
        tensors.incoming_lower_a,
        tensors.preactivation_lower,
        tensors.preactivation_upper,
        tensors.compressed_alpha,
        tensors.sparse_beta,
        tensors.beta_location,
        tensors.beta_sign,
        tensors.linear_weight,
        tensors.linear_bias,
        executor,
    )


__all__ = [
    "BabTerminalTensorsV1",
    "BabTerminalTIRExecutorV1",
    "execute_bab_terminal_tir_v1",
    "validate_bab_terminal_tensors_v1",
]
