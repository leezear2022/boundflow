"""Prepared runtime for fused root input Conv and L-infinity concretization."""

# mypy: disable-error-code=import-untyped
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=too-many-locals,protected-access,duplicate-code
# pylint: disable=import-error,import-outside-toplevel

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from boundflow.backends.tvm.root_crown_input_domain import (
    CompiledRootCrownInputDomainTIRV1,
    RootCrownInputDomainTemplateV1,
    compile_root_crown_input_domain_tir_v1,
)
from boundflow.runtime.root_crown_projection_tir import (
    _ordinal_map,
)


@dataclass(frozen=True)
class RootCrownInputDomainTensorsV1:
    """Dynamic tensors for one input Conv/concretization forward/full-VJP."""

    incoming_lower_a: torch.Tensor
    preactivation_lower: torch.Tensor
    preactivation_upper: torch.Tensor
    raw_alpha: torch.Tensor
    operator_weight: torch.Tensor
    operator_bias: torch.Tensor
    input_center: torch.Tensor
    input_radius: torch.Tensor


def _expected_shapes(
    template: RootCrownInputDomainTemplateV1,
) -> dict[str, tuple[int, ...]]:
    return {
        "incoming_lower_a": template.incoming_shape,
        "preactivation_lower": template.bound_shape,
        "preactivation_upper": template.bound_shape,
        "raw_alpha": (
            2,
            template.spec_count,
            template.domain_count,
            template.alpha_count,
        ),
        "operator_weight": template.weight_shape,
        "operator_bias": (template.output_channels,),
        "input_center": template.input_shape,
        "input_radius": template.input_shape,
    }


def _validate_runtime_structure(
    tensors: RootCrownInputDomainTensorsV1,
    template: RootCrownInputDomainTemplateV1,
) -> None:
    devices: set[torch.device] = set()
    for name, shape in _expected_shapes(template).items():
        value = getattr(tensors, name)
        devices.add(value.device)
        if (
            tuple(value.shape) != shape
            or value.device.type != "cuda"
            or value.dtype != torch.float32
            or not value.is_contiguous()
        ):
            raise ValueError(f"root CROWN input-domain runtime tensor differs: {name}")
    if len(devices) != 1:
        raise ValueError("root CROWN input-domain runtime device differs")


def validate_root_crown_input_domain_tensors_v1(
    tensors: RootCrownInputDomainTensorsV1,
    template: RootCrownInputDomainTemplateV1,
) -> None:
    """Fail closed at static admission, including value constraints."""

    template.validate()
    _validate_runtime_structure(tensors, template)
    for name in _expected_shapes(template):
        if not bool(torch.isfinite(getattr(tensors, name)).all().item()):
            raise ValueError(
                f"root CROWN input-domain nonfinite tensor differs: {name}"
            )
    if bool((tensors.preactivation_lower > tensors.preactivation_upper).any().item()):
        raise ValueError("root CROWN input-domain interval differs")
    if bool(((tensors.raw_alpha < 0) | (tensors.raw_alpha > 1)).any().item()):
        raise ValueError("root CROWN input-domain alpha range differs")
    if bool((tensors.input_radius < 0).any().item()):
        raise ValueError("root CROWN input-domain radius differs")


class RootCrownInputDomainTIRExecutorV1:
    """Prepared module, persistent arenas, and exact current-stream launches."""

    def __init__(self, template: RootCrownInputDomainTemplateV1) -> None:
        template.validate()
        self.template = template
        self.compiled: CompiledRootCrownInputDomainTIRV1 = (
            compile_root_crown_input_domain_tir_v1(template)
        )
        geometry = (
            template.output_channels,
            template.output_height,
            template.output_width,
        )
        self._alpha_map = _ordinal_map(template.alpha_coordinates, geometry)
        self._view_cache: dict[tuple[int, tuple[int, ...], str, str], Any] = {}
        self._concrete_lower: torch.Tensor | None = None
        self._output_bias: torch.Tensor | None = None
        self._incoming_gradient: torch.Tensor | None = None
        self._alpha_gradient: torch.Tensor | None = None
        self.forward_launch_count = 0
        self.backward_launch_count = 0
        self.fallback_count = 0
        self.pointer_count = 0
        self.pointer_exact_count = 0
        self.prepare_count = 0

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
            raise RuntimeError("root CROWN input-domain DLPack pointer differs")
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
                raise RuntimeError("root CROWN input-domain current stream differs")
            views = tuple(self._view(value) for value in (*sources, *outputs))
            self.compiled.executable[symbol](*views)
        count = len(sources) + len(outputs)
        self.pointer_count += count
        self.pointer_exact_count += count

    def forward(
        self, tensors: RootCrownInputDomainTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_runtime_structure(tensors, self.template)
        if self._concrete_lower is None:
            self._concrete_lower = torch.empty(
                (self.template.domain_count, self.template.spec_count),
                dtype=torch.float32,
                device=tensors.incoming_lower_a.device,
            )
            self._output_bias = torch.empty(
                (self.template.spec_count, self.template.domain_count),
                dtype=torch.float32,
                device=tensors.incoming_lower_a.device,
            )
        concrete = self._concrete_lower
        output_bias = self._output_bias
        if concrete is None or output_bias is None:
            raise RuntimeError("root CROWN input-domain output arena differs")
        self._launch(
            self.template.forward_symbol,
            (
                tensors.incoming_lower_a,
                tensors.preactivation_lower,
                tensors.preactivation_upper,
                tensors.raw_alpha,
                self._alpha_map,
                tensors.operator_weight,
                tensors.operator_bias,
                tensors.input_center,
                tensors.input_radius,
            ),
            (concrete, output_bias),
        )
        self.forward_launch_count += 1
        return concrete, output_bias

    def _prepare_backward_arena(
        self, tensors: RootCrownInputDomainTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._incoming_gradient is None:
            self._incoming_gradient = torch.empty_like(tensors.incoming_lower_a)
            self._alpha_gradient = torch.empty_like(tensors.raw_alpha)
        if self._incoming_gradient is None or self._alpha_gradient is None:
            raise RuntimeError("root CROWN input-domain backward arena differs")
        return self._incoming_gradient, self._alpha_gradient

    def backward(
        self,
        tensors: RootCrownInputDomainTensorsV1,
        concrete_gradient: torch.Tensor,
        bias_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for name, value, shape in (
            (
                "concrete gradient",
                concrete_gradient,
                (self.template.domain_count, self.template.spec_count),
            ),
            (
                "bias gradient",
                bias_gradient,
                (self.template.spec_count, self.template.domain_count),
            ),
        ):
            if (
                tuple(value.shape) != shape
                or value.device != tensors.incoming_lower_a.device
                or value.dtype != torch.float32
                or not value.is_contiguous()
            ):
                raise ValueError(f"root CROWN input-domain {name} differs")
        outputs = self._prepare_backward_arena(tensors)
        self._launch(
            self.template.backward_symbol,
            (
                tensors.incoming_lower_a,
                tensors.preactivation_lower,
                tensors.preactivation_upper,
                tensors.raw_alpha,
                self._alpha_map,
                tensors.operator_weight,
                tensors.operator_bias,
                tensors.input_center,
                tensors.input_radius,
                concrete_gradient,
                bias_gradient,
            ),
            outputs,
        )
        self.backward_launch_count += 1
        return outputs

    def prepare(self) -> None:
        """Materialize module and persistent output/VJP arenas before timing."""

        if self.prepare_count:
            raise RuntimeError("root CROWN input-domain executor already prepared")
        template = self.template
        device = torch.device("cuda")
        tensors = RootCrownInputDomainTensorsV1(
            torch.zeros(template.incoming_shape, dtype=torch.float32, device=device),
            torch.full(template.bound_shape, -1.0, dtype=torch.float32, device=device),
            torch.full(template.bound_shape, 1.0, dtype=torch.float32, device=device),
            torch.full(
                (2, template.spec_count, template.domain_count, template.alpha_count),
                0.5,
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(template.weight_shape, dtype=torch.float32, device=device),
            torch.zeros(
                (template.output_channels,), dtype=torch.float32, device=device
            ),
            torch.zeros(template.input_shape, dtype=torch.float32, device=device),
            torch.ones(template.input_shape, dtype=torch.float32, device=device),
        )
        concrete, bias = self.forward(tensors)
        self.backward(tensors, torch.zeros_like(concrete), torch.zeros_like(bias))
        torch.cuda.synchronize(device)
        persistent = {
            value.data_ptr()
            for value in (
                self._alpha_map,
                self._concrete_lower,
                self._output_bias,
                self._incoming_gradient,
                self._alpha_gradient,
            )
            if value is not None
        }
        self._view_cache = {
            key: view for key, view in self._view_cache.items() if key[0] in persistent
        }
        self.forward_launch_count = 0
        self.backward_launch_count = 0
        self.pointer_count = 0
        self.pointer_exact_count = 0
        self.prepare_count = 1


__all__ = [
    "RootCrownInputDomainTIRExecutorV1",
    "RootCrownInputDomainTensorsV1",
    "validate_root_crown_input_domain_tensors_v1",
]
