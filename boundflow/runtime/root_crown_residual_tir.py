"""Runtime and custom-autograd owner for the root residual CROWN TIR."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals
# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=abstract-method,arguments-differ,too-many-return-statements

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from boundflow.backends.tvm.root_crown_residual import (
    CompiledRootCrownResidualTIRV1,
    RootCrownResidualTemplateV1,
    compile_root_crown_residual_tir_v1,
)


@dataclass(frozen=True)
class RootCrownResidualTensorsV1:
    """Native-layout tensors for one two-Conv residual CROWN evaluation."""

    incoming_lower_a: torch.Tensor
    entry_lower: torch.Tensor
    entry_upper: torch.Tensor
    entry_raw_alpha: torch.Tensor
    main_conv_weight: torch.Tensor
    main_conv_bias: torch.Tensor
    inner_lower: torch.Tensor
    inner_upper: torch.Tensor
    inner_raw_alpha: torch.Tensor
    inner_conv_weight: torch.Tensor
    inner_conv_bias: torch.Tensor


def _expected_shapes(
    template: RootCrownResidualTemplateV1,
) -> dict[str, tuple[int, ...]]:
    return {
        "incoming_lower_a": template.coefficient_shape,
        "entry_lower": template.bound_shape,
        "entry_upper": template.bound_shape,
        "entry_raw_alpha": (
            2,
            template.spec_count,
            template.domain_count,
            template.entry_alpha_count,
        ),
        "main_conv_weight": template.weight_shape,
        "main_conv_bias": (template.channels,),
        "inner_lower": template.bound_shape,
        "inner_upper": template.bound_shape,
        "inner_raw_alpha": (
            2,
            template.spec_count,
            template.domain_count,
            template.inner_alpha_count,
        ),
        "inner_conv_weight": template.weight_shape,
        "inner_conv_bias": (template.channels,),
    }


def validate_root_crown_residual_tensors_v1(
    tensors: RootCrownResidualTensorsV1,
    template: RootCrownResidualTemplateV1,
) -> None:
    """Fail closed before admission, including synchronized value checks."""

    template.validate()
    _validate_runtime_structure(tensors, template)
    for name in _expected_shapes(template):
        if not bool(torch.isfinite(getattr(tensors, name)).all().item()):
            raise ValueError(f"root CROWN residual nonfinite tensor differs: {name}")
    if bool((tensors.entry_lower > tensors.entry_upper).any().item()):
        raise ValueError("root CROWN residual entry interval differs")
    if bool((tensors.inner_lower > tensors.inner_upper).any().item()):
        raise ValueError("root CROWN residual inner interval differs")
    for name in ("entry_raw_alpha", "inner_raw_alpha"):
        alpha = getattr(tensors, name)
        if bool(((alpha < 0) | (alpha > 1)).any().item()):
            raise ValueError(f"root CROWN residual alpha range differs: {name}")


def _validate_runtime_structure(
    tensors: RootCrownResidualTensorsV1,
    template: RootCrownResidualTemplateV1,
) -> None:
    """Check O(1) shape/layout/device invariants on the warm path."""

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
            raise ValueError(f"root CROWN residual runtime tensor differs: {name}")
    if len(devices) != 1:
        raise ValueError("root CROWN residual runtime device differs")


def _coordinate_tensors(
    coordinates: tuple[tuple[int, int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinate_tensor = (
        torch.tensor(coordinates, dtype=torch.int32, device="cuda")
        .transpose(0, 1)
        .contiguous()
    )
    return coordinate_tensor, coordinate_tensor.transpose(0, 1).contiguous()


def _ordinal_map(
    coordinates: tuple[tuple[int, int, int], ...],
    shape: tuple[int, int, int],
) -> torch.Tensor:
    result = torch.full(shape, -1, dtype=torch.int32, device="cuda")
    coordinate_tensor, row_coordinates = _coordinate_tensors(coordinates)
    del coordinate_tensor
    ordinals = torch.arange(len(coordinates), dtype=torch.int32, device="cuda")
    result[
        row_coordinates[:, 0].to(torch.int64),
        row_coordinates[:, 1].to(torch.int64),
        row_coordinates[:, 2].to(torch.int64),
    ] = ordinals
    return result


class RootCrownResidualTIRExecutorV1:
    """Prepared module, coordinate metadata, arena, and current-stream launches."""

    def __init__(self, template: RootCrownResidualTemplateV1) -> None:
        template.validate()
        self.template = template
        self.compiled: CompiledRootCrownResidualTIRV1 = (
            compile_root_crown_residual_tir_v1(template)
        )
        coordinate_shape = (template.channels, template.height, template.width)
        self._entry_map = _ordinal_map(
            template.entry_alpha_coordinates, coordinate_shape
        )
        self._entry_coordinates, _ = _coordinate_tensors(
            template.entry_alpha_coordinates
        )
        self._inner_map = _ordinal_map(
            template.inner_alpha_coordinates, coordinate_shape
        )
        self._inner_coordinates, _ = _coordinate_tensors(
            template.inner_alpha_coordinates
        )
        self._view_cache: dict[tuple[int, tuple[int, ...], str, str], Any] = {}
        self._output_a: torch.Tensor | None = None
        self._output_bias: torch.Tensor | None = None
        self._main_a: torch.Tensor | None = None
        self._incoming_gradient: torch.Tensor | None = None
        self._entry_lower_gradient: torch.Tensor | None = None
        self._entry_upper_gradient: torch.Tensor | None = None
        self._entry_alpha_gradient: torch.Tensor | None = None
        self._inner_lower_gradient: torch.Tensor | None = None
        self._inner_upper_gradient: torch.Tensor | None = None
        self._inner_alpha_gradient: torch.Tensor | None = None
        self.forward_launch_count = 0
        self.backward_launch_count = 0
        self.fallback_count = 0
        self.pointer_count = 0
        self.pointer_exact_count = 0
        self.prepare_count = 0

    def _view(self, tensor: torch.Tensor) -> Any:
        import tvm  # type: ignore[import-not-found]

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
            raise RuntimeError("root CROWN residual DLPack pointer differs")
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
                raise RuntimeError("root CROWN residual current stream differs")
            views = tuple(self._view(value) for value in (*sources, *outputs))
            self.compiled.executable[symbol](*views)
        count = len(sources) + len(outputs)
        self.pointer_count += count
        self.pointer_exact_count += count

    def forward(
        self, tensors: RootCrownResidualTensorsV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_runtime_structure(tensors, self.template)
        if self._output_a is None:
            self._main_a = torch.empty_like(tensors.incoming_lower_a)
            self._output_a = torch.empty_like(tensors.incoming_lower_a)
            self._output_bias = torch.empty(
                (self.template.spec_count, self.template.domain_count),
                dtype=torch.float32,
                device=tensors.incoming_lower_a.device,
            )
        output_a = self._output_a
        output_bias = self._output_bias
        main_a = self._main_a
        if output_bias is None or main_a is None:
            raise RuntimeError("root CROWN residual output arena differs")
        self._launch(
            self.template.forward_symbol,
            (
                tensors.incoming_lower_a,
                tensors.entry_lower,
                tensors.entry_upper,
                tensors.entry_raw_alpha,
                self._entry_map,
                tensors.main_conv_weight,
                tensors.main_conv_bias,
                tensors.inner_lower,
                tensors.inner_upper,
                tensors.inner_raw_alpha,
                self._inner_map,
                tensors.inner_conv_weight,
                tensors.inner_conv_bias,
            ),
            (main_a, output_a, output_bias),
        )
        self.forward_launch_count += 1
        return output_a, output_bias

    @property
    def last_main_a(self) -> torch.Tensor:
        """Return the host-required current inner-ReLU input coefficient."""

        if self._main_a is None:
            raise RuntimeError("root CROWN residual main state is absent")
        return self._main_a

    def _prepare_backward_arena(
        self, tensors: RootCrownResidualTensorsV1
    ) -> tuple[torch.Tensor, ...]:
        if self._incoming_gradient is None:
            self._incoming_gradient = torch.empty_like(tensors.incoming_lower_a)
            self._entry_lower_gradient = torch.empty_like(tensors.entry_lower)
            self._entry_upper_gradient = torch.empty_like(tensors.entry_upper)
            self._entry_alpha_gradient = torch.empty_like(tensors.entry_raw_alpha)
            self._inner_lower_gradient = torch.empty_like(tensors.inner_lower)
            self._inner_upper_gradient = torch.empty_like(tensors.inner_upper)
            self._inner_alpha_gradient = torch.empty_like(tensors.inner_raw_alpha)
        values = (
            self._incoming_gradient,
            self._entry_lower_gradient,
            self._entry_upper_gradient,
            self._entry_alpha_gradient,
            self._inner_lower_gradient,
            self._inner_upper_gradient,
            self._inner_alpha_gradient,
        )
        if any(value is None for value in values):
            raise RuntimeError("root CROWN residual backward arena differs")
        return tuple(value for value in values if value is not None)

    def backward(
        self,
        tensors: RootCrownResidualTensorsV1,
        output_a_gradient: torch.Tensor,
        output_bias_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        for name, value, shape in (
            ("output A gradient", output_a_gradient, self.template.coefficient_shape),
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
                raise ValueError(f"root CROWN residual {name} differs")
        outputs = self._prepare_backward_arena(tensors)
        self._launch(
            self.template.backward_symbol,
            (
                tensors.incoming_lower_a,
                tensors.entry_lower,
                tensors.entry_upper,
                tensors.entry_raw_alpha,
                self._entry_map,
                self._entry_coordinates,
                tensors.main_conv_weight,
                tensors.main_conv_bias,
                tensors.inner_lower,
                tensors.inner_upper,
                tensors.inner_raw_alpha,
                self._inner_map,
                self._inner_coordinates,
                tensors.inner_conv_weight,
                tensors.inner_conv_bias,
                output_a_gradient,
                output_bias_gradient,
            ),
            outputs,
        )
        self.backward_launch_count += 1
        return outputs

    def prepare(self) -> None:
        """Materialize CUDA modules and persistent arenas before timed execution."""

        if self.prepare_count:
            raise RuntimeError("root CROWN residual executor already prepared")
        device = torch.device("cuda")
        coefficient = torch.zeros(
            self.template.coefficient_shape, dtype=torch.float32, device=device
        )
        lower = torch.full(
            self.template.bound_shape, -1.0, dtype=torch.float32, device=device
        )
        upper = torch.full(
            self.template.bound_shape, 1.0, dtype=torch.float32, device=device
        )
        entry_alpha = torch.full(
            (
                2,
                self.template.spec_count,
                self.template.domain_count,
                self.template.entry_alpha_count,
            ),
            0.5,
            dtype=torch.float32,
            device=device,
        )
        inner_alpha = torch.full(
            (
                2,
                self.template.spec_count,
                self.template.domain_count,
                self.template.inner_alpha_count,
            ),
            0.5,
            dtype=torch.float32,
            device=device,
        )
        weight = torch.zeros(
            self.template.weight_shape, dtype=torch.float32, device=device
        )
        bias = torch.zeros(
            (self.template.channels,), dtype=torch.float32, device=device
        )
        tensors = RootCrownResidualTensorsV1(
            coefficient,
            lower,
            upper,
            entry_alpha,
            weight,
            bias,
            lower,
            upper,
            inner_alpha,
            weight,
            bias,
        )
        output_a, output_bias = self.forward(tensors)
        self.backward(
            tensors,
            torch.zeros_like(output_a),
            torch.zeros_like(output_bias),
        )
        torch.cuda.synchronize(device)
        persistent_pointers = {
            value.data_ptr()
            for value in (
                self._main_a,
                self._output_a,
                self._output_bias,
                self._incoming_gradient,
                self._entry_lower_gradient,
                self._entry_upper_gradient,
                self._entry_alpha_gradient,
                self._inner_lower_gradient,
                self._inner_upper_gradient,
                self._inner_alpha_gradient,
            )
            if value is not None
        }
        self._view_cache = {
            key: view
            for key, view in self._view_cache.items()
            if key[0] in persistent_pointers
        }
        self.forward_launch_count = 0
        self.backward_launch_count = 0
        self.pointer_count = 0
        self.pointer_exact_count = 0
        self.prepare_count = 1


class _RootCrownResidualTIRFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        incoming_lower_a: torch.Tensor,
        entry_lower: torch.Tensor,
        entry_upper: torch.Tensor,
        entry_raw_alpha: torch.Tensor,
        main_conv_weight: torch.Tensor,
        main_conv_bias: torch.Tensor,
        inner_lower: torch.Tensor,
        inner_upper: torch.Tensor,
        inner_raw_alpha: torch.Tensor,
        inner_conv_weight: torch.Tensor,
        inner_conv_bias: torch.Tensor,
        executor: RootCrownResidualTIRExecutorV1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = RootCrownResidualTensorsV1(
            incoming_lower_a,
            entry_lower,
            entry_upper,
            entry_raw_alpha,
            main_conv_weight,
            main_conv_bias,
            inner_lower,
            inner_upper,
            inner_raw_alpha,
            inner_conv_weight,
            inner_conv_bias,
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
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("root CROWN residual higher-order gradient unsupported")
        gradients = ctx.executor.backward(
            ctx.tensors,
            output_a_gradient.contiguous(),
            output_bias_gradient.contiguous(),
        )
        (
            incoming_gradient,
            entry_lower_gradient,
            entry_upper_gradient,
            entry_alpha_gradient,
            inner_lower_gradient,
            inner_upper_gradient,
            inner_alpha_gradient,
        ) = gradients
        return (
            incoming_gradient,
            entry_lower_gradient,
            entry_upper_gradient,
            entry_alpha_gradient,
            None,
            None,
            inner_lower_gradient,
            inner_upper_gradient,
            inner_alpha_gradient,
            None,
            None,
            None,
        )


def execute_root_crown_residual_tir_v1(
    tensors: RootCrownResidualTensorsV1,
    executor: RootCrownResidualTIRExecutorV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute one fused residual forward and attach the compiled full VJP."""

    return _RootCrownResidualTIRFunction.apply(
        tensors.incoming_lower_a,
        tensors.entry_lower,
        tensors.entry_upper,
        tensors.entry_raw_alpha,
        tensors.main_conv_weight,
        tensors.main_conv_bias,
        tensors.inner_lower,
        tensors.inner_upper,
        tensors.inner_raw_alpha,
        tensors.inner_conv_weight,
        tensors.inner_conv_bias,
        executor,
    )


__all__ = [
    "RootCrownResidualTIRExecutorV1",
    "RootCrownResidualTensorsV1",
    "execute_root_crown_residual_tir_v1",
    "validate_root_crown_residual_tensors_v1",
]
