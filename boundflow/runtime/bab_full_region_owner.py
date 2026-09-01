"""Activation-BaB full-region owner with recompute-only custom backward.

The owner spans terminal Linear, a residual block, a projection residual and
the input-domain concretization.  It keeps the solver-visible state compact:
six sparse alpha tensors plus one sparse beta tensor.  Dense coefficients are
generated inside the transaction and are never saved across the autograd
boundary.
"""

# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals
# pylint: disable=missing-function-docstring,abstract-method,arguments-differ
# pylint: disable=too-many-boolean-expressions,not-callable
# pylint: disable=import-outside-toplevel

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, Sequence

import torch
import torch.nn.functional as torch_functional

BAB_ALPHA_OWNER_COUNT = 6
BAB_MUTABLE_OWNER_COUNT = BAB_ALPHA_OWNER_COUNT + 1


def _coordinates_1d(values: Sequence[torch.Tensor]) -> tuple[int, ...]:
    if len(values) != 1 or values[0].ndim != 1:
        raise ValueError("activation-BaB linear alpha coordinates differ")
    return tuple(int(value) for value in values[0].tolist())


def _coordinates_3d(
    values: Sequence[torch.Tensor],
) -> tuple[tuple[int, int, int], ...]:
    if len(values) != 3 or any(value.ndim != 1 for value in values):
        raise ValueError("activation-BaB convolution alpha coordinates differ")
    lengths = {int(value.numel()) for value in values}
    if len(lengths) != 1:
        raise ValueError("activation-BaB convolution alpha coordinate count differs")
    return cast(
        tuple[tuple[int, int, int], ...],
        tuple(
            tuple(int(values[axis][ordinal]) for axis in range(3))
            for ordinal in range(int(values[0].numel()))
        ),
    )


def _shape3(value: torch.Tensor, *, name: str) -> tuple[int, int, int]:
    shape = tuple(value.shape[1:])
    if len(shape) != 3:
        raise ValueError(f"activation-BaB {name} rank differs")
    return cast(tuple[int, int, int], shape)


def _flat_coordinate_indices(
    coordinates: tuple[tuple[int, int, int], ...],
    shape: tuple[int, int, int],
) -> tuple[int, ...]:
    channels, height, width = shape
    result = tuple(
        channel * height * width + row * width + column
        for channel, row, column in coordinates
    )
    if (
        not result
        or tuple(sorted(result)) != result
        or len(set(result)) != len(result)
        or min(result) < 0
        or max(result) >= channels * height * width
    ):
        raise ValueError("activation-BaB alpha coordinate legality differs")
    return result


def _dense_alpha_1d(
    raw_alpha: torch.Tensor, coordinates: tuple[int, ...], feature_count: int
) -> torch.Tensor:
    indices = torch.tensor(
        coordinates, dtype=torch.int64, device=raw_alpha.device
    ).view(1, 1, 1, -1)
    return torch.zeros(
        (*raw_alpha.shape[:3], feature_count),
        dtype=raw_alpha.dtype,
        device=raw_alpha.device,
    ).scatter(3, indices.expand_as(raw_alpha), raw_alpha)


def _dense_alpha_3d(
    raw_alpha: torch.Tensor,
    coordinates: tuple[tuple[int, int, int], ...],
    shape: tuple[int, int, int],
) -> torch.Tensor:
    indices = torch.tensor(
        _flat_coordinate_indices(coordinates, shape),
        dtype=torch.int64,
        device=raw_alpha.device,
    ).view(1, 1, 1, -1)
    dense = torch.zeros(
        (*raw_alpha.shape[:3], shape[0] * shape[1] * shape[2]),
        dtype=raw_alpha.dtype,
        device=raw_alpha.device,
    ).scatter(3, indices.expand_as(raw_alpha), raw_alpha)
    return dense.reshape(*raw_alpha.shape[:3], *shape)


def _relu_terms(
    incoming: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    dense_alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = (upper - lower).clamp_min(torch.finfo(torch.float32).eps)
    upper_slope = torch.where(
        lower >= 0,
        torch.ones_like(lower),
        torch.where(upper <= 0, torch.zeros_like(lower), upper / denominator),
    )
    ambiguous = (lower < 0) & (upper > 0)
    lower_slope = torch.where(
        ambiguous,
        dense_alpha[0].clamp(0, 1),
        (lower >= 0).to(lower.dtype),
    )
    slope = torch.where(incoming >= 0, lower_slope, upper_slope)
    intercept = torch.where(
        (incoming < 0) & ambiguous,
        -lower * upper_slope,
        torch.zeros_like(incoming),
    )
    return incoming * slope, incoming * intercept


@dataclass(frozen=True)
class BabResidualStaticV1:
    """Frozen tensors and sparse layouts for one same-shape residual block."""

    entry_lower: torch.Tensor
    entry_upper: torch.Tensor
    entry_coordinates: tuple[tuple[int, int, int], ...]
    main_weight: torch.Tensor
    main_bias: torch.Tensor
    inner_lower: torch.Tensor
    inner_upper: torch.Tensor
    inner_coordinates: tuple[tuple[int, int, int], ...]
    inner_weight: torch.Tensor
    inner_bias: torch.Tensor


@dataclass(frozen=True)
class BabProjectionStaticV1:
    """Frozen tensors and sparse layouts for one projection residual block."""

    entry_lower: torch.Tensor
    entry_upper: torch.Tensor
    entry_coordinates: tuple[tuple[int, int, int], ...]
    outer_weight: torch.Tensor
    outer_bias: torch.Tensor
    inner_lower: torch.Tensor
    inner_upper: torch.Tensor
    inner_coordinates: tuple[tuple[int, int, int], ...]
    inner_weight: torch.Tensor
    inner_bias: torch.Tensor
    skip_weight: torch.Tensor
    skip_bias: torch.Tensor


@dataclass(frozen=True)
class BabInputStaticV1:
    """Frozen input Conv, interval and perturbation payload."""

    lower: torch.Tensor
    upper: torch.Tensor
    coordinates: tuple[tuple[int, int, int], ...]
    weight: torch.Tensor
    bias: torch.Tensor
    center: torch.Tensor
    radius: torch.Tensor


@dataclass(frozen=True)
class BabFullRegionStaticV1:
    """All immutable data for one shape-specialized activation-BaB region."""

    terminal_lower: torch.Tensor
    terminal_upper: torch.Tensor
    terminal_coordinates: tuple[int, ...]
    terminal_weight: torch.Tensor
    terminal_bias: torch.Tensor
    beta_locations: torch.Tensor
    beta_signs: torch.Tensor
    residual: BabResidualStaticV1
    projection: BabProjectionStaticV1
    input_domain: BabInputStaticV1

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return the complete frozen tensor inventory."""

        return (
            self.terminal_lower,
            self.terminal_upper,
            self.terminal_weight,
            self.terminal_bias,
            self.beta_locations,
            self.beta_signs,
            self.residual.entry_lower,
            self.residual.entry_upper,
            self.residual.main_weight,
            self.residual.main_bias,
            self.residual.inner_lower,
            self.residual.inner_upper,
            self.residual.inner_weight,
            self.residual.inner_bias,
            self.projection.entry_lower,
            self.projection.entry_upper,
            self.projection.outer_weight,
            self.projection.outer_bias,
            self.projection.inner_lower,
            self.projection.inner_upper,
            self.projection.inner_weight,
            self.projection.inner_bias,
            self.projection.skip_weight,
            self.projection.skip_bias,
            self.input_domain.lower,
            self.input_domain.upper,
            self.input_domain.weight,
            self.input_domain.bias,
            self.input_domain.center,
            self.input_domain.radius,
        )

    def validate(self, dynamic: "BabFullRegionDynamicV1") -> None:
        """Fail closed on the captured topology and ownership boundary."""

        dynamic.validate()
        device = dynamic.terminal_incoming.device
        dtype = dynamic.terminal_incoming.dtype
        frozen = self.tensors()
        if (
            dtype != torch.float32
            or any(value.device != device for value in frozen)
            or any(value.requires_grad for value in frozen)
            or any(
                value.dtype != torch.float32
                for value in frozen
                if value is not self.beta_locations
            )
            or self.beta_locations.dtype != torch.int64
            or not all(value.is_contiguous() for value in frozen)
        ):
            raise ValueError("activation-BaB full-region frozen owner differs")
        spec, domain, terminal_features = dynamic.terminal_incoming.shape
        residual_shape = _shape3(self.residual.entry_lower, name="residual bound")
        projection_output_shape = _shape3(self.input_domain.lower, name="input bound")
        if (
            tuple(self.terminal_lower.shape) != (domain, terminal_features)
            or self.terminal_upper.shape != self.terminal_lower.shape
            or tuple(self.terminal_weight.shape[:1]) != (terminal_features,)
            or tuple(self.terminal_bias.shape) != (terminal_features,)
            or self.terminal_weight.shape[1]
            != residual_shape[0] * residual_shape[1] * residual_shape[2]
            or tuple(self.beta_locations.shape) != tuple(dynamic.beta.shape)
            or self.beta_signs.shape != dynamic.beta.shape
            or len(self.terminal_coordinates) != dynamic.terminal_alpha.shape[-1]
            or tuple(self.residual.entry_lower.shape) != (domain, *residual_shape)
            or self.residual.entry_upper.shape != self.residual.entry_lower.shape
            or self.residual.inner_lower.shape != self.residual.entry_lower.shape
            or self.residual.inner_upper.shape != self.residual.entry_lower.shape
            or tuple(self.projection.entry_lower.shape) != (domain, *residual_shape)
            or self.projection.entry_upper.shape != self.projection.entry_lower.shape
            or self.projection.inner_lower.shape != self.projection.entry_lower.shape
            or self.projection.inner_upper.shape != self.projection.entry_lower.shape
            or tuple(self.input_domain.lower.shape)
            != (domain, *projection_output_shape)
            or self.input_domain.upper.shape != self.input_domain.lower.shape
            or tuple(self.input_domain.center.shape[:1]) != (domain,)
            or self.input_domain.radius.shape != self.input_domain.center.shape
            or spec < 1
            or domain < 1
        ):
            raise ValueError("activation-BaB full-region topology differs")
        for coordinates, alpha, shape in (
            (
                self.residual.entry_coordinates,
                dynamic.residual_entry_alpha,
                residual_shape,
            ),
            (
                self.residual.inner_coordinates,
                dynamic.residual_inner_alpha,
                residual_shape,
            ),
            (
                self.projection.entry_coordinates,
                dynamic.projection_entry_alpha,
                residual_shape,
            ),
            (
                self.projection.inner_coordinates,
                dynamic.projection_inner_alpha,
                residual_shape,
            ),
            (
                self.input_domain.coordinates,
                dynamic.input_alpha,
                projection_output_shape,
            ),
        ):
            _flat_coordinate_indices(coordinates, shape)
            if len(coordinates) != alpha.shape[-1]:
                raise ValueError("activation-BaB sparse alpha ownership differs")
        if bool(
            (
                (self.beta_locations < 0)
                | (self.beta_locations >= terminal_features)
                | ~torch.isin(self.beta_signs, self.beta_signs.new_tensor((-1.0, 1.0)))
            )
            .any()
            .item()
        ):
            raise ValueError("activation-BaB sparse beta legality differs")


@dataclass(frozen=True)
class BabFullRegionDynamicV1:
    """The only differentiable owners crossing the full-region boundary."""

    terminal_incoming: torch.Tensor
    terminal_alpha: torch.Tensor
    residual_entry_alpha: torch.Tensor
    residual_inner_alpha: torch.Tensor
    projection_entry_alpha: torch.Tensor
    projection_inner_alpha: torch.Tensor
    input_alpha: torch.Tensor
    beta: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.terminal_incoming,
            self.terminal_alpha,
            self.residual_entry_alpha,
            self.residual_inner_alpha,
            self.projection_entry_alpha,
            self.projection_inner_alpha,
            self.input_alpha,
            self.beta,
        )

    def validate(self) -> None:
        tensors = self.tensors()
        devices = {value.device for value in tensors}
        if (
            self.terminal_incoming.ndim != 3
            or any(value.ndim != 4 for value in tensors[1:7])
            or self.beta.ndim != 2
            or len(devices) != 1
            or any(value.dtype != torch.float32 for value in tensors)
            or any(not value.is_contiguous() for value in tensors)
            or any(value.shape[0] != 2 for value in tensors[1:7])
            or any(
                tuple(value.shape[1:3]) != tuple(self.terminal_alpha.shape[1:3])
                for value in tensors[2:7]
            )
            or tuple(self.terminal_alpha.shape[1:3])
            != tuple(self.terminal_incoming.shape[:2])
            or self.beta.shape[0] != self.terminal_incoming.shape[1]
        ):
            raise ValueError("activation-BaB mutable owner differs")


@dataclass(frozen=True)
class BabFullRegionTraceV1:
    """Intermediate oracle trace; never retained by the custom backward."""

    terminal_a: torch.Tensor
    terminal_bias: torch.Tensor
    residual_a: torch.Tensor
    residual_bias: torch.Tensor
    projection_a: torch.Tensor
    projection_bias: torch.Tensor
    concrete: torch.Tensor
    input_bias: torch.Tensor
    final_lower: torch.Tensor


def _residual(
    incoming: torch.Tensor,
    entry_alpha: torch.Tensor,
    inner_alpha: torch.Tensor,
    static: BabResidualStaticV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = _shape3(static.entry_lower, name="residual bound")
    entry_dense = _dense_alpha_3d(entry_alpha, static.entry_coordinates, shape)
    entry_a, entry_intercept = _relu_terms(
        incoming, static.entry_lower, static.entry_upper, entry_dense
    )
    spec, domain, channels, height, width = incoming.shape
    merged_shape = (spec * domain, channels, height, width)
    main_a = torch_functional.conv_transpose2d(
        entry_a.reshape(merged_shape), static.main_weight, padding=1
    ).reshape_as(incoming)
    inner_dense = _dense_alpha_3d(inner_alpha, static.inner_coordinates, shape)
    inner_a, inner_intercept = _relu_terms(
        main_a, static.inner_lower, static.inner_upper, inner_dense
    )
    residual_a = torch_functional.conv_transpose2d(
        inner_a.reshape(merged_shape), static.inner_weight, padding=1
    ).reshape_as(incoming)
    bias = (
        entry_intercept.sum(dim=(-3, -2, -1))
        + (entry_a * static.main_bias.view(1, 1, channels, 1, 1)).sum(dim=(-3, -2, -1))
        + inner_intercept.sum(dim=(-3, -2, -1))
        + (inner_a * static.inner_bias.view(1, 1, channels, 1, 1)).sum(dim=(-3, -2, -1))
    )
    return entry_a + residual_a, bias


def _projection(
    incoming: torch.Tensor,
    entry_alpha: torch.Tensor,
    inner_alpha: torch.Tensor,
    static: BabProjectionStaticV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = _shape3(static.entry_lower, name="projection bound")
    entry_dense = _dense_alpha_3d(entry_alpha, static.entry_coordinates, shape)
    entry_a, entry_intercept = _relu_terms(
        incoming, static.entry_lower, static.entry_upper, entry_dense
    )
    spec, domain, channels, height, width = incoming.shape
    merged_shape = (spec * domain, channels, height, width)
    main_a = torch_functional.conv_transpose2d(
        entry_a.reshape(merged_shape), static.outer_weight, padding=1
    ).reshape_as(incoming)
    inner_dense = _dense_alpha_3d(inner_alpha, static.inner_coordinates, shape)
    inner_a, inner_intercept = _relu_terms(
        main_a, static.inner_lower, static.inner_upper, inner_dense
    )
    output_channels = int(static.inner_weight.shape[1])
    output_shape = (spec, domain, output_channels, height * 2, width * 2)
    main_output = torch_functional.conv_transpose2d(
        inner_a.reshape(merged_shape),
        static.inner_weight,
        stride=2,
        padding=1,
        output_padding=1,
    ).reshape(output_shape)
    skip_output = torch_functional.conv_transpose2d(
        entry_a.reshape(merged_shape),
        static.skip_weight,
        stride=2,
        output_padding=1,
    ).reshape(output_shape)
    bias = (
        entry_intercept.sum(dim=(-3, -2, -1))
        + (entry_a * static.outer_bias.view(1, 1, channels, 1, 1)).sum(dim=(-3, -2, -1))
        + inner_intercept.sum(dim=(-3, -2, -1))
        + (inner_a * static.inner_bias.view(1, 1, channels, 1, 1)).sum(dim=(-3, -2, -1))
        + (entry_a * static.skip_bias.view(1, 1, channels, 1, 1)).sum(dim=(-3, -2, -1))
    )
    return main_output + skip_output, bias


def _input_domain(
    incoming: torch.Tensor,
    alpha: torch.Tensor,
    static: BabInputStaticV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = _shape3(static.lower, name="input bound")
    dense = _dense_alpha_3d(alpha, static.coordinates, shape)
    transformed, intercept = _relu_terms(incoming, static.lower, static.upper, dense)
    spec, domain, channels, height, width = incoming.shape
    coefficient = torch_functional.conv_transpose2d(
        transformed.reshape(spec * domain, channels, height, width),
        static.weight,
        stride=2,
        padding=1,
        output_padding=1,
    ).reshape(spec, domain, *static.center.shape[1:])
    concrete = (
        (
            coefficient * static.center.unsqueeze(0)
            - coefficient.abs() * static.radius.unsqueeze(0)
        )
        .sum(dim=(-3, -2, -1))
        .transpose(0, 1)
    )
    bias = intercept.sum(dim=(-3, -2, -1)) + (
        transformed * static.bias.view(1, 1, channels, 1, 1)
    ).sum(dim=(-3, -2, -1))
    return concrete, bias


def evaluate_bab_full_region_trace_v1(
    dynamic: BabFullRegionDynamicV1,
    static: BabFullRegionStaticV1,
    *,
    terminal_executor: Any | None = None,
    residual_executor: Any | None = None,
    projection_executor: Any | None = None,
) -> BabFullRegionTraceV1:
    """Evaluate the full activation-BaB region and expose an oracle trace."""

    static.validate(dynamic)
    if terminal_executor is None:
        terminal_dense = _dense_alpha_1d(
            dynamic.terminal_alpha,
            static.terminal_coordinates,
            dynamic.terminal_incoming.shape[-1],
        )
        relu_a, relu_intercept = _relu_terms(
            dynamic.terminal_incoming,
            static.terminal_lower,
            static.terminal_upper,
            terminal_dense,
        )
        beta_delta = torch.zeros_like(relu_a).scatter_add(
            2,
            static.beta_locations.unsqueeze(0).expand(relu_a.shape[0], -1, -1),
            (-dynamic.beta * static.beta_signs)
            .unsqueeze(0)
            .expand(relu_a.shape[0], -1, -1),
        )
        terminal_linear_incoming = relu_a + beta_delta
        terminal_a = terminal_linear_incoming.matmul(static.terminal_weight)
        terminal_bias = relu_intercept.sum(dim=-1) + (
            terminal_linear_incoming * static.terminal_bias
        ).sum(dim=-1)
    else:
        from boundflow.runtime.bab_terminal_tir import (
            BabTerminalTensorsV1,
            execute_bab_terminal_tir_v1,
        )

        terminal_a, terminal_bias = execute_bab_terminal_tir_v1(
            BabTerminalTensorsV1(
                incoming_lower_a=dynamic.terminal_incoming,
                preactivation_lower=static.terminal_lower,
                preactivation_upper=static.terminal_upper,
                compressed_alpha=dynamic.terminal_alpha,
                sparse_beta=dynamic.beta,
                beta_location=static.beta_locations,
                beta_sign=static.beta_signs,
                linear_weight=static.terminal_weight,
                linear_bias=static.terminal_bias,
            ),
            terminal_executor,
        )
    residual_shape = _shape3(static.residual.entry_lower, name="residual bound")
    residual_incoming = terminal_a.reshape(
        terminal_a.shape[0], terminal_a.shape[1], *residual_shape
    )
    if residual_executor is None:
        residual_a, residual_bias = _residual(
            residual_incoming,
            dynamic.residual_entry_alpha,
            dynamic.residual_inner_alpha,
            static.residual,
        )
    else:
        from boundflow.runtime.root_crown_residual_tir import (
            execute_root_crown_residual_tir_v1,
            RootCrownResidualTensorsV1,
        )

        residual_a, residual_bias = execute_root_crown_residual_tir_v1(
            RootCrownResidualTensorsV1(
                incoming_lower_a=residual_incoming,
                entry_lower=static.residual.entry_lower,
                entry_upper=static.residual.entry_upper,
                entry_raw_alpha=dynamic.residual_entry_alpha,
                main_conv_weight=static.residual.main_weight,
                main_conv_bias=static.residual.main_bias,
                inner_lower=static.residual.inner_lower,
                inner_upper=static.residual.inner_upper,
                inner_raw_alpha=dynamic.residual_inner_alpha,
                inner_conv_weight=static.residual.inner_weight,
                inner_conv_bias=static.residual.inner_bias,
            ),
            residual_executor,
        )
    if projection_executor is None:
        projection_a, projection_bias = _projection(
            residual_a,
            dynamic.projection_entry_alpha,
            dynamic.projection_inner_alpha,
            static.projection,
        )
    else:
        from boundflow.runtime.root_crown_projection_tir import (
            execute_root_crown_projection_tir_v1,
            RootCrownProjectionTensorsV1,
        )

        projection_a, projection_bias = execute_root_crown_projection_tir_v1(
            RootCrownProjectionTensorsV1(
                incoming_lower_a=residual_a,
                entry_lower=static.projection.entry_lower,
                entry_upper=static.projection.entry_upper,
                entry_raw_alpha=dynamic.projection_entry_alpha,
                main_outer_conv_weight=static.projection.outer_weight,
                main_outer_conv_bias=static.projection.outer_bias,
                inner_lower=static.projection.inner_lower,
                inner_upper=static.projection.inner_upper,
                inner_raw_alpha=dynamic.projection_inner_alpha,
                main_inner_conv_weight=static.projection.inner_weight,
                main_inner_conv_bias=static.projection.inner_bias,
                skip_conv_weight=static.projection.skip_weight,
                skip_conv_bias=static.projection.skip_bias,
            ),
            projection_executor,
        )
    concrete, input_bias = _input_domain(
        projection_a, dynamic.input_alpha, static.input_domain
    )
    total_bias = terminal_bias + residual_bias + projection_bias + input_bias
    return BabFullRegionTraceV1(
        terminal_a=terminal_a,
        terminal_bias=terminal_bias,
        residual_a=residual_a,
        residual_bias=residual_bias,
        projection_a=projection_a,
        projection_bias=projection_bias,
        concrete=concrete,
        input_bias=input_bias,
        final_lower=concrete + total_bias.transpose(0, 1),
    )


class _BabFullRegionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        terminal_incoming: torch.Tensor,
        terminal_alpha: torch.Tensor,
        residual_entry_alpha: torch.Tensor,
        residual_inner_alpha: torch.Tensor,
        projection_entry_alpha: torch.Tensor,
        projection_inner_alpha: torch.Tensor,
        input_alpha: torch.Tensor,
        beta: torch.Tensor,
        owner: "PreparedBabFullRegionOwnerV1",
    ) -> torch.Tensor:
        dynamic = BabFullRegionDynamicV1(
            terminal_incoming,
            terminal_alpha,
            residual_entry_alpha,
            residual_inner_alpha,
            projection_entry_alpha,
            projection_inner_alpha,
            input_alpha,
            beta,
        )
        owner.static.validate(dynamic)
        ctx.save_for_backward(*dynamic.tensors())
        ctx.owner = owner
        ctx.set_materialize_grads(False)
        owner.forward_count += 1
        return evaluate_bab_full_region_trace_v1(
            dynamic,
            owner.static,
            terminal_executor=owner.terminal_executor,
            residual_executor=owner.residual_executor,
            projection_executor=owner.projection_executor,
        ).final_lower

    @staticmethod
    def backward(ctx: Any, lower_gradient: torch.Tensor) -> tuple[Any, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("activation-BaB higher-order gradient unsupported")
        if lower_gradient is None:
            raise RuntimeError("activation-BaB lower gradient is absent")
        leaves = tuple(
            value.detach().requires_grad_(True) for value in ctx.saved_tensors
        )
        with torch.enable_grad():
            dynamic = BabFullRegionDynamicV1(*leaves)
            lower = evaluate_bab_full_region_trace_v1(
                dynamic,
                ctx.owner.static,
                terminal_executor=ctx.owner.terminal_executor,
                residual_executor=ctx.owner.residual_executor,
                projection_executor=ctx.owner.projection_executor,
            ).final_lower
            gradients = torch.autograd.grad(
                lower,
                leaves,
                grad_outputs=lower_gradient.contiguous(),
            )
        ctx.owner.backward_count += 1
        return (*gradients, None)


@dataclass(frozen=True)
class BabFullRegionOwnerReceiptV1:
    """Structural evidence for the no-dense-save correctness owner."""

    forward_count: int
    backward_count: int
    mutable_owner_count: int = BAB_MUTABLE_OWNER_COUNT
    saved_dense_coefficient_count: int = 0
    frozen_bound_gradient_count: int = 0
    fallback_count: int = 0
    terminal_backend: str = "pytorch-reference"
    terminal_forward_launch_count: int = 0
    terminal_backward_launch_count: int = 0
    compiled_segment_count: int = 0
    compiled_forward_launch_count: int = 0
    compiled_backward_launch_count: int = 0
    timing_recorded: bool = False
    performance_claimed: bool = False

    def validate(self) -> None:
        if (
            self.forward_count < 1
            or self.backward_count < 0
            or self.backward_count > self.forward_count
            or self.mutable_owner_count != BAB_MUTABLE_OWNER_COUNT
            or self.saved_dense_coefficient_count
            or self.frozen_bound_gradient_count
            or self.fallback_count
            or self.terminal_backend
            not in {"pytorch-reference", "tvm-beta-terminal-v1"}
            or (
                self.terminal_backend == "pytorch-reference"
                and (
                    self.terminal_forward_launch_count
                    or self.terminal_backward_launch_count
                )
            )
            or (
                self.terminal_backend == "tvm-beta-terminal-v1"
                and (
                    self.terminal_forward_launch_count < self.forward_count
                    or self.terminal_backward_launch_count < self.backward_count
                )
            )
            or self.compiled_segment_count not in {0, 1, 2, 3}
            or (
                self.compiled_segment_count == 0
                and (
                    self.compiled_forward_launch_count
                    or self.compiled_backward_launch_count
                )
            )
            or (
                self.compiled_segment_count > 0
                and (
                    self.compiled_forward_launch_count
                    < self.compiled_segment_count * self.forward_count
                    or self.compiled_backward_launch_count
                    < self.compiled_segment_count * self.backward_count
                )
            )
            or self.timing_recorded
            or self.performance_claimed
        ):
            raise ValueError("activation-BaB full-region owner receipt differs")


class PreparedBabFullRegionOwnerV1:
    """Reusable correctness owner for a fixed production topology."""

    def __init__(
        self,
        static: BabFullRegionStaticV1,
        *,
        terminal_executor: Any | None = None,
        residual_executor: Any | None = None,
        projection_executor: Any | None = None,
    ) -> None:
        self.static = static
        self.terminal_executor = terminal_executor
        self.residual_executor = residual_executor
        self.projection_executor = projection_executor
        self.forward_count = 0
        self.backward_count = 0

    def evaluate(self, dynamic: BabFullRegionDynamicV1) -> torch.Tensor:
        """Execute one evaluation behind the single custom-autograd boundary."""

        self.static.validate(dynamic)
        return _BabFullRegionFunction.apply(*dynamic.tensors(), self)

    def receipt(self) -> BabFullRegionOwnerReceiptV1:
        backend = (
            "pytorch-reference"
            if self.terminal_executor is None
            else "tvm-beta-terminal-v1"
        )
        executors = tuple(
            value
            for value in (
                self.terminal_executor,
                self.residual_executor,
                self.projection_executor,
            )
            if value is not None
        )
        result = BabFullRegionOwnerReceiptV1(
            forward_count=self.forward_count,
            backward_count=self.backward_count,
            fallback_count=(
                0
                if self.terminal_executor is None
                else int(self.terminal_executor.fallback_count)
            ),
            terminal_backend=backend,
            terminal_forward_launch_count=(
                0
                if self.terminal_executor is None
                else int(self.terminal_executor.forward_launch_count)
            ),
            terminal_backward_launch_count=(
                0
                if self.terminal_executor is None
                else int(self.terminal_executor.backward_launch_count)
            ),
            compiled_segment_count=len(executors),
            compiled_forward_launch_count=sum(
                int(value.forward_launch_count) for value in executors
            ),
            compiled_backward_launch_count=sum(
                int(value.backward_launch_count) for value in executors
            ),
        )
        result.validate()
        return result


def bab_full_region_inputs_from_capture_v1(
    segments: dict[str, Any], ordinal: int, *, device: torch.device | str
) -> tuple[BabFullRegionDynamicV1, BabFullRegionStaticV1]:
    """Build a correctness instance from the versioned production capture."""

    if tuple(sorted(segments)) != (
        "input_domain",
        "projection",
        "residual",
        "terminal",
    ):
        raise ValueError("activation-BaB captured segment inventory differs")

    def evaluation(name: str) -> dict[str, Any]:
        values = segments[name].get("evaluations")
        if not isinstance(values, list) or len(values) != 10:
            raise ValueError("activation-BaB captured evaluation inventory differs")
        result = values[ordinal]
        if not isinstance(result, dict) or result.get("ordinal") != ordinal:
            raise ValueError("activation-BaB captured evaluation order differs")
        return result

    def frozen(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device).contiguous()

    def mutable(value: torch.Tensor) -> torch.Tensor:
        return frozen(value).clone().requires_grad_(True)

    terminal = evaluation("terminal")
    residual = evaluation("residual")
    projection = evaluation("projection")
    input_domain = evaluation("input_domain")
    beta_rows = segments["terminal"].get("beta_evidence")
    if not isinstance(beta_rows, list) or len(beta_rows) != 10:
        raise ValueError("activation-BaB captured beta inventory differs")
    beta = beta_rows[ordinal]
    dynamic = BabFullRegionDynamicV1(
        terminal_incoming=mutable(terminal["incoming_lower_a"]),
        terminal_alpha=mutable(terminal["raw_alpha"]),
        residual_entry_alpha=mutable(residual["entry_raw_alpha"]),
        residual_inner_alpha=mutable(residual["inner_raw_alpha"]),
        projection_entry_alpha=mutable(projection["entry_raw_alpha"]),
        projection_inner_alpha=mutable(projection["inner_raw_alpha"]),
        input_alpha=mutable(input_domain["raw_alpha"]),
        beta=mutable(beta["value"]),
    )
    static = BabFullRegionStaticV1(
        terminal_lower=frozen(terminal["preactivation_lower"]),
        terminal_upper=frozen(terminal["preactivation_upper"]),
        terminal_coordinates=_coordinates_1d(terminal["alpha_feature_indices"]),
        terminal_weight=frozen(terminal["operator_weight"]),
        terminal_bias=frozen(terminal["operator_bias"]),
        beta_locations=frozen(beta["location"]).to(torch.int64),
        beta_signs=frozen(beta["sign"]),
        residual=BabResidualStaticV1(
            entry_lower=frozen(residual["entry_lower"]),
            entry_upper=frozen(residual["entry_upper"]),
            entry_coordinates=_coordinates_3d(residual["entry_alpha_feature_indices"]),
            main_weight=frozen(residual["main_conv_weight"]),
            main_bias=frozen(residual["main_conv_bias"]),
            inner_lower=frozen(residual["inner_lower"]),
            inner_upper=frozen(residual["inner_upper"]),
            inner_coordinates=_coordinates_3d(residual["inner_alpha_feature_indices"]),
            inner_weight=frozen(residual["inner_conv_weight"]),
            inner_bias=frozen(residual["inner_conv_bias"]),
        ),
        projection=BabProjectionStaticV1(
            entry_lower=frozen(projection["entry_lower"]),
            entry_upper=frozen(projection["entry_upper"]),
            entry_coordinates=_coordinates_3d(
                projection["entry_alpha_feature_indices"]
            ),
            outer_weight=frozen(projection["main_outer_conv_weight"]),
            outer_bias=frozen(projection["main_outer_conv_bias"]),
            inner_lower=frozen(projection["inner_lower"]),
            inner_upper=frozen(projection["inner_upper"]),
            inner_coordinates=_coordinates_3d(
                projection["inner_alpha_feature_indices"]
            ),
            inner_weight=frozen(projection["main_inner_conv_weight"]),
            inner_bias=frozen(projection["main_inner_conv_bias"]),
            skip_weight=frozen(projection["skip_conv_weight"]),
            skip_bias=frozen(projection["skip_conv_bias"]),
        ),
        input_domain=BabInputStaticV1(
            lower=frozen(input_domain["preactivation_lower"]),
            upper=frozen(input_domain["preactivation_upper"]),
            coordinates=_coordinates_3d(input_domain["alpha_feature_indices"]),
            weight=frozen(input_domain["operator_weight"]),
            bias=frozen(input_domain["operator_bias"]),
            center=frozen(input_domain["input_center"]),
            radius=frozen(
                (input_domain["input_upper"] - input_domain["input_lower"]) * 0.5
            ),
        ),
    )
    static.validate(dynamic)
    return dynamic, static


__all__ = [
    "bab_full_region_inputs_from_capture_v1",
    "evaluate_bab_full_region_trace_v1",
    "BabFullRegionDynamicV1",
    "BabFullRegionOwnerReceiptV1",
    "BabFullRegionStaticV1",
    "BabFullRegionTraceV1",
    "PreparedBabFullRegionOwnerV1",
]
