"""Provider-independent RVIR-v4 native lA and intermediate-bound export."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=too-many-instance-attributes,missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import torch

from ..domains.interval import IntervalState
from ..ir.task import BFTaskModule
from .alpha_beta_crown import BetaState, _beta_to_relu_pre_add_coeff
from .crown_ibp import (
    _forward_ibp_trace_mlp,
    run_crown_ibp_mlp_with_relu_lower_coefficients_from_forward_trace,
)
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizationState
from .rvir_v4_pre_state_initializer import ProductionReluTopologyV4
from .rvir_v4_production_state import production_tensor_sha256
from .task_executor import InputSpec

NATIVE_BACKWARD_EXPORT_SCHEMA = "boundflow.rvir-v4-native-backward-export/v1"
PARITY_ATOL = 2e-4
PARITY_RTOL = 2e-4


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _tensor_identity(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": [int(dimension) for dimension in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "content_sha256": production_tensor_sha256(value),
    }


@dataclass(frozen=True)
class NativeBackwardExportV4:
    """Six-layer native lower adjoints and shared input intermediate bounds."""

    lower: torch.Tensor
    l_a_by_provider_activation: tuple[tuple[str, torch.Tensor], ...]
    intermediate_by_provider_preactivation: tuple[tuple[str, IntervalState], ...]
    intermediate_source: str
    provider_core_callback_count: int = 0
    provider_compute_bounds_callback_count: int = 0
    provider_update_bounds_callback_count: int = 0
    fallback_dispatch_count: int = 0
    schema_version: str = NATIVE_BACKWARD_EXPORT_SCHEMA

    @property
    def l_as(self) -> dict[str, torch.Tensor]:
        return dict(self.l_a_by_provider_activation)

    @property
    def intermediates(self) -> dict[str, IntervalState]:
        return dict(self.intermediate_by_provider_preactivation)

    def validate(self) -> None:
        l_as = self.l_as
        intermediates = self.intermediates
        if (
            self.schema_version != NATIVE_BACKWARD_EXPORT_SCHEMA
            or tuple(sorted(l_as))
            != ("/45", "/48", "/input-12", "/input-16", "/input-24", "/input-4")
            or tuple(sorted(intermediates))
            != ("/39", "/44", "/input", "/input-20", "/input-28", "/input-8")
            or len(l_as) != len(self.l_a_by_provider_activation)
            or len(intermediates) != len(self.intermediate_by_provider_preactivation)
            or tuple(self.lower.shape) != (6, 1)
            or not bool(torch.isfinite(self.lower).all())
            or self.intermediate_source != "shared-pre-result-external-bounds"
            or self.provider_core_callback_count != 0
            or self.provider_compute_bounds_callback_count != 0
            or self.provider_update_bounds_callback_count != 0
            or self.fallback_dispatch_count != 0
        ):
            raise ValueError("RVIR-v4 native backward export contract differs")
        for coefficient in l_as.values():
            if (
                coefficient.shape[0] != 6
                or coefficient.shape[1] != 1
                or not torch.is_floating_point(coefficient)
                or not bool(torch.isfinite(coefficient).all())
            ):
                raise ValueError("RVIR-v4 native lA tensor schema differs")
        for interval in intermediates.values():
            interval.validate()
            if (
                interval.lower.shape[0] != 6
                or interval.lower.shape != interval.upper.shape
                or not bool(torch.isfinite(interval.lower).all())
                or not bool(torch.isfinite(interval.upper).all())
            ):
                raise ValueError("RVIR-v4 native intermediate tensor schema differs")

    def metadata(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "lower": _tensor_identity(self.lower),
            "lAs": {
                name: _tensor_identity(value)
                for name, value in self.l_a_by_provider_activation
            },
            "intermediates": {
                name: {
                    "lower": _tensor_identity(value.lower),
                    "upper": _tensor_identity(value.upper),
                }
                for name, value in self.intermediate_by_provider_preactivation
            },
            "intermediate_source": self.intermediate_source,
            "provider_core_callback_count": self.provider_core_callback_count,
            "provider_compute_bounds_callback_count": (
                self.provider_compute_bounds_callback_count
            ),
            "provider_update_bounds_callback_count": (
                self.provider_update_bounds_callback_count
            ),
            "fallback_dispatch_count": self.fallback_dispatch_count,
            "whole_core_replacement_admitted": False,
            "b2_same_solver_timing_admitted": False,
            "performance_claimed": False,
        }
        payload["export_hash"] = _canonical_hash(payload)
        return payload


@dataclass(frozen=True)
class NativeBackwardParityV4:
    """Formal numeric and discrete parity for one native backward export."""

    lower_maximum_absolute_difference: float
    l_a_maximum_absolute_difference: float
    intermediate_maximum_absolute_difference: float
    l_a_sign_exact: bool
    intermediate_sign_exact: bool
    lower_sign_exact: bool
    l_a_tensor_count: int
    intermediate_tensor_count: int

    def validate(self) -> None:
        if (
            not all(
                math.isfinite(value)
                for value in (
                    self.lower_maximum_absolute_difference,
                    self.l_a_maximum_absolute_difference,
                    self.intermediate_maximum_absolute_difference,
                )
            )
            or self.lower_maximum_absolute_difference > PARITY_ATOL
            or self.l_a_maximum_absolute_difference > PARITY_ATOL
            or self.intermediate_maximum_absolute_difference > PARITY_ATOL
            or self.l_a_sign_exact is not True
            or self.intermediate_sign_exact is not True
            or self.lower_sign_exact is not True
            or self.l_a_tensor_count != 6
            or self.intermediate_tensor_count != 12
        ):
            raise ValueError("RVIR-v4 native backward parity differs")

    def metadata(self) -> dict[str, object]:
        self.validate()
        return {
            **self.__dict__,
            "atol": PARITY_ATOL,
            "rtol": PARITY_RTOL,
            "native_backward_export_admitted": True,
            "whole_core_replacement_admitted": False,
            "b2_same_solver_timing_admitted": False,
            "performance_claimed": False,
        }


def export_rvir_v4_native_backward(
    *,
    module: BFTaskModule,
    input_spec: InputSpec,
    linear_spec_C: torch.Tensor,
    relu_pre: Mapping[str, IntervalState],
    terminal_state: NativeAlphaBetaOptimizationState,
    topology: tuple[ProductionReluTopologyV4, ...],
) -> NativeBackwardExportV4:
    """Export lA and shared intermediate bounds without any provider callback."""

    terminal_state.validate()
    if len(topology) != 6:
        raise ValueError("RVIR-v4 native backward topology inventory differs")
    for item in topology:
        item.validate()
    native_names = {item.native_preactivation for item in topology}
    if (
        set(relu_pre) != native_names
        or set(terminal_state.alphas) != native_names
        or set(terminal_state.betas) != native_names
        or set(terminal_state.splits) != native_names
    ):
        raise ValueError("RVIR-v4 native backward state topology differs")
    interval_env, local_pre = _forward_ibp_trace_mlp(
        module, input_spec, relu_split_state=terminal_state.splits
    )
    if set(local_pre) != native_names:
        raise ValueError("RVIR-v4 native backward local graph topology differs")
    beta_add = _beta_to_relu_pre_add_coeff(
        BetaState(terminal_state.betas),
        relu_pre=dict(relu_pre),
        relu_split_state=terminal_state.splits,
    )
    bounds, native_l_as = (
        run_crown_ibp_mlp_with_relu_lower_coefficients_from_forward_trace(
            module,
            input_spec,
            interval_env=interval_env,
            relu_pre=dict(relu_pre),
            linear_spec_C=linear_spec_C,
            relu_alpha=terminal_state.alphas,
            relu_pre_add_coeff_l=beta_add,
        )
    )
    export = NativeBackwardExportV4(
        lower=bounds.lower.detach().contiguous().clone(),
        l_a_by_provider_activation=tuple(
            sorted(
                (
                    item.provider_activation,
                    native_l_as[item.native_preactivation]
                    .detach()
                    .contiguous()
                    .clone(),
                )
                for item in topology
            )
        ),
        intermediate_by_provider_preactivation=tuple(
            sorted(
                (
                    item.provider_preactivation,
                    IntervalState(
                        lower=relu_pre[item.native_preactivation]
                        .lower.detach()
                        .contiguous()
                        .clone(),
                        upper=relu_pre[item.native_preactivation]
                        .upper.detach()
                        .contiguous()
                        .clone(),
                    ),
                )
                for item in topology
            )
        ),
        intermediate_source="shared-pre-result-external-bounds",
    )
    export.validate()
    return export


def _tensor_parity(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, bool]:
    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or actual.device != expected.device
        or not torch.allclose(
            actual, expected, atol=PARITY_ATOL, rtol=PARITY_RTOL, equal_nan=False
        )
    ):
        raise ValueError("RVIR-v4 native backward numeric tensor parity differs")
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    maximum = (
        float(torch.max(torch.abs(actual[finite] - expected[finite])).item())
        if bool(finite.any())
        else 0.0
    )
    return maximum, torch.equal(torch.sign(actual), torch.sign(expected))


def compare_rvir_v4_native_backward_export(
    export: NativeBackwardExportV4,
    *,
    expected_lower: torch.Tensor,
    expected_l_as: Mapping[str, torch.Tensor],
    expected_intermediates: Mapping[str, IntervalState],
) -> NativeBackwardParityV4:
    """Compare native export against the separately frozen V4-3A truth."""

    export.validate()
    if set(export.l_as) != set(expected_l_as) or set(export.intermediates) != set(
        expected_intermediates
    ):
        raise ValueError("RVIR-v4 native backward comparator inventory differs")
    lower_max, lower_sign = _tensor_parity(export.lower, expected_lower)
    l_a_rows = [
        _tensor_parity(export.l_as[name], expected_l_as[name])
        for name in sorted(expected_l_as)
    ]
    intermediate_rows = [
        _tensor_parity(
            getattr(export.intermediates[name], polarity),
            getattr(expected_intermediates[name], polarity),
        )
        for name in sorted(expected_intermediates)
        for polarity in ("lower", "upper")
    ]
    parity = NativeBackwardParityV4(
        lower_maximum_absolute_difference=lower_max,
        l_a_maximum_absolute_difference=max(row[0] for row in l_a_rows),
        intermediate_maximum_absolute_difference=max(
            row[0] for row in intermediate_rows
        ),
        l_a_sign_exact=all(row[1] for row in l_a_rows),
        intermediate_sign_exact=all(row[1] for row in intermediate_rows),
        lower_sign_exact=lower_sign,
        l_a_tensor_count=len(l_a_rows),
        intermediate_tensor_count=len(intermediate_rows),
    )
    parity.validate()
    return parity


__all__ = [
    "compare_rvir_v4_native_backward_export",
    "export_rvir_v4_native_backward",
    "NativeBackwardExportV4",
    "NativeBackwardParityV4",
    "PARITY_ATOL",
    "PARITY_RTOL",
]
