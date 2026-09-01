"""Provider-independent RVIR-v4 KFSB candidate evaluation."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=too-many-instance-attributes,missing-function-docstring

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import torch

from ..domains.interval import IntervalState
from ..ir.task import BFTaskModule, TaskOp
from .crown_ibp import _forward_ibp_trace_mlp, run_crown_ibp_mlp_from_forward_trace
from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizationState
from .rvir_v4_native_backward_export import (
    NativeBackwardExportV4,
    PARITY_ATOL,
    PARITY_RTOL,
)
from .rvir_v4_pre_state_initializer import ProductionReluTopologyV4
from .rvir_v4_production_state import production_tensor_sha256
from .task_executor import InputSpec

NATIVE_KFSB_SCHEMA = "boundflow.rvir-v4-native-kfsb/v1"
KFSB_CANDIDATE_COUNT = 3
KFSB_SCORE_THRESHOLD = 1e-4
KFSB_INVALID_PENALTY = 999999.0
Decision = tuple[int, int]


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


def _decision_payload(decisions: Sequence[Decision]) -> list[list[int]]:
    return [[int(layer), int(neuron)] for layer, neuron in decisions]


@dataclass(frozen=True)
class NativeKfsbEvaluationV4:
    """Three native KFSB candidates, their child bounds, and final decision."""

    candidate_splits: tuple[tuple[Decision, ...], ...]
    candidate_child_lowers: tuple[torch.Tensor, ...]
    final_decision: tuple[Decision, ...]
    alpha_score_topk_values: torch.Tensor
    intercept_score_topk_values: torch.Tensor
    reduced_candidate_values: torch.Tensor
    unstable_mask_by_provider_preactivation: tuple[tuple[str, torch.Tensor], ...]
    layer_sizes: tuple[int, ...]
    unstable_counts: tuple[int, ...]
    provider_core_callback_count: int = 0
    provider_compute_bounds_callback_count: int = 0
    provider_update_bounds_callback_count: int = 0
    fallback_dispatch_count: int = 0
    schema_version: str = NATIVE_KFSB_SCHEMA

    @property
    def unstable_masks(self) -> dict[str, torch.Tensor]:
        return dict(self.unstable_mask_by_provider_preactivation)

    def validate(self) -> None:
        batch = len(self.final_decision)
        unstable_masks = self.unstable_masks
        if (
            self.schema_version != NATIVE_KFSB_SCHEMA
            or len(self.layer_sizes) != 6
            or len(self.unstable_counts) != 6
            or any(size <= 0 for size in self.layer_sizes)
            or any(count <= 0 for count in self.unstable_counts)
            or len(self.candidate_splits) != KFSB_CANDIDATE_COUNT
            or len(self.candidate_child_lowers) != KFSB_CANDIDATE_COUNT
            or batch != 6
            or tuple(self.alpha_score_topk_values.shape)
            != (batch, KFSB_CANDIDATE_COUNT)
            or tuple(self.intercept_score_topk_values.shape)
            != (batch, KFSB_CANDIDATE_COUNT)
            or tuple(self.reduced_candidate_values.shape)
            != (KFSB_CANDIDATE_COUNT, batch * 2)
            or len(unstable_masks) != 6
            or len(unstable_masks) != len(self.unstable_mask_by_provider_preactivation)
            or self.provider_core_callback_count != 0
            or self.provider_compute_bounds_callback_count != 0
            or self.provider_update_bounds_callback_count != 0
            or self.fallback_dispatch_count != 0
        ):
            raise ValueError("RVIR-v4 native KFSB contract differs")
        tensors = (
            *self.candidate_child_lowers,
            self.alpha_score_topk_values,
            self.intercept_score_topk_values,
            self.reduced_candidate_values,
        )
        if any(
            not torch.is_floating_point(value) or not bool(torch.isfinite(value).all())
            for value in tensors
        ):
            raise ValueError("RVIR-v4 native KFSB numeric tensor differs")
        for child_lower in self.candidate_child_lowers:
            if tuple(child_lower.shape) != (batch * 4, 1):
                raise ValueError("RVIR-v4 native KFSB child-lower shape differs")
        for (name, mask), size, count in zip(
            self.unstable_mask_by_provider_preactivation,
            self.layer_sizes,
            self.unstable_counts,
            strict=True,
        ):
            if (
                not name
                or mask.dtype != torch.bool
                or tuple(mask.shape) != (batch, size)
                or int(mask.sum().item()) != count
            ):
                raise ValueError("RVIR-v4 native KFSB unstable mask differs")
        for decisions in self.candidate_splits:
            if len(decisions) != batch * 2:
                raise ValueError("RVIR-v4 native KFSB candidate width differs")
            self._validate_decisions(decisions)
        self._validate_decisions(self.final_decision)

    def _validate_decisions(self, decisions: Sequence[Decision]) -> None:
        for layer, neuron in decisions:
            if (
                layer < 0
                or layer >= len(self.layer_sizes)
                or neuron < 0
                or neuron >= self.layer_sizes[layer]
            ):
                raise ValueError("RVIR-v4 native KFSB decision differs")

    def metadata(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "candidate_splits": [
                _decision_payload(decisions) for decisions in self.candidate_splits
            ],
            "candidate_child_lowers": [
                _tensor_identity(value) for value in self.candidate_child_lowers
            ],
            "final_decision": _decision_payload(self.final_decision),
            "alpha_score_topk_values": _tensor_identity(self.alpha_score_topk_values),
            "intercept_score_topk_values": _tensor_identity(
                self.intercept_score_topk_values
            ),
            "reduced_candidate_values": _tensor_identity(self.reduced_candidate_values),
            "unstable_masks": {
                name: _tensor_identity(value)
                for name, value in self.unstable_mask_by_provider_preactivation
            },
            "layer_sizes": list(self.layer_sizes),
            "unstable_counts": list(self.unstable_counts),
            "provider_core_callback_count": self.provider_core_callback_count,
            "provider_compute_bounds_callback_count": (
                self.provider_compute_bounds_callback_count
            ),
            "provider_update_bounds_callback_count": (
                self.provider_update_bounds_callback_count
            ),
            "fallback_dispatch_count": self.fallback_dispatch_count,
            "native_kfsb_admitted": True,
            "whole_core_replacement_admitted": False,
            "b2_same_solver_timing_admitted": False,
            "performance_claimed": False,
        }
        payload["evaluation_hash"] = _canonical_hash(payload)
        return payload


@dataclass(frozen=True)
class NativeKfsbParityV4:
    """Formal candidate, child-bound, and decision parity with V4-3A truth."""

    candidate_splits_exact: bool
    final_decision_exact: bool
    unstable_masks_exact: bool
    child_lower_sign_exact: bool
    child_lower_maximum_absolute_difference: float
    candidate_count: int
    child_lower_tensor_count: int
    child_lower_element_count: int
    unstable_mask_tensor_count: int
    unstable_mask_element_count: int

    def validate(self) -> None:
        if (
            self.candidate_splits_exact is not True
            or self.final_decision_exact is not True
            or self.unstable_masks_exact is not True
            or self.child_lower_sign_exact is not True
            or not math.isfinite(self.child_lower_maximum_absolute_difference)
            or self.child_lower_maximum_absolute_difference > PARITY_ATOL
            or self.candidate_count != KFSB_CANDIDATE_COUNT
            or self.child_lower_tensor_count != KFSB_CANDIDATE_COUNT
            or self.child_lower_element_count != 72
            or self.unstable_mask_tensor_count != 6
            or self.unstable_mask_element_count != 37464
        ):
            raise ValueError("RVIR-v4 native KFSB parity differs")

    def metadata(self) -> dict[str, object]:
        self.validate()
        return {
            **self.__dict__,
            "atol": PARITY_ATOL,
            "rtol": PARITY_RTOL,
            "native_kfsb_admitted": True,
            "whole_core_replacement_admitted": False,
            "b2_same_solver_timing_admitted": False,
            "performance_claimed": False,
        }


def _params(module: BFTaskModule) -> dict[str, object]:
    raw = module.bindings.get("params")
    if not isinstance(raw, dict):
        raise TypeError("RVIR-v4 native KFSB requires parameter bindings")
    return dict(raw)


def _producer_by_output(module: BFTaskModule) -> dict[str, TaskOp]:
    producers: dict[str, TaskOp] = {}
    for op in module.get_entry_task().ops:
        for output in op.outputs:
            if output in producers:
                raise ValueError("RVIR-v4 native KFSB producer identity repeats")
            producers[output] = op
    return producers


def _preactivation_bias(
    module: BFTaskModule, value_name: str, *, expected_channels: int
) -> torch.Tensor | None:
    params = _params(module)
    producers = _producer_by_output(module)

    def visit(name: str) -> torch.Tensor | None:
        op = producers.get(name)
        if op is None:
            return None
        if op.op_type in {"conv2d", "linear"}:
            if len(op.inputs) < 3:
                return None
            value = params.get(op.inputs[2])
            if not torch.is_tensor(value):
                raise TypeError("RVIR-v4 native KFSB affine bias differs")
            return value.detach()
        if op.op_type == "add":
            values = [visit(input_name) for input_name in op.inputs]
            tensors = [value for value in values if value is not None]
            if not tensors:
                return None
            result = tensors[0]
            for value in tensors[1:]:
                if value.shape != result.shape:
                    raise ValueError("RVIR-v4 native KFSB residual bias differs")
                result = result + value
            return result
        return None

    producer = producers.get(value_name)
    if producer is None or producer.op_type not in {"conv2d", "linear", "add"}:
        raise ValueError("RVIR-v4 native KFSB preactivation producer differs")
    bias = visit(value_name)
    if bias is not None and bias.numel() != expected_channels:
        raise ValueError("RVIR-v4 native KFSB bias width differs")
    return bias


def _repeat_batch(value: torch.Tensor, copies: int) -> torch.Tensor:
    return value.repeat(copies, *([1] * (value.ndim - 1)))


def _flat_decision(flat_index: int, cumulative: Sequence[int]) -> Decision:
    layer = bisect_right(cumulative, flat_index) - 1
    if layer < 0 or layer >= len(cumulative) - 1:
        raise ValueError("RVIR-v4 native KFSB flattened decision differs")
    return layer, flat_index - cumulative[layer]


def _native_scores(
    module: BFTaskModule,
    export: NativeBackwardExportV4,
    terminal_state: NativeAlphaBetaOptimizationState,
    topology: tuple[ProductionReluTopologyV4, ...],
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    tuple[int, ...],
    tuple[int, ...],
    dict[str, torch.Tensor],
    dict[str, IntervalState],
]:
    l_as = export.l_as
    provider_intermediates = export.intermediates
    splits = terminal_state.splits
    scores: list[torch.Tensor] = []
    intercept_scores: list[torch.Tensor] = []
    masks: dict[str, torch.Tensor] = {}
    native_intermediates: dict[str, IntervalState] = {}
    layer_sizes: list[int] = []
    unstable_counts: list[int] = []
    for item in topology:
        coefficient = l_as[item.provider_activation]
        interval = provider_intermediates[item.provider_preactivation]
        native_name = item.native_preactivation
        split = splits[native_name]
        if (
            interval.lower.shape != split.shape
            or coefficient.shape[0] != interval.lower.shape[0]
            or coefficient.shape[2:] != interval.lower.shape[1:]
            or coefficient.shape[1] != 1
        ):
            raise ValueError("RVIR-v4 native KFSB layer shape differs")
        mask = ((interval.lower < 0) & (interval.upper > 0) & (split == 0)).reshape(
            int(interval.lower.shape[0]), -1
        )
        lower_temp = interval.lower.clamp(max=0)
        upper_temp = interval.upper.clamp(min=0)
        slope = upper_temp / (upper_temp - lower_temp)
        intercept = -lower_temp * slope
        intercept_candidate = coefficient.clamp(max=0) * intercept.unsqueeze(1)
        masked_intercept = (
            intercept_candidate.reshape(int(mask.shape[0]), 1, -1) * mask.unsqueeze(1)
        ).mean(1)
        bias = _preactivation_bias(
            module,
            native_name,
            expected_channels=int(coefficient.shape[2]),
        )
        if bias is None:
            bias_term: torch.Tensor | int = 0
        else:
            bias_term = bias.to(device=coefficient.device, dtype=coefficient.dtype)
            bias_term = bias_term.reshape(-1, *([1] * (coefficient.ndim - 3)))
        weighted_bias = bias_term * coefficient
        slope_with_spec = slope.unsqueeze(1)
        bias_candidate = torch.minimum(
            weighted_bias * (slope_with_spec - 1),
            weighted_bias * slope_with_spec,
        )
        score = (
            (bias_candidate + intercept_candidate)
            .abs()
            .reshape(int(mask.shape[0]), 1, -1)
            * mask.unsqueeze(1)
        ).mean(1)
        if not bool(torch.isfinite(score).all()) or not bool(
            torch.isfinite(masked_intercept).all()
        ):
            raise ValueError("RVIR-v4 native KFSB score is not finite")
        scores.append(score)
        intercept_scores.append(masked_intercept)
        masks[native_name] = mask
        native_intermediates[native_name] = interval
        layer_sizes.append(int(mask.shape[1]))
        unstable_counts.append(int(mask.sum().item()))
    return (
        scores,
        intercept_scores,
        tuple(layer_sizes),
        tuple(unstable_counts),
        masks,
        native_intermediates,
    )


def _evaluate_children(
    *,
    module: BFTaskModule,
    input_spec: InputSpec,
    linear_spec_C: torch.Tensor,
    terminal_state: NativeAlphaBetaOptimizationState,
    topology: tuple[ProductionReluTopologyV4, ...],
    native_intermediates: Mapping[str, IntervalState],
    decisions: Sequence[Decision],
) -> torch.Tensor:
    batch = int(input_spec.center.shape[0])
    if len(decisions) != batch * 2:
        raise ValueError("RVIR-v4 native KFSB child decision inventory differs")
    child_splits = {
        name: _repeat_batch(value, 4).clone()
        for name, value in terminal_state.splits.items()
    }
    child_intermediates = {
        name: IntervalState(
            lower=_repeat_batch(value.lower, 4).clone(),
            upper=_repeat_batch(value.upper, 4).clone(),
        )
        for name, value in native_intermediates.items()
    }
    for row, (layer, neuron) in enumerate(decisions):
        name = topology[layer].native_preactivation
        split = child_splits[name].reshape(batch * 4, -1)
        lower = child_intermediates[name].lower.reshape(batch * 4, -1)
        upper = child_intermediates[name].upper.reshape(batch * 4, -1)
        if (
            split[row, neuron] != 0
            or lower[row, neuron] >= 0
            or upper[row, neuron] <= 0
        ):
            raise ValueError("RVIR-v4 native KFSB candidate is not unstable")
        split[row, neuron] = 1
        lower[row, neuron] = 0
        opposite_row = batch * 2 + row
        if split[opposite_row, neuron] != 0:
            raise ValueError("RVIR-v4 native KFSB opposite split is not free")
        split[opposite_row, neuron] = -1
        upper[opposite_row, neuron] = 0
    input_lower, input_upper = input_spec.perturbation.bounding_box(input_spec.center)
    child_input = InputSpec.box(
        value_name=input_spec.value_name,
        lower=_repeat_batch(input_lower, 4),
        upper=_repeat_batch(input_upper, 4),
    )
    interval_env, _local_pre = _forward_ibp_trace_mlp(
        module,
        child_input,
        relu_split_state=child_splits,
    )
    child_lower = run_crown_ibp_mlp_from_forward_trace(
        module,
        child_input,
        interval_env=interval_env,
        relu_pre=child_intermediates,
        linear_spec_C=_repeat_batch(linear_spec_C, 4),
        relu_alpha={
            name: _repeat_batch(value, 4)
            for name, value in terminal_state.alphas.items()
        },
        relu_pre_add_coeff_l={},
    ).lower
    if tuple(child_lower.shape) != (batch * 4, 1):
        raise ValueError("RVIR-v4 native KFSB evaluated child shape differs")
    return child_lower.detach().contiguous().clone()


@torch.no_grad()
def evaluate_rvir_v4_native_kfsb(
    *,
    module: BFTaskModule,
    input_spec: InputSpec,
    linear_spec_C: torch.Tensor,
    thresholds: torch.Tensor,
    terminal_state: NativeAlphaBetaOptimizationState,
    topology: tuple[ProductionReluTopologyV4, ...],
    backward_export: NativeBackwardExportV4,
) -> NativeKfsbEvaluationV4:
    """Run the fixed production KFSB policy using BoundFlow only."""

    module.validate()
    terminal_state.validate()
    backward_export.validate()
    if len(topology) != 6 or int(input_spec.center.shape[0]) != 6:
        raise ValueError("RVIR-v4 native KFSB fixed topology/batch differs")
    for item in topology:
        item.validate()
    batch = int(input_spec.center.shape[0])
    if (
        tuple(linear_spec_C.shape[:2]) != (batch, 1)
        or tuple(thresholds.shape) != (batch, 1)
        or linear_spec_C.device != input_spec.center.device
        or thresholds.device != input_spec.center.device
        or linear_spec_C.dtype != input_spec.center.dtype
        or thresholds.dtype != input_spec.center.dtype
        or not bool(torch.isfinite(thresholds).all())
    ):
        raise ValueError("RVIR-v4 native KFSB query tensor differs")
    (
        score_layers,
        intercept_layers,
        layer_sizes,
        unstable_counts,
        masks,
        native_intermediates,
    ) = _native_scores(module, backward_export, terminal_state, topology)
    if sum(unstable_counts) < KFSB_CANDIDATE_COUNT:
        raise ValueError("RVIR-v4 native KFSB candidate population differs")
    all_score = torch.cat(score_layers, dim=1)
    all_intercept = torch.cat(intercept_layers, dim=1)
    score_topk = torch.topk(all_score, KFSB_CANDIDATE_COUNT)
    intercept_topk = torch.topk(all_intercept, KFSB_CANDIDATE_COUNT, largest=False)
    cumulative = [0]
    for size in layer_sizes:
        cumulative.append(cumulative[-1] + size)
    candidate_splits: list[tuple[Decision, ...]] = []
    candidate_child_lowers: list[torch.Tensor] = []
    reduced_rows: list[torch.Tensor] = []
    repeated_thresholds = _repeat_batch(thresholds, 4)
    for candidate_index in range(KFSB_CANDIDATE_COUNT):
        alpha_decisions = tuple(
            _flat_decision(int(score_topk.indices[row, candidate_index]), cumulative)
            for row in range(batch)
        )
        intercept_decisions = tuple(
            _flat_decision(
                int(intercept_topk.indices[row, candidate_index]), cumulative
            )
            for row in range(batch)
        )
        decisions = alpha_decisions + intercept_decisions
        for row, (layer, neuron) in enumerate(decisions):
            native_name = topology[layer].native_preactivation
            if not bool(masks[native_name][row % batch, neuron].item()):
                score_value = (
                    score_topk.values[row, candidate_index]
                    if row < batch
                    else intercept_topk.values[row - batch, candidate_index]
                )
                if (
                    row < batch and float(score_value.item()) > KFSB_SCORE_THRESHOLD
                ) or (
                    row >= batch and float(score_value.item()) < -KFSB_SCORE_THRESHOLD
                ):
                    raise ValueError(
                        "RVIR-v4 native KFSB valid score selected stable node"
                    )
        child_lower = _evaluate_children(
            module=module,
            input_spec=input_spec,
            linear_spec_C=linear_spec_C,
            terminal_state=terminal_state,
            topology=topology,
            native_intermediates=native_intermediates,
            decisions=decisions,
        )
        adjusted = (child_lower - repeated_thresholds).max(-1).values
        invalid_alpha = (
            score_topk.values[:, candidate_index] <= KFSB_SCORE_THRESHOLD
        ).to(dtype=adjusted.dtype)
        invalid_intercept = (
            intercept_topk.values[:, candidate_index] >= -KFSB_SCORE_THRESHOLD
        ).to(dtype=adjusted.dtype)
        invalid = torch.cat((invalid_alpha, invalid_intercept)).repeat(2)
        reduced = (
            (adjusted.reshape(-1) - invalid * KFSB_INVALID_PENALTY)
            .reshape(2, -1)
            .min(dim=0)
            .values
        )
        candidate_splits.append(decisions)
        candidate_child_lowers.append(child_lower)
        reduced_rows.append(reduced)
    reduced_candidates = torch.stack(reduced_rows)
    best = reduced_candidates.topk(1, dim=0)
    final_decision: list[Decision] = []
    for row in range(batch):
        use_alpha = bool(best.values[0, row] > best.values[0, row + batch])
        score_row = row if use_alpha else row + batch
        candidate_index = int(best.indices[0, score_row].item())
        decision = candidate_splits[candidate_index][score_row]
        native_name = topology[decision[0]].native_preactivation
        if not bool(masks[native_name][row, decision[1]].item()):
            raise ValueError("RVIR-v4 native KFSB final decision is not unstable")
        final_decision.append(decision)
    evaluation = NativeKfsbEvaluationV4(
        candidate_splits=tuple(candidate_splits),
        candidate_child_lowers=tuple(candidate_child_lowers),
        final_decision=tuple(final_decision),
        alpha_score_topk_values=score_topk.values.detach().contiguous().clone(),
        intercept_score_topk_values=(
            intercept_topk.values.detach().contiguous().clone()
        ),
        reduced_candidate_values=reduced_candidates.detach().contiguous().clone(),
        unstable_mask_by_provider_preactivation=tuple(
            (
                item.provider_preactivation,
                masks[item.native_preactivation].detach().contiguous().clone(),
            )
            for item in topology
        ),
        layer_sizes=layer_sizes,
        unstable_counts=unstable_counts,
    )
    evaluation.validate()
    return evaluation


def _tensor_parity(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, bool]:
    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or actual.device != expected.device
        or not torch.allclose(
            actual, expected, atol=PARITY_ATOL, rtol=PARITY_RTOL, equal_nan=False
        )
    ):
        raise ValueError("RVIR-v4 native KFSB child numeric parity differs")
    maximum = float(torch.max(torch.abs(actual - expected)).item())
    return maximum, torch.equal(torch.sign(actual), torch.sign(expected))


def compare_rvir_v4_native_kfsb(
    evaluation: NativeKfsbEvaluationV4,
    *,
    expected_candidate_splits: Sequence[Sequence[Decision]],
    expected_candidate_child_lowers: Sequence[torch.Tensor],
    expected_final_decision: Sequence[Decision],
    expected_unstable_masks: Mapping[str, torch.Tensor],
) -> NativeKfsbParityV4:
    """Compare native KFSB output with the separately frozen V4-3A truth."""

    evaluation.validate()
    normalized_candidates = tuple(
        tuple((int(layer), int(neuron)) for layer, neuron in decisions)
        for decisions in expected_candidate_splits
    )
    normalized_final = tuple(
        (int(layer), int(neuron)) for layer, neuron in expected_final_decision
    )
    if set(evaluation.unstable_masks) != set(expected_unstable_masks):
        raise ValueError("RVIR-v4 native KFSB mask inventory differs")
    rows = [
        _tensor_parity(actual, expected)
        for actual, expected in zip(
            evaluation.candidate_child_lowers,
            expected_candidate_child_lowers,
            strict=True,
        )
    ]
    parity = NativeKfsbParityV4(
        candidate_splits_exact=evaluation.candidate_splits == normalized_candidates,
        final_decision_exact=evaluation.final_decision == normalized_final,
        unstable_masks_exact=all(
            torch.equal(evaluation.unstable_masks[name], expected_unstable_masks[name])
            for name in sorted(expected_unstable_masks)
        ),
        child_lower_sign_exact=all(row[1] for row in rows),
        child_lower_maximum_absolute_difference=max(row[0] for row in rows),
        candidate_count=len(evaluation.candidate_splits),
        child_lower_tensor_count=len(rows),
        child_lower_element_count=sum(
            int(value.numel()) for value in evaluation.candidate_child_lowers
        ),
        unstable_mask_tensor_count=len(evaluation.unstable_masks),
        unstable_mask_element_count=sum(
            int(value.numel()) for value in evaluation.unstable_masks.values()
        ),
    )
    parity.validate()
    return parity


__all__ = [
    "compare_rvir_v4_native_kfsb",
    "evaluate_rvir_v4_native_kfsb",
    "KFSB_CANDIDATE_COUNT",
    "NATIVE_KFSB_SCHEMA",
    "NativeKfsbEvaluationV4",
    "NativeKfsbParityV4",
]
