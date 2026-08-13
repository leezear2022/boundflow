"""Provider-independent RVIR-v4 native optimizer trace and parity contracts."""

# pylint: disable=too-many-locals,too-many-statements,too-many-arguments
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=too-many-instance-attributes

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
from .crown_ibp import _forward_ibp_trace_mlp, run_crown_ibp_mlp_from_forward_trace
from .native_alpha_beta_optimization_state import (
    build_native_alpha_beta_scope,
    NativeAlphaBetaOptimizationState,
)
from .rvir_v4_optimizer_mutation import (
    ProductionMutationPolicyV4,
    ProductionOptimizerStepTraceV4,
)
from .rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
    ProductionReluTopologyV4,
)
from .rvir_v4_production_state import (
    ProductionStateSnapshotV4,
    production_tensor_sha256,
)
from .task_executor import InputSpec

RVIR_V4_NATIVE_OPTIMIZER_TRACE_SCHEMA = "boundflow.rvir-v4-native-optimizer-trace/v1"
RVIR_V4_NATIVE_OPTIMIZER_PARITY_SCHEMA = "boundflow.rvir-v4-native-optimizer-parity/v1"
PARITY_ATOL = 2e-4
PARITY_RTOL = 2e-4


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _tensor_rows(values: Mapping[str, torch.Tensor]) -> dict[str, object]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "content_sha256": production_tensor_sha256(value),
        }
        for name, value in sorted(values.items())
    }


@dataclass(frozen=True)
class NativeProductionOptimizerStepV4:
    """One native evaluation and its mutable state before an optional update."""

    evaluation_ordinal: int
    updates_before: int
    update_after: bool
    alpha_learning_rate: float
    beta_learning_rate: float
    lower: torch.Tensor
    alpha_by_relu_input: tuple[tuple[str, torch.Tensor], ...]
    beta_by_relu_input: tuple[tuple[str, torch.Tensor], ...]

    @property
    def alphas(self) -> dict[str, torch.Tensor]:
        return dict(self.alpha_by_relu_input)

    @property
    def betas(self) -> dict[str, torch.Tensor]:
        return dict(self.beta_by_relu_input)

    def validate(self) -> None:
        if (
            self.evaluation_ordinal < 0
            or self.updates_before != self.evaluation_ordinal
            or not math.isfinite(self.alpha_learning_rate)
            or not math.isfinite(self.beta_learning_rate)
            or self.alpha_learning_rate <= 0.0
            or self.beta_learning_rate <= 0.0
            or not torch.is_tensor(self.lower)
            or tuple(self.lower.shape) != (6, 1)
            or not torch.is_floating_point(self.lower)
            or not bool(torch.isfinite(self.lower).all())
        ):
            raise ValueError("RVIR-v4 native optimizer step identity differs")
        alphas = self.alphas
        betas = self.betas
        if (
            len(alphas) != len(self.alpha_by_relu_input)
            or len(betas) != len(self.beta_by_relu_input)
            or set(alphas) != set(betas)
            or len(alphas) != 6
        ):
            raise ValueError("RVIR-v4 native optimizer state inventory differs")
        for name in sorted(alphas):
            alpha = alphas[name]
            beta = betas[name]
            if (
                alpha.shape != beta.shape
                or alpha.dtype != beta.dtype
                or alpha.device != beta.device
                or not bool(torch.isfinite(alpha).all())
                or not bool(torch.isfinite(beta).all())
                or not bool(((alpha >= 0.0) & (alpha <= 1.0)).all())
                or not bool((beta >= 0.0).all())
            ):
                raise ValueError(f"RVIR-v4 native optimizer tensor differs: {name}")

    def metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "evaluation_ordinal": self.evaluation_ordinal,
            "updates_before": self.updates_before,
            "update_after": self.update_after,
            "alpha_learning_rate": self.alpha_learning_rate,
            "beta_learning_rate": self.beta_learning_rate,
            "lower_shape": list(self.lower.shape),
            "lower_dtype": str(self.lower.dtype),
            "lower_sha256": production_tensor_sha256(self.lower),
            "alphas": _tensor_rows(self.alphas),
            "betas": _tensor_rows(self.betas),
        }


@dataclass(frozen=True)
class NativeProductionOptimizerTraceV4:
    """Ten evaluations produced without a provider callback or reference trace."""

    source_state_hash: str
    scope_hash: str
    mutation_policy_hash: str
    steps: tuple[NativeProductionOptimizerStepV4, ...]
    schema_version: str = RVIR_V4_NATIVE_OPTIMIZER_TRACE_SCHEMA

    def validate(self) -> None:
        if (
            self.schema_version != RVIR_V4_NATIVE_OPTIMIZER_TRACE_SCHEMA
            or len(self.source_state_hash) != 64
            or len(self.scope_hash) != 64
            or len(self.mutation_policy_hash) != 64
            or len(self.steps) != 10
        ):
            raise ValueError("RVIR-v4 native optimizer trace identity differs")
        for ordinal, step in enumerate(self.steps):
            step.validate()
            if (
                step.evaluation_ordinal != ordinal
                or step.updates_before != ordinal
                or step.update_after != (ordinal < 9)
            ):
                raise ValueError("RVIR-v4 native optimizer loop semantics differ")

    def metadata(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "source_state_hash": self.source_state_hash,
            "scope_hash": self.scope_hash,
            "mutation_policy_hash": self.mutation_policy_hash,
            "evaluation_count": len(self.steps),
            "update_count": sum(step.update_after for step in self.steps),
            "steps": [step.metadata() for step in self.steps],
            "provider_callback_count": 0,
            "performance_claimed": False,
        }
        payload["trace_hash"] = _canonical_hash(payload)
        return payload


@dataclass(frozen=True)
class NativeProductionOptimizerParityV4:
    """Per-step cross-device numerical parity against the frozen provider truth."""

    native_trace_hash: str
    production_trace_hash: str
    step_rows: tuple[Mapping[str, object], ...]
    lower_maximum_absolute_difference: float
    alpha_maximum_absolute_difference: float
    beta_maximum_absolute_difference: float
    schema_version: str = RVIR_V4_NATIVE_OPTIMIZER_PARITY_SCHEMA

    def validate(self) -> None:
        maximums = (
            self.lower_maximum_absolute_difference,
            self.alpha_maximum_absolute_difference,
            self.beta_maximum_absolute_difference,
        )
        if (
            self.schema_version != RVIR_V4_NATIVE_OPTIMIZER_PARITY_SCHEMA
            or len(self.native_trace_hash) != 64
            or len(self.production_trace_hash) != 64
            or len(self.step_rows) != 10
            or not all(
                math.isfinite(value) and value <= PARITY_ATOL for value in maximums
            )
            or not all(row.get("allclose") is True for row in self.step_rows)
            or not all(row.get("sign_exact") is True for row in self.step_rows)
        ):
            raise ValueError("RVIR-v4 native optimizer parity gate failed")

    def metadata(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "native_trace_hash": self.native_trace_hash,
            "production_trace_hash": self.production_trace_hash,
            "step_rows": [dict(row) for row in self.step_rows],
            "lower_maximum_absolute_difference": self.lower_maximum_absolute_difference,
            "alpha_maximum_absolute_difference": self.alpha_maximum_absolute_difference,
            "beta_maximum_absolute_difference": self.beta_maximum_absolute_difference,
            "atol": PARITY_ATOL,
            "rtol": PARITY_RTOL,
            "optimizer_replacement_admitted": False,
            "b2_same_solver_timing_admitted": False,
            "performance_claimed": False,
        }
        payload["parity_hash"] = _canonical_hash(payload)
        return payload


def execute_rvir_v4_native_optimizer_trace(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    relu_pre: Mapping[str, IntervalState],
    initial_state: NativeAlphaBetaOptimizationState,
    mutation_policy: ProductionMutationPolicyV4,
) -> NativeProductionOptimizerTraceV4:
    """Run the admitted production loop without reading its expected step trace."""

    mutation_policy.validate()
    initial_state.validate()
    native_policy = mutation_policy.to_native_policy()
    expected_scope = build_native_alpha_beta_scope(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        relu_pre=relu_pre,
        relu_split_state=initial_state.splits,
        policy=native_policy,
    )
    if initial_state.scope != expected_scope or set(relu_pre) != set(
        initial_state.splits
    ):
        raise ValueError("RVIR-v4 native optimizer initial scope differs")
    interval_env, _local_pre = _forward_ibp_trace_mlp(
        module, input_spec, relu_split_state=initial_state.splits
    )
    alphas = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in sorted(initial_state.alphas.items())
    }
    betas = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in sorted(initial_state.betas.items())
    }
    optimizer = torch.optim.Adam(
        (
            {"params": list(alphas.values()), "lr": native_policy.lr},
            {"params": list(betas.values()), "lr": native_policy.effective_beta_lr},
        )
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=mutation_policy.controls.lr_decay
    )
    steps: list[NativeProductionOptimizerStepV4] = []
    for ordinal in range(mutation_policy.evaluation_count):
        relu_pre_add = _beta_to_relu_pre_add_coeff(
            BetaState(betas),
            relu_pre=dict(relu_pre),
            relu_split_state=initial_state.splits,
        )
        bounds = run_crown_ibp_mlp_from_forward_trace(
            module,
            input_spec,
            interval_env=interval_env,
            relu_pre=dict(relu_pre),
            linear_spec_C=linear_spec_C,
            relu_alpha=alphas,
            relu_pre_add_coeff_l=relu_pre_add,
        )
        row = NativeProductionOptimizerStepV4(
            evaluation_ordinal=ordinal,
            updates_before=ordinal,
            update_after=ordinal < mutation_policy.update_count,
            alpha_learning_rate=float(optimizer.param_groups[0]["lr"]),
            beta_learning_rate=float(optimizer.param_groups[1]["lr"]),
            lower=bounds.lower.detach().contiguous().clone(),
            alpha_by_relu_input=tuple(
                (name, value.detach().contiguous().clone())
                for name, value in sorted(alphas.items())
            ),
            beta_by_relu_input=tuple(
                (name, value.detach().contiguous().clone())
                for name, value in sorted(betas.items())
            ),
        )
        row.validate()
        steps.append(row)
        if row.update_after:
            optimizer.zero_grad(set_to_none=True)
            (-bounds.lower.sum()).backward()
            optimizer.step()
            with torch.no_grad():
                for value in alphas.values():
                    value.clamp_(0.0, 1.0)
                for value in betas.values():
                    value.clamp_(min=0.0)
            scheduler.step()
    trace = NativeProductionOptimizerTraceV4(
        source_state_hash=initial_state.stable_hash(),
        scope_hash=initial_state.scope.stable_hash(),
        mutation_policy_hash=mutation_policy.stable_hash(),
        steps=tuple(steps),
    )
    trace.validate()
    return trace


def _maximum_difference(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> tuple[float, bool, bool]:
    if set(left) != set(right):
        raise ValueError("RVIR-v4 native optimizer parity tensor keys differ")
    maximum = 0.0
    allclose = True
    sign_exact = True
    for name in sorted(left):
        lhs = left[name]
        rhs = right[name]
        if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
            raise ValueError(f"RVIR-v4 native optimizer parity schema differs: {name}")
        maximum = max(maximum, float((lhs - rhs).abs().max().item()))
        allclose = allclose and bool(
            torch.allclose(lhs, rhs, atol=PARITY_ATOL, rtol=PARITY_RTOL)
        )
        sign_exact = sign_exact and torch.equal(torch.sign(lhs), torch.sign(rhs))
    return maximum, allclose, sign_exact


def compare_rvir_v4_native_optimizer_trace(
    native: NativeProductionOptimizerTraceV4,
    production: ProductionOptimizerStepTraceV4,
    *,
    base_snapshot: ProductionStateSnapshotV4,
    topology: tuple[ProductionReluTopologyV4, ...],
) -> NativeProductionOptimizerParityV4:
    """Map every provider step independently and compare it to the native trace."""

    native.validate()
    production.validate()
    base_snapshot.validate()
    if native.mutation_policy_hash != production.mutation_policy.stable_hash():
        raise ValueError("RVIR-v4 native/production mutation policy differs")
    rows: list[Mapping[str, object]] = []
    lower_maximum = 0.0
    alpha_maximum = 0.0
    beta_maximum = 0.0
    for native_step, production_step in zip(native.steps, production.steps):
        tensor_map = base_snapshot.tensor_map()
        tensor_map.update(production_step.tensor_map)
        snapshot = ProductionStateSnapshotV4(
            snapshot_id=base_snapshot.snapshot_id,
            tensors=tuple(
                sorted(tensor_map.values(), key=lambda item: item.semantic_path)
            ),
            history=base_snapshot.history,
            optimizer_policy=base_snapshot.optimizer_policy,
        )
        expected = initialize_rvir_v4_native_pre_state(snapshot, topology)
        alpha_diff, alpha_close, alpha_sign = _maximum_difference(
            native_step.alphas, expected.alphas
        )
        beta_diff, beta_close, beta_sign = _maximum_difference(
            native_step.betas, expected.betas
        )
        lower_diff = float(
            (native_step.lower - production_step.lower).abs().max().item()
        )
        lower_close = bool(
            torch.allclose(
                native_step.lower,
                production_step.lower,
                atol=PARITY_ATOL,
                rtol=PARITY_RTOL,
            )
        )
        lower_sign = torch.equal(
            torch.sign(native_step.lower), torch.sign(production_step.lower)
        )
        lower_maximum = max(lower_maximum, lower_diff)
        alpha_maximum = max(alpha_maximum, alpha_diff)
        beta_maximum = max(beta_maximum, beta_diff)
        rows.append(
            {
                "evaluation_ordinal": native_step.evaluation_ordinal,
                "lower_maximum_absolute_difference": lower_diff,
                "alpha_maximum_absolute_difference": alpha_diff,
                "beta_maximum_absolute_difference": beta_diff,
                "allclose": lower_close and alpha_close and beta_close,
                "sign_exact": lower_sign and alpha_sign and beta_sign,
            }
        )
    parity = NativeProductionOptimizerParityV4(
        native_trace_hash=str(native.metadata()["trace_hash"]),
        production_trace_hash=str(production.metadata()["trace_hash"]),
        step_rows=tuple(rows),
        lower_maximum_absolute_difference=lower_maximum,
        alpha_maximum_absolute_difference=alpha_maximum,
        beta_maximum_absolute_difference=beta_maximum,
    )
    parity.validate()
    return parity


__all__ = [
    "compare_rvir_v4_native_optimizer_trace",
    "execute_rvir_v4_native_optimizer_trace",
    "NativeProductionOptimizerParityV4",
    "NativeProductionOptimizerStepV4",
    "NativeProductionOptimizerTraceV4",
    "PARITY_ATOL",
    "PARITY_RTOL",
]
