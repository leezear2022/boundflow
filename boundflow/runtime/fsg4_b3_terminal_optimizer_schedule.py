"""Terminal-only optimizer Schedule IR and forward-trace handoff for B3-B."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes
# pylint: disable=missing-function-docstring,too-many-boolean-expressions
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    plain_crown_primal_graph_hash,
    relu_split_state_hash,
)
from ..ir.task import BFTaskModule
from .alpha_beta_crown import BetaState, _beta_to_relu_pre_add_coeff
from .crown_ibp import _forward_ibp_trace_mlp, run_crown_ibp_mlp_from_forward_trace
from .fsg4_b3_prepared_core import CorePlanInstanceV1
from .fsg4_b4b_production_region_capture import B4BRegionLiveObserverV1
from .native_alpha_beta_optimization_state import (
    build_native_alpha_beta_scope,
    NativeAlphaBetaOptimizationState,
)
from .rvir_v4_optimizer_mutation import ProductionMutationPolicyV4
from .rvir_v4_production_state import production_tensor_sha256
from .task_executor import InputSpec

FSG4_B3_TERMINAL_SCHEDULE_SCHEMA = "boundflow.fsg4-b3-terminal-schedule/v1"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class TerminalOptimizerActionV1:
    """One immutable evaluate-then-optional-update Schedule IR action."""

    evaluation_ordinal: int
    update_after: bool
    alpha_learning_rate: float
    beta_learning_rate: float

    def validate(self) -> None:
        if (
            self.evaluation_ordinal < 0
            or not math.isfinite(self.alpha_learning_rate)
            or not math.isfinite(self.beta_learning_rate)
            or self.alpha_learning_rate <= 0.0
            or self.beta_learning_rate <= 0.0
        ):
            raise ValueError("FSG4/B3-B terminal optimizer action differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "evaluation_ordinal": self.evaluation_ordinal,
            "update_after": self.update_after,
            "alpha_learning_rate": self.alpha_learning_rate,
            "beta_learning_rate": self.beta_learning_rate,
        }


@dataclass(frozen=True)
class NativeTerminalOptimizerScheduleV1:
    """First-class fixed production loop, separate from formal trace capture."""

    actions: tuple[TerminalOptimizerActionV1, ...]
    policy_contract: str = "rvir-v4-production-mutation/admitted-v1"
    schema_version: str = FSG4_B3_TERMINAL_SCHEDULE_SCHEMA

    @property
    def evaluation_count(self) -> int:
        return len(self.actions)

    @property
    def update_count(self) -> int:
        return sum(action.update_after for action in self.actions)

    def validate(self) -> None:
        if (
            self.schema_version != FSG4_B3_TERMINAL_SCHEDULE_SCHEMA
            or self.policy_contract != "rvir-v4-production-mutation/admitted-v1"
            or self.evaluation_count != 10
            or self.update_count != 9
        ):
            raise ValueError("FSG4/B3-B terminal optimizer schedule differs")
        for ordinal, action in enumerate(self.actions):
            action.validate()
            if (
                action.evaluation_ordinal != ordinal
                or action.update_after != (ordinal < 9)
                or not math.isclose(
                    action.alpha_learning_rate,
                    0.01 * 0.98**ordinal,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    action.beta_learning_rate,
                    0.05 * 0.98**ordinal,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError("FSG4/B3-B terminal optimizer action sequence differs")

    def metadata(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "policy_contract": self.policy_contract,
            "evaluation_count": self.evaluation_count,
            "update_count": self.update_count,
            "actions": [action.to_dict() for action in self.actions],
        }
        payload["schedule_hash"] = _canonical_hash(payload)
        return payload

    def stable_hash(self) -> str:
        return str(self.metadata()["schedule_hash"])


def compile_terminal_optimizer_schedule_v1() -> NativeTerminalOptimizerScheduleV1:
    """Compile the preregistered production action sequence without query state."""

    schedule = NativeTerminalOptimizerScheduleV1(
        actions=tuple(
            TerminalOptimizerActionV1(
                evaluation_ordinal=ordinal,
                update_after=ordinal < 9,
                alpha_learning_rate=0.01 * 0.98**ordinal,
                beta_learning_rate=0.05 * 0.98**ordinal,
            )
            for ordinal in range(10)
        )
    )
    schedule.validate()
    return schedule


@dataclass(frozen=True)
class NativeOptimizerForwardTraceV1:
    """Parent forward trace owned by optimizer and handed to terminal backward."""

    scope_hash: str
    primal_graph_hash: str
    split_state_hash: str
    interval_by_value: tuple[tuple[str, IntervalState], ...]
    local_relu_pre_by_input: tuple[tuple[str, IntervalState], ...]
    schema_version: str = FSG4_B3_TERMINAL_SCHEDULE_SCHEMA

    @property
    def interval_env(self) -> dict[str, IntervalState]:
        return dict(self.interval_by_value)

    @property
    def local_relu_pre(self) -> dict[str, IntervalState]:
        return dict(self.local_relu_pre_by_input)

    def validate(
        self,
        *,
        module: BFTaskModule,
        terminal_state: NativeAlphaBetaOptimizationState,
    ) -> None:
        module.validate()
        terminal_state.validate()
        interval_env = self.interval_env
        local_pre = self.local_relu_pre
        expected_outputs = {
            output for op in module.get_entry_task().ops for output in op.outputs
        }
        reference = next(iter(terminal_state.alphas.values()))
        if (
            self.schema_version != FSG4_B3_TERMINAL_SCHEDULE_SCHEMA
            or not _is_sha256(self.scope_hash)
            or not _is_sha256(self.primal_graph_hash)
            or not _is_sha256(self.split_state_hash)
            or self.scope_hash != terminal_state.scope.stable_hash()
            or self.primal_graph_hash != plain_crown_primal_graph_hash(module)
            or self.split_state_hash != relu_split_state_hash(terminal_state.splits)
            or set(interval_env) != expected_outputs
            or len(interval_env) != len(self.interval_by_value)
            or set(local_pre) != set(terminal_state.splits)
            or len(local_pre) != len(self.local_relu_pre_by_input)
        ):
            raise ValueError("FSG4/B3-B optimizer forward trace identity differs")
        for interval in (*interval_env.values(), *local_pre.values()):
            interval.validate()
            if (
                interval.lower.device != reference.device
                or interval.lower.dtype != reference.dtype
                or not bool(torch.isfinite(interval.lower).all())
                or not bool(torch.isfinite(interval.upper).all())
            ):
                raise ValueError("FSG4/B3-B optimizer forward tensor differs")

    def metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "scope_hash": self.scope_hash,
            "primal_graph_hash": self.primal_graph_hash,
            "split_state_hash": self.split_state_hash,
            "intervals": {
                name: {
                    "lower": production_tensor_sha256(value.lower),
                    "upper": production_tensor_sha256(value.upper),
                }
                for name, value in self.interval_by_value
            },
            "local_relu_pre": {
                name: {
                    "lower": production_tensor_sha256(value.lower),
                    "upper": production_tensor_sha256(value.upper),
                }
                for name, value in self.local_relu_pre_by_input
            },
        }
        payload["forward_trace_hash"] = _canonical_hash(payload)
        return payload


@dataclass(frozen=True)
class NativeTerminalOptimizerResultV1:
    """Only terminal mutable state/lower and one reusable parent forward trace."""

    source_state_hash: str
    mutation_policy_hash: str
    schedule_hash: str
    terminal_lower: torch.Tensor
    terminal_state: NativeAlphaBetaOptimizationState
    forward_trace: NativeOptimizerForwardTraceV1
    evaluation_count: int = 10
    update_count: int = 9
    full_step_snapshot_count: int = 0
    schema_version: str = FSG4_B3_TERMINAL_SCHEDULE_SCHEMA

    def validate(
        self, *, module: BFTaskModule, schedule: NativeTerminalOptimizerScheduleV1
    ) -> None:
        schedule.validate()
        self.terminal_state.validate()
        self.forward_trace.validate(module=module, terminal_state=self.terminal_state)
        if (
            self.schema_version != FSG4_B3_TERMINAL_SCHEDULE_SCHEMA
            or not _is_sha256(self.source_state_hash)
            or not _is_sha256(self.mutation_policy_hash)
            or self.schedule_hash != schedule.stable_hash()
            or self.evaluation_count != schedule.evaluation_count
            or self.update_count != schedule.update_count
            or self.full_step_snapshot_count != 0
            or tuple(self.terminal_lower.shape) != (6, 1)
            or not bool(torch.isfinite(self.terminal_lower).all())
        ):
            raise ValueError("FSG4/B3-B terminal optimizer result differs")

    def metadata(
        self, *, module: BFTaskModule, schedule: NativeTerminalOptimizerScheduleV1
    ) -> dict[str, object]:
        self.validate(module=module, schedule=schedule)
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "source_state_hash": self.source_state_hash,
            "mutation_policy_hash": self.mutation_policy_hash,
            "schedule_hash": self.schedule_hash,
            "terminal_lower_sha256": production_tensor_sha256(self.terminal_lower),
            "terminal_state_hash": self.terminal_state.stable_hash(),
            "forward_trace": self.forward_trace.metadata(),
            "evaluation_count": self.evaluation_count,
            "update_count": self.update_count,
            "full_step_snapshot_count": self.full_step_snapshot_count,
            "provider_callback_count": 0,
            "performance_claimed": False,
        }
        payload["result_hash"] = _canonical_hash(payload)
        return payload


def _expected_scope(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    relu_pre: Mapping[str, IntervalState],
    initial_state: NativeAlphaBetaOptimizationState,
    mutation_policy: ProductionMutationPolicyV4,
    prevalidated_plan: CorePlanInstanceV1 | None,
):
    native_policy = mutation_policy.to_native_policy()
    if prevalidated_plan is None:
        return build_native_alpha_beta_scope(
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
            relu_pre=relu_pre,
            relu_split_state=initial_state.splits,
            policy=native_policy,
        )
    # pylint: disable-next=unidiomatic-typecheck
    if type(prevalidated_plan) is not CorePlanInstanceV1:
        raise TypeError("FSG4/B3-B prevalidated plan differs")
    if (
        prevalidated_plan.initial_state.stable_hash() != initial_state.stable_hash()
        or prevalidated_plan.scope != initial_state.scope
    ):
        raise ValueError("FSG4/B3-B prevalidated state differs")
    return prevalidated_plan.scope


def execute_terminal_optimizer_schedule_v1(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    relu_pre: Mapping[str, IntervalState],
    initial_state: NativeAlphaBetaOptimizationState,
    mutation_policy: ProductionMutationPolicyV4,
    schedule: NativeTerminalOptimizerScheduleV1,
    prevalidated_plan: CorePlanInstanceV1 | None = None,
    b4b_region_observer: B4BRegionLiveObserverV1 | None = None,
) -> NativeTerminalOptimizerResultV1:
    """Execute 10/9 semantics while retaining no per-step state snapshots."""

    schedule.validate()
    mutation_policy.validate()
    initial_state.validate()
    expected_scope = _expected_scope(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        relu_pre=relu_pre,
        initial_state=initial_state,
        mutation_policy=mutation_policy,
        prevalidated_plan=prevalidated_plan,
    )
    if (
        initial_state.scope != expected_scope
        or set(relu_pre) != set(initial_state.splits)
        or schedule.evaluation_count != mutation_policy.evaluation_count
        or schedule.update_count != mutation_policy.update_count
    ):
        raise ValueError("FSG4/B3-B terminal optimizer admission differs")
    interval_env, local_pre = _forward_ibp_trace_mlp(
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
    native_policy = mutation_policy.to_native_policy()
    optimizer = torch.optim.Adam(
        (
            {"params": list(alphas.values()), "lr": native_policy.lr},
            {"params": list(betas.values()), "lr": native_policy.effective_beta_lr},
        )
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=mutation_policy.controls.lr_decay
    )
    terminal_lower: torch.Tensor | None = None
    for action in schedule.actions:
        if not math.isclose(
            float(optimizer.param_groups[0]["lr"]),
            action.alpha_learning_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        ) or not math.isclose(
            float(optimizer.param_groups[1]["lr"]),
            action.beta_learning_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("FSG4/B3-B runtime learning rate differs from schedule")
        relu_pre_add = _beta_to_relu_pre_add_coeff(
            BetaState(betas),
            relu_pre=dict(relu_pre),
            relu_split_state=initial_state.splits,
        )
        if b4b_region_observer is not None:
            b4b_region_observer.begin_evaluation(
                action.evaluation_ordinal,
                native_alphas=alphas,
                native_betas=betas,
                relu_pre_add_coeff_l=relu_pre_add,
            )
        bounds = run_crown_ibp_mlp_from_forward_trace(
            module,
            input_spec,
            interval_env=interval_env,
            relu_pre=dict(relu_pre),
            linear_spec_C=linear_spec_C,
            relu_alpha=alphas,
            relu_pre_add_coeff_l=relu_pre_add,
            b4b_region_observer=b4b_region_observer,
        )
        if action.update_after:
            optimizer.zero_grad(set_to_none=True)
            (-bounds.lower.sum()).backward()
            if b4b_region_observer is not None and action.evaluation_ordinal == 0:
                b4b_region_observer.complete_evaluation(
                    loss_seed=-torch.ones_like(bounds.lower)
                )
            optimizer.step()
            with torch.no_grad():
                for value in alphas.values():
                    value.clamp_(0.0, 1.0)
                for value in betas.values():
                    value.clamp_(min=0.0)
            scheduler.step()
        else:
            terminal_lower = bounds.lower.detach().contiguous().clone()
    if terminal_lower is None:
        raise ValueError("FSG4/B3-B terminal optimizer produced no terminal lower")
    terminal_state = NativeAlphaBetaOptimizationState(
        scope=initial_state.scope,
        split_by_relu_input=initial_state.split_by_relu_input,
        alpha_by_relu_input=tuple(
            (name, value.detach().contiguous().clone())
            for name, value in sorted(alphas.items())
        ),
        beta_by_relu_input=tuple(
            (name, value.detach().contiguous().clone())
            for name, value in sorted(betas.items())
        ),
    )
    forward_trace = NativeOptimizerForwardTraceV1(
        scope_hash=terminal_state.scope.stable_hash(),
        primal_graph_hash=plain_crown_primal_graph_hash(module),
        split_state_hash=relu_split_state_hash(terminal_state.splits),
        interval_by_value=tuple(sorted(interval_env.items())),
        local_relu_pre_by_input=tuple(sorted(local_pre.items())),
    )
    result = NativeTerminalOptimizerResultV1(
        source_state_hash=initial_state.stable_hash(),
        mutation_policy_hash=mutation_policy.stable_hash(),
        schedule_hash=schedule.stable_hash(),
        terminal_lower=terminal_lower,
        terminal_state=terminal_state,
        forward_trace=forward_trace,
    )
    result.validate(module=module, schedule=schedule)
    return result


__all__ = [
    "compile_terminal_optimizer_schedule_v1",
    "execute_terminal_optimizer_schedule_v1",
    "FSG4_B3_TERMINAL_SCHEDULE_SCHEMA",
    "NativeOptimizerForwardTraceV1",
    "NativeTerminalOptimizerResultV1",
    "NativeTerminalOptimizerScheduleV1",
    "TerminalOptimizerActionV1",
]
