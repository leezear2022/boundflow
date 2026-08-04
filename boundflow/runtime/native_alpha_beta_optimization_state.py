"""First-class native alpha/beta state and warm-start validity contracts."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes
# pylint: disable=missing-function-docstring,too-many-boolean-expressions
# pylint: disable=too-many-return-statements

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal, Mapping, Optional

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    plain_crown_primal_graph_hash,
    relu_optimization_state_hash,
    relu_split_state_hash,
    tensor_content_hash,
)
from ..ir.bound import IntermediateBoundSource
from ..ir.task import BFTaskModule
from .alpha_beta_crown import BetaState, run_alpha_beta_crown_mlp
from .alpha_crown import AlphaObjective, AlphaState, SpecReduce
from .crown_ibp import _forward_ibp_trace_mlp
from .native_verifier_ir_integration import (
    NativePlainCrownRepresentationCompilation,
    compile_native_plain_crown_representation_query,
    execute_native_plain_crown_representation_query,
)
from .task_executor import InputSpec
from .task_ir_executor import TaskExecutionTrace

NATIVE_ALPHA_BETA_STATE_SCHEMA_VERSION = "boundflow.native-alpha-beta-state/v1"
WarmStartKind = Literal["exact", "monotonic_split_refinement", "rejected"]
AlphaInitializationMode = Literal["constant", "adaptive"]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NativeAlphaBetaOptimizerPolicy:
    """Frozen runtime optimizer policy that produced one state payload."""

    steps: int
    lr: float
    alpha_init: float = 0.5
    beta_init: float = 0.0
    objective: AlphaObjective = "lower"
    spec_reduce: SpecReduce = "mean"
    soft_tau: float = 1.0
    alpha_initialization_mode: AlphaInitializationMode = "constant"

    def validate(self) -> None:
        if (
            self.steps < 0
            or self.lr <= 0.0
            or not 0.0 <= self.alpha_init <= 1.0
            or self.beta_init < 0.0
            or self.objective not in {"lower", "upper", "gap", "both"}
            or self.spec_reduce not in {"mean", "min", "softmin"}
            or self.alpha_initialization_mode not in {"constant", "adaptive"}
            or self.soft_tau <= 0.0
            or not all(
                torch.isfinite(torch.tensor(value)).item()
                for value in (
                    self.lr,
                    self.alpha_init,
                    self.beta_init,
                    self.soft_tau,
                )
            )
        ):
            raise ValueError("native alpha/beta optimizer policy is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "steps": self.steps,
            "lr": self.lr,
            "alpha_init": self.alpha_init,
            "beta_init": self.beta_init,
            "objective": self.objective,
            "spec_reduce": self.spec_reduce,
            "soft_tau": self.soft_tau,
            "per_batch_params": True,
            "optimizer": "torch.optim.Adam",
        }
        # Compatibility: the historical constant policy keeps its exact v1 hash.
        if self.alpha_initialization_mode != "constant":
            payload["alpha_initialization_mode"] = self.alpha_initialization_mode
        return payload

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeAlphaBetaStateScope:
    """Exact semantic scope in which an optimized state was produced."""

    primal_graph_hash: str
    input_region_hash: str
    objective_hash: str
    intermediate_bounds_hash: str
    split_state_hash: str
    optimizer_policy_hash: str

    def validate(self) -> None:
        for name in (
            "primal_graph_hash",
            "input_region_hash",
            "objective_hash",
            "intermediate_bounds_hash",
            "split_state_hash",
            "optimizer_policy_hash",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"native alpha/beta scope {name} is invalid")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "primal_graph_hash": self.primal_graph_hash,
            "input_region_hash": self.input_region_hash,
            "objective_hash": self.objective_hash,
            "intermediate_bounds_hash": self.intermediate_bounds_hash,
            "split_state_hash": self.split_state_hash,
            "optimizer_policy_hash": self.optimizer_policy_hash,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeAlphaBetaOptimizationState:
    """Exact split/alpha/beta tensors plus their semantic scope."""

    scope: NativeAlphaBetaStateScope
    split_by_relu_input: tuple[tuple[str, torch.Tensor], ...]
    alpha_by_relu_input: tuple[tuple[str, torch.Tensor], ...]
    beta_by_relu_input: tuple[tuple[str, torch.Tensor], ...]
    schema_version: str = NATIVE_ALPHA_BETA_STATE_SCHEMA_VERSION

    @property
    def splits(self) -> dict[str, torch.Tensor]:
        return dict(self.split_by_relu_input)

    @property
    def alphas(self) -> dict[str, torch.Tensor]:
        return dict(self.alpha_by_relu_input)

    @property
    def betas(self) -> dict[str, torch.Tensor]:
        return dict(self.beta_by_relu_input)

    def validate(self) -> None:
        self.scope.validate()
        if self.schema_version != NATIVE_ALPHA_BETA_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported native alpha/beta state schema")
        splits = self.splits
        alphas = self.alphas
        betas = self.betas
        if (
            not splits
            or len(splits) != len(self.split_by_relu_input)
            or len(alphas) != len(self.alpha_by_relu_input)
            or len(betas) != len(self.beta_by_relu_input)
            or set(splits) != set(alphas)
            or set(splits) != set(betas)
        ):
            raise ValueError("native alpha/beta state keys repeat or differ")
        for name in sorted(splits):
            split = splits[name]
            alpha = alphas[name]
            beta = betas[name]
            if (
                not torch.is_tensor(split)
                or not torch.is_tensor(alpha)
                or not torch.is_tensor(beta)
                or split.dtype != torch.int8
                or not torch.is_floating_point(alpha)
                or alpha.dtype != beta.dtype
                or split.shape != alpha.shape
                or split.shape != beta.shape
                or split.device != alpha.device
                or split.device != beta.device
                or not bool(((split >= -1) & (split <= 1)).all().item())
                or not bool(torch.isfinite(alpha).all().item())
                or not bool(torch.isfinite(beta).all().item())
                or not bool(((alpha >= 0) & (alpha <= 1)).all().item())
                or not bool((beta >= 0).all().item())
            ):
                raise ValueError(
                    f"native alpha/beta tensor contract differs for {name}"
                )
        if relu_split_state_hash(splits) != self.scope.split_state_hash:
            raise ValueError("native alpha/beta split hash differs from scope")

    def payload_hash(self) -> str:
        self.validate()
        return relu_optimization_state_hash(self.splits, self.alphas, self.betas)

    def stable_hash(self) -> str:
        return _canonical_hash(
            {
                "schema_version": self.schema_version,
                "scope": self.scope.to_dict(),
                "payload_hash": self.payload_hash(),
            }
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "scope": self.scope.to_dict(),
            "payload_hash": self.payload_hash(),
            "state_hash": self.stable_hash(),
            "relu_states": {
                name: {
                    "split_hash": tensor_content_hash(self.splits[name]),
                    "alpha_hash": tensor_content_hash(self.alphas[name]),
                    "beta_hash": tensor_content_hash(self.betas[name]),
                    "shape": list(self.splits[name].shape),
                    "dtype": str(self.alphas[name].dtype).removeprefix("torch."),
                    "device": str(self.alphas[name].device),
                }
                for name in sorted(self.splits)
            },
        }


@dataclass(frozen=True)
class NativeWarmStartDecision:
    """Fail-closed classification of one source state for a target scope."""

    kind: WarmStartKind
    reason: str
    source_state_hash: str
    target_scope_hash: str
    alpha_initialization_allowed: bool
    beta_initialization_allowed: bool
    exact_state_reuse_allowed: bool

    def validate(self) -> None:
        if (
            self.kind not in {"exact", "monotonic_split_refinement", "rejected"}
            or not self.reason
            or not _is_sha256(self.source_state_hash)
            or not _is_sha256(self.target_scope_hash)
        ):
            raise ValueError("native warm-start decision identity is invalid")
        expected = {
            "exact": (True, True, True),
            "monotonic_split_refinement": (True, True, False),
            "rejected": (False, False, False),
        }[self.kind]
        if (
            self.alpha_initialization_allowed,
            self.beta_initialization_allowed,
            self.exact_state_reuse_allowed,
        ) != expected:
            raise ValueError("native warm-start permissions differ from kind")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "kind": self.kind,
            "reason": self.reason,
            "source_state_hash": self.source_state_hash,
            "target_scope_hash": self.target_scope_hash,
            "alpha_initialization_allowed": self.alpha_initialization_allowed,
            "beta_initialization_allowed": self.beta_initialization_allowed,
            "exact_state_reuse_allowed": self.exact_state_reuse_allowed,
        }


@dataclass(frozen=True)
class NativeAlphaBetaOptimizationResult:
    """Optimizer output together with the exact native state and local trace."""

    bounds: IntervalState
    state: NativeAlphaBetaOptimizationState
    interval_env: Mapping[str, IntervalState]
    relu_pre: Mapping[str, IntervalState]
    warm_start_decision: Optional[NativeWarmStartDecision]

    def validate(self) -> None:
        self.state.validate()
        if self.bounds.lower.shape != self.bounds.upper.shape:
            raise ValueError("native alpha/beta result bounds shape differs")
        if not bool(torch.isfinite(self.bounds.lower).all().item()) or not bool(
            torch.isfinite(self.bounds.upper).all().item()
        ):
            raise ValueError("native alpha/beta result bounds are non-finite")
        if self.warm_start_decision is not None:
            self.warm_start_decision.validate()


def build_native_alpha_beta_scope(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    relu_pre: Mapping[str, IntervalState],
    relu_split_state: Mapping[str, torch.Tensor],
    policy: NativeAlphaBetaOptimizerPolicy,
) -> NativeAlphaBetaStateScope:
    """Build the exact target scope from model, query, bounds, split, and policy."""

    module.validate()
    policy.validate()
    lower, upper = input_spec.perturbation.bounding_box(input_spec.center)
    input_region_hash = _canonical_hash(
        {
            "value_name": input_spec.value_name,
            "center": tensor_content_hash(input_spec.center),
            "lower": tensor_content_hash(lower),
            "upper": tensor_content_hash(upper),
            "perturbation_id": input_spec.perturbation.perturbation_id,
        }
    )
    intermediate_bounds_hash = _canonical_hash(
        {
            name: {
                "lower": tensor_content_hash(state.lower),
                "upper": tensor_content_hash(state.upper),
            }
            for name, state in sorted(relu_pre.items())
        }
    )
    scope = NativeAlphaBetaStateScope(
        primal_graph_hash=plain_crown_primal_graph_hash(module),
        input_region_hash=input_region_hash,
        objective_hash=tensor_content_hash(linear_spec_C),
        intermediate_bounds_hash=intermediate_bounds_hash,
        split_state_hash=relu_split_state_hash(relu_split_state),
        optimizer_policy_hash=policy.stable_hash(),
    )
    scope.validate()
    return scope


def classify_native_alpha_beta_warm_start(
    source: NativeAlphaBetaOptimizationState,
    *,
    target_scope: NativeAlphaBetaStateScope,
    target_split_state: Mapping[str, torch.Tensor],
) -> NativeWarmStartDecision:
    """Allow only exact scope or monotonic split refinement as initialization."""

    source.validate()
    target_scope.validate()
    source_hash = source.stable_hash()
    target_hash = target_scope.stable_hash()

    def decision(kind: WarmStartKind, reason: str) -> NativeWarmStartDecision:
        allowed = kind != "rejected"
        result = NativeWarmStartDecision(
            kind=kind,
            reason=reason,
            source_state_hash=source_hash,
            target_scope_hash=target_hash,
            alpha_initialization_allowed=allowed,
            beta_initialization_allowed=allowed,
            exact_state_reuse_allowed=kind == "exact",
        )
        result.validate()
        return result

    if relu_split_state_hash(target_split_state) != target_scope.split_state_hash:
        return decision("rejected", "target_split_hash_mismatch")
    source_scope = source.scope
    stable_fields = (
        "primal_graph_hash",
        "input_region_hash",
        "objective_hash",
        "optimizer_policy_hash",
    )
    drift = [
        name
        for name in stable_fields
        if getattr(source_scope, name) != getattr(target_scope, name)
    ]
    if drift:
        return decision("rejected", f"semantic_scope_drift:{','.join(drift)}")
    source_splits = source.splits
    if set(source_splits) != set(target_split_state):
        return decision("rejected", "split_key_drift")
    exact = True
    refined = False
    for name in sorted(source_splits):
        old = source_splits[name]
        new = target_split_state[name]
        if old.shape != new.shape or old.dtype != new.dtype or old.device != new.device:
            return decision("rejected", f"split_tensor_schema_drift:{name}")
        if not torch.equal(old, new):
            exact = False
        if bool(((old != 0) & (new != old)).any().item()):
            return decision("rejected", f"split_reversal_or_removal:{name}")
        if bool(((old == 0) & (new != 0)).any().item()):
            refined = True
    if exact:
        if source_scope != target_scope:
            return decision("rejected", "exact_split_scope_drift")
        return decision("exact", "exact_scope_and_split")
    if not refined:
        return decision("rejected", "split_not_monotonic_refinement")
    return decision(
        "monotonic_split_refinement",
        "initialization_only_parent_exact_state_invalid_for_child",
    )


def optimize_native_alpha_beta_state(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    relu_split_state: Mapping[str, torch.Tensor],
    policy: NativeAlphaBetaOptimizerPolicy,
    warm_start: Optional[NativeAlphaBetaOptimizationState] = None,
) -> NativeAlphaBetaOptimizationResult:
    """Run the legacy optimizer, then freeze and validate its native state."""

    policy.validate()
    interval_env, relu_pre = _forward_ibp_trace_mlp(
        module, input_spec, relu_split_state=dict(relu_split_state)
    )
    scope = build_native_alpha_beta_scope(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        relu_pre=relu_pre,
        relu_split_state=relu_split_state,
        policy=policy,
    )
    warm_decision: Optional[NativeWarmStartDecision] = None
    warm_alpha: Optional[AlphaState] = None
    warm_beta: Optional[BetaState] = None
    if warm_start is not None:
        warm_decision = classify_native_alpha_beta_warm_start(
            warm_start,
            target_scope=scope,
            target_split_state=relu_split_state,
        )
        if warm_decision.kind == "rejected":
            raise ValueError(f"native warm start rejected: {warm_decision.reason}")
        warm_alpha = AlphaState(
            {name: value.detach().clone() for name, value in warm_start.alphas.items()}
        )
        warm_beta = BetaState(
            {name: value.detach().clone() for name, value in warm_start.betas.items()}
        )
    bounds, alpha, beta, stats = run_alpha_beta_crown_mlp(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        relu_split_state=dict(relu_split_state),
        steps=policy.steps,
        lr=policy.lr,
        alpha_init=policy.alpha_init,
        beta_init=policy.beta_init,
        warm_start_alpha=warm_alpha,
        warm_start_beta=warm_beta,
        objective=policy.objective,
        spec_reduce=policy.spec_reduce,
        soft_tau=policy.soft_tau,
        per_batch_params=True,
    )
    if stats.feasibility != "unknown" or not alpha.alpha_by_relu_input:
        raise ValueError("native alpha/beta v1 does not freeze infeasible empty state")
    normalized_splits = tuple(
        (name, value.detach().contiguous().clone())
        for name, value in sorted(relu_split_state.items())
    )
    state = NativeAlphaBetaOptimizationState(
        scope=scope,
        split_by_relu_input=normalized_splits,
        alpha_by_relu_input=tuple(
            (name, value.detach().contiguous().clone())
            for name, value in sorted(alpha.alpha_by_relu_input.items())
        ),
        beta_by_relu_input=tuple(
            (name, value.detach().contiguous().clone())
            for name, value in sorted(beta.beta_by_relu_input.items())
        ),
    )
    result = NativeAlphaBetaOptimizationResult(
        bounds=bounds,
        state=state,
        interval_env=interval_env,
        relu_pre=relu_pre,
        warm_start_decision=warm_decision,
    )
    result.validate()
    return result


def compile_native_alpha_beta_state_query(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    optimization: NativeAlphaBetaOptimizationResult,
    query_id: str,
    available_memory_bytes: int = 1 << 30,
    memory_budget_bytes: int = 1 << 30,
    intermediate_bound_source: IntermediateBoundSource = (
        IntermediateBoundSource.LOCAL_FORWARD
    ),
) -> NativePlainCrownRepresentationCompilation:
    """Compile one frozen optimized state through all five native IR layers."""

    optimization.validate()
    if not isinstance(intermediate_bound_source, IntermediateBoundSource):
        raise TypeError("native alpha/beta intermediate-bound source is invalid")
    state = optimization.state
    if state.scope.primal_graph_hash != plain_crown_primal_graph_hash(
        module
    ) or state.scope.objective_hash != tensor_content_hash(linear_spec_C):
        raise ValueError("native alpha/beta compile query scope differs")
    return compile_native_plain_crown_representation_query(
        module,
        input_spec,
        interval_env=optimization.interval_env,
        relu_pre=optimization.relu_pre,
        linear_spec_C=linear_spec_C,
        intermediate_bounds_hash=state.scope.intermediate_bounds_hash,
        query_id=query_id,
        available_memory_bytes=available_memory_bytes,
        memory_budget_bytes=memory_budget_bytes,
        relu_split_state=state.splits,
        split_state_hash=state.scope.split_state_hash,
        relu_alpha_state=state.alphas,
        relu_beta_state=state.betas,
        optimization_state_hash=state.payload_hash(),
        intermediate_bound_source=intermediate_bound_source,
    )


def execute_native_alpha_beta_state_query(
    compilation: NativePlainCrownRepresentationCompilation,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    optimization: NativeAlphaBetaOptimizationResult,
) -> tuple[IntervalState, TaskExecutionTrace]:
    """Execute one frozen optimized state through Task/Schedule semantics."""

    optimization.validate()
    state = optimization.state
    if compilation.build.module.domain.alpha_enabled is not True or (
        compilation.build.module.domain.beta_enabled is not True
    ):
        raise ValueError("native alpha/beta execution received a plain compilation")
    return execute_native_plain_crown_representation_query(
        compilation,
        legacy_task_module=module,
        input_spec=input_spec,
        relu_pre=optimization.relu_pre,
        linear_spec_C=linear_spec_C,
        relu_split_state=state.splits,
        relu_alpha_state=state.alphas,
        relu_beta_state=state.betas,
    )
