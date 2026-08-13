"""Map RVIR-v4 production state into the native frozen-state evaluator."""

# pylint: disable=too-many-locals,too-many-arguments,too-many-branches
# pylint: disable=too-many-statements,missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ..domains.interval import IntervalState
from ..ir.task import BFTaskModule
from .crown_ibp import _forward_ibp_trace_mlp
from .native_alpha_beta_optimization_state import (
    NativeAlphaBetaOptimizationResult,
    NativeAlphaBetaOptimizationState,
    NativeAlphaBetaOptimizerPolicy,
    build_native_alpha_beta_scope,
    compile_native_alpha_beta_state_query,
    execute_native_alpha_beta_state_query,
)
from .rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
    ProductionReluTopologyV4,
)
from .rvir_v4_production_state import (
    ProductionStateSnapshotV4,
    ProductionTensorRole,
)
from .task_executor import InputSpec


@dataclass(frozen=True)
class FrozenStateEvaluationV4:
    """Native execution result and its five-layer IR identities."""

    lower: torch.Tensor
    ir_hashes: Mapping[str, str]
    state_hash: str


def _one_role(
    snapshot: ProductionStateSnapshotV4, role: ProductionTensorRole
) -> torch.Tensor:
    values = [tensor.value for tensor in snapshot.tensors if tensor.role == role]
    if len(values) != 1:
        raise ValueError(f"RVIR-v4 frozen state requires one {role.value}")
    return values[0]


def evaluate_rvir_v4_frozen_state(
    *,
    module: BFTaskModule,
    input_value_name: str,
    pre: ProductionStateSnapshotV4,
    post: ProductionStateSnapshotV4,
    topology: tuple[ProductionReluTopologyV4, ...],
    query_id: str,
    expected_lower: torch.Tensor,
    available_memory_bytes: int = 1 << 40,
) -> FrozenStateEvaluationV4:
    """Execute post-optimized α/β/split state with no provider callback."""

    pre.validate()
    post.validate()
    if not topology or len({item.native_preactivation for item in topology}) != len(
        topology
    ):
        raise ValueError("RVIR-v4 frozen topology keys differ")
    for item in topology:
        item.validate()
    lower = _one_role(pre, ProductionTensorRole.INPUT_LOWER)
    upper = _one_role(pre, ProductionTensorRole.INPUT_UPPER)
    linear_spec = _one_role(pre, ProductionTensorRole.LINEAR_SPEC)
    input_spec = InputSpec.box(
        value_name=input_value_name,
        lower=lower,
        upper=upper,
    )
    mapping = initialize_rvir_v4_native_pre_state(post, topology)
    relu_pre = mapping.relu_pre
    alphas = mapping.alphas
    betas = mapping.betas
    splits = mapping.splits
    policy = NativeAlphaBetaOptimizerPolicy(
        steps=post.optimizer_policy.iteration,
        lr=post.optimizer_policy.alpha_learning_rate,
    )
    interval_env, _local_pre = _forward_ibp_trace_mlp(
        module, input_spec, relu_split_state=splits
    )
    scope = build_native_alpha_beta_scope(
        module,
        input_spec,
        linear_spec_C=linear_spec,
        relu_pre=relu_pre,
        relu_split_state=splits,
        policy=policy,
    )
    state = NativeAlphaBetaOptimizationState(
        scope=scope,
        split_by_relu_input=tuple(sorted(splits.items())),
        alpha_by_relu_input=tuple(sorted(alphas.items())),
        beta_by_relu_input=tuple(sorted(betas.items())),
    )
    optimization = NativeAlphaBetaOptimizationResult(
        bounds=IntervalState(expected_lower, expected_lower),
        state=state,
        interval_env=interval_env,
        relu_pre=relu_pre,
        warm_start_decision=None,
    )
    compilation = compile_native_alpha_beta_state_query(
        module,
        input_spec,
        linear_spec_C=linear_spec,
        optimization=optimization,
        query_id=query_id,
        available_memory_bytes=available_memory_bytes,
        memory_budget_bytes=available_memory_bytes,
    )
    result, _trace = execute_native_alpha_beta_state_query(
        compilation,
        module,
        input_spec,
        linear_spec_C=linear_spec,
        optimization=optimization,
    )
    return FrozenStateEvaluationV4(
        lower=result.lower,
        ir_hashes=compilation.hashes(),
        state_hash=state.stable_hash(),
    )


__all__ = [
    "FrozenStateEvaluationV4",
    "ProductionReluTopologyV4",
    "evaluate_rvir_v4_frozen_state",
]
