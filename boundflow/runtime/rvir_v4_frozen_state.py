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
from .rvir_v4_production_state import (
    ProductionStateSnapshotV4,
    ProductionTensorRole,
)
from .task_executor import InputSpec


def _encode(value: str) -> str:
    return value.replace("%", "%25").replace("/", "%2F")


@dataclass(frozen=True)
class ProductionReluTopologyV4:
    """Exact provider activation/preactivation to native primal-value linkage."""

    provider_activation: str
    provider_preactivation: str
    native_preactivation: str
    provider_start_node: str

    def validate(self) -> None:
        if any(
            not value
            for value in (
                self.provider_activation,
                self.provider_preactivation,
                self.native_preactivation,
                self.provider_start_node,
            )
        ):
            raise ValueError("RVIR-v4 frozen ReLU topology differs")


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
    pre_map = pre.tensor_map()
    post_map = post.tensor_map()
    lower = _one_role(pre, ProductionTensorRole.INPUT_LOWER)
    upper = _one_role(pre, ProductionTensorRole.INPUT_UPPER)
    linear_spec = _one_role(pre, ProductionTensorRole.LINEAR_SPEC)
    input_spec = InputSpec.box(
        value_name=input_value_name,
        lower=lower,
        upper=upper,
    )
    relu_pre: dict[str, IntervalState] = {}
    alphas: dict[str, torch.Tensor] = {}
    betas: dict[str, torch.Tensor] = {}
    splits: dict[str, torch.Tensor] = {}
    preactivation_to_native: dict[str, str] = {}
    for link in topology:
        encoded_pre = _encode(link.provider_preactivation)
        encoded_activation = _encode(link.provider_activation)
        encoded_start = _encode(link.provider_start_node)
        native = link.native_preactivation
        preactivation_to_native[link.provider_preactivation] = native
        relu_pre[native] = IntervalState(
            pre_map[f"intermediate/{encoded_pre}/lower"].value,
            pre_map[f"intermediate/{encoded_pre}/upper"].value,
        )
        alpha = post_map[f"alpha/{encoded_activation}/{encoded_start}"].value
        feature_shape = tuple(
            int(value)
            for value in pre_map[
                f"alpha_layout/{encoded_activation}/feature_shape"
            ].value.tolist()
        )
        dense_alpha = torch.zeros((alpha.shape[2],) + feature_shape, dtype=alpha.dtype)
        indices: list[torch.Tensor] = []
        ordinal = 0
        while f"alpha_layout/{encoded_activation}/feature_index/{ordinal}" in pre_map:
            indices.append(
                pre_map[
                    f"alpha_layout/{encoded_activation}/feature_index/{ordinal}"
                ].value
            )
            ordinal += 1
        if indices:
            dense_alpha[(slice(None),) + tuple(indices)] = alpha[0, 0]
        else:
            dense_alpha.copy_(alpha[0, 0].reshape_as(dense_alpha))
        alphas[native] = dense_alpha
        betas[native] = torch.zeros_like(dense_alpha)
        splits[native] = torch.zeros_like(dense_alpha, dtype=torch.int8)
    for history in post.history:
        if not history.locations:
            continue
        native = preactivation_to_native[history.layer_name]
        dense_split = splits[native].reshape(splits[native].shape[0], -1)
        for location, coefficient in zip(history.locations, history.coefficients):
            dense_split[history.domain_ordinal, location] = int(coefficient)
    for link in topology:
        encoded_pre = _encode(link.provider_preactivation)
        prefix = f"beta/{encoded_pre}/0"
        value_item = post_map.get(f"{prefix}/value")
        if value_item is None:
            continue
        locations = post_map[f"{prefix}/location"].value
        values = value_item.value
        dense_beta = betas[link.native_preactivation].reshape(values.shape[0], -1)
        for domain in range(values.shape[0]):
            for slot in range(values.shape[1]):
                dense_beta[domain, int(locations[domain, slot])] = values[domain, slot]
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
