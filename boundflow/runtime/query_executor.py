"""Physical batched executors for state-versioned bound queries."""

# Option forwarding deliberately mirrors the serial reference executor so the
# physical batch changes layout only, never verifier semantics.
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..domains.interval import IntervalState
from ..ir.task import BFTaskModule
from ..planner.materialization import BoundMethod
from .alpha_beta_crown import BetaState, run_alpha_beta_crown_mlp
from .alpha_crown import AlphaState
from .bab_query import (
    BoundQueryRequest,
    BoundQueryResult,
    QueryBatch,
    require_float_option,
    require_int_option,
    require_str_option,
    result_from_execution,
)
from .task_executor import InputSpec


def _gather_batch_linear_spec(
    requests: Sequence[BoundQueryRequest],
) -> Optional[torch.Tensor]:
    values = [request.payload.linear_spec_c for request in requests]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("query batch mixes implicit and explicit output specs")
    rows: list[torch.Tensor] = []
    for value in values:
        assert value is not None
        if value.dim() == 2:
            rows.append(value.unsqueeze(0))
        elif value.dim() == 3 and int(value.shape[0]) == 1:
            rows.append(value)
        else:
            raise ValueError(
                "batched query output spec expects rank-2 or singleton rank-3, "
                f"got {tuple(value.shape)}"
            )
    return torch.cat(rows, dim=0)


def _gather_named_state(
    requests: Sequence[BoundQueryRequest],
    *,
    source: str,
    default: float,
) -> dict[str, torch.Tensor]:
    split_names = tuple(requests[0].payload.split_by_relu_input)
    gathered: dict[str, torch.Tensor] = {}
    for name in split_names:
        rows: list[torch.Tensor] = []
        for request in requests:
            split = request.payload.split_by_relu_input.get(name)
            if split is None:
                raise ValueError(f"query is missing split tensor: {name}")
            if source == "split":
                value = split
            elif source == "alpha":
                alpha_value = request.payload.warm_alpha_by_relu_input.get(name)
                if alpha_value is None:
                    alpha_value = torch.full(
                        tuple(split.shape),
                        default,
                        device=split.device,
                        dtype=request.payload.input_spec.center.dtype,
                    )
                value = alpha_value
            elif source == "beta":
                beta_value = request.payload.warm_beta_by_relu_input.get(name)
                if beta_value is None:
                    beta_value = torch.full(
                        tuple(split.shape),
                        default,
                        device=split.device,
                        dtype=request.payload.input_spec.center.dtype,
                    )
                value = beta_value
            else:
                raise AssertionError(f"unknown state source: {source}")
            rows.append(value)
        gathered[name] = torch.stack(rows, dim=0)
    return gathered


def execute_alpha_beta_query_batch(  # pylint: disable=too-many-locals
    module: BFTaskModule,
    batch: QueryBatch,
) -> list[tuple[str, BoundQueryResult]]:
    """Execute one homogeneous αβ batch with per-query split and warm state."""

    batch.validate()
    if batch.key.bound_method != BoundMethod.ALPHA_BETA_CROWN.value:
        raise ValueError("physical αβ batch executor received a different method")
    if batch.key.backend_capability_class != "alpha_beta_dense_split":
        raise ValueError("physical αβ batch executor received an invalid capability")
    requests = batch.requests
    first_spec = requests[0].payload.input_spec
    first_perturbation_id = first_spec.perturbation.perturbation_id
    if any(
        request.payload.input_spec.perturbation.perturbation_id != first_perturbation_id
        for request in requests
    ):
        raise ValueError("query batch mixes perturbation contracts")
    centers = torch.cat(
        [request.payload.input_spec.center for request in requests], dim=0
    )
    batch_spec = InputSpec(
        value_name=first_spec.value_name,
        center=centers,
        perturbation=first_spec.perturbation,
    )
    options = requests[0].query.execution_options
    alpha_init = require_float_option(options, "alpha_init")
    beta_init = require_float_option(options, "beta_init")
    split_batch = _gather_named_state(requests, source="split", default=0.0)
    alpha_batch = _gather_named_state(requests, source="alpha", default=alpha_init)
    beta_batch = _gather_named_state(requests, source="beta", default=beta_init)
    bounds, alpha, beta, stats = run_alpha_beta_crown_mlp(
        module,
        batch_spec,
        linear_spec_C=_gather_batch_linear_spec(requests),
        relu_split_state=split_batch,
        steps=require_int_option(options, "alpha_steps"),
        lr=require_float_option(options, "alpha_lr"),
        alpha_init=alpha_init,
        beta_init=beta_init,
        warm_start_alpha=AlphaState(alpha_batch),
        warm_start_beta=BetaState(beta_batch),
        objective=require_str_option(options, "objective"),  # type: ignore[arg-type]
        spec_reduce=require_str_option(options, "spec_reduce"),  # type: ignore[arg-type]
        soft_tau=require_float_option(options, "soft_tau"),
        lb_weight=require_float_option(options, "lb_weight"),
        ub_weight=require_float_option(options, "ub_weight"),
        per_batch_params=True,
    )
    if stats.feasibility == "infeasible":
        return [
            (
                request.query.query_id,
                result_from_execution(
                    None,
                    None,
                    None,
                    None,
                    status="infeasible",
                    hash_state_versions=False,
                ),
            )
            for request in requests
        ]
    results: list[tuple[str, BoundQueryResult]] = []
    branch_choices = stats.branch_choices or []
    for index, request in enumerate(requests):
        node_bounds = IntervalState(
            lower=bounds.lower[index : index + 1],
            upper=bounds.upper[index : index + 1],
        )
        node_alpha = AlphaState(
            {
                name: value[index].detach().clone()
                for name, value in alpha.alpha_by_relu_input.items()
            }
        )
        node_beta = BetaState(
            {
                name: value[index].detach().clone()
                for name, value in beta.beta_by_relu_input.items()
            }
        )
        branch = branch_choices[index] if index < len(branch_choices) else None
        results.append(
            (
                request.query.query_id,
                result_from_execution(
                    node_bounds,
                    node_alpha,
                    node_beta,
                    branch,
                    # Query lineage already versions warm starts.  Hashing each
                    # GPU slice would force one host synchronization per node;
                    # exact content hashes are reserved for offline traces.
                    hash_state_versions=False,
                ),
            )
        )
    return results


__all__ = ["execute_alpha_beta_query_batch"]
