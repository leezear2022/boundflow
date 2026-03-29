from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch

from ..domains.interval import IntervalState
from ..ir.task import BFTaskModule, TaskKind
from .crown_ibp import _forward_ibp_trace_mlp, run_crown_ibp_mlp_from_forward_trace
from .relu_shape_utils import coerce_relu_param_shape, relu_input_shapes
from .task_executor import InputSpecLike, _normalize_input_spec

AlphaObjective = Literal["lower", "upper", "gap", "both"]
SpecReduce = Literal["mean", "min", "softmin"]


@dataclass
class AlphaState:
    alpha_by_relu_input: Dict[str, torch.Tensor]

    def detach_clone(self) -> "AlphaState":
        return AlphaState({k: v.detach().clone() for k, v in self.alpha_by_relu_input.items()})


@dataclass(frozen=True)
class AlphaCrownStats:
    steps: int
    lr: float
    objective: AlphaObjective
    spec_reduce: SpecReduce
    soft_tau: float
    relu_inputs: Tuple[str, ...]


def _init_alpha_state(
    relu_shapes: Dict[str, Tuple[int, ...]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    alpha_init: float,
    warm_start: Optional[AlphaState],
) -> AlphaState:
    alpha_by_relu: Dict[str, torch.Tensor] = {}
    for name, shape in relu_shapes.items():
        if warm_start is not None and name in warm_start.alpha_by_relu_input:
            alpha_by_relu[name] = coerce_relu_param_shape(
                warm_start.alpha_by_relu_input[name],
                shape=shape,
                batch_size=1,
                per_batch=False,
                name=name,
                label="warm_start alpha",
                device=device,
                dtype=dtype,
            ).detach().clone()
        else:
            alpha_by_relu[name] = torch.full(shape, float(alpha_init), device=device, dtype=dtype)
    return AlphaState(alpha_by_relu_input=alpha_by_relu)


def _reduce_over_specs(
    x: torch.Tensor,
    *,
    mode: SpecReduce,
    soft_tau: float,
    direction: Literal["min", "max"],
) -> torch.Tensor:
    if x.dim() != 2:
        return x.mean()
    if mode == "mean":
        return x.mean()
    if mode == "min":
        if direction == "min":
            return x.min(dim=1).values.mean()
        return x.max(dim=1).values.mean()
    if mode == "softmin":
        tau = float(soft_tau)
        if tau <= 0.0:
            raise ValueError(f"soft_tau must be >0, got {soft_tau}")
        if direction == "min":
            return (-tau * torch.logsumexp(-x / tau, dim=1)).mean()
        return (tau * torch.logsumexp(x / tau, dim=1)).mean()
    raise AssertionError(f"unreachable spec_reduce: {mode}")


def _objective_metric(
    bounds: IntervalState,
    objective: AlphaObjective,
    *,
    spec_reduce: SpecReduce,
    soft_tau: float,
    lb_weight: float = 1.0,
    ub_weight: float = 1.0,
) -> torch.Tensor:
    if objective == "lower":
        return _reduce_over_specs(bounds.lower, mode=spec_reduce, soft_tau=soft_tau, direction="min")
    if objective == "upper":
        return -_reduce_over_specs(bounds.upper, mode=spec_reduce, soft_tau=soft_tau, direction="max")
    if objective == "gap":
        return -_reduce_over_specs(bounds.upper - bounds.lower, mode=spec_reduce, soft_tau=soft_tau, direction="max")
    if objective == "both":
        lb = _reduce_over_specs(bounds.lower, mode=spec_reduce, soft_tau=soft_tau, direction="min")
        ub = _reduce_over_specs(bounds.upper, mode=spec_reduce, soft_tau=soft_tau, direction="max")
        return float(lb_weight) * lb - float(ub_weight) * ub
    raise AssertionError(f"unreachable objective: {objective}")


def _loss(
    bounds: IntervalState,
    objective: AlphaObjective,
    *,
    spec_reduce: SpecReduce,
    soft_tau: float,
    lb_weight: float = 1.0,
    ub_weight: float = 1.0,
) -> torch.Tensor:
    if objective == "lower":
        return -_reduce_over_specs(bounds.lower, mode=spec_reduce, soft_tau=soft_tau, direction="min")
    if objective == "upper":
        return _reduce_over_specs(bounds.upper, mode=spec_reduce, soft_tau=soft_tau, direction="max")
    if objective == "gap":
        return _reduce_over_specs(bounds.upper - bounds.lower, mode=spec_reduce, soft_tau=soft_tau, direction="max")
    if objective == "both":
        lb = _reduce_over_specs(bounds.lower, mode=spec_reduce, soft_tau=soft_tau, direction="min")
        ub = _reduce_over_specs(bounds.upper, mode=spec_reduce, soft_tau=soft_tau, direction="max")
        return float(ub_weight) * ub - float(lb_weight) * lb
    raise AssertionError(f"unreachable objective: {objective}")


def run_alpha_crown_mlp(
    module: BFTaskModule,
    input_spec: InputSpecLike,
    *,
    linear_spec_C: Optional[torch.Tensor] = None,
    steps: int = 20,
    lr: float = 0.2,
    alpha_init: float = 0.5,
    objective: AlphaObjective = "lower",
    spec_reduce: SpecReduce = "mean",
    soft_tau: float = 1.0,
    lb_weight: float = 1.0,
    ub_weight: float = 1.0,
    warm_start: Optional[AlphaState] = None,
    relu_split_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[IntervalState, AlphaState, AlphaCrownStats]:
    """
    Minimal alpha-CROWN-style optimization loop over the shared single-task CROWN backward path.

    This implementation parameterizes unstable ReLU lower relaxation as y >= alpha * x with alpha in [0,1],
    and optimizes alpha with autograd to tighten a chosen objective.

    Limitations:
    - Single-task graphs only.
    - Supports the same operator subset as run_crown_ibp_mlp, including conv2d/flatten and the current add/concat DAG path.
    - alpha is stored per ReLU node, shared across batch/spec for each element.
    """
    module.validate()
    task = module.get_entry_task()
    if task.kind != TaskKind.INTERVAL_IBP:
        raise NotImplementedError(f"alpha-crown only supports INTERVAL_IBP, got {task.kind}")
    input_spec = _normalize_input_spec(input_spec)
    device = input_spec.center.device
    dtype = input_spec.center.dtype
    if not torch.is_floating_point(input_spec.center):
        raise TypeError(f"alpha-crown expects floating input center, got dtype={input_spec.center.dtype}")

    interval_env, relu_pre = _forward_ibp_trace_mlp(module, input_spec, relu_split_state=relu_split_state)
    state = _init_alpha_state(
        relu_input_shapes(relu_pre),
        device=device,
        dtype=dtype,
        alpha_init=alpha_init,
        warm_start=warm_start,
    )
    params: List[torch.Tensor] = []
    for k, a in state.alpha_by_relu_input.items():
        if not torch.is_floating_point(a):
            raise TypeError(f"alpha[{k}] must be floating, got dtype={a.dtype}")
        a = a.detach().clone().requires_grad_(True)
        state.alpha_by_relu_input[k] = a
        params.append(a)

    opt = torch.optim.Adam(params, lr=float(lr)) if params else None

    best_metric = None
    best_bounds: Optional[IntervalState] = None
    best_state: Optional[AlphaState] = None

    for step in range(int(steps) + 1):
        bounds = run_crown_ibp_mlp_from_forward_trace(
            module,
            input_spec,
            interval_env=interval_env,
            relu_pre=relu_pre,
            linear_spec_C=linear_spec_C,
            relu_alpha=state.alpha_by_relu_input,
        )
        metric = _objective_metric(
            bounds,
            objective,
            spec_reduce=spec_reduce,
            soft_tau=float(soft_tau),
            lb_weight=lb_weight,
            ub_weight=ub_weight,
        )
        metric_val = float(metric.detach().cpu().item())
        best_val = None if best_metric is None else float(best_metric.detach().cpu().item())
        if best_metric is None or (best_val is not None and metric_val > best_val):
            best_metric = metric.detach()
            best_bounds = IntervalState(lower=bounds.lower.detach().clone(), upper=bounds.upper.detach().clone())
            best_state = state.detach_clone()

        if step == int(steps):
            break
        if opt is None:
            break

        loss = _loss(
            bounds,
            objective,
            spec_reduce=spec_reduce,
            soft_tau=float(soft_tau),
            lb_weight=lb_weight,
            ub_weight=ub_weight,
        )
        if not loss.requires_grad:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for a in state.alpha_by_relu_input.values():
                a.clamp_(0.0, 1.0)

    assert best_bounds is not None
    assert best_state is not None
    stats = AlphaCrownStats(
        steps=int(steps),
        lr=float(lr),
        objective=objective,
        spec_reduce=spec_reduce,
        soft_tau=float(soft_tau),
        relu_inputs=tuple(best_state.alpha_by_relu_input.keys()),
    )
    return best_bounds, best_state, stats
