from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from ..domains.interval import IntervalState
from ..ir.task import BFTaskModule, TaskKind
from .alpha_crown import AlphaCrownStats, AlphaObjective, AlphaState, SpecReduce
from .crown_ibp import _forward_ibp_trace_mlp, run_crown_ibp_mlp_from_forward_trace
from .linear_operator import DenseLinearOperator
from .relu_shape_utils import broadcast_relu_split_like_pre, coerce_relu_param_shape, relu_input_shapes, shape_numel
from .task_executor import InputSpecLike, _normalize_input_spec


@dataclass
class BetaState:
    """
    Phase 6F (PR-1): dense beta storage per ReLU input, shape [H] per node.

    Note: this is a correctness-first representation that keeps the computation graph dense/continuous.
    """

    beta_by_relu_input: Dict[str, torch.Tensor]

    def detach_clone(self) -> "BetaState":
        return BetaState({k: v.detach().clone() for k, v in self.beta_by_relu_input.items()})


@dataclass(frozen=True)
class AlphaBetaCrownStats:
    feasibility: Literal["unknown", "infeasible"]
    reason: str
    alpha_stats: Optional[AlphaCrownStats] = None
    infeasible_certificate: Optional[Dict[str, Any]] = None
    branch_choices: Optional[List[Optional[Tuple[str, int]]]] = None


def _has_nonzero_split_state(relu_split_state: Dict[str, torch.Tensor]) -> bool:
    for split in relu_split_state.values():
        if split is None:
            continue
        split_t = split if torch.is_tensor(split) else torch.as_tensor(split)
        if split_t.numel() > 0 and bool((split_t != 0).any().item()):
            return True
    return False


def _flatten_param_for_pre(
    value: torch.Tensor,
    *,
    pre: IntervalState,
    name: str,
    label: str,
) -> torch.Tensor:
    shape = tuple(int(dim) for dim in pre.lower.shape[1:])
    flat_dim = shape_numel(shape)
    batch = int(pre.lower.shape[0])

    if value.dim() == 0:
        return value.reshape(1, 1).expand(batch, flat_dim)
    if tuple(value.shape) == tuple(shape):
        return value.reshape(1, flat_dim).expand(batch, flat_dim)
    if value.dim() == 1 and int(value.shape[0]) == flat_dim:
        return value.reshape(1, flat_dim).expand(batch, flat_dim)
    if value.dim() == len(shape) + 1 and int(value.shape[0]) == 1 and tuple(value.shape[1:]) == tuple(shape):
        return value.reshape(1, flat_dim).expand(batch, flat_dim)
    if value.dim() == 2 and int(value.shape[0]) == 1 and int(value.shape[1]) == flat_dim:
        return value.expand(batch, flat_dim)
    if value.dim() == len(shape) + 1 and int(value.shape[0]) == batch and tuple(value.shape[1:]) == tuple(shape):
        return value.reshape(batch, flat_dim)
    if value.dim() == 2 and int(value.shape[0]) == batch and int(value.shape[1]) == flat_dim:
        return value
    raise ValueError(
        f"{label}[{name}] shape {tuple(value.shape)} cannot broadcast to logical shape {shape} with batch {batch}"
    )


def check_first_layer_infeasible_split(
    module: BFTaskModule,
    input_spec: InputSpecLike,
    *,
    relu_split_state: Dict[str, torch.Tensor],
) -> AlphaBetaCrownStats:
    """
    Best-effort infeasibility check for first-layer split constraints (Phase 6F PR-2).

    This is a thin wrapper that returns an AlphaBetaCrownStats with `feasibility="infeasible"` when a certificate
    is found, otherwise `feasibility="unknown"`.
    """
    module.validate()
    spec = _normalize_input_spec(input_spec)
    infeasible, reason, certificate = _is_infeasible_split_first_layer_convex_combo(
        module,
        spec,
        relu_split_state=relu_split_state,
    )
    if infeasible:
        return AlphaBetaCrownStats(
            feasibility="infeasible",
            reason=reason,
            alpha_stats=None,
            infeasible_certificate=certificate,
            branch_choices=None,
        )
    return AlphaBetaCrownStats(
        feasibility="unknown",
        reason=reason,
        alpha_stats=None,
        infeasible_certificate=None,
        branch_choices=None,
    )


def _branch_choices_from_relu_pre(relu_pre: Dict[str, IntervalState]) -> List[Optional[Tuple[str, int]]]:
    if not relu_pre:
        return []
    # Infer batch size from any relu pre bound.
    any_pre = next(iter(relu_pre.values()))
    bsz = int(any_pre.lower.shape[0])
    best_name: List[Optional[str]] = [None] * bsz
    best_idx: List[int] = [0] * bsz
    best_gap = torch.full((bsz,), float("-inf"), device=any_pre.lower.device, dtype=any_pre.lower.dtype)

    for name, pre in relu_pre.items():
        l = pre.lower.reshape(int(pre.lower.shape[0]), -1)
        u = pre.upper.reshape(int(pre.upper.shape[0]), -1)
        amb = (l < 0) & (u > 0)
        if not amb.any():
            continue
        gap = (u - l).clamp_min(0.0)
        gap = torch.where(amb, gap, torch.full_like(gap, float("-inf")))
        row_best = gap.max(dim=1).values  # [B]
        row_idx = gap.argmax(dim=1)  # [B]
        improve = row_best > best_gap
        if improve.any():
            best_gap = torch.where(improve, row_best, best_gap)
            for i in improve.nonzero(as_tuple=False).flatten().tolist():
                best_name[i] = name
                best_idx[i] = int(row_idx[i].item())

    out: List[Optional[Tuple[str, int]]] = []
    for i in range(bsz):
        if best_name[i] is None:
            out.append(None)
        else:
            out.append((best_name[i], int(best_idx[i])))
    return out


def _init_alpha_state(
    shape_by_relu_input: Dict[str, Tuple[int, ...]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    alpha_init: float,
    warm_start: Optional[AlphaState],
    per_batch_params: bool,
) -> AlphaState:
    alpha_by_relu: Dict[str, torch.Tensor] = {}
    for name, shape in shape_by_relu_input.items():
        if warm_start is not None and name in warm_start.alpha_by_relu_input:
            alpha_by_relu[name] = coerce_relu_param_shape(
                warm_start.alpha_by_relu_input[name],
                shape=shape,
                batch_size=batch_size,
                per_batch=per_batch_params,
                name=name,
                label="warm_start alpha",
                device=device,
                dtype=dtype,
            ).detach().clone()
        else:
            target_shape = (int(batch_size),) + tuple(shape) if per_batch_params else tuple(shape)
            alpha_by_relu[name] = torch.full(target_shape, float(alpha_init), device=device, dtype=dtype)
    return AlphaState(alpha_by_relu_input=alpha_by_relu)


def _init_beta_state(
    shape_by_relu_input: Dict[str, Tuple[int, ...]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    beta_init: float,
    warm_start: Optional[BetaState],
    per_batch_params: bool,
) -> BetaState:
    beta_by_relu: Dict[str, torch.Tensor] = {}
    for name, shape in shape_by_relu_input.items():
        if warm_start is not None and name in warm_start.beta_by_relu_input:
            beta_by_relu[name] = coerce_relu_param_shape(
                warm_start.beta_by_relu_input[name],
                shape=shape,
                batch_size=batch_size,
                per_batch=per_batch_params,
                name=name,
                label="warm_start beta",
                device=device,
                dtype=dtype,
            ).detach().clone()
        else:
            target_shape = (int(batch_size),) + tuple(shape) if per_batch_params else tuple(shape)
            beta_by_relu[name] = torch.full(target_shape, float(beta_init), device=device, dtype=dtype)
    return BetaState(beta_by_relu_input=beta_by_relu)


def _collect_first_layer_split_halfspaces(
    module: BFTaskModule,
    input_spec: InputSpecLike,
    *,
    relu_split_state: Dict[str, torch.Tensor],
) -> List[Tuple[str, int, torch.Tensor, torch.Tensor]]:
    if not relu_split_state or not _has_nonzero_split_state(relu_split_state):
        return []
    spec = _normalize_input_spec(input_spec)
    module.validate()
    task = module.get_entry_task()
    raw_params = module.bindings.get("params", {})
    params: Dict[str, object] = dict(raw_params) if isinstance(raw_params, dict) else {}
    interval_env, relu_pre = _forward_ibp_trace_mlp(module, spec)
    producer_by_output = {op.outputs[0]: op for op in task.ops if op.outputs}

    halfspaces: List[Tuple[str, int, torch.Tensor, torch.Tensor]] = []
    for op in task.ops:
        if op.op_type != "relu":
            continue
        x_name = op.inputs[0]
        split = relu_split_state.get(x_name)
        if split is None or x_name not in relu_pre:
            continue
        split_flat = broadcast_relu_split_like_pre(split, pre=relu_pre[x_name], x_name=x_name, device=spec.center.device)

        prod = producer_by_output.get(x_name)
        if prod is None:
            continue
        if not prod.inputs or prod.inputs[0] != spec.value_name:
            continue
        if int(split_flat.shape[0]) != 1:
            continue
        split_vec = split_flat[0]

        if prod.op_type == "linear":
            w_name = prod.inputs[1]
            b_name = prod.inputs[2] if len(prod.inputs) == 3 else None
            w = params.get(w_name)
            if w is None:
                continue
            W = w if torch.is_tensor(w) else torch.as_tensor(w)
            if W.dim() != 2 or int(W.shape[0]) != int(split_vec.shape[0]):
                continue
            if b_name is None:
                b = torch.zeros(int(W.shape[0]), dtype=W.dtype, device=W.device)
            else:
                bb = params.get(b_name)
                if bb is None:
                    continue
                b = bb if torch.is_tensor(bb) else torch.as_tensor(bb)
                if b.dim() != 1 or int(b.shape[0]) != int(W.shape[0]):
                    continue

            for i in range(int(split_vec.shape[0])):
                s = int(split_vec[i].item())
                if s == 0:
                    continue
                s_t = torch.tensor(float(s), device=W.device, dtype=W.dtype)
                a = s_t * W[i]
                c = s_t * b[i]
                halfspaces.append((x_name, int(i), a, c))
            continue

        if prod.op_type != "conv2d":
            continue

        w_name = prod.inputs[1]
        b_name = prod.inputs[2] if len(prod.inputs) == 3 else None
        weight_raw = params.get(w_name)
        if weight_raw is None:
            continue
        bias_raw = params.get(b_name) if b_name is not None else None
        weight = weight_raw if torch.is_tensor(weight_raw) else torch.as_tensor(weight_raw, device=spec.center.device, dtype=spec.center.dtype)
        weight = weight.to(device=spec.center.device, dtype=spec.center.dtype)
        if weight.dim() != 4:
            continue
        if len(spec.center.shape) != 4:
            continue
        output_shape = tuple(int(dim) for dim in relu_pre[x_name].lower.shape[1:])
        if len(output_shape) != 3:
            continue
        if b_name is None or bias_raw is None:
            bias_vec = torch.zeros(int(weight.shape[0]), device=weight.device, dtype=weight.dtype)
        else:
            bias_vec = bias_raw if torch.is_tensor(bias_raw) else torch.as_tensor(bias_raw, device=weight.device, dtype=weight.dtype)
            bias_vec = bias_vec.to(device=weight.device, dtype=weight.dtype)
            if bias_vec.dim() != 1 or int(bias_vec.shape[0]) != int(weight.shape[0]):
                continue

        out_h = int(output_shape[1])
        out_w = int(output_shape[2])
        for flat_idx in range(int(split_vec.shape[0])):
            s = int(split_vec[flat_idx].item())
            if s == 0:
                continue
            one_hot = torch.zeros((1, 1) + output_shape, device=weight.device, dtype=weight.dtype)
            one_hot.view(-1)[flat_idx] = 1.0
            row_op = DenseLinearOperator(one_hot, input_shape=output_shape).conv2d_right(
                weight,
                stride=prod.attrs.get("stride", 1),
                padding=prod.attrs.get("padding", 0),
                dilation=prod.attrs.get("dilation", 1),
                groups=int(prod.attrs.get("groups", 1)),
                input_shape=tuple(int(dim) for dim in spec.center.shape[1:]),
            )
            a = row_op.to_dense()[0, 0]
            channel = flat_idx // (out_h * out_w)
            c = bias_vec[channel]
            s_t = torch.tensor(float(s), device=a.device, dtype=a.dtype)
            halfspaces.append((x_name, int(flat_idx), s_t * a, s_t * c))
    return halfspaces


def _is_infeasible_split_first_layer_convex_combo(
    module: BFTaskModule,
    input_spec: InputSpecLike,
    *,
    relu_split_state: Dict[str, torch.Tensor],
    steps: int = 200,
    lr: float = 0.1,
    tol: float = 1e-6,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Best-effort infeasibility detector (PR-2):

    If first-layer split constraints {a_i·x + c_i >= 0} have no intersection within the input perturbation set,
    then there exists a convex combination (weights w on simplex) such that:

      max_{x in S}  ( (Σ w_i a_i)·x + Σ w_i c_i ) < 0

    which is a sound certificate of infeasibility.
    """
    spec = _normalize_input_spec(input_spec)
    halfspaces = _collect_first_layer_split_halfspaces(module, spec, relu_split_state=relu_split_state)
    if not halfspaces:
        return False, "ok (no first-layer split halfspaces)", None
    if int(spec.center.shape[0]) != 1:
        return False, "skip (batch!=1)", None
    if not torch.is_floating_point(spec.center):
        return False, "skip (non-floating center)", None

    device = spec.center.device
    dtype = spec.center.dtype
    m = len(halfspaces)
    input_shape = tuple(int(dim) for dim in spec.center.shape[1:])

    a_mat = torch.stack([h[2].to(device=device, dtype=dtype) for h in halfspaces], dim=0)  # [M,I]
    c_vec = torch.stack([h[3].to(device=device, dtype=dtype) for h in halfspaces], dim=0)  # [M]
    if m == 1:
        a = a_mat[0]
        c = c_vec[0]
        A = DenseLinearOperator(a.view(1, 1, -1), input_shape=input_shape)
        b = c.view(1, 1)
        _lb, ub = spec.perturbation.concretize_affine(center=spec.center, A=A, b=b)
        ub0 = float(ub.squeeze().detach().cpu().item())
        if ub0 < -float(tol):
            cert: Dict[str, Any] = {
                "type": "single_halfspace_max_negative",
                "max_value": float(ub0),
                "weights": [1.0],
                "a": a.detach().cpu().tolist(),
                "c": float(c.detach().cpu().item()),
                "halfspaces": [
                    {
                        "relu_input": halfspaces[0][0],
                        "neuron": int(halfspaces[0][1]),
                        "a": a.detach().cpu().tolist(),
                        "c": float(c.detach().cpu().item()),
                    }
                ],
            }
            return True, f"infeasible: single halfspace has max<{ -float(tol):.2e}", cert
        return False, "ok", None

    logits = torch.zeros(m, device=device, dtype=dtype, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=float(lr))

    best = None
    for _ in range(int(steps)):
        w = torch.softmax(logits, dim=0)  # simplex
        a = (w.unsqueeze(1) * a_mat).sum(dim=0)  # [I]
        c = (w * c_vec).sum()  # []
        A = DenseLinearOperator(a.view(1, 1, -1), input_shape=input_shape)  # [B=1,K=1,I]
        b = c.view(1, 1)
        _lb, ub = spec.perturbation.concretize_affine(center=spec.center, A=A, b=b)
        ub0 = ub.squeeze()
        val = float(ub0.detach().cpu().item())
        if best is None or val < best[0]:
            best = (val, w.detach().clone(), a.detach().clone(), c.detach().clone())
        # Early exit: current convex combo already proves infeasible.
        if val < -float(tol):
            cert: Dict[str, Any] = {
                "type": "convex_combo_max_negative",
                "max_value": float(val),
                "weights": w.detach().cpu().tolist(),
                "a": a.detach().cpu().tolist(),
                "c": float(c.detach().cpu().item()),
                "halfspaces": [
                    {"relu_input": name, "neuron": int(i), "a": aa.detach().cpu().tolist(), "c": float(cc.detach().cpu().item())}
                    for (name, i, aa, cc) in halfspaces
                ],
            }
            return True, f"infeasible: found convex-combo certificate with max<{ -float(tol):.2e}", cert
        opt.zero_grad(set_to_none=True)
        ub0.backward()
        opt.step()

    assert best is not None
    best_ub, best_w, best_a, best_c = best
    if best_ub < -float(tol):
        cert: Dict[str, Any] = {
            "type": "convex_combo_max_negative",
            "max_value": float(best_ub),
            "weights": best_w.detach().cpu().tolist(),
            "a": best_a.detach().cpu().tolist(),
            "c": float(best_c.detach().cpu().item()),
            "halfspaces": [
                {"relu_input": name, "neuron": int(i), "a": a.detach().cpu().tolist(), "c": float(c.detach().cpu().item())}
                for (name, i, a, c) in halfspaces
            ],
        }
        return True, f"infeasible: found convex-combo certificate with max<{ -float(tol):.2e}", cert
    return False, "ok", None


def _objective_metric(
    bounds: IntervalState,
    objective: AlphaObjective,
    *,
    spec_reduce: SpecReduce,
    soft_tau: float,
    lb_weight: float = 1.0,
    ub_weight: float = 1.0,
) -> torch.Tensor:
    def _reduce(x: torch.Tensor, *, direction: Literal["min", "max"]) -> torch.Tensor:
        if x.dim() != 2:
            return x.mean()
        if spec_reduce == "mean":
            return x.mean()
        if spec_reduce == "min":
            if direction == "min":
                return x.min(dim=1).values.mean()
            return x.max(dim=1).values.mean()
        if spec_reduce == "softmin":
            tau = float(soft_tau)
            if tau <= 0.0:
                raise ValueError(f"soft_tau must be >0, got {soft_tau}")
            if direction == "min":
                return (-tau * torch.logsumexp(-x / tau, dim=1)).mean()
            return (tau * torch.logsumexp(x / tau, dim=1)).mean()
        raise AssertionError(f"unreachable spec_reduce: {spec_reduce}")

    if objective == "lower":
        return _reduce(bounds.lower, direction="min")
    if objective == "upper":
        return -_reduce(bounds.upper, direction="max")
    if objective == "gap":
        return -_reduce(bounds.upper - bounds.lower, direction="max")
    if objective == "both":
        lb = _reduce(bounds.lower, direction="min")
        ub = _reduce(bounds.upper, direction="max")
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
    def _reduce(x: torch.Tensor, *, direction: Literal["min", "max"]) -> torch.Tensor:
        if x.dim() != 2:
            return x.mean()
        if spec_reduce == "mean":
            return x.mean()
        if spec_reduce == "min":
            if direction == "min":
                return x.min(dim=1).values.mean()
            return x.max(dim=1).values.mean()
        if spec_reduce == "softmin":
            tau = float(soft_tau)
            if tau <= 0.0:
                raise ValueError(f"soft_tau must be >0, got {soft_tau}")
            if direction == "min":
                return (-tau * torch.logsumexp(-x / tau, dim=1)).mean()
            return (tau * torch.logsumexp(x / tau, dim=1)).mean()
        raise AssertionError(f"unreachable spec_reduce: {spec_reduce}")

    if objective == "lower":
        return -_reduce(bounds.lower, direction="min")
    if objective == "upper":
        return _reduce(bounds.upper, direction="max")
    if objective == "gap":
        return _reduce(bounds.upper - bounds.lower, direction="max")
    if objective == "both":
        lb = _reduce(bounds.lower, direction="min")
        ub = _reduce(bounds.upper, direction="max")
        return float(ub_weight) * ub - float(lb_weight) * lb
    raise AssertionError(f"unreachable objective: {objective}")


def _beta_to_relu_pre_add_coeff(
    beta_state: BetaState,
    *,
    relu_pre: Dict[str, IntervalState],
    relu_split_state: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if not relu_split_state:
        return {}
    out: Dict[str, torch.Tensor] = {}
    for name, beta in beta_state.beta_by_relu_input.items():
        split = relu_split_state.get(name)
        pre = relu_pre.get(name)
        if split is None or pre is None:
            continue
        s_b = broadcast_relu_split_like_pre(split, pre=pre, x_name=name, device=beta.device)
        beta_b = _flatten_param_for_pre(beta, pre=pre, name=name, label="beta")
        mask = s_b != 0
        if not mask.any():
            continue
        # split constraint: s * z >= 0  <=>  g(z) = -s*z <= 0
        # Lagrangian term for lower-bound dual: + beta * g(z) = -beta*s*z, beta>=0.
        s_f = s_b.to(dtype=beta.dtype)
        out[name] = -beta_b * s_f * mask.to(dtype=beta.dtype)
    return out


def run_alpha_beta_crown_mlp(
    module: BFTaskModule,
    input_spec: InputSpecLike,
    *,
    linear_spec_C: Optional[torch.Tensor] = None,
    relu_split_state: Optional[Dict[str, torch.Tensor]] = None,
    steps: int = 20,
    lr: float = 0.2,
    alpha_init: float = 0.5,
    beta_init: float = 0.0,
    warm_start_alpha: Optional[AlphaState] = None,
    warm_start_beta: Optional[BetaState] = None,
    objective: AlphaObjective = "lower",
    spec_reduce: SpecReduce = "mean",
    soft_tau: float = 1.0,
    lb_weight: float = 1.0,
    ub_weight: float = 1.0,
    per_batch_params: bool = False,
) -> Tuple[IntervalState, AlphaState, BetaState, AlphaBetaCrownStats]:
    """
    Phase 7A (PR-6): alpha-beta oracle for single-task chain graphs.

    - alpha parameterizes unstable ReLU lower relaxation (Phase 6D).
    - beta encodes split constraints by adding a Lagrangian term on ReLU pre-activations:
        split: s*z >= 0 (s in {-1,+1})  => add beta*(s*z) to the lower-bound dual objective, beta>=0.

    Supported subset:
    - linear / relu / linear
    - conv2d / relu / ... / flatten(start_dim=1,end_dim=-1) / linear

    BaB on conv graphs remains unsupported; this function only extends the alpha-beta oracle.
    """
    module.validate()
    task = module.get_entry_task()
    spec = _normalize_input_spec(input_spec)
    device = spec.center.device
    dtype = spec.center.dtype
    if not torch.is_floating_point(spec.center):
        raise TypeError(f"alpha-beta-crown expects floating input center, got dtype={spec.center.dtype}")
    batch_size = int(spec.center.shape[0])

    split_state = relu_split_state or {}
    do_infeasible_check = batch_size == 1 and _has_nonzero_split_state(split_state)
    if do_infeasible_check:
        infeasible, reason, certificate = _is_infeasible_split_first_layer_convex_combo(
            module, spec, relu_split_state=split_state
        )
        if infeasible:
            zeros = torch.zeros((int(spec.center.shape[0]), 1), device=device, dtype=dtype)
            stats = AlphaBetaCrownStats(
                feasibility="infeasible",
                reason=reason,
                alpha_stats=None,
                infeasible_certificate=certificate,
            )
            return (
                IntervalState(lower=zeros, upper=zeros),
                AlphaState(alpha_by_relu_input={}),
                BetaState(beta_by_relu_input={}),
                stats,
            )

    interval_env, relu_pre = _forward_ibp_trace_mlp(module, spec, relu_split_state=relu_split_state)
    shape_by_relu_input = relu_input_shapes(relu_pre)
    alpha_state = _init_alpha_state(
        shape_by_relu_input,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        alpha_init=alpha_init,
        warm_start=warm_start_alpha,
        per_batch_params=per_batch_params,
    )
    beta_state = _init_beta_state(
        shape_by_relu_input,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        beta_init=beta_init,
        warm_start=warm_start_beta,
        per_batch_params=per_batch_params,
    )

    params: List[torch.Tensor] = []
    for k, a in alpha_state.alpha_by_relu_input.items():
        alpha_state.alpha_by_relu_input[k] = a.detach().clone().requires_grad_(True)
        params.append(alpha_state.alpha_by_relu_input[k])
    for k, b in beta_state.beta_by_relu_input.items():
        beta_state.beta_by_relu_input[k] = b.detach().clone().requires_grad_(True)
        params.append(beta_state.beta_by_relu_input[k])

    opt = torch.optim.Adam(params, lr=float(lr))

    best_metric = None
    best_bounds: Optional[IntervalState] = None
    best_alpha: Optional[AlphaState] = None
    best_beta: Optional[BetaState] = None
    best_metric_scalar: Optional[torch.Tensor] = None
    branch_choices = _branch_choices_from_relu_pre(relu_pre)

    for step in range(int(steps) + 1):
        relu_pre_add = _beta_to_relu_pre_add_coeff(
            beta_state,
            relu_pre=relu_pre,
            relu_split_state=relu_split_state,
        )

        if objective == "lower":
            bounds = run_crown_ibp_mlp_from_forward_trace(
                module,
                spec,
                interval_env=interval_env,
                relu_pre=relu_pre,
                linear_spec_C=linear_spec_C,
                relu_alpha=alpha_state.alpha_by_relu_input,
                relu_pre_add_coeff_l=relu_pre_add,
            )
        else:
            bounds = run_crown_ibp_mlp_from_forward_trace(
                module,
                spec,
                interval_env=interval_env,
                relu_pre=relu_pre,
                linear_spec_C=linear_spec_C,
                relu_alpha=alpha_state.alpha_by_relu_input,
            )

        def _reduce_specs(x: torch.Tensor, *, direction: Literal["min", "max"]) -> torch.Tensor:
            if x.dim() != 2:
                return x.mean().expand(batch_size)
            if spec_reduce == "mean":
                return x.mean(dim=1)
            if spec_reduce == "min":
                if direction == "min":
                    return x.min(dim=1).values
                return x.max(dim=1).values
            if spec_reduce == "softmin":
                tau = float(soft_tau)
                if tau <= 0.0:
                    raise ValueError(f"soft_tau must be >0, got {soft_tau}")
                if direction == "min":
                    return -tau * torch.logsumexp(-x / tau, dim=1)
                return tau * torch.logsumexp(x / tau, dim=1)
            raise AssertionError(f"unreachable spec_reduce: {spec_reduce}")

        if objective == "lower":
            metric_b = _reduce_specs(bounds.lower, direction="min")
        elif objective == "upper":
            metric_b = -_reduce_specs(bounds.upper, direction="max")
        elif objective == "gap":
            metric_b = -_reduce_specs(bounds.upper - bounds.lower, direction="max")
        elif objective == "both":
            metric_b = float(lb_weight) * _reduce_specs(bounds.lower, direction="min") - float(ub_weight) * _reduce_specs(
                bounds.upper, direction="max"
            )
        else:
            raise AssertionError(f"unreachable objective: {objective}")

        metric = metric_b.mean()
        if best_metric is None:
            best_metric = metric_b.detach().clone()
            best_metric_scalar = metric.detach().clone()
            best_bounds = IntervalState(lower=bounds.lower.detach().clone(), upper=bounds.upper.detach().clone())
            best_alpha = AlphaState({k: v.detach().clone() for k, v in alpha_state.alpha_by_relu_input.items()})
            best_beta = BetaState({k: v.detach().clone() for k, v in beta_state.beta_by_relu_input.items()})
        else:
            if per_batch_params and isinstance(best_metric, torch.Tensor) and best_metric.dim() == 1 and metric_b.shape == best_metric.shape:
                improve = metric_b.detach() > best_metric
                if improve.any():
                    best_metric = torch.where(improve, metric_b.detach(), best_metric)
                    assert best_bounds is not None
                    best_bounds = IntervalState(
                        lower=torch.where(improve.unsqueeze(1), bounds.lower.detach(), best_bounds.lower),
                        upper=torch.where(improve.unsqueeze(1), bounds.upper.detach(), best_bounds.upper),
                    )
                    assert best_alpha is not None and best_beta is not None
                    for name, cur in alpha_state.alpha_by_relu_input.items():
                        prev = best_alpha.alpha_by_relu_input[name]
                        if cur.shape == prev.shape and int(cur.shape[0]) == batch_size:
                            mask = improve.view(batch_size, *([1] * (cur.dim() - 1)))
                            best_alpha.alpha_by_relu_input[name] = torch.where(mask, cur.detach(), prev)
                    for name, cur in beta_state.beta_by_relu_input.items():
                        prev = best_beta.beta_by_relu_input[name]
                        if cur.shape == prev.shape and int(cur.shape[0]) == batch_size:
                            mask = improve.view(batch_size, *([1] * (cur.dim() - 1)))
                            best_beta.beta_by_relu_input[name] = torch.where(mask, cur.detach(), prev)
            metric_val = float(metric.detach().cpu().item())
            best_val = (
                float(best_metric_scalar.detach().cpu().item())
                if best_metric_scalar is not None
                else float("-inf")
            )
            if metric_val > best_val:
                best_metric_scalar = metric.detach().clone()
                if not per_batch_params:
                    best_metric = metric_b.detach().clone()
                    best_bounds = IntervalState(lower=bounds.lower.detach().clone(), upper=bounds.upper.detach().clone())
                    best_alpha = AlphaState({k: v.detach().clone() for k, v in alpha_state.alpha_by_relu_input.items()})
                    best_beta = BetaState({k: v.detach().clone() for k, v in beta_state.beta_by_relu_input.items()})

        if step == int(steps):
            break

        loss = -metric_b.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for a in alpha_state.alpha_by_relu_input.values():
                a.clamp_(0.0, 1.0)
            for b in beta_state.beta_by_relu_input.values():
                b.clamp_(0.0)

    assert best_bounds is not None
    assert best_alpha is not None
    assert best_beta is not None
    stats = AlphaBetaCrownStats(
        feasibility="unknown",
        reason="ok",
        alpha_stats=AlphaCrownStats(
            steps=int(steps),
            lr=float(lr),
            objective=objective,
            spec_reduce=spec_reduce,
            soft_tau=float(soft_tau),
            relu_inputs=tuple(best_alpha.alpha_by_relu_input.keys()),
        ),
        branch_choices=branch_choices,
    )
    return best_bounds, best_alpha, best_beta, stats
