from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from ..domains.interval import IntervalDomain, IntervalState
from ..ir.task import BFTaskModule, TaskKind
from .linear_operator import DenseLinearOperator, LinearOperator
from .perturbation import InputPerturbationState
from .relu_shape_utils import broadcast_relu_split_like_pre
from .task_executor import InputSpec, InputSpecLike, _normalize_input_spec


@dataclass
class CrownIbpStats:
    supported: bool
    reason: str = ""
    ops_seen: Tuple[str, ...] = ()


def _apply_relu_split(pre: IntervalState, split: torch.Tensor, *, relu_input_name: str) -> IntervalState:
    split_b = broadcast_relu_split_like_pre(split, pre=pre, x_name=relu_input_name, device=pre.lower.device)
    lower_flat = pre.lower.reshape(int(pre.lower.shape[0]), -1)
    upper_flat = pre.upper.reshape(int(pre.upper.shape[0]), -1)
    active = split_b > 0
    inactive = split_b < 0
    if not active.any() and not inactive.any():
        return pre
    lower = lower_flat
    upper = upper_flat
    if active.any():
        lower = torch.where(active, torch.maximum(lower, torch.zeros_like(lower)), lower)
    if inactive.any():
        upper = torch.where(inactive, torch.minimum(upper, torch.zeros_like(upper)), upper)
    if (lower > upper).any():
        raise ValueError(f"infeasible relu split for {relu_input_name}: lower>upper after applying split")
    return IntervalState(lower=lower.reshape_as(pre.lower), upper=upper.reshape_as(pre.upper))


def _forward_ibp_trace_mlp(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    relu_split_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, IntervalState], Dict[str, IntervalState]]:
    task = module.get_entry_task()
    raw_params = module.bindings.get("params", {})
    params: Dict[str, Any] = dict(raw_params) if isinstance(raw_params, dict) else {}

    domain = IntervalDomain()
    env: Dict[str, Any] = {
        input_spec.value_name: InputPerturbationState(center=input_spec.center, perturbation=input_spec.perturbation)
    }
    interval_env: Dict[str, IntervalState] = {}
    relu_pre: Dict[str, IntervalState] = {}

    def _get_tensor(name: str) -> Any:
        if name not in params:
            raise KeyError(f"missing param tensor: {name}")
        return params[name]

    def _ensure_interval(state: Any) -> IntervalState:
        if isinstance(state, IntervalState):
            return state
        if isinstance(state, InputPerturbationState):
            lb, ub = state.perturbation.bounding_box(state.center)
            return IntervalState(lower=lb, upper=ub)
        raise TypeError(f"expected IntervalState/InputPerturbationState, got {type(state)}")

    def _get_state(name: str) -> Any:
        if name in env:
            return env[name]
        if name in params:
            t = params[name]
            if not torch.is_tensor(t):
                t = torch.as_tensor(t, device=input_spec.center.device)
            return IntervalState(lower=t, upper=t)
        raise KeyError(f"missing value in env/params: {name}")

    for op in task.ops:
        if op.op_type == "linear":
            x_state = _get_state(op.inputs[0])
            w = _get_tensor(op.inputs[1])
            b = _get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None
            if isinstance(x_state, InputPerturbationState):
                if not torch.is_tensor(w):
                    w = torch.as_tensor(w, device=input_spec.center.device)
                if b is not None and not torch.is_tensor(b):
                    b = torch.as_tensor(b, device=input_spec.center.device)
                lb, ub = x_state.perturbation.concretize_matmul(center=x_state.center, weight=w, bias=b)
                y = IntervalState(lower=lb, upper=ub)
            else:
                x = _ensure_interval(x_state)
                y = domain.affine_transformer(x, w, b, op="linear")
            env[op.outputs[0]] = y
            interval_env[op.outputs[0]] = y
            continue

        if op.op_type == "conv2d":
            x_state = _get_state(op.inputs[0])
            w = _get_tensor(op.inputs[1])
            b = _get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None
            attrs = dict(op.attrs)
            attrs.setdefault("op", "conv2d")
            if isinstance(x_state, InputPerturbationState):
                lb, ub = x_state.perturbation.bounding_box(x_state.center)
                x = IntervalState(lower=lb, upper=ub)
            else:
                x = _ensure_interval(x_state)
            y = domain.affine_transformer(x, w, b, **attrs)
            env[op.outputs[0]] = y
            interval_env[op.outputs[0]] = y
            continue

        if op.op_type == "relu":
            x_name = op.inputs[0]
            x = _ensure_interval(_get_state(x_name))
            if relu_split_state is not None and x_name in relu_split_state:
                x = _apply_relu_split(x, relu_split_state[x_name], relu_input_name=x_name)
            relu_pre[x_name] = x
            y = domain.relu_transformer(x)
            env[op.outputs[0]] = y
            interval_env[op.outputs[0]] = y
            continue

        if op.op_type == "flatten":
            start_dim = int(op.attrs.get("start_dim", 1))
            end_dim = int(op.attrs.get("end_dim", -1))
            if start_dim != 1 or end_dim != -1:
                raise NotImplementedError(
                    "_forward_ibp_trace_mlp only supports flatten(start_dim=1, end_dim=-1)"
                )
            x_state = _get_state(op.inputs[0])
            if isinstance(x_state, InputPerturbationState):
                env[op.outputs[0]] = InputPerturbationState(
                    center=torch.flatten(x_state.center, start_dim=start_dim, end_dim=end_dim),
                    perturbation=x_state.perturbation,
                )
            else:
                x = _ensure_interval(x_state)
                y = IntervalState(
                    lower=torch.flatten(x.lower, start_dim=start_dim, end_dim=end_dim),
                    upper=torch.flatten(x.upper, start_dim=start_dim, end_dim=end_dim),
                )
                env[op.outputs[0]] = y
                interval_env[op.outputs[0]] = y
            continue

        raise NotImplementedError(f"_forward_ibp_trace_mlp unsupported op_type: {op.op_type}")

    return interval_env, relu_pre


@dataclass
class AffineBackwardState:
    A_u: LinearOperator
    A_l: LinearOperator
    b_u: torch.Tensor
    b_l: torch.Tensor


def _resolve_output_value(task: Any, output_value: Optional[str], *, caller: str) -> str:
    if output_value is None:
        if len(task.output_values) != 1:
            raise ValueError(f"task has {len(task.output_values)} outputs; specify output_value explicitly")
        output_value = task.output_values[0]
    last_out = task.ops[-1].outputs[0] if task.ops[-1].outputs else None
    if output_value != last_out:
        raise NotImplementedError(
            f"{caller} currently supports using the last op output only (got output_value={output_value}, last={last_out})"
        )
    return output_value


def _get_output_meta(
    *,
    interval_env: Dict[str, IntervalState],
    output_value: str,
    caller: str,
) -> Tuple[IntervalState, int, int, torch.device, torch.dtype]:
    if output_value not in interval_env:
        raise KeyError(f"output_value not produced in forward trace: {output_value}")
    y_out = interval_env[output_value]
    if y_out.lower.dim() != 2:
        raise ValueError(f"{caller} expects output rank-2 [B,O], got {tuple(y_out.lower.shape)}")
    if not torch.is_floating_point(y_out.lower) or not torch.is_floating_point(y_out.upper):
        raise TypeError(f"{caller} expects floating bounds, got dtype={y_out.lower.dtype}")
    batch = int(y_out.lower.shape[0])
    out_dim = int(y_out.lower.shape[1])
    device = y_out.lower.device
    dtype = y_out.lower.dtype
    return y_out, batch, out_dim, device, dtype


def _init_backward_state(
    *,
    batch: int,
    out_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    linear_spec_C: Optional[torch.Tensor],
) -> AffineBackwardState:
    if linear_spec_C is None:
        A = torch.eye(out_dim, device=device, dtype=dtype).unsqueeze(0).expand(batch, out_dim, out_dim).clone()
        b = torch.zeros(batch, out_dim, device=device, dtype=dtype)
        return AffineBackwardState(
            A_u=DenseLinearOperator(A),
            A_l=DenseLinearOperator(A.clone()),
            b_u=b,
            b_l=b.clone(),
        )

    C = linear_spec_C
    if not torch.is_tensor(C):
        C = torch.as_tensor(C, device=device, dtype=dtype)
    else:
        C = C.to(device=device, dtype=dtype)
    if C.dim() == 2:
        if int(C.shape[1]) != out_dim:
            raise ValueError(f"linear_spec_C shape mismatch: C={tuple(C.shape)} out=({batch}, {out_dim})")
        C = C.unsqueeze(0).expand(batch, int(C.shape[0]), out_dim).clone()
    if C.dim() != 3:
        raise ValueError(f"linear_spec_C expects rank-3 [B,S,O], got {tuple(C.shape)}")
    if int(C.shape[0]) != batch or int(C.shape[2]) != out_dim:
        raise ValueError(f"linear_spec_C shape mismatch: C={tuple(C.shape)} out=({batch}, {out_dim})")
    b = torch.zeros(int(C.shape[0]), int(C.shape[1]), device=device, dtype=dtype)
    return AffineBackwardState(
        A_u=DenseLinearOperator(C),
        A_l=DenseLinearOperator(C.clone()),
        b_u=b,
        b_l=b.clone(),
    )


def _relu_relax(l: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.is_floating_point(l) or not torch.is_floating_point(u):
        raise TypeError(f"_relu_relax expects floating tensors, got l={l.dtype} u={u.dtype}")
    pos = l >= 0
    neg = u <= 0
    amb = (~pos) & (~neg)

    alpha_u = torch.empty_like(l)
    beta_u = torch.empty_like(l)
    alpha_l = torch.empty_like(l)
    beta_l = torch.empty_like(l)

    alpha_u[pos] = 1.0
    beta_u[pos] = 0.0
    alpha_l[pos] = 1.0
    beta_l[pos] = 0.0

    alpha_u[neg] = 0.0
    beta_u[neg] = 0.0
    alpha_l[neg] = 0.0
    beta_l[neg] = 0.0

    if amb.any():
        l_amb = l[amb]
        u_amb = u[amb]
        denom = (u_amb - l_amb).clamp_min(torch.finfo(l.dtype).eps)
        a = u_amb / denom
        alpha_u[amb] = a
        beta_u[amb] = -l_amb * a
        alpha_l[amb] = 0.0
        beta_l[amb] = 0.0

    return alpha_u, beta_u, alpha_l, beta_l


def _normalize_linear_inputs(
    w_raw: Any,
    bias_raw: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    caller: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    w = w_raw if torch.is_tensor(w_raw) else torch.as_tensor(w_raw, device=device, dtype=dtype)
    w = w.to(device=device, dtype=dtype)
    if w.dim() != 2:
        raise NotImplementedError(f"{caller} currently supports non-batched linear weights only")
    out_dim = int(w.shape[0])

    if bias_raw is None:
        bias = torch.zeros(out_dim, device=device, dtype=dtype)
        return w, bias

    bias = bias_raw if torch.is_tensor(bias_raw) else torch.as_tensor(bias_raw, device=device, dtype=dtype)
    bias = bias.to(device=device, dtype=dtype)
    if bias.dim() == 0:
        return w, bias.expand(out_dim)
    if bias.dim() == 1 and int(bias.shape[0]) == out_dim:
        return w, bias
    raise NotImplementedError(f"{caller} expects linear bias scalar or rank-1 [O], got {tuple(bias.shape)}")


def _as_pair(value: Any, *, name: str) -> Tuple[int, int]:
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return (int(value[0]), int(value[0]))
        if len(value) == 2:
            return (int(value[0]), int(value[1]))
    raise ValueError(f"{name} expects int or pair, got {value}")


def _normalize_conv2d_inputs(
    weight_raw: Any,
    bias_raw: Any,
    *,
    attrs: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    caller: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int], Tuple[int, int], Tuple[int, int], int]:
    w = weight_raw if torch.is_tensor(weight_raw) else torch.as_tensor(weight_raw, device=device, dtype=dtype)
    w = w.to(device=device, dtype=dtype)
    if w.dim() != 4:
        raise NotImplementedError(f"{caller} currently supports rank-4 conv2d weights only, got {tuple(w.shape)}")

    bias: Optional[torch.Tensor]
    if bias_raw is None:
        bias = None
    else:
        bias_t = bias_raw if torch.is_tensor(bias_raw) else torch.as_tensor(bias_raw, device=device, dtype=dtype)
        bias_t = bias_t.to(device=device, dtype=dtype)
        if bias_t.dim() != 1 or int(bias_t.shape[0]) != int(w.shape[0]):
            raise NotImplementedError(
                f"{caller} expects conv2d bias rank-1 [O], got {tuple(bias_t.shape)} for weight {tuple(w.shape)}"
            )
        bias = bias_t

    stride = _as_pair(attrs.get("stride", 1), name="stride")
    padding = _as_pair(attrs.get("padding", 0), name="padding")
    dilation = _as_pair(attrs.get("dilation", 1), name="dilation")
    groups = int(attrs.get("groups", 1))
    return w, bias, stride, padding, dilation, groups


def _value_shape(
    *,
    input_spec: InputSpec,
    interval_env: Dict[str, IntervalState],
    value_name: str,
) -> Tuple[int, ...]:
    if value_name == input_spec.value_name:
        return tuple(int(dim) for dim in input_spec.center.shape[1:])
    if value_name not in interval_env:
        raise KeyError(f"missing interval trace for value shape lookup: {value_name}")
    return tuple(int(dim) for dim in interval_env[value_name].lower.shape[1:])


def _broadcast_relu_alpha(
    alpha_raw: Any,
    *,
    pre: IntervalState,
    x_name: str,
    device: torch.device,
    dtype: torch.dtype,
    caller: str,
) -> torch.Tensor:
    alpha = alpha_raw if torch.is_tensor(alpha_raw) else torch.as_tensor(alpha_raw, device=device, dtype=dtype)
    alpha = alpha.to(device=device, dtype=dtype)
    if not torch.is_floating_point(alpha):
        raise TypeError(f"relu_alpha[{x_name}] must be floating, got dtype={alpha.dtype}")
    logical_shape = tuple(int(dim) for dim in pre.lower.shape[1:])
    flat_dim = 1
    for dim in logical_shape:
        flat_dim *= int(dim)
    batch = int(pre.lower.shape[0])
    target_shape = (batch, flat_dim)

    if alpha.dim() == 0:
        out = alpha.reshape(1, 1).expand(target_shape)
    elif tuple(alpha.shape) == logical_shape:
        out = alpha.reshape(1, flat_dim).expand(target_shape)
    elif alpha.dim() == 1 and int(alpha.shape[0]) == flat_dim:
        out = alpha.reshape(1, flat_dim).expand(target_shape)
    elif alpha.dim() == len(logical_shape) + 1 and int(alpha.shape[0]) == 1 and tuple(alpha.shape[1:]) == logical_shape:
        out = alpha.reshape(1, flat_dim).expand(target_shape)
    elif alpha.dim() == 2 and int(alpha.shape[0]) == 1 and int(alpha.shape[1]) == flat_dim:
        out = alpha.expand(target_shape)
    elif alpha.dim() == len(logical_shape) + 1 and int(alpha.shape[0]) == batch and tuple(alpha.shape[1:]) == logical_shape:
        out = alpha.reshape(batch, flat_dim)
    elif alpha.dim() == 2 and int(alpha.shape[0]) == batch and int(alpha.shape[1]) == flat_dim:
        out = alpha
    else:
        raise ValueError(
            f"relu_alpha[{x_name}] shape {tuple(alpha.shape)} cannot broadcast to logical shape "
            f"{logical_shape} (flat_dim={flat_dim}, batch={batch})"
        )
    return out.clamp(0.0, 1.0)


def _apply_relu_pre_add_coeff(
    A: torch.Tensor,
    add_raw: Any,
    *,
    x_name: str,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    add = add_raw if torch.is_tensor(add_raw) else torch.as_tensor(add_raw, device=device, dtype=dtype)
    add = add.to(device=device, dtype=dtype)
    flat_dim = int(A.shape[2])
    batch = int(A.shape[0])
    if add.dim() == 0:
        add = add.expand(flat_dim)
    if add.dim() == 1:
        if int(add.shape[0]) != flat_dim:
            raise ValueError(
                f"{label}[{x_name}] shape {tuple(add.shape)} does not match expected ({flat_dim},)"
            )
        return A + add.view(1, 1, -1)
    if add.dim() == 2:
        if int(add.shape[1]) != flat_dim:
            raise ValueError(
                f"{label}[{x_name}] shape {tuple(add.shape)} does not match expected (*,{flat_dim})"
            )
        if int(add.shape[0]) == 1:
            add_b = add.expand(batch, -1)
        elif int(add.shape[0]) == batch:
            add_b = add
        else:
            raise ValueError(f"{label}[{x_name}] shape {tuple(add.shape)} does not match batch {batch}")
        return A + add_b.unsqueeze(1)
    total = int(add.numel())
    if total == flat_dim:
        return A + add.reshape(1, 1, flat_dim)
    if total == batch * flat_dim and int(add.shape[0]) == batch:
        return A + add.reshape(batch, flat_dim).unsqueeze(1)
    raise ValueError(f"{label}[{x_name}] expects shape broadcastable to [B,{flat_dim}], got {tuple(add.shape)}")


def _backprop_linear_step(
    state: AffineBackwardState,
    *,
    weight: Any,
    bias: Any,
    device: torch.device,
    dtype: torch.dtype,
    caller: str,
) -> AffineBackwardState:
    w, bias_vec = _normalize_linear_inputs(weight, bias, device=device, dtype=dtype, caller=caller)
    return AffineBackwardState(
        A_u=state.A_u.matmul_right(w),
        A_l=state.A_l.matmul_right(w),
        b_u=state.b_u + state.A_u.contract_input(bias_vec),
        b_l=state.b_l + state.A_l.contract_input(bias_vec),
    )


def _backprop_flatten_step(
    state: AffineBackwardState,
    *,
    pre_shape: Tuple[int, ...],
) -> AffineBackwardState:
    return AffineBackwardState(
        A_u=state.A_u.reshape_input(pre_shape),
        A_l=state.A_l.reshape_input(pre_shape),
        b_u=state.b_u,
        b_l=state.b_l,
    )


def _backprop_conv2d_step(
    state: AffineBackwardState,
    *,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    weight: Any,
    bias: Any,
    attrs: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    caller: str,
) -> AffineBackwardState:
    if len(input_shape) != 3:
        raise NotImplementedError(f"{caller} currently supports NCHW conv2d inputs only, got {input_shape}")
    if len(output_shape) != 3:
        raise NotImplementedError(f"{caller} expects conv2d output rank-3 without batch, got {output_shape}")

    w, bias_vec, stride, padding, dilation, groups = _normalize_conv2d_inputs(
        weight,
        bias,
        attrs=attrs,
        device=device,
        dtype=dtype,
        caller=caller,
    )

    A_u_base = state.A_u if tuple(state.A_u.input_shape) == tuple(output_shape) else state.A_u.reshape_input(output_shape)
    A_l_base = state.A_l if tuple(state.A_l.input_shape) == tuple(output_shape) else state.A_l.reshape_input(output_shape)

    b_u = state.b_u
    b_l = state.b_l
    if bias_vec is not None:
        bias_map = bias_vec.view(-1, 1, 1).expand(output_shape)
        b_u = b_u + A_u_base.contract_input(bias_map)
        b_l = b_l + A_l_base.contract_input(bias_map)

    return AffineBackwardState(
        A_u=A_u_base.conv2d_right(
            w,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            input_shape=input_shape,
        ),
        A_l=A_l_base.conv2d_right(
            w,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            input_shape=input_shape,
        ),
        b_u=b_u,
        b_l=b_l,
    )


def _backprop_relu_step(
    state: AffineBackwardState,
    *,
    pre: IntervalState,
    x_name: str,
    relu_alpha: Optional[Dict[str, torch.Tensor]],
    relu_pre_add_coeff_u: Optional[Dict[str, torch.Tensor]],
    relu_pre_add_coeff_l: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    caller: str,
) -> AffineBackwardState:
    orig_input_shape = tuple(int(dim) for dim in pre.lower.shape[1:])
    pre_flat = IntervalState(
        lower=pre.lower.reshape(int(pre.lower.shape[0]), -1),
        upper=pre.upper.reshape(int(pre.upper.shape[0]), -1),
    )
    if pre_flat.lower.shape[1] != state.A_u.input_numel or pre_flat.lower.shape[1] != state.A_l.input_numel:
        raise ValueError(
            f"{caller} relu backward shape mismatch: pre={tuple(pre.lower.shape)} "
            f"A_u.input_shape={state.A_u.input_shape} A_l.input_shape={state.A_l.input_shape}"
        )
    if pre.lower.dim() != 2:
        if relu_pre_add_coeff_u is not None and x_name in relu_pre_add_coeff_u:
            raise NotImplementedError(
                f"{caller} only supports relu_pre_add_coeff_u on rank-2 pre-activations; got {tuple(pre.lower.shape)} for {x_name}"
            )

    A_u = state.A_u.to_dense()
    A_l = state.A_l.to_dense()
    b_u = state.b_u
    b_l = state.b_l

    alpha_u, beta_u, alpha_l, beta_l = _relu_relax(pre_flat.lower, pre_flat.upper)
    if relu_alpha is not None and x_name in relu_alpha:
        alpha_broadcast = _broadcast_relu_alpha(
            relu_alpha[x_name],
            pre=pre,
            x_name=x_name,
            device=device,
            dtype=dtype,
            caller=caller,
        )
        amb = (pre_flat.lower < 0) & (pre_flat.upper > 0)
        if amb.any():
            alpha_l = torch.where(amb, alpha_broadcast, alpha_l)

    sel_alpha_u = torch.where(A_u >= 0, alpha_u.unsqueeze(1), alpha_l.unsqueeze(1))
    sel_beta_u = torch.where(A_u >= 0, beta_u.unsqueeze(1), beta_l.unsqueeze(1))
    b_u = b_u + (A_u * sel_beta_u).sum(dim=2)
    A_u = A_u * sel_alpha_u
    if relu_pre_add_coeff_u is not None and x_name in relu_pre_add_coeff_u:
        A_u = _apply_relu_pre_add_coeff(
            A_u,
            relu_pre_add_coeff_u[x_name],
            x_name=x_name,
            label="relu_pre_add_coeff_u",
            device=device,
            dtype=dtype,
        )

    sel_alpha_l = torch.where(A_l >= 0, alpha_l.unsqueeze(1), alpha_u.unsqueeze(1))
    sel_beta_l = torch.where(A_l >= 0, beta_l.unsqueeze(1), beta_u.unsqueeze(1))
    b_l = b_l + (A_l * sel_beta_l).sum(dim=2)
    A_l = A_l * sel_alpha_l
    if relu_pre_add_coeff_l is not None and x_name in relu_pre_add_coeff_l:
        A_l = _apply_relu_pre_add_coeff(
            A_l,
            relu_pre_add_coeff_l[x_name],
            x_name=x_name,
            label="relu_pre_add_coeff_l",
            device=device,
            dtype=dtype,
        )

    return AffineBackwardState(
        A_u=DenseLinearOperator(A_u, input_shape=orig_input_shape),
        A_l=DenseLinearOperator(A_l, input_shape=orig_input_shape),
        b_u=b_u,
        b_l=b_l,
    )


def _run_crown_backward_from_trace(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    interval_env: Dict[str, IntervalState],
    relu_pre: Dict[str, IntervalState],
    linear_spec_C: Optional[torch.Tensor],
    output_value: Optional[str],
    relu_alpha: Optional[Dict[str, torch.Tensor]],
    relu_pre_add_coeff_u: Optional[Dict[str, torch.Tensor]],
    relu_pre_add_coeff_l: Optional[Dict[str, torch.Tensor]],
    caller: str,
) -> IntervalState:
    task = module.get_entry_task()
    raw_params = module.bindings.get("params", {})
    params: Dict[str, Any] = dict(raw_params) if isinstance(raw_params, dict) else {}

    def _get_tensor(name: str) -> Any:
        if name not in params:
            raise KeyError(f"missing param tensor: {name}")
        return params[name]

    resolved_output = _resolve_output_value(task, output_value, caller=caller)
    _y_out, batch, out_dim, device, dtype = _get_output_meta(
        interval_env=interval_env,
        output_value=resolved_output,
        caller=caller,
    )
    state = _init_backward_state(batch=batch, out_dim=out_dim, device=device, dtype=dtype, linear_spec_C=linear_spec_C)

    for op in reversed(task.ops):
        if op.op_type == "linear":
            state = _backprop_linear_step(
                state,
                weight=_get_tensor(op.inputs[1]),
                bias=_get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None,
                device=device,
                dtype=dtype,
                caller=caller,
            )
            continue

        if op.op_type == "flatten":
            start_dim = int(op.attrs.get("start_dim", 1))
            end_dim = int(op.attrs.get("end_dim", -1))
            if start_dim != 1 or end_dim != -1:
                raise NotImplementedError(
                    f"{caller} only supports flatten(start_dim=1, end_dim=-1), got attrs={op.attrs}"
                )
            state = _backprop_flatten_step(
                state,
                pre_shape=_value_shape(input_spec=input_spec, interval_env=interval_env, value_name=op.inputs[0]),
            )
            continue

        if op.op_type == "relu":
            x_name = op.inputs[0]
            if x_name not in relu_pre:
                raise KeyError(f"missing relu pre-activation bounds for value: {x_name}")
            state = _backprop_relu_step(
                state,
                pre=relu_pre[x_name],
                x_name=x_name,
                relu_alpha=relu_alpha,
                relu_pre_add_coeff_u=relu_pre_add_coeff_u,
                relu_pre_add_coeff_l=relu_pre_add_coeff_l,
                device=device,
                dtype=dtype,
                caller=caller,
            )
            continue

        if op.op_type == "conv2d":
            state = _backprop_conv2d_step(
                state,
                input_shape=_value_shape(input_spec=input_spec, interval_env=interval_env, value_name=op.inputs[0]),
                output_shape=_value_shape(input_spec=input_spec, interval_env=interval_env, value_name=op.outputs[0]),
                weight=_get_tensor(op.inputs[1]),
                bias=_get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None,
                attrs=dict(op.attrs),
                device=device,
                dtype=dtype,
                caller=caller,
            )
            continue

        raise NotImplementedError(f"run_crown_ibp_mlp unsupported op_type in backward: {op.op_type}")

    x0 = input_spec.center
    _lb_u, ub_u = input_spec.perturbation.concretize_affine(center=x0, A=state.A_u, b=state.b_u)
    lb_l, _ub_l = input_spec.perturbation.concretize_affine(center=x0, A=state.A_l, b=state.b_l)
    return IntervalState(lower=lb_l, upper=ub_u)


def run_crown_ibp_mlp(
    module: BFTaskModule,
    input_spec: InputSpecLike,
    *,
    linear_spec_C: Optional[torch.Tensor] = None,
    output_value: Optional[str] = None,
    relu_alpha: Optional[Dict[str, torch.Tensor]] = None,
    relu_pre_add_coeff_u: Optional[Dict[str, torch.Tensor]] = None,
    relu_pre_add_coeff_l: Optional[Dict[str, torch.Tensor]] = None,
    relu_split_state: Optional[Dict[str, torch.Tensor]] = None,
) -> IntervalState:
    """
    Minimal CROWN-IBP for chain-structured graphs.

    - Forward: interval IBP to get pre-activation bounds for ReLU.
    - Backward: CROWN-style linear bound propagation using ReLU relaxations fixed by IBP bounds.

    Limitations:
    - Single task only.
    - Supports chain op_type in {"linear", "relu", "conv2d", "flatten"}.
    - `flatten` is restricted to `start_dim=1, end_dim=-1`.
    - Conv support is plain CROWN-IBP only; alpha/beta/BaB remain MLP-only.
    - Returns bounds for a single output value (rank-2 [B,O]).
    """
    module.validate()
    task = module.get_entry_task()
    if task.kind != TaskKind.INTERVAL_IBP:
        raise NotImplementedError(f"run_crown_ibp_mlp only supports INTERVAL_IBP, got {task.kind}")
    if module.task_graph is not None or len(module.tasks) != 1:
        raise NotImplementedError("run_crown_ibp_mlp currently supports single-task BFTaskModule only")
    if not task.ops:
        raise ValueError("run_crown_ibp_mlp expects a non-empty task")
    for i in range(len(task.ops) - 1):
        cur = task.ops[i]
        nxt = task.ops[i + 1]
        if len(cur.outputs) != 1:
            raise NotImplementedError(f"run_crown_ibp_mlp expects single-output ops, got outputs={cur.outputs}")
        if not nxt.inputs:
            raise NotImplementedError("run_crown_ibp_mlp expects each op to have at least one input")
        if cur.outputs[0] != nxt.inputs[0]:
            raise NotImplementedError(
                "run_crown_ibp_mlp currently supports chain-structured graphs only "
                f"(expected next input {nxt.inputs[0]} to equal prev output {cur.outputs[0]})"
            )

    input_spec = _normalize_input_spec(input_spec)
    if output_value is None:
        if len(task.output_values) != 1:
            raise ValueError(f"task has {len(task.output_values)} outputs; specify output_value explicitly")
        output_value = task.output_values[0]

    interval_env, relu_pre = _forward_ibp_trace_mlp(module, input_spec, relu_split_state=relu_split_state)
    return _run_crown_backward_from_trace(
        module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec_C,
        output_value=output_value,
        relu_alpha=relu_alpha,
        relu_pre_add_coeff_u=relu_pre_add_coeff_u,
        relu_pre_add_coeff_l=relu_pre_add_coeff_l,
        caller="run_crown_ibp_mlp",
    )


def run_crown_ibp_mlp_from_forward_trace(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    interval_env: Dict[str, IntervalState],
    relu_pre: Dict[str, IntervalState],
    linear_spec_C: Optional[torch.Tensor] = None,
    output_value: Optional[str] = None,
    relu_alpha: Optional[Dict[str, torch.Tensor]] = None,
    relu_pre_add_coeff_u: Optional[Dict[str, torch.Tensor]] = None,
    relu_pre_add_coeff_l: Optional[Dict[str, torch.Tensor]] = None,
) -> IntervalState:
    """
    Backward-only CROWN-IBP given a precomputed forward trace (interval_env + relu_pre).

    The forward trace should be computed by `_forward_ibp_trace_mlp(module, input_spec, relu_split_state=...)`.
    This enables higher-level drivers (alpha/alpha-beta/BaB) to reuse the forward IBP results across multiple
    backward/optimization iterations and branch picking without re-running forward.
    """
    module.validate()
    task = module.get_entry_task()
    if task.kind != TaskKind.INTERVAL_IBP:
        raise NotImplementedError(f"run_crown_ibp_mlp only supports INTERVAL_IBP, got {task.kind}")
    if module.task_graph is not None or len(module.tasks) != 1:
        raise NotImplementedError("run_crown_ibp_mlp_from_forward_trace currently supports single-task BFTaskModule only")

    return _run_crown_backward_from_trace(
        module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec_C,
        output_value=output_value,
        relu_alpha=relu_alpha,
        relu_pre_add_coeff_u=relu_pre_add_coeff_u,
        relu_pre_add_coeff_l=relu_pre_add_coeff_l,
        caller="run_crown_ibp_mlp_from_forward_trace",
    )


def get_crown_ibp_mlp_stats(module: BFTaskModule) -> CrownIbpStats:
    try:
        module.validate()
        task = module.get_entry_task()
        if task.kind != TaskKind.INTERVAL_IBP:
            return CrownIbpStats(supported=False, reason=f"TaskKind={task.kind}", ops_seen=tuple())
        if module.task_graph is not None or len(module.tasks) != 1:
            return CrownIbpStats(supported=False, reason="multi-task module not supported", ops_seen=tuple())
        ops = tuple(op.op_type for op in task.ops)
        if not task.ops:
            return CrownIbpStats(supported=False, reason="empty task", ops_seen=ops)
        bad = [t for t in ops if t not in {"linear", "relu", "conv2d", "flatten"}]
        if bad:
            return CrownIbpStats(supported=False, reason=f"unsupported ops: {bad}", ops_seen=ops)
        for i, op in enumerate(task.ops):
            if op.op_type == "flatten":
                start_dim = int(op.attrs.get("start_dim", 1))
                end_dim = int(op.attrs.get("end_dim", -1))
                if start_dim != 1 or end_dim != -1:
                    return CrownIbpStats(
                        supported=False,
                        reason=f"unsupported flatten attrs at op {i}: {op.attrs}",
                        ops_seen=ops,
                    )
        for i in range(len(task.ops) - 1):
            cur = task.ops[i]
            nxt = task.ops[i + 1]
            if len(cur.outputs) != 1:
                return CrownIbpStats(
                    supported=False,
                    reason=f"non-chain: op {i} outputs={cur.outputs}",
                    ops_seen=ops,
                )
            if not nxt.inputs:
                return CrownIbpStats(
                    supported=False,
                    reason=f"non-chain: op {i+1} has no inputs",
                    ops_seen=ops,
                )
            if cur.outputs[0] != nxt.inputs[0]:
                return CrownIbpStats(
                    supported=False,
                    reason=f"non-chain: op {i} -> {i+1} value mismatch ({cur.outputs[0]} != {nxt.inputs[0]})",
                    ops_seen=ops,
                )
        return CrownIbpStats(supported=True, ops_seen=ops)
    except Exception as e:  # pragma: no cover
        return CrownIbpStats(supported=False, reason=str(e), ops_seen=tuple())
