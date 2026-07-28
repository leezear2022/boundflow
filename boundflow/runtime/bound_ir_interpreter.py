"""Independent dense reference interpreter for Bound IR v1 plain CROWN."""

# CROWN literature and the existing runtime use A_u/A_l for affine coefficients.
# pylint: disable=too-many-locals,invalid-name,duplicate-code,not-callable

from __future__ import annotations

from typing import Any, Mapping, Optional, cast

import torch
import torch.nn.functional as torch_functional

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    plain_crown_primal_graph_hash,
    tensor_content_hash,
)
from ..ir.bound import (
    AddBackwardAttrs,
    BFBoundModule,
    BoundAffineStateRef,
    BoundMethodKind,
    BoundOp,
    BoundOpKind,
    ConcatBackwardAttrs,
    ConcretizeAttrs,
    Conv2dBackwardAttrs,
    LinearBackwardAttrs,
    ReluRelaxationAttrs,
    ReshapeAttrs,
    SpecBindAttrs,
)
from ..ir.task import BFTaskModule
from .task_executor import InputSpec


def execute_plain_crown_bound_ir(  # pylint: disable=too-many-branches,too-many-statements
    bound_module: BFBoundModule,
    *,
    task_module: BFTaskModule,
    input_spec: InputSpec,
    relu_pre: Mapping[str, IntervalState],
    linear_spec_C: Optional[torch.Tensor] = None,
) -> IntervalState:
    """Execute only the explicit BoundOp sequence, using query payload as bindings."""

    bound_module.validate()
    if bound_module.domain.method != BoundMethodKind.CROWN:
        raise NotImplementedError("dense Bound IR interpreter supports CROWN only")
    task_module.validate()
    if bound_module.primal_graph_hash != plain_crown_primal_graph_hash(task_module):
        raise ValueError("runtime task/parameter fingerprint does not match Bound IR")
    objective_spec = bound_module.spec.objectives[0]
    if objective_spec.payload_hash is None:
        if linear_spec_C is not None:
            raise ValueError("identity Bound IR cannot bind a linear objective")
    elif linear_spec_C is None or objective_spec.payload_hash != tensor_content_hash(
        linear_spec_C
    ):
        raise ValueError("runtime objective payload does not match Bound IR")
    params = _parameter_bindings(task_module)
    values = {value.value_id: value for value in bound_module.graph.values}
    env: dict[str, torch.Tensor] = {}

    for input_value_id in bound_module.graph.inputs:
        input_value = values[input_value_id]
        if input_value.role.value != "objective":
            raise NotImplementedError(
                "plain-CROWN interpreter only accepts objective graph inputs"
            )
        env[input_value_id] = _objective_tensor(
            input_value.tensor_type.shape,
            input_spec=input_spec,
            linear_spec_C=linear_spec_C,
        )

    for op in bound_module.graph.ops:
        if op.kind == BoundOpKind.SPEC_BIND:
            attrs = _attrs(op, SpecBindAttrs)
            del attrs
            objective = _get(env, op.inputs[0])
            if len(op.outputs) != 4:
                raise NotImplementedError(
                    "plain-CROWN interpreter requires affine-state spec binding"
                )
            bias = torch.zeros(
                int(objective.shape[0]),
                int(objective.shape[1]),
                device=objective.device,
                dtype=objective.dtype,
            )
            _store_state(
                env,
                op.outputs,
                (
                    objective.clone(),
                    bias,
                    objective.clone(),
                    bias.clone(),
                ),
            )
            continue

        if op.kind == BoundOpKind.COEFFICIENT_COMPOSE:
            states = tuple(
                _load_state(env, op.inputs[index : index + 4])
                for index in range(0, len(op.inputs), 4)
            )
            result = (
                sum((state[0] for state in states[1:]), states[0][0]),
                sum((state[1] for state in states[1:]), states[0][1]),
                sum((state[2] for state in states[1:]), states[0][2]),
                sum((state[3] for state in states[1:]), states[0][3]),
            )
            _store_state(env, op.outputs, result)
            continue

        if op.kind == BoundOpKind.LINEAR_BACKWARD:
            attrs = _attrs(op, LinearBackwardAttrs)
            A_u, b_u, A_l, b_l = _load_state(env, op.inputs)
            weight = _tensor_binding(params, attrs.weight_primal_value_id, like=A_u)
            if weight.dim() != 2:
                raise ValueError("linear weight must have rank 2")
            bias = (
                torch.zeros(int(weight.shape[0]), device=A_u.device, dtype=A_u.dtype)
                if attrs.bias_primal_value_id is None
                else _tensor_binding(params, attrs.bias_primal_value_id, like=A_u)
            )
            if bias.dim() == 0:
                bias = bias.expand(int(weight.shape[0]))
            if bias.dim() != 1 or int(bias.shape[0]) != int(weight.shape[0]):
                raise ValueError("linear bias must be scalar or rank 1 [output]")
            A_u_flat = A_u.reshape(int(A_u.shape[0]), int(A_u.shape[1]), -1)
            A_l_flat = A_l.reshape(int(A_l.shape[0]), int(A_l.shape[1]), -1)
            if int(A_u_flat.shape[2]) != int(weight.shape[0]):
                raise ValueError("linear coefficient/weight output shape mismatch")
            _store_state(
                env,
                op.outputs,
                (
                    A_u_flat.matmul(weight),
                    b_u + (A_u_flat * bias.view(1, 1, -1)).sum(dim=2),
                    A_l_flat.matmul(weight),
                    b_l + (A_l_flat * bias.view(1, 1, -1)).sum(dim=2),
                ),
            )
            continue

        if op.kind == BoundOpKind.CONV2D_BACKWARD:
            attrs = _attrs(op, Conv2dBackwardAttrs)
            A_u, b_u, A_l, b_l = _load_state(env, op.inputs)
            target_shape = _coefficient_primal_shape(values, op.outputs[0])
            source_shape = _coefficient_primal_shape(values, op.inputs[0])
            weight = _tensor_binding(params, attrs.weight_primal_value_id, like=A_u)
            conv_bias: Optional[torch.Tensor] = (
                None
                if attrs.bias_primal_value_id is None
                else _tensor_binding(params, attrs.bias_primal_value_id, like=A_u)
            )
            out_b_u = b_u
            out_b_l = b_l
            if conv_bias is not None:
                if conv_bias.dim() != 1 or int(conv_bias.shape[0]) != source_shape[0]:
                    raise ValueError("conv2d bias must be rank 1 [output_channels]")
                bias_map = conv_bias.view(-1, 1, 1).expand(source_shape)
                out_b_u = out_b_u + _contract(A_u, bias_map)
                out_b_l = out_b_l + _contract(A_l, bias_map)
            _store_state(
                env,
                op.outputs,
                (
                    _conv2d_backward(
                        A_u,
                        target_shape=target_shape,
                        source_shape=source_shape,
                        weight=weight,
                        attrs=attrs,
                    ),
                    out_b_u,
                    _conv2d_backward(
                        A_l,
                        target_shape=target_shape,
                        source_shape=source_shape,
                        weight=weight,
                        attrs=attrs,
                    ),
                    out_b_l,
                ),
            )
            continue

        if op.kind == BoundOpKind.RELU_RELAXATION:
            attrs = _attrs(op, ReluRelaxationAttrs)
            preactivation = attrs.preactivation_primal_value_id
            if preactivation is None or preactivation not in relu_pre:
                raise KeyError(f"missing ReLU pre-activation binding '{preactivation}'")
            pre = relu_pre[preactivation]
            A_u, b_u, A_l, b_l = _load_state(env, op.inputs)
            result = _relu_backward(A_u, b_u, A_l, b_l, pre=pre)
            output_shape = _coefficient_primal_shape(values, op.outputs[0])
            _store_state(
                env,
                op.outputs,
                (
                    result[0].reshape(
                        int(A_u.shape[0]), int(A_u.shape[1]), *output_shape
                    ),
                    result[1],
                    result[2].reshape(
                        int(A_l.shape[0]), int(A_l.shape[1]), *output_shape
                    ),
                    result[3],
                ),
            )
            continue

        if op.kind == BoundOpKind.RESHAPE:
            attrs = _attrs(op, ReshapeAttrs)
            A_u, b_u, A_l, b_l = _load_state(env, op.inputs)
            target_shape = tuple(int(dim) for dim in attrs.target_shape)
            _store_state(
                env,
                op.outputs,
                (
                    A_u.reshape(int(A_u.shape[0]), int(A_u.shape[1]), *target_shape),
                    b_u,
                    A_l.reshape(int(A_l.shape[0]), int(A_l.shape[1]), *target_shape),
                    b_l,
                ),
            )
            continue

        if op.kind == BoundOpKind.ADD_BACKWARD:
            attrs = _attrs(op, AddBackwardAttrs)
            A_u, b_u, A_l, b_l = _load_state(env, op.inputs)
            for constant_id in attrs.constant_input_primal_value_ids:
                constant = _tensor_binding(params, constant_id, like=A_u)
                b_u = b_u + _contract(A_u, constant)
                b_l = b_l + _contract(A_l, constant)
            output_states = _state_refs(op.outputs)
            for index, state in enumerate(output_states):
                child_b_u = b_u if index == 0 else torch.zeros_like(b_u)
                child_b_l = b_l if index == 0 else torch.zeros_like(b_l)
                _store_state(
                    env,
                    state.value_ids,
                    (A_u.clone(), child_b_u, A_l.clone(), child_b_l),
                )
            continue

        if op.kind == BoundOpKind.CONCAT_BACKWARD:
            attrs = _attrs(op, ConcatBackwardAttrs)
            A_u, b_u, A_l, b_l = _load_state(env, op.inputs)
            output_states = _state_refs(op.outputs)
            start = 0
            for index, (state, shape) in enumerate(
                zip(output_states, attrs.input_shapes)
            ):
                stop = start + int(shape[attrs.axis])
                slices = [slice(None)] * A_u.dim()
                slices[attrs.axis + 2] = slice(start, stop)
                child_b_u = b_u if index == 0 else torch.zeros_like(b_u)
                child_b_l = b_l if index == 0 else torch.zeros_like(b_l)
                _store_state(
                    env,
                    state.value_ids,
                    (
                        A_u[tuple(slices)].contiguous(),
                        child_b_u,
                        A_l[tuple(slices)].contiguous(),
                        child_b_l,
                    ),
                )
                start = stop
            if start != int(A_u.shape[attrs.axis + 2]):
                raise ValueError("concat backward slices do not cover source axis")
            continue

        if op.kind == BoundOpKind.CONCRETIZE:
            attrs = _attrs(op, ConcretizeAttrs)
            if attrs.perturbation_id != input_spec.perturbation.perturbation_id:
                raise ValueError("runtime perturbation does not match Bound IR")
            A_u, b_u, A_l, b_l = _load_state(env, op.inputs)
            _unused_lower, upper = input_spec.perturbation.concretize_affine(
                center=input_spec.center, A=A_u, b=b_u
            )
            lower, _unused_upper = input_spec.perturbation.concretize_affine(
                center=input_spec.center, A=A_l, b=b_l
            )
            env[op.outputs[0]] = lower
            env[op.outputs[1]] = upper
            continue

        raise NotImplementedError(
            f"dense Bound IR interpreter does not support {op.kind.value}"
        )

    lower_id, upper_id = bound_module.graph.outputs
    return IntervalState(lower=_get(env, lower_id), upper=_get(env, upper_id))


def _objective_tensor(
    expected_shape: tuple[Optional[int], ...],
    *,
    input_spec: InputSpec,
    linear_spec_C: Optional[torch.Tensor],
) -> torch.Tensor:
    if any(dimension is None for dimension in expected_shape):
        raise NotImplementedError("objective binding requires static dimensions")
    static_shape = cast(tuple[int, int, int], expected_shape)
    batch, specs, output_dim = static_shape
    device = input_spec.center.device
    dtype = input_spec.center.dtype
    if linear_spec_C is None:
        if specs != output_dim:
            raise ValueError("identity objective requires specs == output dimension")
        return (
            torch.eye(output_dim, device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch, output_dim, output_dim)
            .clone()
        )
    objective = linear_spec_C.to(device=device, dtype=dtype)
    if objective.dim() == 2:
        objective = objective.unsqueeze(0).expand(batch, specs, output_dim).clone()
    if tuple(objective.shape) != (batch, specs, output_dim):
        raise ValueError("runtime objective shape does not match Bound IR")
    return objective


def _relu_backward(
    A_u: torch.Tensor,
    b_u: torch.Tensor,
    A_l: torch.Tensor,
    b_l: torch.Tensor,
    *,
    pre: IntervalState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lower = pre.lower.reshape(int(pre.lower.shape[0]), -1)
    upper = pre.upper.reshape(int(pre.upper.shape[0]), -1)
    if lower.shape != upper.shape:
        raise ValueError("ReLU pre-activation lower/upper shapes differ")
    A_u_flat = A_u.reshape(int(A_u.shape[0]), int(A_u.shape[1]), -1)
    A_l_flat = A_l.reshape(int(A_l.shape[0]), int(A_l.shape[1]), -1)
    if int(A_u_flat.shape[2]) != int(lower.shape[1]):
        raise ValueError("ReLU coefficient/pre-activation shape mismatch")

    positive = lower >= 0
    negative = upper <= 0
    ambiguous = ~(positive | negative)
    alpha_u = torch.zeros_like(lower)
    beta_u = torch.zeros_like(lower)
    alpha_l = torch.zeros_like(lower)
    alpha_u[positive] = 1
    alpha_l[positive] = 1
    if ambiguous.any():
        denominator = (upper[ambiguous] - lower[ambiguous]).clamp_min(
            torch.finfo(lower.dtype).eps
        )
        alpha_u[ambiguous] = upper[ambiguous] / denominator
        beta_u[ambiguous] = -lower[ambiguous] * alpha_u[ambiguous]

    upper_alpha = torch.where(A_u_flat >= 0, alpha_u.unsqueeze(1), alpha_l.unsqueeze(1))
    upper_beta = torch.where(
        A_u_flat >= 0, beta_u.unsqueeze(1), torch.zeros_like(beta_u).unsqueeze(1)
    )
    lower_alpha = torch.where(A_l_flat >= 0, alpha_l.unsqueeze(1), alpha_u.unsqueeze(1))
    lower_beta = torch.where(
        A_l_flat >= 0, torch.zeros_like(beta_u).unsqueeze(1), beta_u.unsqueeze(1)
    )
    return (
        A_u_flat * upper_alpha,
        b_u + (A_u_flat * upper_beta).sum(dim=2),
        A_l_flat * lower_alpha,
        b_l + (A_l_flat * lower_beta).sum(dim=2),
    )


def _conv2d_backward(
    coefficient: torch.Tensor,
    *,
    target_shape: tuple[int, ...],
    source_shape: tuple[int, ...],
    weight: torch.Tensor,
    attrs: Conv2dBackwardAttrs,
) -> torch.Tensor:
    if len(target_shape) != 3 or len(source_shape) != 3:
        raise ValueError("conv2d backward requires CHW coefficient shapes")
    coefficient = coefficient.reshape(
        int(coefficient.shape[0]), int(coefficient.shape[1]), *source_shape
    )
    output_padding = tuple(
        int(target_shape[index + 1])
        - (
            (int(source_shape[index + 1]) - 1) * attrs.stride[index]
            - 2 * attrs.padding[index]
            + attrs.dilation[index] * (int(weight.shape[index + 2]) - 1)
            + 1
        )
        for index in range(2)
    )
    if any(
        value < 0 or value >= attrs.stride[index]
        for index, value in enumerate(output_padding)
    ):
        raise ValueError(f"invalid conv2d backward output_padding {output_padding}")
    flat = coefficient.reshape(
        int(coefficient.shape[0]) * int(coefficient.shape[1]), *source_shape
    )
    result = torch_functional.conv_transpose2d(
        flat,
        weight,
        bias=None,
        stride=attrs.stride,
        padding=attrs.padding,
        output_padding=output_padding,
        groups=attrs.groups,
        dilation=attrs.dilation,
    )
    return result.reshape(
        int(coefficient.shape[0]), int(coefficient.shape[1]), *target_shape
    )


def _contract(coefficient: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    flat_coefficient = coefficient.reshape(
        int(coefficient.shape[0]), int(coefficient.shape[1]), -1
    )
    if value.dim() == coefficient.dim() - 2:
        flat_value = value.reshape(1, 1, -1)
    elif value.dim() == coefficient.dim() - 1:
        flat_value = value.reshape(int(value.shape[0]), 1, -1)
    else:
        raise ValueError("affine contraction value rank is incompatible")
    if int(flat_value.shape[2]) != int(flat_coefficient.shape[2]):
        raise ValueError("affine contraction value shape is incompatible")
    return (flat_coefficient * flat_value).sum(dim=2)


def _coefficient_primal_shape(
    values: Mapping[str, Any], value_id: str
) -> tuple[int, ...]:
    shape = values[value_id].tensor_type.shape[2:]
    if any(dimension is None for dimension in shape):
        raise NotImplementedError(
            "dense interpreter requires static coefficient shapes"
        )
    return tuple(int(dimension) for dimension in shape)


def _state_refs(value_ids: tuple[str, ...]) -> tuple[BoundAffineStateRef, ...]:
    return tuple(
        BoundAffineStateRef(
            upper_coefficient=value_ids[index],
            upper_bias=value_ids[index + 1],
            lower_coefficient=value_ids[index + 2],
            lower_bias=value_ids[index + 3],
        )
        for index in range(0, len(value_ids), 4)
    )


def _load_state(
    env: Mapping[str, torch.Tensor], value_ids: tuple[str, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(value_ids) != 4:
        raise ValueError("affine state must contain four values")
    return (
        _get(env, value_ids[0]),
        _get(env, value_ids[1]),
        _get(env, value_ids[2]),
        _get(env, value_ids[3]),
    )


def _store_state(
    env: dict[str, torch.Tensor],
    value_ids: tuple[str, ...],
    state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    if len(value_ids) != 4:
        raise ValueError("affine state output must contain four values")
    for value_id, tensor in zip(value_ids, state):
        env[value_id] = tensor


def _get(env: Mapping[str, torch.Tensor], value_id: str) -> torch.Tensor:
    if value_id not in env:
        raise KeyError(f"Bound IR value '{value_id}' is not bound")
    return env[value_id]


def _parameter_bindings(module: BFTaskModule) -> dict[str, Any]:
    raw = module.bindings.get("params", {})
    if not isinstance(raw, dict):
        raise TypeError("BFTaskModule bindings['params'] must be a dictionary")
    return dict(raw)


def _tensor_binding(
    params: Mapping[str, Any], value_id: str, *, like: torch.Tensor
) -> torch.Tensor:
    if value_id not in params:
        raise KeyError(f"missing runtime parameter binding '{value_id}'")
    value = params[value_id]
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.to(device=like.device, dtype=like.dtype)


def _attrs(op: BoundOp, expected_type: type[Any]) -> Any:
    if not isinstance(op.attrs, expected_type):
        raise TypeError(
            f"BoundOp '{op.op_id}' expects {expected_type.__name__} attributes"
        )
    return op.attrs
