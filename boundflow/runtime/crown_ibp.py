from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    cast,
    Tuple,
)

import torch

from ..domains.interval import IntervalDomain, IntervalState
from ..ir.task import BFTaskModule, TaskKind
from ..planner.materialization import (
    BoundMethod,
    MaterializationAction,
    MaterializationContext,
    MaterializationPlan,
    MaterializationPlannerOptions,
    OptimizationStage,
    estimate_operator_tree_summary,
    plan_materialization,
)
from ..planner.materialization_placement import MaterializationPlacementPlan
from .dag_utils import (
    normalize_concat_axis,
    validate_concat_tensor_shapes,
    validate_concat_value_shapes,
)
from .fused_crown import (
    FusedCrownExecutionContext,
    FusedCrownExecutionStep,
    FusedCrownExecutor,
    FusedReluAffineDescriptor,
    FusedReluAffineRequest,
    validate_fused_crown_execution_steps,
)
from .fsg4_b4b_production_region_capture import B4BRegionLiveObserverProtocol
from .linear_operator import (
    DenseLinearOperator,
    LinearOperator,
    SignSplitLinearOperator,
)
from .materialization import materialize_linear_operator
from .perturbation import BoxPerturbation, InputPerturbationState
from .relu_shape_utils import broadcast_relu_split_like_pre
from .scheduler import (
    PlacementRetryStats,
    execute_bounded_placement_candidates_with_retry,
    execute_placement_candidates_with_retry,
)
from .task_executor import InputSpec, InputSpecLike, _normalize_input_spec


@dataclass
class CrownIbpStats:
    supported: bool
    reason: str = ""
    ops_seen: Tuple[str, ...] = ()


def _apply_relu_split(
    pre: IntervalState, split: torch.Tensor, *, relu_input_name: str
) -> IntervalState:
    split_b = broadcast_relu_split_like_pre(
        split, pre=pre, x_name=relu_input_name, device=pre.lower.device
    )
    lower_flat = pre.lower.reshape(int(pre.lower.shape[0]), -1)
    upper_flat = pre.upper.reshape(int(pre.upper.shape[0]), -1)
    active = split_b > 0
    inactive = split_b < 0
    if not active.any() and not inactive.any():
        return pre
    lower = lower_flat
    upper = upper_flat
    if active.any():
        lower = torch.where(
            active, torch.maximum(lower, torch.zeros_like(lower)), lower
        )
    if inactive.any():
        upper = torch.where(
            inactive, torch.minimum(upper, torch.zeros_like(upper)), upper
        )
    if (lower > upper).any():
        raise ValueError(
            f"infeasible relu split for {relu_input_name}: lower>upper after applying split"
        )
    return IntervalState(
        lower=lower.reshape_as(pre.lower), upper=upper.reshape_as(pre.upper)
    )


def _forward_ibp_trace_mlp(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    relu_split_state: Optional[Dict[str, torch.Tensor]] = None,
    relu_pre_constraints: Optional[Mapping[str, IntervalState]] = None,
) -> Tuple[Dict[str, IntervalState], Dict[str, IntervalState]]:
    task = module.get_entry_task()
    raw_params = module.bindings.get("params", {})
    params: Dict[str, Any] = dict(raw_params) if isinstance(raw_params, dict) else {}

    domain = IntervalDomain()
    env: Dict[str, Any] = {
        input_spec.value_name: InputPerturbationState(
            center=input_spec.center, perturbation=input_spec.perturbation
        )
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
        raise TypeError(
            f"expected IntervalState/InputPerturbationState, got {type(state)}"
        )

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
                lb, ub = x_state.perturbation.concretize_matmul(
                    center=x_state.center, weight=w, bias=b
                )
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
            if relu_pre_constraints is not None:
                constraint = relu_pre_constraints.get(x_name)
                if (
                    not isinstance(constraint, IntervalState)
                    or constraint.lower.shape != x.lower.shape
                    or constraint.upper.shape != x.upper.shape
                    or constraint.lower.dtype != x.lower.dtype
                    or constraint.upper.dtype != x.upper.dtype
                    or constraint.lower.device != x.lower.device
                    or constraint.upper.device != x.upper.device
                    or not bool(torch.isfinite(constraint.lower).all())
                    or not bool(torch.isfinite(constraint.upper).all())
                    or not bool((constraint.lower <= constraint.upper).all())
                ):
                    raise ValueError(
                        f"ReLU pre-activation constraint schema differs for {x_name}"
                    )
                lower = torch.maximum(x.lower, constraint.lower)
                upper = torch.minimum(x.upper, constraint.upper)
                if bool((lower > upper).any().item()):
                    raise ValueError(
                        f"infeasible ReLU pre-activation intersection for {x_name}"
                    )
                x = IntervalState(lower=lower, upper=upper)
            if relu_split_state is not None and x_name in relu_split_state:
                x = _apply_relu_split(
                    x, relu_split_state[x_name], relu_input_name=x_name
                )
            relu_pre[x_name] = x
            y = domain.relu_transformer(x)
            env[op.outputs[0]] = y
            interval_env[op.outputs[0]] = y
            continue

        if op.op_type == "add":
            a = _ensure_interval(_get_state(op.inputs[0]))
            b = _ensure_interval(_get_state(op.inputs[1]))
            if tuple(a.lower.shape) != tuple(b.lower.shape) or tuple(
                a.upper.shape
            ) != tuple(b.upper.shape):
                raise NotImplementedError(
                    "_forward_ibp_trace_mlp only supports add with exact same-shape inputs; "
                    f"got {tuple(a.lower.shape)} and {tuple(b.lower.shape)}"
                )
            y = IntervalState(lower=a.lower + b.lower, upper=a.upper + b.upper)
            env[op.outputs[0]] = y
            interval_env[op.outputs[0]] = y
            continue

        if op.op_type == "concat":
            if len(op.inputs) < 2:
                raise ValueError(
                    f"concat expects at least 2 inputs, got {len(op.inputs)}"
                )
            parts = [_ensure_interval(_get_state(name)) for name in op.inputs]
            axis = normalize_concat_axis(
                op.attrs.get("axis", 1),
                rank_with_batch=int(parts[0].lower.dim()),
                caller="_forward_ibp_trace_mlp",
            )
            _ = validate_concat_tensor_shapes(
                [tuple(int(dim) for dim in part.lower.shape) for part in parts],
                axis=axis,
                caller="_forward_ibp_trace_mlp",
            )
            y = IntervalState(
                lower=torch.cat([part.lower for part in parts], dim=axis),
                upper=torch.cat([part.upper for part in parts], dim=axis),
            )
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
                perturbation = x_state.perturbation
                if isinstance(perturbation, BoxPerturbation):
                    perturbation = BoxPerturbation(
                        lower=torch.flatten(
                            perturbation.lower,
                            start_dim=start_dim,
                            end_dim=end_dim,
                        ),
                        upper=torch.flatten(
                            perturbation.upper,
                            start_dim=start_dim,
                            end_dim=end_dim,
                        ),
                    )
                shaped_state = InputPerturbationState(
                    center=torch.flatten(
                        x_state.center, start_dim=start_dim, end_dim=end_dim
                    ),
                    perturbation=perturbation,
                )
                env[op.outputs[0]] = shaped_state
                if isinstance(perturbation, BoxPerturbation):
                    interval_env[op.outputs[0]] = IntervalState(
                        lower=perturbation.lower,
                        upper=perturbation.upper,
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

        if op.op_type == "reshape":
            shape = tuple(int(dim) for dim in op.attrs.get("shape", ()))
            if not shape:
                raise ValueError("reshape requires non-empty attrs['shape']")
            x_state = _get_state(op.inputs[0])
            if isinstance(x_state, InputPerturbationState):
                perturbation = x_state.perturbation
                if isinstance(perturbation, BoxPerturbation):
                    perturbation = BoxPerturbation(
                        lower=perturbation.lower.reshape(shape),
                        upper=perturbation.upper.reshape(shape),
                    )
                shaped_state = InputPerturbationState(
                    center=x_state.center.reshape(shape), perturbation=perturbation
                )
                env[op.outputs[0]] = shaped_state
                if isinstance(perturbation, BoxPerturbation):
                    interval_env[op.outputs[0]] = IntervalState(
                        lower=perturbation.lower,
                        upper=perturbation.upper,
                    )
            else:
                x = _ensure_interval(x_state)
                y = IntervalState(
                    lower=x.lower.reshape(shape), upper=x.upper.reshape(shape)
                )
                env[op.outputs[0]] = y
                interval_env[op.outputs[0]] = y
            continue

        raise NotImplementedError(
            f"_forward_ibp_trace_mlp unsupported op_type: {op.op_type}"
        )

    if relu_pre_constraints is not None and tuple(relu_pre_constraints) != tuple(
        relu_pre
    ):
        raise ValueError("ReLU pre-activation constraint identities differ")
    return interval_env, relu_pre


@dataclass
class AffineBackwardState:
    A_u: LinearOperator
    A_l: LinearOperator
    b_u: torch.Tensor
    b_l: torch.Tensor


@dataclass(frozen=True)
class DenseReluBackwardResult:
    """Dense reference outputs for one ReLU backward transformer step."""

    A_u: torch.Tensor
    A_l: torch.Tensor
    b_u: torch.Tensor
    b_l: torch.Tensor


@dataclass(frozen=True)
class ReluBackwardRelaxation:
    """Broadcast ReLU slopes/intercepts shared by dense and structured paths."""

    pre_flat: IntervalState
    alpha_u: torch.Tensor
    beta_u: torch.Tensor
    alpha_l: torch.Tensor
    beta_l: torch.Tensor


ReluBackwardMode = Literal["dense", "structured"]
_DEFAULT_RELU_BACKWARD_MODE: ReluBackwardMode = (
    "structured"
    if os.environ.get("BOUNDFLOW_RELU_BACKWARD_MODE") == "structured"
    else "dense"
)
_RELU_BACKWARD_MODE: ContextVar[ReluBackwardMode] = ContextVar(
    "boundflow_relu_backward_mode",
    default=_DEFAULT_RELU_BACKWARD_MODE,
)
_RELU_BACKWARD_PLACEMENTS: ContextVar[Optional[Dict[str, ReluBackwardMode]]] = (
    ContextVar(
        "boundflow_relu_backward_placements",
        default=None,
    )
)


@contextmanager
def _relu_backward_mode(mode: ReluBackwardMode) -> Iterator[None]:
    """Temporarily select dense-reference or structured ReLU backward."""

    if mode not in {"dense", "structured"}:
        raise ValueError(f"unsupported ReLU backward mode: {mode}")
    token = _RELU_BACKWARD_MODE.set(mode)
    try:
        yield
    finally:
        _RELU_BACKWARD_MODE.reset(token)


class MaterializationReplanRequired(RuntimeError):
    """The host runtime must shrink and resubmit the current query batch."""

    def __init__(self, plan: MaterializationPlan) -> None:
        self.plan = plan
        super().__init__(
            "materialization plan requires a smaller domain batch: "
            f"recommended_domain_batch_size={plan.recommended_domain_batch_size} "
            f"reason={plan.reason}"
        )


class MaterializationPlacementReplanRequired(RuntimeError):
    """The mixed placement plan requires host-side batch reduction."""

    def __init__(self, plan: MaterializationPlacementPlan) -> None:
        self.plan = plan
        super().__init__(
            "materialization placement requires a smaller domain batch: "
            f"recommended_domain_batch_size={plan.recommended_domain_batch_size} "
            f"reason={plan.reason}"
        )


@contextmanager
def _apply_materialization_plan(plan: Optional[MaterializationPlan]) -> Iterator[None]:
    """Apply an explicit query-level plan without changing the legacy default path."""

    if plan is None:
        yield
        return
    if plan.action == MaterializationAction.REDUCE_BATCH:
        raise MaterializationReplanRequired(plan)
    mode: ReluBackwardMode = (
        "structured" if plan.action == MaterializationAction.STRUCTURED else "dense"
    )
    with _relu_backward_mode(mode):
        yield


@contextmanager
def _apply_materialization_placement_plan(
    plan: Optional[MaterializationPlacementPlan],
) -> Iterator[None]:
    """Apply a mixed plan keyed by ReLU pre-activation/source value."""

    if plan is None:
        yield
        return
    if plan.requires_replan:
        raise MaterializationPlacementReplanRequired(plan)
    placements: Dict[str, ReluBackwardMode] = {}
    for placement in plan.placements:
        if placement.action == MaterializationAction.REDUCE_BATCH:
            raise ValueError("placement entries cannot use reduce_batch")
        placements[placement.barrier_id] = (
            "structured"
            if placement.action == MaterializationAction.STRUCTURED
            else "dense"
        )
    token = _RELU_BACKWARD_PLACEMENTS.set(placements)
    try:
        yield
    finally:
        _RELU_BACKWARD_PLACEMENTS.reset(token)


@contextmanager
def _apply_execution_materialization(
    plan: Optional[MaterializationPlan],
    placement_plan: Optional[MaterializationPlacementPlan],
) -> Iterator[None]:
    """Apply exactly one query-level or mixed materialization plan."""

    if plan is not None and placement_plan is not None:
        raise ValueError(
            "materialization_plan and materialization_placement_plan are mutually exclusive"
        )
    with _apply_materialization_plan(plan):
        with _apply_materialization_placement_plan(placement_plan):
            yield


def validate_optimized_bound_materialization_plan(
    plan: Optional[MaterializationPlan],
    *,
    placement_plan: Optional[MaterializationPlacementPlan] = None,
    caller: str,
) -> None:
    """Reject structured autograd until its backend capability is validated."""

    if plan is not None and plan.action == MaterializationAction.STRUCTURED:
        raise ValueError(
            f"{caller} cannot execute structured optimized bounds: "
            "supports_structured_autograd=false and supports_optimized_bound_structured=false"
        )
    if placement_plan is not None and any(
        placement.action == MaterializationAction.STRUCTURED
        for placement in placement_plan.placements
    ):
        raise ValueError(
            f"{caller} cannot execute structured optimized-bound placements: "
            "structured autograd capability is not validated"
        )


def _tensor_dict_bytes(values: Optional[Dict[str, torch.Tensor]]) -> int:
    if values is None:
        return 0
    return sum(
        int(value.numel()) * int(value.element_size()) for value in values.values()
    )


def build_crown_materialization_context(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    interval_env: Dict[str, IntervalState],
    relu_pre: Dict[str, IntervalState],
    linear_spec_C: Optional[torch.Tensor],
    output_value: str,
    options: MaterializationPlannerOptions,
    bound_method: BoundMethod = BoundMethod.CROWN,
    requires_grad: bool = False,
    optimization_stage: OptimizationStage = OptimizationStage.INFERENCE,
    alpha_enabled: bool = False,
    beta_enabled: bool = False,
    split_state_present: bool = False,
    relu_alpha: Optional[Dict[str, torch.Tensor]] = None,
    relu_pre_add_coeff_u: Optional[Dict[str, torch.Tensor]] = None,
    relu_pre_add_coeff_l: Optional[Dict[str, torch.Tensor]] = None,
) -> MaterializationContext:
    """Derive an explainable PR-11 context from one concrete CROWN query."""

    options.validate()
    task = module.get_entry_task()
    domain_batch = int(input_spec.center.shape[0])
    element_size = int(input_spec.center.element_size())
    if linear_spec_C is None:
        spec_size = int(interval_env[output_value].lower[0].numel())
    else:
        if linear_spec_C.dim() < 2:
            raise ValueError(
                "linear_spec_C must have at least two dimensions, "
                f"got shape={tuple(linear_spec_C.shape)}"
            )
        spec_size = int(linear_spec_C.shape[-2])
    relu_numels = tuple(int(pre.lower[0].numel()) for pre in relu_pre.values())
    output_numel = int(interval_env[output_value].lower[0].numel())
    beta_state_bytes = _tensor_dict_bytes(relu_pre_add_coeff_u) + _tensor_dict_bytes(
        relu_pre_add_coeff_l
    )
    available = (
        int(options.available_memory_bytes)
        if options.available_memory_bytes is not None
        else int(options.memory_budget_bytes)
    )
    return MaterializationContext(
        bound_method=bound_method,
        requires_grad=requires_grad,
        optimization_stage=optimization_stage,
        alpha_enabled=alpha_enabled,
        beta_enabled=beta_enabled,
        split_state_present=split_state_present,
        batch_size=domain_batch,
        spec_size=spec_size,
        domain_batch_size=domain_batch,
        operator_summary=estimate_operator_tree_summary(
            domain_batch_size=domain_batch,
            spec_size=spec_size,
            output_numel=output_numel,
            relu_numels=relu_numels,
            element_size=element_size,
            operator_nodes=len(task.ops),
            alpha_state_bytes=_tensor_dict_bytes(relu_alpha),
            beta_state_bytes=beta_state_bytes,
        ),
        memory_budget_bytes=int(options.memory_budget_bytes),
        available_memory_bytes=available,
        safety_margin=float(options.safety_margin),
        expected_query_reuse=int(options.expected_query_reuse),
        target=options.target,
    )


def plan_crown_materialization(
    module: BFTaskModule,
    input_spec: InputSpecLike,
    *,
    options: MaterializationPlannerOptions,
    linear_spec_C: Optional[torch.Tensor] = None,
    output_value: Optional[str] = None,
) -> tuple[MaterializationContext, MaterializationPlan]:
    """Build and plan one real plain-CROWN query before execution."""

    module.validate()
    task = module.get_entry_task()
    spec = _normalize_input_spec(input_spec)
    resolved_output = _resolve_output_value(
        task, output_value, caller="plan_crown_materialization"
    )
    interval_env, relu_pre = _forward_ibp_trace_mlp(module, spec)
    context = build_crown_materialization_context(
        module,
        spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec_C,
        output_value=resolved_output,
        options=options,
    )
    return context, plan_materialization(context, policy=options.policy)


def _resolve_output_value(
    task: Any, output_value: Optional[str], *, caller: str
) -> str:
    if output_value is None:
        if len(task.output_values) != 1:
            raise ValueError(
                f"task has {len(task.output_values)} outputs; specify output_value explicitly"
            )
        output_value = task.output_values[0]
    produced = {value_name for op in task.ops for value_name in tuple(op.outputs)}
    if output_value not in produced:
        raise ValueError(
            f"{caller} output_value is not produced by the task: {output_value}"
        )
    return output_value


def _get_output_meta(
    *,
    interval_env: Dict[str, IntervalState],
    output_value: str,
    caller: str,
) -> Tuple[
    IntervalState,
    int,
    int,
    Tuple[int, ...],
    torch.device,
    torch.dtype,
]:
    if output_value not in interval_env:
        raise KeyError(f"output_value not produced in forward trace: {output_value}")
    y_out = interval_env[output_value]
    if y_out.lower.dim() < 2 or y_out.lower.shape != y_out.upper.shape:
        raise ValueError(
            f"{caller} expects matching output bounds with rank>=2, "
            f"got lower={tuple(y_out.lower.shape)} upper={tuple(y_out.upper.shape)}"
        )
    if not torch.is_floating_point(y_out.lower) or not torch.is_floating_point(
        y_out.upper
    ):
        raise TypeError(
            f"{caller} expects floating bounds, got dtype={y_out.lower.dtype}"
        )
    batch = int(y_out.lower.shape[0])
    output_shape = tuple(int(dimension) for dimension in y_out.lower.shape[1:])
    out_dim = int(y_out.lower[0].numel())
    device = y_out.lower.device
    dtype = y_out.lower.dtype
    return y_out, batch, out_dim, output_shape, device, dtype


def _init_backward_state(
    *,
    batch: int,
    out_dim: int,
    output_shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    linear_spec_C: Optional[torch.Tensor],
) -> AffineBackwardState:
    if linear_spec_C is None:
        A = (
            torch.eye(out_dim, device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch, out_dim, out_dim)
            .clone()
        )
        b = torch.zeros(batch, out_dim, device=device, dtype=dtype)
        return AffineBackwardState(
            A_u=DenseLinearOperator(A, input_shape=output_shape),
            A_l=DenseLinearOperator(A.clone(), input_shape=output_shape),
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
            raise ValueError(
                f"linear_spec_C shape mismatch: C={tuple(C.shape)} out=({batch}, {out_dim})"
            )
        C = C.unsqueeze(0).expand(batch, int(C.shape[0]), out_dim).clone()
    if C.dim() != 3:
        raise ValueError(f"linear_spec_C expects rank-3 [B,S,O], got {tuple(C.shape)}")
    if int(C.shape[0]) != batch or int(C.shape[2]) != out_dim:
        raise ValueError(
            f"linear_spec_C shape mismatch: C={tuple(C.shape)} out=({batch}, {out_dim})"
        )
    b = torch.zeros(int(C.shape[0]), int(C.shape[1]), device=device, dtype=dtype)
    return AffineBackwardState(
        A_u=DenseLinearOperator(C, input_shape=output_shape),
        A_l=DenseLinearOperator(C.clone(), input_shape=output_shape),
        b_u=b,
        b_l=b.clone(),
    )


def _relu_relax(
    l: torch.Tensor, u: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.is_floating_point(l) or not torch.is_floating_point(u):
        raise TypeError(
            f"_relu_relax expects floating tensors, got l={l.dtype} u={u.dtype}"
        )
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
    w = (
        w_raw
        if torch.is_tensor(w_raw)
        else torch.as_tensor(w_raw, device=device, dtype=dtype)
    )
    w = w.to(device=device, dtype=dtype)
    if w.dim() != 2:
        raise NotImplementedError(
            f"{caller} currently supports non-batched linear weights only"
        )
    out_dim = int(w.shape[0])

    if bias_raw is None:
        bias = torch.zeros(out_dim, device=device, dtype=dtype)
        return w, bias

    bias = (
        bias_raw
        if torch.is_tensor(bias_raw)
        else torch.as_tensor(bias_raw, device=device, dtype=dtype)
    )
    bias = bias.to(device=device, dtype=dtype)
    if bias.dim() == 0:
        return w, bias.expand(out_dim)
    if bias.dim() == 1 and int(bias.shape[0]) == out_dim:
        return w, bias
    raise NotImplementedError(
        f"{caller} expects linear bias scalar or rank-1 [O], got {tuple(bias.shape)}"
    )


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
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Tuple[int, int],
    Tuple[int, int],
    Tuple[int, int],
    int,
]:
    w = (
        weight_raw
        if torch.is_tensor(weight_raw)
        else torch.as_tensor(weight_raw, device=device, dtype=dtype)
    )
    w = w.to(device=device, dtype=dtype)
    if w.dim() != 4:
        raise NotImplementedError(
            f"{caller} currently supports rank-4 conv2d weights only, got {tuple(w.shape)}"
        )

    bias: Optional[torch.Tensor]
    if bias_raw is None:
        bias = None
    else:
        bias_t = (
            bias_raw
            if torch.is_tensor(bias_raw)
            else torch.as_tensor(bias_raw, device=device, dtype=dtype)
        )
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


def _align_backward_state_input_shape(
    state: AffineBackwardState,
    *,
    input_shape: Tuple[int, ...],
) -> AffineBackwardState:
    A_u = (
        state.A_u
        if tuple(state.A_u.input_shape) == tuple(input_shape)
        else state.A_u.reshape_input(input_shape)
    )
    A_l = (
        state.A_l
        if tuple(state.A_l.input_shape) == tuple(input_shape)
        else state.A_l.reshape_input(input_shape)
    )
    return AffineBackwardState(A_u=A_u, A_l=A_l, b_u=state.b_u, b_l=state.b_l)


def _accumulate_backward_state(
    current: Optional[AffineBackwardState],
    update: AffineBackwardState,
    *,
    input_shape: Tuple[int, ...],
) -> AffineBackwardState:
    aligned_update = _align_backward_state_input_shape(update, input_shape=input_shape)
    if current is None:
        return aligned_update
    aligned_current = _align_backward_state_input_shape(
        current, input_shape=input_shape
    )
    return AffineBackwardState(
        A_u=aligned_current.A_u.add(aligned_update.A_u),
        A_l=aligned_current.A_l.add(aligned_update.A_l),
        b_u=aligned_current.b_u + aligned_update.b_u,
        b_l=aligned_current.b_l + aligned_update.b_l,
    )


def _dynamic_value_names(
    *,
    input_spec: InputSpec,
    interval_env: Dict[str, IntervalState],
) -> set[str]:
    names = set(interval_env.keys())
    names.add(input_spec.value_name)
    return names


def _split_bias_once(
    state: AffineBackwardState,
    *,
    num_children: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if num_children < 0:
        raise ValueError(f"num_children must be >= 0, got {num_children}")
    zero_u = torch.zeros_like(state.b_u)
    zero_l = torch.zeros_like(state.b_l)
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for idx in range(num_children):
        if idx == 0:
            out.append((state.b_u, state.b_l))
        else:
            out.append((zero_u, zero_l))
    return out


def _broadcast_relu_alpha(
    alpha_raw: Any,
    *,
    pre: IntervalState,
    x_name: str,
    device: torch.device,
    dtype: torch.dtype,
    caller: str,
) -> torch.Tensor:
    alpha = (
        alpha_raw
        if torch.is_tensor(alpha_raw)
        else torch.as_tensor(alpha_raw, device=device, dtype=dtype)
    )
    alpha = alpha.to(device=device, dtype=dtype)
    if not torch.is_floating_point(alpha):
        raise TypeError(
            f"relu_alpha[{x_name}] must be floating, got dtype={alpha.dtype}"
        )
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
    elif (
        alpha.dim() == len(logical_shape) + 1
        and int(alpha.shape[0]) == 1
        and tuple(alpha.shape[1:]) == logical_shape
    ):
        out = alpha.reshape(1, flat_dim).expand(target_shape)
    elif (
        alpha.dim() == 2
        and int(alpha.shape[0]) == 1
        and int(alpha.shape[1]) == flat_dim
    ):
        out = alpha.expand(target_shape)
    elif (
        alpha.dim() == len(logical_shape) + 1
        and int(alpha.shape[0]) == batch
        and tuple(alpha.shape[1:]) == logical_shape
    ):
        out = alpha.reshape(batch, flat_dim)
    elif (
        alpha.dim() == 2
        and int(alpha.shape[0]) == batch
        and int(alpha.shape[1]) == flat_dim
    ):
        out = alpha
    else:
        raise ValueError(
            f"relu_alpha[{x_name}] shape {tuple(alpha.shape)} cannot broadcast to logical shape "
            f"{logical_shape} (flat_dim={flat_dim}, batch={batch})"
        )
    return out.clamp(0.0, 1.0)


def _broadcast_relu_pre_add_coeff(
    add_raw: Any,
    *,
    batch: int,
    flat_dim: int,
    x_name: str,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    add = (
        add_raw
        if torch.is_tensor(add_raw)
        else torch.as_tensor(add_raw, device=device, dtype=dtype)
    )
    add = add.to(device=device, dtype=dtype)
    if add.dim() == 0:
        add = add.expand(flat_dim)
    if add.dim() == 1:
        if int(add.shape[0]) != flat_dim:
            raise ValueError(
                f"{label}[{x_name}] shape {tuple(add.shape)} does not match expected ({flat_dim},)"
            )
        return add.view(1, -1).expand(batch, -1)
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
            raise ValueError(
                f"{label}[{x_name}] shape {tuple(add.shape)} does not match batch {batch}"
            )
        return add_b
    total = int(add.numel())
    if total == flat_dim:
        return add.reshape(1, flat_dim).expand(batch, -1)
    if total == batch * flat_dim and int(add.shape[0]) == batch:
        return add.reshape(batch, flat_dim)
    raise ValueError(
        f"{label}[{x_name}] expects shape broadcastable to [B,{flat_dim}], got {tuple(add.shape)}"
    )


def _apply_relu_pre_add_coeff(
    A: torch.Tensor,
    add_raw: Any,
    *,
    x_name: str,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    add = _broadcast_relu_pre_add_coeff(
        add_raw,
        batch=int(A.shape[0]),
        flat_dim=int(A.shape[2]),
        x_name=x_name,
        label=label,
        device=device,
        dtype=dtype,
    )
    return A + add.unsqueeze(1)


def _backprop_linear_step(
    state: AffineBackwardState,
    *,
    weight: Any,
    bias: Any,
    device: torch.device,
    dtype: torch.dtype,
    caller: str,
) -> AffineBackwardState:
    w, bias_vec = _normalize_linear_inputs(
        weight, bias, device=device, dtype=dtype, caller=caller
    )
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
        raise NotImplementedError(
            f"{caller} currently supports NCHW conv2d inputs only, got {input_shape}"
        )
    if len(output_shape) != 3:
        raise NotImplementedError(
            f"{caller} expects conv2d output rank-3 without batch, got {output_shape}"
        )

    w, bias_vec, stride, padding, dilation, groups = _normalize_conv2d_inputs(
        weight,
        bias,
        attrs=attrs,
        device=device,
        dtype=dtype,
        caller=caller,
    )

    A_u_base = (
        state.A_u
        if tuple(state.A_u.input_shape) == tuple(output_shape)
        else state.A_u.reshape_input(output_shape)
    )
    A_l_base = (
        state.A_l
        if tuple(state.A_l.input_shape) == tuple(output_shape)
        else state.A_l.reshape_input(output_shape)
    )

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


def _relu_backward_relaxation(
    *,
    pre: IntervalState,
    x_name: str,
    relu_alpha: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    caller: str,
) -> ReluBackwardRelaxation:
    pre_flat = IntervalState(
        lower=pre.lower.reshape(int(pre.lower.shape[0]), -1),
        upper=pre.upper.reshape(int(pre.upper.shape[0]), -1),
    )
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
    return ReluBackwardRelaxation(
        pre_flat=pre_flat,
        alpha_u=alpha_u,
        beta_u=beta_u,
        alpha_l=alpha_l,
        beta_l=beta_l,
    )


def _backprop_relu_step_dense_reference(
    A_u: torch.Tensor,
    A_l: torch.Tensor,
    b_u: torch.Tensor,
    b_l: torch.Tensor,
    *,
    pre: IntervalState,
    x_name: str,
    relu_alpha: Optional[Dict[str, torch.Tensor]],
    relu_pre_add_coeff_u: Optional[Dict[str, torch.Tensor]],
    relu_pre_add_coeff_l: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    caller: str,
) -> DenseReluBackwardResult:
    """Apply the exact eager dense ReLU backward equations used as PR-10 oracle."""

    relaxation = _relu_backward_relaxation(
        pre=pre,
        x_name=x_name,
        relu_alpha=relu_alpha,
        device=device,
        dtype=dtype,
        caller=caller,
    )
    pre_flat = relaxation.pre_flat
    expected_prefix = (int(pre_flat.lower.shape[0]),)
    if A_u.dim() != 3 or A_l.dim() != 3:
        raise ValueError(f"{caller} dense ReLU reference expects rank-3 A tensors")
    if (
        tuple(A_u.shape[:1]) != expected_prefix
        or tuple(A_l.shape[:1]) != expected_prefix
    ):
        raise ValueError(f"{caller} dense ReLU reference batch mismatch")
    if int(A_u.shape[2]) != int(pre_flat.lower.shape[1]) or int(A_l.shape[2]) != int(
        pre_flat.lower.shape[1]
    ):
        raise ValueError(
            f"{caller} relu backward shape mismatch: pre={tuple(pre.lower.shape)} "
            f"A_u={tuple(A_u.shape)} A_l={tuple(A_l.shape)}"
        )
    if tuple(b_u.shape) != tuple(A_u.shape[:2]) or tuple(b_l.shape) != tuple(
        A_l.shape[:2]
    ):
        raise ValueError(f"{caller} dense ReLU reference bias shape mismatch")
    if pre.lower.dim() != 2:
        if relu_pre_add_coeff_u is not None and x_name in relu_pre_add_coeff_u:
            raise NotImplementedError(
                f"{caller} only supports relu_pre_add_coeff_u on rank-2 pre-activations; got {tuple(pre.lower.shape)} for {x_name}"
            )

    sel_alpha_u = torch.where(
        A_u >= 0, relaxation.alpha_u.unsqueeze(1), relaxation.alpha_l.unsqueeze(1)
    )
    sel_beta_u = torch.where(
        A_u >= 0, relaxation.beta_u.unsqueeze(1), relaxation.beta_l.unsqueeze(1)
    )
    out_b_u = b_u + (A_u * sel_beta_u).sum(dim=2)
    out_A_u = A_u * sel_alpha_u
    if relu_pre_add_coeff_u is not None and x_name in relu_pre_add_coeff_u:
        out_A_u = _apply_relu_pre_add_coeff(
            out_A_u,
            relu_pre_add_coeff_u[x_name],
            x_name=x_name,
            label="relu_pre_add_coeff_u",
            device=device,
            dtype=dtype,
        )

    sel_alpha_l = torch.where(
        A_l >= 0, relaxation.alpha_l.unsqueeze(1), relaxation.alpha_u.unsqueeze(1)
    )
    sel_beta_l = torch.where(
        A_l >= 0, relaxation.beta_l.unsqueeze(1), relaxation.beta_u.unsqueeze(1)
    )
    out_b_l = b_l + (A_l * sel_beta_l).sum(dim=2)
    out_A_l = A_l * sel_alpha_l
    if relu_pre_add_coeff_l is not None and x_name in relu_pre_add_coeff_l:
        out_A_l = _apply_relu_pre_add_coeff(
            out_A_l,
            relu_pre_add_coeff_l[x_name],
            x_name=x_name,
            label="relu_pre_add_coeff_l",
            device=device,
            dtype=dtype,
        )

    return DenseReluBackwardResult(A_u=out_A_u, A_l=out_A_l, b_u=out_b_u, b_l=out_b_l)


def _backprop_relu_step_dense(
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
    pre_numel = int(pre.lower[0].numel())
    if pre_numel != state.A_u.input_numel or pre_numel != state.A_l.input_numel:
        raise ValueError(
            f"{caller} relu backward shape mismatch: pre={tuple(pre.lower.shape)} "
            f"A_u.input_shape={state.A_u.input_shape} A_l.input_shape={state.A_l.input_shape}"
        )

    dense_A_u = materialize_linear_operator(
        state.A_u,
        reason="relu_sign_split",
        operator_site=f"{x_name}:upper",
        source_value=x_name,
        source_primal_op="relu",
        persistent_or_ephemeral="persistent",
        logical_lifetime_begin="relu_backward_step",
        logical_lifetime_end="backward_end",
        alpha_related=relu_alpha is not None and x_name in relu_alpha,
        beta_related=(
            (relu_pre_add_coeff_u is not None and x_name in relu_pre_add_coeff_u)
            or (relu_pre_add_coeff_l is not None and x_name in relu_pre_add_coeff_l)
        ),
    )
    dense_A_l = materialize_linear_operator(
        state.A_l,
        reason="relu_sign_split",
        operator_site=f"{x_name}:lower",
        source_value=x_name,
        source_primal_op="relu",
        persistent_or_ephemeral="persistent",
        logical_lifetime_begin="relu_backward_step",
        logical_lifetime_end="backward_end",
        alpha_related=relu_alpha is not None and x_name in relu_alpha,
        beta_related=(
            (relu_pre_add_coeff_u is not None and x_name in relu_pre_add_coeff_u)
            or (relu_pre_add_coeff_l is not None and x_name in relu_pre_add_coeff_l)
        ),
    )
    dense = _backprop_relu_step_dense_reference(
        dense_A_u,
        dense_A_l,
        state.b_u,
        state.b_l,
        pre=pre,
        x_name=x_name,
        relu_alpha=relu_alpha,
        relu_pre_add_coeff_u=relu_pre_add_coeff_u,
        relu_pre_add_coeff_l=relu_pre_add_coeff_l,
        device=device,
        dtype=dtype,
        caller=caller,
    )

    return AffineBackwardState(
        A_u=DenseLinearOperator(dense.A_u, input_shape=orig_input_shape),
        A_l=DenseLinearOperator(dense.A_l, input_shape=orig_input_shape),
        b_u=dense.b_u,
        b_l=dense.b_l,
    )


def _add_structured_relu_pre_coeff(
    operator: LinearOperator,
    add_raw: Any,
    *,
    x_name: str,
    label: str,
    input_shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> LinearOperator:
    add = _broadcast_relu_pre_add_coeff(
        add_raw,
        batch=operator.shape[0],
        flat_dim=operator.input_numel,
        x_name=x_name,
        label=label,
        device=device,
        dtype=dtype,
    )
    coeffs = add.unsqueeze(1).expand(
        operator.shape[0], operator.spec_dim, operator.input_numel
    )
    return operator.add(DenseLinearOperator(coeffs, input_shape=input_shape))


def _backprop_relu_step_structured(
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
    """Keep the main post-ReLU coefficient structured; materialize only bias reduction."""

    input_shape = tuple(int(dim) for dim in pre.lower.shape[1:])
    pre_numel = int(pre.lower[0].numel())
    if pre_numel != state.A_u.input_numel or pre_numel != state.A_l.input_numel:
        raise ValueError(
            f"{caller} relu backward shape mismatch: pre={tuple(pre.lower.shape)} "
            f"A_u.input_shape={state.A_u.input_shape} A_l.input_shape={state.A_l.input_shape}"
        )
    if (
        pre.lower.dim() != 2
        and relu_pre_add_coeff_u is not None
        and x_name in relu_pre_add_coeff_u
    ):
        raise NotImplementedError(
            f"{caller} only supports relu_pre_add_coeff_u on rank-2 pre-activations; "
            f"got {tuple(pre.lower.shape)} for {x_name}"
        )
    relaxation = _relu_backward_relaxation(
        pre=pre,
        x_name=x_name,
        relu_alpha=relu_alpha,
        device=device,
        dtype=dtype,
        caller=caller,
    )
    beta_related = (
        relu_pre_add_coeff_u is not None and x_name in relu_pre_add_coeff_u
    ) or (relu_pre_add_coeff_l is not None and x_name in relu_pre_add_coeff_l)
    bias_A_u = materialize_linear_operator(
        state.A_u,
        reason="relu_bias_sign_reduce",
        operator_site=f"{x_name}:upper:bias",
        source_value=x_name,
        source_primal_op="relu",
        persistent_or_ephemeral="ephemeral",
        logical_lifetime_begin="relu_bias_reduce",
        logical_lifetime_end="relu_bias_reduce:return",
        beta_related=beta_related,
    )
    bias_A_l = materialize_linear_operator(
        state.A_l,
        reason="relu_bias_sign_reduce",
        operator_site=f"{x_name}:lower:bias",
        source_value=x_name,
        source_primal_op="relu",
        persistent_or_ephemeral="ephemeral",
        logical_lifetime_begin="relu_bias_reduce",
        logical_lifetime_end="relu_bias_reduce:return",
        beta_related=beta_related,
    )
    sel_beta_u = torch.where(
        bias_A_u >= 0,
        relaxation.beta_u.unsqueeze(1),
        relaxation.beta_l.unsqueeze(1),
    )
    sel_beta_l = torch.where(
        bias_A_l >= 0,
        relaxation.beta_l.unsqueeze(1),
        relaxation.beta_u.unsqueeze(1),
    )
    out_b_u = state.b_u + (bias_A_u * sel_beta_u).sum(dim=2)
    out_b_l = state.b_l + (bias_A_l * sel_beta_l).sum(dim=2)

    batch = int(pre.lower.shape[0])
    upper: LinearOperator = SignSplitLinearOperator(
        base=state.A_u,
        positive_scale=relaxation.alpha_u.reshape(batch, *input_shape),
        negative_scale=relaxation.alpha_l.reshape(batch, *input_shape),
        source_value=x_name,
        bound_direction="upper",
    )
    lower: LinearOperator = SignSplitLinearOperator(
        base=state.A_l,
        positive_scale=relaxation.alpha_l.reshape(batch, *input_shape),
        negative_scale=relaxation.alpha_u.reshape(batch, *input_shape),
        source_value=x_name,
        bound_direction="lower",
    )
    if relu_pre_add_coeff_u is not None and x_name in relu_pre_add_coeff_u:
        upper = _add_structured_relu_pre_coeff(
            upper,
            relu_pre_add_coeff_u[x_name],
            x_name=x_name,
            label="relu_pre_add_coeff_u",
            input_shape=input_shape,
            device=device,
            dtype=dtype,
        )
    if relu_pre_add_coeff_l is not None and x_name in relu_pre_add_coeff_l:
        lower = _add_structured_relu_pre_coeff(
            lower,
            relu_pre_add_coeff_l[x_name],
            x_name=x_name,
            label="relu_pre_add_coeff_l",
            input_shape=input_shape,
            device=device,
            dtype=dtype,
        )
    return AffineBackwardState(A_u=upper, A_l=lower, b_u=out_b_u, b_l=out_b_l)


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
    placements = _RELU_BACKWARD_PLACEMENTS.get()
    mode = (
        placements.get(x_name, _RELU_BACKWARD_MODE.get())
        if placements
        else _RELU_BACKWARD_MODE.get()
    )
    implementation = (
        _backprop_relu_step_dense if mode == "dense" else _backprop_relu_step_structured
    )
    return implementation(
        state,
        pre=pre,
        x_name=x_name,
        relu_alpha=relu_alpha,
        relu_pre_add_coeff_u=relu_pre_add_coeff_u,
        relu_pre_add_coeff_l=relu_pre_add_coeff_l,
        device=device,
        dtype=dtype,
        caller=caller,
    )


def _execute_fused_relu_affine_step(  # pylint: disable=too-many-arguments,too-many-locals
    state: AffineBackwardState,
    *,
    affine_op: Any,
    pre: IntervalState,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    weight_raw: Any,
    bias_raw: Any,
    executor: FusedCrownExecutor,
    context: FusedCrownExecutionContext,
    device: torch.device,
    dtype: torch.dtype,
    caller: str,
) -> Optional[AffineBackwardState]:
    """Execute one planned dense-boundary region or request deterministic fallback."""
    if (
        not context.plain_crown
        or context.requires_grad
        or context.alpha_enabled
        or context.beta_enabled
        or context.split_state_present
    ):
        return None

    attrs: Dict[str, object] = dict(affine_op.attrs)
    weight: torch.Tensor
    bias: Optional[torch.Tensor]
    if affine_op.op_type == "linear":
        weight, bias = _normalize_linear_inputs(
            weight_raw, bias_raw, device=device, dtype=dtype, caller=caller
        )
    else:
        weight, bias, stride, padding, dilation, groups = _normalize_conv2d_inputs(
            weight_raw,
            bias_raw,
            attrs=dict(affine_op.attrs),
            device=device,
            dtype=dtype,
            caller=caller,
        )
        attrs.update(stride=stride, padding=padding, dilation=dilation, groups=groups)
        attrs["output_padding"] = tuple(
            int(input_shape[axis + 1])
            - (
                (int(output_shape[axis + 1]) - 1) * int(stride[axis])
                - 2 * int(padding[axis])
                + int(dilation[axis]) * (int(weight.shape[axis + 2]) - 1)
                + 1
            )
            for axis in range(2)
        )
    descriptor = FusedReluAffineDescriptor(
        kind=affine_op.op_type,
        coefficient_shape=state.A_u.shape,
        weight=weight,
        bias=bias,
        input_shape=input_shape,
        output_shape=output_shape,
        attrs=attrs,
        device=device,
        dtype=dtype,
    )
    if not executor.supports_descriptor(descriptor, context):
        return None

    A_u = materialize_linear_operator(
        state.A_u,
        reason="fused_region_dense_boundary",
        operator_site=f"{affine_op.outputs[0]}:upper:fused",
        source_value=affine_op.outputs[0],
        source_primal_op=affine_op.op_type,
        persistent_or_ephemeral="ephemeral",
        logical_lifetime_begin="fused_region",
        logical_lifetime_end="fused_region:return",
        beta_related=False,
    ).contiguous()
    A_l = materialize_linear_operator(
        state.A_l,
        reason="fused_region_dense_boundary",
        operator_site=f"{affine_op.outputs[0]}:lower:fused",
        source_value=affine_op.outputs[0],
        source_primal_op=affine_op.op_type,
        persistent_or_ephemeral="ephemeral",
        logical_lifetime_begin="fused_region",
        logical_lifetime_end="fused_region:return",
        beta_related=False,
    ).contiguous()
    relaxation = _relu_backward_relaxation(
        pre=pre,
        x_name=affine_op.outputs[0],
        relu_alpha=None,
        device=device,
        dtype=dtype,
        caller=caller,
    )
    request = FusedReluAffineRequest(
        kind=affine_op.op_type,
        A_u=A_u,
        A_l=A_l,
        alpha_u=relaxation.alpha_u.contiguous(),
        alpha_l=relaxation.alpha_l.contiguous(),
        beta_u=relaxation.beta_u.contiguous(),
        beta_l=relaxation.beta_l.contiguous(),
        weight=weight.contiguous(),
        bias=None if bias is None else bias.contiguous(),
        input_shape=input_shape,
        output_shape=output_shape,
        attrs=attrs,
    )
    if not executor.supports(request, context):
        return None
    stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    result = executor.run(request, stream=stream)
    return AffineBackwardState(
        A_u=DenseLinearOperator(result.A_prev_u, input_shape=input_shape),
        A_l=DenseLinearOperator(result.A_prev_l, input_shape=input_shape),
        b_u=state.b_u + result.bias_delta_u,
        b_l=state.b_l + result.bias_delta_l,
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
    fused_crown_executor: Optional[FusedCrownExecutor],
    fused_crown_steps: Sequence[FusedCrownExecutionStep],
    fused_crown_context: FusedCrownExecutionContext,
    relu_objective_influence_out: Optional[Dict[str, torch.Tensor]],
    relu_lower_coefficients_out: Optional[Dict[str, torch.Tensor]],
    caller: str,
    b4b_region_observer: B4BRegionLiveObserverProtocol | None = None,
) -> IntervalState:
    task = module.get_entry_task()
    raw_params = module.bindings.get("params", {})
    params: Dict[str, Any] = dict(raw_params) if isinstance(raw_params, dict) else {}

    def _get_tensor(name: str) -> Any:
        if name not in params:
            raise KeyError(f"missing param tensor: {name}")
        return params[name]

    def _get_state(name: str) -> Any:
        if name == input_spec.value_name:
            return InputPerturbationState(
                center=input_spec.center, perturbation=input_spec.perturbation
            )
        if name in interval_env:
            return interval_env[name]
        if name in params:
            t = params[name]
            if not torch.is_tensor(t):
                t = torch.as_tensor(
                    t, device=input_spec.center.device, dtype=input_spec.center.dtype
                )
            return IntervalState(lower=t, upper=t)
        raise KeyError(f"missing value in interval_env/params: {name}")

    def _ensure_interval(state_like: Any) -> IntervalState:
        if isinstance(state_like, IntervalState):
            return state_like
        if isinstance(state_like, InputPerturbationState):
            lb, ub = state_like.perturbation.bounding_box(state_like.center)
            return IntervalState(lower=lb, upper=ub)
        raise TypeError(
            f"expected IntervalState/InputPerturbationState, got {type(state_like)}"
        )

    resolved_output = _resolve_output_value(task, output_value, caller=caller)
    _y_out, batch, out_dim, output_shape, device, dtype = _get_output_meta(
        interval_env=interval_env,
        output_value=resolved_output,
        caller=caller,
    )
    init_state = _init_backward_state(
        batch=batch,
        out_dim=out_dim,
        output_shape=output_shape,
        device=device,
        dtype=dtype,
        linear_spec_C=linear_spec_C,
    )
    adjoints: Dict[str, AffineBackwardState] = {resolved_output: init_state}
    dynamic_names = _dynamic_value_names(
        input_spec=input_spec, interval_env=interval_env
    )

    validated_fused_steps = validate_fused_crown_execution_steps(
        task.ops, fused_crown_steps
    )
    fused_by_relu = {step.relu_op_index: step for step in validated_fused_steps}
    consumed_affine_indices: set[int] = set()
    for op_index in range(len(task.ops) - 1, -1, -1):
        if op_index in consumed_affine_indices:
            continue
        op = task.ops[op_index]
        if len(op.outputs) != 1:
            raise NotImplementedError(
                f"{caller} expects single-output ops, got outputs={op.outputs}"
            )
        out_name = op.outputs[0]
        state = adjoints.pop(out_name, None)
        if state is None:
            continue

        if op.op_type == "linear":
            weight_raw = _get_tensor(op.inputs[1])
            bias_raw = _get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None
            contrib = _backprop_linear_step(
                state,
                weight=weight_raw,
                bias=bias_raw,
                device=device,
                dtype=dtype,
                caller=caller,
            )
            in_name = op.inputs[0]
            in_shape = _value_shape(
                input_spec=input_spec, interval_env=interval_env, value_name=in_name
            )
            if b4b_region_observer is not None and b4b_region_observer.wants(out_name):
                weight_tensor, normalized_bias = _normalize_linear_inputs(
                    weight_raw,
                    bias_raw,
                    device=device,
                    dtype=dtype,
                    caller=caller,
                )
                replacement = b4b_region_observer.observe_affine_output(
                    out_name,
                    operator_weight=weight_tensor,
                    operator_bias=(None if bias_raw is None else normalized_bias),
                    output_lower_a=contrib.A_l.to_dense()
                    .reshape(batch, contrib.A_l.spec_dim, *in_shape)
                    .contiguous(),
                    output_bias=contrib.b_l,
                    operator_attributes={
                        "operator_kind": "linear",
                        "weight_shape": list(weight_tensor.shape),
                    },
                )
                if replacement is not None:
                    replaced_lower_a, replaced_bias = replacement
                    contrib = AffineBackwardState(
                        A_u=contrib.A_u,
                        A_l=cast(
                            LinearOperator,
                            DenseLinearOperator(
                                replaced_lower_a,
                                input_shape=in_shape,
                            ),
                        ),
                        b_u=contrib.b_u,
                        b_l=replaced_bias,
                    )
            adjoints[in_name] = _accumulate_backward_state(
                adjoints.get(in_name), contrib, input_shape=in_shape
            )
            continue

        if op.op_type == "flatten":
            start_dim = int(op.attrs.get("start_dim", 1))
            end_dim = int(op.attrs.get("end_dim", -1))
            if start_dim != 1 or end_dim != -1:
                raise NotImplementedError(
                    f"{caller} only supports flatten(start_dim=1, end_dim=-1), got attrs={op.attrs}"
                )
            in_name = op.inputs[0]
            in_shape = _value_shape(
                input_spec=input_spec, interval_env=interval_env, value_name=in_name
            )
            contrib = _backprop_flatten_step(state, pre_shape=in_shape)
            adjoints[in_name] = _accumulate_backward_state(
                adjoints.get(in_name), contrib, input_shape=in_shape
            )
            continue

        if op.op_type == "reshape":
            in_name = op.inputs[0]
            in_shape = _value_shape(
                input_spec=input_spec, interval_env=interval_env, value_name=in_name
            )
            contrib = _backprop_flatten_step(state, pre_shape=in_shape)
            adjoints[in_name] = _accumulate_backward_state(
                adjoints.get(in_name), contrib, input_shape=in_shape
            )
            continue

        if op.op_type == "relu":
            x_name = op.inputs[0]
            if x_name not in relu_pre:
                raise KeyError(
                    f"missing relu pre-activation bounds for value: {x_name}"
                )
            if b4b_region_observer is not None and b4b_region_observer.wants(x_name):
                observed_shape = tuple(
                    int(dimension) for dimension in relu_pre[x_name].lower.shape[1:]
                )
                aligned = _align_backward_state_input_shape(
                    state, input_shape=observed_shape
                )
                incoming_lower_a = (
                    aligned.A_l.to_dense()
                    .reshape(batch, aligned.A_l.spec_dim, *observed_shape)
                    .contiguous()
                )
                b4b_region_observer.observe_relu_input(
                    x_name,
                    incoming_lower_a=incoming_lower_a,
                    preactivation_lower=relu_pre[x_name].lower,
                    preactivation_upper=relu_pre[x_name].upper,
                    incoming_lower_bias=aligned.b_l,
                )
                observed_incoming = b4b_region_observer.observed_incoming_lower_a(
                    x_name
                )
                state = AffineBackwardState(
                    A_u=aligned.A_u,
                    A_l=cast(
                        LinearOperator,
                        DenseLinearOperator(
                            observed_incoming.reshape(
                                batch, aligned.A_l.spec_dim, aligned.A_l.input_numel
                            ),
                            input_shape=observed_shape,
                        ),
                    ),
                    b_u=aligned.b_u,
                    b_l=aligned.b_l,
                )
            if relu_lower_coefficients_out is not None:
                if x_name in relu_lower_coefficients_out:
                    raise ValueError(
                        f"duplicate ReLU lower-coefficient identity: {x_name}"
                    )
                coefficient_shape = tuple(
                    int(dimension) for dimension in relu_pre[x_name].lower.shape[1:]
                )
                aligned_lower = _align_backward_state_input_shape(
                    state, input_shape=coefficient_shape
                ).A_l.to_dense()
                relu_lower_coefficients_out[x_name] = (
                    aligned_lower.reshape(
                        batch, int(aligned_lower.shape[1]), *coefficient_shape
                    )
                    .detach()
                    .contiguous()
                    .clone()
                )
            if relu_objective_influence_out is not None:
                if x_name in relu_objective_influence_out:
                    raise ValueError(
                        f"duplicate ReLU objective influence identity: {x_name}"
                    )
                input_shape = tuple(
                    int(dimension) for dimension in relu_pre[x_name].lower.shape[1:]
                )
                aligned = _align_backward_state_input_shape(
                    state, input_shape=input_shape
                )
                upper_coefficients = aligned.A_u.to_dense().reshape(
                    batch, aligned.A_u.spec_dim, *input_shape
                )
                lower_coefficients = aligned.A_l.to_dense().reshape(
                    batch, aligned.A_l.spec_dim, *input_shape
                )
                relu_objective_influence_out[x_name] = (
                    torch.maximum(
                        upper_coefficients.abs().amax(dim=1),
                        lower_coefficients.abs().amax(dim=1),
                    )
                    .detach()
                    .contiguous()
                )
            step = fused_by_relu.get(op_index)
            if (
                fused_crown_executor is not None
                and step is not None
                and x_name not in adjoints
            ):
                if step.affine_op_index != op_index - 1:
                    raise ValueError(
                        "fused CROWN v1 requires adjacent Affine->ReLU ops"
                    )
                affine_op = task.ops[step.affine_op_index]
                if tuple(step.consumed_outputs) != (out_name, x_name):
                    raise ValueError(
                        "fused CROWN execution step no longer matches the task graph"
                    )
                affine_input = affine_op.inputs[0]
                fused = _execute_fused_relu_affine_step(
                    state,
                    affine_op=affine_op,
                    pre=relu_pre[x_name],
                    input_shape=_value_shape(
                        input_spec=input_spec,
                        interval_env=interval_env,
                        value_name=affine_input,
                    ),
                    output_shape=_value_shape(
                        input_spec=input_spec,
                        interval_env=interval_env,
                        value_name=x_name,
                    ),
                    weight_raw=_get_tensor(affine_op.inputs[1]),
                    bias_raw=(
                        _get_tensor(affine_op.inputs[2])
                        if len(affine_op.inputs) == 3
                        else None
                    ),
                    executor=fused_crown_executor,
                    context=fused_crown_context,
                    device=device,
                    dtype=dtype,
                    caller=caller,
                )
                if fused is not None:
                    adjoints[affine_input] = _accumulate_backward_state(
                        adjoints.get(affine_input),
                        fused,
                        input_shape=fused.A_u.input_shape,
                    )
                    consumed_affine_indices.add(step.affine_op_index)
                    continue
            contrib = _backprop_relu_step(
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
            in_shape = _value_shape(
                input_spec=input_spec, interval_env=interval_env, value_name=x_name
            )
            adjoints[x_name] = _accumulate_backward_state(
                adjoints.get(x_name), contrib, input_shape=in_shape
            )
            continue

        if op.op_type == "conv2d":
            in_name = op.inputs[0]
            in_shape = _value_shape(
                input_spec=input_spec, interval_env=interval_env, value_name=in_name
            )
            out_shape = _value_shape(
                input_spec=input_spec, interval_env=interval_env, value_name=out_name
            )
            weight_raw = _get_tensor(op.inputs[1])
            bias_raw = _get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None
            contrib = _backprop_conv2d_step(
                state,
                input_shape=in_shape,
                output_shape=out_shape,
                weight=weight_raw,
                bias=bias_raw,
                attrs=dict(op.attrs),
                device=device,
                dtype=dtype,
                caller=caller,
            )
            if b4b_region_observer is not None and b4b_region_observer.wants(out_name):
                weight_tensor, conv_bias, stride, padding, dilation, groups = (
                    _normalize_conv2d_inputs(
                        weight_raw,
                        bias_raw,
                        attrs=dict(op.attrs),
                        device=device,
                        dtype=dtype,
                        caller=caller,
                    )
                )
                replacement = b4b_region_observer.observe_affine_output(
                    out_name,
                    operator_weight=weight_tensor,
                    operator_bias=conv_bias,
                    output_lower_a=contrib.A_l.to_dense()
                    .reshape(batch, contrib.A_l.spec_dim, *in_shape)
                    .contiguous(),
                    output_bias=contrib.b_l,
                    operator_attributes={
                        "operator_kind": "conv2d",
                        "weight_shape": list(weight_tensor.shape),
                        "stride": list(stride),
                        "padding": list(padding),
                        "dilation": list(dilation),
                        "groups": groups,
                    },
                )
                if replacement is not None:
                    replaced_lower_a, replaced_bias = replacement
                    contrib = AffineBackwardState(
                        A_u=contrib.A_u,
                        A_l=cast(
                            LinearOperator,
                            DenseLinearOperator(
                                replaced_lower_a,
                                input_shape=in_shape,
                            ),
                        ),
                        b_u=contrib.b_u,
                        b_l=replaced_bias,
                    )
            adjoints[in_name] = _accumulate_backward_state(
                adjoints.get(in_name), contrib, input_shape=in_shape
            )
            continue

        if op.op_type == "add":
            out_shape = _value_shape(
                input_spec=input_spec, interval_env=interval_env, value_name=out_name
            )
            base = _align_backward_state_input_shape(state, input_shape=out_shape)
            a_state = base
            b_state = base
            const_bias_u = base.b_u
            const_bias_l = base.b_l
            dynamic_inputs: List[str] = []
            for in_name in op.inputs:
                val = _ensure_interval(_get_state(in_name))
                if tuple(int(dim) for dim in val.lower.shape[1:]) != tuple(out_shape):
                    raise NotImplementedError(
                        f"{caller} only supports add with exact same-shape non-broadcast inputs; "
                        f"got output_shape={out_shape} input_shape={tuple(int(dim) for dim in val.lower.shape[1:])}"
                    )
                if in_name in dynamic_names:
                    dynamic_inputs.append(in_name)
                else:
                    const_bias_u = const_bias_u + a_state.A_u.contract_input(val.lower)
                    const_bias_l = const_bias_l + a_state.A_l.contract_input(val.lower)
            bias_parts = _split_bias_once(
                AffineBackwardState(
                    A_u=a_state.A_u, A_l=a_state.A_l, b_u=const_bias_u, b_l=const_bias_l
                ),
                num_children=len(dynamic_inputs),
            )
            for idx, in_name in enumerate(dynamic_inputs):
                in_shape = _value_shape(
                    input_spec=input_spec, interval_env=interval_env, value_name=in_name
                )
                contrib = AffineBackwardState(
                    A_u=a_state.A_u,
                    A_l=a_state.A_l,
                    b_u=bias_parts[idx][0],
                    b_l=bias_parts[idx][1],
                )
                adjoints[in_name] = _accumulate_backward_state(
                    adjoints.get(in_name), contrib, input_shape=in_shape
                )
            continue

        if op.op_type == "concat":
            out_shape = _value_shape(
                input_spec=input_spec, interval_env=interval_env, value_name=out_name
            )
            axis = normalize_concat_axis(
                op.attrs.get("axis", 1),
                rank_with_batch=len(out_shape) + 1,
                caller=caller,
            )
            if axis != 1:
                raise AssertionError(
                    "supported concat axes must normalize to batch-preserving axis=1"
                )
            base = _align_backward_state_input_shape(state, input_shape=out_shape)
            bias_parts = _split_bias_once(base, num_children=len(op.inputs))
            input_shapes = [
                _value_shape(
                    input_spec=input_spec, interval_env=interval_env, value_name=in_name
                )
                for in_name in op.inputs
            ]
            total = validate_concat_value_shapes(input_shapes, caller=caller)
            start = 0
            for idx, (in_name, in_shape) in enumerate(zip(op.inputs, input_shapes)):
                if in_name not in dynamic_names:
                    raise NotImplementedError(
                        f"{caller} only supports concat over dynamic tensor values, got {in_name}"
                    )
                stop = start + int(in_shape[0])
                contrib = AffineBackwardState(
                    A_u=(
                        base.A_u
                        if tuple(base.A_u.input_shape) == tuple(out_shape)
                        else base.A_u.reshape_input(out_shape)
                    ).slice_input(
                        in_shape,
                        start=start,
                        stop=stop,
                    ),
                    A_l=(
                        base.A_l
                        if tuple(base.A_l.input_shape) == tuple(out_shape)
                        else base.A_l.reshape_input(out_shape)
                    ).slice_input(
                        in_shape,
                        start=start,
                        stop=stop,
                    ),
                    b_u=bias_parts[idx][0],
                    b_l=bias_parts[idx][1],
                )
                adjoints[in_name] = _accumulate_backward_state(
                    adjoints.get(in_name), contrib, input_shape=in_shape
                )
                start = stop
            if start != total or total != int(out_shape[0]):
                raise ValueError(
                    f"{caller} concat backward shape mismatch: sliced={start} validated={total} but output axis size is {int(out_shape[0])}"
                )
            continue

        raise NotImplementedError(
            f"run_crown_ibp_mlp unsupported op_type in backward: {op.op_type}"
        )

    if input_spec.value_name not in adjoints:
        raise RuntimeError(
            f"{caller} backward did not reach input value {input_spec.value_name}"
        )
    input_state = _align_backward_state_input_shape(
        adjoints[input_spec.value_name],
        input_shape=tuple(int(dim) for dim in input_spec.center.shape[1:]),
    )
    x0 = input_spec.center
    _lb_u, ub_u = input_spec.perturbation.concretize_affine(
        center=x0, A=input_state.A_u, b=input_state.b_u
    )
    lb_l, _ub_l = input_spec.perturbation.concretize_affine(
        center=x0, A=input_state.A_l, b=input_state.b_l
    )
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
    materialization_plan: Optional[MaterializationPlan] = None,
    materialization_placement_plan: Optional[MaterializationPlacementPlan] = None,
    fused_crown_executor: Optional[FusedCrownExecutor] = None,
    fused_crown_steps: Sequence[FusedCrownExecutionStep] = (),
) -> IntervalState:
    """
    Minimal CROWN-IBP for a single-task general DAG subset.

    - Forward: interval IBP to get pre-activation bounds for ReLU.
    - Backward: CROWN-style linear bound propagation using ReLU relaxations fixed by IBP bounds.

    Limitations:
    - Single task only.
    - Supports op_type in {"linear", "relu", "conv2d", "flatten", "add", "concat"}.
    - `flatten` is restricted to `start_dim=1, end_dim=-1`.
    - `add` only supports exact same-shape inputs (no broadcast).
    - `concat` only supports feature-axis concat on [B,F] and channel-axis concat on [B,C,H,W].
    - Conv support is plain CROWN-IBP only; alpha/beta/BaB remain MLP-only.
    - Returns bounds for selected rows of any produced output value; non-rank-2
      intermediate values require an explicit flattened `linear_spec_C`.
    """
    module.validate()
    task = module.get_entry_task()
    if task.kind != TaskKind.INTERVAL_IBP:
        raise NotImplementedError(
            f"run_crown_ibp_mlp only supports INTERVAL_IBP, got {task.kind}"
        )
    if module.task_graph is not None or len(module.tasks) != 1:
        raise NotImplementedError(
            "run_crown_ibp_mlp currently supports single-task BFTaskModule only"
        )
    if not task.ops:
        raise ValueError("run_crown_ibp_mlp expects a non-empty task")
    input_spec = _normalize_input_spec(input_spec)
    if output_value is None:
        if len(task.output_values) != 1:
            raise ValueError(
                f"task has {len(task.output_values)} outputs; specify output_value explicitly"
            )
        output_value = task.output_values[0]

    interval_env, relu_pre = _forward_ibp_trace_mlp(
        module, input_spec, relu_split_state=relu_split_state
    )
    fused_context = FusedCrownExecutionContext(
        requires_grad=input_spec.center.requires_grad,
        alpha_enabled=relu_alpha is not None,
        beta_enabled=relu_pre_add_coeff_u is not None
        or relu_pre_add_coeff_l is not None,
        split_state_present=relu_split_state is not None,
    )
    with _apply_execution_materialization(
        materialization_plan, materialization_placement_plan
    ):
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
            fused_crown_executor=fused_crown_executor,
            fused_crown_steps=fused_crown_steps,
            fused_crown_context=fused_context,
            relu_objective_influence_out=None,
            relu_lower_coefficients_out=None,
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
    materialization_plan: Optional[MaterializationPlan] = None,
    materialization_placement_plan: Optional[MaterializationPlacementPlan] = None,
    fused_crown_executor: Optional[FusedCrownExecutor] = None,
    fused_crown_steps: Sequence[FusedCrownExecutionStep] = (),
    fused_crown_context: Optional[FusedCrownExecutionContext] = None,
    b4b_region_observer: B4BRegionLiveObserverProtocol | None = None,
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
        raise NotImplementedError(
            f"run_crown_ibp_mlp only supports INTERVAL_IBP, got {task.kind}"
        )
    if module.task_graph is not None or len(module.tasks) != 1:
        raise NotImplementedError(
            "run_crown_ibp_mlp_from_forward_trace currently supports single-task BFTaskModule only"
        )

    with _apply_execution_materialization(
        materialization_plan, materialization_placement_plan
    ):
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
            fused_crown_executor=fused_crown_executor,
            fused_crown_steps=fused_crown_steps,
            fused_crown_context=(
                fused_crown_context
                if fused_crown_context is not None
                else FusedCrownExecutionContext(
                    requires_grad=input_spec.center.requires_grad,
                    alpha_enabled=relu_alpha is not None,
                    beta_enabled=relu_pre_add_coeff_u is not None
                    or relu_pre_add_coeff_l is not None,
                )
            ),
            relu_objective_influence_out=None,
            relu_lower_coefficients_out=None,
            caller="run_crown_ibp_mlp_from_forward_trace",
            b4b_region_observer=b4b_region_observer,
        )


def run_crown_ibp_mlp_with_relu_influence_from_forward_trace(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    interval_env: Dict[str, IntervalState],
    relu_pre: Dict[str, IntervalState],
    linear_spec_C: torch.Tensor,
    output_value: Optional[str] = None,
) -> tuple[IntervalState, Dict[str, torch.Tensor]]:
    """Return sound objective bounds and per-ReLU backward coefficient influence."""

    module.validate()
    task = module.get_entry_task()
    if task.kind != TaskKind.INTERVAL_IBP:
        raise NotImplementedError(
            "ReLU influence extraction only supports INTERVAL_IBP tasks"
        )
    if module.task_graph is not None or len(module.tasks) != 1:
        raise NotImplementedError(
            "ReLU influence extraction currently supports one task only"
        )
    influence: Dict[str, torch.Tensor] = {}
    bounds = _run_crown_backward_from_trace(
        module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec_C,
        output_value=output_value,
        relu_alpha=None,
        relu_pre_add_coeff_u=None,
        relu_pre_add_coeff_l=None,
        fused_crown_executor=None,
        fused_crown_steps=(),
        fused_crown_context=FusedCrownExecutionContext(),
        relu_objective_influence_out=influence,
        relu_lower_coefficients_out=None,
        caller="run_crown_ibp_mlp_with_relu_influence_from_forward_trace",
    )
    if set(influence) != set(relu_pre):
        raise ValueError("ReLU objective influence coverage differs")
    return bounds, {name: influence[name] for name in relu_pre}


def run_crown_ibp_mlp_with_relu_lower_coefficients_from_forward_trace(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    interval_env: Dict[str, IntervalState],
    relu_pre: Dict[str, IntervalState],
    linear_spec_C: torch.Tensor,
    relu_alpha: Dict[str, torch.Tensor],
    relu_pre_add_coeff_l: Dict[str, torch.Tensor],
    output_value: Optional[str] = None,
) -> tuple[IntervalState, Dict[str, torch.Tensor]]:
    """Return optimized lower bounds and the lower adjoint at every ReLU."""

    coefficients: Dict[str, torch.Tensor] = {}
    bounds = _run_crown_backward_from_trace(
        module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=linear_spec_C,
        output_value=output_value,
        relu_alpha=relu_alpha,
        relu_pre_add_coeff_u=None,
        relu_pre_add_coeff_l=relu_pre_add_coeff_l,
        fused_crown_executor=None,
        fused_crown_steps=(),
        fused_crown_context=FusedCrownExecutionContext(
            requires_grad=False,
            alpha_enabled=True,
            beta_enabled=True,
            split_state_present=True,
        ),
        relu_objective_influence_out=None,
        relu_lower_coefficients_out=coefficients,
        caller="run_crown_ibp_mlp_with_relu_lower_coefficients_from_forward_trace",
    )
    if set(coefficients) != set(relu_pre):
        raise ValueError("ReLU lower-coefficient coverage differs")
    return bounds, {name: coefficients[name] for name in relu_pre}


def run_crown_ibp_mlp_with_placement_retry(
    module: BFTaskModule,
    input_spec: InputSpecLike,
    *,
    placement_plans: Tuple[MaterializationPlacementPlan, ...],
    linear_spec_C: Optional[torch.Tensor] = None,
    output_value: Optional[str] = None,
    relu_split_state: Optional[Dict[str, torch.Tensor]] = None,
    memory_budget_bytes: Optional[int] = None,
    max_attempts: int = 6,
    prediction_budget_factor: float = 1.3,
) -> tuple[IntervalState, PlacementRetryStats]:
    """Execute placement candidates and retry only after real CUDA OOM."""

    def execute(plan: MaterializationPlacementPlan) -> IntervalState:
        return run_crown_ibp_mlp(
            module,
            input_spec,
            linear_spec_C=linear_spec_C,
            output_value=output_value,
            relu_split_state=relu_split_state,
            materialization_placement_plan=plan,
        )

    if memory_budget_bytes is not None:
        return execute_bounded_placement_candidates_with_retry(
            placement_plans,
            execute,
            memory_budget_bytes=int(memory_budget_bytes),
            max_attempts=int(max_attempts),
            prediction_budget_factor=float(prediction_budget_factor),
        )
    return execute_placement_candidates_with_retry(placement_plans, execute)


def get_crown_ibp_mlp_stats(module: BFTaskModule) -> CrownIbpStats:
    try:
        module.validate()
        task = module.get_entry_task()
        if task.kind != TaskKind.INTERVAL_IBP:
            return CrownIbpStats(
                supported=False, reason=f"TaskKind={task.kind}", ops_seen=tuple()
            )
        if module.task_graph is not None or len(module.tasks) != 1:
            return CrownIbpStats(
                supported=False,
                reason="multi-task module not supported",
                ops_seen=tuple(),
            )
        ops = tuple(op.op_type for op in task.ops)
        if not task.ops:
            return CrownIbpStats(supported=False, reason="empty task", ops_seen=ops)
        bad = [
            t
            for t in ops
            if t
            not in {"linear", "relu", "conv2d", "flatten", "reshape", "add", "concat"}
        ]
        if bad:
            return CrownIbpStats(
                supported=False, reason=f"unsupported ops: {bad}", ops_seen=ops
            )
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
            if len(op.outputs) != 1:
                return CrownIbpStats(
                    supported=False,
                    reason=f"multi-output op not supported at op {i}: outputs={op.outputs}",
                    ops_seen=ops,
                )
            if op.op_type == "add" and len(op.inputs) != 2:
                return CrownIbpStats(
                    supported=False,
                    reason=f"add expects 2 inputs at op {i}, got {len(op.inputs)}",
                    ops_seen=ops,
                )
            if op.op_type == "concat":
                if len(op.inputs) < 2:
                    return CrownIbpStats(
                        supported=False,
                        reason=f"concat expects at least 2 inputs at op {i}, got {len(op.inputs)}",
                        ops_seen=ops,
                    )
                try:
                    normalize_concat_axis(
                        op.attrs.get("axis", 1),
                        rank_with_batch=2,
                        caller="get_crown_ibp_mlp_stats",
                    )
                except NotImplementedError:
                    return CrownIbpStats(
                        supported=False,
                        reason=f"unsupported concat axis at op {i}: {op.attrs.get('axis', 1)}",
                        ops_seen=ops,
                    )
        return CrownIbpStats(supported=True, ops_seen=ops)
    except Exception as e:  # pragma: no cover
        return CrownIbpStats(supported=False, reason=str(e), ops_seen=tuple())
