"""Reference abstract and concrete executors for legacy BFTaskModule programs."""

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-locals,too-many-branches
# pylint: disable=too-many-statements,invalid-name,line-too-long,not-callable

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Union

import torch
import torch.nn.functional as F

from ..domains.interval import IntervalDomain, IntervalState
from ..ir.task import BFTaskModule, TaskKind
from ..ir.task import StoragePlan
from .dag_utils import normalize_concat_axis, validate_concat_tensor_shapes
from .perturbation import BoxPerturbation
from .perturbation import InputPerturbationState, LpBallPerturbation, PerturbationSet


@dataclass(frozen=True)
class LinfInputSpec:
    value_name: str
    center: torch.Tensor
    eps: float


@dataclass(frozen=True)
class InputSpec:
    value_name: str
    center: torch.Tensor
    perturbation: PerturbationSet

    @staticmethod
    def linf(*, value_name: str, center: torch.Tensor, eps: float) -> "InputSpec":
        return InputSpec(
            value_name=value_name,
            center=center,
            perturbation=LpBallPerturbation(p="inf", eps=eps),
        )

    @staticmethod
    def l2(*, value_name: str, center: torch.Tensor, eps: float) -> "InputSpec":
        return InputSpec(
            value_name=value_name,
            center=center,
            perturbation=LpBallPerturbation(p=2, eps=eps),
        )

    @staticmethod
    def l1(*, value_name: str, center: torch.Tensor, eps: float) -> "InputSpec":
        return InputSpec(
            value_name=value_name,
            center=center,
            perturbation=LpBallPerturbation(p=1, eps=eps),
        )

    @staticmethod
    def box(
        *, value_name: str, lower: torch.Tensor, upper: torch.Tensor
    ) -> "InputSpec":
        """Build an exact per-element box input specification."""

        return InputSpec(
            value_name=value_name,
            center=(lower + upper) / 2.0,
            perturbation=BoxPerturbation(lower=lower, upper=upper),
        )


InputSpecLike = Union[LinfInputSpec, InputSpec]


@dataclass(frozen=True)
class ConcreteTaskExecution:
    """Deterministic primal Task IR result with intermediate value tensors."""

    output_value: str
    output: torch.Tensor
    values: tuple[tuple[str, torch.Tensor], ...]
    gradients_preserved: bool = False

    def validate(self) -> None:
        value_map = dict(self.values)
        if (
            not self.output_value
            or len(value_map) != len(self.values)
            or self.output_value not in value_map
            or not torch.equal(self.output, value_map[self.output_value])
        ):
            raise ValueError("concrete Task IR execution result is invalid")
        if any(
            not name
            or not torch.is_tensor(value)
            or not bool(torch.isfinite(value).all())
            for name, value in self.values
        ):
            raise ValueError("concrete Task IR execution values are invalid")

    def value_map(self) -> dict[str, torch.Tensor]:
        self.validate()
        return dict(self.values)


class TaskExecutor(Protocol):
    def run_ibp(
        self,
        module: BFTaskModule,
        input_spec: InputSpecLike,
        *,
        output_value: Optional[str] = None,
    ) -> IntervalState: ...


class PythonTaskExecutor:
    """
    执行 BFTaskModule 的 reference executor。
    v0.1 仅支持 TaskKind.INTERVAL_IBP，并覆盖少量 primitive（linear/conv2d/relu/add/mul/flatten/reshape）。
    """

    def __init__(self, domain: Optional[IntervalDomain] = None):
        self.domain = domain or IntervalDomain()

    def run_ibp(
        self,
        module: BFTaskModule,
        input_spec: InputSpecLike,
        *,
        output_value: Optional[str] = None,
    ) -> IntervalState:
        module.validate()
        task = module.get_entry_task()
        if task.kind != TaskKind.INTERVAL_IBP:
            raise NotImplementedError(
                f"PythonTaskExecutor only supports INTERVAL_IBP, got {task.kind}"
            )

        input_spec = _normalize_input_spec(input_spec)
        if input_spec.value_name not in task.input_values:
            raise ValueError(
                f"input_spec.value_name '{input_spec.value_name}' not in task.input_values: {task.input_values}"
            )

        params: Dict[str, Any] = {}
        raw_params = module.bindings.get("params", {})
        if isinstance(raw_params, dict):
            params.update(raw_params)

        x0 = input_spec.center
        env: Dict[str, Any] = {
            input_spec.value_name: InputPerturbationState(
                center=x0, perturbation=input_spec.perturbation
            )
        }

        def get_state(value_name: str) -> Any:
            if value_name in env:
                return env[value_name]
            if value_name in params:
                t = params[value_name]
                if not torch.is_tensor(t):
                    t = torch.as_tensor(t, device=x0.device)
                return IntervalState(lower=t, upper=t)
            raise KeyError(f"missing value in env/params: {value_name}")

        def ensure_interval(state: Any) -> IntervalState:
            if isinstance(state, IntervalState):
                return state
            if isinstance(state, InputPerturbationState):
                lb, ub = state.perturbation.bounding_box(state.center)
                return IntervalState(lower=lb, upper=ub)
            raise TypeError(
                f"expected IntervalState or InputPerturbationState, got {type(state)}"
            )

        def get_tensor(value_name: str) -> Any:
            if value_name in params:
                return params[value_name]
            raise KeyError(f"missing param tensor: {value_name}")

        for op in task.ops:
            if op.op_type == "spec_linear":
                # y = C @ logits, where C shape [B,S,O], logits shape [B,O]
                logits = ensure_interval(get_state(op.inputs[0]))
                C = get_tensor(op.inputs[1])
                if not torch.is_tensor(C):
                    C = torch.as_tensor(C, device=x0.device)
                if C.dim() != 3:
                    raise ValueError(
                        f"spec_linear expects C rank-3 [B,S,O], got {tuple(C.shape)}"
                    )
                if logits.lower.dim() != 2:
                    raise ValueError(
                        f"spec_linear expects logits rank-2 [B,O], got {tuple(logits.lower.shape)}"
                    )
                if (
                    C.shape[0] != logits.lower.shape[0]
                    or C.shape[2] != logits.lower.shape[1]
                ):
                    raise ValueError(
                        f"spec_linear shape mismatch: C={tuple(C.shape)} logits={tuple(logits.lower.shape)}"
                    )
                C_pos = torch.clamp(C, min=0.0)
                C_neg = torch.clamp(C, max=0.0)
                l = logits.lower.unsqueeze(1)
                u = logits.upper.unsqueeze(1)
                lb = (C_pos * l + C_neg * u).sum(dim=-1)
                ub = (C_pos * u + C_neg * l).sum(dim=-1)
                env[op.outputs[0]] = IntervalState(lower=lb, upper=ub)  # type: ignore[assignment]
                continue

            if op.op_type == "linear":
                x_state = get_state(op.inputs[0])
                w = get_tensor(op.inputs[1])
                b = get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None
                if isinstance(x_state, InputPerturbationState):
                    if not torch.is_tensor(w):
                        w = torch.as_tensor(w, device=x0.device)
                    if b is not None and not torch.is_tensor(b):
                        b = torch.as_tensor(b, device=x0.device)
                    lb, ub = x_state.perturbation.concretize_matmul(
                        center=x_state.center, weight=w, bias=b
                    )
                    env[op.outputs[0]] = IntervalState(lower=lb, upper=ub)  # type: ignore[assignment]
                    continue

                x = ensure_interval(x_state)
                if torch.is_tensor(w) and w.dim() == 3:
                    # Batched linear: w shape [B, O, I], b shape [B, O] (optional)
                    if b is None:
                        b = 0.0
                    if not torch.is_tensor(b):
                        b = torch.as_tensor(b, device=x0.device)
                    mid = (x.lower + x.upper) / 2.0
                    diff = (x.upper - x.lower) / 2.0
                    center = torch.bmm(mid.unsqueeze(1), w.transpose(-1, -2)).squeeze(1)
                    deviation = torch.bmm(
                        diff.unsqueeze(1), w.abs().transpose(-1, -2)
                    ).squeeze(1)
                    lb = center - deviation + b
                    ub = center + deviation + b
                    env[op.outputs[0]] = IntervalState(lower=lb, upper=ub)  # type: ignore[assignment]
                else:
                    out = self.domain.affine_transformer(x, w, b, op="linear")
                    env[op.outputs[0]] = out  # type: ignore[assignment]
                continue

            if op.op_type == "conv2d":
                x = ensure_interval(get_state(op.inputs[0]))
                w = get_tensor(op.inputs[1])
                b = get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None
                attrs = dict(op.attrs)
                attrs.setdefault("op", "conv2d")
                out = self.domain.affine_transformer(x, w, b, **attrs)
                env[op.outputs[0]] = out  # type: ignore[assignment]
                continue

            if op.op_type == "relu":
                x = ensure_interval(get_state(op.inputs[0]))
                env[op.outputs[0]] = self.domain.relu_transformer(x)  # type: ignore[assignment]
                continue

            if op.op_type == "add":
                a = ensure_interval(get_state(op.inputs[0]))
                b = ensure_interval(get_state(op.inputs[1]))
                if tuple(a.lower.shape) != tuple(b.lower.shape) or tuple(
                    a.upper.shape
                ) != tuple(b.upper.shape):
                    raise NotImplementedError(
                        "PythonTaskExecutor only supports add with exact same-shape inputs; "
                        f"got {tuple(a.lower.shape)} and {tuple(b.lower.shape)}"
                    )
                env[op.outputs[0]] = IntervalState(  # type: ignore[assignment]
                    lower=a.lower + b.lower,
                    upper=a.upper + b.upper,
                )
                continue

            if op.op_type == "mul":
                a = ensure_interval(get_state(op.inputs[0]))
                b = ensure_interval(get_state(op.inputs[1]))
                env[op.outputs[0]] = self.domain.elementwise_transformer(  # type: ignore[assignment]
                    [a, b], op=op.op_type
                )
                continue

            if op.op_type == "concat":
                if len(op.inputs) < 2:
                    raise ValueError(
                        f"concat expects at least 2 inputs, got {len(op.inputs)}"
                    )
                parts = [ensure_interval(get_state(name)) for name in op.inputs]
                axis = normalize_concat_axis(
                    op.attrs.get("axis", 1),
                    rank_with_batch=int(parts[0].lower.dim()),
                    caller="PythonTaskExecutor.run_ibp",
                )
                _ = validate_concat_tensor_shapes(
                    [tuple(int(dim) for dim in part.lower.shape) for part in parts],
                    axis=axis,
                    caller="PythonTaskExecutor only supports concat",
                )
                env[op.outputs[0]] = IntervalState(  # type: ignore[assignment]
                    lower=torch.cat([part.lower for part in parts], dim=axis),
                    upper=torch.cat([part.upper for part in parts], dim=axis),
                )
                continue

            if op.op_type == "flatten":
                x = ensure_interval(get_state(op.inputs[0]))
                start_dim = int(op.attrs.get("start_dim", 0))
                end_dim = int(op.attrs.get("end_dim", -1))
                env[op.outputs[0]] = IntervalState(  # type: ignore[assignment]
                    lower=torch.flatten(x.lower, start_dim=start_dim, end_dim=end_dim),
                    upper=torch.flatten(x.upper, start_dim=start_dim, end_dim=end_dim),
                )
                continue

            if op.op_type == "reshape":
                x = ensure_interval(get_state(op.inputs[0]))
                shape = op.attrs.get("shape")
                if shape is None:
                    env[op.outputs[0]] = x
                    continue
                env[op.outputs[0]] = IntervalState(  # type: ignore[assignment]
                    lower=x.lower.reshape(shape),
                    upper=x.upper.reshape(shape),
                )
                continue

            if op.op_type in ("permute", "transpose"):
                x = ensure_interval(get_state(op.inputs[0]))
                dims = op.attrs.get("dims")
                if not isinstance(dims, (list, tuple)):
                    raise ValueError(
                        f"transpose missing dims for op '{op.name}': {dims}"
                    )
                dims = [int(d) for d in dims]
                env[op.outputs[0]] = IntervalState(  # type: ignore[assignment]
                    lower=x.lower.permute(*dims),
                    upper=x.upper.permute(*dims),
                )
                continue

            raise NotImplementedError(
                f"unsupported op_type in task executor: {op.op_type}"
            )

        if output_value is None:
            if len(task.output_values) != 1:
                raise ValueError(
                    f"task has {len(task.output_values)} outputs; specify output_value explicitly"
                )
            output_value = task.output_values[0]
        return ensure_interval(get_state(output_value))

    def run_ibp_task(
        self,
        task,
        *,
        env: Dict[str, IntervalState],
        params: Dict[str, Any],
        storage_plan: StoragePlan,
    ) -> None:
        """
        Execute a single INTERVAL_IBP task in-place on a shared env.

        This is the building block for Phase 5 TaskGraph scheduling.
        """
        if task.kind != TaskKind.INTERVAL_IBP:
            raise NotImplementedError(
                f"PythonTaskExecutor only supports INTERVAL_IBP, got {task.kind}"
            )

        # BufferEnv: env maps buffer_id -> IntervalState.
        # Pick a device anchor from existing env or any param tensor.
        device = None
        for v in env.values():
            device = v.lower.device
            break
        if device is None:
            for p in params.values():
                if torch.is_tensor(p):
                    device = p.device
                    break

        def _buf(value_name: str) -> str:
            logical = storage_plan.value_to_buffer.get(value_name)
            if logical is None:
                raise KeyError(f"value not found in storage_plan: {value_name}")
            phys = storage_plan.to_physical(logical)
            if (
                storage_plan.physical_buffers
                and phys not in storage_plan.physical_buffers
            ):
                raise KeyError(
                    f"physical buffer_id not found in storage_plan.physical_buffers: {phys} (value={value_name})"
                )
            return phys

        def get_interval(value_name: str) -> IntervalState:
            bid = _buf(value_name)
            if bid in env:
                return env[bid]
            if value_name in params:
                t = params[value_name]
                if not torch.is_tensor(t):
                    t = torch.as_tensor(t, device=device)
                s = IntervalState(lower=t, upper=t)
                env[bid] = s
                return s
            raise KeyError(f"missing value in env/params: {value_name}")

        def get_tensor(value_name: str) -> Any:
            if value_name in params:
                return params[value_name]
            raise KeyError(f"missing param tensor: {value_name}")

        for op in task.ops:
            if op.op_type == "spec_linear":
                logits = get_interval(op.inputs[0])
                C = get_tensor(op.inputs[1])
                if not torch.is_tensor(C):
                    C = torch.as_tensor(C, device=device)
                if C.dim() != 3:
                    raise ValueError(
                        f"spec_linear expects C rank-3 [B,S,O], got {tuple(C.shape)}"
                    )
                C_pos = torch.clamp(C, min=0.0)
                C_neg = torch.clamp(C, max=0.0)
                l = logits.lower.unsqueeze(1)
                u = logits.upper.unsqueeze(1)
                lb = (C_pos * l + C_neg * u).sum(dim=-1)
                ub = (C_pos * u + C_neg * l).sum(dim=-1)
                env[_buf(op.outputs[0])] = IntervalState(lower=lb, upper=ub)  # type: ignore[assignment]
                continue

            if op.op_type == "linear":
                x = get_interval(op.inputs[0])
                w = get_tensor(op.inputs[1])
                b = get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None
                if torch.is_tensor(w) and w.dim() == 3:
                    if b is None:
                        b = 0.0
                    if not torch.is_tensor(b):
                        b = torch.as_tensor(b, device=device)
                    mid = (x.lower + x.upper) / 2.0
                    diff = (x.upper - x.lower) / 2.0
                    center = torch.bmm(mid.unsqueeze(1), w.transpose(-1, -2)).squeeze(1)
                    deviation = torch.bmm(
                        diff.unsqueeze(1), w.abs().transpose(-1, -2)
                    ).squeeze(1)
                    lb = center - deviation + b
                    ub = center + deviation + b
                    env[_buf(op.outputs[0])] = IntervalState(lower=lb, upper=ub)  # type: ignore[assignment]
                else:
                    out = self.domain.affine_transformer(x, w, b, op="linear")
                    env[_buf(op.outputs[0])] = out  # type: ignore[assignment]
                continue

            if op.op_type == "conv2d":
                x = get_interval(op.inputs[0])
                w = get_tensor(op.inputs[1])
                b = get_tensor(op.inputs[2]) if len(op.inputs) == 3 else None
                attrs = dict(op.attrs)
                attrs.setdefault("op", "conv2d")
                out = self.domain.affine_transformer(x, w, b, **attrs)
                env[_buf(op.outputs[0])] = out  # type: ignore[assignment]
                continue

            if op.op_type == "relu":
                x = get_interval(op.inputs[0])
                env[_buf(op.outputs[0])] = self.domain.relu_transformer(x)  # type: ignore[assignment]
                continue

            if op.op_type == "add":
                a = get_interval(op.inputs[0])
                b = get_interval(op.inputs[1])
                if tuple(a.lower.shape) != tuple(b.lower.shape) or tuple(
                    a.upper.shape
                ) != tuple(b.upper.shape):
                    raise NotImplementedError(
                        "PythonTaskExecutor only supports add with exact same-shape inputs; "
                        f"got {tuple(a.lower.shape)} and {tuple(b.lower.shape)}"
                    )
                env[_buf(op.outputs[0])] = IntervalState(  # type: ignore[assignment]
                    lower=a.lower + b.lower,
                    upper=a.upper + b.upper,
                )
                continue

            if op.op_type == "mul":
                a = get_interval(op.inputs[0])
                b = get_interval(op.inputs[1])
                env[_buf(op.outputs[0])] = self.domain.elementwise_transformer(  # type: ignore[assignment]
                    [a, b], op=op.op_type
                )
                continue

            if op.op_type == "concat":
                if len(op.inputs) < 2:
                    raise ValueError(
                        f"concat expects at least 2 inputs, got {len(op.inputs)}"
                    )
                parts = [get_interval(name) for name in op.inputs]
                axis = normalize_concat_axis(
                    op.attrs.get("axis", 1),
                    rank_with_batch=int(parts[0].lower.dim()),
                    caller="PythonTaskExecutor.run_ibp_task",
                )
                _ = validate_concat_tensor_shapes(
                    [tuple(int(dim) for dim in part.lower.shape) for part in parts],
                    axis=axis,
                    caller="PythonTaskExecutor only supports concat",
                )
                env[_buf(op.outputs[0])] = IntervalState(  # type: ignore[assignment]
                    lower=torch.cat([part.lower for part in parts], dim=axis),
                    upper=torch.cat([part.upper for part in parts], dim=axis),
                )
                continue

            if op.op_type == "flatten":
                x = get_interval(op.inputs[0])
                start_dim = int(op.attrs.get("start_dim", 0))
                end_dim = int(op.attrs.get("end_dim", -1))
                env[_buf(op.outputs[0])] = IntervalState(  # type: ignore[assignment]
                    lower=torch.flatten(x.lower, start_dim=start_dim, end_dim=end_dim),
                    upper=torch.flatten(x.upper, start_dim=start_dim, end_dim=end_dim),
                )
                continue

            if op.op_type == "reshape":
                x = get_interval(op.inputs[0])
                shape = op.attrs.get("shape")
                if shape is None:
                    env[_buf(op.outputs[0])] = x
                    continue
                env[_buf(op.outputs[0])] = IntervalState(  # type: ignore[assignment]
                    lower=x.lower.reshape(shape),
                    upper=x.upper.reshape(shape),
                )
                continue

            if op.op_type in ("permute", "transpose"):
                x = get_interval(op.inputs[0])
                dims = op.attrs.get("dims")
                if not isinstance(dims, (list, tuple)):
                    raise ValueError(
                        f"transpose missing dims for op '{op.name}': {dims}"
                    )
                dims = [int(d) for d in dims]
                env[_buf(op.outputs[0])] = IntervalState(  # type: ignore[assignment]
                    lower=x.lower.permute(*dims),
                    upper=x.upper.permute(*dims),
                )
                continue

            raise NotImplementedError(
                f"unsupported op_type in task executor: {op.op_type}"
            )


def _normalize_input_spec(spec: InputSpecLike) -> InputSpec:
    if isinstance(spec, InputSpec):
        return spec
    return InputSpec.linf(
        value_name=spec.value_name, center=spec.center, eps=float(spec.eps)
    )


def execute_task_module_concrete(
    module: BFTaskModule,
    input_value: torch.Tensor,
    *,
    input_value_name: Optional[str] = None,
    output_value: Optional[str] = None,
    preserve_gradients: bool = False,
) -> ConcreteTaskExecution:
    """Execute the entry task as a concrete primal program.

    This path intentionally does not consume abstract bounds.  It is used for
    independently replaying property witnesses against the primal Task IR.
    Candidate search may opt into autograd without changing operator semantics.
    """

    module.validate()
    task = module.get_entry_task()
    if (
        not torch.is_tensor(input_value)
        or not torch.is_floating_point(input_value)
        or not bool(torch.isfinite(input_value).all())
    ):
        raise ValueError("concrete Task IR input must be a finite floating tensor")
    if input_value_name is None:
        if len(task.input_values) != 1:
            raise ValueError("concrete Task IR requires an explicit input value name")
        input_value_name = task.input_values[0]
    if input_value_name not in task.input_values:
        raise ValueError("concrete Task IR input value is not an entry-task input")

    raw_params = module.bindings.get("params", {})
    if not isinstance(raw_params, dict):
        raise TypeError("concrete Task IR parameter binding must be a mapping")
    params: Dict[str, Any] = dict(raw_params)
    env: Dict[str, torch.Tensor] = {
        input_value_name: (
            input_value.contiguous()
            if preserve_gradients
            else input_value.detach().contiguous().clone()
        )
    }

    def get_value(name: str) -> torch.Tensor:
        if name in env:
            return env[name]
        if name not in params:
            raise KeyError(f"missing concrete Task IR value: {name}")
        value = params[name]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=input_value.device)
        if value.device != input_value.device:
            raise ValueError("concrete Task IR parameter device differs from input")
        return value

    def bind(op_name: str, outputs: list[str], value: torch.Tensor) -> None:
        if len(outputs) != 1 or not outputs[0] or outputs[0] in env:
            raise ValueError(
                f"concrete Task IR op '{op_name}' must define one fresh output"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(
                f"concrete Task IR op '{op_name}' produced non-finite data"
            )
        env[outputs[0]] = value

    with nullcontext() if preserve_gradients else torch.no_grad():
        for op in task.ops:
            if op.op_type == "spec_linear":
                logits = get_value(op.inputs[0])
                objective = get_value(op.inputs[1])
                if (
                    objective.dim() != 3
                    or logits.dim() != 2
                    or (
                        int(objective.shape[0]) != int(logits.shape[0])
                        or int(objective.shape[2]) != int(logits.shape[1])
                    )
                ):
                    raise ValueError("concrete spec_linear shape mismatch")
                bind(
                    op.name,
                    op.outputs,
                    torch.einsum("bso,bo->bs", objective, logits),
                )
                continue

            if op.op_type == "linear":
                value = get_value(op.inputs[0])
                weight = get_value(op.inputs[1])
                bias = get_value(op.inputs[2]) if len(op.inputs) == 3 else None
                if weight.dim() == 2:
                    result = F.linear(value, weight, bias)
                elif weight.dim() == 3 and value.dim() == 2:
                    result = torch.bmm(weight, value.unsqueeze(-1)).squeeze(-1)
                    if bias is not None:
                        result = result + bias
                else:
                    raise ValueError("concrete linear shape is unsupported")
                bind(op.name, op.outputs, result)
                continue

            if op.op_type == "conv2d":
                value = get_value(op.inputs[0])
                weight = get_value(op.inputs[1])
                bias = get_value(op.inputs[2]) if len(op.inputs) == 3 else None
                bind(
                    op.name,
                    op.outputs,
                    F.conv2d(
                        value,
                        weight,
                        bias=bias,
                        stride=op.attrs.get("stride", 1),
                        padding=op.attrs.get("padding", 0),
                        dilation=op.attrs.get("dilation", 1),
                        groups=int(op.attrs.get("groups", 1)),
                    ),
                )
                continue

            if op.op_type == "relu":
                bind(op.name, op.outputs, torch.relu(get_value(op.inputs[0])))
                continue

            if op.op_type == "add":
                bind(
                    op.name,
                    op.outputs,
                    get_value(op.inputs[0]) + get_value(op.inputs[1]),
                )
                continue

            if op.op_type == "mul":
                bind(
                    op.name,
                    op.outputs,
                    get_value(op.inputs[0]) * get_value(op.inputs[1]),
                )
                continue

            if op.op_type == "concat":
                parts = [get_value(name) for name in op.inputs]
                if len(parts) < 2:
                    raise ValueError("concrete concat requires at least two inputs")
                axis = normalize_concat_axis(
                    op.attrs.get("axis", 1),
                    rank_with_batch=int(parts[0].dim()),
                    caller="execute_task_module_concrete",
                )
                _ = validate_concat_tensor_shapes(
                    [tuple(int(dim) for dim in part.shape) for part in parts],
                    axis=axis,
                    caller="execute_task_module_concrete",
                )
                bind(op.name, op.outputs, torch.cat(parts, dim=axis))
                continue

            if op.op_type == "flatten":
                bind(
                    op.name,
                    op.outputs,
                    torch.flatten(
                        get_value(op.inputs[0]),
                        start_dim=int(op.attrs.get("start_dim", 0)),
                        end_dim=int(op.attrs.get("end_dim", -1)),
                    ),
                )
                continue

            if op.op_type == "reshape":
                shape = op.attrs.get("shape")
                if shape is None:
                    raise ValueError("concrete reshape requires attrs['shape']")
                bind(op.name, op.outputs, get_value(op.inputs[0]).reshape(shape))
                continue

            if op.op_type in {"permute", "transpose"}:
                dims = op.attrs.get("dims")
                if not isinstance(dims, (list, tuple)):
                    raise ValueError("concrete transpose requires attrs['dims']")
                bind(
                    op.name,
                    op.outputs,
                    get_value(op.inputs[0]).permute(*(int(dim) for dim in dims)),
                )
                continue

            raise NotImplementedError(
                f"unsupported concrete Task IR op_type: {op.op_type}"
            )

    if output_value is None:
        if len(task.output_values) != 1:
            raise ValueError("concrete Task IR requires an explicit output value")
        output_value = task.output_values[0]
    if output_value not in env:
        raise KeyError(f"concrete Task IR output is unavailable: {output_value}")

    def result_value(value: torch.Tensor) -> torch.Tensor:
        if preserve_gradients:
            return value
        return value.detach().contiguous().clone()

    execution = ConcreteTaskExecution(
        output_value=output_value,
        output=result_value(env[output_value]),
        values=tuple((name, result_value(value)) for name, value in env.items()),
        gradients_preserved=preserve_gradients,
    )
    execution.validate()
    return execution
