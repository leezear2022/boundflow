"""Lower the existing plain-CROWN task subset into first-class Bound IR."""

# pylint: disable=invalid-name

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Optional, Sequence

import torch

from ..domains.interval import IntervalState
from ..ir.bound import (
    AddBackwardAttrs,
    BFBoundGraph,
    BFBoundModule,
    BatchAxisKind,
    BoundAffineStateRef,
    BoundBatchAxis,
    BoundDomainConfig,
    BoundMethodKind,
    BoundOp,
    BoundOpKind,
    BoundPolarity,
    BoundRepresentation,
    BoundTensorType,
    BoundValue,
    BoundValueRole,
    ConcatBackwardAttrs,
    ConcretizeAttrs,
    Conv2dBackwardAttrs,
    LinearBackwardAttrs,
    NoBoundOpAttrs,
    ObjectiveKind,
    ObjectiveSpec,
    PerturbationKind,
    PerturbationSpec,
    IntermediateBoundSource,
    ReluLowerSlopePolicy,
    ReluRelaxationAttrs,
    ReshapeAttrs,
    SpecBindAttrs,
    VerificationSpec,
)
from ..ir.task import BFTaskModule, TaskKind, TaskOp
from ..runtime.perturbation import BoxPerturbation, LpBallPerturbation
from ..runtime.task_executor import InputSpec


@dataclass(frozen=True)
class PlainCrownBoundIRBuild:
    """Bound IR plus stable IDs needed to bind its runtime query payload."""

    module: BFBoundModule
    objective_input_value_id: str
    input_affine_state: BoundAffineStateRef
    lower_output_value_id: str
    upper_output_value_id: str


class _GraphBuilder:
    """Small deterministic SSA builder local to the lowering adapter."""

    def __init__(self, *, batch: int, specs: int, dtype: str, device: str) -> None:
        self.batch = batch
        self.specs = specs
        self.dtype = dtype
        self.device = device
        self.values: list[BoundValue] = []
        self.ops: list[BoundOp] = []
        self._serial = 0

    def _next_prefix(self, label: str) -> str:
        """Allocate one deterministic SSA/op prefix."""

        self._serial += 1
        safe = "".join(
            char if char.isalnum() or char in "._-" else "_" for char in label
        )
        return f"{self._serial:04d}.{safe}"

    def coefficient_type(self, primal_shape: Sequence[int]) -> BoundTensorType:
        """Build the dense coefficient type for one primal value."""

        return BoundTensorType(
            shape=(self.batch, self.specs, *(int(dim) for dim in primal_shape)),
            dtype=self.dtype,
            layout="contiguous",
            device=self.device,
            batch_axes=(
                BoundBatchAxis(BatchAxisKind.DOMAIN, 0),
                BoundBatchAxis(BatchAxisKind.SPEC, 1),
            ),
        )

    def bias_type(self) -> BoundTensorType:
        """Build the shared affine-bias type."""

        return BoundTensorType(
            shape=(self.batch, self.specs),
            dtype=self.dtype,
            layout="contiguous",
            device=self.device,
            batch_axes=(
                BoundBatchAxis(BatchAxisKind.DOMAIN, 0),
                BoundBatchAxis(BatchAxisKind.SPEC, 1),
            ),
        )

    def add_objective_input(
        self, *, value_id: str, primal_value_id: str, primal_shape: Sequence[int]
    ) -> BoundValue:
        """Create the query objective graph input."""

        value = BoundValue(
            value_id=value_id,
            tensor_type=self.coefficient_type(primal_shape),
            role=BoundValueRole.OBJECTIVE,
            polarity=BoundPolarity.BOTH,
            representation=BoundRepresentation.DENSE,
            state_version="plain-crown-v1",
            source_primal_value_id=primal_value_id,
        )
        self.values.append(value)
        return value

    def add_state(
        self,
        *,
        label: str,
        primal_value_id: str,
        primal_shape: Sequence[int],
        state_version: str = "plain-crown-v1",
    ) -> BoundAffineStateRef:
        """Create one four-component affine SSA state."""

        prefix = self._next_prefix(label)
        coefficient_type = self.coefficient_type(primal_shape)
        bias_type = self.bias_type()
        components = (
            ("A_u", coefficient_type, BoundValueRole.COEFFICIENT, BoundPolarity.UPPER),
            ("b_u", bias_type, BoundValueRole.BIAS, BoundPolarity.UPPER),
            ("A_l", coefficient_type, BoundValueRole.COEFFICIENT, BoundPolarity.LOWER),
            ("b_l", bias_type, BoundValueRole.BIAS, BoundPolarity.LOWER),
        )
        created: list[BoundValue] = []
        for suffix, tensor_type, role, polarity in components:
            created.append(
                BoundValue(
                    value_id=f"{prefix}.{suffix}",
                    tensor_type=tensor_type,
                    role=role,
                    polarity=polarity,
                    representation=BoundRepresentation.DENSE,
                    state_version=state_version,
                    source_primal_value_id=primal_value_id,
                )
            )
        self.values.extend(created)
        return BoundAffineStateRef(
            upper_coefficient=created[0].value_id,
            upper_bias=created[1].value_id,
            lower_coefficient=created[2].value_id,
            lower_bias=created[3].value_id,
        )

    def add_result(self, *, label: str, polarity: BoundPolarity) -> BoundValue:
        """Create one concretized objective result."""

        prefix = self._next_prefix(label)
        value = BoundValue(
            value_id=f"{prefix}.objective",
            tensor_type=self.bias_type(),
            role=BoundValueRole.OBJECTIVE,
            polarity=polarity,
            representation=BoundRepresentation.DENSE,
            state_version="plain-crown-v1",
        )
        self.values.append(value)
        return value

    def add_op(  # pylint: disable=too-many-arguments
        self,
        *,
        label: str,
        kind: BoundOpKind,
        inputs: Sequence[str],
        outputs: Sequence[str],
        attrs: object,
    ) -> None:
        """Append one BoundOp with a deterministic ID."""

        self.ops.append(
            BoundOp(
                op_id=self._next_prefix(label),
                kind=kind,
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                attrs=attrs,  # type: ignore[arg-type]
            )
        )


def build_plain_crown_bound_ir(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    task_module: BFTaskModule,
    input_spec: InputSpec,
    *,
    interval_env: Mapping[str, IntervalState],
    relu_pre: Mapping[str, IntervalState],
    linear_spec_C: Optional[torch.Tensor] = None,
    output_value: Optional[str] = None,
    intermediate_bound_source: IntermediateBoundSource = (
        IntermediateBoundSource.LOCAL_FORWARD
    ),
    intermediate_bounds_hash: Optional[str] = None,
    relu_lower_slope_policy: ReluLowerSlopePolicy = ReluLowerSlopePolicy.ZERO,
) -> PlainCrownBoundIRBuild:
    """Lower a validated plain-CROWN query shape into deterministic Bound IR."""

    task_module.validate()
    if not isinstance(intermediate_bound_source, IntermediateBoundSource):
        raise TypeError("intermediate_bound_source must be an IntermediateBoundSource")
    if not isinstance(relu_lower_slope_policy, ReluLowerSlopePolicy):
        raise TypeError("relu_lower_slope_policy must be a ReluLowerSlopePolicy")
    if intermediate_bound_source == IntermediateBoundSource.EXTERNAL_VERIFIER:
        if not _is_sha256(intermediate_bounds_hash):
            raise ValueError(
                "external intermediate bounds require an exact SHA-256 identity"
            )
    elif intermediate_bounds_hash is not None:
        raise ValueError("local intermediate bounds cannot declare an external hash")
    if len(task_module.tasks) != 1 or task_module.task_graph is not None:
        raise NotImplementedError("Bound IR v1 lowering supports one task only")
    task = task_module.get_entry_task()
    if task.kind != TaskKind.INTERVAL_IBP:
        raise NotImplementedError(
            f"Bound IR v1 lowering expects INTERVAL_IBP, got {task.kind}"
        )
    if not task.ops:
        raise ValueError("Bound IR v1 lowering expects a non-empty task")
    if input_spec.value_name not in task.input_values:
        raise ValueError("input specification does not bind a task input")

    resolved_output = _resolve_output(task.ops, task.output_values, output_value)
    output_state = _require_interval(interval_env, resolved_output)
    if output_state.lower.dim() != 2:
        raise ValueError(
            "Bound IR v1 plain-CROWN output must be rank-2 [domain, output]"
        )
    _validate_trace_tensor_pair(output_state, label=resolved_output)
    batch, output_dim = (int(dim) for dim in output_state.lower.shape)
    specs = _validate_objective(linear_spec_C, batch=batch, output_dim=output_dim)
    dtype = str(output_state.lower.dtype).removeprefix("torch.")
    device = str(output_state.lower.device)
    builder = _GraphBuilder(batch=batch, specs=specs, dtype=dtype, device=device)

    objective_id, objective_kind, objective_hash = _objective_identity(linear_spec_C)
    perturbation = _perturbation_spec(input_spec)
    verification_spec = VerificationSpec(
        perturbations=(perturbation,),
        objectives=(
            ObjectiveSpec(
                objective_id=objective_id,
                output_primal_value_id=resolved_output,
                kind=objective_kind,
                num_objectives=specs,
                payload_hash=objective_hash,
            ),
        ),
        requested_bounds=(BoundPolarity.BOTH,),
        numeric_policy=f"{dtype}_dense_reference",
    )

    objective_input = builder.add_objective_input(
        value_id="query.objective",
        primal_value_id=resolved_output,
        primal_shape=(output_dim,),
    )
    seed = builder.add_state(
        label="spec_seed", primal_value_id=resolved_output, primal_shape=(output_dim,)
    )
    builder.add_op(
        label="spec_bind",
        kind=BoundOpKind.SPEC_BIND,
        inputs=(objective_input.value_id,),
        outputs=seed.value_ids,
        attrs=SpecBindAttrs(primal_value_id=resolved_output, objective_id=objective_id),
    )

    params = _parameter_bindings(task_module)
    dynamic_names = set(interval_env)
    dynamic_names.add(input_spec.value_name)
    adjoints: dict[str, list[BoundAffineStateRef]] = {resolved_output: [seed]}

    def value_shape(value_id: str) -> tuple[int, ...]:
        if value_id == input_spec.value_name:
            shape = tuple(int(dim) for dim in input_spec.center.shape[1:])
            if not shape:
                raise ValueError("plain-CROWN input must include a feature dimension")
            return shape
        if value_id in params:
            tensor = _as_tensor(params[value_id])
            if tensor.dim() < 2:
                raise NotImplementedError(
                    f"constant '{value_id}' must carry an explicit domain batch axis"
                )
            return tuple(int(dim) for dim in tensor.shape[1:])
        state = _require_interval(interval_env, value_id)
        _validate_trace_tensor_pair(state, label=value_id)
        return tuple(int(dim) for dim in state.lower.shape[1:])

    def consume(value_id: str) -> Optional[BoundAffineStateRef]:
        contributions = adjoints.pop(value_id, None)
        if not contributions:
            return None
        if len(contributions) == 1:
            return contributions[0]
        merged = builder.add_state(
            label=f"compose_{value_id}",
            primal_value_id=value_id,
            primal_shape=value_shape(value_id),
        )
        builder.add_op(
            label=f"compose_{value_id}",
            kind=BoundOpKind.COEFFICIENT_COMPOSE,
            inputs=tuple(
                component
                for contribution in contributions
                for component in contribution.value_ids
            ),
            outputs=merged.value_ids,
            attrs=NoBoundOpAttrs(),
        )
        return merged

    def route(value_id: str, state: BoundAffineStateRef) -> None:
        adjoints.setdefault(value_id, []).append(state)

    for op in reversed(task.ops):
        if len(op.outputs) != 1:
            raise NotImplementedError(
                f"Bound IR v1 expects single-output TaskOps, got {op.outputs}"
            )
        source = consume(op.outputs[0])
        if source is None:
            continue
        out_name = op.outputs[0]

        if op.op_type == "linear":
            _require_param_inputs(op, params, minimum=2, maximum=3)
            target_name = op.inputs[0]
            target = builder.add_state(
                label=op.name,
                primal_value_id=target_name,
                primal_shape=value_shape(target_name),
            )
            builder.add_op(
                label=op.name,
                kind=BoundOpKind.LINEAR_BACKWARD,
                inputs=source.value_ids,
                outputs=target.value_ids,
                attrs=LinearBackwardAttrs(
                    primal_node_id=op.name,
                    weight_primal_value_id=op.inputs[1],
                    bias_primal_value_id=op.inputs[2] if len(op.inputs) == 3 else None,
                ),
            )
            route(target_name, target)
            continue

        if op.op_type == "conv2d":
            _require_param_inputs(op, params, minimum=2, maximum=3)
            target_name = op.inputs[0]
            target = builder.add_state(
                label=op.name,
                primal_value_id=target_name,
                primal_shape=value_shape(target_name),
            )
            builder.add_op(
                label=op.name,
                kind=BoundOpKind.CONV2D_BACKWARD,
                inputs=source.value_ids,
                outputs=target.value_ids,
                attrs=Conv2dBackwardAttrs(
                    primal_node_id=op.name,
                    weight_primal_value_id=op.inputs[1],
                    bias_primal_value_id=op.inputs[2] if len(op.inputs) == 3 else None,
                    stride=_as_pair(op.attrs.get("stride", 1), label="stride"),
                    padding=_as_pair(op.attrs.get("padding", 0), label="padding"),
                    dilation=_as_pair(op.attrs.get("dilation", 1), label="dilation"),
                    groups=int(op.attrs.get("groups", 1)),
                ),
            )
            route(target_name, target)
            continue

        if op.op_type == "relu":
            target_name = op.inputs[0]
            if target_name not in relu_pre:
                raise KeyError(f"missing ReLU pre-activation trace for '{target_name}'")
            _validate_trace_tensor_pair(relu_pre[target_name], label=target_name)
            target = builder.add_state(
                label=op.name,
                primal_value_id=target_name,
                primal_shape=value_shape(target_name),
                state_version=(
                    "plain-crown-v1"
                    if intermediate_bounds_hash is None
                    else f"external-intermediate-bounds:{intermediate_bounds_hash}"
                ),
            )
            builder.add_op(
                label=op.name,
                kind=BoundOpKind.RELU_RELAXATION,
                inputs=source.value_ids,
                outputs=target.value_ids,
                attrs=ReluRelaxationAttrs(
                    primal_node_id=op.name,
                    preactivation_primal_value_id=target_name,
                    intermediate_bound_source=intermediate_bound_source,
                    lower_slope_policy=relu_lower_slope_policy,
                ),
            )
            route(target_name, target)
            continue

        if op.op_type in {"flatten", "reshape"}:
            if op.op_type == "flatten" and (
                int(op.attrs.get("start_dim", 1)) != 1
                or int(op.attrs.get("end_dim", -1)) != -1
            ):
                raise NotImplementedError(
                    "Bound IR v1 only supports flatten(start_dim=1, end_dim=-1)"
                )
            target_name = op.inputs[0]
            target_shape = value_shape(target_name)
            target = builder.add_state(
                label=op.name,
                primal_value_id=target_name,
                primal_shape=target_shape,
            )
            builder.add_op(
                label=op.name,
                kind=BoundOpKind.RESHAPE,
                inputs=source.value_ids,
                outputs=target.value_ids,
                attrs=ReshapeAttrs(target_shape=target_shape),
            )
            route(target_name, target)
            continue

        if op.op_type == "add":
            out_shape = value_shape(out_name)
            dynamic_inputs = tuple(
                value_id for value_id in op.inputs if value_id in dynamic_names
            )
            constant_inputs = tuple(
                value_id for value_id in op.inputs if value_id not in dynamic_names
            )
            if not dynamic_inputs:
                raise ValueError(f"add '{op.name}' has no dynamic inputs")
            for value_id in op.inputs:
                if value_shape(value_id) != out_shape:
                    raise NotImplementedError(
                        "Bound IR v1 add requires exact same-shape inputs"
                    )
                if value_id in constant_inputs and value_id not in params:
                    raise KeyError(f"missing constant add parameter '{value_id}'")
            targets = tuple(
                builder.add_state(
                    label=f"{op.name}_{index}",
                    primal_value_id=value_id,
                    primal_shape=out_shape,
                )
                for index, value_id in enumerate(dynamic_inputs)
            )
            builder.add_op(
                label=op.name,
                kind=BoundOpKind.ADD_BACKWARD,
                inputs=source.value_ids,
                outputs=tuple(
                    component for target in targets for component in target.value_ids
                ),
                attrs=AddBackwardAttrs(
                    primal_node_id=op.name,
                    dynamic_input_primal_value_ids=dynamic_inputs,
                    constant_input_primal_value_ids=constant_inputs,
                ),
            )
            for value_id, target in zip(dynamic_inputs, targets):
                route(value_id, target)
            continue

        if op.op_type == "concat":
            if len(op.inputs) < 2:
                raise ValueError("Bound IR v1 concat requires at least two inputs")
            if any(value_id not in dynamic_names for value_id in op.inputs):
                raise NotImplementedError(
                    "Bound IR v1 concat supports dynamic inputs only"
                )
            out_shape = value_shape(out_name)
            axis = _normalize_concat_primal_axis(
                op.attrs.get("axis", 1), primal_rank=len(out_shape)
            )
            input_shapes = tuple(value_shape(value_id) for value_id in op.inputs)
            _validate_concat_shapes(input_shapes, out_shape=out_shape, axis=axis)
            targets = tuple(
                builder.add_state(
                    label=f"{op.name}_{index}",
                    primal_value_id=value_id,
                    primal_shape=shape,
                )
                for index, (value_id, shape) in enumerate(zip(op.inputs, input_shapes))
            )
            builder.add_op(
                label=op.name,
                kind=BoundOpKind.CONCAT_BACKWARD,
                inputs=source.value_ids,
                outputs=tuple(
                    component for target in targets for component in target.value_ids
                ),
                attrs=ConcatBackwardAttrs(
                    primal_node_id=op.name,
                    input_primal_value_ids=tuple(op.inputs),
                    input_shapes=input_shapes,
                    axis=axis,
                ),
            )
            for value_id, target in zip(op.inputs, targets):
                route(value_id, target)
            continue

        raise NotImplementedError(
            f"Bound IR v1 lowering does not support TaskOp '{op.op_type}'"
        )

    input_state = consume(input_spec.value_name)
    if input_state is None:
        raise RuntimeError(
            f"Bound IR backward graph did not reach input '{input_spec.value_name}'"
        )
    if adjoints:
        raise RuntimeError(
            f"Bound IR lowering left unconsumed adjoints: {sorted(adjoints)}"
        )
    lower = builder.add_result(label="concretize_lower", polarity=BoundPolarity.LOWER)
    upper = builder.add_result(label="concretize_upper", polarity=BoundPolarity.UPPER)
    builder.add_op(
        label="concretize",
        kind=BoundOpKind.CONCRETIZE,
        inputs=input_state.value_ids,
        outputs=(lower.value_id, upper.value_id),
        attrs=ConcretizeAttrs(perturbation_id=perturbation.perturbation_id),
    )

    graph = BFBoundGraph(
        values=tuple(builder.values),
        ops=tuple(builder.ops),
        inputs=(objective_input.value_id,),
        outputs=(lower.value_id, upper.value_id),
    )
    primal_hash = plain_crown_primal_graph_hash(task_module)
    bound_module = BFBoundModule(
        module_id=f"plain-crown-{primal_hash[:16]}",
        primal_graph_hash=primal_hash,
        spec=verification_spec,
        domain=BoundDomainConfig(method=BoundMethodKind.CROWN),
        graph=graph,
    )
    bound_module.validate()
    return PlainCrownBoundIRBuild(
        module=bound_module,
        objective_input_value_id=objective_input.value_id,
        input_affine_state=input_state,
        lower_output_value_id=lower.value_id,
        upper_output_value_id=upper.value_id,
    )


def _resolve_output(
    ops: Sequence[TaskOp],
    task_outputs: Sequence[str],
    output_value: Optional[str],
) -> str:
    if output_value is None:
        if len(task_outputs) != 1:
            raise ValueError("multiple task outputs require explicit output_value")
        output_value = task_outputs[0]
    if output_value not in task_outputs:
        raise ValueError(f"output '{output_value}' is not a task output")
    if ops[-1].outputs != [output_value]:
        raise NotImplementedError("Bound IR v1 supports the last TaskOp output only")
    return output_value


def _require_interval(
    interval_env: Mapping[str, IntervalState], value_id: str
) -> IntervalState:
    if value_id not in interval_env:
        raise KeyError(f"missing interval trace for '{value_id}'")
    return interval_env[value_id]


def _validate_trace_tensor_pair(state: IntervalState, *, label: str) -> None:
    if state.lower.shape != state.upper.shape:
        raise ValueError(f"interval trace '{label}' has mismatched shapes")
    if (
        state.lower.dtype != state.upper.dtype
        or state.lower.device != state.upper.device
    ):
        raise ValueError(f"interval trace '{label}' has mismatched dtype/device")
    if not torch.is_floating_point(state.lower):
        raise TypeError(f"interval trace '{label}' must be floating point")


def _validate_objective(
    linear_spec_C: Optional[torch.Tensor], *, batch: int, output_dim: int
) -> int:
    if linear_spec_C is None:
        return output_dim
    if not torch.is_tensor(linear_spec_C):
        raise TypeError("linear_spec_C must be a torch.Tensor when present")
    if linear_spec_C.dim() == 2:
        if int(linear_spec_C.shape[1]) != output_dim:
            raise ValueError("linear_spec_C output dimension mismatch")
        return int(linear_spec_C.shape[0])
    if linear_spec_C.dim() == 3:
        if (
            int(linear_spec_C.shape[0]) != batch
            or int(linear_spec_C.shape[2]) != output_dim
        ):
            raise ValueError("linear_spec_C batch/output dimension mismatch")
        return int(linear_spec_C.shape[1])
    raise ValueError("linear_spec_C must have rank 2 [S,O] or rank 3 [B,S,O]")


def _objective_identity(
    linear_spec_C: Optional[torch.Tensor],
) -> tuple[str, ObjectiveKind, Optional[str]]:
    if linear_spec_C is None:
        return "identity-objective", ObjectiveKind.IDENTITY, None
    payload_hash = tensor_content_hash(linear_spec_C)
    return (
        f"linear-objective-{payload_hash[:16]}",
        ObjectiveKind.LINEAR,
        payload_hash,
    )


def _perturbation_spec(input_spec: InputSpec) -> PerturbationSpec:
    perturbation = input_spec.perturbation
    if isinstance(perturbation, LpBallPerturbation):
        normalized = perturbation._normalize_p()  # pylint: disable=protected-access
        kind = {
            "inf": PerturbationKind.LINF,
            "2": PerturbationKind.L2,
            "1": PerturbationKind.L1,
        }[normalized]
        return PerturbationSpec(
            perturbation_id=perturbation.perturbation_id,
            input_primal_value_id=input_spec.value_name,
            kind=kind,
            radius=float(perturbation.eps),
        )
    if isinstance(perturbation, BoxPerturbation):
        perturbation_id = perturbation.perturbation_id
        payload_hash = perturbation_id.removeprefix("box(sha256=").removesuffix(")")
        return PerturbationSpec(
            perturbation_id=perturbation_id,
            input_primal_value_id=input_spec.value_name,
            kind=PerturbationKind.BOX,
            payload_hash=payload_hash,
        )
    raise NotImplementedError(
        f"Bound IR v1 does not support perturbation {type(perturbation).__name__}"
    )


def _parameter_bindings(module: BFTaskModule) -> dict[str, Any]:
    raw = module.bindings.get("params", {})
    if not isinstance(raw, dict):
        raise TypeError("BFTaskModule bindings['params'] must be a dictionary")
    return dict(raw)


def _require_param_inputs(
    op: TaskOp, params: Mapping[str, Any], *, minimum: int, maximum: int
) -> None:
    if len(op.inputs) < minimum or len(op.inputs) > maximum:
        raise ValueError(f"TaskOp '{op.name}' has invalid input arity")
    for value_id in op.inputs[1:]:
        if value_id not in params:
            raise KeyError(f"missing parameter '{value_id}' for TaskOp '{op.name}'")


def _as_pair(value: object, *, label: str) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) in {1, 2}:
        first = int(value[0])
        second = first if len(value) == 1 else int(value[1])
        return (first, second)
    raise ValueError(f"{label} must be an int or length-1/2 sequence")


def _normalize_concat_primal_axis(value: object, *, primal_rank: int) -> int:
    if not isinstance(value, int):
        raise TypeError("concat axis must be an integer")
    axis_with_batch = value
    rank_with_batch = primal_rank + 1
    if axis_with_batch < 0:
        axis_with_batch += rank_with_batch
    if axis_with_batch != 1:
        raise NotImplementedError(
            "Bound IR v1 concat supports the first non-batch axis only"
        )
    return 0


def _validate_concat_shapes(
    input_shapes: Sequence[tuple[int, ...]],
    *,
    out_shape: tuple[int, ...],
    axis: int,
) -> None:
    if not input_shapes or any(len(shape) != len(out_shape) for shape in input_shapes):
        raise ValueError("concat input/output ranks do not match")
    expected = list(input_shapes[0])
    expected[axis] = sum(shape[axis] for shape in input_shapes)
    for shape in input_shapes[1:]:
        for dimension, (left, right) in enumerate(zip(input_shapes[0], shape)):
            if dimension != axis and left != right:
                raise ValueError("concat non-axis dimensions do not match")
    if tuple(expected) != out_shape:
        raise ValueError("concat trace output shape does not match inputs")


def plain_crown_primal_graph_hash(module: BFTaskModule) -> str:
    """Fingerprint TaskOps and parameter contents used by Bound IR references."""

    task = module.get_entry_task()
    params = _parameter_bindings(module)
    payload = {
        "task_id": task.task_id,
        "inputs": list(task.input_values),
        "outputs": list(task.output_values),
        "ops": [
            {
                "type": op.op_type,
                "name": op.name,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
                "attrs": _jsonable(op.attrs),
            }
            for op in task.ops
        ],
        "params": {
            value_id: tensor_content_hash(_as_tensor(value))
            for value_id, value in sorted(params.items())
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tensor_content_hash(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and exact contiguous bytes."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _as_tensor(value: Any) -> torch.Tensor:
    return value if torch.is_tensor(value) else torch.as_tensor(value)


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise TypeError(f"TaskOp attribute is not deterministic JSON data: {type(value)}")
