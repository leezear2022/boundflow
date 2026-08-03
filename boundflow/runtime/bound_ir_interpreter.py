"""Independent dense/structured reference interpreter for Bound IR v1 CROWN."""

# CROWN literature and the existing runtime use A_u/A_l for affine coefficients.
# pylint: disable=too-many-locals,invalid-name,duplicate-code,not-callable
# pylint: disable=too-many-lines,too-many-return-statements

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, cast

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
    BoundRepresentation,
    ConcatBackwardAttrs,
    ConcretizeAttrs,
    Conv2dBackwardAttrs,
    LinearBackwardAttrs,
    ReluRelaxationAttrs,
    RepresentationChangeAttrs,
    ReshapeAttrs,
    SpecBindAttrs,
)
from ..ir.task import BFTaskModule
from .linear_operator import DenseLinearOperator, LinearOperator
from .fused_crown import (
    FusedCrownExecutionContext,
    FusedCrownExecutor,
    FusedReluAffineDescriptor,
    FusedReluAffineRequest,
)
from .task_executor import InputSpec

Coefficient = torch.Tensor | LinearOperator
RuntimeValue = torch.Tensor | LinearOperator


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
        raise NotImplementedError("Bound IR reference interpreter supports CROWN only")
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
    env: dict[str, RuntimeValue] = {}

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
            objective = _get_tensor(env, op.inputs[0])
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

        if op.kind in {
            BoundOpKind.REPRESENTATION_CAST,
            BoundOpKind.MATERIALIZE,
        }:
            attrs = _attrs(op, RepresentationChangeAttrs)
            source = _get(env, op.inputs[0])
            primal_shape = _coefficient_primal_shape(values, op.outputs[0])
            if attrs.target == BoundRepresentation.STRUCTURED:
                if not torch.is_tensor(source):
                    raise ValueError("dense-to-structured cast expects a tensor")
                env[op.outputs[0]] = cast(
                    LinearOperator,
                    DenseLinearOperator(source, input_shape=primal_shape),
                )
            elif attrs.target == BoundRepresentation.DENSE:
                if torch.is_tensor(source):
                    raise ValueError(
                        "structured-to-dense transition expects an operator"
                    )
                dense = source.to_dense()
                env[op.outputs[0]] = dense.reshape(
                    int(dense.shape[0]), int(dense.shape[1]), *primal_shape
                )
            else:
                raise NotImplementedError(
                    f"interpreter cannot transition to {attrs.target.value}"
                )
            continue

        if op.kind == BoundOpKind.COEFFICIENT_COMPOSE:
            states = tuple(
                _load_state(env, op.inputs[index : index + 4])
                for index in range(0, len(op.inputs), 4)
            )
            result = (
                _coefficient_sum(tuple(state[0] for state in states)),
                sum((state[1] for state in states[1:]), states[0][1]),
                _coefficient_sum(tuple(state[2] for state in states)),
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
                torch.zeros(
                    int(weight.shape[0]),
                    device=_coefficient_device(A_u),
                    dtype=_coefficient_dtype(A_u),
                )
                if attrs.bias_primal_value_id is None
                else _tensor_binding(params, attrs.bias_primal_value_id, like=A_u)
            )
            if bias.dim() == 0:
                bias = bias.expand(int(weight.shape[0]))
            if bias.dim() != 1 or int(bias.shape[0]) != int(weight.shape[0]):
                raise ValueError("linear bias must be scalar or rank 1 [output]")
            if _coefficient_input_numel(A_u) != int(weight.shape[0]):
                raise ValueError("linear coefficient/weight output shape mismatch")
            _store_state(
                env,
                op.outputs,
                (
                    _linear_backward_coefficient(A_u, weight),
                    b_u + _contract(A_u, bias),
                    _linear_backward_coefficient(A_l, weight),
                    b_l + _contract(A_l, bias),
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
            if not torch.is_tensor(A_u) or not torch.is_tensor(A_l):
                raise ValueError(
                    "ReLU relaxation requires an explicit dense materialization"
                )
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
                    _reshape_coefficient(A_u, target_shape),
                    b_u,
                    _reshape_coefficient(A_l, target_shape),
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
                    (
                        _clone_coefficient(A_u),
                        child_b_u,
                        _clone_coefficient(A_l),
                        child_b_l,
                    ),
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
                child_b_u = b_u if index == 0 else torch.zeros_like(b_u)
                child_b_l = b_l if index == 0 else torch.zeros_like(b_l)
                _store_state(
                    env,
                    state.value_ids,
                    (
                        _slice_coefficient(
                            A_u, shape=shape, axis=attrs.axis, start=start, stop=stop
                        ),
                        child_b_u,
                        _slice_coefficient(
                            A_l, shape=shape, axis=attrs.axis, start=start, stop=stop
                        ),
                        child_b_l,
                    ),
                )
                start = stop
            if start != _coefficient_input_shape(A_u)[attrs.axis]:
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
            f"Bound IR reference interpreter does not support {op.kind.value}"
        )

    lower_id, upper_id = bound_module.graph.outputs
    return IntervalState(
        lower=_get_tensor(env, lower_id), upper=_get_tensor(env, upper_id)
    )


@dataclass(frozen=True)
class BoundIRTaskStepResult:
    """Deterministic semantic outputs produced by one task step."""

    op_ids: Tuple[str, ...]
    output_value_hashes: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class PreparedPlainCrownBoundIRProgram:
    """One-time validated static program reused by dynamic query sessions."""

    bound_module: BFBoundModule
    task_module: BFTaskModule
    bound_module_hash: str
    params: Mapping[str, Any]
    values: Mapping[str, Any]

    @classmethod
    def prepare(
        cls,
        bound_module: BFBoundModule,
        task_module: BFTaskModule,
    ) -> "PreparedPlainCrownBoundIRProgram":
        """Validate immutable compiler/model identity and cache static lookups."""

        bound_module.validate()
        if bound_module.domain.method != BoundMethodKind.CROWN:
            raise NotImplementedError(
                "Bound IR reference interpreter supports CROWN only"
            )
        task_module.validate()
        if bound_module.primal_graph_hash != plain_crown_primal_graph_hash(task_module):
            raise ValueError(
                "runtime task/parameter fingerprint does not match Bound IR"
            )
        return cls(
            bound_module=bound_module,
            task_module=task_module,
            bound_module_hash=bound_module.stable_hash(),
            params={
                name: (value.detach().clone() if torch.is_tensor(value) else value)
                for name, value in _parameter_bindings(task_module).items()
            },
            values={value.value_id: value for value in bound_module.graph.values},
        )


class PlainCrownBoundIRSession:  # pylint: disable=too-many-instance-attributes
    # The keyword-only inputs make static program/query ownership explicit.
    # pylint: disable=too-many-arguments
    """Stateful reference session that executes contiguous Bound IR task regions."""

    def __init__(
        self,
        bound_module: BFBoundModule,
        *,
        task_module: BFTaskModule,
        input_spec: InputSpec,
        relu_pre: Mapping[str, IntervalState],
        linear_spec_C: Optional[torch.Tensor] = None,
        prepared_program: Optional[PreparedPlainCrownBoundIRProgram] = None,
        capture_output_hashes: bool = True,
    ) -> None:
        if prepared_program is None:
            prepared_program = PreparedPlainCrownBoundIRProgram.prepare(
                bound_module, task_module
            )
        elif (
            prepared_program.bound_module is not bound_module
            or prepared_program.task_module is not task_module
        ):
            raise ValueError("prepared Bound IR program identity differs from session")
        objective_spec = bound_module.spec.objectives[0]
        if objective_spec.payload_hash is None:
            if linear_spec_C is not None:
                raise ValueError("identity Bound IR cannot bind a linear objective")
        elif (
            linear_spec_C is None
            or objective_spec.payload_hash != tensor_content_hash(linear_spec_C)
        ):
            raise ValueError("runtime objective payload does not match Bound IR")
        self.bound_module = bound_module
        self.bound_module_hash = prepared_program.bound_module_hash
        self.input_spec = input_spec
        self.relu_pre = relu_pre
        self.params = prepared_program.params
        self.values = prepared_program.values
        self.capture_output_hashes = capture_output_hashes
        self.env: dict[str, RuntimeValue] = {}
        self.next_op_index = 0
        for input_value_id in bound_module.graph.inputs:
            input_value = self.values[input_value_id]
            if input_value.role.value != "objective":
                raise NotImplementedError(
                    "plain-CROWN interpreter only accepts objective graph inputs"
                )
            self.env[input_value_id] = _objective_tensor(
                input_value.tensor_type.shape,
                input_spec=input_spec,
                linear_spec_C=linear_spec_C,
            )

    def execute_task(
        self,
        op_ids: Tuple[str, ...],
        *,
        output_value_ids: Tuple[str, ...],
    ) -> BoundIRTaskStepResult:
        """Execute exactly the next contiguous task-owned Bound operations."""

        if not op_ids:
            raise ValueError("Bound IR task step requires op IDs")
        stop = self.next_op_index + len(op_ids)
        expected = self.bound_module.graph.ops[self.next_op_index : stop]
        if tuple(op.op_id for op in expected) != op_ids:
            raise ValueError(
                "Bound IR task step is non-contiguous, reordered, or repeated"
            )
        for op in expected:
            self._execute_op(op)
        self.next_op_index = stop
        missing = tuple(
            value_id for value_id in output_value_ids if value_id not in self.env
        )
        if missing:
            raise ValueError(f"Bound IR task step omits outputs: {missing}")
        return BoundIRTaskStepResult(
            op_ids=op_ids,
            output_value_hashes=self._output_hashes(output_value_ids),
        )

    def load_state_value(
        self,
        value_id: str,
        *,
        state_version: str,
        value: torch.Tensor,
    ) -> None:
        """Bind one exact dense cached value before its producer task runs."""

        bound_value = self.values.get(value_id)
        if bound_value is None:
            raise ValueError(f"runtime state references unknown value: {value_id}")
        if bound_value.state_version != state_version:
            raise ValueError("runtime state version differs from Bound IR")
        if bound_value.representation != BoundRepresentation.DENSE:
            raise NotImplementedError(
                "runtime state payload v1 supports dense values only"
            )
        expected_shape = bound_value.tensor_type.shape
        if any(dim is None for dim in expected_shape):
            raise NotImplementedError(
                "runtime state payload v1 requires static Bound value shapes"
            )
        static_shape = cast(Tuple[int, ...], expected_shape)
        if tuple(int(dim) for dim in value.shape) != tuple(
            int(dim) for dim in static_shape
        ):
            raise ValueError("runtime state tensor shape differs from Bound IR")
        if str(value.dtype).removeprefix("torch.") != bound_value.tensor_type.dtype:
            raise ValueError("runtime state tensor dtype differs from Bound IR")
        if str(value.device) != bound_value.tensor_type.device:
            raise ValueError("runtime state tensor device differs from Bound IR")
        existing = self.env.get(value_id)
        if existing is not None and _runtime_value_hash(
            existing
        ) != _runtime_value_hash(value):
            raise ValueError("runtime state conflicts with an existing value")
        self.env[value_id] = value.detach().clone()

    def export_state_value(self, value_id: str, *, state_version: str) -> torch.Tensor:
        """Return an owned dense value only after exact-version validation."""

        bound_value = self.values.get(value_id)
        if bound_value is None or bound_value.state_version != state_version:
            raise ValueError("runtime state export identity/version mismatch")
        value = self.env.get(value_id)
        if value is None:
            raise ValueError("runtime state export requested before value definition")
        if not torch.is_tensor(value):
            raise NotImplementedError(
                "runtime state payload v1 does not serialize structured operators"
            )
        return value.detach().clone()

    def skip_task_with_loaded_outputs(
        self,
        op_ids: Tuple[str, ...],
        *,
        output_value_ids: Tuple[str, ...],
    ) -> BoundIRTaskStepResult:
        """Advance past a task only when every boundary output was loaded."""

        if not op_ids or not output_value_ids:
            raise ValueError("state-reused task requires ops and boundary outputs")
        stop = self.next_op_index + len(op_ids)
        expected = self.bound_module.graph.ops[self.next_op_index : stop]
        if tuple(op.op_id for op in expected) != op_ids:
            raise ValueError("state-reused Bound task is reordered or non-contiguous")
        missing = tuple(
            value_id for value_id in output_value_ids if value_id not in self.env
        )
        if missing:
            raise ValueError(
                f"state reuse cannot skip task with unloaded outputs: {missing}"
            )
        self.next_op_index = stop
        return BoundIRTaskStepResult(
            op_ids=op_ids,
            output_value_hashes=self._output_hashes(output_value_ids),
        )

    def execute_fused_relu_affine_task(
        self,
        op_ids: Tuple[str, ...],
        *,
        output_value_ids: Tuple[str, ...],
        executor: FusedCrownExecutor,
    ) -> BoundIRTaskStepResult:
        """Execute one exact ReLU→Affine Bound region through a fused backend."""

        if len(op_ids) != 2:
            raise ValueError("fused Bound task requires exactly two op IDs")
        expected = self.bound_module.graph.ops[
            self.next_op_index : self.next_op_index + 2
        ]
        if tuple(op.op_id for op in expected) != op_ids:
            raise ValueError("fused Bound task is non-contiguous or reordered")
        relu_op, affine_op = expected
        if relu_op.kind != BoundOpKind.RELU_RELAXATION or affine_op.kind not in {
            BoundOpKind.LINEAR_BACKWARD,
            BoundOpKind.CONV2D_BACKWARD,
        }:
            raise ValueError("fused Bound task must be ReLU followed by affine")
        relu_attrs = _attrs(relu_op, ReluRelaxationAttrs)
        preactivation = relu_attrs.preactivation_primal_value_id
        if preactivation is None or preactivation not in self.relu_pre:
            raise KeyError(f"missing ReLU pre-activation binding '{preactivation}'")
        A_u, b_u, A_l, b_l = _load_state(self.env, relu_op.inputs)
        if not torch.is_tensor(A_u) or not torch.is_tensor(A_l):
            raise ValueError("fused Bound task requires dense coefficient inputs")
        pre = self.relu_pre[preactivation]
        alpha_u, beta_u, alpha_l, beta_l = _relu_relaxation_parameters(pre)
        source_shape = _coefficient_primal_shape(self.values, affine_op.inputs[0])
        target_shape = _coefficient_primal_shape(self.values, affine_op.outputs[0])
        attrs: dict[str, object] = {}
        if affine_op.kind == BoundOpKind.LINEAR_BACKWARD:
            affine_attrs = _attrs(affine_op, LinearBackwardAttrs)
            kind = "linear"
            weight = _tensor_binding(
                self.params, affine_attrs.weight_primal_value_id, like=A_u
            )
            bias = (
                None
                if affine_attrs.bias_primal_value_id is None
                else _tensor_binding(
                    self.params, affine_attrs.bias_primal_value_id, like=A_u
                )
            )
        else:
            affine_attrs = _attrs(affine_op, Conv2dBackwardAttrs)
            kind = "conv2d"
            weight = _tensor_binding(
                self.params, affine_attrs.weight_primal_value_id, like=A_u
            )
            bias = (
                None
                if affine_attrs.bias_primal_value_id is None
                else _tensor_binding(
                    self.params, affine_attrs.bias_primal_value_id, like=A_u
                )
            )
            attrs.update(
                stride=affine_attrs.stride,
                padding=affine_attrs.padding,
                dilation=affine_attrs.dilation,
                groups=affine_attrs.groups,
                output_padding=tuple(
                    int(target_shape[axis + 1])
                    - (
                        (int(source_shape[axis + 1]) - 1)
                        * int(affine_attrs.stride[axis])
                        - 2 * int(affine_attrs.padding[axis])
                        + int(affine_attrs.dilation[axis])
                        * (int(weight.shape[axis + 2]) - 1)
                        + 1
                    )
                    for axis in range(2)
                ),
            )
        context = FusedCrownExecutionContext()
        descriptor = FusedReluAffineDescriptor(
            kind=kind,  # type: ignore[arg-type]
            coefficient_shape=(
                int(A_u.shape[0]),
                int(A_u.shape[1]),
                int(A_u[0, 0].numel()),
            ),
            weight=weight,
            bias=bias,
            input_shape=target_shape,
            output_shape=source_shape,
            attrs=attrs,
            device=A_u.device,
            dtype=A_u.dtype,
        )
        if not executor.supports_descriptor(descriptor, context):
            raise ValueError("selected fused backend rejects Task IR descriptor")
        request = FusedReluAffineRequest(
            kind=kind,  # type: ignore[arg-type]
            A_u=A_u.contiguous(),
            A_l=A_l.contiguous(),
            alpha_u=alpha_u.contiguous(),
            alpha_l=alpha_l.contiguous(),
            beta_u=beta_u.contiguous(),
            beta_l=beta_l.contiguous(),
            weight=weight.contiguous(),
            bias=None if bias is None else bias.contiguous(),
            input_shape=target_shape,
            output_shape=source_shape,
            attrs=attrs,
        )
        if not executor.supports(request, context):
            raise ValueError("selected fused backend rejects Task IR request")
        stream = (
            torch.cuda.current_stream(A_u.device) if A_u.device.type == "cuda" else None
        )
        result = executor.run(request, stream=stream)
        _store_state(
            self.env,
            affine_op.outputs,
            (
                result.A_prev_u,
                b_u + result.bias_delta_u,
                result.A_prev_l,
                b_l + result.bias_delta_l,
            ),
        )
        self.next_op_index += 2
        missing = tuple(
            value_id for value_id in output_value_ids if value_id not in self.env
        )
        if missing:
            raise ValueError(f"fused Bound task omits outputs: {missing}")
        return BoundIRTaskStepResult(
            op_ids=op_ids,
            output_value_hashes=self._output_hashes(output_value_ids),
        )

    def _output_hashes(
        self, output_value_ids: Tuple[str, ...]
    ) -> Tuple[Tuple[str, str], ...]:
        if not self.capture_output_hashes:
            return ()
        return tuple(
            (value_id, _runtime_value_hash(self.env[value_id]))
            for value_id in output_value_ids
        )

    def result(self) -> IntervalState:
        """Return final bounds only after every Bound operation has executed."""

        if self.next_op_index != len(self.bound_module.graph.ops):
            raise ValueError("Bound IR session result requested before task completion")
        lower_id, upper_id = self.bound_module.graph.outputs
        return IntervalState(
            lower=_get_tensor(self.env, lower_id),
            upper=_get_tensor(self.env, upper_id),
        )

    def _execute_op(  # pylint: disable=too-many-branches,too-many-statements
        self, op: BoundOp
    ) -> None:
        env = self.env
        values = self.values
        params = self.params
        input_spec = self.input_spec
        relu_pre = self.relu_pre

        if op.kind == BoundOpKind.SPEC_BIND:
            _attrs(op, SpecBindAttrs)
            objective = _get_tensor(env, op.inputs[0])
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
                (objective.clone(), bias, objective.clone(), bias.clone()),
            )
            return

        if op.kind in {
            BoundOpKind.REPRESENTATION_CAST,
            BoundOpKind.MATERIALIZE,
        }:
            attrs = _attrs(op, RepresentationChangeAttrs)
            source = _get(env, op.inputs[0])
            primal_shape = _coefficient_primal_shape(values, op.outputs[0])
            if attrs.target == BoundRepresentation.STRUCTURED:
                if not torch.is_tensor(source):
                    raise ValueError("dense-to-structured cast expects a tensor")
                env[op.outputs[0]] = cast(
                    LinearOperator,
                    DenseLinearOperator(source, input_shape=primal_shape),
                )
            elif attrs.target == BoundRepresentation.DENSE:
                if torch.is_tensor(source):
                    raise ValueError(
                        "structured-to-dense transition expects an operator"
                    )
                dense = source.to_dense()
                env[op.outputs[0]] = dense.reshape(
                    int(dense.shape[0]), int(dense.shape[1]), *primal_shape
                )
            else:
                raise NotImplementedError(
                    f"interpreter cannot transition to {attrs.target.value}"
                )
            return

        if op.kind == BoundOpKind.COEFFICIENT_COMPOSE:
            states = tuple(
                _load_state(env, op.inputs[index : index + 4])
                for index in range(0, len(op.inputs), 4)
            )
            result = (
                _coefficient_sum(tuple(state[0] for state in states)),
                sum((state[1] for state in states[1:]), states[0][1]),
                _coefficient_sum(tuple(state[2] for state in states)),
                sum((state[3] for state in states[1:]), states[0][3]),
            )
            _store_state(env, op.outputs, result)
            return

        if op.kind == BoundOpKind.LINEAR_BACKWARD:
            attrs = _attrs(op, LinearBackwardAttrs)
            A_u, b_u, A_l, b_l = _load_state(env, op.inputs)
            weight = _tensor_binding(params, attrs.weight_primal_value_id, like=A_u)
            if weight.dim() != 2:
                raise ValueError("linear weight must have rank 2")
            bias = (
                torch.zeros(
                    int(weight.shape[0]),
                    device=_coefficient_device(A_u),
                    dtype=_coefficient_dtype(A_u),
                )
                if attrs.bias_primal_value_id is None
                else _tensor_binding(params, attrs.bias_primal_value_id, like=A_u)
            )
            if bias.dim() == 0:
                bias = bias.expand(int(weight.shape[0]))
            if bias.dim() != 1 or int(bias.shape[0]) != int(weight.shape[0]):
                raise ValueError("linear bias must be scalar or rank 1 [output]")
            if _coefficient_input_numel(A_u) != int(weight.shape[0]):
                raise ValueError("linear coefficient/weight output shape mismatch")
            _store_state(
                env,
                op.outputs,
                (
                    _linear_backward_coefficient(A_u, weight),
                    b_u + _contract(A_u, bias),
                    _linear_backward_coefficient(A_l, weight),
                    b_l + _contract(A_l, bias),
                ),
            )
            return

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
            return

        if op.kind == BoundOpKind.RELU_RELAXATION:
            attrs = _attrs(op, ReluRelaxationAttrs)
            preactivation = attrs.preactivation_primal_value_id
            if preactivation is None or preactivation not in relu_pre:
                raise KeyError(f"missing ReLU pre-activation binding '{preactivation}'")
            A_u, b_u, A_l, b_l = _load_state(env, op.inputs)
            if not torch.is_tensor(A_u) or not torch.is_tensor(A_l):
                raise ValueError(
                    "ReLU relaxation requires an explicit dense materialization"
                )
            result = _relu_backward(A_u, b_u, A_l, b_l, pre=relu_pre[preactivation])
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
            return

        if op.kind == BoundOpKind.RESHAPE:
            attrs = _attrs(op, ReshapeAttrs)
            A_u, b_u, A_l, b_l = _load_state(env, op.inputs)
            target_shape = tuple(int(dim) for dim in attrs.target_shape)
            _store_state(
                env,
                op.outputs,
                (
                    _reshape_coefficient(A_u, target_shape),
                    b_u,
                    _reshape_coefficient(A_l, target_shape),
                    b_l,
                ),
            )
            return

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
                    (
                        _clone_coefficient(A_u),
                        child_b_u,
                        _clone_coefficient(A_l),
                        child_b_l,
                    ),
                )
            return

        if op.kind == BoundOpKind.CONCAT_BACKWARD:
            attrs = _attrs(op, ConcatBackwardAttrs)
            A_u, b_u, A_l, b_l = _load_state(env, op.inputs)
            output_states = _state_refs(op.outputs)
            start = 0
            for index, (state, shape) in enumerate(
                zip(output_states, attrs.input_shapes)
            ):
                stop = start + int(shape[attrs.axis])
                child_b_u = b_u if index == 0 else torch.zeros_like(b_u)
                child_b_l = b_l if index == 0 else torch.zeros_like(b_l)
                _store_state(
                    env,
                    state.value_ids,
                    (
                        _slice_coefficient(
                            A_u,
                            shape=shape,
                            axis=attrs.axis,
                            start=start,
                            stop=stop,
                        ),
                        child_b_u,
                        _slice_coefficient(
                            A_l,
                            shape=shape,
                            axis=attrs.axis,
                            start=start,
                            stop=stop,
                        ),
                        child_b_l,
                    ),
                )
                start = stop
            if start != _coefficient_input_shape(A_u)[attrs.axis]:
                raise ValueError("concat backward slices do not cover source axis")
            return

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
            return

        raise NotImplementedError(
            f"Bound IR reference interpreter does not support {op.kind.value}"
        )


def _runtime_value_hash(value: RuntimeValue) -> str:
    tensor = value if torch.is_tensor(value) else value.to_dense()
    return tensor_content_hash(tensor)


def runtime_value_hash(value: RuntimeValue) -> str:
    """Return the canonical content hash used by runtime state payloads."""

    return _runtime_value_hash(value)


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
    alpha_u, beta_u, alpha_l, _beta_l = _relu_relaxation_parameters(pre)
    A_u_flat = A_u.reshape(int(A_u.shape[0]), int(A_u.shape[1]), -1)
    A_l_flat = A_l.reshape(int(A_l.shape[0]), int(A_l.shape[1]), -1)
    if int(A_u_flat.shape[2]) != int(alpha_u.shape[1]):
        raise ValueError("ReLU coefficient/pre-activation shape mismatch")

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


def _relu_relaxation_parameters(
    pre: IntervalState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lower = pre.lower.reshape(int(pre.lower.shape[0]), -1)
    upper = pre.upper.reshape(int(pre.upper.shape[0]), -1)
    if lower.shape != upper.shape:
        raise ValueError("ReLU pre-activation lower/upper shapes differ")
    positive = lower >= 0
    negative = upper <= 0
    ambiguous = ~(positive | negative)
    alpha_u = torch.zeros_like(lower)
    beta_u = torch.zeros_like(lower)
    alpha_l = torch.zeros_like(lower)
    beta_l = torch.zeros_like(lower)
    alpha_u[positive] = 1
    alpha_l[positive] = 1
    if ambiguous.any():
        denominator = (upper[ambiguous] - lower[ambiguous]).clamp_min(
            torch.finfo(lower.dtype).eps
        )
        alpha_u[ambiguous] = upper[ambiguous] / denominator
        beta_u[ambiguous] = -lower[ambiguous] * alpha_u[ambiguous]
    return alpha_u, beta_u, alpha_l, beta_l


def _conv2d_backward(
    coefficient: Coefficient,
    *,
    target_shape: tuple[int, ...],
    source_shape: tuple[int, ...],
    weight: torch.Tensor,
    attrs: Conv2dBackwardAttrs,
) -> Coefficient:
    if len(target_shape) != 3 or len(source_shape) != 3:
        raise ValueError("conv2d backward requires CHW coefficient shapes")
    if not torch.is_tensor(coefficient):
        return coefficient.conv2d_right(
            weight,
            stride=attrs.stride,
            padding=attrs.padding,
            dilation=attrs.dilation,
            groups=attrs.groups,
            input_shape=target_shape,
        )
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


def _contract(coefficient: Coefficient, value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(coefficient):
        return coefficient.contract_input(value)
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
            "Bound IR interpreter requires static coefficient shapes"
        )
    return tuple(int(dimension) for dimension in shape)


def _coefficient_device(coefficient: Coefficient) -> torch.device:
    return coefficient.device


def _coefficient_dtype(coefficient: Coefficient) -> torch.dtype:
    return coefficient.dtype


def _coefficient_input_shape(coefficient: Coefficient) -> tuple[int, ...]:
    if torch.is_tensor(coefficient):
        return tuple(int(dimension) for dimension in coefficient.shape[2:])
    return tuple(int(dimension) for dimension in coefficient.input_shape)


def _coefficient_input_numel(coefficient: Coefficient) -> int:
    if not torch.is_tensor(coefficient):
        return int(coefficient.input_numel)
    result = 1
    for dimension in coefficient.shape[2:]:
        result *= int(dimension)
    return result


def _linear_backward_coefficient(
    coefficient: Coefficient, weight: torch.Tensor
) -> Coefficient:
    if not torch.is_tensor(coefficient):
        return coefficient.matmul_right(weight)
    flat = coefficient.reshape(int(coefficient.shape[0]), int(coefficient.shape[1]), -1)
    return flat.matmul(weight)


def _reshape_coefficient(
    coefficient: Coefficient, target_shape: tuple[int, ...]
) -> Coefficient:
    if not torch.is_tensor(coefficient):
        return coefficient.reshape_input(target_shape)
    return coefficient.reshape(
        int(coefficient.shape[0]), int(coefficient.shape[1]), *target_shape
    )


def _clone_coefficient(coefficient: Coefficient) -> Coefficient:
    return coefficient.clone() if torch.is_tensor(coefficient) else coefficient


def _slice_coefficient(
    coefficient: Coefficient,
    *,
    shape: tuple[int, ...],
    axis: int,
    start: int,
    stop: int,
) -> Coefficient:
    if axis != 0:
        raise NotImplementedError(
            "Bound IR v1 structured concat supports the first primal axis"
        )
    if not torch.is_tensor(coefficient):
        return coefficient.slice_input(shape, start=start, stop=stop)
    slices = [slice(None)] * coefficient.dim()
    slices[axis + 2] = slice(start, stop)
    return coefficient[tuple(slices)].contiguous()


def _coefficient_sum(coefficients: tuple[Coefficient, ...]) -> Coefficient:
    if not coefficients:
        raise ValueError("coefficient sum requires at least one input")
    result = coefficients[0]
    for coefficient in coefficients[1:]:
        if torch.is_tensor(result) and torch.is_tensor(coefficient):
            result = result + coefficient
        elif not torch.is_tensor(result) and not torch.is_tensor(coefficient):
            result = result.add(coefficient)
        else:
            raise ValueError("coefficient sum requires one consistent representation")
    return result


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
    env: Mapping[str, RuntimeValue], value_ids: tuple[str, ...]
) -> tuple[Coefficient, torch.Tensor, Coefficient, torch.Tensor]:
    if len(value_ids) != 4:
        raise ValueError("affine state must contain four values")
    return (
        _get_coefficient(env, value_ids[0]),
        _get_tensor(env, value_ids[1]),
        _get_coefficient(env, value_ids[2]),
        _get_tensor(env, value_ids[3]),
    )


def _store_state(
    env: dict[str, RuntimeValue],
    value_ids: tuple[str, ...],
    state: tuple[Coefficient, torch.Tensor, Coefficient, torch.Tensor],
) -> None:
    if len(value_ids) != 4:
        raise ValueError("affine state output must contain four values")
    for value_id, tensor in zip(value_ids, state):
        env[value_id] = tensor


def _get(env: Mapping[str, RuntimeValue], value_id: str) -> RuntimeValue:
    if value_id not in env:
        raise KeyError(f"Bound IR value '{value_id}' is not bound")
    return env[value_id]


def _get_tensor(env: Mapping[str, RuntimeValue], value_id: str) -> torch.Tensor:
    value = _get(env, value_id)
    if not torch.is_tensor(value):
        raise TypeError(f"Bound IR value '{value_id}' is not a tensor")
    return value


def _get_coefficient(env: Mapping[str, RuntimeValue], value_id: str) -> Coefficient:
    value = _get(env, value_id)
    if torch.is_tensor(value) or isinstance(value, LinearOperator):
        return value
    raise TypeError(f"Bound IR value '{value_id}' is not a coefficient")


def _parameter_bindings(module: BFTaskModule) -> dict[str, Any]:
    raw = module.bindings.get("params", {})
    if not isinstance(raw, dict):
        raise TypeError("BFTaskModule bindings['params'] must be a dictionary")
    return dict(raw)


def _tensor_binding(
    params: Mapping[str, Any], value_id: str, *, like: Coefficient
) -> torch.Tensor:
    if value_id not in params:
        raise KeyError(f"missing runtime parameter binding '{value_id}'")
    value = params[value_id]
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.to(device=_coefficient_device(like), dtype=_coefficient_dtype(like))


def _attrs(op: BoundOp, expected_type: type[Any]) -> Any:
    if not isinstance(op.attrs, expected_type):
        raise TypeError(
            f"BoundOp '{op.op_id}' expects {expected_type.__name__} attributes"
        )
    return op.attrs
