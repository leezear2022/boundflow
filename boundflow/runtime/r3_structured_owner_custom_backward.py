"""R3-1 full lower-region owner with a rematerializing custom backward."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes
# pylint: disable=too-many-statements,too-many-boolean-expressions,protected-access
# pylint: disable=missing-function-docstring,abstract-method,arguments-differ

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import torch
from torch.autograd.function import once_differentiable

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    plain_crown_primal_graph_hash,
)
from ..ir.structured_lower_region import assert_tensor_free_context
from ..ir.task import BFTaskModule
from .alpha_beta_crown import BetaState, _beta_to_relu_pre_add_coeff
from .crown_ibp import _forward_ibp_trace_mlp, run_crown_ibp_mlp_from_forward_trace
from .rvir_v4_pre_state_initializer import (
    ProductionNativePreStateV4,
    ProductionReluTopologyV4,
)
from .rvir_v4_production_state import (
    ProductionStateSnapshotV4,
    ProductionTensorRole,
    production_tensor_sha256,
)
from .task_executor import InputSpec

R31_PLAN_SCHEMA = "boundflow.r3-1-full-region-plan/v1"
R31_RECEIPT_SCHEMA = "boundflow.r3-1-custom-backward-receipt/v1"
R31_START_NODE = "25/Conv_8"
R31_P_NATIVE_PREACTIVATION = "25"
R31_P_ALPHA_PATH = "alpha/%2Finput-24/%2F49"
R31_P_BETA_PATH = "beta/%2Finput-20/0/value"
R31_CONTEXT_SCHEMA = "boundflow.r3-1-autograd-context/v1"

_PLAN_REGISTRY: dict[str, "R31FullRegionPlanV1"] = {}
_COUNTER_REGISTRY: dict[str, "R31ExecutionCountersV1"] = {}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _encode(value: str) -> str:
    return value.replace("%", "%25").replace("/", "%2F")


def _one_role(
    snapshot: ProductionStateSnapshotV4, role: ProductionTensorRole
) -> torch.Tensor:
    values = [item.value for item in snapshot.tensors if item.role == role]
    if len(values) != 1:
        raise ValueError(f"R3-1 requires one {role.value}")
    return values[0]


def _ravel_indices(
    coordinates: Sequence[Sequence[int]], shape: tuple[int, ...]
) -> tuple[int, ...]:
    if len(coordinates) != len(shape):
        raise ValueError("R3-1 alpha coordinate rank differs")
    count = len(coordinates[0]) if coordinates else math.prod(shape)
    if any(len(axis) != count for axis in coordinates):
        raise ValueError("R3-1 alpha coordinate count differs")
    result = []
    for ordinal in range(count):
        flat = 0
        for axis, extent in enumerate(shape):
            coordinate = int(coordinates[axis][ordinal])
            if coordinate < 0 or coordinate >= extent:
                raise ValueError("R3-1 alpha coordinate is out of range")
            flat = flat * extent + coordinate
        result.append(flat)
    if len(set(result)) != len(result):
        raise ValueError("R3-1 alpha coordinates repeat")
    return tuple(result)


@dataclass(frozen=True)
class R31TensorSpecV1:
    """One frozen dynamic input slot owned by the custom Function boundary."""

    name: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    content_sha256: str

    def validate(self) -> None:
        if (
            not self.name
            or not self.role
            or any(dimension < 0 for dimension in self.shape)
            or not self.dtype
            or not _is_hash(self.content_sha256)
        ):
            raise ValueError("R3-1 tensor spec differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "name": self.name,
            "role": self.role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class R31ReluLayoutV1:
    """Tensor-free production compressed-state layout for one native ReLU."""

    native_preactivation: str
    provider_activation: str
    provider_preactivation: str
    alpha_path: str
    beta_path: str
    feature_shape: tuple[int, ...]
    alpha_flat_indices: tuple[int, ...]
    beta_locations: tuple[tuple[int, ...], ...]
    split_values: tuple[tuple[int, ...], ...]

    def validate(self, *, domain_count: int) -> None:
        feature_count = math.prod(self.feature_shape)
        if (
            not self.native_preactivation
            or not self.provider_activation
            or not self.provider_preactivation
            or not self.alpha_path
            or not self.beta_path
            or not self.feature_shape
            or any(dimension <= 0 for dimension in self.feature_shape)
            or not self.alpha_flat_indices
            or len(set(self.alpha_flat_indices)) != len(self.alpha_flat_indices)
            or any(
                index < 0 or index >= feature_count for index in self.alpha_flat_indices
            )
            or len(self.beta_locations) != domain_count
            or len(self.split_values) != domain_count
            or any(len(row) != feature_count for row in self.split_values)
            or any(
                value not in {-1, 0, 1} for row in self.split_values for value in row
            )
        ):
            raise ValueError("R3-1 ReLU layout differs")
        beta_widths = {len(row) for row in self.beta_locations}
        if len(beta_widths) != 1:
            raise ValueError("R3-1 beta width differs across domains")
        for row in self.beta_locations:
            if len(set(row)) != len(row) or any(
                location < 0 or location >= feature_count for location in row
            ):
                raise ValueError("R3-1 beta locations differ")

    def to_dict(self) -> dict[str, object]:
        return {
            "native_preactivation": self.native_preactivation,
            "provider_activation": self.provider_activation,
            "provider_preactivation": self.provider_preactivation,
            "alpha_path": self.alpha_path,
            "beta_path": self.beta_path,
            "feature_shape": list(self.feature_shape),
            "alpha_flat_indices": list(self.alpha_flat_indices),
            "beta_locations": [list(row) for row in self.beta_locations],
            "split_values": [list(row) for row in self.split_values],
        }


@dataclass(frozen=True)
class R31FullRegionPlanV1:
    """Static tensor-free plan plus exact ordering for full-region rematerialization."""

    module_template: BFTaskModule
    primal_graph_hash: str
    source_state_hash: str
    input_value_name: str
    parameter_names: tuple[str, ...]
    relu_layouts: tuple[R31ReluLayoutV1, ...]
    tensor_specs: tuple[R31TensorSpecV1, ...]
    domain_count: int
    spec_count: int
    start_node_id: str = R31_START_NODE
    scratch_slot_count: int = 2
    schema_version: str = R31_PLAN_SCHEMA

    @property
    def p_layout_ordinal(self) -> int:
        values = [
            ordinal
            for ordinal, layout in enumerate(self.relu_layouts)
            if layout.native_preactivation == R31_P_NATIVE_PREACTIVATION
        ]
        if len(values) != 1:
            raise ValueError("R3-1 P-anchor layout differs")
        return values[0]

    @property
    def p_alpha_input_ordinal(self) -> int:
        name = f"relu/{R31_P_NATIVE_PREACTIVATION}/alpha"
        values = [
            ordinal
            for ordinal, spec in enumerate(self.tensor_specs)
            if spec.name == name
        ]
        if len(values) != 1:
            raise ValueError("R3-1 P-anchor alpha slot differs")
        return values[0]

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "primal_graph_hash": self.primal_graph_hash,
            "source_state_hash": self.source_state_hash,
            "input_value_name": self.input_value_name,
            "parameter_names": list(self.parameter_names),
            "relu_layouts": [layout.to_dict() for layout in self.relu_layouts],
            "tensor_specs": [spec.to_dict() for spec in self.tensor_specs],
            "domain_count": self.domain_count,
            "spec_count": self.spec_count,
            "start_node_id": self.start_node_id,
            "scratch_slot_count": self.scratch_slot_count,
        }

    def stable_hash(self) -> str:
        return _hash(self.identity_payload())

    def validate(self) -> None:
        if (
            self.schema_version != R31_PLAN_SCHEMA
            or not _is_hash(self.primal_graph_hash)
            or not _is_hash(self.source_state_hash)
            or not self.input_value_name
            or not self.parameter_names
            or tuple(sorted(self.parameter_names)) != self.parameter_names
            or len(set(self.parameter_names)) != len(self.parameter_names)
            or len(self.relu_layouts) != 6
            or len({layout.native_preactivation for layout in self.relu_layouts}) != 6
            or len(self.tensor_specs) != 3 + len(self.parameter_names) + 4 * 6
            or len({spec.name for spec in self.tensor_specs}) != len(self.tensor_specs)
            or self.domain_count != 6
            or self.spec_count != 1
            or self.start_node_id != R31_START_NODE
            or self.scratch_slot_count > 2
        ):
            raise ValueError("R3-1 full-region plan differs")
        self.module_template.validate()
        if self.module_template.bindings.get("params") != {}:
            raise ValueError("R3-1 static plan retains parameter tensors")
        for layout in self.relu_layouts:
            layout.validate(domain_count=self.domain_count)
        p_layout = self.relu_layouts[self.p_layout_ordinal]
        if (
            p_layout.alpha_path != R31_P_ALPHA_PATH
            or p_layout.beta_path != R31_P_BETA_PATH
            or p_layout.feature_shape != (16, 8, 8)
            or len(p_layout.alpha_flat_indices) != 86
            or any(p_layout.beta_locations)
        ):
            raise ValueError("R3-1 frozen P-anchor layout differs")
        for spec in self.tensor_specs:
            spec.validate()
        p_alpha = self.tensor_specs[self.p_alpha_input_ordinal]
        p_beta = next(
            spec
            for spec in self.tensor_specs
            if spec.name == f"relu/{R31_P_NATIVE_PREACTIVATION}/beta"
        )
        if p_alpha.shape != (2, 1, 6, 86) or p_beta.shape != (6, 0):
            raise ValueError("R3-1 production P-anchor state shape differs")
        assert_tensor_free_context(self)


@dataclass
class R31ExecutionCountersV1:
    """Mutable scalar-only counters for one admitted custom Function call."""

    forward_count: int = 0
    backward_count: int = 0
    fallback_count: int = 0
    eager_escape_count: int = 0
    native_shadow_count: int = 0
    optimizer_mutation_count: int = 0

    def begin_forward(self) -> None:
        self.forward_count += 1
        if self.forward_count != 1:
            raise RuntimeError("R3-1 custom forward count differs")

    def begin_backward(self) -> None:
        self.backward_count += 1
        if self.backward_count != 1:
            raise RuntimeError("R3-1 custom backward count differs")

    def validate(self) -> None:
        if (
            self.forward_count != 1
            or self.backward_count != 1
            or self.fallback_count
            or self.eager_escape_count
            or self.native_shadow_count
            or self.optimizer_mutation_count
        ):
            raise ValueError("R3-1 execution counters differ")
        assert_tensor_free_context(self)


@dataclass(frozen=True)
class R31ExecutionReceiptV1:
    """Correctness-only receipt for one production-shaped custom backward."""

    plan_hash: str
    forward_count: int
    backward_count: int
    saved_tensor_count: int
    saved_logical_bytes: int
    saved_unique_storage_bytes: int
    saved_dense_a_count: int
    scratch_slot_count: int
    alpha_version_unchanged: bool
    beta_version_unchanged: bool
    fallback_count: int
    eager_escape_count: int
    native_shadow_count: int
    optimizer_mutation_count: int
    production_connected: bool = True
    timing_recorded: bool = False
    performance_claimed: bool = False
    schema_version: str = R31_RECEIPT_SCHEMA

    def validate(self) -> None:
        if (
            self.schema_version != R31_RECEIPT_SCHEMA
            or not _is_hash(self.plan_hash)
            or self.forward_count != 1
            or self.backward_count != 1
            or self.saved_tensor_count <= 0
            or self.saved_logical_bytes <= 0
            or self.saved_unique_storage_bytes <= 0
            or self.saved_unique_storage_bytes > self.saved_logical_bytes
            or self.saved_dense_a_count != 0
            or self.scratch_slot_count > 2
            or self.alpha_version_unchanged is not True
            or self.beta_version_unchanged is not True
            or self.fallback_count
            or self.eager_escape_count
            or self.native_shadow_count
            or self.optimizer_mutation_count
            or self.production_connected is not True
            or self.timing_recorded
            or self.performance_claimed
        ):
            raise ValueError("R3-1 execution receipt differs")


@dataclass(frozen=True)
class R31ExecutionResultV1:
    """Final lower and compressed P-alpha VJP returned by one candidate call."""

    final_lower: torch.Tensor
    compressed_alpha_gradient: torch.Tensor
    receipt: R31ExecutionReceiptV1

    def validate(self) -> None:
        self.receipt.validate()
        if (
            tuple(self.final_lower.shape) != (6, 1)
            or tuple(self.compressed_alpha_gradient.shape) != (2, 1, 6, 86)
            or not bool(torch.isfinite(self.final_lower).all().item())
            or not bool(torch.isfinite(self.compressed_alpha_gradient).all().item())
        ):
            raise ValueError("R3-1 execution result differs")


def _layout_from_snapshot(
    snapshot: ProductionStateSnapshotV4,
    mapping: ProductionNativePreStateV4,
    link: ProductionReluTopologyV4,
) -> R31ReluLayoutV1:
    tensor_map = snapshot.tensor_map()
    activation = _encode(link.provider_activation)
    preactivation = _encode(link.provider_preactivation)
    start = _encode(link.provider_start_node)
    alpha_path = f"alpha/{activation}/{start}"
    beta_prefix = f"beta/{preactivation}/0"
    feature_shape = tuple(
        int(value)
        for value in tensor_map[
            f"alpha_layout/{activation}/feature_shape"
        ].value.tolist()
    )
    coordinates: list[list[int]] = []
    ordinal = 0
    while f"alpha_layout/{activation}/feature_index/{ordinal}" in tensor_map:
        coordinates.append(
            [
                int(value)
                for value in tensor_map[
                    f"alpha_layout/{activation}/feature_index/{ordinal}"
                ].value.tolist()
            ]
        )
        ordinal += 1
    alpha_item = tensor_map[alpha_path]
    if coordinates:
        flat_indices = _ravel_indices(coordinates, feature_shape)
    else:
        flat_indices = tuple(range(math.prod(feature_shape)))
    if len(flat_indices) != int(alpha_item.value.shape[-1]):
        raise ValueError("R3-1 compressed alpha width differs")
    beta_locations = tuple(
        tuple(int(value) for value in row)
        for row in tensor_map[f"{beta_prefix}/location"].value.tolist()
    )
    split_values = tuple(
        tuple(int(value) for value in row)
        for row in mapping.splits[link.native_preactivation].reshape(6, -1).tolist()
    )
    result = R31ReluLayoutV1(
        native_preactivation=link.native_preactivation,
        provider_activation=link.provider_activation,
        provider_preactivation=link.provider_preactivation,
        alpha_path=alpha_path,
        beta_path=f"{beta_prefix}/value",
        feature_shape=feature_shape,
        alpha_flat_indices=flat_indices,
        beta_locations=beta_locations,
        split_values=split_values,
    )
    result.validate(domain_count=6)
    return result


def _tensor_spec(name: str, role: str, value: torch.Tensor) -> R31TensorSpecV1:
    result = R31TensorSpecV1(
        name=name,
        role=role,
        shape=tuple(int(dimension) for dimension in value.shape),
        dtype=str(value.dtype),
        content_sha256=production_tensor_sha256(value),
    )
    result.validate()
    return result


def compile_r31_full_region_plan_v1(
    module: BFTaskModule,
    snapshot: ProductionStateSnapshotV4,
    mapping: ProductionNativePreStateV4,
    topology: tuple[ProductionReluTopologyV4, ...],
) -> R31FullRegionPlanV1:
    """Compile exact frozen R3-1 input ordering without retaining live tensors."""

    module.validate()
    snapshot.validate()
    mapping.validate()
    params = module.bindings.get("params")
    if not isinstance(params, Mapping) or not params:
        raise ValueError("R3-1 module parameters differ")
    parameter_names = tuple(sorted(str(name) for name in params))
    if any(not torch.is_tensor(params[name]) for name in parameter_names):
        raise TypeError("R3-1 module parameter is not a tensor")
    tensor_map = snapshot.tensor_map()
    layouts = tuple(
        sorted(
            (_layout_from_snapshot(snapshot, mapping, link) for link in topology),
            key=lambda layout: layout.native_preactivation,
        )
    )
    lower = _one_role(snapshot, ProductionTensorRole.INPUT_LOWER)
    upper = _one_role(snapshot, ProductionTensorRole.INPUT_UPPER)
    objective = _one_role(snapshot, ProductionTensorRole.LINEAR_SPEC)
    specs = [
        _tensor_spec("input/lower", "input_lower", lower),
        _tensor_spec("input/upper", "input_upper", upper),
        _tensor_spec("objective", "linear_spec", objective),
    ]
    specs.extend(
        _tensor_spec(f"param/{name}", "weight", params[name])
        for name in parameter_names
    )
    for layout in layouts:
        encoded_pre = _encode(layout.provider_preactivation)
        specs.extend(
            (
                _tensor_spec(
                    f"relu/{layout.native_preactivation}/lower",
                    "bounds",
                    tensor_map[f"intermediate/{encoded_pre}/lower"].value,
                ),
                _tensor_spec(
                    f"relu/{layout.native_preactivation}/upper",
                    "bounds",
                    tensor_map[f"intermediate/{encoded_pre}/upper"].value,
                ),
                _tensor_spec(
                    f"relu/{layout.native_preactivation}/alpha",
                    "compressed_alpha",
                    tensor_map[layout.alpha_path].value,
                ),
                _tensor_spec(
                    f"relu/{layout.native_preactivation}/beta",
                    "compressed_beta",
                    tensor_map[layout.beta_path].value,
                ),
            )
        )
    template = copy.copy(module)
    template.bindings = {
        name: copy.deepcopy(value)
        for name, value in module.bindings.items()
        if name != "params"
    }
    template.bindings["params"] = {}
    plan = R31FullRegionPlanV1(
        module_template=template,
        primal_graph_hash=plain_crown_primal_graph_hash(module),
        source_state_hash=mapping.stable_hash(),
        input_value_name=module.get_entry_task().input_values[0],
        parameter_names=parameter_names,
        relu_layouts=layouts,
        tensor_specs=tuple(specs),
        domain_count=int(lower.shape[0]),
        spec_count=int(objective.shape[1]),
    )
    plan.validate()
    return plan


def bind_r31_runtime_inputs_v1(
    plan: R31FullRegionPlanV1,
    module: BFTaskModule,
    snapshot: ProductionStateSnapshotV4,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    """Bind and hash-check the exact frozen production state before hot execution."""

    plan.validate()
    module.validate()
    snapshot.validate()
    if plain_crown_primal_graph_hash(module) != plan.primal_graph_hash:
        raise ValueError("R3-1 runtime graph differs")
    params = module.bindings.get("params")
    if not isinstance(params, Mapping):
        raise ValueError("R3-1 runtime parameters differ")
    tensor_map = snapshot.tensor_map()
    values: list[torch.Tensor] = [
        _one_role(snapshot, ProductionTensorRole.INPUT_LOWER),
        _one_role(snapshot, ProductionTensorRole.INPUT_UPPER),
        _one_role(snapshot, ProductionTensorRole.LINEAR_SPEC),
    ]
    values.extend(params[name] for name in plan.parameter_names)
    for layout in plan.relu_layouts:
        encoded_pre = _encode(layout.provider_preactivation)
        values.extend(
            (
                tensor_map[f"intermediate/{encoded_pre}/lower"].value,
                tensor_map[f"intermediate/{encoded_pre}/upper"].value,
                tensor_map[layout.alpha_path].value,
                tensor_map[layout.beta_path].value,
            )
        )
    if len(values) != len(plan.tensor_specs):
        raise ValueError("R3-1 runtime tensor count differs")
    bound = []
    for spec, value in zip(plan.tensor_specs, values):
        if (
            tuple(value.shape) != spec.shape
            or str(value.dtype) != spec.dtype
            or production_tensor_sha256(value) != spec.content_sha256
        ):
            raise ValueError(f"R3-1 runtime tensor identity differs: {spec.name}")
        target_dtype = dtype if value.is_floating_point() else value.dtype
        bound.append(value.detach().to(device=device, dtype=target_dtype).contiguous())
    p_alpha = bound[plan.p_alpha_input_ordinal].clone().requires_grad_(True)
    bound[plan.p_alpha_input_ordinal] = p_alpha
    return tuple(bound)


def _runtime_parts(
    plan: R31FullRegionPlanV1, tensors: tuple[torch.Tensor, ...]
) -> tuple[
    BFTaskModule,
    InputSpec,
    torch.Tensor,
    dict[str, IntervalState],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    if len(tensors) != len(plan.tensor_specs):
        raise ValueError("R3-1 custom Function tensor count differs")
    for spec, value in zip(plan.tensor_specs, tensors):
        if tuple(value.shape) != spec.shape or str(value.dtype) != spec.dtype:
            raise ValueError(f"R3-1 custom Function tensor schema differs: {spec.name}")
    lower, upper, objective = tensors[:3]
    offset = 3
    params = dict(
        zip(plan.parameter_names, tensors[offset : offset + len(plan.parameter_names)])
    )
    offset += len(plan.parameter_names)
    relu_pre: dict[str, IntervalState] = {}
    alphas: dict[str, torch.Tensor] = {}
    betas: dict[str, torch.Tensor] = {}
    splits: dict[str, torch.Tensor] = {}
    for layout in plan.relu_layouts:
        pre_lower, pre_upper, compressed_alpha, compressed_beta = tensors[
            offset : offset + 4
        ]
        offset += 4
        relu_pre[layout.native_preactivation] = IntervalState(pre_lower, pre_upper)
        alpha_source = compressed_alpha[0, 0]
        flat_alpha = torch.zeros(
            (plan.domain_count, math.prod(layout.feature_shape)),
            device=alpha_source.device,
            dtype=alpha_source.dtype,
        )
        alpha_index = (
            torch.tensor(
                layout.alpha_flat_indices, device=alpha_source.device, dtype=torch.int64
            )
            .reshape(1, -1)
            .expand(plan.domain_count, -1)
        )
        dense_alpha = flat_alpha.scatter(1, alpha_index, alpha_source)
        alphas[layout.native_preactivation] = dense_alpha.reshape(
            plan.domain_count, *layout.feature_shape
        )
        flat_beta = torch.zeros_like(flat_alpha)
        beta_locations = torch.tensor(
            layout.beta_locations, device=compressed_beta.device, dtype=torch.int64
        )
        if compressed_beta.numel():
            flat_beta = flat_beta.scatter(1, beta_locations, compressed_beta)
        betas[layout.native_preactivation] = flat_beta.reshape(
            plan.domain_count, *layout.feature_shape
        )
        splits[layout.native_preactivation] = torch.tensor(
            layout.split_values, device=compressed_alpha.device, dtype=torch.int8
        ).reshape(plan.domain_count, *layout.feature_shape)
    module = copy.copy(plan.module_template)
    module.bindings = dict(plan.module_template.bindings)
    module.bindings["params"] = params
    input_spec = InputSpec.box(
        value_name=plan.input_value_name, lower=lower, upper=upper
    )
    return module, input_spec, objective, relu_pre, alphas, betas, splits


def _evaluate_full_region(
    plan: R31FullRegionPlanV1, tensors: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    module, spec, objective, relu_pre, alphas, betas, splits = _runtime_parts(
        plan, tensors
    )
    interval_env, _ = _forward_ibp_trace_mlp(module, spec, relu_split_state=splits)
    beta_pre_add = _beta_to_relu_pre_add_coeff(
        BetaState(betas), relu_pre=relu_pre, relu_split_state=splits
    )
    bounds = run_crown_ibp_mlp_from_forward_trace(
        module,
        spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=objective,
        relu_alpha=alphas,
        relu_pre_add_coeff_l=beta_pre_add,
    )
    return bounds.lower


class _R31FullRegionFunction(torch.autograd.Function):
    """One output lower; backward rematerializes without retaining dense A."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: object,
        plan_key: str,
        execution_key: str,
        *tensors: torch.Tensor,
    ) -> torch.Tensor:
        plan = _PLAN_REGISTRY.get(plan_key)
        counters = _COUNTER_REGISTRY.get(execution_key)
        if not isinstance(plan, R31FullRegionPlanV1) or not isinstance(
            counters, R31ExecutionCountersV1
        ):
            raise RuntimeError("R3-1 registry admission differs")
        plan.validate()
        counters.begin_forward()
        ctx.set_materialize_grads(False)  # type: ignore[attr-defined]
        setattr(ctx, "plan_key", plan_key)
        setattr(ctx, "execution_key", execution_key)
        setattr(ctx, "schema_version", R31_CONTEXT_SCHEMA)
        setattr(ctx, "p_alpha_input_ordinal", plan.p_alpha_input_ordinal)
        assert_tensor_free_context(
            {
                "plan_key": plan_key,
                "execution_key": execution_key,
                "schema_version": R31_CONTEXT_SCHEMA,
                "p_alpha_input_ordinal": plan.p_alpha_input_ordinal,
            }
        )
        ctx.save_for_backward(*tensors)  # type: ignore[attr-defined]
        return _evaluate_full_region(plan, tuple(tensors))

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx: object, grad_output: torch.Tensor
    ) -> tuple[object, ...]:
        plan_key = getattr(ctx, "plan_key", None)
        execution_key = getattr(ctx, "execution_key", None)
        plan = _PLAN_REGISTRY.get(plan_key) if isinstance(plan_key, str) else None
        counters = (
            _COUNTER_REGISTRY.get(execution_key)
            if isinstance(execution_key, str)
            else None
        )
        if (
            not isinstance(plan, R31FullRegionPlanV1)
            or not isinstance(counters, R31ExecutionCountersV1)
            or getattr(ctx, "schema_version", None) != R31_CONTEXT_SCHEMA
        ):
            raise TypeError("R3-1 custom backward context differs")
        counters.begin_backward()
        saved = tuple(ctx.saved_tensors)  # type: ignore[attr-defined]
        rematerialized = [value.detach() for value in saved]
        p_ordinal = plan.p_alpha_input_ordinal
        p_alpha = rematerialized[p_ordinal].requires_grad_(True)
        rematerialized[p_ordinal] = p_alpha
        with torch.enable_grad():
            lower = _evaluate_full_region(plan, tuple(rematerialized))
            gradient = torch.autograd.grad(
                lower,
                p_alpha,
                grad_outputs=grad_output,
                retain_graph=False,
                create_graph=False,
            )[0]
        tensor_gradients: list[torch.Tensor | None] = [None] * len(saved)
        tensor_gradients[p_ordinal] = gradient
        return (None, None, *tensor_gradients)


def _saved_state_receipt(
    plan: R31FullRegionPlanV1,
    tensors: tuple[torch.Tensor, ...],
    counters: R31ExecutionCountersV1,
    *,
    alpha_versions: tuple[int, ...],
    beta_versions: tuple[int, ...],
) -> R31ExecutionReceiptV1:
    alpha_ordinals = tuple(
        ordinal
        for ordinal, spec in enumerate(plan.tensor_specs)
        if spec.role == "compressed_alpha"
    )
    beta_ordinals = tuple(
        ordinal
        for ordinal, spec in enumerate(plan.tensor_specs)
        if spec.role == "compressed_beta"
    )
    logical = sum(value.numel() * value.element_size() for value in tensors)
    storages: dict[tuple[str, int], int] = {}
    for value in tensors:
        storage = value.untyped_storage()
        key = (str(value.device), int(storage.data_ptr()))
        storages[key] = max(storages.get(key, 0), int(storage.nbytes()))
    dense_a_count = sum(
        value.ndim == 5
        and tuple(value.shape[:2]) == (plan.domain_count, plan.spec_count)
        for value in tensors
    )
    receipt = R31ExecutionReceiptV1(
        plan_hash=plan.stable_hash(),
        forward_count=counters.forward_count,
        backward_count=counters.backward_count,
        saved_tensor_count=len(tensors),
        saved_logical_bytes=logical,
        saved_unique_storage_bytes=sum(storages.values()),
        saved_dense_a_count=dense_a_count,
        scratch_slot_count=plan.scratch_slot_count,
        alpha_version_unchanged=alpha_versions
        == tuple(tensors[ordinal]._version for ordinal in alpha_ordinals),
        beta_version_unchanged=beta_versions
        == tuple(tensors[ordinal]._version for ordinal in beta_ordinals),
        fallback_count=counters.fallback_count,
        eager_escape_count=counters.eager_escape_count,
        native_shadow_count=counters.native_shadow_count,
        optimizer_mutation_count=counters.optimizer_mutation_count,
    )
    receipt.validate()
    return receipt


def execute_r31_custom_backward_v1(
    plan: R31FullRegionPlanV1, tensors: tuple[torch.Tensor, ...]
) -> R31ExecutionResultV1:
    """Run exactly one candidate forward and one rematerializing custom backward."""

    plan.validate()
    if len(tensors) != len(plan.tensor_specs):
        raise ValueError("R3-1 execution tensor count differs")
    alpha_ordinals = tuple(
        ordinal
        for ordinal, spec in enumerate(plan.tensor_specs)
        if spec.role == "compressed_alpha"
    )
    beta_ordinals = tuple(
        ordinal
        for ordinal, spec in enumerate(plan.tensor_specs)
        if spec.role == "compressed_beta"
    )
    alpha_versions = tuple(tensors[ordinal]._version for ordinal in alpha_ordinals)
    beta_versions = tuple(tensors[ordinal]._version for ordinal in beta_ordinals)
    counters = R31ExecutionCountersV1()
    plan_key = plan.stable_hash()
    registered = _PLAN_REGISTRY.setdefault(plan_key, plan)
    if registered.identity_payload() != plan.identity_payload():
        raise RuntimeError("R3-1 plan registry collision")
    execution_key = f"{plan_key}:{id(counters)}"
    if execution_key in _COUNTER_REGISTRY:
        raise RuntimeError("R3-1 execution key repeats")
    _COUNTER_REGISTRY[execution_key] = counters
    lower = _R31FullRegionFunction.apply(plan_key, execution_key, *tensors)
    p_alpha = tensors[plan.p_alpha_input_ordinal]
    try:
        compressed_gradient = torch.autograd.grad(-lower.sum(), p_alpha)[0]
    finally:
        _COUNTER_REGISTRY.pop(execution_key, None)
    counters.validate()
    receipt = _saved_state_receipt(
        plan,
        tensors,
        counters,
        alpha_versions=alpha_versions,
        beta_versions=beta_versions,
    )
    result = R31ExecutionResultV1(
        final_lower=lower.detach().contiguous().clone(),
        compressed_alpha_gradient=compressed_gradient.detach().contiguous().clone(),
        receipt=receipt,
    )
    result.validate()
    return result


def execute_r31_native_oracle_v1(
    plan: R31FullRegionPlanV1, tensors: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent eager-autograd one-evaluation oracle; never call from candidate."""

    plan.validate()
    p_ordinal = plan.p_alpha_input_ordinal
    native_values = [value.detach() for value in tensors]
    native_alpha = native_values[p_ordinal].clone().requires_grad_(True)
    native_values[p_ordinal] = native_alpha
    lower = _evaluate_full_region(plan, tuple(native_values))
    gradient = torch.autograd.grad(-lower.sum(), native_alpha)[0]
    return lower.detach().contiguous(), gradient.detach().contiguous()


__all__ = [
    "bind_r31_runtime_inputs_v1",
    "compile_r31_full_region_plan_v1",
    "execute_r31_custom_backward_v1",
    "execute_r31_native_oracle_v1",
    "R31ExecutionCountersV1",
    "R31ExecutionReceiptV1",
    "R31ExecutionResultV1",
    "R31FullRegionPlanV1",
    "R31ReluLayoutV1",
    "R31TensorSpecV1",
]
