"""Typed static-template/dynamic-instance boundary for FSG4/B3-A."""

# pylint: disable=too-many-arguments,too-many-instance-attributes
# pylint: disable=too-many-locals,missing-function-docstring
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, cast, Mapping, Sequence

import torch

from ..frontends.plain_crown_bound_ir import (
    plain_crown_primal_graph_hash,
    relu_split_state_hash,
    tensor_content_hash,
)
from ..ir.task import BFTaskModule
from .native_alpha_beta_optimization_state import (
    build_native_alpha_beta_scope,
    NativeAlphaBetaOptimizationState,
    NativeAlphaBetaStateScope,
)
from .rvir_v4_optimizer_mutation import ProductionMutationPolicyV4
from .rvir_v4_pre_state_initializer import (
    ProductionNativePreStateV4,
    ProductionReluTopologyV4,
)
from .rvir_v4_production_state import (
    ProductionStateSnapshotV4,
    ProductionTensorOwnership,
)
from .task_executor import InputSpec

FSG4_B3_PREPARED_CORE_SCHEMA = "boundflow.fsg4-b3-prepared-core/v1"
FSG4_B3_POLICY_CONTRACT = "rvir-v4-production-mutation/admitted-v1"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_device(value: torch.device | str) -> str:
    return str(torch.device(value))


def _canonical_dtype(value: torch.dtype) -> str:
    if not isinstance(value, torch.dtype):
        raise TypeError("FSG4/B3-A target dtype must be torch.dtype")
    if not value.is_floating_point:
        raise ValueError("FSG4/B3-A target dtype must be floating point")
    return str(value)


def _topology_hash(topology: tuple[ProductionReluTopologyV4, ...]) -> str:
    if len(topology) != 6:
        raise ValueError("FSG4/B3-A topology count differs")
    for item in topology:
        item.validate()
    if (
        len({item.provider_activation for item in topology}) != len(topology)
        or len({item.provider_preactivation for item in topology}) != len(topology)
        or len({item.native_preactivation for item in topology}) != len(topology)
    ):
        raise ValueError("FSG4/B3-A topology keys repeat")
    return _canonical_hash([item.to_dict() for item in topology])


def _move_binding_value(
    value: object, *, device: torch.device, dtype: torch.dtype
) -> object:
    if torch.is_tensor(value):
        tensor = cast(torch.Tensor, value)
        return tensor.to(
            device=device, dtype=dtype if tensor.is_floating_point() else None
        )
    if isinstance(value, dict):
        return {
            key: _move_binding_value(item, device=device, dtype=dtype)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_binding_value(item, device=device, dtype=dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _move_binding_value(item, device=device, dtype=dtype) for item in value
        )
    return value


def _tensor_leaves(
    value: object, *, path: str = "bindings"
) -> list[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        return [(path, value)]
    if isinstance(value, Mapping):
        leaves: list[tuple[str, torch.Tensor]] = []
        for key in sorted(value, key=str):
            leaves.extend(_tensor_leaves(value[key], path=f"{path}/{key}"))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(_tensor_leaves(item, path=f"{path}/{index}"))
        return leaves
    return []


@dataclass(frozen=True)
class PreparedBindingV1:
    """One immutable binding leaf expected by a prepared core template."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    content_sha256: str

    def validate(self) -> None:
        if (
            not self.path
            or not self.dtype
            or not self.device
            or any(dimension < 0 for dimension in self.shape)
            or not _is_sha256(self.content_sha256)
        ):
            raise ValueError("FSG4/B3-A prepared binding differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "path": self.path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "content_sha256": self.content_sha256,
        }


def _binding_inventory(module: BFTaskModule) -> tuple[PreparedBindingV1, ...]:
    rows = tuple(
        PreparedBindingV1(
            path=path,
            shape=tuple(int(item) for item in value.shape),
            dtype=str(value.dtype),
            device=str(value.device),
            content_sha256=tensor_content_hash(value),
        )
        for path, value in _tensor_leaves(module.bindings)
    )
    if not rows or len({row.path for row in rows}) != len(rows):
        raise ValueError("FSG4/B3-A module binding inventory differs")
    for row in rows:
        row.validate()
    return rows


def _mutable_paths(snapshot: ProductionStateSnapshotV4) -> tuple[str, ...]:
    snapshot.validate()
    paths = tuple(
        sorted(
            tensor.semantic_path
            for tensor in snapshot.tensors
            if tensor.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
        )
    )
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("FSG4/B3-A mutable path inventory differs")
    return paths


@dataclass(frozen=True)
class PreparedCoreTemplateV1:  # pylint: disable=too-many-instance-attributes
    """Static graph/device/policy-shape ownership prepared outside the core."""

    template_id: str
    primal_graph_hash: str
    topology_hash: str
    device: str
    dtype: str
    input_value_name: str
    input_shape: tuple[int, ...]
    objective_shape: tuple[int, ...]
    mutable_paths: tuple[str, ...]
    binding_inventory: tuple[PreparedBindingV1, ...]
    program: Any
    module: BFTaskModule
    policy_contract: str = FSG4_B3_POLICY_CONTRACT
    schema_version: str = FSG4_B3_PREPARED_CORE_SCHEMA

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "template_id": self.template_id,
            "primal_graph_hash": self.primal_graph_hash,
            "topology_hash": self.topology_hash,
            "device": self.device,
            "dtype": self.dtype,
            "input_value_name": self.input_value_name,
            "input_shape": list(self.input_shape),
            "objective_shape": list(self.objective_shape),
            "mutable_paths": list(self.mutable_paths),
            "binding_inventory": [row.to_dict() for row in self.binding_inventory],
            "policy_contract": self.policy_contract,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.identity_payload())

    def validate(
        self, *, topology: tuple[ProductionReluTopologyV4, ...] | None = None
    ) -> None:
        if (
            self.schema_version != FSG4_B3_PREPARED_CORE_SCHEMA
            or not self.template_id
            or self.policy_contract != FSG4_B3_POLICY_CONTRACT
            or not _is_sha256(self.primal_graph_hash)
            or not _is_sha256(self.topology_hash)
            or not self.device
            or not self.dtype
            or not self.input_value_name
            or not self.input_shape
            or not self.objective_shape
            or any(dimension <= 0 for dimension in self.input_shape)
            or any(dimension <= 0 for dimension in self.objective_shape)
            or not self.mutable_paths
            or tuple(sorted(self.mutable_paths)) != self.mutable_paths
            or len(set(self.mutable_paths)) != len(self.mutable_paths)
            or not self.binding_inventory
        ):
            raise ValueError("FSG4/B3-A prepared template identity differs")
        self.module.validate()
        graph_inputs = tuple(self.program.graph.inputs)
        if not graph_inputs or graph_inputs[0] != self.input_value_name:
            raise ValueError("FSG4/B3-A prepared program input differs")
        if plain_crown_primal_graph_hash(self.module) != self.primal_graph_hash:
            raise ValueError("FSG4/B3-A prepared module graph is stale")
        observed_bindings = _binding_inventory(self.module)
        if observed_bindings != self.binding_inventory:
            raise ValueError("FSG4/B3-A prepared module bindings are stale")
        if any(
            row.device != self.device
            or (row.dtype.startswith("torch.float") and row.dtype != self.dtype)
            for row in observed_bindings
        ):
            raise ValueError("FSG4/B3-A prepared binding placement differs")
        if topology is not None and _topology_hash(topology) != self.topology_hash:
            raise ValueError("FSG4/B3-A prepared topology differs")


def prepare_core_template_v1(
    *,
    template_id: str,
    program: Any,
    module: BFTaskModule,
    topology: tuple[ProductionReluTopologyV4, ...],
    device: torch.device | str,
    dtype: torch.dtype,
    input_shape: Sequence[int],
    objective_shape: Sequence[int],
    mutable_paths: Sequence[str],
) -> PreparedCoreTemplateV1:
    """Move static bindings once and seal their exact reusable identity."""

    module.validate()
    target_device = torch.device(device)
    target_dtype = _canonical_dtype(dtype)
    module.bindings = cast(
        dict[str, Any],
        _move_binding_value(module.bindings, device=target_device, dtype=dtype),
    )
    template = PreparedCoreTemplateV1(
        template_id=template_id,
        primal_graph_hash=plain_crown_primal_graph_hash(module),
        topology_hash=_topology_hash(topology),
        device=_canonical_device(target_device),
        dtype=target_dtype,
        input_value_name=str(program.graph.inputs[0]),
        input_shape=tuple(int(item) for item in input_shape),
        objective_shape=tuple(int(item) for item in objective_shape),
        mutable_paths=tuple(sorted(str(path) for path in mutable_paths)),
        binding_inventory=_binding_inventory(module),
        program=program,
        module=module,
    )
    template.validate(topology=topology)
    return template


class PreparedCoreTemplateCache:
    """Exact-hash cache with observable miss/compile and hit cardinalities."""

    def __init__(self) -> None:
        self._templates: dict[str, PreparedCoreTemplateV1] = {}
        self.compile_count = 0
        self.hit_count = 0

    def insert(self, template: PreparedCoreTemplateV1) -> str:
        template.validate()
        key = template.stable_hash()
        if key in self._templates:
            raise ValueError("FSG4/B3-A prepared template is already cached")
        self._templates[key] = template
        self.compile_count += 1
        return key

    def resolve(
        self,
        template_hash: str,
        *,
        topology: tuple[ProductionReluTopologyV4, ...],
    ) -> PreparedCoreTemplateV1:
        if not _is_sha256(template_hash) or template_hash not in self._templates:
            raise KeyError("FSG4/B3-A prepared template cache miss")
        template = self._templates[template_hash]
        template.validate(topology=topology)
        if template.stable_hash() != template_hash:
            raise ValueError("FSG4/B3-A prepared template hash is stale")
        self.hit_count += 1
        return template


@dataclass(frozen=True)
class CorePlanInstanceV1:  # pylint: disable=too-many-instance-attributes
    """Dynamic query/state binding to one exact prepared template."""

    template_hash: str
    snapshot_hash: str
    mapping_hash: str
    mutation_policy_hash: str
    scope: NativeAlphaBetaStateScope
    initial_state: NativeAlphaBetaOptimizationState
    instance_hash: str
    schema_version: str = FSG4_B3_PREPARED_CORE_SCHEMA

    def validate(
        self,
        *,
        template: PreparedCoreTemplateV1,
        snapshot: ProductionStateSnapshotV4,
        mapping: ProductionNativePreStateV4,
        input_spec: InputSpec,
        linear_spec_C: torch.Tensor,
        mutation_policy: ProductionMutationPolicyV4,
    ) -> None:
        template.validate()
        snapshot.validate()
        mapping.validate()
        mutation_policy.validate()
        self.scope.validate()
        self.initial_state.validate()
        lower, upper = input_spec.perturbation.bounding_box(input_spec.center)
        observed_payload = {
            "schema_version": self.schema_version,
            "template_hash": template.stable_hash(),
            "snapshot_hash": snapshot.stable_hash(),
            "mapping_hash": mapping.stable_hash(),
            "mutation_policy_hash": mutation_policy.stable_hash(),
            "scope_hash": self.scope.stable_hash(),
            "initial_state_hash": self.initial_state.stable_hash(),
        }
        if (
            self.schema_version != FSG4_B3_PREPARED_CORE_SCHEMA
            or self.template_hash != observed_payload["template_hash"]
            or self.snapshot_hash != observed_payload["snapshot_hash"]
            or self.mapping_hash != observed_payload["mapping_hash"]
            or self.mutation_policy_hash != observed_payload["mutation_policy_hash"]
            or self.instance_hash != _canonical_hash(observed_payload)
            or input_spec.value_name != template.input_value_name
            or tuple(lower.shape) != template.input_shape
            or tuple(upper.shape) != template.input_shape
            or tuple(linear_spec_C.shape) != template.objective_shape
            or str(lower.device) != template.device
            or str(upper.device) != template.device
            or str(linear_spec_C.device) != template.device
            or str(lower.dtype) != template.dtype
            or str(upper.dtype) != template.dtype
            or str(linear_spec_C.dtype) != template.dtype
            or _mutable_paths(snapshot) != template.mutable_paths
            or mapping.identity.topology_hash != template.topology_hash
            or self.scope.primal_graph_hash != template.primal_graph_hash
            or self.scope.objective_hash != tensor_content_hash(linear_spec_C)
            or self.scope.split_state_hash != relu_split_state_hash(mapping.splits)
            or self.scope.optimizer_policy_hash
            != mutation_policy.to_native_policy().stable_hash()
            or self.initial_state.scope != self.scope
        ):
            raise ValueError("FSG4/B3-A core plan instance differs")


def instantiate_core_plan_v1(
    *,
    template: PreparedCoreTemplateV1,
    topology: tuple[ProductionReluTopologyV4, ...],
    snapshot: ProductionStateSnapshotV4,
    mapping: ProductionNativePreStateV4,
    input_spec: InputSpec,
    linear_spec_C: torch.Tensor,
    mutation_policy: ProductionMutationPolicyV4,
) -> CorePlanInstanceV1:
    """Bind all dynamic state and construct the exact native scope once."""

    template.validate(topology=topology)
    snapshot.validate()
    mapping.validate()
    mutation_policy.validate()
    if _mutable_paths(snapshot) != template.mutable_paths:
        raise ValueError("FSG4/B3-A dynamic mutable inventory differs")
    scope = build_native_alpha_beta_scope(
        template.module,
        input_spec,
        linear_spec_C=linear_spec_C,
        relu_pre=mapping.relu_pre,
        relu_split_state=mapping.splits,
        policy=mutation_policy.to_native_policy(),
    )
    initial_state = mapping.to_native_state(scope)
    payload = {
        "schema_version": FSG4_B3_PREPARED_CORE_SCHEMA,
        "template_hash": template.stable_hash(),
        "snapshot_hash": snapshot.stable_hash(),
        "mapping_hash": mapping.stable_hash(),
        "mutation_policy_hash": mutation_policy.stable_hash(),
        "scope_hash": scope.stable_hash(),
        "initial_state_hash": initial_state.stable_hash(),
    }
    instance = CorePlanInstanceV1(
        template_hash=str(payload["template_hash"]),
        snapshot_hash=str(payload["snapshot_hash"]),
        mapping_hash=str(payload["mapping_hash"]),
        mutation_policy_hash=str(payload["mutation_policy_hash"]),
        scope=scope,
        initial_state=initial_state,
        instance_hash=_canonical_hash(payload),
    )
    instance.validate(
        template=template,
        snapshot=snapshot,
        mapping=mapping,
        input_spec=input_spec,
        linear_spec_C=linear_spec_C,
        mutation_policy=mutation_policy,
    )
    return instance


__all__ = [
    "CorePlanInstanceV1",
    "FSG4_B3_POLICY_CONTRACT",
    "FSG4_B3_PREPARED_CORE_SCHEMA",
    "instantiate_core_plan_v1",
    "PreparedBindingV1",
    "PreparedCoreTemplateCache",
    "PreparedCoreTemplateV1",
    "prepare_core_template_v1",
]
