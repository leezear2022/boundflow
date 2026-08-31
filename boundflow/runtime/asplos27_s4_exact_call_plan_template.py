"""Tensor-free AOT plan template for the S4 RVIR exact-call region.

The production exact call owns values, versions, and solver state.  This
module owns only query-independent graph/layout/shape metadata and persistent
GPU storage.  A profile snapshot may be used once by an offline compiler to
derive the template, but it is never needed by the runtime that consumes the
serialized template.
"""

# mypy: disable-error-code=import-untyped
# pylint: disable=too-many-arguments,too-many-locals,protected-access
# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=too-many-boolean-expressions,import-outside-toplevel
# pylint: disable=import-error,too-few-public-methods,duplicate-code

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, cast

import torch

from boundflow.frontends.plain_crown_bound_ir import plain_crown_primal_graph_hash
from boundflow.ir.task import BFTaskModule
from boundflow.runtime.fsg4_b3_prepared_core import PreparedCoreTemplateV1
from boundflow.runtime.r3_structured_owner_custom_backward import (
    R31FullRegionPlanV1,
    R31ReluLayoutV1,
    R31TensorSpecV1,
    compile_r31_full_region_plan_v1,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    ProductionNativePreStateV4,
    ProductionReluTopologyV4,
)
from boundflow.runtime.rvir_v4_production_state import (
    ProductionStateSnapshotV4,
    ProductionTensorRole,
    production_tensor_sha256,
)

S4_EXACT_CALL_PLAN_TEMPLATE_SCHEMA = "boundflow.asplos27-s4-exact-call-plan/v1"
S4_EXACT_CALL_TEMPLATE_BUFFER_SCHEMA = (
    "boundflow.asplos27-s4-exact-call-template-buffers/v1"
)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _topology_hash(topology: tuple[ProductionReluTopologyV4, ...]) -> str:
    for item in topology:
        item.validate()
    return _canonical_hash([item.to_dict() for item in topology])


def _one_role(
    snapshot: ProductionStateSnapshotV4, role: ProductionTensorRole
) -> torch.Tensor:
    values = [item.value for item in snapshot.tensors if item.role == role]
    if len(values) != 1:
        raise ValueError(f"S4 AOT template requires one {role.value}")
    return values[0]


def _encode(value: str) -> str:
    return value.replace("%", "%25").replace("/", "%2F")


def _snapshot_layout_signature_hash(
    snapshot: ProductionStateSnapshotV4,
    topology: tuple[ProductionReluTopologyV4, ...],
) -> str:
    """Hash precomputed snapshot digests without reading device tensor values."""

    tensor_map = snapshot.tensor_map()
    rows: list[dict[str, object]] = []
    for link in sorted(topology, key=lambda item: item.native_preactivation):
        activation = _encode(link.provider_activation)
        preactivation = _encode(link.provider_preactivation)
        layout_prefix = f"alpha_layout/{activation}"
        beta_prefix = f"beta/{preactivation}/0"
        index_items = tuple(
            sorted(
                (
                    item
                    for item in snapshot.tensors
                    if item.semantic_path.startswith(f"{layout_prefix}/feature_index/")
                ),
                key=lambda item: int(item.semantic_path.rsplit("/", 1)[1]),
            )
        )
        history = tuple(
            entry.to_dict()
            for entry in sorted(
                (
                    entry
                    for entry in snapshot.history
                    if entry.layer_name == link.provider_preactivation
                ),
                key=lambda entry: entry.domain_ordinal,
            )
        )
        required = (
            tensor_map.get(f"{layout_prefix}/feature_shape"),
            tensor_map.get(f"{beta_prefix}/location"),
            tensor_map.get(f"{beta_prefix}/sign"),
        )
        if any(item is None for item in required) or len(history) != 6:
            raise ValueError("S4 AOT layout signature coverage differs")
        rows.append(
            {
                "native_preactivation": link.native_preactivation,
                "provider_activation": link.provider_activation,
                "provider_preactivation": link.provider_preactivation,
                "provider_start_node": link.provider_start_node,
                "feature_shape_hash": required[0].content_sha256,  # type: ignore[union-attr]
                "feature_index_hashes": [item.content_sha256 for item in index_items],
                "beta_location_hash": required[1].content_sha256,  # type: ignore[union-attr]
                "beta_sign_hash": required[2].content_sha256,  # type: ignore[union-attr]
                "history": list(history),
            }
        )
    return _canonical_hash(rows)


@dataclass(frozen=True)
class S4ExactCallTensorSlotV1:
    """Shape/dtype contract for one persistent physical-plan input."""

    name: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    static_content_hash: str | None

    def validate(self) -> None:
        if (
            not self.name
            or not self.role
            or any(dimension < 0 for dimension in self.shape)
            or self.dtype != "torch.float32"
            or (
                self.static_content_hash is not None
                and not _is_sha256(self.static_content_hash)
            )
            or (
                self.name.startswith("param/") != (self.static_content_hash is not None)
            )
        ):
            raise ValueError("S4 AOT tensor slot differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "name": self.name,
            "role": self.role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "static_content_hash": self.static_content_hash,
        }


def _layout_from_dict(value: Mapping[str, object]) -> R31ReluLayoutV1:
    def strings(name: str) -> str:
        result = value.get(name)
        if not isinstance(result, str):
            raise TypeError(f"S4 AOT layout {name} differs")
        return result

    def integer_tuple(name: str) -> tuple[int, ...]:
        result = value.get(name)
        if not isinstance(result, list) or any(
            not isinstance(item, int) or isinstance(item, bool) for item in result
        ):
            raise TypeError(f"S4 AOT layout {name} differs")
        return tuple(int(item) for item in result)

    def rows(name: str) -> tuple[tuple[int, ...], ...]:
        result = value.get(name)
        if not isinstance(result, list) or any(
            not isinstance(row, list) for row in result
        ):
            raise TypeError(f"S4 AOT layout {name} differs")
        typed_rows = cast(list[list[object]], result)
        if any(
            not isinstance(item, int) or isinstance(item, bool)
            for row in typed_rows
            for item in row
        ):
            raise TypeError(f"S4 AOT layout {name} differs")
        return tuple(tuple(cast(int, item) for item in row) for row in typed_rows)

    return R31ReluLayoutV1(
        native_preactivation=strings("native_preactivation"),
        provider_activation=strings("provider_activation"),
        provider_preactivation=strings("provider_preactivation"),
        alpha_path=strings("alpha_path"),
        beta_path=strings("beta_path"),
        feature_shape=integer_tuple("feature_shape"),
        alpha_flat_indices=integer_tuple("alpha_flat_indices"),
        beta_locations=rows("beta_locations"),
        split_values=rows("split_values"),
    )


@dataclass(frozen=True)
class S4ExactCallPlanTemplateV1:  # pylint: disable=too-many-instance-attributes
    """Serializable static signature from which a persistent region is built."""

    template_id: str
    core_template_hash: str
    primal_graph_hash: str
    topology_hash: str
    layout_signature_hash: str
    device: str
    dtype: str
    input_value_name: str
    input_shape: tuple[int, ...]
    objective_shape: tuple[int, ...]
    parameter_names: tuple[str, ...]
    relu_layouts: tuple[R31ReluLayoutV1, ...]
    tensor_slots: tuple[S4ExactCallTensorSlotV1, ...]
    domain_count: int
    spec_count: int
    compute_capability: str
    schema_version: str = S4_EXACT_CALL_PLAN_TEMPLATE_SCHEMA

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "template_id": self.template_id,
            "core_template_hash": self.core_template_hash,
            "primal_graph_hash": self.primal_graph_hash,
            "topology_hash": self.topology_hash,
            "layout_signature_hash": self.layout_signature_hash,
            "device": self.device,
            "dtype": self.dtype,
            "input_value_name": self.input_value_name,
            "input_shape": list(self.input_shape),
            "objective_shape": list(self.objective_shape),
            "parameter_names": list(self.parameter_names),
            "relu_layouts": [layout.to_dict() for layout in self.relu_layouts],
            "tensor_slots": [slot.to_dict() for slot in self.tensor_slots],
            "domain_count": self.domain_count,
            "spec_count": self.spec_count,
            "compute_capability": self.compute_capability,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.identity_payload())

    def to_dict(self) -> dict[str, object]:
        payload = self.identity_payload()
        payload["template_hash"] = self.stable_hash()
        return payload

    def validate(
        self,
        *,
        core_template: PreparedCoreTemplateV1 | None = None,
        topology: tuple[ProductionReluTopologyV4, ...] | None = None,
    ) -> None:
        if (
            self.schema_version != S4_EXACT_CALL_PLAN_TEMPLATE_SCHEMA
            or not self.template_id
            or not _is_sha256(self.core_template_hash)
            or not _is_sha256(self.primal_graph_hash)
            or not _is_sha256(self.topology_hash)
            or not _is_sha256(self.layout_signature_hash)
            or not self.device.startswith("cuda:")
            or self.dtype != "torch.float32"
            or not self.input_value_name
            or not self.input_shape
            or not self.objective_shape
            or any(dimension <= 0 for dimension in self.input_shape)
            or any(dimension <= 0 for dimension in self.objective_shape)
            or tuple(sorted(self.parameter_names)) != self.parameter_names
            or len(set(self.parameter_names)) != len(self.parameter_names)
            or tuple(layout.native_preactivation for layout in self.relu_layouts)
            != ("17", "19", "23", "25", "28", "31")
            or len(self.tensor_slots) != 3 + len(self.parameter_names) + 24
            or len({slot.name for slot in self.tensor_slots}) != len(self.tensor_slots)
            or self.domain_count != 6
            or self.spec_count != 1
            or not self.compute_capability.startswith("sm_")
        ):
            raise ValueError("S4 AOT plan template identity differs")
        for layout in self.relu_layouts:
            layout.validate(domain_count=self.domain_count)
        for slot in self.tensor_slots:
            slot.validate()
        parameter_slots = tuple(
            slot.name.removeprefix("param/")
            for slot in self.tensor_slots
            if slot.name.startswith("param/")
        )
        if parameter_slots != self.parameter_names:
            raise ValueError("S4 AOT parameter slot order differs")
        if core_template is not None:
            core_template.validate(topology=topology)
            if (
                core_template.stable_hash() != self.core_template_hash
                or core_template.primal_graph_hash != self.primal_graph_hash
                or core_template.topology_hash != self.topology_hash
                or core_template.device != self.device
                or core_template.dtype != self.dtype
                or core_template.input_value_name != self.input_value_name
                or core_template.input_shape != self.input_shape
                or core_template.objective_shape != self.objective_shape
            ):
                raise ValueError("S4 AOT/core template binding differs")
            params = core_template.module.bindings.get("params")
            if not isinstance(params, Mapping):
                raise ValueError("S4 AOT core parameter binding differs")
            for slot in self.tensor_slots:
                if not slot.name.startswith("param/"):
                    continue
                parameter = params.get(slot.name.removeprefix("param/"))
                if not torch.is_tensor(parameter):
                    raise ValueError(f"S4 AOT parameter differs: {slot.name}")
                tensor = cast(torch.Tensor, parameter)
                if (
                    tuple(tensor.shape) != slot.shape
                    or str(tensor.dtype) != slot.dtype
                    or production_tensor_sha256(tensor) != slot.static_content_hash
                ):
                    raise ValueError(f"S4 AOT parameter differs: {slot.name}")
        if topology is not None and _topology_hash(topology) != self.topology_hash:
            raise ValueError("S4 AOT topology differs")

    def validate_dynamic_signature(
        self,
        *,
        module: BFTaskModule,
        snapshot: ProductionStateSnapshotV4,
        mapping: ProductionNativePreStateV4,
        topology: tuple[ProductionReluTopologyV4, ...],
    ) -> None:
        """Reject a query whose sparse layout or physical slot schema drifts."""

        if (
            plain_crown_primal_graph_hash(module) != self.primal_graph_hash
            or mapping.identity.topology_hash != self.topology_hash
            or _topology_hash(topology) != self.topology_hash
        ):
            raise ValueError("S4 AOT dynamic graph/topology differs")
        if (
            _snapshot_layout_signature_hash(snapshot, topology)
            != self.layout_signature_hash
        ):
            raise ValueError("S4 AOT dynamic sparse layout differs")
        params = module.bindings.get("params")
        if not isinstance(params, Mapping):
            raise ValueError("S4 AOT dynamic parameters differ")
        tensor_map = snapshot.tensor_map()
        values: list[torch.Tensor] = [
            _one_role(snapshot, ProductionTensorRole.INPUT_LOWER),
            _one_role(snapshot, ProductionTensorRole.INPUT_UPPER),
            _one_role(snapshot, ProductionTensorRole.LINEAR_SPEC),
        ]
        values.extend(cast(torch.Tensor, params[name]) for name in self.parameter_names)
        for layout in self.relu_layouts:
            encoded = _encode(layout.provider_preactivation)
            values.extend(
                (
                    tensor_map[f"intermediate/{encoded}/lower"].value,
                    tensor_map[f"intermediate/{encoded}/upper"].value,
                    tensor_map[layout.alpha_path].value,
                    tensor_map[layout.beta_path].value,
                )
            )
        if len(values) != len(self.tensor_slots):
            raise ValueError("S4 AOT dynamic tensor count differs")
        for slot, value in zip(self.tensor_slots, values):
            if tuple(value.shape) != slot.shape or str(value.dtype) != slot.dtype:
                raise ValueError(f"S4 AOT dynamic tensor schema differs: {slot.name}")


def compile_s4_exact_call_plan_template_v1(
    *,
    template_id: str,
    core_template: PreparedCoreTemplateV1,
    snapshot: ProductionStateSnapshotV4,
    mapping: ProductionNativePreStateV4,
    topology: tuple[ProductionReluTopologyV4, ...],
    compute_capability: str,
) -> S4ExactCallPlanTemplateV1:
    """Compile a tensor-free AOT template from one admitted shape profile."""

    core_template.validate(topology=topology)
    plan = compile_r31_full_region_plan_v1(
        core_template.module, snapshot, mapping, topology
    )
    slots = tuple(
        S4ExactCallTensorSlotV1(
            name=spec.name,
            role=spec.role,
            shape=spec.shape,
            dtype=spec.dtype,
            static_content_hash=(
                spec.content_sha256 if spec.name.startswith("param/") else None
            ),
        )
        for spec in plan.tensor_specs
    )
    result = S4ExactCallPlanTemplateV1(
        template_id=template_id,
        core_template_hash=core_template.stable_hash(),
        primal_graph_hash=plan.primal_graph_hash,
        topology_hash=core_template.topology_hash,
        layout_signature_hash=_snapshot_layout_signature_hash(snapshot, topology),
        device=core_template.device,
        dtype=core_template.dtype,
        input_value_name=plan.input_value_name,
        input_shape=core_template.input_shape,
        objective_shape=core_template.objective_shape,
        parameter_names=plan.parameter_names,
        relu_layouts=plan.relu_layouts,
        tensor_slots=slots,
        domain_count=plan.domain_count,
        spec_count=plan.spec_count,
        compute_capability=compute_capability,
    )
    result.validate(core_template=core_template, topology=topology)
    return result


def s4_exact_call_plan_template_from_dict_v1(
    value: Mapping[str, object],
) -> S4ExactCallPlanTemplateV1:
    def text(name: str) -> str:
        result = value.get(name)
        if not isinstance(result, str):
            raise TypeError(f"S4 AOT template {name} differs")
        return result

    def integer_tuple(name: str) -> tuple[int, ...]:
        result = value.get(name)
        if not isinstance(result, list) or any(
            not isinstance(item, int) or isinstance(item, bool) for item in result
        ):
            raise TypeError(f"S4 AOT template {name} differs")
        return tuple(int(item) for item in result)

    raw_parameters = value.get("parameter_names")
    raw_layouts = value.get("relu_layouts")
    raw_slots = value.get("tensor_slots")
    if (
        not isinstance(raw_parameters, list)
        or any(not isinstance(item, str) for item in raw_parameters)
        or not isinstance(raw_layouts, list)
        or any(not isinstance(item, Mapping) for item in raw_layouts)
        or not isinstance(raw_slots, list)
        or any(not isinstance(item, Mapping) for item in raw_slots)
    ):
        raise TypeError("S4 AOT template collections differ")
    slots = []
    for raw in cast(list[Mapping[str, object]], raw_slots):
        shape = raw.get("shape")
        static_hash = raw.get("static_content_hash")
        if (
            not isinstance(raw.get("name"), str)
            or not isinstance(raw.get("role"), str)
            or not isinstance(shape, list)
            or any(
                not isinstance(item, int) or isinstance(item, bool) for item in shape
            )
            or not isinstance(raw.get("dtype"), str)
            or (static_hash is not None and not isinstance(static_hash, str))
        ):
            raise TypeError("S4 AOT tensor slot payload differs")
        slots.append(
            S4ExactCallTensorSlotV1(
                name=cast(str, raw["name"]),
                role=cast(str, raw["role"]),
                shape=tuple(int(item) for item in shape),
                dtype=cast(str, raw["dtype"]),
                static_content_hash=cast(str | None, static_hash),
            )
        )
    raw_domain_count = value.get("domain_count")
    raw_spec_count = value.get("spec_count")
    if (
        not isinstance(raw_domain_count, int)
        or isinstance(raw_domain_count, bool)
        or not isinstance(raw_spec_count, int)
        or isinstance(raw_spec_count, bool)
    ):
        raise TypeError("S4 AOT template domain/spec count differs")
    result = S4ExactCallPlanTemplateV1(
        template_id=text("template_id"),
        core_template_hash=text("core_template_hash"),
        primal_graph_hash=text("primal_graph_hash"),
        topology_hash=text("topology_hash"),
        layout_signature_hash=text("layout_signature_hash"),
        device=text("device"),
        dtype=text("dtype"),
        input_value_name=text("input_value_name"),
        input_shape=integer_tuple("input_shape"),
        objective_shape=integer_tuple("objective_shape"),
        parameter_names=tuple(cast(list[str], raw_parameters)),
        relu_layouts=tuple(
            _layout_from_dict(cast(Mapping[str, object], item)) for item in raw_layouts
        ),
        tensor_slots=tuple(slots),
        domain_count=raw_domain_count,
        spec_count=raw_spec_count,
        compute_capability=text("compute_capability"),
        schema_version=text("schema_version"),
    )
    result.validate()
    if value.get("template_hash") != result.stable_hash():
        raise ValueError("S4 AOT serialized template hash differs")
    return result


def load_s4_exact_call_plan_template_v1(
    path: Path,
    *,
    core_template: PreparedCoreTemplateV1,
    topology: tuple[ProductionReluTopologyV4, ...],
) -> S4ExactCallPlanTemplateV1:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("S4 AOT template root differs")
    result = s4_exact_call_plan_template_from_dict_v1(cast(Mapping[str, object], raw))
    result.validate(core_template=core_template, topology=topology)
    return result


def instantiate_s4_physical_plan_v1(
    template: S4ExactCallPlanTemplateV1,
    *,
    core_template: PreparedCoreTemplateV1,
    topology: tuple[ProductionReluTopologyV4, ...],
) -> tuple[R31FullRegionPlanV1, tuple[torch.Tensor, ...]]:
    """Allocate seed storage for a persistent region without query state."""

    template.validate(core_template=core_template, topology=topology)
    device = torch.device(template.device)
    params = core_template.module.bindings.get("params")
    if not isinstance(params, Mapping):
        raise ValueError("S4 AOT physical parameters differ")
    tensors: list[torch.Tensor] = []
    specs: list[R31TensorSpecV1] = []
    for slot in template.tensor_slots:
        if slot.name.startswith("param/"):
            tensor = cast(
                torch.Tensor, params[slot.name.removeprefix("param/")]
            ).contiguous()
        else:
            tensor = torch.zeros(slot.shape, dtype=torch.float32, device=device)
        if tuple(tensor.shape) != slot.shape or str(tensor.dtype) != slot.dtype:
            raise ValueError(f"S4 AOT physical tensor differs: {slot.name}")
        tensors.append(tensor)
        specs.append(
            R31TensorSpecV1(
                name=slot.name,
                role=slot.role,
                shape=slot.shape,
                dtype=slot.dtype,
                content_sha256=production_tensor_sha256(tensor),
            )
        )
    module_template = copy.copy(core_template.module)
    module_template.bindings = {
        name: copy.deepcopy(value)
        for name, value in core_template.module.bindings.items()
        if name != "params"
    }
    module_template.bindings["params"] = {}
    plan = R31FullRegionPlanV1(
        module_template=module_template,
        primal_graph_hash=template.primal_graph_hash,
        source_state_hash=template.stable_hash(),
        input_value_name=template.input_value_name,
        parameter_names=template.parameter_names,
        relu_layouts=template.relu_layouts,
        tensor_specs=tuple(specs),
        domain_count=template.domain_count,
        spec_count=template.spec_count,
    )
    plan.validate()
    return plan, tuple(tensors)


@dataclass(frozen=True)
class S4ExactCallTemplateBufferReceiptV1:
    """Truthful receipt for query-independent persistent buffer allocation."""

    plan_template_hash: str
    physical_plan_hash: str
    exact_call_id: str
    device: str
    parameter_shapes: tuple[tuple[int, ...], ...]
    gradient_shapes: tuple[tuple[int, ...], ...]
    candidate_storage_count: int
    base_dlpack_view_count: int
    static_source_copy_count: int
    static_source_copy_bytes: int
    fallback_count: int = 0
    performance_claimed: bool = False
    schema_version: str = S4_EXACT_CALL_TEMPLATE_BUFFER_SCHEMA

    def validate(self) -> None:
        widths = (164, 132, 121, 86, 178, 27)
        expected = tuple((6, width) for width in widths) + ((6, 1),)
        if (
            self.schema_version != S4_EXACT_CALL_TEMPLATE_BUFFER_SCHEMA
            or not _is_sha256(self.plan_template_hash)
            or not _is_sha256(self.physical_plan_hash)
            or not self.exact_call_id
            or not self.device.startswith("cuda:")
            or self.parameter_shapes != expected
            or self.gradient_shapes != expected
            or self.candidate_storage_count != 16
            or self.base_dlpack_view_count != 16
            or self.static_source_copy_count
            or self.static_source_copy_bytes
            or self.fallback_count
            or self.performance_claimed
        ):
            raise ValueError("S4 AOT template buffer receipt differs")

    def stable_hash(self) -> str:
        self.validate()
        return _canonical_hash(asdict(self))

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["receipt_hash"] = self.stable_hash()
        return payload


class _S4TemplateBufferResourcesV1:
    """Minimal resource owner matching the existing prepared evaluator ABI."""

    def __init__(self, plan: R31FullRegionPlanV1, device: torch.device) -> None:
        import tvm

        self._parameters: list[torch.Tensor] = [
            torch.zeros(
                (plan.domain_count, len(layout.alpha_flat_indices)),
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            for layout in plan.relu_layouts
        ]
        active = [layout for layout in plan.relu_layouts if any(layout.beta_locations)]
        if len(active) != 1:
            raise ValueError("S4 AOT active beta inventory differs")
        beta_widths = {len(row) for row in active[0].beta_locations}
        if beta_widths != {1}:
            raise ValueError("S4 AOT active beta width differs")
        self._parameters.append(
            torch.zeros(
                (plan.domain_count, 1),
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
        )
        self._gradients: list[torch.Tensor] = [
            torch.empty_like(value) for value in self._parameters
        ]
        self._lower: torch.Tensor | None = torch.empty(
            plan.domain_count, dtype=torch.float32, device=device
        )
        self._upstream: torch.Tensor | None = torch.full(
            (plan.domain_count, 1), -1.0, dtype=torch.float32, device=device
        )
        tensors = (*self._parameters, *self._gradients, self._lower, self._upstream)
        if (
            len(tensors) != 16
            or len({int(value.untyped_storage()._cdata) for value in tensors}) != 16
        ):
            raise ValueError("S4 AOT persistent buffer storage aliases")
        self._views = [tvm.runtime.from_dlpack(value) for value in tensors]
        self._private_view_keys = [
            (
                value.data_ptr(),
                tuple(value.shape),
                tuple(value.stride()),
                str(value.dtype),
                str(value.device),
            )
            for value in tensors
        ]
        self._initialized = [True] * 7 + [False] * 8 + [True]
        self._state = "PREPARED"

    def close(self) -> None:
        self._views.clear()
        self._private_view_keys.clear()
        self._upstream = None
        self._lower = None
        self._gradients.clear()
        self._parameters.clear()
        self._initialized.clear()
        self._state = "CLOSED"


class PreparedS4ExactCallTemplateBuffersV1:
    """Persistent candidate storage created without a production snapshot."""

    def __init__(
        self,
        plan: R31FullRegionPlanV1,
        template: S4ExactCallPlanTemplateV1,
        *,
        exact_call_id: str,
    ) -> None:
        device = torch.device(template.device)
        resources = _S4TemplateBufferResourcesV1(plan, device)
        self._resources: _S4TemplateBufferResourcesV1 | None = resources
        parameter_shapes = tuple(tuple(value.shape) for value in resources._parameters)
        gradient_shapes = tuple(tuple(value.shape) for value in resources._gradients)
        self.receipt = S4ExactCallTemplateBufferReceiptV1(
            plan_template_hash=template.stable_hash(),
            physical_plan_hash=plan.stable_hash(),
            exact_call_id=exact_call_id,
            device=template.device,
            parameter_shapes=parameter_shapes,
            gradient_shapes=gradient_shapes,
            candidate_storage_count=16,
            base_dlpack_view_count=16,
            static_source_copy_count=0,
            static_source_copy_bytes=0,
        )
        self.receipt.validate()

    def close(self) -> None:
        if self._resources is not None:
            self._resources.close()
            self._resources = None


__all__ = [
    "compile_s4_exact_call_plan_template_v1",
    "instantiate_s4_physical_plan_v1",
    "load_s4_exact_call_plan_template_v1",
    "PreparedS4ExactCallTemplateBuffersV1",
    "S4ExactCallPlanTemplateV1",
    "S4ExactCallTemplateBufferReceiptV1",
    "S4ExactCallTensorSlotV1",
    "s4_exact_call_plan_template_from_dict_v1",
]
