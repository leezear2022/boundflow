"""R3-0 first-class structured lower-region contracts and validators."""

# pylint: disable=missing-function-docstring,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,too-many-locals,too-many-lines

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Mapping

import torch

TEMPLATE_SCHEMA = "boundflow.structured-lower-region-template/v1"
INSTANCE_SCHEMA = "boundflow.structured-lower-region-instance/v1"
RECEIPT_SCHEMA = "boundflow.structured-lower-region-r3-0-receipt/v1"
MAX_NODE_EXPANSION = 4
MAX_SCRATCH_SLOTS = 2


class StructuredDenseEscapeError(RuntimeError):
    """Raised when production code tries to materialize a structured handle."""


class StructuredLowerOpKind(str, Enum):
    """Closed R3 v1 lower-region node vocabulary."""

    SPEC_SEED = "spec_seed"
    RELU_LOWER_TRANSFORM = "relu_lower_transform"
    LINEAR_RIGHT = "linear_right"
    CONV2D_RIGHT = "conv2d_right"
    ADD = "add"
    RESHAPE = "reshape"
    SLICE = "slice"
    BIAS_SPLIT = "bias_split"
    INPUT_CONCRETIZE = "input_concretize"


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


def _require_exact(
    payload: Mapping[str, object], expected: set[str], name: str
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} fields differ")


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _shape(value: object, name: str, *, allow_zero: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty integer list")
    minimum = 0 if allow_zero else 1
    return tuple(_integer(item, name, minimum=minimum) for item in value)


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a string list")
    result = tuple(_string(item, name) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique")
    return result


@dataclass(frozen=True)
class StructuredLowerAttributeV1:
    """Typed node attribute without an unbounded ``Any`` payload."""

    name: str
    integers: tuple[int, ...] = ()
    text: str = ""
    flag: bool | None = None

    def validate(self) -> None:
        if not self.name or any(not isinstance(value, int) for value in self.integers):
            raise ValueError("structured attribute differs")
        populated = (
            int(bool(self.integers)) + int(bool(self.text)) + int(self.flag is not None)
        )
        if populated != 1:
            raise ValueError("structured attribute must have exactly one typed value")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "name": self.name,
            "integers": list(self.integers),
            "text": self.text,
            "flag": self.flag,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StructuredLowerAttributeV1":
        _require_exact(payload, {"name", "integers", "text", "flag"}, "attribute")
        integers = payload["integers"]
        if not isinstance(integers, list):
            raise ValueError("attribute integers must be a list")
        flag = payload["flag"]
        if flag is not None and not isinstance(flag, bool):
            raise ValueError("attribute flag must be boolean or null")
        result = cls(
            name=_string(payload["name"], "attribute name"),
            integers=tuple(_integer(value, "attribute integer") for value in integers),
            text=payload["text"] if isinstance(payload["text"], str) else "",
            flag=flag,
        )
        if not isinstance(payload["text"], str):
            raise ValueError("attribute text must be a string")
        result.validate()
        return result


@dataclass(frozen=True)
class StructuredLowerNodeV1:
    """One immutable node in a lower-coefficient DAG."""

    node_id: str
    ordinal: int
    op_kind: StructuredLowerOpKind
    input_ids: tuple[str, ...]
    output_shape: tuple[int, ...]
    source_op_ids: tuple[str, ...]
    declared_consumer_count: int
    external_consumer_count: int = 0
    attributes: tuple[StructuredLowerAttributeV1, ...] = ()

    def validate_local(self) -> None:
        if (
            not self.node_id
            or self.ordinal < 0
            or not self.output_shape
            or any(dimension <= 0 for dimension in self.output_shape)
            or not self.source_op_ids
            or len(set(self.source_op_ids)) != len(self.source_op_ids)
            or self.declared_consumer_count < 0
            or self.external_consumer_count not in (0, 1)
        ):
            raise ValueError("structured lower node differs")
        if len({attribute.name for attribute in self.attributes}) != len(
            self.attributes
        ):
            raise ValueError("structured node attribute names must be unique")
        for attribute in self.attributes:
            attribute.validate()

        arity = len(self.input_ids)
        expected_arities: dict[StructuredLowerOpKind, tuple[int, ...]] = {
            StructuredLowerOpKind.SPEC_SEED: (0,),
            StructuredLowerOpKind.RELU_LOWER_TRANSFORM: (1,),
            StructuredLowerOpKind.LINEAR_RIGHT: (1,),
            StructuredLowerOpKind.CONV2D_RIGHT: (1,),
            StructuredLowerOpKind.ADD: (2,),
            StructuredLowerOpKind.RESHAPE: (1,),
            StructuredLowerOpKind.SLICE: (1,),
            StructuredLowerOpKind.BIAS_SPLIT: (1,),
            StructuredLowerOpKind.INPUT_CONCRETIZE: (1,),
        }
        if arity not in expected_arities[self.op_kind]:
            raise ValueError(f"{self.op_kind.value} input arity differs")
        required_attributes: dict[StructuredLowerOpKind, set[str]] = {
            StructuredLowerOpKind.SPEC_SEED: {"start_node"},
            StructuredLowerOpKind.RELU_LOWER_TRANSFORM: {"alpha_layout", "beta_layout"},
            StructuredLowerOpKind.LINEAR_RIGHT: {"weight_layout"},
            StructuredLowerOpKind.CONV2D_RIGHT: {
                "stride",
                "padding",
                "dilation",
                "groups",
            },
            StructuredLowerOpKind.ADD: set(),
            StructuredLowerOpKind.RESHAPE: {"source_shape"},
            StructuredLowerOpKind.SLICE: {"axis", "start", "stop"},
            StructuredLowerOpKind.BIAS_SPLIT: {"token"},
            StructuredLowerOpKind.INPUT_CONCRETIZE: {"perturbation"},
        }
        names = {attribute.name for attribute in self.attributes}
        if names != required_attributes[self.op_kind]:
            raise ValueError(f"{self.op_kind.value} attributes differ")

    def to_dict(self) -> dict[str, object]:
        self.validate_local()
        return {
            "node_id": self.node_id,
            "ordinal": self.ordinal,
            "op_kind": self.op_kind.value,
            "input_ids": list(self.input_ids),
            "output_shape": list(self.output_shape),
            "source_op_ids": list(self.source_op_ids),
            "declared_consumer_count": self.declared_consumer_count,
            "external_consumer_count": self.external_consumer_count,
            "attributes": [attribute.to_dict() for attribute in self.attributes],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StructuredLowerNodeV1":
        expected = {
            "node_id",
            "ordinal",
            "op_kind",
            "input_ids",
            "output_shape",
            "source_op_ids",
            "declared_consumer_count",
            "external_consumer_count",
            "attributes",
        }
        _require_exact(payload, expected, "node")
        raw_attributes = payload["attributes"]
        if not isinstance(raw_attributes, list) or any(
            not isinstance(value, dict) for value in raw_attributes
        ):
            raise ValueError("node attributes must be an object list")
        try:
            op_kind = StructuredLowerOpKind(_string(payload["op_kind"], "op_kind"))
        except ValueError as error:
            raise ValueError("structured op kind differs") from error
        result = cls(
            node_id=_string(payload["node_id"], "node_id"),
            ordinal=_integer(payload["ordinal"], "ordinal"),
            op_kind=op_kind,
            input_ids=_strings(payload["input_ids"], "input_ids"),
            output_shape=_shape(payload["output_shape"], "output_shape"),
            source_op_ids=_strings(payload["source_op_ids"], "source_op_ids"),
            declared_consumer_count=_integer(
                payload["declared_consumer_count"], "declared_consumer_count"
            ),
            external_consumer_count=_integer(
                payload["external_consumer_count"], "external_consumer_count"
            ),
            attributes=tuple(
                StructuredLowerAttributeV1.from_dict(value) for value in raw_attributes
            ),
        )
        result.validate_local()
        return result


@dataclass(frozen=True)
class BiasSplitWitnessV1:
    """Integer ownership fractions for one fanout bias token."""

    parent_node_id: str
    child_node_ids: tuple[str, ...]
    numerators: tuple[int, ...]
    denominator: int

    def validate(self) -> None:
        if (
            not self.parent_node_id
            or len(self.child_node_ids) < 2
            or len(self.child_node_ids) != len(self.numerators)
            or len(set(self.child_node_ids)) != len(self.child_node_ids)
            or self.denominator <= 0
            or any(value <= 0 for value in self.numerators)
            or sum(self.numerators) != self.denominator
        ):
            raise ValueError("bias split ownership differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "parent_node_id": self.parent_node_id,
            "child_node_ids": list(self.child_node_ids),
            "numerators": list(self.numerators),
            "denominator": self.denominator,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BiasSplitWitnessV1":
        _require_exact(
            payload,
            {"parent_node_id", "child_node_ids", "numerators", "denominator"},
            "bias split witness",
        )
        raw_numerators = payload["numerators"]
        if not isinstance(raw_numerators, list):
            raise ValueError("bias split numerators must be a list")
        result = cls(
            parent_node_id=_string(payload["parent_node_id"], "parent_node_id"),
            child_node_ids=_strings(payload["child_node_ids"], "child_node_ids"),
            numerators=tuple(
                _integer(value, "bias numerator", minimum=1) for value in raw_numerators
            ),
            denominator=_integer(payload["denominator"], "denominator", minimum=1),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ScratchIntervalV1:
    """One statically bounded use of a plan-owned scratch slot."""

    slot_id: int
    first_ordinal: int
    last_ordinal: int
    size_bytes: int

    def validate(self) -> None:
        if (
            self.slot_id not in (0, 1)
            or self.first_ordinal < 0
            or self.last_ordinal < self.first_ordinal
            or self.size_bytes <= 0
        ):
            raise ValueError("scratch interval differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "slot_id": self.slot_id,
            "first_ordinal": self.first_ordinal,
            "last_ordinal": self.last_ordinal,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ScratchIntervalV1":
        _require_exact(
            payload,
            {"slot_id", "first_ordinal", "last_ordinal", "size_bytes"},
            "scratch interval",
        )
        result = cls(
            slot_id=_integer(payload["slot_id"], "slot_id"),
            first_ordinal=_integer(payload["first_ordinal"], "first_ordinal"),
            last_ordinal=_integer(payload["last_ordinal"], "last_ordinal"),
            size_bytes=_integer(payload["size_bytes"], "size_bytes", minimum=1),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class StructuredLowerRegionTemplateV1:
    """Immutable lower-region DAG and ownership/liveness contract."""

    nodes: tuple[StructuredLowerNodeV1, ...]
    root_node_id: str
    source_op_count: int
    start_node_id: str
    source_hash: str
    topology_hash: str
    lineage_hash: str
    bias_witnesses: tuple[BiasSplitWitnessV1, ...] = ()
    scratch_intervals: tuple[ScratchIntervalV1, ...] = ()
    performance_claimed: bool = False
    production_connected: bool = False
    schema_version: str = TEMPLATE_SCHEMA

    def validate(self) -> None:  # pylint: disable=too-many-branches,too-many-statements
        if (
            self.schema_version != TEMPLATE_SCHEMA
            or not self.nodes
            or self.source_op_count <= 0
            or len(self.nodes) > MAX_NODE_EXPANSION * self.source_op_count
            or not self.start_node_id
            or any(
                not _is_hash(value)
                for value in (self.source_hash, self.topology_hash, self.lineage_hash)
            )
            or self.performance_claimed
            or self.production_connected
        ):
            raise ValueError("structured lower template differs")
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)) or self.root_node_id not in ids:
            raise ValueError("structured node identity differs")
        if [node.ordinal for node in self.nodes] != list(range(len(self.nodes))):
            raise ValueError("structured node ordinals differ")

        nodes_by_id = {node.node_id: node for node in self.nodes}
        consumers: dict[str, list[str]] = {node_id: [] for node_id in ids}
        for node in self.nodes:
            node.validate_local()
            for input_id in node.input_ids:
                producer = nodes_by_id.get(input_id)
                if producer is None or producer.ordinal >= node.ordinal:
                    raise ValueError("structured DAG topology differs")
                consumers[input_id].append(node.node_id)

        root = nodes_by_id[self.root_node_id]
        if root.op_kind != StructuredLowerOpKind.INPUT_CONCRETIZE:
            raise ValueError("structured root must be input concretize")
        for node in self.nodes:
            actual = len(consumers[node.node_id]) + node.external_consumer_count
            if actual != node.declared_consumer_count:
                raise ValueError("structured consumer count differs")
            expected_external = 1 if node.node_id == self.root_node_id else 0
            if node.external_consumer_count != expected_external:
                raise ValueError("structured region has an escaped consumer")

        reachable = {self.root_node_id}
        pending = [self.root_node_id]
        while pending:
            current = pending.pop()
            for input_id in nodes_by_id[current].input_ids:
                if input_id not in reachable:
                    reachable.add(input_id)
                    pending.append(input_id)
        if reachable != set(ids):
            raise ValueError("structured region contains nodes outside root closure")

        witnesses_by_parent: dict[str, BiasSplitWitnessV1] = {}
        for witness in self.bias_witnesses:
            witness.validate()
            if witness.parent_node_id in witnesses_by_parent:
                raise ValueError("duplicate bias split parent")
            parent = nodes_by_id.get(witness.parent_node_id)
            if parent is None or parent.op_kind != StructuredLowerOpKind.BIAS_SPLIT:
                raise ValueError("bias split parent differs")
            if tuple(consumers[parent.node_id]) != witness.child_node_ids:
                raise ValueError("bias split children differ")
            witnesses_by_parent[parent.node_id] = witness
        bias_parents = {
            node.node_id
            for node in self.nodes
            if node.op_kind == StructuredLowerOpKind.BIAS_SPLIT
        }
        if set(witnesses_by_parent) != bias_parents:
            raise ValueError("bias split witness coverage differs")

        for interval in self.scratch_intervals:
            interval.validate()
            if interval.last_ordinal >= len(self.nodes):
                raise ValueError("scratch interval exceeds topology")
        slots = {interval.slot_id for interval in self.scratch_intervals}
        if len(slots) > MAX_SCRATCH_SLOTS:
            raise ValueError("structured scratch slot count differs")
        for slot_id in slots:
            ordered = sorted(
                (item for item in self.scratch_intervals if item.slot_id == slot_id),
                key=lambda item: item.first_ordinal,
            )
            if any(
                previous.last_ordinal >= current.first_ordinal
                for previous, current in zip(ordered, ordered[1:])
            ):
                raise ValueError("structured scratch intervals overlap")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "nodes": [node.to_dict() for node in self.nodes],
            "root_node_id": self.root_node_id,
            "source_op_count": self.source_op_count,
            "start_node_id": self.start_node_id,
            "source_hash": self.source_hash,
            "topology_hash": self.topology_hash,
            "lineage_hash": self.lineage_hash,
            "bias_witnesses": [witness.to_dict() for witness in self.bias_witnesses],
            "scratch_intervals": [
                interval.to_dict() for interval in self.scratch_intervals
            ],
            "performance_claimed": self.performance_claimed,
            "production_connected": self.production_connected,
        }

    def stable_hash(self) -> str:
        return _hash(self.to_dict())

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "StructuredLowerRegionTemplateV1":
        expected = {
            "schema_version",
            "nodes",
            "root_node_id",
            "source_op_count",
            "start_node_id",
            "source_hash",
            "topology_hash",
            "lineage_hash",
            "bias_witnesses",
            "scratch_intervals",
            "performance_claimed",
            "production_connected",
        }
        _require_exact(payload, expected, "template")
        raw_nodes = payload["nodes"]
        raw_witnesses = payload["bias_witnesses"]
        raw_scratch = payload["scratch_intervals"]
        if not isinstance(raw_nodes, list) or any(
            not isinstance(value, dict) for value in raw_nodes
        ):
            raise ValueError("template nodes must be an object list")
        if not isinstance(raw_witnesses, list) or any(
            not isinstance(value, dict) for value in raw_witnesses
        ):
            raise ValueError("template witnesses must be an object list")
        if not isinstance(raw_scratch, list) or any(
            not isinstance(value, dict) for value in raw_scratch
        ):
            raise ValueError("template scratch must be an object list")
        result = cls(
            nodes=tuple(StructuredLowerNodeV1.from_dict(value) for value in raw_nodes),
            root_node_id=_string(payload["root_node_id"], "root_node_id"),
            source_op_count=_integer(
                payload["source_op_count"], "source_op_count", minimum=1
            ),
            start_node_id=_string(payload["start_node_id"], "start_node_id"),
            source_hash=_string(payload["source_hash"], "source_hash"),
            topology_hash=_string(payload["topology_hash"], "topology_hash"),
            lineage_hash=_string(payload["lineage_hash"], "lineage_hash"),
            bias_witnesses=tuple(
                BiasSplitWitnessV1.from_dict(value) for value in raw_witnesses
            ),
            scratch_intervals=tuple(
                ScratchIntervalV1.from_dict(value) for value in raw_scratch
            ),
            performance_claimed=_boolean(
                payload["performance_claimed"], "performance_claimed"
            ),
            production_connected=_boolean(
                payload["production_connected"], "production_connected"
            ),
            schema_version=_string(payload["schema_version"], "schema_version"),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class StructuredCoefficientHandleV1:
    """Opaque production handle; materialization is deliberately unavailable."""

    template_hash: str
    root_node_id: str
    output_shape: tuple[int, ...]

    def validate(self) -> None:
        if (
            not _is_hash(self.template_hash)
            or not self.root_node_id
            or any(value <= 0 for value in self.output_shape)
        ):
            raise ValueError("structured coefficient handle differs")

    def to_dense(self) -> torch.Tensor:
        raise StructuredDenseEscapeError(
            "R3 structured coefficient cannot escape as dense"
        )


@dataclass(frozen=True)
class StructuredTensorBindingV1:
    """Evaluation-local tensor identity metadata; no Tensor is stored in the IR."""

    name: str
    role: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    device: str
    data_ptr: int
    storage_ptr: int
    version: int
    requires_grad: bool

    def validate(self) -> None:
        allowed_roles = {
            "alpha",
            "beta",
            "bound",
            "weight",
            "bias",
            "input",
            "spec",
            "scratch",
        }
        if (
            not self.name
            or self.role not in allowed_roles
            or not self.shape
            or len(self.shape) != len(self.strides)
            or any(value < 0 for value in self.shape)
            or any(value < 0 for value in self.strides)
            or not self.dtype
            or not self.device
            or self.data_ptr <= 0
            or self.storage_ptr <= 0
            or self.version < 0
            or (self.requires_grad and self.role not in {"alpha", "beta"})
        ):
            raise ValueError("structured tensor binding differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "name": self.name,
            "role": self.role,
            "shape": list(self.shape),
            "strides": list(self.strides),
            "dtype": self.dtype,
            "device": self.device,
            "data_ptr": self.data_ptr,
            "storage_ptr": self.storage_ptr,
            "version": self.version,
            "requires_grad": self.requires_grad,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StructuredTensorBindingV1":
        expected = {
            "name",
            "role",
            "shape",
            "strides",
            "dtype",
            "device",
            "data_ptr",
            "storage_ptr",
            "version",
            "requires_grad",
        }
        _require_exact(payload, expected, "tensor binding")
        shape = _shape(payload["shape"], "binding shape", allow_zero=True)
        strides = _shape(payload["strides"], "binding strides", allow_zero=True)
        result = cls(
            name=_string(payload["name"], "binding name"),
            role=_string(payload["role"], "binding role"),
            shape=shape,
            strides=strides,
            dtype=_string(payload["dtype"], "binding dtype"),
            device=_string(payload["device"], "binding device"),
            data_ptr=_integer(payload["data_ptr"], "data_ptr", minimum=1),
            storage_ptr=_integer(payload["storage_ptr"], "storage_ptr", minimum=1),
            version=_integer(payload["version"], "version"),
            requires_grad=_boolean(payload["requires_grad"], "requires_grad"),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class StructuredLowerRegionInstanceV1:
    """Tensor-free binding receipt for one optimizer evaluation."""

    template_hash: str
    start_node_id: str
    evaluation_ordinal: int
    mutation_ordinal: int
    current_stream: int
    split_history_hash: str
    domain_hash: str
    bindings: tuple[StructuredTensorBindingV1, ...]
    schema_version: str = INSTANCE_SCHEMA

    def validate(self) -> None:
        if (
            self.schema_version != INSTANCE_SCHEMA
            or not _is_hash(self.template_hash)
            or not self.start_node_id
            or self.evaluation_ordinal < 0
            or self.mutation_ordinal < 0
            or self.mutation_ordinal > self.evaluation_ordinal
            or self.current_stream < 0
            or not _is_hash(self.split_history_hash)
            or not _is_hash(self.domain_hash)
            or not self.bindings
        ):
            raise ValueError("structured lower instance differs")
        names = [binding.name for binding in self.bindings]
        if len(names) != len(set(names)):
            raise ValueError("structured binding names differ")
        for binding in self.bindings:
            binding.validate()
        if not any(binding.role == "alpha" for binding in self.bindings):
            raise ValueError("structured instance lacks compressed alpha")
        if (
            sum(binding.role == "scratch" for binding in self.bindings)
            > MAX_SCRATCH_SLOTS
        ):
            raise ValueError("structured instance scratch count differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "template_hash": self.template_hash,
            "start_node_id": self.start_node_id,
            "evaluation_ordinal": self.evaluation_ordinal,
            "mutation_ordinal": self.mutation_ordinal,
            "current_stream": self.current_stream,
            "split_history_hash": self.split_history_hash,
            "domain_hash": self.domain_hash,
            "bindings": [binding.to_dict() for binding in self.bindings],
        }

    def stable_hash(self) -> str:
        return _hash(self.to_dict())

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "StructuredLowerRegionInstanceV1":
        expected = {
            "schema_version",
            "template_hash",
            "start_node_id",
            "evaluation_ordinal",
            "mutation_ordinal",
            "current_stream",
            "split_history_hash",
            "domain_hash",
            "bindings",
        }
        _require_exact(payload, expected, "instance")
        raw_bindings = payload["bindings"]
        if not isinstance(raw_bindings, list) or any(
            not isinstance(value, dict) for value in raw_bindings
        ):
            raise ValueError("instance bindings must be an object list")
        result = cls(
            template_hash=_string(payload["template_hash"], "template_hash"),
            start_node_id=_string(payload["start_node_id"], "start_node_id"),
            evaluation_ordinal=_integer(
                payload["evaluation_ordinal"], "evaluation_ordinal"
            ),
            mutation_ordinal=_integer(payload["mutation_ordinal"], "mutation_ordinal"),
            current_stream=_integer(payload["current_stream"], "current_stream"),
            split_history_hash=_string(
                payload["split_history_hash"], "split_history_hash"
            ),
            domain_hash=_string(payload["domain_hash"], "domain_hash"),
            bindings=tuple(
                StructuredTensorBindingV1.from_dict(value) for value in raw_bindings
            ),
            schema_version=_string(payload["schema_version"], "schema_version"),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class SavedTensorEntryV1:
    """One saved-tensor ledger row for the future custom VJP boundary."""

    role: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    storage_id: str
    logical_bytes: int
    version: int
    coefficient_lineage: bool

    def validate(self) -> None:
        if (
            not self.role
            or not self.shape
            or any(value < 0 for value in self.shape)
            or not self.dtype
            or not self.device
            or not self.storage_id
            or self.logical_bytes < 0
            or self.version < 0
        ):
            raise ValueError("saved tensor entry differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "role": self.role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "storage_id": self.storage_id,
            "logical_bytes": self.logical_bytes,
            "version": self.version,
            "coefficient_lineage": self.coefficient_lineage,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SavedTensorEntryV1":
        expected = {
            "role",
            "shape",
            "dtype",
            "device",
            "storage_id",
            "logical_bytes",
            "version",
            "coefficient_lineage",
        }
        _require_exact(payload, expected, "saved tensor entry")
        result = cls(
            role=_string(payload["role"], "saved role"),
            shape=_shape(payload["shape"], "saved shape", allow_zero=True),
            dtype=_string(payload["dtype"], "saved dtype"),
            device=_string(payload["device"], "saved device"),
            storage_id=_string(payload["storage_id"], "saved storage_id"),
            logical_bytes=_integer(payload["logical_bytes"], "saved logical_bytes"),
            version=_integer(payload["version"], "saved version"),
            coefficient_lineage=_boolean(
                payload["coefficient_lineage"], "coefficient_lineage"
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class SavedTensorLedgerV1:
    """Saved state proof: semantic leaves are allowed, dense coefficient A is not."""

    entries: tuple[SavedTensorEntryV1, ...]

    def validate(self) -> None:
        for entry in self.entries:
            entry.validate()
        if any(entry.coefficient_lineage for entry in self.entries):
            raise ValueError("saved dense coefficient lineage is forbidden")

    @property
    def logical_bytes(self) -> int:
        return sum(entry.logical_bytes for entry in self.entries)

    @property
    def unique_storage_bytes(self) -> int:
        sizes: dict[str, int] = {}
        for entry in self.entries:
            sizes[entry.storage_id] = max(
                sizes.get(entry.storage_id, 0), entry.logical_bytes
            )
        return sum(sizes.values())

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "logical_bytes": self.logical_bytes,
            "unique_storage_bytes": self.unique_storage_bytes,
            "coefficient_bytes": 0,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SavedTensorLedgerV1":
        _require_exact(
            payload,
            {"entries", "logical_bytes", "unique_storage_bytes", "coefficient_bytes"},
            "saved tensor ledger",
        )
        raw_entries = payload["entries"]
        if not isinstance(raw_entries, list) or any(
            not isinstance(value, dict) for value in raw_entries
        ):
            raise ValueError("saved tensor entries must be an object list")
        result = cls(
            tuple(SavedTensorEntryV1.from_dict(value) for value in raw_entries)
        )
        result.validate()
        if (
            payload["logical_bytes"] != result.logical_bytes
            or payload["unique_storage_bytes"] != result.unique_storage_bytes
            or payload["coefficient_bytes"] != 0
        ):
            raise ValueError("saved tensor ledger derivation differs")
        return result


def assert_tensor_free_context(value: object) -> None:
    """Reject tensors recursively reachable from an autograd context-like object."""

    seen: set[int] = set()

    def visit(current: object, path: str) -> None:
        if torch.is_tensor(current):
            raise ValueError(f"context tensor reachable at {path}")
        if current is None or isinstance(
            current, (str, bytes, int, float, bool, Enum, type)
        ):
            return
        identity = id(current)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(current, Mapping):
            for key, item in current.items():
                visit(item, f"{path}[{key!r}]")
            return
        if isinstance(current, (tuple, list, set, frozenset)):
            for index, item in enumerate(current):
                visit(item, f"{path}[{index}]")
            return
        if is_dataclass(current) and not isinstance(current, type):
            for field in fields(current):
                visit(getattr(current, field.name), f"{path}.{field.name}")
            return
        attributes = getattr(current, "__dict__", None)
        if isinstance(attributes, dict):
            for name, item in attributes.items():
                visit(item, f"{path}.{name}")

    visit(value, "ctx")


@dataclass(frozen=True)
class StructuredLowerR30ReceiptV1:
    """Fail-closed R3-0 contract receipt; it cannot carry a performance result."""

    template_hash: str
    instance_hash: str
    node_count: int
    source_op_count: int
    edge_count: int
    root_node_id: str
    scratch_slot_count: int
    saved_logical_bytes: int
    saved_unique_storage_bytes: int
    saved_coefficient_bytes: int
    dense_escape_count: int
    context_tensor_count: int
    production_connected: bool = False
    timing_recorded: bool = False
    performance_claimed: bool = False
    status: str = "validated-r3-0-contract"
    schema_version: str = RECEIPT_SCHEMA

    def validate(
        self,
        *,
        template: StructuredLowerRegionTemplateV1,
        instance: StructuredLowerRegionInstanceV1,
        ledger: SavedTensorLedgerV1,
    ) -> None:
        template.validate()
        instance.validate()
        ledger.validate()
        expected = {
            "template_hash": template.stable_hash(),
            "instance_hash": instance.stable_hash(),
            "node_count": len(template.nodes),
            "source_op_count": template.source_op_count,
            "edge_count": sum(len(node.input_ids) for node in template.nodes),
            "root_node_id": template.root_node_id,
            "scratch_slot_count": len(
                {item.slot_id for item in template.scratch_intervals}
            ),
            "saved_logical_bytes": ledger.logical_bytes,
            "saved_unique_storage_bytes": ledger.unique_storage_bytes,
        }
        actual = {name: getattr(self, name) for name in expected}
        if (
            self.schema_version != RECEIPT_SCHEMA
            or actual != expected
            or instance.template_hash != template.stable_hash()
            or instance.start_node_id != template.start_node_id
            or self.saved_coefficient_bytes != 0
            or self.dense_escape_count != 0
            or self.context_tensor_count != 0
            or self.production_connected
            or self.timing_recorded
            or self.performance_claimed
            or self.status != "validated-r3-0-contract"
        ):
            raise ValueError("R3-0 receipt differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "template_hash": self.template_hash,
            "instance_hash": self.instance_hash,
            "node_count": self.node_count,
            "source_op_count": self.source_op_count,
            "edge_count": self.edge_count,
            "root_node_id": self.root_node_id,
            "scratch_slot_count": self.scratch_slot_count,
            "saved_logical_bytes": self.saved_logical_bytes,
            "saved_unique_storage_bytes": self.saved_unique_storage_bytes,
            "saved_coefficient_bytes": self.saved_coefficient_bytes,
            "dense_escape_count": self.dense_escape_count,
            "context_tensor_count": self.context_tensor_count,
            "production_connected": self.production_connected,
            "timing_recorded": self.timing_recorded,
            "performance_claimed": self.performance_claimed,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StructuredLowerR30ReceiptV1":
        expected = {
            "schema_version",
            "template_hash",
            "instance_hash",
            "node_count",
            "source_op_count",
            "edge_count",
            "root_node_id",
            "scratch_slot_count",
            "saved_logical_bytes",
            "saved_unique_storage_bytes",
            "saved_coefficient_bytes",
            "dense_escape_count",
            "context_tensor_count",
            "production_connected",
            "timing_recorded",
            "performance_claimed",
            "status",
        }
        _require_exact(payload, expected, "R3-0 receipt")
        result = cls(
            template_hash=_string(payload["template_hash"], "template_hash"),
            instance_hash=_string(payload["instance_hash"], "instance_hash"),
            node_count=_integer(payload["node_count"], "node_count", minimum=1),
            source_op_count=_integer(
                payload["source_op_count"], "source_op_count", minimum=1
            ),
            edge_count=_integer(payload["edge_count"], "edge_count"),
            root_node_id=_string(payload["root_node_id"], "root_node_id"),
            scratch_slot_count=_integer(
                payload["scratch_slot_count"], "scratch_slot_count"
            ),
            saved_logical_bytes=_integer(
                payload["saved_logical_bytes"], "saved_logical_bytes"
            ),
            saved_unique_storage_bytes=_integer(
                payload["saved_unique_storage_bytes"], "saved_unique_storage_bytes"
            ),
            saved_coefficient_bytes=_integer(
                payload["saved_coefficient_bytes"], "saved_coefficient_bytes"
            ),
            dense_escape_count=_integer(
                payload["dense_escape_count"], "dense_escape_count"
            ),
            context_tensor_count=_integer(
                payload["context_tensor_count"], "context_tensor_count"
            ),
            production_connected=_boolean(
                payload["production_connected"], "production_connected"
            ),
            timing_recorded=_boolean(payload["timing_recorded"], "timing_recorded"),
            performance_claimed=_boolean(
                payload["performance_claimed"], "performance_claimed"
            ),
            status=_string(payload["status"], "status"),
            schema_version=_string(payload["schema_version"], "schema_version"),
        )
        return result


__all__ = [
    "BiasSplitWitnessV1",
    "INSTANCE_SCHEMA",
    "MAX_NODE_EXPANSION",
    "MAX_SCRATCH_SLOTS",
    "RECEIPT_SCHEMA",
    "SavedTensorEntryV1",
    "SavedTensorLedgerV1",
    "ScratchIntervalV1",
    "StructuredCoefficientHandleV1",
    "StructuredDenseEscapeError",
    "StructuredLowerAttributeV1",
    "StructuredLowerNodeV1",
    "StructuredLowerOpKind",
    "StructuredLowerR30ReceiptV1",
    "StructuredLowerRegionInstanceV1",
    "StructuredLowerRegionTemplateV1",
    "StructuredTensorBindingV1",
    "TEMPLATE_SCHEMA",
    "assert_tensor_free_context",
]
