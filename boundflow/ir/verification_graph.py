"""Generic verification-graph ABI frozen by GC0-0.

This module is schema-only.  It deliberately contains no capture pass, graph
legalizer, TVM lowering, physical allocator, runtime launch, or timing path.
"""

# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-branches
# pylint: disable=too-many-locals,too-many-statements,too-many-boolean-expressions
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Mapping, NoReturn, TypeVar, cast

VERIFICATION_GRAPH_SCHEMA_V1 = "boundflow.verification-graph/v1"
VERIFICATION_GRAPH_MODULE_SCHEMA_V1 = "boundflow.verification-graph-module/v1"
VERIFICATION_RULE_REGISTRY_SCHEMA_V1 = "boundflow.verification-rule-registry/v1"
VERIFICATION_LEGALITY_RESULT_SCHEMA_V1 = "boundflow.verification-legality-result/v1"
VERIFICATION_SEMANTIC_OWNER_V1 = "boundflow.verification-graph/v1"


class VerificationValueRole(str, Enum):
    """Semantic role of one verification graph value."""

    COEFFICIENT = "coefficient"
    BIAS_TOKEN = "bias-token"
    LOWER_BOUND = "lower-bound"
    UPPER_BOUND = "upper-bound"
    ALPHA = "alpha"
    BETA = "beta"
    SPLIT = "split"
    HISTORY = "history"
    PARAMETER = "parameter"
    OPTIMIZER_STATE = "optimizer-state"
    STATUS = "status"
    COMMIT_TOKEN = "commit-token"
    SCRATCH = "scratch"
    INCOMING_ADJOINT = "incoming-adjoint"
    GRADIENT = "gradient"
    INDEX = "index"
    MASK = "mask"


class VerificationAxisRole(str, Enum):
    """Verification-specific logical axis identity."""

    DOMAIN = "domain"
    SPEC = "spec"
    CHANNEL = "channel"
    HEIGHT = "height"
    WIDTH = "width"
    FEATURE = "feature"
    BETA_SLOT = "beta-slot"
    DIRECTION = "direction"
    INPUT_CHANNEL = "input-channel"
    OUTPUT_CHANNEL = "output-channel"
    KERNEL_HEIGHT = "kernel-height"
    KERNEL_WIDTH = "kernel-width"
    NONE = "none"


class VerificationPolarity(str, Enum):
    """Bound direction owned by a value or operation."""

    LOWER = "lower"
    UPPER = "upper"
    BOTH = "both"
    NONE = "none"


class VerificationRepresentation(str, Enum):
    """Logical representation before physical storage planning."""

    DENSE = "dense"
    COMPRESSED_INDEXED = "compressed-indexed"
    SPARSE_LOCATION = "sparse-location"
    SCALAR = "scalar"
    TOKEN = "token"


class VerificationStorageClass(str, Enum):
    """Storage ownership class carried by graph values."""

    EXTERNAL_BORROWED = "external-borrowed"
    PARAMETER_RESIDENT = "parameter-resident"
    ARENA_PERSISTENT = "arena-persistent"
    ARENA_SCRATCH = "arena-scratch"
    SAVED_MINIMAL = "saved-minimal"
    HOST_STATUS = "host-status"


class VerificationFinitePolicy(str, Enum):
    """Validation policy for a value payload."""

    FINITE_REQUIRED = "finite-required"
    INTEGER_EXACT = "integer-exact"
    TOKEN = "token"


class VerificationOpKind(str, Enum):
    """Closed GC0-0 operation vocabulary; execution is not implemented here."""

    SPEC_SEED = "spec-seed"
    INCOMING_COEFFICIENT = "incoming-coefficient"
    COMPRESSED_ALPHA_GATHER = "compressed-alpha-gather"
    SPARSE_BETA_INJECT = "sparse-beta-inject"
    RELU_RELAXATION = "relu-relaxation"
    SIGN_SELECT = "sign-select"
    LINEAR_RIGHT = "linear-right"
    CONV2D_RIGHT = "conv2d-right"
    RESHAPE_VIEW = "reshape-view"
    LAYOUT_NORMALIZE = "layout-normalize"
    RESIDUAL_DIAMOND = "residual-diamond"
    BIAS_REDUCE = "bias-reduce"
    BIAS_ACCUMULATE = "bias-accumulate"
    INPUT_CONCRETIZE = "input-concretize"
    MINIMAL_STATE_VJP = "minimal-state-vjp"
    COMPACT_STATUS = "compact-status"
    COARSE_COMMIT = "coarse-commit"


class VerificationEffectKind(str, Enum):
    """Versioned verification/runtime effect resource."""

    ALPHA_STATE = "alpha-state"
    BETA_STATE = "beta-state"
    SPLIT_HISTORY = "split-history"
    OPTIMIZER_STATE = "optimizer-state"
    DOMAIN_LINEAGE = "domain-lineage"
    QUEUE_STATE = "queue-state"
    COMMIT_STATE = "commit-state"
    RUNTIME_ARENA = "runtime-arena"


class VerificationEffectAccess(str, Enum):
    """One effect token's version transition."""

    READ = "read"
    WRITE = "write"
    READ_WRITE = "read-write"
    EXTERNAL_BOUNDARY = "external-boundary"


class VerificationFallbackPolicy(str, Enum):
    """Fallback policy for candidate regions and rewrite rules."""

    REJECT_BEFORE_LAUNCH = "reject-before-launch"


class VerificationRulePatternKind(str, Enum):
    """Static pattern family.  Matching belongs to GC0-1/GC-1."""

    EXACT_CHAIN = "exact-chain"
    DATAFLOW_PATTERN = "dataflow-pattern"
    RESIDUAL_DIAMOND = "residual-diamond"
    TERMINAL_REGION = "terminal-region"
    VJP_REGION = "vjp-region"
    MEMORY_REUSE = "memory-reuse"


class VerificationRejectionReason(str, Enum):
    """Frozen fail-closed rejection vocabulary from the GC-0 preregistration."""

    UNSUPPORTED_OP_KIND = "UNSUPPORTED_OP_KIND"
    DYNAMIC_SHAPE_UNBOUND = "DYNAMIC_SHAPE_UNBOUND"
    DTYPE_OR_DEVICE_MISMATCH = "DTYPE_OR_DEVICE_MISMATCH"
    LAYOUT_NOT_NORMALIZABLE = "LAYOUT_NOT_NORMALIZABLE"
    REGION_EXTERNAL_USE = "REGION_EXTERNAL_USE"
    REGION_NOT_POSTDOMINATED = "REGION_NOT_POSTDOMINATED"
    STATE_VERSION_MISMATCH = "STATE_VERSION_MISMATCH"
    EFFECT_ORDER_CONFLICT = "EFFECT_ORDER_CONFLICT"
    ALPHA_START_NODE_MISMATCH = "ALPHA_START_NODE_MISMATCH"
    ALPHA_INDEX_OR_DIRECTION_MISMATCH = "ALPHA_INDEX_OR_DIRECTION_MISMATCH"
    BETA_ACTIVE_EMPTY_MISMATCH = "BETA_ACTIVE_EMPTY_MISMATCH"
    BETA_LOCATION_SIGN_HISTORY_MISMATCH = "BETA_LOCATION_SIGN_HISTORY_MISMATCH"
    BOUND_POLARITY_MISMATCH = "BOUND_POLARITY_MISMATCH"
    ENDPOINT_POLICY_MISMATCH = "ENDPOINT_POLICY_MISMATCH"
    RESIDUAL_BIAS_TOKEN_UNCLOSED = "RESIDUAL_BIAS_TOKEN_UNCLOSED"
    UNSAFE_ALIAS_OR_LIFETIME = "UNSAFE_ALIAS_OR_LIFETIME"
    DENSE_A_ESCAPE = "DENSE_A_ESCAPE"
    VJP_OWNER_OR_SAVED_STATE_MISMATCH = "VJP_OWNER_OR_SAVED_STATE_MISMATCH"
    HIGHER_ORDER_GRAD_UNSUPPORTED = "HIGHER_ORDER_GRAD_UNSUPPORTED"
    QUEUE_OR_TERMINATION_EFFECT_CROSSED = "QUEUE_OR_TERMINATION_EFFECT_CROSSED"
    RUNTIME_FALLBACK_REQUIRED = "RUNTIME_FALLBACK_REQUIRED"
    RECEIPT_IDENTITY_MISMATCH = "RECEIPT_IDENTITY_MISMATCH"


GC0_DIRECT_REJECTION_REASONS = (
    VerificationRejectionReason.UNSUPPORTED_OP_KIND,
    VerificationRejectionReason.DYNAMIC_SHAPE_UNBOUND,
    VerificationRejectionReason.DTYPE_OR_DEVICE_MISMATCH,
    VerificationRejectionReason.LAYOUT_NOT_NORMALIZABLE,
    VerificationRejectionReason.STATE_VERSION_MISMATCH,
    VerificationRejectionReason.ALPHA_START_NODE_MISMATCH,
    VerificationRejectionReason.ALPHA_INDEX_OR_DIRECTION_MISMATCH,
    VerificationRejectionReason.BETA_ACTIVE_EMPTY_MISMATCH,
    VerificationRejectionReason.BETA_LOCATION_SIGN_HISTORY_MISMATCH,
    VerificationRejectionReason.BOUND_POLARITY_MISMATCH,
    VerificationRejectionReason.ENDPOINT_POLICY_MISMATCH,
    VerificationRejectionReason.VJP_OWNER_OR_SAVED_STATE_MISMATCH,
    VerificationRejectionReason.HIGHER_ORDER_GRAD_UNSUPPORTED,
    VerificationRejectionReason.RUNTIME_FALLBACK_REQUIRED,
    VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
)

GC01_ANALYSIS_REJECTION_REASONS = tuple(
    reason
    for reason in VerificationRejectionReason
    if reason not in GC0_DIRECT_REJECTION_REASONS
)

REQUIRED_VERIFICATION_RULE_IDS_V1 = (
    "V-C1-terminal-concretize-v1",
    "V-D1-residual-diamond-v1",
    "V-H1-lower-upper-tuple-v1",
    "V-M1-certified-arena-reuse-v1",
    "V-R1-relax-sign-affine-v1",
    "V-R2-compressed-alpha-gather-v1",
    "V-R3-sparse-beta-inject-v1",
    "V-VJP1-minimal-saved-state-v1",
)

_SUPPORTED_DTYPES = frozenset(
    {
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int32",
        "int64",
        "bool",
        "token",
    }
)
_SUPPORTED_DEVICES = frozenset({"cpu", "cuda", "host"})
_SUPPORTED_LAYOUTS = frozenset(
    {"contiguous-strided", "channels-last", "opaque-compressed", "scalar", "token"}
)
_BOUND_OPS = frozenset(
    {
        VerificationOpKind.COMPRESSED_ALPHA_GATHER,
        VerificationOpKind.SPARSE_BETA_INJECT,
        VerificationOpKind.RELU_RELAXATION,
        VerificationOpKind.SIGN_SELECT,
        VerificationOpKind.LINEAR_RIGHT,
        VerificationOpKind.CONV2D_RIGHT,
        VerificationOpKind.BIAS_REDUCE,
        VerificationOpKind.BIAS_ACCUMULATE,
        VerificationOpKind.INPUT_CONCRETIZE,
        VerificationOpKind.MINIMAL_STATE_VJP,
    }
)
_STATE_ROLES = frozenset(
    {
        VerificationValueRole.ALPHA,
        VerificationValueRole.BETA,
        VerificationValueRole.SPLIT,
        VerificationValueRole.HISTORY,
        VerificationValueRole.OPTIMIZER_STATE,
        VerificationValueRole.COMMIT_TOKEN,
    }
)


class VerificationGraphValidationError(ValueError):
    """Validation failure carrying one stable preregistered reason."""

    def __init__(self, reason: VerificationRejectionReason, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}")


def _reject(reason: VerificationRejectionReason, detail: str) -> NoReturn:
    raise VerificationGraphValidationError(reason, detail)


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


def _require_identifier(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        _reject(
            VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
            f"{label} is empty or unstable",
        )
    return value


def _require_unique(values: tuple[str, ...], label: str) -> None:
    if len(values) != len(set(values)):
        _reject(
            VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
            f"{label} are not unique",
        )


def _freeze_attribute(value: object) -> object:
    if isinstance(value, list):
        return tuple(_freeze_attribute(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_attribute(item) for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("verification attribute float must be finite")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError("verification attribute is not canonical JSON")


def _json_attribute(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_attribute(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError("verification attribute is not frozen canonical JSON")


def freeze_verification_attributes(
    attributes: Mapping[str, object],
) -> tuple[tuple[str, object], ...]:
    """Freeze an attribute mapping into a sorted, immutable canonical form."""

    result = tuple(
        (str(name), _freeze_attribute(value))
        for name, value in sorted(attributes.items())
    )
    if any(not name for name, _value in result):
        raise TypeError("verification attribute name is empty")
    return result


class _CanonicalSchemaObject:
    """Shared canonical JSON and digest surface for leaf schema objects."""

    def to_dict(self) -> dict[str, object]:
        raise NotImplementedError

    def canonical_json(self) -> str:
        return _canonical(self.to_dict())

    def stable_hash(self) -> str:
        return _hash(self.to_dict())


@dataclass(frozen=True)
class VerificationValueV1(_CanonicalSchemaObject):
    """Typed SSA/state value independent of any model or provider site name."""

    value_id: str
    role: VerificationValueRole
    shape: tuple[int | None, ...]
    dtype: str
    device_kind: str
    layout: str
    strides: tuple[int, ...]
    axis_roles: tuple[VerificationAxisRole, ...]
    polarity: VerificationPolarity
    representation: VerificationRepresentation
    requires_grad: bool
    state_version: str | None
    lineage_id: str | None
    storage_class: VerificationStorageClass
    alias_set: str | None
    producer_op_id: str | None
    consumer_op_ids: tuple[str, ...]
    external_use_count: int
    present: bool
    finite_policy: VerificationFinitePolicy

    def validate(self) -> None:
        _require_identifier(self.value_id, "value_id")
        if not isinstance(self.role, VerificationValueRole):
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                "value role is not typed",
            )
        if any(dimension is None for dimension in self.shape):
            _reject(
                VerificationRejectionReason.DYNAMIC_SHAPE_UNBOUND,
                f"value {self.value_id} has an unbound dimension",
            )
        concrete_shape = cast(tuple[int, ...], self.shape)
        if any(dimension < 0 for dimension in concrete_shape):
            _reject(
                VerificationRejectionReason.DYNAMIC_SHAPE_UNBOUND,
                f"value {self.value_id} has a negative dimension",
            )
        zero_dimensions = tuple(
            index for index, dimension in enumerate(concrete_shape) if dimension == 0
        )
        beta_empty = (
            self.role == VerificationValueRole.BETA
            and zero_dimensions == (len(concrete_shape) - 1,)
            and self.representation
            in {
                VerificationRepresentation.COMPRESSED_INDEXED,
                VerificationRepresentation.SPARSE_LOCATION,
            }
        )
        if zero_dimensions and not beta_empty:
            _reject(
                VerificationRejectionReason.BETA_ACTIVE_EMPTY_MISMATCH,
                f"value {self.value_id} has an illegal zero dimension",
            )
        if (
            self.dtype not in _SUPPORTED_DTYPES
            or self.device_kind not in _SUPPORTED_DEVICES
        ):
            _reject(
                VerificationRejectionReason.DTYPE_OR_DEVICE_MISMATCH,
                f"value {self.value_id} dtype/device is unsupported",
            )
        if self.layout not in _SUPPORTED_LAYOUTS:
            _reject(
                VerificationRejectionReason.LAYOUT_NOT_NORMALIZABLE,
                f"value {self.value_id} layout is unsupported",
            )
        if len(self.strides) != len(concrete_shape) or any(
            stride < 0 for stride in self.strides
        ):
            _reject(
                VerificationRejectionReason.LAYOUT_NOT_NORMALIZABLE,
                f"value {self.value_id} strides differ from rank",
            )
        if len(self.axis_roles) != len(concrete_shape) or any(
            not isinstance(axis, VerificationAxisRole) for axis in self.axis_roles
        ):
            _reject(
                VerificationRejectionReason.DTYPE_OR_DEVICE_MISMATCH,
                f"value {self.value_id} axis roles differ from rank",
            )
        if not isinstance(self.polarity, VerificationPolarity):
            _reject(
                VerificationRejectionReason.BOUND_POLARITY_MISMATCH,
                f"value {self.value_id} polarity is not typed",
            )
        if not isinstance(
            self.representation, VerificationRepresentation
        ) or not isinstance(self.storage_class, VerificationStorageClass):
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                f"value {self.value_id} representation/storage is not typed",
            )
        if self.role in _STATE_ROLES and not self.state_version:
            _reject(
                VerificationRejectionReason.STATE_VERSION_MISMATCH,
                f"state value {self.value_id} has no version",
            )
        if self.state_version is not None:
            _require_identifier(self.state_version, "state_version")
        if self.lineage_id is not None:
            _require_identifier(self.lineage_id, "lineage_id")
        if self.alias_set is not None:
            _require_identifier(self.alias_set, "alias_set")
        if self.producer_op_id is not None:
            _require_identifier(self.producer_op_id, "producer_op_id")
        _require_unique(self.consumer_op_ids, f"{self.value_id} consumers")
        for consumer in self.consumer_op_ids:
            _require_identifier(consumer, "consumer_op_id")
        if self.external_use_count < 0:
            _reject(
                VerificationRejectionReason.REGION_EXTERNAL_USE,
                f"value {self.value_id} external use count is negative",
            )
        if beta_empty != (self.role == VerificationValueRole.BETA and not self.present):
            _reject(
                VerificationRejectionReason.BETA_ACTIVE_EMPTY_MISMATCH,
                f"value {self.value_id} beta presence differs from shape",
            )
        if not isinstance(self.finite_policy, VerificationFinitePolicy):
            _reject(
                VerificationRejectionReason.DTYPE_OR_DEVICE_MISMATCH,
                f"value {self.value_id} finite policy is not typed",
            )
        if self.representation == VerificationRepresentation.TOKEN:
            if (
                self.shape
                or self.strides
                or self.axis_roles
                or self.dtype != "token"
                or self.layout != "token"
                or self.finite_policy != VerificationFinitePolicy.TOKEN
            ):
                _reject(
                    VerificationRejectionReason.DTYPE_OR_DEVICE_MISMATCH,
                    f"token value {self.value_id} has tensor fields",
                )
        elif (
            self.dtype == "token"
            or self.finite_policy == VerificationFinitePolicy.TOKEN
        ):
            _reject(
                VerificationRejectionReason.DTYPE_OR_DEVICE_MISMATCH,
                f"non-token value {self.value_id} uses token policy",
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "value_id": self.value_id,
            "role": self.role.value,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device_kind": self.device_kind,
            "layout": self.layout,
            "strides": list(self.strides),
            "axis_roles": [axis.value for axis in self.axis_roles],
            "polarity": self.polarity.value,
            "representation": self.representation.value,
            "requires_grad": self.requires_grad,
            "state_version": self.state_version,
            "lineage_id": self.lineage_id,
            "storage_class": self.storage_class.value,
            "alias_set": self.alias_set,
            "producer_op_id": self.producer_op_id,
            "consumer_op_ids": list(self.consumer_op_ids),
            "external_use_count": self.external_use_count,
            "present": self.present,
            "finite_policy": self.finite_policy.value,
        }


@dataclass(frozen=True)
class VerificationOpV1(_CanonicalSchemaObject):
    """One semantic graph operation with explicit effect references."""

    op_id: str
    op_kind: VerificationOpKind
    semantic_version: str
    input_value_ids: tuple[str, ...]
    output_value_ids: tuple[str, ...]
    parameter_value_ids: tuple[str, ...]
    effect_read_ids: tuple[str, ...]
    effect_write_ids: tuple[str, ...]
    attributes: tuple[tuple[str, object], ...]
    bound_direction: VerificationPolarity
    numeric_policy_id: str
    vjp_contract_id: str | None
    source_op_ids: tuple[str, ...]

    @property
    def attribute_map(self) -> dict[str, object]:
        return dict(self.attributes)

    def validate(self) -> None:
        _require_identifier(self.op_id, "op_id")
        if not isinstance(self.op_kind, VerificationOpKind):
            _reject(
                VerificationRejectionReason.UNSUPPORTED_OP_KIND,
                f"op {self.op_id} kind is unsupported",
            )
        _require_identifier(self.semantic_version, "semantic_version")
        _require_identifier(self.numeric_policy_id, "numeric_policy_id")
        if not self.output_value_ids:
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                f"op {self.op_id} has no output",
            )
        for label, values in (
            ("inputs", self.input_value_ids),
            ("outputs", self.output_value_ids),
            ("parameters", self.parameter_value_ids),
            ("effect reads", self.effect_read_ids),
            ("effect writes", self.effect_write_ids),
            ("source ops", self.source_op_ids),
        ):
            _require_unique(values, f"{self.op_id} {label}")
            for value in values:
                _require_identifier(value, f"{self.op_id} {label}")
        if set(self.effect_read_ids) & set(self.effect_write_ids):
            _reject(
                VerificationRejectionReason.EFFECT_ORDER_CONFLICT,
                f"op {self.op_id} reads and writes the same effect token",
            )
        attribute_names = tuple(name for name, _value in self.attributes)
        if attribute_names != tuple(sorted(attribute_names)) or len(
            attribute_names
        ) != len(set(attribute_names)):
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                f"op {self.op_id} attributes are not sorted and unique",
            )
        for name, attribute_value in self.attributes:
            _require_identifier(name, "attribute name")
            _json_attribute(attribute_value)
        if not isinstance(self.bound_direction, VerificationPolarity) or (
            self.op_kind in _BOUND_OPS
            and self.bound_direction == VerificationPolarity.NONE
        ):
            _reject(
                VerificationRejectionReason.BOUND_POLARITY_MISMATCH,
                f"op {self.op_id} bound direction differs",
            )
        if self.vjp_contract_id is not None:
            _require_identifier(self.vjp_contract_id, "vjp_contract_id")
        attributes = self.attribute_map
        if self.op_kind == VerificationOpKind.COMPRESSED_ALPHA_GATHER:
            if not isinstance(
                attributes.get("start_node_key"), str
            ) or not attributes.get("start_node_key"):
                _reject(
                    VerificationRejectionReason.ALPHA_START_NODE_MISMATCH,
                    f"op {self.op_id} alpha start-node key is missing",
                )
            if any(
                not isinstance(attributes.get(name), int)
                or cast(int, attributes[name]) < 0
                for name in ("direction_index", "spec_index")
            ) or not isinstance(attributes.get("feature_index_value_id"), str):
                _reject(
                    VerificationRejectionReason.ALPHA_INDEX_OR_DIRECTION_MISMATCH,
                    f"op {self.op_id} alpha lookup identity differs",
                )
            if not isinstance(
                attributes.get("state_version"), str
            ) or not attributes.get("state_version"):
                _reject(
                    VerificationRejectionReason.STATE_VERSION_MISMATCH,
                    f"op {self.op_id} alpha state version differs",
                )
        if self.op_kind == VerificationOpKind.SPARSE_BETA_INJECT:
            active = attributes.get("active")
            if not isinstance(active, bool):
                _reject(
                    VerificationRejectionReason.BETA_ACTIVE_EMPTY_MISMATCH,
                    f"op {self.op_id} beta active flag is not typed",
                )
            active_fields = ("location_value_id", "sign_value_id", "history_value_id")
            present = tuple(
                isinstance(attributes.get(name), str) for name in active_fields
            )
            if (active and not all(present)) or (not active and any(present)):
                _reject(
                    VerificationRejectionReason.BETA_LOCATION_SIGN_HISTORY_MISMATCH,
                    f"op {self.op_id} beta location/sign/history differs",
                )
        if self.op_kind in {
            VerificationOpKind.RELU_RELAXATION,
            VerificationOpKind.SIGN_SELECT,
        } and not isinstance(attributes.get("endpoint_policy"), str):
            _reject(
                VerificationRejectionReason.ENDPOINT_POLICY_MISMATCH,
                f"op {self.op_id} endpoint policy is missing",
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "op_id": self.op_id,
            "op_kind": self.op_kind.value,
            "semantic_version": self.semantic_version,
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
            "parameter_value_ids": list(self.parameter_value_ids),
            "effect_read_ids": list(self.effect_read_ids),
            "effect_write_ids": list(self.effect_write_ids),
            "attributes": [
                [name, _json_attribute(value)] for name, value in self.attributes
            ],
            "bound_direction": self.bound_direction.value,
            "numeric_policy_id": self.numeric_policy_id,
            "vjp_contract_id": self.vjp_contract_id,
            "source_op_ids": list(self.source_op_ids),
        }


@dataclass(frozen=True)
class VerificationEffectTokenV1(_CanonicalSchemaObject):
    """One typed and versioned effect transition."""

    effect_id: str
    kind: VerificationEffectKind
    resource_id: str
    input_version: str
    output_version: str
    access: VerificationEffectAccess
    ordinal: int

    def validate(self) -> None:
        for label, value in (
            ("effect_id", self.effect_id),
            ("resource_id", self.resource_id),
            ("input_version", self.input_version),
            ("output_version", self.output_version),
        ):
            _require_identifier(value, label)
        if not isinstance(self.kind, VerificationEffectKind) or not isinstance(
            self.access, VerificationEffectAccess
        ):
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                f"effect {self.effect_id} kind/access is not typed",
            )
        if self.ordinal < 0:
            _reject(
                VerificationRejectionReason.EFFECT_ORDER_CONFLICT,
                f"effect {self.effect_id} ordinal is negative",
            )
        unchanged = self.input_version == self.output_version
        if (
            self.access
            in {
                VerificationEffectAccess.READ,
                VerificationEffectAccess.EXTERNAL_BOUNDARY,
            }
            and not unchanged
        ):
            _reject(
                VerificationRejectionReason.STATE_VERSION_MISMATCH,
                f"read effect {self.effect_id} changes version",
            )
        if (
            self.access
            in {
                VerificationEffectAccess.WRITE,
                VerificationEffectAccess.READ_WRITE,
            }
            and unchanged
        ):
            _reject(
                VerificationRejectionReason.STATE_VERSION_MISMATCH,
                f"write effect {self.effect_id} does not change version",
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "effect_id": self.effect_id,
            "kind": self.kind.value,
            "resource_id": self.resource_id,
            "input_version": self.input_version,
            "output_version": self.output_version,
            "access": self.access.value,
            "ordinal": self.ordinal,
        }


@dataclass(frozen=True)
class VerificationVJPContractV1(_CanonicalSchemaObject):
    """Minimal-saved-state first-order VJP contract."""

    contract_id: str
    primal_input_value_ids: tuple[str, ...]
    primal_output_value_ids: tuple[str, ...]
    incoming_adjoint_value_ids: tuple[str, ...]
    alpha_gradient_owner_value_ids: tuple[str, ...]
    beta_gradient_owner_value_ids: tuple[str, ...]
    compressed_output_layouts: tuple[str, ...]
    saved_value_ids: tuple[str, ...]
    recomputed_value_ids: tuple[str, ...]
    endpoint_policy: str
    higher_order_policy: str = "reject"
    dense_a_escape_policy: str = "forbid"
    mutation_policy: str = "none-inside-vjp"

    def validate(self) -> None:
        _require_identifier(self.contract_id, "VJP contract_id")
        for label, values in (
            ("primal inputs", self.primal_input_value_ids),
            ("primal outputs", self.primal_output_value_ids),
            ("incoming adjoints", self.incoming_adjoint_value_ids),
            ("alpha owners", self.alpha_gradient_owner_value_ids),
            ("beta owners", self.beta_gradient_owner_value_ids),
            ("saved values", self.saved_value_ids),
            ("recomputed values", self.recomputed_value_ids),
        ):
            _require_unique(values, f"{self.contract_id} {label}")
            for value in values:
                _require_identifier(value, f"{self.contract_id} {label}")
        if (
            not self.primal_input_value_ids
            or not self.primal_output_value_ids
            or not self.incoming_adjoint_value_ids
            or not (
                self.alpha_gradient_owner_value_ids
                or self.beta_gradient_owner_value_ids
            )
            or len(self.compressed_output_layouts)
            != len(self.alpha_gradient_owner_value_ids)
            + len(self.beta_gradient_owner_value_ids)
            or set(self.saved_value_ids) & set(self.recomputed_value_ids)
        ):
            _reject(
                VerificationRejectionReason.VJP_OWNER_OR_SAVED_STATE_MISMATCH,
                f"VJP contract {self.contract_id} ownership/saved state differs",
            )
        for layout in self.compressed_output_layouts:
            _require_identifier(layout, "compressed output layout")
        if not self.endpoint_policy:
            _reject(
                VerificationRejectionReason.ENDPOINT_POLICY_MISMATCH,
                f"VJP contract {self.contract_id} endpoint policy is empty",
            )
        if self.higher_order_policy != "reject":
            _reject(
                VerificationRejectionReason.HIGHER_ORDER_GRAD_UNSUPPORTED,
                f"VJP contract {self.contract_id} enables higher-order gradients",
            )
        if self.dense_a_escape_policy != "forbid":
            _reject(
                VerificationRejectionReason.DENSE_A_ESCAPE,
                f"VJP contract {self.contract_id} permits dense-A escape",
            )
        if self.mutation_policy != "none-inside-vjp":
            _reject(
                VerificationRejectionReason.VJP_OWNER_OR_SAVED_STATE_MISMATCH,
                f"VJP contract {self.contract_id} mutates state",
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "contract_id": self.contract_id,
            "primal_input_value_ids": list(self.primal_input_value_ids),
            "primal_output_value_ids": list(self.primal_output_value_ids),
            "incoming_adjoint_value_ids": list(self.incoming_adjoint_value_ids),
            "alpha_gradient_owner_value_ids": list(self.alpha_gradient_owner_value_ids),
            "beta_gradient_owner_value_ids": list(self.beta_gradient_owner_value_ids),
            "compressed_output_layouts": list(self.compressed_output_layouts),
            "saved_value_ids": list(self.saved_value_ids),
            "recomputed_value_ids": list(self.recomputed_value_ids),
            "endpoint_policy": self.endpoint_policy,
            "higher_order_policy": self.higher_order_policy,
            "dense_a_escape_policy": self.dense_a_escape_policy,
            "mutation_policy": self.mutation_policy,
        }


@dataclass(frozen=True)
class VerificationRegionV1(_CanonicalSchemaObject):
    """One candidate region boundary; truth of witnesses belongs to GC0-1."""

    region_id: str
    op_ids: tuple[str, ...]
    input_value_ids: tuple[str, ...]
    output_value_ids: tuple[str, ...]
    parameter_value_ids: tuple[str, ...]
    external_use_ids: tuple[str, ...]
    effect_input_ids: tuple[str, ...]
    effect_output_ids: tuple[str, ...]
    saved_state_ids: tuple[str, ...]
    gradient_owner_ids: tuple[str, ...]
    entry_op_ids: tuple[str, ...]
    exit_op_ids: tuple[str, ...]
    postdominator_witness: str | None
    closed_world: bool
    fallback_policy: VerificationFallbackPolicy

    def validate(self) -> None:
        _require_identifier(self.region_id, "region_id")
        if not self.op_ids or not self.entry_op_ids or not self.exit_op_ids:
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                f"region {self.region_id} has no operations/entry/exit",
            )
        for label, values in (
            ("ops", self.op_ids),
            ("inputs", self.input_value_ids),
            ("outputs", self.output_value_ids),
            ("parameters", self.parameter_value_ids),
            ("external uses", self.external_use_ids),
            ("effect inputs", self.effect_input_ids),
            ("effect outputs", self.effect_output_ids),
            ("saved state", self.saved_state_ids),
            ("gradient owners", self.gradient_owner_ids),
            ("entry ops", self.entry_op_ids),
            ("exit ops", self.exit_op_ids),
        ):
            _require_unique(values, f"{self.region_id} {label}")
            for value in values:
                _require_identifier(value, f"{self.region_id} {label}")
        if not set(self.entry_op_ids).issubset(self.op_ids) or not set(
            self.exit_op_ids
        ).issubset(self.op_ids):
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                f"region {self.region_id} entry/exit is outside the region",
            )
        if set(self.input_value_ids) & set(self.output_value_ids):
            _reject(
                VerificationRejectionReason.UNSAFE_ALIAS_OR_LIFETIME,
                f"region {self.region_id} reuses boundary value identity",
            )
        if self.closed_world and (
            self.external_use_ids or not self.postdominator_witness
        ):
            _reject(
                (
                    VerificationRejectionReason.REGION_EXTERNAL_USE
                    if self.external_use_ids
                    else VerificationRejectionReason.REGION_NOT_POSTDOMINATED
                ),
                f"region {self.region_id} closed-world witness differs",
            )
        if not isinstance(self.fallback_policy, VerificationFallbackPolicy) or (
            self.fallback_policy != VerificationFallbackPolicy.REJECT_BEFORE_LAUNCH
        ):
            _reject(
                VerificationRejectionReason.RUNTIME_FALLBACK_REQUIRED,
                f"region {self.region_id} fallback is not fail-closed",
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "region_id": self.region_id,
            "op_ids": list(self.op_ids),
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
            "parameter_value_ids": list(self.parameter_value_ids),
            "external_use_ids": list(self.external_use_ids),
            "effect_input_ids": list(self.effect_input_ids),
            "effect_output_ids": list(self.effect_output_ids),
            "saved_state_ids": list(self.saved_state_ids),
            "gradient_owner_ids": list(self.gradient_owner_ids),
            "entry_op_ids": list(self.entry_op_ids),
            "exit_op_ids": list(self.exit_op_ids),
            "postdominator_witness": self.postdominator_witness,
            "closed_world": self.closed_world,
            "fallback_policy": self.fallback_policy.value,
        }


@dataclass(frozen=True)
class VerificationRuleV1(_CanonicalSchemaObject):
    """Guarded rewrite rule descriptor; this schema does not execute rules."""

    rule_id: str
    rule_version: str
    pattern_kind: VerificationRulePatternKind
    input_op_kinds: tuple[VerificationOpKind, ...]
    output_op_kinds: tuple[VerificationOpKind, ...]
    semantic_guards: tuple[str, ...]
    shape_guards: tuple[str, ...]
    effect_guards: tuple[str, ...]
    alias_guards: tuple[str, ...]
    external_use_guards: tuple[str, ...]
    vjp_guards: tuple[str, ...]
    replacement_builder_id: str
    estimated_boundary_elimination: int
    estimated_materialization_elimination: int
    fallback_policy: VerificationFallbackPolicy

    def validate(self) -> None:
        _require_identifier(self.rule_id, "rule_id")
        _require_identifier(self.rule_version, "rule_version")
        _require_identifier(self.replacement_builder_id, "replacement_builder_id")
        if not isinstance(self.pattern_kind, VerificationRulePatternKind):
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                f"rule {self.rule_id} pattern kind is not typed",
            )
        if (
            not self.input_op_kinds
            or not self.output_op_kinds
            or any(
                not isinstance(kind, VerificationOpKind)
                for kind in (*self.input_op_kinds, *self.output_op_kinds)
            )
        ):
            _reject(
                VerificationRejectionReason.UNSUPPORTED_OP_KIND,
                f"rule {self.rule_id} has unsupported operation kinds",
            )
        for label, values in (
            ("semantic guards", self.semantic_guards),
            ("shape guards", self.shape_guards),
            ("effect guards", self.effect_guards),
            ("alias guards", self.alias_guards),
            ("external-use guards", self.external_use_guards),
            ("VJP guards", self.vjp_guards),
        ):
            _require_unique(values, f"{self.rule_id} {label}")
            for value in values:
                _require_identifier(value, f"{self.rule_id} {label}")
        if (
            self.estimated_boundary_elimination < 0
            or self.estimated_materialization_elimination < 0
        ):
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                f"rule {self.rule_id} estimates are negative",
            )
        if not isinstance(self.fallback_policy, VerificationFallbackPolicy):
            _reject(
                VerificationRejectionReason.RUNTIME_FALLBACK_REQUIRED,
                f"rule {self.rule_id} fallback is not typed",
            )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "pattern_kind": self.pattern_kind.value,
            "input_op_kinds": [kind.value for kind in self.input_op_kinds],
            "output_op_kinds": [kind.value for kind in self.output_op_kinds],
            "semantic_guards": list(self.semantic_guards),
            "shape_guards": list(self.shape_guards),
            "effect_guards": list(self.effect_guards),
            "alias_guards": list(self.alias_guards),
            "external_use_guards": list(self.external_use_guards),
            "vjp_guards": list(self.vjp_guards),
            "replacement_builder_id": self.replacement_builder_id,
            "estimated_boundary_elimination": self.estimated_boundary_elimination,
            "estimated_materialization_elimination": self.estimated_materialization_elimination,
            "fallback_policy": self.fallback_policy.value,
        }


@dataclass(frozen=True)
class VerificationRuleRegistryV1:
    """Frozen rule registry; execution remains disabled in GC0-0."""

    registry_id: str
    rules: tuple[VerificationRuleV1, ...]
    execution_enabled: bool = False
    timing_recorded: bool = False
    performance_claimed: bool = False
    schema_version: str = VERIFICATION_RULE_REGISTRY_SCHEMA_V1

    def validate(self) -> None:
        _require_identifier(self.registry_id, "rule registry ID")
        if self.schema_version != VERIFICATION_RULE_REGISTRY_SCHEMA_V1:
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                "rule registry schema differs",
            )
        for rule in self.rules:
            rule.validate()
        rule_ids = tuple(rule.rule_id for rule in self.rules)
        _require_unique(rule_ids, "rule registry IDs")
        if tuple(sorted(rule_ids)) != REQUIRED_VERIFICATION_RULE_IDS_V1:
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                "rule registry does not contain the frozen v1 rule set",
            )
        if self.execution_enabled or self.timing_recorded or self.performance_claimed:
            _reject(
                VerificationRejectionReason.RUNTIME_FALLBACK_REQUIRED,
                "GC0-0 rule execution/timing/performance is not admitted",
            )

    def identity_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "registry_id": self.registry_id,
            "rules": [rule.to_dict() for rule in self.rules],
            "execution_enabled": self.execution_enabled,
            "timing_recorded": self.timing_recorded,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(self) -> str:
        return _hash(self.identity_payload())

    def canonical_json(self) -> str:
        return _canonical(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        payload = self.identity_payload()
        payload["registry_hash"] = self.stable_hash()
        return payload


@dataclass(frozen=True)
class LegalityResultV1:
    """Serializable analysis result; GC0-0 does not produce admitted results."""

    admitted: bool
    region_id: str
    ordered_op_ids: tuple[str, ...]
    boundary_input_ids: tuple[str, ...]
    boundary_output_ids: tuple[str, ...]
    external_use_witnesses: tuple[str, ...]
    effect_order_witnesses: tuple[str, ...]
    alias_witnesses: tuple[str, ...]
    dense_escape_witnesses: tuple[str, ...]
    vjp_witnesses: tuple[str, ...]
    rejection_reasons: tuple[VerificationRejectionReason, ...]
    schema_version: str = VERIFICATION_LEGALITY_RESULT_SCHEMA_V1

    def validate(self) -> None:
        _require_identifier(self.region_id, "legality region ID")
        if self.schema_version != VERIFICATION_LEGALITY_RESULT_SCHEMA_V1:
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                "legality result schema differs",
            )
        for label, values in (
            ("ordered ops", self.ordered_op_ids),
            ("boundary inputs", self.boundary_input_ids),
            ("boundary outputs", self.boundary_output_ids),
            ("external-use witnesses", self.external_use_witnesses),
            ("effect witnesses", self.effect_order_witnesses),
            ("alias witnesses", self.alias_witnesses),
            ("dense-escape witnesses", self.dense_escape_witnesses),
            ("VJP witnesses", self.vjp_witnesses),
        ):
            _require_unique(values, f"legality {label}")
            for value in values:
                _require_identifier(value, f"legality {label}")
        if any(
            not isinstance(reason, VerificationRejectionReason)
            for reason in self.rejection_reasons
        ) or len(self.rejection_reasons) != len(set(self.rejection_reasons)):
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                "legality rejection reasons are not typed and unique",
            )
        witness_groups = (
            self.external_use_witnesses,
            self.effect_order_witnesses,
            self.alias_witnesses,
            self.dense_escape_witnesses,
            self.vjp_witnesses,
        )
        if self.admitted and (
            self.rejection_reasons
            or not self.ordered_op_ids
            or any(not group for group in witness_groups)
        ):
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                "admitted legality result lacks complete witnesses",
            )
        if not self.admitted and not self.rejection_reasons:
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                "rejected legality result has no stable reason",
            )

    def identity_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "admitted": self.admitted,
            "region_id": self.region_id,
            "ordered_op_ids": list(self.ordered_op_ids),
            "boundary_input_ids": list(self.boundary_input_ids),
            "boundary_output_ids": list(self.boundary_output_ids),
            "external_use_witnesses": list(self.external_use_witnesses),
            "effect_order_witnesses": list(self.effect_order_witnesses),
            "alias_witnesses": list(self.alias_witnesses),
            "dense_escape_witnesses": list(self.dense_escape_witnesses),
            "vjp_witnesses": list(self.vjp_witnesses),
            "rejection_reasons": [reason.value for reason in self.rejection_reasons],
        }

    def stable_hash(self) -> str:
        return _hash(self.identity_payload())

    def canonical_json(self) -> str:
        return _canonical(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        payload = self.identity_payload()
        payload["analysis_hash"] = self.stable_hash()
        return payload


@dataclass(frozen=True)
class VerificationProgramV1:
    """Program identity referencing separately serialized graph objects."""

    program_id: str
    source_graph_hash: str
    parameter_schema_hash: str
    numeric_policy_id: str
    target_contract_id: str
    region_ids: tuple[str, ...]
    entry_region_ids: tuple[str, ...]
    external_value_ids: tuple[str, ...]
    external_effect_ids: tuple[str, ...]
    rule_registry_hash: str
    semantic_owner: str = VERIFICATION_SEMANTIC_OWNER_V1
    schema_version: str = VERIFICATION_GRAPH_SCHEMA_V1

    def validate(self) -> None:
        for label, value in (
            ("program_id", self.program_id),
            ("numeric_policy_id", self.numeric_policy_id),
            ("target_contract_id", self.target_contract_id),
        ):
            _require_identifier(value, label)
        for label, value in (
            ("source_graph_hash", self.source_graph_hash),
            ("parameter_schema_hash", self.parameter_schema_hash),
            ("rule_registry_hash", self.rule_registry_hash),
        ):
            if not _is_hash(value):
                _reject(
                    VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                    f"program {label} is not SHA-256",
                )
        if (
            self.schema_version != VERIFICATION_GRAPH_SCHEMA_V1
            or self.semantic_owner != VERIFICATION_SEMANTIC_OWNER_V1
        ):
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                "program schema/semantic owner differs",
            )
        for label, values in (
            ("regions", self.region_ids),
            ("entry regions", self.entry_region_ids),
            ("external values", self.external_value_ids),
            ("external effects", self.external_effect_ids),
        ):
            _require_unique(values, f"program {label}")
            for value in values:
                _require_identifier(value, f"program {label}")
        if (
            not self.region_ids
            or not self.entry_region_ids
            or not set(self.entry_region_ids).issubset(self.region_ids)
        ):
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                "program entry regions differ",
            )

    def identity_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "program_id": self.program_id,
            "semantic_owner": self.semantic_owner,
            "source_graph_hash": self.source_graph_hash,
            "parameter_schema_hash": self.parameter_schema_hash,
            "numeric_policy_id": self.numeric_policy_id,
            "target_contract_id": self.target_contract_id,
            "region_ids": list(self.region_ids),
            "entry_region_ids": list(self.entry_region_ids),
            "external_value_ids": list(self.external_value_ids),
            "external_effect_ids": list(self.external_effect_ids),
            "rule_registry_hash": self.rule_registry_hash,
        }

    def stable_hash(self) -> str:
        return _hash(self.identity_payload())

    def canonical_json(self) -> str:
        return _canonical(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        payload = self.identity_payload()
        payload["program_hash"] = self.stable_hash()
        return payload


@dataclass(frozen=True)
class VerificationGraphModuleV1:
    """Canonical container for GC0-0 graph schema round-trips."""

    module_id: str
    program: VerificationProgramV1
    regions: tuple[VerificationRegionV1, ...]
    values: tuple[VerificationValueV1, ...]
    ops: tuple[VerificationOpV1, ...]
    effects: tuple[VerificationEffectTokenV1, ...]
    vjp_contracts: tuple[VerificationVJPContractV1, ...]
    rule_registry: VerificationRuleRegistryV1
    timing_recorded: bool = False
    performance_claimed: bool = False
    schema_version: str = VERIFICATION_GRAPH_MODULE_SCHEMA_V1

    def validate(self) -> None:  # pylint: disable=too-many-branches
        _require_identifier(self.module_id, "module_id")
        if self.schema_version != VERIFICATION_GRAPH_MODULE_SCHEMA_V1:
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                "verification graph module schema differs",
            )
        if self.timing_recorded or self.performance_claimed:
            _reject(
                VerificationRejectionReason.RUNTIME_FALLBACK_REQUIRED,
                "GC0-0 module records timing/performance",
            )
        self.rule_registry.validate()
        self.program.validate()
        for region_item in self.regions:
            region_item.validate()
        for value_item in self.values:
            value_item.validate()
        for op_item in self.ops:
            op_item.validate()
        for effect_item in self.effects:
            effect_item.validate()
        for vjp_item in self.vjp_contracts:
            vjp_item.validate()
        region_map = {item.region_id: item for item in self.regions}
        value_map = {item.value_id: item for item in self.values}
        op_map = {item.op_id: item for item in self.ops}
        effect_map = {item.effect_id: item for item in self.effects}
        vjp_map = {item.contract_id: item for item in self.vjp_contracts}
        for label, sequence, mapping in (
            ("regions", self.regions, region_map),
            ("values", self.values, value_map),
            ("ops", self.ops, op_map),
            ("effects", self.effects, effect_map),
            ("VJP contracts", self.vjp_contracts, vjp_map),
        ):
            if len(sequence) != len(mapping):
                _reject(
                    VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                    f"module {label} IDs repeat",
                )
        identity_sets = (
            set(region_map),
            set(value_map),
            set(op_map),
            set(effect_map),
            set(vjp_map),
        )
        for ordinal, left in enumerate(identity_sets):
            if any(left & right for right in identity_sets[ordinal + 1 :]):
                _reject(
                    VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                    "graph object IDs are not globally unique",
                )
        if (
            tuple(region_map) != self.program.region_ids
            or self.program.rule_registry_hash != self.rule_registry.stable_hash()
            or not set(self.program.external_value_ids).issubset(value_map)
            or not set(self.program.external_effect_ids).issubset(effect_map)
        ):
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                "program references differ from module objects",
            )
        for op in self.ops:
            if not set(
                (*op.input_value_ids, *op.output_value_ids, *op.parameter_value_ids)
            ).issubset(value_map) or not set(
                (*op.effect_read_ids, *op.effect_write_ids)
            ).issubset(
                effect_map
            ):
                _reject(
                    VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                    f"op {op.op_id} references unknown values/effects",
                )
            if op.vjp_contract_id is not None and op.vjp_contract_id not in vjp_map:
                _reject(
                    VerificationRejectionReason.VJP_OWNER_OR_SAVED_STATE_MISMATCH,
                    f"op {op.op_id} references unknown VJP contract",
                )
        for region in self.regions:
            if (
                not set(region.op_ids).issubset(op_map)
                or not set(
                    (
                        *region.input_value_ids,
                        *region.output_value_ids,
                        *region.parameter_value_ids,
                        *region.saved_state_ids,
                        *region.gradient_owner_ids,
                    )
                ).issubset(value_map)
                or not set(
                    (*region.effect_input_ids, *region.effect_output_ids)
                ).issubset(effect_map)
            ):
                _reject(
                    VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                    f"region {region.region_id} references unknown objects",
                )
        for contract in self.vjp_contracts:
            referenced = set(
                (
                    *contract.primal_input_value_ids,
                    *contract.primal_output_value_ids,
                    *contract.incoming_adjoint_value_ids,
                    *contract.alpha_gradient_owner_value_ids,
                    *contract.beta_gradient_owner_value_ids,
                    *contract.saved_value_ids,
                    *contract.recomputed_value_ids,
                )
            )
            if not referenced.issubset(value_map):
                _reject(
                    VerificationRejectionReason.VJP_OWNER_OR_SAVED_STATE_MISMATCH,
                    f"VJP contract {contract.contract_id} references unknown values",
                )
            if any(
                value_map[value_id].role == VerificationValueRole.COEFFICIENT
                and value_map[value_id].representation
                == VerificationRepresentation.DENSE
                for value_id in contract.saved_value_ids
            ):
                _reject(
                    VerificationRejectionReason.DENSE_A_ESCAPE,
                    f"VJP contract {contract.contract_id} saves dense coefficients",
                )

    def identity_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "module_id": self.module_id,
            "program": self.program.to_dict(),
            "regions": [item.to_dict() for item in self.regions],
            "values": [item.to_dict() for item in self.values],
            "ops": [item.to_dict() for item in self.ops],
            "effects": [item.to_dict() for item in self.effects],
            "vjp_contracts": [item.to_dict() for item in self.vjp_contracts],
            "rule_registry": self.rule_registry.to_dict(),
            "timing_recorded": self.timing_recorded,
            "performance_claimed": self.performance_claimed,
        }

    def canonical_json(self) -> str:
        payload = self.identity_payload()
        payload["module_hash"] = _hash(payload)
        return _canonical(payload)

    def stable_hash(self) -> str:
        return _hash(self.identity_payload())

    @classmethod
    def from_canonical_json(cls, encoded: str) -> "VerificationGraphModuleV1":
        """Reconstruct and validate a canonical module without executing it."""

        raw = json.loads(encoded)
        if not isinstance(raw, dict) or _canonical(raw) != encoded:
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                "verification graph JSON is not canonical",
            )
        payload = cast(dict[str, object], raw)
        expected_hash = payload.pop("module_hash", None)
        module = _module_from_dict(payload)
        if expected_hash != module.stable_hash() or module.canonical_json() != encoded:
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                "verification graph module hash differs",
            )
        return module


EnumT = TypeVar("EnumT", bound=Enum)


def _enum(enum_type: type[EnumT], value: object) -> EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        _reject(
            VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
            f"{enum_type.__name__} value differs",
        )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        _reject(
            VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
            f"{label} is not an object",
        )
    return cast(Mapping[str, object], value)


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        _reject(
            VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
            f"{label} is not an array",
        )
    return tuple(cast(list[object], value))


def _strings(value: object, label: str) -> tuple[str, ...]:
    result = _sequence(value, label)
    if any(not isinstance(item, str) for item in result):
        _reject(
            VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
            f"{label} contains a non-string",
        )
    return cast(tuple[str, ...], result)


def _value_from_dict(raw: object) -> VerificationValueV1:
    item = _mapping(raw, "value")
    shape_raw = _sequence(item["shape"], "value shape")
    if any(value is not None and not isinstance(value, int) for value in shape_raw):
        _reject(
            VerificationRejectionReason.DYNAMIC_SHAPE_UNBOUND,
            "value shape contains an invalid dimension",
        )
    return VerificationValueV1(
        value_id=cast(str, item["value_id"]),
        role=_enum(VerificationValueRole, item["role"]),
        shape=cast(tuple[int | None, ...], shape_raw),
        dtype=cast(str, item["dtype"]),
        device_kind=cast(str, item["device_kind"]),
        layout=cast(str, item["layout"]),
        strides=cast(tuple[int, ...], _sequence(item["strides"], "value strides")),
        axis_roles=tuple(
            _enum(VerificationAxisRole, value)
            for value in _sequence(item["axis_roles"], "value axis roles")
        ),
        polarity=_enum(VerificationPolarity, item["polarity"]),
        representation=_enum(VerificationRepresentation, item["representation"]),
        requires_grad=cast(bool, item["requires_grad"]),
        state_version=cast(str | None, item["state_version"]),
        lineage_id=cast(str | None, item["lineage_id"]),
        storage_class=_enum(VerificationStorageClass, item["storage_class"]),
        alias_set=cast(str | None, item["alias_set"]),
        producer_op_id=cast(str | None, item["producer_op_id"]),
        consumer_op_ids=_strings(item["consumer_op_ids"], "value consumers"),
        external_use_count=cast(int, item["external_use_count"]),
        present=cast(bool, item["present"]),
        finite_policy=_enum(VerificationFinitePolicy, item["finite_policy"]),
    )


def _op_from_dict(raw: object) -> VerificationOpV1:
    item = _mapping(raw, "op")
    attributes: list[tuple[str, object]] = []
    for pair_raw in _sequence(item["attributes"], "op attributes"):
        pair = _sequence(pair_raw, "op attribute pair")
        if len(pair) != 2 or not isinstance(pair[0], str):
            _reject(
                VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
                "op attribute pair differs",
            )
        attributes.append((pair[0], _freeze_attribute(pair[1])))
    return VerificationOpV1(
        op_id=cast(str, item["op_id"]),
        op_kind=_enum(VerificationOpKind, item["op_kind"]),
        semantic_version=cast(str, item["semantic_version"]),
        input_value_ids=_strings(item["input_value_ids"], "op inputs"),
        output_value_ids=_strings(item["output_value_ids"], "op outputs"),
        parameter_value_ids=_strings(item["parameter_value_ids"], "op parameters"),
        effect_read_ids=_strings(item["effect_read_ids"], "op effect reads"),
        effect_write_ids=_strings(item["effect_write_ids"], "op effect writes"),
        attributes=tuple(attributes),
        bound_direction=_enum(VerificationPolarity, item["bound_direction"]),
        numeric_policy_id=cast(str, item["numeric_policy_id"]),
        vjp_contract_id=cast(str | None, item["vjp_contract_id"]),
        source_op_ids=_strings(item["source_op_ids"], "op source IDs"),
    )


def _effect_from_dict(raw: object) -> VerificationEffectTokenV1:
    item = _mapping(raw, "effect")
    return VerificationEffectTokenV1(
        effect_id=cast(str, item["effect_id"]),
        kind=_enum(VerificationEffectKind, item["kind"]),
        resource_id=cast(str, item["resource_id"]),
        input_version=cast(str, item["input_version"]),
        output_version=cast(str, item["output_version"]),
        access=_enum(VerificationEffectAccess, item["access"]),
        ordinal=cast(int, item["ordinal"]),
    )


def _vjp_from_dict(raw: object) -> VerificationVJPContractV1:
    item = _mapping(raw, "VJP contract")
    return VerificationVJPContractV1(
        contract_id=cast(str, item["contract_id"]),
        primal_input_value_ids=_strings(item["primal_input_value_ids"], "VJP inputs"),
        primal_output_value_ids=_strings(
            item["primal_output_value_ids"], "VJP outputs"
        ),
        incoming_adjoint_value_ids=_strings(
            item["incoming_adjoint_value_ids"], "VJP adjoints"
        ),
        alpha_gradient_owner_value_ids=_strings(
            item["alpha_gradient_owner_value_ids"], "VJP alpha owners"
        ),
        beta_gradient_owner_value_ids=_strings(
            item["beta_gradient_owner_value_ids"], "VJP beta owners"
        ),
        compressed_output_layouts=_strings(
            item["compressed_output_layouts"], "VJP output layouts"
        ),
        saved_value_ids=_strings(item["saved_value_ids"], "VJP saved values"),
        recomputed_value_ids=_strings(
            item["recomputed_value_ids"], "VJP recomputed values"
        ),
        endpoint_policy=cast(str, item["endpoint_policy"]),
        higher_order_policy=cast(str, item["higher_order_policy"]),
        dense_a_escape_policy=cast(str, item["dense_a_escape_policy"]),
        mutation_policy=cast(str, item["mutation_policy"]),
    )


def _region_from_dict(raw: object) -> VerificationRegionV1:
    item = _mapping(raw, "region")
    return VerificationRegionV1(
        region_id=cast(str, item["region_id"]),
        op_ids=_strings(item["op_ids"], "region ops"),
        input_value_ids=_strings(item["input_value_ids"], "region inputs"),
        output_value_ids=_strings(item["output_value_ids"], "region outputs"),
        parameter_value_ids=_strings(item["parameter_value_ids"], "region parameters"),
        external_use_ids=_strings(item["external_use_ids"], "region external uses"),
        effect_input_ids=_strings(item["effect_input_ids"], "region effect inputs"),
        effect_output_ids=_strings(item["effect_output_ids"], "region effect outputs"),
        saved_state_ids=_strings(item["saved_state_ids"], "region saved state"),
        gradient_owner_ids=_strings(
            item["gradient_owner_ids"], "region gradient owners"
        ),
        entry_op_ids=_strings(item["entry_op_ids"], "region entry ops"),
        exit_op_ids=_strings(item["exit_op_ids"], "region exit ops"),
        postdominator_witness=cast(str | None, item["postdominator_witness"]),
        closed_world=cast(bool, item["closed_world"]),
        fallback_policy=_enum(VerificationFallbackPolicy, item["fallback_policy"]),
    )


def _rule_from_dict(raw: object) -> VerificationRuleV1:
    item = _mapping(raw, "rule")
    return VerificationRuleV1(
        rule_id=cast(str, item["rule_id"]),
        rule_version=cast(str, item["rule_version"]),
        pattern_kind=_enum(VerificationRulePatternKind, item["pattern_kind"]),
        input_op_kinds=tuple(
            _enum(VerificationOpKind, value)
            for value in _sequence(item["input_op_kinds"], "rule input kinds")
        ),
        output_op_kinds=tuple(
            _enum(VerificationOpKind, value)
            for value in _sequence(item["output_op_kinds"], "rule output kinds")
        ),
        semantic_guards=_strings(item["semantic_guards"], "semantic guards"),
        shape_guards=_strings(item["shape_guards"], "shape guards"),
        effect_guards=_strings(item["effect_guards"], "effect guards"),
        alias_guards=_strings(item["alias_guards"], "alias guards"),
        external_use_guards=_strings(
            item["external_use_guards"], "external-use guards"
        ),
        vjp_guards=_strings(item["vjp_guards"], "VJP guards"),
        replacement_builder_id=cast(str, item["replacement_builder_id"]),
        estimated_boundary_elimination=cast(
            int, item["estimated_boundary_elimination"]
        ),
        estimated_materialization_elimination=cast(
            int, item["estimated_materialization_elimination"]
        ),
        fallback_policy=_enum(VerificationFallbackPolicy, item["fallback_policy"]),
    )


def _registry_from_dict(raw: object) -> VerificationRuleRegistryV1:
    item = _mapping(raw, "rule registry")
    registry = VerificationRuleRegistryV1(
        registry_id=cast(str, item["registry_id"]),
        rules=tuple(
            _rule_from_dict(value)
            for value in _sequence(item["rules"], "registry rules")
        ),
        execution_enabled=cast(bool, item["execution_enabled"]),
        timing_recorded=cast(bool, item["timing_recorded"]),
        performance_claimed=cast(bool, item["performance_claimed"]),
        schema_version=cast(str, item["schema_version"]),
    )
    if item.get("registry_hash") != registry.stable_hash():
        _reject(
            VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
            "rule registry hash differs",
        )
    return registry


def _program_from_dict(raw: object) -> VerificationProgramV1:
    item = _mapping(raw, "program")
    program = VerificationProgramV1(
        program_id=cast(str, item["program_id"]),
        source_graph_hash=cast(str, item["source_graph_hash"]),
        parameter_schema_hash=cast(str, item["parameter_schema_hash"]),
        numeric_policy_id=cast(str, item["numeric_policy_id"]),
        target_contract_id=cast(str, item["target_contract_id"]),
        region_ids=_strings(item["region_ids"], "program regions"),
        entry_region_ids=_strings(item["entry_region_ids"], "program entry regions"),
        external_value_ids=_strings(
            item["external_value_ids"], "program external values"
        ),
        external_effect_ids=_strings(
            item["external_effect_ids"], "program external effects"
        ),
        rule_registry_hash=cast(str, item["rule_registry_hash"]),
        semantic_owner=cast(str, item["semantic_owner"]),
        schema_version=cast(str, item["schema_version"]),
    )
    if item.get("program_hash") != program.stable_hash():
        _reject(
            VerificationRejectionReason.RECEIPT_IDENTITY_MISMATCH,
            "program hash differs",
        )
    return program


def _module_from_dict(item: Mapping[str, object]) -> VerificationGraphModuleV1:
    return VerificationGraphModuleV1(
        module_id=cast(str, item["module_id"]),
        program=_program_from_dict(item["program"]),
        regions=tuple(
            _region_from_dict(value)
            for value in _sequence(item["regions"], "module regions")
        ),
        values=tuple(
            _value_from_dict(value)
            for value in _sequence(item["values"], "module values")
        ),
        ops=tuple(
            _op_from_dict(value) for value in _sequence(item["ops"], "module ops")
        ),
        effects=tuple(
            _effect_from_dict(value)
            for value in _sequence(item["effects"], "module effects")
        ),
        vjp_contracts=tuple(
            _vjp_from_dict(value)
            for value in _sequence(item["vjp_contracts"], "module VJP contracts")
        ),
        rule_registry=_registry_from_dict(item["rule_registry"]),
        timing_recorded=cast(bool, item["timing_recorded"]),
        performance_claimed=cast(bool, item["performance_claimed"]),
        schema_version=cast(str, item["schema_version"]),
    )


def build_gc0_rule_registry_v1() -> VerificationRuleRegistryV1:
    """Build the frozen, non-executable GC0-0 rule registry."""

    specifications = (
        (
            "V-C1-terminal-concretize-v1",
            VerificationRulePatternKind.TERMINAL_REGION,
            (VerificationOpKind.INPUT_CONCRETIZE,),
        ),
        (
            "V-D1-residual-diamond-v1",
            VerificationRulePatternKind.RESIDUAL_DIAMOND,
            (VerificationOpKind.RESIDUAL_DIAMOND,),
        ),
        (
            "V-H1-lower-upper-tuple-v1",
            VerificationRulePatternKind.DATAFLOW_PATTERN,
            (VerificationOpKind.CONV2D_RIGHT,),
        ),
        (
            "V-M1-certified-arena-reuse-v1",
            VerificationRulePatternKind.MEMORY_REUSE,
            (VerificationOpKind.LAYOUT_NORMALIZE,),
        ),
        (
            "V-R1-relax-sign-affine-v1",
            VerificationRulePatternKind.EXACT_CHAIN,
            (
                VerificationOpKind.RELU_RELAXATION,
                VerificationOpKind.SIGN_SELECT,
                VerificationOpKind.LINEAR_RIGHT,
            ),
        ),
        (
            "V-R2-compressed-alpha-gather-v1",
            VerificationRulePatternKind.EXACT_CHAIN,
            (VerificationOpKind.COMPRESSED_ALPHA_GATHER,),
        ),
        (
            "V-R3-sparse-beta-inject-v1",
            VerificationRulePatternKind.EXACT_CHAIN,
            (VerificationOpKind.SPARSE_BETA_INJECT,),
        ),
        (
            "V-VJP1-minimal-saved-state-v1",
            VerificationRulePatternKind.VJP_REGION,
            (VerificationOpKind.MINIMAL_STATE_VJP,),
        ),
    )
    rules = tuple(
        VerificationRuleV1(
            rule_id=rule_id,
            rule_version="1",
            pattern_kind=pattern_kind,
            input_op_kinds=op_kinds,
            output_op_kinds=op_kinds,
            semantic_guards=("verification-semantic-owner",),
            shape_guards=("static-shape",),
            effect_guards=("effect-versions-exact",),
            alias_guards=("alias-analysis-required",),
            external_use_guards=("closed-world-required",),
            vjp_guards=("first-order-minimal-saved-state",),
            replacement_builder_id=f"builder:{rule_id}",
            estimated_boundary_elimination=0,
            estimated_materialization_elimination=0,
            fallback_policy=VerificationFallbackPolicy.REJECT_BEFORE_LAUNCH,
        )
        for rule_id, pattern_kind, op_kinds in specifications
    )
    registry = VerificationRuleRegistryV1(
        registry_id="verification-rule-registry-v1",
        rules=rules,
    )
    registry.validate()
    return registry


__all__ = [
    "GC01_ANALYSIS_REJECTION_REASONS",
    "GC0_DIRECT_REJECTION_REASONS",
    "LegalityResultV1",
    "REQUIRED_VERIFICATION_RULE_IDS_V1",
    "VerificationAxisRole",
    "VerificationEffectAccess",
    "VerificationEffectKind",
    "VerificationEffectTokenV1",
    "VerificationFallbackPolicy",
    "VerificationFinitePolicy",
    "VerificationGraphModuleV1",
    "VerificationGraphValidationError",
    "VerificationOpKind",
    "VerificationOpV1",
    "VerificationPolarity",
    "VerificationProgramV1",
    "VerificationRegionV1",
    "VerificationRejectionReason",
    "VerificationRepresentation",
    "VerificationRulePatternKind",
    "VerificationRuleRegistryV1",
    "VerificationRuleV1",
    "VerificationStorageClass",
    "VerificationVJPContractV1",
    "VerificationValueRole",
    "VerificationValueV1",
    "build_gc0_rule_registry_v1",
    "freeze_verification_attributes",
]
