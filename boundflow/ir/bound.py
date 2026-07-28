"""First-class Bound IR schemas and deterministic validation.

The IR in this module is deliberately independent from PyTorch, TVM, and the
runtime CROWN implementation. Runtime domain-state classes still inherit from
``DomainState`` for compatibility, but those mutable runtime objects are not
part of the serialized Bound IR.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Optional, Tuple, TypeAlias

BOUND_IR_SCHEMA_VERSION = "boundflow.bound_ir/v1.0"


@dataclass
class DomainState:
    """Compatibility base for runtime states; not a serialized IR value."""


class BoundValueRole(Enum):
    """Semantic role carried by one Bound IR value."""

    PERTURBATION = "perturbation"
    INTERVAL = "interval"
    COEFFICIENT = "coefficient"
    BIAS = "bias"
    RELAXATION = "relaxation"
    SPLIT = "split"
    OBJECTIVE = "objective"


class BoundPolarity(Enum):
    """Whether a value belongs to the lower, upper, or paired computation."""

    LOWER = "lower"
    UPPER = "upper"
    BOTH = "both"


class BoundRepresentation(Enum):
    """Physical/logical representation selected for a bound value."""

    DENSE = "dense"
    STRUCTURED = "structured"
    CHUNKED = "chunked"
    SCALAR = "scalar"


class BatchAxisKind(Enum):
    """Verification-specific batch dimensions kept distinct in the IR."""

    SAMPLE = "sample"
    SPEC = "spec"
    DOMAIN = "domain"


@dataclass(frozen=True)
class BoundBatchAxis:
    """One named batch axis and its tensor dimension."""

    kind: BatchAxisKind
    dimension: int

    def validate(self, *, rank: int) -> None:
        """Reject an axis outside its owning tensor rank."""

        if self.dimension < 0 or self.dimension >= rank:
            raise ValueError(
                f"batch axis {self.kind.value} dimension {self.dimension} "
                f"is outside rank {rank}"
            )

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible axis fields."""

        return {"kind": self.kind.value, "dimension": self.dimension}


@dataclass(frozen=True)
class BoundTensorType:
    """Backend-neutral tensor type used by Bound IR values."""

    shape: Tuple[Optional[int], ...]
    dtype: str
    layout: Optional[str] = None
    device: Optional[str] = None
    batch_axes: Tuple[BoundBatchAxis, ...] = ()

    def validate(self) -> None:
        """Validate shape metadata and verification batch-axis identity."""

        if not self.dtype:
            raise ValueError("bound tensor dtype must be non-empty")
        if self.layout is not None and not self.layout:
            raise ValueError("bound tensor layout must be non-empty when present")
        if self.device is not None and not self.device:
            raise ValueError("bound tensor device must be non-empty when present")
        for dimension in self.shape:
            if dimension is not None and dimension <= 0:
                raise ValueError(
                    f"bound tensor dimensions must be positive or dynamic, got {self.shape}"
                )
        for axis in self.batch_axes:
            axis.validate(rank=len(self.shape))
        kinds = [axis.kind for axis in self.batch_axes]
        dimensions = [axis.dimension for axis in self.batch_axes]
        if len(kinds) != len(set(kinds)):
            raise ValueError("bound tensor contains duplicate batch-axis kinds")
        if len(dimensions) != len(set(dimensions)):
            raise ValueError(
                "bound tensor contains multiple batch axes on one dimension"
            )
        if dimensions != sorted(dimensions):
            raise ValueError("bound tensor batch axes must be sorted by dimension")

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible tensor-type fields."""

        self.validate()
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "layout": self.layout,
            "device": self.device,
            "batch_axes": [axis.to_dict() for axis in self.batch_axes],
        }


@dataclass(frozen=True)
class BoundValue:
    """SSA value in the Bound IR graph."""

    value_id: str
    tensor_type: BoundTensorType
    role: BoundValueRole
    polarity: BoundPolarity
    representation: BoundRepresentation
    state_version: Optional[str] = None
    source_primal_value_id: Optional[str] = None

    def validate(self) -> None:
        """Validate one SSA value and its source identity."""

        if not self.value_id:
            raise ValueError("bound value_id must be non-empty")
        if self.state_version is not None and not self.state_version:
            raise ValueError("state_version must be non-empty when present")
        if self.source_primal_value_id is not None and not self.source_primal_value_id:
            raise ValueError("source_primal_value_id must be non-empty when present")
        self.tensor_type.validate()

    def semantic_signature(self) -> tuple[object, ...]:
        """Return fields that a representation-only conversion must preserve."""

        return (
            self.tensor_type,
            self.role,
            self.polarity,
            self.state_version,
            self.source_primal_value_id,
        )

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible value fields."""

        self.validate()
        return {
            "value_id": self.value_id,
            "tensor_type": self.tensor_type.to_dict(),
            "role": self.role.value,
            "polarity": self.polarity.value,
            "representation": self.representation.value,
            "state_version": self.state_version,
            "source_primal_value_id": self.source_primal_value_id,
        }


class PerturbationKind(Enum):
    """Perturbation families described by the verification spec."""

    LINF = "linf"
    L2 = "l2"
    L1 = "l1"
    BOX = "box"


class ObjectiveKind(Enum):
    """Objective forms supported by the first schema."""

    IDENTITY = "identity"
    LINEAR = "linear"
    MARGIN = "margin"


@dataclass(frozen=True)
class PerturbationSpec:
    """Typed identity for one perturbed primal input."""

    perturbation_id: str
    input_primal_value_id: str
    kind: PerturbationKind
    radius: Optional[float] = None
    payload_hash: Optional[str] = None

    def validate(self) -> None:
        """Validate one perturbation identity without runtime payloads."""

        if not self.perturbation_id or not self.input_primal_value_id:
            raise ValueError("perturbation IDs and primal input IDs must be non-empty")
        if self.kind == PerturbationKind.BOX:
            if self.payload_hash is None:
                raise ValueError("box perturbation requires payload_hash")
        elif self.radius is None:
            raise ValueError(f"{self.kind.value} perturbation requires radius")
        if self.radius is not None and (
            not math.isfinite(self.radius) or self.radius < 0.0
        ):
            raise ValueError("perturbation radius must be finite and non-negative")
        if self.payload_hash is not None and not self.payload_hash:
            raise ValueError("perturbation payload_hash must be non-empty when present")

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible perturbation fields."""

        self.validate()
        return {
            "perturbation_id": self.perturbation_id,
            "input_primal_value_id": self.input_primal_value_id,
            "kind": self.kind.value,
            "radius": self.radius,
            "payload_hash": self.payload_hash,
        }


@dataclass(frozen=True)
class ObjectiveSpec:
    """Typed identity for one verification objective set."""

    objective_id: str
    output_primal_value_id: str
    kind: ObjectiveKind
    num_objectives: Optional[int] = None
    payload_hash: Optional[str] = None

    def validate(self) -> None:
        """Validate objective cardinality and payload identity."""

        if not self.objective_id or not self.output_primal_value_id:
            raise ValueError("objective IDs and primal output IDs must be non-empty")
        if self.num_objectives is not None and self.num_objectives <= 0:
            raise ValueError("num_objectives must be positive when present")
        if self.kind in {ObjectiveKind.LINEAR, ObjectiveKind.MARGIN}:
            if self.payload_hash is None:
                raise ValueError(f"{self.kind.value} objective requires payload_hash")
        if self.payload_hash is not None and not self.payload_hash:
            raise ValueError("objective payload_hash must be non-empty when present")

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible objective fields."""

        self.validate()
        return {
            "objective_id": self.objective_id,
            "output_primal_value_id": self.output_primal_value_id,
            "kind": self.kind.value,
            "num_objectives": self.num_objectives,
            "payload_hash": self.payload_hash,
        }


@dataclass(frozen=True)
class VerificationSpec:
    """Strongly typed verification input contract."""

    perturbations: Tuple[PerturbationSpec, ...]
    objectives: Tuple[ObjectiveSpec, ...]
    requested_bounds: Tuple[BoundPolarity, ...]
    numeric_policy: str

    def validate(self) -> None:
        """Validate the complete verification input contract."""

        if not self.perturbations:
            raise ValueError("verification spec requires at least one perturbation")
        if not self.objectives:
            raise ValueError("verification spec requires at least one objective")
        if not self.requested_bounds:
            raise ValueError("verification spec requires requested bounds")
        if not self.numeric_policy:
            raise ValueError("verification spec numeric_policy must be non-empty")
        for perturbation in self.perturbations:
            perturbation.validate()
        for objective in self.objectives:
            objective.validate()
        perturbation_ids = [
            perturbation.perturbation_id for perturbation in self.perturbations
        ]
        objective_ids = [objective.objective_id for objective in self.objectives]
        if len(perturbation_ids) != len(set(perturbation_ids)):
            raise ValueError("verification spec contains duplicate perturbation IDs")
        if len(objective_ids) != len(set(objective_ids)):
            raise ValueError("verification spec contains duplicate objective IDs")
        if len(self.requested_bounds) != len(set(self.requested_bounds)):
            raise ValueError("verification spec contains duplicate requested bounds")
        if (
            BoundPolarity.BOTH in self.requested_bounds
            and len(self.requested_bounds) != 1
        ):
            raise ValueError("requested bound BOTH cannot be combined with LOWER/UPPER")

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible specification fields."""

        self.validate()
        return {
            "perturbations": [
                perturbation.to_dict() for perturbation in self.perturbations
            ],
            "objectives": [objective.to_dict() for objective in self.objectives],
            "requested_bounds": [bound.value for bound in self.requested_bounds],
            "numeric_policy": self.numeric_policy,
        }


class BoundMethodKind(Enum):
    """Bound method whose state semantics the module represents."""

    INTERVAL = "interval"
    CROWN = "crown"
    ALPHA_CROWN = "alpha_crown"
    ALPHA_BETA_CROWN = "alpha_beta_crown"


@dataclass(frozen=True)
class BoundDomainConfig:
    """Method/state capability carried by a Bound IR module."""

    method: BoundMethodKind
    requires_grad: bool = False
    alpha_enabled: bool = False
    beta_enabled: bool = False
    split_state_present: bool = False

    def validate(self) -> None:
        """Reject method/state combinations with incompatible semantics."""

        if self.beta_enabled and not self.alpha_enabled:
            raise ValueError("beta state requires alpha state")
        if self.method in {BoundMethodKind.INTERVAL, BoundMethodKind.CROWN}:
            if self.alpha_enabled or self.beta_enabled or self.split_state_present:
                raise ValueError(
                    f"{self.method.value} cannot carry alpha/beta/split state"
                )
        elif self.method == BoundMethodKind.ALPHA_CROWN:
            if not self.alpha_enabled or self.beta_enabled or self.split_state_present:
                raise ValueError(
                    "alpha_crown requires alpha and rejects beta/split state"
                )
        elif self.method == BoundMethodKind.ALPHA_BETA_CROWN:
            if not self.alpha_enabled or not self.beta_enabled:
                raise ValueError("alpha_beta_crown requires alpha and beta state")

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible domain-state fields."""

        self.validate()
        return {
            "method": self.method.value,
            "requires_grad": self.requires_grad,
            "alpha_enabled": self.alpha_enabled,
            "beta_enabled": self.beta_enabled,
            "split_state_present": self.split_state_present,
        }


class BoundOpKind(Enum):
    """Semantic operations represented in Bound IR v1."""

    INPUT_BIND = "input_bind"
    SPEC_BIND = "spec_bind"
    LINEAR_BACKWARD = "linear_backward"
    CONV2D_BACKWARD = "conv2d_backward"
    RELU_RELAXATION = "relu_relaxation"
    COEFFICIENT_COMPOSE = "coefficient_compose"
    BIAS_ACCUMULATE = "bias_accumulate"
    ADD = "add"
    RESHAPE = "reshape"
    MATERIALIZE = "materialize"
    REPRESENTATION_CAST = "representation_cast"
    CONCRETIZE = "concretize"
    OBJECTIVE_REDUCE = "objective_reduce"


@dataclass(frozen=True)
class NoBoundOpAttrs:
    """Explicit empty attribute set for operations without parameters."""


@dataclass(frozen=True)
class InputBindAttrs:
    """Bind a perturbed graph value to a primal input."""

    primal_value_id: str
    perturbation_id: str

    def validate(self) -> None:
        """Validate input and perturbation references."""

        if not self.primal_value_id or not self.perturbation_id:
            raise ValueError("input binding IDs must be non-empty")


@dataclass(frozen=True)
class SpecBindAttrs:
    """Bind an objective graph value to a primal output."""

    primal_value_id: str
    objective_id: str

    def validate(self) -> None:
        """Validate primal-output and objective references."""

        if not self.primal_value_id or not self.objective_id:
            raise ValueError("spec binding IDs must be non-empty")


@dataclass(frozen=True)
class LinearBackwardAttrs:
    """Static primal references for linear backward propagation."""

    primal_node_id: str
    weight_primal_value_id: str
    bias_primal_value_id: Optional[str] = None

    def validate(self) -> None:
        """Validate linear parameter references."""

        if not self.primal_node_id or not self.weight_primal_value_id:
            raise ValueError("linear backward primal/weight IDs must be non-empty")
        if self.bias_primal_value_id is not None and not self.bias_primal_value_id:
            raise ValueError("linear backward bias ID must be non-empty when present")


@dataclass(frozen=True)
class Conv2dBackwardAttrs:
    """Static primal references and geometry for Conv2d backward propagation."""

    primal_node_id: str
    weight_primal_value_id: str
    bias_primal_value_id: Optional[str] = None
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (0, 0)
    dilation: Tuple[int, int] = (1, 1)
    groups: int = 1

    def validate(self) -> None:
        """Validate Conv2d parameter references and geometry."""

        if not self.primal_node_id or not self.weight_primal_value_id:
            raise ValueError("conv backward primal/weight IDs must be non-empty")
        if self.bias_primal_value_id is not None and not self.bias_primal_value_id:
            raise ValueError("conv backward bias ID must be non-empty when present")
        for name, pair, allow_zero in (
            ("stride", self.stride, False),
            ("padding", self.padding, True),
            ("dilation", self.dilation, False),
        ):
            if len(pair) != 2:
                raise ValueError(f"conv {name} must contain two dimensions")
            minimum = 0 if allow_zero else 1
            if any(value < minimum for value in pair):
                raise ValueError(f"conv {name} contains an invalid value: {pair}")
        if self.groups <= 0:
            raise ValueError("conv groups must be positive")


@dataclass(frozen=True)
class ReluRelaxationAttrs:
    """Reference the primal ReLU whose relaxation is applied."""

    primal_node_id: str

    def validate(self) -> None:
        """Validate the referenced primal ReLU identity."""

        if not self.primal_node_id:
            raise ValueError("ReLU relaxation primal_node_id must be non-empty")


@dataclass(frozen=True)
class ReshapeAttrs:
    """Target shape for a semantic-preserving view change."""

    target_shape: Tuple[Optional[int], ...]

    def validate(self) -> None:
        """Validate target dimensions."""

        if any(
            dimension is not None and dimension <= 0 for dimension in self.target_shape
        ):
            raise ValueError("reshape target dimensions must be positive or dynamic")


@dataclass(frozen=True)
class RepresentationChangeAttrs:
    """Describe an explicit representation transition and its reason."""

    source: BoundRepresentation
    target: BoundRepresentation
    reason: str

    def validate(self) -> None:
        """Validate a nontrivial, auditable representation transition."""

        if self.source == self.target:
            raise ValueError(
                "representation change requires distinct source and target"
            )
        if not self.reason:
            raise ValueError("representation change reason must be non-empty")


@dataclass(frozen=True)
class ConcretizeAttrs:
    """Reference the perturbation used to concretize coefficients."""

    perturbation_id: str

    def validate(self) -> None:
        """Validate the perturbation reference."""

        if not self.perturbation_id:
            raise ValueError("concretize perturbation_id must be non-empty")


class ObjectiveReduction(Enum):
    """Reduction applied after objective-bound computation."""

    NONE = "none"
    MIN = "min"
    MAX = "max"
    SUM = "sum"


@dataclass(frozen=True)
class ObjectiveReduceAttrs:
    """Describe an optional objective-axis reduction."""

    reduction: ObjectiveReduction
    dimension: Optional[int] = None

    def validate(self) -> None:
        """Validate the reduction/dimension combination."""

        if self.reduction == ObjectiveReduction.NONE and self.dimension is not None:
            raise ValueError("NONE objective reduction cannot specify a dimension")
        if self.dimension is not None and self.dimension < 0:
            raise ValueError("objective reduction dimension must be non-negative")


BoundOpAttrs: TypeAlias = (
    NoBoundOpAttrs
    | InputBindAttrs
    | SpecBindAttrs
    | LinearBackwardAttrs
    | Conv2dBackwardAttrs
    | ReluRelaxationAttrs
    | ReshapeAttrs
    | RepresentationChangeAttrs
    | ConcretizeAttrs
    | ObjectiveReduceAttrs
)


_EXPECTED_ATTRS: dict[BoundOpKind, type[object]] = {
    BoundOpKind.INPUT_BIND: InputBindAttrs,
    BoundOpKind.SPEC_BIND: SpecBindAttrs,
    BoundOpKind.LINEAR_BACKWARD: LinearBackwardAttrs,
    BoundOpKind.CONV2D_BACKWARD: Conv2dBackwardAttrs,
    BoundOpKind.RELU_RELAXATION: ReluRelaxationAttrs,
    BoundOpKind.COEFFICIENT_COMPOSE: NoBoundOpAttrs,
    BoundOpKind.BIAS_ACCUMULATE: NoBoundOpAttrs,
    BoundOpKind.ADD: NoBoundOpAttrs,
    BoundOpKind.RESHAPE: ReshapeAttrs,
    BoundOpKind.MATERIALIZE: RepresentationChangeAttrs,
    BoundOpKind.REPRESENTATION_CAST: RepresentationChangeAttrs,
    BoundOpKind.CONCRETIZE: ConcretizeAttrs,
    BoundOpKind.OBJECTIVE_REDUCE: ObjectiveReduceAttrs,
}


@dataclass(frozen=True)
class BoundOp:
    """One typed operation in topological Bound IR order."""

    op_id: str
    kind: BoundOpKind
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    attrs: BoundOpAttrs

    def validate(self, *, values: dict[str, BoundValue]) -> None:
        """Validate operation arity, attributes, references, and value contracts."""

        if not self.op_id:
            raise ValueError("bound op_id must be non-empty")
        expected_attrs = _EXPECTED_ATTRS[self.kind]
        if not isinstance(self.attrs, expected_attrs):
            raise ValueError(
                f"{self.kind.value} expects {expected_attrs.__name__}, "
                f"got {type(self.attrs).__name__}"
            )
        validator = getattr(self.attrs, "validate", None)
        if validator is not None:
            validator()
        if not self.inputs or not self.outputs:
            raise ValueError(f"bound op '{self.op_id}' requires inputs and outputs")
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError(f"bound op '{self.op_id}' contains duplicate outputs")
        for value_id in self.inputs + self.outputs:
            if value_id not in values:
                raise ValueError(
                    f"bound op '{self.op_id}' references unknown value '{value_id}'"
                )
        if len(self.outputs) != 1:
            raise ValueError(
                f"Bound IR v1 op '{self.op_id}' requires exactly one output"
            )
        if (
            self.kind
            not in {
                BoundOpKind.COEFFICIENT_COMPOSE,
                BoundOpKind.BIAS_ACCUMULATE,
                BoundOpKind.ADD,
            }
            and len(self.inputs) != 1
        ):
            raise ValueError(f"bound op '{self.op_id}' requires exactly one input")
        if (
            self.kind
            in {
                BoundOpKind.COEFFICIENT_COMPOSE,
                BoundOpKind.BIAS_ACCUMULATE,
                BoundOpKind.ADD,
            }
            and len(self.inputs) < 2
        ):
            raise ValueError(f"bound op '{self.op_id}' requires at least two inputs")
        self._validate_value_contract(values)

    def _validate_value_contract(  # pylint: disable=too-many-branches,too-many-statements
        self, values: dict[str, BoundValue]
    ) -> None:
        """Validate semantic relationships between input and output values."""

        input_values = tuple(values[value_id] for value_id in self.inputs)
        output = values[self.outputs[0]]
        representation_only = {
            BoundOpKind.MATERIALIZE,
            BoundOpKind.REPRESENTATION_CAST,
        }
        if self.kind in representation_only:
            source = input_values[0]
            attrs = self.attrs
            if not isinstance(attrs, RepresentationChangeAttrs):
                raise AssertionError("representation attributes checked above")
            if source.semantic_signature() != output.semantic_signature():
                raise ValueError(
                    f"bound op '{self.op_id}' representation change alters semantics"
                )
            if source.representation != attrs.source:
                raise ValueError(
                    f"bound op '{self.op_id}' source representation mismatch"
                )
            if output.representation != attrs.target:
                raise ValueError(
                    f"bound op '{self.op_id}' target representation mismatch"
                )
            if (
                self.kind == BoundOpKind.MATERIALIZE
                and attrs.target != BoundRepresentation.DENSE
            ):
                raise ValueError("materialize must produce a dense representation")
        if self.kind in {
            BoundOpKind.RELU_RELAXATION,
            BoundOpKind.COEFFICIENT_COMPOSE,
            BoundOpKind.BIAS_ACCUMULATE,
            BoundOpKind.ADD,
            BoundOpKind.RESHAPE,
        }:
            if any(value.polarity != output.polarity for value in input_values):
                raise ValueError(
                    f"bound op '{self.op_id}' changes lower/upper polarity"
                )
        if self.kind in {
            BoundOpKind.RELU_RELAXATION,
            BoundOpKind.COEFFICIENT_COMPOSE,
            BoundOpKind.BIAS_ACCUMULATE,
            BoundOpKind.ADD,
        }:
            if any(value.tensor_type != output.tensor_type for value in input_values):
                raise ValueError(
                    f"bound op '{self.op_id}' requires matching tensor types"
                )
        if self.kind in {
            BoundOpKind.COEFFICIENT_COMPOSE,
            BoundOpKind.ADD,
        } and any(
            value.representation != output.representation for value in input_values
        ):
            raise ValueError(
                f"bound op '{self.op_id}' requires matching representations"
            )
        if self.kind == BoundOpKind.INPUT_BIND:
            if input_values[0].role != BoundValueRole.PERTURBATION:
                raise ValueError("input binding input must be a perturbation")
            if output.role != BoundValueRole.INTERVAL:
                raise ValueError("input binding output must be an interval")
        if self.kind == BoundOpKind.SPEC_BIND:
            if input_values[0].role != BoundValueRole.OBJECTIVE:
                raise ValueError("spec binding input must be an objective")
            if output.role != BoundValueRole.COEFFICIENT:
                raise ValueError("spec binding output must be a coefficient")
        if self.kind in {
            BoundOpKind.LINEAR_BACKWARD,
            BoundOpKind.CONV2D_BACKWARD,
            BoundOpKind.RELU_RELAXATION,
        }:
            if input_values[0].role != BoundValueRole.COEFFICIENT:
                raise ValueError(f"{self.kind.value} input must be a coefficient")
            if output.role != BoundValueRole.COEFFICIENT:
                raise ValueError(f"{self.kind.value} output must be a coefficient")
        if self.kind == BoundOpKind.COEFFICIENT_COMPOSE and any(
            value.role != BoundValueRole.COEFFICIENT
            for value in input_values + (output,)
        ):
            raise ValueError("coefficient compose accepts only coefficient values")
        if self.kind == BoundOpKind.ADD and any(
            value.role != output.role for value in input_values
        ):
            raise ValueError("add requires matching value roles")
        if self.kind == BoundOpKind.RESHAPE:
            self._validate_reshape(input_values[0], output)
        if self.kind == BoundOpKind.CONCRETIZE:
            if input_values[0].role != BoundValueRole.COEFFICIENT:
                raise ValueError("concretize input must be a coefficient")
            if output.role not in {
                BoundValueRole.INTERVAL,
                BoundValueRole.OBJECTIVE,
            }:
                raise ValueError("concretize output must be interval/objective")
            if input_values[0].polarity != output.polarity:
                raise ValueError("concretize cannot change polarity")
        if self.kind == BoundOpKind.OBJECTIVE_REDUCE:
            if input_values[0].role != BoundValueRole.OBJECTIVE:
                raise ValueError("objective reduction input must be an objective")
            if output.role != BoundValueRole.OBJECTIVE:
                raise ValueError("objective reduction output must be an objective")

    def _validate_reshape(self, source: BoundValue, target: BoundValue) -> None:
        """Validate reshape semantics when static dimensions are available."""

        attrs = self.attrs
        if not isinstance(attrs, ReshapeAttrs):
            raise AssertionError("reshape attributes checked above")
        if target.tensor_type.shape != attrs.target_shape:
            raise ValueError("reshape output shape does not match target_shape")
        if source.tensor_type.dtype != target.tensor_type.dtype:
            raise ValueError("reshape cannot change dtype")
        if source.role != target.role or source.representation != target.representation:
            raise ValueError("reshape cannot change value role/representation")
        source_numel = _static_numel(source.tensor_type.shape)
        target_numel = _static_numel(target.tensor_type.shape)
        if (
            source_numel is not None
            and target_numel is not None
            and source_numel != target_numel
        ):
            raise ValueError("reshape changes the static element count")

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible operation fields."""

        payload = _strict_jsonable(asdict(self.attrs))
        if not isinstance(payload, dict):
            raise AssertionError("Bound IR attributes must serialize as an object")
        return {
            "op_id": self.op_id,
            "kind": self.kind.value,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs_type": type(self.attrs).__name__,
            "attrs": payload,
        }


@dataclass(frozen=True)
class BFBoundGraph:
    """Validated SSA graph of semantic bound operations."""

    values: Tuple[BoundValue, ...]
    ops: Tuple[BoundOp, ...]
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]

    def validate(self) -> None:  # pylint: disable=too-many-branches
        """Validate SSA identity, topological use-def, and graph IO."""

        if not self.values:
            raise ValueError("Bound IR graph requires values")
        if not self.ops:
            raise ValueError("Bound IR graph requires operations")
        if not self.inputs or not self.outputs:
            raise ValueError("Bound IR graph requires inputs and outputs")
        value_ids = [value.value_id for value in self.values]
        op_ids = [op.op_id for op in self.ops]
        if len(value_ids) != len(set(value_ids)):
            raise ValueError("Bound IR graph contains duplicate value IDs")
        if len(op_ids) != len(set(op_ids)):
            raise ValueError("Bound IR graph contains duplicate op IDs")
        if len(self.inputs) != len(set(self.inputs)):
            raise ValueError("Bound IR graph contains duplicate inputs")
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError("Bound IR graph contains duplicate outputs")
        values = {value.value_id: value for value in self.values}
        for value in self.values:
            value.validate()
        for value_id in self.inputs + self.outputs:
            if value_id not in values:
                raise ValueError(f"Bound IR graph references unknown IO '{value_id}'")

        available = set(self.inputs)
        for op in self.ops:
            op.validate(values=values)
            for value_id in op.inputs:
                if value_id not in available:
                    raise ValueError(
                        f"bound op '{op.op_id}' uses '{value_id}' before definition"
                    )
            for value_id in op.outputs:
                if value_id in available:
                    raise ValueError(
                        f"bound op '{op.op_id}' redefines value '{value_id}'"
                    )
                available.add(value_id)
        undeclared_producers = set(values) - available
        if undeclared_producers:
            raise ValueError(
                "Bound IR values lack graph input or producer: "
                f"{sorted(undeclared_producers)}"
            )
        for output in self.outputs:
            if output not in available:
                raise ValueError(f"Bound IR output '{output}' is unavailable")

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible graph fields."""

        self.validate()
        return {
            "values": [value.to_dict() for value in self.values],
            "ops": [op.to_dict() for op in self.ops],
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
        }


@dataclass(frozen=True)
class BFBoundModule:
    """Top-level, hashable Bound IR unit."""

    module_id: str
    primal_graph_hash: str
    spec: VerificationSpec
    domain: BoundDomainConfig
    graph: BFBoundGraph
    schema_version: str = BOUND_IR_SCHEMA_VERSION

    def validate(self) -> None:
        """Validate schema identity and all nested IR contracts."""

        if self.schema_version != BOUND_IR_SCHEMA_VERSION:
            raise ValueError(f"unsupported Bound IR schema: {self.schema_version}")
        if not self.module_id or not self.primal_graph_hash:
            raise ValueError("Bound IR module/primal graph IDs must be non-empty")
        self.spec.validate()
        self.domain.validate()
        self.graph.validate()
        self._validate_spec_references()

    def _validate_spec_references(self) -> None:
        """Resolve binding/concretization IDs against the typed module spec."""

        perturbations = {
            perturbation.perturbation_id: perturbation
            for perturbation in self.spec.perturbations
        }
        objectives = {
            objective.objective_id: objective for objective in self.spec.objectives
        }
        for op in self.graph.ops:
            attrs = op.attrs
            if isinstance(attrs, InputBindAttrs):
                perturbation = perturbations.get(attrs.perturbation_id)
                if perturbation is None:
                    raise ValueError(
                        f"input bind references unknown perturbation "
                        f"'{attrs.perturbation_id}'"
                    )
                if perturbation.input_primal_value_id != attrs.primal_value_id:
                    raise ValueError("input bind primal value/spec mismatch")
            elif isinstance(attrs, SpecBindAttrs):
                objective = objectives.get(attrs.objective_id)
                if objective is None:
                    raise ValueError(
                        f"spec bind references unknown objective '{attrs.objective_id}'"
                    )
                if objective.output_primal_value_id != attrs.primal_value_id:
                    raise ValueError("spec bind primal value/spec mismatch")
            elif isinstance(attrs, ConcretizeAttrs):
                if attrs.perturbation_id not in perturbations:
                    raise ValueError(
                        f"concretize references unknown perturbation "
                        f"'{attrs.perturbation_id}'"
                    )

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible module fields."""

        self.validate()
        return {
            "schema_version": self.schema_version,
            "module_id": self.module_id,
            "primal_graph_hash": self.primal_graph_hash,
            "spec": self.spec.to_dict(),
            "domain": self.domain.to_dict(),
            "graph": self.graph.to_dict(),
        }

    def canonical_json(self) -> str:
        """Serialize deterministically for artifacts and cache keys."""

        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        """Return a content hash over the validated canonical module."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _static_numel(shape: Tuple[Optional[int], ...]) -> Optional[int]:
    """Return a static element count, or None for dynamic shapes."""

    if any(dimension is None for dimension in shape):
        return None
    result = 1
    for dimension in shape:
        if dimension is None:
            return None
        result *= dimension
    return result


def _strict_jsonable(value: object) -> object:
    """Convert only schema-approved immutable values into JSON data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_strict_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_strict_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _strict_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise TypeError(f"Bound IR value is not JSON serializable: {type(value).__name__}")


# Transitional aliases for the unused Phase-4 names. They point at the new
# first-class schemas instead of preserving the old Any/dict containers.
Spec = VerificationSpec
ApplyTransformer = BoundOp
BFBoundProgram = BFBoundModule
