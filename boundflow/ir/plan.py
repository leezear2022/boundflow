"""First-class Plan IR v1 schemas and cross-decision verification."""

# Declarative schemas intentionally expose all semantic fields; validators
# centralize cross-object checks and therefore have high structural complexity.
# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-branches,too-many-locals,too-many-statements,missing-function-docstring

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Optional, Tuple

from .bound import (
    BFBoundModule,
    BoundMethodKind,
    BoundOpKind,
    BoundRepresentation,
)

PLAN_IR_SCHEMA_VERSION = "boundflow.plan_ir/v1.0"


class RegionKind(Enum):
    """Semantic class used for capability and lowering decisions."""

    BINDING = "binding"
    AFFINE = "affine"
    RELAXATION = "relaxation"
    ROUTING = "routing"
    CONCRETIZATION = "concretization"
    MIXED = "mixed"
    EXTERNAL_VERIFIER = "external_verifier"


class BackendKind(Enum):
    """Backend families represented by Plan IR v1."""

    PYTORCH_DENSE = "pytorch_dense"
    PYTORCH_STRUCTURED = "pytorch_structured"
    PYTORCH_CHUNKED = "pytorch_chunked"
    TORCH_COMPILE = "torch_compile"
    TVM_RELAX_UNFUSED = "tvm_relax_unfused"
    TVM_TIR_UNFUSED = "tvm_tir_unfused"
    TVM_FUSED_TIR = "tvm_fused_tir"
    REFERENCE = "reference"
    EXTERNAL_ABCROWN = "external_abcrown"


class TransitionKind(Enum):
    """Explicit representation boundary chosen by a plan."""

    CAST = "cast"
    MATERIALIZE = "materialize"


class StateAction(Enum):
    """Planner action for reusable runtime state."""

    REUSE = "reuse"
    CACHE = "cache"
    RECOMPUTE = "recompute"
    EVICT = "evict"


@dataclass(frozen=True)
class PlanProvenance:
    """One deterministic non-semantic provenance entry."""

    key: str
    value: str

    def validate(self) -> None:
        if not self.key or not self.value:
            raise ValueError("plan provenance key/value must be non-empty")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {"key": self.key, "value": self.value}


@dataclass(frozen=True)
class PlanCost:
    """Comparable prediction attached to every Plan IR candidate."""

    predicted_latency_ms: float
    predicted_peak_bytes: int
    compile_cost_ms: float
    setup_cost_ms: float
    confidence: float
    risk_tags: Tuple[str, ...] = ()

    def validate(self) -> None:
        for name in (
            "predicted_latency_ms",
            "compile_cost_ms",
            "setup_cost_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.predicted_peak_bytes < 0:
            raise ValueError("predicted_peak_bytes must be non-negative")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("plan cost confidence must be in [0, 1]")
        if any(not tag for tag in self.risk_tags):
            raise ValueError("plan risk tags must be non-empty")
        if len(self.risk_tags) != len(set(self.risk_tags)):
            raise ValueError("plan cost contains duplicate risk tags")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "predicted_latency_ms": self.predicted_latency_ms,
            "predicted_peak_bytes": self.predicted_peak_bytes,
            "compile_cost_ms": self.compile_cost_ms,
            "setup_cost_ms": self.setup_cost_ms,
            "confidence": self.confidence,
            "risk_tags": list(self.risk_tags),
        }


@dataclass(frozen=True)
class HardwareProfile:
    """Static hardware facts available to template construction."""

    profile_id: str
    device: str
    total_memory_bytes: int
    supported_dtypes: Tuple[str, ...]
    backend_capability_ids: Tuple[str, ...]
    alignment_bytes: int = 1

    def validate(self) -> None:
        if not self.profile_id or not self.device:
            raise ValueError("hardware profile identity/device must be non-empty")
        if self.total_memory_bytes <= 0 or self.alignment_bytes <= 0:
            raise ValueError("hardware memory/alignment must be positive")
        if not self.supported_dtypes or any(
            not dtype for dtype in self.supported_dtypes
        ):
            raise ValueError("hardware profile requires supported dtypes")
        if not self.backend_capability_ids or any(
            not capability_id for capability_id in self.backend_capability_ids
        ):
            raise ValueError("hardware profile requires backend capabilities")
        _require_unique(self.supported_dtypes, label="hardware dtypes")
        _require_unique(
            self.backend_capability_ids, label="hardware backend capabilities"
        )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "profile_id": self.profile_id,
            "device": self.device,
            "total_memory_bytes": self.total_memory_bytes,
            "supported_dtypes": list(self.supported_dtypes),
            "backend_capability_ids": list(self.backend_capability_ids),
            "alignment_bytes": self.alignment_bytes,
        }


@dataclass(frozen=True)
class WorkloadProfile:
    """Static and query-bucket properties used by Plan IR legality checks."""

    profile_id: str
    method: BoundMethodKind
    requires_grad: bool
    alpha_enabled: bool
    beta_enabled: bool
    split_state_present: bool
    static_shapes: bool
    domain_batch_size: int
    spec_batch_size: int
    sample_batch_size: int
    dtype: str
    device: str
    numeric_policy: str

    def validate(self) -> None:
        for name in ("profile_id", "dtype", "device", "numeric_policy"):
            if not getattr(self, name):
                raise ValueError(f"workload {name} must be non-empty")
        for name in (
            "domain_batch_size",
            "spec_batch_size",
            "sample_batch_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"workload {name} must be positive")
        if self.beta_enabled and not self.alpha_enabled:
            raise ValueError("workload beta state requires alpha state")
        if self.method in {BoundMethodKind.INTERVAL, BoundMethodKind.CROWN} and (
            self.alpha_enabled or self.beta_enabled or self.split_state_present
        ):
            raise ValueError(f"{self.method.value} workload has invalid state")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["method"] = self.method.value
        return payload


@dataclass(frozen=True)
class BackendCapabilitySpec:
    """Backend legality facts independent from legacy planner classes."""

    capability_id: str
    backend: BackendKind
    supported_methods: Tuple[BoundMethodKind, ...]
    supported_op_kinds: Tuple[BoundOpKind, ...]
    supported_representations: Tuple[BoundRepresentation, ...]
    supported_dtypes: Tuple[str, ...]
    supported_devices: Tuple[str, ...]
    supports_grad: bool
    supports_alpha: bool
    supports_beta: bool
    supports_split_state: bool
    static_shape_only: bool

    def validate(self) -> None:
        if not self.capability_id:
            raise ValueError("backend capability_id must be non-empty")
        for name in (
            "supported_methods",
            "supported_op_kinds",
            "supported_representations",
            "supported_dtypes",
            "supported_devices",
        ):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"backend {name} must be non-empty")
            _require_unique(values, label=f"backend {name}")

    def rejection_reasons(
        self,
        *,
        workload: WorkloadProfile,
        op_kinds: Tuple[BoundOpKind, ...],
        representation: BoundRepresentation,
    ) -> Tuple[str, ...]:
        """Return deterministic cross-layer capability rejection reasons."""

        self.validate()
        workload.validate()
        reasons: list[str] = []
        if workload.method not in self.supported_methods:
            reasons.append("bound_method_unsupported")
        if workload.requires_grad and not self.supports_grad:
            reasons.append("requires_grad_unsupported")
        if workload.alpha_enabled and not self.supports_alpha:
            reasons.append("alpha_unsupported")
        if workload.beta_enabled and not self.supports_beta:
            reasons.append("beta_unsupported")
        if workload.split_state_present and not self.supports_split_state:
            reasons.append("split_state_unsupported")
        if any(op_kind not in self.supported_op_kinds for op_kind in op_kinds):
            reasons.append("op_kind_unsupported")
        if representation not in self.supported_representations:
            reasons.append("representation_unsupported")
        if workload.dtype not in self.supported_dtypes:
            reasons.append("dtype_unsupported")
        if workload.device not in self.supported_devices:
            reasons.append("device_unsupported")
        if self.static_shape_only and not workload.static_shapes:
            reasons.append("dynamic_shape_unsupported")
        return tuple(reasons)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "capability_id": self.capability_id,
            "backend": self.backend.value,
            "supported_methods": [method.value for method in self.supported_methods],
            "supported_op_kinds": [
                op_kind.value for op_kind in self.supported_op_kinds
            ],
            "supported_representations": [
                representation.value
                for representation in self.supported_representations
            ],
            "supported_dtypes": list(self.supported_dtypes),
            "supported_devices": list(self.supported_devices),
            "supports_grad": self.supports_grad,
            "supports_alpha": self.supports_alpha,
            "supports_beta": self.supports_beta,
            "supports_split_state": self.supports_split_state,
            "static_shape_only": self.static_shape_only,
        }


@dataclass(frozen=True)
class RegionCandidate:
    """One candidate partition/fusion region over Bound IR operations."""

    candidate_id: str
    region_id: str
    kind: RegionKind
    op_ids: Tuple[str, ...]
    input_value_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]
    fused: bool
    cost: PlanCost

    def validate(self) -> None:
        if not self.candidate_id or not self.region_id:
            raise ValueError("region candidate/region IDs must be non-empty")
        if not self.op_ids or not self.input_value_ids or not self.output_value_ids:
            raise ValueError("region candidate requires ops and boundary values")
        for name in ("op_ids", "input_value_ids", "output_value_ids"):
            values = getattr(self, name)
            if any(not value for value in values):
                raise ValueError(f"region {name} contains an empty ID")
            _require_unique(values, label=f"region {name}")
        if self.fused and len(self.op_ids) < 2:
            raise ValueError("fused region must contain at least two ops")
        self.cost.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "candidate_id": self.candidate_id,
            "region_id": self.region_id,
            "kind": self.kind.value,
            "op_ids": list(self.op_ids),
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
            "fused": self.fused,
            "cost": self.cost.to_dict(),
        }


@dataclass(frozen=True)
class RepresentationCandidate:
    """One legal representation and its required explicit transitions."""

    candidate_id: str
    region_id: str
    representation: BoundRepresentation
    required_transition_candidate_ids: Tuple[str, ...]
    static_legal: bool
    rejection_reasons: Tuple[str, ...]
    cost: PlanCost

    def validate(self) -> None:
        _validate_candidate_identity(self.candidate_id, self.region_id)
        _validate_static_legality(
            self.static_legal, self.rejection_reasons, label=self.candidate_id
        )
        _require_unique(
            self.required_transition_candidate_ids,
            label="required transition candidates",
        )
        self.cost.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "candidate_id": self.candidate_id,
            "region_id": self.region_id,
            "representation": self.representation.value,
            "required_transition_candidate_ids": list(
                self.required_transition_candidate_ids
            ),
            "static_legal": self.static_legal,
            "rejection_reasons": list(self.rejection_reasons),
            "cost": self.cost.to_dict(),
        }


@dataclass(frozen=True)
class MaterializationCandidate:
    """One explicit representation transition at a Bound IR boundary."""

    candidate_id: str
    source_value_id: str
    before_op_id: str
    kind: TransitionKind
    source_representation: BoundRepresentation
    target_representation: BoundRepresentation
    static_legal: bool
    rejection_reasons: Tuple[str, ...]
    cost: PlanCost

    def validate(self) -> None:
        if not self.candidate_id or not self.source_value_id or not self.before_op_id:
            raise ValueError("materialization candidate IDs must be non-empty")
        if self.source_representation == self.target_representation:
            raise ValueError("materialization candidate must change representation")
        if (
            self.kind == TransitionKind.MATERIALIZE
            and self.target_representation != BoundRepresentation.DENSE
        ):
            raise ValueError("materialize candidate must target dense")
        _validate_static_legality(
            self.static_legal, self.rejection_reasons, label=self.candidate_id
        )
        self.cost.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "candidate_id": self.candidate_id,
            "source_value_id": self.source_value_id,
            "before_op_id": self.before_op_id,
            "kind": self.kind.value,
            "source_representation": self.source_representation.value,
            "target_representation": self.target_representation.value,
            "static_legal": self.static_legal,
            "rejection_reasons": list(self.rejection_reasons),
            "cost": self.cost.to_dict(),
        }


@dataclass(frozen=True)
class BackendCandidate:
    """One backend/capability choice for a logical region."""

    candidate_id: str
    region_id: str
    backend: BackendKind
    capability_id: str
    compatible_representation_candidate_ids: Tuple[str, ...]
    compiled_artifact_key: Optional[str]
    static_legal: bool
    rejection_reasons: Tuple[str, ...]
    cost: PlanCost

    def validate(self) -> None:
        _validate_candidate_identity(self.candidate_id, self.region_id)
        if not self.capability_id:
            raise ValueError("backend candidate capability_id must be non-empty")
        if self.compiled_artifact_key is not None and not self.compiled_artifact_key:
            raise ValueError("compiled artifact key must be non-empty when present")
        if not self.compatible_representation_candidate_ids:
            raise ValueError("backend candidate requires compatible representations")
        _require_unique(
            self.compatible_representation_candidate_ids,
            label="backend compatible representations",
        )
        _validate_static_legality(
            self.static_legal, self.rejection_reasons, label=self.candidate_id
        )
        self.cost.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "candidate_id": self.candidate_id,
            "region_id": self.region_id,
            "backend": self.backend.value,
            "capability_id": self.capability_id,
            "compatible_representation_candidate_ids": list(
                self.compatible_representation_candidate_ids
            ),
            "compiled_artifact_key": self.compiled_artifact_key,
            "static_legal": self.static_legal,
            "rejection_reasons": list(self.rejection_reasons),
            "cost": self.cost.to_dict(),
        }


@dataclass(frozen=True)
class BatchCandidate:
    """Independent domain/spec/sample batching choice."""

    candidate_id: str
    domain_batch_size: int
    spec_batch_size: int
    sample_batch_size: int
    estimated_payload_bytes: int
    static_legal: bool
    rejection_reasons: Tuple[str, ...]
    cost: PlanCost

    def validate(self) -> None:
        if not self.candidate_id:
            raise ValueError("batch candidate_id must be non-empty")
        for name in (
            "domain_batch_size",
            "spec_batch_size",
            "sample_batch_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"batch {name} must be positive")
        if self.estimated_payload_bytes < 0:
            raise ValueError("batch payload bytes must be non-negative")
        _validate_static_legality(
            self.static_legal, self.rejection_reasons, label=self.candidate_id
        )
        self.cost.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            **asdict(self),
            "rejection_reasons": list(self.rejection_reasons),
            "cost": self.cost.to_dict(),
        }


@dataclass(frozen=True)
class StorageBinding:
    """One logical Bound IR value assigned to a physical arena/allocation."""

    value_id: str
    arena_id: str
    offset_bytes: int
    logical_size_bytes: int
    size_bytes: int
    representation: BoundRepresentation
    live_from_op_id: str
    live_to_op_id: str

    def validate(self) -> None:
        for name in ("value_id", "arena_id", "live_from_op_id", "live_to_op_id"):
            if not getattr(self, name):
                raise ValueError(f"storage binding {name} must be non-empty")
        if (
            self.offset_bytes < 0
            or self.logical_size_bytes <= 0
            or self.size_bytes <= 0
        ):
            raise ValueError("storage binding offset/size are invalid")
        if (
            self.representation == BoundRepresentation.DENSE
            and self.size_bytes < self.logical_size_bytes
        ):
            raise ValueError("dense storage is smaller than its logical value")

    @property
    def end_bytes(self) -> int:
        return self.offset_bytes + self.size_bytes

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["representation"] = self.representation.value
        return payload


@dataclass(frozen=True)
class StorageCandidate:
    """Whole-plan storage/lifetime candidate with exact compatibility links."""

    candidate_id: str
    bindings: Tuple[StorageBinding, ...]
    compatible_batch_candidate_ids: Tuple[str, ...]
    compatible_representation_candidate_ids: Tuple[str, ...]
    static_legal: bool
    rejection_reasons: Tuple[str, ...]
    cost: PlanCost

    def validate(self) -> None:
        if not self.candidate_id or not self.bindings:
            raise ValueError("storage candidate requires ID and bindings")
        for binding in self.bindings:
            binding.validate()
        _require_unique(
            tuple(binding.value_id for binding in self.bindings),
            label="storage values",
        )
        if not self.compatible_batch_candidate_ids:
            raise ValueError("storage candidate requires compatible batches")
        _require_unique(
            self.compatible_batch_candidate_ids,
            label="storage compatible batches",
        )
        _require_unique(
            self.compatible_representation_candidate_ids,
            label="storage compatible representations",
        )
        _validate_static_legality(
            self.static_legal, self.rejection_reasons, label=self.candidate_id
        )
        self.cost.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "candidate_id": self.candidate_id,
            "bindings": [binding.to_dict() for binding in self.bindings],
            "compatible_batch_candidate_ids": list(self.compatible_batch_candidate_ids),
            "compatible_representation_candidate_ids": list(
                self.compatible_representation_candidate_ids
            ),
            "static_legal": self.static_legal,
            "rejection_reasons": list(self.rejection_reasons),
            "cost": self.cost.to_dict(),
        }


@dataclass(frozen=True)
class StateCandidate:
    """Cache/recompute/evict choice tied to a versioned Bound IR value."""

    candidate_id: str
    state_id: str
    source_value_id: str
    action: StateAction
    state_version: str
    size_bytes: int
    static_legal: bool
    rejection_reasons: Tuple[str, ...]
    cost: PlanCost

    def validate(self) -> None:
        for name in (
            "candidate_id",
            "state_id",
            "source_value_id",
            "state_version",
        ):
            if not getattr(self, name):
                raise ValueError(f"state candidate {name} must be non-empty")
        if self.size_bytes < 0:
            raise ValueError("state candidate size_bytes must be non-negative")
        _validate_static_legality(
            self.static_legal, self.rejection_reasons, label=self.candidate_id
        )
        self.cost.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["action"] = self.action.value
        payload["rejection_reasons"] = list(self.rejection_reasons)
        payload["cost"] = self.cost.to_dict()
        return payload


PlanCandidate = (
    RegionCandidate
    | RepresentationCandidate
    | MaterializationCandidate
    | BackendCandidate
    | BatchCandidate
    | StorageCandidate
    | StateCandidate
)


@dataclass(frozen=True)
class PlanTemplate:
    """Static candidate space determined before a dynamic query instance."""

    template_id: str
    bound_module_hash: str
    planner_config_hash: str
    hardware: HardwareProfile
    workload: WorkloadProfile
    capabilities: Tuple[BackendCapabilitySpec, ...]
    region_candidates: Tuple[RegionCandidate, ...]
    representation_candidates: Tuple[RepresentationCandidate, ...]
    materialization_candidates: Tuple[MaterializationCandidate, ...]
    backend_candidates: Tuple[BackendCandidate, ...]
    batch_candidates: Tuple[BatchCandidate, ...]
    storage_candidates: Tuple[StorageCandidate, ...]
    state_candidates: Tuple[StateCandidate, ...] = ()
    provenance: Tuple[PlanProvenance, ...] = ()
    schema_version: str = PLAN_IR_SCHEMA_VERSION

    def validate(self, *, bound_module: BFBoundModule) -> None:
        """Validate static candidates and every Bound IR/capability reference."""

        if self.schema_version != PLAN_IR_SCHEMA_VERSION:
            raise ValueError(f"unsupported Plan IR schema: {self.schema_version}")
        for name in ("template_id", "bound_module_hash", "planner_config_hash"):
            if not getattr(self, name):
                raise ValueError(f"plan template {name} must be non-empty")
        bound_module.validate()
        if self.bound_module_hash != bound_module.stable_hash():
            raise ValueError("plan template Bound IR hash mismatch")
        self.hardware.validate()
        self.workload.validate()
        if self.workload.method != bound_module.domain.method:
            raise ValueError("plan workload method/Bound IR domain mismatch")
        if self.workload.requires_grad != bound_module.domain.requires_grad:
            raise ValueError("plan workload grad/Bound IR domain mismatch")
        if self.workload.alpha_enabled != bound_module.domain.alpha_enabled:
            raise ValueError("plan workload alpha/Bound IR domain mismatch")
        if self.workload.beta_enabled != bound_module.domain.beta_enabled:
            raise ValueError("plan workload beta/Bound IR domain mismatch")
        if self.workload.split_state_present != bound_module.domain.split_state_present:
            raise ValueError("plan workload split/Bound IR domain mismatch")
        if self.workload.device != self.hardware.device:
            raise ValueError("workload/hardware device mismatch")
        if self.workload.dtype not in self.hardware.supported_dtypes:
            raise ValueError("workload dtype is unsupported by hardware")

        if not self.capabilities:
            raise ValueError("plan template requires backend capabilities")
        for capability in self.capabilities:
            capability.validate()
        capability_ids = tuple(
            capability.capability_id for capability in self.capabilities
        )
        _require_unique(capability_ids, label="capability IDs")
        if set(capability_ids) != set(self.hardware.backend_capability_ids):
            raise ValueError("hardware/template capability sets differ")

        required_groups = (
            self.region_candidates,
            self.representation_candidates,
            self.backend_candidates,
            self.batch_candidates,
            self.storage_candidates,
        )
        if any(not group for group in required_groups):
            raise ValueError("plan template has an empty required candidate group")
        for candidate in self.all_candidates():
            candidate.validate()
        candidate_ids = tuple(
            candidate.candidate_id for candidate in self.all_candidates()
        )
        _require_unique(candidate_ids, label="global Plan IR candidate IDs")
        for item in self.provenance:
            item.validate()
        _require_unique(
            tuple(item.key for item in self.provenance),
            label="plan provenance keys",
        )
        self._validate_bound_references(bound_module)
        self._validate_candidate_cross_references(bound_module)

    def all_candidates(self) -> Tuple[PlanCandidate, ...]:
        """Return candidates in canonical decision-category order."""

        return (
            *self.region_candidates,
            *self.representation_candidates,
            *self.materialization_candidates,
            *self.backend_candidates,
            *self.batch_candidates,
            *self.storage_candidates,
            *self.state_candidates,
        )

    def candidate_map(self) -> dict[str, PlanCandidate]:
        return {
            candidate.candidate_id: candidate for candidate in self.all_candidates()
        }

    def _validate_bound_references(self, bound_module: BFBoundModule) -> None:
        graph = bound_module.graph
        ops = {op.op_id: op for op in graph.ops}
        values = {value.value_id: value for value in graph.values}
        op_index = {op.op_id: index for index, op in enumerate(graph.ops)}
        producer = {value_id: op.op_id for op in graph.ops for value_id in op.outputs}
        users: dict[str, set[str]] = {}
        for op in graph.ops:
            for value_id in op.inputs:
                users.setdefault(value_id, set()).add(op.op_id)

        for region in self.region_candidates:
            if any(op_id not in ops for op_id in region.op_ids):
                raise ValueError(f"region '{region.region_id}' references unknown op")
            indices = [op_index[op_id] for op_id in region.op_ids]
            if indices != sorted(indices):
                raise ValueError("region op_ids must follow Bound IR topology")
            region_ops = set(region.op_ids)
            expected_inputs = {
                value_id
                for op_id in region.op_ids
                for value_id in ops[op_id].inputs
                if producer.get(value_id) not in region_ops
            }
            expected_outputs = {
                value_id
                for op_id in region.op_ids
                for value_id in ops[op_id].outputs
                if value_id in graph.outputs
                or any(user not in region_ops for user in users.get(value_id, set()))
            }
            if set(region.input_value_ids) != expected_inputs:
                raise ValueError(f"region '{region.region_id}' input boundary mismatch")
            if set(region.output_value_ids) != expected_outputs:
                raise ValueError(
                    f"region '{region.region_id}' output boundary mismatch"
                )

        for transition_candidate in self.materialization_candidates:
            if transition_candidate.source_value_id not in values:
                raise ValueError("materialization references unknown Bound IR value")
            if transition_candidate.before_op_id not in ops:
                raise ValueError("materialization references unknown Bound IR op")
            if (
                transition_candidate.source_value_id
                not in ops[transition_candidate.before_op_id].inputs
            ):
                raise ValueError(
                    "materialization source is not consumed by before_op_id"
                )
        for storage_candidate in self.storage_candidates:
            for binding in storage_candidate.bindings:
                if binding.value_id not in values:
                    raise ValueError("storage references unknown Bound IR value")
                if (
                    binding.live_from_op_id not in ops
                    or binding.live_to_op_id not in ops
                ):
                    raise ValueError("storage lifetime references unknown Bound IR op")
                if op_index[binding.live_from_op_id] > op_index[binding.live_to_op_id]:
                    raise ValueError("storage lifetime is reversed")
        for state_candidate in self.state_candidates:
            if state_candidate.source_value_id not in values:
                raise ValueError("state candidate references unknown Bound IR value")
            value_version = values[state_candidate.source_value_id].state_version
            if value_version != state_candidate.state_version:
                raise ValueError("state candidate version/Bound IR value mismatch")

    def _validate_candidate_cross_references(self, bound_module: BFBoundModule) -> None:
        regions = {region.region_id: region for region in self.region_candidates}
        transitions = {
            candidate.candidate_id: candidate
            for candidate in self.materialization_candidates
        }
        representations = {
            candidate.candidate_id: candidate
            for candidate in self.representation_candidates
        }
        batches = {
            candidate.candidate_id: candidate for candidate in self.batch_candidates
        }
        capabilities = {
            capability.capability_id: capability for capability in self.capabilities
        }
        ops = {op.op_id: op for op in bound_module.graph.ops}

        for representation_candidate in self.representation_candidates:
            if representation_candidate.region_id not in regions:
                raise ValueError("representation references unknown region")
            for (
                transition_id
            ) in representation_candidate.required_transition_candidate_ids:
                if transition_id not in transitions:
                    raise ValueError(
                        "representation references unknown transition candidate"
                    )
        for backend_candidate in self.backend_candidates:
            region = regions.get(backend_candidate.region_id)
            if region is None:
                raise ValueError("backend candidate references unknown region")
            capability = capabilities.get(backend_candidate.capability_id)
            if capability is None:
                raise ValueError("backend candidate references unknown capability")
            if capability.backend != backend_candidate.backend:
                raise ValueError("backend candidate/capability kind mismatch")
            for (
                representation_id
            ) in backend_candidate.compatible_representation_candidate_ids:
                representation = representations.get(representation_id)
                if representation is None:
                    raise ValueError(
                        "backend references unknown representation candidate"
                    )
                if representation.region_id != backend_candidate.region_id:
                    raise ValueError(
                        "backend/representation candidates reference different regions"
                    )
                reasons = capability.rejection_reasons(
                    workload=self.workload,
                    op_kinds=tuple(ops[op_id].kind for op_id in region.op_ids),
                    representation=representation.representation,
                )
                if (
                    reasons
                    and backend_candidate.static_legal
                    and representation.static_legal
                ):
                    raise ValueError(
                        "backend candidate marked legal despite capability rejection"
                    )
                if reasons and not set(reasons).issubset(
                    set(backend_candidate.rejection_reasons)
                    | set(representation.rejection_reasons)
                ):
                    raise ValueError(
                        "candidate rejection omits capability rejection reasons"
                    )
        for storage_candidate in self.storage_candidates:
            if any(
                batch_id not in batches
                for batch_id in storage_candidate.compatible_batch_candidate_ids
            ):
                raise ValueError("storage references unknown batch candidate")
            if any(
                representation_id not in representations
                for representation_id in (
                    storage_candidate.compatible_representation_candidate_ids
                )
            ):
                raise ValueError("storage references unknown representation candidate")
            self._validate_storage_candidate(
                storage_candidate,
                bound_module=bound_module,
            )

    def _validate_storage_candidate(
        self,
        candidate: StorageCandidate,
        *,
        bound_module: BFBoundModule,
    ) -> None:
        """Validate sizes, alignment, lifetimes, and physical alias safety."""

        values = {value.value_id: value for value in bound_module.graph.values}
        op_index = {op.op_id: index for index, op in enumerate(bound_module.graph.ops)}
        producer = {
            value_id: op.op_id
            for op in bound_module.graph.ops
            for value_id in op.outputs
        }
        users: dict[str, list[str]] = {}
        for op in bound_module.graph.ops:
            for value_id in op.inputs:
                users.setdefault(value_id, []).append(op.op_id)
        last_index = len(bound_module.graph.ops) - 1
        for binding in candidate.bindings:
            value = values[binding.value_id]
            if binding.offset_bytes % self.hardware.alignment_bytes != 0:
                raise ValueError("storage binding violates hardware alignment")
            logical_bytes = _static_tensor_bytes(
                value.tensor_type.shape, value.tensor_type.dtype
            )
            if (
                logical_bytes is not None
                and binding.logical_size_bytes != logical_bytes
            ):
                raise ValueError("storage logical size/Bound IR value mismatch")
            live_from = op_index[binding.live_from_op_id]
            live_to = op_index[binding.live_to_op_id]
            required_from = (
                op_index[producer[binding.value_id]]
                if binding.value_id in producer
                else min(
                    (op_index[user] for user in users.get(binding.value_id, [])),
                    default=0,
                )
            )
            required_to = max(
                (op_index[user] for user in users.get(binding.value_id, [])),
                default=(
                    last_index
                    if binding.value_id in bound_module.graph.outputs
                    else required_from
                ),
            )
            if live_from > required_from or live_to < required_to:
                raise ValueError("storage lifetime does not cover value uses")

        for index, left in enumerate(candidate.bindings):
            for right in candidate.bindings[index + 1 :]:
                if left.arena_id != right.arena_id:
                    continue
                byte_overlap = (
                    left.offset_bytes < right.end_bytes
                    and right.offset_bytes < left.end_bytes
                )
                lifetime_overlap = (
                    op_index[left.live_from_op_id] <= op_index[right.live_to_op_id]
                    and op_index[right.live_from_op_id] <= op_index[left.live_to_op_id]
                )
                if byte_overlap and lifetime_overlap:
                    raise ValueError(
                        "storage candidate aliases simultaneously live values"
                    )

    def to_dict(self) -> dict[str, object]:
        """Return canonical JSON-compatible PlanTemplate fields."""

        return {
            "schema_version": self.schema_version,
            "template_id": self.template_id,
            "bound_module_hash": self.bound_module_hash,
            "planner_config_hash": self.planner_config_hash,
            "hardware": self.hardware.to_dict(),
            "workload": self.workload.to_dict(),
            "capabilities": [capability.to_dict() for capability in self.capabilities],
            "region_candidates": [
                candidate.to_dict() for candidate in self.region_candidates
            ],
            "representation_candidates": [
                candidate.to_dict() for candidate in self.representation_candidates
            ],
            "materialization_candidates": [
                candidate.to_dict() for candidate in self.materialization_candidates
            ],
            "backend_candidates": [
                candidate.to_dict() for candidate in self.backend_candidates
            ],
            "batch_candidates": [
                candidate.to_dict() for candidate in self.batch_candidates
            ],
            "storage_candidates": [
                candidate.to_dict() for candidate in self.storage_candidates
            ],
            "state_candidates": [
                candidate.to_dict() for candidate in self.state_candidates
            ],
            "provenance": [item.to_dict() for item in self.provenance],
        }

    def canonical_json(self, *, bound_module: BFBoundModule) -> str:
        self.validate(bound_module=bound_module)
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self, *, bound_module: BFBoundModule) -> str:
        return hashlib.sha256(
            self.canonical_json(bound_module=bound_module).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class RegionDecision:
    """Select one region/partition candidate."""

    region_id: str
    candidate_id: str

    def validate(self) -> None:
        _validate_candidate_identity(self.candidate_id, self.region_id)


@dataclass(frozen=True)
class RepresentationDecision:
    """Select one representation candidate for a region."""

    region_id: str
    candidate_id: str

    def validate(self) -> None:
        _validate_candidate_identity(self.candidate_id, self.region_id)


@dataclass(frozen=True)
class MaterializationDecision:
    """Select one explicit representation transition."""

    candidate_id: str

    def validate(self) -> None:
        if not self.candidate_id:
            raise ValueError("materialization decision candidate_id is empty")


@dataclass(frozen=True)
class BackendDecision:
    """Select one backend candidate for a region."""

    region_id: str
    candidate_id: str

    def validate(self) -> None:
        _validate_candidate_identity(self.candidate_id, self.region_id)


@dataclass(frozen=True)
class BatchDecision:
    """Select one domain/spec/sample batching candidate."""

    candidate_id: str

    def validate(self) -> None:
        if not self.candidate_id:
            raise ValueError("batch decision candidate_id is empty")


@dataclass(frozen=True)
class StorageDecision:
    """Select one whole-plan storage/lifetime candidate."""

    candidate_id: str

    def validate(self) -> None:
        if not self.candidate_id:
            raise ValueError("storage decision candidate_id is empty")


@dataclass(frozen=True)
class StateDecision:
    """Select one reuse/cache/recompute/evict candidate for a state."""

    state_id: str
    candidate_id: str

    def validate(self) -> None:
        if not self.state_id or not self.candidate_id:
            raise ValueError("state decision IDs must be non-empty")


@dataclass(frozen=True)
class StateValidity:
    """Query-time evidence describing one available or invalid cached state."""

    state_id: str
    source_value_id: str
    state_version: str
    valid: bool
    invalidation_reason: Optional[str] = None

    def validate(self) -> None:
        for name in ("state_id", "source_value_id", "state_version"):
            if not getattr(self, name):
                raise ValueError(f"state validity {name} must be non-empty")
        if self.valid and self.invalidation_reason is not None:
            raise ValueError("valid state cannot have an invalidation reason")
        if not self.valid and not self.invalidation_reason:
            raise ValueError("invalid state requires an invalidation reason")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class RejectedCandidate:
    """One unselected candidate and deterministic rejection reasons."""

    candidate_id: str
    reasons: Tuple[str, ...]

    def validate(self) -> None:
        if not self.candidate_id or not self.reasons:
            raise ValueError("rejected candidate requires ID and reasons")
        if any(not reason for reason in self.reasons):
            raise ValueError("rejected candidate reasons must be non-empty")
        _require_unique(self.reasons, label="rejection reasons")


@dataclass(frozen=True)
class PlanInstance:  # pylint: disable=too-many-instance-attributes
    """Dynamic, fully accounted selection from one PlanTemplate."""

    instance_id: str
    template_hash: str
    query_bucket_id: str
    available_memory_bytes: int
    memory_budget_bytes: int
    deadline_us: Optional[int]
    region_decisions: Tuple[RegionDecision, ...]
    representation_decisions: Tuple[RepresentationDecision, ...]
    materialization_decisions: Tuple[MaterializationDecision, ...]
    backend_decisions: Tuple[BackendDecision, ...]
    batch_decision: BatchDecision
    storage_decision: StorageDecision
    state_decisions: Tuple[StateDecision, ...]
    rejected_candidates: Tuple[RejectedCandidate, ...]
    cost_summary: PlanCost
    state_validities: Tuple[StateValidity, ...] = ()
    provenance: Tuple[PlanProvenance, ...] = ()
    schema_version: str = PLAN_IR_SCHEMA_VERSION

    def validate(
        self, *, template: PlanTemplate, bound_module: BFBoundModule
    ) -> None:  # pylint: disable=too-many-locals,too-many-statements
        """Cross-check partition, legality, memory, state, and full accounting."""

        if self.schema_version != PLAN_IR_SCHEMA_VERSION:
            raise ValueError(f"unsupported Plan IR schema: {self.schema_version}")
        for name in ("instance_id", "template_hash", "query_bucket_id"):
            if not getattr(self, name):
                raise ValueError(f"plan instance {name} must be non-empty")
        if self.available_memory_bytes <= 0 or self.memory_budget_bytes <= 0:
            raise ValueError("plan instance memory limits must be positive")
        if self.deadline_us is not None and self.deadline_us <= 0:
            raise ValueError("plan instance deadline must be positive when present")
        template.validate(bound_module=bound_module)
        if self.template_hash != template.stable_hash(bound_module=bound_module):
            raise ValueError("plan instance template hash mismatch")
        for region_decision in self.region_decisions:
            region_decision.validate()
        for representation_decision in self.representation_decisions:
            representation_decision.validate()
        for materialization_decision in self.materialization_decisions:
            materialization_decision.validate()
        for backend_decision in self.backend_decisions:
            backend_decision.validate()
        for state_decision in self.state_decisions:
            state_decision.validate()
        for state_validity in self.state_validities:
            state_validity.validate()
        self.batch_decision.validate()
        self.storage_decision.validate()
        for rejection in self.rejected_candidates:
            rejection.validate()
        for item in self.provenance:
            item.validate()
        _require_unique(
            tuple(item.key for item in self.provenance),
            label="instance provenance keys",
        )
        self.cost_summary.validate()

        candidate_map = template.candidate_map()
        regions = {
            candidate.candidate_id: candidate
            for candidate in template.region_candidates
        }
        representations = {
            candidate.candidate_id: candidate
            for candidate in template.representation_candidates
        }
        transitions = {
            candidate.candidate_id: candidate
            for candidate in template.materialization_candidates
        }
        backends = {
            candidate.candidate_id: candidate
            for candidate in template.backend_candidates
        }
        batches = {
            candidate.candidate_id: candidate for candidate in template.batch_candidates
        }
        storages = {
            candidate.candidate_id: candidate
            for candidate in template.storage_candidates
        }
        states = {
            candidate.candidate_id: candidate for candidate in template.state_candidates
        }

        selected_region_ids = tuple(
            decision.candidate_id for decision in self.region_decisions
        )
        selected_representation_ids = tuple(
            decision.candidate_id for decision in self.representation_decisions
        )
        selected_transition_ids = tuple(
            decision.candidate_id for decision in self.materialization_decisions
        )
        selected_backend_ids = tuple(
            decision.candidate_id for decision in self.backend_decisions
        )
        selected_state_ids = tuple(
            decision.candidate_id for decision in self.state_decisions
        )
        selected_ids = (
            *selected_region_ids,
            *selected_representation_ids,
            *selected_transition_ids,
            *selected_backend_ids,
            self.batch_decision.candidate_id,
            self.storage_decision.candidate_id,
            *selected_state_ids,
        )
        _require_unique(selected_ids, label="selected candidate IDs")
        if any(candidate_id not in candidate_map for candidate_id in selected_ids):
            raise ValueError("plan instance selects an unknown candidate")

        selected_regions = tuple(
            regions[candidate_id] for candidate_id in selected_region_ids
        )
        graph_op_ids = {op.op_id for op in bound_module.graph.ops}
        covered_op_ids = [
            op_id for region in selected_regions for op_id in region.op_ids
        ]
        if len(covered_op_ids) != len(set(covered_op_ids)):
            raise ValueError("selected region partition overlaps")
        if set(covered_op_ids) != graph_op_ids:
            raise ValueError("selected region partition does not cover Bound IR")
        logical_region_ids = tuple(region.region_id for region in selected_regions)
        _require_unique(logical_region_ids, label="selected logical region IDs")

        representation_by_region = self._representation_decision_map(
            self.representation_decisions,
            representations,
            logical_region_ids=logical_region_ids,
        )
        backend_by_region = self._backend_decision_map(
            self.backend_decisions,
            backends,
            logical_region_ids=logical_region_ids,
        )
        selected_transition_set = set(selected_transition_ids)
        for region_id, representation in representation_by_region.items():
            if not representation.static_legal:
                raise ValueError("plan selects statically illegal representation")
            required = set(representation.required_transition_candidate_ids)
            if not required.issubset(selected_transition_set):
                raise ValueError("plan omits a required representation transition")
            backend = backend_by_region[region_id]
            if not backend.static_legal:
                raise ValueError("plan selects statically illegal backend")
            if representation.candidate_id not in (
                backend.compatible_representation_candidate_ids
            ):
                raise ValueError("selected backend/representation are incompatible")
        required_transition_ids = {
            transition_id
            for representation in representation_by_region.values()
            for transition_id in representation.required_transition_candidate_ids
        }
        if selected_transition_set != required_transition_ids:
            raise ValueError("plan selects unused or misses required transitions")
        if any(
            not transitions[candidate_id].static_legal
            for candidate_id in selected_transition_ids
        ):
            raise ValueError("plan selects statically illegal transition")

        batch = batches.get(self.batch_decision.candidate_id)
        storage = storages.get(self.storage_decision.candidate_id)
        if batch is None or storage is None:
            raise ValueError("batch/storage decision has the wrong candidate type")
        if not batch.static_legal or not storage.static_legal:
            raise ValueError("plan selects illegal batch/storage candidate")
        if batch.candidate_id not in storage.compatible_batch_candidate_ids:
            raise ValueError("selected storage/batch candidates are incompatible")
        if any(
            representation.candidate_id
            not in storage.compatible_representation_candidate_ids
            for representation in representation_by_region.values()
        ):
            raise ValueError(
                "selected storage/representation candidates are incompatible"
            )
        if batch.domain_batch_size > template.workload.domain_batch_size:
            raise ValueError("selected domain batch exceeds workload bucket")
        if batch.spec_batch_size > template.workload.spec_batch_size:
            raise ValueError("selected spec batch exceeds workload bucket")
        if batch.sample_batch_size > template.workload.sample_batch_size:
            raise ValueError("selected sample batch exceeds workload bucket")

        validity_by_state: dict[str, StateValidity] = {}
        bound_values = {value.value_id: value for value in bound_module.graph.values}
        template_state_ids = {candidate.state_id for candidate in states.values()}
        for validity in self.state_validities:
            if validity.state_id in validity_by_state:
                raise ValueError("duplicate query-time state validity")
            if validity.state_id not in template_state_ids:
                raise ValueError("state validity references an unknown template state")
            value = bound_values.get(validity.source_value_id)
            if value is None:
                raise ValueError("state validity references an unknown Bound IR value")
            if validity.valid and value.state_version != validity.state_version:
                raise ValueError("valid cached state has a stale Bound IR version")
            validity_by_state[validity.state_id] = validity

        state_ids: list[str] = []
        for state_decision in self.state_decisions:
            candidate = states.get(state_decision.candidate_id)
            if candidate is None or candidate.state_id != state_decision.state_id:
                raise ValueError("state decision/candidate identity mismatch")
            if not candidate.static_legal:
                raise ValueError("plan selects illegal state candidate")
            if candidate.action == StateAction.REUSE:
                selected_validity = validity_by_state.get(candidate.state_id)
                if (
                    selected_validity is None
                    or not selected_validity.valid
                    or selected_validity.source_value_id != candidate.source_value_id
                    or selected_validity.state_version != candidate.state_version
                ):
                    raise ValueError(
                        "plan selects state reuse without exact valid cache evidence"
                    )
            state_ids.append(state_decision.state_id)
        _require_unique(tuple(state_ids), label="selected state IDs")
        if set(state_ids) != template_state_ids:
            raise ValueError("state decisions do not cover every template state")

        rejected_ids = tuple(
            rejection.candidate_id for rejection in self.rejected_candidates
        )
        _require_unique(rejected_ids, label="rejected candidate IDs")
        if any(candidate_id not in candidate_map for candidate_id in rejected_ids):
            raise ValueError("plan rejects an unknown candidate")
        if set(selected_ids) & set(rejected_ids):
            raise ValueError("candidate cannot be both selected and rejected")
        if set(selected_ids) | set(rejected_ids) != set(candidate_map):
            raise ValueError("plan instance does not account for every candidate")

        effective_budget = min(
            self.available_memory_bytes,
            self.memory_budget_bytes,
            template.hardware.total_memory_bytes,
        )
        if self.cost_summary.predicted_peak_bytes != storage.cost.predicted_peak_bytes:
            raise ValueError("plan cost summary/storage peak mismatch")
        if self.cost_summary.predicted_peak_bytes > effective_budget:
            raise ValueError("selected plan exceeds effective memory budget")
        selected_candidates = tuple(
            candidate_map[candidate_id] for candidate_id in selected_ids
        )
        if self.cost_summary.predicted_latency_ms < max(
            candidate.cost.predicted_latency_ms for candidate in selected_candidates
        ):
            raise ValueError("plan summary under-reports selected latency evidence")
        if self.cost_summary.compile_cost_ms < max(
            candidate.cost.compile_cost_ms for candidate in selected_candidates
        ):
            raise ValueError("plan summary under-reports selected compile cost")
        if self.cost_summary.setup_cost_ms < max(
            candidate.cost.setup_cost_ms for candidate in selected_candidates
        ):
            raise ValueError("plan summary under-reports selected setup cost")
        if self.cost_summary.confidence > min(
            candidate.cost.confidence for candidate in selected_candidates
        ):
            raise ValueError("plan summary overstates selected confidence")
        selected_risks = {
            risk
            for candidate in selected_candidates
            for risk in candidate.cost.risk_tags
        }
        if not selected_risks.issubset(set(self.cost_summary.risk_tags)):
            raise ValueError("plan summary omits selected candidate risks")

    @staticmethod
    def _representation_decision_map(
        decisions: Tuple[RepresentationDecision, ...],
        candidates: dict[str, RepresentationCandidate],
        *,
        logical_region_ids: Tuple[str, ...],
    ) -> dict[str, RepresentationCandidate]:
        result: dict[str, RepresentationCandidate] = {}
        for decision in decisions:
            candidate = candidates.get(decision.candidate_id)
            if candidate is None:
                raise ValueError("representation decision has the wrong candidate type")
            if candidate.region_id != decision.region_id:
                raise ValueError("representation decision/candidate region mismatch")
            if decision.region_id in result:
                raise ValueError("duplicate representation decision for one region")
            result[decision.region_id] = candidate
        if set(result) != set(logical_region_ids):
            raise ValueError("representation decisions do not cover selected regions")
        return result

    @staticmethod
    def _backend_decision_map(
        decisions: Tuple[BackendDecision, ...],
        candidates: dict[str, BackendCandidate],
        *,
        logical_region_ids: Tuple[str, ...],
    ) -> dict[str, BackendCandidate]:
        result: dict[str, BackendCandidate] = {}
        for decision in decisions:
            candidate = candidates.get(decision.candidate_id)
            if candidate is None:
                raise ValueError("backend decision has the wrong candidate type")
            if candidate.region_id != decision.region_id:
                raise ValueError("backend decision/candidate region mismatch")
            if decision.region_id in result:
                raise ValueError("duplicate backend decision for one region")
            result[decision.region_id] = candidate
        if set(result) != set(logical_region_ids):
            raise ValueError("backend decisions do not cover selected regions")
        return result

    def to_dict(self) -> dict[str, object]:
        """Return canonical JSON-compatible PlanInstance fields."""

        return {
            "schema_version": self.schema_version,
            "instance_id": self.instance_id,
            "template_hash": self.template_hash,
            "query_bucket_id": self.query_bucket_id,
            "available_memory_bytes": self.available_memory_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "deadline_us": self.deadline_us,
            "region_decisions": [asdict(item) for item in self.region_decisions],
            "representation_decisions": [
                asdict(item) for item in self.representation_decisions
            ],
            "materialization_decisions": [
                asdict(item) for item in self.materialization_decisions
            ],
            "backend_decisions": [asdict(item) for item in self.backend_decisions],
            "batch_decision": asdict(self.batch_decision),
            "storage_decision": asdict(self.storage_decision),
            "state_decisions": [asdict(item) for item in self.state_decisions],
            "state_validities": [item.to_dict() for item in self.state_validities],
            "rejected_candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "reasons": list(item.reasons),
                }
                for item in self.rejected_candidates
            ],
            "cost_summary": self.cost_summary.to_dict(),
            "provenance": [item.to_dict() for item in self.provenance],
        }

    def canonical_json(
        self, *, template: PlanTemplate, bound_module: BFBoundModule
    ) -> str:
        self.validate(template=template, bound_module=bound_module)
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(
        self, *, template: PlanTemplate, bound_module: BFBoundModule
    ) -> str:
        return hashlib.sha256(
            self.canonical_json(template=template, bound_module=bound_module).encode(
                "utf-8"
            )
        ).hexdigest()

    @classmethod
    def from_canonical_json(
        cls,
        encoded: str,
        *,
        template: PlanTemplate,
        bound_module: BFBoundModule,
    ) -> "PlanInstance":
        """Replay a serialized selection against an exact PlanTemplate hash."""

        try:
            raw: object = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise ValueError("invalid PlanInstance JSON") from error
        payload = _expect_object(raw, label="PlanInstance")
        _expect_exact_keys(
            payload,
            {
                "schema_version",
                "instance_id",
                "template_hash",
                "query_bucket_id",
                "available_memory_bytes",
                "memory_budget_bytes",
                "deadline_us",
                "region_decisions",
                "representation_decisions",
                "materialization_decisions",
                "backend_decisions",
                "batch_decision",
                "storage_decision",
                "state_decisions",
                "state_validities",
                "rejected_candidates",
                "cost_summary",
                "provenance",
            },
            label="PlanInstance",
        )
        deadline_raw = payload["deadline_us"]
        deadline = (
            None
            if deadline_raw is None
            else _expect_int(deadline_raw, label="deadline_us")
        )
        instance = cls(
            schema_version=_expect_string(
                payload["schema_version"], label="schema_version"
            ),
            instance_id=_expect_string(payload["instance_id"], label="instance_id"),
            template_hash=_expect_string(
                payload["template_hash"], label="template_hash"
            ),
            query_bucket_id=_expect_string(
                payload["query_bucket_id"], label="query_bucket_id"
            ),
            available_memory_bytes=_expect_int(
                payload["available_memory_bytes"],
                label="available_memory_bytes",
            ),
            memory_budget_bytes=_expect_int(
                payload["memory_budget_bytes"], label="memory_budget_bytes"
            ),
            deadline_us=deadline,
            region_decisions=tuple(
                _parse_region_decision(item)
                for item in _expect_list(
                    payload["region_decisions"], label="region_decisions"
                )
            ),
            representation_decisions=tuple(
                _parse_representation_decision(item)
                for item in _expect_list(
                    payload["representation_decisions"],
                    label="representation_decisions",
                )
            ),
            materialization_decisions=tuple(
                _parse_materialization_decision(item)
                for item in _expect_list(
                    payload["materialization_decisions"],
                    label="materialization_decisions",
                )
            ),
            backend_decisions=tuple(
                _parse_backend_decision(item)
                for item in _expect_list(
                    payload["backend_decisions"], label="backend_decisions"
                )
            ),
            batch_decision=_parse_batch_decision(payload["batch_decision"]),
            storage_decision=_parse_storage_decision(payload["storage_decision"]),
            state_decisions=tuple(
                _parse_state_decision(item)
                for item in _expect_list(
                    payload["state_decisions"], label="state_decisions"
                )
            ),
            state_validities=tuple(
                _parse_state_validity(item)
                for item in _expect_list(
                    payload["state_validities"], label="state_validities"
                )
            ),
            rejected_candidates=tuple(
                _parse_rejected_candidate(item)
                for item in _expect_list(
                    payload["rejected_candidates"],
                    label="rejected_candidates",
                )
            ),
            cost_summary=_parse_plan_cost(payload["cost_summary"]),
            provenance=tuple(
                _parse_provenance(item)
                for item in _expect_list(payload["provenance"], label="provenance")
            ),
        )
        instance.validate(template=template, bound_module=bound_module)
        if (
            instance.canonical_json(template=template, bound_module=bound_module)
            != encoded
        ):
            raise ValueError("PlanInstance JSON is not canonical")
        return instance


def _validate_candidate_identity(candidate_id: str, region_id: str) -> None:
    if not candidate_id or not region_id:
        raise ValueError("candidate/region IDs must be non-empty")


def _validate_static_legality(
    static_legal: bool, reasons: Tuple[str, ...], *, label: str
) -> None:
    if static_legal and reasons:
        raise ValueError(f"legal candidate '{label}' cannot have rejection reasons")
    if not static_legal and not reasons:
        raise ValueError(f"illegal candidate '{label}' requires rejection reasons")
    if any(not reason for reason in reasons):
        raise ValueError(f"candidate '{label}' has an empty rejection reason")
    _require_unique(reasons, label=f"candidate '{label}' rejection reasons")


def _require_unique(values: Tuple[object, ...], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _static_tensor_bytes(shape: Tuple[Optional[int], ...], dtype: str) -> Optional[int]:
    if any(dimension is None for dimension in shape):
        return None
    dtype_bytes = {
        "bool": 1,
        "int8": 1,
        "uint8": 1,
        "float16": 2,
        "bfloat16": 2,
        "int16": 2,
        "float32": 4,
        "int32": 4,
        "float64": 8,
        "int64": 8,
    }.get(dtype)
    if dtype_bytes is None:
        raise ValueError(f"Plan IR cannot size unsupported dtype '{dtype}'")
    elements = 1
    for dimension in shape:
        if dimension is None:
            return None
        elements *= dimension
    return elements * dtype_bytes


def _parse_region_decision(value: object) -> RegionDecision:
    payload = _expect_object(value, label="RegionDecision")
    _expect_exact_keys(payload, {"region_id", "candidate_id"}, label="RegionDecision")
    return RegionDecision(
        region_id=_expect_string(payload["region_id"], label="region_id"),
        candidate_id=_expect_string(payload["candidate_id"], label="candidate_id"),
    )


def _parse_representation_decision(value: object) -> RepresentationDecision:
    payload = _expect_object(value, label="RepresentationDecision")
    _expect_exact_keys(
        payload, {"region_id", "candidate_id"}, label="RepresentationDecision"
    )
    return RepresentationDecision(
        region_id=_expect_string(payload["region_id"], label="region_id"),
        candidate_id=_expect_string(payload["candidate_id"], label="candidate_id"),
    )


def _parse_materialization_decision(value: object) -> MaterializationDecision:
    payload = _expect_object(value, label="MaterializationDecision")
    _expect_exact_keys(payload, {"candidate_id"}, label="MaterializationDecision")
    return MaterializationDecision(
        candidate_id=_expect_string(payload["candidate_id"], label="candidate_id")
    )


def _parse_backend_decision(value: object) -> BackendDecision:
    payload = _expect_object(value, label="BackendDecision")
    _expect_exact_keys(payload, {"region_id", "candidate_id"}, label="BackendDecision")
    return BackendDecision(
        region_id=_expect_string(payload["region_id"], label="region_id"),
        candidate_id=_expect_string(payload["candidate_id"], label="candidate_id"),
    )


def _parse_batch_decision(value: object) -> BatchDecision:
    payload = _expect_object(value, label="BatchDecision")
    _expect_exact_keys(payload, {"candidate_id"}, label="BatchDecision")
    return BatchDecision(
        candidate_id=_expect_string(payload["candidate_id"], label="candidate_id")
    )


def _parse_storage_decision(value: object) -> StorageDecision:
    payload = _expect_object(value, label="StorageDecision")
    _expect_exact_keys(payload, {"candidate_id"}, label="StorageDecision")
    return StorageDecision(
        candidate_id=_expect_string(payload["candidate_id"], label="candidate_id")
    )


def _parse_state_decision(value: object) -> StateDecision:
    payload = _expect_object(value, label="StateDecision")
    _expect_exact_keys(payload, {"state_id", "candidate_id"}, label="StateDecision")
    return StateDecision(
        state_id=_expect_string(payload["state_id"], label="state_id"),
        candidate_id=_expect_string(payload["candidate_id"], label="candidate_id"),
    )


def _parse_state_validity(value: object) -> StateValidity:
    payload = _expect_object(value, label="StateValidity")
    _expect_exact_keys(
        payload,
        {
            "state_id",
            "source_value_id",
            "state_version",
            "valid",
            "invalidation_reason",
        },
        label="StateValidity",
    )
    invalidation_raw = payload["invalidation_reason"]
    if invalidation_raw is not None and not isinstance(invalidation_raw, str):
        raise ValueError("invalidation_reason must be a string or null")
    valid_raw = payload["valid"]
    if not isinstance(valid_raw, bool):
        raise ValueError("state validity valid must be a boolean")
    return StateValidity(
        state_id=_expect_string(payload["state_id"], label="state_id"),
        source_value_id=_expect_string(
            payload["source_value_id"], label="source_value_id"
        ),
        state_version=_expect_string(payload["state_version"], label="state_version"),
        valid=valid_raw,
        invalidation_reason=invalidation_raw,
    )


def _parse_rejected_candidate(value: object) -> RejectedCandidate:
    payload = _expect_object(value, label="RejectedCandidate")
    _expect_exact_keys(payload, {"candidate_id", "reasons"}, label="RejectedCandidate")
    return RejectedCandidate(
        candidate_id=_expect_string(payload["candidate_id"], label="candidate_id"),
        reasons=tuple(
            _expect_string(reason, label="rejection reason")
            for reason in _expect_list(payload["reasons"], label="reasons")
        ),
    )


def _parse_plan_cost(value: object) -> PlanCost:
    payload = _expect_object(value, label="PlanCost")
    _expect_exact_keys(
        payload,
        {
            "predicted_latency_ms",
            "predicted_peak_bytes",
            "compile_cost_ms",
            "setup_cost_ms",
            "confidence",
            "risk_tags",
        },
        label="PlanCost",
    )
    return PlanCost(
        predicted_latency_ms=_expect_float(
            payload["predicted_latency_ms"], label="predicted_latency_ms"
        ),
        predicted_peak_bytes=_expect_int(
            payload["predicted_peak_bytes"], label="predicted_peak_bytes"
        ),
        compile_cost_ms=_expect_float(
            payload["compile_cost_ms"], label="compile_cost_ms"
        ),
        setup_cost_ms=_expect_float(payload["setup_cost_ms"], label="setup_cost_ms"),
        confidence=_expect_float(payload["confidence"], label="confidence"),
        risk_tags=tuple(
            _expect_string(risk, label="risk tag")
            for risk in _expect_list(payload["risk_tags"], label="risk_tags")
        ),
    )


def _parse_provenance(value: object) -> PlanProvenance:
    payload = _expect_object(value, label="PlanProvenance")
    _expect_exact_keys(payload, {"key", "value"}, label="PlanProvenance")
    return PlanProvenance(
        key=_expect_string(payload["key"], label="provenance key"),
        value=_expect_string(payload["value"], label="provenance value"),
    )


def _expect_object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    return {str(key): item for key, item in value.items()}


def _expect_list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return list(value)


def _expect_string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _expect_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _expect_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _expect_exact_keys(
    payload: dict[str, object], expected: set[str], *, label: str
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} fields do not match the Plan IR schema")
