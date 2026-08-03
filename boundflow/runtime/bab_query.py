"""State-versioned query contracts and fixed-stream replay for PR-13."""

# The query contract intentionally keeps all semantic identity fields in one
# immutable record.  Splitting it only to satisfy style limits would weaken the
# audit boundary this module provides.
# pylint: disable=too-many-lines

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

import torch

from ..domains.interval import IntervalState
from ..ir.task import BFTaskModule
from ..planner.materialization import BoundMethod, OptimizationStage
from .alpha_beta_crown import BetaState, run_alpha_beta_crown_mlp
from .alpha_crown import AlphaState, run_alpha_crown_mlp
from .perturbation import BoxPerturbation, LpBallPerturbation
from .task_executor import InputSpec, InputSpecLike, _normalize_input_spec

QUERY_SCHEMA_VERSION = "boundflow.bab-query/v1"
QUERY_TRACE_SCHEMA_VERSION = "boundflow.bab-query-trace/v1"


class ReuseClass(Enum):
    """Allowed semantic relationship between cached state and a new query."""

    EXACT_REUSE = "exact_reuse"
    CONDITIONAL_REUSE = "conditional_reuse"
    WARM_START_ONLY = "warm_start_only"
    INVALIDATE = "invalidate"


class StateArtifactKind(Enum):
    """Objects whose parent/child validity differs materially."""

    GRAPH_STRUCTURE = "graph_structure"
    COMPILED_MODULE = "compiled_module"
    PLANNER_TEMPLATE = "planner_template"
    WEIGHT_PREPROCESS = "weight_preprocess"
    INTERMEDIATE_BOUNDS = "intermediate_bounds"
    ALPHA_STATE = "alpha_state"
    BETA_STATE = "beta_state"
    CUTS = "cuts"
    FINAL_BOUNDS = "final_bounds"
    COMPILE_PROFILE = "compile_profile"


@dataclass(frozen=True)
class QueryIdentityVersions:
    """Precomputed stable versions that avoid per-node device synchronization."""

    model_structure_hash: Optional[str] = None
    weight_version: Optional[str] = None
    input_region_hash: Optional[str] = None
    output_spec_hash: Optional[str] = None
    split_signature: Optional[str] = None
    alpha_state_version: Optional[str] = None
    beta_state_version: Optional[str] = None


@dataclass(frozen=True)
class QueryCompatibilityKey:  # pylint: disable=too-many-instance-attributes
    """Exact fields required for legal GPU batching and compiled-plan reuse."""

    model_structure_hash: str
    weight_version: str
    bound_method: str
    optimization_stage: str
    requires_grad: bool
    input_value_name: str
    input_shape: Tuple[int, ...]
    spec_shape: Tuple[int, ...]
    split_tensor_shapes: Tuple[Tuple[str, Tuple[int, ...]], ...]
    dtype: str
    device: str
    perturbation_signature: str
    execution_options_hash: str
    backend_capability_class: str
    numeric_policy: str

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible fields."""

        payload = asdict(self)
        payload["input_shape"] = list(self.input_shape)
        payload["spec_shape"] = list(self.spec_shape)
        payload["split_tensor_shapes"] = [
            [name, list(shape)] for name, shape in self.split_tensor_shapes
        ]
        return payload


@dataclass(frozen=True)
class BoundQuery:  # pylint: disable=too-many-instance-attributes
    """Serializable identity and semantic context for one BaB bound call."""

    query_id: str
    parent_query_id: Optional[str]
    sequence_number: int
    example_idx: int
    model_structure_hash: str
    weight_version: str
    input_region_hash: str
    output_spec_hash: str
    split_signature: str
    bound_method: BoundMethod
    optimization_stage: OptimizationStage
    requires_grad: bool
    alpha_state_version: Optional[str]
    beta_state_version: Optional[str]
    cuts_version: Optional[str]
    dtype: str
    device: str
    numeric_policy: str
    requested_outputs: Tuple[str, ...]
    compatibility_key: QueryCompatibilityKey
    execution_options: Mapping[str, object]
    schema_version: str = QUERY_SCHEMA_VERSION

    def validate(self) -> None:
        """Reject incomplete identities and mismatched compatibility metadata."""

        if self.schema_version != QUERY_SCHEMA_VERSION:
            raise ValueError(f"unsupported query schema: {self.schema_version}")
        for name in (
            "query_id",
            "model_structure_hash",
            "weight_version",
            "input_region_hash",
            "output_spec_hash",
            "split_signature",
            "dtype",
            "device",
            "numeric_policy",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.sequence_number < 0 or self.example_idx < 0:
            raise ValueError("sequence_number and example_idx must be non-negative")
        if not self.requested_outputs:
            raise ValueError("requested_outputs must be non-empty")
        if self.compatibility_key.model_structure_hash != self.model_structure_hash:
            raise ValueError("compatibility model hash mismatch")
        if self.compatibility_key.weight_version != self.weight_version:
            raise ValueError("compatibility weight version mismatch")

    def to_dict(self) -> dict[str, object]:
        """Return canonical JSON-compatible data."""

        self.validate()
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            "parent_query_id": self.parent_query_id,
            "sequence_number": self.sequence_number,
            "example_idx": self.example_idx,
            "model_structure_hash": self.model_structure_hash,
            "weight_version": self.weight_version,
            "input_region_hash": self.input_region_hash,
            "output_spec_hash": self.output_spec_hash,
            "split_signature": self.split_signature,
            "bound_method": self.bound_method.value,
            "optimization_stage": self.optimization_stage.value,
            "requires_grad": self.requires_grad,
            "alpha_state_version": self.alpha_state_version,
            "beta_state_version": self.beta_state_version,
            "cuts_version": self.cuts_version,
            "dtype": self.dtype,
            "device": self.device,
            "numeric_policy": self.numeric_policy,
            "requested_outputs": list(self.requested_outputs),
            "compatibility_key": self.compatibility_key.to_dict(),
            "execution_options": dict(self.execution_options),
        }

    def canonical_json(self) -> str:
        """Serialize deterministically for hashing and fixed-stream artifacts."""

        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )


@dataclass(frozen=True)
class BoundQueryPayload:
    """Runtime values kept outside the serializable query identity."""

    input_spec: InputSpec
    linear_spec_c: Optional[torch.Tensor]
    split_by_relu_input: Mapping[str, torch.Tensor]
    warm_alpha_by_relu_input: Mapping[str, torch.Tensor]
    warm_beta_by_relu_input: Mapping[str, torch.Tensor]

    def detached_clone(self) -> "BoundQueryPayload":
        """Own all tensors so later solver mutation cannot alter the trace."""

        spec = self.input_spec
        return BoundQueryPayload(
            input_spec=InputSpec(
                value_name=spec.value_name,
                center=spec.center.detach().clone(),
                perturbation=spec.perturbation,
            ),
            linear_spec_c=(
                None
                if self.linear_spec_c is None
                else self.linear_spec_c.detach().clone()
            ),
            split_by_relu_input={
                name: tensor.detach().clone()
                for name, tensor in self.split_by_relu_input.items()
            },
            warm_alpha_by_relu_input={
                name: tensor.detach().clone()
                for name, tensor in self.warm_alpha_by_relu_input.items()
            },
            warm_beta_by_relu_input={
                name: tensor.detach().clone()
                for name, tensor in self.warm_beta_by_relu_input.items()
            },
        )


@dataclass(frozen=True)
class BoundQueryResult:  # pylint: disable=too-many-instance-attributes
    """Expected or replayed per-query outputs."""

    status: str
    lower: Optional[torch.Tensor]
    upper: Optional[torch.Tensor]
    branch: Optional[Tuple[str, int]]
    alpha_state_version: Optional[str]
    beta_state_version: Optional[str]
    alpha_by_relu_input: Mapping[str, torch.Tensor] = field(default_factory=dict)
    beta_by_relu_input: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def detached_clone(self) -> "BoundQueryResult":
        """Own result tensors for deterministic later comparison."""

        return BoundQueryResult(
            status=self.status,
            lower=None if self.lower is None else self.lower.detach().clone(),
            upper=None if self.upper is None else self.upper.detach().clone(),
            branch=self.branch,
            alpha_state_version=self.alpha_state_version,
            beta_state_version=self.beta_state_version,
            alpha_by_relu_input={
                name: value.detach().clone()
                for name, value in self.alpha_by_relu_input.items()
            },
            beta_by_relu_input={
                name: value.detach().clone()
                for name, value in self.beta_by_relu_input.items()
            },
        )


@dataclass(frozen=True)
class BoundQueryRequest:
    """One immutable query and its owned dynamic payload."""

    query: BoundQuery
    payload: BoundQueryPayload

    def detached_clone(self) -> "BoundQueryRequest":
        """Clone dynamic values while preserving the immutable query identity."""

        return BoundQueryRequest(
            query=self.query,
            payload=self.payload.detached_clone(),
        )


@dataclass(frozen=True)
class QueryBatch:
    """Compatibility-homogeneous logical batch used by PR-13B scheduling."""

    key: QueryCompatibilityKey
    requests: Tuple[BoundQueryRequest, ...]
    estimated_peak_bytes: int
    memory_budget_bytes: int

    def validate(self) -> None:
        """Reject mixed capability keys, duplicate IDs, and invalid budgets."""

        if not self.requests:
            raise ValueError("query batch must be non-empty")
        if self.estimated_peak_bytes < 0:
            raise ValueError("estimated_peak_bytes must be non-negative")
        if self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive")
        query_ids = [request.query.query_id for request in self.requests]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query batch contains duplicate query IDs")
        for request in self.requests:
            request.query.validate()
            if request.query.compatibility_key != self.key:
                raise ValueError(
                    f"incompatible query in batch: {request.query.query_id}"
                )

    @property
    def domain_batch_size(self) -> int:
        """Return the total leading input-domain batch size."""

        return sum(
            int(request.payload.input_spec.center.shape[0]) for request in self.requests
        )

    @property
    def spec_batch_size(self) -> int:
        """Return the common spec size, or one for implicit output specs."""

        request = self.requests[0]
        linear_spec_c = request.payload.linear_spec_c
        if linear_spec_c is None:
            return 1
        if linear_spec_c.dim() == 2:
            return int(linear_spec_c.shape[0])
        if linear_spec_c.dim() == 3:
            return int(linear_spec_c.shape[1])
        raise ValueError(
            f"linear_spec_c expects rank-2/3, got {tuple(linear_spec_c.shape)}"
        )


@dataclass
class BoundQueryTraceEntry:
    """One submitted call plus its eventual result."""

    query: BoundQuery
    payload: BoundQueryPayload
    result: Optional[BoundQueryResult] = None


class BabQueryRecorder(Protocol):
    """Observer contract used by the unchanged host-side BaB algorithm."""

    def submit(self, query: BoundQuery, payload: BoundQueryPayload) -> None:
        """Record one actual bound-oracle submission."""

    def complete(self, query_id: str, result: BoundQueryResult) -> None:
        """Attach exactly one terminal result to a submitted query."""


@dataclass
class FixedBabQueryRecorder:
    """In-memory ordered trace with duplicate/loss checks and JSONL export."""

    entries: list[BoundQueryTraceEntry] = field(default_factory=list)
    _by_id: dict[str, BoundQueryTraceEntry] = field(default_factory=dict, init=False)

    def submit(self, query: BoundQuery, payload: BoundQueryPayload) -> None:
        """Record exactly one actual oracle submission."""

        query.validate()
        if query.query_id in self._by_id:
            raise ValueError(f"duplicate query_id: {query.query_id}")
        entry = BoundQueryTraceEntry(query=query, payload=payload.detached_clone())
        self.entries.append(entry)
        self._by_id[query.query_id] = entry

    def complete(self, query_id: str, result: BoundQueryResult) -> None:
        """Attach exactly one result to a submitted query."""

        entry = self._by_id.get(query_id)
        if entry is None:
            raise KeyError(f"result for unknown query_id: {query_id}")
        if entry.result is not None:
            raise ValueError(f"duplicate result for query_id: {query_id}")
        entry.result = result.detached_clone()

    def validate_complete(self) -> None:
        """Prove no query loss, duplicate ID, or incomplete result."""

        if len(self.entries) != len(self._by_id):
            raise ValueError("query trace contains duplicate identities")
        incomplete = [
            entry.query.query_id for entry in self.entries if entry.result is None
        ]
        if incomplete:
            raise ValueError(f"query trace has incomplete results: {incomplete}")
        sequences = [entry.query.sequence_number for entry in self.entries]
        if sequences != list(range(len(sequences))):
            raise ValueError(f"query sequence is not contiguous: {sequences}")

    def write_jsonl(self, path: Path) -> None:
        """Write deterministic metadata/result summaries without tensor payload values."""

        self.validate_complete()
        lines = []
        for entry in self.entries:
            assert entry.result is not None
            lines.append(
                json.dumps(
                    {
                        "schema_version": QUERY_TRACE_SCHEMA_VERSION,
                        "query": entry.query.to_dict(),
                        "result": result_summary(entry.result),
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
        path.write_text("".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class ReplayComparison:  # pylint: disable=too-many-instance-attributes
    """Per-query correctness outcome for fixed-stream replay."""

    query_id: str
    status_match: bool
    branch_match: bool
    state_version_match: bool
    state_values_allclose: bool
    finite: bool
    ordered: bool
    allclose: bool
    max_abs_diff: float

    @property
    def passed(self) -> bool:
        """Return the full correctness conjunction."""

        return (
            self.status_match
            and self.branch_match
            # Physical batching may change floating-point reduction order.
            # Exact content hashes remain diagnostic, while numerical state
            # equality is the semantic correctness criterion.
            and self.state_values_allclose
            and self.finite
            and self.ordered
            and self.allclose
        )


class StateValidityManager:  # pylint: disable=too-few-public-methods
    """Central parent/child cache-reuse policy; no implicit parent reuse."""

    def classify(  # pylint: disable=too-many-return-statements
        self,
        source: BoundQuery,
        target: BoundQuery,
        artifact: StateArtifactKind,
    ) -> ReuseClass:
        """Classify reuse using identity, versions, split, and compatibility."""

        same_query = source.query_id == target.query_id
        same_model = (
            source.model_structure_hash == target.model_structure_hash
            and source.weight_version == target.weight_version
        )
        same_compatibility = source.compatibility_key == target.compatibility_key
        parent_child = target.parent_query_id == source.query_id
        same_split = source.split_signature == target.split_signature
        if not same_model:
            return ReuseClass.INVALIDATE
        if artifact in {
            StateArtifactKind.GRAPH_STRUCTURE,
            StateArtifactKind.WEIGHT_PREPROCESS,
        }:
            return ReuseClass.EXACT_REUSE
        if artifact == StateArtifactKind.COMPILED_MODULE:
            return (
                ReuseClass.EXACT_REUSE
                if same_compatibility
                else ReuseClass.CONDITIONAL_REUSE
            )
        if artifact in {
            StateArtifactKind.PLANNER_TEMPLATE,
            StateArtifactKind.COMPILE_PROFILE,
        }:
            return (
                ReuseClass.CONDITIONAL_REUSE
                if same_compatibility
                else ReuseClass.INVALIDATE
            )
        if artifact == StateArtifactKind.ALPHA_STATE:
            if same_query and source.alpha_state_version == target.alpha_state_version:
                return ReuseClass.EXACT_REUSE
            return ReuseClass.WARM_START_ONLY if parent_child else ReuseClass.INVALIDATE
        if artifact == StateArtifactKind.INTERMEDIATE_BOUNDS:
            return ReuseClass.WARM_START_ONLY if parent_child else ReuseClass.INVALIDATE
        if artifact == StateArtifactKind.BETA_STATE:
            return (
                ReuseClass.EXACT_REUSE
                if same_query
                and same_split
                and source.beta_state_version == target.beta_state_version
                else ReuseClass.INVALIDATE
            )
        if artifact == StateArtifactKind.CUTS:
            return (
                ReuseClass.CONDITIONAL_REUSE
                if source.cuts_version is not None
                and source.cuts_version == target.cuts_version
                and same_compatibility
                else ReuseClass.INVALIDATE
            )
        if artifact == StateArtifactKind.FINAL_BOUNDS:
            return (
                ReuseClass.EXACT_REUSE
                if same_query and same_split
                else ReuseClass.INVALIDATE
            )
        raise AssertionError(f"unhandled artifact kind: {artifact}")


def _hash_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(memoryview(value.numpy()).tobytes())
    return digest.hexdigest()


def _hash_tensor_mapping(values: Mapping[str, torch.Tensor]) -> Optional[str]:
    if not values:
        return None
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode("utf-8"))
        digest.update(_hash_tensor(values[name]).encode("utf-8"))
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def model_versions(module: BFTaskModule) -> tuple[str, str]:
    """Hash graph structure separately from parameter values."""

    module.validate()
    structure = {
        "entry_task_id": module.entry_task_id,
        "tasks": [
            {
                "task_id": task.task_id,
                "kind": task.kind.value,
                "inputs": task.input_values,
                "outputs": task.output_values,
                "ops": [
                    {
                        "op_type": op.op_type,
                        "name": op.name,
                        "inputs": op.inputs,
                        "outputs": op.outputs,
                        "attrs": _json_value(op.attrs),
                    }
                    for op in task.ops
                ],
            }
            for task in module.tasks
        ],
    }
    structure_hash = hashlib.sha256(
        json.dumps(structure, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    params = module.bindings.get("params", {})
    digest = hashlib.sha256()
    if isinstance(params, Mapping):
        for name in sorted(params):
            value = params[name]
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
            digest.update(str(name).encode("utf-8"))
            digest.update(_hash_tensor(tensor).encode("utf-8"))
    return structure_hash, digest.hexdigest()


def tensor_version(tensor: torch.Tensor) -> str:
    """Return a content version; callers should cache it outside hot loops."""

    return _hash_tensor(tensor)


def input_spec_version(input_spec: InputSpecLike) -> str:
    """Version one input region, including center and perturbation identity."""

    spec = _normalize_input_spec(input_spec)
    digest = hashlib.sha256()
    digest.update(_hash_tensor(spec.center).encode("utf-8"))
    digest.update(spec.perturbation.perturbation_id.encode("utf-8"))
    return digest.hexdigest()


def make_bound_query(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    module: BFTaskModule,
    query_id: str,
    parent_query_id: Optional[str],
    sequence_number: int,
    example_idx: int,
    input_spec: InputSpecLike,
    linear_spec_c: Optional[torch.Tensor],
    split_by_relu_input: Mapping[str, torch.Tensor],
    warm_alpha_by_relu_input: Mapping[str, torch.Tensor],
    warm_beta_by_relu_input: Mapping[str, torch.Tensor],
    bound_method: BoundMethod,
    execution_options: Mapping[str, object],
    identity_versions: Optional[QueryIdentityVersions] = None,
) -> tuple[BoundQuery, BoundQueryPayload]:
    """Build one immutable query and owned runtime payload."""

    spec = _normalize_input_spec(input_spec)
    versions = identity_versions or QueryIdentityVersions()
    if (
        versions.model_structure_hash is not None
        and versions.weight_version is not None
    ):
        structure_hash = versions.model_structure_hash
        weight_version = versions.weight_version
    else:
        computed_structure_hash, computed_weight_version = model_versions(module)
        structure_hash = versions.model_structure_hash or computed_structure_hash
        weight_version = versions.weight_version or computed_weight_version
    perturbation = spec.perturbation
    if isinstance(perturbation, LpBallPerturbation):
        perturbation_payload = {
            "kind": "lp_ball",
            "id": perturbation.perturbation_id,
            "eps": float(perturbation.eps),
        }
    elif isinstance(perturbation, BoxPerturbation):
        perturbation_payload = {
            "kind": "box",
            "id": perturbation.perturbation_id,
        }
    else:
        perturbation_payload = {"kind": type(perturbation).__name__}
    normalized_execution_options = {
        **dict(execution_options),
        "input_perturbation": perturbation_payload,
    }
    execution_options_hash = hashlib.sha256(
        json.dumps(
            _json_value(normalized_execution_options),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if versions.input_region_hash is not None:
        input_region_hash = versions.input_region_hash
    else:
        input_digest = hashlib.sha256()
        input_digest.update(_hash_tensor(spec.center).encode("utf-8"))
        input_digest.update(
            json.dumps(perturbation_payload, sort_keys=True).encode("utf-8")
        )
        input_region_hash = input_digest.hexdigest()
    output_spec_hash = (
        versions.output_spec_hash
        if versions.output_spec_hash is not None
        else "none" if linear_spec_c is None else _hash_tensor(linear_spec_c)
    )
    split_signature = (
        versions.split_signature
        if versions.split_signature is not None
        else _hash_tensor_mapping(split_by_relu_input) or "empty"
    )
    alpha_version = (
        versions.alpha_state_version
        if identity_versions is not None
        else _hash_tensor_mapping(warm_alpha_by_relu_input)
    )
    beta_version = (
        versions.beta_state_version
        if identity_versions is not None
        else _hash_tensor_mapping(warm_beta_by_relu_input)
    )
    split_shapes = tuple(
        (name, tuple(int(dim) for dim in split_by_relu_input[name].shape))
        for name in sorted(split_by_relu_input)
    )
    spec_shape = (
        () if linear_spec_c is None else tuple(int(dim) for dim in linear_spec_c.shape)
    )
    if bound_method == BoundMethod.ALPHA_BETA_CROWN:
        capability = "alpha_beta_dense_split"
    elif bound_method == BoundMethod.ALPHA_CROWN:
        capability = "alpha_dense"
    elif bound_method == BoundMethod.CROWN:
        capability = "plain_crown_typed_ir"
    else:
        raise ValueError(
            f"PR-13 query contract rejects unsupported method: {bound_method.value}"
        )
    plain_crown = bound_method == BoundMethod.CROWN
    optimization_stage = (
        OptimizationStage.FINAL_BOUND
        if plain_crown
        else OptimizationStage.BAB_NODE_EVAL
    )
    compatibility = QueryCompatibilityKey(
        model_structure_hash=structure_hash,
        weight_version=weight_version,
        bound_method=bound_method.value,
        optimization_stage=optimization_stage.value,
        requires_grad=not plain_crown,
        input_value_name=spec.value_name,
        input_shape=tuple(int(dim) for dim in spec.center.shape),
        spec_shape=spec_shape,
        split_tensor_shapes=split_shapes,
        dtype=str(spec.center.dtype),
        device=str(spec.center.device),
        perturbation_signature=json.dumps(
            perturbation_payload, sort_keys=True, separators=(",", ":")
        ),
        execution_options_hash=execution_options_hash,
        backend_capability_class=capability,
        numeric_policy=(
            "fp32_strict"
            if spec.center.dtype == torch.float32
            else str(spec.center.dtype)
        ),
    )
    query = BoundQuery(
        query_id=query_id,
        parent_query_id=parent_query_id,
        sequence_number=sequence_number,
        example_idx=example_idx,
        model_structure_hash=structure_hash,
        weight_version=weight_version,
        input_region_hash=input_region_hash,
        output_spec_hash=output_spec_hash,
        split_signature=split_signature,
        bound_method=bound_method,
        optimization_stage=optimization_stage,
        requires_grad=not plain_crown,
        alpha_state_version=alpha_version,
        beta_state_version=beta_version,
        cuts_version=None,
        dtype=str(spec.center.dtype),
        device=str(spec.center.device),
        numeric_policy=compatibility.numeric_policy,
        requested_outputs=(
            ("bounds",)
            if plain_crown
            else ("bounds", "alpha_state", "beta_state", "branch_hint")
        ),
        compatibility_key=compatibility,
        execution_options=normalized_execution_options,
    )
    query.validate()
    payload = BoundQueryPayload(
        input_spec=spec,
        linear_spec_c=linear_spec_c,
        split_by_relu_input=split_by_relu_input,
        warm_alpha_by_relu_input=warm_alpha_by_relu_input,
        warm_beta_by_relu_input=warm_beta_by_relu_input,
    ).detached_clone()
    return query, payload


def result_from_execution(  # pylint: disable=too-many-arguments
    bounds: Optional[IntervalState],
    alpha: Optional[AlphaState],
    beta: Optional[BetaState],
    branch: Optional[Tuple[str, int]],
    *,
    status: str = "ok",
    hash_state_versions: bool = True,
) -> BoundQueryResult:
    """Convert executor outputs to an owned trace result."""

    return BoundQueryResult(
        status=status,
        lower=None if bounds is None else bounds.lower,
        upper=None if bounds is None else bounds.upper,
        branch=branch,
        alpha_state_version=(
            None
            if alpha is None or not hash_state_versions
            else _hash_tensor_mapping(alpha.alpha_by_relu_input)
        ),
        beta_state_version=(
            None
            if beta is None or not hash_state_versions
            else _hash_tensor_mapping(beta.beta_by_relu_input)
        ),
        alpha_by_relu_input=({} if alpha is None else alpha.alpha_by_relu_input),
        beta_by_relu_input={} if beta is None else beta.beta_by_relu_input,
    ).detached_clone()


def require_int_option(options: Mapping[str, object], name: str) -> int:
    """Read one required integer execution option."""

    value = options.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"execution option {name} must be an int, got {value!r}")
    return value


def require_float_option(options: Mapping[str, object], name: str) -> float:
    """Read one required numeric execution option as ``float``."""

    value = options.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"execution option {name} must be numeric, got {value!r}")
    return float(value)


def require_str_option(options: Mapping[str, object], name: str) -> str:
    """Read one required string execution option."""

    value = options.get(name)
    if not isinstance(value, str):
        raise ValueError(f"execution option {name} must be a string, got {value!r}")
    return value


def _execute_bound_query_unchecked(
    module: BFTaskModule, query: BoundQuery, payload: BoundQueryPayload
) -> BoundQueryResult:
    """Replay one query while allowing executor rejection to propagate."""

    options = query.execution_options
    warm_alpha = (
        None
        if not payload.warm_alpha_by_relu_input
        else AlphaState(dict(payload.warm_alpha_by_relu_input))
    )
    if query.bound_method == BoundMethod.ALPHA_BETA_CROWN:
        warm_beta = (
            None
            if not payload.warm_beta_by_relu_input
            else BetaState(dict(payload.warm_beta_by_relu_input))
        )
        bounds, alpha, beta, stats = run_alpha_beta_crown_mlp(
            module,
            payload.input_spec,
            linear_spec_C=payload.linear_spec_c,
            relu_split_state=dict(payload.split_by_relu_input),
            steps=require_int_option(options, "alpha_steps"),
            lr=require_float_option(options, "alpha_lr"),
            alpha_init=require_float_option(options, "alpha_init"),
            beta_init=require_float_option(options, "beta_init"),
            warm_start_alpha=warm_alpha,
            warm_start_beta=warm_beta,
            objective=require_str_option(options, "objective"),  # type: ignore[arg-type]
            spec_reduce=require_str_option(options, "spec_reduce"),  # type: ignore[arg-type]
            soft_tau=require_float_option(options, "soft_tau"),
            lb_weight=require_float_option(options, "lb_weight"),
            ub_weight=require_float_option(options, "ub_weight"),
        )
        if stats.feasibility == "infeasible":
            return result_from_execution(None, alpha, beta, None, status="infeasible")
        branch = (
            stats.branch_choices[0]
            if stats.branch_choices and len(stats.branch_choices) == 1
            else None
        )
        return result_from_execution(bounds, alpha, beta, branch)
    if query.bound_method == BoundMethod.ALPHA_CROWN:
        bounds, alpha, _stats = run_alpha_crown_mlp(
            module,
            payload.input_spec,
            linear_spec_C=payload.linear_spec_c,
            steps=require_int_option(options, "alpha_steps"),
            lr=require_float_option(options, "alpha_lr"),
            alpha_init=require_float_option(options, "alpha_init"),
            objective=require_str_option(options, "objective"),  # type: ignore[arg-type]
            spec_reduce=require_str_option(options, "spec_reduce"),  # type: ignore[arg-type]
            soft_tau=require_float_option(options, "soft_tau"),
            lb_weight=require_float_option(options, "lb_weight"),
            ub_weight=require_float_option(options, "ub_weight"),
            warm_start=warm_alpha,
            relu_split_state=dict(payload.split_by_relu_input),
        )
        return result_from_execution(bounds, alpha, None, None)
    raise ValueError(f"unsupported BaB query method: {query.bound_method.value}")


def execute_bound_query(
    module: BFTaskModule, query: BoundQuery, payload: BoundQueryPayload
) -> BoundQueryResult:
    """Replay one query and preserve the solver's rejection semantics."""

    try:
        return _execute_bound_query_unchecked(module, query, payload)
    except ValueError:
        return BoundQueryResult(
            status="rejected",
            lower=None,
            upper=None,
            branch=None,
            alpha_state_version=None,
            beta_state_version=None,
        )


def _state_mapping_allclose(
    expected: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
    *,
    rtol: float,
    atol: float,
) -> bool:
    """Compare named solver state without requiring bitwise-identical hashes."""

    return set(expected) == set(actual) and all(
        torch.allclose(expected[name], actual[name], rtol=rtol, atol=atol)
        for name in expected
    )


def compare_query_results(
    query_id: str,
    expected: BoundQueryResult,
    actual: BoundQueryResult,
    *,
    rtol: float = 2e-4,
    atol: float = 2e-4,
) -> ReplayComparison:
    """Compare status/branch/bounds without hiding infeasible queries."""

    status_match = expected.status == actual.status
    branch_match = expected.branch == actual.branch
    state_version_match = (
        expected.alpha_state_version == actual.alpha_state_version
        and expected.beta_state_version == actual.beta_state_version
    )
    state_values_allclose = _state_mapping_allclose(
        expected.alpha_by_relu_input,
        actual.alpha_by_relu_input,
        rtol=rtol,
        atol=atol,
    ) and _state_mapping_allclose(
        expected.beta_by_relu_input,
        actual.beta_by_relu_input,
        rtol=rtol,
        atol=atol,
    )
    if expected.lower is None or expected.upper is None:
        empty_match = actual.lower is None and actual.upper is None
        return ReplayComparison(
            query_id=query_id,
            status_match=status_match,
            branch_match=branch_match,
            state_version_match=state_version_match,
            state_values_allclose=state_values_allclose,
            finite=empty_match,
            ordered=empty_match,
            allclose=empty_match,
            max_abs_diff=0.0 if empty_match else float("inf"),
        )
    if actual.lower is None or actual.upper is None:
        return ReplayComparison(
            query_id=query_id,
            status_match=status_match,
            branch_match=branch_match,
            state_version_match=state_version_match,
            state_values_allclose=state_values_allclose,
            finite=False,
            ordered=False,
            allclose=False,
            max_abs_diff=float("inf"),
        )
    finite = bool(
        torch.isfinite(actual.lower).all() and torch.isfinite(actual.upper).all()
    )
    ordered = bool((actual.lower <= actual.upper).all())
    lower_close = torch.allclose(actual.lower, expected.lower, rtol=rtol, atol=atol)
    upper_close = torch.allclose(actual.upper, expected.upper, rtol=rtol, atol=atol)
    max_abs = max(
        float((actual.lower - expected.lower).abs().max().item()),
        float((actual.upper - expected.upper).abs().max().item()),
    )
    return ReplayComparison(
        query_id=query_id,
        status_match=status_match,
        branch_match=branch_match,
        state_version_match=state_version_match,
        state_values_allclose=state_values_allclose,
        finite=finite,
        ordered=ordered,
        allclose=bool(lower_close and upper_close),
        max_abs_diff=max_abs,
    )


def replay_fixed_query_trace(
    module: BFTaskModule,
    entries: Sequence[BoundQueryTraceEntry],
) -> list[ReplayComparison]:
    """Replay in original order and return one comparison per query."""

    comparisons: list[ReplayComparison] = []
    for entry in entries:
        if entry.result is None:
            raise ValueError(f"query has no expected result: {entry.query.query_id}")
        actual = execute_bound_query(module, entry.query, entry.payload)
        comparisons.append(
            compare_query_results(entry.query.query_id, entry.result, actual)
        )
    return comparisons


def build_query_batch(
    requests: Sequence[BoundQueryRequest],
    *,
    estimated_peak_bytes: int,
    memory_budget_bytes: int,
) -> QueryBatch:
    """Build a logical batch without weakening exact compatibility equality."""

    if not requests:
        raise ValueError("cannot build an empty query batch")
    batch = QueryBatch(
        key=requests[0].query.compatibility_key,
        # Requests created by ``make_bound_query`` already own detached tensor
        # payloads.  QueryBatch is immutable, so cloning again here only adds
        # hot-path GPU copies and allocator pressure.
        requests=tuple(requests),
        estimated_peak_bytes=int(estimated_peak_bytes),
        memory_budget_bytes=int(memory_budget_bytes),
    )
    batch.validate()
    return batch


def execute_query_batch_reference(
    module: BFTaskModule,
    batch: QueryBatch,
) -> list[tuple[str, BoundQueryResult]]:
    """Execute a logical batch serially and preserve its declared order."""

    batch.validate()
    return [
        (
            request.query.query_id,
            execute_bound_query(module, request.query, request.payload),
        )
        for request in batch.requests
    ]


def result_summary(result: BoundQueryResult) -> dict[str, object]:
    """Return JSON-safe result metadata and hashes."""

    return {
        "status": result.status,
        "lower_hash": None if result.lower is None else _hash_tensor(result.lower),
        "upper_hash": None if result.upper is None else _hash_tensor(result.upper),
        "lower_shape": None if result.lower is None else list(result.lower.shape),
        "upper_shape": None if result.upper is None else list(result.upper.shape),
        "branch": None if result.branch is None else list(result.branch),
        "alpha_state_version": result.alpha_state_version,
        "beta_state_version": result.beta_state_version,
    }


def compatibility_groups(
    queries: Iterable[BoundQuery],
) -> dict[QueryCompatibilityKey, list[BoundQuery]]:
    """Group without weakening exact compatibility equality."""

    groups: dict[QueryCompatibilityKey, list[BoundQuery]] = {}
    for query in queries:
        query.validate()
        groups.setdefault(query.compatibility_key, []).append(query)
    return groups


__all__ = [
    "BabQueryRecorder",
    "BoundQuery",
    "BoundQueryPayload",
    "BoundQueryRequest",
    "BoundQueryResult",
    "BoundQueryTraceEntry",
    "FixedBabQueryRecorder",
    "QueryCompatibilityKey",
    "QueryIdentityVersions",
    "QueryBatch",
    "ReplayComparison",
    "ReuseClass",
    "StateArtifactKind",
    "StateValidityManager",
    "compatibility_groups",
    "build_query_batch",
    "compare_query_results",
    "execute_bound_query",
    "execute_query_batch_reference",
    "make_bound_query",
    "model_versions",
    "input_spec_version",
    "replay_fixed_query_trace",
    "require_float_option",
    "require_int_option",
    "require_str_option",
    "result_from_execution",
    "tensor_version",
]
