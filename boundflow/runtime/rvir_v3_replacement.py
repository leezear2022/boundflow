"""Executable RVIR-v3 payload and no-provider-callback replacement runtime."""

# pylint: disable=missing-function-docstring,too-few-public-methods
# pylint: disable=too-many-statements,too-many-locals,too-many-branches
# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Mapping, Protocol, Tuple

import torch

RVIR_V3_SCHEMA_VERSION = "boundflow.rvir-executable-payload/v3"


class VerifierPhase(str, Enum):
    """Solver phase whose bound call is being replaced."""

    INITIAL_CROWN = "initial_crown"
    ALPHA_OPTIMIZE = "alpha_optimize"
    BETA_SPLIT = "beta_split"


class VerifierTensorRole(str, Enum):
    """Executable tensor ownership role."""

    INPUT_LOWER = "input_lower"
    INPUT_UPPER = "input_upper"
    LINEAR_SPEC = "linear_spec"
    PROGRAM_WEIGHT = "program_weight"
    PROGRAM_BIAS = "program_bias"
    PROGRAM_PARAMETER = "program_parameter"
    INTERMEDIATE_LOWER = "intermediate_lower"
    INTERMEDIATE_UPPER = "intermediate_upper"
    ALPHA_STATE = "alpha_state"
    BETA_STATE = "beta_state"
    SPLIT_LOWER = "split_lower"
    SPLIT_UPPER = "split_upper"


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor type, shape, device-independent contiguous bytes."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class OwnedVerifierTensor:
    """One tensor whose content is owned and digest-bound by the payload."""

    tensor_id: str
    role: VerifierTensorRole
    value: torch.Tensor
    content_sha256: str

    @classmethod
    def own(
        cls, tensor_id: str, role: VerifierTensorRole, value: torch.Tensor
    ) -> "OwnedVerifierTensor":
        if not torch.is_tensor(value):
            raise TypeError("RVIR-v3 owned value must be a tensor")
        owned = value.detach().clone().contiguous()
        return cls(tensor_id, role, owned, tensor_sha256(owned))

    def validate(self) -> None:
        if not self.tensor_id:
            raise ValueError("RVIR-v3 tensor ID must be non-empty")
        if not torch.is_tensor(self.value):
            raise TypeError("RVIR-v3 owned value must remain a tensor")
        if self.content_sha256 != tensor_sha256(self.value):
            raise ValueError("RVIR-v3 owned tensor content differs")
        if self.value.is_floating_point() and not bool(
            torch.isfinite(self.value).all()
        ):
            raise ValueError("RVIR-v3 owned tensor must be finite")

    def metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "tensor_id": self.tensor_id,
            "role": self.role.value,
            "shape": list(self.value.shape),
            "dtype": str(self.value.dtype),
            "device": str(self.value.device),
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class DomainSlice:
    """One contiguous ragged-domain slice."""

    slice_id: str
    start: int
    end: int

    def validate(self) -> None:
        if not self.slice_id or self.start < 0 or self.end <= self.start:
            raise ValueError("RVIR-v3 domain slice differs")


@dataclass(frozen=True)
class ExecutableVerifierPayload:  # pylint: disable=too-many-instance-attributes
    """Self-contained executable identity and tensors for one replacement call."""

    query_id: str
    sequence_number: int
    parent_query_id: str | None
    phase: VerifierPhase
    method: str
    requested_polarities: Tuple[str, ...]
    tensors: Tuple[OwnedVerifierTensor, ...]
    expected_result_shape: Tuple[int, ...]
    ragged_slices: Tuple[DomainSlice, ...] = ()
    mutable_state_ids: Tuple[str, ...] = ()
    copy_out_state_ids: Tuple[str, ...] = ()
    numeric_policy: str = "fp32_strict"
    schema_version: str = RVIR_V3_SCHEMA_VERSION

    def tensor_map(self) -> dict[str, OwnedVerifierTensor]:
        return {item.tensor_id: item for item in self.tensors}

    def tensors_with_role(
        self, role: VerifierTensorRole
    ) -> Tuple[OwnedVerifierTensor, ...]:
        return tuple(item for item in self.tensors if item.role == role)

    def one_tensor(self, role: VerifierTensorRole) -> OwnedVerifierTensor:
        values = self.tensors_with_role(role)
        if len(values) != 1:
            raise ValueError(f"RVIR-v3 requires one {role.value} tensor")
        return values[0]

    def validate(self) -> None:  # pylint: disable=too-many-branches
        if self.schema_version != RVIR_V3_SCHEMA_VERSION:
            raise ValueError("RVIR-v3 payload schema differs")
        if not self.query_id or not self.method or not self.numeric_policy:
            raise ValueError("RVIR-v3 payload identity is incomplete")
        if self.sequence_number < 0:
            raise ValueError("RVIR-v3 sequence number is negative")
        if self.parent_query_id is not None and not self.parent_query_id:
            raise ValueError("RVIR-v3 parent query ID is empty")
        if self.requested_polarities not in {
            ("lower",),
            ("upper",),
            ("lower", "upper"),
        }:
            raise ValueError("RVIR-v3 requested polarity differs")
        if not self.expected_result_shape or any(
            dimension <= 0 for dimension in self.expected_result_shape
        ):
            raise ValueError("RVIR-v3 expected result shape differs")
        if not self.tensors:
            raise ValueError("RVIR-v3 executable tensors are empty")
        for item in self.tensors:
            item.validate()
        tensor_ids = tuple(item.tensor_id for item in self.tensors)
        if len(set(tensor_ids)) != len(tensor_ids):
            raise ValueError("RVIR-v3 tensor IDs duplicate")
        for required_role in (
            VerifierTensorRole.INPUT_LOWER,
            VerifierTensorRole.INPUT_UPPER,
            VerifierTensorRole.LINEAR_SPEC,
        ):
            self.one_tensor(required_role)
        lower = self.one_tensor(VerifierTensorRole.INPUT_LOWER).value
        upper = self.one_tensor(VerifierTensorRole.INPUT_UPPER).value
        linear_spec = self.one_tensor(VerifierTensorRole.LINEAR_SPEC).value
        shape_checks = (
            lower.shape == upper.shape,
            lower.ndim >= 2,
            linear_spec.ndim >= 2,
            tuple(self.expected_result_shape)
            == (lower.shape[0], linear_spec.shape[-2]),
        )
        if not all(shape_checks):
            raise ValueError("RVIR-v3 executable tensor shapes differ")
        if linear_spec.ndim >= 3 and linear_spec.shape[0] not in {1, lower.shape[0]}:
            raise ValueError("RVIR-v3 spec batch differs")
        if lower.dtype != upper.dtype or lower.device != upper.device:
            raise ValueError("RVIR-v3 input bound tensor types differ")
        if not bool((lower <= upper).all()):
            raise ValueError("RVIR-v3 input lower exceeds upper")
        intermediate_lowers = self.tensors_with_role(
            VerifierTensorRole.INTERMEDIATE_LOWER
        )
        intermediate_uppers = self.tensors_with_role(
            VerifierTensorRole.INTERMEDIATE_UPPER
        )
        if len(intermediate_lowers) != len(intermediate_uppers) or any(
            lower_item.value.shape != upper_item.value.shape
            for lower_item, upper_item in zip(intermediate_lowers, intermediate_uppers)
        ):
            raise ValueError("RVIR-v3 intermediate-bound tensors differ")
        roles = {item.role for item in self.tensors}
        if self.phase == VerifierPhase.ALPHA_OPTIMIZE and (
            VerifierTensorRole.ALPHA_STATE not in roles
        ):
            raise ValueError("RVIR-v3 alpha phase omits alpha state")
        if (
            self.phase == VerifierPhase.BETA_SPLIT
            and not {
                VerifierTensorRole.ALPHA_STATE,
                VerifierTensorRole.BETA_STATE,
                VerifierTensorRole.SPLIT_LOWER,
                VerifierTensorRole.SPLIT_UPPER,
            }
            <= roles
        ):
            raise ValueError("RVIR-v3 beta phase omits executable state")
        state_roles = {
            VerifierTensorRole.ALPHA_STATE,
            VerifierTensorRole.BETA_STATE,
        }
        state_ids = {
            item.tensor_id for item in self.tensors if item.role in state_roles
        }
        if (
            len(set(self.mutable_state_ids)) != len(self.mutable_state_ids)
            or not set(self.mutable_state_ids) <= state_ids
            or len(set(self.copy_out_state_ids)) != len(self.copy_out_state_ids)
            or not set(self.copy_out_state_ids) <= set(self.mutable_state_ids)
        ):
            raise ValueError("RVIR-v3 mutation ownership differs")
        if self.ragged_slices:
            cursor = 0
            seen: set[str] = set()
            for domain_slice in self.ragged_slices:
                domain_slice.validate()
                if domain_slice.slice_id in seen or domain_slice.start != cursor:
                    raise ValueError("RVIR-v3 ragged slices have a gap or overlap")
                seen.add(domain_slice.slice_id)
                cursor = domain_slice.end
            if cursor != lower.shape[0]:
                raise ValueError("RVIR-v3 ragged slices do not cover the domain batch")

    def stable_hash(self) -> str:
        self.validate()
        payload = {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            "sequence_number": self.sequence_number,
            "parent_query_id": self.parent_query_id,
            "phase": self.phase.value,
            "method": self.method,
            "requested_polarities": list(self.requested_polarities),
            "tensors": [item.metadata() for item in self.tensors],
            "expected_result_shape": list(self.expected_result_shape),
            "ragged_slices": [item.__dict__ for item in self.ragged_slices],
            "mutable_state_ids": list(self.mutable_state_ids),
            "copy_out_state_ids": list(self.copy_out_state_ids),
            "numeric_policy": self.numeric_policy,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReplacementStateUpdate:
    """One backend-produced state value, applied only after result validation."""

    tensor_id: str
    value: torch.Tensor


@dataclass(frozen=True)
class ReplacementBackendResult:
    """Raw result from an independent BoundFlow replacement backend."""

    lower: torch.Tensor | None
    upper: torch.Tensor | None
    state_updates: Tuple[ReplacementStateUpdate, ...] = ()


class BoundFlowVerifierReplacementBackend(Protocol):
    """Backend API intentionally excludes any original-provider callable."""

    backend_id: str

    def execute(
        self,
        payload: ExecutableVerifierPayload,
        tensors: Mapping[str, torch.Tensor],
    ) -> ReplacementBackendResult: ...


@dataclass(frozen=True)
class StateMutationReceipt:
    """Digest-bound copy-in/copy-out evidence for one mutable state tensor."""

    tensor_id: str
    before_sha256: str
    after_sha256: str
    copied_out: bool


@dataclass(frozen=True)
class ReplacementExecution:
    """Validated replacement result and no-fallback execution receipt."""

    query_id: str
    sequence_number: int
    backend_id: str
    lower: torch.Tensor | None
    upper: torch.Tensor | None
    payload_hash: str
    result_hash: str
    mutations: Tuple[StateMutationReceipt, ...]
    replacement_dispatch_count: int = 1
    original_callback_count: int = 0
    fallback_dispatch_count: int = 0
    performance_claimed: bool = False


class TorchAffineRVIRV3Backend:
    """Independent Torch affine interval/reference backend for the v3 contract."""

    backend_id = "boundflow.torch-affine-rvir-v3/v1"

    def execute(
        self,
        payload: ExecutableVerifierPayload,
        tensors: Mapping[str, torch.Tensor],
    ) -> ReplacementBackendResult:
        weight_items = payload.tensors_with_role(VerifierTensorRole.PROGRAM_WEIGHT)
        bias_items = payload.tensors_with_role(VerifierTensorRole.PROGRAM_BIAS)
        if len(weight_items) != 1 or len(bias_items) != 1:
            raise ValueError("RVIR-v3 affine backend program differs")
        by_role = {item.role: tensors[item.tensor_id] for item in payload.tensors}
        lower = by_role[VerifierTensorRole.INPUT_LOWER]
        upper = by_role[VerifierTensorRole.INPUT_UPPER]
        if payload.phase == VerifierPhase.BETA_SPLIT:
            lower = torch.maximum(lower, by_role[VerifierTensorRole.SPLIT_LOWER])
            upper = torch.minimum(upper, by_role[VerifierTensorRole.SPLIT_UPPER])
            if not bool((lower <= upper).all()):
                raise ValueError("RVIR-v3 split state creates an empty domain")
        weight = by_role[VerifierTensorRole.PROGRAM_WEIGHT]
        bias = by_role[VerifierTensorRole.PROGRAM_BIAS]
        linear_spec = by_role[VerifierTensorRole.LINEAR_SPEC]
        if (
            lower.ndim != 2
            or weight.ndim != 2
            or bias.ndim != 1
            or weight.shape != (bias.shape[0], lower.shape[1])
            or linear_spec.shape[-1] != weight.shape[0]
        ):
            raise ValueError("RVIR-v3 affine backend tensor shapes differ")
        positive_weight = weight.clamp(min=0)
        negative_weight = weight.clamp(max=0)
        output_lower = lower @ positive_weight.T + upper @ negative_weight.T + bias
        output_upper = upper @ positive_weight.T + lower @ negative_weight.T + bias
        if linear_spec.ndim == 2:
            linear_spec = linear_spec.unsqueeze(0).expand(lower.shape[0], -1, -1)
        elif linear_spec.shape[0] == 1:
            linear_spec = linear_spec.expand(lower.shape[0], -1, -1)
        positive_spec = linear_spec.clamp(min=0)
        negative_spec = linear_spec.clamp(max=0)
        bound_lower = (
            positive_spec * output_lower.unsqueeze(1)
            + negative_spec * output_upper.unsqueeze(1)
        ).sum(dim=-1)
        bound_upper = (
            positive_spec * output_upper.unsqueeze(1)
            + negative_spec * output_lower.unsqueeze(1)
        ).sum(dim=-1)
        return ReplacementBackendResult(
            lower=bound_lower if "lower" in payload.requested_polarities else None,
            upper=bound_upper if "upper" in payload.requested_polarities else None,
        )


def _result_hash(lower: torch.Tensor | None, upper: torch.Tensor | None) -> str:
    digest = hashlib.sha256()
    for name, value in (("lower", lower), ("upper", upper)):
        digest.update(name.encode("utf-8"))
        digest.update(
            b"none" if value is None else tensor_sha256(value).encode("utf-8")
        )
    return digest.hexdigest()


def execute_rvir_v3_replacement(  # pylint: disable=too-many-locals
    payload: ExecutableVerifierPayload,
    backend: BoundFlowVerifierReplacementBackend,
    *,
    copy_out_targets: Mapping[str, torch.Tensor] | None = None,
) -> ReplacementExecution:
    """Execute exactly one independent replacement and atomically commit state."""

    payload.validate()
    backend_id = getattr(backend, "backend_id", "")
    if (
        not isinstance(backend_id, str)
        or not backend_id.startswith("boundflow.")
        or "external" in backend_id
        or "provider" in backend_id
    ):
        raise ValueError("RVIR-v3 replacement backend identity differs")
    owned = payload.tensor_map()
    workspace = {
        tensor_id: item.value.detach().clone().contiguous()
        for tensor_id, item in owned.items()
    }
    targets = dict(copy_out_targets or {})
    if set(targets) != set(payload.copy_out_state_ids):
        raise ValueError("RVIR-v3 copy-out targets differ")
    for tensor_id, target in targets.items():
        source = owned[tensor_id]
        if (
            target.shape != source.value.shape
            or target.dtype != source.value.dtype
            or target.device != source.value.device
            or tensor_sha256(target) != source.content_sha256
        ):
            raise ValueError("RVIR-v3 live copy-out state differs from captured state")
    raw = backend.execute(payload, workspace)
    if not isinstance(raw, ReplacementBackendResult):
        raise TypeError("RVIR-v3 backend returned an invalid result")
    expected = tuple(payload.expected_result_shape)
    for polarity, value in (("lower", raw.lower), ("upper", raw.upper)):
        requested = polarity in payload.requested_polarities
        if requested != (value is not None):
            raise ValueError("RVIR-v3 backend result polarity differs")
        if value is not None and (
            tuple(value.shape) != expected
            or not torch.is_floating_point(value)
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("RVIR-v3 backend result tensor differs")
    if (
        raw.lower is not None
        and raw.upper is not None
        and not bool((raw.lower <= raw.upper).all())
    ):
        raise ValueError("RVIR-v3 backend lower exceeds upper")
    updates = {item.tensor_id: item.value for item in raw.state_updates}
    if len(updates) != len(raw.state_updates) or not set(updates) <= set(
        payload.mutable_state_ids
    ):
        raise ValueError("RVIR-v3 backend produced an undeclared state mutation")
    for tensor_id, update_value in updates.items():
        source_tensor = owned[tensor_id].value
        if (
            update_value.shape != source_tensor.shape
            or update_value.dtype != source_tensor.dtype
            or update_value.device != source_tensor.device
            or (
                update_value.is_floating_point()
                and not bool(torch.isfinite(update_value).all())
            )
        ):
            raise ValueError("RVIR-v3 state update tensor differs")
    if set(updates) != set(payload.copy_out_state_ids):
        raise ValueError("RVIR-v3 backend copy-out mutation coverage differs")
    for tensor_id, target in targets.items():
        target.copy_(updates[tensor_id])
    receipts = tuple(
        StateMutationReceipt(
            tensor_id=tensor_id,
            before_sha256=owned[tensor_id].content_sha256,
            after_sha256=tensor_sha256(updates.get(tensor_id, workspace[tensor_id])),
            copied_out=tensor_id in targets,
        )
        for tensor_id in payload.mutable_state_ids
    )
    lower = None if raw.lower is None else raw.lower.detach().clone().contiguous()
    upper = None if raw.upper is None else raw.upper.detach().clone().contiguous()
    return ReplacementExecution(
        query_id=payload.query_id,
        sequence_number=payload.sequence_number,
        backend_id=backend_id,
        lower=lower,
        upper=upper,
        payload_hash=payload.stable_hash(),
        result_hash=_result_hash(lower, upper),
        mutations=receipts,
    )


__all__ = [
    "RVIR_V3_SCHEMA_VERSION",
    "BoundFlowVerifierReplacementBackend",
    "DomainSlice",
    "ExecutableVerifierPayload",
    "OwnedVerifierTensor",
    "ReplacementBackendResult",
    "ReplacementExecution",
    "ReplacementStateUpdate",
    "StateMutationReceipt",
    "TorchAffineRVIRV3Backend",
    "VerifierPhase",
    "VerifierTensorRole",
    "execute_rvir_v3_replacement",
    "tensor_sha256",
]
