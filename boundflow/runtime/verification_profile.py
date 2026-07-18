"""Coverage-first profiles derived from the PR-13 bound-query contract."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from ..ir.task import BFTaskModule
from ..planner.execution_candidate import (
    BackendCapability,
    ExecutionContext,
    OperatorFamily,
    capability_rejections,
    fused_tir_conv_v1_capability,
    fused_tir_linear_v1_capability,
)
from ..planner.materialization import BoundMethod
from .bab_query import BoundQuery

VERIFICATION_QUERY_PROFILE_SCHEMA_VERSION = "boundflow.verification-query-profile/v1"
VERIFICATION_COVERAGE_REPORT_SCHEMA_VERSION = "boundflow.verification-coverage/v1"


def module_layer_pattern(module: BFTaskModule) -> Tuple[str, ...]:
    """Return the stable operation sequence seen by the BoundFlow task module."""

    module.validate()
    return tuple(op.op_type for task in module.tasks for op in task.ops)


def _normalize_dtype(dtype: str) -> str:
    return dtype.removeprefix("torch.").lower()


def _normalize_device(device: str) -> str:
    return device.split(":", maxsplit=1)[0].lower()


def _candidate_families(layer_pattern: Sequence[str]) -> Tuple[OperatorFamily, ...]:
    """Find affine/ReLU regions implemented by the current fused candidates."""

    families: list[OperatorFamily] = []
    for affine, following in zip(layer_pattern, layer_pattern[1:]):
        if following != "relu":
            continue
        if affine == "linear" and OperatorFamily.LINEAR not in families:
            families.append(OperatorFamily.LINEAR)
        if affine == "conv2d" and OperatorFamily.CONV2D not in families:
            families.append(OperatorFamily.CONV2D)
    return tuple(families)


def _spec_size(spec_shape: Sequence[int]) -> int:
    if not spec_shape:
        return 1
    if len(spec_shape) >= 3:
        return int(spec_shape[1])
    return int(spec_shape[0])


def _domain_size(input_shape: Sequence[int]) -> int:
    return int(input_shape[0]) if input_shape else 1


def _query_flags(query: BoundQuery) -> tuple[bool, bool, bool]:
    alpha_enabled = query.bound_method in {
        BoundMethod.ALPHA_FORWARD,
        BoundMethod.ALPHA_CROWN,
        BoundMethod.ALPHA_BETA_CROWN,
    }
    beta_enabled = query.bound_method == BoundMethod.ALPHA_BETA_CROWN
    external_split = query.execution_options.get("split_state_present")
    if isinstance(external_split, bool):
        split_state = external_split
    else:
        split_state = bool(query.compatibility_key.split_tensor_shapes)
    return alpha_enabled, beta_enabled, split_state


def _eligibility(
    query: BoundQuery,
    layer_pattern: Sequence[str],
    capabilities: Sequence[BackendCapability],
) -> tuple[bool, Tuple[str, ...], Tuple[str, ...]]:
    families = _candidate_families(layer_pattern)
    if not families:
        return False, ("no_fused_affine_relu_region",), ()

    alpha_enabled, beta_enabled, split_state = _query_flags(query)
    eligible_ids: list[str] = []
    rejected: list[str] = []
    for family in families:
        layout = "nchw" if family == OperatorFamily.CONV2D else "contiguous"
        context = ExecutionContext(
            bound_method=query.bound_method,
            requires_grad=query.requires_grad,
            optimization_stage=query.optimization_stage,
            alpha_enabled=alpha_enabled,
            beta_enabled=beta_enabled,
            split_state_present=split_state,
            operator_family=family,
            device=_normalize_device(query.device),
            dtype=_normalize_dtype(query.dtype),
            layout=layout,
            static_shape=all(
                int(dim) > 0
                for dim in (
                    *query.compatibility_key.input_shape,
                    *query.compatibility_key.spec_shape,
                )
            ),
        )
        matching_capabilities = [
            capability
            for capability in capabilities
            if (family == OperatorFamily.LINEAR and capability.supports_linear)
            or (family == OperatorFamily.CONV2D and capability.supports_conv2d)
        ]
        if not matching_capabilities:
            rejected.append(f"{family.value}_capability_unavailable")
            continue
        for capability in matching_capabilities:
            reasons = capability_rejections(context, capability)
            if reasons:
                rejected.extend(reasons)
            else:
                eligible_ids.append(capability.capability_id)

    if eligible_ids:
        return True, (), tuple(dict.fromkeys(eligible_ids))
    return False, tuple(dict.fromkeys(rejected)), ()


@dataclass(frozen=True)
class VerificationQueryProfile:  # pylint: disable=too-many-instance-attributes
    """One coverage row; the identity remains owned by :class:`BoundQuery`."""

    query_id: str
    solver_phase: str
    bound_method: str
    requires_grad: bool
    alpha_enabled: bool
    beta_enabled: bool
    split_state: bool
    spec_size: int
    domain_size: int
    layer_pattern: Tuple[str, ...]
    backend_eligible: bool
    reason_if_not: Tuple[str, ...]
    eligible_capability_ids: Tuple[str, ...]
    parent_query_id: str | None
    sequence_number: int
    source: str
    schema_version: str = VERIFICATION_QUERY_PROFILE_SCHEMA_VERSION

    @classmethod
    def from_bound_query(  # pylint: disable=too-many-arguments
        cls,
        query: BoundQuery,
        *,
        solver_phase: str,
        layer_pattern: Sequence[str],
        source: str = "boundflow",
        capabilities: Sequence[BackendCapability] | None = None,
        precondition_rejections: Sequence[str] = (),
    ) -> "VerificationQueryProfile":
        """Project an existing query onto coverage and capability fields."""

        query.validate()
        if not solver_phase:
            raise ValueError("solver_phase must be non-empty")
        if not layer_pattern:
            raise ValueError("layer_pattern must be non-empty")
        active_capabilities = (
            tuple(capabilities)
            if capabilities is not None
            else (
                fused_tir_linear_v1_capability(),
                fused_tir_conv_v1_capability(),
            )
        )
        eligible, reasons, capability_ids = _eligibility(
            query, layer_pattern, active_capabilities
        )
        if precondition_rejections:
            eligible = False
            reasons = tuple(dict.fromkeys((*precondition_rejections, *reasons)))
            capability_ids = ()
        alpha_enabled, beta_enabled, split_state = _query_flags(query)
        profile = cls(
            query_id=query.query_id,
            solver_phase=solver_phase,
            bound_method=query.bound_method.value,
            requires_grad=query.requires_grad,
            alpha_enabled=alpha_enabled,
            beta_enabled=beta_enabled,
            split_state=split_state,
            spec_size=_spec_size(query.compatibility_key.spec_shape),
            domain_size=_domain_size(query.compatibility_key.input_shape),
            layer_pattern=tuple(layer_pattern),
            backend_eligible=eligible,
            reason_if_not=reasons,
            eligible_capability_ids=capability_ids,
            parent_query_id=query.parent_query_id,
            sequence_number=query.sequence_number,
            source=source,
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        """Reject incomplete or internally inconsistent coverage records."""

        if self.schema_version != VERIFICATION_QUERY_PROFILE_SCHEMA_VERSION:
            raise ValueError(f"unsupported profile schema: {self.schema_version}")
        for name in ("query_id", "solver_phase", "bound_method", "source"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if not self.layer_pattern:
            raise ValueError("layer_pattern must be non-empty")
        if self.spec_size <= 0 or self.domain_size <= 0:
            raise ValueError("spec_size and domain_size must be positive")
        if self.backend_eligible and self.reason_if_not:
            raise ValueError("eligible profile cannot contain rejection reasons")
        if self.backend_eligible and not self.eligible_capability_ids:
            raise ValueError("eligible profile must name at least one capability")
        if not self.backend_eligible and not self.reason_if_not:
            raise ValueError("ineligible profile must explain why")

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-compatible fields."""

        self.validate()
        payload = asdict(self)
        payload["layer_pattern"] = list(self.layer_pattern)
        payload["reason_if_not"] = list(self.reason_if_not)
        payload["eligible_capability_ids"] = list(self.eligible_capability_ids)
        return payload


@dataclass(frozen=True)
class VerificationCoverageReport:
    """Aggregate coverage without dropping unsupported calls."""

    total_queries: int
    eligible_queries: int
    eligible_percent: float
    by_solver_phase: dict[str, int]
    by_bound_method: dict[str, int]
    rejection_reasons: dict[str, int]
    schema_version: str = VERIFICATION_COVERAGE_REPORT_SCHEMA_VERSION

    @classmethod
    def from_profiles(
        cls, profiles: Iterable[VerificationQueryProfile]
    ) -> "VerificationCoverageReport":
        """Count all profiles and preserve every rejection category."""

        rows = list(profiles)
        for profile in rows:
            profile.validate()
        total = len(rows)
        eligible = sum(profile.backend_eligible for profile in rows)
        return cls(
            total_queries=total,
            eligible_queries=eligible,
            eligible_percent=(0.0 if total == 0 else 100.0 * eligible / total),
            by_solver_phase=dict(
                sorted(Counter(profile.solver_phase for profile in rows).items())
            ),
            by_bound_method=dict(
                sorted(Counter(profile.bound_method for profile in rows).items())
            ),
            rejection_reasons=dict(
                sorted(
                    Counter(
                        reason for profile in rows for reason in profile.reason_if_not
                    ).items()
                )
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible aggregate fields."""

        return asdict(self)


def write_verification_profiles_jsonl(
    path: Path, profiles: Iterable[VerificationQueryProfile]
) -> None:
    """Write a stable profile-only artifact; tensor payloads stay elsewhere."""

    rows = list(profiles)
    lines = [
        json.dumps(profile.to_dict(), sort_keys=True, allow_nan=False) + "\n"
        for profile in rows
    ]
    path.write_text("".join(lines), encoding="utf-8")


__all__ = [
    "VERIFICATION_COVERAGE_REPORT_SCHEMA_VERSION",
    "VERIFICATION_QUERY_PROFILE_SCHEMA_VERSION",
    "VerificationCoverageReport",
    "VerificationQueryProfile",
    "module_layer_pattern",
    "write_verification_profiles_jsonl",
]
