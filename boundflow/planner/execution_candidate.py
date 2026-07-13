"""Backend-aware execution candidates introduced after the frozen PR-11 planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Tuple

from .materialization import BoundMethod, OptimizationStage

BACKEND_CANDIDATE_SCHEMA_VERSION = "boundflow.backend_candidate/v1.0"
BACKEND_PROFILE_SCHEMA_VERSION = "boundflow.backend_profile/v2.0"


class PlacementKind(Enum):
    """Logical coefficient representation, independent from its executor."""

    DENSE = "dense"
    STRUCTURED = "structured"


class BackendVariant(Enum):
    """Execution implementations that can realize a placement."""

    PYTORCH_EAGER = "pytorch_eager"
    TORCH_COMPILE = "torch_compile"
    TVM_RELAX_UNFUSED = "tvm_relax_unfused"
    TVM_TIR_DEFAULT = "tvm_tir_default"
    TVM_FUSED_TIR = "tvm_fused_tir"


class OperatorFamily(Enum):
    """Affine operator family adjacent to the ReLU backward task."""

    LINEAR = "linear"
    CONV2D = "conv2d"


@dataclass(frozen=True)
class BackendCapability:  # pylint: disable=too-many-instance-attributes
    """Machine-checkable legality contract for one backend capability id."""

    capability_id: str
    supported_bound_methods: Tuple[BoundMethod, ...]
    supports_grad: bool
    supports_alpha: bool
    supports_beta: bool
    supports_split_state: bool
    supports_linear: bool
    supports_conv2d: bool
    supported_dtypes: Tuple[str, ...]
    supported_layouts: Tuple[str, ...]
    supported_devices: Tuple[str, ...]
    supported_optimization_stages: Tuple[OptimizationStage, ...]
    static_shape_only: bool

    def validate(self) -> None:
        """Reject incomplete capabilities before candidate filtering."""

        if not self.capability_id:
            raise ValueError("capability_id must be non-empty")
        for name in (
            "supported_dtypes",
            "supported_layouts",
            "supported_devices",
            "supported_optimization_stages",
            "supported_bound_methods",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if not self.supports_linear and not self.supports_conv2d:
            raise ValueError("capability must support at least one operator family")


@dataclass(frozen=True)
class ExecutionContext:  # pylint: disable=too-many-instance-attributes
    """Dynamic query properties used exclusively for capability filtering."""

    bound_method: BoundMethod
    requires_grad: bool
    optimization_stage: OptimizationStage
    alpha_enabled: bool
    beta_enabled: bool
    split_state_present: bool
    operator_family: OperatorFamily
    device: str
    dtype: str
    layout: str
    static_shape: bool


@dataclass(frozen=True)
class ExecutionCandidate:  # pylint: disable=too-many-instance-attributes
    """One placement/backend pair exposed to post-PR-11 planning."""

    placement: PlacementKind
    backend: BackendVariant
    domain_batch_size: int
    spec_batch_size: int
    materialization_points: Tuple[str, ...]
    capability_id: str
    schedule_id: str
    reason: str
    schema_version: str = BACKEND_CANDIDATE_SCHEMA_VERSION

    def validate(self) -> None:
        """Validate stable identity and query dimensions."""

        if self.schema_version != BACKEND_CANDIDATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported candidate schema: {self.schema_version}")
        if self.domain_batch_size <= 0 or self.spec_batch_size <= 0:
            raise ValueError("domain/spec batch sizes must be positive")
        for name in ("capability_id", "schedule_id", "reason"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible candidate dump."""

        self.validate()
        payload = asdict(self)
        payload["placement"] = self.placement.value
        payload["backend"] = self.backend.value
        payload["materialization_points"] = list(self.materialization_points)
        return payload


def capability_rejections(
    context: ExecutionContext, capability: BackendCapability
) -> Tuple[str, ...]:
    """Return explicit reasons why a query cannot use a backend capability."""

    capability.validate()
    reasons: list[str] = []
    if context.bound_method not in capability.supported_bound_methods:
        reasons.append("bound_method_unsupported")
    if context.requires_grad and not capability.supports_grad:
        reasons.append("requires_grad_unsupported")
    if context.alpha_enabled and not capability.supports_alpha:
        reasons.append("alpha_unsupported")
    if context.beta_enabled and not capability.supports_beta:
        reasons.append("beta_unsupported")
    if context.split_state_present and not capability.supports_split_state:
        reasons.append("split_state_unsupported")
    if (
        context.operator_family == OperatorFamily.LINEAR
        and not capability.supports_linear
    ):
        reasons.append("linear_unsupported")
    if (
        context.operator_family == OperatorFamily.CONV2D
        and not capability.supports_conv2d
    ):
        reasons.append("conv2d_unsupported")
    if context.dtype not in capability.supported_dtypes:
        reasons.append("dtype_unsupported")
    if context.layout not in capability.supported_layouts:
        reasons.append("layout_unsupported")
    if context.device not in capability.supported_devices:
        reasons.append("device_unsupported")
    if context.optimization_stage not in capability.supported_optimization_stages:
        reasons.append("optimization_stage_unsupported")
    if capability.static_shape_only and not context.static_shape:
        reasons.append("dynamic_shape_unsupported")
    return tuple(reasons)


def fused_tir_linear_v1_capability() -> BackendCapability:
    """Return the capability actually implemented by the first PR-12 slice."""

    return BackendCapability(
        capability_id="tvm_fused_tir_linear_plain_crown_fp32_static_v1",
        supported_bound_methods=(BoundMethod.CROWN,),
        supports_grad=False,
        supports_alpha=False,
        supports_beta=False,
        supports_split_state=False,
        supports_linear=True,
        supports_conv2d=False,
        supported_dtypes=("float32",),
        supported_layouts=("contiguous", "nchw"),
        supported_devices=("cuda",),
        supported_optimization_stages=(
            OptimizationStage.INFERENCE,
            OptimizationStage.FINAL_BOUND,
        ),
        static_shape_only=True,
    )


__all__ = [
    "BACKEND_CANDIDATE_SCHEMA_VERSION",
    "BACKEND_PROFILE_SCHEMA_VERSION",
    "BackendCapability",
    "BackendVariant",
    "ExecutionCandidate",
    "ExecutionContext",
    "OperatorFamily",
    "PlacementKind",
    "capability_rejections",
    "fused_tir_linear_v1_capability",
]
