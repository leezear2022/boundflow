"""Backend-neutral execution contract for fused plain-CROWN regions."""

# pylint: disable=invalid-name,import-outside-toplevel,not-callable,too-many-locals
# mypy: disable-error-code=import-untyped

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence, Tuple, cast

import torch
from torch.nn import functional


class BoundaryRepresentation(Enum):
    """Representation visible at a fused region boundary."""

    DENSE = "dense"
    STRUCTURED = "structured"


class InternalMaterializationPolicy(Enum):
    """Intermediate tensors intentionally retained or removed within a region."""

    EAGER = "eager"
    ELIDE_RELU_SCALED_A = "elide_relu_scaled_a"


@dataclass(frozen=True)
class FusedReluAffineRequest:  # pylint: disable=too-many-instance-attributes
    """Dense-boundary request shared by reference and compiled executors."""

    kind: Literal["linear", "conv2d"]
    A_u: torch.Tensor
    A_l: torch.Tensor
    alpha_u: torch.Tensor
    alpha_l: torch.Tensor
    beta_u: torch.Tensor
    beta_l: torch.Tensor
    weight: torch.Tensor
    bias: Optional[torch.Tensor]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    attrs: Mapping[str, object]


@dataclass(frozen=True)
class FusedReluAffineResult:
    """Four outputs produced by one fused backward region."""

    A_prev_u: torch.Tensor
    A_prev_l: torch.Tensor
    bias_delta_u: torch.Tensor
    bias_delta_l: torch.Tensor


@dataclass(frozen=True)
class FusedCrownExecutionContext:
    """Legality properties that must never be inferred from tensor shapes."""

    plain_crown: bool = True
    requires_grad: bool = False
    alpha_enabled: bool = False
    beta_enabled: bool = False
    split_state_present: bool = False


@dataclass(frozen=True)
class FusedCrownExecutionStep:  # pylint: disable=too-many-instance-attributes
    """Explicit planner output consumed by the backward runtime."""

    kind: Literal["fused_relu_linear", "fused_relu_conv2d"]
    relu_op_index: int
    affine_op_index: int
    consumed_outputs: Tuple[str, str]
    graph_fingerprint: str
    boundary_representation: BoundaryRepresentation = BoundaryRepresentation.DENSE
    internal_materialization: InternalMaterializationPolicy = (
        InternalMaterializationPolicy.ELIDE_RELU_SCALED_A
    )
    backend: str = "tvm_fused_tir"


class FusedCrownExecutor(Protocol):
    """Executor contract kept independent from TVM and solver internals."""

    def supports_descriptor(
        self,
        descriptor: "FusedReluAffineDescriptor",
        context: FusedCrownExecutionContext,
    ) -> bool:
        """Reject an unsupported region before dense-boundary materialization."""
        raise NotImplementedError

    def supports(
        self,
        request: FusedReluAffineRequest,
        context: FusedCrownExecutionContext,
    ) -> bool:
        """Return whether this executor can legally run the request."""
        raise NotImplementedError

    def run(
        self,
        request: FusedReluAffineRequest,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> FusedReluAffineResult:
        """Execute one legal request on the selected CUDA stream."""
        raise NotImplementedError


def _pair_attr(
    attrs: Mapping[str, object], name: str, default: Tuple[int, int]
) -> Tuple[int, int]:
    value = attrs.get(name, default)
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"{name} expects an integer pair, got {value!r}")


@dataclass(frozen=True)
class FusedReluAffineDescriptor:  # pylint: disable=too-many-instance-attributes
    """Static region metadata available before coefficient materialization."""

    kind: Literal["linear", "conv2d"]
    coefficient_shape: Tuple[int, int, int]
    weight: torch.Tensor
    bias: Optional[torch.Tensor]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    attrs: Mapping[str, object]
    device: torch.device
    dtype: torch.dtype


def fused_crown_graph_fingerprint(ops: Sequence[Any]) -> str:
    """Return a stable identity for the exact task-op schedule being planned."""

    payload = [
        {
            "index": index,
            "op_type": op.op_type,
            "name": op.name,
            "inputs": list(op.inputs),
            "outputs": list(op.outputs),
            "attrs": sorted((str(key), repr(value)) for key, value in op.attrs.items()),
        }
        for index, op in enumerate(ops)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _consumer_indices(ops: Sequence[Any]) -> Mapping[str, Tuple[int, ...]]:
    consumers: dict[str, list[int]] = {}
    for index, op in enumerate(ops):
        for input_name in op.inputs:
            consumers.setdefault(input_name, []).append(index)
    return {name: tuple(indices) for name, indices in consumers.items()}


def plan_fused_crown_regions(
    ops: Sequence[Any],
) -> Tuple[FusedCrownExecutionStep, ...]:
    """Match forward Affine->ReLU pairs once and emit an explicit schedule."""

    steps: list[FusedCrownExecutionStep] = []
    consumers = _consumer_indices(ops)
    fingerprint = fused_crown_graph_fingerprint(ops)
    for relu_index, relu_op in enumerate(ops):
        if relu_op.op_type != "relu" or relu_index == 0:
            continue
        affine_index = relu_index - 1
        affine_op = ops[affine_index]
        if affine_op.op_type not in {"linear", "conv2d"}:
            continue
        if len(affine_op.outputs) != 1 or tuple(relu_op.inputs) != tuple(
            affine_op.outputs
        ):
            continue
        if consumers.get(affine_op.outputs[0], ()) != (relu_index,):
            continue
        kind: Literal["fused_relu_linear", "fused_relu_conv2d"] = (
            "fused_relu_linear"
            if affine_op.op_type == "linear"
            else "fused_relu_conv2d"
        )
        steps.append(
            FusedCrownExecutionStep(
                kind=kind,
                relu_op_index=relu_index,
                affine_op_index=affine_index,
                consumed_outputs=(relu_op.outputs[0], affine_op.outputs[0]),
                graph_fingerprint=fingerprint,
            )
        )
    return tuple(steps)


def validate_fused_crown_execution_steps(  # pylint: disable=too-many-branches
    ops: Sequence[Any], steps: Sequence[FusedCrownExecutionStep]
) -> Tuple[FusedCrownExecutionStep, ...]:
    """Keep only graph-current, single-consumer, v1 TVM execution steps."""

    fingerprint = fused_crown_graph_fingerprint(ops)
    consumers = _consumer_indices(ops)
    valid: list[FusedCrownExecutionStep] = []
    seen_relu: set[int] = set()
    seen_affine: set[int] = set()
    for step in steps:
        if step.graph_fingerprint != fingerprint:
            continue
        if step.relu_op_index in seen_relu or step.affine_op_index in seen_affine:
            continue
        if not 0 <= step.affine_op_index < step.relu_op_index < len(ops):
            continue
        if step.relu_op_index != step.affine_op_index + 1:
            continue
        affine_op = ops[step.affine_op_index]
        relu_op = ops[step.relu_op_index]
        expected_kind = (
            "fused_relu_linear"
            if affine_op.op_type == "linear"
            else "fused_relu_conv2d"
        )
        if affine_op.op_type not in {"linear", "conv2d"} or step.kind != expected_kind:
            continue
        if relu_op.op_type != "relu" or tuple(relu_op.inputs) != tuple(
            affine_op.outputs
        ):
            continue
        if len(affine_op.outputs) != 1 or len(relu_op.outputs) != 1:
            continue
        if step.consumed_outputs != (relu_op.outputs[0], affine_op.outputs[0]):
            continue
        if consumers.get(affine_op.outputs[0], ()) != (step.relu_op_index,):
            continue
        if step.boundary_representation != BoundaryRepresentation.DENSE:
            continue
        if (
            step.internal_materialization
            != InternalMaterializationPolicy.ELIDE_RELU_SCALED_A
        ):
            continue
        if step.backend != "tvm_fused_tir":
            continue
        valid.append(step)
        seen_relu.add(step.relu_op_index)
        seen_affine.add(step.affine_op_index)
    return tuple(valid)


def _plain_static_cuda_fp32(
    request: FusedReluAffineRequest,
    context: FusedCrownExecutionContext,
) -> bool:
    tensors = [
        request.A_u,
        request.A_l,
        request.alpha_u,
        request.alpha_l,
        request.beta_u,
        request.beta_l,
        request.weight,
    ]
    if request.bias is not None:
        tensors.append(request.bias)
    return bool(
        context.plain_crown
        and not context.requires_grad
        and not context.alpha_enabled
        and not context.beta_enabled
        and not context.split_state_present
        and all(t.device.type == "cuda" for t in tensors)
        and all(t.dtype == torch.float32 for t in tensors)
        and all(t.is_contiguous() for t in tensors)
        and all(not t.requires_grad for t in tensors)
    )


class TorchDenseFusedCrownReference:
    """Allocation-visible eager oracle implementing the same region contract."""

    def supports_descriptor(
        self,
        descriptor: FusedReluAffineDescriptor,
        context: FusedCrownExecutionContext,
    ) -> bool:
        """Reference execution accepts either modeled family after context filtering."""

        del context
        return descriptor.kind in {"linear", "conv2d"}

    def supports(
        self,
        request: FusedReluAffineRequest,
        context: FusedCrownExecutionContext,
    ) -> bool:
        """The reference covers both currently modeled affine families."""

        del context
        return request.kind in {"linear", "conv2d"}

    def run(
        self,
        request: FusedReluAffineRequest,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> FusedReluAffineResult:
        """Run the allocation-visible dense equations."""

        del stream

        A_u = request.A_u.reshape(request.A_u.shape[0], request.A_u.shape[1], -1)
        A_l = request.A_l.reshape(request.A_l.shape[0], request.A_l.shape[1], -1)
        scaled_u = torch.where(
            A_u >= 0, A_u * request.alpha_u[:, None], A_u * request.alpha_l[:, None]
        )
        scaled_l = torch.where(
            A_l >= 0, A_l * request.alpha_l[:, None], A_l * request.alpha_u[:, None]
        )
        delta_u = torch.where(
            A_u >= 0, A_u * request.beta_u[:, None], A_u * request.beta_l[:, None]
        ).sum(2)
        delta_l = torch.where(
            A_l >= 0, A_l * request.beta_l[:, None], A_l * request.beta_u[:, None]
        ).sum(2)
        if request.kind == "linear":
            if request.bias is not None:
                delta_u = delta_u + (scaled_u * request.bias).sum(2)
                delta_l = delta_l + (scaled_l * request.bias).sum(2)
            return FusedReluAffineResult(
                scaled_u @ request.weight,
                scaled_l @ request.weight,
                delta_u,
                delta_l,
            )

        stride = _pair_attr(request.attrs, "stride", (1, 1))
        padding = _pair_attr(request.attrs, "padding", (0, 0))
        dilation = _pair_attr(request.attrs, "dilation", (1, 1))
        groups = cast(int, request.attrs.get("groups", 1))
        out_pad = _pair_attr(request.attrs, "output_padding", (0, 0))
        shaped_u = scaled_u.reshape(
            request.A_u.shape[0] * request.A_u.shape[1], *request.output_shape
        )
        shaped_l = scaled_l.reshape(
            request.A_l.shape[0] * request.A_l.shape[1], *request.output_shape
        )
        previous_u = functional.conv_transpose2d(
            shaped_u,
            request.weight,
            stride=stride,
            padding=padding,
            output_padding=out_pad,
            groups=groups,
            dilation=dilation,
        ).reshape(request.A_u.shape[0], request.A_u.shape[1], *request.input_shape)
        previous_l = functional.conv_transpose2d(
            shaped_l,
            request.weight,
            stride=stride,
            padding=padding,
            output_padding=out_pad,
            groups=groups,
            dilation=dilation,
        ).reshape(request.A_l.shape[0], request.A_l.shape[1], *request.input_shape)
        if request.bias is not None:
            bias_map = (
                request.bias.view(-1, 1, 1).expand(request.output_shape).reshape(-1)
            )
            delta_u = delta_u + (scaled_u * bias_map).sum(2)
            delta_l = delta_l + (scaled_l * bias_map).sum(2)
        return FusedReluAffineResult(previous_u, previous_l, delta_u, delta_l)


class TVMFusedCrownExecutor:
    """Zero-copy DLPack adapter for the specialized CUDA TIR candidates."""

    def supports_descriptor(
        self,
        descriptor: FusedReluAffineDescriptor,
        context: FusedCrownExecutionContext,
    ) -> bool:
        """Reject illegal contexts and signatures before dense A is materialized."""

        tensors = [descriptor.weight]
        if descriptor.bias is not None:
            tensors.append(descriptor.bias)
        if not (
            context.plain_crown
            and not context.requires_grad
            and not context.alpha_enabled
            and not context.beta_enabled
            and not context.split_state_present
            and descriptor.device.type == "cuda"
            and descriptor.dtype == torch.float32
            and all(t.device == descriptor.device for t in tensors)
            and all(t.dtype == descriptor.dtype for t in tensors)
            and all(t.is_contiguous() for t in tensors)
            and all(not t.requires_grad for t in tensors)
        ):
            return False
        try:
            self._compile_descriptor_signature(descriptor)
        except (IndexError, NotImplementedError, ValueError):
            return False
        return True

    def supports(
        self,
        request: FusedReluAffineRequest,
        context: FusedCrownExecutionContext,
    ) -> bool:
        """Filter the strict plain-CROWN CUDA v1 capability."""

        if not _plain_static_cuda_fp32(request, context):
            return False
        try:
            self._compile_signature(request)
        except (NotImplementedError, ValueError):
            return False
        return True

    @staticmethod
    def _compute_capability(device: torch.device) -> str:
        major, minor = torch.cuda.get_device_capability(device)
        return f"sm_{major}{minor}"

    def _compile_descriptor_signature(
        self, descriptor: FusedReluAffineDescriptor
    ) -> object:
        domain, spec = descriptor.coefficient_shape[:2]
        capability = self._compute_capability(descriptor.device)
        if descriptor.kind == "linear":
            from ..backends.tvm.fused_crown_linear import FusedCrownLinearKey

            key = FusedCrownLinearKey(
                domain,
                spec,
                int(descriptor.output_shape[0]),
                int(descriptor.input_shape[0]),
                compute_capability=capability,
            )
            key.validate()
            return key
        from ..backends.tvm.fused_crown_conv2d import FusedCrownConv2dSignature

        signature = FusedCrownConv2dSignature(
            domain_batch=domain,
            spec_batch=spec,
            input_channels=int(descriptor.input_shape[0]),
            input_height=int(descriptor.input_shape[1]),
            input_width=int(descriptor.input_shape[2]),
            output_channels=int(descriptor.output_shape[0]),
            output_height=int(descriptor.output_shape[1]),
            output_width=int(descriptor.output_shape[2]),
            kernel_height=int(descriptor.weight.shape[2]),
            kernel_width=int(descriptor.weight.shape[3]),
            stride=_pair_attr(descriptor.attrs, "stride", (1, 1)),
            padding=_pair_attr(descriptor.attrs, "padding", (0, 0)),
            dilation=_pair_attr(descriptor.attrs, "dilation", (1, 1)),
            groups=cast(int, descriptor.attrs.get("groups", 1)),
            bias_present=descriptor.bias is not None,
            compute_capability=capability,
        )
        signature.validate()
        return signature

    def _compile_signature(self, request: FusedReluAffineRequest) -> object:
        descriptor = FusedReluAffineDescriptor(
            kind=request.kind,
            coefficient_shape=(
                int(request.A_u.shape[0]),
                int(request.A_u.shape[1]),
                int(request.A_u.shape[2]),
            ),
            weight=request.weight,
            bias=request.bias,
            input_shape=request.input_shape,
            output_shape=request.output_shape,
            attrs=request.attrs,
            device=request.A_u.device,
            dtype=request.A_u.dtype,
        )
        return self._compile_descriptor_signature(descriptor)

    def run(
        self,
        request: FusedReluAffineRequest,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> FusedReluAffineResult:
        """Compile/cache the candidate and launch it through zero-copy DLPack views."""

        import tvm  # pylint: disable=import-outside-toplevel,import-error
        import tvm_ffi  # pylint: disable=import-outside-toplevel,import-error

        signature = self._compile_signature(request)
        if request.kind == "linear":
            from ..backends.tvm.fused_crown_linear import (
                build_fused_crown_linear_module,
            )

            compiled = build_fused_crown_linear_module(signature)
        else:
            from ..backends.tvm.fused_crown_conv2d import (
                build_fused_crown_conv2d_module,
            )

            compiled = build_fused_crown_conv2d_module(signature)
        domain, spec = int(request.A_u.shape[0]), int(request.A_u.shape[1])
        output_shape = (domain, spec, *request.input_shape)
        outputs = (
            torch.empty(
                output_shape, device=request.A_u.device, dtype=request.A_u.dtype
            ),
            torch.empty(
                output_shape, device=request.A_u.device, dtype=request.A_u.dtype
            ),
            torch.empty(
                (domain, spec), device=request.A_u.device, dtype=request.A_u.dtype
            ),
            torch.empty(
                (domain, spec), device=request.A_u.device, dtype=request.A_u.dtype
            ),
        )
        alpha_shape = (domain, *request.output_shape)
        inputs = [
            request.A_u.reshape(domain, spec, *request.output_shape),
            request.A_l.reshape(domain, spec, *request.output_shape),
            request.alpha_u.reshape(alpha_shape),
            request.beta_u.reshape(alpha_shape),
            request.alpha_l.reshape(alpha_shape),
            request.beta_l.reshape(alpha_shape),
            request.weight,
        ]
        if request.kind == "linear" or request.bias is not None:
            inputs.append(
                request.bias
                if request.bias is not None
                else torch.zeros(request.output_shape[0], device=request.A_u.device)
            )
        current = stream or torch.cuda.current_stream(request.A_u.device)
        with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
            compiled(
                *[tvm.runtime.from_dlpack(tensor) for tensor in inputs],
                *[tvm.runtime.from_dlpack(tensor) for tensor in outputs],
            )
        return FusedReluAffineResult(*outputs)


__all__ = [
    "BoundaryRepresentation",
    "FusedCrownExecutionContext",
    "FusedCrownExecutionStep",
    "FusedCrownExecutor",
    "FusedReluAffineRequest",
    "FusedReluAffineResult",
    "InternalMaterializationPolicy",
    "TVMFusedCrownExecutor",
    "TorchDenseFusedCrownReference",
    "fused_crown_graph_fingerprint",
    "plan_fused_crown_regions",
    "validate_fused_crown_execution_steps",
]
