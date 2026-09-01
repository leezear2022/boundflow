"""Native plain-CROWN backend for executable RVIR-v3 payloads."""

# pylint: disable=too-many-arguments,too-many-locals,missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import torch

from ..domains.interval import IntervalState
from ..ir.task import BFTaskModule
from .crown_ibp import _forward_ibp_trace_mlp
from .native_verifier_ir_integration import (
    compile_native_plain_crown_query,
    execute_native_plain_crown_query,
)
from .rvir_v3_replacement import (
    ExecutableVerifierPayload,
    OwnedVerifierTensor,
    ReplacementBackendResult,
    VerifierPhase,
    VerifierTensorRole,
    tensor_sha256,
)
from .task_executor import InputSpec


def _program_parameters(module: BFTaskModule) -> Mapping[str, torch.Tensor]:
    raw = module.bindings.get("params", {})
    if not isinstance(raw, Mapping) or not all(
        isinstance(name, str) and torch.is_tensor(value) for name, value in raw.items()
    ):
        raise TypeError("RVIR-v3 native module parameters differ")
    return raw  # type: ignore[return-value]


def _intermediate_digest(
    lowers: Sequence[torch.Tensor], uppers: Sequence[torch.Tensor]
) -> str:
    digest = hashlib.sha256()
    for ordinal, (lower, upper) in enumerate(zip(lowers, uppers)):
        digest.update(str(ordinal).encode("utf-8"))
        digest.update(tensor_sha256(lower).encode("utf-8"))
        digest.update(tensor_sha256(upper).encode("utf-8"))
    return digest.hexdigest()


def build_native_initial_crown_payload(
    *,
    query_id: str,
    sequence_number: int,
    parent_query_id: str | None,
    module: BFTaskModule,
    input_lower: torch.Tensor,
    input_upper: torch.Tensor,
    linear_spec_c: torch.Tensor,
    intermediate_lowers: Sequence[torch.Tensor],
    intermediate_uppers: Sequence[torch.Tensor],
    requested_polarities: tuple[str, ...],
) -> ExecutableVerifierPayload:
    """Own a complete native plain-CROWN payload without a provider closure."""

    if len(intermediate_lowers) != len(intermediate_uppers):
        raise ValueError("RVIR-v3 native intermediate count differs")
    tensors = [
        OwnedVerifierTensor.own(
            "query.input.lower", VerifierTensorRole.INPUT_LOWER, input_lower
        ),
        OwnedVerifierTensor.own(
            "query.input.upper", VerifierTensorRole.INPUT_UPPER, input_upper
        ),
        OwnedVerifierTensor.own(
            "query.objective", VerifierTensorRole.LINEAR_SPEC, linear_spec_c
        ),
    ]
    for name, value in sorted(_program_parameters(module).items()):
        tensors.append(
            OwnedVerifierTensor.own(
                f"program.parameter:{name}",
                VerifierTensorRole.PROGRAM_PARAMETER,
                value,
            )
        )
    for ordinal, (lower, upper) in enumerate(
        zip(intermediate_lowers, intermediate_uppers)
    ):
        tensors.extend(
            (
                OwnedVerifierTensor.own(
                    f"intermediate.{ordinal:06d}.lower",
                    VerifierTensorRole.INTERMEDIATE_LOWER,
                    lower,
                ),
                OwnedVerifierTensor.own(
                    f"intermediate.{ordinal:06d}.upper",
                    VerifierTensorRole.INTERMEDIATE_UPPER,
                    upper,
                ),
            )
        )
    payload = ExecutableVerifierPayload(
        query_id=query_id,
        sequence_number=sequence_number,
        parent_query_id=parent_query_id,
        phase=VerifierPhase.INITIAL_CROWN,
        method="CROWN",
        requested_polarities=requested_polarities,
        tensors=tuple(tensors),
        expected_result_shape=(input_lower.shape[0], linear_spec_c.shape[-2]),
    )
    payload.validate()
    return payload


@dataclass
class NativePlainCrownRVIRV3Backend:
    """Execute native Bound/Plan/Task/Schedule IR with no original callback."""

    module: BFTaskModule
    input_value_name: str
    available_memory_bytes: int = 1 << 40
    backend_id: str = "boundflow.native-plain-crown-rvir-v3/v1"
    last_ir_hashes: Mapping[str, str] | None = None

    def execute(
        self,
        payload: ExecutableVerifierPayload,
        tensors: Mapping[str, torch.Tensor],
    ) -> ReplacementBackendResult:
        if payload.phase != VerifierPhase.INITIAL_CROWN:
            raise ValueError("RVIR-v3 native plain-CROWN phase differs")
        parameters = _program_parameters(self.module)
        for name, value in parameters.items():
            tensor_id = f"program.parameter:{name}"
            if tensor_id not in tensors or tensor_sha256(
                tensors[tensor_id]
            ) != tensor_sha256(value):
                raise ValueError("RVIR-v3 native program parameter differs")
        lower = payload.one_tensor(VerifierTensorRole.INPUT_LOWER).value
        upper = payload.one_tensor(VerifierTensorRole.INPUT_UPPER).value
        linear_spec = payload.one_tensor(VerifierTensorRole.LINEAR_SPEC).value
        input_spec = InputSpec.box(
            value_name=self.input_value_name,
            lower=lower,
            upper=upper,
        )
        interval_env, local_relu_pre = _forward_ibp_trace_mlp(self.module, input_spec)
        lower_items = payload.tensors_with_role(VerifierTensorRole.INTERMEDIATE_LOWER)
        upper_items = payload.tensors_with_role(VerifierTensorRole.INTERMEDIATE_UPPER)
        if len(lower_items) != len(local_relu_pre) or len(upper_items) != len(
            local_relu_pre
        ):
            raise ValueError("RVIR-v3 native intermediate topology differs")
        relu_pre: dict[str, IntervalState] = {}
        for (local_name, local), lower_item, upper_item in zip(
            local_relu_pre.items(), lower_items, upper_items
        ):
            if (
                lower_item.value.shape != local.lower.shape
                or upper_item.value.shape != local.upper.shape
            ):
                raise ValueError("RVIR-v3 native intermediate shape differs")
            relu_pre[local_name] = IntervalState(
                lower=lower_item.value.to(local.lower).contiguous(),
                upper=upper_item.value.to(local.upper).contiguous(),
            )
        intermediate_hash = _intermediate_digest(
            [item.value for item in lower_items],
            [item.value for item in upper_items],
        )
        compilation = compile_native_plain_crown_query(
            self.module,
            input_spec,
            interval_env=interval_env,
            relu_pre=relu_pre,
            linear_spec_C=linear_spec,
            intermediate_bounds_hash=intermediate_hash,
            query_id=payload.query_id,
            available_memory_bytes=self.available_memory_bytes,
        )
        result, _trace = execute_native_plain_crown_query(
            compilation,
            legacy_task_module=self.module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=linear_spec,
        )
        self.last_ir_hashes = compilation.hashes()
        return ReplacementBackendResult(
            lower=result.lower if "lower" in payload.requested_polarities else None,
            upper=result.upper if "upper" in payload.requested_polarities else None,
        )


__all__ = [
    "NativePlainCrownRVIRV3Backend",
    "build_native_initial_crown_payload",
]
