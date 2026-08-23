"""First-class compiler and execution receipts for differentiable lower TIR."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

DIFFERENTIABLE_LOWER_TIR_TEMPLATE_SCHEMA = (
    "boundflow.differentiable-lower-tir-template/v1"
)
DIFFERENTIABLE_LOWER_TIR_INSTANCE_SCHEMA = (
    "boundflow.differentiable-lower-tir-instance/v1"
)
DIFFERENTIABLE_LOWER_TIR_SCHEDULE_SCHEMA = (
    "boundflow.differentiable-lower-tir-schedule/v1"
)
DIFFERENTIABLE_LOWER_TIR_MODULE_RECEIPT_SCHEMA = (
    "boundflow.differentiable-lower-tir-module-receipt/v1"
)
DIFFERENTIABLE_LOWER_TIR_LAUNCH_RECEIPT_SCHEMA = (
    "boundflow.differentiable-lower-tir-launch-receipt/v1"
)
IDENTITY_FORWARD_SYMBOL = "boundflow_b4b2_identity_forward"
IDENTITY_BACKWARD_SYMBOL = "boundflow_b4b2_identity_backward"
FROZEN_TVM_COMMIT = "6248b5db43505fbcfb13cc289d11877d5d2649e8"
FROZEN_TVM_FFI_COMMIT = "438f6439148b059d424ce2cc2a348736923f6948"


def canonical_tir_hash(value: object) -> str:
    """Hash one canonical JSON value."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha1(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _integer(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _boolean(payload: Mapping[str, object], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _strings(payload: Mapping[str, object], name: str) -> tuple[str, ...]:
    value = payload.get(name)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a string list")
    return tuple(value)


def _integers(payload: Mapping[str, object], name: str) -> tuple[int, ...]:
    value = payload.get(name)
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{name} must be an integer list")
    return tuple(value)


@dataclass(frozen=True)
class DifferentiableLowerTIRTemplateV1:  # pylint: disable=too-many-instance-attributes
    """Static semantic-to-compiler boundary for one differentiable region."""

    lower_region_ir_hash: str
    anchor_id: str
    operator_kind: str
    mapping_layout_hash: str
    operator_attributes_hash: str
    abi: str
    dtype: str
    device_kind: str
    target: str
    compute_capability: str
    tensor_numel: int
    gradient_targets: tuple[str, ...]
    forward_symbol: str = IDENTITY_FORWARD_SYMBOL
    backward_symbol: str = IDENTITY_BACKWARD_SYMBOL
    enabled_by_default: bool = False
    performance_admitted: bool = False
    schema_version: str = DIFFERENTIABLE_LOWER_TIR_TEMPLATE_SCHEMA

    def validate(self) -> None:
        if (
            self.schema_version != DIFFERENTIABLE_LOWER_TIR_TEMPLATE_SCHEMA
            or not _is_sha256(self.lower_region_ir_hash)
            or not self.anchor_id
            or self.operator_kind not in {"linear", "conv2d"}
            or not _is_sha256(self.mapping_layout_hash)
            or not _is_sha256(self.operator_attributes_hash)
            or self.abi != "identity-probe-v1"
            or self.dtype != "torch.float32"
            or self.device_kind != "cuda"
            or self.target != "cuda"
            or not self.compute_capability.startswith("sm_")
            or self.tensor_numel < 1
            or self.gradient_targets != ("input",)
            or self.forward_symbol != IDENTITY_FORWARD_SYMBOL
            or self.backward_symbol != IDENTITY_BACKWARD_SYMBOL
            or self.enabled_by_default is not False
            or self.performance_admitted is not False
        ):
            raise ValueError("differentiable lower TIR template differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "lower_region_ir_hash": self.lower_region_ir_hash,
            "anchor_id": self.anchor_id,
            "operator_kind": self.operator_kind,
            "mapping_layout_hash": self.mapping_layout_hash,
            "operator_attributes_hash": self.operator_attributes_hash,
            "abi": self.abi,
            "dtype": self.dtype,
            "device_kind": self.device_kind,
            "target": self.target,
            "compute_capability": self.compute_capability,
            "tensor_numel": self.tensor_numel,
            "gradient_targets": list(self.gradient_targets),
            "forward_symbol": self.forward_symbol,
            "backward_symbol": self.backward_symbol,
            "enabled_by_default": self.enabled_by_default,
            "performance_admitted": self.performance_admitted,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "DifferentiableLowerTIRTemplateV1":
        value = cls(
            schema_version=_string(payload, "schema_version"),
            lower_region_ir_hash=_string(payload, "lower_region_ir_hash"),
            anchor_id=_string(payload, "anchor_id"),
            operator_kind=_string(payload, "operator_kind"),
            mapping_layout_hash=_string(payload, "mapping_layout_hash"),
            operator_attributes_hash=_string(payload, "operator_attributes_hash"),
            abi=_string(payload, "abi"),
            dtype=_string(payload, "dtype"),
            device_kind=_string(payload, "device_kind"),
            target=_string(payload, "target"),
            compute_capability=_string(payload, "compute_capability"),
            tensor_numel=_integer(payload, "tensor_numel"),
            gradient_targets=_strings(payload, "gradient_targets"),
            forward_symbol=_string(payload, "forward_symbol"),
            backward_symbol=_string(payload, "backward_symbol"),
            enabled_by_default=_boolean(payload, "enabled_by_default"),
            performance_admitted=_boolean(payload, "performance_admitted"),
        )
        value.validate()
        return value

    def stable_hash(self) -> str:
        return canonical_tir_hash(self.to_dict())


@dataclass(frozen=True)
class DifferentiableLowerTIRInstanceV1:
    """Dynamic launch identity kept separate from the static template."""

    template_hash: str
    lower_region_instance_hash: str
    tensor_shape: tuple[int, ...]
    input_tensor_hash: str
    upstream_gradient_hash: str
    device_ordinal: int
    dtype: str = "torch.float32"
    schema_version: str = DIFFERENTIABLE_LOWER_TIR_INSTANCE_SCHEMA

    def validate_against(self, template: DifferentiableLowerTIRTemplateV1) -> None:
        template.validate()
        if (
            self.schema_version != DIFFERENTIABLE_LOWER_TIR_INSTANCE_SCHEMA
            or self.template_hash != template.stable_hash()
            or not _is_sha256(self.lower_region_instance_hash)
            or not self.tensor_shape
            or any(dimension < 1 for dimension in self.tensor_shape)
            or self.numel != template.tensor_numel
            or not _is_sha256(self.input_tensor_hash)
            or not _is_sha256(self.upstream_gradient_hash)
            or self.device_ordinal < 0
            or self.dtype != template.dtype
        ):
            raise ValueError("differentiable lower TIR instance differs")

    @property
    def numel(self) -> int:
        result = 1
        for dimension in self.tensor_shape:
            result *= dimension
        return result

    def to_dict(self, template: DifferentiableLowerTIRTemplateV1) -> dict[str, object]:
        self.validate_against(template)
        return {
            "schema_version": self.schema_version,
            "template_hash": self.template_hash,
            "lower_region_instance_hash": self.lower_region_instance_hash,
            "tensor_shape": list(self.tensor_shape),
            "input_tensor_hash": self.input_tensor_hash,
            "upstream_gradient_hash": self.upstream_gradient_hash,
            "device_ordinal": self.device_ordinal,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        template: DifferentiableLowerTIRTemplateV1,
    ) -> "DifferentiableLowerTIRInstanceV1":
        value = cls(
            schema_version=_string(payload, "schema_version"),
            template_hash=_string(payload, "template_hash"),
            lower_region_instance_hash=_string(payload, "lower_region_instance_hash"),
            tensor_shape=_integers(payload, "tensor_shape"),
            input_tensor_hash=_string(payload, "input_tensor_hash"),
            upstream_gradient_hash=_string(payload, "upstream_gradient_hash"),
            device_ordinal=_integer(payload, "device_ordinal"),
            dtype=_string(payload, "dtype"),
        )
        value.validate_against(template)
        return value

    def stable_hash(self, template: DifferentiableLowerTIRTemplateV1) -> str:
        return canonical_tir_hash(self.to_dict(template))


@dataclass(frozen=True)
class DifferentiableLowerTIRScheduleV1:
    """A mathematical-semantics-preserving identity CUDA schedule."""

    template_hash: str
    tensor_numel: int
    thread_extent: int
    block_extent: int
    schedule_family: str = "identity-copy-1d-v1"
    vector_width: int = 1
    workspace_bytes: int = 0
    deterministic: bool = True
    candidate_ordinal: int = 0
    schema_version: str = DIFFERENTIABLE_LOWER_TIR_SCHEDULE_SCHEMA

    def validate_against(self, template: DifferentiableLowerTIRTemplateV1) -> None:
        template.validate()
        expected_blocks = (
            self.tensor_numel + self.thread_extent - 1
        ) // self.thread_extent
        if (
            self.schema_version != DIFFERENTIABLE_LOWER_TIR_SCHEDULE_SCHEMA
            or self.template_hash != template.stable_hash()
            or self.tensor_numel != template.tensor_numel
            or self.thread_extent not in {64, 128, 256}
            or self.block_extent != expected_blocks
            or self.schedule_family != "identity-copy-1d-v1"
            or self.vector_width != 1
            or self.workspace_bytes != 0
            or self.deterministic is not True
            or self.candidate_ordinal != 0
        ):
            raise ValueError("differentiable lower TIR schedule differs")

    def to_dict(self, template: DifferentiableLowerTIRTemplateV1) -> dict[str, object]:
        self.validate_against(template)
        return {
            "schema_version": self.schema_version,
            "template_hash": self.template_hash,
            "tensor_numel": self.tensor_numel,
            "thread_extent": self.thread_extent,
            "block_extent": self.block_extent,
            "schedule_family": self.schedule_family,
            "vector_width": self.vector_width,
            "workspace_bytes": self.workspace_bytes,
            "deterministic": self.deterministic,
            "candidate_ordinal": self.candidate_ordinal,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        template: DifferentiableLowerTIRTemplateV1,
    ) -> "DifferentiableLowerTIRScheduleV1":
        value = cls(
            schema_version=_string(payload, "schema_version"),
            template_hash=_string(payload, "template_hash"),
            tensor_numel=_integer(payload, "tensor_numel"),
            thread_extent=_integer(payload, "thread_extent"),
            block_extent=_integer(payload, "block_extent"),
            schedule_family=_string(payload, "schedule_family"),
            vector_width=_integer(payload, "vector_width"),
            workspace_bytes=_integer(payload, "workspace_bytes"),
            deterministic=_boolean(payload, "deterministic"),
            candidate_ordinal=_integer(payload, "candidate_ordinal"),
        )
        value.validate_against(template)
        return value

    def stable_hash(self, template: DifferentiableLowerTIRTemplateV1) -> str:
        return canonical_tir_hash(self.to_dict(template))


@dataclass(frozen=True)
class DifferentiableLowerTIRModuleReceiptV1:  # pylint: disable=too-many-instance-attributes
    """Hash-bound receipt for one compiled identity forward/backward module."""

    template_hash: str
    schedule_hash: str
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    cache_key: str
    target: str
    compute_capability: str
    tvm_version: str
    tvm_commit: str
    tvm_ffi_commit: str
    torch_version: str
    exported_symbols: tuple[str, ...]
    performance_claimed: bool = False
    schema_version: str = DIFFERENTIABLE_LOWER_TIR_MODULE_RECEIPT_SCHEMA

    def validate_against(
        self,
        template: DifferentiableLowerTIRTemplateV1,
        schedule: DifferentiableLowerTIRScheduleV1,
    ) -> None:
        template.validate()
        schedule.validate_against(template)
        hashes = (
            self.unscheduled_tir_hash,
            self.scheduled_tir_hash,
            self.device_source_hash,
            self.cache_key,
        )
        if (
            self.schema_version != DIFFERENTIABLE_LOWER_TIR_MODULE_RECEIPT_SCHEMA
            or self.template_hash != template.stable_hash()
            or self.schedule_hash != schedule.stable_hash(template)
            or any(not _is_sha256(value) for value in hashes)
            or self.target != template.target
            or self.compute_capability != template.compute_capability
            or not self.tvm_version
            or not _is_git_sha1(self.tvm_commit)
            or self.tvm_commit != FROZEN_TVM_COMMIT
            or not _is_git_sha1(self.tvm_ffi_commit)
            or self.tvm_ffi_commit != FROZEN_TVM_FFI_COMMIT
            or not self.torch_version
            or self.exported_symbols
            != (template.forward_symbol, template.backward_symbol)
            or self.cache_key != self.expected_cache_key(template, schedule)
            or self.performance_claimed is not False
        ):
            raise ValueError("differentiable lower TIR module receipt differs")

    @staticmethod
    def expected_cache_key(
        template: DifferentiableLowerTIRTemplateV1,
        schedule: DifferentiableLowerTIRScheduleV1,
    ) -> str:
        return canonical_tir_hash(
            {
                "schema": DIFFERENTIABLE_LOWER_TIR_MODULE_RECEIPT_SCHEMA,
                "template_hash": template.stable_hash(),
                "schedule_hash": schedule.stable_hash(template),
                "symbols": [template.forward_symbol, template.backward_symbol],
                "target": template.target,
                "compute_capability": template.compute_capability,
                "tvm_commit": FROZEN_TVM_COMMIT,
                "tvm_ffi_commit": FROZEN_TVM_FFI_COMMIT,
            }
        )

    def to_dict(
        self,
        template: DifferentiableLowerTIRTemplateV1,
        schedule: DifferentiableLowerTIRScheduleV1,
    ) -> dict[str, object]:
        self.validate_against(template, schedule)
        return {
            "schema_version": self.schema_version,
            "template_hash": self.template_hash,
            "schedule_hash": self.schedule_hash,
            "unscheduled_tir_hash": self.unscheduled_tir_hash,
            "scheduled_tir_hash": self.scheduled_tir_hash,
            "device_source_hash": self.device_source_hash,
            "cache_key": self.cache_key,
            "target": self.target,
            "compute_capability": self.compute_capability,
            "tvm_version": self.tvm_version,
            "tvm_commit": self.tvm_commit,
            "tvm_ffi_commit": self.tvm_ffi_commit,
            "torch_version": self.torch_version,
            "exported_symbols": list(self.exported_symbols),
            "performance_claimed": self.performance_claimed,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        template: DifferentiableLowerTIRTemplateV1,
        schedule: DifferentiableLowerTIRScheduleV1,
    ) -> "DifferentiableLowerTIRModuleReceiptV1":
        value = cls(
            schema_version=_string(payload, "schema_version"),
            template_hash=_string(payload, "template_hash"),
            schedule_hash=_string(payload, "schedule_hash"),
            unscheduled_tir_hash=_string(payload, "unscheduled_tir_hash"),
            scheduled_tir_hash=_string(payload, "scheduled_tir_hash"),
            device_source_hash=_string(payload, "device_source_hash"),
            cache_key=_string(payload, "cache_key"),
            target=_string(payload, "target"),
            compute_capability=_string(payload, "compute_capability"),
            tvm_version=_string(payload, "tvm_version"),
            tvm_commit=_string(payload, "tvm_commit"),
            tvm_ffi_commit=_string(payload, "tvm_ffi_commit"),
            torch_version=_string(payload, "torch_version"),
            exported_symbols=_strings(payload, "exported_symbols"),
            performance_claimed=_boolean(payload, "performance_claimed"),
        )
        value.validate_against(template, schedule)
        return value

    def stable_hash(
        self,
        template: DifferentiableLowerTIRTemplateV1,
        schedule: DifferentiableLowerTIRScheduleV1,
    ) -> str:
        return canonical_tir_hash(self.to_dict(template, schedule))


@dataclass(frozen=True)
class DifferentiableLowerTIRLaunchReceiptV1:  # pylint: disable=too-many-instance-attributes
    """Fail-closed zero-copy, stream, cache, alias and launch ledger."""

    template_hash: str
    instance_hash: str
    schedule_hash: str
    module_receipt_hash: str
    stream_id: int
    tvm_ffi_stream_id: int
    input_data_ptr: int
    output_data_ptr: int
    upstream_gradient_data_ptr: int
    input_gradient_data_ptr: int
    input_roundtrip_ptr_exact: bool
    output_roundtrip_ptr_exact: bool
    upstream_gradient_roundtrip_ptr_exact: bool
    input_gradient_roundtrip_ptr_exact: bool
    output_aliases_input: bool
    input_gradient_aliases_upstream: bool
    output_tensor_hash: str
    input_gradient_hash: str
    cache_event: str
    forward_launch_count: int
    backward_launch_count: int
    fallback_count: int
    eager_backward_count: int
    performance_claimed: bool = False
    schema_version: str = DIFFERENTIABLE_LOWER_TIR_LAUNCH_RECEIPT_SCHEMA

    def validate_against(
        self,
        template: DifferentiableLowerTIRTemplateV1,
        instance: DifferentiableLowerTIRInstanceV1,
        schedule: DifferentiableLowerTIRScheduleV1,
        module: DifferentiableLowerTIRModuleReceiptV1,
    ) -> None:
        instance.validate_against(template)
        schedule.validate_against(template)
        module.validate_against(template, schedule)
        pointer_exact = (
            self.input_roundtrip_ptr_exact,
            self.output_roundtrip_ptr_exact,
            self.upstream_gradient_roundtrip_ptr_exact,
            self.input_gradient_roundtrip_ptr_exact,
        )
        pointers = (
            self.input_data_ptr,
            self.output_data_ptr,
            self.upstream_gradient_data_ptr,
            self.input_gradient_data_ptr,
        )
        if (
            self.schema_version != DIFFERENTIABLE_LOWER_TIR_LAUNCH_RECEIPT_SCHEMA
            or self.template_hash != template.stable_hash()
            or self.instance_hash != instance.stable_hash(template)
            or self.schedule_hash != schedule.stable_hash(template)
            or self.module_receipt_hash != module.stable_hash(template, schedule)
            or self.stream_id < 0
            or self.tvm_ffi_stream_id != self.stream_id
            or any(pointer <= 0 for pointer in pointers)
            or any(value is not True for value in pointer_exact)
            or self.output_aliases_input is not False
            or self.input_gradient_aliases_upstream is not False
            or self.output_data_ptr == self.input_data_ptr
            or self.input_gradient_data_ptr == self.upstream_gradient_data_ptr
            or not _is_sha256(self.output_tensor_hash)
            or not _is_sha256(self.input_gradient_hash)
            or self.output_tensor_hash != instance.input_tensor_hash
            or self.input_gradient_hash != instance.upstream_gradient_hash
            or self.cache_event not in {"hit", "miss"}
            or self.forward_launch_count != 1
            or self.backward_launch_count != 1
            or self.fallback_count != 0
            or self.eager_backward_count != 0
            or self.performance_claimed is not False
        ):
            raise ValueError("differentiable lower TIR launch receipt differs")

    def to_dict(
        self,
        template: DifferentiableLowerTIRTemplateV1,
        instance: DifferentiableLowerTIRInstanceV1,
        schedule: DifferentiableLowerTIRScheduleV1,
        module: DifferentiableLowerTIRModuleReceiptV1,
    ) -> dict[str, object]:
        self.validate_against(template, instance, schedule, module)
        return {
            "schema_version": self.schema_version,
            "template_hash": self.template_hash,
            "instance_hash": self.instance_hash,
            "schedule_hash": self.schedule_hash,
            "module_receipt_hash": self.module_receipt_hash,
            "stream_id": self.stream_id,
            "tvm_ffi_stream_id": self.tvm_ffi_stream_id,
            "input_data_ptr": self.input_data_ptr,
            "output_data_ptr": self.output_data_ptr,
            "upstream_gradient_data_ptr": self.upstream_gradient_data_ptr,
            "input_gradient_data_ptr": self.input_gradient_data_ptr,
            "input_roundtrip_ptr_exact": self.input_roundtrip_ptr_exact,
            "output_roundtrip_ptr_exact": self.output_roundtrip_ptr_exact,
            "upstream_gradient_roundtrip_ptr_exact": self.upstream_gradient_roundtrip_ptr_exact,
            "input_gradient_roundtrip_ptr_exact": self.input_gradient_roundtrip_ptr_exact,
            "output_aliases_input": self.output_aliases_input,
            "input_gradient_aliases_upstream": self.input_gradient_aliases_upstream,
            "output_tensor_hash": self.output_tensor_hash,
            "input_gradient_hash": self.input_gradient_hash,
            "cache_event": self.cache_event,
            "forward_launch_count": self.forward_launch_count,
            "backward_launch_count": self.backward_launch_count,
            "fallback_count": self.fallback_count,
            "eager_backward_count": self.eager_backward_count,
            "performance_claimed": self.performance_claimed,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        template: DifferentiableLowerTIRTemplateV1,
        instance: DifferentiableLowerTIRInstanceV1,
        schedule: DifferentiableLowerTIRScheduleV1,
        module: DifferentiableLowerTIRModuleReceiptV1,
    ) -> "DifferentiableLowerTIRLaunchReceiptV1":
        value = cls(
            schema_version=_string(payload, "schema_version"),
            template_hash=_string(payload, "template_hash"),
            instance_hash=_string(payload, "instance_hash"),
            schedule_hash=_string(payload, "schedule_hash"),
            module_receipt_hash=_string(payload, "module_receipt_hash"),
            stream_id=_integer(payload, "stream_id"),
            tvm_ffi_stream_id=_integer(payload, "tvm_ffi_stream_id"),
            input_data_ptr=_integer(payload, "input_data_ptr"),
            output_data_ptr=_integer(payload, "output_data_ptr"),
            upstream_gradient_data_ptr=_integer(payload, "upstream_gradient_data_ptr"),
            input_gradient_data_ptr=_integer(payload, "input_gradient_data_ptr"),
            input_roundtrip_ptr_exact=_boolean(payload, "input_roundtrip_ptr_exact"),
            output_roundtrip_ptr_exact=_boolean(payload, "output_roundtrip_ptr_exact"),
            upstream_gradient_roundtrip_ptr_exact=_boolean(
                payload, "upstream_gradient_roundtrip_ptr_exact"
            ),
            input_gradient_roundtrip_ptr_exact=_boolean(
                payload, "input_gradient_roundtrip_ptr_exact"
            ),
            output_aliases_input=_boolean(payload, "output_aliases_input"),
            input_gradient_aliases_upstream=_boolean(
                payload, "input_gradient_aliases_upstream"
            ),
            output_tensor_hash=_string(payload, "output_tensor_hash"),
            input_gradient_hash=_string(payload, "input_gradient_hash"),
            cache_event=_string(payload, "cache_event"),
            forward_launch_count=_integer(payload, "forward_launch_count"),
            backward_launch_count=_integer(payload, "backward_launch_count"),
            fallback_count=_integer(payload, "fallback_count"),
            eager_backward_count=_integer(payload, "eager_backward_count"),
            performance_claimed=_boolean(payload, "performance_claimed"),
        )
        value.validate_against(template, instance, schedule, module)
        return value

    def stable_hash(
        self,
        template: DifferentiableLowerTIRTemplateV1,
        instance: DifferentiableLowerTIRInstanceV1,
        schedule: DifferentiableLowerTIRScheduleV1,
        module: DifferentiableLowerTIRModuleReceiptV1,
    ) -> str:
        return canonical_tir_hash(self.to_dict(template, instance, schedule, module))


__all__ = [
    "DIFFERENTIABLE_LOWER_TIR_INSTANCE_SCHEMA",
    "DIFFERENTIABLE_LOWER_TIR_LAUNCH_RECEIPT_SCHEMA",
    "DIFFERENTIABLE_LOWER_TIR_MODULE_RECEIPT_SCHEMA",
    "DIFFERENTIABLE_LOWER_TIR_SCHEDULE_SCHEMA",
    "DIFFERENTIABLE_LOWER_TIR_TEMPLATE_SCHEMA",
    "IDENTITY_BACKWARD_SYMBOL",
    "IDENTITY_FORWARD_SYMBOL",
    "FROZEN_TVM_COMMIT",
    "FROZEN_TVM_FFI_COMMIT",
    "DifferentiableLowerTIRInstanceV1",
    "DifferentiableLowerTIRLaunchReceiptV1",
    "DifferentiableLowerTIRModuleReceiptV1",
    "DifferentiableLowerTIRScheduleV1",
    "DifferentiableLowerTIRTemplateV1",
    "canonical_tir_hash",
]
