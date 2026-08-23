"""First-class B4-B2 B2-3 dense ConvTranspose TIR receipts."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,too-many-arguments
# pylint: disable=too-many-positional-arguments

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .differentiable_lower_tir import (
    FROZEN_TVM_COMMIT,
    FROZEN_TVM_FFI_COMMIT,
    canonical_tir_hash,
)

DENSE_CONV_TEMPLATE_SCHEMA = "boundflow.differentiable-lower-dense-conv-template/v1"
DENSE_CONV_INSTANCE_SCHEMA = "boundflow.differentiable-lower-dense-conv-instance/v1"
DENSE_CONV_SCHEDULE_SCHEMA = "boundflow.differentiable-lower-dense-conv-schedule/v1"
DENSE_CONV_MODULE_SCHEMA = "boundflow.differentiable-lower-dense-conv-module/v1"
DENSE_CONV_LAUNCH_SCHEMA = "boundflow.differentiable-lower-dense-conv-launch/v1"
DENSE_CONV_FORWARD_SYMBOL = "boundflow_b4b2_dense_conv_forward"
DENSE_CONV_BACKWARD_SYMBOL = "boundflow_b4b2_dense_conv_backward"
DENSE_CONV_INPUT_NAMES = tuple(
    sorted(
        (
            "incoming_lower_a",
            "incoming_lower_bias",
            "native_alpha",
            "operator_bias",
            "operator_weight",
            "output_bias_gradient",
            "output_lower_a_gradient",
            "preactivation_lower",
            "preactivation_upper",
        )
    )
)
DENSE_CONV_OUTPUT_NAMES = tuple(
    sorted(
        (
            "incoming_lower_a_gradient",
            "native_alpha_gradient",
            "output_bias",
            "output_lower_a",
        )
    )
)
DENSE_CONV_WORKSPACE_INVENTORY = (
    ("adjoint_conv", (6, 1, 16, 8, 8)),
    ("output_bias_delta", (6, 1)),
)


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
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


def _string_pairs(
    payload: Mapping[str, object], name: str
) -> tuple[tuple[str, str], ...]:
    raw = payload.get(name)
    if not isinstance(raw, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw.items()
    ):
        raise ValueError(f"{name} must be a string map")
    return tuple(sorted(raw.items()))


def _pointer_pairs(
    payload: Mapping[str, object], name: str
) -> tuple[tuple[str, int], ...]:
    raw = payload.get(name)
    if not isinstance(raw, dict) or any(
        not isinstance(key, str)
        or not isinstance(value, int)
        or isinstance(value, bool)
        for key, value in raw.items()
    ):
        raise ValueError(f"{name} must be an integer map")
    return tuple(sorted(raw.items()))


def _integer_tuple(payload: Mapping[str, object], name: str) -> tuple[int, ...]:
    raw = payload.get(name)
    if not isinstance(raw, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in raw
    ):
        raise ValueError(f"{name} must be an integer list")
    return tuple(raw)


def _workspace_inventory(
    payload: Mapping[str, object], name: str
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    raw = payload.get(name)
    if not isinstance(raw, list):
        raise ValueError(f"{name} must be a workspace list")
    result: list[tuple[str, tuple[int, ...]]] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"name", "shape"}:
            raise ValueError(f"{name} must be a workspace list")
        buffer_name = item.get("name")
        shape = item.get("shape")
        if (
            not isinstance(buffer_name, str)
            or not isinstance(shape, list)
            or any(
                not isinstance(value, int) or isinstance(value, bool) for value in shape
            )
        ):
            raise ValueError(f"{name} must be a workspace list")
        result.append((buffer_name, tuple(shape)))
    return tuple(result)


def _workspace_payload(
    inventory: tuple[tuple[str, tuple[int, ...]], ...],
) -> list[dict[str, object]]:
    return [{"name": name, "shape": list(shape)} for name, shape in inventory]


@dataclass(frozen=True)
class DifferentiableLowerDenseConvTIRTemplateV1:
    """Frozen P-anchor dense ConvTranspose correctness template."""

    lower_region_ir_hash: str
    mapping_layout_hash: str
    operator_attributes_hash: str
    domain_count: int = 6
    spec_count: int = 1
    channels: int = 16
    height: int = 8
    width: int = 8
    kernel_height: int = 3
    kernel_width: int = 3
    anchor_id: str = "performance-conv-8-candidate"
    abi: str = "dense-conv-transpose-semantic-v1"
    stride: tuple[int, ...] = (1, 1)
    padding: tuple[int, ...] = (1, 1)
    dilation: tuple[int, ...] = (1, 1)
    output_padding: tuple[int, ...] = (0, 0)
    groups: int = 1
    compressed_beta_shape: tuple[int, ...] = (6, 0)
    beta_gradient_present: bool = False
    dtype: str = "torch.float32"
    device_kind: str = "cuda"
    target: str = "cuda"
    compute_capability: str = "sm_89"
    gradient_targets: tuple[str, ...] = ("incoming_lower_a", "native_alpha")
    enabled_by_default: bool = False
    performance_admitted: bool = False
    forward_symbol: str = DENSE_CONV_FORWARD_SYMBOL
    backward_symbol: str = DENSE_CONV_BACKWARD_SYMBOL
    schema_version: str = DENSE_CONV_TEMPLATE_SCHEMA

    def validate(self) -> None:
        if (
            self.schema_version != DENSE_CONV_TEMPLATE_SCHEMA
            or any(
                not _is_hex(value, 64)
                for value in (
                    self.lower_region_ir_hash,
                    self.mapping_layout_hash,
                    self.operator_attributes_hash,
                )
            )
            or (self.domain_count, self.spec_count) != (6, 1)
            or (self.channels, self.height, self.width) != (16, 8, 8)
            or (self.kernel_height, self.kernel_width) != (3, 3)
            or self.anchor_id != "performance-conv-8-candidate"
            or self.abi != "dense-conv-transpose-semantic-v1"
            or self.stride != (1, 1)
            or self.padding != (1, 1)
            or self.dilation != (1, 1)
            or self.output_padding != (0, 0)
            or self.groups != 1
            or self.compressed_beta_shape != (6, 0)
            or self.beta_gradient_present is not False
            or self.dtype != "torch.float32"
            or self.device_kind != "cuda"
            or self.target != "cuda"
            or self.compute_capability != "sm_89"
            or self.gradient_targets != ("incoming_lower_a", "native_alpha")
            or self.enabled_by_default is not False
            or self.performance_admitted is not False
            or self.forward_symbol != DENSE_CONV_FORWARD_SYMBOL
            or self.backward_symbol != DENSE_CONV_BACKWARD_SYMBOL
        ):
            raise ValueError("dense Conv TIR template differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "lower_region_ir_hash": self.lower_region_ir_hash,
            "mapping_layout_hash": self.mapping_layout_hash,
            "operator_attributes_hash": self.operator_attributes_hash,
            "domain_count": self.domain_count,
            "spec_count": self.spec_count,
            "channels": self.channels,
            "height": self.height,
            "width": self.width,
            "kernel_height": self.kernel_height,
            "kernel_width": self.kernel_width,
            "anchor_id": self.anchor_id,
            "abi": self.abi,
            "stride": list(self.stride),
            "padding": list(self.padding),
            "dilation": list(self.dilation),
            "output_padding": list(self.output_padding),
            "groups": self.groups,
            "compressed_beta_shape": list(self.compressed_beta_shape),
            "beta_gradient_present": self.beta_gradient_present,
            "dtype": self.dtype,
            "device_kind": self.device_kind,
            "target": self.target,
            "compute_capability": self.compute_capability,
            "gradient_targets": list(self.gradient_targets),
            "enabled_by_default": self.enabled_by_default,
            "performance_admitted": self.performance_admitted,
            "forward_symbol": self.forward_symbol,
            "backward_symbol": self.backward_symbol,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "DifferentiableLowerDenseConvTIRTemplateV1":
        targets = payload.get("gradient_targets")
        if not isinstance(targets, list) or any(
            not isinstance(v, str) for v in targets
        ):
            raise ValueError("gradient_targets must be a string list")
        value = cls(
            schema_version=_string(payload, "schema_version"),
            lower_region_ir_hash=_string(payload, "lower_region_ir_hash"),
            mapping_layout_hash=_string(payload, "mapping_layout_hash"),
            operator_attributes_hash=_string(payload, "operator_attributes_hash"),
            domain_count=_integer(payload, "domain_count"),
            spec_count=_integer(payload, "spec_count"),
            channels=_integer(payload, "channels"),
            height=_integer(payload, "height"),
            width=_integer(payload, "width"),
            kernel_height=_integer(payload, "kernel_height"),
            kernel_width=_integer(payload, "kernel_width"),
            anchor_id=_string(payload, "anchor_id"),
            abi=_string(payload, "abi"),
            stride=_integer_tuple(payload, "stride"),
            padding=_integer_tuple(payload, "padding"),
            dilation=_integer_tuple(payload, "dilation"),
            output_padding=_integer_tuple(payload, "output_padding"),
            groups=_integer(payload, "groups"),
            compressed_beta_shape=_integer_tuple(payload, "compressed_beta_shape"),
            beta_gradient_present=_boolean(payload, "beta_gradient_present"),
            dtype=_string(payload, "dtype"),
            device_kind=_string(payload, "device_kind"),
            target=_string(payload, "target"),
            compute_capability=_string(payload, "compute_capability"),
            gradient_targets=tuple(targets),
            enabled_by_default=_boolean(payload, "enabled_by_default"),
            performance_admitted=_boolean(payload, "performance_admitted"),
            forward_symbol=_string(payload, "forward_symbol"),
            backward_symbol=_string(payload, "backward_symbol"),
        )
        value.validate()
        return value

    def stable_hash(self) -> str:
        return canonical_tir_hash(self.to_dict())


@dataclass(frozen=True)
class DifferentiableLowerDenseConvTIRInstanceV1:
    """Dynamic P-anchor tensor identity for one frozen raw capture."""

    template_hash: str
    lower_region_instance_hash: str
    reference_capture_hash: str
    tensor_hashes: tuple[tuple[str, str], ...]
    fresh_run_ordinal: int
    device_ordinal: int
    schema_version: str = DENSE_CONV_INSTANCE_SCHEMA

    @property
    def tensor_hash_map(self) -> dict[str, str]:
        return dict(self.tensor_hashes)

    def validate_against(
        self, template: DifferentiableLowerDenseConvTIRTemplateV1
    ) -> None:
        template.validate()
        hashes = self.tensor_hash_map
        if (
            self.schema_version != DENSE_CONV_INSTANCE_SCHEMA
            or self.template_hash != template.stable_hash()
            or not _is_hex(self.lower_region_instance_hash, 64)
            or not _is_hex(self.reference_capture_hash, 64)
            or tuple(sorted(hashes)) != DENSE_CONV_INPUT_NAMES
            or len(hashes) != len(self.tensor_hashes)
            or any(not _is_hex(value, 64) for value in hashes.values())
            or self.fresh_run_ordinal not in range(5)
            or self.device_ordinal < 0
        ):
            raise ValueError("dense Conv TIR instance differs")

    def to_dict(
        self, template: DifferentiableLowerDenseConvTIRTemplateV1
    ) -> dict[str, object]:
        self.validate_against(template)
        return {
            "schema_version": self.schema_version,
            "template_hash": self.template_hash,
            "lower_region_instance_hash": self.lower_region_instance_hash,
            "reference_capture_hash": self.reference_capture_hash,
            "tensor_hashes": dict(self.tensor_hashes),
            "fresh_run_ordinal": self.fresh_run_ordinal,
            "device_ordinal": self.device_ordinal,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        template: DifferentiableLowerDenseConvTIRTemplateV1,
    ) -> "DifferentiableLowerDenseConvTIRInstanceV1":
        value = cls(
            schema_version=_string(payload, "schema_version"),
            template_hash=_string(payload, "template_hash"),
            lower_region_instance_hash=_string(payload, "lower_region_instance_hash"),
            reference_capture_hash=_string(payload, "reference_capture_hash"),
            tensor_hashes=_string_pairs(payload, "tensor_hashes"),
            fresh_run_ordinal=_integer(payload, "fresh_run_ordinal"),
            device_ordinal=_integer(payload, "device_ordinal"),
        )
        value.validate_against(template)
        return value

    def stable_hash(self, template: DifferentiableLowerDenseConvTIRTemplateV1) -> str:
        return canonical_tir_hash(self.to_dict(template))


@dataclass(frozen=True)
class DifferentiableLowerDenseConvTIRScheduleV1:
    """Deterministic Conv correctness schedule with structural workspace policy."""

    template_hash: str
    thread_extent: int = 128
    schedule_family: str = "dense-conv-transpose-serial-reduction-v1"
    workspace_inventory: tuple[tuple[str, tuple[int, ...]], ...] = (
        DENSE_CONV_WORKSPACE_INVENTORY
    )
    deterministic: bool = True
    candidate_ordinal: int = 0
    performance_admitted: bool = False
    schema_version: str = DENSE_CONV_SCHEDULE_SCHEMA

    def validate_against(
        self, template: DifferentiableLowerDenseConvTIRTemplateV1
    ) -> None:
        template.validate()
        if (
            self.schema_version != DENSE_CONV_SCHEDULE_SCHEMA
            or self.template_hash != template.stable_hash()
            or self.thread_extent != 128
            or self.schedule_family != "dense-conv-transpose-serial-reduction-v1"
            or self.workspace_inventory != DENSE_CONV_WORKSPACE_INVENTORY
            or self.deterministic is not True
            or self.candidate_ordinal != 0
            or self.performance_admitted is not False
        ):
            raise ValueError("dense Conv TIR schedule differs")

    def to_dict(
        self, template: DifferentiableLowerDenseConvTIRTemplateV1
    ) -> dict[str, object]:
        self.validate_against(template)
        return {
            "schema_version": self.schema_version,
            "template_hash": self.template_hash,
            "thread_extent": self.thread_extent,
            "schedule_family": self.schedule_family,
            "workspace_inventory": _workspace_payload(self.workspace_inventory),
            "deterministic": self.deterministic,
            "candidate_ordinal": self.candidate_ordinal,
            "performance_admitted": self.performance_admitted,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        template: DifferentiableLowerDenseConvTIRTemplateV1,
    ) -> "DifferentiableLowerDenseConvTIRScheduleV1":
        value = cls(
            schema_version=_string(payload, "schema_version"),
            template_hash=_string(payload, "template_hash"),
            thread_extent=_integer(payload, "thread_extent"),
            schedule_family=_string(payload, "schedule_family"),
            workspace_inventory=_workspace_inventory(payload, "workspace_inventory"),
            deterministic=_boolean(payload, "deterministic"),
            candidate_ordinal=_integer(payload, "candidate_ordinal"),
            performance_admitted=_boolean(payload, "performance_admitted"),
        )
        value.validate_against(template)
        return value

    def stable_hash(self, template: DifferentiableLowerDenseConvTIRTemplateV1) -> str:
        return canonical_tir_hash(self.to_dict(template))


@dataclass(frozen=True)
class DifferentiableLowerDenseConvTIRModuleReceiptV1:
    """Compiled Conv module and structurally observed alloc-buffer inventory."""

    template_hash: str
    schedule_hash: str
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    cache_key: str
    tvm_version: str
    torch_version: str
    exported_symbols: tuple[str, ...]
    observed_workspace_inventory: tuple[tuple[str, tuple[int, ...]], ...]
    structural_workspace_check: bool
    tvm_commit: str = FROZEN_TVM_COMMIT
    tvm_ffi_commit: str = FROZEN_TVM_FFI_COMMIT
    performance_claimed: bool = False
    schema_version: str = DENSE_CONV_MODULE_SCHEMA

    @staticmethod
    def expected_cache_key(
        template: DifferentiableLowerDenseConvTIRTemplateV1,
        schedule: DifferentiableLowerDenseConvTIRScheduleV1,
    ) -> str:
        return canonical_tir_hash(
            {
                "schema": DENSE_CONV_MODULE_SCHEMA,
                "template_hash": template.stable_hash(),
                "schedule_hash": schedule.stable_hash(template),
                "symbols": [template.forward_symbol, template.backward_symbol],
                "target": template.target,
                "compute_capability": template.compute_capability,
                "tvm_commit": FROZEN_TVM_COMMIT,
                "tvm_ffi_commit": FROZEN_TVM_FFI_COMMIT,
            }
        )

    def validate_against(
        self,
        template: DifferentiableLowerDenseConvTIRTemplateV1,
        schedule: DifferentiableLowerDenseConvTIRScheduleV1,
    ) -> None:
        schedule.validate_against(template)
        if (
            self.schema_version != DENSE_CONV_MODULE_SCHEMA
            or self.template_hash != template.stable_hash()
            or self.schedule_hash != schedule.stable_hash(template)
            or any(
                not _is_hex(value, 64)
                for value in (
                    self.unscheduled_tir_hash,
                    self.scheduled_tir_hash,
                    self.device_source_hash,
                    self.cache_key,
                )
            )
            or self.cache_key != self.expected_cache_key(template, schedule)
            or not self.tvm_version
            or not self.torch_version
            or self.exported_symbols
            != (template.forward_symbol, template.backward_symbol)
            or self.observed_workspace_inventory != schedule.workspace_inventory
            or self.structural_workspace_check is not True
            or self.tvm_commit != FROZEN_TVM_COMMIT
            or self.tvm_ffi_commit != FROZEN_TVM_FFI_COMMIT
            or self.performance_claimed is not False
        ):
            raise ValueError("dense Conv TIR module receipt differs")

    def to_dict(
        self,
        template: DifferentiableLowerDenseConvTIRTemplateV1,
        schedule: DifferentiableLowerDenseConvTIRScheduleV1,
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
            "tvm_version": self.tvm_version,
            "torch_version": self.torch_version,
            "exported_symbols": list(self.exported_symbols),
            "observed_workspace_inventory": _workspace_payload(
                self.observed_workspace_inventory
            ),
            "structural_workspace_check": self.structural_workspace_check,
            "tvm_commit": self.tvm_commit,
            "tvm_ffi_commit": self.tvm_ffi_commit,
            "performance_claimed": self.performance_claimed,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        template: DifferentiableLowerDenseConvTIRTemplateV1,
        schedule: DifferentiableLowerDenseConvTIRScheduleV1,
    ) -> "DifferentiableLowerDenseConvTIRModuleReceiptV1":
        symbols = payload.get("exported_symbols")
        if not isinstance(symbols, list):
            raise ValueError("exported_symbols must be a list")
        value = cls(
            schema_version=_string(payload, "schema_version"),
            template_hash=_string(payload, "template_hash"),
            schedule_hash=_string(payload, "schedule_hash"),
            unscheduled_tir_hash=_string(payload, "unscheduled_tir_hash"),
            scheduled_tir_hash=_string(payload, "scheduled_tir_hash"),
            device_source_hash=_string(payload, "device_source_hash"),
            cache_key=_string(payload, "cache_key"),
            tvm_version=_string(payload, "tvm_version"),
            torch_version=_string(payload, "torch_version"),
            exported_symbols=tuple(str(item) for item in symbols),
            observed_workspace_inventory=_workspace_inventory(
                payload, "observed_workspace_inventory"
            ),
            structural_workspace_check=_boolean(payload, "structural_workspace_check"),
            tvm_commit=_string(payload, "tvm_commit"),
            tvm_ffi_commit=_string(payload, "tvm_ffi_commit"),
            performance_claimed=_boolean(payload, "performance_claimed"),
        )
        value.validate_against(template, schedule)
        return value

    def stable_hash(
        self,
        template: DifferentiableLowerDenseConvTIRTemplateV1,
        schedule: DifferentiableLowerDenseConvTIRScheduleV1,
    ) -> str:
        return canonical_tir_hash(self.to_dict(template, schedule))


@dataclass(frozen=True)
class DifferentiableLowerDenseConvTIRLaunchReceiptV1:
    """P-anchor launch, zero-copy, gradient-presence and claim ledger."""

    template_hash: str
    instance_hash: str
    schedule_hash: str
    module_receipt_hash: str
    stream_id: int
    tvm_ffi_stream_id: int
    input_data_ptrs: tuple[tuple[str, int], ...]
    output_data_ptrs: tuple[tuple[str, int], ...]
    output_tensor_hashes: tuple[tuple[str, str], ...]
    dlpack_pointer_exact_count: int
    dlpack_pointer_count: int
    cache_event: str
    forward_launch_count: int
    backward_launch_count: int
    fallback_count: int
    eager_backward_count: int
    incoming_a_gradient_present: bool
    native_alpha_gradient_present: bool
    beta_gradient_present: bool
    semantic_passed: bool
    performance_claimed: bool = False
    schema_version: str = DENSE_CONV_LAUNCH_SCHEMA

    def validate_against(
        self,
        template: DifferentiableLowerDenseConvTIRTemplateV1,
        instance: DifferentiableLowerDenseConvTIRInstanceV1,
        schedule: DifferentiableLowerDenseConvTIRScheduleV1,
        module: DifferentiableLowerDenseConvTIRModuleReceiptV1,
    ) -> None:
        instance.validate_against(template)
        module.validate_against(template, schedule)
        inputs = dict(self.input_data_ptrs)
        outputs = dict(self.output_data_ptrs)
        hashes = dict(self.output_tensor_hashes)
        if (
            self.schema_version != DENSE_CONV_LAUNCH_SCHEMA
            or self.template_hash != template.stable_hash()
            or self.instance_hash != instance.stable_hash(template)
            or self.schedule_hash != schedule.stable_hash(template)
            or self.module_receipt_hash != module.stable_hash(template, schedule)
            or self.stream_id < 0
            or self.tvm_ffi_stream_id != self.stream_id
            or tuple(sorted(inputs)) != DENSE_CONV_INPUT_NAMES
            or tuple(sorted(outputs)) != DENSE_CONV_OUTPUT_NAMES
            or tuple(sorted(hashes)) != DENSE_CONV_OUTPUT_NAMES
            or len(inputs) != len(self.input_data_ptrs)
            or len(outputs) != len(self.output_data_ptrs)
            or len(hashes) != len(self.output_tensor_hashes)
            or any(pointer <= 0 for pointer in (*inputs.values(), *outputs.values()))
            or set(inputs.values()) & set(outputs.values())
            or len(set(outputs.values())) != len(outputs)
            or any(not _is_hex(value, 64) for value in hashes.values())
            or self.dlpack_pointer_count != 19
            or self.dlpack_pointer_exact_count != self.dlpack_pointer_count
            or self.cache_event not in {"hit", "miss"}
            or self.forward_launch_count != 1
            or self.backward_launch_count != 1
            or self.fallback_count != 0
            or self.eager_backward_count != 0
            or self.incoming_a_gradient_present is not True
            or self.native_alpha_gradient_present is not True
            or self.beta_gradient_present is not False
            or self.semantic_passed is not True
            or self.performance_claimed is not False
        ):
            raise ValueError("dense Conv TIR launch receipt differs")

    def to_dict(
        self,
        template: DifferentiableLowerDenseConvTIRTemplateV1,
        instance: DifferentiableLowerDenseConvTIRInstanceV1,
        schedule: DifferentiableLowerDenseConvTIRScheduleV1,
        module: DifferentiableLowerDenseConvTIRModuleReceiptV1,
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
            "input_data_ptrs": dict(self.input_data_ptrs),
            "output_data_ptrs": dict(self.output_data_ptrs),
            "output_tensor_hashes": dict(self.output_tensor_hashes),
            "dlpack_pointer_exact_count": self.dlpack_pointer_exact_count,
            "dlpack_pointer_count": self.dlpack_pointer_count,
            "cache_event": self.cache_event,
            "forward_launch_count": self.forward_launch_count,
            "backward_launch_count": self.backward_launch_count,
            "fallback_count": self.fallback_count,
            "eager_backward_count": self.eager_backward_count,
            "incoming_a_gradient_present": self.incoming_a_gradient_present,
            "native_alpha_gradient_present": self.native_alpha_gradient_present,
            "beta_gradient_present": self.beta_gradient_present,
            "semantic_passed": self.semantic_passed,
            "performance_claimed": self.performance_claimed,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        template: DifferentiableLowerDenseConvTIRTemplateV1,
        instance: DifferentiableLowerDenseConvTIRInstanceV1,
        schedule: DifferentiableLowerDenseConvTIRScheduleV1,
        module: DifferentiableLowerDenseConvTIRModuleReceiptV1,
    ) -> "DifferentiableLowerDenseConvTIRLaunchReceiptV1":
        value = cls(
            schema_version=_string(payload, "schema_version"),
            template_hash=_string(payload, "template_hash"),
            instance_hash=_string(payload, "instance_hash"),
            schedule_hash=_string(payload, "schedule_hash"),
            module_receipt_hash=_string(payload, "module_receipt_hash"),
            stream_id=_integer(payload, "stream_id"),
            tvm_ffi_stream_id=_integer(payload, "tvm_ffi_stream_id"),
            input_data_ptrs=_pointer_pairs(payload, "input_data_ptrs"),
            output_data_ptrs=_pointer_pairs(payload, "output_data_ptrs"),
            output_tensor_hashes=_string_pairs(payload, "output_tensor_hashes"),
            dlpack_pointer_exact_count=_integer(payload, "dlpack_pointer_exact_count"),
            dlpack_pointer_count=_integer(payload, "dlpack_pointer_count"),
            cache_event=_string(payload, "cache_event"),
            forward_launch_count=_integer(payload, "forward_launch_count"),
            backward_launch_count=_integer(payload, "backward_launch_count"),
            fallback_count=_integer(payload, "fallback_count"),
            eager_backward_count=_integer(payload, "eager_backward_count"),
            incoming_a_gradient_present=_boolean(
                payload, "incoming_a_gradient_present"
            ),
            native_alpha_gradient_present=_boolean(
                payload, "native_alpha_gradient_present"
            ),
            beta_gradient_present=_boolean(payload, "beta_gradient_present"),
            semantic_passed=_boolean(payload, "semantic_passed"),
            performance_claimed=_boolean(payload, "performance_claimed"),
        )
        value.validate_against(template, instance, schedule, module)
        return value

    def stable_hash(
        self,
        template: DifferentiableLowerDenseConvTIRTemplateV1,
        instance: DifferentiableLowerDenseConvTIRInstanceV1,
        schedule: DifferentiableLowerDenseConvTIRScheduleV1,
        module: DifferentiableLowerDenseConvTIRModuleReceiptV1,
    ) -> str:
        return canonical_tir_hash(self.to_dict(template, instance, schedule, module))


__all__ = [
    "DENSE_CONV_BACKWARD_SYMBOL",
    "DENSE_CONV_FORWARD_SYMBOL",
    "DENSE_CONV_INPUT_NAMES",
    "DENSE_CONV_OUTPUT_NAMES",
    "DENSE_CONV_WORKSPACE_INVENTORY",
    "DifferentiableLowerDenseConvTIRInstanceV1",
    "DifferentiableLowerDenseConvTIRLaunchReceiptV1",
    "DifferentiableLowerDenseConvTIRModuleReceiptV1",
    "DifferentiableLowerDenseConvTIRScheduleV1",
    "DifferentiableLowerDenseConvTIRTemplateV1",
]
