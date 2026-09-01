"""First-class B4-B2 B2-4 sparse-source P-anchor Conv TIR receipts."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=missing-class-docstring

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .differentiable_lower_dense_conv_tir import DENSE_CONV_WORKSPACE_INVENTORY
from .differentiable_lower_tir import (
    FROZEN_TVM_COMMIT,
    FROZEN_TVM_FFI_COMMIT,
    canonical_tir_hash,
)

SPARSE_CONV_TEMPLATE_SCHEMA = "boundflow.differentiable-lower-sparse-conv-template/v1"
SPARSE_CONV_INSTANCE_SCHEMA = "boundflow.differentiable-lower-sparse-conv-instance/v1"
SPARSE_CONV_SCHEDULE_SCHEMA = "boundflow.differentiable-lower-sparse-conv-schedule/v1"
SPARSE_CONV_MODULE_SCHEMA = "boundflow.differentiable-lower-sparse-conv-module/v1"
SPARSE_CONV_PROJECTION_SCHEMA = (
    "boundflow.differentiable-lower-sparse-conv-projection/v1"
)
SPARSE_CONV_LAUNCH_SCHEMA = "boundflow.differentiable-lower-sparse-conv-launch/v1"
SPARSE_CONV_LEDGER_SCHEMA = "boundflow.differentiable-lower-sparse-conv-ledger/v1"
SPARSE_CONV_FORWARD_SYMBOL = "boundflow_b4b2_sparse_conv_forward"
SPARSE_CONV_BACKWARD_SYMBOL = "boundflow_b4b2_sparse_conv_backward"
SPARSE_CONV_INPUT_NAMES = tuple(
    sorted(
        (
            "compressed_alpha",
            "incoming_lower_a",
            "incoming_lower_bias",
            "operator_bias",
            "operator_weight",
            "output_bias_gradient",
            "output_lower_a_gradient",
            "preactivation_lower",
            "preactivation_upper",
        )
    )
)
SPARSE_CONV_OUTPUT_NAMES = tuple(
    sorted(
        (
            "compressed_alpha_gradient",
            "incoming_lower_a_gradient",
            "output_bias",
            "output_lower_a",
        )
    )
)
SPARSE_CONV_CANDIDATE_KNOBS = (
    (128, 16, 1, 1),
    (256, 16, 1, 1),
    (128, 8, 1, 1),
    (256, 8, 1, 1),
    (128, 4, 1, 1),
    (256, 4, 1, 1),
    (128, 16, 2, 1),
    (256, 16, 2, 1),
    (128, 8, 2, 1),
    (256, 8, 2, 1),
    (128, 16, 1, 3),
    (256, 16, 1, 3),
)


def _is_hex(value: object, length: int = 64) -> bool:
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


def _integer_tuple(payload: Mapping[str, object], name: str) -> tuple[int, ...]:
    value = payload.get(name)
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{name} must be an integer list")
    return tuple(value)


def _string_pairs(
    payload: Mapping[str, object], name: str
) -> tuple[tuple[str, str], ...]:
    value = payload.get(name)
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise ValueError(f"{name} must be a string map")
    return tuple(sorted(value.items()))


def _pointer_pairs(
    payload: Mapping[str, object], name: str
) -> tuple[tuple[str, int], ...]:
    value = payload.get(name)
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, int) or isinstance(item, bool)
        for key, item in value.items()
    ):
        raise ValueError(f"{name} must be an integer map")
    return tuple(sorted(value.items()))


def _workspace_inventory(
    payload: Mapping[str, object], name: str
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    value = payload.get(name)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a workspace list")
    result = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"name", "shape"}:
            raise ValueError(f"{name} must be a workspace list")
        buffer_name = item.get("name")
        shape = item.get("shape")
        if (
            not isinstance(buffer_name, str)
            or not isinstance(shape, list)
            or any(
                not isinstance(dimension, int) or isinstance(dimension, bool)
                for dimension in shape
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
class DifferentiableLowerSparseConvTIRTemplateV1:
    """Frozen compressed-alpha, empty-beta P-anchor compiler template."""

    lower_region_ir_hash: str
    dense_template_hash: str
    operator_attributes_hash: str
    alpha_coordinate_hash: str
    alpha_channels: tuple[int, ...]
    alpha_heights: tuple[int, ...]
    alpha_widths: tuple[int, ...]
    domain_count: int = 6
    spec_count: int = 1
    channels: int = 16
    height: int = 8
    width: int = 8
    kernel_height: int = 3
    kernel_width: int = 3
    compressed_alpha_features: int = 86
    compressed_beta_entries: int = 0
    anchor_id: str = "performance-conv-8-candidate"
    abi: str = "sparse-source-conv-transpose-fused-v1"
    stride: tuple[int, ...] = (1, 1)
    padding: tuple[int, ...] = (1, 1)
    dilation: tuple[int, ...] = (1, 1)
    output_padding: tuple[int, ...] = (0, 0)
    groups: int = 1
    dtype: str = "torch.float32"
    device_kind: str = "cuda"
    target: str = "cuda"
    compute_capability: str = "sm_89"
    gradient_targets: tuple[str, ...] = ("compressed_alpha", "incoming_lower_a")
    mapping_ownership: str = "plan-template-constant"
    mapping_inline: bool = True
    enabled_by_default: bool = False
    sparse_source_admitted: bool = True
    performance_admitted: bool = False
    forward_symbol: str = SPARSE_CONV_FORWARD_SYMBOL
    backward_symbol: str = SPARSE_CONV_BACKWARD_SYMBOL
    schema_version: str = SPARSE_CONV_TEMPLATE_SCHEMA

    @property
    def alpha_coordinates(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(zip(self.alpha_channels, self.alpha_heights, self.alpha_widths))

    def validate(self) -> None:
        coordinates = self.alpha_coordinates
        if (
            self.schema_version != SPARSE_CONV_TEMPLATE_SCHEMA
            or any(
                not _is_hex(value)
                for value in (
                    self.lower_region_ir_hash,
                    self.dense_template_hash,
                    self.operator_attributes_hash,
                    self.alpha_coordinate_hash,
                )
            )
            or (self.domain_count, self.spec_count) != (6, 1)
            or (self.channels, self.height, self.width) != (16, 8, 8)
            or (self.kernel_height, self.kernel_width) != (3, 3)
            or self.compressed_alpha_features != 86
            or self.compressed_beta_entries != 0
            or any(
                len(axis) != 86
                for axis in (self.alpha_channels, self.alpha_heights, self.alpha_widths)
            )
            or len(set(coordinates)) != 86
            or any(
                channel not in range(16)
                or height not in range(8)
                or width not in range(8)
                for channel, height, width in coordinates
            )
            or self.alpha_coordinate_hash
            != canonical_tir_hash([list(item) for item in coordinates])
            or self.anchor_id != "performance-conv-8-candidate"
            or self.abi != "sparse-source-conv-transpose-fused-v1"
            or self.stride != (1, 1)
            or self.padding != (1, 1)
            or self.dilation != (1, 1)
            or self.output_padding != (0, 0)
            or self.groups != 1
            or self.dtype != "torch.float32"
            or self.device_kind != "cuda"
            or self.target != "cuda"
            or self.compute_capability != "sm_89"
            or self.gradient_targets != ("compressed_alpha", "incoming_lower_a")
            or self.mapping_ownership != "plan-template-constant"
            or self.mapping_inline is not True
            or self.enabled_by_default is not False
            or self.sparse_source_admitted is not True
            or self.performance_admitted is not False
            or self.forward_symbol != SPARSE_CONV_FORWARD_SYMBOL
            or self.backward_symbol != SPARSE_CONV_BACKWARD_SYMBOL
        ):
            raise ValueError("sparse-source Conv TIR template differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "lower_region_ir_hash": self.lower_region_ir_hash,
            "dense_template_hash": self.dense_template_hash,
            "operator_attributes_hash": self.operator_attributes_hash,
            "alpha_coordinate_hash": self.alpha_coordinate_hash,
            "alpha_channels": list(self.alpha_channels),
            "alpha_heights": list(self.alpha_heights),
            "alpha_widths": list(self.alpha_widths),
            "domain_count": self.domain_count,
            "spec_count": self.spec_count,
            "channels": self.channels,
            "height": self.height,
            "width": self.width,
            "kernel_height": self.kernel_height,
            "kernel_width": self.kernel_width,
            "compressed_alpha_features": self.compressed_alpha_features,
            "compressed_beta_entries": self.compressed_beta_entries,
            "anchor_id": self.anchor_id,
            "abi": self.abi,
            "stride": list(self.stride),
            "padding": list(self.padding),
            "dilation": list(self.dilation),
            "output_padding": list(self.output_padding),
            "groups": self.groups,
            "dtype": self.dtype,
            "device_kind": self.device_kind,
            "target": self.target,
            "compute_capability": self.compute_capability,
            "gradient_targets": list(self.gradient_targets),
            "mapping_ownership": self.mapping_ownership,
            "mapping_inline": self.mapping_inline,
            "enabled_by_default": self.enabled_by_default,
            "sparse_source_admitted": self.sparse_source_admitted,
            "performance_admitted": self.performance_admitted,
            "forward_symbol": self.forward_symbol,
            "backward_symbol": self.backward_symbol,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "DifferentiableLowerSparseConvTIRTemplateV1":
        targets = payload.get("gradient_targets")
        if not isinstance(targets, list) or any(
            not isinstance(item, str) for item in targets
        ):
            raise ValueError("gradient_targets must be a string list")
        value = cls(
            schema_version=_string(payload, "schema_version"),
            lower_region_ir_hash=_string(payload, "lower_region_ir_hash"),
            dense_template_hash=_string(payload, "dense_template_hash"),
            operator_attributes_hash=_string(payload, "operator_attributes_hash"),
            alpha_coordinate_hash=_string(payload, "alpha_coordinate_hash"),
            alpha_channels=_integer_tuple(payload, "alpha_channels"),
            alpha_heights=_integer_tuple(payload, "alpha_heights"),
            alpha_widths=_integer_tuple(payload, "alpha_widths"),
            domain_count=_integer(payload, "domain_count"),
            spec_count=_integer(payload, "spec_count"),
            channels=_integer(payload, "channels"),
            height=_integer(payload, "height"),
            width=_integer(payload, "width"),
            kernel_height=_integer(payload, "kernel_height"),
            kernel_width=_integer(payload, "kernel_width"),
            compressed_alpha_features=_integer(payload, "compressed_alpha_features"),
            compressed_beta_entries=_integer(payload, "compressed_beta_entries"),
            anchor_id=_string(payload, "anchor_id"),
            abi=_string(payload, "abi"),
            stride=_integer_tuple(payload, "stride"),
            padding=_integer_tuple(payload, "padding"),
            dilation=_integer_tuple(payload, "dilation"),
            output_padding=_integer_tuple(payload, "output_padding"),
            groups=_integer(payload, "groups"),
            dtype=_string(payload, "dtype"),
            device_kind=_string(payload, "device_kind"),
            target=_string(payload, "target"),
            compute_capability=_string(payload, "compute_capability"),
            gradient_targets=tuple(targets),
            mapping_ownership=_string(payload, "mapping_ownership"),
            mapping_inline=_boolean(payload, "mapping_inline"),
            enabled_by_default=_boolean(payload, "enabled_by_default"),
            sparse_source_admitted=_boolean(payload, "sparse_source_admitted"),
            performance_admitted=_boolean(payload, "performance_admitted"),
            forward_symbol=_string(payload, "forward_symbol"),
            backward_symbol=_string(payload, "backward_symbol"),
        )
        value.validate()
        return value

    def stable_hash(self) -> str:
        return canonical_tir_hash(self.to_dict())


@dataclass(frozen=True)
class DifferentiableLowerSparseConvTIRInstanceV1:
    template_hash: str
    lower_region_instance_hash: str
    reference_capture_hash: str
    tensor_hashes: tuple[tuple[str, str], ...]
    fresh_run_ordinal: int
    device_ordinal: int
    schema_version: str = SPARSE_CONV_INSTANCE_SCHEMA

    def validate_against(
        self, template: DifferentiableLowerSparseConvTIRTemplateV1
    ) -> None:
        template.validate()
        hashes = dict(self.tensor_hashes)
        if (
            self.schema_version != SPARSE_CONV_INSTANCE_SCHEMA
            or self.template_hash != template.stable_hash()
            or not _is_hex(self.lower_region_instance_hash)
            or not _is_hex(self.reference_capture_hash)
            or tuple(sorted(hashes)) != SPARSE_CONV_INPUT_NAMES
            or len(hashes) != len(self.tensor_hashes)
            or any(not _is_hex(value) for value in hashes.values())
            or self.fresh_run_ordinal not in range(5)
            or self.device_ordinal < 0
        ):
            raise ValueError("sparse-source Conv TIR instance differs")

    def to_dict(
        self, template: DifferentiableLowerSparseConvTIRTemplateV1
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
        template: DifferentiableLowerSparseConvTIRTemplateV1,
    ):
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

    def stable_hash(self, template: DifferentiableLowerSparseConvTIRTemplateV1) -> str:
        return canonical_tir_hash(self.to_dict(template))


@dataclass(frozen=True)
class DifferentiableLowerSparseConvTIRScheduleV1:
    template_hash: str
    candidate_ordinal: int
    thread_extent: int
    output_channel_tile: int
    spatial_tile: int
    reduction_unroll: int
    mapping_inline: bool = True
    schedule_family: str = "sparse-source-conv-bounded-ledger-v1"
    workspace_inventory: tuple[tuple[str, tuple[int, ...]], ...] = (
        DENSE_CONV_WORKSPACE_INVENTORY
    )
    deterministic: bool = True
    sparse_source_admitted: bool = True
    performance_admitted: bool = False
    schema_version: str = SPARSE_CONV_SCHEDULE_SCHEMA

    @property
    def knob_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.thread_extent,
            self.output_channel_tile,
            self.spatial_tile,
            self.reduction_unroll,
        )

    def validate_against(
        self, template: DifferentiableLowerSparseConvTIRTemplateV1
    ) -> None:
        template.validate()
        if (
            self.schema_version != SPARSE_CONV_SCHEDULE_SCHEMA
            or self.template_hash != template.stable_hash()
            or self.candidate_ordinal not in range(len(SPARSE_CONV_CANDIDATE_KNOBS))
            or self.knob_tuple != SPARSE_CONV_CANDIDATE_KNOBS[self.candidate_ordinal]
            or self.thread_extent not in {128, 256}
            or self.output_channel_tile not in {4, 8, 16}
            or self.spatial_tile not in {1, 2}
            or self.reduction_unroll not in {1, 3}
            or self.mapping_inline is not True
            or self.schedule_family != "sparse-source-conv-bounded-ledger-v1"
            or self.workspace_inventory != DENSE_CONV_WORKSPACE_INVENTORY
            or self.deterministic is not True
            or self.sparse_source_admitted is not True
            or self.performance_admitted is not False
        ):
            raise ValueError("sparse-source Conv TIR schedule differs")

    def to_dict(
        self, template: DifferentiableLowerSparseConvTIRTemplateV1
    ) -> dict[str, object]:
        self.validate_against(template)
        return {
            "schema_version": self.schema_version,
            "template_hash": self.template_hash,
            "candidate_ordinal": self.candidate_ordinal,
            "thread_extent": self.thread_extent,
            "output_channel_tile": self.output_channel_tile,
            "spatial_tile": self.spatial_tile,
            "reduction_unroll": self.reduction_unroll,
            "mapping_inline": self.mapping_inline,
            "schedule_family": self.schedule_family,
            "workspace_inventory": _workspace_payload(self.workspace_inventory),
            "deterministic": self.deterministic,
            "sparse_source_admitted": self.sparse_source_admitted,
            "performance_admitted": self.performance_admitted,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        template: DifferentiableLowerSparseConvTIRTemplateV1,
    ):
        value = cls(
            schema_version=_string(payload, "schema_version"),
            template_hash=_string(payload, "template_hash"),
            candidate_ordinal=_integer(payload, "candidate_ordinal"),
            thread_extent=_integer(payload, "thread_extent"),
            output_channel_tile=_integer(payload, "output_channel_tile"),
            spatial_tile=_integer(payload, "spatial_tile"),
            reduction_unroll=_integer(payload, "reduction_unroll"),
            mapping_inline=_boolean(payload, "mapping_inline"),
            schedule_family=_string(payload, "schedule_family"),
            workspace_inventory=_workspace_inventory(payload, "workspace_inventory"),
            deterministic=_boolean(payload, "deterministic"),
            sparse_source_admitted=_boolean(payload, "sparse_source_admitted"),
            performance_admitted=_boolean(payload, "performance_admitted"),
        )
        value.validate_against(template)
        return value

    def stable_hash(self, template: DifferentiableLowerSparseConvTIRTemplateV1) -> str:
        return canonical_tir_hash(self.to_dict(template))


@dataclass(frozen=True)
class DifferentiableLowerSparseConvCandidateLedgerV1:
    template_hash: str
    schedule_hashes: tuple[str, ...]
    generated_before_timing: bool = True
    timing_raw_present: bool = False
    winner_selected: bool = False
    performance_claimed: bool = False
    schema_version: str = SPARSE_CONV_LEDGER_SCHEMA

    def validate_against(
        self,
        template: DifferentiableLowerSparseConvTIRTemplateV1,
        schedules: tuple[DifferentiableLowerSparseConvTIRScheduleV1, ...],
    ) -> None:
        for schedule in schedules:
            schedule.validate_against(template)
        expected = tuple(schedule.stable_hash(template) for schedule in schedules)
        if (
            self.schema_version != SPARSE_CONV_LEDGER_SCHEMA
            or self.template_hash != template.stable_hash()
            or len(schedules) != 12
            or tuple(schedule.candidate_ordinal for schedule in schedules)
            != tuple(range(12))
            or self.schedule_hashes != expected
            or len(set(self.schedule_hashes)) != 12
            or any(not _is_hex(value) for value in self.schedule_hashes)
            or self.generated_before_timing is not True
            or self.timing_raw_present is not False
            or self.winner_selected is not False
            or self.performance_claimed is not False
        ):
            raise ValueError("sparse-source Conv candidate ledger differs")

    def to_dict(
        self,
        template: DifferentiableLowerSparseConvTIRTemplateV1,
        schedules: tuple[DifferentiableLowerSparseConvTIRScheduleV1, ...],
    ) -> dict[str, object]:
        self.validate_against(template, schedules)
        return {
            "schema_version": self.schema_version,
            "template_hash": self.template_hash,
            "schedule_hashes": list(self.schedule_hashes),
            "generated_before_timing": self.generated_before_timing,
            "timing_raw_present": self.timing_raw_present,
            "winner_selected": self.winner_selected,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(
        self,
        template: DifferentiableLowerSparseConvTIRTemplateV1,
        schedules: tuple[DifferentiableLowerSparseConvTIRScheduleV1, ...],
    ) -> str:
        return canonical_tir_hash(self.to_dict(template, schedules))

    @classmethod
    def from_dict(cls, payload, template, schedules):
        hashes = payload.get("schedule_hashes")
        if not isinstance(hashes, list) or any(
            not isinstance(item, str) for item in hashes
        ):
            raise ValueError("schedule_hashes must be a string list")
        value = cls(
            schema_version=_string(payload, "schema_version"),
            template_hash=_string(payload, "template_hash"),
            schedule_hashes=tuple(hashes),
            generated_before_timing=_boolean(payload, "generated_before_timing"),
            timing_raw_present=_boolean(payload, "timing_raw_present"),
            winner_selected=_boolean(payload, "winner_selected"),
            performance_claimed=_boolean(payload, "performance_claimed"),
        )
        value.validate_against(template, schedules)
        return value


@dataclass(frozen=True)
class DifferentiableLowerSparseConvTIRModuleReceiptV1:
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
    sparse_source_admitted: bool = True
    performance_claimed: bool = False
    schema_version: str = SPARSE_CONV_MODULE_SCHEMA

    @staticmethod
    def expected_cache_key(template, schedule) -> str:
        return canonical_tir_hash(
            {
                "schema": SPARSE_CONV_MODULE_SCHEMA,
                "template_hash": template.stable_hash(),
                "schedule_hash": schedule.stable_hash(template),
                "symbols": [template.forward_symbol, template.backward_symbol],
                "target": template.target,
                "compute_capability": template.compute_capability,
                "tvm_commit": FROZEN_TVM_COMMIT,
                "tvm_ffi_commit": FROZEN_TVM_FFI_COMMIT,
            }
        )

    def validate_against(self, template, schedule) -> None:
        schedule.validate_against(template)
        if (
            self.schema_version != SPARSE_CONV_MODULE_SCHEMA
            or self.template_hash != template.stable_hash()
            or self.schedule_hash != schedule.stable_hash(template)
            or any(
                not _is_hex(value)
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
            or self.sparse_source_admitted is not True
            or self.performance_claimed is not False
        ):
            raise ValueError("sparse-source Conv TIR module receipt differs")

    def to_dict(self, template, schedule) -> dict[str, object]:
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
            "sparse_source_admitted": self.sparse_source_admitted,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(self, template, schedule) -> str:
        return canonical_tir_hash(self.to_dict(template, schedule))

    @classmethod
    def from_dict(cls, payload, template, schedule):
        symbols = payload.get("exported_symbols")
        if not isinstance(symbols, list) or any(
            not isinstance(item, str) for item in symbols
        ):
            raise ValueError("exported_symbols must be a string list")
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
            exported_symbols=tuple(symbols),
            observed_workspace_inventory=_workspace_inventory(
                payload, "observed_workspace_inventory"
            ),
            structural_workspace_check=_boolean(payload, "structural_workspace_check"),
            tvm_commit=_string(payload, "tvm_commit"),
            tvm_ffi_commit=_string(payload, "tvm_ffi_commit"),
            sparse_source_admitted=_boolean(payload, "sparse_source_admitted"),
            performance_claimed=_boolean(payload, "performance_claimed"),
        )
        value.validate_against(template, schedule)
        return value


@dataclass(frozen=True)
class DifferentiableLowerSparseConvGradientProjectionReceiptV1:
    template_hash: str
    instance_hash: str
    reference_native_alpha_gradient_hash: str
    reference_compressed_alpha_gradient_hash: str
    candidate_compressed_alpha_gradient_hash: str
    projected_native_alpha_gradient_hash: str
    alpha_owned_element_count: int
    coordinate_mapping_exact: bool
    alpha_numerical_passed: bool
    nonzero_sign_exact: bool
    unowned_native_zero_exact: bool
    beta_gradient_absent: bool
    schema_version: str = SPARSE_CONV_PROJECTION_SCHEMA

    def validate_against(self, template, instance) -> None:
        instance.validate_against(template)
        hashes = (
            self.reference_native_alpha_gradient_hash,
            self.reference_compressed_alpha_gradient_hash,
            self.candidate_compressed_alpha_gradient_hash,
            self.projected_native_alpha_gradient_hash,
        )
        if (
            self.schema_version != SPARSE_CONV_PROJECTION_SCHEMA
            or self.template_hash != template.stable_hash()
            or self.instance_hash != instance.stable_hash(template)
            or any(not _is_hex(value) for value in hashes)
            or self.alpha_owned_element_count != 6 * 86
            or self.coordinate_mapping_exact is not True
            or self.alpha_numerical_passed is not True
            or self.nonzero_sign_exact is not True
            or self.unowned_native_zero_exact is not True
            or self.beta_gradient_absent is not True
        ):
            raise ValueError("sparse-source Conv gradient projection differs")

    def to_dict(self, template, instance) -> dict[str, object]:
        self.validate_against(template, instance)
        return {"schema_version": self.schema_version, **self.__dict__}

    def stable_hash(self, template, instance) -> str:
        return canonical_tir_hash(self.to_dict(template, instance))

    @classmethod
    def from_dict(cls, payload, template, instance):
        value = cls(
            schema_version=_string(payload, "schema_version"),
            template_hash=_string(payload, "template_hash"),
            instance_hash=_string(payload, "instance_hash"),
            reference_native_alpha_gradient_hash=_string(
                payload, "reference_native_alpha_gradient_hash"
            ),
            reference_compressed_alpha_gradient_hash=_string(
                payload, "reference_compressed_alpha_gradient_hash"
            ),
            candidate_compressed_alpha_gradient_hash=_string(
                payload, "candidate_compressed_alpha_gradient_hash"
            ),
            projected_native_alpha_gradient_hash=_string(
                payload, "projected_native_alpha_gradient_hash"
            ),
            alpha_owned_element_count=_integer(payload, "alpha_owned_element_count"),
            coordinate_mapping_exact=_boolean(payload, "coordinate_mapping_exact"),
            alpha_numerical_passed=_boolean(payload, "alpha_numerical_passed"),
            nonzero_sign_exact=_boolean(payload, "nonzero_sign_exact"),
            unowned_native_zero_exact=_boolean(payload, "unowned_native_zero_exact"),
            beta_gradient_absent=_boolean(payload, "beta_gradient_absent"),
        )
        value.validate_against(template, instance)
        return value


@dataclass(frozen=True)
class DifferentiableLowerSparseConvTIRLaunchReceiptV1:
    template_hash: str
    instance_hash: str
    schedule_hash: str
    module_receipt_hash: str
    projection_receipt_hash: str
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
    semantic_passed: bool
    beta_gradient_present: bool = False
    sparse_source_admitted: bool = True
    performance_claimed: bool = False
    schema_version: str = SPARSE_CONV_LAUNCH_SCHEMA

    def validate_against(
        self, template, instance, schedule, module, projection
    ) -> None:
        instance.validate_against(template)
        module.validate_against(template, schedule)
        projection.validate_against(template, instance)
        inputs = dict(self.input_data_ptrs)
        outputs = dict(self.output_data_ptrs)
        hashes = dict(self.output_tensor_hashes)
        if (
            self.schema_version != SPARSE_CONV_LAUNCH_SCHEMA
            or self.template_hash != template.stable_hash()
            or self.instance_hash != instance.stable_hash(template)
            or self.schedule_hash != schedule.stable_hash(template)
            or self.module_receipt_hash != module.stable_hash(template, schedule)
            or self.projection_receipt_hash
            != projection.stable_hash(template, instance)
            or self.stream_id < 0
            or self.tvm_ffi_stream_id != self.stream_id
            or tuple(sorted(inputs)) != SPARSE_CONV_INPUT_NAMES
            or tuple(sorted(outputs)) != SPARSE_CONV_OUTPUT_NAMES
            or tuple(sorted(hashes)) != SPARSE_CONV_OUTPUT_NAMES
            or any(pointer <= 0 for pointer in (*inputs.values(), *outputs.values()))
            or set(inputs.values()) & set(outputs.values())
            or any(not _is_hex(value) for value in hashes.values())
            or self.dlpack_pointer_count != 19
            or self.dlpack_pointer_exact_count != 19
            or self.cache_event not in {"hit", "miss"}
            or (self.forward_launch_count, self.backward_launch_count) != (1, 1)
            or (self.fallback_count, self.eager_backward_count) != (0, 0)
            or self.semantic_passed is not True
            or self.beta_gradient_present is not False
            or self.sparse_source_admitted is not True
            or self.performance_claimed is not False
        ):
            raise ValueError("sparse-source Conv TIR launch receipt differs")

    def to_dict(
        self, template, instance, schedule, module, projection
    ) -> dict[str, object]:
        self.validate_against(template, instance, schedule, module, projection)
        return {
            "schema_version": self.schema_version,
            **self.__dict__,
            "input_data_ptrs": dict(self.input_data_ptrs),
            "output_data_ptrs": dict(self.output_data_ptrs),
            "output_tensor_hashes": dict(self.output_tensor_hashes),
        }

    @classmethod
    def from_dict(cls, payload, template, instance, schedule, module, projection):
        value = cls(
            schema_version=_string(payload, "schema_version"),
            template_hash=_string(payload, "template_hash"),
            instance_hash=_string(payload, "instance_hash"),
            schedule_hash=_string(payload, "schedule_hash"),
            module_receipt_hash=_string(payload, "module_receipt_hash"),
            projection_receipt_hash=_string(payload, "projection_receipt_hash"),
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
            semantic_passed=_boolean(payload, "semantic_passed"),
            beta_gradient_present=_boolean(payload, "beta_gradient_present"),
            sparse_source_admitted=_boolean(payload, "sparse_source_admitted"),
            performance_claimed=_boolean(payload, "performance_claimed"),
        )
        value.validate_against(template, instance, schedule, module, projection)
        return value

    def stable_hash(self, template, instance, schedule, module, projection) -> str:
        return canonical_tir_hash(
            self.to_dict(template, instance, schedule, module, projection)
        )


__all__ = [
    "SPARSE_CONV_BACKWARD_SYMBOL",
    "SPARSE_CONV_CANDIDATE_KNOBS",
    "SPARSE_CONV_FORWARD_SYMBOL",
    "SPARSE_CONV_INPUT_NAMES",
    "SPARSE_CONV_OUTPUT_NAMES",
    "DifferentiableLowerSparseConvCandidateLedgerV1",
    "DifferentiableLowerSparseConvGradientProjectionReceiptV1",
    "DifferentiableLowerSparseConvTIRInstanceV1",
    "DifferentiableLowerSparseConvTIRLaunchReceiptV1",
    "DifferentiableLowerSparseConvTIRModuleReceiptV1",
    "DifferentiableLowerSparseConvTIRScheduleV1",
    "DifferentiableLowerSparseConvTIRTemplateV1",
]
