"""First-class B4-B2 B2-2 sparse-source Linear TIR receipts."""

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

SPARSE_LINEAR_TEMPLATE_SCHEMA = (
    "boundflow.differentiable-lower-sparse-linear-template/v1"
)
SPARSE_LINEAR_INSTANCE_SCHEMA = (
    "boundflow.differentiable-lower-sparse-linear-instance/v1"
)
SPARSE_LINEAR_SCHEDULE_SCHEMA = (
    "boundflow.differentiable-lower-sparse-linear-schedule/v1"
)
SPARSE_LINEAR_MODULE_SCHEMA = "boundflow.differentiable-lower-sparse-linear-module/v1"
SPARSE_LINEAR_LAUNCH_SCHEMA = "boundflow.differentiable-lower-sparse-linear-launch/v1"
SPARSE_LINEAR_PROJECTION_SCHEMA = (
    "boundflow.differentiable-lower-sparse-linear-projection/v1"
)
SPARSE_LINEAR_FORWARD_SYMBOL = "boundflow_b4b2_sparse_linear_forward"
SPARSE_LINEAR_BACKWARD_SYMBOL = "boundflow_b4b2_sparse_linear_backward"
SPARSE_LINEAR_INPUT_NAMES = tuple(
    sorted(
        (
            "compressed_alpha",
            "compressed_beta",
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
SPARSE_LINEAR_OUTPUT_NAMES = tuple(
    sorted(
        (
            "compressed_alpha_gradient",
            "compressed_beta_gradient",
            "output_bias",
            "output_lower_a",
        )
    )
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


@dataclass(frozen=True)
class DifferentiableLowerSparseLinearTIRTemplateV1:
    """Frozen S-anchor sparse-source compiler template and mapping constants."""

    lower_region_ir_hash: str
    dense_template_hash: str
    operator_attributes_hash: str
    alpha_feature_index_hash: str
    beta_location_hash: str
    beta_sign_hash: str
    alpha_feature_indices: tuple[int, ...]
    beta_locations: tuple[int, ...]
    beta_signs: tuple[int, ...]
    domain_count: int = 6
    spec_count: int = 1
    current_features: int = 100
    previous_features: int = 1024
    compressed_alpha_features: int = 27
    compressed_beta_entries: int = 1
    anchor_id: str = "semantic-active-beta-gemm-14"
    abi: str = "sparse-source-linear-fused-v1"
    dtype: str = "torch.float32"
    device_kind: str = "cuda"
    target: str = "cuda"
    compute_capability: str = "sm_89"
    gradient_targets: tuple[str, ...] = ("compressed_alpha", "compressed_beta")
    mapping_ownership: str = "plan-template-constant"
    enabled_by_default: bool = False
    sparse_source_admitted: bool = True
    performance_admitted: bool = False
    forward_symbol: str = SPARSE_LINEAR_FORWARD_SYMBOL
    backward_symbol: str = SPARSE_LINEAR_BACKWARD_SYMBOL
    schema_version: str = SPARSE_LINEAR_TEMPLATE_SCHEMA

    def validate(self) -> None:
        hashes = (
            self.lower_region_ir_hash,
            self.dense_template_hash,
            self.operator_attributes_hash,
            self.alpha_feature_index_hash,
            self.beta_location_hash,
            self.beta_sign_hash,
        )
        if (
            self.schema_version != SPARSE_LINEAR_TEMPLATE_SCHEMA
            or any(not _is_hex(value, 64) for value in hashes)
            or self.domain_count != 6
            or self.spec_count != 1
            or self.current_features != 100
            or self.previous_features != 1024
            or self.compressed_alpha_features != 27
            or self.compressed_beta_entries != 1
            or len(self.alpha_feature_indices) != self.compressed_alpha_features
            or tuple(sorted(self.alpha_feature_indices)) != self.alpha_feature_indices
            or len(set(self.alpha_feature_indices)) != len(self.alpha_feature_indices)
            or any(
                value not in range(self.current_features)
                for value in self.alpha_feature_indices
            )
            or len(self.beta_locations) != self.domain_count
            or any(
                value not in range(self.current_features)
                for value in self.beta_locations
            )
            or len(self.beta_signs) != self.domain_count
            or any(value not in {-1, 1} for value in self.beta_signs)
            or self.alpha_feature_index_hash
            != canonical_tir_hash(list(self.alpha_feature_indices))
            or self.beta_location_hash != canonical_tir_hash(list(self.beta_locations))
            or self.beta_sign_hash != canonical_tir_hash(list(self.beta_signs))
            or self.anchor_id != "semantic-active-beta-gemm-14"
            or self.abi != "sparse-source-linear-fused-v1"
            or self.dtype != "torch.float32"
            or self.device_kind != "cuda"
            or self.target != "cuda"
            or self.compute_capability != "sm_89"
            or self.gradient_targets != ("compressed_alpha", "compressed_beta")
            or self.mapping_ownership != "plan-template-constant"
            or self.enabled_by_default is not False
            or self.sparse_source_admitted is not True
            or self.performance_admitted is not False
            or self.forward_symbol != SPARSE_LINEAR_FORWARD_SYMBOL
            or self.backward_symbol != SPARSE_LINEAR_BACKWARD_SYMBOL
        ):
            raise ValueError("sparse-source Linear TIR template differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "lower_region_ir_hash": self.lower_region_ir_hash,
            "dense_template_hash": self.dense_template_hash,
            "operator_attributes_hash": self.operator_attributes_hash,
            "alpha_feature_index_hash": self.alpha_feature_index_hash,
            "beta_location_hash": self.beta_location_hash,
            "beta_sign_hash": self.beta_sign_hash,
            "alpha_feature_indices": list(self.alpha_feature_indices),
            "beta_locations": list(self.beta_locations),
            "beta_signs": list(self.beta_signs),
            "domain_count": self.domain_count,
            "spec_count": self.spec_count,
            "current_features": self.current_features,
            "previous_features": self.previous_features,
            "compressed_alpha_features": self.compressed_alpha_features,
            "compressed_beta_entries": self.compressed_beta_entries,
            "anchor_id": self.anchor_id,
            "abi": self.abi,
            "dtype": self.dtype,
            "device_kind": self.device_kind,
            "target": self.target,
            "compute_capability": self.compute_capability,
            "gradient_targets": list(self.gradient_targets),
            "mapping_ownership": self.mapping_ownership,
            "enabled_by_default": self.enabled_by_default,
            "sparse_source_admitted": self.sparse_source_admitted,
            "performance_admitted": self.performance_admitted,
            "forward_symbol": self.forward_symbol,
            "backward_symbol": self.backward_symbol,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "DifferentiableLowerSparseLinearTIRTemplateV1":
        targets = payload.get("gradient_targets")
        if not isinstance(targets, list) or any(
            not isinstance(v, str) for v in targets
        ):
            raise ValueError("gradient_targets must be a string list")
        value = cls(
            schema_version=_string(payload, "schema_version"),
            lower_region_ir_hash=_string(payload, "lower_region_ir_hash"),
            dense_template_hash=_string(payload, "dense_template_hash"),
            operator_attributes_hash=_string(payload, "operator_attributes_hash"),
            alpha_feature_index_hash=_string(payload, "alpha_feature_index_hash"),
            beta_location_hash=_string(payload, "beta_location_hash"),
            beta_sign_hash=_string(payload, "beta_sign_hash"),
            alpha_feature_indices=_integer_tuple(payload, "alpha_feature_indices"),
            beta_locations=_integer_tuple(payload, "beta_locations"),
            beta_signs=_integer_tuple(payload, "beta_signs"),
            domain_count=_integer(payload, "domain_count"),
            spec_count=_integer(payload, "spec_count"),
            current_features=_integer(payload, "current_features"),
            previous_features=_integer(payload, "previous_features"),
            compressed_alpha_features=_integer(payload, "compressed_alpha_features"),
            compressed_beta_entries=_integer(payload, "compressed_beta_entries"),
            anchor_id=_string(payload, "anchor_id"),
            abi=_string(payload, "abi"),
            dtype=_string(payload, "dtype"),
            device_kind=_string(payload, "device_kind"),
            target=_string(payload, "target"),
            compute_capability=_string(payload, "compute_capability"),
            gradient_targets=tuple(targets),
            mapping_ownership=_string(payload, "mapping_ownership"),
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
class DifferentiableLowerSparseLinearTIRInstanceV1:
    """Dynamic compressed-source values for one frozen raw capture."""

    template_hash: str
    lower_region_instance_hash: str
    reference_capture_hash: str
    tensor_hashes: tuple[tuple[str, str], ...]
    fresh_run_ordinal: int
    device_ordinal: int
    schema_version: str = SPARSE_LINEAR_INSTANCE_SCHEMA

    @property
    def tensor_hash_map(self) -> dict[str, str]:
        return dict(self.tensor_hashes)

    def validate_against(
        self, template: DifferentiableLowerSparseLinearTIRTemplateV1
    ) -> None:
        template.validate()
        hashes = self.tensor_hash_map
        if (
            self.schema_version != SPARSE_LINEAR_INSTANCE_SCHEMA
            or self.template_hash != template.stable_hash()
            or not _is_hex(self.lower_region_instance_hash, 64)
            or not _is_hex(self.reference_capture_hash, 64)
            or tuple(sorted(hashes)) != SPARSE_LINEAR_INPUT_NAMES
            or len(hashes) != len(self.tensor_hashes)
            or any(not _is_hex(value, 64) for value in hashes.values())
            or self.fresh_run_ordinal not in range(5)
            or self.device_ordinal < 0
        ):
            raise ValueError("sparse-source Linear TIR instance differs")

    def to_dict(
        self, template: DifferentiableLowerSparseLinearTIRTemplateV1
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
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
    ) -> "DifferentiableLowerSparseLinearTIRInstanceV1":
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

    def stable_hash(
        self, template: DifferentiableLowerSparseLinearTIRTemplateV1
    ) -> str:
        return canonical_tir_hash(self.to_dict(template))


@dataclass(frozen=True)
class DifferentiableLowerSparseLinearTIRScheduleV1:
    """Deterministic B2-2 correctness schedule with an explicit workspace ledger."""

    template_hash: str
    thread_extent: int = 128
    schedule_family: str = "sparse-source-linear-serial-reduction-v1"
    workspace_names: tuple[str, ...] = ("adjoint_matmul", "output_bias_delta")
    forbidden_global_workspaces: tuple[str, ...] = (
        "native_alpha",
        "native_beta",
        "relu_lower_a",
        "scaled_a",
    )
    deterministic: bool = True
    candidate_ordinal: int = 0
    performance_admitted: bool = False
    schema_version: str = SPARSE_LINEAR_SCHEDULE_SCHEMA

    def validate_against(
        self, template: DifferentiableLowerSparseLinearTIRTemplateV1
    ) -> None:
        template.validate()
        if (
            self.schema_version != SPARSE_LINEAR_SCHEDULE_SCHEMA
            or self.template_hash != template.stable_hash()
            or self.thread_extent != 128
            or self.schedule_family != "sparse-source-linear-serial-reduction-v1"
            or self.workspace_names != ("adjoint_matmul", "output_bias_delta")
            or self.forbidden_global_workspaces
            != ("native_alpha", "native_beta", "relu_lower_a", "scaled_a")
            or self.deterministic is not True
            or self.candidate_ordinal != 0
            or self.performance_admitted is not False
        ):
            raise ValueError("sparse-source Linear TIR schedule differs")

    def to_dict(
        self, template: DifferentiableLowerSparseLinearTIRTemplateV1
    ) -> dict[str, object]:
        self.validate_against(template)
        return {
            "schema_version": self.schema_version,
            "template_hash": self.template_hash,
            "thread_extent": self.thread_extent,
            "schedule_family": self.schedule_family,
            "workspace_names": list(self.workspace_names),
            "forbidden_global_workspaces": list(self.forbidden_global_workspaces),
            "deterministic": self.deterministic,
            "candidate_ordinal": self.candidate_ordinal,
            "performance_admitted": self.performance_admitted,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
    ) -> "DifferentiableLowerSparseLinearTIRScheduleV1":
        workspaces = payload.get("workspace_names")
        forbidden = payload.get("forbidden_global_workspaces")
        if not isinstance(workspaces, list) or not isinstance(forbidden, list):
            raise ValueError("workspace inventory must be a list")
        value = cls(
            schema_version=_string(payload, "schema_version"),
            template_hash=_string(payload, "template_hash"),
            thread_extent=_integer(payload, "thread_extent"),
            schedule_family=_string(payload, "schedule_family"),
            workspace_names=tuple(str(item) for item in workspaces),
            forbidden_global_workspaces=tuple(str(item) for item in forbidden),
            deterministic=_boolean(payload, "deterministic"),
            candidate_ordinal=_integer(payload, "candidate_ordinal"),
            performance_admitted=_boolean(payload, "performance_admitted"),
        )
        value.validate_against(template)
        return value

    def stable_hash(
        self, template: DifferentiableLowerSparseLinearTIRTemplateV1
    ) -> str:
        return canonical_tir_hash(self.to_dict(template))


@dataclass(frozen=True)
class DifferentiableLowerSparseLinearTIRModuleReceiptV1:
    """Hash-bound sparse-source module and exact scheduled workspace inventory."""

    template_hash: str
    schedule_hash: str
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    cache_key: str
    tvm_version: str
    torch_version: str
    exported_symbols: tuple[str, ...]
    observed_workspace_names: tuple[str, ...]
    forbidden_workspace_count: int
    tvm_commit: str = FROZEN_TVM_COMMIT
    tvm_ffi_commit: str = FROZEN_TVM_FFI_COMMIT
    sparse_source_admitted: bool = True
    performance_claimed: bool = False
    schema_version: str = SPARSE_LINEAR_MODULE_SCHEMA

    @staticmethod
    def expected_cache_key(
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
        schedule: DifferentiableLowerSparseLinearTIRScheduleV1,
    ) -> str:
        return canonical_tir_hash(
            {
                "schema": SPARSE_LINEAR_MODULE_SCHEMA,
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
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
        schedule: DifferentiableLowerSparseLinearTIRScheduleV1,
    ) -> None:
        schedule.validate_against(template)
        if (
            self.schema_version != SPARSE_LINEAR_MODULE_SCHEMA
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
            or self.observed_workspace_names != schedule.workspace_names
            or self.forbidden_workspace_count != 0
            or self.tvm_commit != FROZEN_TVM_COMMIT
            or self.tvm_ffi_commit != FROZEN_TVM_FFI_COMMIT
            or self.sparse_source_admitted is not True
            or self.performance_claimed is not False
        ):
            raise ValueError("sparse-source Linear TIR module receipt differs")

    def to_dict(
        self,
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
        schedule: DifferentiableLowerSparseLinearTIRScheduleV1,
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
            "observed_workspace_names": list(self.observed_workspace_names),
            "forbidden_workspace_count": self.forbidden_workspace_count,
            "tvm_commit": self.tvm_commit,
            "tvm_ffi_commit": self.tvm_ffi_commit,
            "sparse_source_admitted": self.sparse_source_admitted,
            "performance_claimed": self.performance_claimed,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
        schedule: DifferentiableLowerSparseLinearTIRScheduleV1,
    ) -> "DifferentiableLowerSparseLinearTIRModuleReceiptV1":
        symbols = payload.get("exported_symbols")
        workspaces = payload.get("observed_workspace_names")
        if not isinstance(symbols, list) or not isinstance(workspaces, list):
            raise ValueError("module inventory must be a list")
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
            observed_workspace_names=tuple(str(item) for item in workspaces),
            forbidden_workspace_count=_integer(payload, "forbidden_workspace_count"),
            tvm_commit=_string(payload, "tvm_commit"),
            tvm_ffi_commit=_string(payload, "tvm_ffi_commit"),
            sparse_source_admitted=_boolean(payload, "sparse_source_admitted"),
            performance_claimed=_boolean(payload, "performance_claimed"),
        )
        value.validate_against(template, schedule)
        return value

    def stable_hash(
        self,
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
        schedule: DifferentiableLowerSparseLinearTIRScheduleV1,
    ) -> str:
        return canonical_tir_hash(self.to_dict(template, schedule))


@dataclass(frozen=True)
class DifferentiableLowerSparseLinearGradientProjectionReceiptV1:
    """Compressed-gradient gather/scatter equality against the B4-B1 native oracle."""

    template_hash: str
    instance_hash: str
    reference_native_alpha_gradient_hash: str
    reference_native_beta_gradient_hash: str
    reference_compressed_alpha_gradient_hash: str
    reference_compressed_beta_gradient_hash: str
    candidate_compressed_alpha_gradient_hash: str
    candidate_compressed_beta_gradient_hash: str
    projected_native_alpha_gradient_hash: str
    projected_native_beta_gradient_hash: str
    alpha_owned_element_count: int
    beta_owned_element_count: int
    alpha_mapping_exact: bool
    beta_mapping_exact: bool
    alpha_numerical_passed: bool
    beta_numerical_passed: bool
    nonzero_sign_exact: bool
    unowned_native_zero_exact: bool
    schema_version: str = SPARSE_LINEAR_PROJECTION_SCHEMA

    def validate_against(
        self,
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
        instance: DifferentiableLowerSparseLinearTIRInstanceV1,
    ) -> None:
        instance.validate_against(template)
        hashes = (
            self.reference_native_alpha_gradient_hash,
            self.reference_native_beta_gradient_hash,
            self.reference_compressed_alpha_gradient_hash,
            self.reference_compressed_beta_gradient_hash,
            self.candidate_compressed_alpha_gradient_hash,
            self.candidate_compressed_beta_gradient_hash,
            self.projected_native_alpha_gradient_hash,
            self.projected_native_beta_gradient_hash,
        )
        if (
            self.schema_version != SPARSE_LINEAR_PROJECTION_SCHEMA
            or self.template_hash != template.stable_hash()
            or self.instance_hash != instance.stable_hash(template)
            or any(not _is_hex(value, 64) for value in hashes)
            or self.alpha_owned_element_count != 6 * 27
            or self.beta_owned_element_count != 6
            or self.alpha_mapping_exact is not True
            or self.beta_mapping_exact is not True
            or self.alpha_numerical_passed is not True
            or self.beta_numerical_passed is not True
            or self.nonzero_sign_exact is not True
            or self.unowned_native_zero_exact is not True
        ):
            raise ValueError("sparse-source Linear gradient projection differs")

    def to_dict(
        self,
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
        instance: DifferentiableLowerSparseLinearTIRInstanceV1,
    ) -> dict[str, object]:
        self.validate_against(template, instance)
        return {
            "schema_version": self.schema_version,
            "template_hash": self.template_hash,
            "instance_hash": self.instance_hash,
            "reference_native_alpha_gradient_hash": self.reference_native_alpha_gradient_hash,
            "reference_native_beta_gradient_hash": self.reference_native_beta_gradient_hash,
            "reference_compressed_alpha_gradient_hash": (
                self.reference_compressed_alpha_gradient_hash
            ),
            "reference_compressed_beta_gradient_hash": (
                self.reference_compressed_beta_gradient_hash
            ),
            "candidate_compressed_alpha_gradient_hash": (
                self.candidate_compressed_alpha_gradient_hash
            ),
            "candidate_compressed_beta_gradient_hash": (
                self.candidate_compressed_beta_gradient_hash
            ),
            "projected_native_alpha_gradient_hash": self.projected_native_alpha_gradient_hash,
            "projected_native_beta_gradient_hash": self.projected_native_beta_gradient_hash,
            "alpha_owned_element_count": self.alpha_owned_element_count,
            "beta_owned_element_count": self.beta_owned_element_count,
            "alpha_mapping_exact": self.alpha_mapping_exact,
            "beta_mapping_exact": self.beta_mapping_exact,
            "alpha_numerical_passed": self.alpha_numerical_passed,
            "beta_numerical_passed": self.beta_numerical_passed,
            "nonzero_sign_exact": self.nonzero_sign_exact,
            "unowned_native_zero_exact": self.unowned_native_zero_exact,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
        instance: DifferentiableLowerSparseLinearTIRInstanceV1,
    ) -> "DifferentiableLowerSparseLinearGradientProjectionReceiptV1":
        value = cls(
            schema_version=_string(payload, "schema_version"),
            template_hash=_string(payload, "template_hash"),
            instance_hash=_string(payload, "instance_hash"),
            reference_native_alpha_gradient_hash=_string(
                payload, "reference_native_alpha_gradient_hash"
            ),
            reference_native_beta_gradient_hash=_string(
                payload, "reference_native_beta_gradient_hash"
            ),
            reference_compressed_alpha_gradient_hash=_string(
                payload, "reference_compressed_alpha_gradient_hash"
            ),
            reference_compressed_beta_gradient_hash=_string(
                payload, "reference_compressed_beta_gradient_hash"
            ),
            candidate_compressed_alpha_gradient_hash=_string(
                payload, "candidate_compressed_alpha_gradient_hash"
            ),
            candidate_compressed_beta_gradient_hash=_string(
                payload, "candidate_compressed_beta_gradient_hash"
            ),
            projected_native_alpha_gradient_hash=_string(
                payload, "projected_native_alpha_gradient_hash"
            ),
            projected_native_beta_gradient_hash=_string(
                payload, "projected_native_beta_gradient_hash"
            ),
            alpha_owned_element_count=_integer(payload, "alpha_owned_element_count"),
            beta_owned_element_count=_integer(payload, "beta_owned_element_count"),
            alpha_mapping_exact=_boolean(payload, "alpha_mapping_exact"),
            beta_mapping_exact=_boolean(payload, "beta_mapping_exact"),
            alpha_numerical_passed=_boolean(payload, "alpha_numerical_passed"),
            beta_numerical_passed=_boolean(payload, "beta_numerical_passed"),
            nonzero_sign_exact=_boolean(payload, "nonzero_sign_exact"),
            unowned_native_zero_exact=_boolean(payload, "unowned_native_zero_exact"),
        )
        value.validate_against(template, instance)
        return value

    def stable_hash(
        self,
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
        instance: DifferentiableLowerSparseLinearTIRInstanceV1,
    ) -> str:
        return canonical_tir_hash(self.to_dict(template, instance))


@dataclass(frozen=True)
class DifferentiableLowerSparseLinearTIRLaunchReceiptV1:
    """Sparse-source launch, zero-copy, stream, counter and claim ledger."""

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
    sparse_source_admitted: bool = True
    performance_claimed: bool = False
    schema_version: str = SPARSE_LINEAR_LAUNCH_SCHEMA

    def validate_against(
        self,
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
        instance: DifferentiableLowerSparseLinearTIRInstanceV1,
        schedule: DifferentiableLowerSparseLinearTIRScheduleV1,
        module: DifferentiableLowerSparseLinearTIRModuleReceiptV1,
        projection: DifferentiableLowerSparseLinearGradientProjectionReceiptV1,
    ) -> None:
        instance.validate_against(template)
        module.validate_against(template, schedule)
        projection.validate_against(template, instance)
        inputs = dict(self.input_data_ptrs)
        outputs = dict(self.output_data_ptrs)
        hashes = dict(self.output_tensor_hashes)
        if (
            self.schema_version != SPARSE_LINEAR_LAUNCH_SCHEMA
            or self.template_hash != template.stable_hash()
            or self.instance_hash != instance.stable_hash(template)
            or self.schedule_hash != schedule.stable_hash(template)
            or self.module_receipt_hash != module.stable_hash(template, schedule)
            or self.projection_receipt_hash
            != projection.stable_hash(template, instance)
            or self.stream_id < 0
            or self.tvm_ffi_stream_id != self.stream_id
            or tuple(sorted(inputs)) != SPARSE_LINEAR_INPUT_NAMES
            or tuple(sorted(outputs)) != SPARSE_LINEAR_OUTPUT_NAMES
            or tuple(sorted(hashes)) != SPARSE_LINEAR_OUTPUT_NAMES
            or len(inputs) != len(self.input_data_ptrs)
            or len(outputs) != len(self.output_data_ptrs)
            or len(hashes) != len(self.output_tensor_hashes)
            or any(pointer <= 0 for pointer in (*inputs.values(), *outputs.values()))
            or set(inputs.values()) & set(outputs.values())
            or len(set(outputs.values())) != len(outputs)
            or any(not _is_hex(value, 64) for value in hashes.values())
            or self.dlpack_pointer_count != 21
            or self.dlpack_pointer_exact_count != self.dlpack_pointer_count
            or self.cache_event not in {"hit", "miss"}
            or self.forward_launch_count != 1
            or self.backward_launch_count != 1
            or self.fallback_count != 0
            or self.eager_backward_count != 0
            or self.semantic_passed is not True
            or self.sparse_source_admitted is not True
            or self.performance_claimed is not False
        ):
            raise ValueError("sparse-source Linear TIR launch receipt differs")

    def to_dict(
        self,
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
        instance: DifferentiableLowerSparseLinearTIRInstanceV1,
        schedule: DifferentiableLowerSparseLinearTIRScheduleV1,
        module: DifferentiableLowerSparseLinearTIRModuleReceiptV1,
        projection: DifferentiableLowerSparseLinearGradientProjectionReceiptV1,
    ) -> dict[str, object]:
        self.validate_against(template, instance, schedule, module, projection)
        return {
            "schema_version": self.schema_version,
            "template_hash": self.template_hash,
            "instance_hash": self.instance_hash,
            "schedule_hash": self.schedule_hash,
            "module_receipt_hash": self.module_receipt_hash,
            "projection_receipt_hash": self.projection_receipt_hash,
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
            "semantic_passed": self.semantic_passed,
            "sparse_source_admitted": self.sparse_source_admitted,
            "performance_claimed": self.performance_claimed,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
        instance: DifferentiableLowerSparseLinearTIRInstanceV1,
        schedule: DifferentiableLowerSparseLinearTIRScheduleV1,
        module: DifferentiableLowerSparseLinearTIRModuleReceiptV1,
        projection: DifferentiableLowerSparseLinearGradientProjectionReceiptV1,
    ) -> "DifferentiableLowerSparseLinearTIRLaunchReceiptV1":
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
            sparse_source_admitted=_boolean(payload, "sparse_source_admitted"),
            performance_claimed=_boolean(payload, "performance_claimed"),
        )
        value.validate_against(template, instance, schedule, module, projection)
        return value

    def stable_hash(
        self,
        template: DifferentiableLowerSparseLinearTIRTemplateV1,
        instance: DifferentiableLowerSparseLinearTIRInstanceV1,
        schedule: DifferentiableLowerSparseLinearTIRScheduleV1,
        module: DifferentiableLowerSparseLinearTIRModuleReceiptV1,
        projection: DifferentiableLowerSparseLinearGradientProjectionReceiptV1,
    ) -> str:
        return canonical_tir_hash(
            self.to_dict(template, instance, schedule, module, projection)
        )


__all__ = [
    "SPARSE_LINEAR_BACKWARD_SYMBOL",
    "SPARSE_LINEAR_FORWARD_SYMBOL",
    "SPARSE_LINEAR_INPUT_NAMES",
    "SPARSE_LINEAR_OUTPUT_NAMES",
    "DifferentiableLowerSparseLinearGradientProjectionReceiptV1",
    "DifferentiableLowerSparseLinearTIRInstanceV1",
    "DifferentiableLowerSparseLinearTIRLaunchReceiptV1",
    "DifferentiableLowerSparseLinearTIRModuleReceiptV1",
    "DifferentiableLowerSparseLinearTIRScheduleV1",
    "DifferentiableLowerSparseLinearTIRTemplateV1",
]
