"""Typed RVIR-v4 production optimizer mutation policy contracts."""

# pylint: disable=too-many-boolean-expressions,too-many-instance-attributes
# pylint: disable=too-many-locals,too-many-branches,too-many-statements

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import sys
from typing import Any, cast, Mapping

import torch

from .native_alpha_beta_optimization_state import NativeAlphaBetaOptimizerPolicy
from .rvir_v4_production_state import (
    OwnedProductionTensorV4,
    ProductionOptimizerPolicyV4,
    ProductionTensorOwnership,
    ProductionTensorRole,
    production_tensor_sha256,
)

PRODUCTION_ITERATION_SEMANTICS_V4 = "evaluate-n-update-n-minus-one/v1"
OPTIMIZER_STEP_TRACE_SCHEMA_V4 = "boundflow.rvir-v4-optimizer-step-trace/v1"


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_number(payload: Mapping[str, object], name: str, *, label: str) -> float:
    raw = payload.get(name)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise TypeError(f"RVIR-v4 {label} {name} differs")
    return float(raw)


def _callable_id(value: object) -> str:
    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", type(value).__name__)
    loaded = sys.modules.get(module)
    if loaded is not None:
        aliases = sorted(
            name
            for name, candidate in vars(loaded).items()
            if candidate is value and not name.startswith("_")
        )
        if aliases:
            qualname = aliases[0]
    return f"{module}.{qualname}"


@dataclass(frozen=True)
class ProductionOptimizerControlsV4:
    """Full optimize-bound controls observed at the production core boundary."""

    optimizer: str
    lr_decay: float
    keep_best: bool
    loss_reduction_id: str
    early_stop_patience: int
    start_save_best: float
    use_float64_in_last_iteration: bool
    pruning_in_iteration: bool
    pruning_in_iteration_threshold: float
    max_time: float
    enable_alpha_crown: bool
    enable_beta_crown: bool
    init_alpha: bool
    use_shared_alpha: bool
    apply_output_constraints_to: tuple[str, ...]
    directly_optimize: tuple[str, ...]
    tighten_input_bounds: bool
    cuts_enabled: bool

    def validate(self) -> None:
        """Validate a lossless, finite production controls snapshot."""

        if (
            not self.optimizer
            or not self.loss_reduction_id
            or self.early_stop_patience < 0
            or not 0.0 <= self.start_save_best <= 1.0
            or self.pruning_in_iteration_threshold < 0.0
            or self.max_time <= 0.0
            or not all(
                math.isfinite(value)
                for value in (
                    self.lr_decay,
                    self.start_save_best,
                    self.pruning_in_iteration_threshold,
                    self.max_time,
                )
            )
            or self.lr_decay <= 0.0
        ):
            raise ValueError("RVIR-v4 production optimizer controls differ")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical production controls payload."""

        self.validate()
        return {
            "optimizer": self.optimizer,
            "lr_decay": self.lr_decay,
            "keep_best": self.keep_best,
            "loss_reduction_id": self.loss_reduction_id,
            "early_stop_patience": self.early_stop_patience,
            "start_save_best": self.start_save_best,
            "use_float64_in_last_iteration": self.use_float64_in_last_iteration,
            "pruning_in_iteration": self.pruning_in_iteration,
            "pruning_in_iteration_threshold": self.pruning_in_iteration_threshold,
            "max_time": self.max_time,
            "enable_alpha_crown": self.enable_alpha_crown,
            "enable_beta_crown": self.enable_beta_crown,
            "init_alpha": self.init_alpha,
            "use_shared_alpha": self.use_shared_alpha,
            "apply_output_constraints_to": list(self.apply_output_constraints_to),
            "directly_optimize": list(self.directly_optimize),
            "tighten_input_bounds": self.tighten_input_bounds,
            "cuts_enabled": self.cuts_enabled,
        }

    def stable_hash(self) -> str:
        """Return the full production optimizer controls identity."""

        return _canonical_hash(self.to_dict())


def capture_production_optimizer_controls_v4(
    optimize_bound_args: Mapping[str, Any], *, cuts_enabled: bool
) -> ProductionOptimizerControlsV4:
    """Capture every V4-2-relevant control from a live BoundedModule."""

    required = {
        "optimizer",
        "lr_decay",
        "keep_best",
        "loss_reduction_func",
        "early_stop_patience",
        "start_save_best",
        "use_float64_in_last_iteration",
        "pruning_in_iteration",
        "pruning_in_iteration_threshold",
        "max_time",
        "enable_alpha_crown",
        "enable_beta_crown",
        "init_alpha",
        "use_shared_alpha",
        "apply_output_constraints_to",
        "directly_optimize",
        "tighten_input_bounds",
    }
    if not required.issubset(optimize_bound_args):
        missing = sorted(required - set(optimize_bound_args))
        raise ValueError(f"RVIR-v4 production optimizer controls missing: {missing}")

    bool_fields = (
        "keep_best",
        "use_float64_in_last_iteration",
        "pruning_in_iteration",
        "enable_alpha_crown",
        "enable_beta_crown",
        "init_alpha",
        "use_shared_alpha",
        "tighten_input_bounds",
    )
    if not isinstance(cuts_enabled, bool) or not all(
        isinstance(optimize_bound_args[name], bool) for name in bool_fields
    ):
        raise TypeError("RVIR-v4 production optimizer live boolean fields differ")
    if not isinstance(optimize_bound_args["optimizer"], str):
        raise TypeError("RVIR-v4 production optimizer live name differs")
    if not isinstance(optimize_bound_args["early_stop_patience"], int) or isinstance(
        optimize_bound_args["early_stop_patience"], bool
    ):
        raise TypeError("RVIR-v4 production optimizer live patience differs")

    def number(name: str) -> float:
        raw = optimize_bound_args[name]
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise TypeError(f"RVIR-v4 production optimizer live {name} differs")
        return float(raw)

    def strings(name: str) -> tuple[str, ...]:
        raw = optimize_bound_args[name]
        if not isinstance(raw, (tuple, list)) or not all(
            isinstance(value, str) for value in raw
        ):
            raise TypeError(f"RVIR-v4 production optimizer {name} differs")
        return tuple(raw)

    controls = ProductionOptimizerControlsV4(
        optimizer=optimize_bound_args["optimizer"],
        lr_decay=number("lr_decay"),
        keep_best=optimize_bound_args["keep_best"],
        loss_reduction_id=_callable_id(optimize_bound_args["loss_reduction_func"]),
        early_stop_patience=optimize_bound_args["early_stop_patience"],
        start_save_best=number("start_save_best"),
        use_float64_in_last_iteration=optimize_bound_args[
            "use_float64_in_last_iteration"
        ],
        pruning_in_iteration=optimize_bound_args["pruning_in_iteration"],
        pruning_in_iteration_threshold=number("pruning_in_iteration_threshold"),
        max_time=number("max_time"),
        enable_alpha_crown=optimize_bound_args["enable_alpha_crown"],
        enable_beta_crown=optimize_bound_args["enable_beta_crown"],
        init_alpha=optimize_bound_args["init_alpha"],
        use_shared_alpha=optimize_bound_args["use_shared_alpha"],
        apply_output_constraints_to=strings("apply_output_constraints_to"),
        directly_optimize=strings("directly_optimize"),
        tighten_input_bounds=optimize_bound_args["tighten_input_bounds"],
        cuts_enabled=cuts_enabled,
    )
    controls.validate()
    return controls


def production_optimizer_controls_from_payload_v4(
    payload: Mapping[str, object],
) -> ProductionOptimizerControlsV4:
    """Rebuild and validate a canonical controls payload for replay."""

    expected_keys = {
        "optimizer",
        "lr_decay",
        "keep_best",
        "loss_reduction_id",
        "early_stop_patience",
        "start_save_best",
        "use_float64_in_last_iteration",
        "pruning_in_iteration",
        "pruning_in_iteration_threshold",
        "max_time",
        "enable_alpha_crown",
        "enable_beta_crown",
        "init_alpha",
        "use_shared_alpha",
        "apply_output_constraints_to",
        "directly_optimize",
        "tighten_input_bounds",
        "cuts_enabled",
    }
    if set(payload) != expected_keys:
        raise ValueError("RVIR-v4 optimizer controls payload fields differ")
    bool_fields = (
        "keep_best",
        "use_float64_in_last_iteration",
        "pruning_in_iteration",
        "enable_alpha_crown",
        "enable_beta_crown",
        "init_alpha",
        "use_shared_alpha",
        "tighten_input_bounds",
        "cuts_enabled",
    )
    if not all(isinstance(payload[name], bool) for name in bool_fields):
        raise TypeError("RVIR-v4 optimizer controls boolean fields differ")

    def strings(name: str) -> tuple[str, ...]:
        raw = payload[name]
        if not isinstance(raw, list) or not all(
            isinstance(value, str) for value in raw
        ):
            raise TypeError(f"RVIR-v4 optimizer controls {name} payload differs")
        return tuple(raw)

    def number(name: str) -> float:
        raw = payload[name]
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise TypeError(f"RVIR-v4 optimizer controls {name} payload differs")
        return float(raw)

    raw_patience = payload["early_stop_patience"]
    if not isinstance(raw_patience, int) or isinstance(raw_patience, bool):
        raise TypeError("RVIR-v4 optimizer controls patience payload differs")

    controls = ProductionOptimizerControlsV4(
        optimizer=str(payload["optimizer"]),
        lr_decay=number("lr_decay"),
        keep_best=payload["keep_best"],  # type: ignore[arg-type]
        loss_reduction_id=str(payload["loss_reduction_id"]),
        early_stop_patience=raw_patience,
        start_save_best=number("start_save_best"),
        use_float64_in_last_iteration=payload[  # type: ignore[arg-type]
            "use_float64_in_last_iteration"
        ],
        pruning_in_iteration=payload["pruning_in_iteration"],  # type: ignore[arg-type]
        pruning_in_iteration_threshold=number("pruning_in_iteration_threshold"),
        max_time=number("max_time"),
        enable_alpha_crown=payload["enable_alpha_crown"],  # type: ignore[arg-type]
        enable_beta_crown=payload["enable_beta_crown"],  # type: ignore[arg-type]
        init_alpha=payload["init_alpha"],  # type: ignore[arg-type]
        use_shared_alpha=payload["use_shared_alpha"],  # type: ignore[arg-type]
        apply_output_constraints_to=strings("apply_output_constraints_to"),
        directly_optimize=strings("directly_optimize"),
        tighten_input_bounds=payload["tighten_input_bounds"],  # type: ignore[arg-type]
        cuts_enabled=payload["cuts_enabled"],  # type: ignore[arg-type]
    )
    controls.validate()
    if controls.to_dict() != dict(payload):
        raise ValueError("RVIR-v4 optimizer controls payload canonicalization differs")
    return controls


def production_mutation_policy_from_payload_v4(
    payload: Mapping[str, object],
) -> ProductionMutationPolicyV4:
    """Rebuild the admitted production mutation policy from canonical JSON."""

    expected = {
        "production",
        "controls",
        "iteration_semantics",
        "evaluation_count",
        "update_count",
    }
    if set(payload) != expected:
        raise ValueError("RVIR-v4 mutation policy payload fields differ")
    raw_production = payload["production"]
    raw_controls = payload["controls"]
    if not isinstance(raw_production, Mapping) or not isinstance(raw_controls, Mapping):
        raise TypeError("RVIR-v4 mutation policy payload structure differs")
    production_fields = {
        "iteration",
        "alpha_learning_rate",
        "beta_learning_rate",
        "bound_lower",
        "bound_upper",
        "fix_intermediate_bounds",
        "deterministic",
        "stop_criterion_id",
    }
    if set(raw_production) != production_fields:
        raise ValueError("RVIR-v4 production policy payload fields differ")
    for name in (
        "bound_lower",
        "bound_upper",
        "fix_intermediate_bounds",
        "deterministic",
    ):
        if not isinstance(raw_production[name], bool):
            raise TypeError("RVIR-v4 production policy boolean fields differ")
    iteration = raw_production["iteration"]
    if not isinstance(iteration, int) or isinstance(iteration, bool):
        raise TypeError("RVIR-v4 production policy iteration differs")

    def number(name: str) -> float:
        value = raw_production[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"RVIR-v4 production policy {name} differs")
        return float(value)

    production = ProductionOptimizerPolicyV4(
        iteration=iteration,
        alpha_learning_rate=number("alpha_learning_rate"),
        beta_learning_rate=number("beta_learning_rate"),
        bound_lower=cast(bool, raw_production["bound_lower"]),
        bound_upper=cast(bool, raw_production["bound_upper"]),
        fix_intermediate_bounds=cast(bool, raw_production["fix_intermediate_bounds"]),
        deterministic=cast(bool, raw_production["deterministic"]),
        stop_criterion_id=str(raw_production["stop_criterion_id"]),
    )
    policy = ProductionMutationPolicyV4(
        production=production,
        controls=production_optimizer_controls_from_payload_v4(raw_controls),
        iteration_semantics=str(payload["iteration_semantics"]),
    )
    policy.validate()
    if policy.to_dict() != dict(payload):
        raise ValueError("RVIR-v4 mutation policy payload canonicalization differs")
    return policy


@dataclass(frozen=True)
class ProductionMutationPolicyV4:
    """Admitted lower-only production policy and exact loop cardinalities."""

    production: ProductionOptimizerPolicyV4
    controls: ProductionOptimizerControlsV4
    iteration_semantics: str = PRODUCTION_ITERATION_SEMANTICS_V4

    @property
    def evaluation_count(self) -> int:
        """Number of production bound evaluations."""

        return self.production.iteration

    @property
    def update_count(self) -> int:
        """Number of production backward/optimizer updates."""

        return max(self.production.iteration - 1, 0)

    def validate(self) -> None:
        """Reject policy variants outside the preregistered V4-2 subset."""

        self.production.validate()
        self.controls.validate()
        if (
            self.iteration_semantics != PRODUCTION_ITERATION_SEMANTICS_V4
            or self.production.iteration <= 0
            or self.production.bound_lower is not True
            or self.production.bound_upper is not False
            or self.production.fix_intermediate_bounds is not True
            or "stop_criterion_batch_any" not in self.production.stop_criterion_id
            or self.controls.optimizer != "adam"
            or self.controls.keep_best is not True
            or not self.controls.loss_reduction_id.endswith("reduction_sum")
            or self.controls.use_float64_in_last_iteration is not False
            or self.controls.enable_alpha_crown is not True
            or self.controls.enable_beta_crown is not True
            or self.controls.init_alpha is not False
            or self.controls.use_shared_alpha is not False
            or self.controls.apply_output_constraints_to
            or self.controls.directly_optimize
            or self.controls.tighten_input_bounds is not False
            or self.controls.cuts_enabled is not False
        ):
            raise ValueError("RVIR-v4 production mutation policy is not admitted")
        if self.evaluation_count != self.update_count + 1:
            raise ValueError("RVIR-v4 production optimizer loop cardinality differs")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical mutation-policy payload."""

        self.validate()
        return {
            "production": self.production.to_dict(),
            "controls": self.controls.to_dict(),
            "iteration_semantics": self.iteration_semantics,
            "evaluation_count": self.evaluation_count,
            "update_count": self.update_count,
        }

    def stable_hash(self) -> str:
        """Return the canonical policy and loop-semantics identity."""

        return _canonical_hash(self.to_dict())

    def to_native_policy(self) -> NativeAlphaBetaOptimizerPolicy:
        """Map 10 production evaluations to nine native optimizer updates."""

        self.validate()
        policy = NativeAlphaBetaOptimizerPolicy(
            steps=self.update_count,
            lr=self.production.alpha_learning_rate,
            beta_lr=self.production.beta_learning_rate,
            objective="lower",
            spec_reduce="mean",
        )
        policy.validate()
        return policy


@dataclass(frozen=True)
class ProductionOptimizerStepV4:
    """One production evaluation and its optimizer state before evaluation."""

    core_id: int
    call_id: int
    parent_call_id: int
    evaluation_ordinal: int
    updates_before: int
    update_after: bool
    optimizer_step_ordinal: int | None
    alpha_learning_rate: float
    beta_learning_rate: float
    state_tensors: tuple[OwnedProductionTensorV4, ...]
    lower: torch.Tensor
    lower_sha256: str

    @property
    def tensor_map(self) -> dict[str, OwnedProductionTensorV4]:
        """Return state tensors indexed by stable semantic path."""

        return {tensor.semantic_path: tensor for tensor in self.state_tensors}

    @property
    def state_hash(self) -> str:
        """Hash tensor metadata and content identities for this evaluation."""

        return _canonical_hash([tensor.metadata() for tensor in self.state_tensors])

    def validate(self) -> None:
        """Validate one fixed-workload production evaluation row."""

        if (
            self.core_id < 0
            or self.call_id < 0
            or self.parent_call_id < 0
            or self.evaluation_ordinal < 0
            or self.updates_before != self.evaluation_ordinal
            or self.optimizer_step_ordinal
            != (self.evaluation_ordinal if self.update_after else None)
            or not math.isfinite(self.alpha_learning_rate)
            or not math.isfinite(self.beta_learning_rate)
            or self.alpha_learning_rate <= 0.0
            or self.beta_learning_rate <= 0.0
            or not torch.is_tensor(self.lower)
            or not torch.is_floating_point(self.lower)
            or self.lower.ndim != 2
            or tuple(self.lower.shape) != (6, 1)
            or not bool(torch.isfinite(self.lower).all().item())
            or not _is_sha256(self.lower_sha256)
            or production_tensor_sha256(self.lower) != self.lower_sha256
        ):
            raise ValueError("RVIR-v4 optimizer step identity/result differs")
        for tensor in self.state_tensors:
            tensor.validate()
        paths = [tensor.semantic_path for tensor in self.state_tensors]
        if len(paths) != 24 or len(set(paths)) != 24:
            raise ValueError("RVIR-v4 optimizer step tensor inventory differs")
        role_counts = {
            role: sum(tensor.role == role for tensor in self.state_tensors)
            for role in (
                ProductionTensorRole.ALPHA,
                ProductionTensorRole.BETA_VALUE,
                ProductionTensorRole.BETA_LOCATION,
                ProductionTensorRole.BETA_SIGN,
            )
        }
        if set(tensor.role for tensor in self.state_tensors) != set(role_counts) or any(
            count != 6 for count in role_counts.values()
        ):
            raise ValueError("RVIR-v4 optimizer step tensor roles differ")
        for tensor in self.state_tensors:
            expected_ownership = (
                ProductionTensorOwnership.MUTABLE_COPY_OUT
                if tensor.role
                in {ProductionTensorRole.ALPHA, ProductionTensorRole.BETA_VALUE}
                else ProductionTensorOwnership.COPY_IN
            )
            if tensor.ownership != expected_ownership:
                raise ValueError("RVIR-v4 optimizer step tensor ownership differs")

    def metadata(self) -> dict[str, object]:
        """Return canonical metadata without duplicating raw tensor bytes."""

        self.validate()
        return {
            "core_id": self.core_id,
            "call_id": self.call_id,
            "parent_call_id": self.parent_call_id,
            "evaluation_ordinal": self.evaluation_ordinal,
            "updates_before": self.updates_before,
            "update_after": self.update_after,
            "optimizer_step_ordinal": self.optimizer_step_ordinal,
            "alpha_learning_rate": self.alpha_learning_rate,
            "beta_learning_rate": self.beta_learning_rate,
            "state_tensors": [tensor.metadata() for tensor in self.state_tensors],
            "state_hash": self.state_hash,
            "lower_shape": list(self.lower.shape),
            "lower_dtype": str(self.lower.dtype).removeprefix("torch."),
            "lower_sha256": self.lower_sha256,
        }


@dataclass(frozen=True)
class ProductionOptimizerStepTraceV4:
    """Ten production evaluations and nine mutations for one solver core."""

    mutation_policy: ProductionMutationPolicyV4
    steps: tuple[ProductionOptimizerStepV4, ...]
    schema_version: str = OPTIMIZER_STEP_TRACE_SCHEMA_V4

    def validate(self) -> None:
        """Validate lineage, loop cardinality, schemas, and mutation closure."""

        self.mutation_policy.validate()
        if (
            self.schema_version != OPTIMIZER_STEP_TRACE_SCHEMA_V4
            or len(self.steps) != self.mutation_policy.evaluation_count
        ):
            raise ValueError("RVIR-v4 optimizer step trace cardinality differs")
        first_paths: tuple[str, ...] | None = None
        first_schema: dict[str, tuple[object, ...]] | None = None
        copy_in_hashes: dict[str, str] | None = None
        core_id: int | None = None
        parent_call_id: int | None = None
        previous_call_id = -1
        for ordinal, step in enumerate(self.steps):
            step.validate()
            if (
                step.evaluation_ordinal != ordinal
                or step.updates_before != ordinal
                or step.update_after != (ordinal < self.mutation_policy.update_count)
                or step.call_id <= previous_call_id
            ):
                raise ValueError("RVIR-v4 optimizer step loop semantics differ")
            expected_alpha_lr = (
                self.mutation_policy.production.alpha_learning_rate
                * self.mutation_policy.controls.lr_decay**ordinal
            )
            expected_beta_lr = (
                self.mutation_policy.production.beta_learning_rate
                * self.mutation_policy.controls.lr_decay**ordinal
            )
            if not math.isclose(
                step.alpha_learning_rate, expected_alpha_lr, rel_tol=1e-12
            ) or not math.isclose(
                step.beta_learning_rate, expected_beta_lr, rel_tol=1e-12
            ):
                raise ValueError(
                    "RVIR-v4 optimizer step learning-rate schedule differs"
                )
            previous_call_id = step.call_id
            core_id = step.core_id if core_id is None else core_id
            parent_call_id = (
                step.parent_call_id if parent_call_id is None else parent_call_id
            )
            if step.core_id != core_id or step.parent_call_id != parent_call_id:
                raise ValueError("RVIR-v4 optimizer step lineage differs")
            tensor_map = step.tensor_map
            paths = tuple(sorted(tensor_map))
            schema: dict[str, tuple[object, ...]] = {
                path: (
                    tensor_map[path].role,
                    tensor_map[path].axes,
                    tuple(tensor_map[path].value.shape),
                    tensor_map[path].value.dtype,
                    tensor_map[path].ownership,
                )
                for path in paths
            }
            current_copy_in = {
                path: tensor.content_sha256
                for path, tensor in tensor_map.items()
                if tensor.ownership == ProductionTensorOwnership.COPY_IN
            }
            if first_paths is None:
                first_paths = paths
                first_schema = schema
                copy_in_hashes = current_copy_in
            elif (
                paths != first_paths
                or schema != first_schema
                or current_copy_in != copy_in_hashes
            ):
                raise ValueError("RVIR-v4 optimizer step state schema/copy-in drift")
        for left, right in zip(self.steps, self.steps[1:]):
            changed = sum(
                left.tensor_map[path].content_sha256
                != right.tensor_map[path].content_sha256
                for path in left.tensor_map
                if left.tensor_map[path].ownership
                == ProductionTensorOwnership.MUTABLE_COPY_OUT
            )
            if changed != 7:
                raise ValueError("RVIR-v4 optimizer step mutation count differs")

    def metadata(self) -> dict[str, object]:
        """Return the canonical replay projection for the complete trace."""

        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "mutation_policy": self.mutation_policy.to_dict(),
            "mutation_policy_hash": self.mutation_policy.stable_hash(),
            "evaluation_count": len(self.steps),
            "update_count": sum(step.update_after for step in self.steps),
            "steps": [step.metadata() for step in self.steps],
            "performance_claimed": False,
        }
        payload["trace_hash"] = _canonical_hash(payload)
        return payload


def production_optimizer_step_trace_to_payload_v4(
    trace: ProductionOptimizerStepTraceV4,
) -> dict[str, object]:
    """Serialize a trace with raw tensors for torch.save artifacts."""

    metadata = trace.metadata()
    return metadata | {
        "steps": [
            step.metadata()
            | {
                "state_tensors": [
                    tensor.metadata() | {"value": tensor.value.detach().cpu().clone()}
                    for tensor in step.state_tensors
                ],
                "lower": step.lower.detach().cpu().clone(),
            }
            for step in trace.steps
        ]
    }


def production_optimizer_step_trace_from_payload_v4(
    payload: Mapping[str, object],
) -> ProductionOptimizerStepTraceV4:
    """Rebuild a raw step trace and re-run all semantic validation gates."""

    expected_fields = {
        "schema_version",
        "mutation_policy",
        "mutation_policy_hash",
        "evaluation_count",
        "update_count",
        "steps",
        "performance_claimed",
        "trace_hash",
    }
    if set(payload) != expected_fields or payload["performance_claimed"] is not False:
        raise ValueError("RVIR-v4 optimizer step trace payload fields differ")
    raw_policy = payload["mutation_policy"]
    raw_steps = payload["steps"]
    if not isinstance(raw_policy, Mapping) or not isinstance(raw_steps, list):
        raise TypeError("RVIR-v4 optimizer step trace payload structure differs")
    policy = production_mutation_policy_from_payload_v4(raw_policy)
    if payload["mutation_policy_hash"] != policy.stable_hash():
        raise ValueError("RVIR-v4 optimizer step trace policy hash differs")
    steps: list[ProductionOptimizerStepV4] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, Mapping):
            raise TypeError("RVIR-v4 optimizer step row differs")
        raw_tensors = raw_step.get("state_tensors")
        lower = raw_step.get("lower")
        if not isinstance(raw_tensors, list) or not torch.is_tensor(lower):
            raise TypeError("RVIR-v4 optimizer step raw tensor payload differs")
        tensors: list[OwnedProductionTensorV4] = []
        for raw_tensor in raw_tensors:
            if not isinstance(raw_tensor, Mapping) or not torch.is_tensor(
                raw_tensor.get("value")
            ):
                raise TypeError("RVIR-v4 optimizer step state tensor payload differs")
            axes = raw_tensor.get("axes")
            if not isinstance(axes, list) or not all(
                isinstance(axis, str) for axis in axes
            ):
                raise TypeError("RVIR-v4 optimizer step tensor axes differ")
            tensor = OwnedProductionTensorV4(
                semantic_path=str(raw_tensor.get("semantic_path", "")),
                role=ProductionTensorRole(str(raw_tensor.get("role", ""))),
                axes=tuple(axes),
                value=cast(torch.Tensor, raw_tensor["value"])
                .detach()
                .cpu()
                .contiguous()
                .clone(),
                content_sha256=str(raw_tensor.get("content_sha256", "")),
                source_device=str(raw_tensor.get("source_device", "")),
                ownership=ProductionTensorOwnership(
                    str(raw_tensor.get("ownership", ""))
                ),
                alias_group=str(raw_tensor.get("alias_group", "")),
            )
            tensor.validate()
            expected_tensor = tensor.metadata() | {"value": raw_tensor["value"]}
            if set(raw_tensor) != set(expected_tensor) or any(
                raw_tensor[key] != expected_tensor[key]
                for key in expected_tensor
                if key != "value"
            ):
                raise ValueError("RVIR-v4 optimizer step tensor metadata differs")
            tensors.append(tensor)
        integer_fields = (
            "core_id",
            "call_id",
            "parent_call_id",
            "evaluation_ordinal",
            "updates_before",
        )
        if not all(
            isinstance(raw_step.get(name), int)
            and not isinstance(raw_step.get(name), bool)
            for name in integer_fields
        ) or not isinstance(raw_step.get("update_after"), bool):
            raise TypeError("RVIR-v4 optimizer step scalar fields differ")
        raw_optimizer_step = raw_step.get("optimizer_step_ordinal")
        if raw_optimizer_step is not None and (
            not isinstance(raw_optimizer_step, int)
            or isinstance(raw_optimizer_step, bool)
        ):
            raise TypeError("RVIR-v4 optimizer step ordinal differs")

        step = ProductionOptimizerStepV4(
            core_id=cast(int, raw_step["core_id"]),
            call_id=cast(int, raw_step["call_id"]),
            parent_call_id=cast(int, raw_step["parent_call_id"]),
            evaluation_ordinal=cast(int, raw_step["evaluation_ordinal"]),
            updates_before=cast(int, raw_step["updates_before"]),
            update_after=cast(bool, raw_step["update_after"]),
            optimizer_step_ordinal=cast(int | None, raw_optimizer_step),
            alpha_learning_rate=_strict_number(
                raw_step, "alpha_learning_rate", label="optimizer step"
            ),
            beta_learning_rate=_strict_number(
                raw_step, "beta_learning_rate", label="optimizer step"
            ),
            state_tensors=tuple(tensors),
            lower=cast(torch.Tensor, lower).detach().cpu().contiguous().clone(),
            lower_sha256=str(raw_step.get("lower_sha256", "")),
        )
        step.validate()
        serialized_tensors = []
        for tensor_item, raw_tensor_item in zip(tensors, raw_tensors):
            serialized_tensors.append(
                tensor_item.metadata() | {"value": raw_tensor_item["value"]}
            )
        expected_step = step.metadata() | {
            "state_tensors": serialized_tensors,
            "lower": lower,
        }
        if set(raw_step) != set(expected_step) or any(
            raw_step[key] != expected_step[key]
            for key in expected_step
            if key not in {"state_tensors", "lower"}
        ):
            raise ValueError("RVIR-v4 optimizer step metadata replay differs")
        steps.append(step)
    trace = ProductionOptimizerStepTraceV4(
        mutation_policy=policy,
        steps=tuple(steps),
        schema_version=str(payload["schema_version"]),
    )
    trace.validate()
    metadata = trace.metadata()
    comparable_payload = dict(payload)
    comparable_payload["steps"] = [step.metadata() for step in trace.steps]
    if comparable_payload != metadata:
        raise ValueError("RVIR-v4 optimizer step trace metadata replay differs")
    return trace


__all__ = [
    "PRODUCTION_ITERATION_SEMANTICS_V4",
    "OPTIMIZER_STEP_TRACE_SCHEMA_V4",
    "ProductionMutationPolicyV4",
    "ProductionOptimizerControlsV4",
    "ProductionOptimizerStepTraceV4",
    "ProductionOptimizerStepV4",
    "capture_production_optimizer_controls_v4",
    "production_optimizer_controls_from_payload_v4",
    "production_mutation_policy_from_payload_v4",
    "production_optimizer_step_trace_from_payload_v4",
    "production_optimizer_step_trace_to_payload_v4",
]
