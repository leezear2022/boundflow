"""Deterministic concrete candidate search over primal Task IR."""

# pylint: disable=too-many-arguments,too-many-locals,missing-function-docstring
# pylint: disable=too-many-boolean-expressions,too-many-instance-attributes
# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch

from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.task import BFTaskModule
from .native_relu_split_bab_runtime import _normalize_scalar_objective
from .perturbation import BoxPerturbation
from .task_executor import InputSpec, execute_task_module_concrete

NATIVE_CANDIDATE_SEARCH_SCHEMA_VERSION = "boundflow.native-candidate-search/v1"
NATIVE_CANDIDATE_SEARCH_COMPILER_VERSION = "boundflow.native-candidate-search/v1"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class NativeProjectedGradientSearchPolicy:
    """Frozen deterministic center-start sign-gradient search policy."""

    steps: int
    step_size: float
    early_stop: bool = True

    def validate(self) -> None:
        if (
            self.steps < 0
            or self.step_size <= 0.0
            or not torch.isfinite(torch.tensor(self.step_size)).item()
            or self.early_stop is not True
        ):
            raise ValueError("native projected-gradient search policy is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "steps": self.steps,
            "step_size": self.step_size,
            "early_stop": self.early_stop,
            "start": "input_box_center",
            "update": "sign_gradient_descent",
            "projection": "exact_box_clamp",
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeCandidateSearchTrace:
    """Replayable search trajectory summary; never a verification proof."""

    objective_hash: str
    threshold: float
    policy: NativeProjectedGradientSearchPolicy
    initial_input_hash: str
    best_input_hash: str
    objective_values: tuple[float, ...]
    gradient_l1_values: tuple[float, ...]
    steps_executed: int
    projected_update_count: int
    best_iteration: int
    best_objective_value: float
    counterexample_found: bool
    early_stopped: bool
    performance_claimed: bool = False
    schema_version: str = NATIVE_CANDIDATE_SEARCH_SCHEMA_VERSION

    def validate(self) -> None:
        self.policy.validate()
        if (
            self.schema_version != NATIVE_CANDIDATE_SEARCH_SCHEMA_VERSION
            or not _is_sha256(self.objective_hash)
            or not _is_sha256(self.initial_input_hash)
            or not _is_sha256(self.best_input_hash)
            or self.steps_executed < 0
            or self.steps_executed > self.policy.steps
            or self.projected_update_count != self.steps_executed
            or len(self.objective_values) != self.steps_executed + 1
            or len(self.gradient_l1_values) != self.steps_executed
            or not 0 <= self.best_iteration < len(self.objective_values)
            or self.best_objective_value != self.objective_values[self.best_iteration]
            or self.best_objective_value != min(self.objective_values)
            or self.counterexample_found != (self.best_objective_value < self.threshold)
            or self.early_stopped
            != (self.counterexample_found and self.steps_executed < self.policy.steps)
            or self.performance_claimed is not False
            or not all(
                torch.isfinite(torch.tensor(value)).item()
                for value in (
                    self.threshold,
                    self.best_objective_value,
                    *self.objective_values,
                    *self.gradient_l1_values,
                )
            )
            or any(value < 0.0 for value in self.gradient_l1_values)
        ):
            raise ValueError("native candidate search trace is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "compiler_version": NATIVE_CANDIDATE_SEARCH_COMPILER_VERSION,
            "objective_hash": self.objective_hash,
            "threshold": self.threshold,
            "policy": self.policy.to_dict(),
            "initial_input_hash": self.initial_input_hash,
            "best_input_hash": self.best_input_hash,
            "objective_values": list(self.objective_values),
            "gradient_l1_values": list(self.gradient_l1_values),
            "steps_executed": self.steps_executed,
            "projected_update_count": self.projected_update_count,
            "best_iteration": self.best_iteration,
            "best_objective_value": self.best_objective_value,
            "counterexample_found": self.counterexample_found,
            "early_stopped": self.early_stopped,
            "performance_claimed": self.performance_claimed,
            "proof_claimed": False,
        }

    def stable_hash(self) -> str:
        self.validate()
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeCandidateSearchExecution:
    """Search trace plus the non-serialized best candidate tensor."""

    trace: NativeCandidateSearchTrace
    best_input: torch.Tensor

    def validate_against(
        self,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        linear_spec_C: torch.Tensor,
    ) -> None:
        self.trace.validate()
        objective = _normalize_scalar_objective(linear_spec_C)
        if tensor_content_hash(objective) != self.trace.objective_hash:
            raise ValueError("candidate search objective identity differs")
        if not isinstance(input_spec.perturbation, BoxPerturbation):
            raise NotImplementedError("native candidate search v1 requires a box input")
        lower, upper = input_spec.perturbation.bounding_box(input_spec.center)
        if (
            self.best_input.shape != lower.shape
            or self.best_input.dtype != lower.dtype
            or self.best_input.device != lower.device
            or not bool(torch.isfinite(self.best_input).all())
            or not bool(((self.best_input >= lower) & (self.best_input <= upper)).all())
            or tensor_content_hash(self.best_input) != self.trace.best_input_hash
        ):
            raise ValueError("candidate search best input is outside its exact box")
        execution = execute_task_module_concrete(
            module,
            self.best_input,
            input_value_name=input_spec.value_name,
        )
        output = execution.output
        if (
            output.dim() != 2
            or int(output.shape[0]) != 1
            or int(output.shape[1]) != int(objective.shape[1])
        ):
            raise ValueError("candidate search output/objective shape differs")
        actual = float((output * objective).sum().item())
        if actual != self.trace.best_objective_value:
            raise ValueError("candidate search best input replay differs")


def search_native_box_counterexample(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    threshold: float,
    policy: NativeProjectedGradientSearchPolicy,
) -> NativeCandidateSearchExecution:
    """Search for one concrete violating input; not-found is never a proof."""

    module.validate()
    policy.validate()
    objective = _normalize_scalar_objective(linear_spec_C)
    if not isinstance(input_spec.perturbation, BoxPerturbation):
        raise NotImplementedError("native candidate search v1 requires a box input")
    lower, upper = input_spec.perturbation.bounding_box(input_spec.center)
    if int(lower.shape[0]) != 1:
        raise ValueError("native candidate search v1 requires one input domain")
    if not torch.isfinite(torch.tensor(threshold)).item():
        raise ValueError("candidate search threshold must be finite")

    candidate = input_spec.center.detach().contiguous().clone().requires_grad_(True)
    initial_input_hash = tensor_content_hash(candidate)
    objective_values: list[float] = []
    gradient_l1_values: list[float] = []
    best_input = candidate.detach().contiguous().clone()
    best_value = float("inf")
    best_iteration = 0
    updates = 0
    early_stopped = False

    for iteration in range(policy.steps + 1):
        execution = execute_task_module_concrete(
            module,
            candidate,
            input_value_name=input_spec.value_name,
            preserve_gradients=True,
        )
        value_tensor = (execution.output * objective).sum()
        value = float(value_tensor.detach().item())
        objective_values.append(value)
        if value < best_value:
            best_value = value
            best_iteration = iteration
            best_input = candidate.detach().contiguous().clone()
        if value < threshold:
            early_stopped = iteration < policy.steps
            break
        if iteration == policy.steps:
            break
        gradient = torch.autograd.grad(value_tensor, candidate, only_inputs=True)[0]
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("candidate search produced a non-finite input gradient")
        gradient_l1_values.append(float(gradient.abs().sum().item()))
        with torch.no_grad():
            candidate = torch.clamp(
                candidate - policy.step_size * gradient.sign(),
                min=lower,
                max=upper,
            ).detach()
        candidate.requires_grad_(True)
        updates += 1

    trace = NativeCandidateSearchTrace(
        objective_hash=tensor_content_hash(objective),
        threshold=float(threshold),
        policy=policy,
        initial_input_hash=initial_input_hash,
        best_input_hash=tensor_content_hash(best_input),
        objective_values=tuple(objective_values),
        gradient_l1_values=tuple(gradient_l1_values),
        steps_executed=updates,
        projected_update_count=updates,
        best_iteration=best_iteration,
        best_objective_value=best_value,
        counterexample_found=best_value < threshold,
        early_stopped=early_stopped,
    )
    result = NativeCandidateSearchExecution(trace=trace, best_input=best_input)
    result.validate_against(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
    )
    return result


__all__ = [
    "NATIVE_CANDIDATE_SEARCH_COMPILER_VERSION",
    "NATIVE_CANDIDATE_SEARCH_SCHEMA_VERSION",
    "NativeCandidateSearchExecution",
    "NativeCandidateSearchTrace",
    "NativeProjectedGradientSearchPolicy",
    "search_native_box_counterexample",
]
