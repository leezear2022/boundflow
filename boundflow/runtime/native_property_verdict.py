"""Sound three-state property verdicts over an optimized native BaB trace."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions,missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal, Optional, Sequence

import torch

from ..frontends.plain_crown_bound_ir import tensor_content_hash
from ..ir.task import BFTaskModule
from .native_optimized_relu_split_bab_runtime import (
    NativeOptimizedReluSplitBabExecution,
    NativeOptimizedReluSplitBabTrace,
)
from .native_relu_split_bab_runtime import (
    _normalize_scalar_objective,
    _root_box_bounds,
)
from .task_executor import InputSpec, execute_task_module_concrete

NATIVE_PROPERTY_VERDICT_SCHEMA_VERSION = "boundflow.native-property-verdict/v1"
NATIVE_PROPERTY_VERDICT_COMPILER_VERSION = "boundflow.native-property-verdict/v1"
WITNESS_SPLIT_TOLERANCE = 1e-6
PropertyVerdictStatus = Literal["verified", "unsafe", "unknown"]


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _leaf_accounting(
    queue: NativeOptimizedReluSplitBabTrace,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    decision_by_id = {item.node_id: item for item in queue.decisions}
    evaluation_by_id = {item.node.node_id: item for item in queue.evaluations}
    evaluation_ids = tuple(evaluation_by_id)
    sound_pruned = tuple(
        node_id
        for node_id in evaluation_ids
        if node_id in decision_by_id
        and decision_by_id[node_id].kind == "prune"
        and decision_by_id[node_id].reason == "lower_bound_meets_threshold"
        and evaluation_by_id[node_id].lower >= queue.config.threshold
    )
    sound_pruned_set = set(sound_pruned)
    unresolved = tuple(
        node_id
        for node_id in evaluation_ids
        if node_id in queue.final_frontier_node_ids
        or (
            node_id in decision_by_id
            and (
                decision_by_id[node_id].kind == "terminal"
                or (
                    decision_by_id[node_id].kind == "prune"
                    and node_id not in sound_pruned_set
                )
            )
        )
    )
    return sound_pruned, unresolved


def _unknown_reason(queue: NativeOptimizedReluSplitBabTrace) -> str:
    if queue.final_frontier_node_ids:
        return "node_budget_frontier_open"
    terminal_reasons = {
        decision.reason for decision in queue.decisions if decision.kind == "terminal"
    }
    if "configured_depth_limit" in terminal_reasons:
        return "configured_depth_terminal_open"
    if terminal_reasons:
        return "unresolved_exact_activation_leaf"
    return "unproven_prune_open"


@dataclass(frozen=True)
class NativeConcreteCounterexampleTrace:
    """Digest-bound concrete witness replayed through primal Task IR."""

    node_id: str
    split_state_hash: str
    input_hash: str
    input_shape: tuple[int, ...]
    input_dtype: str
    output_hash: str
    value_trace_hash: str
    objective_value: float
    threshold: float
    objective_margin: float
    input_box_max_violation: float
    satisfied_split_count: int
    split_constraint_min_margin: Optional[float]
    split_tolerance: float = WITNESS_SPLIT_TOLERANCE

    def validate(self) -> None:
        if (
            not self.node_id
            or not _is_sha256(self.split_state_hash)
            or not _is_sha256(self.input_hash)
            or not _is_sha256(self.output_hash)
            or not _is_sha256(self.value_trace_hash)
            or not self.input_shape
            or any(dim < 1 for dim in self.input_shape)
            or not self.input_dtype
            or self.satisfied_split_count < 0
            or self.split_tolerance < 0.0
            or not all(
                torch.isfinite(torch.tensor(value)).item()
                for value in (
                    self.objective_value,
                    self.threshold,
                    self.objective_margin,
                    self.input_box_max_violation,
                    self.split_tolerance,
                )
            )
            or self.input_box_max_violation > 0.0
            or self.objective_value >= self.threshold
            or self.objective_margin != self.objective_value - self.threshold
        ):
            raise ValueError("native concrete counterexample trace is invalid")
        if self.satisfied_split_count == 0:
            if self.split_constraint_min_margin is not None:
                raise ValueError("root counterexample cannot declare a split margin")
        elif (
            self.split_constraint_min_margin is None
            or not torch.isfinite(torch.tensor(self.split_constraint_min_margin)).item()
            or self.split_constraint_min_margin < -self.split_tolerance
        ):
            raise ValueError("counterexample does not satisfy its ReLU split path")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "node_id": self.node_id,
            "split_state_hash": self.split_state_hash,
            "input_hash": self.input_hash,
            "input_shape": list(self.input_shape),
            "input_dtype": self.input_dtype,
            "output_hash": self.output_hash,
            "value_trace_hash": self.value_trace_hash,
            "objective_value": self.objective_value,
            "threshold": self.threshold,
            "objective_margin": self.objective_margin,
            "input_box_max_violation": self.input_box_max_violation,
            "satisfied_split_count": self.satisfied_split_count,
            "split_constraint_min_margin": self.split_constraint_min_margin,
            "split_tolerance": self.split_tolerance,
        }


@dataclass(frozen=True)
class NativePropertyVerdictTrace:
    """Serialized verdict proof bound to one immutable optimized queue trace."""

    run_id: str
    status: PropertyVerdictStatus
    reason: str
    queue_trace_hash: str
    objective_hash: str
    threshold: float
    sound_pruned_leaf_node_ids: tuple[str, ...]
    unresolved_leaf_node_ids: tuple[str, ...]
    counterexample: Optional[NativeConcreteCounterexampleTrace]
    performance_claimed: bool = False
    schema_version: str = NATIVE_PROPERTY_VERDICT_SCHEMA_VERSION

    def validate(self, queue: NativeOptimizedReluSplitBabTrace) -> None:
        queue.validate()
        expected_pruned, expected_unresolved = _leaf_accounting(queue)
        if (
            self.schema_version != NATIVE_PROPERTY_VERDICT_SCHEMA_VERSION
            or not self.run_id
            or self.run_id != queue.run_id
            or self.status not in {"verified", "unsafe", "unknown"}
            or not self.reason
            or self.queue_trace_hash != queue.stable_hash()
            or self.objective_hash != queue.objective_hash
            or self.threshold != queue.config.threshold
            or not torch.isfinite(torch.tensor(self.threshold)).item()
            or self.sound_pruned_leaf_node_ids != expected_pruned
            or self.unresolved_leaf_node_ids != expected_unresolved
            or self.performance_claimed is not False
        ):
            raise ValueError("native property verdict trace does not match its queue")

        if self.status == "verified":
            if (
                self.reason != "all_leaves_soundly_pruned"
                or queue.status != "complete"
                or expected_unresolved
                or self.counterexample is not None
                or not expected_pruned
            ):
                raise ValueError("verified verdict lacks a closed sound-prune proof")
        elif self.status == "unsafe":
            if (
                self.reason != "concrete_counterexample_reexecuted"
                or self.counterexample is None
            ):
                raise ValueError("unsafe verdict lacks a concrete counterexample")
            self.counterexample.validate()
            evaluation = next(
                (
                    item
                    for item in queue.evaluations
                    if item.node.node_id == self.counterexample.node_id
                ),
                None,
            )
            if (
                evaluation is None
                or evaluation.node.split_state_hash
                != self.counterexample.split_state_hash
                or self.counterexample.threshold != self.threshold
            ):
                raise ValueError("counterexample node identity differs from queue")
        elif (
            self.counterexample is not None
            or not expected_unresolved
            or self.reason != _unknown_reason(queue)
        ):
            raise ValueError("unknown verdict lacks an explicit unresolved leaf")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "compiler_version": NATIVE_PROPERTY_VERDICT_COMPILER_VERSION,
            "run_id": self.run_id,
            "status": self.status,
            "reason": self.reason,
            "performance_claimed": self.performance_claimed,
            "queue_trace_hash": self.queue_trace_hash,
            "objective_hash": self.objective_hash,
            "threshold": self.threshold,
            "sound_pruned_leaf_node_ids": list(self.sound_pruned_leaf_node_ids),
            "unresolved_leaf_node_ids": list(self.unresolved_leaf_node_ids),
            "counterexample": (
                None if self.counterexample is None else self.counterexample.to_dict()
            ),
        }

    def stable_hash(self, queue: NativeOptimizedReluSplitBabTrace) -> str:
        self.validate(queue)
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativePropertyVerdictExecution:
    """Verdict trace plus the non-serialized concrete witness tensor, if any."""

    trace: NativePropertyVerdictTrace
    counterexample_input: Optional[torch.Tensor]

    def validate_against(
        self,
        module: BFTaskModule,
        input_spec: InputSpec,
        *,
        linear_spec_C: torch.Tensor,
        queue_execution: NativeOptimizedReluSplitBabExecution,
    ) -> None:
        queue_execution.validate()
        self.trace.validate(queue_execution.trace)
        objective = _validate_semantic_identity(
            input_spec,
            linear_spec_C=linear_spec_C,
            queue=queue_execution.trace,
        )
        if self.trace.status != "unsafe":
            if self.counterexample_input is not None:
                raise ValueError("non-unsafe verdict cannot retain a counterexample")
            return
        if self.counterexample_input is None or self.trace.counterexample is None:
            raise ValueError("unsafe verdict lacks its concrete input payload")
        actual = _reexecute_counterexample(
            module,
            input_spec,
            objective=objective,
            queue_execution=queue_execution,
            node_id=self.trace.counterexample.node_id,
            candidate=self.counterexample_input,
        )
        if actual.to_dict() != self.trace.counterexample.to_dict():
            raise ValueError("counterexample replay differs from serialized trace")


def _validate_semantic_identity(
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    queue: NativeOptimizedReluSplitBabTrace,
) -> torch.Tensor:
    lower, upper = _root_box_bounds(input_spec)
    objective = _normalize_scalar_objective(linear_spec_C)
    if (
        tensor_content_hash(lower) != queue.root_input_lower_hash
        or tensor_content_hash(upper) != queue.root_input_upper_hash
        or tensor_content_hash(objective) != queue.objective_hash
    ):
        raise ValueError("property verdict input/objective identity differs from queue")
    return objective


def _reexecute_counterexample(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    objective: torch.Tensor,
    queue_execution: NativeOptimizedReluSplitBabExecution,
    node_id: str,
    candidate: torch.Tensor,
) -> NativeConcreteCounterexampleTrace:
    lower, upper = _root_box_bounds(input_spec)
    if (
        not torch.is_tensor(candidate)
        or candidate.shape != lower.shape
        or candidate.dtype != lower.dtype
        or candidate.device != lower.device
        or not bool(torch.isfinite(candidate).all())
    ):
        raise ValueError(
            "counterexample input shape/dtype/device differs from root box"
        )
    max_box_violation = float(
        torch.maximum(
            torch.clamp(lower - candidate, min=0.0),
            torch.clamp(candidate - upper, min=0.0),
        )
        .max()
        .item()
    )
    if max_box_violation > 0.0:
        raise ValueError("counterexample input lies outside the root box")

    state = queue_execution.state_map().get(node_id)
    evaluation = next(
        (
            item
            for item in queue_execution.trace.evaluations
            if item.node.node_id == node_id
        ),
        None,
    )
    if state is None or evaluation is None:
        raise ValueError("counterexample references an unknown queue node")
    execution = execute_task_module_concrete(
        module,
        candidate,
        input_value_name=input_spec.value_name,
    )
    values = execution.value_map()
    split_margins: list[torch.Tensor] = []
    satisfied_split_count = 0
    for name, split in state.splits.items():
        preactivation = values.get(name)
        if preactivation is None or preactivation.shape != split.shape:
            raise ValueError("counterexample ReLU split value is missing or mismatched")
        active = split == 1
        inactive = split == -1
        if bool(active.any()):
            split_margins.append(preactivation[active])
            satisfied_split_count += int(active.sum().item())
        if bool(inactive.any()):
            split_margins.append(-preactivation[inactive])
            satisfied_split_count += int(inactive.sum().item())
    min_split_margin = (
        None
        if not split_margins
        else float(
            torch.cat(tuple(value.reshape(-1) for value in split_margins)).min().item()
        )
    )
    if min_split_margin is not None and min_split_margin < -WITNESS_SPLIT_TOLERANCE:
        raise ValueError(
            "counterexample input does not satisfy its queue-node split path"
        )

    output = execution.output
    if (
        output.dim() != 2
        or int(output.shape[0]) != 1
        or int(output.shape[1]) != int(objective.shape[1])
        or output.dtype != objective.dtype
        or output.device != objective.device
    ):
        raise ValueError("counterexample output/objective shape differs")
    objective_value = float((output * objective).sum().item())
    threshold = queue_execution.trace.config.threshold
    value_trace_hash = _canonical_hash(
        [
            {"name": name, "tensor_hash": tensor_content_hash(value)}
            for name, value in execution.values
        ]
    )
    trace = NativeConcreteCounterexampleTrace(
        node_id=node_id,
        split_state_hash=evaluation.node.split_state_hash,
        input_hash=tensor_content_hash(candidate),
        input_shape=tuple(int(dim) for dim in candidate.shape),
        input_dtype=str(candidate.dtype).removeprefix("torch."),
        output_hash=tensor_content_hash(output),
        value_trace_hash=value_trace_hash,
        objective_value=objective_value,
        threshold=threshold,
        objective_margin=objective_value - threshold,
        input_box_max_violation=max_box_violation,
        satisfied_split_count=satisfied_split_count,
        split_constraint_min_margin=min_split_margin,
    )
    return trace


def derive_native_property_verdict(
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    queue_execution: NativeOptimizedReluSplitBabExecution,
    candidate_counterexamples: Sequence[tuple[str, torch.Tensor]] = (),
) -> NativePropertyVerdictExecution:
    """Derive a sound verdict; supplied candidates are never trusted without replay."""

    module.validate()
    queue_execution.validate()
    queue = queue_execution.trace
    objective = _validate_semantic_identity(
        input_spec,
        linear_spec_C=linear_spec_C,
        queue=queue,
    )
    pruned, unresolved = _leaf_accounting(queue)

    for node_id, candidate in candidate_counterexamples:
        witness = _reexecute_counterexample(
            module,
            input_spec,
            objective=objective,
            queue_execution=queue_execution,
            node_id=node_id,
            candidate=candidate,
        )
        if witness.objective_value < queue.config.threshold:
            trace = NativePropertyVerdictTrace(
                run_id=queue.run_id,
                status="unsafe",
                reason="concrete_counterexample_reexecuted",
                queue_trace_hash=queue.stable_hash(),
                objective_hash=queue.objective_hash,
                threshold=queue.config.threshold,
                sound_pruned_leaf_node_ids=pruned,
                unresolved_leaf_node_ids=unresolved,
                counterexample=witness,
            )
            result = NativePropertyVerdictExecution(
                trace=trace,
                counterexample_input=candidate.detach().contiguous().clone(),
            )
            result.validate_against(
                module,
                input_spec,
                linear_spec_C=linear_spec_C,
                queue_execution=queue_execution,
            )
            return result

    if not unresolved and queue.status == "complete":
        status: PropertyVerdictStatus = "verified"
        reason = "all_leaves_soundly_pruned"
    else:
        status = "unknown"
        reason = _unknown_reason(queue)
    trace = NativePropertyVerdictTrace(
        run_id=queue.run_id,
        status=status,
        reason=reason,
        queue_trace_hash=queue.stable_hash(),
        objective_hash=queue.objective_hash,
        threshold=queue.config.threshold,
        sound_pruned_leaf_node_ids=pruned,
        unresolved_leaf_node_ids=unresolved,
        counterexample=None,
    )
    result = NativePropertyVerdictExecution(trace=trace, counterexample_input=None)
    result.validate_against(
        module,
        input_spec,
        linear_spec_C=linear_spec_C,
        queue_execution=queue_execution,
    )
    return result


__all__ = [
    "NATIVE_PROPERTY_VERDICT_COMPILER_VERSION",
    "NATIVE_PROPERTY_VERDICT_SCHEMA_VERSION",
    "NativeConcreteCounterexampleTrace",
    "NativePropertyVerdictExecution",
    "NativePropertyVerdictTrace",
    "PropertyVerdictStatus",
    "WITNESS_SPLIT_TOLERANCE",
    "derive_native_property_verdict",
]
