"""Deterministic native ReLU-split BaB queue over Bound/Plan/Task/Schedule IR."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-instance-attributes,invalid-name
# pylint: disable=too-many-lines,missing-function-docstring
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from typing import Literal, Mapping, Optional, Sequence

import torch

from ..domains.interval import IntervalState
from ..frontends.plain_crown_bound_ir import (
    relu_split_state_hash,
    tensor_content_hash,
)
from ..ir.bound import (
    BoundOpKind,
    BoundValueRole,
    IntermediateBoundSource,
    SplitReluRelaxationAttrs,
)
from ..ir.schedule import LaunchAction
from ..ir.task import BFTaskModule
from .crown_ibp import _forward_ibp_trace_mlp
from .native_verifier_ir_integration import (
    NativePlainCrownRepresentationCompilation,
    compile_native_plain_crown_representation_query,
    execute_native_plain_crown_representation_query,
)
from .perturbation import BoxPerturbation
from .task_executor import InputSpec

NATIVE_RELU_SPLIT_BAB_COMPILER_VERSION = "boundflow.native-relu-split-bab/v1"
NATIVE_RELU_SPLIT_BAB_TRACE_SCHEMA_VERSION = "boundflow.relu-split-bab-trace/v1"
PARENT_EXACT_STATE_VALIDITY = "discrete_split_inheritance_only"
BabQueueStatus = Literal["complete", "budget_exhausted"]
BabDecisionKind = Literal["prune", "expand", "terminal"]
_IR_HASH_KEYS = {
    "source_bound_module_hash",
    "source_plan_template_hash",
    "source_plan_instance_hash",
    "source_schedule_hash",
    "representation_binding_hash",
    "execution_bound_module_hash",
    "execution_plan_template_hash",
    "execution_plan_instance_hash",
    "task_module_hash",
    "schedule_hash",
}


@dataclass(frozen=True)
class NativeReluSplitBabConfig:
    """Frozen correctness-only queue and native node batching policy."""

    max_nodes: int
    max_depth: int
    expansion_batch_size: int
    max_eval_batch_size: int
    threshold: float = 0.0
    available_memory_bytes: int = 1 << 30
    memory_budget_bytes: int = 1 << 30

    def validate(self) -> None:
        if (
            self.max_nodes < 1
            or self.max_depth < 0
            or self.expansion_batch_size < 1
            or self.max_eval_batch_size < 1
            or self.available_memory_bytes < 1
            or self.memory_budget_bytes < 1
            or self.memory_budget_bytes > self.available_memory_bytes
        ):
            raise ValueError("native ReLU-split BaB config is invalid")
        if not torch.isfinite(torch.tensor(self.threshold)):
            raise ValueError("native ReLU-split BaB threshold must be finite")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "max_nodes": self.max_nodes,
            "max_depth": self.max_depth,
            "expansion_batch_size": self.expansion_batch_size,
            "max_eval_batch_size": self.max_eval_batch_size,
            "threshold": self.threshold,
            "available_memory_bytes": self.available_memory_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
        }


@dataclass(frozen=True)
class ReluSplitBranch:
    """One deterministic ambiguous ReLU branch candidate."""

    relu_input: str
    neuron_index: int
    lower: float
    upper: float
    width: float

    def validate(self) -> None:
        if (
            not self.relu_input
            or self.neuron_index < 0
            or not self.lower < 0.0 < self.upper
            or self.width <= 0.0
            or not math.isclose(
                self.width,
                self.upper - self.lower,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            raise ValueError("native ReLU branch candidate is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "relu_input": self.relu_input,
            "neuron_index": self.neuron_index,
            "lower": self.lower,
            "upper": self.upper,
            "width": self.width,
        }


@dataclass(frozen=True)
class NativeReluSplitBabNode:
    """Typed queue-node lineage and exact split-state identity."""

    node_id: str
    parent_node_id: Optional[str]
    depth: int
    branch_relu_input: Optional[str]
    branch_neuron_index: Optional[int]
    branch_value: int
    split_state_hash: str

    def validate(self) -> None:
        if not self.node_id or self.depth < 0 or not _is_sha256(self.split_state_hash):
            raise ValueError("native ReLU-split BaB node identity is invalid")
        if self.depth == 0:
            if (
                self.parent_node_id is not None
                or self.branch_relu_input is not None
                or self.branch_neuron_index is not None
                or self.branch_value != 0
            ):
                raise ValueError("native ReLU-split BaB root lineage is invalid")
        elif (
            not self.parent_node_id
            or not self.branch_relu_input
            or self.branch_neuron_index is None
            or self.branch_neuron_index < 0
            or self.branch_value not in (-1, 1)
        ):
            raise ValueError("native ReLU-split BaB child lineage is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "node_id": self.node_id,
            "parent_node_id": self.parent_node_id,
            "depth": self.depth,
            "branch_relu_input": self.branch_relu_input,
            "branch_neuron_index": self.branch_neuron_index,
            "branch_value": self.branch_value,
            "split_state_hash": self.split_state_hash,
        }


@dataclass(frozen=True)
class NativeReluSplitBabEvaluation:
    """One node result from an actual native Bound/Plan/Task/Schedule stack."""

    node: NativeReluSplitBabNode
    lower: float
    upper: float
    priority: float
    exact_state_hash: str
    parent_exact_state_hash: Optional[str]
    eval_batch_id: str
    eval_batch_position: int
    native_ir_hashes: tuple[tuple[str, str], ...]
    branch_candidate: Optional[ReluSplitBranch]
    parent_state_validity: str = PARENT_EXACT_STATE_VALIDITY
    parent_state_consumed_as_exact: bool = False

    def validate(self) -> None:
        self.node.validate()
        hashes = dict(self.native_ir_hashes)
        if (
            not self.eval_batch_id
            or self.eval_batch_position < 0
            or not _is_sha256(self.exact_state_hash)
            or set(hashes) != _IR_HASH_KEYS
            or len(hashes) != len(self.native_ir_hashes)
            or any(not _is_sha256(value) for value in hashes.values())
            or not all(
                torch.isfinite(torch.tensor(value))
                for value in (self.lower, self.upper, self.priority)
            )
            or self.lower > self.upper
            or self.parent_state_validity != PARENT_EXACT_STATE_VALIDITY
            or self.parent_state_consumed_as_exact is not False
        ):
            raise ValueError("native ReLU-split BaB evaluation is invalid")
        if self.node.depth == 0:
            if self.parent_exact_state_hash is not None:
                raise ValueError(
                    "native ReLU-split root cannot bind parent exact state"
                )
        elif (
            not _is_sha256(self.parent_exact_state_hash)
            or self.parent_exact_state_hash == self.exact_state_hash
        ):
            raise ValueError("parent exact state was reused as child exact state")
        if self.branch_candidate is not None:
            self.branch_candidate.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "node": self.node.to_dict(),
            "lower": self.lower,
            "upper": self.upper,
            "priority": self.priority,
            "exact_state_hash": self.exact_state_hash,
            "parent_exact_state_hash": self.parent_exact_state_hash,
            "parent_state_validity": self.parent_state_validity,
            "parent_state_consumed_as_exact": self.parent_state_consumed_as_exact,
            "eval_batch_id": self.eval_batch_id,
            "eval_batch_position": self.eval_batch_position,
            "native_ir_hashes": dict(self.native_ir_hashes),
            "branch_candidate": (
                None
                if self.branch_candidate is None
                else self.branch_candidate.to_dict()
            ),
        }


@dataclass(frozen=True)
class NativeReluSplitBabDecision:
    """Deterministic prune/expand/terminal action for one evaluated node."""

    decision_index: int
    node_id: str
    kind: BabDecisionKind
    reason: str
    child_node_ids: tuple[str, ...] = ()
    branch_candidate: Optional[ReluSplitBranch] = None

    def validate(self) -> None:
        if self.decision_index < 0 or not self.node_id or not self.reason:
            raise ValueError("native ReLU-split BaB decision identity is invalid")
        if self.kind == "expand":
            if len(self.child_node_ids) != 2 or self.branch_candidate is None:
                raise ValueError("native ReLU-split expansion requires two children")
            self.branch_candidate.validate()
        elif self.kind in {"prune", "terminal"}:
            if self.child_node_ids or self.branch_candidate is not None:
                raise ValueError("non-expansion decision cannot declare children")
        else:
            raise ValueError("native ReLU-split BaB decision kind is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "decision_index": self.decision_index,
            "node_id": self.node_id,
            "kind": self.kind,
            "reason": self.reason,
            "child_node_ids": list(self.child_node_ids),
            "branch_candidate": (
                None
                if self.branch_candidate is None
                else self.branch_candidate.to_dict()
            ),
        }


@dataclass(frozen=True)
class NativeReluSplitBabStackTrace:
    """Structural proof that one node batch owned a full native IR stack."""

    stack_id: str
    node_ids: tuple[str, ...]
    domain_batch_size: int
    bound_split_input_count: int
    bound_split_relu_op_count: int
    bound_local_forward_relu_op_count: int
    source_plan_split_state_present: bool
    execution_plan_split_state_present: bool
    split_capability_count: int
    local_forward_provenance_count: int
    task_count: int
    schedule_launch_count: int
    representation_policy_id: str
    storage_candidate_id: str
    native_ir_hashes: tuple[tuple[str, str], ...]

    def validate(self) -> None:
        hashes = dict(self.native_ir_hashes)
        if (
            not self.stack_id
            or not self.node_ids
            or len(self.node_ids) != len(set(self.node_ids))
            or self.domain_batch_size != len(self.node_ids)
            or self.bound_split_input_count < 1
            or self.bound_split_relu_op_count != self.bound_split_input_count
            or self.bound_local_forward_relu_op_count != self.bound_split_relu_op_count
            or self.source_plan_split_state_present is not True
            or self.execution_plan_split_state_present is not True
            or self.split_capability_count < 2
            or self.local_forward_provenance_count < 2
            or self.task_count < 1
            or self.schedule_launch_count != self.task_count
            or not self.representation_policy_id
            or not self.storage_candidate_id
            or set(hashes) != _IR_HASH_KEYS
            or len(hashes) != len(self.native_ir_hashes)
            or any(not _is_sha256(value) for value in hashes.values())
        ):
            raise ValueError("native ReLU-split IR stack trace is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "stack_id": self.stack_id,
            "node_ids": list(self.node_ids),
            "domain_batch_size": self.domain_batch_size,
            "bound_split_input_count": self.bound_split_input_count,
            "bound_split_relu_op_count": self.bound_split_relu_op_count,
            "bound_local_forward_relu_op_count": (
                self.bound_local_forward_relu_op_count
            ),
            "source_plan_split_state_present": (self.source_plan_split_state_present),
            "execution_plan_split_state_present": (
                self.execution_plan_split_state_present
            ),
            "split_capability_count": self.split_capability_count,
            "local_forward_provenance_count": (self.local_forward_provenance_count),
            "task_count": self.task_count,
            "schedule_launch_count": self.schedule_launch_count,
            "representation_policy_id": self.representation_policy_id,
            "storage_candidate_id": self.storage_candidate_id,
            "native_ir_hashes": dict(self.native_ir_hashes),
        }


@dataclass(frozen=True)
class NativeReluSplitBabTrace:
    """Replayable bounded queue trace without a full-verifier property claim."""

    run_id: str
    status: BabQueueStatus
    termination_reason: str
    config: NativeReluSplitBabConfig
    root_input_lower_hash: str
    root_input_upper_hash: str
    objective_hash: str
    evaluations: tuple[NativeReluSplitBabEvaluation, ...]
    decisions: tuple[NativeReluSplitBabDecision, ...]
    final_frontier_node_ids: tuple[str, ...]
    native_stacks: tuple[NativeReluSplitBabStackTrace, ...]
    native_stack_count: int
    max_queue_size: int
    performance_claimed: bool = False
    property_status: str = "not_claimed"
    schema_version: str = NATIVE_RELU_SPLIT_BAB_TRACE_SCHEMA_VERSION

    def validate(self) -> None:  # pylint: disable=too-many-statements
        self.config.validate()
        if (
            self.schema_version != NATIVE_RELU_SPLIT_BAB_TRACE_SCHEMA_VERSION
            or not self.run_id
            or self.status not in {"complete", "budget_exhausted"}
            or not self.termination_reason
            or any(
                not _is_sha256(value)
                for value in (
                    self.root_input_lower_hash,
                    self.root_input_upper_hash,
                    self.objective_hash,
                )
            )
            or not self.evaluations
            or not self.native_stacks
            or self.native_stack_count < 1
            or self.max_queue_size < 1
            or self.performance_claimed is not False
            or self.property_status != "not_claimed"
        ):
            raise ValueError("native ReLU-split BaB trace header is invalid")
        if len(self.evaluations) > self.config.max_nodes:
            raise ValueError("native ReLU-split BaB trace exceeds node budget")
        evaluation_by_id: dict[str, NativeReluSplitBabEvaluation] = {}
        position: dict[str, int] = {}
        batches: dict[str, list[int]] = {}
        for index, evaluation in enumerate(self.evaluations):
            evaluation.validate()
            node = evaluation.node
            if node.node_id in evaluation_by_id:
                raise ValueError("native ReLU-split BaB node was evaluated twice")
            if node.depth == 0 and index != 0:
                raise ValueError("native ReLU-split BaB root is not first")
            if node.depth > 0:
                parent = evaluation_by_id.get(node.parent_node_id or "")
                if parent is None or node.depth != parent.node.depth + 1:
                    raise ValueError("native ReLU-split BaB parent is absent or late")
                if evaluation.parent_exact_state_hash != parent.exact_state_hash:
                    raise ValueError(
                        "native ReLU-split child parent-state link differs"
                    )
            evaluation_by_id[node.node_id] = evaluation
            position[node.node_id] = index
            batches.setdefault(evaluation.eval_batch_id, []).append(
                evaluation.eval_batch_position
            )
        if len(batches) != self.native_stack_count or any(
            positions != list(range(len(positions))) for positions in batches.values()
        ):
            raise ValueError("native ReLU-split evaluation batch accounting differs")
        stack_by_id: dict[str, NativeReluSplitBabStackTrace] = {}
        for stack in self.native_stacks:
            stack.validate()
            if stack.stack_id in stack_by_id:
                raise ValueError("native ReLU-split IR stack ID repeats")
            stack_by_id[stack.stack_id] = stack
        if (
            set(stack_by_id) != set(batches)
            or len(stack_by_id) != self.native_stack_count
        ):
            raise ValueError("native ReLU-split stack/evaluation batches differ")
        for batch_id, positions in batches.items():
            batch_evaluations = tuple(
                item for item in self.evaluations if item.eval_batch_id == batch_id
            )
            stack = stack_by_id[batch_id]
            if (
                stack.node_ids != tuple(item.node.node_id for item in batch_evaluations)
                or stack.domain_batch_size != len(positions)
                or any(
                    item.native_ir_hashes != stack.native_ir_hashes
                    for item in batch_evaluations
                )
            ):
                raise ValueError("native ReLU-split stack node/hash binding differs")
        decision_nodes: set[str] = set()
        expanded_children: set[str] = set()
        for index, decision in enumerate(self.decisions):
            decision.validate()
            if decision.decision_index != index or decision.node_id in decision_nodes:
                raise ValueError("native ReLU-split decisions repeat or reorder")
            if decision.node_id not in evaluation_by_id:
                raise ValueError("native ReLU-split decision references unknown node")
            decision_nodes.add(decision.node_id)
            if decision.kind == "expand":
                parent_position = position[decision.node_id]
                for child_index, child_id in enumerate(decision.child_node_ids):
                    child = evaluation_by_id.get(child_id)
                    if child is None or child.node.parent_node_id != decision.node_id:
                        raise ValueError(
                            "native ReLU-split expansion child link differs"
                        )
                    branch = decision.branch_candidate
                    if (
                        branch is None
                        or child.node.branch_relu_input != branch.relu_input
                        or child.node.branch_neuron_index != branch.neuron_index
                        or child.node.branch_value != (-1 if child_index == 0 else 1)
                    ):
                        raise ValueError("native ReLU-split expansion branch differs")
                    if position[child_id] <= parent_position:
                        raise ValueError("native ReLU-split child precedes parent")
                    expanded_children.add(child_id)
        frontier = set(self.final_frontier_node_ids)
        if (
            len(frontier) != len(self.final_frontier_node_ids)
            or not frontier <= set(evaluation_by_id)
            or frontier & decision_nodes
            or decision_nodes | frontier != set(evaluation_by_id)
            or expanded_children
            != {
                node_id
                for node_id in evaluation_by_id
                if node_id != self.evaluations[0].node.node_id
            }
        ):
            raise ValueError("native ReLU-split queue accounting does not close")
        if self.status == "complete" and frontier:
            raise ValueError("complete native ReLU-split queue retains a frontier")
        if self.status == "budget_exhausted" and not frontier:
            raise ValueError("budget-exhausted native ReLU-split queue lacks frontier")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "compiler_version": NATIVE_RELU_SPLIT_BAB_COMPILER_VERSION,
            "run_id": self.run_id,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "performance_claimed": self.performance_claimed,
            "property_status": self.property_status,
            "config": self.config.to_dict(),
            "root_input_lower_hash": self.root_input_lower_hash,
            "root_input_upper_hash": self.root_input_upper_hash,
            "objective_hash": self.objective_hash,
            "evaluations": [item.to_dict() for item in self.evaluations],
            "decisions": [item.to_dict() for item in self.decisions],
            "final_frontier_node_ids": list(self.final_frontier_node_ids),
            "native_stacks": [item.to_dict() for item in self.native_stacks],
            "native_stack_count": self.native_stack_count,
            "max_queue_size": self.max_queue_size,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def stable_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def logical_queue_signature(self) -> tuple[tuple[object, ...], ...]:
        """Return batching-independent node lineage/decision semantics."""

        decisions = {decision.node_id: decision for decision in self.decisions}
        return tuple(
            (
                item.node.node_id,
                item.node.parent_node_id,
                item.node.depth,
                item.node.branch_relu_input,
                item.node.branch_neuron_index,
                item.node.branch_value,
                (
                    None
                    if item.branch_candidate is None
                    else (
                        item.branch_candidate.relu_input,
                        item.branch_candidate.neuron_index,
                    )
                ),
                (
                    "frontier"
                    if item.node.node_id not in decisions
                    else decisions[item.node.node_id].kind
                ),
                (
                    "frontier"
                    if item.node.node_id not in decisions
                    else decisions[item.node.node_id].reason
                ),
            )
            for item in self.evaluations
        )


@dataclass(frozen=True)
class _RuntimeNode:
    node: NativeReluSplitBabNode
    split_state: tuple[tuple[str, torch.Tensor], ...]

    def mapping(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in self.split_state}


@dataclass(order=True)
class _QueueEntry:
    priority: float
    serial: int
    node_id: str


@dataclass(frozen=True)
class _EvaluatedRuntimeNode:
    runtime_node: _RuntimeNode
    evaluation: NativeReluSplitBabEvaluation


def run_native_relu_split_bab(
    legacy_task_module: BFTaskModule,
    input_spec: InputSpec,
    *,
    linear_spec_C: torch.Tensor,
    run_id: str,
    config: NativeReluSplitBabConfig,
) -> NativeReluSplitBabTrace:
    """Run the bounded best-first queue with native batched node evaluation."""

    if not run_id:
        raise ValueError("native ReLU-split BaB run ID must be non-empty")
    config.validate()
    legacy_task_module.validate()
    lower, upper = _root_box_bounds(input_spec)
    objective = _normalize_scalar_objective(linear_spec_C)
    _root_interval, root_pre = _forward_ibp_trace_mlp(legacy_task_module, input_spec)
    root_splits = tuple(
        (
            name,
            torch.zeros(
                tuple(int(dim) for dim in pre.lower.shape[1:]),
                dtype=torch.int8,
                device=pre.lower.device,
            ),
        )
        for name, pre in sorted(root_pre.items())
    )
    if not root_splits:
        raise ValueError("native ReLU-split BaB requires at least one ReLU")
    root_mapping = {name: tensor.unsqueeze(0) for name, tensor in root_splits}
    root = _RuntimeNode(
        node=NativeReluSplitBabNode(
            node_id=f"{run_id}:n000000",
            parent_node_id=None,
            depth=0,
            branch_relu_input=None,
            branch_neuron_index=None,
            branch_value=0,
            split_state_hash=relu_split_state_hash(root_mapping),
        ),
        split_state=root_splits,
    )

    evaluations: list[NativeReluSplitBabEvaluation] = []
    decisions: list[NativeReluSplitBabDecision] = []
    native_stacks: list[NativeReluSplitBabStackTrace] = []
    runtime_by_id: dict[str, _EvaluatedRuntimeNode] = {}
    stack_count = 0
    batch_serial = 0
    next_node_serial = 1

    def evaluate(nodes: Sequence[_RuntimeNode]) -> None:
        nonlocal stack_count, batch_serial
        for start in range(0, len(nodes), config.max_eval_batch_size):
            chunk = tuple(nodes[start : start + config.max_eval_batch_size])
            batch_id = f"{run_id}:eval:{batch_serial:04d}"
            batch_serial += 1
            evaluated, compilation = _evaluate_native_node_batch(
                legacy_task_module,
                input_spec,
                objective=objective,
                nodes=chunk,
                batch_id=batch_id,
                config=config,
                parent_by_id=runtime_by_id,
            )
            stack_count += 1
            native_stacks.append(
                _native_stack_trace(
                    compilation,
                    stack_id=batch_id,
                    node_ids=tuple(item.node.node_id for item in chunk),
                )
            )
            evaluations.extend(item.evaluation for item in evaluated)
            runtime_by_id.update(
                {item.runtime_node.node.node_id: item for item in evaluated}
            )

    evaluate((root,))
    heap: list[_QueueEntry] = []
    root_evaluation = runtime_by_id[root.node.node_id].evaluation
    heapq.heappush(
        heap,
        _QueueEntry(root_evaluation.priority, 0, root.node.node_id),
    )
    max_queue_size = 1
    budget_exhausted = False

    while heap and not budget_exhausted:
        selected: list[_QueueEntry] = []
        for _unused in range(min(config.expansion_batch_size, len(heap))):
            selected.append(heapq.heappop(heap))
        generated: list[_RuntimeNode] = []
        for selected_index, entry in enumerate(selected):
            evaluated = runtime_by_id[entry.node_id]
            node = evaluated.runtime_node.node
            result = evaluated.evaluation
            if result.lower >= config.threshold:
                decisions.append(
                    NativeReluSplitBabDecision(
                        decision_index=len(decisions),
                        node_id=node.node_id,
                        kind="prune",
                        reason="lower_bound_meets_threshold",
                    )
                )
                continue
            if node.depth >= config.max_depth:
                decisions.append(
                    NativeReluSplitBabDecision(
                        decision_index=len(decisions),
                        node_id=node.node_id,
                        kind="terminal",
                        reason="configured_depth_limit",
                    )
                )
                continue
            branch = result.branch_candidate
            if branch is None:
                decisions.append(
                    NativeReluSplitBabDecision(
                        decision_index=len(decisions),
                        node_id=node.node_id,
                        kind="terminal",
                        reason="no_unsplit_ambiguous_relu",
                    )
                )
                continue
            if len(evaluations) + len(generated) + 2 > config.max_nodes:
                for pending in selected[selected_index:]:
                    heapq.heappush(heap, pending)
                budget_exhausted = True
                break
            child_nodes: list[_RuntimeNode] = []
            for branch_value in (-1, 1):
                child_id = f"{run_id}:n{next_node_serial:06d}"
                next_node_serial += 1
                child_nodes.append(
                    _make_child_runtime_node(
                        evaluated.runtime_node,
                        child_id=child_id,
                        branch=branch,
                        branch_value=branch_value,
                    )
                )
            generated.extend(child_nodes)
            decisions.append(
                NativeReluSplitBabDecision(
                    decision_index=len(decisions),
                    node_id=node.node_id,
                    kind="expand",
                    reason="widest_unsplit_ambiguous_relu",
                    child_node_ids=tuple(item.node.node_id for item in child_nodes),
                    branch_candidate=branch,
                )
            )
        if generated:
            evaluate(generated)
            for child in generated:
                evaluation = runtime_by_id[child.node.node_id].evaluation
                heapq.heappush(
                    heap,
                    _QueueEntry(
                        evaluation.priority,
                        next_node_serial,
                        child.node.node_id,
                    ),
                )
            max_queue_size = max(max_queue_size, len(heap))

    frontier = tuple(
        entry.node_id
        for entry in sorted(heap, key=lambda item: (item.priority, item.serial))
    )
    status: BabQueueStatus = "budget_exhausted" if budget_exhausted else "complete"
    trace = NativeReluSplitBabTrace(
        run_id=run_id,
        status=status,
        termination_reason=(
            "node_budget_exhausted"
            if budget_exhausted
            else "configured_bounded_tree_exhausted"
        ),
        config=config,
        root_input_lower_hash=tensor_content_hash(lower),
        root_input_upper_hash=tensor_content_hash(upper),
        objective_hash=tensor_content_hash(objective),
        evaluations=tuple(evaluations),
        decisions=tuple(decisions),
        final_frontier_node_ids=frontier,
        native_stacks=tuple(native_stacks),
        native_stack_count=stack_count,
        max_queue_size=max_queue_size,
    )
    trace.validate()
    return trace


def _evaluate_native_node_batch(
    legacy_task_module: BFTaskModule,
    root_input_spec: InputSpec,
    *,
    objective: torch.Tensor,
    nodes: tuple[_RuntimeNode, ...],
    batch_id: str,
    config: NativeReluSplitBabConfig,
    parent_by_id: Mapping[str, _EvaluatedRuntimeNode],
) -> tuple[
    tuple[_EvaluatedRuntimeNode, ...],
    NativePlainCrownRepresentationCompilation,
]:
    if not nodes:
        raise ValueError("native ReLU-split node batch cannot be empty")
    batch_input = _repeat_box_input_spec(root_input_spec, count=len(nodes))
    names = tuple(name for name, _tensor in nodes[0].split_state)
    if any(
        tuple(name for name, _tensor in node.split_state) != names for node in nodes
    ):
        raise ValueError("native ReLU-split node batch changes split schema")
    split_batch = {
        name: torch.stack(
            tuple(dict(node.split_state)[name] for node in nodes), dim=0
        ).contiguous()
        for name in names
    }
    interval_env, relu_pre = _forward_ibp_trace_mlp(
        legacy_task_module,
        batch_input,
        relu_split_state=split_batch,
    )
    batch_state_hash = _exact_state_hash(
        batch_input,
        interval_env=interval_env,
        relu_pre=relu_pre,
        relu_split_state=split_batch,
    )
    compilation = compile_native_plain_crown_representation_query(
        legacy_task_module,
        batch_input,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=objective,
        intermediate_bounds_hash=batch_state_hash,
        query_id=batch_id,
        available_memory_bytes=config.available_memory_bytes,
        memory_budget_bytes=config.memory_budget_bytes,
        relu_split_state=split_batch,
        split_state_hash=relu_split_state_hash(split_batch),
        intermediate_bound_source=IntermediateBoundSource.LOCAL_FORWARD,
    )
    result, _task_trace = execute_native_plain_crown_representation_query(
        compilation,
        legacy_task_module=legacy_task_module,
        input_spec=batch_input,
        relu_pre=relu_pre,
        linear_spec_C=objective,
        relu_split_state=split_batch,
    )
    if (
        tuple(result.lower.shape) != (len(nodes), 1)
        or result.upper.shape != result.lower.shape
    ):
        raise ValueError(
            "native ReLU-split node batch must return one scalar objective"
        )
    hashes = tuple(sorted(compilation.hashes().items()))
    evaluated: list[_EvaluatedRuntimeNode] = []
    for index, runtime_node in enumerate(nodes):
        node_interval = {
            name: _slice_interval(state, index=index)
            for name, state in interval_env.items()
        }
        node_relu_pre = {
            name: _slice_interval(state, index=index)
            for name, state in relu_pre.items()
        }
        node_split = {
            name: tensor[index : index + 1].contiguous()
            for name, tensor in split_batch.items()
        }
        exact_hash = _exact_state_hash(
            _repeat_box_input_spec(root_input_spec, count=1),
            interval_env=node_interval,
            relu_pre=node_relu_pre,
            relu_split_state=node_split,
        )
        parent = (
            None
            if runtime_node.node.parent_node_id is None
            else parent_by_id.get(runtime_node.node.parent_node_id)
        )
        if runtime_node.node.depth > 0 and parent is None:
            raise ValueError("native ReLU-split node batch lacks evaluated parent")
        branch = _select_branch(
            node_relu_pre,
            relu_split_state=runtime_node.mapping(),
        )
        lower = float(result.lower[index, 0].item())
        upper = float(result.upper[index, 0].item())
        evaluation = NativeReluSplitBabEvaluation(
            node=runtime_node.node,
            lower=lower,
            upper=upper,
            priority=_priority(lower),
            exact_state_hash=exact_hash,
            parent_exact_state_hash=(
                None if parent is None else parent.evaluation.exact_state_hash
            ),
            eval_batch_id=batch_id,
            eval_batch_position=index,
            native_ir_hashes=hashes,
            branch_candidate=branch,
        )
        evaluation.validate()
        evaluated.append(
            _EvaluatedRuntimeNode(
                runtime_node=runtime_node,
                evaluation=evaluation,
            )
        )
    return tuple(evaluated), compilation


def _native_stack_trace(
    compilation: NativePlainCrownRepresentationCompilation,
    *,
    stack_id: str,
    node_ids: tuple[str, ...],
) -> NativeReluSplitBabStackTrace:
    source = compilation.source_bound_module
    split_inputs = tuple(
        value for value in source.graph.values if value.role == BoundValueRole.SPLIT
    )
    split_attrs = tuple(
        op.attrs
        for op in source.graph.ops
        if op.kind == BoundOpKind.RELU_RELAXATION
        and isinstance(op.attrs, SplitReluRelaxationAttrs)
    )
    local_forward_attrs = tuple(
        attrs
        for attrs in split_attrs
        if attrs.intermediate_bound_source == IntermediateBoundSource.LOCAL_FORWARD
    )
    split_capabilities = sum(
        capability.supports_split_state
        for capability in (
            *compilation.source_template.capabilities,
            *compilation.execution_template.capabilities,
        )
    )
    local_forward_provenance = sum(
        item.key == "intermediate_bound_source"
        and item.value == IntermediateBoundSource.LOCAL_FORWARD.value
        for item in (
            *compilation.source_template.provenance,
            *compilation.execution_template.provenance,
        )
    )
    launches = sum(
        isinstance(action, LaunchAction) for action in compilation.schedule.actions
    )
    trace = NativeReluSplitBabStackTrace(
        stack_id=stack_id,
        node_ids=node_ids,
        domain_batch_size=len(node_ids),
        bound_split_input_count=len(split_inputs),
        bound_split_relu_op_count=len(split_attrs),
        bound_local_forward_relu_op_count=len(local_forward_attrs),
        source_plan_split_state_present=(
            compilation.source_template.workload.split_state_present
        ),
        execution_plan_split_state_present=(
            compilation.execution_template.workload.split_state_present
        ),
        split_capability_count=split_capabilities,
        local_forward_provenance_count=local_forward_provenance,
        task_count=len(compilation.task_module.tasks),
        schedule_launch_count=launches,
        representation_policy_id=compilation.binding.trace.policy_id,
        storage_candidate_id=(
            compilation.source_instance.storage_decision.candidate_id
        ),
        native_ir_hashes=tuple(sorted(compilation.hashes().items())),
    )
    trace.validate()
    return trace


def _make_child_runtime_node(
    parent: _RuntimeNode,
    *,
    child_id: str,
    branch: ReluSplitBranch,
    branch_value: int,
) -> _RuntimeNode:
    if branch_value not in (-1, 1):
        raise ValueError("native ReLU-split child branch must be inactive/active")
    mapping = parent.mapping()
    if branch.relu_input not in mapping:
        raise ValueError("native ReLU-split branch references unknown ReLU")
    tensor = mapping[branch.relu_input].clone()
    flat = tensor.reshape(-1)
    if int(flat[branch.neuron_index].item()) != 0:
        raise ValueError("native ReLU-split branch repeats a fixed neuron")
    flat[branch.neuron_index] = branch_value
    mapping[branch.relu_input] = tensor
    batched = {name: value.unsqueeze(0) for name, value in mapping.items()}
    node = NativeReluSplitBabNode(
        node_id=child_id,
        parent_node_id=parent.node.node_id,
        depth=parent.node.depth + 1,
        branch_relu_input=branch.relu_input,
        branch_neuron_index=branch.neuron_index,
        branch_value=branch_value,
        split_state_hash=relu_split_state_hash(batched),
    )
    node.validate()
    return _RuntimeNode(node=node, split_state=tuple(sorted(mapping.items())))


def _select_branch(
    relu_pre: Mapping[str, IntervalState],
    *,
    relu_split_state: Mapping[str, torch.Tensor],
) -> Optional[ReluSplitBranch]:
    candidates: list[tuple[float, str, int, float, float]] = []
    for name in sorted(relu_pre):
        pre = relu_pre[name]
        split = relu_split_state.get(name)
        if split is None:
            raise ValueError("native ReLU branch selection lacks split state")
        lower = pre.lower.reshape(int(pre.lower.shape[0]), -1)
        upper = pre.upper.reshape(int(pre.upper.shape[0]), -1)
        if int(lower.shape[0]) != 1 or split.numel() != lower.numel():
            raise ValueError("native ReLU branch selection expects one node")
        split_flat = split.reshape(-1)
        for index in range(int(split_flat.numel())):
            low = float(lower[0, index].item())
            high = float(upper[0, index].item())
            if int(split_flat[index].item()) == 0 and low < 0.0 < high:
                candidates.append((high - low, name, index, low, high))
    if not candidates:
        return None
    width, name, index, low, high = min(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    )
    branch = ReluSplitBranch(
        relu_input=name,
        neuron_index=index,
        lower=low,
        upper=high,
        width=width,
    )
    branch.validate()
    return branch


def _normalize_scalar_objective(linear_spec_C: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(linear_spec_C) or not torch.is_floating_point(linear_spec_C):
        raise TypeError("native ReLU-split objective must be floating point tensor")
    objective = linear_spec_C.detach().contiguous()
    if objective.dim() == 3:
        if int(objective.shape[0]) != 1:
            raise ValueError("native ReLU-split objective batch must be one")
        objective = objective[0].contiguous()
    if objective.dim() != 2 or int(objective.shape[0]) != 1:
        raise ValueError("native ReLU-split v1 requires one scalar objective")
    if not bool(torch.isfinite(objective).all()):
        raise ValueError("native ReLU-split objective must be finite")
    return objective


def _root_box_bounds(input_spec: InputSpec) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(input_spec.perturbation, BoxPerturbation):
        raise NotImplementedError("native ReLU-split BaB v1 requires a box input")
    lower, upper = input_spec.perturbation.bounding_box(input_spec.center)
    if int(lower.shape[0]) != 1:
        raise ValueError("native ReLU-split BaB v1 requires one root domain")
    return lower, upper


def _repeat_box_input_spec(input_spec: InputSpec, *, count: int) -> InputSpec:
    lower, upper = _root_box_bounds(input_spec)
    if count < 1:
        raise ValueError("native ReLU-split domain count must be positive")
    repeats = (count, *(1 for _unused in lower.shape[1:]))
    return InputSpec.box(
        value_name=input_spec.value_name,
        lower=lower.repeat(repeats),
        upper=upper.repeat(repeats),
    )


def _slice_interval(state: IntervalState, *, index: int) -> IntervalState:
    return IntervalState(
        lower=state.lower[index : index + 1].contiguous(),
        upper=state.upper[index : index + 1].contiguous(),
    )


def _exact_state_hash(
    input_spec: InputSpec,
    *,
    interval_env: Mapping[str, IntervalState],
    relu_pre: Mapping[str, IntervalState],
    relu_split_state: Mapping[str, torch.Tensor],
) -> str:
    lower, upper = input_spec.perturbation.bounding_box(input_spec.center)
    payload = {
        "input_lower": tensor_content_hash(lower),
        "input_upper": tensor_content_hash(upper),
        "interval": {
            name: {
                "lower": tensor_content_hash(state.lower),
                "upper": tensor_content_hash(state.upper),
            }
            for name, state in sorted(interval_env.items())
        },
        "relu_pre": {
            name: {
                "lower": tensor_content_hash(state.lower),
                "upper": tensor_content_hash(state.upper),
            }
            for name, state in sorted(relu_pre.items())
        },
        "relu_split_state": relu_split_state_hash(relu_split_state),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _priority(lower: float) -> float:
    if not torch.isfinite(torch.tensor(lower)):
        raise ValueError("native ReLU-split queue bound must be finite")
    return float(f"{lower:.7g}")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
