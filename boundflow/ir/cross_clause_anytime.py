"""Typed cross-clause floor plus anytime packed-escalation IR."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Literal, Optional, Tuple

CROSS_CLAUSE_ANYTIME_PLAN_SCHEMA_VERSION = "boundflow.cross-clause-anytime-plan/v1"
CROSS_CLAUSE_ANYTIME_DECISION_SCHEMA_VERSION = (
    "boundflow.cross-clause-anytime-decision/v1"
)
CROSS_CLAUSE_ANYTIME_TASK_SCHEMA_VERSION = "boundflow.cross-clause-anytime-task/v1"
CROSS_CLAUSE_ANYTIME_SCHEDULE_SCHEMA_VERSION = (
    "boundflow.cross-clause-anytime-schedule/v1"
)
CROSS_CLAUSE_ANYTIME_AGGREGATE_SCHEMA_VERSION = (
    "boundflow.cross-clause-anytime-aggregate/v1"
)
FinalStatus = Literal["verified", "unsafe", "unknown"]


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class NativeCrossClauseAnytimeTaskKind(Enum):
    """Closed staged action vocabulary."""

    EXECUTE_FLOOR = "execute_floor"
    DECIDE_ANYTIME = "decide_anytime"
    COMPILE_PACKED_PLAN = "compile_packed_plan"
    EXECUTE_PACKED_QUEUE = "execute_packed_queue"
    AGGREGATE_ORIGINAL_ORDINALS = "aggregate_original_ordinals"
    EMIT_RESULT = "emit_result"


@dataclass(frozen=True)
class NativeCrossClauseAnytimePlanIR:
    """Static NRIR-31 floor and optional NRIR-34 escalation contract."""

    plan_id: str
    floor_plan_hash: str
    floor_task_ir_hash: str
    floor_schedule_hash: str
    objective_matrix_hash: str
    thresholds_hash: str
    search_policy_hash: str
    optimizer_policy_hash: str
    clause_count: int = 9
    packed_original_clause_index: int = 0
    packed_max_nodes: int = 31
    packed_max_depth: int = 4
    sibling_group_size: int = 2
    child_refinement_cap: int = 128
    whole_query_timeout_ns: int = 60 * 1_000_000_000
    semantics_owner: str = "boundflow_cross_clause_anytime_objective_evaluator"
    schema_version: str = CROSS_CLAUSE_ANYTIME_PLAN_SCHEMA_VERSION

    def validate(self) -> None:
        hashes = (
            self.floor_plan_hash,
            self.floor_task_ir_hash,
            self.floor_schedule_hash,
            self.objective_matrix_hash,
            self.thresholds_hash,
            self.search_policy_hash,
            self.optimizer_policy_hash,
        )
        if (
            self.schema_version != CROSS_CLAUSE_ANYTIME_PLAN_SCHEMA_VERSION
            or not self.plan_id
            or any(not _is_sha256(value) for value in hashes)
            or self.clause_count != 9
            or self.packed_original_clause_index != 0
            or (self.packed_max_nodes, self.packed_max_depth) != (31, 4)
            or self.sibling_group_size != 2
            or self.child_refinement_cap != 128
            or self.whole_query_timeout_ns != 60 * 1_000_000_000
            or self.semantics_owner
            != "boundflow_cross_clause_anytime_objective_evaluator"
        ):
            raise ValueError("cross-clause anytime Plan IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "floor_plan_hash": self.floor_plan_hash,
            "floor_task_ir_hash": self.floor_task_ir_hash,
            "floor_schedule_hash": self.floor_schedule_hash,
            "objective_matrix_hash": self.objective_matrix_hash,
            "thresholds_hash": self.thresholds_hash,
            "search_policy_hash": self.search_policy_hash,
            "optimizer_policy_hash": self.optimizer_policy_hash,
            "clause_count": self.clause_count,
            "packed_original_clause_index": self.packed_original_clause_index,
            "packed_max_nodes": self.packed_max_nodes,
            "packed_max_depth": self.packed_max_depth,
            "sibling_group_size": self.sibling_group_size,
            "child_refinement_cap": self.child_refinement_cap,
            "whole_query_timeout_ns": self.whole_query_timeout_ns,
            "execution_order": [
                "nrir31_floor",
                "anytime_decision",
                "optional_nrir34_packed_clause",
                "monotone_original_ordinal_aggregate",
            ],
            "semantics_owner": self.semantics_owner,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeCrossClauseAnytimeDecisionIR:
    """Dynamic admission bound to the accepted floor clause source."""

    plan_hash: str
    floor_trace_hash: str
    floor_completed_original_clause_indices: Tuple[int, ...]
    floor_status: FinalStatus
    floor_verified_clause_indices: Tuple[int, ...]
    floor_unresolved_clause_indices: Tuple[int, ...]
    floor_unsafe_clause_index: Optional[int]
    admitted_original_clause_index: Optional[int]
    root_refinement_plan_hash: Optional[str]
    root_refinement_semantic_trace_hash: Optional[str]
    root_final_intermediate_bounds_hash: Optional[str]
    admitted: bool
    reason: str
    schema_version: str = CROSS_CLAUSE_ANYTIME_DECISION_SCHEMA_VERSION

    def validate_against(self, plan: NativeCrossClauseAnytimePlanIR) -> None:
        plan.validate()
        sequences = (
            self.floor_completed_original_clause_indices,
            self.floor_verified_clause_indices,
            self.floor_unresolved_clause_indices,
        )
        source_hashes = (
            self.root_refinement_plan_hash,
            self.root_refinement_semantic_trace_hash,
            self.root_final_intermediate_bounds_hash,
        )
        if (
            self.schema_version != CROSS_CLAUSE_ANYTIME_DECISION_SCHEMA_VERSION
            or self.plan_hash != plan.stable_hash()
            or not _is_sha256(self.floor_trace_hash)
            or any(tuple(sorted(set(values))) != values for values in sequences)
            or self.floor_status not in {"verified", "unsafe", "unknown"}
            or any(
                not set(values) <= set(range(plan.clause_count)) for values in sequences
            )
            or set(self.floor_verified_clause_indices)
            & set(self.floor_unresolved_clause_indices)
            or not set(self.floor_completed_original_clause_indices)
            <= set(range(plan.clause_count))
            or self.floor_unsafe_clause_index is not None
            and not 0 <= self.floor_unsafe_clause_index < plan.clause_count
            or not self.reason
        ):
            raise ValueError("cross-clause anytime Decision IR is invalid")
        if self.floor_status == "verified" and (
            self.floor_verified_clause_indices != tuple(range(plan.clause_count))
            or self.floor_unresolved_clause_indices
            or self.floor_unsafe_clause_index is not None
        ):
            raise ValueError("cross-clause anytime verified floor differs")
        if self.floor_status == "unsafe" and self.floor_unsafe_clause_index is None:
            raise ValueError("cross-clause anytime unsafe floor lacks witness ordinal")
        if self.floor_status == "unknown" and (
            self.floor_unsafe_clause_index is not None
            or not self.floor_unresolved_clause_indices
        ):
            raise ValueError("cross-clause anytime unknown floor lacks open ordinal")
        if self.admitted:
            if (
                self.reason != "floor_complete_unresolved_clause_admitted"
                or self.floor_status != "unknown"
                or self.floor_completed_original_clause_indices
                != tuple(range(plan.clause_count))
                or self.floor_unsafe_clause_index is not None
                or self.admitted_original_clause_index
                != plan.packed_original_clause_index
                or self.admitted_original_clause_index
                not in self.floor_unresolved_clause_indices
                or any(not _is_sha256(value) for value in source_hashes)
            ):
                raise ValueError("cross-clause anytime admitted Decision differs")
        elif self.admitted_original_clause_index is not None or any(
            value is not None for value in source_hashes
        ):
            raise ValueError("cross-clause anytime skipped Decision retains source")

    def to_dict(self, plan: NativeCrossClauseAnytimePlanIR) -> dict[str, object]:
        self.validate_against(plan)
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "floor_trace_hash": self.floor_trace_hash,
            "floor_completed_original_clause_indices": list(
                self.floor_completed_original_clause_indices
            ),
            "floor_status": self.floor_status,
            "floor_verified_clause_indices": list(self.floor_verified_clause_indices),
            "floor_unresolved_clause_indices": list(
                self.floor_unresolved_clause_indices
            ),
            "floor_unsafe_clause_index": self.floor_unsafe_clause_index,
            "admitted_original_clause_index": self.admitted_original_clause_index,
            "root_refinement_plan_hash": self.root_refinement_plan_hash,
            "root_refinement_semantic_trace_hash": (
                self.root_refinement_semantic_trace_hash
            ),
            "root_final_intermediate_bounds_hash": (
                self.root_final_intermediate_bounds_hash
            ),
            "admitted": self.admitted,
            "reason": self.reason,
        }

    def stable_hash(self, plan: NativeCrossClauseAnytimePlanIR) -> str:
        return _canonical_hash(self.to_dict(plan))


@dataclass(frozen=True)
class NativeCrossClauseAnytimeTaskIRUnit:
    """One static floor, decision, optional packed, aggregate, or emit action."""

    sequence: int
    task_id: str
    kind: NativeCrossClauseAnytimeTaskKind
    dependency_task_ids: Tuple[str, ...]
    guard: str
    original_clause_index: Optional[int]
    input_contract_hash: str
    output_contract: str

    def validate(self) -> None:
        packed = self.kind in {
            NativeCrossClauseAnytimeTaskKind.COMPILE_PACKED_PLAN,
            NativeCrossClauseAnytimeTaskKind.EXECUTE_PACKED_QUEUE,
        }
        if (
            self.sequence < 0
            or not self.task_id
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or self.guard not in {"always", "decision_admitted_before_deadline"}
            or packed != (self.guard == "decision_admitted_before_deadline")
            or packed != (self.original_clause_index is not None)
            or not _is_sha256(self.input_contract_hash)
            or not self.output_contract
        ):
            raise ValueError("cross-clause anytime Task unit is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "sequence": self.sequence,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "dependency_task_ids": list(self.dependency_task_ids),
            "guard": self.guard,
            "original_clause_index": self.original_clause_index,
            "input_contract_hash": self.input_contract_hash,
            "output_contract": self.output_contract,
        }


@dataclass(frozen=True)
class NativeCrossClauseAnytimeTaskIRModule:
    """Six-stage static control-flow Task IR."""

    plan_hash: str
    tasks: Tuple[NativeCrossClauseAnytimeTaskIRUnit, ...]
    schema_version: str = CROSS_CLAUSE_ANYTIME_TASK_SCHEMA_VERSION

    def validate(self) -> None:
        expected_kinds = tuple(NativeCrossClauseAnytimeTaskKind)
        if (
            self.schema_version != CROSS_CLAUSE_ANYTIME_TASK_SCHEMA_VERSION
            or not _is_sha256(self.plan_hash)
            or len(self.tasks) != len(expected_kinds)
            or tuple(task.sequence for task in self.tasks)
            != tuple(range(len(self.tasks)))
            or tuple(task.kind for task in self.tasks) != expected_kinds
            or len({task.task_id for task in self.tasks}) != len(self.tasks)
        ):
            raise ValueError("cross-clause anytime Task IR is invalid")
        available: set[str] = set()
        for task in self.tasks:
            task.validate()
            if any(item not in available for item in task.dependency_task_ids):
                raise ValueError("cross-clause anytime Task dependency order differs")
            available.add(task.task_id)
        if self.tasks[-1].dependency_task_ids != (self.tasks[-2].task_id,):
            raise ValueError("cross-clause anytime emit dependency differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeCrossClauseAnytimeScheduleActionIR:
    """One sequential Schedule action bound to a static Task."""

    sequence: int
    action_id: str
    task_id: str
    kind: NativeCrossClauseAnytimeTaskKind
    guard: str

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "action_id": self.action_id,
            "task_id": self.task_id,
            "kind": self.kind.value,
            "guard": self.guard,
        }


@dataclass(frozen=True)
class NativeCrossClauseAnytimeScheduleIR:
    """Sequential floor-first anytime Schedule IR."""

    plan_hash: str
    task_ir_hash: str
    actions: Tuple[NativeCrossClauseAnytimeScheduleActionIR, ...]
    schema_version: str = CROSS_CLAUSE_ANYTIME_SCHEDULE_SCHEMA_VERSION

    def validate_against(self, task_ir: NativeCrossClauseAnytimeTaskIRModule) -> None:
        task_ir.validate()
        if (
            self.schema_version != CROSS_CLAUSE_ANYTIME_SCHEDULE_SCHEMA_VERSION
            or self.plan_hash != task_ir.plan_hash
            or self.task_ir_hash != task_ir.stable_hash()
            or len(self.actions) != len(task_ir.tasks)
        ):
            raise ValueError("cross-clause anytime Schedule IR differs")
        for index, (action, task) in enumerate(zip(self.actions, task_ir.tasks)):
            if (
                action.sequence != index
                or action.action_id != f"cross-clause-anytime.launch.{index:04d}"
                or action.task_id != task.task_id
                or action.kind != task.kind
                or action.guard != task.guard
            ):
                raise ValueError("cross-clause anytime Schedule/Task binding differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "actions": [action.to_dict() for action in self.actions],
            "dispatch": "sequential_floor_then_optional_anytime",
        }

    def stable_hash(self, task_ir: NativeCrossClauseAnytimeTaskIRModule) -> str:
        self.validate_against(task_ir)
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeCrossClauseAnytimeAggregateIR:
    """Monotone original-ordinal aggregate over floor and optional packed verdict."""

    plan_hash: str
    decision_hash: str
    floor_trace_hash: str
    packed_queue_trace_hash: Optional[str]
    packed_status: Optional[FinalStatus]
    floor_status: FinalStatus
    floor_verified_clause_indices: Tuple[int, ...]
    floor_unresolved_clause_indices: Tuple[int, ...]
    floor_unsafe_clause_index: Optional[int]
    final_status: FinalStatus
    final_verified_clause_indices: Tuple[int, ...]
    final_unresolved_clause_indices: Tuple[int, ...]
    final_unsafe_clause_index: Optional[int]
    original_clause_indices: Tuple[int, ...] = tuple(range(9))
    schema_version: str = CROSS_CLAUSE_ANYTIME_AGGREGATE_SCHEMA_VERSION

    def validate_against(
        self,
        plan: NativeCrossClauseAnytimePlanIR,
        decision: NativeCrossClauseAnytimeDecisionIR,
    ) -> None:
        decision.validate_against(plan)
        packed_present = self.packed_queue_trace_hash is not None
        if (
            self.schema_version != CROSS_CLAUSE_ANYTIME_AGGREGATE_SCHEMA_VERSION
            or self.plan_hash != plan.stable_hash()
            or self.decision_hash != decision.stable_hash(plan)
            or self.floor_trace_hash != decision.floor_trace_hash
            or self.floor_status != decision.floor_status
            or self.floor_verified_clause_indices
            != decision.floor_verified_clause_indices
            or self.floor_unresolved_clause_indices
            != decision.floor_unresolved_clause_indices
            or self.floor_unsafe_clause_index != decision.floor_unsafe_clause_index
            or self.original_clause_indices != tuple(range(plan.clause_count))
            or self.floor_status not in {"verified", "unsafe", "unknown"}
            or self.final_status not in {"verified", "unsafe", "unknown"}
            or packed_present != (self.packed_status is not None)
            or packed_present
            and not _is_sha256(self.packed_queue_trace_hash)
            or packed_present
            and not decision.admitted
            or self.packed_status not in {None, "verified", "unsafe", "unknown"}
            or not set(self.floor_verified_clause_indices)
            <= set(self.final_verified_clause_indices)
        ):
            raise ValueError("cross-clause anytime aggregate is invalid")
        if not packed_present or self.packed_status == "unknown":
            expected = (
                self.floor_status,
                self.floor_verified_clause_indices,
                self.floor_unresolved_clause_indices,
                self.floor_unsafe_clause_index,
            )
        elif self.packed_status == "verified":
            verified = tuple(
                sorted(
                    {
                        *self.floor_verified_clause_indices,
                        plan.packed_original_clause_index,
                    }
                )
            )
            unresolved = tuple(
                item
                for item in self.floor_unresolved_clause_indices
                if item != plan.packed_original_clause_index
            )
            expected = (
                "verified" if len(verified) == plan.clause_count else "unknown",
                verified,
                unresolved,
                None,
            )
        else:
            expected = (
                "unsafe",
                self.floor_verified_clause_indices,
                self.floor_unresolved_clause_indices,
                plan.packed_original_clause_index,
            )
        actual = (
            self.final_status,
            self.final_verified_clause_indices,
            self.final_unresolved_clause_indices,
            self.final_unsafe_clause_index,
        )
        if actual != expected:
            raise ValueError("cross-clause anytime aggregate is non-monotone")

    def to_dict(
        self,
        plan: NativeCrossClauseAnytimePlanIR,
        decision: NativeCrossClauseAnytimeDecisionIR,
    ) -> dict[str, object]:
        self.validate_against(plan, decision)
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "decision_hash": self.decision_hash,
            "floor_trace_hash": self.floor_trace_hash,
            "packed_queue_trace_hash": self.packed_queue_trace_hash,
            "packed_status": self.packed_status,
            "floor_status": self.floor_status,
            "floor_verified_clause_indices": list(self.floor_verified_clause_indices),
            "floor_unresolved_clause_indices": list(
                self.floor_unresolved_clause_indices
            ),
            "floor_unsafe_clause_index": self.floor_unsafe_clause_index,
            "final_status": self.final_status,
            "final_verified_clause_indices": list(self.final_verified_clause_indices),
            "final_unresolved_clause_indices": list(
                self.final_unresolved_clause_indices
            ),
            "final_unsafe_clause_index": self.final_unsafe_clause_index,
            "original_clause_indices": list(self.original_clause_indices),
            "aggregation": "floor_monotone_optional_packed_upgrade",
        }

    def stable_hash(
        self,
        plan: NativeCrossClauseAnytimePlanIR,
        decision: NativeCrossClauseAnytimeDecisionIR,
    ) -> str:
        return _canonical_hash(self.to_dict(plan, decision))


def lower_native_cross_clause_anytime_ir(
    plan: NativeCrossClauseAnytimePlanIR,
) -> tuple[NativeCrossClauseAnytimeTaskIRModule, NativeCrossClauseAnytimeScheduleIR]:
    plan.validate()
    kinds = tuple(NativeCrossClauseAnytimeTaskKind)
    dependencies = ((), (0,), (1,), (2,), (1, 3), (4,))
    guards = (
        "always",
        "always",
        "decision_admitted_before_deadline",
        "decision_admitted_before_deadline",
        "always",
        "always",
    )
    outputs = (
        "native_objective_hard_clause_escalation_execution",
        "cross_clause_anytime_decision_ir",
        "sibling_pack_plan_ir",
        "sibling_pack_execution",
        "cross_clause_anytime_aggregate_ir",
        "cross_clause_anytime_execution",
    )
    task_ids = tuple(f"{plan.plan_id}:{kind.value}" for kind in kinds)
    tasks = tuple(
        NativeCrossClauseAnytimeTaskIRUnit(
            sequence=index,
            task_id=task_ids[index],
            kind=kind,
            dependency_task_ids=tuple(task_ids[item] for item in dependencies[index]),
            guard=guards[index],
            original_clause_index=(
                plan.packed_original_clause_index
                if guards[index] == "decision_admitted_before_deadline"
                else None
            ),
            input_contract_hash=_canonical_hash(
                {
                    "plan_hash": plan.stable_hash(),
                    "kind": kind.value,
                    "dependencies": list(dependencies[index]),
                }
            ),
            output_contract=outputs[index],
        )
        for index, kind in enumerate(kinds)
    )
    task_ir = NativeCrossClauseAnytimeTaskIRModule(
        plan_hash=plan.stable_hash(), tasks=tasks
    )
    task_ir.validate()
    schedule = NativeCrossClauseAnytimeScheduleIR(
        plan_hash=plan.stable_hash(),
        task_ir_hash=task_ir.stable_hash(),
        actions=tuple(
            NativeCrossClauseAnytimeScheduleActionIR(
                sequence=index,
                action_id=f"cross-clause-anytime.launch.{index:04d}",
                task_id=task.task_id,
                kind=task.kind,
                guard=task.guard,
            )
            for index, task in enumerate(tasks)
        ),
    )
    schedule.validate_against(task_ir)
    return task_ir, schedule


__all__ = [
    "FinalStatus",
    "NativeCrossClauseAnytimeAggregateIR",
    "NativeCrossClauseAnytimeDecisionIR",
    "NativeCrossClauseAnytimePlanIR",
    "NativeCrossClauseAnytimeScheduleIR",
    "NativeCrossClauseAnytimeTaskIRModule",
    "NativeCrossClauseAnytimeTaskIRUnit",
    "NativeCrossClauseAnytimeTaskKind",
    "lower_native_cross_clause_anytime_ir",
]
