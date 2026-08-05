"""Typed ragged IR for cross-axis verification lower batches."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,missing-class-docstring,duplicate-code

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from typing import Tuple

CROSS_AXIS_VERIFICATION_BATCH_PLAN_SCHEMA_VERSION = (
    "boundflow.cross-axis-verification-batch-plan/v1"
)
CROSS_AXIS_VERIFICATION_BATCH_INSTANCE_SCHEMA_VERSION = (
    "boundflow.cross-axis-verification-batch-instance/v1"
)
CROSS_AXIS_VERIFICATION_BATCH_TASK_SCHEMA_VERSION = (
    "boundflow.cross-axis-verification-batch-task-ir/v1"
)
CROSS_AXIS_VERIFICATION_BATCH_SCHEDULE_SCHEMA_VERSION = (
    "boundflow.cross-axis-verification-batch-schedule-ir/v1"
)
CROSS_AXIS_VERIFICATION_BATCH_TRACE_SCHEMA_VERSION = (
    "boundflow.cross-axis-verification-batch-trace/v1"
)


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


class NativeCrossAxisVerificationBatchTaskKind(str, Enum):
    """Exact stages of one cross-axis branch-score launch."""

    ADMIT_READY_SET = "admit_ready_set"
    PACK_CHILD_DOMAINS = "pack_child_domains"
    EXECUTE_LOWER_BATCH = "execute_lower_batch"
    SEGMENT_REDUCE = "segment_reduce"
    COMMIT_BRANCHES = "commit_branches"


@dataclass(frozen=True)
class NativeCrossAxisVerificationBatchSegmentIR:
    """One clause/node owner inside a ragged candidate/child batch."""

    clause_ordinal: int
    node_id: str
    branch_plan_hash: str
    capsule_hash: str
    objective_hash: str
    selected_state_hash: str
    candidate_offset: int
    candidate_count: int
    child_domain_offset: int
    child_domain_count: int

    def validate(self) -> None:
        hashes = (
            self.branch_plan_hash,
            self.capsule_hash,
            self.objective_hash,
            self.selected_state_hash,
        )
        if (
            self.clause_ordinal < 0
            or not self.node_id
            or any(not _is_sha256(value) for value in hashes)
            or self.candidate_offset < 0
            or self.candidate_count < 1
            or self.child_domain_offset < 0
            or self.child_domain_count != 2 * self.candidate_count
        ):
            raise ValueError("cross-axis verification batch segment is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "clause_ordinal": self.clause_ordinal,
            "node_id": self.node_id,
            "branch_plan_hash": self.branch_plan_hash,
            "capsule_hash": self.capsule_hash,
            "objective_hash": self.objective_hash,
            "selected_state_hash": self.selected_state_hash,
            "candidate_offset": self.candidate_offset,
            "candidate_count": self.candidate_count,
            "child_domain_offset": self.child_domain_offset,
            "child_domain_count": self.child_domain_count,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeCrossAxisVerificationBatchPlanIR:
    """Plan-owned ragged layout for already-admitted scorer programs."""

    plan_id: str
    optimizer_policy_hash: str
    branch_policy_hash: str
    segments: Tuple[NativeCrossAxisVerificationBatchSegmentIR, ...]
    clause_count: int
    node_count: int
    candidate_count: int
    child_domain_count: int
    max_child_domains: int
    semantics_owner: str = "boundflow_cross_axis_verification_batch"
    performance_claimed: bool = False
    schema_version: str = CROSS_AXIS_VERIFICATION_BATCH_PLAN_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != CROSS_AXIS_VERIFICATION_BATCH_PLAN_SCHEMA_VERSION
            or not self.plan_id
            or not _is_sha256(self.optimizer_policy_hash)
            or not _is_sha256(self.branch_policy_hash)
            or not self.segments
            or self.clause_count != len({item.clause_ordinal for item in self.segments})
            or self.node_count != len(self.segments)
            or self.node_count != len({item.node_id for item in self.segments})
            or self.candidate_count < self.node_count
            or self.child_domain_count != 2 * self.candidate_count
            or self.max_child_domains < self.child_domain_count
            or self.semantics_owner != "boundflow_cross_axis_verification_batch"
            or self.performance_claimed is not False
        ):
            raise ValueError("cross-axis verification batch Plan IR is invalid")
        candidate_cursor = 0
        child_cursor = 0
        for segment in self.segments:
            segment.validate()
            if (
                segment.candidate_offset != candidate_cursor
                or segment.child_domain_offset != child_cursor
            ):
                raise ValueError(
                    "cross-axis verification batch segments are not packed"
                )
            candidate_cursor += segment.candidate_count
            child_cursor += segment.child_domain_count
        if (
            candidate_cursor != self.candidate_count
            or child_cursor != self.child_domain_count
        ):
            raise ValueError("cross-axis verification batch segment coverage differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "optimizer_policy_hash": self.optimizer_policy_hash,
            "branch_policy_hash": self.branch_policy_hash,
            "segments": [item.to_dict() for item in self.segments],
            "clause_count": self.clause_count,
            "node_count": self.node_count,
            "candidate_count": self.candidate_count,
            "child_domain_count": self.child_domain_count,
            "max_child_domains": self.max_child_domains,
            "semantics_owner": self.semantics_owner,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeCrossAxisVerificationBatchInstanceIR:
    """One ready-set binding of a cross-axis batch Plan."""

    instance_id: str
    plan_hash: str
    segment_hashes: Tuple[str, ...]
    ready_set_hash: str
    child_domain_count: int
    semantic_token: str
    schema_version: str = CROSS_AXIS_VERIFICATION_BATCH_INSTANCE_SCHEMA_VERSION

    def semantic_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "instance_id": self.instance_id,
            "plan_hash": self.plan_hash,
            "segment_hashes": list(self.segment_hashes),
            "ready_set_hash": self.ready_set_hash,
            "child_domain_count": self.child_domain_count,
        }

    def validate(self, *, plan: NativeCrossAxisVerificationBatchPlanIR) -> None:
        plan.validate()
        if (
            self.schema_version != CROSS_AXIS_VERIFICATION_BATCH_INSTANCE_SCHEMA_VERSION
            or not self.instance_id
            or self.plan_hash != plan.stable_hash()
            or self.segment_hashes
            != tuple(segment.stable_hash() for segment in plan.segments)
            or self.ready_set_hash
            != _canonical_hash([segment.to_dict() for segment in plan.segments])
            or self.child_domain_count != plan.child_domain_count
            or not _is_sha256(self.semantic_token)
            or self.semantic_token != _canonical_hash(self.semantic_dict())
        ):
            raise ValueError("cross-axis verification batch Instance IR differs")

    @classmethod
    def from_plan(
        cls, plan: NativeCrossAxisVerificationBatchPlanIR
    ) -> "NativeCrossAxisVerificationBatchInstanceIR":
        plan.validate()
        instance = cls(
            instance_id=f"{plan.plan_id}:instance",
            plan_hash=plan.stable_hash(),
            segment_hashes=tuple(segment.stable_hash() for segment in plan.segments),
            ready_set_hash=_canonical_hash(
                [segment.to_dict() for segment in plan.segments]
            ),
            child_domain_count=plan.child_domain_count,
            semantic_token="0" * 64,
        )
        instance = replace(
            instance, semantic_token=_canonical_hash(instance.semantic_dict())
        )
        instance.validate(plan=plan)
        return instance

    def to_dict(
        self, *, plan: NativeCrossAxisVerificationBatchPlanIR
    ) -> dict[str, object]:
        self.validate(plan=plan)
        return {**self.semantic_dict(), "semantic_token": self.semantic_token}

    def stable_hash(self, *, plan: NativeCrossAxisVerificationBatchPlanIR) -> str:
        return _canonical_hash(self.to_dict(plan=plan))


@dataclass(frozen=True)
class NativeCrossAxisVerificationBatchTaskIRUnit:
    task_id: str
    kind: NativeCrossAxisVerificationBatchTaskKind
    dependency_task_ids: Tuple[str, ...]
    input_value_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]

    def validate(self) -> None:
        if (
            not self.task_id
            or not self.input_value_ids
            or not self.output_value_ids
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or len(self.input_value_ids) != len(set(self.input_value_ids))
            or len(self.output_value_ids) != len(set(self.output_value_ids))
        ):
            raise ValueError("cross-axis verification batch Task unit is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "task_id": self.task_id,
            "kind": self.kind.value,
            "dependency_task_ids": list(self.dependency_task_ids),
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
        }


@dataclass(frozen=True)
class NativeCrossAxisVerificationBatchTaskIRModule:
    module_id: str
    plan_hash: str
    instance_hash: str
    tasks: Tuple[NativeCrossAxisVerificationBatchTaskIRUnit, ...]
    output_task_id: str
    schema_version: str = CROSS_AXIS_VERIFICATION_BATCH_TASK_SCHEMA_VERSION

    def validate(
        self,
        *,
        plan: NativeCrossAxisVerificationBatchPlanIR,
        instance: NativeCrossAxisVerificationBatchInstanceIR,
    ) -> None:
        instance.validate(plan=plan)
        expected_kinds = tuple(NativeCrossAxisVerificationBatchTaskKind)
        if (
            self.schema_version != CROSS_AXIS_VERIFICATION_BATCH_TASK_SCHEMA_VERSION
            or not self.module_id
            or self.plan_hash != plan.stable_hash()
            or self.instance_hash != instance.stable_hash(plan=plan)
            or tuple(item.kind for item in self.tasks) != expected_kinds
            or self.output_task_id != self.tasks[-1].task_id
        ):
            raise ValueError("cross-axis verification batch Task module is invalid")
        completed: set[str] = set()
        available = {"batch.plan", "batch.instance", "batch.programs"}
        for task in self.tasks:
            task.validate()
            if any(item not in completed for item in task.dependency_task_ids):
                raise ValueError("cross-axis verification batch dependency is late")
            if any(item not in available for item in task.input_value_ids):
                raise ValueError("cross-axis verification batch input is late")
            if any(item in available for item in task.output_value_ids):
                raise ValueError("cross-axis verification batch output is redefined")
            completed.add(task.task_id)
            available.update(task.output_value_ids)

    def to_dict(
        self,
        *,
        plan: NativeCrossAxisVerificationBatchPlanIR,
        instance: NativeCrossAxisVerificationBatchInstanceIR,
    ) -> dict[str, object]:
        self.validate(plan=plan, instance=instance)
        return {
            "schema_version": self.schema_version,
            "module_id": self.module_id,
            "plan_hash": self.plan_hash,
            "instance_hash": self.instance_hash,
            "tasks": [item.to_dict() for item in self.tasks],
            "output_task_id": self.output_task_id,
        }

    def stable_hash(
        self,
        *,
        plan: NativeCrossAxisVerificationBatchPlanIR,
        instance: NativeCrossAxisVerificationBatchInstanceIR,
    ) -> str:
        return _canonical_hash(self.to_dict(plan=plan, instance=instance))


@dataclass(frozen=True)
class NativeCrossAxisVerificationBatchScheduleAction:
    action_id: str
    sequence: int
    task_id: str

    def validate(self) -> None:
        if not self.action_id or self.sequence < 0 or not self.task_id:
            raise ValueError("cross-axis verification batch Schedule action is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "action_id": self.action_id,
            "sequence": self.sequence,
            "task_id": self.task_id,
        }


@dataclass(frozen=True)
class NativeCrossAxisVerificationBatchScheduleIR:
    schedule_id: str
    plan_hash: str
    instance_hash: str
    task_module_hash: str
    actions: Tuple[NativeCrossAxisVerificationBatchScheduleAction, ...]
    lower_launch_count: int
    schema_version: str = CROSS_AXIS_VERIFICATION_BATCH_SCHEDULE_SCHEMA_VERSION

    def validate(
        self,
        *,
        plan: NativeCrossAxisVerificationBatchPlanIR,
        instance: NativeCrossAxisVerificationBatchInstanceIR,
        task_module: NativeCrossAxisVerificationBatchTaskIRModule,
    ) -> None:
        task_module.validate(plan=plan, instance=instance)
        if (
            self.schema_version != CROSS_AXIS_VERIFICATION_BATCH_SCHEDULE_SCHEMA_VERSION
            or not self.schedule_id
            or self.plan_hash != plan.stable_hash()
            or self.instance_hash != instance.stable_hash(plan=plan)
            or self.task_module_hash
            != task_module.stable_hash(plan=plan, instance=instance)
            or len(self.actions) != len(task_module.tasks)
            or self.lower_launch_count != 1
        ):
            raise ValueError("cross-axis verification batch Schedule is invalid")
        for sequence, (action, task) in enumerate(zip(self.actions, task_module.tasks)):
            action.validate()
            if action.sequence != sequence or action.task_id != task.task_id:
                raise ValueError("cross-axis verification batch Schedule/Task differs")

    def to_dict(
        self,
        *,
        plan: NativeCrossAxisVerificationBatchPlanIR,
        instance: NativeCrossAxisVerificationBatchInstanceIR,
        task_module: NativeCrossAxisVerificationBatchTaskIRModule,
    ) -> dict[str, object]:
        self.validate(plan=plan, instance=instance, task_module=task_module)
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "plan_hash": self.plan_hash,
            "instance_hash": self.instance_hash,
            "task_module_hash": self.task_module_hash,
            "actions": [item.to_dict() for item in self.actions],
            "lower_launch_count": self.lower_launch_count,
        }

    def stable_hash(
        self,
        *,
        plan: NativeCrossAxisVerificationBatchPlanIR,
        instance: NativeCrossAxisVerificationBatchInstanceIR,
        task_module: NativeCrossAxisVerificationBatchTaskIRModule,
    ) -> str:
        return _canonical_hash(
            self.to_dict(plan=plan, instance=instance, task_module=task_module)
        )


@dataclass(frozen=True)
class NativeCrossAxisVerificationBatchTraceIR:
    plan_hash: str
    instance_hash: str
    task_module_hash: str
    schedule_hash: str
    batch_child_lower_hash: str
    segment_child_lower_hashes: Tuple[str, ...]
    segment_score_hashes: Tuple[str, ...]
    selected_candidate_ordinals: Tuple[int, ...]
    lower_launch_count: int
    performance_claimed: bool = False
    schema_version: str = CROSS_AXIS_VERIFICATION_BATCH_TRACE_SCHEMA_VERSION

    def validate(
        self,
        *,
        plan: NativeCrossAxisVerificationBatchPlanIR,
        instance: NativeCrossAxisVerificationBatchInstanceIR,
        task_module: NativeCrossAxisVerificationBatchTaskIRModule,
        schedule: NativeCrossAxisVerificationBatchScheduleIR,
    ) -> None:
        schedule.validate(plan=plan, instance=instance, task_module=task_module)
        hashes = (
            self.plan_hash,
            self.instance_hash,
            self.task_module_hash,
            self.schedule_hash,
            self.batch_child_lower_hash,
            *self.segment_child_lower_hashes,
            *self.segment_score_hashes,
        )
        if (
            self.schema_version != CROSS_AXIS_VERIFICATION_BATCH_TRACE_SCHEMA_VERSION
            or any(not _is_sha256(value) for value in hashes)
            or self.plan_hash != plan.stable_hash()
            or self.instance_hash != instance.stable_hash(plan=plan)
            or self.task_module_hash
            != task_module.stable_hash(plan=plan, instance=instance)
            or self.schedule_hash
            != schedule.stable_hash(
                plan=plan, instance=instance, task_module=task_module
            )
            or len(self.segment_child_lower_hashes) != plan.node_count
            or len(self.segment_score_hashes) != plan.node_count
            or len(self.selected_candidate_ordinals) != plan.node_count
            or any(
                ordinal < 0 or ordinal >= segment.candidate_count
                for ordinal, segment in zip(
                    self.selected_candidate_ordinals, plan.segments
                )
            )
            or self.lower_launch_count != 1
            or self.performance_claimed is not False
        ):
            raise ValueError("cross-axis verification batch Trace IR differs")

    def to_dict(
        self,
        *,
        plan: NativeCrossAxisVerificationBatchPlanIR,
        instance: NativeCrossAxisVerificationBatchInstanceIR,
        task_module: NativeCrossAxisVerificationBatchTaskIRModule,
        schedule: NativeCrossAxisVerificationBatchScheduleIR,
    ) -> dict[str, object]:
        self.validate(
            plan=plan,
            instance=instance,
            task_module=task_module,
            schedule=schedule,
        )
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "instance_hash": self.instance_hash,
            "task_module_hash": self.task_module_hash,
            "schedule_hash": self.schedule_hash,
            "batch_child_lower_hash": self.batch_child_lower_hash,
            "segment_child_lower_hashes": list(self.segment_child_lower_hashes),
            "segment_score_hashes": list(self.segment_score_hashes),
            "selected_candidate_ordinals": list(self.selected_candidate_ordinals),
            "lower_launch_count": self.lower_launch_count,
            "performance_claimed": self.performance_claimed,
        }

    def stable_hash(
        self,
        *,
        plan: NativeCrossAxisVerificationBatchPlanIR,
        instance: NativeCrossAxisVerificationBatchInstanceIR,
        task_module: NativeCrossAxisVerificationBatchTaskIRModule,
        schedule: NativeCrossAxisVerificationBatchScheduleIR,
    ) -> str:
        return _canonical_hash(
            self.to_dict(
                plan=plan,
                instance=instance,
                task_module=task_module,
                schedule=schedule,
            )
        )


def lower_native_cross_axis_verification_batch_ir(
    plan: NativeCrossAxisVerificationBatchPlanIR,
    instance: NativeCrossAxisVerificationBatchInstanceIR,
) -> tuple[
    NativeCrossAxisVerificationBatchTaskIRModule,
    NativeCrossAxisVerificationBatchScheduleIR,
]:
    """Lower a ragged ready set to one lower-bound launch."""

    instance.validate(plan=plan)
    definitions = (
        (
            NativeCrossAxisVerificationBatchTaskKind.ADMIT_READY_SET,
            ("batch.plan", "batch.instance", "batch.programs"),
            ("batch.admitted_programs",),
        ),
        (
            NativeCrossAxisVerificationBatchTaskKind.PACK_CHILD_DOMAINS,
            ("batch.admitted_programs",),
            ("batch.child_domains", "batch.segments"),
        ),
        (
            NativeCrossAxisVerificationBatchTaskKind.EXECUTE_LOWER_BATCH,
            ("batch.child_domains",),
            ("batch.child_lowers",),
        ),
        (
            NativeCrossAxisVerificationBatchTaskKind.SEGMENT_REDUCE,
            ("batch.child_lowers", "batch.segments"),
            ("batch.segment_scores",),
        ),
        (
            NativeCrossAxisVerificationBatchTaskKind.COMMIT_BRANCHES,
            ("batch.segment_scores",),
            ("batch.selected_branches",),
        ),
    )
    tasks: list[NativeCrossAxisVerificationBatchTaskIRUnit] = []
    dependencies: tuple[str, ...] = ()
    for kind, inputs, outputs in definitions:
        task_id = f"{plan.plan_id}:{kind.value}"
        tasks.append(
            NativeCrossAxisVerificationBatchTaskIRUnit(
                task_id=task_id,
                kind=kind,
                dependency_task_ids=dependencies,
                input_value_ids=inputs,
                output_value_ids=outputs,
            )
        )
        dependencies = (task_id,)
    task_module = NativeCrossAxisVerificationBatchTaskIRModule(
        module_id=f"{plan.plan_id}:tasks",
        plan_hash=plan.stable_hash(),
        instance_hash=instance.stable_hash(plan=plan),
        tasks=tuple(tasks),
        output_task_id=tasks[-1].task_id,
    )
    actions = tuple(
        NativeCrossAxisVerificationBatchScheduleAction(
            action_id=f"{plan.plan_id}:launch:{index:04d}:{task.kind.value}",
            sequence=index,
            task_id=task.task_id,
        )
        for index, task in enumerate(tasks)
    )
    schedule = NativeCrossAxisVerificationBatchScheduleIR(
        schedule_id=f"{plan.plan_id}:schedule",
        plan_hash=plan.stable_hash(),
        instance_hash=instance.stable_hash(plan=plan),
        task_module_hash=task_module.stable_hash(plan=plan, instance=instance),
        actions=actions,
        lower_launch_count=1,
    )
    schedule.validate(plan=plan, instance=instance, task_module=task_module)
    return task_module, schedule


__all__ = [
    "NativeCrossAxisVerificationBatchInstanceIR",
    "NativeCrossAxisVerificationBatchPlanIR",
    "NativeCrossAxisVerificationBatchScheduleIR",
    "NativeCrossAxisVerificationBatchSegmentIR",
    "NativeCrossAxisVerificationBatchTaskIRModule",
    "NativeCrossAxisVerificationBatchTaskKind",
    "NativeCrossAxisVerificationBatchTraceIR",
    "lower_native_cross_axis_verification_batch_ir",
]
