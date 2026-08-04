"""Typed Plan, Task, and Schedule IR for fixed-wall-clock BaB scaling."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Tuple

from .workload import VerificationWorkloadSourceIR

SEARCH_SCALING_PLAN_IR_SCHEMA_VERSION = "boundflow.search-scaling-plan-ir/v1"
SEARCH_SCALING_TASK_IR_SCHEMA_VERSION = "boundflow.search-scaling-task-ir/v1"
SEARCH_SCALING_SCHEDULE_IR_SCHEMA_VERSION = "boundflow.search-scaling-schedule-ir/v1"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class NativeBabSearchBudgetIR:
    """One registered node/depth budget; all other policies stay invariant."""

    budget_id: str
    max_nodes: int
    max_depth: int

    def validate(self) -> None:
        if (
            not self.budget_id
            or self.max_nodes < 1
            or self.max_depth < 0
            or self.max_nodes > 2 ** (self.max_depth + 1) - 1
        ):
            raise ValueError("search-scaling budget IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "budget_id": self.budget_id,
            "max_nodes": self.max_nodes,
            "max_depth": self.max_depth,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeBabSearchScalingPlanIR:
    """Frozen workloads, budgets, resources, and evidence boundary."""

    plan_id: str
    benchmark_commit: str
    native_code_revision: str
    workloads: Tuple[VerificationWorkloadSourceIR, ...]
    budgets: Tuple[NativeBabSearchBudgetIR, ...]
    repeats: int
    timeout_seconds: int
    torch_threads: int
    optimizer_steps: int
    search_steps: int
    expansion_batch_size: int
    max_eval_batch_size: int
    timing_boundary: str = "fresh_process_start_to_structured_result"
    claim_boundary: str = "same_algorithm_cpu_fixed_deadline_search_coverage"
    schema_version: str = SEARCH_SCALING_PLAN_IR_SCHEMA_VERSION

    def validate(self) -> None:
        workload_ids = tuple(item.workload_id for item in self.workloads)
        budget_ids = tuple(item.budget_id for item in self.budgets)
        budget_pairs = tuple((item.max_nodes, item.max_depth) for item in self.budgets)
        if (
            self.schema_version != SEARCH_SCALING_PLAN_IR_SCHEMA_VERSION
            or not self.plan_id
            or not _is_revision(self.benchmark_commit)
            or not _is_sha256(self.native_code_revision)
            or len(self.workloads) != 3
            or len(workload_ids) != len(set(workload_ids))
            or len(self.budgets) != 3
            or len(budget_ids) != len(set(budget_ids))
            or budget_pairs != ((7, 2), (31, 4), (127, 6))
            or self.repeats != 3
            or self.timeout_seconds != 60
            or self.torch_threads < 1
            or self.optimizer_steps < 0
            or self.search_steps < 0
            or self.expansion_batch_size < 1
            or self.max_eval_batch_size < self.expansion_batch_size
            or self.timing_boundary != "fresh_process_start_to_structured_result"
            or self.claim_boundary
            != "same_algorithm_cpu_fixed_deadline_search_coverage"
        ):
            raise ValueError("search-scaling Plan IR is invalid")
        for workload in self.workloads:
            workload.validate()
        for budget in self.budgets:
            budget.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "benchmark_commit": self.benchmark_commit,
            "native_code_revision": self.native_code_revision,
            "workloads": [item.to_dict() for item in self.workloads],
            "budgets": [item.to_dict() for item in self.budgets],
            "repeats": self.repeats,
            "timeout_seconds": self.timeout_seconds,
            "torch_threads": self.torch_threads,
            "optimizer_steps": self.optimizer_steps,
            "search_steps": self.search_steps,
            "expansion_batch_size": self.expansion_batch_size,
            "max_eval_batch_size": self.max_eval_batch_size,
            "timing_boundary": self.timing_boundary,
            "process_isolation": "one_fresh_process_per_workload_budget_repeat",
            "claim_boundary": self.claim_boundary,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeBabSearchScalingTaskIR:
    """One fresh-process workload/budget/repeat execution."""

    task_id: str
    plan_hash: str
    workload_id: str
    workload_source_hash: str
    budget_id: str
    budget_hash: str
    repeat_index: int
    group_order_index: int

    def validate(self) -> None:
        if (
            not self.task_id
            or not self.workload_id
            or not self.budget_id
            or any(
                not _is_sha256(value)
                for value in (
                    self.plan_hash,
                    self.workload_source_hash,
                    self.budget_hash,
                )
            )
            or self.repeat_index < 0
            or self.group_order_index < 0
        ):
            raise ValueError("search-scaling Task IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "task_id": self.task_id,
            "plan_hash": self.plan_hash,
            "workload_id": self.workload_id,
            "workload_source_hash": self.workload_source_hash,
            "budget_id": self.budget_id,
            "budget_hash": self.budget_hash,
            "repeat_index": self.repeat_index,
            "group_order_index": self.group_order_index,
            "fresh_process": True,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeBabSearchScalingTaskIRModule:
    """Closed 3 workload x 3 budget x 3 repeat experiment graph."""

    plan_hash: str
    tasks: Tuple[NativeBabSearchScalingTaskIR, ...]
    schema_version: str = SEARCH_SCALING_TASK_IR_SCHEMA_VERSION

    def validate_against(self, plan: NativeBabSearchScalingPlanIR) -> None:
        plan.validate()
        plan_hash = plan.stable_hash()
        workload_by_id = {item.workload_id: item for item in plan.workloads}
        budget_by_id = {item.budget_id: item for item in plan.budgets}
        task_ids = tuple(item.task_id for item in self.tasks)
        if (
            self.schema_version != SEARCH_SCALING_TASK_IR_SCHEMA_VERSION
            or self.plan_hash != plan_hash
            or len(self.tasks) != len(plan.workloads) * len(plan.budgets) * plan.repeats
            or len(task_ids) != len(set(task_ids))
        ):
            raise ValueError("search-scaling Task IR module differs")
        observed: set[tuple[str, str, int]] = set()
        for task in self.tasks:
            task.validate()
            workload = workload_by_id.get(task.workload_id)
            budget = budget_by_id.get(task.budget_id)
            identity = (task.workload_id, task.budget_id, task.repeat_index)
            if (
                task.plan_hash != plan_hash
                or workload is None
                or budget is None
                or task.workload_source_hash != workload.stable_hash()
                or task.budget_hash != budget.stable_hash()
                or task.repeat_index not in range(plan.repeats)
                or task.group_order_index not in range(len(plan.budgets))
                or identity in observed
            ):
                raise ValueError("search-scaling Task/Plan binding differs")
            observed.add(identity)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "tasks": [item.to_dict() for item in self.tasks],
        }

    def stable_hash(self, plan: NativeBabSearchScalingPlanIR) -> str:
        self.validate_against(plan)
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class NativeBabSearchScalingScheduleIR:
    """Latin-rotated budget order with an exact fresh-process dispatch list."""

    plan_hash: str
    task_ir_hash: str
    budget_orders: Tuple[Tuple[str, ...], ...]
    ordered_task_ids: Tuple[str, ...]
    fresh_process_task_ids: Tuple[str, ...]
    schema_version: str = SEARCH_SCALING_SCHEDULE_IR_SCHEMA_VERSION

    def validate_against(
        self,
        plan: NativeBabSearchScalingPlanIR,
        task_ir: NativeBabSearchScalingTaskIRModule,
    ) -> None:
        task_ir.validate_against(plan)
        task_hash = task_ir.stable_hash(plan)
        expected_orders = _budget_orders(plan)
        task_ids = tuple(item.task_id for item in task_ir.tasks)
        if (
            self.schema_version != SEARCH_SCALING_SCHEDULE_IR_SCHEMA_VERSION
            or self.plan_hash != plan.stable_hash()
            or self.task_ir_hash != task_hash
            or self.budget_orders != expected_orders
            or self.ordered_task_ids != task_ids
            or self.fresh_process_task_ids != task_ids
        ):
            raise ValueError("search-scaling Schedule IR differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "budget_orders": [list(item) for item in self.budget_orders],
            "ordered_task_ids": list(self.ordered_task_ids),
            "fresh_process_task_ids": list(self.fresh_process_task_ids),
            "dispatch": "sequential_workload_repeat_then_latin_rotated_budget",
            "timing_boundary": "fresh_process_start_to_structured_result",
        }

    def stable_hash(
        self,
        plan: NativeBabSearchScalingPlanIR,
        task_ir: NativeBabSearchScalingTaskIRModule,
    ) -> str:
        self.validate_against(plan, task_ir)
        return _canonical_hash(self.to_dict())


def _budget_orders(
    plan: NativeBabSearchScalingPlanIR,
) -> Tuple[Tuple[str, ...], ...]:
    budget_ids = tuple(item.budget_id for item in plan.budgets)
    return tuple(
        budget_ids[offset:] + budget_ids[:offset] for offset in range(plan.repeats)
    )


def compile_search_scaling_task_ir(
    plan: NativeBabSearchScalingPlanIR,
) -> NativeBabSearchScalingTaskIRModule:
    plan.validate()
    plan_hash = plan.stable_hash()
    budget_by_id = {item.budget_id: item for item in plan.budgets}
    tasks: list[NativeBabSearchScalingTaskIR] = []
    for workload in plan.workloads:
        for repeat_index, order in enumerate(_budget_orders(plan)):
            for group_order_index, budget_id in enumerate(order):
                budget = budget_by_id[budget_id]
                tasks.append(
                    NativeBabSearchScalingTaskIR(
                        task_id=(
                            f"{workload.workload_id}:repeat:{repeat_index}:"
                            f"budget:{budget_id}"
                        ),
                        plan_hash=plan_hash,
                        workload_id=workload.workload_id,
                        workload_source_hash=workload.stable_hash(),
                        budget_id=budget_id,
                        budget_hash=budget.stable_hash(),
                        repeat_index=repeat_index,
                        group_order_index=group_order_index,
                    )
                )
    task_ir = NativeBabSearchScalingTaskIRModule(
        plan_hash=plan_hash,
        tasks=tuple(tasks),
    )
    task_ir.validate_against(plan)
    return task_ir


def compile_search_scaling_schedule_ir(
    plan: NativeBabSearchScalingPlanIR,
    task_ir: NativeBabSearchScalingTaskIRModule,
) -> NativeBabSearchScalingScheduleIR:
    task_ir.validate_against(plan)
    task_ids = tuple(item.task_id for item in task_ir.tasks)
    schedule = NativeBabSearchScalingScheduleIR(
        plan_hash=plan.stable_hash(),
        task_ir_hash=task_ir.stable_hash(plan),
        budget_orders=_budget_orders(plan),
        ordered_task_ids=task_ids,
        fresh_process_task_ids=task_ids,
    )
    schedule.validate_against(plan, task_ir)
    return schedule


__all__ = [
    "SEARCH_SCALING_PLAN_IR_SCHEMA_VERSION",
    "SEARCH_SCALING_SCHEDULE_IR_SCHEMA_VERSION",
    "SEARCH_SCALING_TASK_IR_SCHEMA_VERSION",
    "NativeBabSearchBudgetIR",
    "NativeBabSearchScalingPlanIR",
    "NativeBabSearchScalingScheduleIR",
    "NativeBabSearchScalingTaskIR",
    "NativeBabSearchScalingTaskIRModule",
    "compile_search_scaling_schedule_ir",
    "compile_search_scaling_task_ir",
]
