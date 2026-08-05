"""Typed single-pass target-selection admission IR."""

# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Optional, Tuple

TARGET_ADMISSION_RECEIPT_SCHEMA_VERSION = "boundflow.target-admission-receipt/v1"
TARGET_ADMISSION_TASK_SCHEMA_VERSION = "boundflow.target-admission-task-ir/v1"
TARGET_ADMISSION_SCHEDULE_SCHEMA_VERSION = "boundflow.target-admission-schedule-ir/v1"


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
class NativeTargetAdmissionReceiptIR:
    """One exact selector invocation bound to its ordered inputs and outputs."""

    receipt_id: str
    plan_id: str
    primal_graph_hash: str
    input_bounds_hash: str
    split_state_hash: str
    initial_intermediate_bounds_hash: str
    effective_policy_hash: str
    target_table_hash: str
    target_count: int
    admission_receipt_hash: str
    objective_hash: Optional[str] = None
    objective_influence_hash: Optional[str] = None
    selector_schema: str = "boundflow.exact-target-selector/v1"
    selection_count: int = 1
    semantics_owner: str = "boundflow_single_pass_target_admission"
    performance_claimed: bool = False
    schema_version: str = TARGET_ADMISSION_RECEIPT_SCHEMA_VERSION

    def receipt_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "plan_id": self.plan_id,
            "primal_graph_hash": self.primal_graph_hash,
            "input_bounds_hash": self.input_bounds_hash,
            "split_state_hash": self.split_state_hash,
            "initial_intermediate_bounds_hash": (self.initial_intermediate_bounds_hash),
            "effective_policy_hash": self.effective_policy_hash,
            "target_table_hash": self.target_table_hash,
            "target_count": self.target_count,
            "selector_schema": self.selector_schema,
            "selection_count": self.selection_count,
            "semantics_owner": self.semantics_owner,
            "performance_claimed": self.performance_claimed,
        }
        if self.objective_hash is not None:
            payload["objective_hash"] = self.objective_hash
            payload["objective_influence_hash"] = self.objective_influence_hash
        return payload

    def expected_receipt_hash(self) -> str:
        return _canonical_hash(self.receipt_dict())

    def validate(self) -> None:
        required_hashes = (
            self.primal_graph_hash,
            self.input_bounds_hash,
            self.split_state_hash,
            self.initial_intermediate_bounds_hash,
            self.effective_policy_hash,
            self.target_table_hash,
            self.admission_receipt_hash,
        )
        objective_present = self.objective_hash is not None
        if (
            self.schema_version != TARGET_ADMISSION_RECEIPT_SCHEMA_VERSION
            or not self.receipt_id
            or not self.plan_id
            or any(not _is_sha256(value) for value in required_hashes)
            or objective_present != (self.objective_influence_hash is not None)
            or (
                objective_present
                and (
                    not _is_sha256(self.objective_hash)
                    or not _is_sha256(self.objective_influence_hash)
                )
            )
            or self.target_count < 1
            or self.selector_schema != "boundflow.exact-target-selector/v1"
            or self.selection_count != 1
            or self.semantics_owner != "boundflow_single_pass_target_admission"
            or self.performance_claimed is not False
            or self.admission_receipt_hash != self.expected_receipt_hash()
        ):
            raise ValueError("target admission receipt is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            **self.receipt_dict(),
            "admission_receipt_hash": self.admission_receipt_hash,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


class NativeTargetAdmissionTaskKind(str, Enum):
    SELECT_EXACT_TARGETS = "select_exact_targets"
    ADMIT_TARGET_RECEIPT = "admit_target_receipt"


@dataclass(frozen=True)
class NativeTargetAdmissionTaskIRUnit:
    task_id: str
    kind: NativeTargetAdmissionTaskKind
    dependency_task_ids: Tuple[str, ...]
    input_value_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]

    def validate(self) -> None:
        if (
            not self.task_id
            or not self.input_value_ids
            or not self.output_value_ids
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
        ):
            raise ValueError("target admission Task is invalid")

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
class NativeTargetAdmissionTaskIRModule:
    module_id: str
    source_plan_hash: str
    receipt_hash: str
    tasks: Tuple[NativeTargetAdmissionTaskIRUnit, ...]
    output_task_id: str
    schema_version: str = TARGET_ADMISSION_TASK_SCHEMA_VERSION

    def validate(self, *, receipt: NativeTargetAdmissionReceiptIR) -> None:
        receipt.validate()
        if (
            self.schema_version != TARGET_ADMISSION_TASK_SCHEMA_VERSION
            or not self.module_id
            or not _is_sha256(self.source_plan_hash)
            or self.receipt_hash != receipt.stable_hash()
            or tuple(task.kind for task in self.tasks)
            != tuple(NativeTargetAdmissionTaskKind)
            or self.output_task_id != self.tasks[-1].task_id
        ):
            raise ValueError("target admission Task module differs")
        completed: set[str] = set()
        available = {
            "admission.initial_bounds",
            "admission.effective_policy",
            "admission.objective_influence",
        }
        for task in self.tasks:
            task.validate()
            if any(item not in completed for item in task.dependency_task_ids):
                raise ValueError("target admission Task dependency is late")
            if any(item not in available for item in task.input_value_ids):
                raise ValueError("target admission Task input is late")
            if any(item in available for item in task.output_value_ids):
                raise ValueError("target admission Task output is redefined")
            completed.add(task.task_id)
            available.update(task.output_value_ids)

    def to_dict(self, *, receipt: NativeTargetAdmissionReceiptIR) -> dict[str, object]:
        self.validate(receipt=receipt)
        return {
            "schema_version": self.schema_version,
            "module_id": self.module_id,
            "source_plan_hash": self.source_plan_hash,
            "receipt_hash": self.receipt_hash,
            "tasks": [task.to_dict() for task in self.tasks],
            "output_task_id": self.output_task_id,
        }

    def stable_hash(self, *, receipt: NativeTargetAdmissionReceiptIR) -> str:
        return _canonical_hash(self.to_dict(receipt=receipt))


@dataclass(frozen=True)
class NativeTargetAdmissionScheduleAction:
    action_id: str
    sequence: int
    task_id: str

    def validate(self) -> None:
        if not self.action_id or self.sequence < 0 or not self.task_id:
            raise ValueError("target admission Schedule action is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "action_id": self.action_id,
            "sequence": self.sequence,
            "task_id": self.task_id,
        }


@dataclass(frozen=True)
class NativeTargetAdmissionScheduleIR:
    schedule_id: str
    receipt_hash: str
    task_module_hash: str
    actions: Tuple[NativeTargetAdmissionScheduleAction, ...]
    production_selection_launches: int = 1
    full_replay_selection_launches: int = 0
    schema_version: str = TARGET_ADMISSION_SCHEDULE_SCHEMA_VERSION

    def validate(
        self,
        *,
        receipt: NativeTargetAdmissionReceiptIR,
        task_module: NativeTargetAdmissionTaskIRModule,
    ) -> None:
        task_module.validate(receipt=receipt)
        if (
            self.schema_version != TARGET_ADMISSION_SCHEDULE_SCHEMA_VERSION
            or not self.schedule_id
            or self.receipt_hash != receipt.stable_hash()
            or self.task_module_hash != task_module.stable_hash(receipt=receipt)
            or len(self.actions) != len(task_module.tasks)
            or self.production_selection_launches != 1
            or self.full_replay_selection_launches != 0
        ):
            raise ValueError("target admission Schedule differs")
        for index, (action, task) in enumerate(zip(self.actions, task_module.tasks)):
            action.validate()
            if action.sequence != index or action.task_id != task.task_id:
                raise ValueError("target admission Schedule/Task differs")

    def to_dict(
        self,
        *,
        receipt: NativeTargetAdmissionReceiptIR,
        task_module: NativeTargetAdmissionTaskIRModule,
    ) -> dict[str, object]:
        self.validate(receipt=receipt, task_module=task_module)
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "receipt_hash": self.receipt_hash,
            "task_module_hash": self.task_module_hash,
            "actions": [action.to_dict() for action in self.actions],
            "production_selection_launches": self.production_selection_launches,
            "full_replay_selection_launches": (self.full_replay_selection_launches),
        }

    def stable_hash(
        self,
        *,
        receipt: NativeTargetAdmissionReceiptIR,
        task_module: NativeTargetAdmissionTaskIRModule,
    ) -> str:
        return _canonical_hash(self.to_dict(receipt=receipt, task_module=task_module))


def lower_native_target_admission_ir(
    *, source_plan_hash: str, receipt: NativeTargetAdmissionReceiptIR
) -> tuple[NativeTargetAdmissionTaskIRModule, NativeTargetAdmissionScheduleIR]:
    """Lower one receipt into an explicit two-stage admission schedule."""

    receipt.validate()
    if not _is_sha256(source_plan_hash):
        raise ValueError("target admission source Plan hash is invalid")
    tasks = (
        NativeTargetAdmissionTaskIRUnit(
            task_id=f"{receipt.receipt_id}:select",
            kind=NativeTargetAdmissionTaskKind.SELECT_EXACT_TARGETS,
            dependency_task_ids=(),
            input_value_ids=(
                "admission.initial_bounds",
                "admission.effective_policy",
                "admission.objective_influence",
            ),
            output_value_ids=("admission.ordered_targets",),
        ),
        NativeTargetAdmissionTaskIRUnit(
            task_id=f"{receipt.receipt_id}:admit",
            kind=NativeTargetAdmissionTaskKind.ADMIT_TARGET_RECEIPT,
            dependency_task_ids=(f"{receipt.receipt_id}:select",),
            input_value_ids=("admission.ordered_targets",),
            output_value_ids=("admission.receipt",),
        ),
    )
    task_module = NativeTargetAdmissionTaskIRModule(
        module_id=f"{receipt.receipt_id}:tasks",
        source_plan_hash=source_plan_hash,
        receipt_hash=receipt.stable_hash(),
        tasks=tasks,
        output_task_id=tasks[-1].task_id,
    )
    task_module.validate(receipt=receipt)
    schedule = NativeTargetAdmissionScheduleIR(
        schedule_id=f"{receipt.receipt_id}:schedule",
        receipt_hash=receipt.stable_hash(),
        task_module_hash=task_module.stable_hash(receipt=receipt),
        actions=tuple(
            NativeTargetAdmissionScheduleAction(
                action_id=f"{task.task_id}:launch",
                sequence=index,
                task_id=task.task_id,
            )
            for index, task in enumerate(tasks)
        ),
    )
    schedule.validate(receipt=receipt, task_module=task_module)
    return task_module, schedule


__all__ = [
    "NativeTargetAdmissionReceiptIR",
    "NativeTargetAdmissionScheduleIR",
    "NativeTargetAdmissionTaskIRModule",
    "NativeTargetAdmissionTaskIRUnit",
    "NativeTargetAdmissionTaskKind",
    "lower_native_target_admission_ir",
]
