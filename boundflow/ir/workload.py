"""Typed Plan, Task, and Schedule IR for multi-workload verifier evaluation."""

# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions
# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import PurePosixPath
from typing import Tuple

MULTIWORKLOAD_PLAN_IR_SCHEMA_VERSION = "boundflow.multiworkload_plan_ir/v1"
MULTIWORKLOAD_TASK_IR_SCHEMA_VERSION = "boundflow.multiworkload_task_ir/v1"
MULTIWORKLOAD_SCHEDULE_IR_SCHEMA_VERSION = "boundflow.multiworkload_schedule_ir/v1"


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


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _relative_source_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


class VerifierBackendKind(Enum):
    """Backends admitted by the frozen multi-workload protocol."""

    BOUNDFLOW_NATIVE = "boundflow_native"
    EXTERNAL_ABCROWN = "external_abcrown"


class MultiWorkloadTaskKind(Enum):
    """Closed acquisition-to-result pipeline."""

    ACQUIRE_SOURCES = "acquire_sources"
    PARSE_QUERY = "parse_query"
    IMPORT_ONNX = "import_onnx"
    COMPILE_NATIVE = "compile_native"
    EXECUTE_NATIVE = "execute_native"
    EXECUTE_COMPETITOR = "execute_competitor"
    EMIT_RESULT = "emit_result"


@dataclass(frozen=True)
class VerificationWorkloadSourceIR:
    """One exact model/property pair selected from a benchmark CSV."""

    workload_id: str
    category: str
    csv_ordinal: int
    csv_relative_path: str
    model_relative_path: str
    property_relative_path: str
    csv_sha256: str
    model_sha256: str
    property_sha256: str
    query_ir_hash: str
    model_input_shape: Tuple[int, ...]
    model_output_dim: int
    onnx_ops: Tuple[str, ...]

    def validate(self) -> None:
        if (
            not self.workload_id
            or not self.category
            or self.csv_ordinal < 0
            or any(
                not _relative_source_path(value)
                for value in (
                    self.csv_relative_path,
                    self.model_relative_path,
                    self.property_relative_path,
                )
            )
            or any(
                not _is_sha256(value)
                for value in (
                    self.csv_sha256,
                    self.model_sha256,
                    self.property_sha256,
                    self.query_ir_hash,
                )
            )
            or not self.model_input_shape
            or any(dimension < 1 for dimension in self.model_input_shape)
            or self.model_output_dim < 1
            or not self.onnx_ops
            or tuple(sorted(set(self.onnx_ops))) != self.onnx_ops
        ):
            raise ValueError("verification workload source IR is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "workload_id": self.workload_id,
            "category": self.category,
            "csv_ordinal": self.csv_ordinal,
            "csv_relative_path": self.csv_relative_path,
            "model_relative_path": self.model_relative_path,
            "property_relative_path": self.property_relative_path,
            "csv_sha256": self.csv_sha256,
            "model_sha256": self.model_sha256,
            "property_sha256": self.property_sha256,
            "query_ir_hash": self.query_ir_hash,
            "model_input_shape": list(self.model_input_shape),
            "model_output_dim": self.model_output_dim,
            "onnx_ops": list(self.onnx_ops),
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class VerifierExecutionPolicyIR:
    """Auditable method and resource contract for one verifier backend."""

    backend: VerifierBackendKind
    implementation_id: str
    implementation_revision: str
    device: str
    torch_threads: int
    timeout_seconds: int
    alpha_steps: int
    beta_steps: int
    search_steps: int
    max_nodes: int
    attack_policy: str
    complete_verifier: str

    def validate(self) -> None:
        if (
            not self.implementation_id
            or not _is_revision(self.implementation_revision)
            or self.device not in {"cpu", "cuda"}
            or self.torch_threads < 1
            or self.timeout_seconds < 1
            or min(
                self.alpha_steps,
                self.beta_steps,
                self.search_steps,
                self.max_nodes,
            )
            < 0
            or not self.attack_policy
            or not self.complete_verifier
        ):
            raise ValueError("verifier execution policy IR is invalid")
        if self.backend == VerifierBackendKind.BOUNDFLOW_NATIVE:
            if (
                self.attack_policy != "native_projected_gradient"
                or self.complete_verifier != "bounded_relu_bab"
                or self.max_nodes < 1
            ):
                raise ValueError("BoundFlow native execution policy differs")
        elif self.backend == VerifierBackendKind.EXTERNAL_ABCROWN:
            if (
                self.attack_policy != "skip"
                or self.complete_verifier != "bab"
                or self.max_nodes != 0
            ):
                raise ValueError("external alpha-beta-CROWN execution policy differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "backend": self.backend.value,
            "implementation_id": self.implementation_id,
            "implementation_revision": self.implementation_revision,
            "device": self.device,
            "torch_threads": self.torch_threads,
            "timeout_seconds": self.timeout_seconds,
            "alpha_steps": self.alpha_steps,
            "beta_steps": self.beta_steps,
            "search_steps": self.search_steps,
            "max_nodes": self.max_nodes,
            "attack_policy": self.attack_policy,
            "complete_verifier": self.complete_verifier,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class MultiWorkloadPlanIR:
    """Frozen workload selection and dual-verifier execution contract."""

    plan_id: str
    benchmark_commit: str
    workloads: Tuple[VerificationWorkloadSourceIR, ...]
    policies: Tuple[VerifierExecutionPolicyIR, ...]
    timing_boundary: str = "fresh_process_start_to_structured_result"
    claim_boundary: str = "cpu_diagnostic_no_speedup"
    schema_version: str = MULTIWORKLOAD_PLAN_IR_SCHEMA_VERSION

    def validate(self) -> None:
        workload_ids = tuple(item.workload_id for item in self.workloads)
        backends = tuple(item.backend for item in self.policies)
        if (
            self.schema_version != MULTIWORKLOAD_PLAN_IR_SCHEMA_VERSION
            or not self.plan_id
            or not _is_revision(self.benchmark_commit)
            or len(self.workloads) < 3
            or len(workload_ids) != len(set(workload_ids))
            or set(backends)
            != {
                VerifierBackendKind.BOUNDFLOW_NATIVE,
                VerifierBackendKind.EXTERNAL_ABCROWN,
            }
            or len(backends) != len(set(backends))
            or self.timing_boundary != "fresh_process_start_to_structured_result"
            or self.claim_boundary != "cpu_diagnostic_no_speedup"
        ):
            raise ValueError("multi-workload Plan IR is invalid")
        for workload in self.workloads:
            workload.validate()
        for policy in self.policies:
            policy.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "benchmark_commit": self.benchmark_commit,
            "workloads": [item.to_dict() for item in self.workloads],
            "policies": [item.to_dict() for item in self.policies],
            "timing_boundary": self.timing_boundary,
            "process_isolation": "one_fresh_process_per_workload_backend",
            "claim_boundary": self.claim_boundary,
            "performance_claimed": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class MultiWorkloadTaskIRUnit:
    """One typed task in the dual-verifier evaluation graph."""

    task_id: str
    workload_id: str
    kind: MultiWorkloadTaskKind
    backend: str
    dependency_task_ids: Tuple[str, ...]
    output_value_ids: Tuple[str, ...]

    def validate(self) -> None:
        if (
            not self.task_id
            or not self.workload_id
            or self.backend
            not in {
                "shared",
                VerifierBackendKind.BOUNDFLOW_NATIVE.value,
                VerifierBackendKind.EXTERNAL_ABCROWN.value,
            }
            or len(self.dependency_task_ids) != len(set(self.dependency_task_ids))
            or not self.output_value_ids
            or len(self.output_value_ids) != len(set(self.output_value_ids))
        ):
            raise ValueError("multi-workload Task IR unit is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "task_id": self.task_id,
            "workload_id": self.workload_id,
            "kind": self.kind.value,
            "backend": self.backend,
            "dependency_task_ids": list(self.dependency_task_ids),
            "output_value_ids": list(self.output_value_ids),
        }


@dataclass(frozen=True)
class MultiWorkloadTaskIR:
    """Exact task graph compiled from a multi-workload Plan IR."""

    plan_hash: str
    tasks: Tuple[MultiWorkloadTaskIRUnit, ...]
    schema_version: str = MULTIWORKLOAD_TASK_IR_SCHEMA_VERSION

    def validate(self) -> None:
        task_ids = tuple(task.task_id for task in self.tasks)
        if (
            self.schema_version != MULTIWORKLOAD_TASK_IR_SCHEMA_VERSION
            or not _is_sha256(self.plan_hash)
            or not self.tasks
            or len(task_ids) != len(set(task_ids))
        ):
            raise ValueError("multi-workload Task IR is invalid")
        available: set[str] = set()
        for task in self.tasks:
            task.validate()
            if any(
                dependency not in available for dependency in task.dependency_task_ids
            ):
                raise ValueError("multi-workload Task IR dependency order differs")
            available.add(task.task_id)

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
class MultiWorkloadScheduleIR:
    """Deterministic fresh-process ordering for all workload/backend tasks."""

    plan_hash: str
    task_ir_hash: str
    ordered_task_ids: Tuple[str, ...]
    fresh_process_task_ids: Tuple[str, ...]
    schema_version: str = MULTIWORKLOAD_SCHEDULE_IR_SCHEMA_VERSION

    def validate_against(self, task_ir: MultiWorkloadTaskIR) -> None:
        task_ir.validate()
        task_ids = tuple(task.task_id for task in task_ir.tasks)
        task_by_id = {task.task_id: task for task in task_ir.tasks}
        if (
            self.schema_version != MULTIWORKLOAD_SCHEDULE_IR_SCHEMA_VERSION
            or self.plan_hash != task_ir.plan_hash
            or self.task_ir_hash != task_ir.stable_hash()
            or self.ordered_task_ids != task_ids
            or len(self.fresh_process_task_ids) != len(set(self.fresh_process_task_ids))
            or any(task_id not in task_by_id for task_id in self.fresh_process_task_ids)
            or any(
                task_by_id[task_id].kind
                not in {
                    MultiWorkloadTaskKind.EXECUTE_NATIVE,
                    MultiWorkloadTaskKind.EXECUTE_COMPETITOR,
                }
                for task_id in self.fresh_process_task_ids
            )
        ):
            raise ValueError("multi-workload Schedule IR differs")
        required = tuple(
            task.task_id
            for task in task_ir.tasks
            if task.kind
            in {
                MultiWorkloadTaskKind.EXECUTE_NATIVE,
                MultiWorkloadTaskKind.EXECUTE_COMPETITOR,
            }
        )
        if self.fresh_process_task_ids != required:
            raise ValueError("multi-workload fresh-process coverage differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "task_ir_hash": self.task_ir_hash,
            "ordered_task_ids": list(self.ordered_task_ids),
            "fresh_process_task_ids": list(self.fresh_process_task_ids),
            "dispatch": "sequential_stable_workload_then_backend",
            "timing_boundary": "fresh_process_start_to_structured_result",
        }

    def stable_hash(self, task_ir: MultiWorkloadTaskIR) -> str:
        self.validate_against(task_ir)
        return _canonical_hash(self.to_dict())


def compile_multiworkload_task_ir(plan: MultiWorkloadPlanIR) -> MultiWorkloadTaskIR:
    plan.validate()
    tasks: list[MultiWorkloadTaskIRUnit] = []
    for workload in plan.workloads:
        prefix = workload.workload_id
        acquire = f"{prefix}:acquire"
        parse = f"{prefix}:parse-query"
        import_model = f"{prefix}:import-onnx"
        compile_native = f"{prefix}:compile-native"
        execute_native = f"{prefix}:execute-native"
        execute_competitor = f"{prefix}:execute-abcrown"
        tasks.extend(
            (
                MultiWorkloadTaskIRUnit(
                    acquire,
                    prefix,
                    MultiWorkloadTaskKind.ACQUIRE_SOURCES,
                    "shared",
                    (),
                    (f"{prefix}:source-lock",),
                ),
                MultiWorkloadTaskIRUnit(
                    parse,
                    prefix,
                    MultiWorkloadTaskKind.PARSE_QUERY,
                    "shared",
                    (acquire,),
                    (f"{prefix}:query-ir",),
                ),
                MultiWorkloadTaskIRUnit(
                    import_model,
                    prefix,
                    MultiWorkloadTaskKind.IMPORT_ONNX,
                    "shared",
                    (acquire,),
                    (f"{prefix}:primal-ir",),
                ),
                MultiWorkloadTaskIRUnit(
                    compile_native,
                    prefix,
                    MultiWorkloadTaskKind.COMPILE_NATIVE,
                    VerifierBackendKind.BOUNDFLOW_NATIVE.value,
                    (parse, import_model),
                    (f"{prefix}:bound-plan-task-schedule",),
                ),
                MultiWorkloadTaskIRUnit(
                    execute_native,
                    prefix,
                    MultiWorkloadTaskKind.EXECUTE_NATIVE,
                    VerifierBackendKind.BOUNDFLOW_NATIVE.value,
                    (compile_native,),
                    (f"{prefix}:native-result",),
                ),
                MultiWorkloadTaskIRUnit(
                    execute_competitor,
                    prefix,
                    MultiWorkloadTaskKind.EXECUTE_COMPETITOR,
                    VerifierBackendKind.EXTERNAL_ABCROWN.value,
                    (acquire,),
                    (f"{prefix}:abcrown-result",),
                ),
                MultiWorkloadTaskIRUnit(
                    f"{prefix}:emit",
                    prefix,
                    MultiWorkloadTaskKind.EMIT_RESULT,
                    "shared",
                    (execute_native, execute_competitor),
                    (f"{prefix}:comparison-record",),
                ),
            )
        )
    task_ir = MultiWorkloadTaskIR(plan_hash=plan.stable_hash(), tasks=tuple(tasks))
    task_ir.validate()
    return task_ir


def compile_multiworkload_schedule_ir(
    plan: MultiWorkloadPlanIR, task_ir: MultiWorkloadTaskIR
) -> MultiWorkloadScheduleIR:
    plan.validate()
    task_ir.validate()
    if task_ir.plan_hash != plan.stable_hash():
        raise ValueError("multi-workload Plan/Task hash differs")
    fresh = tuple(
        task.task_id
        for task in task_ir.tasks
        if task.kind
        in {
            MultiWorkloadTaskKind.EXECUTE_NATIVE,
            MultiWorkloadTaskKind.EXECUTE_COMPETITOR,
        }
    )
    schedule = MultiWorkloadScheduleIR(
        plan_hash=plan.stable_hash(),
        task_ir_hash=task_ir.stable_hash(),
        ordered_task_ids=tuple(task.task_id for task in task_ir.tasks),
        fresh_process_task_ids=fresh,
    )
    schedule.validate_against(task_ir)
    return schedule


__all__ = [
    "MULTIWORKLOAD_PLAN_IR_SCHEMA_VERSION",
    "MULTIWORKLOAD_SCHEDULE_IR_SCHEMA_VERSION",
    "MULTIWORKLOAD_TASK_IR_SCHEMA_VERSION",
    "MultiWorkloadPlanIR",
    "MultiWorkloadScheduleIR",
    "MultiWorkloadTaskIR",
    "MultiWorkloadTaskIRUnit",
    "MultiWorkloadTaskKind",
    "VerificationWorkloadSourceIR",
    "VerifierBackendKind",
    "VerifierExecutionPolicyIR",
    "compile_multiworkload_schedule_ir",
    "compile_multiworkload_task_ir",
]
