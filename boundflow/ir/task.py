from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from enum import Enum

# Placeholder for TVM imports to avoid hard dependency at import time
# import tvm
# from tvm import relax, tir

if TYPE_CHECKING:
    from .task_graph import TaskGraph

class TaskKind(Enum):
    INTERVAL_IBP = "interval_ibp"


class TaskLowering(Enum):
    TVM_TIR = "tvm_tir"
    CONSTRAINT = "constraint"

class MemoryEffect(Enum):
    READ = "read"
    WRITE = "write"
    READWRITE = "readwrite"
    ALLOC = "alloc"
    FREE = "free"


@dataclass(frozen=True)
class BufferSpec:
    """
    Buffer/Storage 抽象：用于表达 value 与底层存储（buffer_id）的绑定与复用计划。

    v0.1 只提供 schema 与默认“一值一 buffer”的填充方式，后续 planner 才会做 aliasing/复用优化。
    """

    buffer_id: str
    dtype: str
    shape: List[Optional[int]]
    scope: str = "global"  # e.g. global/shared/local/param

    # Placeholders for backend/planner (Phase 4B/5):
    device: Optional[str] = None  # e.g. "cpu", "cuda"
    layout: Optional[str] = None  # e.g. "NCHW", "NHWC"
    strides: Optional[List[int]] = None
    alignment: Optional[int] = None
    alias_group: Optional[str] = None  # allow planner to express reuse/aliasing groups


@dataclass
class StoragePlan:
    buffers: Dict[str, BufferSpec] = field(default_factory=dict)  # buffer_id -> spec
    value_to_buffer: Dict[str, str] = field(default_factory=dict)  # value_name -> buffer_id

    # Phase 5B: logical vs physical storage.
    #
    # - `buffers` holds logical buffers (one per value by default).
    # - `logical_to_physical` optionally maps logical buffer_id -> physical buffer_id
    # - `physical_buffers` enumerates allocated physical buffers (by id).
    #
    # When `logical_to_physical` is empty, the plan is equivalent to "physical == logical".
    physical_buffers: Dict[str, BufferSpec] = field(default_factory=dict)  # physical_buffer_id -> spec
    logical_to_physical: Dict[str, str] = field(default_factory=dict)  # logical_buffer_id -> physical_buffer_id

    def to_physical(self, logical_buffer_id: str) -> str:
        return self.logical_to_physical.get(logical_buffer_id, logical_buffer_id)

    def get_logical_spec(self, logical_buffer_id: str) -> BufferSpec:
        if logical_buffer_id not in self.buffers:
            raise KeyError(f"logical buffer_id not found: {logical_buffer_id}")
        return self.buffers[logical_buffer_id]

    def get_physical_spec(self, logical_buffer_id: str) -> BufferSpec:
        phys = self.to_physical(logical_buffer_id)
        if self.physical_buffers:
            if phys not in self.physical_buffers:
                raise KeyError(f"physical buffer_id not found: {phys} (from logical {logical_buffer_id})")
            return self.physical_buffers[phys]
        return self.get_logical_spec(phys)

    def num_logical_buffers(self) -> int:
        return len(self.buffers)

    def num_physical_buffers(self) -> int:
        if self.physical_buffers:
            return len(self.physical_buffers)
        # Identity mapping.
        return len(self.buffers)

    def validate(self) -> None:
        for value_name, buffer_id in self.value_to_buffer.items():
            if buffer_id not in self.buffers:
                raise ValueError(f"storage_plan references missing buffer_id: {buffer_id} (value={value_name})")

        # Validate logical_to_physical only when present.
        for logical_id, physical_id in self.logical_to_physical.items():
            if logical_id not in self.buffers:
                raise ValueError(f"logical_to_physical references missing logical buffer_id: {logical_id}")
            if self.physical_buffers:
                if physical_id not in self.physical_buffers:
                    raise ValueError(
                        f"logical_to_physical references missing physical buffer_id: {physical_id} (logical={logical_id})"
                    )
            else:
                # If physical_buffers is empty, we treat it as identity physical storage.
                if physical_id != logical_id:
                    raise ValueError(
                        "logical_to_physical is non-identity but physical_buffers is empty; "
                        f"logical={logical_id} physical={physical_id}"
                    )

        for bid, spec in self.physical_buffers.items():
            if spec.buffer_id != bid:
                raise ValueError(f"physical_buffers spec.buffer_id mismatch: key={bid} spec={spec.buffer_id}")


@dataclass
class TaskOp:
    """
    TaskOp 是可执行任务的最小算子表示。
    v0.1 先复用 Primal 的 op_type/value-name 语义：inputs/outputs 是 value 名。
    """

    op_type: str
    name: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)
    # Placeholder for future alias/memory-effect modeling (Phase 5B+ / 5C+ / 5E+).
    memory_effect: Optional[MemoryEffect] = None


@dataclass
class BoundTask:
    task_id: str
    kind: TaskKind
    ops: List[TaskOp]

    input_values: List[str]
    output_values: List[str]

    # Phase 5 TaskIO contract: buffer-level IO (stable for reuse/lowering).
    # When BFTaskModule.task_graph is present, planner should populate these.
    input_buffers: List[str] = field(default_factory=list)
    output_buffers: List[str] = field(default_factory=list)
    
    # Batching axes info
    batch_axes: Dict[str, str] = field(default_factory=dict)  # e.g. {"spec": "axis_0", "neuron": "axis_1"}

    # Strategy
    memory_plan: Dict[str, Any] = field(default_factory=dict)  # Which states to cache
    
    lowering: TaskLowering = TaskLowering.TVM_TIR
    
    # Metadata for the backend (e.g. function name in the TVM mod)
    target_func_name: str = ""
    # Parameters/Constants referenced by this task (value names).
    params: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not self.ops:
            raise ValueError(f"task '{self.task_id}' has no ops")
        if not self.input_values:
            raise ValueError(f"task '{self.task_id}' has no inputs")
        if not self.output_values:
            raise ValueError(f"task '{self.task_id}' has no outputs")
        if len(set(self.input_buffers)) != len(self.input_buffers):
            raise ValueError(f"task '{self.task_id}' has duplicate input_buffers")
        if len(set(self.output_buffers)) != len(self.output_buffers):
            raise ValueError(f"task '{self.task_id}' has duplicate output_buffers")

@dataclass
class BFTaskModule:
    """
    Container for the executable task.
    In the TVM backend case, `tvm_mod` holds both Relax (driver) and TIR (kernels).
    """
    tasks: List[BoundTask]
    entry_task_id: str

    # Optional: TVM IRModule when lowered.
    tvm_mod: Optional[Any] = None # tvm.IRModule
    
    # Input/param binding info: how to feed args to the entry function
    bindings: Dict[str, Any] = field(default_factory=dict)

    # Memory/storage planning (optional in v0.1, but planner should try to fill it).
    storage_plan: StoragePlan = field(default_factory=StoragePlan)

    # Optional: task dependency graph for multi-task scheduling (Phase 5).
    task_graph: Optional["TaskGraph"] = None

    def get_task(self, task_id: str) -> BoundTask:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(f"task_id not found: {task_id}")

    def get_entry_task(self) -> BoundTask:
        return self.get_task(self.entry_task_id)

    def validate(self) -> None:
        if not self.tasks:
            raise ValueError("BFTaskModule.tasks must be non-empty")
        ids = [t.task_id for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate task_id in BFTaskModule")
        _ = self.get_entry_task()
        for task in self.tasks:
            task.validate()
        self.storage_plan.validate()

        # Phase 5 contract: when task_graph exists, tasks must have buffer IO populated and consistent.
        if self.task_graph is not None:
            for task in self.tasks:
                if not task.input_buffers:
                    raise ValueError(f"task '{task.task_id}' missing input_buffers (required when task_graph exists)")
                if not task.output_buffers:
                    raise ValueError(f"task '{task.task_id}' missing output_buffers (required when task_graph exists)")
                for v in task.input_values:
                    b = self.storage_plan.value_to_buffer.get(v)
                    if b is None:
                        raise ValueError(f"task '{task.task_id}' input value missing in storage_plan: {v}")
                    if b not in task.input_buffers:
                        raise ValueError(f"task '{task.task_id}' input_buffers missing buffer for value '{v}': {b}")
                for v in task.output_values:
                    b = self.storage_plan.value_to_buffer.get(v)
                    if b is None:
                        raise ValueError(f"task '{task.task_id}' output value missing in storage_plan: {v}")
                    if b not in task.output_buffers:
                        raise ValueError(f"task '{task.task_id}' output_buffers missing buffer for value '{v}': {b}")

        if self.task_graph is not None:
            tasks_by_id = {t.task_id: t for t in self.tasks}
            self.task_graph.validate(
                tasks_by_id=tasks_by_id, entry_task_id=self.entry_task_id, storage_plan=self.storage_plan
            )
