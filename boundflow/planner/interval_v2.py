from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from ..ir.task import BFTaskModule, BoundTask, TaskKind, TaskLowering, TaskOp
from ..ir.task_graph import TaskBufferDep, TaskDepEdge, TaskGraph
from .interval_v0 import plan_interval_ibp_v0


@dataclass(frozen=True)
class IntervalV2PartitionConfig:
    """
    v2 baseline partition config.
    """

    min_tasks: int = 2


def plan_interval_ibp_v2(program, *, config: IntervalV2PartitionConfig | None = None) -> BFTaskModule:
    """
    Planner v2 (Phase 5A PR#2 baseline): lower a primal graph to multiple INTERVAL_IBP tasks + TaskGraph.

    Notes:
    - v2 reuses v0 lowering to get a canonical TaskOp list and StoragePlan.
    - v2 only changes scheduling shape (single task -> multi task) without changing semantics.
    """
    cfg = config or IntervalV2PartitionConfig()

    base = plan_interval_ibp_v0(program)
    base.validate()
    if len(base.tasks) != 1:
        raise ValueError(f"plan_interval_ibp_v2 expects v0 to produce exactly 1 task, got {len(base.tasks)}")

    full = base.tasks[0]
    segments = _partition_ops(full.ops, min_tasks=cfg.min_tasks)
    tasks = _lower_segments_to_tasks(
        segments,
        module_inputs=list(full.input_values),
        module_outputs=list(full.output_values),
        params=list(base.bindings.get("params", {}).keys()) if isinstance(base.bindings.get("params"), dict) else [],
    )
    graph = _build_task_graph(tasks, storage_plan=base.storage_plan)

    module = BFTaskModule(
        tasks=tasks,
        entry_task_id=tasks[0].task_id,
        tvm_mod=None,
        bindings=dict(base.bindings),
        storage_plan=base.storage_plan,
        task_graph=graph,
    )
    module.validate()
    return module


def _partition_ops(ops: Sequence[TaskOp], *, min_tasks: int) -> List[List[TaskOp]]:
    segments: List[List[TaskOp]] = []
    current: List[TaskOp] = []

    def flush() -> None:
        nonlocal current
        if current:
            segments.append(current)
            current = []

    for op in ops:
        is_layout_only = op.op_type in ("permute", "transpose") or bool(op.attrs.get("layout_only"))
        if is_layout_only:
            flush()
            segments.append([op])
            continue
        current.append(op)
    flush()

    if len(segments) >= min_tasks:
        return segments

    # Fallback: split the whole list in half (keep it deterministic).
    if len(ops) < 2:
        return [list(ops)]
    mid = max(1, len(ops) // 2)
    return [list(ops[:mid]), list(ops[mid:])]


def _lower_segments_to_tasks(
    segments: List[List[TaskOp]],
    *,
    module_inputs: List[str],
    module_outputs: List[str],
    params: List[str],
) -> List[BoundTask]:
    # Precompute produced/consumed sets for each segment.
    produced: List[set[str]] = []
    consumed: List[set[str]] = []
    for seg in segments:
        p: set[str] = set()
        c: set[str] = set()
        for op in seg:
            p.update(op.outputs)
            c.update(op.inputs)
        produced.append(p)
        consumed.append(c)

    all_future_consumed: List[set[str]] = []
    future: set[str] = set()
    for i in reversed(range(len(segments))):
        all_future_consumed.append(set(future))
        future.update(consumed[i])
    all_future_consumed.reverse()

    tasks: List[BoundTask] = []
    for i, seg in enumerate(segments):
        seg_produced = produced[i]
        seg_consumed = consumed[i]

        external_inputs = set(seg_consumed) - set(seg_produced)
        # Remove values produced in earlier segments? No: they are external to this segment (task inputs).
        input_values = sorted(external_inputs)
        if not input_values:
            # Ensure non-empty; keep module input if available.
            if module_inputs:
                input_values = [module_inputs[0]]
            else:
                # fall back to any consumed value
                input_values = sorted(seg_consumed)[:1]

        # Boundary outputs: produced values used later or final outputs.
        needed = all_future_consumed[i] | set(module_outputs)
        output_values = sorted([v for v in seg_produced if v in needed])
        if not output_values:
            # Ensure non-empty: for the last segment, emit module_outputs; otherwise emit last produced.
            if i == len(segments) - 1 and module_outputs:
                output_values = list(module_outputs)
            elif seg:
                output_values = list(seg[-1].outputs[:1])
            else:
                raise ValueError("empty segment cannot form a task")

        task_params = sorted([v for v in params if v in seg_consumed])

        tasks.append(
            BoundTask(
                task_id=f"ibp_t{i}",
                kind=TaskKind.INTERVAL_IBP,
                ops=[TaskOp(op_type=o.op_type, name=o.name, inputs=list(o.inputs), outputs=list(o.outputs), attrs=dict(o.attrs)) for o in seg],
                input_values=input_values,
                output_values=output_values,
                params=task_params,
                batch_axes={},
                memory_plan={},
                lowering=TaskLowering.TVM_TIR,
            )
        )
    return tasks


def _build_task_graph(tasks: List[BoundTask], *, storage_plan) -> TaskGraph:
    # Producer map (SSA): value -> task_id
    producer: Dict[str, str] = {}
    for t in tasks:
        for v in t.output_values:
            producer[v] = t.task_id
        for op in t.ops:
            for v in op.outputs:
                producer[v] = t.task_id

    edge_map: Dict[Tuple[str, str], List[TaskBufferDep]] = {}
    for t in tasks:
        for v in t.input_values:
            src_task = producer.get(v)
            if src_task is None:
                continue
            if src_task == t.task_id:
                continue
            src_buf = storage_plan.value_to_buffer.get(v)
            dst_buf = storage_plan.value_to_buffer.get(v)
            if src_buf is None or dst_buf is None:
                continue
            edge_map.setdefault((src_task, t.task_id), []).append(
                TaskBufferDep(src_value=v, src_buffer_id=src_buf, dst_value=v, dst_buffer_id=dst_buf)
            )

    edges = [
        TaskDepEdge(src_task_id=src, dst_task_id=dst, deps=_dedup_deps(deps))
        for (src, dst), deps in edge_map.items()
    ]
    return TaskGraph(task_ids=[t.task_id for t in tasks], edges=edges)


def _dedup_deps(deps: List[TaskBufferDep]) -> List[TaskBufferDep]:
    seen = set()
    out: List[TaskBufferDep] = []
    for d in deps:
        key = (d.src_value, d.src_buffer_id, d.dst_value, d.dst_buffer_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out

