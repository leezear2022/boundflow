from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import torch

from ..ir.task import BFTaskModule, BoundTask, TaskKind
from ..ir.task_graph import TaskGraph
from ..domains.interval import IntervalState
from .task_executor import LinfInputSpec, PythonTaskExecutor


class IBPTaskStepExecutor(Protocol):
    def run_ibp_task(
        self,
        task: BoundTask,
        *,
        env: Dict[str, IntervalState],
        params: Dict[str, Any],
        storage_plan,
    ) -> None: ...


@dataclass
class ScheduleStats:
    task_order: list[str]


def run_ibp_scheduled(
    module: BFTaskModule,
    input_spec: LinfInputSpec,
    *,
    executor: Optional[IBPTaskStepExecutor] = None,
    output_value: Optional[str] = None,
) -> IntervalState:
    """
    Execute BFTaskModule by scheduling tasks in topo order if module.task_graph is present.

    v0: supports INTERVAL_IBP only.
    """
    module.validate()
    if executor is None:
        executor = PythonTaskExecutor()

    entry = module.get_entry_task()
    if entry.kind != TaskKind.INTERVAL_IBP:
        raise NotImplementedError(f"scheduler only supports INTERVAL_IBP, got {entry.kind}")

    raw_params = module.bindings.get("params", {})
    params: Dict[str, Any] = dict(raw_params) if isinstance(raw_params, dict) else {}

    x0 = input_spec.center
    eps = float(input_spec.eps)
    input_logical = module.storage_plan.value_to_buffer.get(input_spec.value_name)
    if input_logical is None:
        raise KeyError(f"input_spec.value_name not found in storage_plan: {input_spec.value_name}")
    input_phys = module.storage_plan.to_physical(input_logical)
    env: Dict[str, IntervalState] = {input_phys: IntervalState(lower=x0 - eps, upper=x0 + eps)}

    if module.task_graph is None:
        # Fallback: behave like phase-4 single-task execution.
        if not hasattr(executor, "run_ibp"):
            raise TypeError("executor does not support run_ibp and module has no task_graph")
        return executor.run_ibp(module, input_spec, output_value=output_value)  # type: ignore[attr-defined]

    graph: TaskGraph = module.task_graph
    tasks_by_id = {t.task_id: t for t in module.tasks}
    order = graph.topo_sort(tasks_by_id=tasks_by_id, entry_task_id=module.entry_task_id)
    for task_id in order:
        task = tasks_by_id[task_id]
        if task.kind != TaskKind.INTERVAL_IBP:
            raise NotImplementedError(f"mixed TaskKind not supported in v0 scheduler: {task.kind}")
        executor.run_ibp_task(task, env=env, params=params, storage_plan=module.storage_plan)

    if output_value is None:
        # Try to infer a unique "sink output" in the reachable task subgraph.
        reachable = graph.reachable_from(module.entry_task_id)
        out_deg: Dict[str, int] = {t: 0 for t in reachable}
        for e in graph.edges:
            if e.src_task_id in reachable and e.dst_task_id in reachable:
                out_deg[e.src_task_id] += 1
        sinks = [t for t, d in out_deg.items() if d == 0]
        if len(sinks) != 1:
            raise ValueError(
                f"task_graph has {len(sinks)} sink tasks; specify output_value explicitly (sinks={sinks})"
            )
        sink_task = tasks_by_id[sinks[0]]
        if len(sink_task.output_values) != 1:
            raise ValueError(
                f"sink task '{sink_task.task_id}' has {len(sink_task.output_values)} outputs; "
                "specify output_value explicitly"
            )
        output_value = sink_task.output_values[0]

    out_logical = module.storage_plan.value_to_buffer.get(output_value)
    if out_logical is None:
        raise KeyError(f"output_value not found in storage_plan: {output_value}")
    out_phys = module.storage_plan.to_physical(out_logical)

    if out_phys not in env and output_value in params:
        t = params[output_value]  # type: ignore[index]
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=x0.device)
        return IntervalState(lower=t, upper=t)

    if out_phys not in env:
        raise KeyError(f"missing output buffer in env: {out_phys} (value={output_value}, logical={out_logical})")
    return env[out_phys]
