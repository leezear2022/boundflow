from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import torch

from ..ir.task import BFTaskModule, BoundTask, TaskKind
from ..ir.task_graph import TaskGraph
from ..domains.interval import IntervalState
from .task_executor import LinfInputSpec, PythonTaskExecutor


class IBPTaskStepExecutor(Protocol):
    def run_ibp_task(self, task: BoundTask, *, env: Dict[str, IntervalState], params: Dict[str, Any]) -> None: ...


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
    env: Dict[str, IntervalState] = {input_spec.value_name: IntervalState(lower=x0 - eps, upper=x0 + eps)}

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
        executor.run_ibp_task(task, env=env, params=params)

    if output_value is None:
        if len(entry.output_values) != 1:
            raise ValueError(f"entry task has {len(entry.output_values)} outputs; specify output_value explicitly")
        output_value = entry.output_values[0]

    if output_value not in env and output_value in params:
        t = params[output_value]
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=x0.device)
        return IntervalState(lower=t, upper=t)

    if output_value not in env:
        raise KeyError(f"missing output_value in env: {output_value}")
    return env[output_value]

