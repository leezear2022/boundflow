from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple, TypeVar

import torch

from ..ir.task import BFTaskModule, BoundTask, TaskKind
from ..ir.task_graph import TaskGraph
from ..planner.materialization_placement import (
    MaterializationPlacementPlan,
    PlacementRetryCandidate,
    rank_bounded_placement_retry_candidates,
)
from ..domains.interval import IntervalState
from .perturbation import LpBallPerturbation
from .task_executor import InputSpec, InputSpecLike, LinfInputSpec, PythonTaskExecutor


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


RetryResult = TypeVar("RetryResult")


@dataclass(frozen=True)
class PlacementRetryStats:
    """Observable host feedback for one placement candidate sequence."""

    attempts: int
    oom_failures: int
    selected_index: Optional[int]
    attempted_patterns: Tuple[str, ...]


class PlacementRetryExhausted(RuntimeError):
    """Every placement candidate failed with a real CUDA OOM."""

    def __init__(self, stats: PlacementRetryStats, messages: Tuple[str, ...]) -> None:
        self.stats = stats
        self.messages = messages
        super().__init__(
            "all materialization placement candidates failed with CUDA OOM: "
            f"attempts={stats.attempts} patterns={list(stats.attempted_patterns)}"
        )


def _placement_pattern(plan: MaterializationPlacementPlan) -> str:
    if plan.requires_replan:
        return "REPLAN"
    return "".join(
        "D" if placement.action.value == "dense" else "S"
        for placement in plan.placements
    )


def execute_placement_candidates_with_retry(
    plans: Sequence[MaterializationPlacementPlan],
    execute: Callable[[MaterializationPlacementPlan], RetryResult],
    *,
    clear_cuda_cache: bool = True,
) -> tuple[RetryResult, PlacementRetryStats]:
    """Try candidates in order, blacklisting only real CUDA OOM failures."""

    if not plans:
        raise ValueError("placement retry requires at least one candidate plan")
    if any(plan.requires_replan for plan in plans):
        raise ValueError("placement retry candidates must be executable plans")
    attempted: list[str] = []
    messages: list[str] = []
    for index, plan in enumerate(plans):
        attempted.append(_placement_pattern(plan))
        try:
            result = execute(plan)
        except torch.cuda.OutOfMemoryError as error:
            messages.append(str(error))
            if clear_cuda_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        stats = PlacementRetryStats(
            attempts=index + 1,
            oom_failures=index,
            selected_index=index,
            attempted_patterns=tuple(attempted),
        )
        return result, stats
    stats = PlacementRetryStats(
        attempts=len(plans),
        oom_failures=len(plans),
        selected_index=None,
        attempted_patterns=tuple(attempted),
    )
    raise PlacementRetryExhausted(stats, tuple(messages))


def execute_bounded_placement_candidates_with_retry(
    plans: Sequence[MaterializationPlacementPlan],
    execute: Callable[[MaterializationPlacementPlan], RetryResult],
    *,
    memory_budget_bytes: int,
    max_attempts: int = 6,
    prediction_budget_factor: float = 1.3,
    clear_cuda_cache: bool = True,
) -> tuple[RetryResult, PlacementRetryStats]:
    """Rank placement plans into a bounded ladder, then apply CUDA OOM retry."""

    if (
        not math.isfinite(float(prediction_budget_factor))
        or float(prediction_budget_factor) < 1.0
    ):
        raise ValueError("prediction_budget_factor must be finite and >= 1")

    ladder = rank_bounded_placement_retry_candidates(
        (
            PlacementRetryCandidate(
                candidate_id=str(index),
                predicted_peak_bytes=int(plan.predicted_peak_bytes),
                predicted_latency_ms=float(plan.predicted_latency_ms),
                conservative=all(
                    placement.action.value == "structured"
                    for placement in plan.placements
                ),
                structured_count=sum(
                    placement.action.value == "structured"
                    for placement in plan.placements
                ),
                barrier_count=len(plan.placements),
                action_transition_count=sum(
                    lhs.action != rhs.action
                    for lhs, rhs in zip(plan.placements, plan.placements[1:])
                ),
            )
            for index, plan in enumerate(plans)
        ),
        memory_budget_bytes=int(
            round(float(memory_budget_bytes) * float(prediction_budget_factor))
        ),
        max_attempts=int(max_attempts),
    )
    ordered = tuple(plans[int(candidate_id)] for candidate_id in ladder)
    return execute_placement_candidates_with_retry(
        ordered, execute, clear_cuda_cache=clear_cuda_cache
    )


def run_ibp_scheduled(
    module: BFTaskModule,
    input_spec: InputSpecLike,
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

    if isinstance(input_spec, InputSpec):
        # Phase 5 scheduler currently assumes box initialization at buffer-level env.
        # For non-L∞ perturbations, prefer single-task execution (PythonTaskExecutor.run_ibp) for now.
        ptb = input_spec.perturbation
        if isinstance(ptb, LpBallPerturbation) and ptb.perturbation_id.startswith("lp(p=inf"):
            input_spec = LinfInputSpec(value_name=input_spec.value_name, center=input_spec.center, eps=float(ptb.eps))
        else:
            raise NotImplementedError(
                "run_ibp_scheduled currently supports LinfInputSpec (box/L∞) only; "
                "use PythonTaskExecutor.run_ibp for non-L∞ perturbations"
            )

    raw_params = module.bindings.get("params", {})
    params: Dict[str, Any] = dict(raw_params) if isinstance(raw_params, dict) else {}

    x0 = input_spec.center
    eps = float(input_spec.eps)
    input_logical = module.storage_plan.value_to_buffer.get(input_spec.value_name)
    if input_logical is None:
        raise KeyError(f"input_spec.value_name not found in storage_plan: {input_spec.value_name}")
    input_phys = module.storage_plan.to_physical(input_logical)
    if module.storage_plan.physical_buffers and input_phys not in module.storage_plan.physical_buffers:
        raise KeyError(f"input physical buffer_id not found in storage_plan.physical_buffers: {input_phys}")
    env: Dict[str, IntervalState] = {input_phys: IntervalState(lower=x0 - eps, upper=x0 + eps)}

    if module.task_graph is None:
        # Fallback: behave like phase-4 single-task execution.
        if not hasattr(executor, "run_ibp"):
            raise TypeError("executor does not support run_ibp and module has no task_graph")
        return executor.run_ibp(module, input_spec, output_value=output_value)  # type: ignore[union-attr]

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
    if module.storage_plan.physical_buffers and out_phys not in module.storage_plan.physical_buffers:
        raise KeyError(f"output physical buffer_id not found in storage_plan.physical_buffers: {out_phys}")

    if out_phys not in env and output_value in params:
        t = params[output_value]  # type: ignore[index]
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=x0.device)
        return IntervalState(lower=t, upper=t)

    if out_phys not in env:
        raise KeyError(f"missing output buffer in env: {out_phys} (value={output_value}, logical={out_logical})")
    return env[out_phys]
