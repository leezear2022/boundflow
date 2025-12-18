from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from ..ir.primal import BFPrimalProgram
from ..ir.task import BFTaskModule, StoragePlan
from ..ir.task_graph import TaskGraph
from .storage_reuse import StorageReuseOptions
from .options import LayoutOptions, LifetimeOptions, PartitionOptions, PlannerDebugOptions


@dataclass(frozen=True)
class PlannerConfig:
    """
    Phase 5 baseline config container.

    v0: keep it minimal; add fields as planner passes become real.
    """

    enable_task_graph: bool = False
    enable_storage_reuse: bool = False
    enable_cache: bool = False
    tvm_target: Optional[str] = None
    storage_reuse: StorageReuseOptions = field(default_factory=StorageReuseOptions)
    partition: PartitionOptions = field(default_factory=PartitionOptions)
    lifetime: LifetimeOptions = field(default_factory=LifetimeOptions)
    layout: LayoutOptions = field(default_factory=LayoutOptions)
    debug: PlannerDebugOptions = field(default_factory=PlannerDebugOptions)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanBundle:
    """
    The planner output bundle (Phase 5).

    We keep BFTaskModule as the execution container, and attach additional plans
    (task_graph/cache/layout/lowering) incrementally.
    """

    program: BFPrimalProgram
    task_module: BFTaskModule
    task_graph: Optional[TaskGraph] = None
    storage_plan: Optional[StoragePlan] = None
    cache_plan: Optional[Any] = None
    layout_plan: Optional[Any] = None
    lowering_plan: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.program.graph.validate()
        self.task_module.validate()
        if self.task_graph is not None:
            tasks_by_id = {t.task_id: t for t in self.task_module.tasks}
            self.task_graph.validate(
                tasks_by_id=tasks_by_id,
                entry_task_id=self.task_module.entry_task_id,
                storage_plan=self.task_module.storage_plan,
            )
        if self.storage_plan is not None:
            self.storage_plan.validate()


class PlannerPass(Protocol):
    """
    A pluggable planner pass.
    """

    pass_id: str

    def run(self, bundle: PlanBundle, *, config: PlannerConfig) -> PlanBundle: ...


def run_planner_passes(
    bundle: PlanBundle, *, config: PlannerConfig, passes: List[PlannerPass]
) -> PlanBundle:
    for p in passes:
        bundle = p.run(bundle, config=config)
    return bundle
