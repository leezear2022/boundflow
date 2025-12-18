from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Dict, List

from ..ir.primal import BFPrimalProgram
from ..ir.task import BFTaskModule
from .core import PlanBundle, PlannerConfig
from .interval_v0 import plan_interval_ibp_v0
from .interval_v2 import IntervalV2PartitionConfig, plan_interval_ibp_v2
from .options import PartitionPolicy
from .passes.buffer_reuse_pass import apply_conservative_buffer_reuse
from .storage_reuse import StorageReuseOptions


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return repr(obj)


def plan(program: BFPrimalProgram, *, config: PlannerConfig) -> PlanBundle:
    """
    Phase 5C.1 unified planner entry.

    v0: interval-IBP only; focuses on making the pipeline configurable and reproducible.
    """
    program.graph.validate()

    # Decide lowering shape (single-task vs multi-task).
    use_v2 = False
    if config.enable_task_graph:
        use_v2 = True
    if config.partition.policy == PartitionPolicy.V2_BASELINE:
        use_v2 = use_v2 or int(config.partition.min_tasks) > 1
    if config.partition.policy == PartitionPolicy.V0_SINGLE_TASK:
        use_v2 = False

    module: BFTaskModule
    planner_steps: List[str] = []
    if use_v2:
        planner_steps.append("interval_v2_partition")
        module = plan_interval_ibp_v2(program, config=IntervalV2PartitionConfig(min_tasks=int(config.partition.min_tasks)))
    else:
        planner_steps.append("interval_v0_single_task")
        module = plan_interval_ibp_v0(program)

    # Build bundle.
    meta: Dict[str, Any] = {}
    if config.debug.dump_config:
        meta["config_dump"] = _to_jsonable(config)
    meta["planner_steps"] = list(planner_steps)

    bundle = PlanBundle(
        program=program,
        task_module=module,
        task_graph=module.task_graph,
        storage_plan=module.storage_plan,
        meta=meta,
    )

    # Apply storage reuse as a planner step (keep interval_v2 as a pure partitioner in this pipeline).
    reuse_enabled = bool(config.enable_storage_reuse) or bool(config.storage_reuse.enabled)
    if reuse_enabled and module.task_graph is not None:
        opt: StorageReuseOptions = config.storage_reuse
        if not opt.enabled:
            opt = StorageReuseOptions(
                enabled=True,
                include_scopes=opt.include_scopes,
                reuse_entry_buffers=opt.reuse_entry_buffers,
                key_mode=opt.key_mode,
                policy=opt.policy,
                respect_memory_effect=opt.respect_memory_effect,
                meta=dict(opt.meta),
            )
        stats = apply_conservative_buffer_reuse(module, options=opt)
        bundle.meta = dict(bundle.meta)
        bundle.meta["reuse_stats"] = stats
        bundle.storage_plan = module.storage_plan

    bundle.validate()
    return bundle

