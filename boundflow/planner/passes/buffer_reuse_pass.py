from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from ...ir.liveness import (
    BufferReuseKey,
    LivenessInfo,
    compute_liveness_task_level,
)
from ...ir.task import BFTaskModule, StoragePlan
from ..core import PlanBundle, PlannerConfig, PlannerPass
from ..storage_reuse import (
    BufferReuseStats,
    ReuseKeyMode,
    ReuseMissReason,
    ReusePolicy,
    StorageReuseOptions,
    estimate_bytes_saved,
    make_reuse_key_fn,
)


@dataclass(frozen=True)
class BufferReuseConfig:
    """
    Phase 5B.1 conservative reuse config.

    v0: reuse only buffers produced inside tasks (exclude ENTRY buffers), and only in "global" scope.
    """

    include_scopes: Tuple[str, ...] = ("global",)
    reuse_entry_buffers: bool = False


ReuseKeyFn = Callable[[object], BufferReuseKey]
ReusePolicyFn = Callable[[List[str], str, BufferReuseKey], Optional[str]]


def lifo_reuse_policy(pool: List[str], logical_buffer_id: str, key: BufferReuseKey) -> Optional[str]:
    _ = (logical_buffer_id, key)
    if not pool:
        return None
    return pool.pop()


def apply_conservative_buffer_reuse(
    module: BFTaskModule,
    *,
    liveness: Optional[LivenessInfo] = None,
    config: BufferReuseConfig = BufferReuseConfig(),
    options: Optional[StorageReuseOptions] = None,
    reuse_key_fn: Optional[Callable] = None,
    reuse_policy_fn: ReusePolicyFn = lifo_reuse_policy,
) -> BufferReuseStats:
    """
    Mutate `module.storage_plan` in-place by introducing logical->physical buffer aliasing.

    This is a conservative, task-level reuse model:
    - A buffer can be reused only after its last_use task completes.
    - Buffers produced within the same task are treated as overlapping (not reusable within-task).
    """
    module.validate()
    if module.task_graph is None:
        raise ValueError("apply_conservative_buffer_reuse requires module.task_graph")

    opt = options or StorageReuseOptions(enabled=True)
    if not opt.enabled:
        return BufferReuseStats()

    if reuse_key_fn is None:
        # Safe loosening: IGNORE_LAYOUT keeps strides.
        reuse_key_fn = make_reuse_key_fn(mode=opt.key_mode)

    # Policy selection hook (v0: only LIFO; FIFO reserved).
    if opt.policy == ReusePolicy.LIFO:
        reuse_policy_fn = lifo_reuse_policy
    elif opt.policy == ReusePolicy.FIFO:
        def fifo_policy(pool: List[str], logical_buffer_id: str, key: BufferReuseKey) -> Optional[str]:
            _ = (logical_buffer_id, key)
            if not pool:
                return None
            return pool.pop(0)

        reuse_policy_fn = fifo_policy
    else:
        raise ValueError(f"unsupported ReusePolicy: {opt.policy}")

    stats = BufferReuseStats()

    if liveness is None:
        liveness = compute_liveness_task_level(module, reuse_key_fn=reuse_key_fn)

    include_scopes = set(opt.include_scopes or config.include_scopes)
    topo = liveness.topo_order
    tasks_by_id = {t.task_id: t for t in module.tasks}

    # Buffer sets by task index.
    produced_by_task: Dict[int, List[str]] = {}
    released_by_task: Dict[int, List[str]] = {}
    for bid, lt in liveness.lifetimes.items():
        if lt.producer_index >= 0:
            produced_by_task.setdefault(lt.producer_index, []).append(bid)
        released_by_task.setdefault(lt.last_use_index, []).append(bid)

    for k in list(produced_by_task.keys()):
        produced_by_task[k] = sorted(set(produced_by_task[k]))
    for k in list(released_by_task.keys()):
        released_by_task[k] = sorted(set(released_by_task[k]))

    # Physical pool keyed by (dtype/shape/device/layout/alignment/...).
    free_pool: Dict[BufferReuseKey, List[str]] = {}

    logical_to_physical: Dict[str, str] = {}

    def _is_reusable(bid: str) -> bool:
        lt = liveness.lifetimes.get(bid)
        if lt is None:
            return False
        if lt.scope not in include_scopes:
            stats.inc(ReuseMissReason.NOT_REUSABLE_SCOPE)
            return False
        if not opt.reuse_entry_buffers and lt.producer_task_id == "ENTRY":
            stats.inc(ReuseMissReason.ENTRY_BUFFER)
            return False
        # Conservative: only reuse buffers produced inside tasks.
        if lt.producer_task_id == "ENTRY":
            return False
        return True

    def _key(bid: str) -> BufferReuseKey:
        spec = module.storage_plan.buffers[bid]
        return reuse_key_fn(spec)

    def _alloc_identity(bid: str) -> str:
        logical_to_physical[bid] = bid
        return bid

    # Simulate execution in topo order: allocate buffers when first produced, release at last_use.
    for idx, tid in enumerate(topo):
        _ = tasks_by_id[tid]  # ensure task exists

        # Allocate produced buffers for this task.
        for bid in produced_by_task.get(idx, []):
            if bid in logical_to_physical:
                continue
            if bid not in module.storage_plan.buffers:
                stats.inc(ReuseMissReason.NOT_IN_STORAGE_PLAN)
                continue

            if not _is_reusable(bid):
                _alloc_identity(bid)
                continue

            k = _key(bid)
            avail = free_pool.get(k)
            if avail:
                phys = reuse_policy_fn(avail, bid, k)
                if phys is None:
                    stats.pool_miss += 1
                    stats.inc(ReuseMissReason.POLICY_DECLINED)
                    _alloc_identity(bid)
                else:
                    stats.pool_hit += 1
                    logical_to_physical[bid] = phys
            else:
                stats.pool_miss += 1
                # If some other pool has buffers, this miss is more likely a key mismatch than "nothing freed yet".
                any_other_free = any(v for v in free_pool.values())
                stats.inc(ReuseMissReason.KEY_MISMATCH if any_other_free else ReuseMissReason.NO_FREE_BUFFER)
                _alloc_identity(bid)

        # Release buffers whose last use is this task.
        for bid in released_by_task.get(idx, []):
            if bid not in module.storage_plan.buffers:
                continue
            if not _is_reusable(bid):
                continue
            phys = logical_to_physical.get(bid, bid)
            k = _key(bid)
            free_pool.setdefault(k, []).append(phys)

    # Ensure every logical buffer has a mapping (identity by default).
    for bid, spec in module.storage_plan.buffers.items():
        if bid in logical_to_physical:
            continue
        # Never reuse non-global scopes into the pool; keep identity.
        logical_to_physical[bid] = bid

    # Build physical_buffers as the unique physical ids that appear in the mapping.
    physical_ids = sorted(set(logical_to_physical.values()))
    physical_buffers = {pid: module.storage_plan.buffers[pid] for pid in physical_ids if pid in module.storage_plan.buffers}

    new_sp = StoragePlan(
        buffers=dict(module.storage_plan.buffers),
        value_to_buffer=dict(module.storage_plan.value_to_buffer),
        physical_buffers=physical_buffers,
        logical_to_physical=logical_to_physical,
    )
    new_sp.validate()
    module.storage_plan = new_sp
    module.validate()
    saved, unknown = estimate_bytes_saved(new_sp, include_scopes=tuple(include_scopes))
    stats.bytes_saved_est = int(saved)
    stats.unknown_bytes_buffers = int(unknown)
    return stats


@dataclass(frozen=True)
class BufferReusePass(PlannerPass):
    """
    Phase 5B.1: apply conservative physical buffer reuse to StoragePlan.

    The updated StoragePlan is written back into bundle.task_module.storage_plan.
    """

    pass_id: str = "buffer_reuse_v0"
    reuse_config: BufferReuseConfig = BufferReuseConfig()

    def run(self, bundle: PlanBundle, *, config: PlannerConfig) -> PlanBundle:
        opt = config.storage_reuse
        if config.enable_storage_reuse and not opt.enabled:
            opt = StorageReuseOptions(
                enabled=True,
                include_scopes=opt.include_scopes,
                reuse_entry_buffers=opt.reuse_entry_buffers,
                key_mode=opt.key_mode,
                policy=opt.policy,
                meta=dict(opt.meta),
            )
        stats = apply_conservative_buffer_reuse(
            bundle.task_module,
            config=self.reuse_config,
            options=opt,
        )
        bundle.meta = dict(bundle.meta)
        bundle.meta["reuse_stats"] = stats
        bundle.storage_plan = bundle.task_module.storage_plan
        return bundle
