from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .task import BFTaskModule, BufferSpec


BufferReuseKey = Tuple[Any, ...]


@dataclass(frozen=True)
class BufferLifetime:
    logical_buffer_id: str
    producer_task_id: str  # "ENTRY" or task_id
    last_use_task_id: str
    producer_index: int  # -1 for "ENTRY"
    last_use_index: int
    key: BufferReuseKey
    size_bytes: Optional[int]
    scope: str


@dataclass
class LivenessInfo:
    topo_order: List[str]
    task_index: Dict[str, int]
    lifetimes: Dict[str, BufferLifetime] = field(default_factory=dict)  # logical_buffer_id -> lifetime

    def validate(self) -> None:
        for bid, lt in self.lifetimes.items():
            if bid != lt.logical_buffer_id:
                raise ValueError(f"liveness logical_buffer_id mismatch: key={bid} lt={lt.logical_buffer_id}")
            if lt.last_use_index < lt.producer_index:
                raise ValueError(
                    f"invalid lifetime for {bid}: producer_index={lt.producer_index} last_use_index={lt.last_use_index}"
                )
            if lt.producer_index >= 0 and lt.producer_task_id not in self.task_index:
                raise ValueError(f"producer_task_id not in task_index: {lt.producer_task_id}")
            if lt.last_use_task_id not in self.task_index:
                raise ValueError(f"last_use_task_id not in task_index: {lt.last_use_task_id}")


def dtype_nbytes(dtype: str) -> Optional[int]:
    d = str(dtype)
    table = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "float64": 8,
        "int8": 1,
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "int32": 4,
        "uint32": 4,
        "int64": 8,
        "uint64": 8,
        "bool": 1,
    }
    return table.get(d)


def shape_numel(shape: Sequence[Optional[int]]) -> Optional[int]:
    n = 1
    for d in shape:
        if d is None:
            return None
        if int(d) < 0:
            return None
        n *= int(d)
    return int(n)


def buffer_size_bytes(spec: BufferSpec) -> Optional[int]:
    nbytes = dtype_nbytes(spec.dtype)
    numel = shape_numel(spec.shape)
    if nbytes is None or numel is None:
        return None
    return int(nbytes * numel)


def default_reuse_key(spec: BufferSpec) -> BufferReuseKey:
    # Keep it conservative and stable for Phase 5B.1: require exact match.
    return (
        spec.scope,
        spec.dtype,
        tuple(spec.shape),
        spec.device,
        spec.layout,
        tuple(spec.strides) if spec.strides is not None else None,
        spec.alignment,
    )


def compute_liveness_task_level(
    module: BFTaskModule,
    *,
    reuse_key_fn: Callable[[BufferSpec], BufferReuseKey] = default_reuse_key,
) -> LivenessInfo:
    """
    Compute buffer lifetimes at task granularity (conservative).

    - Uses TaskGraph topo order for the reachable subgraph from entry.
    - Tracks producers/uses by scanning task ops (value->buffer via StoragePlan).
    - Treats each buffer as live for the whole producer/consumer task(s).
    """
    module.validate()
    if module.task_graph is None:
        raise ValueError("compute_liveness_task_level requires module.task_graph (multi-task module)")

    graph = module.task_graph
    tasks_by_id = {t.task_id: t for t in module.tasks}
    topo = graph.topo_sort(tasks_by_id=tasks_by_id, entry_task_id=module.entry_task_id)
    t_index = {tid: i for i, tid in enumerate(topo)}

    v2b = module.storage_plan.value_to_buffer

    producer_task: Dict[str, str] = {}  # logical_buffer_id -> task_id (or "ENTRY")
    producer_index: Dict[str, int] = {}
    last_use_index: Dict[str, int] = {}

    def _touch_use(bid: str, *, idx: int) -> None:
        cur = last_use_index.get(bid)
        if cur is None or idx > cur:
            last_use_index[bid] = idx

    for tid in topo:
        task = tasks_by_id[tid]
        idx = t_index[tid]

        # Uses: task inputs (buffer-level), plus op inputs (value-level).
        for bid in task.input_buffers:
            _touch_use(bid, idx=idx)

        for op in task.ops:
            for v in op.inputs:
                bid = v2b.get(v)
                if bid is None:
                    continue
                _touch_use(bid, idx=idx)

            for v in op.outputs:
                bid = v2b.get(v)
                if bid is None:
                    continue
                if bid in producer_task and producer_task[bid] != tid:
                    raise ValueError(
                        f"buffer has multiple producers (not SSA?): {bid} producers={producer_task[bid]} and {tid}"
                    )
                producer_task[bid] = tid
                producer_index[bid] = idx
                # If an output is never used, keep it live at least in its producer task.
                _touch_use(bid, idx=idx)

    # Cross-task uses must be derived from TaskGraph edges (stable contract for multi-task).
    for e in graph.edges:
        if e.src_task_id not in t_index or e.dst_task_id not in t_index:
            continue
        dst_idx = t_index[e.dst_task_id]
        for dep in e.deps:
            _touch_use(dep.src_buffer_id, idx=dst_idx)

    # Any used buffer without a producer is an entry/param buffer; treat as producer=ENTRY at index -1.
    for bid in list(last_use_index.keys()):
        if bid not in producer_task:
            producer_task[bid] = "ENTRY"
            producer_index[bid] = -1

    lifetimes: Dict[str, BufferLifetime] = {}
    for bid, p_tid in producer_task.items():
        if bid not in module.storage_plan.buffers:
            # Planner may choose not to allocate buffers for certain values; skip.
            continue
        spec = module.storage_plan.buffers[bid]
        p_idx = int(producer_index.get(bid, -1))
        u_idx = int(last_use_index.get(bid, p_idx))
        if u_idx < 0:
            # Should not happen for reachable tasks.
            continue
        last_tid = topo[u_idx]
        lifetimes[bid] = BufferLifetime(
            logical_buffer_id=bid,
            producer_task_id=p_tid,
            last_use_task_id=last_tid,
            producer_index=p_idx,
            last_use_index=u_idx,
            key=reuse_key_fn(spec),
            size_bytes=buffer_size_bytes(spec),
            scope=spec.scope,
        )

    info = LivenessInfo(topo_order=topo, task_index=t_index, lifetimes=lifetimes)
    info.validate()
    return info


@dataclass(frozen=True)
class PeakMemoryStats:
    peak_physical_buffers_count: int
    peak_physical_bytes: int
    unknown_bytes_buffers: int


def compute_peak_physical_memory_task_level(
    module: BFTaskModule,
    liveness: LivenessInfo,
    *,
    include_scopes: Iterable[str] = ("global",),
) -> PeakMemoryStats:
    """
    Conservative peak memory stats at task boundaries (ignores within-task peaks).
    """
    include_scopes = set(include_scopes)
    sp = module.storage_plan

    # Pre-group logical lifetimes by physical buffer id.
    phys_to_logical: Dict[str, List[BufferLifetime]] = {}
    for lt in liveness.lifetimes.values():
        if lt.scope not in include_scopes:
            continue
        phys = sp.to_physical(lt.logical_buffer_id)
        phys_to_logical.setdefault(phys, []).append(lt)

    topo_len = len(liveness.topo_order)
    peak_count = 0
    peak_bytes = 0
    peak_unknown = 0

    for idx in range(topo_len):
        active_phys: List[str] = []
        total = 0
        unknown = 0
        for phys, lts in phys_to_logical.items():
            # Active if any mapped logical lifetime overlaps this task index.
            active = False
            for lt in lts:
                if lt.producer_index <= idx <= lt.last_use_index:
                    active = True
                    break
            if not active:
                continue
            active_phys.append(phys)
            # Use physical spec size.
            try:
                spec = sp.get_physical_spec(phys)
                sz = buffer_size_bytes(spec)
            except KeyError:
                sz = None
            if sz is None:
                unknown += 1
            else:
                total += int(sz)
        if len(active_phys) > peak_count:
            peak_count = len(active_phys)
        if total > peak_bytes:
            peak_bytes = total
        if unknown > peak_unknown:
            peak_unknown = unknown

    return PeakMemoryStats(
        peak_physical_buffers_count=int(peak_count),
        peak_physical_bytes=int(peak_bytes),
        unknown_bytes_buffers=int(peak_unknown),
    )
