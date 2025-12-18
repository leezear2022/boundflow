from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from ..ir.liveness import BufferReuseKey, buffer_size_bytes
from ..ir.task import BufferSpec, StoragePlan


class ReuseKeyMode(Enum):
    STRICT = "strict"
    IGNORE_LAYOUT = "ignore_layout"


class ReusePolicy(Enum):
    LIFO = "lifo"
    FIFO = "fifo"


@dataclass(frozen=True)
class StorageReuseOptions:
    """
    Planner-facing storage reuse configuration (Phase 5B).

    v0: conservative defaults; looseness comes from key_mode/policy changes.
    """

    enabled: bool = False
    include_scopes: Tuple[str, ...] = ("global",)
    reuse_entry_buffers: bool = False

    key_mode: ReuseKeyMode = ReuseKeyMode.STRICT
    policy: ReusePolicy = ReusePolicy.LIFO

    # Reserved for Phase 5B.2+: when enabled, reuse/copy decisions should consider
    # memory effects (read/write conflicts) and alias groups.
    respect_memory_effect: bool = False

    meta: Dict[str, Any] = field(default_factory=dict)


class ReuseMissReason(Enum):
    NOT_REUSABLE_SCOPE = "not_reusable_scope"
    ENTRY_BUFFER = "entry_buffer"
    NOT_IN_STORAGE_PLAN = "not_in_storage_plan"
    NO_FREE_BUFFER = "no_free_buffer"
    POLICY_DECLINED = "policy_declined"
    KEY_MISMATCH = "key_mismatch"
    LIFETIME_OVERLAP = "lifetime_overlap"


@dataclass
class BufferReuseStats:
    pool_hit: int = 0
    pool_miss: int = 0
    bytes_saved_est: int = 0
    unknown_bytes_buffers: int = 0
    max_free_pool_keys: int = 0
    max_free_pool_buffers: int = 0
    overlap_blockers: Dict[str, int] = field(default_factory=dict)  # task_id -> count
    miss_reasons: Dict[ReuseMissReason, int] = field(default_factory=dict)

    def inc(self, reason: ReuseMissReason) -> None:
        self.miss_reasons[reason] = int(self.miss_reasons.get(reason, 0)) + 1

    def inc_overlap_blocker(self, task_id: str) -> None:
        self.overlap_blockers[task_id] = int(self.overlap_blockers.get(task_id, 0)) + 1


def make_reuse_key_fn(*, mode: ReuseKeyMode) -> Callable[[BufferSpec], BufferReuseKey]:
    """
    Produce a buffer reuse key function.

    Note: we do NOT ignore strides here; ignoring strides is only safe with an explicit "view"
    representation or boundary materialization, which is a Phase 5B.2+ topic.
    """

    if mode == ReuseKeyMode.STRICT:
        def _key(spec: BufferSpec) -> BufferReuseKey:
            return (
                spec.scope,
                spec.dtype,
                tuple(spec.shape),
                spec.device,
                spec.layout,
                tuple(spec.strides) if spec.strides is not None else None,
                spec.alignment,
            )

        return _key

    if mode == ReuseKeyMode.IGNORE_LAYOUT:
        def _key(spec: BufferSpec) -> BufferReuseKey:
            return (
                spec.scope,
                spec.dtype,
                tuple(spec.shape),
                spec.device,
                # layout ignored
                tuple(spec.strides) if spec.strides is not None else None,
                spec.alignment,
            )

        return _key

    raise ValueError(f"unsupported ReuseKeyMode: {mode}")


def estimate_bytes_saved(
    storage_plan: StoragePlan, *, include_scopes: Sequence[str] = ("global",)
) -> tuple[int, int]:
    """
    Returns (bytes_saved_est, unknown_bytes_buffers) using StoragePlan specs.
    """
    include = set(include_scopes)
    logical_bytes = 0
    physical_bytes = 0
    unknown = 0

    # Logical sum (by logical buffer ids).
    for bid, spec in storage_plan.buffers.items():
        if spec.scope not in include:
            continue
        sz = buffer_size_bytes(spec)
        if sz is None:
            unknown += 1
        else:
            logical_bytes += int(sz)

    # Physical sum (by physical buffer ids).
    phys_specs = storage_plan.physical_buffers or storage_plan.buffers
    for bid, spec in phys_specs.items():
        if spec.scope not in include:
            continue
        sz = buffer_size_bytes(spec)
        if sz is None:
            unknown += 1
        else:
            physical_bytes += int(sz)

    return max(0, int(logical_bytes - physical_bytes)), int(unknown)
