from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class PartitionPolicy(Enum):
    V0_SINGLE_TASK = "v0_single_task"
    V2_BASELINE = "v2_baseline"


class LifetimeModel(Enum):
    TASK_LEVEL = "task_level"
    OP_LEVEL = "op_level"  # reserved (Phase 5B.3+)


class LayoutPolicy(Enum):
    NONE = "none"
    PROPAGATE = "propagate"  # reserved


@dataclass(frozen=True)
class PartitionOptions:
    policy: PartitionPolicy = PartitionPolicy.V2_BASELINE
    min_tasks: int = 1
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LifetimeOptions:
    model: LifetimeModel = LifetimeModel.TASK_LEVEL
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LayoutOptions:
    policy: LayoutPolicy = LayoutPolicy.NONE
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannerDebugOptions:
    dump_config: bool = True
    validate_after_each_pass: bool = False  # reserved for PR#6 validators
    meta: Dict[str, Any] = field(default_factory=dict)

