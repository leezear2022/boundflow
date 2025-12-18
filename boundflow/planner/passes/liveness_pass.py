from __future__ import annotations

from dataclasses import dataclass

from ...ir.liveness import LivenessInfo, compute_liveness_task_level
from ..core import PlanBundle, PlannerConfig, PlannerPass


@dataclass(frozen=True)
class LivenessPass(PlannerPass):
    """
    Phase 5B.1: compute conservative (task-level) buffer liveness.

    The result is stored in bundle.meta["liveness_info"].
    """

    pass_id: str = "liveness_v0"

    def run(self, bundle: PlanBundle, *, config: PlannerConfig) -> PlanBundle:
        info: LivenessInfo = compute_liveness_task_level(bundle.task_module)
        bundle.meta = dict(bundle.meta)
        bundle.meta["liveness_info"] = info
        return bundle

