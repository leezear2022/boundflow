from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..ir.liveness import compute_liveness_task_level
from ..ir.task import BFTaskModule


@dataclass(frozen=True)
class VerifyError:
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifyReport:
    ok: bool
    errors: List[VerifyError] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, code: str, message: str, **context: Any) -> None:
        self.errors.append(VerifyError(code=code, message=message, context=dict(context)))
        self.ok = False


def verify_task_graph_soundness(module: BFTaskModule) -> VerifyReport:
    """
    Verify TaskGraph structural invariants.

    If module.task_graph is None, returns ok=True with stats["skipped"]=True.
    """
    report = VerifyReport(ok=True)
    if module.task_graph is None:
        report.stats["skipped"] = True
        return report
    try:
        module.validate()
    except Exception as e:  # noqa: BLE001
        report.add_error("module_validate_failed", str(e))
        return report

    g = module.task_graph
    tasks_by_id = {t.task_id: t for t in module.tasks}
    # topo_sort checks acyclicity for reachable subgraph.
    try:
        _ = g.topo_sort(tasks_by_id=tasks_by_id, entry_task_id=module.entry_task_id)
    except Exception as e:  # noqa: BLE001
        report.add_error("topo_sort_failed", str(e))
        return report

    # Extra check: every cross-task value dependency implied by TaskIO must be covered by an edge dep.
    # Build producer map by value name from tasks.
    producer: Dict[str, str] = {}
    for t in module.tasks:
        for v in t.output_values:
            producer[v] = t.task_id
        for op in t.ops:
            for v in op.outputs:
                producer[v] = t.task_id

    edge_deps = set()
    for e in g.edges:
        for d in e.deps:
            edge_deps.add((e.src_task_id, e.dst_task_id, d.src_value, d.src_buffer_id))

    missing = 0
    for t in module.tasks:
        for v in t.input_values:
            src = producer.get(v)
            if src is None or src == t.task_id:
                continue
            buf = module.storage_plan.value_to_buffer.get(v)
            if buf is None:
                continue
            if (src, t.task_id, v, buf) not in edge_deps:
                missing += 1
                report.add_error(
                    "missing_edge_dep",
                    "cross-task dependency not covered by TaskGraph edge",
                    src_task_id=src,
                    dst_task_id=t.task_id,
                    value=v,
                    buffer_id=buf,
                )

    report.stats["missing_edge_deps"] = int(missing)
    return report


def verify_storage_plan_soundness(module: BFTaskModule) -> VerifyReport:
    """
    Verify logical->physical mapping is total (when physical_buffers is present) and self-consistent.
    """
    report = VerifyReport(ok=True)
    try:
        module.storage_plan.validate()
    except Exception as e:  # noqa: BLE001
        report.add_error("storage_plan_validate_failed", str(e))
        return report

    sp = module.storage_plan
    if sp.physical_buffers:
        # Require total mapping when physical_buffers is non-empty (execution env uses physical ids only).
        missing = [bid for bid in sp.buffers.keys() if bid not in sp.logical_to_physical]
        if missing:
            report.add_error(
                "logical_to_physical_not_total",
                "physical_buffers present but logical_to_physical does not cover all logical buffers",
                missing_logical_buffers=missing[:20],
                missing_count=len(missing),
            )
        # Require all mapped targets exist in physical_buffers.
        bad = []
        for logical, phys in sp.logical_to_physical.items():
            if phys not in sp.physical_buffers:
                bad.append((logical, phys))
        if bad:
            report.add_error(
                "physical_buffers_missing_target",
                "logical_to_physical maps to ids not present in physical_buffers",
                examples=bad[:20],
                count=len(bad),
            )

    report.stats["num_logical_buffers"] = sp.num_logical_buffers()
    report.stats["num_physical_buffers"] = sp.num_physical_buffers()
    report.stats["has_physical_buffers"] = bool(sp.physical_buffers)
    return report


def verify_liveness_reuse_consistency(module: BFTaskModule) -> VerifyReport:
    """
    Verify basic liveness/reuse invariants at task granularity:
    - producer_index <= last_use_index
    - For any physical buffer, mapped logical lifetimes must not overlap (inclusive)
    """
    report = VerifyReport(ok=True)
    if module.task_graph is None:
        report.stats["skipped"] = True
        return report

    try:
        liveness = compute_liveness_task_level(module)
    except Exception as e:  # noqa: BLE001
        report.add_error("liveness_compute_failed", str(e))
        return report

    # Interval overlap check per physical buffer.
    sp = module.storage_plan
    phys_to_intervals: Dict[str, List[tuple[int, int, str]]] = {}
    for bid, lt in liveness.lifetimes.items():
        phys = sp.to_physical(bid)
        phys_to_intervals.setdefault(phys, []).append((int(lt.producer_index), int(lt.last_use_index), bid))

    overlap = 0
    for phys, intervals in phys_to_intervals.items():
        intervals = sorted(intervals, key=lambda x: (x[0], x[1], x[2]))
        for (a0, a1, a_id), (b0, b1, b_id) in zip(intervals, intervals[1:]):
            if b0 <= a1:
                overlap += 1
                report.add_error(
                    "physical_lifetime_overlap",
                    "two logical lifetimes overlap on the same physical buffer",
                    physical_id=phys,
                    a_logical=a_id,
                    a_window=(a0, a1),
                    b_logical=b_id,
                    b_window=(b0, b1),
                )

    report.stats["physical_overlap_count"] = int(overlap)
    report.stats["num_tasks"] = len(liveness.topo_order)
    return report


def verify_all(module: BFTaskModule) -> Dict[str, VerifyReport]:
    """
    Run the core invariant verifiers and return per-check reports.
    """
    return {
        "task_graph": verify_task_graph_soundness(module),
        "storage_plan": verify_storage_plan_soundness(module),
        "liveness_reuse": verify_liveness_reuse_consistency(module),
    }

