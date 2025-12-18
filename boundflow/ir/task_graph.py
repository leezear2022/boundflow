from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from .task import BoundTask


@dataclass(frozen=True)
class TaskDepEdge:
    """
    A dependency edge between tasks, expressed at value granularity.

    In Phase 5 we will likely switch to buffer-granularity edges (via StoragePlan),
    but value edges are sufficient to establish a correct topo schedule in v0.
    """

    src_task_id: str
    dst_task_id: str
    values: List[str]


@dataclass
class TaskGraph:
    """
    A DAG of BoundTasks.

    v0: explicit edges provided by planner; scheduler uses it for topo order.
    """

    task_ids: List[str]
    edges: List[TaskDepEdge] = field(default_factory=list)

    def validate(self, *, tasks_by_id: Dict[str, BoundTask], entry_task_id: str) -> None:
        if entry_task_id not in tasks_by_id:
            raise KeyError(f"entry_task_id not found in tasks: {entry_task_id}")
        if not self.task_ids:
            raise ValueError("TaskGraph.task_ids must be non-empty")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("TaskGraph.task_ids contains duplicates")
        for tid in self.task_ids:
            if tid not in tasks_by_id:
                raise KeyError(f"TaskGraph references missing task_id: {tid}")
        for edge in self.edges:
            if edge.src_task_id not in tasks_by_id:
                raise KeyError(f"edge.src_task_id missing: {edge.src_task_id}")
            if edge.dst_task_id not in tasks_by_id:
                raise KeyError(f"edge.dst_task_id missing: {edge.dst_task_id}")
            src = tasks_by_id[edge.src_task_id]
            dst = tasks_by_id[edge.dst_task_id]
            for v in edge.values:
                if v not in src.output_values:
                    raise ValueError(f"edge value '{v}' not in src task outputs: {src.task_id}")
                if v not in dst.input_values:
                    raise ValueError(f"edge value '{v}' not in dst task inputs: {dst.task_id}")

        _ = self.topo_sort(tasks_by_id=tasks_by_id, entry_task_id=entry_task_id)

    def reachable_from(self, entry_task_id: str) -> Set[str]:
        adj: Dict[str, List[str]] = {}
        for e in self.edges:
            adj.setdefault(e.src_task_id, []).append(e.dst_task_id)
        seen: Set[str] = set()
        stack = [entry_task_id]
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            for n in adj.get(t, []):
                if n not in seen:
                    stack.append(n)
        return seen

    def topo_sort(self, *, tasks_by_id: Dict[str, BoundTask], entry_task_id: str) -> List[str]:
        """
        Return a topo order of tasks reachable from entry_task_id.
        Raises ValueError if the reachable subgraph contains cycles.
        """
        reachable = self.reachable_from(entry_task_id)
        if not reachable:
            return [entry_task_id]

        indeg: Dict[str, int] = {t: 0 for t in reachable}
        succ: Dict[str, List[str]] = {t: [] for t in reachable}
        for e in self.edges:
            if e.src_task_id not in reachable or e.dst_task_id not in reachable:
                continue
            succ[e.src_task_id].append(e.dst_task_id)
            indeg[e.dst_task_id] += 1

        queue: List[str] = [t for t, d in indeg.items() if d == 0]
        out: List[str] = []
        while queue:
            t = queue.pop()
            out.append(t)
            for n in succ.get(t, []):
                indeg[n] -= 1
                if indeg[n] == 0:
                    queue.append(n)

        if len(out) != len(reachable):
            raise ValueError(f"TaskGraph contains a cycle in reachable subgraph: {sorted(reachable)}")
        for tid in out:
            if tid not in tasks_by_id:
                raise KeyError(f"topo_sort referenced missing task_id: {tid}")
        return out

    @staticmethod
    def chain(*task_ids: str, values: Optional[List[str]] = None) -> "TaskGraph":
        """
        Convenience helper for tests: t0 -> t1 -> ... -> tn with a single value edge list reused.
        """
        if len(task_ids) < 1:
            raise ValueError("TaskGraph.chain expects at least one task_id")
        edges: List[TaskDepEdge] = []
        if values is None:
            values = []
        for a, b in zip(task_ids[:-1], task_ids[1:]):
            edges.append(TaskDepEdge(src_task_id=a, dst_task_id=b, values=list(values)))
        return TaskGraph(task_ids=list(task_ids), edges=edges)

