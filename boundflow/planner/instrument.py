from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from .core import PlanBundle
from .verify import VerifyReport, verify_all


class PlannerInstrument(Protocol):
    instrument_id: str

    def should_run(self, step_name: str, bundle: PlanBundle) -> bool: ...

    def before_step(self, step_name: str, bundle: PlanBundle) -> None: ...

    def after_step(
        self, step_name: str, bundle: PlanBundle, *, verify: Optional[Dict[str, VerifyReport]] = None
    ) -> None: ...


@dataclass
class TimingInstrument:
    instrument_id: str = "timing_v0"
    _start_ns: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def should_run(self, step_name: str, bundle: PlanBundle) -> bool:
        _ = (step_name, bundle)
        return True

    def before_step(self, step_name: str, bundle: PlanBundle) -> None:
        _ = bundle
        self._start_ns[step_name] = time.perf_counter_ns()

    def after_step(self, step_name: str, bundle: PlanBundle, *, verify=None) -> None:
        _ = verify
        t0 = self._start_ns.get(step_name)
        if t0 is None:
            return
        t0 = int(t0)
        t1 = time.perf_counter_ns()
        ms = (t1 - t0) / 1e6 if t0 else None
        if ms is None:
            return
        bundle.meta = dict(bundle.meta)
        timings = dict(bundle.meta.get("timings_ms", {}))
        if timings.get(step_name) is not None:
            return
        timings[step_name] = ms
        bundle.meta["timings_ms"] = timings


@dataclass
class VerifyInstrument:
    instrument_id: str = "verify_v0"
    fail_fast: bool = True

    def should_run(self, step_name: str, bundle: PlanBundle) -> bool:
        _ = (step_name, bundle)
        return True

    def before_step(self, step_name: str, bundle: PlanBundle) -> None:
        _ = (step_name, bundle)

    def after_step(self, step_name: str, bundle: PlanBundle, *, verify=None) -> None:
        if verify is None:
            verify = verify_all(bundle.task_module)
        ok = all(r.ok for r in verify.values())
        bundle.meta = dict(bundle.meta)
        v = dict(bundle.meta.get("verify", {}))
        v[step_name] = {
            "ok": bool(ok),
            "reports": {
                k: {
                    "ok": bool(r.ok),
                    "errors": [
                        {"code": e.code, "message": e.message, "where": dict(e.where), "context": dict(e.context)}
                        for e in r.errors
                    ],
                    "stats": dict(r.stats),
                }
                for k, r in verify.items()
            },
        }
        bundle.meta["verify"] = v
        if self.fail_fast and not ok:
            raise ValueError(f"planner verify failed after step '{step_name}'")


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


@dataclass
class DumpPlanInstrument:
    """
    Debug utility: dump a small JSON snapshot of the current PlanBundle after a step.

    The dump is designed to be safe for large models by only writing summaries + top-k samples.
    """

    instrument_id: str = "dump_plan_v0"
    dump_dir: str = ".benchmarks/plans"
    run_id: Optional[str] = None
    max_items: int = 50
    _seq: int = field(default=0, init=False, repr=False)

    def should_run(self, step_name: str, bundle: PlanBundle) -> bool:
        _ = (step_name, bundle)
        return True

    def before_step(self, step_name: str, bundle: PlanBundle) -> None:
        _ = (step_name, bundle)

    def after_step(self, step_name: str, bundle: PlanBundle, *, verify=None) -> None:
        _ = verify
        run_id = self.run_id or os.environ.get("BOUNDFLOW_PLAN_RUN_ID") or "run"
        out_dir = Path(self.dump_dir) / str(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        tm = bundle.task_module
        sp = tm.storage_plan
        tg = tm.task_graph

        logical_to_physical = sp.logical_to_physical or {}
        l2p_items = sorted(logical_to_physical.items())

        snapshot: Dict[str, Any] = {
            "step": str(step_name),
            "meta": _to_jsonable(bundle.meta),
            "task_graph": _to_jsonable(tg),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "kind": t.kind.value,
                    "num_ops": int(len(t.ops)),
                    "input_buffers": list(t.input_buffers),
                    "output_buffers": list(t.output_buffers),
                }
                for t in tm.tasks
            ],
            "storage_plan": {
                "num_logical_buffers": int(sp.num_logical_buffers()),
                "num_physical_buffers": int(sp.num_physical_buffers()),
                "logical_to_physical_count": int(len(logical_to_physical)),
                "logical_to_physical_topk": _to_jsonable(l2p_items[: int(self.max_items)]),
                "physical_buffers": sorted(list((sp.physical_buffers or {}).keys()))[: int(self.max_items)],
                "logical_buffers": sorted(list(sp.buffers.keys()))[: int(self.max_items)],
            },
        }

        fname = f"step_{self._seq:02d}_{step_name}.json"
        self._seq += 1
        (out_dir / fname).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
