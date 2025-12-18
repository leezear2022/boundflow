from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

from .core import PlanBundle
from .verify import VerifyReport, verify_all


class PlannerInstrument(Protocol):
    instrument_id: str

    def before_step(self, step_name: str, bundle: PlanBundle) -> None: ...

    def after_step(
        self, step_name: str, bundle: PlanBundle, *, verify: Optional[Dict[str, VerifyReport]] = None
    ) -> None: ...


@dataclass
class TimingInstrument:
    instrument_id: str = "timing_v0"
    _start_ns: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def before_step(self, step_name: str, bundle: PlanBundle) -> None:
        _ = bundle
        self._start_ns[step_name] = time.perf_counter_ns()

    def after_step(self, step_name: str, bundle: PlanBundle, *, verify=None) -> None:
        _ = verify
        t0 = int(self._start_ns.get(step_name, 0))
        t1 = time.perf_counter_ns()
        ms = (t1 - t0) / 1e6 if t0 else None
        bundle.meta = dict(bundle.meta)
        timings = dict(bundle.meta.get("timings_ms", {}))
        timings[step_name] = ms
        bundle.meta["timings_ms"] = timings


@dataclass
class VerifyInstrument:
    instrument_id: str = "verify_v0"
    fail_fast: bool = True

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
                    "errors": [{"code": e.code, "message": e.message, "context": e.context} for e in r.errors],
                    "stats": dict(r.stats),
                }
                for k, r in verify.items()
            },
        }
        bundle.meta["verify"] = v
        if self.fail_fast and not ok:
            raise ValueError(f"planner verify failed after step '{step_name}'")

