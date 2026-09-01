"""Prepare process-local root optimizer kernels outside a warm query window."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, MutableMapping


@dataclass(frozen=True)
class PreparedRootOptimizerWarmupReceiptV1:
    """Account for one exact root-only warmup transaction."""

    warmup_wall_ns: int
    status: str
    lower_bound_tensor_count: int
    lower_bound_element_count: int
    performance_claimed: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a fail-closed JSON-safe projection."""

        if (
            self.warmup_wall_ns <= 0
            or not self.status
            or self.lower_bound_tensor_count <= 0
            or self.lower_bound_element_count <= 0
            or self.performance_claimed
        ):
            raise ValueError("prepared root optimizer warmup receipt differs")
        payload = asdict(self)
        payload.update(
            {
                "schema_version": "boundflow.prepared-root-optimizer-warmup/v1",
                "exact_model_property_warmup": True,
                "query_timing_excluded": True,
            }
        )
        return payload


def _restore_attribute(owner: Any, name: str, existed: bool, value: Any) -> None:
    if existed:
        setattr(owner, name, value)
    elif hasattr(owner, name):
        delattr(owner, name)


def prepare_root_optimizer_warmup_v1(
    *, solver: Any, constraints: Any, torch_module: Any
) -> PreparedRootOptimizerWarmupReceiptV1:
    """Run the exact incomplete verifier once while disabling complete BaB.

    The result is intentionally discarded.  The only retained state is process-local
    CUDA/library kernel setup and allocator caching.  Solver-visible mutable fields and
    the complete-verifier policy are restored before the measured query starts.
    """

    config = getattr(solver, "config", None)
    if not isinstance(config, MutableMapping):
        raise TypeError("prepared root warmup solver config differs")
    general = config.get("general")
    if not isinstance(general, MutableMapping) or "complete_verifier" not in general:
        raise TypeError("prepared root warmup general config differs")
    original_complete_verifier = general["complete_verifier"]
    state_names = (
        "constraint",
        "spec",
        "vnnlib_handler",
        "spec_handler_incomplete",
        "_last_result",
    )
    state = {
        name: (hasattr(solver, name), getattr(solver, name, None))
        for name in state_names
    }
    started_ns = time.perf_counter_ns()
    result: Any = None
    try:
        general["complete_verifier"] = "skip"
        result = solver.verify(constraints=constraints)
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
    finally:
        general["complete_verifier"] = original_complete_verifier
        for name in state_names:
            _restore_attribute(solver, name, state[name][0], state[name][1])
    warmup_wall_ns = time.perf_counter_ns() - started_ns
    reference = getattr(result, "reference", None)
    lower_bounds = (
        reference.get("lower_bounds") if isinstance(reference, dict) else None
    )
    if not isinstance(lower_bounds, dict):
        raise ValueError("prepared root warmup lower bounds differ")
    tensors = [
        value for value in lower_bounds.values() if torch_module.is_tensor(value)
    ]
    if len(tensors) != len(lower_bounds):
        raise TypeError("prepared root warmup lower tensor set differs")
    return PreparedRootOptimizerWarmupReceiptV1(
        warmup_wall_ns=warmup_wall_ns,
        status=str(getattr(result, "status", "")),
        lower_bound_tensor_count=len(tensors),
        lower_bound_element_count=sum(int(value.numel()) for value in tensors),
    )
