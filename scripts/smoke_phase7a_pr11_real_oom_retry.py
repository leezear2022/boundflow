#!/usr/bin/env python
"""Trigger a process-local real CUDA OOM and verify placement retry."""

# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner import plan_interval_ibp_v0
from boundflow.planner.materialization import MaterializationAction
from boundflow.planner.materialization_placement import (
    BarrierPlacement,
    MaterializationPlacementPlan,
    PLACEMENT_SCHEMA_VERSION,
    PlacementPolicy,
)
from boundflow.runtime.crown_ibp import (
    _forward_ibp_trace_mlp,
    run_crown_ibp_mlp_with_placement_retry,
)
from boundflow.runtime.scheduler import PlacementRetryExhausted
from boundflow.runtime.task_executor import InputSpec
from scripts.profile_phase7a_pr10_materialization import _make_workload


def _uniform_plan(
    barrier_ids: tuple[str, ...], action: MaterializationAction, domain_batch: int
) -> MaterializationPlacementPlan:
    return MaterializationPlacementPlan(
        schema_version=PLACEMENT_SCHEMA_VERSION,
        policy=PlacementPolicy.GLOBAL_EXHAUSTIVE,
        placements=tuple(
            BarrierPlacement(
                barrier_id=barrier_id,
                action=action,
                persistent_bytes=0,
                ephemeral_bytes=0,
                latency_ms=0.0,
                reason="real_oom_smoke_candidate",
            )
            for barrier_id in barrier_ids
        ),
        predicted_peak_bytes=0,
        predicted_latency_ms=0.0,
        safe_memory_budget_bytes=1 << 62,
        requires_replan=False,
        recommended_domain_batch_size=int(domain_batch),
        reason="real_oom_smoke_candidate",
    )


# pylint: disable-next=too-many-locals
def main(argv: list[str] | None = None) -> int:
    """Run dense then structured under a process-local allocator cap."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-mib", type=int, required=True)
    parser.add_argument("--spec-size", type=int, default=128)
    parser.add_argument("--domain-batch", type=int, default=32)
    args = parser.parse_args(argv)
    if args.cap_mib <= 0 or args.spec_size <= 0 or args.domain_batch <= 0:
        parser.error("cap/spec/domain must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    cap_bytes = int(args.cap_mib) * 1024 * 1024
    fraction = float(cap_bytes) / float(properties.total_memory)
    if not 0.0 < fraction <= 1.0:
        parser.error("cap exceeds device memory")
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    torch.cuda.empty_cache()
    torch.manual_seed(0)

    workload = _make_workload("mini_resnet", device)
    dummy = torch.zeros((1, *workload.input_shape), device=device)
    program = import_torch(
        workload.model, (dummy,), export_mode="export", normalize=True
    )
    module = plan_interval_ibp_v0(program)
    torch.manual_seed(1)
    center = torch.randn((args.domain_batch, *workload.input_shape), device=device)
    spec = InputSpec.linf(
        value_name=module.get_entry_task().input_values[0], center=center, eps=0.03
    )
    linear_spec = torch.randn((args.domain_batch, args.spec_size, 10), device=device)
    _interval_env, relu_pre = _forward_ibp_trace_mlp(module, spec)
    barrier_ids = tuple(relu_pre)
    dense = _uniform_plan(barrier_ids, MaterializationAction.DENSE, args.domain_batch)
    structured = _uniform_plan(
        barrier_ids, MaterializationAction.STRUCTURED, args.domain_batch
    )

    try:
        bounds, stats = run_crown_ibp_mlp_with_placement_retry(
            module,
            spec,
            placement_plans=(dense, structured),
            linear_spec_C=linear_spec,
            memory_budget_bytes=cap_bytes,
            max_attempts=2,
        )
    except PlacementRetryExhausted as error:
        print(
            json.dumps(
                {
                    "schema_version": "boundflow.pr11-real-oom-retry/v2",
                    "status": "exhausted",
                    "cap_mib": args.cap_mib,
                    "spec_size": args.spec_size,
                    "domain_batch": args.domain_batch,
                    "stats": {
                        "attempts": error.stats.attempts,
                        "oom_failures": error.stats.oom_failures,
                        "selected_index": error.stats.selected_index,
                        "attempted_patterns": list(error.stats.attempted_patterns),
                    },
                },
                sort_keys=True,
            )
        )
        return 1

    finite = bool(
        torch.isfinite(bounds.lower).all() and torch.isfinite(bounds.upper).all()
    )
    ordered = bool((bounds.lower <= bounds.upper + 1e-6).all())
    payload = {
        "schema_version": "boundflow.pr11-real-oom-retry/v2",
        "retry_strategy": "latency_topology_density_stratified_v3",
        "status": "ok" if finite and ordered else "correctness_fail",
        "cap_mib": args.cap_mib,
        "cap_fraction": fraction,
        "device_total_memory": int(properties.total_memory),
        "spec_size": args.spec_size,
        "domain_batch": args.domain_batch,
        "barrier_count": len(barrier_ids),
        "correctness": {"finite": finite, "lower_le_upper": ordered},
        "stats": {
            "attempts": stats.attempts,
            "oom_failures": stats.oom_failures,
            "selected_index": stats.selected_index,
            "attempted_patterns": list(stats.attempted_patterns),
        },
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
