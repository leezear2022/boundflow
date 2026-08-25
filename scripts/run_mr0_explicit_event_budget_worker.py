#!/usr/bin/env python3
"""Run one fresh MR0 explicit CUDA-event budget worker."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=too-many-statements,missing-function-docstring,duplicate-code
# pylint: disable=too-many-arguments

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, cast, Callable, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.cibc_ibp_graph import CIBCIBPCUDAGraphPlanV1
from boundflow.runtime.mr0_explicit_event_budget import (
    MR0_BUDGETS,
    MR0_GROUP_COUNT,
    MR0_ORDERS,
    MR0_REPEATS,
    MR0_WARMUP,
    canonical_hash,
    derive_budget_row,
)
from scripts import run_cibc_ibp_horizontal_worker as horizontal_worker

SCHEMA = "boundflow.mr0-explicit-event-budget-worker/v1"
THREADS_PER_BLOCK = 128


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_snapshot() -> dict[str, object]:
    query = (
        "name,temperature.gpu,power.draw,clocks.current.graphics,"
        "clocks.current.memory,enforced.power.limit,driver_version"
    )
    completed = subprocess.run(
        ("nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"),
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("MR0 GPU inventory differs")
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != 7:
        raise RuntimeError("MR0 GPU snapshot differs")
    return {
        "name": values[0],
        "temperature_celsius": int(values[1]),
        "power_draw_watts": float(values[2]),
        "graphics_clock_mhz": int(values[3]),
        "memory_clock_mhz": int(values[4]),
        "enforced_power_limit_watts": float(values[5]),
        "driver_version": values[6],
    }


def _timed_group(
    call: Callable[[], object],
    *,
    repeats: int,
    budget: int,
    outer_start: torch.cuda.Event,
    outer_end: torch.cuda.Event,
    starts: list[torch.cuda.Event],
    ends: list[torch.cuda.Event],
) -> float:
    if budget not in (0, *MR0_BUDGETS) or len(starts) != 17 or len(ends) != 17:
        raise ValueError("MR0 event budget differs")
    outer_start.record()
    for _ in range(repeats):
        for event in starts[:budget]:
            event.record()
        call()
        for event in ends[:budget]:
            event.record()
    outer_end.record()
    outer_end.synchronize()
    milliseconds = float(outer_start.elapsed_time(outer_end)) / repeats
    if not math.isfinite(milliseconds) or milliseconds <= 0.0:
        raise RuntimeError("MR0 timing differs")
    return milliseconds


def _output_state(
    plan: CIBCIBPCUDAGraphPlanV1,
) -> tuple[dict[str, torch.Tensor], tuple[int, ...], tuple[tuple[object, ...], ...]]:
    snapshots = {}
    pointers = []
    contracts = []
    for name in sorted(plan.outputs):
        state = plan.outputs[name]
        for side in ("lower", "upper"):
            tensor = getattr(state, side)
            key = f"{name}:{side}"
            snapshots[key] = tensor.detach().clone()
            pointers.append(tensor.data_ptr())
            contracts.append(
                (
                    key,
                    tuple(int(value) for value in tensor.shape),
                    str(tensor.dtype),
                    str(tensor.device),
                )
            )
    return snapshots, tuple(pointers), tuple(contracts)


def _semantic_receipt(
    plan: CIBCIBPCUDAGraphPlanV1,
    reference: dict[str, torch.Tensor],
    pointers: tuple[int, ...],
    contracts: tuple[tuple[object, ...], ...],
) -> dict[str, object]:
    maximum = 0.0
    exact = True
    current_pointers = []
    current_contracts = []
    element_count = 0
    for name in sorted(plan.outputs):
        state = plan.outputs[name]
        for side in ("lower", "upper"):
            tensor = getattr(state, side)
            key = f"{name}:{side}"
            maximum = max(maximum, float((reference[key] - tensor).abs().max().item()))
            exact = exact and torch.equal(reference[key], tensor)
            current_pointers.append(tensor.data_ptr())
            current_contracts.append(
                (
                    key,
                    tuple(int(value) for value in tensor.shape),
                    str(tensor.dtype),
                    str(tensor.device),
                )
            )
            element_count += tensor.numel()
    receipt = {
        "exact": exact,
        "maximum_absolute_difference": maximum,
        "pointer_stable": tuple(current_pointers) == pointers,
        "contract_stable": tuple(current_contracts) == contracts,
        "tensor_count": len(reference),
        "element_count": element_count,
        "candidate_conv_launch_count": plan.launch_count,
        "fallback_count": 0,
        "eager_shadow_count": 0,
    }
    if (
        receipt["exact"] is not True
        or receipt["pointer_stable"] is not True
        or receipt["contract_stable"] is not True
        or maximum != 0.0
        or plan.launch_count != 6
    ):
        raise RuntimeError("MR0 semantic receipt differs")
    return receipt


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("MR0 worker requires CUDA")
    expected_order = MR0_ORDERS[args.run_ordinal]
    if args.order != expected_order:
        raise ValueError("MR0 worker order differs")
    source_args = argparse.Namespace(
        source_capture=args.source_capture,
        model=args.model,
        run_ordinal=args.run_ordinal,
    )
    module, spec, input_lower, input_upper = horizontal_worker._prepare(source_args)
    inventory = Counter(op.op_type for op in module.get_entry_task().ops)
    if inventory != Counter(
        {"conv2d": 6, "linear": 2, "relu": 6, "add": 2, "flatten": 1}
    ):
        raise ValueError("MR0 production topology differs")
    plan = CIBCIBPCUDAGraphPlanV1(
        module, spec.value_name, input_lower, input_upper, THREADS_PER_BLOCK
    )

    def call() -> object:
        return plan.replay(input_lower=input_lower, input_upper=input_upper)

    for _ in range(MR0_WARMUP):
        call()
    torch.cuda.synchronize()
    reference, pointers, contracts = _output_state(plan)
    outer_start = torch.cuda.Event(enable_timing=True)
    outer_end = torch.cuda.Event(enable_timing=True)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(17)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(17)]
    stream_before = int(torch.cuda.current_stream().cuda_stream)
    gpu_before = _gpu_snapshot()
    budget_rows = []
    for budget in MR0_BUDGETS:
        control, instrumented = [], []
        for _ in range(MR0_GROUP_COUNT):
            if args.order == "CI":
                control.append(
                    _timed_group(
                        call,
                        repeats=MR0_REPEATS,
                        budget=0,
                        outer_start=outer_start,
                        outer_end=outer_end,
                        starts=starts,
                        ends=ends,
                    )
                )
                instrumented.append(
                    _timed_group(
                        call,
                        repeats=MR0_REPEATS,
                        budget=budget,
                        outer_start=outer_start,
                        outer_end=outer_end,
                        starts=starts,
                        ends=ends,
                    )
                )
            else:
                instrumented.append(
                    _timed_group(
                        call,
                        repeats=MR0_REPEATS,
                        budget=budget,
                        outer_start=outer_start,
                        outer_end=outer_end,
                        starts=starts,
                        ends=ends,
                    )
                )
                control.append(
                    _timed_group(
                        call,
                        repeats=MR0_REPEATS,
                        budget=0,
                        outer_start=outer_start,
                        outer_end=outer_end,
                        starts=starts,
                        ends=ends,
                    )
                )
        budget_rows.append(
            derive_budget_row(
                budget=budget,
                control_ms=control,
                instrumented_ms=instrumented,
            )
        )
    call()
    torch.cuda.synchronize()
    stream_after = int(torch.cuda.current_stream().cuda_stream)
    semantic = _semantic_receipt(plan, reference, pointers, contracts)
    gpu_after = _gpu_snapshot()
    payload: dict[str, object] = {
        "schema_version": SCHEMA,
        "run_ordinal": args.run_ordinal,
        "order": args.order,
        "source_capture_sha256": _file_hash(args.source_capture),
        "model_sha256": _file_hash(args.model),
        "threads_per_block": THREADS_PER_BLOCK,
        "topology_inventory": dict(sorted(inventory.items())),
        "budget_rows": budget_rows,
        "warmup_count": MR0_WARMUP,
        "group_count": MR0_GROUP_COUNT,
        "repeats_per_side": MR0_REPEATS,
        "event_budgets": list(MR0_BUDGETS),
        "event_object_count": 36,
        "semantic_receipt": semantic,
        "semantic_admitted": True,
        "stream_before": stream_before,
        "stream_after": stream_after,
        "stream_admitted": stream_before == stream_after,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "environment": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "measurement_backend": "torch-cuda-event-no-profiler",
        },
        "compile_excluded": True,
        "cuda_graph": True,
        "input_copy_included": True,
        "performance_claimed": False,
    }
    if payload["stream_admitted"] is not True:
        raise RuntimeError("MR0 stream drift")
    payload["worker_hash"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-ordinal", type=int, choices=range(5), required=True)
    parser.add_argument("--order", choices=("CI", "IC"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    decision = cast(list[Mapping[str, Any]], payload["budget_rows"])[-1]
    print(
        f"MR0 worker={args.run_ordinal} order={args.order} "
        f"budget17_ratio={decision['overhead_ratio']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
