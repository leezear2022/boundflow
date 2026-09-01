#!/usr/bin/env python3
"""Run one fresh CIBC R1-A control or smoke-profile process."""

# pylint: disable=wrong-import-position,protected-access,duplicate-code
# pylint: disable=too-many-locals,too-many-statements,missing-function-docstring
# pylint: disable=too-many-arguments

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterator, Mapping, cast

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.cibc_ibp_graph import CIBCIBPCUDAGraphPlanV1
from boundflow.runtime.cibc_r1_attribution import (
    CuptiTimestampSource,
    R1OpType,
    R1TopologyLedger,
    canonical_hash,
    canonical_json,
    collect_cupti_triplets,
    derive_clock_calibration,
    topology_from_task,
)
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from scripts import run_cibc_ibp_horizontal_worker as horizontal_worker

WORKER_SCHEMA = "boundflow.cibc-r1-attribution-worker/v1"
CONTROL_WARMUP = 10
CONTROL_GROUP_COUNT = 20
CONTROL_REPEATS = 50
PROFILE_WARMUP = 10
PROFILE_GROUP_COUNT = 20
PROFILE_REPEATS = 5
THREADS_PER_BLOCK = 128
SEMANTIC_ATOL = 3.0e-4
EXPECTED_BUCKET_COUNTS = {
    R1OpType.CIBC_CONV: 6,
    R1OpType.LINEAR: 2,
    R1OpType.RELU: 6,
    R1OpType.RESIDUAL_ADD: 2,
    R1OpType.FLATTEN_VIEW: 1,
}


def _prepare(args: argparse.Namespace):
    source_args = argparse.Namespace(
        source_capture=args.source_capture,
        model=args.model,
        run_ordinal=args.pair_ordinal % 5,
    )
    return horizontal_worker._prepare(source_args)


def _topology(module, spec) -> R1TopologyLedger:
    interval_env, _relu_pre = _forward_ibp_trace_mlp(module, spec)
    params = cast(Mapping[str, torch.Tensor], module.bindings["params"])
    value_shapes: dict[str, tuple[int, ...]] = {}
    value_dtypes: dict[str, str] = {}
    value_devices: dict[str, str] = {}
    for name, state in interval_env.items():
        value_shapes[name] = tuple(int(item) for item in state.lower.shape)
        value_dtypes[name] = str(state.lower.dtype)
        value_devices[name] = str(state.lower.device)
    for name, tensor in params.items():
        value_shapes[name] = tuple(int(item) for item in tensor.shape)
        value_dtypes[name] = str(tensor.dtype)
        value_devices[name] = str(tensor.device)
    value_shapes[spec.value_name] = tuple(int(item) for item in spec.center.shape)
    value_dtypes[spec.value_name] = str(spec.center.dtype)
    value_devices[spec.value_name] = str(spec.center.device)
    task = module.get_entry_task()
    topology = topology_from_task(
        task,
        external_values=(*task.input_values, *task.params),
        value_shapes=value_shapes,
        value_dtypes=value_dtypes,
        value_devices=value_devices,
        single_stream=True,
    )
    observed = Counter(node.op_type for node in topology.nodes)
    if observed != Counter(EXPECTED_BUCKET_COUNTS):
        raise ValueError("R1-A production topology bucket inventory differs")
    return topology


class _NVTXMarkerFactory:
    def __init__(self, topology: R1TopologyLedger) -> None:
        self.topology = topology
        self.invocations = {node.ordinal: 0 for node in topology.nodes}

    @contextmanager
    def __call__(self, ordinal: int, op_type: str) -> Iterator[None]:
        if ordinal not in range(len(self.topology.nodes)):
            raise ValueError("R1-A marker ordinal differs")
        node = self.topology.nodes[ordinal]
        expected_source_type = {
            R1OpType.CIBC_CONV: "conv2d",
            R1OpType.LINEAR: "linear",
            R1OpType.RELU: "relu",
            R1OpType.RESIDUAL_ADD: "add",
            R1OpType.FLATTEN_VIEW: "flatten",
        }[node.op_type]
        if op_type != expected_source_type:
            raise ValueError("R1-A marker source op differs")
        marker = self.topology.marker_for(ordinal)
        self.invocations[ordinal] += 1
        torch.cuda.nvtx.range_push(marker)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def receipt(self) -> dict[str, object]:
        if any(count != 4 for count in self.invocations.values()):
            raise ValueError("R1-A capture marker invocation count differs")
        return {
            "markers": [
                self.topology.marker_for(node.ordinal) for node in self.topology.nodes
            ],
            "invocations": {
                str(ordinal): self.invocations[ordinal]
                for ordinal in sorted(self.invocations)
            },
            "expected_invocations_per_marker": 4,
            "capture_only": True,
        }


def _metric(
    reference: torch.Tensor, candidate: torch.Tensor
) -> tuple[float, bool, int]:
    if reference.shape != candidate.shape:
        raise ValueError("R1-A semantic tensor shape differs")
    return (
        float((reference - candidate).abs().max().item()),
        bool(torch.equal(torch.sign(reference), torch.sign(candidate))),
        reference.numel(),
    )


def _semantic_receipt(
    baseline: CIBCIBPCUDAGraphPlanV1, candidate: CIBCIBPCUDAGraphPlanV1
) -> dict[str, object]:
    if set(baseline.outputs) != set(candidate.outputs):
        raise ValueError("R1-A semantic output inventory differs")
    maximum = 0.0
    sign_exact = True
    element_count = 0
    for name in sorted(baseline.outputs):
        for side in ("lower", "upper"):
            difference, signs, count = _metric(
                getattr(baseline.outputs[name], side),
                getattr(candidate.outputs[name], side),
            )
            maximum = max(maximum, difference)
            sign_exact = sign_exact and signs
            element_count += count
    if maximum > SEMANTIC_ATOL or not sign_exact:
        raise ValueError("R1-A candidate graph semantic receipt differs")
    return {
        "maximum_absolute_difference": maximum,
        "sign_exact": sign_exact,
        "element_count": element_count,
        "atol": SEMANTIC_ATOL,
        "rtol": SEMANTIC_ATOL,
        "baseline_launch_count": baseline.launch_count,
        "candidate_launch_count": candidate.launch_count,
        "fallback_count": 0,
        "eager_shadow_count": 0,
    }


def _timed_group(
    plan: CIBCIBPCUDAGraphPlanV1, lower: torch.Tensor, upper: torch.Tensor, repeats: int
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        plan.replay(input_lower=lower, input_upper=upper)
    end.record()
    end.synchronize()
    milliseconds = float(start.elapsed_time(end)) / repeats
    if not math.isfinite(milliseconds) or milliseconds <= 0.0:
        raise ValueError("R1-A CUDA-event timing differs")
    return milliseconds


def _timed_profile_group(
    plan: CIBCIBPCUDAGraphPlanV1,
    lower: torch.Tensor,
    upper: torch.Tensor,
    repeats: int,
    *,
    group_ordinal: int,
    profile_backend: str,
) -> float:
    if profile_backend != "nsys":
        return _timed_group(plan, lower, upper, repeats)
    marker = f"boundflow.r1/profile-group/{group_ordinal}/{repeats}"
    torch.cuda.nvtx.range_push(marker)
    try:
        return _timed_group(plan, lower, upper, repeats)
    finally:
        torch.cuda.nvtx.range_pop()


def _warmup(
    plan: CIBCIBPCUDAGraphPlanV1, lower: torch.Tensor, upper: torch.Tensor, count: int
) -> None:
    for _ in range(count):
        plan.replay(input_lower=lower, input_upper=upper)
    torch.cuda.synchronize()


def _profile_inventory(profiler: Any) -> dict[str, object]:
    device_counts: Counter[str] = Counter()
    name_counts: Counter[str] = Counter()
    for event in profiler.events():
        device_counts[str(event.device_type)] += 1
        name_counts[str(event.name)] += 1
    return {
        "backend": "torch_profiler_smoke",
        "device_event_counts": dict(sorted(device_counts.items())),
        "event_name_counts": dict(sorted(name_counts.items())),
        "event_count": sum(device_counts.values()),
        "owner_ledger_available": False,
        "formal_attribution_available": False,
        "reason": "nsys_export_unavailable",
    }


def _nsys_anchor(source: CuptiTimestampSource, ordinal: int) -> dict[str, object]:
    host_before = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    gpu_before = source.timestamp_ns()
    marker = f"boundflow.r1/calibration-anchor/{ordinal}"
    torch.cuda.nvtx.mark(marker)
    gpu_after = source.timestamp_ns()
    host_after = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    return {
        "ordinal": ordinal,
        "marker": marker,
        "gpu_before_ns": gpu_before,
        "gpu_after_ns": gpu_after,
        "gpu_timestamp_ns": (gpu_before + gpu_after) / 2.0,
        "gpu_bracket_width_ns": gpu_after - gpu_before,
        "host_before_ns": host_before,
        "host_after_ns": host_after,
    }


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("R1-A worker requires CUDA")
    module, spec, lower, upper = _prepare(args)
    topology = _topology(module, spec)
    marker_factory = _NVTXMarkerFactory(topology)
    baseline = CIBCIBPCUDAGraphPlanV1(module, spec.value_name, lower, upper, None)
    candidate = CIBCIBPCUDAGraphPlanV1(
        module,
        spec.value_name,
        lower,
        upper,
        THREADS_PER_BLOCK,
        op_context_factory=marker_factory,
    )
    baseline.replay(input_lower=lower, input_upper=upper)
    candidate.replay(input_lower=lower, input_upper=upper)
    torch.cuda.synchronize()
    semantics = _semantic_receipt(baseline, candidate)
    warmup_count = CONTROL_WARMUP if args.mode == "control" else PROFILE_WARMUP
    group_count = CONTROL_GROUP_COUNT if args.mode == "control" else PROFILE_GROUP_COUNT
    repeats = CONTROL_REPEATS if args.mode == "control" else PROFILE_REPEATS
    _warmup(candidate, lower, upper, warmup_count)
    calibration = None
    profile_inventory = None
    if args.mode == "profile":
        timestamp_source = CuptiTimestampSource(args.cupti_library)
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        if args.profile_backend == "torch":
            # The first ActivityProfiler session initializes CUPTI's timestamp
            # epoch. Exclude that initialization before taking the frozen fit.
            with torch.profiler.profile(activities=activities):
                candidate.replay(input_lower=lower, input_upper=upper)
                torch.cuda.synchronize()
        before = collect_cupti_triplets(timestamp_source, phase="before")
        if args.profile_backend == "torch":
            with torch.profiler.profile(activities=activities) as profiler:
                groups = [
                    _timed_profile_group(
                        candidate,
                        lower,
                        upper,
                        repeats,
                        group_ordinal=group,
                        profile_backend=args.profile_backend,
                    )
                    for group in range(group_count)
                ]
            profile_inventory = _profile_inventory(profiler)
        else:
            anchors = [_nsys_anchor(timestamp_source, 0)]
            midpoint = group_count // 2
            groups = [
                _timed_profile_group(
                    candidate,
                    lower,
                    upper,
                    repeats,
                    group_ordinal=group,
                    profile_backend=args.profile_backend,
                )
                for group in range(midpoint)
            ]
            anchors.append(_nsys_anchor(timestamp_source, 1))
            groups.extend(
                _timed_profile_group(
                    candidate,
                    lower,
                    upper,
                    repeats,
                    group_ordinal=group,
                    profile_backend=args.profile_backend,
                )
                for group in range(midpoint, group_count)
            )
            anchors.append(_nsys_anchor(timestamp_source, 2))
            profile_inventory = {
                "backend": "nsys_pending_export",
                "anchors": anchors,
                "owner_ledger_available": False,
                "formal_attribution_available": False,
                "reason": "nsys_export_pending",
            }
        after = collect_cupti_triplets(timestamp_source, phase="after")
        calibration = derive_clock_calibration((*before, *after)).to_dict()
    else:
        groups = [
            _timed_group(candidate, lower, upper, repeats) for _ in range(group_count)
        ]
    median_ms = statistics.median(groups)
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "mode": args.mode,
        "pair_ordinal": args.pair_ordinal,
        "topology": topology.to_dict(),
        "marker_receipt": marker_factory.receipt(),
        "semantic_receipt": semantics,
        "groups_ms": groups,
        "median_ms": median_ms,
        "warmup_count": warmup_count,
        "group_count": group_count,
        "repeats_per_group": repeats,
        "input_copy_included": True,
        "threads_per_block": THREADS_PER_BLOCK,
        "calibration_receipt": calibration,
        "profile_inventory": profile_inventory,
        "profiler_epoch_warmup_excluded": (
            args.mode == "profile" and args.profile_backend == "torch"
        ),
        "environment": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "stream_id": int(torch.cuda.current_stream().cuda_stream),
        },
        "cupti_admitted": bool(calibration and calibration["cupti_admitted"]),
        "formal_attribution_admitted": False,
        "performance_claimed": False,
    }
    payload["worker_hash"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("control", "profile"), required=True)
    parser.add_argument("--pair-ordinal", type=int, choices=range(6), required=True)
    parser.add_argument("--cupti-library", default="/opt/cuda/lib64/libcupti.so")
    parser.add_argument("--profile-backend", choices=("torch", "nsys"), default="torch")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = _run(args)
    encoded = canonical_json(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
