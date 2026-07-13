#!/usr/bin/env python
"""Measure PR-12 runtime Pareto and evaluate the frozen held-out planner."""

# mypy: disable-error-code=import-untyped
# pylint: disable=too-many-lines,too-many-locals,too-many-statements,too-many-branches

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any, Callable, Iterable, Optional, Sequence

import torch

from boundflow.backends.tvm.fused_crown_conv2d import (
    build_fused_crown_conv2d_module,
)
from boundflow.backends.tvm.fused_crown_linear import build_fused_crown_linear_module
from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.planner.execution_candidate import BackendVariant, OperatorFamily
from boundflow.planner.fused_crown_backend import (
    FusedCrownBackendObservation,
    FusedCrownBackendPlanner,
    FusedCrownMultiBackendPlanner,
)
from boundflow.runtime.crown_ibp import run_crown_ibp_mlp
from boundflow.runtime.fused_crown import (
    FusedCrownExecutor,
    FusedReluAffineRequest,
    TVMFusedCrownExecutor,
    build_fused_crown_runtime_selection,
    plan_fused_crown_regions,
)
from boundflow.runtime.task_executor import InputSpec

SCHEMA_VERSION = "boundflow.backend_profile/v2.0"
MANIFEST_SCHEMA_VERSION = "boundflow.pr12-runtime-pareto-manifest/v1"
PLANNER_SCHEMA_VERSION = "boundflow.pr12-heldout-planner/v1"
HELDOUT_SPLIT_ID = "pr12-final-heldout-v1"
DEFAULT_STREAMS = ("default", "custom")
DEFAULT_BACKENDS = (
    BackendVariant.PYTORCH_EAGER,
    BackendVariant.TVM_FUSED_TIR,
)


def _frozen_candidate_backends(split: dict[str, Any]) -> tuple[str, ...]:
    """Normalize candidate ids stored by a frozen benchmark split."""

    return tuple(
        "pytorch_chunked" if item.startswith("pytorch_chunked_r") else item
        for item in map(str, split.get("candidate_set", ()))
    )


@dataclass(frozen=True)
class RuntimeWorkload:  # pylint: disable=too-many-instance-attributes
    """Static benchmark workload derived from the frozen PR-12 split."""

    case_id: str
    family: str
    planner_family: OperatorFamily
    split_role: str
    domain: int
    spec: int
    budget_bytes: int
    boundary_bytes: int
    expected_regions: int
    config: dict[str, Any]


class CountingTVMFusedCrownExecutor(TVMFusedCrownExecutor):
    """Expose fused-region dispatch counts without changing runtime semantics."""

    def __init__(self) -> None:
        self.run_calls = 0

    def run(
        self,
        request: FusedReluAffineRequest,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ):  # type: ignore[no-untyped-def]
        self.run_calls += 1
        return super().run(request, stream=stream)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile expects non-empty values")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    lower = _percentile(ordered, 0.25)
    upper = _percentile(ordered, 0.75)
    return {
        "median_ms": statistics.median(ordered),
        "iqr_ms": upper - lower,
        "p95_ms": _percentile(ordered, 0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _conv_attrs(
    *, stride: int = 1, padding: int = 1, groups: int = 1
) -> dict[str, object]:
    return {
        "stride": (stride, stride),
        "padding": (padding, padding),
        "dilation": (1, 1),
        "groups": groups,
    }


def _linear_module(workload: RuntimeWorkload, device: torch.device) -> BFTaskModule:
    current = int(workload.config["current"])
    previous = int(workload.config["previous"])
    task = BoundTask(
        task_id=workload.case_id,
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp("linear", "region_affine", ["input", "w1", "b1"], ["h1"]),
            TaskOp("relu", "region_relu", ["h1"], ["r1"]),
            TaskOp("linear", "head", ["r1", "w2", "b2"], ["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id=workload.case_id,
        bindings={
            "params": {
                "w1": torch.randn(current, previous, device=device),
                "b1": torch.randn(current, device=device),
                "w2": torch.randn(workload.spec, current, device=device),
                "b2": torch.randn(workload.spec, device=device),
            }
        },
    )


def _conv_module(workload: RuntimeWorkload, device: torch.device) -> BFTaskModule:
    channels = int(workload.config["channels"])
    height = int(workload.config["height"])
    width = int(workload.config["width"])
    kernel = int(workload.config.get("kernel", 3))
    padding = kernel // 2
    task = BoundTask(
        task_id=workload.case_id,
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                "conv2d",
                "region_affine",
                ["input", "w1", "b1"],
                ["h1"],
                _conv_attrs(padding=padding),
            ),
            TaskOp("relu", "region_relu", ["h1"], ["r1"]),
            TaskOp(
                "flatten",
                "flatten",
                ["r1"],
                ["flat"],
                {"start_dim": 1, "end_dim": -1},
            ),
            TaskOp("linear", "head", ["flat", "wh", "bh"], ["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id=workload.case_id,
        bindings={
            "params": {
                "w1": torch.randn(channels, channels, kernel, kernel, device=device),
                "b1": torch.randn(channels, device=device),
                "wh": torch.randn(
                    workload.spec, channels * height * width, device=device
                ),
                "bh": torch.randn(workload.spec, device=device),
            }
        },
    )


def _fanout_control_module(
    workload: RuntimeWorkload, device: torch.device
) -> BFTaskModule:
    current = int(workload.config["current"])
    previous = int(workload.config["previous"])
    task = BoundTask(
        task_id=workload.case_id,
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp("linear", "shared", ["input", "w1", "b1"], ["h"]),
            TaskOp("relu", "relu", ["h"], ["r"]),
            TaskOp("linear", "direct", ["h", "wd", "bd"], ["d"]),
            TaskOp("linear", "relu_path", ["r", "wr", "br"], ["q"]),
            TaskOp("add", "merge", ["d", "q"], ["out"]),
        ],
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id=workload.case_id,
        bindings={
            "params": {
                "w1": torch.randn(current, previous, device=device),
                "b1": torch.randn(current, device=device),
                "wd": torch.randn(workload.spec, current, device=device),
                "bd": torch.randn(workload.spec, device=device),
                "wr": torch.randn(workload.spec, current, device=device),
                "br": torch.randn(workload.spec, device=device),
            }
        },
    )


def _mini_resnet_module(
    workload: RuntimeWorkload, device: torch.device
) -> BFTaskModule:
    width = int(workload.config["width"])
    blocks = int(workload.config["blocks"])
    if blocks < 2 or width < 4 or width % 2:
        raise ValueError("mini_resnet requires blocks>=2 and positive even width>=4")
    stem_width = width // 2
    ops: list[TaskOp] = [
        TaskOp(
            "conv2d",
            "stem",
            ["input", "w_stem", "b_stem"],
            ["h_stem"],
            _conv_attrs(),
        ),
        TaskOp("relu", "stem_relu", ["h_stem"], ["r0"]),
    ]
    params: dict[str, torch.Tensor] = {
        "w_stem": torch.randn(stem_width, 3, 3, 3, device=device),
        "b_stem": torch.randn(stem_width, device=device),
    }
    previous = "r0"
    channels = stem_width
    spatial = 16
    for block in range(blocks):
        downsample = block == 1
        output_channels = width if downsample else channels
        stride = 2 if downsample else 1
        h1, r1, h2, merged, rout = (
            f"b{block}_h1",
            f"b{block}_r1",
            f"b{block}_h2",
            f"b{block}_sum",
            f"b{block}_out",
        )
        ops.extend(
            [
                TaskOp(
                    "conv2d",
                    f"b{block}_conv1",
                    [previous, f"w{block}1", f"b{block}1"],
                    [h1],
                    _conv_attrs(stride=stride),
                ),
                TaskOp("relu", f"b{block}_relu1", [h1], [r1]),
                TaskOp(
                    "conv2d",
                    f"b{block}_conv2",
                    [r1, f"w{block}2", f"b{block}2"],
                    [h2],
                    _conv_attrs(),
                ),
            ]
        )
        params[f"w{block}1"] = torch.randn(
            output_channels, channels, 3, 3, device=device
        )
        params[f"b{block}1"] = torch.randn(output_channels, device=device)
        params[f"w{block}2"] = torch.randn(
            output_channels, output_channels, 3, 3, device=device
        )
        params[f"b{block}2"] = torch.randn(output_channels, device=device)
        skip = previous
        if downsample:
            skip = f"b{block}_skip"
            ops.append(
                TaskOp(
                    "conv2d",
                    f"b{block}_projection",
                    [previous, f"wp{block}", f"bp{block}"],
                    [skip],
                    _conv_attrs(stride=2, padding=0),
                )
            )
            params[f"wp{block}"] = torch.randn(
                output_channels, channels, 1, 1, device=device
            )
            params[f"bp{block}"] = torch.randn(output_channels, device=device)
            spatial //= 2
        ops.extend(
            [
                TaskOp("add", f"b{block}_add", [h2, skip], [merged]),
                TaskOp("relu", f"b{block}_relu_out", [merged], [rout]),
            ]
        )
        previous = rout
        channels = output_channels
    ops.extend(
        [
            TaskOp(
                "flatten",
                "flatten",
                [previous],
                ["flat"],
                {"start_dim": 1, "end_dim": -1},
            ),
            TaskOp("linear", "head", ["flat", "wh", "bh"], ["out"]),
        ]
    )
    params["wh"] = torch.randn(
        workload.spec, channels * spatial * spatial, device=device
    )
    params["bh"] = torch.randn(workload.spec, device=device)
    task = BoundTask(
        task_id=workload.case_id,
        kind=TaskKind.INTERVAL_IBP,
        ops=ops,
        input_values=["input"],
        output_values=["out"],
    )
    return BFTaskModule(
        tasks=[task],
        entry_task_id=workload.case_id,
        bindings={"params": params},
    )


def _build_query(
    workload: RuntimeWorkload, device: torch.device
) -> tuple[BFTaskModule, InputSpec]:
    torch.manual_seed(12000 + sum(ord(char) for char in workload.case_id))
    center_shape: tuple[int, ...]
    if workload.family == "linear":
        module = _linear_module(workload, device)
        center_shape = (workload.domain, int(workload.config["previous"]))
    elif workload.family == "conv2d":
        module = _conv_module(workload, device)
        center_shape = (
            workload.domain,
            int(workload.config["channels"]),
            int(workload.config["height"]),
            int(workload.config["width"]),
        )
    elif workload.family == "mini_resnet":
        module = _mini_resnet_module(workload, device)
        center_shape = (workload.domain, 3, 16, 16)
    elif workload.family == "fanout_control":
        module = _fanout_control_module(workload, device)
        center_shape = (workload.domain, int(workload.config["previous"]))
    else:
        raise ValueError(f"unsupported workload family: {workload.family}")
    center = torch.randn(*center_shape, device=device)
    return module, InputSpec.linf(value_name="input", center=center, eps=0.03)


def _workload(record: dict[str, Any], *, split_role: str) -> RuntimeWorkload:
    family = str(record["family"])
    domain = int(record["domain"])
    spec = int(record["spec"])
    budget_bytes = int(record.get("budget_mib", 1024)) * 1024 * 1024
    if family == "linear":
        elements = int(record["current"])
        boundary_bytes = 2 * domain * spec * elements * 4
        regions = 1
        planner_family = OperatorFamily.LINEAR
    elif family == "conv2d":
        elements = (
            int(record["channels"]) * int(record["height"]) * int(record["width"])
        )
        boundary_bytes = 2 * domain * spec * elements * 4
        regions = 1
        planner_family = OperatorFamily.CONV2D
    elif family == "mini_resnet":
        width = int(record["width"])
        blocks = int(record["blocks"])
        stem_elements = (width // 2) * 16 * 16
        block_elements = stem_elements + width * 8 * 8 * (blocks - 1)
        boundary_bytes = 2 * domain * spec * (stem_elements + block_elements) * 4
        regions = blocks + 1
        planner_family = OperatorFamily.CONV2D
    else:
        raise ValueError(f"unsupported split family: {family}")
    case_id = (
        str(record["case_id"]) if "case_id" in record else _calibration_case_id(record)
    )
    return RuntimeWorkload(
        case_id=case_id,
        family=family,
        planner_family=planner_family,
        split_role=split_role,
        domain=domain,
        spec=spec,
        budget_bytes=budget_bytes,
        boundary_bytes=boundary_bytes,
        expected_regions=regions,
        config=dict(record),
    )


def _fallback_control_workload() -> RuntimeWorkload:
    """Return an explicit graph-ineligible control outside the frozen held-out set."""

    config: dict[str, Any] = {
        "case_id": "fanout-graph-fallback-control",
        "family": "fanout_control",
        "domain": 3,
        "spec": 17,
        "current": 29,
        "previous": 13,
        "budget_mib": 256,
    }
    return RuntimeWorkload(
        case_id=str(config["case_id"]),
        family="fanout_control",
        planner_family=OperatorFamily.LINEAR,
        split_role="fallback_control",
        domain=int(config["domain"]),
        spec=int(config["spec"]),
        budget_bytes=int(config["budget_mib"]) * 1024 * 1024,
        boundary_bytes=(
            2 * int(config["domain"]) * int(config["spec"]) * int(config["current"]) * 4
        ),
        expected_regions=0,
        config=config,
    )


def _calibration_case_id(record: dict[str, Any]) -> str:
    family = str(record["family"])
    if family == "linear":
        return (
            f"cal-linear-d{record['domain']}-s{record['spec']}-"
            f"i{record['current']}-j{record['previous']}"
        )
    return (
        f"cal-conv-d{record['domain']}-s{record['spec']}-"
        f"c{record['channels']}-h{record['height']}-w{record['width']}"
    )


def _event_call(
    call: Callable[[], Any], stream: torch.cuda.Stream
) -> tuple[float, float, Any]:
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    with torch.cuda.stream(stream):
        started.record(stream)
        result = call()
        finished.record(stream)
    finished.synchronize()
    wall_ms = (time.perf_counter() - wall_started) * 1000.0
    return wall_ms, float(started.elapsed_time(finished)), result


def _warm_groups(
    call: Callable[[], Any],
    stream: torch.cuda.Stream,
    *,
    warmup: int,
    groups: int,
    repeats: int,
) -> tuple[list[float], list[float]]:
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            call()
    stream.synchronize()
    host_group_ms: list[float] = []
    event_samples_ms: list[float] = []
    for _ in range(groups):
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        started_wall = time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(repeats):
                started = torch.cuda.Event(enable_timing=True)
                finished = torch.cuda.Event(enable_timing=True)
                started.record(stream)
                call()
                finished.record(stream)
                events.append((started, finished))
        stream.synchronize()
        host_group_ms.append((time.perf_counter() - started_wall) * 1000.0 / repeats)
        event_samples_ms.extend(
            float(started.elapsed_time(finished)) for started, finished in events
        )
    return host_group_ms, event_samples_ms


def _measure_memory(
    call: Callable[[], Any], stream: torch.cuda.Stream, device: torch.device
) -> dict[str, int]:
    gc.collect()
    torch.cuda.empty_cache()
    stream.synchronize()
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.cuda.stream(stream):
        result = call()
    stream.synchronize()
    output_bytes = int(result.lower.numel() + result.upper.numel()) * int(
        result.lower.element_size()
    )
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    del result
    return {
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": max(0, peak_allocated - baseline_allocated),
        "peak_reserved_delta_bytes": max(0, peak_reserved - baseline_reserved),
        "output_bytes": output_bytes,
        "temporary_workspace_upper_bound_bytes": max(
            0, peak_allocated - baseline_allocated - output_bytes
        ),
    }


def _max_abs_diff(actual: Any, expected: Any) -> float:
    return max(
        float((actual.lower - expected.lower).abs().max().item()),
        float((actual.upper - expected.upper).abs().max().item()),
    )


def _max_rel_diff(actual: Any, expected: Any) -> float:
    values = []
    for actual_tensor, expected_tensor in (
        (actual.lower, expected.lower),
        (actual.upper, expected.upper),
    ):
        denominator = expected_tensor.abs().clamp_min(1e-6)
        values.append(
            float(((actual_tensor - expected_tensor).abs() / denominator).max())
        )
    return max(values)


def _clear_fused_compile_cache() -> None:
    build_fused_crown_linear_module.cache_clear()
    build_fused_crown_conv2d_module.cache_clear()


def _benchmark_candidate(  # pylint: disable=too-many-arguments
    workload: RuntimeWorkload,
    module: BFTaskModule,
    input_spec: InputSpec,
    *,
    backend: BackendVariant,
    stream_name: str,
    expected: Any,
    warmup: int,
    groups: int,
    repeats: int,
    chunk_rows: int,
    split_id: str,
) -> dict[str, Any]:
    device = input_spec.center.device
    stream = (
        torch.cuda.default_stream(device)
        if stream_name == "default"
        else torch.cuda.Stream(device=device)
    )
    selection = build_fused_crown_runtime_selection(
        module.get_entry_task().ops,
        backend=backend.value,
        chunk_rows=chunk_rows,
    )
    steps = selection.steps
    executor: Optional[FusedCrownExecutor] = selection.executor
    if backend == BackendVariant.TVM_FUSED_TIR:
        executor = CountingTVMFusedCrownExecutor()
        _clear_fused_compile_cache()

    def call() -> Any:
        return run_crown_ibp_mlp(
            module,
            input_spec,
            fused_crown_executor=executor,
            fused_crown_steps=steps if executor is not None else (),
        )

    first_wall, first_event, first_result = _event_call(call, stream)
    max_diff = _max_abs_diff(first_result, expected)
    max_rel_diff = _max_rel_diff(first_result, expected)
    finite = bool(
        torch.isfinite(first_result.lower).all()
        and torch.isfinite(first_result.upper).all()
    )
    ordered = bool((first_result.lower <= first_result.upper).all())
    allclose = bool(
        torch.allclose(first_result.lower, expected.lower, rtol=2e-4, atol=2e-4)
        and torch.allclose(first_result.upper, expected.upper, rtol=2e-4, atol=2e-4)
    )
    del first_result
    cold_wall, cold_event, cold_result = _event_call(call, stream)
    del cold_result
    host_groups, event_samples = _warm_groups(
        call, stream, warmup=warmup, groups=groups, repeats=repeats
    )
    memory = _measure_memory(call, stream, device)
    fused_calls_per_query = len(steps) if executor is not None else 0
    compile_cache = {
        "linear": build_fused_crown_linear_module.cache_info()._asdict(),
        "conv2d": build_fused_crown_conv2d_module.cache_info()._asdict(),
    }
    correct = finite and ordered and allclose
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if correct else "fail",
        "error": None if correct else {"error_type": "CorrectnessFailure"},
        "split": {
            "split_id": split_id,
            "role": workload.split_role,
        },
        "workload": {
            "case_id": workload.case_id,
            "family": workload.family,
            "planner_family": workload.planner_family.value,
            "domain": workload.domain,
            "spec": workload.spec,
            "boundary_bytes": workload.boundary_bytes,
            "budget_bytes": workload.budget_bytes,
            "expected_fused_regions": workload.expected_regions,
            "config": workload.config,
        },
        "candidate": {
            "backend": backend.value,
            "stream": stream_name,
            "eligible": bool(steps) if executor is not None else True,
            "planned_fused_regions": len(steps),
            "fused_dispatches_per_query": fused_calls_per_query,
            "chunk_rows": (
                chunk_rows if backend == BackendVariant.PYTORCH_CHUNKED else None
            ),
        },
        "runtime": {
            "compile_first_run_wall_ms": first_wall,
            "compile_first_run_cuda_event_ms": first_event,
            "cold_wall_ms": cold_wall,
            "cold_cuda_event_ms": cold_event,
            "estimated_compile_overhead_ms": max(0.0, first_wall - cold_wall),
            "host_group_per_query": _summary(host_groups),
            "cuda_event_per_query": _summary(event_samples),
            "warmup": warmup,
            "independent_groups": groups,
            "repeats_per_group": repeats,
            "host_group_samples_ms": host_groups,
            "cuda_event_samples_ms": event_samples,
        },
        "memory": memory,
        "compile_cache": compile_cache,
        "correctness": {
            "max_abs_diff": max_diff,
            "max_rel_diff": max_rel_diff,
            "finite": finite,
            "lower_le_upper": ordered,
            "allclose": allclose,
            "rtol": 2e-4,
            "atol": 2e-4,
        },
    }


def _observations_from_rows(
    rows: Iterable[dict[str, Any]], *, stream: str = "default"
) -> list[FusedCrownBackendObservation]:
    observations: list[FusedCrownBackendObservation] = []
    for row in rows:
        if row.get("status") != "ok" or row["candidate"]["stream"] != stream:
            continue
        backend = BackendVariant(row["candidate"]["backend"])
        observations.append(
            FusedCrownBackendObservation(
                case_id=row["workload"]["case_id"],
                family=OperatorFamily(row["workload"]["planner_family"]),
                backend=backend,
                boundary_bytes=int(row["workload"]["boundary_bytes"]),
                region_count=int(row["workload"]["expected_fused_regions"]),
                warm_latency_ms=float(
                    row["runtime"]["host_group_per_query"]["median_ms"]
                ),
                peak_allocated_bytes=int(row["memory"]["peak_allocated_delta_bytes"]),
                eligible=bool(row["candidate"]["eligible"]),
                correct=True,
            )
        )
    return observations


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _planner_evaluation(
    workload: RuntimeWorkload,
    rows: Sequence[dict[str, Any]],
    planner: FusedCrownBackendPlanner | FusedCrownMultiBackendPlanner,
    *,
    split_id: str,
) -> dict[str, Any]:
    fused_eligible = workload.expected_regions > 0
    planner_started = time.perf_counter()
    if isinstance(planner, FusedCrownMultiBackendPlanner):
        eligible_backends = [BackendVariant.PYTORCH_EAGER]
        if fused_eligible:
            eligible_backends.extend(
                [BackendVariant.PYTORCH_CHUNKED, BackendVariant.TVM_FUSED_TIR]
            )
        multi_decision = planner.decide(
            family=workload.planner_family,
            boundary_bytes=workload.boundary_bytes,
            region_count=max(1, workload.expected_regions),
            budget_bytes=workload.budget_bytes,
            eligible_backends=eligible_backends,
        )
        decision_backend = multi_decision.backend
        decision_payload = multi_decision.to_dict()
        use_fused = decision_backend != BackendVariant.PYTORCH_EAGER
        schema_version = "boundflow.pr12-heldout-planner/v2"
    else:
        legacy_decision = planner.decide(
            family=workload.planner_family,
            boundary_bytes=workload.boundary_bytes,
            region_count=max(1, workload.expected_regions),
            budget_bytes=workload.budget_bytes,
            eligible=fused_eligible,
        )
        decision_backend = legacy_decision.backend
        decision_payload = legacy_decision.to_dict()
        use_fused = legacy_decision.use_fused
        schema_version = PLANNER_SCHEMA_VERSION
    planner_overhead_ms = (time.perf_counter() - planner_started) * 1000.0
    candidates = {
        BackendVariant(row["candidate"]["backend"]): row
        for row in rows
        if row["candidate"]["stream"] == "default"
        and row["status"] == "ok"
        and (
            row["candidate"]["backend"] == BackendVariant.PYTORCH_EAGER.value
            or row["candidate"]["eligible"]
        )
    }
    feasible = {
        backend: row
        for backend, row in candidates.items()
        if int(row["memory"]["peak_allocated_delta_bytes"]) <= workload.budget_bytes
    }
    oracle_pool = feasible or candidates
    oracle_backend = min(
        oracle_pool,
        key=lambda backend: float(
            oracle_pool[backend]["runtime"]["host_group_per_query"]["median_ms"]
        ),
    )
    selected = candidates[decision_backend]
    selected_latency = float(selected["runtime"]["host_group_per_query"]["median_ms"])
    oracle_latency = float(
        candidates[oracle_backend]["runtime"]["host_group_per_query"]["median_ms"]
    )
    return {
        "schema_version": schema_version,
        "split_id": split_id,
        "role": workload.split_role,
        "case_id": workload.case_id,
        "family": workload.family,
        "boundary_bytes": workload.boundary_bytes,
        "budget_bytes": workload.budget_bytes,
        "decision": decision_payload,
        "oracle_backend": oracle_backend.value,
        "selected_latency_ms": selected_latency,
        "planner_overhead_ms": planner_overhead_ms,
        "oracle_latency_ms": oracle_latency,
        "latency_regret": selected_latency / oracle_latency,
        "selected_peak_allocated_bytes": int(
            selected["memory"]["peak_allocated_delta_bytes"]
        ),
        "selected_budget_feasible": int(
            selected["memory"]["peak_allocated_delta_bytes"]
        )
        <= workload.budget_bytes,
        "unsafe_fusion": bool(use_fused and not fused_eligible),
    }


def _environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": properties.total_memory,
        "tvm_home": os.environ.get("TVM_HOME", ""),
    }


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run calibration or frozen final held-out evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=("calibration", "development", "heldout"), required=True
    )
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--calibration-jsonl", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--chunk-rows", type=int)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--streams", default=",".join(DEFAULT_STREAMS))
    parser.add_argument(
        "--backends",
        help="comma-separated candidates; defaults to the frozen split candidate set",
    )
    parser.add_argument("--planner-version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.split == "heldout" and args.calibration_jsonl is None:
        parser.error("--calibration-jsonl is required for heldout")
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    chunk_rows = (
        int(args.chunk_rows)
        if args.chunk_rows is not None
        else int(split.get("chunk_rows", 128))
    )
    if min(args.warmup, args.groups, args.repeats, chunk_rows) <= 0:
        parser.error("warmup/groups/repeats/chunk-rows must be positive")
    streams = tuple(item.strip() for item in args.streams.split(",") if item.strip())
    if not streams or any(item not in DEFAULT_STREAMS for item in streams):
        parser.error("streams must be a comma-separated subset of default,custom")
    frozen_candidates = _frozen_candidate_backends(split)
    backend_values = args.backends or ",".join(
        frozen_candidates or tuple(backend.value for backend in DEFAULT_BACKENDS)
    )
    try:
        backends = tuple(
            BackendVariant(item.strip())
            for item in backend_values.split(",")
            if item.strip()
        )
    except ValueError as error:
        parser.error(str(error))
    allowed_backends = {
        BackendVariant.PYTORCH_EAGER,
        BackendVariant.PYTORCH_CHUNKED,
        BackendVariant.TVM_FUSED_TIR,
    }
    if not backends or any(backend not in allowed_backends for backend in backends):
        parser.error("unsupported PR-12 runtime backend")
    if args.split != "development" and frozen_candidates:
        if tuple(backend.value for backend in backends) != frozen_candidates:
            parser.error("formal split must use its complete frozen candidate set")
        if chunk_rows != int(split["chunk_rows"]):
            parser.error("formal split must use its frozen chunk_rows")
    split_id = str(split.get("split_id", ""))
    if not split_id:
        raise ValueError("held-out split must have a non-empty split_id")
    records = split["calibration" if args.split == "calibration" else "final_heldout"]
    workloads = [_workload(record, split_role=args.split) for record in records]
    selected_case_ids = {
        item.strip() for item in args.case_ids.split(",") if item.strip()
    }
    if selected_case_ids:
        workloads = [
            workload for workload in workloads if workload.case_id in selected_case_ids
        ]
        missing = selected_case_ids - {workload.case_id for workload in workloads}
        if missing:
            parser.error(f"unknown case ids: {sorted(missing)}")
    if args.split == "heldout":
        workloads.append(_fallback_control_workload())
    planner: Optional[FusedCrownBackendPlanner | FusedCrownMultiBackendPlanner] = None
    calibration_hash: Optional[str] = None
    if args.split == "heldout":
        assert args.calibration_jsonl is not None
        calibration_rows = _read_jsonl(args.calibration_jsonl)
        observations = _observations_from_rows(calibration_rows)
        planner = (
            FusedCrownBackendPlanner.fit(observations)
            if args.planner_version == "v1"
            else FusedCrownMultiBackendPlanner.fit(observations)
        )
        calibration_hash = _sha256(args.calibration_jsonl)

    args.out_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    planner_rows: list[dict[str, Any]] = []
    for workload in workloads:
        module, input_spec = _build_query(workload, device)
        planned_regions = len(plan_fused_crown_regions(module.get_entry_task().ops))
        if planned_regions != workload.expected_regions:
            raise RuntimeError(
                f"{workload.case_id} expected {workload.expected_regions} fused regions, "
                f"planner produced {planned_regions}"
            )
        with torch.cuda.stream(torch.cuda.default_stream(device)):
            expected = run_crown_ibp_mlp(module, input_spec)
        torch.cuda.default_stream(device).synchronize()
        case_rows: list[dict[str, Any]] = []
        for stream in streams:
            for backend in backends:
                row = _benchmark_candidate(
                    workload,
                    module,
                    input_spec,
                    backend=backend,
                    stream_name=stream,
                    expected=expected,
                    warmup=args.warmup,
                    groups=args.groups,
                    repeats=args.repeats,
                    chunk_rows=chunk_rows,
                    split_id=split_id,
                )
                rows.append(row)
                case_rows.append(row)
        if planner is not None:
            planner_rows.append(
                _planner_evaluation(workload, case_rows, planner, split_id=split_id)
            )
        del expected, input_spec, module
        gc.collect()
        torch.cuda.empty_cache()

    raw_path = args.out_dir / "raw.jsonl"
    _write_jsonl(raw_path, rows)
    outputs = {"raw.jsonl": _sha256(raw_path)}
    if planner is not None:
        planner_path = args.out_dir / "planner.jsonl"
        model_path = args.out_dir / "planner_model.json"
        _write_jsonl(planner_path, planner_rows)
        model_path.write_text(
            json.dumps(planner.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outputs.update(
            {
                "planner.jsonl": _sha256(planner_path),
                "planner_model.json": _sha256(model_path),
            }
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "split": args.split,
        "split_id": split_id,
        "planner_version": args.planner_version,
        "split_file": str(args.split_file),
        "split_file_sha256": _sha256(args.split_file),
        "calibration_jsonl_sha256": calibration_hash,
        "measurement_protocol": {
            "streams": list(streams),
            "warmup": args.warmup,
            "independent_groups": args.groups,
            "repeats_per_group": args.repeats,
            "cuda_timing": "events_on_measured_stream",
            "host_timing": "per-group wall time with one stream synchronize",
            "global_synchronize_in_timed_region": False,
            "memory": "torch_allocator_peak_delta",
        },
        "environment": _environment(),
        "row_count": len(rows),
        "planner_row_count": len(planner_rows),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        },
        "outputs": outputs,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
