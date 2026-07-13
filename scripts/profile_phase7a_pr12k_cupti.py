#!/usr/bin/env python
"""Collect PR-12K CUPTI activity traces when hardware counters are unavailable."""

# mypy: disable-error-code=import-untyped
# pylint: disable=broad-exception-caught,duplicate-code,too-many-arguments
# pylint: disable=too-many-branches,too-many-lines,too-many-locals,too-many-statements

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Literal, Optional, Sequence

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from boundflow.ir.task import BFTaskModule, BoundTask, TaskKind, TaskOp
from boundflow.planner.execution_candidate import BackendVariant, OperatorFamily
from boundflow.runtime.crown_ibp import _relu_backward_mode, run_crown_ibp_mlp
from boundflow.runtime.fused_crown import build_fused_crown_runtime_selection
from boundflow.runtime.task_executor import InputSpec
from scripts.benchmark_phase7a_pr12_runtime_pareto import (
    RuntimeWorkload,
    _build_query,
    _conv_attrs,
    _environment,
    _max_abs_diff,
    _max_rel_diff,
    _sha256,
    _workload,
    _write_jsonl,
)

SCHEMA_VERSION = "boundflow.pr12k-cupti-activity/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.pr12k-cupti-activity-manifest/v1"
DEFAULT_BACKENDS = (
    BackendVariant.PYTORCH_EAGER,
    BackendVariant.PYTORCH_STRUCTURED,
    BackendVariant.PYTORCH_CHUNKED,
    BackendVariant.TVM_TIR_UNFUSED,
    BackendVariant.TVM_FUSED_TIR,
)


def _audit(*, probe_counters: bool) -> dict[str, Any]:
    params_path = Path("/proc/driver/nvidia/params")
    params = (
        params_path.read_text(encoding="utf-8", errors="replace")
        if params_path.is_file()
        else ""
    )
    admin_only = next(
        (
            line.split(":", 1)[1].strip()
            for line in params.splitlines()
            if line.startswith("RmProfilingAdminOnly:")
        ),
        "unknown",
    )
    cupti_candidates = (
        Path("/opt/cuda/targets/x86_64-linux/lib/libcupti.so"),
        Path("/opt/cuda/extras/CUPTI/lib64/libcupti.so"),
    )
    cupti_path = next((path for path in cupti_candidates if path.exists()), None)
    ncu_path = shutil.which("ncu")
    ncu_version = None
    counter_probe: dict[str, Any] = {"attempted": False}
    if ncu_path is not None:
        version = subprocess.run(
            [ncu_path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        ncu_version = (version.stdout + version.stderr).strip()
    if probe_counters and ncu_path is not None:
        completed = subprocess.run(
            [
                ncu_path,
                "--section",
                "LaunchStats",
                "--target-processes",
                "all",
                sys.executable,
                "-c",
                "import torch; print(float(torch.ones(16, device='cuda').sum()))",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        output = completed.stdout + completed.stderr
        counter_probe = {
            "attempted": True,
            "returncode": completed.returncode,
            "permission_error": "ERR_NVGPUCTRPERM" in output,
            "output_tail": output[-8000:],
        }
    counters_available = bool(
        ncu_path
        and admin_only != "1"
        and not counter_probe.get("permission_error", False)
    )
    return {
        "ncu": ncu_path,
        "ncu_version": ncu_version,
        "ncu_counter_probe": counter_probe,
        "nsys": shutil.which("nsys"),
        "nvprof": shutil.which("nvprof"),
        "cupti_library": str(cupti_path) if cupti_path is not None else None,
        "rm_profiling_admin_only": admin_only,
        "hardware_counter_sections_available": counters_available,
        "hardware_counter_missing_reason": (
            None
            if counters_available
            else "ncu_missing_and_or_RmProfilingAdminOnly=1_or_ERR_NVGPUCTRPERM"
        ),
        "fallback": (
            "torch.profiler CUPTI activity trace: kernel names/count/time "
            "and CUDA runtime APIs"
        ),
        "forbidden_claims": [
            "SpeedOfLight utilization",
            "MemoryWorkloadAnalysis bandwidth/cache counters",
            "achieved occupancy",
            "scheduler stall reasons",
        ],
    }


def _synthetic_workload(
    *, case_id: str, family: str, domain: int, spec: int, config: dict[str, Any]
) -> RuntimeWorkload:
    return RuntimeWorkload(
        case_id=case_id,
        family=family,
        planner_family=OperatorFamily.CONV2D,
        split_role="profiler_calibration",
        domain=domain,
        spec=spec,
        budget_bytes=1024 * 1024 * 1024,
        boundary_bytes=0,
        expected_regions=1 if family == "conv2d_stride2" else int(config["blocks"]) + 1,
        config=config,
    )


def _profile_suite(split: dict[str, Any]) -> list[RuntimeWorkload]:
    calibration = split["calibration"]
    linear_small = _workload(calibration[0], split_role="profiler_calibration")
    linear_memory = _workload(
        next(
            row
            for row in calibration
            if row.get("case_id") == "linear-memory-sensitive"
        ),
        split_role="profiler_calibration",
    )
    conv_stride1 = _workload(calibration[2], split_role="profiler_calibration")
    mini_resnet = _workload(
        next(
            row
            for row in calibration
            if row.get("case_id") == "mini-resnet-unseen-width"
        ),
        split_role="profiler_calibration",
    )
    conv_s2_config = {
        "case_id": "profile-conv-stride2",
        "family": "conv2d_stride2",
        "domain": 2,
        "spec": 16,
        "channels": 8,
        "height": 16,
        "width": 16,
        "kernel": 3,
        "stride": 2,
    }
    residual_config = {
        "case_id": "profile-residual-two-block",
        "family": "mini_resnet",
        "domain": 2,
        "spec": 16,
        "width": 8,
        "blocks": 2,
    }
    return [
        linear_small,
        linear_memory,
        conv_stride1,
        _synthetic_workload(
            case_id="profile-conv-stride2",
            family="conv2d_stride2",
            domain=2,
            spec=16,
            config=conv_s2_config,
        ),
        _synthetic_workload(
            case_id="profile-residual-two-block",
            family="mini_resnet",
            domain=2,
            spec=16,
            config=residual_config,
        ),
        mini_resnet,
    ]


def _stride2_query(
    workload: RuntimeWorkload, device: torch.device
) -> tuple[BFTaskModule, InputSpec]:
    torch.manual_seed(14000 + sum(ord(char) for char in workload.case_id))
    channels = int(workload.config["channels"])
    height, width = int(workload.config["height"]), int(workload.config["width"])
    output_h, output_w = (height + 1) // 2, (width + 1) // 2
    task = BoundTask(
        task_id=workload.case_id,
        kind=TaskKind.INTERVAL_IBP,
        ops=[
            TaskOp(
                "conv2d",
                "region_affine",
                ["input", "w1", "b1"],
                ["h1"],
                _conv_attrs(stride=2, padding=1),
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
    module = BFTaskModule(
        tasks=[task],
        entry_task_id=workload.case_id,
        bindings={
            "params": {
                "w1": torch.randn(channels, channels, 3, 3, device=device),
                "b1": torch.randn(channels, device=device),
                "wh": torch.randn(
                    workload.spec, channels * output_h * output_w, device=device
                ),
                "bh": torch.randn(workload.spec, device=device),
            }
        },
    )
    center = torch.randn(workload.domain, channels, height, width, device=device)
    return module, InputSpec.linf(value_name="input", center=center, eps=0.03)


def _build_profile_query(
    workload: RuntimeWorkload, device: torch.device
) -> tuple[BFTaskModule, InputSpec]:
    if workload.family == "conv2d_stride2":
        return _stride2_query(workload, device)
    return _build_query(workload, device)


def _call(
    module: BFTaskModule,
    input_spec: InputSpec,
    backend: BackendVariant,
) -> tuple[Callable[[], Any], int]:
    mode: Optional[Literal["dense", "structured"]]
    if backend == BackendVariant.PYTORCH_STRUCTURED:
        mode = "structured"
        selection_backend = BackendVariant.PYTORCH_STRUCTURED.value
    elif backend == BackendVariant.PYTORCH_EAGER:
        mode = "dense"
        selection_backend = BackendVariant.PYTORCH_EAGER.value
    else:
        mode = None
        selection_backend = backend.value
    initial = build_fused_crown_runtime_selection(
        module.get_entry_task().ops, backend=selection_backend
    )
    executor = initial.executor

    def call() -> Any:
        current = build_fused_crown_runtime_selection(
            module.get_entry_task().ops, backend=selection_backend
        )
        context = _relu_backward_mode(mode) if mode is not None else nullcontext()
        with context:
            return run_crown_ibp_mlp(
                module,
                input_spec,
                fused_crown_executor=executor,
                fused_crown_steps=current.steps,
            )

    return call, len(initial.steps)


def _aggregate_events(profiler, *, iterations: int) -> dict[str, Any]:
    kernels: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "device_time_us": 0.0}
    )
    runtime_apis: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "self_cpu_time_us": 0.0}
    )
    for event in profiler.events():
        device_name = getattr(event.device_type, "name", str(event.device_type))
        if device_name == "CUDA" and not event.name.startswith("boundflow_pr12k::"):
            item = kernels[event.name]
            item["count"] += 1
            item["device_time_us"] += float(event.device_time_total)
        elif event.name.startswith(("cudaLaunch", "cudaMalloc", "cudaFree")):
            item = runtime_apis[event.name]
            item["count"] += 1
            item["self_cpu_time_us"] += float(event.self_cpu_time_total)
    kernel_rows: list[dict[str, Any]] = [
        {
            "name": name,
            "count": int(value["count"]),
            "device_time_us": float(value["device_time_us"]),
        }
        for name, value in kernels.items()
    ]
    kernel_rows.sort(key=lambda item: float(item["device_time_us"]), reverse=True)
    launch_rows: list[dict[str, Any]] = [
        {
            "name": name,
            "count": int(value["count"]),
            "self_cpu_time_us": float(value["self_cpu_time_us"]),
        }
        for name, value in runtime_apis.items()
    ]
    launch_rows.sort(key=lambda item: float(item["self_cpu_time_us"]), reverse=True)
    total_kernels = sum(int(item["count"]) for item in kernel_rows)
    total_device_us = sum(float(item["device_time_us"]) for item in kernel_rows)
    vendor_us = sum(
        float(item["device_time_us"])
        for item in kernel_rows
        if any(
            token in str(item["name"]).lower()
            for token in ("cublas", "cudnn", "gemm", "cutlass", "ampere", "volta")
        )
    )
    launch_count = sum(
        int(item["count"])
        for item in launch_rows
        if str(item["name"]).startswith("cudaLaunch")
    )
    launch_cpu_us = sum(
        float(item["self_cpu_time_us"])
        for item in launch_rows
        if str(item["name"]).startswith("cudaLaunch")
    )
    return {
        "iterations": iterations,
        "kernel_launches": total_kernels,
        "kernel_launches_per_query": total_kernels / iterations,
        "device_time_us": total_device_us,
        "device_time_us_per_query": total_device_us / iterations,
        "mean_kernel_device_us": (
            total_device_us / total_kernels if total_kernels else 0
        ),
        "unique_kernel_names": len(kernel_rows),
        "vendor_library_device_time_share": (
            vendor_us / total_device_us if total_device_us else 0
        ),
        "cuda_launch_api_count": launch_count,
        "cuda_launch_api_count_per_query": launch_count / iterations,
        "cuda_launch_self_cpu_us": launch_cpu_us,
        "cuda_launch_self_cpu_us_per_query": launch_cpu_us / iterations,
        "top_kernels": kernel_rows[:20],
        "cuda_runtime_apis": launch_rows,
    }


def _profile_one(
    workload: RuntimeWorkload,
    backend: BackendVariant,
    trace_dir: Path,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    module, input_spec = _build_profile_query(workload, device)
    expected = run_crown_ibp_mlp(module, input_spec)
    call, planned_regions = _call(module, input_spec, backend)
    for _ in range(warmup):
        call()
    torch.cuda.synchronize(device)
    wall_started = time.perf_counter_ns()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=False,
        with_stack=False,
    ) as profiler:
        actual = None
        with record_function(f"boundflow_pr12k::{workload.case_id}::{backend.value}"):
            for _ in range(iterations):
                actual = call()
        torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter_ns() - wall_started) / 1e6
    assert actual is not None
    trace_path = trace_dir / f"{workload.case_id}__{backend.value}.json.gz"
    profiler.export_chrome_trace(str(trace_path))
    correctness = {
        "max_abs_diff": _max_abs_diff(actual, expected),
        "max_rel_diff": _max_rel_diff(actual, expected),
        "finite": bool(
            torch.isfinite(actual.lower).all() and torch.isfinite(actual.upper).all()
        ),
        "lower_le_upper": bool((actual.lower <= actual.upper).all()),
        "allclose": bool(
            torch.allclose(actual.lower, expected.lower, rtol=2e-4, atol=2e-4)
            and torch.allclose(actual.upper, expected.upper, rtol=2e-4, atol=2e-4)
        ),
    }
    correct = bool(
        correctness["finite"]
        and correctness["lower_le_upper"]
        and correctness["allclose"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if correct else "fail",
        "workload": {
            "case_id": workload.case_id,
            "family": workload.family,
            "domain": workload.domain,
            "spec": workload.spec,
            "config": workload.config,
        },
        "candidate": {"backend": backend.value},
        "planned_fused_regions": planned_regions,
        "profile_wall_ms": wall_ms,
        "activity": _aggregate_events(profiler, iterations=iterations),
        "correctness": correctness,
        "trace": str(trace_path.name),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Profile the frozen six-workload/five-backend calibration matrix."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--backends", default=",".join(backend.value for backend in DEFAULT_BACKENDS)
    )
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--skip-counter-probe", action="store_true")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.warmup, args.iterations) <= 0:
        parser.error("warmup/iterations must be positive")
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    workloads = _profile_suite(split)
    requested = {item for item in args.case_ids.split(",") if item}
    if requested:
        workloads = [
            workload for workload in workloads if workload.case_id in requested
        ]
        missing = requested - {workload.case_id for workload in workloads}
        if missing:
            parser.error(f"unknown profiler case ids: {sorted(missing)}")
    try:
        backends = tuple(
            BackendVariant(item) for item in args.backends.split(",") if item
        )
    except ValueError as error:
        parser.error(str(error))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    trace_dir = args.out_dir / "traces"
    trace_dir.mkdir()
    audit = _audit(probe_counters=not args.skip_counter_probe)
    audit_path = args.out_dir / "profiler_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for workload in workloads:
        for backend in backends:
            try:
                row = _profile_one(
                    workload,
                    backend,
                    trace_dir,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            except Exception as error:
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "workload": {
                        "case_id": workload.case_id,
                        "family": workload.family,
                        "domain": workload.domain,
                        "spec": workload.spec,
                        "config": workload.config,
                    },
                    "candidate": {"backend": backend.value},
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            rows.append(row)
    raw_path = args.out_dir / "raw.jsonl"
    _write_jsonl(raw_path, rows)
    trace_outputs = {
        str(path.relative_to(args.out_dir)): _sha256(path)
        for path in sorted(trace_dir.glob("*.json.gz"))
    }
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "split_id": split["split_id"],
        "split_sha256": _sha256(args.split_file),
        "environment": _environment(),
        "audit": audit,
        "case_ids": [workload.case_id for workload in workloads],
        "backends": [backend.value for backend in backends],
        "measurement": {"warmup": args.warmup, "iterations": args.iterations},
        "row_count": len(rows),
        "status_counts": status_counts,
        "outputs": {
            "raw.jsonl": _sha256(raw_path),
            "profiler_audit.json": _sha256(audit_path),
            **trace_outputs,
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
