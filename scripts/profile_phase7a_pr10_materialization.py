#!/usr/bin/env python
"""Characterize PR-10 materialization barriers with trace-on/off separation."""

# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.alpha_beta_crown import run_alpha_beta_crown_mlp
from boundflow.runtime.alpha_crown import run_alpha_crown_mlp
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp, run_crown_ibp_mlp
from boundflow.runtime.materialization import (
    TRACE_SCHEMA_VERSION,
    trace_materializations,
)
from boundflow.runtime.task_executor import InputSpec

PROFILE_SCHEMA_VERSION = "boundflow.pr10-profile/v1"
METHODS = ("CROWN", "alpha-CROWN", "alpha-beta-CROWN")
WORKLOADS = (
    "mlp_chain",
    "cnn_chain",
    "residual_block",
    "add_concat_dag",
    "mini_resnet",
)


class MLPChain(nn.Module):
    """Two-ReLU MLP mechanism workload."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the MLP."""

        return self.layers(x)


class CNNChain(nn.Module):
    """Two-ReLU stride-convolution mechanism workload."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.head = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the CNN chain."""

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.head(x.flatten(1))


class ResidualBlockNet(nn.Module):
    """Single residual-block mechanism workload."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.head = nn.Linear(8 * 16 * 16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the residual block and dense head."""

        skip = torch.relu(self.stem(x))
        branch = torch.relu(self.conv1(skip))
        branch = self.conv2(branch)
        return self.head(torch.relu(branch + skip).flatten(1))


class AddConcatDAG(nn.Module):
    """DAG mechanism workload containing both add and concat."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.right = nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.head = nn.Linear(8 * 16 * 16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate both branches, merge and concatenate."""

        left = torch.relu(self.left(x))
        right = torch.relu(self.right(x))
        merged = torch.relu(left + right)
        return self.head(torch.cat((merged, left), dim=1).flatten(1))


class BasicBlock(nn.Module):
    """Minimal batch-norm-free residual block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate one residual block."""

        return torch.relu(self.conv2(torch.relu(self.conv1(x))) + x)


class MiniResNet(nn.Module):
    """Three-BasicBlock non-toy structural workload."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.block1 = BasicBlock(8)
        self.block2 = BasicBlock(8)
        self.block3 = BasicBlock(8)
        self.head = nn.Linear(8 * 16 * 16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the mini-ResNet."""

        x = torch.relu(self.stem(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x.flatten(1))


@dataclass(frozen=True)
class Workload:
    """Model, shape and evidence tier used by the profile matrix."""

    name: str
    model: nn.Module
    input_shape: tuple[int, ...]
    tier: str


def _make_workload(name: str, device: torch.device) -> Workload:
    torch.manual_seed(0)
    builders: dict[str, tuple[Callable[[], nn.Module], tuple[int, ...], str]] = {
        "mlp_chain": (MLPChain, (64,), "mechanism"),
        "cnn_chain": (CNNChain, (3, 32, 32), "mechanism"),
        "residual_block": (ResidualBlockNet, (3, 32, 32), "mechanism"),
        "add_concat_dag": (AddConcatDAG, (3, 32, 32), "mechanism"),
        "mini_resnet": (MiniResNet, (3, 32, 32), "non_toy_structure"),
    }
    if name not in builders:
        raise ValueError(f"unknown workload: {name}")
    builder, input_shape, tier = builders[name]
    return Workload(
        name=name, model=builder().eval().to(device), input_shape=input_shape, tier=tier
    )


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel()) * int(value.element_size())


def _state_dict_bytes(values: dict[str, torch.Tensor]) -> int:
    return sum(_tensor_bytes(value) for value in values.values())


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fixed_split_state(relu_pre: dict[str, object]) -> dict[str, torch.Tensor]:
    split_state: dict[str, torch.Tensor] = {}
    for name, state in relu_pre.items():
        lower = state.lower
        upper = state.upper
        midpoint = (lower + upper) * 0.5
        split = torch.zeros_like(lower, dtype=torch.int8)
        for batch_index in range(int(lower.shape[0])):
            ambiguous = ((lower[batch_index] < 0) & (upper[batch_index] > 0)).reshape(-1)
            candidates = ambiguous.nonzero()
            if int(candidates.numel()) == 0:
                continue
            index = int(candidates[0].item())
            midpoint_value = float(midpoint[batch_index].reshape(-1)[index].item())
            split[batch_index].reshape(-1)[index] = 1 if midpoint_value >= 0 else -1
        split_state[name] = split
    return split_state


def _run_method(  # pylint: disable=too-many-arguments
    method: str,
    module: object,
    spec: InputSpec,
    linear_spec: torch.Tensor,
    split_state: dict[str, torch.Tensor],
    *,
    optimization_steps: int,
) -> tuple[object, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if method == "CROWN":
        return run_crown_ibp_mlp(module, spec, linear_spec_C=linear_spec), {}, {}
    if method == "alpha-CROWN":
        bounds, alpha, _stats = run_alpha_crown_mlp(
            module,
            spec,
            linear_spec_C=linear_spec,
            steps=optimization_steps,
        )
        return bounds, alpha.alpha_by_relu_input, {}
    if method == "alpha-beta-CROWN":
        bounds, alpha, beta, _stats = run_alpha_beta_crown_mlp(
            module,
            spec,
            linear_spec_C=linear_spec,
            relu_split_state=split_state,
            steps=optimization_steps,
            beta_init=0.1,
            per_batch_params=int(spec.center.shape[0]) > 1,
        )
        return bounds, alpha.alpha_by_relu_input, beta.beta_by_relu_input
    raise ValueError(f"unknown method: {method}")


def _measure_trace_off(
    function: Callable[
        [], tuple[object, dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    for _ in range(warmup):
        warmup_result = function()
        del warmup_result
    _sync(device)
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        timing_result = function()
        _sync(device)
        timings.append((time.perf_counter() - start) * 1000.0)
        del timing_result

    peak_allocated: int | None = None
    peak_reserved: int | None = None
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        memory_result = function()
        _sync(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        del memory_result
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "trace_enabled": False,
        "warmup": warmup,
        "repeats": repeats,
        "latency_ms_median": statistics.median(timings),
        "latency_ms_p90": _percentile(timings, 0.90),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "peak_measurement_repeats": 1 if device.type == "cuda" else 0,
        "allocator_cache_cleared_before_peak": device.type == "cuda",
    }


def _profile_query(  # pylint: disable=too-many-arguments,too-many-locals
    workload: Workload,
    module: object,
    *,
    method: str,
    spec_size: int,
    domain_batch: int,
    device: torch.device,
    run_id: str,
    warmup: int,
    repeats: int,
    optimization_steps: int,
) -> dict[str, object]:
    query_id = f"{workload.name}:{method}:s{spec_size}:d{domain_batch}"
    base = {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "status": "ok",
        "error": None,
        "run_id": run_id,
        "query_id": query_id,
        "workload": {"name": workload.name, "tier": workload.tier},
        "method": method,
        "spec_batch": spec_size,
        "domain_batch": domain_batch,
        "domain_source": "fixed_batch_replay",
        "device": str(device),
    }
    try:
        torch.manual_seed(1)
        center = torch.randn((domain_batch, *workload.input_shape), device=device)
        spec = InputSpec.linf(
            value_name=module.get_entry_task().input_values[0], center=center, eps=0.03
        )
        linear_spec = torch.randn((domain_batch, spec_size, 10), device=device)
        interval_env, relu_pre = _forward_ibp_trace_mlp(module, spec)
        split_state = _fixed_split_state(relu_pre)
        intermediate_bytes = sum(
            _tensor_bytes(state.lower) + _tensor_bytes(state.upper)
            for state in interval_env.values()
        )
        params = module.bindings.get("params", {})
        weight_bytes = sum(
            _tensor_bytes(value) for value in params.values() if torch.is_tensor(value)
        )

        def _invoke():
            return _run_method(
                method,
                module,
                spec,
                linear_spec,
                split_state,
                optimization_steps=optimization_steps,
            )

        timing = _measure_trace_off(
            _invoke, device=device, warmup=warmup, repeats=repeats
        )
        with trace_materializations(
            run_id=run_id,
            query_id=query_id,
            bound_method=method,
            solver_phase="backward",
            spec_batch=spec_size,
            domain_batch=domain_batch,
            capture_cuda_memory=device.type == "cuda",
        ) as trace:
            bounds, alpha_state, beta_state = _invoke()
            trace.record_state_bytes(
                "alpha_state_bytes", _state_dict_bytes(alpha_state)
            )
            trace.record_state_bytes("beta_state_bytes", _state_dict_bytes(beta_state))
            trace.record_state_bytes("intermediate_bound_bytes", intermediate_bytes)
            trace.record_state_bytes("weight_bytes", weight_bytes)
            trace.record_state_bytes("operator_state_bytes", 0)

        finite = bool(
            torch.isfinite(bounds.lower).all() and torch.isfinite(bounds.upper).all()
        )
        ordered = bool((bounds.lower <= bounds.upper + 1e-6).all())
        base.update(
            {
                "timing_trace_off": timing,
                "trace_on": trace.to_record(),
                "correctness": {"finite": finite, "lower_le_upper": ordered},
            }
        )
        if not finite or not ordered:
            base["status"] = "fail"
            base["error"] = {
                "type": "CorrectnessGate",
                "message": str(base["correctness"]),
            }
    except torch.cuda.OutOfMemoryError as error:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        base["status"] = "oom"
        base["error"] = {"type": type(error).__name__, "message": str(error)}
    except Exception as error:  # pylint: disable=broad-exception-caught
        rendered = traceback.format_exc()
        base["status"] = "fail"
        base["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        }
    return base


def _parse_csv_list(value: str, *, allowed: Iterable[str]) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported values: {invalid}")
    return values


def _parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _flatten_row(row: dict[str, object]) -> dict[str, object]:
    trace = row.get("trace_on") or {}
    materialization = trace.get("materialization") or {}
    timing = row.get("timing_trace_off") or {}
    workload = row.get("workload") or {}
    return {
        "status": row.get("status"),
        "run_id": row.get("run_id"),
        "query_id": row.get("query_id"),
        "workload": workload.get("name"),
        "tier": workload.get("tier"),
        "method": row.get("method"),
        "spec_batch": row.get("spec_batch"),
        "domain_batch": row.get("domain_batch"),
        "event_count": materialization.get("event_count"),
        "logical_materialized_bytes": materialization.get("logical_materialized_bytes"),
        "max_event_logical_bytes": max(
            (event.get("logical_bytes", 0) for event in trace.get("events", [])),
            default=0,
        ),
        "latency_ms_median_trace_off": timing.get("latency_ms_median"),
        "latency_ms_p90_trace_off": timing.get("latency_ms_p90"),
        "peak_cuda_allocated_bytes_trace_off": timing.get("peak_cuda_allocated_bytes"),
        "peak_cuda_reserved_bytes_trace_off": timing.get("peak_cuda_reserved_bytes"),
    }


def _write_outputs(
    out_dir: Path, rows: list[dict[str, object]], command: Sequence[str]
) -> None:
    profile_dir = out_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    raw_path = profile_dir / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    csv_path = profile_dir / "normalized.csv"
    flat = [_flatten_row(row) for row in rows]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(flat[0]) if flat else ["status"]
        )
        writer.writeheader()
        writer.writerows(flat)
    manifest = {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": list(command),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rows": len(rows),
        "status_counts": {
            status: sum(row.get("status") == status for row in rows)
            for status in ("ok", "fail", "oom", "unsupported")
        },
        "outputs": {
            "raw.jsonl": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "normalized.csv": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        },
    }
    (profile_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:  # pylint: disable=too-many-locals
    """Run the selected profile matrix and write JSONL/CSV/manifest evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out-root", default="artifacts/phase7a-pr10")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--spec-sizes", default="1,9")
    parser.add_argument("--domain-batches", default="1,8")
    parser.add_argument("--optimization-steps", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    workloads = _parse_csv_list(args.workloads, allowed=WORKLOADS)
    methods = _parse_csv_list(args.methods, allowed=METHODS)
    spec_sizes = _parse_int_list(args.spec_sizes)
    domain_batches = _parse_int_list(args.domain_batches)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.optimization_steps < 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error(
            "optimization-steps/warmup must be non-negative and repeats must be positive"
        )
    device = torch.device(device_name)
    torch.set_num_threads(1)
    run_id = args.run_id or f"{int(time.time())}-{os.getpid()}"
    rows: list[dict[str, object]] = []
    for workload_name in workloads:
        workload = _make_workload(workload_name, device)
        dummy = torch.zeros((1, *workload.input_shape), device=device)
        program = import_torch(
            workload.model, (dummy,), export_mode="export", normalize=True
        )
        module = plan_interval_ibp_v0(program)
        for method in methods:
            for spec_size in spec_sizes:
                for domain_batch in domain_batches:
                    rows.append(
                        _profile_query(
                            workload,
                            module,
                            method=method,
                            spec_size=spec_size,
                            domain_batch=domain_batch,
                            device=device,
                            run_id=run_id,
                            warmup=args.warmup,
                            repeats=args.repeats,
                            optimization_steps=args.optimization_steps,
                        )
                    )
    out_dir = Path(args.out_root) / run_id
    effective_argv = list(sys.argv if argv is None else [Path(sys.argv[0]).name, *argv])
    _write_outputs(out_dir, rows, effective_argv)
    return (
        1 if any(row.get("status") not in {"ok", "unsupported"} for row in rows) else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
