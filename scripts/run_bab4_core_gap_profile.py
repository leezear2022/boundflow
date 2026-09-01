#!/usr/bin/env python3
"""Profile the live BAB4 optimizer without promoting profiler timings to claims."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,protected-access,too-many-locals
# pylint: disable=too-many-statements,missing-function-docstring
# pylint: disable=wrong-import-position,import-outside-toplevel
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, cast

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime.bab_four_segment_optimizer import (
    PreparedBabFourSegmentOptimizerV1,
)
from boundflow.runtime.r3_d0_microphysics_attribution import (
    canonical_hash,
    derive_worker_ledger,
    extract_torch_profiler_events,
)
from scripts import run_asplos27_s4_same_solver_worker as live_worker
from scripts import run_fsg3_same_solver_timing as fsg3_timing

SCHEMA = "boundflow.bab4-core-gap-profile/v1"
FORMAL_ARTIFACT = ROOT / "artifacts/bab4-same-solver-five-fresh/resnet2b-prop0-v1"
SEGMENTS = ("terminal", "residual", "projection", "input")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"BAB4 core-gap {label} differs")
    return cast(Mapping[str, Any], value)


def unprofiled_optimizer_samples_ns(
    artifact: Path = FORMAL_ARTIFACT,
) -> tuple[int, ...]:
    """Load the five admitted unprofiled optimizer spans from frozen raw."""

    rows: list[int] = []
    for path in sorted(artifact.glob("raw/pair-*/BAB4/worker.json")):
        worker = _mapping(json.loads(path.read_text(encoding="utf-8")), "worker")
        receipts = worker.get("s4_exact_call_receipts")
        if not isinstance(receipts, list) or len(receipts) != 1:
            raise ValueError("BAB4 core-gap exact-call receipt cardinality differs")
        receipt = _mapping(receipts[0], "receipt")
        value = receipt.get("optimizer_ns")
        if (
            receipt.get("schema_version") != "boundflow.bab-four-segment-exact-call/v1"
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or receipt.get("compiled_forward_launch_count") != 76
            or receipt.get("compiled_backward_launch_count") != 36
            or receipt.get("fallback_count") != 0
        ):
            raise ValueError("BAB4 core-gap frozen optimizer receipt differs")
        rows.append(value)
    if len(rows) != 5:
        raise ValueError("BAB4 core-gap frozen sample count differs")
    return tuple(rows)


@contextmanager
def _patch(target: object, name: str, replacement: object) -> Iterator[None]:
    original = getattr(target, name)
    setattr(target, name, replacement)
    try:
        yield
    finally:
        setattr(target, name, original)


@contextmanager
def _segment_markers(optimizer: PreparedBabFourSegmentOptimizerV1) -> Iterator[None]:
    from torch.profiler import record_function

    executors = {
        "terminal": optimizer.owner.terminal_executor,
        "residual": optimizer.owner.residual_executor,
        "projection": optimizer.owner.projection_executor,
        "input": optimizer.owner.input_executor,
    }
    if tuple(executors) != SEGMENTS or any(
        value is None for value in executors.values()
    ):
        raise ValueError("BAB4 core-gap compiled segment inventory differs")
    with ExitStack() as stack:
        for segment, executor in executors.items():
            assert executor is not None
            original_launch = executor._launch

            def make_launch(label: str, target: Any, original: Any) -> Any:
                def launch(symbol: str, sources: Any, outputs: Any) -> Any:
                    direction = (
                        "forward"
                        if symbol == target.template.forward_symbol
                        else "backward"
                    )
                    marker = f"boundflow::r3d0::segment.{label}.{direction}::{label}-{direction}"
                    with record_function(marker):
                        return original(symbol, sources, outputs)

                return launch

            stack.enter_context(
                _patch(
                    executor, "_launch", make_launch(segment, executor, original_launch)
                )
            )
        yield


@contextmanager
def _optimizer_markers(optimizer: PreparedBabFourSegmentOptimizerV1) -> Iterator[None]:
    from torch.profiler import record_function

    original_evaluate = optimizer.owner.evaluate
    original_grad = torch.autograd.grad
    original_adam_step = torch.optim.Adam.step
    original_zero_grad = torch.optim.Optimizer.zero_grad
    original_scheduler_step = torch.optim.lr_scheduler.ExponentialLR.step
    grad_depth = 0

    def evaluate(dynamic: Any) -> Any:
        with record_function("boundflow::r3d0::region.forward::full-region-forward"):
            return original_evaluate(dynamic)

    def grad(*args: Any, **kwargs: Any) -> Any:
        nonlocal grad_depth
        depth = grad_depth
        grad_depth += 1
        try:
            label = "outer" if depth == 0 else "recompute"
            with record_function(
                f"boundflow::r3d0::autograd.{label}::autograd-{label}"
            ):
                return original_grad(*args, **kwargs)
        finally:
            grad_depth -= 1

    def adam_step(instance: Any, *args: Any, **kwargs: Any) -> Any:
        with record_function("boundflow::r3d0::optimizer.adam::adam-step"):
            return original_adam_step(instance, *args, **kwargs)

    def zero_grad(instance: Any, *args: Any, **kwargs: Any) -> Any:
        with record_function("boundflow::r3d0::optimizer.zero::zero-grad"):
            return original_zero_grad(instance, *args, **kwargs)

    def scheduler_step(instance: Any, *args: Any, **kwargs: Any) -> Any:
        with record_function("boundflow::r3d0::optimizer.scheduler::scheduler-step"):
            return original_scheduler_step(instance, *args, **kwargs)

    with (
        _patch(optimizer.owner, "evaluate", evaluate),
        _patch(torch.autograd, "grad", grad),
        _patch(torch.optim.Adam, "step", adam_step),
        _patch(torch.optim.Optimizer, "zero_grad", zero_grad),
        _patch(torch.optim.lr_scheduler.ExponentialLR, "step", scheduler_step),
        _segment_markers(optimizer),
    ):
        yield


def _profile_live_worker(args: argparse.Namespace) -> dict[str, object]:
    from torch.profiler import profile, ProfilerActivity, record_function

    samples = unprofiled_optimizer_samples_ns(args.formal_artifact)
    unprofiled_median = round(statistics.median(samples))
    profile_payload: dict[str, object] = {}
    run_count = 0
    original_run = PreparedBabFourSegmentOptimizerV1.run

    def profiled_run(
        optimizer: PreparedBabFourSegmentOptimizerV1, stream: torch.cuda.Stream
    ) -> Any:
        nonlocal run_count, profile_payload
        run_count += 1
        if run_count == 1:
            return original_run(optimizer, stream)
        if run_count != 2 or profile_payload:
            raise ValueError("BAB4 core-gap optimizer run cardinality differs")
        stream.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with profile(
            activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
            record_shapes=False,
            with_stack=False,
        ) as profiler:
            host_started = time.perf_counter_ns()
            with record_function("boundflow::r3d0::wrapper::bab4-optimizer"):
                with _optimizer_markers(optimizer):
                    start.record(stream)
                    result = original_run(optimizer, stream)
                    end.record(stream)
                    end.synchronize()
            host_wall_ns = time.perf_counter_ns() - host_started
        cuda_event_ns = round(start.elapsed_time(end) * 1_000_000.0)
        events = extract_torch_profiler_events(profiler.events(), mode="candidate")
        ledger = derive_worker_ledger(
            events,
            mode="candidate",
            unprofiled_median_ns=unprofiled_median,
            profiled_host_wall_ns=host_wall_ns,
            cuda_event_elapsed_ns=cuda_event_ns,
        )
        profile_payload = {
            "unprofiled_optimizer_samples_ns": list(samples),
            "unprofiled_optimizer_median_ns": unprofiled_median,
            "profiled_host_wall_ns": host_wall_ns,
            "cuda_event_elapsed_ns": cuda_event_ns,
            "events": [event.to_dict() for event in events],
            "event_hash": canonical_hash([event.to_dict() for event in events]),
            "ledger": ledger,
        }
        return result

    worker_result = args.output / "worker.json"
    worker_args = argparse.Namespace(
        configuration="BAB4",
        mode="control",
        run_id="bab4-core-gap-profile-live",
        block_index=0,
        sequence_position=0,
        benchmark_root=args.benchmark_root,
        abcrown_root=args.abcrown_root,
        model=args.model,
        property=args.property,
        result=worker_result,
        attribute_root_incomplete=False,
    )
    old_window = fsg3_timing.POST_PREPARE_ENVIRONMENT_WINDOW
    fsg3_timing.POST_PREPARE_ENVIRONMENT_WINDOW = True
    try:
        with _patch(PreparedBabFourSegmentOptimizerV1, "run", profiled_run):
            live_worker._worker(worker_args)
    finally:
        fsg3_timing.POST_PREPARE_ENVIRONMENT_WINDOW = old_window
    if run_count != 2 or not profile_payload or not worker_result.is_file():
        raise ValueError("BAB4 core-gap live profile did not close")
    worker = _mapping(json.loads(worker_result.read_text(encoding="utf-8")), "worker")
    run = _mapping(worker.get("run"), "run")
    metrics = _mapping(run.get("metrics"), "metrics")
    receipt = _mapping(worker.get("s4_exact_call_receipts", [None])[0], "receipt")
    if receipt.get("evaluation_count") != 10:
        raise ValueError("BAB4 core-gap evaluation count differs")
    result_payload: dict[str, object] = {
        "schema_version": SCHEMA,
        "source_git_head": _git_head(),
        "worker_sha256": _file_hash(worker_result),
        "formal_artifact_relative_path": str(args.formal_artifact.relative_to(ROOT)),
        "formal_manifest_sha256": _file_hash(args.formal_artifact / "manifest.json"),
        "optimizer_receipt": {
            key: receipt[key]
            for key in (
                "assets_hash",
                "production_plan_hash",
                "evaluation_count",
                "mutation_count",
                "compiled_forward_launch_count",
                "compiled_backward_launch_count",
                "fallback_count",
            )
        },
        "same_solver_metrics": {
            "core_wall_ns": metrics["core_wall_ns"],
            "query_wall_ns": metrics["query_wall_ns"],
        },
        "profile": profile_payload,
        "environment": {
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "profiler": "torch-profiler-cupti",
        },
        "profile_timing_claimed": False,
        "performance_claimed": False,
    }
    result_payload["profile_hash"] = canonical_hash(result_payload)
    return result_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--formal-artifact", type=Path, default=FORMAL_ARTIFACT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for name in (
        "benchmark_root",
        "abcrown_root",
        "model",
        "property",
        "formal_artifact",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        raise FileExistsError("BAB4 core-gap output already exists")
    args.output.mkdir(parents=True)
    payload = _profile_live_worker(args)
    output = args.output / "profile.json"
    output.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    ledger = cast(
        Mapping[str, Any], cast(Mapping[str, Any], payload["profile"])["ledger"]
    )
    print(
        json.dumps(
            {
                "unprofiled_optimizer_median_ns": cast(
                    Mapping[str, Any], payload["profile"]
                )["unprofiled_optimizer_median_ns"],
                "profiled_host_wall_ns": cast(Mapping[str, Any], payload["profile"])[
                    "profiled_host_wall_ns"
                ],
                "cuda_event_elapsed_ns": cast(Mapping[str, Any], payload["profile"])[
                    "cuda_event_elapsed_ns"
                ],
                "kernel_count": ledger["kernel_count"],
                "calibration_admitted": ledger["calibration_admitted"],
                "performance_claimed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
