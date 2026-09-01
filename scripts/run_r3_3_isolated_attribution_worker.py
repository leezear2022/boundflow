#!/usr/bin/env python3
"""Run one fresh diagnostic-only R3-3 isolated attribution worker."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=too-many-statements,missing-function-docstring,duplicate-code
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Iterator

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.runtime import fsg4_b4b2_sparse_linear_tir as sparse_linear
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.r3_3_active_beta_timing import (
    PreparedR3ActiveBetaTimingV1,
    R3ActiveBetaExecutionV1,
    compare_r3_active_beta_executions_v1,
    cuda_event_wrapper_ms_v1,
)
from boundflow.runtime.r3_3_isolated_attribution import (
    canonical_hash,
    derive_ledger,
    extract_profiler_events,
)

CAPTURE = ROOT / "artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1"
SCHEMA = "boundflow.r3-3-isolated-attribution-worker/v1"
WARMUP_COUNT = 10
SAMPLE_COUNT = 30
CALIBRATION_ELEMENT_COUNT = 1 << 20


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(run_index: int):  # type: ignore[no-untyped-def]
    path = CAPTURE / f"run_{run_index:02d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return path, production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][0]
    )


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
        raise RuntimeError("R3-3 attribution GPU inventory differs")
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != 7:
        raise RuntimeError("R3-3 attribution GPU snapshot differs")
    return {
        "name": values[0],
        "temperature_celsius": int(values[1]),
        "power_draw_watts": float(values[2]),
        "graphics_clock_mhz": int(values[3]),
        "memory_clock_mhz": int(values[4]),
        "enforced_power_limit_watts": float(values[5]),
        "driver_version": values[6],
    }


@contextmanager
def _executor_markers(
    executor: sparse_linear._SparseLinearTIRExecutor,
) -> Iterator[None]:
    from torch.profiler import record_function

    original_launch = executor._launch
    original_forward = executor.forward
    original_backward = executor.backward

    def launch(symbol, sources, outputs):  # type: ignore[no-untyped-def]
        phase = (
            "forward-ffi"
            if symbol == executor.template.forward_symbol
            else "backward-ffi"
        )
        with record_function(f"boundflow::r33attr::{phase}"):
            return original_launch(symbol, sources, outputs)

    def forward(tensors):  # type: ignore[no-untyped-def]
        with record_function("boundflow::r33attr::output-allocation-forward"):
            return original_forward(tensors)

    def backward(output_a_gradient, output_bias_gradient):  # type: ignore[no-untyped-def]
        with record_function("boundflow::r33attr::output-allocation-backward"):
            return original_backward(output_a_gradient, output_bias_gradient)

    executor._launch = launch  # type: ignore[method-assign]
    executor.forward = forward  # type: ignore[method-assign]
    executor.backward = backward  # type: ignore[method-assign]
    try:
        yield
    finally:
        executor._launch = original_launch  # type: ignore[method-assign]
        executor.forward = original_forward  # type: ignore[method-assign]
        executor.backward = original_backward  # type: ignore[method-assign]


def _diagnostic_candidate(
    prepared: PreparedR3ActiveBetaTimingV1,
) -> R3ActiveBetaExecutionV1:
    from torch.profiler import record_function

    with record_function("boundflow::r33attr::prepare-executor"):
        executor = sparse_linear._SparseLinearTIRExecutor(
            prepared.template, prepared.schedule, prepared.cache
        )
        if executor.cache_event != "hit":
            raise RuntimeError("R3-3 attribution candidate includes cache miss")
        executor.prime(prepared.tensors)
    with _executor_markers(executor):
        with record_function("boundflow::r33attr::autograd-apply"):
            output_a, output_bias = sparse_linear._SparseLinearTIRFunction.apply(
                prepared.tensors.incoming_lower_a,
                prepared.tensors.preactivation_lower,
                prepared.tensors.preactivation_upper,
                prepared.tensors.compressed_alpha,
                prepared.tensors.compressed_beta,
                prepared.tensors.incoming_lower_bias,
                prepared.tensors.operator_weight,
                prepared.tensors.operator_bias,
                executor,
            )
        with record_function("boundflow::r33attr::autograd-grad"):
            alpha_gradient, beta_gradient = torch.autograd.grad(
                (output_a, output_bias),
                (
                    prepared.tensors.compressed_alpha,
                    prepared.tensors.compressed_beta,
                ),
                grad_outputs=(
                    prepared.tensors.output_lower_a_gradient,
                    prepared.tensors.output_bias_gradient,
                ),
                create_graph=False,
                retain_graph=False,
            )
    if (
        executor.forward_launch_count != 1
        or executor.backward_launch_count != 1
        or executor.fallback_count != 0
        or executor.eager_backward_count != 0
    ):
        raise RuntimeError("R3-3 attribution launch receipt differs")
    return R3ActiveBetaExecutionV1(
        output_lower_a=output_a,
        output_bias=output_bias,
        compressed_alpha_gradient=alpha_gradient,
        compressed_beta_gradient=beta_gradient,
    )


def _run(run_index: int) -> dict[str, object]:
    from torch.profiler import profile, ProfilerActivity, record_function

    if not torch.cuda.is_available():
        raise RuntimeError("R3-3 attribution worker requires CUDA")
    capture_path, capture = _capture(run_index)
    prepared = PreparedR3ActiveBetaTimingV1(capture)
    parity = compare_r3_active_beta_executions_v1(
        prepared.baseline_once(), prepared.candidate_once()
    )
    if not parity.allclose or not parity.sign_exact:
        raise RuntimeError("R3-3 attribution parity differs")
    for _ in range(WARMUP_COUNT):
        prepared.candidate_once()
    torch.cuda.synchronize()
    samples_ns = [
        round(cuda_event_wrapper_ms_v1(prepared.candidate_once) * 1_000_000.0)
        for _ in range(SAMPLE_COUNT)
    ]
    median_ns = round(statistics.median(samples_ns))
    gpu_before = _gpu_snapshot()
    calibration_start = torch.cuda.Event(enable_timing=True)
    calibration_end = torch.cuda.Event(enable_timing=True)
    wrapper_start = torch.cuda.Event(enable_timing=True)
    wrapper_end = torch.cuda.Event(enable_timing=True)
    calibration_tensor = torch.ones(
        CALIBRATION_ELEMENT_COUNT, device="cuda:0", dtype=torch.float32
    )
    calibration_tensor.add_(1.0)
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as profiler:
        with record_function("boundflow::r33attr::calibration"):
            calibration_start.record()
            calibration_tensor.add_(1.0)
            calibration_end.record()
            calibration_end.synchronize()
        with record_function("boundflow::r33attr::wrapper"):
            wrapper_start.record()
            result = _diagnostic_candidate(prepared)
            wrapper_end.record()
            wrapper_end.synchronize()
    calibration_ns = round(
        calibration_start.elapsed_time(calibration_end) * 1_000_000.0
    )
    profiled_ns = round(wrapper_start.elapsed_time(wrapper_end) * 1_000_000.0)
    events = extract_profiler_events(profiler.events())
    ledger = derive_ledger(
        events,
        unprofiled_median_ns=median_ns,
        profiled_cuda_event_ns=profiled_ns,
        calibration_cuda_event_ns=calibration_ns,
    )
    gpu_after = _gpu_snapshot()
    payload: dict[str, object] = {
        "schema_version": SCHEMA,
        "run_index": run_index,
        "capture_sha256": _file_hash(capture_path),
        "warmup_count": WARMUP_COUNT,
        "sample_count": SAMPLE_COUNT,
        "latency_ns": samples_ns,
        "median_latency_ns": median_ns,
        "profiled_cuda_event_ns": profiled_ns,
        "calibration_cuda_event_ns": calibration_ns,
        "calibration_element_count": CALIBRATION_ELEMENT_COUNT,
        "parity": parity.to_dict(),
        "template_hash": prepared.template.stable_hash(),
        "schedule_hash": prepared.schedule.stable_hash(prepared.template),
        "module_receipt_hash": prepared.module_receipt.stable_hash(
            prepared.template, prepared.schedule
        ),
        "events": [event.to_dict() for event in events],
        "event_hash": canonical_hash([event.to_dict() for event in events]),
        "ledger": ledger,
        "output_shapes": [list(tensor.shape) for tensor in result.tensors],
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "environment": {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "profiler": "torch-profiler-cupti",
        },
        "performance_claimed": False,
    }
    payload["worker_hash"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.run_index not in range(5):
        raise ValueError("R3-3 attribution run index differs")
    payload = _run(args.run_index)
    args.result.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    ledger = payload["ledger"]
    if not isinstance(ledger, dict):
        raise TypeError("R3-3 attribution worker ledger differs")
    print(
        f"R3-3 attribution run={args.run_index} "
        f"median_ns={payload['median_latency_ns']} "
        f"admitted={ledger['attribution_admitted']} "
        f"unexplained={ledger['bucket_share']['unexplained']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
