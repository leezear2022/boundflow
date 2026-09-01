#!/usr/bin/env python3
"""Calibrate fixed 64/128/256-thread D1-B serial-reduction schedules."""

# mypy: disable-error-code=import-untyped
# pylint: disable=too-many-locals,too-many-statements,protected-access,import-error
# pylint: disable=missing-function-docstring,import-outside-toplevel

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Callable, cast, Mapping

import torch

from boundflow.backends.tvm.r3_d1b_serial_schedule import (
    CompiledR3D1BModuleV1,
    R3D1B_SERIAL_THREADS,
    compile_r3d1b_serial_candidate,
    compile_r3d1b_v1_baseline,
)
from boundflow.backends.tvm.r3_d1_residual11_staged import (
    R3D1_RESIDUAL11_STAGE1_SYMBOL,
    R3D1_RESIDUAL11_STAGE2_SYMBOL,
)
from boundflow.backends.tvm.r3_d1_residual6_staged import (
    R3D1_RESIDUAL6_STAGE1_SYMBOL,
    R3D1_RESIDUAL6_STAGE2_SYMBOL,
)
from boundflow.backends.tvm.r3_full_lower_forward import (
    R31B1_RESIDUAL11_SYMBOL,
    R31B1_RESIDUAL6_SYMBOL,
)

ROOT = Path(__file__).resolve().parents[1]
RESIDUAL11 = ROOT / "artifacts/r3-structured-owner/r3-d1a-residual11-staged-v1"
RESIDUAL6 = ROOT / "artifacts/r3-structured-owner/r3-d1a-residual6-staged-v1"
DEFAULT_OUTPUT = (
    ROOT / "artifacts/r3-structured-owner/r3-d1b-serial-calibration-v1.json"
)
WARMUP_COUNT = 2
SAMPLE_COUNT = 10
ATOL = 2.0e-4


def _hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError("R3-D1B calibration input differs")
    return value


def _cuda_inputs(raw: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    values = raw["inputs"]
    if not isinstance(values, Mapping):
        raise TypeError("R3-D1B input mapping differs")
    return {
        name: value.cuda().contiguous()
        for name, value in values.items()
        if torch.is_tensor(value)
    }


def _view_map(tensors: tuple[torch.Tensor, ...]) -> dict[int, Any]:
    import tvm

    views = {tensor.data_ptr(): tvm.runtime.from_dlpack(tensor) for tensor in tensors}
    if len(views) != len(tensors):
        raise ValueError("R3-D1B DLPack pointer identity differs")
    return views


def _launch(executable: Any, symbol: str, views: Mapping[int, Any], *args) -> None:
    executable[symbol](*(views[value.data_ptr()] for value in args))


def _prepare_launchers(
    baseline: CompiledR3D1BModuleV1,
    candidate: CompiledR3D1BModuleV1,
    residual11_raw: Mapping[str, Any],
    residual6_raw: Mapping[str, Any],
) -> tuple[Callable[[], None], Callable[[], None], dict[str, torch.Tensor]]:
    input11 = _cuda_inputs(residual11_raw)
    input6 = _cuda_inputs(residual6_raw)
    output11_v1 = torch.empty(18_432, device="cuda")
    output6_v1 = torch.empty(18_432, device="cuda")
    bias11_v1 = input11["bias_in"].clone()
    bias6_v1 = input6["bias_in"].clone()
    scratch11 = torch.empty(6144, device="cuda")
    output11 = torch.empty(6144, device="cuda")
    bias11 = torch.empty(6, device="cuda")
    scratch6 = torch.empty(6144, device="cuda")
    output6 = torch.empty(12_288, device="cuda")
    bias6 = torch.empty(6, device="cuda")
    tensors = (
        tuple(input11.values())
        + tuple(input6.values())
        + (
            output11_v1,
            output6_v1,
            bias11_v1,
            bias6_v1,
            scratch11,
            output11,
            bias11,
            scratch6,
            output6,
            bias6,
        )
    )
    views = _view_map(tensors)

    def launch_baseline() -> None:
        bias11_v1.copy_(input11["bias_in"])
        bias6_v1.copy_(input6["bias_in"])
        _launch(
            baseline.executable,
            R31B1_RESIDUAL11_SYMBOL,
            views,
            input11["incoming"],
            input11["weight10"],
            input11["bias10"],
            input11["lower25"],
            input11["upper25"],
            input11["alpha25"],
            input11["alpha_map25"],
            input11["weight8"],
            input11["bias8"],
            bias11_v1,
            output11_v1,
        )
        _launch(
            baseline.executable,
            R31B1_RESIDUAL6_SYMBOL,
            views,
            input6["incoming"],
            input6["weight4"],
            input6["bias4"],
            input6["lower19"],
            input6["upper19"],
            input6["alpha19"],
            input6["alpha_map19"],
            input6["weight2"],
            input6["bias2"],
            input6["weight5"],
            input6["bias5"],
            bias6_v1,
            output6_v1,
        )

    def launch_candidate() -> None:
        _launch(
            candidate.executable,
            R3D1_RESIDUAL11_STAGE1_SYMBOL,
            views,
            input11["incoming"],
            input11["weight10"],
            scratch11,
        )
        _launch(
            candidate.executable,
            R3D1_RESIDUAL11_STAGE2_SYMBOL,
            views,
            input11["incoming"],
            scratch11,
            input11["lower25"],
            input11["upper25"],
            input11["alpha25"],
            input11["alpha_map25"],
            input11["weight8"],
            input11["bias10"],
            input11["bias8"],
            input11["bias_in"],
            output11,
            bias11,
        )
        _launch(
            candidate.executable,
            R3D1_RESIDUAL6_STAGE1_SYMBOL,
            views,
            input6["incoming"],
            input6["weight4"],
            scratch6,
        )
        _launch(
            candidate.executable,
            R3D1_RESIDUAL6_STAGE2_SYMBOL,
            views,
            input6["incoming"],
            scratch6,
            input6["lower19"],
            input6["upper19"],
            input6["alpha19"],
            input6["alpha_map19"],
            input6["weight2"],
            input6["weight5"],
            input6["bias4"],
            input6["bias2"],
            input6["bias5"],
            input6["bias_in"],
            output6,
            bias6,
        )

    outputs = {
        "residual11_output": output11,
        "residual11_bias": bias11,
        "residual6_output": output6,
        "residual6_bias": bias6,
    }
    return launch_baseline, launch_candidate, outputs


def _time(stream: torch.cuda.Stream, launch: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    launch()
    end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end))


def _validate_outputs(
    outputs: Mapping[str, torch.Tensor],
    residual11_raw: Mapping[str, Any],
    residual6_raw: Mapping[str, Any],
) -> float:
    expected = {
        "residual11_output": residual11_raw["reference_output"],
        "residual11_bias": residual11_raw["reference_bias"],
        "residual6_output": residual6_raw["reference_output"],
        "residual6_bias": residual6_raw["reference_bias"],
    }
    maximum = 0.0
    for name, output in outputs.items():
        reference = expected[name]
        if not torch.is_tensor(reference):
            raise TypeError("R3-D1B reference tensor differs")
        actual = output.detach().cpu()
        maximum = max(maximum, float((actual - reference).abs().max().item()))
        if not torch.allclose(
            actual, reference, atol=ATOL, rtol=0.0
        ) or not torch.equal(torch.sign(actual), torch.sign(reference)):
            raise ValueError(f"R3-D1B {name} semantics differ")
    return maximum


def _measure_candidate(
    stream: torch.cuda.Stream,
    baseline: CompiledR3D1BModuleV1,
    candidate: CompiledR3D1BModuleV1,
    residual11_raw: Mapping[str, Any],
    residual6_raw: Mapping[str, Any],
) -> dict[str, object]:
    import tvm_ffi

    baseline_launch, candidate_launch, outputs = _prepare_launchers(
        baseline, candidate, residual11_raw, residual6_raw
    )
    with torch.cuda.stream(stream), tvm_ffi.use_torch_stream(torch.cuda.stream(stream)):
        for _ in range(WARMUP_COUNT):
            baseline_launch()
            candidate_launch()
        stream.synchronize()
        candidate_launch()
        stream.synchronize()
        maximum_diff = _validate_outputs(outputs, residual11_raw, residual6_raw)
        baseline_ms = []
        candidate_ms = []
        for ordinal in range(SAMPLE_COUNT):
            if ordinal % 2 == 0:
                baseline_ms.append(_time(stream, baseline_launch))
                candidate_ms.append(_time(stream, candidate_launch))
            else:
                candidate_ms.append(_time(stream, candidate_launch))
                baseline_ms.append(_time(stream, baseline_launch))
    baseline_median = statistics.median(baseline_ms)
    candidate_median = statistics.median(candidate_ms)
    return {
        "threads_per_block": candidate.threads_per_block,
        "schedule_kind": candidate.schedule_kind,
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "speedup": baseline_median / candidate_median,
        "maximum_diff": maximum_diff,
        "sign_exact": True,
        "candidate_scheduled_tir_hash": candidate.scheduled_tir_hash,
        "candidate_device_source_hash": candidate.device_source_hash,
        "baseline_scheduled_tir_hash": baseline.scheduled_tir_hash,
        "baseline_device_source_hash": baseline.device_source_hash,
        "launch_count": 4,
        "scratch_count": 2,
        "persistent_dense_a": False,
    }


def run(output: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("R3-D1B calibration requires CUDA")
    raw11 = _load(RESIDUAL11 / "raw/run-00.pt")
    raw6 = _load(RESIDUAL6 / "raw/run-00.pt")
    baseline = compile_r3d1b_v1_baseline()
    stream = torch.cuda.Stream()
    rows = []
    for threads in R3D1B_SERIAL_THREADS:
        candidate = compile_r3d1b_serial_candidate(threads)
        row = _measure_candidate(stream, baseline, candidate, raw11, raw6)
        rows.append(row)
        print(
            f"R3-D1B serial threads={threads} median={row['candidate_median_ms']:.6f}ms "
            f"speedup={row['speedup']:.4f}x",
            flush=True,
        )
    winner = min(rows, key=lambda row: cast(float, row["candidate_median_ms"]))
    result: dict[str, object] = {
        "schema_version": "boundflow.r3-d1b-serial-calibration/v1",
        "warmup_count": WARMUP_COUNT,
        "sample_count": SAMPLE_COUNT,
        "candidates": rows,
        "winner_threads_per_block": winner["threads_per_block"],
        "winner_schedule_kind": winner["schedule_kind"],
        "winner_speedup": winner["speedup"],
        "isolated_opportunity_gate": 15.5,
        "winner_gate_pass": cast(float, winner["speedup"]) >= 15.5,
        "calibration_only": True,
        "formal_performance_claimed": False,
    }
    result["result_hash"] = _hash(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output.absolute())
    print(
        f"R3-D1B calibration winner={result['winner_threads_per_block']} "
        f"gate_pass={str(result['winner_gate_pass']).lower()} "
        "formal_performance_claimed=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
