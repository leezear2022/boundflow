#!/usr/bin/env python3
"""Run one fresh PyTorch/direct-CIBC/canonical-pipeline S1 timing triple."""

# pylint: disable=wrong-import-position,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring,protected-access

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, cast

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.asplos27_s1_cibc_pipeline import (
    PreparedS1CIBCCUDAGraphV1,
    prepare_s1_cibc_program_v1,
)
from boundflow.runtime.cibc_ibp_graph import CIBCIBPCUDAGraphPlanV1
from scripts import run_cibc_ibp_horizontal_worker as source_worker

WORKER_SCHEMA = "boundflow.asplos27-s1-cibc-worker/v1"
GROUP_COUNT = 30
REPEATS = 200
ATOL = 3.0e-4


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def timed(call: Callable[[], object]) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(REPEATS):
        call()
    torch.cuda.synchronize()
    duration = (time.perf_counter() - started) * 1000.0 / REPEATS
    if not math.isfinite(duration) or duration <= 0.0:
        raise RuntimeError("S1 CIBC worker duration differs")
    return duration


def metric(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    maximum = float((reference - candidate).abs().max().item())
    return {
        "element_count": reference.numel(),
        "maximum_absolute_difference": maximum,
        "allclose": bool(torch.allclose(reference, candidate, atol=ATOL, rtol=ATOL)),
        "sign_exact": bool(torch.equal(torch.sign(reference), torch.sign(candidate))),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("S1 CIBC worker requires CUDA")
    source_args = argparse.Namespace(
        source_capture=args.source_capture,
        model=args.model,
        run_ordinal=args.run_ordinal,
    )
    module, spec, input_lower, input_upper = source_worker._prepare(source_args)
    task = module.get_entry_task()
    schedules = tuple((op.name, 128) for op in task.ops if op.op_type == "conv2d")
    baseline = CIBCIBPCUDAGraphPlanV1(
        module, spec.value_name, input_lower, input_upper, None
    )
    direct = CIBCIBPCUDAGraphPlanV1(
        module, spec.value_name, input_lower, input_upper, 128
    )
    prepared = prepare_s1_cibc_program_v1(
        module,
        input_lower=input_lower,
        input_upper=input_upper,
        cibc_threads_by_op=schedules,
    )
    pipeline = PreparedS1CIBCCUDAGraphV1(prepared)
    baseline_outputs = baseline.replay(input_lower=input_lower, input_upper=input_upper)
    direct_outputs = direct.replay(input_lower=input_lower, input_upper=input_upper)
    pipeline_output, execution_receipt = pipeline.run(
        input_lower=input_lower, input_upper=input_upper
    )
    torch.cuda.synchronize()
    final_name = task.output_values[0]
    final_baseline = baseline_outputs[final_name]
    final_direct = direct_outputs[final_name]
    metrics = {
        "baseline_direct_lower": metric(final_baseline.lower, final_direct.lower),
        "baseline_direct_upper": metric(final_baseline.upper, final_direct.upper),
        "baseline_pipeline_lower": metric(final_baseline.lower, pipeline_output.lower),
        "baseline_pipeline_upper": metric(final_baseline.upper, pipeline_output.upper),
        "direct_pipeline_lower": metric(final_direct.lower, pipeline_output.lower),
        "direct_pipeline_upper": metric(final_direct.upper, pipeline_output.upper),
    }
    if not all(
        bool(value["allclose"]) and bool(value["sign_exact"])
        for value in metrics.values()
    ):
        raise ValueError("S1 CIBC worker semantics differ")
    calls = {
        "B": lambda: baseline.replay(input_lower=input_lower, input_upper=input_upper),
        "D": lambda: direct.replay(input_lower=input_lower, input_upper=input_upper),
        "P": lambda: pipeline.run(input_lower=input_lower, input_upper=input_upper),
    }
    for _ in range(20):
        for label in args.order:
            calls[label]()
    groups = []
    for ordinal in range(GROUP_COUNT):
        durations = {label: timed(calls[label]) for label in args.order}
        groups.append(
            {
                "group": ordinal,
                "baseline_ms": durations["B"],
                "direct_ms": durations["D"],
                "pipeline_ms": durations["P"],
            }
        )
    medians = {
        name: statistics.median(float(row[name]) for row in groups)
        for name in ("baseline_ms", "direct_ms", "pipeline_ms")
    }
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "run_ordinal": args.run_ordinal,
        "order": args.order,
        "groups": groups,
        "group_count": GROUP_COUNT,
        "repeats": REPEATS,
        **medians,
        "direct_speedup": medians["baseline_ms"] / medians["direct_ms"],
        "pipeline_speedup": medians["baseline_ms"] / medians["pipeline_ms"],
        "pipeline_direct_propagation": medians["direct_ms"] / medians["pipeline_ms"],
        "metrics": metrics,
        "maximum_absolute_difference": max(
            float(cast(Any, value["maximum_absolute_difference"]))
            for value in metrics.values()
        ),
        "allclose": True,
        "sign_exact": True,
        "compile_receipt": prepared.compile_receipt.to_dict(),
        "execution_receipt": execution_receipt.to_dict(),
        "environment": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "input_copy_included": True,
        "pipeline_cuda_graph": True,
        "direct_cuda_graph": True,
        "baseline_cuda_graph": True,
        "performance_claimed": False,
    }
    payload["worker_hash"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-ordinal", type=int, choices=range(6), required=True)
    parser.add_argument(
        "--order", choices=("BDP", "BPD", "DBP", "DPB", "PBD", "PDB"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(canonical_json(payload))


if __name__ == "__main__":
    main()
