#!/usr/bin/env python3
"""Fresh operator-tuning or whole-model CIBC IBP timing worker."""

# pylint: disable=wrong-import-position,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring,protected-access,too-many-arguments
# pylint: disable=assignment-from-no-return,unpacking-non-sequence,not-callable
# pylint: disable=cell-var-from-loop,consider-using-dict-items
# pylint: disable=unnecessary-lambda-assignment
# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import cast

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.backends.tvm.cibc_ibp_conv import (
    CIBCIBPConvSignatureV1,
    compile_cibc_ibp_conv_tir_v1,
)
from boundflow.runtime.cibc_ibp_conv import CIBCIBPConvExecutorV1
from boundflow.runtime.cibc_ibp_graph import CIBCIBPCUDAGraphPlanV1
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from scripts import run_fsg4_b4b3_cibc_exact_worker as exact_worker

SCHEMA = "boundflow.cibc-ibp-horizontal-worker/v1"
GROUP_COUNT = 30
OPERATOR_REPEATS = 500
MODEL_REPEATS = 100
ATOL = 3.0e-4
CONV_ORDINALS = (0, 2, 4, 5, 8, 10)


def _timed(call, repeats: int) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        call()
    torch.cuda.synchronize()
    duration = (time.perf_counter() - started) * 1000.0 / repeats
    if not math.isfinite(duration) or duration <= 0.0:
        raise RuntimeError("CIBC IBP duration differs")
    return duration


def _prepare(args: argparse.Namespace):
    source_args = argparse.Namespace(
        source_capture=args.source_capture,
        model=args.model,
        run_ordinal=args.run_ordinal,
    )
    module, _mapping, _production, _objective, spec, _instance = exact_worker._prepare(
        source_args
    )
    lower, upper = spec.perturbation.bounding_box(spec.center)
    return module, spec, lower.contiguous(), upper.contiguous()


def _operator_worker(args: argparse.Namespace) -> dict[str, object]:
    module, spec, input_lower, input_upper = _prepare(args)
    interval_env, _relu_pre = _forward_ibp_trace_mlp(module, spec)
    params = dict(module.bindings["params"])
    major, minor = torch.cuda.get_device_capability()
    rows = []
    for op_ordinal, op in enumerate(module.get_entry_task().ops):
        if op.op_type != "conv2d":
            continue
        if op.inputs[0] == spec.value_name:
            lower, upper = input_lower, input_upper
        else:
            source = interval_env[op.inputs[0]]
            lower, upper = source.lower.contiguous(), source.upper.contiguous()
        weight = params[op.inputs[1]].contiguous()
        bias = params[op.inputs[2]].contiguous()
        stride = tuple(int(value) for value in op.attrs["stride"])
        padding = tuple(int(value) for value in op.attrs["padding"])
        dilation = tuple(int(value) for value in op.attrs["dilation"])
        groups = int(op.attrs["groups"])
        signature = CIBCIBPConvSignatureV1(
            input_shape=cast(
                tuple[int, int, int, int], tuple(int(value) for value in lower.shape)
            ),
            weight_shape=cast(
                tuple[int, int, int, int], tuple(int(value) for value in weight.shape)
            ),
            stride=cast(tuple[int, int], stride),
            padding=cast(tuple[int, int], padding),
            dilation=cast(tuple[int, int], dilation),
            groups=groups,
        )
        compiled = compile_cibc_ibp_conv_tir_v1(
            signature,
            threads_per_block=args.threads,
            compute_capability=f"sm_{major}{minor}",
        )
        executor = CIBCIBPConvExecutorV1(lower, upper, weight, bias, compiled=compiled)
        executor.validate_stream()
        kwargs = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }

        def baseline_call():
            positive = weight.clamp_min(0)
            negative = weight.clamp_max(0)
            lower_out = (
                F.conv2d(lower, positive, None, **kwargs)
                + F.conv2d(upper, negative, None, **kwargs)
                + bias.view(1, -1, 1, 1)
            )
            upper_out = (
                F.conv2d(upper, positive, None, **kwargs)
                + F.conv2d(lower, negative, None, **kwargs)
                + bias.view(1, -1, 1, 1)
            )
            return lower_out, upper_out

        reference = baseline_call()
        observed = executor.run()
        maximum = max(
            float((reference[0] - observed[0]).abs().max().item()),
            float((reference[1] - observed[1]).abs().max().item()),
        )
        sign_exact = all(
            torch.equal(torch.sign(left), torch.sign(right))
            for left, right in zip(reference, observed)
        )
        if maximum > ATOL or not sign_exact:
            raise ValueError("CIBC IBP operator semantic pair differs")
        for _ in range(20):
            baseline_call()
            executor.run()
        groups_payload = []
        for group in range(GROUP_COUNT):
            if args.order == "BC":
                baseline_ms = _timed(baseline_call, OPERATOR_REPEATS)
                candidate_ms = _timed(executor.run, OPERATOR_REPEATS)
            else:
                candidate_ms = _timed(executor.run, OPERATOR_REPEATS)
                baseline_ms = _timed(baseline_call, OPERATOR_REPEATS)
            groups_payload.append(
                {
                    "group": group,
                    "baseline_ms": baseline_ms,
                    "candidate_ms": candidate_ms,
                    "speedup": baseline_ms / candidate_ms,
                }
            )
        baseline_median = statistics.median(
            item["baseline_ms"] for item in groups_payload
        )
        candidate_median = statistics.median(
            item["candidate_ms"] for item in groups_payload
        )
        rows.append(
            {
                "op_ordinal": op_ordinal,
                "signature": {
                    "input_shape": list(signature.input_shape),
                    "weight_shape": list(signature.weight_shape),
                    "stride": list(signature.stride),
                    "padding": list(signature.padding),
                },
                "groups": groups_payload,
                "baseline_median_ms": baseline_median,
                "candidate_median_ms": candidate_median,
                "speedup": baseline_median / candidate_median,
                "maximum_absolute_difference": maximum,
                "sign_exact": sign_exact,
                "module_hash": compiled.module_hash,
                "device_source_hash": compiled.device_source_hash,
                "launch_count": executor.launch_count,
            }
        )
    observed_ordinals = tuple(cast(int, row["op_ordinal"]) for row in rows)
    if observed_ordinals != CONV_ORDINALS:
        raise ValueError("CIBC IBP production Conv inventory differs")
    return {
        "operators": rows,
        "operator_count": len(rows),
        "group_count": GROUP_COUNT,
        "operator_repeats": OPERATOR_REPEATS,
        "plan_owned": True,
    }


def _model_worker(args: argparse.Namespace) -> dict[str, object]:
    module, spec, input_lower, input_upper = _prepare(args)
    baseline = CIBCIBPCUDAGraphPlanV1(
        module, spec.value_name, input_lower, input_upper, None
    )
    candidate = CIBCIBPCUDAGraphPlanV1(
        module, spec.value_name, input_lower, input_upper, args.threads
    )
    baseline.replay(input_lower=input_lower, input_upper=input_upper)
    candidate.replay(input_lower=input_lower, input_upper=input_upper)
    torch.cuda.synchronize()
    maximum = 0.0
    sign_exact = True
    for name in baseline.outputs:
        for side in ("lower", "upper"):
            reference = getattr(baseline.outputs[name], side)
            observed = getattr(candidate.outputs[name], side)
            maximum = max(maximum, float((reference - observed).abs().max().item()))
            sign_exact = sign_exact and torch.equal(
                torch.sign(reference), torch.sign(observed)
            )
    final_name = module.get_entry_task().ops[-1].outputs[0]
    final_maximum = max(
        float(
            (
                getattr(baseline.outputs[final_name], side)
                - getattr(candidate.outputs[final_name], side)
            )
            .abs()
            .max()
            .item()
        )
        for side in ("lower", "upper")
    )
    if maximum > ATOL or not sign_exact:
        raise ValueError("CIBC IBP model semantic pair differs")
    for _ in range(20):
        baseline.replay(input_lower=input_lower, input_upper=input_upper)
        candidate.replay(input_lower=input_lower, input_upper=input_upper)
    groups_payload = []
    baseline_call = lambda: baseline.replay(  # noqa: E731
        input_lower=input_lower, input_upper=input_upper
    )
    candidate_call = lambda: candidate.replay(  # noqa: E731
        input_lower=input_lower, input_upper=input_upper
    )
    for group in range(GROUP_COUNT):
        if args.order == "BC":
            baseline_ms = _timed(baseline_call, MODEL_REPEATS)
            candidate_ms = _timed(candidate_call, MODEL_REPEATS)
        else:
            candidate_ms = _timed(candidate_call, MODEL_REPEATS)
            baseline_ms = _timed(baseline_call, MODEL_REPEATS)
        groups_payload.append(
            {
                "group": group,
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "speedup": baseline_ms / candidate_ms,
            }
        )
    baseline_median = statistics.median(item["baseline_ms"] for item in groups_payload)
    candidate_median = statistics.median(
        item["candidate_ms"] for item in groups_payload
    )
    return {
        "groups": groups_payload,
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "speedup": baseline_median / candidate_median,
        "maximum_absolute_difference": maximum,
        "final_maximum_absolute_difference": final_maximum,
        "sign_exact": sign_exact,
        "conv_coverage": candidate.launch_count,
        "group_count": GROUP_COUNT,
        "model_repeats": MODEL_REPEATS,
        "input_copy_included": True,
        "baseline_cuda_graph": True,
        "candidate_cuda_graph": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("operator", "model"), required=True)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-ordinal", type=int, required=True)
    parser.add_argument("--order", choices=("BC", "CB"), required=True)
    parser.add_argument("--threads", type=int, choices=(64, 128, 256), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CIBC IBP worker requires CUDA")
    result = _operator_worker(args) if args.mode == "operator" else _model_worker(args)
    payload: dict[str, object] = {
        "schema_version": SCHEMA,
        "mode": args.mode,
        "run_ordinal": args.run_ordinal,
        "order": args.order,
        "threads_per_block": args.threads,
        "environment": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "performance_claimed": False,
        **result,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    payload["worker_hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
