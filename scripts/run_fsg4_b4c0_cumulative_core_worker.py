#!/usr/bin/env python3
"""Fresh warmed/interleaved B3 versus B4-B3 cumulative core timing worker."""

# pylint: disable=wrong-import-position,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring,protected-access,duplicate-code

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.backends.tvm.cibc_dense_exact_conv import (
    compile_cibc_dense_exact_conv_tir_v3,
)
from boundflow.runtime.fsg4_b3_terminal_optimizer_schedule import (
    compile_terminal_optimizer_schedule_v1,
    execute_terminal_optimizer_schedule_v1,
)
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.fsg4_b4b3_cibc_exact_call import (
    B4B3CIBCExactCallObserverV1,
)
from boundflow.runtime.fsg4_b4c2_materialization_frontier import (
    B4C2MaterializationFrontierObserverV1,
)
from scripts import run_fsg4_b4b3_cibc_exact_worker as exact_worker

WORKER_SCHEMA = "boundflow.fsg4-b4c0-cumulative-core-worker/v1"
REFERENCE_ARTIFACT = REPOSITORY_ROOT / (
    "artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1"
)
WARMUPS_PER_SIDE = 3
GROUP_COUNT = 30
ATOL = 2.0e-4


def _semantic_maximum(baseline, candidate) -> tuple[float, bool, bool]:
    pairs = [(baseline.terminal_lower, candidate.terminal_lower)]
    pairs.extend(
        (baseline.terminal_state.alphas[name], candidate.terminal_state.alphas[name])
        for name in sorted(baseline.terminal_state.alphas)
    )
    pairs.extend(
        (baseline.terminal_state.betas[name], candidate.terminal_state.betas[name])
        for name in sorted(baseline.terminal_state.betas)
    )
    maximum = 0.0
    allclose = True
    sign_exact = True
    for reference, observed in pairs:
        maximum = max(maximum, float((reference - observed).abs().max().item()))
        allclose = allclose and bool(
            torch.allclose(reference, observed, atol=ATOL, rtol=ATOL)
        )
        sign_exact = sign_exact and bool(
            torch.equal(torch.sign(reference), torch.sign(observed))
        )
    return maximum, allclose, sign_exact


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("B4-C0 cumulative core worker requires CUDA")
    module, mapping, production, objective, spec, instance = exact_worker._prepare(args)
    schedule = compile_terminal_optimizer_schedule_v1()
    reference_payload = torch.load(
        REFERENCE_ARTIFACT / f"run_{args.run_ordinal % 5:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    reference_capture = production_differentiable_reference_capture_from_payload_v1(
        reference_payload["captures"][1]
    )
    major, minor = torch.cuda.get_device_capability()
    compiled = compile_cibc_dense_exact_conv_tir_v3(
        compute_capability=f"sm_{major}{minor}"
    )

    def baseline_call():
        return execute_terminal_optimizer_schedule_v1(
            module,
            spec,
            linear_spec_C=objective,
            relu_pre=mapping.relu_pre,
            initial_state=instance.initial_state,
            mutation_policy=production.mutation_policy,
            schedule=schedule,
            prevalidated_plan=instance,
        )

    def candidate_call():
        if args.materialization_frontier:
            observer = B4C2MaterializationFrontierObserverV1()
        else:
            observer = B4B3CIBCExactCallObserverV1(
                reference_capture,
                compiled=compiled,
                record_local_parity=False,
                capture_evaluation_zero=False,
                native_value_bridge=not args.provider_owned,
                provider_owns_lower_path=args.provider_owned,
            )
        result = execute_terminal_optimizer_schedule_v1(
            module,
            spec,
            linear_spec_C=objective,
            relu_pre=mapping.relu_pre,
            initial_state=instance.initial_state,
            mutation_policy=production.mutation_policy,
            schedule=schedule,
            prevalidated_plan=instance,
            b4b_region_observer=observer,
        )
        return result, observer.receipt().to_dict()

    def elapsed(call):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = call()
        torch.cuda.synchronize()
        milliseconds = (time.perf_counter() - started) * 1000.0
        if not math.isfinite(milliseconds) or milliseconds <= 0.0:
            raise RuntimeError("B4-C0 cumulative core duration differs")
        return result, milliseconds

    for _ in range(WARMUPS_PER_SIDE):
        baseline_call()
        candidate_call()
    torch.cuda.synchronize()
    groups = []
    maximum = 0.0
    sign_exact = True
    receipt = None
    for ordinal in range(GROUP_COUNT):
        if args.order == "BC":
            baseline, baseline_ms = elapsed(baseline_call)
            (candidate, candidate_receipt), candidate_ms = elapsed(candidate_call)
        else:
            (candidate, candidate_receipt), candidate_ms = elapsed(candidate_call)
            baseline, baseline_ms = elapsed(baseline_call)
        observed_maximum, observed_allclose, observed_sign = _semantic_maximum(
            baseline, candidate
        )
        if not args.allow_semantic_drift and (
            not observed_allclose or not observed_sign or observed_maximum > ATOL
        ):
            raise ValueError("B4-C0 cumulative core semantic pair differs")
        maximum = max(maximum, observed_maximum)
        sign_exact = sign_exact and observed_sign
        if receipt is None:
            receipt = candidate_receipt
        elif receipt != candidate_receipt:
            raise ValueError("B4-C0 cumulative core receipt differs")
        groups.append(
            {
                "group_ordinal": ordinal,
                "order": args.order,
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "speedup": baseline_ms / candidate_ms,
            }
        )

    def peak(call) -> tuple[int, int]:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        call()
        torch.cuda.synchronize()
        return (
            torch.cuda.max_memory_allocated(),
            torch.cuda.max_memory_reserved(),
        )

    baseline_peak = peak(baseline_call)
    candidate_peak = peak(candidate_call)
    baseline_median = statistics.median(row["baseline_ms"] for row in groups)
    candidate_median = statistics.median(row["candidate_ms"] for row in groups)
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "run_ordinal": args.run_ordinal,
        "order": args.order,
        "warmups_per_side": WARMUPS_PER_SIDE,
        "group_count": GROUP_COUNT,
        "groups": groups,
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "paired_speedup": baseline_median / candidate_median,
        "maximum_absolute_difference": maximum,
        "allclose": maximum <= ATOL,
        "sign_exact": sign_exact,
        "receipt": receipt,
        "baseline_peak_allocated_bytes": baseline_peak[0],
        "baseline_peak_reserved_bytes": baseline_peak[1],
        "candidate_peak_allocated_bytes": candidate_peak[0],
        "candidate_peak_reserved_bytes": candidate_peak[1],
        "environment": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "compute_capability": [major, minor],
        },
        "performance_claimed": False,
        "candidate_mode": (
            "materialization-frontier"
            if args.materialization_frontier
            else (
                "provider-owned-lower" if args.provider_owned else "native-value-bridge"
            )
        ),
    }
    payload["worker_hash"] = exact_worker.canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-ordinal", type=int, choices=range(6), required=True)
    parser.add_argument("--order", choices=("BC", "CB"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider-owned", action="store_true")
    parser.add_argument("--materialization-frontier", action="store_true")
    parser.add_argument("--allow-semantic-drift", action="store_true")
    args = parser.parse_args()
    if args.provider_owned and args.materialization_frontier:
        parser.error(
            "provider-owned and materialization-frontier are mutually exclusive"
        )
    payload = _run(args)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
