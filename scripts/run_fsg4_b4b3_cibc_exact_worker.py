#!/usr/bin/env python3
"""Run one fresh B3 versus B4-B3 CIBC production exact-call pair."""

# pylint: disable=wrong-import-position,too-many-locals,missing-function-docstring
# pylint: disable=protected-access,duplicate-code

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, cast

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.fsg4_b3_prepared_core import (
    instantiate_core_plan_v1,
    prepare_core_template_v1,
)
from boundflow.runtime.fsg4_b3_terminal_optimizer_schedule import (
    NativeTerminalOptimizerResultV1,
    compile_terminal_optimizer_schedule_v1,
    execute_terminal_optimizer_schedule_v1,
)
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.fsg4_b4b3_cibc_exact_call import (
    B4B3CIBCExactCallObserverV1,
)
from boundflow.runtime.rvir_v4_optimizer_mutation import (
    production_optimizer_step_trace_from_payload_v4,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    ProductionTensorOwnership,
    ProductionTensorRole,
    production_snapshot_from_payload_v4,
)
from boundflow.runtime.task_executor import InputSpec
from scripts import run_fsg4_b4b_capture_worker as b4b0
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

WORKER_SCHEMA = "boundflow.fsg4-b4b3-cibc-exact-worker/v1"
REFERENCE_ARTIFACT = REPOSITORY_ROOT / (
    "artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1"
)
ATOL = 2.0e-4
RTOL = 2.0e-4


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _metric(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    if reference.shape != candidate.shape:
        raise ValueError("B4-B3 CIBC metric shape differs")
    return {
        "element_count": reference.numel(),
        "maximum_absolute_difference": float(
            (reference - candidate).abs().max().item()
        ),
        "allclose": bool(torch.allclose(reference, candidate, atol=ATOL, rtol=RTOL)),
        "sign_exact": bool(torch.equal(torch.sign(reference), torch.sign(candidate))),
    }


def _prepare(args: argparse.Namespace):
    device = torch.device("cuda:0")
    raw = torch.load(args.source_capture, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("B4-B3 CIBC source capture root differs")
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    production = production_optimizer_step_trace_from_payload_v4(
        raw["optimizer_step_traces"][0]
    )
    cpu_mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    mapping = cpu_mapping.to(device=device, dtype=torch.float32)
    program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    module.bindings = copy.deepcopy(
        cast(dict[str, Any], b4b0._move(module.bindings, device=device))
    )
    lower = b4b0._one(snapshot, ProductionTensorRole.INPUT_LOWER).to(device)
    upper = b4b0._one(snapshot, ProductionTensorRole.INPUT_UPPER).to(device)
    objective = b4b0._one(snapshot, ProductionTensorRole.LINEAR_SPEC).to(device)
    mutable_paths = tuple(
        sorted(
            item.semantic_path
            for item in snapshot.tensors
            if item.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
        )
    )
    template = prepare_core_template_v1(
        template_id="resnet2b-prop0-b4-b3-cibc-exact",
        program=program,
        module=module,
        topology=TOPOLOGY,
        device=str(device),
        dtype=torch.float32,
        input_shape=lower.shape,
        objective_shape=objective.shape,
        mutable_paths=mutable_paths,
    )
    spec = InputSpec.box(value_name=program.graph.inputs[0], lower=lower, upper=upper)
    instance = instantiate_core_plan_v1(
        template=template,
        topology=TOPOLOGY,
        snapshot=snapshot,
        mapping=mapping,
        input_spec=spec,
        linear_spec_C=objective,
        mutation_policy=production.mutation_policy,
    )
    return module, mapping, production, objective, spec, instance


def _execute(
    call: Callable[[], NativeTerminalOptimizerResultV1],
) -> tuple[NativeTerminalOptimizerResultV1, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("B4-B3 CIBC worker requires CUDA")
    module, mapping, production, objective, spec, instance = _prepare(args)
    schedule = compile_terminal_optimizer_schedule_v1()
    reference_payload = torch.load(
        REFERENCE_ARTIFACT / f"run_{args.run_ordinal:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    reference_capture = production_differentiable_reference_capture_from_payload_v1(
        reference_payload["captures"][1]
    )
    observer = B4B3CIBCExactCallObserverV1(reference_capture)

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
        return execute_terminal_optimizer_schedule_v1(
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

    if args.order == "BC":
        baseline, baseline_ms = _execute(baseline_call)
        candidate, candidate_ms = _execute(candidate_call)
    else:
        candidate, candidate_ms = _execute(candidate_call)
        baseline, baseline_ms = _execute(baseline_call)
    metrics = {
        "terminal_lower": _metric(baseline.terminal_lower, candidate.terminal_lower)
    }
    for name in sorted(baseline.terminal_state.alphas):
        metrics[f"alpha:{name}"] = _metric(
            baseline.terminal_state.alphas[name], candidate.terminal_state.alphas[name]
        )
    for name in sorted(baseline.terminal_state.betas):
        metrics[f"beta:{name}"] = _metric(
            baseline.terminal_state.betas[name], candidate.terminal_state.betas[name]
        )
    receipt = observer.receipt().to_dict()
    maximum = max(
        cast(float, metric["maximum_absolute_difference"])
        for metric in metrics.values()
    )
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "run_ordinal": args.run_ordinal,
        "order": args.order,
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "paired_speedup": baseline_ms / candidate_ms,
        "metrics": metrics,
        "maximum_absolute_difference": maximum,
        "allclose": all(bool(metric["allclose"]) for metric in metrics.values()),
        "sign_exact": all(bool(metric["sign_exact"]) for metric in metrics.values()),
        "receipt": receipt,
        "local_parity": observer.local_parity,
        "baseline_terminal_state_hash": baseline.terminal_state.stable_hash(),
        "candidate_terminal_state_hash": candidate.terminal_state.stable_hash(),
        "performance_claimed": False,
    }
    payload["worker_hash"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-ordinal", type=int, choices=range(5), required=True)
    parser.add_argument("--order", choices=("BC", "CB"), required=True)
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
