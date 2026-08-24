#!/usr/bin/env python3
"""Run one isolated native or R3-1b compiled correctness/memory worker."""

# pylint: disable=wrong-import-position,too-many-locals,too-many-statements
# pylint: disable=protected-access,missing-function-docstring

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
from pathlib import Path
import sys
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_compiled_p_alpha_vjp import (
    PreparedR31B2CompiledCustomBackwardV1,
    _EXECUTOR_REGISTRY,
    _R31B2CompiledFunction,
)
from boundflow.runtime.r3_structured_owner_custom_backward import (
    bind_r31_runtime_inputs_v1,
    compile_r31_full_region_plan_v1,
    execute_r31_native_oracle_v1,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
    production_tensor_sha256,
)
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

WORKER_SCHEMA = "boundflow.r3-1b3-compiled-five-fresh-worker/v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions(plan, tensors, role: str) -> tuple[int, ...]:  # type: ignore[no-untyped-def]
    return tuple(
        tensors[ordinal]._version
        for ordinal, spec in enumerate(plan.tensor_specs)
        if spec.role == role
    )


def _candidate(prepared, tensors):  # type: ignore[no-untyped-def]
    key = hashlib.sha256(
        f"{prepared.plan.stable_hash()}:{id(prepared)}".encode("utf-8")
    ).hexdigest()
    saved: list[torch.Tensor] = []

    def pack(value: torch.Tensor) -> torch.Tensor:
        saved.append(value)
        return value

    _EXECUTOR_REGISTRY[key] = prepared
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            lower = _R31B2CompiledFunction.apply(key, *tensors)
            gradient = torch.autograd.grad(
                lower,
                tensors[prepared.plan.p_alpha_input_ordinal],
                grad_outputs=prepared.upstream_gradient,
            )[0]
    finally:
        _EXECUTOR_REGISTRY.pop(key, None)
    dense = sum(value.ndim == 5 and tuple(value.shape[:2]) == (6, 1) for value in saved)
    receipt = dataclasses.asdict(prepared.receipt(saved_dense_a_count=dense))
    receipt.update(
        execution_kind="r3-1b-compiled-custom-vjp",
        saved_tensor_count=len(saved),
        compiled_region=True,
    )
    return lower.detach(), gradient.detach(), receipt


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("R3-1b3 worker requires CUDA")
    device = torch.device("cuda:0")
    raw = torch.load(args.source_capture, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
    tensors = bind_r31_runtime_inputs_v1(
        plan, module, snapshot, device=device, dtype=torch.float32
    )
    prepared = (
        PreparedR31B2CompiledCustomBackwardV1(plan, trace, tensors)
        if args.mode == "candidate"
        else None
    )
    alpha_before = _versions(plan, tensors, "compressed_alpha")
    beta_before = _versions(plan, tensors, "compressed_beta")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    if args.mode == "native":
        lower, gradient = execute_r31_native_oracle_v1(plan, tensors)
        receipt: dict[str, object] = {
            "execution_kind": "independent-native-autograd",
            "forward_count": 1,
            "backward_count": 1,
            "optimizer_mutation_count": 0,
            "compiled_region": False,
            "timing_recorded": False,
            "performance_claimed": False,
        }
    else:
        assert prepared is not None
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            lower, gradient, receipt = _candidate(prepared, tensors)
        stream.synchronize()
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    alpha_after = _versions(plan, tensors, "compressed_alpha")
    beta_after = _versions(plan, tensors, "compressed_beta")
    if alpha_before != alpha_after or beta_before != beta_after:
        raise ValueError("R3-1b3 worker mutated production state")
    properties = torch.cuda.get_device_properties(device)
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "run_index": args.run_index,
        "mode": args.mode,
        "source_capture_sha256": _file_sha256(args.source_capture),
        "model_sha256": _file_sha256(args.model),
        "source_state_hash": mapping.stable_hash(),
        "plan_hash": plan.stable_hash(),
        "trace_hash": trace.stable_hash(),
        "final_lower": lower.cpu(),
        "compressed_alpha_gradient": gradient.cpu(),
        "final_lower_sha256": production_tensor_sha256(lower),
        "compressed_alpha_gradient_sha256": production_tensor_sha256(gradient),
        "execution_receipt": receipt,
        "memory": {
            "allocated_before": allocated_before,
            "reserved_before": reserved_before,
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
            "peak_allocated_increment": peak_allocated - allocated_before,
            "peak_reserved_increment": peak_reserved - reserved_before,
        },
        "alpha_versions_before": list(alpha_before),
        "alpha_versions_after": list(alpha_after),
        "beta_versions_before": list(beta_before),
        "beta_versions_after": list(beta_after),
        "environment": {
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "device_index": device.index,
        },
        "timing_recorded": False,
        "performance_claimed": False,
    }
    torch.save(payload, args.result)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "candidate"), required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    args.source_capture = args.source_capture.resolve()
    args.model = args.model.resolve()
    args.result = args.result.resolve()
    if args.run_index < 0:
        raise ValueError("R3-1b3 worker run index differs")
    payload = _run(args)
    print(
        f"R3-1b3 run={payload['run_index']} mode={payload['mode']} "
        f"plan={str(payload['plan_hash'])[:12]} performance_claimed=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
