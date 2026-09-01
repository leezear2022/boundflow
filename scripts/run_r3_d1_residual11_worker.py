#!/usr/bin/env python3
"""Run one fresh D1-A residual11 staged correctness worker."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=too-many-statements,missing-function-docstring,duplicate-code
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.backends.tvm.r3_full_lower_forward import (
    R31B1_LINEAR14_SYMBOL,
    R31B1_LINEAR16_SYMBOL,
    R31B1_RESIDUAL11_SYMBOL,
    R31B1_SEED_SYMBOL,
)
from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_d1_residual11_staged import (
    execute_r3d1_residual11_staged_v1,
    R3D1Residual11ModuleCacheV1,
)
from boundflow.runtime.r3_full_lower_forward_tir import (
    PreparedR31B1FullLowerForwardV1,
    R31B1ModuleCacheV1,
)
from boundflow.runtime.r3_structured_owner_custom_backward import (
    bind_r31_runtime_inputs_v1,
    compile_r31_full_region_plan_v1,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
    production_tensor_sha256,
)
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

WORKER_SCHEMA = "boundflow.r3-d1-residual11-worker/v1"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous().clone()


def _run(args: argparse.Namespace) -> dict[str, object]:
    import tvm_ffi

    if not torch.cuda.is_available():
        raise RuntimeError("R3-D1 worker requires CUDA")
    raw = torch.load(args.source_capture, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
    tensors = bind_r31_runtime_inputs_v1(
        plan, module, snapshot, device=torch.device("cuda:0")
    )
    prepared = PreparedR31B1FullLowerForwardV1(
        plan, trace, tensors, cache=R31B1ModuleCacheV1()
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with tvm_ffi.use_torch_stream(torch.cuda.stream(stream)):
            s0 = prepared.scratch_0
            s1 = prepared.scratch_1
            bias = prepared.bias_accumulator
            prepared._launch(
                R31B1_SEED_SYMBOL, prepared._tensor("objective"), s0[:60], bias
            )
            prepared._launch(
                R31B1_LINEAR16_SYMBOL,
                s0[:60],
                prepared._tensor("param/linear2.weight"),
                prepared._tensor("param/linear2.bias"),
                bias,
                s1[:600],
                bias,
            )
            prepared._relu("31", s1[:600], bias, active_beta=True)
            prepared._launch(
                R31B1_LINEAR14_SYMBOL,
                s1[:600],
                prepared._tensor("param/linear1.weight"),
                prepared._tensor("param/linear1.bias"),
                bias,
                s0[:6144],
                bias,
            )
            prepared._relu("28", s0[:6144], bias)
            bias_before = bias.clone()
            input_payload = {
                "incoming": _cpu(s0),
                "weight10": _cpu(prepared._tensor("param/layer1.1.conv2.weight")),
                "lower25": _cpu(prepared._tensor("relu/25/lower").reshape(6, 1024)),
                "upper25": _cpu(prepared._tensor("relu/25/upper").reshape(6, 1024)),
                "alpha25": _cpu(prepared._tensor("relu/25/alpha")),
                "alpha_map25": _cpu(prepared.alpha_maps["25"]),
                "weight8": _cpu(prepared._tensor("param/layer1.1.conv1.weight")),
                "bias10": _cpu(prepared._tensor("param/layer1.1.conv2.bias")),
                "bias8": _cpu(prepared._tensor("param/layer1.1.conv1.bias")),
                "bias_in": _cpu(bias_before),
            }
            prepared._launch(
                R31B1_RESIDUAL11_SYMBOL,
                s0,
                prepared._tensor("param/layer1.1.conv2.weight"),
                prepared._tensor("param/layer1.1.conv2.bias"),
                prepared._tensor("relu/25/lower").reshape(6, 1024),
                prepared._tensor("relu/25/upper").reshape(6, 1024),
                prepared._tensor("relu/25/alpha"),
                prepared.alpha_maps["25"],
                prepared._tensor("param/layer1.1.conv1.weight"),
                prepared._tensor("param/layer1.1.conv1.bias"),
                bias,
                s1,
            )
            reference_output = _cpu(s1[:6144])
            reference_bias = _cpu(bias)
        scratch = torch.empty(6144, device="cuda", dtype=torch.float32)
        candidate_output = torch.empty_like(scratch)
        candidate_bias = torch.empty(6, device="cuda", dtype=torch.float32)
        receipt = execute_r3d1_residual11_staged_v1(
            s0,
            prepared._tensor("param/layer1.1.conv2.weight"),
            prepared._tensor("relu/25/lower").reshape(6, 1024),
            prepared._tensor("relu/25/upper").reshape(6, 1024),
            prepared._tensor("relu/25/alpha"),
            prepared.alpha_maps["25"],
            prepared._tensor("param/layer1.1.conv1.weight"),
            prepared._tensor("param/layer1.1.conv2.bias"),
            prepared._tensor("param/layer1.1.conv1.bias"),
            bias_before,
            scratch,
            candidate_output,
            candidate_bias,
            cache=R3D1Residual11ModuleCacheV1(),
        )
    stream.synchronize()
    candidate_output_cpu = _cpu(candidate_output)
    candidate_bias_cpu = _cpu(candidate_bias)
    values = {
        "reference_output": reference_output,
        "reference_bias": reference_bias,
        "candidate_output": candidate_output_cpu,
        "candidate_bias": candidate_bias_cpu,
    }
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "run_index": args.run_index,
        "source_capture_sha256": _file_hash(args.source_capture),
        "model_sha256": _file_hash(args.model),
        "plan_hash": plan.stable_hash(),
        "trace_hash": trace.stable_hash(),
        "inputs": input_payload,
        **values,
        "tensor_hashes": {
            name: production_tensor_sha256(value) for name, value in values.items()
        },
        "receipt": asdict(receipt),
        "timing_recorded": False,
        "performance_claimed": False,
    }
    torch.save(payload, args.result)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.run_index < 0:
        raise ValueError("R3-D1 worker run index differs")
    payload = _run(args)
    print(
        f"R3-D1-A run={payload['run_index']} timing_recorded=false "
        "performance_claimed=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
