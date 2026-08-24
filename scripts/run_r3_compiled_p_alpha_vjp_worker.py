#!/usr/bin/env python3
"""Run one fresh-process R3-1b2 native/compiled lower+dalpha pair."""

# pylint: disable=wrong-import-position,too-many-locals,missing-function-docstring

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
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
    execute_r31b2_compiled_custom_backward_v1,
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
)
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

WORKER_SCHEMA = "boundflow.r3-1b2-compiled-p-alpha-vjp-worker/v1"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    return hashlib.sha256(cpu.numpy().tobytes(order="C")).hexdigest()


def _metric(candidate: torch.Tensor, native: torch.Tensor) -> dict[str, object]:
    candidate_cpu = candidate.detach().contiguous().cpu()
    native_cpu = native.detach().contiguous().cpu()
    return {
        "candidate": [float(value) for value in candidate_cpu.flatten()],
        "native": [float(value) for value in native_cpu.flatten()],
        "candidate_sha256": _tensor_sha256(candidate_cpu),
        "native_sha256": _tensor_sha256(native_cpu),
        "element_count": candidate_cpu.numel(),
        "max_abs_diff": float((candidate_cpu - native_cpu).abs().max().item()),
        "allclose": bool(
            torch.allclose(candidate_cpu, native_cpu, atol=2e-4, rtol=2e-4)
        ),
        "finite": bool(torch.isfinite(candidate_cpu).all().item()),
        "sign_exact": bool(
            torch.equal(torch.sign(candidate_cpu), torch.sign(native_cpu))
        ),
        "candidate_nonzero": int(torch.count_nonzero(candidate_cpu).item()),
        "native_nonzero": int(torch.count_nonzero(native_cpu).item()),
        "atol": 2e-4,
        "rtol": 2e-4,
    }


def run(source_capture: Path, model: Path) -> dict[str, object]:
    raw = torch.load(source_capture, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
    tensors = bind_r31_runtime_inputs_v1(
        plan, module, snapshot, device=torch.device("cuda:0")
    )
    native_lower, native_gradient = execute_r31_native_oracle_v1(plan, tensors)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        candidate = execute_r31b2_compiled_custom_backward_v1(plan, trace, tensors)
    stream.synchronize()
    properties = torch.cuda.get_device_properties(0)
    return {
        "schema_version": WORKER_SCHEMA,
        "source_capture_sha256": _file_sha256(source_capture),
        "model_sha256": _file_sha256(model),
        "trace_hash": trace.stable_hash(),
        "production_plan_hash": plan.stable_hash(),
        "lower_metric": _metric(candidate.final_lower, native_lower),
        "gradient_metric": _metric(
            candidate.compressed_alpha_gradient, native_gradient
        ),
        "execution_receipt": asdict(candidate.receipt),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": properties.name,
            "compute_capability": f"sm_{properties.major}{properties.minor}",
            "python": sys.version.split()[0],
        },
        "compiled_full_lower": True,
        "compiled_vjp": True,
        "custom_vjp": True,
        "r3_1_admitted": False,
        "timing_recorded": False,
        "performance_claimed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    print(_canonical(run(args.source_capture, args.model)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
