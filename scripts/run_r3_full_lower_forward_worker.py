#!/usr/bin/env python3
"""Run one fresh-process R3-1b1 native/compiled lower-only pair."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=missing-function-docstring

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
from boundflow.runtime.r3_full_lower_forward_tir import (
    PreparedR31B1FullLowerForwardV1,
    source_receipt_hash,
)
from boundflow.runtime.r3_structured_owner_custom_backward import (
    _evaluate_full_region,
    bind_r31_runtime_inputs_v1,
    compile_r31_full_region_plan_v1,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
)
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

WORKER_SCHEMA = "boundflow.r3-1b1-full-lower-worker/v1"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    with torch.no_grad():
        native = _evaluate_full_region(plan, tensors).detach()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        candidate = PreparedR31B1FullLowerForwardV1(plan, trace, tensors).run()
    stream.synchronize()
    native_cpu = native.detach().cpu()
    candidate_cpu = candidate.lower.detach().cpu()
    properties = torch.cuda.get_device_properties(0)
    metric = {
        "native_lower": [float(value) for value in native_cpu.flatten()],
        "candidate_lower": [float(value) for value in candidate_cpu.flatten()],
        "max_abs_diff": float((native_cpu - candidate_cpu).abs().max().item()),
        "allclose": bool(
            torch.allclose(candidate_cpu, native_cpu, atol=2e-4, rtol=2e-4)
        ),
        "finite": bool(torch.isfinite(candidate_cpu).all().item()),
        "sign_exact": bool(
            torch.equal(torch.sign(candidate_cpu), torch.sign(native_cpu))
        ),
        "atol": 2e-4,
        "rtol": 2e-4,
    }
    result = {
        "schema_version": WORKER_SCHEMA,
        "source_capture_sha256": _sha256(source_capture),
        "model_sha256": _sha256(model),
        "trace_hash": trace.stable_hash(),
        "production_plan_hash": plan.stable_hash(),
        "metric": metric,
        "compilation_receipt": asdict(candidate.compilation_receipt),
        "compilation_receipt_hash": source_receipt_hash(candidate.compilation_receipt),
        "launch_receipt": asdict(candidate.launch_receipt),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": properties.name,
            "compute_capability": f"sm_{properties.major}{properties.minor}",
            "python": sys.version.split()[0],
        },
        "compiled_full_lower": True,
        "custom_vjp": False,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    print(_canonical(run(args.source_capture, args.model)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
