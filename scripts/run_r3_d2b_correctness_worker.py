#!/usr/bin/env python3
"""Run one fresh D1-C or D2-B stepwise correctness trajectory."""

# pylint: disable=wrong-import-position,protected-access,too-many-locals
# pylint: disable=too-many-statements,missing-function-docstring,duplicate-code

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.fsg4_b3_terminal_optimizer_schedule import (
    compile_terminal_optimizer_schedule_v1,
)
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_d1c_cumulative_wrapper import (
    PreparedR3D1CCumulativeCandidateV1,
)
from boundflow.runtime.r3_d2b_staged_backward import (
    PreparedR3D2BStagedBackwardCandidateV1,
)
from boundflow.runtime.r3_optimizer_trajectory_timing import _candidate_evaluate
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

WORKER_SCHEMA = "boundflow.r3-d2b-correctness-worker/v1"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous().clone()


def _optimizer_state(
    optimizer: torch.optim.Optimizer, alpha: torch.Tensor
) -> tuple[float, torch.Tensor, torch.Tensor]:
    state = optimizer.state.get(alpha)
    if not state:
        raise RuntimeError("R3-D2B optimizer state is absent")
    raw_step = state.get("step")
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    if not torch.is_tensor(exp_avg) or not torch.is_tensor(exp_avg_sq):
        raise TypeError("R3-D2B optimizer moment differs")
    step = float(raw_step.item() if torch.is_tensor(raw_step) else raw_step)
    return step, _clone(exp_avg), _clone(exp_avg_sq)


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("R3-D2B correctness worker requires CUDA")
    device = torch.device("cuda:0")
    source = torch.load(args.source_capture, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(source["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
    tensors = bind_r31_runtime_inputs_v1(plan, module, snapshot, device=device)
    alpha = tensors[plan.p_alpha_input_ordinal]
    mode = args.mode
    if mode == "d1c":
        prepared = PreparedR3D1CCumulativeCandidateV1(plan, trace, tensors)
    else:
        prepared = PreparedR3D2BStagedBackwardCandidateV1(plan, trace, tensors)
    schedule = compile_terminal_optimizer_schedule_v1()
    trajectory_id = _hash(
        {
            "plan_hash": plan.stable_hash(),
            "trace_hash": trace.stable_hash(),
            "initial_alpha": production_tensor_sha256(alpha),
            "schedule_hash": schedule.stable_hash(),
            "mode_independent": True,
        }
    )
    optimizer = torch.optim.Adam([alpha], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    stream = torch.cuda.Stream(device=device)
    initial_alpha = _clone(alpha)
    steps = []
    prepared.begin_sample()
    for action in schedule.actions:
        lr = float(optimizer.param_groups[0]["lr"])
        if abs(lr - action.alpha_learning_rate) > 1e-15:
            raise ValueError("R3-D2B learning rate differs")
        alpha_before = _clone(alpha)
        with torch.cuda.stream(stream):
            lower, gradient = _candidate_evaluate(prepared, action.evaluation_ordinal)
            lower_raw = _clone(lower)
            gradient_raw = _clone(gradient)
            if action.update_after:
                optimizer.zero_grad(set_to_none=True)
                alpha.grad = gradient.detach().clone()
                optimizer.step()
                with torch.no_grad():
                    alpha.clamp_(0.0, 1.0)
                scheduler.step()
            alpha_after = _clone(alpha)
        stream.synchronize()
        optimizer_step, exp_avg, exp_avg_sq = _optimizer_state(optimizer, alpha)
        d1c_receipt = prepared.d1c_receipt().__dict__
        d2b_receipt = (
            prepared.d2b_receipt().__dict__
            if isinstance(prepared, PreparedR3D2BStagedBackwardCandidateV1)
            else None
        )
        raw_step = {
            "evaluation_ordinal": action.evaluation_ordinal,
            "update_after": action.update_after,
            "alpha_learning_rate": action.alpha_learning_rate,
            "alpha_before": alpha_before,
            "lower": lower_raw,
            "gradient": gradient_raw,
            "alpha_after": alpha_after,
            "optimizer_step": optimizer_step,
            "optimizer_exp_avg": exp_avg,
            "optimizer_exp_avg_sq": exp_avg_sq,
            "d1c_receipt": d1c_receipt,
            "d2b_receipt": d2b_receipt,
        }
        raw_step["tensor_hashes"] = {
            name: production_tensor_sha256(raw_step[name])  # type: ignore[arg-type]
            for name in (
                "alpha_before",
                "lower",
                "gradient",
                "alpha_after",
                "optimizer_exp_avg",
                "optimizer_exp_avg_sq",
            )
        }
        steps.append(raw_step)
    terminal_alpha = _clone(alpha)
    metadata: dict[str, object] = {
        "trajectory_id": trajectory_id,
        "initial_alpha_sha256": production_tensor_sha256(initial_alpha),
        "terminal_alpha_sha256": production_tensor_sha256(terminal_alpha),
        "step_hashes": [
            _hash(
                {
                    name: value
                    for name, value in step.items()
                    if name
                    not in {
                        "alpha_before",
                        "lower",
                        "gradient",
                        "alpha_after",
                        "optimizer_exp_avg",
                        "optimizer_exp_avg_sq",
                    }
                }
            )
            for step in steps
        ],
        "evaluation_count": 10,
        "optimizer_mutation_count": 9,
        "scheduler_mutation_count": 9,
        "timing_recorded": False,
        "performance_claimed": False,
    }
    metadata["trajectory_hash"] = _hash(metadata)
    properties = torch.cuda.get_device_properties(device)
    payload: dict[str, object] = {
        "schema_version": WORKER_SCHEMA,
        "run_index": args.run_index,
        "mode": mode,
        "source_capture_sha256": _file_hash(args.source_capture),
        "model_sha256": _file_hash(args.model),
        "plan_hash": plan.stable_hash(),
        "trace_hash": trace.stable_hash(),
        "metadata": metadata,
        "initial_alpha": initial_alpha,
        "terminal_alpha": terminal_alpha,
        "steps": steps,
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
    parser.add_argument("--mode", choices=("d1c", "d2b"), required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.run_index not in range(5):
        raise ValueError("R3-D2B run index differs")
    payload = _run(args)
    print(
        f"R3-D2B run={args.run_index} mode={args.mode} "
        f"trajectory={str(payload['metadata']['trajectory_hash'])[:12]} "  # type: ignore[index]
        "performance_claimed=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
