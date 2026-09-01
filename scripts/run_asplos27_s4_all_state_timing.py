"""Run paired S4 all-state versus dense-PyTorch 10/9 timing observations."""

# pylint: disable=import-error,protected-access,too-many-locals,too-many-statements
# pylint: disable=duplicate-code,missing-function-docstring

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.asplos27_s4_all_state_evaluator import (
    PreparedS4AllStateEvaluatorV1,
)
from boundflow.runtime.asplos27_s4_mutable_state_admission import (
    prepare_s4_mutable_state_admission_v1,
)
from boundflow.runtime.asplos27_s4_optimizer_driver import (
    execute_s4_optimizer_v1,
)
from boundflow.runtime.asplos27_s4_ordered_buffer_abi import (
    prepare_s4_mutable_buffers_v1,
)
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_d2b_staged_backward import (
    PreparedR3D2BStagedBackwardCandidateV1,
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
DEFAULT_MODEL = (
    ROOT.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)
SITES = (17, 19, 23, 25, 28, 31)


def _base(
    capture: Path,
    model: Path,
    *,
    exact_call_id: str,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    raw = torch.load(capture, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
    tensors = bind_r31_runtime_inputs_v1(plan, module, snapshot, device=device)
    executor = PreparedR3D2BStagedBackwardCandidateV1(plan, trace, tensors)
    live = {
        path: snapshot.tensor_map()[path]
        .value.to(device)
        .detach()
        .clone()
        .requires_grad_(True)
        for layout in plan.relu_layouts
        for path in (layout.alpha_path, layout.beta_path)
    }
    admission = prepare_s4_mutable_state_admission_v1(
        snapshot, TOPOLOGY, plan, live, exact_call_id=exact_call_id
    )
    buffers = prepare_s4_mutable_buffers_v1(
        admission, live, exact_call_id=exact_call_id
    )
    return device, executor, buffers


def _measure_candidate(capture: Path, model: Path, ordinal: int):
    device, executor, buffers = _base(
        capture, model, exact_call_id=f"s4-timing-candidate-{ordinal}"
    )
    stream = torch.cuda.Stream(device=device)
    evaluator = PreparedS4AllStateEvaluatorV1(
        executor,
        buffers,
        exact_call_id=f"s4-timing-evaluator-{ordinal}",
        stream=stream,
    )
    torch.cuda.synchronize(device)
    base_allocated = torch.cuda.memory_allocated(device)
    base_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
    result = execute_s4_optimizer_v1(evaluator)
    with torch.cuda.stream(stream):
        end.record()
    end.synchronize()
    return {
        "elapsed_ms": float(start.elapsed_time(end)),
        "allocated_delta_bytes": int(
            torch.cuda.max_memory_allocated(device) - base_allocated
        ),
        "reserved_delta_bytes": int(
            torch.cuda.max_memory_reserved(device) - base_reserved
        ),
        "lower": result.terminal_lower.detach().cpu(),
        "parameters": tuple(
            value.detach().cpu() for value in result.terminal_parameters
        ),
    }


def _dense_mutable(executor: PreparedR3D2BStagedBackwardCandidateV1):
    tensors = list(executor.tensors)
    by_name = {
        spec.name: ordinal for ordinal, spec in enumerate(executor.plan.tensor_specs)
    }
    alphas = []
    for site in SITES:
        ordinal = by_name[f"relu/{site}/alpha"]
        tensors[ordinal] = tensors[ordinal].detach().clone().requires_grad_(True)
        alphas.append(tensors[ordinal])
    beta_ordinal = by_name["relu/31/beta"]
    tensors[beta_ordinal] = tensors[beta_ordinal].detach().clone().requires_grad_(True)
    return tuple(tensors), alphas, tensors[beta_ordinal]


def _measure_native(capture: Path, model: Path, ordinal: int):
    device, executor, _buffers = _base(
        capture, model, exact_call_id=f"s4-timing-native-{ordinal}"
    )
    tensors, alphas, beta = _dense_mutable(executor)
    optimizer = torch.optim.Adam(
        ({"params": alphas, "lr": 0.01}, {"params": [beta], "lr": 0.05})
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(device)
    base_allocated = torch.cuda.memory_allocated(device)
    base_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    terminal = None
    with torch.cuda.stream(stream):
        start.record()
        for evaluation in range(10):
            lower = _evaluate_full_region(executor.plan, tensors).reshape(6)
            if evaluation < 9:
                optimizer.zero_grad(set_to_none=True)
                (-lower.sum()).backward()
                optimizer.step()
                with torch.no_grad():
                    for alpha in alphas:
                        alpha.clamp_(0.0, 1.0)
                    beta.clamp_(min=0.0)
                scheduler.step()
            else:
                terminal = lower.detach().clone()
        end.record()
    end.synchronize()
    if terminal is None:
        raise RuntimeError("native terminal lower is absent")
    return {
        "elapsed_ms": float(start.elapsed_time(end)),
        "allocated_delta_bytes": int(
            torch.cuda.max_memory_allocated(device) - base_allocated
        ),
        "reserved_delta_bytes": int(
            torch.cuda.max_memory_reserved(device) - base_reserved
        ),
        "lower": terminal.detach().cpu(),
        "parameters": tuple(
            [value.detach().cpu()[0, 0] for value in alphas] + [beta.detach().cpu()]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pairs < 3 or not args.capture.is_file() or not args.model.is_file():
        raise SystemExit("pairs>=3 and existing capture/model are required")

    # Both paths pay one discarded process-local warm run before paired data.
    # The candidate already executes an evaluator warmup during prepare; this
    # additionally removes PyTorch/autograd and CUDA library first-use skew.
    warm_native = _measure_native(args.capture, args.model, -1)
    warm_candidate = _measure_candidate(args.capture, args.model, -1)
    rows = []
    for ordinal in range(args.pairs):
        if ordinal % 2:
            candidate = _measure_candidate(args.capture, args.model, ordinal)
            native = _measure_native(args.capture, args.model, ordinal)
            order = "candidate-native"
        else:
            native = _measure_native(args.capture, args.model, ordinal)
            candidate = _measure_candidate(args.capture, args.model, ordinal)
            order = "native-candidate"
        lower_diff = float((native["lower"] - candidate["lower"]).abs().max().item())
        parameter_diff = max(
            float((left - right).abs().max().item())
            for left, right in zip(native["parameters"], candidate["parameters"])
        )
        rows.append(
            {
                "pair_ordinal": ordinal,
                "order": order,
                "native_ms": native["elapsed_ms"],
                "candidate_ms": candidate["elapsed_ms"],
                "speedup": native["elapsed_ms"] / candidate["elapsed_ms"],
                "native_allocated_delta_bytes": native["allocated_delta_bytes"],
                "candidate_allocated_delta_bytes": candidate["allocated_delta_bytes"],
                "native_reserved_delta_bytes": native["reserved_delta_bytes"],
                "candidate_reserved_delta_bytes": candidate["reserved_delta_bytes"],
                "lower_max_abs_diff": lower_diff,
                "parameter_max_abs_diff": parameter_diff,
            }
        )
    speedups = [row["speedup"] for row in rows]
    payload = {
        "schema": "boundflow.asplos27-s4-all-state-timing-observation/v1",
        "pair_count": len(rows),
        "scope": "prepared-10-evaluation-9-mutation-region-wrapper",
        "compile_and_prepare_included": False,
        "complete_query_claimed": False,
        "performance_claimed": False,
        "warmup_discarded": {
            "native_ms": warm_native["elapsed_ms"],
            "candidate_ms": warm_candidate["elapsed_ms"],
        },
        "rows": rows,
        "summary": {
            "native_median_ms": statistics.median(row["native_ms"] for row in rows),
            "candidate_median_ms": statistics.median(
                row["candidate_ms"] for row in rows
            ),
            "speedup_geomean": math.prod(speedups) ** (1.0 / len(speedups)),
            "speedup_worst": min(speedups),
            "lower_max_abs_diff": max(row["lower_max_abs_diff"] for row in rows),
            "parameter_max_abs_diff": max(
                row["parameter_max_abs_diff"] for row in rows
            ),
            "native_allocated_delta_bytes": max(
                row["native_allocated_delta_bytes"] for row in rows
            ),
            "candidate_allocated_delta_bytes": max(
                row["candidate_allocated_delta_bytes"] for row in rows
            ),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
