"""R3-1b0 exact reverse trace and two-scratch liveness tests."""

# pylint: disable=missing-function-docstring

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.ir.r3_bounded_arena import R31BStepKind
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_structured_owner_custom_backward import (
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
CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)


def _trace_objects():  # type: ignore[no-untyped-def]
    if not MODEL.is_file():
        pytest.skip("frozen ResNet2B checkout is unavailable")
    raw = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(MODEL), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    production_plan = compile_r31_full_region_plan_v1(
        module, snapshot, mapping, TOPOLOGY
    )
    trace = compile_r31b_bounded_arena_trace_v1(program, module, production_plan)
    return program, module, production_plan, trace


def test_r31b0_compiles_exact_reverse_trace_with_two_scratch_slots() -> None:
    _program, _module, production_plan, trace = _trace_objects()

    assert trace.production_plan_hash == production_plan.stable_hash()
    assert trace.scratch_slot_count == 2
    assert trace.scratch_capacity_elements == 6 * 1 * 3 * 32 * 32
    assert trace.scratch_capacity_elements * 4 == 73_728
    assert tuple(step.kind for step in trace.steps) == (
        R31BStepKind.SPEC_SEED,
        R31BStepKind.LINEAR_RIGHT,
        R31BStepKind.RELU_LOWER,
        R31BStepKind.LINEAR_RIGHT,
        R31BStepKind.RESHAPE_VIEW,
        R31BStepKind.RELU_LOWER,
        R31BStepKind.RESIDUAL_REGION,
        R31BStepKind.RELU_LOWER,
        R31BStepKind.RESIDUAL_REGION,
        R31BStepKind.RELU_LOWER,
        R31BStepKind.CONV2D_RIGHT,
        R31BStepKind.INPUT_CONCRETIZE,
    )
    assert tuple((step.input_slot, step.output_slot) for step in trace.steps) == (
        (-1, 0),
        (0, 1),
        (1, 1),
        (1, 0),
        (0, 0),
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 0),
        (0, 0),
        (0, 1),
        (1, -1),
    )
    assert trace.compiled_region is False
    assert trace.timing_recorded is False
    assert trace.performance_claimed is False


def test_r31b0_residual_regions_freeze_branch_and_join_ownership() -> None:
    _program, _module, _production_plan, trace = _trace_objects()
    residuals = [
        step for step in trace.steps if step.kind == R31BStepKind.RESIDUAL_REGION
    ]

    assert len(residuals) == 2
    assert residuals[0].input_value == "28"
    assert residuals[0].output_value == "24"
    assert tuple(branch.primal_ops for branch in residuals[0].branches) == (
        ("Conv_10", "Relu_9", "Conv_8"),
        (),
    )
    assert residuals[1].input_value == "23"
    assert residuals[1].output_value == "18"
    assert tuple(branch.primal_ops for branch in residuals[1].branches) == (
        ("Conv_4", "Relu_3", "Conv_2"),
        ("Conv_5",),
    )
    assert all(step.accumulate_into_output for step in residuals)


def test_r31b0_slot_branch_shape_and_primal_order_tamper_fail_closed() -> None:
    program, module, production_plan, trace = _trace_objects()
    steps = list(trace.steps)
    steps[6] = replace(steps[6], output_slot=0, in_place=True)
    with pytest.raises(ValueError, match="residual step"):
        replace(trace, steps=tuple(steps)).validate()

    steps = list(trace.steps)
    bad_branch = replace(steps[6].branches[0], join_value="wrong")
    steps[6] = replace(steps[6], branches=(bad_branch, steps[6].branches[1]))
    with pytest.raises(ValueError, match="residual step"):
        replace(trace, steps=tuple(steps)).validate()

    steps = list(trace.steps)
    steps[10] = replace(steps[10], output_shape=(6, 1, 3, 31, 32))
    with pytest.raises(ValueError, match="trace differs"):
        replace(trace, steps=tuple(steps)).validate()

    changed = copy.copy(module)
    task = copy.copy(module.get_entry_task())
    task.ops = list(reversed(task.ops))
    changed.tasks = [task]
    with pytest.raises(ValueError, match="graph identity"):
        compile_r31b_bounded_arena_trace_v1(program, changed, production_plan)
