"""FSG4/B3-B terminal-only Schedule IR and forward-handoff contracts."""

# pylint: disable=missing-function-docstring,redefined-outer-name,duplicate-code

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.fsg4_b3_prepared_core import (
    instantiate_core_plan_v1,
    prepare_core_template_v1,
)
from boundflow.runtime.fsg4_b3_terminal_optimizer_schedule import (
    compile_terminal_optimizer_schedule_v1,
    execute_terminal_optimizer_schedule_v1,
)
from boundflow.runtime.rvir_v4_native_backward_export import (
    export_rvir_v4_native_backward,
)
from boundflow.runtime.rvir_v4_native_optimizer import (
    execute_rvir_v4_native_optimizer_trace,
)
from boundflow.runtime.rvir_v4_optimizer_mutation import (
    production_optimizer_step_trace_from_payload_v4,
)
from boundflow.runtime.rvir_v4_pre_state_initializer import (
    initialize_rvir_v4_native_pre_state,
)
from boundflow.runtime.rvir_v4_production_state import (
    production_snapshot_from_payload_v4,
    ProductionTensorOwnership,
    ProductionTensorRole,
)
from boundflow.runtime.task_executor import InputSpec
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = ROOT.parent / "vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"


@pytest.fixture(scope="module")
def schedule_case():
    capture = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(capture["cores"][0]["pre_snapshot"])
    production = production_optimizer_step_trace_from_payload_v4(
        capture["optimizer_step_traces"][0]
    )
    program = import_onnx(str(MODEL), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)

    def one(role):
        values = [item.value for item in snapshot.tensors if item.role == role]
        assert len(values) == 1
        return values[0]

    lower = one(ProductionTensorRole.INPUT_LOWER)
    upper = one(ProductionTensorRole.INPUT_UPPER)
    objective = one(ProductionTensorRole.LINEAR_SPEC)
    mutable_paths = tuple(
        sorted(
            item.semantic_path
            for item in snapshot.tensors
            if item.ownership == ProductionTensorOwnership.MUTABLE_COPY_OUT
        )
    )
    template = prepare_core_template_v1(
        template_id="resnet2b-prop0-b3-b",
        program=program,
        module=module,
        topology=TOPOLOGY,
        device="cpu",
        dtype=torch.float32,
        input_shape=lower.shape,
        objective_shape=objective.shape,
        mutable_paths=mutable_paths,
    )
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
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
    return {
        "module": module,
        "spec": spec,
        "objective": objective,
        "mapping": mapping,
        "production": production,
        "instance": instance,
    }


def _terminal(schedule_case):
    schedule = compile_terminal_optimizer_schedule_v1()
    result = execute_terminal_optimizer_schedule_v1(
        schedule_case["module"],
        schedule_case["spec"],
        linear_spec_C=schedule_case["objective"],
        relu_pre=schedule_case["mapping"].relu_pre,
        initial_state=schedule_case["instance"].initial_state,
        mutation_policy=schedule_case["production"].mutation_policy,
        schedule=schedule,
        prevalidated_plan=schedule_case["instance"],
    )
    return schedule, result


def test_terminal_schedule_is_exact_ten_evaluate_nine_update_ir() -> None:
    schedule = compile_terminal_optimizer_schedule_v1()
    assert schedule.evaluation_count == 10
    assert schedule.update_count == 9
    assert tuple(action.evaluation_ordinal for action in schedule.actions) == tuple(
        range(10)
    )
    assert tuple(action.update_after for action in schedule.actions) == (True,) * 9 + (
        False,
    )
    assert len(schedule.stable_hash()) == 64


def test_terminal_result_matches_formal_trace_last_step(schedule_case) -> None:
    schedule, result = _terminal(schedule_case)
    formal = execute_rvir_v4_native_optimizer_trace(
        schedule_case["module"],
        schedule_case["spec"],
        linear_spec_C=schedule_case["objective"],
        relu_pre=schedule_case["mapping"].relu_pre,
        initial_state=schedule_case["instance"].initial_state,
        mutation_policy=schedule_case["production"].mutation_policy,
        prevalidated_plan=schedule_case["instance"],
    )
    last = formal.steps[-1]
    assert torch.equal(result.terminal_lower, last.lower)
    assert result.terminal_state.alphas.keys() == last.alphas.keys()
    assert result.terminal_state.betas.keys() == last.betas.keys()
    for name in last.alphas:
        assert torch.equal(result.terminal_state.alphas[name], last.alphas[name])
        assert torch.equal(result.terminal_state.betas[name], last.betas[name])
    metadata = result.metadata(module=schedule_case["module"], schedule=schedule)
    assert metadata["evaluation_count"] == 10
    assert metadata["update_count"] == 9
    assert metadata["full_step_snapshot_count"] == 0
    assert metadata["performance_claimed"] is False


def test_backward_reuses_optimizer_forward_trace_exactly(schedule_case) -> None:
    _schedule, result = _terminal(schedule_case)
    baseline = export_rvir_v4_native_backward(
        module=schedule_case["module"],
        input_spec=schedule_case["spec"],
        linear_spec_C=schedule_case["objective"],
        relu_pre=schedule_case["mapping"].relu_pre,
        terminal_state=result.terminal_state,
        topology=TOPOLOGY,
    )
    reused = export_rvir_v4_native_backward(
        module=schedule_case["module"],
        input_spec=schedule_case["spec"],
        linear_spec_C=schedule_case["objective"],
        relu_pre=schedule_case["mapping"].relu_pre,
        terminal_state=result.terminal_state,
        topology=TOPOLOGY,
        forward_trace=result.forward_trace,
    )
    assert torch.equal(reused.lower, baseline.lower)
    for name in baseline.l_as:
        assert torch.equal(reused.l_as[name], baseline.l_as[name])
    for name in baseline.intermediates:
        assert torch.equal(
            reused.intermediates[name].lower, baseline.intermediates[name].lower
        )
        assert torch.equal(
            reused.intermediates[name].upper, baseline.intermediates[name].upper
        )


def test_terminal_schedule_and_forward_tampering_fail_closed(schedule_case) -> None:
    schedule, result = _terminal(schedule_case)
    bad_action = replace(
        schedule.actions[2],
        alpha_learning_rate=schedule.actions[2].alpha_learning_rate * 2.0,
    )
    with pytest.raises(ValueError, match="action sequence differs"):
        replace(
            schedule,
            actions=schedule.actions[:2] + (bad_action,) + schedule.actions[3:],
        ).validate()
    bad_trace = replace(result.forward_trace, split_state_hash="f" * 64)
    with pytest.raises(ValueError, match="forward trace identity differs"):
        export_rvir_v4_native_backward(
            module=schedule_case["module"],
            input_spec=schedule_case["spec"],
            linear_spec_C=schedule_case["objective"],
            relu_pre=schedule_case["mapping"].relu_pre,
            terminal_state=result.terminal_state,
            topology=TOPOLOGY,
            forward_trace=bad_trace,
        )
