"""FSG4/B4-A terminal lower/lA handoff contracts."""

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
from boundflow.runtime.fsg4_b4a_terminal_lower_adjoint_handoff import (
    assemble_native_backward_from_terminal_handoff_v1,
    execute_terminal_optimizer_with_lower_adjoint_handoff_v1,
    NativeTerminalLowerAdjointLeaseV1,
)
from boundflow.runtime.rvir_v4_native_backward_export import (
    export_rvir_v4_native_backward,
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
def b4a_case():
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
        template_id="resnet2b-prop0-b4a",
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
    schedule = compile_terminal_optimizer_schedule_v1()
    candidate = execute_terminal_optimizer_with_lower_adjoint_handoff_v1(
        module,
        spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        initial_state=instance.initial_state,
        mutation_policy=production.mutation_policy,
        schedule=schedule,
        topology=TOPOLOGY,
        prevalidated_plan=instance,
    )
    control = execute_terminal_optimizer_schedule_v1(
        module,
        spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        initial_state=instance.initial_state,
        mutation_policy=production.mutation_policy,
        schedule=schedule,
        prevalidated_plan=instance,
    )
    control_export = export_rvir_v4_native_backward(
        module=module,
        input_spec=spec,
        linear_spec_C=objective,
        relu_pre=mapping.relu_pre,
        terminal_state=control.terminal_state,
        topology=TOPOLOGY,
        forward_trace=control.forward_trace,
    )
    return {
        "module": module,
        "spec": spec,
        "objective": objective,
        "mapping": mapping,
        "production": production,
        "instance": instance,
        "schedule": schedule,
        "candidate": candidate,
        "control": control,
        "control_export": control_export,
    }


def _assembly(case):
    candidate = case["candidate"]
    result = candidate.optimizer_result
    return assemble_native_backward_from_terminal_handoff_v1(
        module=case["module"],
        relu_pre=case["mapping"].relu_pre,
        terminal_state=result.terminal_state,
        topology=TOPOLOGY,
        forward_trace=result.forward_trace,
        schedule=case["schedule"],
        mutation_policy=case["production"].mutation_policy,
        handoff_lease=NativeTerminalLowerAdjointLeaseV1(candidate.handoff),
    )


def test_b4a_terminal_producer_preserves_b3_state_and_export(b4a_case) -> None:
    candidate = b4a_case["candidate"]
    result = candidate.optimizer_result
    control = b4a_case["control"]
    control_export = b4a_case["control_export"]
    assembly = _assembly(b4a_case)

    assert torch.equal(result.terminal_lower, control.terminal_lower)
    assert result.terminal_state.stable_hash() == control.terminal_state.stable_hash()
    assert (
        assembly.export.metadata()["export_hash"]
        == control_export.metadata()["export_hash"]
    )
    assert candidate.optimizer_evaluation_count == 10
    assert candidate.optimizer_update_count == 9
    assert candidate.terminal_lower_adjoint_handoff_count == 1
    assert candidate.terminal_export_crown_rerun_count == 0
    assert assembly.terminal_lower_adjoint_handoff_count == 1
    assert assembly.terminal_export_crown_rerun_count == 0
    assert assembly.metadata()["performance_claimed"] is False


def test_b4a_lineage_is_bound_to_exact_parent_operator_and_shape(b4a_case) -> None:
    handoff = b4a_case["candidate"].handoff
    assert len(handoff.lower_adjoints) == 6
    assert len(handoff.lineages) == 6
    for item in TOPOLOGY:
        lineage = handoff.lineages[item.native_preactivation]
        coefficient = handoff.lower_adjoints[item.native_preactivation]
        preactivation = b4a_case["mapping"].relu_pre[item.native_preactivation]
        assert lineage.native_preactivation == item.native_preactivation
        assert lineage.provider_activation == item.provider_activation
        assert lineage.provider_preactivation == item.provider_preactivation
        assert lineage.producer_output == item.native_preactivation
        assert lineage.preactivation_shape == tuple(preactivation.lower.shape)
        assert lineage.coefficient_shape == tuple(coefficient.shape)
        assert lineage.coefficient_shape == (
            lineage.preactivation_shape[0],
            1,
            *lineage.preactivation_shape[1:],
        )
        assert lineage.shape_source == "correlation-parent-boundflow-operator"
        assert lineage.kernel_shape_inferred is False
        assert len(lineage.metadata()["lineage_hash"]) == 64


def test_b4a_handoff_is_one_shot_and_lineage_tamper_fails_closed(b4a_case) -> None:
    candidate = b4a_case["candidate"]
    result = candidate.optimizer_result
    lease = NativeTerminalLowerAdjointLeaseV1(candidate.handoff)
    kwargs = {
        "module": b4a_case["module"],
        "relu_pre": b4a_case["mapping"].relu_pre,
        "terminal_state": result.terminal_state,
        "topology": TOPOLOGY,
        "forward_trace": result.forward_trace,
        "schedule": b4a_case["schedule"],
        "mutation_policy": b4a_case["production"].mutation_policy,
        "handoff_lease": lease,
    }
    assemble_native_backward_from_terminal_handoff_v1(**kwargs)
    assert lease.consumed is True
    with pytest.raises(ValueError, match="already consumed"):
        assemble_native_backward_from_terminal_handoff_v1(**kwargs)

    name, lineage = candidate.handoff.lineage_by_native_preactivation[0]
    bad_lineage = replace(lineage, producer_op_ordinal=lineage.producer_op_ordinal + 1)
    bad_handoff = replace(
        candidate.handoff,
        lineage_by_native_preactivation=(
            (name, bad_lineage),
            *candidate.handoff.lineage_by_native_preactivation[1:],
        ),
    )
    with pytest.raises(ValueError, match="operator lineage differs"):
        assemble_native_backward_from_terminal_handoff_v1(
            **{
                **kwargs,
                "handoff_lease": NativeTerminalLowerAdjointLeaseV1(bad_handoff),
            }
        )


def test_b4a_terminal_evaluation_calls_nine_grad_and_one_handoff_runner(
    b4a_case, monkeypatch
) -> None:
    from boundflow.runtime import fsg4_b4a_terminal_lower_adjoint_handoff as runtime

    counts = {"grad": 0, "handoff": 0}
    original_grad = runtime.run_crown_ibp_mlp_from_forward_trace
    original_handoff = (
        runtime.run_crown_ibp_mlp_with_relu_lower_coefficients_from_forward_trace
    )

    def counted_grad(*args, **kwargs):
        counts["grad"] += 1
        return original_grad(*args, **kwargs)

    def counted_handoff(*args, **kwargs):
        counts["handoff"] += 1
        return original_handoff(*args, **kwargs)

    monkeypatch.setattr(runtime, "run_crown_ibp_mlp_from_forward_trace", counted_grad)
    monkeypatch.setattr(
        runtime,
        "run_crown_ibp_mlp_with_relu_lower_coefficients_from_forward_trace",
        counted_handoff,
    )
    execute_terminal_optimizer_with_lower_adjoint_handoff_v1(
        b4a_case["module"],
        b4a_case["spec"],
        linear_spec_C=b4a_case["objective"],
        relu_pre=b4a_case["mapping"].relu_pre,
        initial_state=b4a_case["instance"].initial_state,
        mutation_policy=b4a_case["production"].mutation_policy,
        schedule=b4a_case["schedule"],
        topology=TOPOLOGY,
        prevalidated_plan=b4a_case["instance"],
    )
    assert counts == {"grad": 9, "handoff": 1}
