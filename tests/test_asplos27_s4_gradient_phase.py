"""S4-1C real production-order Pass-C and terminal lease gates."""

# pylint: disable=missing-function-docstring,too-many-locals,duplicate-code
# pylint: disable=protected-access,not-callable

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.backends.tvm.asplos27_s4_six_site_value import (
    compile_s4_six_site_value_v1,
)
from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.asplos27_s4_coefficient_selector_pass import (
    PreparedS4CoefficientSelectorPassV1,
    capture_r31b2_production_selectors_v1,
)
from boundflow.runtime.asplos27_s4_gradient_emitters import (
    PreparedS4GradientEmittersV1,
    S4_NONTERMINAL_ACTIONS_V1,
    S4_TERMINAL_ACTIONS_V1,
    S4GradientRuntimeError,
)
from boundflow.runtime.asplos27_s4_mutable_state_admission import (
    prepare_s4_mutable_state_admission_v1,
)
from boundflow.runtime.asplos27_s4_ordered_buffer_abi import (
    prepare_s4_mutable_buffers_v1,
)
from boundflow.runtime.asplos27_s4_six_site_value import PreparedS4SixSiteValueV1
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_d2b_staged_backward import (
    PreparedR3D2BStagedBackwardCandidateV1,
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
)
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)


def _fixture():
    if not torch.cuda.is_available() or not CAPTURE.is_file() or not MODEL.is_file():
        pytest.skip("S4-1C production CUDA fixture is unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    raw = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(MODEL), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
    tensors = bind_r31_runtime_inputs_v1(plan, module, snapshot, device=device)
    executor = PreparedR3D2BStagedBackwardCandidateV1(plan, trace, tensors)
    by_name = {spec.name: tensor for spec, tensor in zip(plan.tensor_specs, tensors)}
    live = {
        path: snapshot.tensor_map()[path]
        .value.to(device)
        .detach()
        .clone()
        .requires_grad_(True)
        for layout in plan.relu_layouts
        for path in (layout.alpha_path, layout.beta_path)
    }
    call_id = "s4-1c-production-correctness"
    admission = prepare_s4_mutable_state_admission_v1(
        snapshot, TOPOLOGY, plan, live, exact_call_id=call_id
    )
    buffers = prepare_s4_mutable_buffers_v1(admission, live, exact_call_id=call_id)
    owner = PreparedS4CoefficientSelectorPassV1(
        device=device,
        exact_call_id=call_id,
        evaluation_generation=71,
        parameter_generation=72,
        coefficient_generation=73,
        selector_generation=74,
    )
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        capture_r31b2_production_selectors_v1(executor, owner)
    stream.synchronize()

    def tensor(name: str) -> torch.Tensor:
        return by_name[name]

    reads = (
        tensor("input/lower"),
        tensor("input/upper"),
        owner.selector("endpoint_ainput_v2"),
        tensor("param/conv1.weight"),
        tensor("param/conv1.bias"),
        tensor("relu/17/lower"),
        tensor("relu/17/upper"),
        tensor("relu/17/alpha")[0, 0].contiguous(),
        executor.forward_executor.alpha_maps["17"],
        owner.selector("sign_a18"),
        tensor("param/layer1.0.conv1.weight"),
        tensor("param/layer1.0.conv1.bias"),
        tensor("relu/19/lower"),
        tensor("relu/19/upper"),
        tensor("relu/19/alpha")[0, 0].contiguous(),
        executor.forward_executor.alpha_maps["19"],
        owner.selector("sign_a20"),
        tensor("param/layer1.0.conv2.weight"),
        tensor("param/layer1.0.conv2.bias"),
        tensor("param/layer1.0.shortcut.0.weight"),
        tensor("param/layer1.0.shortcut.0.bias"),
        tensor("relu/23/lower"),
        tensor("relu/23/upper"),
        tensor("relu/23/alpha")[0, 0].contiguous(),
        executor.forward_executor.alpha_maps["23"],
        owner.selector("sign_a24"),
        tensor("param/layer1.1.conv1.weight"),
        tensor("param/layer1.1.conv1.bias"),
        tensor("relu/25/lower"),
        tensor("relu/25/upper"),
        tensor("relu/25/alpha")[0, 0].contiguous(),
        executor.forward_executor.alpha_maps["25"],
        owner.selector("sign_a26"),
        tensor("param/layer1.1.conv2.weight"),
        tensor("param/layer1.1.conv2.bias"),
        tensor("relu/28/lower"),
        tensor("relu/28/upper"),
        tensor("relu/28/alpha")[0, 0].contiguous(),
        executor.forward_executor.alpha_maps["28"],
        owner.selector("sign_a29"),
        tensor("param/linear1.weight"),
        tensor("param/linear1.bias"),
    )
    coefficient_arena = executor.forward_executor.scratch_1
    value_runtime = PreparedS4SixSiteValueV1(
        compile_s4_six_site_value_v1(device_index=device.index or 0),
        reads,
        coefficient_arena=coefficient_arena,
        selected_input_alias=coefficient_arena[:18432].view(6, 3, 32, 32),
        device=device,
    )
    with torch.cuda.stream(stream):
        value_runtime.begin_pass_a()
        value_runtime.adopt_selectors(owner)
        value_runtime.run_pass_b()
        value_result = value_runtime.handoff_to_coefficient_recompute()
    stream.synchronize()
    return device, stream, executor, buffers, value_result


@pytest.mark.parametrize("terminal", [False, True])
def test_s4_gradient_real_six_site_action_inventory_and_outputs(terminal: bool) -> None:
    device, stream, executor, buffers, value_result = _fixture()
    runtime = PreparedS4GradientEmittersV1(
        executor,
        value_result,
        buffers,
        evaluation_generation=9 if terminal else 4,
        state_version=81,
    )
    value_before = tuple(value.detach().clone() for value in value_result.values)
    with torch.cuda.stream(stream):
        result = runtime.run(terminal=terminal)
    stream.synchronize()
    result.validate()
    assert result.receipt.action_inventory == (
        S4_TERMINAL_ACTIONS_V1 if terminal else S4_NONTERMINAL_ACTIONS_V1
    )
    assert result.receipt.action_count == (23 if terminal else 17)
    assert result.receipt.dalpha_launch_count == 6
    assert result.receipt.dbeta_launch_count == 1
    assert result.receipt.terminal_copy_count == (6 if terminal else 0)
    assert result.receipt.prepare_dlpack_view_count == 46
    assert result.receipt.value_arena_physical_storage_count == 1
    assert all(torch.isfinite(gradient).all() for gradient in result.gradients)
    assert sum(gradient.numel() for gradient in result.gradients) == 4254
    if terminal:
        lease = runtime.take_terminal_lease()
        terminal_la = lease.consume(evaluation_generation=9)
        assert tuple(tuple(value.shape) for value in terminal_la) == (
            (6, 1, 8, 16, 16),
            (6, 1, 16, 8, 8),
            (6, 1, 16, 8, 8),
            (6, 1, 16, 8, 8),
            (6, 1, 16, 8, 8),
            (6, 1, 100),
        )
        assert len({int(value.untyped_storage()._cdata) for value in terminal_la}) == 1
        for ordinal, (actual, incoming, adjoint, layout) in enumerate(
            zip(
                result.gradients[:6],
                terminal_la,
                value_before,
                executor.plan.relu_layouts,
            )
        ):
            indices = torch.tensor(
                layout.alpha_flat_indices, device=device, dtype=torch.int64
            )
            a_value = incoming.reshape(6, -1).index_select(1, indices)
            v_value = adjoint.reshape(6, -1).index_select(1, indices)
            lower = (
                executor._tensor(f"relu/{layout.native_preactivation}/lower")
                .reshape(6, -1)
                .index_select(1, indices)
            )
            upper = (
                executor._tensor(f"relu/{layout.native_preactivation}/upper")
                .reshape(6, -1)
                .index_select(1, indices)
            )
            reference = torch.where(
                (lower < 0) & (upper > 0) & (a_value >= 0),
                -a_value * v_value,
                torch.zeros_like(a_value),
            )
            torch.testing.assert_close(
                actual,
                reference,
                rtol=2e-5,
                atol=2e-5,
                msg=f"site ordinal {ordinal} differs",
            )
        location = torch.tensor(
            [17, 17, 31, 17, 17, 31], device=device, dtype=torch.int64
        ).view(6, 1)
        sign = torch.tensor(
            [1, 1, 1, -1, -1, -1], device=device, dtype=torch.float32
        ).view(6, 1)
        v31 = value_before[-1].reshape(6, 100).gather(1, location)
        torch.testing.assert_close(result.gradients[6], v31 * sign)
        with pytest.raises(S4GradientRuntimeError, match="S4_TERMINAL_LA_LEASE_REUSED"):
            lease.consume(evaluation_generation=9)
    else:
        with pytest.raises(
            S4GradientRuntimeError, match="S4_TERMINAL_LA_INVENTORY_INCOMPLETE"
        ):
            runtime.take_terminal_lease()
    buffers.close()


def test_s4_gradient_receipt_rejects_claim_and_inventory_drift() -> None:
    _, stream, executor, buffers, value_result = _fixture()
    runtime = PreparedS4GradientEmittersV1(
        executor, value_result, buffers, evaluation_generation=3, state_version=7
    )
    with torch.cuda.stream(stream):
        result = runtime.run(terminal=False)
    stream.synchronize()
    mutations = (
        replace(result.receipt, action_count=16),
        replace(result.receipt, coefficient_action_count=9),
        replace(result.receipt, dalpha_launch_count=5),
        replace(result.receipt, dbeta_launch_count=0),
        replace(result.receipt, emitter_unique_view_count=45),
        replace(result.receipt, full_prepared_descriptor_union_count=109),
        replace(result.receipt, dynamic_output_allocation_count=1),
        replace(result.receipt, saved_dense_a_count=1),
        replace(result.receipt, fallback_count=1),
        replace(result.receipt, timing_recorded=True),
        replace(result.receipt, performance_claimed=True),
    )
    for changed in mutations:
        with pytest.raises(
            S4GradientRuntimeError, match="S4_GRADIENT_RECEIPT_MISMATCH"
        ):
            changed.validate()
    buffers.close()
