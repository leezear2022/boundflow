"""R3-1 production-shaped full-region custom backward tests."""

# pylint: disable=missing-function-docstring,too-many-locals

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.r3_structured_owner_custom_backward import (
    bind_r31_runtime_inputs_v1,
    compile_r31_full_region_plan_v1,
    execute_r31_custom_backward_v1,
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

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)


def _snapshot():  # type: ignore[no-untyped-def]
    raw = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    return production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])


def _module():  # type: ignore[no-untyped-def]
    if not MODEL.is_file():
        pytest.skip("frozen ResNet2B checkout is unavailable")
    program = import_onnx(str(MODEL), do_shape_infer=True, normalize=True)
    return plan_interval_ibp_v0(program)


def _plan_objects():  # type: ignore[no-untyped-def]
    snapshot = _snapshot()
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    module = _module()
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    return snapshot, module, plan


def test_r31_plan_is_tensor_free_and_binds_exact_compressed_p_state() -> None:
    _snapshot_value, _module_value, plan = _plan_objects()

    assert plan.stable_hash() == (
        "39d61775caac6d64a5a2d697073d0caa434d34bb2f054351f474700e9d61910f"
    )
    assert len(plan.tensor_specs) == 43
    assert plan.scratch_slot_count == 2
    assert plan.module_template.bindings["params"] == {}
    p_alpha = plan.tensor_specs[plan.p_alpha_input_ordinal]
    assert p_alpha.name == "relu/25/alpha"
    assert p_alpha.shape == (2, 1, 6, 86)
    p_layout = plan.relu_layouts[plan.p_layout_ordinal]
    assert len(p_layout.alpha_flat_indices) == 86
    assert p_layout.beta_locations == ((), (), (), (), (), ())


def test_r31_plan_and_runtime_identity_tamper_fail_closed() -> None:
    snapshot, module, plan = _plan_objects()
    with pytest.raises(ValueError, match="plan differs"):
        replace(plan, scratch_slot_count=3).validate()
    p_ordinal = plan.p_layout_ordinal
    layouts = list(plan.relu_layouts)
    layouts[p_ordinal] = replace(layouts[p_ordinal], alpha_path="alpha/wrong")
    with pytest.raises(ValueError, match="P-anchor layout"):
        replace(plan, relu_layouts=tuple(layouts)).validate()

    tensor_ordinal = next(
        ordinal
        for ordinal, item in enumerate(snapshot.tensors)
        if item.semantic_path == "alpha/%2Finput-24/%2F49"
    )
    source = snapshot.tensors[tensor_ordinal]
    changed_value = source.value.clone()
    changed_value[0, 0, 0, 0] = changed_value[0, 0, 0, 0] + 1.0e-3
    tensors = list(snapshot.tensors)
    tensors[tensor_ordinal] = replace(
        source,
        value=changed_value,
        content_sha256=production_tensor_sha256(changed_value),
    )
    changed = replace(snapshot, tensors=tuple(tensors))
    changed.validate()
    with pytest.raises(ValueError, match="runtime tensor identity"):
        bind_r31_runtime_inputs_v1(plan, module, changed, device=torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_r31_custom_backward_matches_independent_full_region_oracle() -> None:
    snapshot, module, plan = _plan_objects()
    tensors = bind_r31_runtime_inputs_v1(
        plan, module, snapshot, device=torch.device("cuda:0")
    )

    native_lower, native_gradient = execute_r31_native_oracle_v1(plan, tensors)
    result = execute_r31_custom_backward_v1(plan, tensors)

    assert torch.allclose(result.final_lower, native_lower, atol=2e-4, rtol=2e-4)
    assert torch.equal(torch.sign(result.final_lower), torch.sign(native_lower))
    assert torch.allclose(
        result.compressed_alpha_gradient,
        native_gradient,
        atol=2e-4,
        rtol=2e-4,
    )
    assert torch.equal(
        torch.sign(result.compressed_alpha_gradient), torch.sign(native_gradient)
    )
    assert result.receipt.saved_dense_a_count == 0
    assert result.receipt.scratch_slot_count == 2
    assert result.receipt.forward_count == result.receipt.backward_count == 1
    assert result.receipt.optimizer_mutation_count == 0
    assert result.receipt.performance_claimed is False
    with pytest.raises(ValueError, match="receipt differs"):
        replace(result.receipt, performance_claimed=True).validate()


def test_r31_bind_does_not_mutate_source_module() -> None:
    snapshot, module, plan = _plan_objects()
    before = copy.deepcopy(module.bindings["tensor_meta"])

    values = bind_r31_runtime_inputs_v1(
        plan, module, snapshot, device=torch.device("cpu")
    )

    assert len(values) == len(plan.tensor_specs)
    assert module.bindings["tensor_meta"] == before
    assert len(module.bindings["params"]) == 16
