"""R3-1b1 compiled full-lower forward correctness and ownership gates."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from boundflow.backends.tvm.r3_full_lower_forward import (
    R31B1_EXPORTED_SYMBOLS,
    build_r31b1_full_lower_forward_tir_v1,
)
from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.r3_bounded_arena_trace_compiler import (
    compile_r31b_bounded_arena_trace_v1,
)
from boundflow.runtime.r3_full_lower_forward_tir import (
    PreparedR31B1FullLowerForwardV1,
    R31B1ModuleCacheV1,
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
CAPTURE = (
    ROOT / "artifacts/rvir-v4-pre-state/resnet2b-core-pre-state-v1/source_capture.pt"
)
MODEL = Path(
    "/home/lee/Codes/vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
)


@pytest.fixture(scope="module")
def r31b1_objects():  # type: ignore[no-untyped-def]
    if not MODEL.is_file():
        pytest.skip("frozen ResNet2B checkout is unavailable")
    if not torch.cuda.is_available():
        pytest.skip("R3-1b1 CUDA device is unavailable")
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (8, 9):
        pytest.skip("R3-1b1 frozen sm_89 device is unavailable")
    raw = torch.load(CAPTURE, map_location="cpu", weights_only=True)
    snapshot = production_snapshot_from_payload_v4(raw["cores"][0]["pre_snapshot"])
    mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
    program = import_onnx(str(MODEL), do_shape_infer=True, normalize=True)
    module = plan_interval_ibp_v0(program)
    plan = compile_r31_full_region_plan_v1(module, snapshot, mapping, TOPOLOGY)
    trace = compile_r31b_bounded_arena_trace_v1(program, module, plan)
    tensors = bind_r31_runtime_inputs_v1(
        plan, module, snapshot, device=torch.device("cuda:0")
    )
    return plan, trace, tensors


def test_r31b1_builds_complete_full_lower_module_without_global_workspace() -> None:
    module = build_r31b1_full_lower_forward_tir_v1()
    symbols = tuple(
        sorted(
            str(function.attrs["global_symbol"])
            for function in module.functions.values()
        )
    )

    assert symbols == tuple(sorted(R31B1_EXPORTED_SYMBOLS))
    assert len(symbols) == 15
    assert 'scope="global"' not in module.script()


def test_r31b1_nondefault_stream_full_lower_matches_independent_native_oracle(
    r31b1_objects,
) -> None:
    plan, trace, tensors = r31b1_objects
    with torch.no_grad():
        native = _evaluate_full_region(plan, tensors).detach()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        prepared = PreparedR31B1FullLowerForwardV1(
            plan, trace, tensors, cache=R31B1ModuleCacheV1()
        )
        result = prepared.run()
    stream.synchronize()

    assert torch.allclose(result.lower, native, atol=2e-4, rtol=2e-4)
    assert torch.equal(torch.sign(result.lower), torch.sign(native))
    assert float((result.lower - native).abs().max().item()) <= 2e-4
    assert result.launch_receipt.warm_dynamic_allocated_bytes == 0
    assert result.launch_receipt.coefficient_scratch_count == 2
    assert result.launch_receipt.launch_count == 15
    assert result.launch_receipt.compiled_region is True
    assert result.launch_receipt.performance_claimed is False


def test_r31b1_default_stream_trace_tensor_and_metadata_tamper_fail_closed(
    r31b1_objects,
) -> None:
    plan, trace, tensors = r31b1_objects
    cache = R31B1ModuleCacheV1()
    prepared = PreparedR31B1FullLowerForwardV1(plan, trace, tensors, cache=cache)
    with pytest.raises(RuntimeError, match="non-default stream"):
        prepared.run()

    changed_trace = replace(trace, production_plan_hash="0" * 64)
    changed_trace.validate()
    with pytest.raises(ValueError, match="runtime admission"):
        PreparedR31B1FullLowerForwardV1(plan, changed_trace, tensors, cache=cache)

    changed_tensors = list(tensors)
    changed_tensors[2] = changed_tensors[2].clone()
    changed_tensors[2][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="runtime admission"):
        PreparedR31B1FullLowerForwardV1(
            plan, trace, tuple(changed_tensors), cache=cache
        )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        changed_metadata = PreparedR31B1FullLowerForwardV1(
            plan, trace, tensors, cache=cache
        )
        changed_metadata.alpha_maps["25"][0] = 0
        with pytest.raises(ValueError, match="metadata identity"):
            changed_metadata.run()
