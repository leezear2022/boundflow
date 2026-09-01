"""S4-1B six-site Relax/TIR graph and prepared runtime tests."""

# pylint: disable=missing-function-docstring,duplicate-code,not-callable
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=consider-using-from-import

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.nn.functional as functional

from boundflow.backends.tvm.asplos27_s4_six_site_value import (
    S4_SIX_SITE_ARGUMENTS,
    S4_SIX_SITE_CUDNN_CALLS,
    S4_SIX_SITE_TIR_COUNT,
    S4_VALUE_ARENA_ELEMENTS,
    S4_VALUE_SLOTS_V1,
    build_s4_six_site_value_relax_module_v1,
    compile_s4_six_site_value_v1,
)
from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.asplos27_s4_coefficient_selector_pass import (
    PreparedS4CoefficientSelectorPassV1,
    S4_SELECTOR_ACTIONS,
    S4_SELECTOR_SPECS,
    capture_r31b2_production_selectors_v1,
)
from boundflow.runtime.asplos27_s4_six_site_value import (
    PreparedS4SixSiteValueV1,
    S4_SIX_SITE_READ_SPECS,
    S4SixSitePhase,
    S4SixSiteRuntimeError,
)
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


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("tvm")
    return torch.device("cuda", torch.cuda.current_device())


def _randn(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if dtype == torch.float32:
        return (torch.randn(shape, dtype=dtype, device=device) * 0.05).contiguous()
    if dtype == torch.int32:
        return torch.zeros(shape, dtype=dtype, device=device)
    return torch.zeros(shape, dtype=dtype, device=device)


def _selector_owner(
    device: torch.device, stream: torch.cuda.Stream
) -> PreparedS4CoefficientSelectorPassV1:
    owner = PreparedS4CoefficientSelectorPassV1(
        device=device, exact_call_id="six-site-test"
    )
    sizes = {action: numel for _name, action, numel, _policy in S4_SELECTOR_SPECS}
    coefficients = {
        action: torch.randn(numel, dtype=torch.float32, device=device)
        for action, numel in sizes.items()
    }
    coefficients["pack_ainput"][::257] = 0.0
    owner.bind_compiled_sources(coefficients)
    with torch.cuda.stream(stream):
        owner.begin()
        for action in S4_SELECTOR_ACTIONS:
            owner.record(action, coefficients.get(action))
    return owner


def _read_arguments(
    device: torch.device, owner: PreparedS4CoefficientSelectorPassV1
) -> tuple[torch.Tensor, ...]:
    tensors = [
        _randn(shape, dtype, device) for _name, shape, dtype in S4_SIX_SITE_READ_SPECS
    ]
    selector_indices = {
        2: "endpoint_ainput_v2",
        9: "sign_a18",
        16: "sign_a20",
        25: "sign_a24",
        32: "sign_a26",
        39: "sign_a29",
    }
    for index, name in selector_indices.items():
        tensors[index] = owner.selector(name)
    for lower_index, upper_index in (
        (0, 1),
        (5, 6),
        (12, 13),
        (21, 22),
        (28, 29),
        (35, 36),
    ):
        center = torch.randn_like(tensors[lower_index]) * 0.05
        radius = torch.rand_like(center) + 0.25
        tensors[lower_index] = (center - radius).contiguous()
        tensors[upper_index] = (center + radius).contiguous()
    for map_index, width in ((8, 164), (15, 132), (24, 121), (31, 86), (38, 178)):
        tensors[map_index] = (
            torch.arange(tensors[map_index].numel(), device=device, dtype=torch.int32)
            % width
        ).contiguous()
    for alpha_index in (7, 14, 23, 30, 37):
        tensors[alpha_index] = torch.rand_like(tensors[alpha_index]).contiguous()
    return tuple(tensors)


def _selected_relu(
    pre: torch.Tensor,
    selector: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    alpha: torch.Tensor,
    alpha_map: torch.Tensor,
) -> torch.Tensor:
    domains = pre.shape[0]
    flat_lower = lower.reshape(domains, -1)
    flat_upper = upper.reshape(domains, -1)
    lookup = alpha_map.to(torch.int64)
    compact = alpha[:, lookup.clamp_min(0)]
    lower_alpha = torch.where(lookup.view(1, -1) >= 0, compact.clamp(0, 1), 0)
    ambiguous = (flat_lower < 0) & (flat_upper > 0)
    lower_slope = torch.where(ambiguous, lower_alpha, (flat_lower >= 0).float())
    upper_slope = torch.where(
        flat_lower >= 0,
        torch.ones_like(flat_lower),
        torch.where(
            flat_upper <= 0,
            torch.zeros_like(flat_lower),
            flat_upper
            / (flat_upper - flat_lower).clamp_min(torch.finfo(torch.float32).eps),
        ),
    )
    select = selector.reshape(domains, -1)
    slope = torch.where(select == 1, lower_slope, upper_slope)
    intercept = torch.where((select == 0) & ambiguous, -flat_lower * upper_slope, 0)
    value = pre.reshape(domains, -1) * slope + intercept
    return value.reshape_as(pre)


def _oracle(reads: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    selector = reads[2].reshape(6, 3, 32, 32)
    selected_input = torch.where(
        selector == 1,
        reads[0],
        torch.where(selector == -1, reads[1], (reads[0] + reads[1]) * 0.5),
    )
    conv: Callable[..., torch.Tensor] = functional.conv2d
    v17 = conv(selected_input, reads[3], reads[4], stride=2, padding=1)
    selected17 = _selected_relu(v17, reads[9], reads[5], reads[6], reads[7], reads[8])
    v19 = conv(selected17, reads[10], reads[11], stride=2, padding=1)
    selected19 = _selected_relu(
        v19, reads[16], reads[12], reads[13], reads[14], reads[15]
    )
    v23 = conv(selected19, reads[17], reads[18], padding=1) + conv(
        selected17, reads[19], reads[20], stride=2
    )
    selected23 = _selected_relu(
        v23, reads[25], reads[21], reads[22], reads[23], reads[24]
    )
    v25 = conv(selected23, reads[26], reads[27], padding=1)
    selected25 = _selected_relu(
        v25, reads[32], reads[28], reads[29], reads[30], reads[31]
    )
    v28 = conv(selected25, reads[33], reads[34], padding=1) + selected23
    selected28 = _selected_relu(
        v28, reads[39], reads[35], reads[36], reads[37], reads[38]
    )
    v31 = functional.linear(selected28.reshape(6, 1024), reads[40], reads[41])
    return v17, v19, v23, v25, v28, v31


def test_s4_six_site_ir_has_exact_abi_and_tir_count() -> None:
    pytest.importorskip("tvm")
    module = build_s4_six_site_value_relax_module_v1()
    script = module.script()
    assert len(module.functions) == S4_SIX_SITE_TIR_COUNT + 1
    assert script.count("call_tir_inplace") == 7
    assert "active_alpha" in script
    assert "endpoint_selector" in script
    assert "boundflow_s4_six_site_value" in script
    assert S4_SIX_SITE_ARGUMENTS == 49


def test_s4_six_site_runtime_matches_independent_torch_oracle() -> None:
    device = _require_cuda()
    torch.manual_seed(270831)
    stream = torch.cuda.Stream(device=device)
    owner = _selector_owner(device, stream)
    stream.synchronize()
    reads = _read_arguments(device, owner)
    coefficient_arena = torch.empty(20000, dtype=torch.float32, device=device)
    selected_input_alias = coefficient_arena[:18432].view(6, 3, 32, 32)
    compiled = compile_s4_six_site_value_v1(device_index=device.index or 0)
    assert compiled.cudnn_conv_call_count == S4_SIX_SITE_CUDNN_CALLS
    for changed in (
        replace(compiled, source_relax_ir_json=compiled.source_relax_ir_json + " "),
        replace(
            compiled,
            partitioned_relax_ir_json=compiled.partitioned_relax_ir_json + " ",
        ),
        replace(compiled, lowered_relax_ir_json=compiled.lowered_relax_ir_json + " "),
        replace(compiled, device_sources=compiled.device_sources + ("tampered",)),
        replace(compiled, cudnn_conv_call_count=5),
        replace(compiled, selected_tir_count=11),
        replace(compiled, performance_claimed=True),
    ):
        with pytest.raises(ValueError, match="compiled identity differs"):
            changed.validate()
    runtime = PreparedS4SixSiteValueV1(
        compiled,
        reads,
        coefficient_arena=coefficient_arena,
        selected_input_alias=selected_input_alias,
        device=device,
    )
    reference = _oracle(reads)
    with torch.cuda.stream(stream):
        runtime.begin_pass_a()
        runtime.adopt_selectors(owner)
        values = runtime.run_pass_b()
        result = runtime.handoff_to_coefficient_recompute()
    stream.synchronize()
    result.validate()
    assert runtime.phase == S4SixSitePhase.COEFFICIENT_RECOMPUTE_READY
    assert runtime.value_arena.numel() == S4_VALUE_ARENA_ELEMENTS
    for actual, expected, slot in zip(values, reference, S4_VALUE_SLOTS_V1):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
        assert (
            actual.data_ptr() == runtime.value_arena[slot.offset_elements :].data_ptr()
        )
    assert result.receipt.prepare_dlpack_view_count == 55
    assert result.receipt.warm_dlpack_view_count == 0
    assert result.receipt.performance_claimed is False
    encoded = json.dumps(result.receipt.to_dict(), sort_keys=True)
    assert "data_ptr" not in encoded and "cuda_stream" not in encoded
    mutations = (
        replace(result.receipt, read_argument_count=41),
        replace(result.receipt, write_target_count=6),
        replace(result.receipt, total_argument_count=48),
        replace(result.receipt, s4_1b_union_descriptor_count=89),
        replace(result.receipt, s4_1abc_union_descriptor_count=109),
        replace(result.receipt, base_overlap_count=4),
        replace(result.receipt, value_arena_elements=37463),
        replace(result.receipt, value_slot_count=5),
        replace(result.receipt, selected_input_alias_exact=False),
        replace(result.receipt, selected_input_live_reader_count=1),
        replace(result.receipt, cudnn_conv_call_count=5),
        replace(result.receipt, selected_tir_count=11),
        replace(result.receipt, persistent_output_copy_count=5),
        replace(result.receipt, dynamic_output_allocation_count=1),
        replace(result.receipt, result_owner_capacity=2),
        replace(result.receipt, output_pointer_exact_count=5),
        replace(result.receipt, fallback_count=1),
        replace(result.receipt, eager_candidate_count=1),
        replace(result.receipt, native_shadow_count=1),
        replace(result.receipt, saved_dense_a_count=1),
        replace(result.receipt, timing_recorded=True),
        replace(result.receipt, performance_claimed=True),
    )
    for changed_receipt in mutations:
        with pytest.raises(S4SixSiteRuntimeError):
            changed_receipt.validate()


def test_s4_six_site_runtime_rejects_selector_pointer_substitution() -> None:
    device = _require_cuda()
    stream = torch.cuda.Stream(device=device)
    owner = _selector_owner(device, stream)
    stream.synchronize()
    reads = list(_read_arguments(device, owner))
    reads[9] = reads[9].clone()
    arena = torch.empty(20000, dtype=torch.float32, device=device)
    compiled = compile_s4_six_site_value_v1(device_index=device.index or 0)
    runtime = PreparedS4SixSiteValueV1(
        compiled,
        tuple(reads),
        coefficient_arena=arena,
        selected_input_alias=arena[:18432].view(6, 3, 32, 32),
        device=device,
    )
    with torch.cuda.stream(stream):
        runtime.begin_pass_a()
        with pytest.raises(
            S4SixSiteRuntimeError, match="SIX_SITE_SELECTOR_POINTER_MISMATCH"
        ):
            runtime.adopt_selectors(owner)
    assert runtime.phase == S4SixSitePhase.POISONED


def test_s4_six_site_real_r31b2_capture_and_value_graph() -> None:
    device = _require_cuda()
    if not CAPTURE.is_file() or not MODEL.is_file():
        pytest.skip("S4-1B frozen production fixture is unavailable")
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
    owner = PreparedS4CoefficientSelectorPassV1(
        device=device,
        exact_call_id="s4-1b-production-correctness",
        evaluation_generation=11,
        parameter_generation=12,
        coefficient_generation=13,
        selector_generation=14,
    )
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        selector_receipt = capture_r31b2_production_selectors_v1(executor, owner)
    stream.synchronize()
    selector_receipt.validate()
    assert selector_receipt.action_count == 19

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
    reference = _oracle(reads)
    coefficient_arena = executor.forward_executor.scratch_1
    selected_input_alias = coefficient_arena[:18432].view(6, 3, 32, 32)
    compiled = compile_s4_six_site_value_v1(device_index=device.index or 0)
    runtime = PreparedS4SixSiteValueV1(
        compiled,
        reads,
        coefficient_arena=coefficient_arena,
        selected_input_alias=selected_input_alias,
        device=device,
    )
    with torch.cuda.stream(stream):
        runtime.begin_pass_a()
        runtime.adopt_selectors(owner)
        values = runtime.run_pass_b()
        result = runtime.handoff_to_coefficient_recompute()
    stream.synchronize()
    result.validate()
    for actual, expected in zip(values, reference):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    assert result.receipt.selector_receipt_hash == selector_receipt.stable_hash()
    assert result.receipt.s4_1b_union_descriptor_count == 90
    assert result.receipt.s4_1abc_union_descriptor_count == 110
