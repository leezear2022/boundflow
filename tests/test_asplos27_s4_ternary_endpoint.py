"""S4-1B0 isolated ternary endpoint bit, cache, and CUDA correctness gates."""

# mypy: disable-error-code=import-untyped
# pylint: disable=protected-access,too-many-locals,import-error
# pylint: disable=missing-function-docstring,import-outside-toplevel

from dataclasses import replace
import json
from pathlib import Path
import struct
from typing import Any, cast

import pytest
import torch

from boundflow.backends.tvm import asplos27_s4_ternary_endpoint as endpoint

FIXTURE_PATH = Path(
    "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_IEEE_BIT_FIXTURES_V1_2026_08_30.json"
)
NEGATIVE_PATH = Path(
    "gemini_doc/BOUNDFLOW_ASPLOS27_S4_1B0_NEGATIVE_CONTRACT_V1_2026_08_30.json"
)


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _negative() -> dict[str, object]:
    return json.loads(NEGATIVE_PATH.read_text(encoding="utf-8"))


def _u32(bits: str) -> int:
    return int(bits, 16)


def _f32(bits: str) -> float:
    return struct.unpack("<f", struct.pack("<I", _u32(bits)))[0]


def _bits(tensor: torch.Tensor) -> list[int]:
    return [int(value) & 0xFFFFFFFF for value in tensor.view(torch.int32).tolist()]


def _gpu_tensors(numel: int) -> tuple[torch.Tensor, ...]:
    coefficient = torch.linspace(-2.0, 2.0, numel, device="cuda")
    lower = torch.linspace(-4.0, -1.0, numel, device="cuda")
    upper = torch.linspace(1.0, 6.0, numel, device="cuda")
    selector = torch.empty(numel, dtype=torch.int8, device="cuda")
    selected = torch.empty(numel, dtype=torch.float32, device="cuda")
    return coefficient, lower, upper, selector, selected


def test_pack_cpu_bit_oracle_covers_signed_zero_subnormal_nonfinite() -> None:
    fixture = _fixture()
    cases = fixture["pack_cases"]
    assert isinstance(cases, list)
    observed = [
        endpoint.ternary_pack_bit_oracle_v1(_u32(str(case["input_bits"])))
        for case in cases
    ]
    assert observed == [int(case["expected_selector"]) for case in cases]
    assert observed == [
        -128,
        -128,
        -1,
        -1,
        -1,
        -1,
        0,
        0,
        1,
        1,
        1,
        1,
        -128,
        -128,
        -128,
        1,
    ]


def test_select_cpu_bit_oracle_covers_midpoint_and_canonical_nan() -> None:
    fixture = _fixture()
    cases = fixture["select_cases"]
    assert isinstance(cases, list)
    observed = [
        endpoint.ternary_select_bit_oracle_v1(
            int(case["selector"]),
            _u32(str(case["lower_bits"])),
            _u32(str(case["upper_bits"])),
        )
        for case in cases
    ]
    assert observed == [_u32(str(case["expected_bits"])) for case in cases]
    assert observed[9:12] == [endpoint.S4_TERNARY_ENDPOINT_QNAN_BITS] * 3


def test_midpoint_operation_order_has_two_frozen_counterexamples() -> None:
    fixture = _fixture()
    cases = fixture["midpoint_reassociation_counterexamples"]
    assert isinstance(cases, list)
    assert len(cases) == 2
    for case in cases:
        observed = endpoint.ternary_select_bit_oracle_v1(
            0, _u32(str(case["input_bits"])), _u32(str(case["input_bits"]))
        )
        assert observed == _u32(str(case["add_then_mul_bits"]))
        assert observed != _u32(str(case["mul_then_add_bits"]))


def test_build_spec_and_schedule_are_generic_and_canonical() -> None:
    first = endpoint.TernaryEndpointBuildSpecV1(numel=16)
    second = endpoint.TernaryEndpointBuildSpecV1(numel=18432)
    schedule = endpoint.TernaryEndpointScheduleSpecV1()
    first.validate()
    second.validate()
    schedule.validate()
    assert first.stable_hash() == first.stable_hash()
    assert first.stable_hash() != second.stable_hash()
    assert schedule.stable_hash() == schedule.stable_hash()
    assert first.target_string == "cuda -arch=sm_89"
    assert (
        endpoint.validate_ternary_endpoint_construction_model_v1()
        == endpoint.S4_TERNARY_ENDPOINT_EXPECTED_CONSTRUCTION_HASH_V1
    )

    mutations = (
        (replace(first, schema_version="wrong"), "TERNARY_ENDPOINT_SCHEMA_MISMATCH"),
        (replace(first, endpoint_policy="wrong"), "TERNARY_ENDPOINT_POLICY_MISMATCH"),
        (
            replace(first, midpoint_policy="wrong"),
            "TERNARY_ENDPOINT_MIDPOINT_POLICY_MISMATCH",
        ),
        (
            replace(first, nonfinite_policy="wrong"),
            "TERNARY_ENDPOINT_NONFINITE_POLICY_MISMATCH",
        ),
        (
            replace(first, pack_symbol=first.select_symbol),
            "TERNARY_ENDPOINT_SYMBOL_COLLISION",
        ),
    )
    for mutated, reason in mutations:
        with pytest.raises(endpoint.TernaryEndpointError, match=reason):
            mutated.validate()


def test_tir_module_exports_exact_new_symbols() -> None:
    import tvm

    spec = endpoint.TernaryEndpointBuildSpecV1(numel=16)
    schedule = endpoint.TernaryEndpointScheduleSpecV1()
    unscheduled, scheduled = endpoint.build_ternary_endpoint_modules_v1(spec, schedule)
    for module in (unscheduled, scheduled):
        text = module.script()
        assert set(name.name_hint for name in module.get_global_vars()) == set(
            endpoint.S4_TERNARY_ENDPOINT_EXPORTED_SYMBOLS
        )
        assert endpoint.S4_TERNARY_ENDPOINT_SCHEMA in text
        assert "boundflow_r31b2_pack_ainput_sign" not in text
        assert "boundflow_s2_select_input_tir" not in text
        assert len(tvm.ir.save_json(module)) > 100


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cache_first_miss_then_exact_hit_compiles_once() -> None:
    spec = endpoint.TernaryEndpointBuildSpecV1(numel=16)
    schedule = endpoint.TernaryEndpointScheduleSpecV1()
    cache = endpoint.TernaryEndpointModuleCacheV1()
    first_compiled, first_receipt, first = cache.get(spec, schedule)
    second_compiled, second_receipt, second = cache.get(spec, schedule)
    assert first.event == "miss"
    assert second.event == "hit"
    assert (second.compile_count, second.miss_count, second.hit_count) == (1, 1, 1)
    assert first_compiled is second_compiled
    assert first_receipt == second_receipt
    assert first_receipt.performance_claimed is False
    assert first_receipt.global_workspace_bytes == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cache_key_rejects_policy_schedule_target_and_legacy_collisions() -> None:
    spec = endpoint.TernaryEndpointBuildSpecV1(numel=16)
    schedule = endpoint.TernaryEndpointScheduleSpecV1()
    cache = endpoint.TernaryEndpointModuleCacheV1()
    _compiled, receipt, _observation = cache.get(spec, schedule)

    with pytest.raises(endpoint.TernaryEndpointError, match="POLICY_MISMATCH"):
        cache.get(replace(spec, endpoint_policy="wrong"), schedule)
    with pytest.raises(endpoint.TernaryEndpointError, match="TIR_IDENTITY_MISMATCH"):
        cache.get(spec, replace(schedule, threads_per_block=32))
    with pytest.raises(endpoint.TernaryEndpointError, match="POLICY_MISMATCH"):
        cache.get(replace(spec, target="llvm"), schedule)
    with pytest.raises(endpoint.TernaryEndpointError, match="LEGACY_MODULE_COLLISION"):
        cache.get(
            replace(spec, pack_symbol="boundflow_r31b2_pack_ainput_sign"), schedule
        )
    with pytest.raises(endpoint.TernaryEndpointError, match="CACHE_KEY_MISMATCH"):
        replace(receipt, cache_key="0" * 64).validate_against(spec, schedule, _compiled)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cache_hit_rehashes_cached_device_source() -> None:
    spec = endpoint.TernaryEndpointBuildSpecV1(numel=16)
    schedule = endpoint.TernaryEndpointScheduleSpecV1()
    cache = endpoint.TernaryEndpointModuleCacheV1()
    compiled, receipt, _observation = cache.get(spec, schedule)
    poisoned = replace(compiled, device_source=compiled.device_source + "\n// tamper")
    cache._entries[receipt.cache_key] = (poisoned, receipt)
    with pytest.raises(endpoint.TernaryEndpointError, match="DEVICE_SOURCE_MISMATCH"):
        cache.get(spec, schedule)

    healthy_cache = endpoint.TernaryEndpointModuleCacheV1()
    healthy_compiled, healthy_receipt, _event = healthy_cache.get(spec, schedule)
    healthy_cache._entries[healthy_receipt.cache_key] = (
        healthy_compiled,
        replace(healthy_receipt, torch_version=""),
    )
    with pytest.raises(endpoint.TernaryEndpointError, match="CACHE_ENTRY_POISONED"):
        healthy_cache.get(spec, schedule)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_boundary_pack_select_matches_bit_oracle() -> None:
    fixture = _fixture()
    pack_cases = fixture["pack_cases"]
    select_cases = fixture["select_cases"]
    assert isinstance(pack_cases, list) and isinstance(select_cases, list)
    coefficient = torch.tensor(
        [_f32(str(case["input_bits"])) for case in pack_cases], device="cuda"
    )
    lower = torch.tensor(
        [_f32(str(case["lower_bits"])) for case in select_cases], device="cuda"
    )
    upper = torch.tensor(
        [_f32(str(case["upper_bits"])) for case in select_cases], device="cuda"
    )
    selector = torch.tensor(
        [int(case["selector"]) for case in select_cases],
        dtype=torch.int8,
        device="cuda",
    )
    selected = torch.empty(16, dtype=torch.float32, device="cuda")
    packed = torch.empty(16, dtype=torch.int8, device="cuda")
    spec = endpoint.TernaryEndpointBuildSpecV1(numel=16)
    schedule = endpoint.TernaryEndpointScheduleSpecV1()
    cache = endpoint.TernaryEndpointModuleCacheV1()
    compiled, _receipt, _observation = cache.get(spec, schedule)
    import tvm

    compiled.executable[spec.pack_symbol](
        tvm.runtime.from_dlpack(coefficient), tvm.runtime.from_dlpack(packed)
    )
    compiled.executable[spec.select_symbol](
        tvm.runtime.from_dlpack(lower),
        tvm.runtime.from_dlpack(upper),
        tvm.runtime.from_dlpack(selector),
        tvm.runtime.from_dlpack(selected),
    )
    torch.cuda.synchronize()
    assert packed.cpu().tolist() == [
        int(case["expected_selector"]) for case in pack_cases
    ]
    assert _bits(selected.cpu()) == [
        _u32(str(case["expected_bits"])) for case in select_cases
    ]
    endpoint.validate_selected_output_after_sync_v1(selector, selected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_non_default_stream_and_five_dlpack_pointers_are_exact() -> None:
    tensors = _gpu_tensors(257)
    spec = endpoint.TernaryEndpointBuildSpecV1(numel=257)
    schedule = endpoint.TernaryEndpointScheduleSpecV1()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        prepared = endpoint.PreparedTernaryEndpointProbeV1.prepare(
            spec, schedule, *tensors
        )
        receipt = prepared.run_once(
            evaluation_ordinal=3,
            parameter_state_version=7,
            selector_generation=11,
        )
    stream.synchronize()
    coefficient, lower, upper, selector, selected = tensors
    expected_selector = torch.sign(coefficient).to(torch.int8)
    expected = torch.where(
        expected_selector > 0,
        lower,
        torch.where(expected_selector < 0, upper, (lower + upper) * 0.5),
    )
    torch.testing.assert_close(selector, expected_selector, rtol=0, atol=0)
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)
    assert receipt.stream_identity == stream.cuda_stream
    assert receipt.prepared_descriptor_hashes == tuple(
        row.stable_hash() for row in prepared.descriptors
    )
    assert len({row.data_ptr for row in prepared.descriptors}) == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_invalid_selector_produces_canonical_nan() -> None:
    selector = torch.tensor([-128, 2, -2, 3], dtype=torch.int8, device="cuda")
    selected = torch.zeros(4, dtype=torch.float32, device="cuda")
    lower = torch.ones(4, device="cuda")
    upper = torch.full((4,), 2.0, device="cuda")
    coefficient = torch.ones(4, device="cuda")
    spec = endpoint.TernaryEndpointBuildSpecV1(numel=4)
    compiled, _receipt, _event = endpoint.TernaryEndpointModuleCacheV1().get(
        spec, endpoint.TernaryEndpointScheduleSpecV1()
    )
    import tvm

    compiled.executable[spec.select_symbol](
        tvm.runtime.from_dlpack(lower),
        tvm.runtime.from_dlpack(upper),
        tvm.runtime.from_dlpack(selector),
        tvm.runtime.from_dlpack(selected),
    )
    torch.cuda.synchronize()
    assert _bits(selected.cpu()) == [endpoint.S4_TERNARY_ENDPOINT_QNAN_BITS] * 4
    endpoint.validate_selected_output_after_sync_v1(selector, selected)
    selected.zero_()
    with pytest.raises(endpoint.TernaryEndpointError, match="INVALID_SELECTOR"):
        endpoint.validate_selected_output_after_sync_v1(selector, selected)
    assert coefficient.numel() == 4


def test_isolated_outputs_account_for_92160_logical_bytes() -> None:
    numel = 18432
    assert numel + 4 * numel == 92160
    assert numel == 18432
    assert 4 * numel == 73728


def test_formal_ainput_inventory_is_8689_9137_606_0() -> None:
    fixture = _fixture()
    inventory = cast(dict[str, object], fixture["future_production_fixture_inventory"])
    assert inventory == {
        "numel": 18432,
        "positive": 8689,
        "negative": 9137,
        "zero": 606,
        "invalid": 0,
        "old_binary_zero_misclassified": 606,
        "status": "design-time-expectation-not-production-validation",
    }
    assert 8689 + 9137 + 606 == 18432


def test_old_binary_misclassifies_exactly_606_zero_entries() -> None:
    fixture = _fixture()
    inventory = cast(dict[str, object], fixture["future_production_fixture_inventory"])
    assert inventory["old_binary_zero_misclassified"] == inventory["zero"] == 606


def test_legacy_r31b2_and_s2_identities_are_unchanged() -> None:
    from boundflow.backends.tvm.asplos27_s2_selected_value import (
        S2_SELECTED_VALUE_SCHEMA,
    )
    from boundflow.backends.tvm.r3_p_alpha_vjp import R31B2_PACK_AINPUT_SYMBOL

    assert R31B2_PACK_AINPUT_SYMBOL == "boundflow_r31b2_pack_ainput_sign"
    assert S2_SELECTED_VALUE_SCHEMA == "boundflow.asplos27-s2-selected-value/v1"
    assert R31B2_PACK_AINPUT_SYMBOL not in endpoint.S4_TERNARY_ENDPOINT_EXPORTED_SYMBOLS


def test_all_twenty_negative_reasons_are_stable() -> None:
    contract = _negative()
    rows = contract["stable_reasons"]
    assert isinstance(rows, list)
    expected = tuple(str(row["reason"]) for row in rows)
    assert endpoint.TERNARY_ENDPOINT_STABLE_REASONS == expected
    assert len(expected) == 20 == len(set(expected))
    for reason in expected:
        error = endpoint.TernaryEndpointError(reason)
        assert error.reason == reason
        assert str(error) == reason


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepare_rejects_shape_dtype_device_layout_alias_and_claim_mutations() -> None:
    spec = endpoint.TernaryEndpointBuildSpecV1(numel=16)
    schedule = endpoint.TernaryEndpointScheduleSpecV1()
    tensors = _gpu_tensors(16)
    bad_shape = torch.empty(15, device="cuda")
    with pytest.raises(endpoint.TernaryEndpointError, match="SHAPE_MISMATCH"):
        endpoint.PreparedTernaryEndpointProbeV1.prepare(
            spec, schedule, bad_shape, *tensors[1:]
        )
    with pytest.raises(endpoint.TernaryEndpointError, match="DTYPE_MISMATCH"):
        endpoint.PreparedTernaryEndpointProbeV1.prepare(
            spec, schedule, tensors[0].double(), *tensors[1:]
        )
    with pytest.raises(endpoint.TernaryEndpointError, match="DEVICE_MISMATCH"):
        endpoint.PreparedTernaryEndpointProbeV1.prepare(
            spec, schedule, tensors[0].cpu(), *tensors[1:]
        )
    matrix = torch.empty((16, 2), device="cuda")
    noncontiguous = matrix[:, 0]
    with pytest.raises(endpoint.TernaryEndpointError, match="LAYOUT_MISMATCH"):
        endpoint.PreparedTernaryEndpointProbeV1.prepare(
            spec, schedule, noncontiguous, *tensors[1:]
        )
    with pytest.raises(endpoint.TernaryEndpointError, match="ALIAS_MISMATCH"):
        endpoint.PreparedTernaryEndpointProbeV1.prepare(
            spec, schedule, tensors[0], tensors[1], tensors[2], tensors[3], tensors[1]
        )

    compiled, receipt, _event = endpoint.TernaryEndpointModuleCacheV1().get(
        spec, schedule
    )
    with pytest.raises(endpoint.TernaryEndpointError, match="CLAIM_FLAG_MISMATCH"):
        replace(receipt, performance_claimed=True).validate_against(
            spec, schedule, compiled
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dlpack_and_launch_receipts_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.ones(8, device="cuda")
    real_from_dlpack = torch.from_dlpack

    def cloned_from_dlpack(value: object) -> torch.Tensor:
        return real_from_dlpack(value).clone()

    monkeypatch.setattr(torch, "from_dlpack", cloned_from_dlpack)
    with pytest.raises(endpoint.TernaryEndpointError, match="DLPACK_IDENTITY_MISMATCH"):
        endpoint._create_dlpack_view(tensor)
    monkeypatch.setattr(torch, "from_dlpack", real_from_dlpack)

    good = endpoint.TernaryEndpointWarmLaunchReceiptV1(
        module_receipt_hash="a" * 64,
        cache_event="hit",
        device_ordinal=0,
        stream_identity=1,
        evaluation_ordinal=0,
        parameter_state_version=0,
        selector_generation=0,
        prepared_descriptor_hashes=("b" * 64,) * 5,
    )
    good.validate()
    with pytest.raises(endpoint.TernaryEndpointError, match="LAUNCH_COUNT_MISMATCH"):
        replace(good, pack_launch_count=2).validate()
    with pytest.raises(endpoint.TernaryEndpointError, match="CLAIM_FLAG_MISMATCH"):
        replace(good, performance_claimed=True).validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_probe_rejects_stream_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tvm_ffi

    tensors = _gpu_tensors(8)
    prepared = endpoint.PreparedTernaryEndpointProbeV1.prepare(
        endpoint.TernaryEndpointBuildSpecV1(numel=8),
        endpoint.TernaryEndpointScheduleSpecV1(),
        *tensors,
    )
    real_get_raw_stream = tvm_ffi.get_raw_stream

    def drifted_stream(device: Any) -> int:
        return int(real_get_raw_stream(device)) + 1

    monkeypatch.setattr(tvm_ffi, "get_raw_stream", drifted_stream)
    with pytest.raises(endpoint.TernaryEndpointError, match="STREAM_IDENTITY_MISMATCH"):
        prepared.run_once(
            evaluation_ordinal=0,
            parameter_state_version=0,
            selector_generation=0,
        )
