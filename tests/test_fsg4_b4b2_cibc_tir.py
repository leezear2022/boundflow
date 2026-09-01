"""Manual TVM TIR CIBC-parity horizontal fusion gates."""

# pylint: disable=missing-function-docstring,too-many-locals,duplicate-code

from pathlib import Path
import re

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from boundflow.backends.tvm.cibc_horizontal_fused_conv import (
    CIBC_TIR_BACKWARD_SYMBOL,
    CIBC_TIR_FORWARD_SYMBOL,
    build_cibc_horizontal_conv_tir_v2,
    compile_cibc_horizontal_conv_tir_v2,
)
from boundflow.runtime.fsg4_b4b1_reference_capture import (
    production_differentiable_reference_capture_from_payload_v1,
)
from boundflow.runtime.fsg4_b4b2_cibc_tir import PreparedCIBCHorizontalTIRV2
from boundflow.runtime.fsg4_b4b2_sparse_conv_timing import (
    compare_sparse_conv_executions_v1,
)

ARTIFACT = Path("artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1")


def _capture(run_ordinal: int):
    payload = torch.load(
        ARTIFACT / f"run_{run_ordinal:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    return production_differentiable_reference_capture_from_payload_v1(
        payload["captures"][1]
    )


def test_cibc_horizontal_tir_has_exact_two_symbols() -> None:
    module = build_cibc_horizontal_conv_tir_v2()
    symbols = tuple(
        sorted(
            str(function.attrs["global_symbol"])
            for function in module.functions.values()
        )
    )
    assert symbols == tuple(sorted((CIBC_TIR_FORWARD_SYMBOL, CIBC_TIR_BACKWARD_SYMBOL)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cibc_horizontal_tir_compile_receipt_and_kernel_source() -> None:
    compiled = compile_cibc_horizontal_conv_tir_v2(compute_capability="sm_89")
    assert compiled.global_workspace_bytes == 0
    assert compiled.exported_symbols == (
        CIBC_TIR_FORWARD_SYMBOL,
        CIBC_TIR_BACKWARD_SYMBOL,
    )
    names = re.findall(
        r'extern "C" __global__ void(?: __launch_bounds__\([^)]*\))? ([A-Za-z0-9_]+)\(',
        compiled.device_source,
    )
    assert set(names) == {
        CIBC_TIR_FORWARD_SYMBOL + "_kernel",
        CIBC_TIR_BACKWARD_SYMBOL + "_kernel",
    }
    assert names.count(CIBC_TIR_FORWARD_SYMBOL + "_kernel") == 2
    assert names.count(CIBC_TIR_BACKWARD_SYMBOL + "_kernel") == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cibc_horizontal_tir_five_raw_public_pytorch_parity() -> None:
    compiled = compile_cibc_horizontal_conv_tir_v2(compute_capability="sm_89")
    maximum = 0.0
    for run_ordinal in range(5):
        prepared = PreparedCIBCHorizontalTIRV2(_capture(run_ordinal), compiled=compiled)
        metric = compare_sparse_conv_executions_v1(
            prepared.baseline_once(), prepared.candidate_once()
        )
        maximum = max(maximum, metric.maximum_absolute_difference)
        assert metric.allclose and metric.sign_exact
        assert metric.element_count == 12_810
        assert prepared.executor.fallback_count == 0
        assert prepared.executor.eager_count == 0
    assert maximum <= 2.0e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cibc_horizontal_tir_profiler_inventory_is_one_plus_one() -> None:
    prepared = PreparedCIBCHorizontalTIRV2(_capture(0))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        prepared.candidate_once()
        torch.cuda.synchronize()
    names = [
        event.name
        for event in profiler.events()
        if str(event.device_type).endswith("CUDA")
    ]
    assert names == [
        CIBC_TIR_FORWARD_SYMBOL + "_kernel",
        CIBC_TIR_BACKWARD_SYMBOL + "_kernel",
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cibc_horizontal_tir_nondefault_stream_fails_closed_at_admission() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with pytest.raises(RuntimeError, match="initial stream differs"):
            PreparedCIBCHorizontalTIRV2(_capture(0))
