"""Compile/load/cache contracts for PR-12J fused CROWN modules."""

import json
from pathlib import Path

import pytest
import torch

from boundflow.backends.tvm.fused_crown_cache import FusedCrownModuleCache
from boundflow.runtime.fused_crown import (
    FusedReluAffineRequest,
    TVMFusedCrownExecutor,
    TorchDenseFusedCrownReference,
)


def _request(kind: str) -> FusedReluAffineRequest:
    torch.manual_seed(1210 + int(kind == "conv2d"))
    domain, spec, current, previous = 2, 3, 7, 4
    if kind == "linear":
        input_shape = (previous,)
        output_shape = (current,)
        weight = torch.randn(current, previous, device="cuda")
        bias = torch.randn(current, device="cuda")
        attrs = {}
    else:
        input_shape = (2, 7, 7)
        output_shape = (3, 4, 4)
        current = 3 * 4 * 4
        weight = torch.randn(3, 2, 3, 3, device="cuda")
        bias = torch.randn(3, device="cuda")
        attrs = {
            "stride": (2, 2),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "output_padding": (0, 0),
        }
    return FusedReluAffineRequest(
        kind=kind,  # type: ignore[arg-type]
        A_u=torch.randn(domain, spec, current, device="cuda"),
        A_l=torch.randn(domain, spec, current, device="cuda"),
        alpha_u=torch.rand(domain, current, device="cuda"),
        alpha_l=torch.rand(domain, current, device="cuda"),
        beta_u=torch.randn(domain, current, device="cuda"),
        beta_l=torch.randn(domain, current, device="cuda"),
        weight=weight,
        bias=bias,
        input_shape=input_shape,
        output_shape=output_shape,
        attrs=attrs,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kind", ["linear", "conv2d"])
def test_fused_cache_separates_miss_memory_hit_and_disk_hit(
    tmp_path: Path, kind: str
) -> None:
    request = _request(kind)
    expected = TorchDenseFusedCrownReference().run(request)
    first_cache = FusedCrownModuleCache(tmp_path)
    first_executor = TVMFusedCrownExecutor(compile_cache=first_cache)

    first = first_executor.run(request)
    second = first_executor.run(request)
    restart_cache = FusedCrownModuleCache(tmp_path)
    third = TVMFusedCrownExecutor(compile_cache=restart_cache).run(request)

    assert [event.event for event in first_cache.events] == ["miss", "memory_hit"]
    assert [event.event for event in restart_cache.events] == ["disk_hit"]
    miss = first_cache.events[0]
    assert miss.tir_generation_ms > 0
    assert miss.schedule_ms > 0
    assert miss.tvm_compile_ms > 0
    assert miss.serialization_ms > 0
    assert miss.library_bytes > 0
    assert restart_cache.events[0].module_load_ms > 0
    manifests = list(tmp_path.glob("*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["library_sha256"] == miss.library_sha256

    for actual in (first, second, third):
        torch.testing.assert_close(
            actual.A_prev_u, expected.A_prev_u, rtol=2e-4, atol=2e-4
        )
        torch.testing.assert_close(
            actual.bias_delta_l, expected.bias_delta_l, rtol=2e-4, atol=2e-4
        )
