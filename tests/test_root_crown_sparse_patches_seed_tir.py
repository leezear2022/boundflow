"""Contracts for the `/44` sparse-Patches seed lowering."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from boundflow.backends.tvm.root_crown_sparse_patches_seed import (
    RootCrownSparsePatchesSeedTemplateV1,
    build_root_crown_sparse_patches_seed_modules_v1,
)
from boundflow.runtime.root_crown_sparse_patches_seed_tir import (
    RootCrownSparsePatchesSeedTIRExecutorV1,
)


def _capability() -> str:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}"
    return "sm_89"


def _template() -> RootCrownSparsePatchesSeedTemplateV1:
    return RootCrownSparsePatchesSeedTemplateV1(
        spec_count=7,
        domain_count=1,
        channels=16,
        height=8,
        width=8,
        compute_capability=_capability(),
    )


def test_sparse_patches_seed_template_and_tir_are_deterministic() -> None:
    template = _template()
    template.validate()
    unscheduled, scheduled = build_root_crown_sparse_patches_seed_modules_v1(template)
    assert template.stable_hash() == template.stable_hash()
    assert template.patches_shape == (7, 1, 16, 1, 1)
    assert template.coefficient_shape == (7, 1, 16, 8, 8)
    assert "dense_seed" in unscheduled.script(show_meta=False)
    assert "threadIdx.x" in scheduled.script(show_meta=False)


@pytest.mark.parametrize(
    "changed",
    (
        {"spec_count": 0},
        {"domain_count": 2},
        {"channels": 8},
        {"height": 16},
        {"width": 16},
        {"compute_capability": "cuda"},
        {"target": "llvm"},
    ),
)
def test_sparse_patches_seed_template_rejects_other_abi(
    changed: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="template differs"):
        replace(_template(), **changed).validate()  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sparse_patches_seed_tir_matches_exact_identity_scatter() -> None:
    template = _template()
    executor = RootCrownSparsePatchesSeedTIRExecutorV1(template)
    executor.prepare()
    channel = torch.tensor([0, 2, 4, 6, 8, 10, 12], device="cuda")
    height = torch.tensor([0, 1, 2, 3, 4, 5, 6], device="cuda")
    width = torch.tensor([7, 6, 5, 4, 3, 2, 1], device="cuda")
    patches = torch.zeros(template.patches_shape, device="cuda")
    rows = torch.arange(template.spec_count, device="cuda")
    patches[rows, 0, channel, 0, 0] = 1.0
    patches = patches.contiguous()

    observed = executor.execute(patches, (channel, height, width)).clone()
    expected = torch.zeros(template.coefficient_shape, device="cuda")
    expected[rows, 0, :, height, width] = patches[:, 0, :, 0, 0]
    torch.testing.assert_close(observed, expected, atol=0.0, rtol=0.0)
    receipt = executor.receipt()
    assert receipt["call_count"] == 1
    assert receipt["fallback_count"] == 0
    assert receipt["pointer_count"] == receipt["pointer_exact_count"] == 4
    assert receipt["persistent_dense_seed_arena"] is True

    duplicate_width = width.clone()
    duplicate_width[1] = width[0]
    duplicate_height = height.clone()
    duplicate_height[1] = height[0]
    duplicate_channel = channel.clone()
    duplicate_channel[1] = channel[0]
    duplicate_patches = patches.clone()
    duplicate_patches[1].zero_()
    duplicate_patches[1, 0, duplicate_channel[1], 0, 0] = 1.0
    with pytest.raises(ValueError, match="locations differ"):
        executor.execute(
            duplicate_patches,
            (duplicate_channel, duplicate_height, duplicate_width),
        )
    assert executor.fallback_count == 1
