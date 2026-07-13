#!/usr/bin/env python
"""Run a small calibration-only latency sanity for PR-12 fused tasks."""

# mypy: disable-error-code=import-untyped
# pylint: disable=too-many-locals,too-many-statements,duplicate-code,not-callable

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Callable, Optional, Sequence

import torch
from torch.nn import functional as torch_functional
import tvm

from boundflow.backends.tvm.fused_crown_conv2d import (
    FusedCrownConv2dSignature,
    build_fused_crown_conv2d_module,
)
from boundflow.backends.tvm.fused_crown_linear import (
    FusedCrownLinearKey,
    build_fused_crown_linear_module,
)

SANITY_SCHEMA = "boundflow.pr12-fused-sanity/v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _time_cuda(call: Callable[[], object], *, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def _summary(samples: Sequence[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples),
        "p90_ms": _percentile(samples, 0.9),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _linear_reference(tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    coeff_u, coeff_l, alpha_u, beta_u, alpha_l, beta_l, weight, bias = tensors
    scaled_u = torch.where(
        coeff_u >= 0, coeff_u * alpha_u[:, None], coeff_u * alpha_l[:, None]
    )
    scaled_l = torch.where(
        coeff_l >= 0, coeff_l * alpha_l[:, None], coeff_l * alpha_u[:, None]
    )
    return (
        scaled_u @ weight,
        scaled_l @ weight,
        torch.where(
            coeff_u >= 0, coeff_u * beta_u[:, None], coeff_u * beta_l[:, None]
        ).sum(2)
        + (scaled_u * bias).sum(2),
        torch.where(
            coeff_l >= 0, coeff_l * beta_l[:, None], coeff_l * beta_u[:, None]
        ).sum(2)
        + (scaled_l * bias).sum(2),
    )


def _linear_case(key: FusedCrownLinearKey, *, warmup: int, repeats: int) -> dict:
    torch.manual_seed(1210 + key.spec_batch)
    device = torch.device("cuda")
    tensors = [
        torch.randn(
            key.domain_batch, key.spec_batch, key.current_features, device=device
        ),
        torch.randn(
            key.domain_batch, key.spec_batch, key.current_features, device=device
        ),
        torch.rand(key.domain_batch, key.current_features, device=device),
        torch.randn(key.domain_batch, key.current_features, device=device),
        torch.rand(key.domain_batch, key.current_features, device=device),
        torch.randn(key.domain_batch, key.current_features, device=device),
        torch.randn(key.current_features, key.previous_features, device=device),
        torch.randn(key.current_features, device=device),
    ]
    started = time.perf_counter()
    compiled = build_fused_crown_linear_module(key)
    compile_ms = (time.perf_counter() - started) * 1000.0
    tvm_inputs = [tvm.runtime.from_dlpack(tensor) for tensor in tensors]
    torch_outputs = [
        torch.empty(
            key.domain_batch,
            key.spec_batch,
            key.previous_features,
            device=device,
        ),
        torch.empty(
            key.domain_batch,
            key.spec_batch,
            key.previous_features,
            device=device,
        ),
        torch.empty(key.domain_batch, key.spec_batch, device=device),
        torch.empty(key.domain_batch, key.spec_batch, device=device),
    ]
    tvm_outputs = [tvm.runtime.from_dlpack(tensor) for tensor in torch_outputs]
    expected = _linear_reference(tensors)
    compiled(*tvm_inputs, *tvm_outputs)
    torch.cuda.synchronize()
    max_diff = max(
        float((actual - reference).abs().max().item())
        for actual, reference in zip(torch_outputs, expected)
    )
    tvm_samples = _time_cuda(
        lambda: compiled(*tvm_inputs, *tvm_outputs), warmup=warmup, repeats=repeats
    )
    torch_samples = _time_cuda(
        lambda: _linear_reference(tensors), warmup=warmup, repeats=repeats
    )
    return {
        "schema_version": SANITY_SCHEMA,
        "case_id": (
            f"linear-d{key.domain_batch}-s{key.spec_batch}-"
            f"i{key.current_features}-j{key.previous_features}"
        ),
        "family": "linear",
        "set": "calibration_sanity",
        "compile_ms": compile_ms,
        "fused_tir": _summary(tvm_samples),
        "pytorch_dense_eager": _summary(torch_samples),
        "fused_over_eager_median": statistics.median(tvm_samples)
        / statistics.median(torch_samples),
        "max_abs_diff": max_diff,
        "scaled_a_bytes_avoided": (
            2 * key.domain_batch * key.spec_batch * key.current_features * 4
        ),
    }


def _conv_reference(
    signature: FusedCrownConv2dSignature, tensors: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    coeff_u, coeff_l, alpha_u, beta_u, alpha_l, beta_l, weight = tensors[:7]
    bias = tensors[7] if signature.bias_present else None
    scaled_u = torch.where(
        coeff_u >= 0, coeff_u * alpha_u[:, None], coeff_u * alpha_l[:, None]
    )
    scaled_l = torch.where(
        coeff_l >= 0, coeff_l * alpha_l[:, None], coeff_l * alpha_u[:, None]
    )
    flat = (
        signature.domain_batch * signature.spec_batch,
        signature.output_channels,
        signature.output_height,
        signature.output_width,
    )
    previous_u = torch_functional.conv_transpose2d(
        scaled_u.reshape(flat),
        weight,
        stride=signature.stride,
        padding=signature.padding,
        output_padding=signature.output_padding(),
    ).reshape(
        signature.domain_batch,
        signature.spec_batch,
        signature.input_channels,
        signature.input_height,
        signature.input_width,
    )
    previous_l = torch_functional.conv_transpose2d(
        scaled_l.reshape(flat),
        weight,
        stride=signature.stride,
        padding=signature.padding,
        output_padding=signature.output_padding(),
    ).reshape_as(previous_u)
    delta_u = torch.where(
        coeff_u >= 0, coeff_u * beta_u[:, None], coeff_u * beta_l[:, None]
    )
    delta_l = torch.where(
        coeff_l >= 0, coeff_l * beta_l[:, None], coeff_l * beta_u[:, None]
    )
    if bias is not None:
        bias_view = bias.view(1, 1, -1, 1, 1)
        delta_u = delta_u + scaled_u * bias_view
        delta_l = delta_l + scaled_l * bias_view
    return (
        previous_u,
        previous_l,
        delta_u.sum((2, 3, 4)),
        delta_l.sum((2, 3, 4)),
    )


def _conv_case(
    signature: FusedCrownConv2dSignature, *, warmup: int, repeats: int
) -> dict:
    torch.manual_seed(1220 + signature.spec_batch)
    device = torch.device("cuda")
    coeff_shape = (
        signature.domain_batch,
        signature.spec_batch,
        signature.output_channels,
        signature.output_height,
        signature.output_width,
    )
    relax_shape = (
        signature.domain_batch,
        signature.output_channels,
        signature.output_height,
        signature.output_width,
    )
    tensors = [
        torch.randn(coeff_shape, device=device),
        torch.randn(coeff_shape, device=device),
        torch.rand(relax_shape, device=device),
        torch.randn(relax_shape, device=device),
        torch.rand(relax_shape, device=device),
        torch.randn(relax_shape, device=device),
        torch.randn(
            signature.output_channels,
            signature.input_channels,
            signature.kernel_height,
            signature.kernel_width,
            device=device,
        ),
    ]
    if signature.bias_present:
        tensors.append(torch.randn(signature.output_channels, device=device))
    started = time.perf_counter()
    compiled = build_fused_crown_conv2d_module(signature)
    compile_ms = (time.perf_counter() - started) * 1000.0
    tvm_inputs = [tvm.runtime.from_dlpack(tensor) for tensor in tensors]
    previous_shape = (
        signature.domain_batch,
        signature.spec_batch,
        signature.input_channels,
        signature.input_height,
        signature.input_width,
    )
    torch_outputs = [
        torch.empty(previous_shape, device=device),
        torch.empty(previous_shape, device=device),
        torch.empty(signature.domain_batch, signature.spec_batch, device=device),
        torch.empty(signature.domain_batch, signature.spec_batch, device=device),
    ]
    tvm_outputs = [tvm.runtime.from_dlpack(tensor) for tensor in torch_outputs]
    expected = _conv_reference(signature, tensors)
    compiled(*tvm_inputs, *tvm_outputs)
    torch.cuda.synchronize()
    max_diff = max(
        float((actual - reference).abs().max().item())
        for actual, reference in zip(torch_outputs, expected)
    )
    tvm_samples = _time_cuda(
        lambda: compiled(*tvm_inputs, *tvm_outputs), warmup=warmup, repeats=repeats
    )
    torch_samples = _time_cuda(
        lambda: _conv_reference(signature, tensors), warmup=warmup, repeats=repeats
    )
    return {
        "schema_version": SANITY_SCHEMA,
        "case_id": (
            f"conv-d{signature.domain_batch}-s{signature.spec_batch}-"
            f"ci{signature.input_channels}-co{signature.output_channels}-"
            f"h{signature.input_height}-k{signature.kernel_height}-"
            f"stride{signature.stride[0]}"
        ),
        "family": "conv2d",
        "set": "calibration_sanity",
        "compile_ms": compile_ms,
        "fused_tir": _summary(tvm_samples),
        "pytorch_dense_eager": _summary(torch_samples),
        "fused_over_eager_median": statistics.median(tvm_samples)
        / statistics.median(torch_samples),
        "max_abs_diff": max_diff,
        "scaled_a_bytes_avoided": 2
        * math.prod(coeff_shape)
        * torch.tensor([], dtype=torch.float32).element_size(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run fixed calibration points without consuming the final held-out split."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    cases = [
        _linear_case(
            FusedCrownLinearKey(2, 8, 16, 12), warmup=args.warmup, repeats=args.repeats
        ),
        _linear_case(
            FusedCrownLinearKey(4, 32, 64, 48), warmup=args.warmup, repeats=args.repeats
        ),
        _conv_case(
            FusedCrownConv2dSignature(1, 3, 5, 7, 7, 4, 7, 7, 3, 3, (1, 1), (1, 1)),
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        _conv_case(
            FusedCrownConv2dSignature(2, 8, 8, 16, 16, 8, 8, 8, 3, 3, (2, 2), (1, 1)),
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    ]
    raw_path = args.out_dir / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(case, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "boundflow.pr12-fused-sanity-manifest/v1",
        "final_heldout_consumed": False,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "case_count": len(cases),
        "outputs": {"raw.jsonl": _sha256(raw_path)},
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
