#!/usr/bin/env python
"""Freeze post-codegen spill and kernel evidence for PR-12 fused tasks."""

# mypy: disable-error-code=import-untyped

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Optional, Sequence

import tvm

from boundflow.backends.tvm.fused_crown_conv2d import (
    FusedCrownConv2dSignature,
    allocated_intermediate_buffers as conv_allocations,
    schedule_fused_crown_conv2d,
)
from boundflow.backends.tvm.fused_crown_linear import (
    FusedCrownLinearKey,
    allocated_intermediate_buffers as linear_allocations,
    schedule_fused_crown_linear,
)

CODEGEN_PROFILE_SCHEMA = "boundflow.pr12-codegen-profile/v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_ptxas(stderr: str) -> dict[str, dict[str, int]]:
    """Parse per-kernel stack, spill, and register metrics from ptxas ``-v``."""

    metrics: dict[str, dict[str, int]] = {}
    pattern = re.compile(
        r"Function properties for (?P<name>\S+)\n"
        r"\s+(?P<stack>\d+) bytes stack frame, "
        r"(?P<stores>\d+) bytes spill stores, "
        r"(?P<loads>\d+) bytes spill loads\n"
        r"ptxas info\s+: Used (?P<registers>\d+) registers"
    )
    for match in pattern.finditer(stderr):
        metrics[match.group("name")] = {
            "stack_frame_bytes": int(match.group("stack")),
            "spill_store_bytes": int(match.group("stores")),
            "spill_load_bytes": int(match.group("loads")),
            "registers_per_thread": int(match.group("registers")),
        }
    return metrics


def parse_cuda_kernel_names(source: str) -> list[str]:
    """Return definitions only, excluding identical forward declarations."""

    return sorted(
        set(
            re.findall(
                r'extern "C" __global__ void __launch_bounds__\(\d+\)\s+'
                r"([A-Za-z_][A-Za-z0-9_]*)\([^;]+\)\s*\{",
                source,
            )
        )
    )


def _compile_evidence(
    *, name: str, ir_module: object, target: str, allocations: Sequence[str], out: Path
) -> dict[str, Any]:
    executable = tvm.compile(ir_module, target=target)
    cuda_source = executable.mod.imports[0].inspect_source()
    source_path = out / f"{name}.cu"
    ptx_path = out / f"{name}.ptx"
    cubin_path = out / f"{name}.cubin"
    source_path.write_text(cuda_source, encoding="utf-8")
    cubin = subprocess.run(
        [
            "nvcc",
            "-cubin",
            "-arch=sm_89",
            "-Xptxas=-v",
            str(source_path),
            "-o",
            str(cubin_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        ["nvcc", "-ptx", "-arch=sm_89", str(source_path), "-o", str(ptx_path)],
        check=True,
        text=True,
        capture_output=True,
    )
    ptx = ptx_path.read_text(encoding="utf-8")
    kernel_names = parse_cuda_kernel_names(cuda_source)
    ptxas = parse_ptxas(cubin.stderr)
    if set(kernel_names) != set(ptxas):
        raise RuntimeError(
            f"kernel/PTXAS mismatch for {name}: source={kernel_names}, ptxas={ptxas}"
        )
    return {
        "name": name,
        "target": target,
        "kernel_count": len(kernel_names),
        "kernel_names": kernel_names,
        "intermediate_allocations": list(allocations),
        "forbidden_source_tokens": {
            "A_scaled": cuda_source.count("A_scaled"),
            "im2col": cuda_source.count("im2col"),
        },
        "ptx_local_declarations": len(re.findall(r"\.local", ptx)),
        "ptx_local_loads": len(re.findall(r"ld\.local", ptx)),
        "ptx_local_stores": len(re.findall(r"st\.local", ptx)),
        "ptxas": ptxas,
        "files": {
            source_path.name: _sha256(source_path),
            ptx_path.name: _sha256(ptx_path),
            cubin_path.name: _sha256(cubin_path),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate codegen evidence for one representative Linear and Conv task."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    linear = FusedCrownLinearKey(2, 8, 16, 12)
    conv = FusedCrownConv2dSignature(1, 3, 5, 7, 7, 4, 7, 7, 3, 3, (1, 1), (1, 1))
    conv_stride_two = FusedCrownConv2dSignature(
        2, 8, 8, 16, 16, 8, 8, 8, 3, 3, (2, 2), (1, 1)
    )
    records = [
        _compile_evidence(
            name="linear_d2_s8_i16_j12",
            ir_module=schedule_fused_crown_linear(linear),
            target=linear.target_string,
            allocations=linear_allocations(linear),
            out=args.out_dir,
        ),
        _compile_evidence(
            name="conv_d1_s3_ci5_co4_h7_k3_s1",
            ir_module=schedule_fused_crown_conv2d(conv),
            target=conv.target_string,
            allocations=conv_allocations(conv, scheduled=True),
            out=args.out_dir,
        ),
        _compile_evidence(
            name="conv_d2_s8_ci8_co8_h16_k3_s2",
            ir_module=schedule_fused_crown_conv2d(conv_stride_two),
            target=conv_stride_two.target_string,
            allocations=conv_allocations(conv_stride_two, scheduled=True),
            out=args.out_dir,
        ),
    ]
    raw_path = args.out_dir / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": CODEGEN_PROFILE_SCHEMA,
        "record_count": len(records),
        "nvcc_version": subprocess.check_output(["nvcc", "--version"], text=True),
        "all_zero_spill": all(
            all(
                kernel[metric] == 0
                for metric in (
                    "stack_frame_bytes",
                    "spill_store_bytes",
                    "spill_load_bytes",
                )
            )
            for record in records
            for kernel in record["ptxas"].values()
        ),
        "outputs": {"raw.jsonl": _sha256(raw_path)},
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
