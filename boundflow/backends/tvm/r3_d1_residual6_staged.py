"""D1-A two-stage residual6 factorization for correctness qualification."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,missing-function-docstring
# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)

R3D1_RESIDUAL6_STAGE1_SYMBOL = "boundflow_r3d1_residual6_stage1"
R3D1_RESIDUAL6_STAGE2_SYMBOL = "boundflow_r3d1_residual6_stage2"
R3D1_RESIDUAL6_SYMBOLS = (
    R3D1_RESIDUAL6_STAGE1_SYMBOL,
    R3D1_RESIDUAL6_STAGE2_SYMBOL,
)
R3D1_THREADS = 128


@dataclass(frozen=True)
class CompiledR3D1Residual6StagedV1:
    """Compiled two-symbol residual6 module and receipts."""

    executable: DifferentiableLowerTIRExecutable
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    exported_symbols: tuple[str, ...]
    global_workspace_bytes: int
    tvm_version: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _schedule(tvm, symbol: str, primfunc, blocks):
    schedule = tvm.tir.Schedule(tvm.IRModule({symbol: primfunc}))
    for block_name, reduction, threads in blocks:
        block = schedule.get_block(block_name, func_name=symbol)
        loops = schedule.get_loops(block)
        spatial = loops[:-1] if reduction else loops
        fused = schedule.fuse(*spatial) if len(spatial) > 1 else spatial[0]
        outer, inner = schedule.split(fused, factors=[None, threads])
        schedule.bind(outer, "blockIdx.x")
        schedule.bind(inner, "threadIdx.x")
    return schedule.mod[symbol]


def _stage1_primfunc():
    import tvm
    from tvm import te

    incoming = te.placeholder((18_432,), "float32", name="incoming_arena")
    weight4 = te.placeholder((16, 16, 3, 3), "float32", name="weight4")
    reduction = te.reduce_axis((0, 16 * 3 * 3), "reduce_conv4")

    def value(flat):
        d_idx = flat // 1024
        logical = flat % 1024
        mid_w = logical % 8
        mid_h = logical // 8 % 8
        mid_channel = logical // 64
        kernel_w = reduction % 3
        kernel_h = reduction // 3 % 3
        out_channel = reduction // 9
        out_h = mid_h + 1 - kernel_h
        out_w = mid_w + 1 - kernel_w
        valid = tvm.tir.all(0 <= out_h, out_h < 8, 0 <= out_w, out_w < 8)
        source = d_idx * 1024 + out_channel * 64 + out_h * 8 + out_w
        return te.sum(
            tvm.tir.if_then_else(
                valid,
                incoming[source]
                * weight4[out_channel, mid_channel, kernel_h, kernel_w],
                tvm.tir.const(0.0, "float32"),
            ),
            axis=reduction,
        )

    staged = te.compute((6 * 1024,), value, name="stage1_output")
    return (
        te.create_prim_func([incoming, weight4, staged])
        .with_attr("global_symbol", R3D1_RESIDUAL6_STAGE1_SYMBOL)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "r3-d1-residual6-staged/v1")
    )


def _slope_and_intercept(tvm, incoming, lower, upper, alpha, alpha_map, d_idx, feature):
    zero = tvm.tir.const(0.0, "float32")
    one = tvm.tir.const(1.0, "float32")
    lookup = alpha_map[feature]
    lower_alpha = tvm.tir.if_then_else(
        lookup >= 0,
        tvm.tir.min(tvm.tir.max(alpha[0, 0, d_idx, tvm.tir.max(lookup, 0)], zero), one),
        zero,
    )
    ambiguous = tvm.tir.all(lower[d_idx, feature] < zero, upper[d_idx, feature] > zero)
    lower_slope = tvm.tir.if_then_else(
        ambiguous,
        lower_alpha,
        tvm.tir.if_then_else(lower[d_idx, feature] >= zero, one, zero),
    )
    upper_slope = tvm.tir.if_then_else(
        lower[d_idx, feature] >= zero,
        one,
        tvm.tir.if_then_else(
            upper[d_idx, feature] <= zero,
            zero,
            upper[d_idx, feature]
            / tvm.tir.max(
                upper[d_idx, feature] - lower[d_idx, feature],
                tvm.tir.const(1.1920928955078125e-07, "float32"),
            ),
        ),
    )
    slope = tvm.tir.if_then_else(incoming >= zero, lower_slope, upper_slope)
    intercept = tvm.tir.if_then_else(
        tvm.tir.all(incoming < zero, ambiguous),
        -lower[d_idx, feature] * upper_slope,
        zero,
    )
    return slope, intercept


def _stage2_primfunc():
    import tvm
    from tvm import te

    incoming = te.placeholder((18_432,), "float32", name="incoming_arena")
    staged = te.placeholder((6 * 1024,), "float32", name="staged")
    lower = te.placeholder((6, 1024), "float32", name="lower19")
    upper = te.placeholder((6, 1024), "float32", name="upper19")
    alpha = te.placeholder((2, 1, 6, 132), "float32", name="alpha19")
    alpha_map = te.placeholder((1024,), "int32", name="alpha_map19")
    weight2 = te.placeholder((16, 8, 3, 3), "float32", name="weight2")
    weight5 = te.placeholder((16, 8, 1, 1), "float32", name="weight5")
    bias4 = te.placeholder((16,), "float32", name="bias4")
    bias2 = te.placeholder((16,), "float32", name="bias2")
    bias5 = te.placeholder((16,), "float32", name="bias5")
    bias_in = te.placeholder((6,), "float32", name="bias_in")
    reduction = te.reduce_axis((0, 16 * 3 * 3 + 16), "reduce_main_shortcut")

    def output_value(flat):
        d_idx = flat // 2048
        logical = flat % 2048
        input_w = logical % 16
        input_h = logical // 16 % 16
        input_channel = logical // 256
        main = reduction < 16 * 3 * 3
        safe_main = tvm.tir.min(reduction, 16 * 3 * 3 - 1)
        mid_channel = safe_main // 9
        kernel_h = safe_main // 3 % 3
        kernel_w = safe_main % 3
        numerator_h = input_h + 1 - kernel_h
        numerator_w = input_w + 1 - kernel_w
        mid_h = numerator_h // 2
        mid_w = numerator_w // 2
        valid_main = tvm.tir.all(
            main,
            numerator_h >= 0,
            numerator_w >= 0,
            numerator_h % 2 == 0,
            numerator_w % 2 == 0,
            mid_h < 8,
            mid_w < 8,
        )
        safe_feature = tvm.tir.min(
            tvm.tir.max(mid_channel * 64 + mid_h * 8 + mid_w, 0), 1023
        )
        coefficient = staged[d_idx * 1024 + safe_feature]
        slope, _intercept = _slope_and_intercept(
            tvm, coefficient, lower, upper, alpha, alpha_map, d_idx, safe_feature
        )
        main_term = tvm.tir.if_then_else(
            valid_main,
            coefficient
            * slope
            * weight2[mid_channel, input_channel, kernel_h, kernel_w],
            tvm.tir.const(0.0, "float32"),
        )
        out_channel = tvm.tir.max(reduction - 16 * 3 * 3, 0)
        valid_shortcut = tvm.tir.all(
            reduction >= 16 * 3 * 3, input_h % 2 == 0, input_w % 2 == 0
        )
        source = d_idx * 1024 + out_channel * 64 + (input_h // 2) * 8 + input_w // 2
        shortcut_term = tvm.tir.if_then_else(
            valid_shortcut,
            incoming[source] * weight5[out_channel, input_channel, 0, 0],
            tvm.tir.const(0.0, "float32"),
        )
        return te.sum(
            tvm.tir.if_then_else(main, main_term, shortcut_term), axis=reduction
        )

    output = te.compute((6 * 2048,), output_value, name="stage2_output")
    bias_reduce = te.reduce_axis((0, 1024 * 2 + 1), "reduce_bias")

    def bias_value(d_idx):
        safe = tvm.tir.min(bias_reduce, 2047)
        feature = safe % 1024
        channel = feature // 64
        coefficient = staged[d_idx * 1024 + feature]
        slope, intercept = _slope_and_intercept(
            tvm, coefficient, lower, upper, alpha, alpha_map, d_idx, feature
        )
        contribution = tvm.tir.if_then_else(
            bias_reduce < 1024,
            incoming[d_idx * 1024 + feature] * (bias4[channel] + bias5[channel]),
            tvm.tir.if_then_else(
                bias_reduce < 2048,
                coefficient * intercept + coefficient * slope * bias2[channel],
                bias_in[d_idx],
            ),
        )
        return te.sum(contribution, axis=bias_reduce)

    bias_out = te.compute((6,), bias_value, name="stage2_bias")
    return (
        te.create_prim_func(
            [
                incoming,
                staged,
                lower,
                upper,
                alpha,
                alpha_map,
                weight2,
                weight5,
                bias4,
                bias2,
                bias5,
                bias_in,
                output,
                bias_out,
            ]
        )
        .with_attr("global_symbol", R3D1_RESIDUAL6_STAGE2_SYMBOL)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "r3-d1-residual6-staged/v1")
    )


def build_r3d1_residual6_staged_modules_v1():
    """Build unscheduled and fixed 128-thread scheduled modules."""

    import tvm

    stage1 = _stage1_primfunc()
    stage2 = _stage2_primfunc()
    unscheduled = tvm.IRModule(
        {
            R3D1_RESIDUAL6_STAGE1_SYMBOL: stage1,
            R3D1_RESIDUAL6_STAGE2_SYMBOL: stage2,
        }
    )
    scheduled = tvm.IRModule(
        {
            R3D1_RESIDUAL6_STAGE1_SYMBOL: _schedule(
                tvm,
                R3D1_RESIDUAL6_STAGE1_SYMBOL,
                stage1,
                (("stage1_output", True, R3D1_THREADS),),
            ),
            R3D1_RESIDUAL6_STAGE2_SYMBOL: _schedule(
                tvm,
                R3D1_RESIDUAL6_STAGE2_SYMBOL,
                stage2,
                (("stage2_output", True, R3D1_THREADS), ("stage2_bias", True, 1)),
            ),
        }
    )
    return unscheduled, scheduled


def compile_r3d1_residual6_staged_v1(
    *, compute_capability: str = "sm_89"
) -> CompiledR3D1Residual6StagedV1:
    """Compile D1-A residual6 symbols without a global workspace."""

    import tvm

    unscheduled, scheduled = build_r3d1_residual6_staged_modules_v1()
    executable = tvm.compile(scheduled, target=f"cuda -arch={compute_capability}")
    sources = tuple(module.inspect_source() for module in executable.mod.imports)
    if not sources:
        raise RuntimeError("R3-D1 residual6 compile produced no CUDA source")
    return CompiledR3D1Residual6StagedV1(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        unscheduled_tir_hash=_sha256(tvm.ir.save_json(unscheduled)),
        scheduled_tir_hash=_sha256(tvm.ir.save_json(scheduled)),
        device_source_hash=_sha256("\n".join(sources)),
        exported_symbols=R3D1_RESIDUAL6_SYMBOLS,
        global_workspace_bytes=0,
        tvm_version=str(tvm.__version__),
    )


__all__ = [
    "R3D1_RESIDUAL6_SYMBOLS",
    "build_r3d1_residual6_staged_modules_v1",
    "compile_r3d1_residual6_staged_v1",
]
