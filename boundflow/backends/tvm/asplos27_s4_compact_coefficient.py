"""Compact-parameter coefficient kernels for the S4 all-state evaluator.

The old R3 kernels consume production-shaped ``[2, 1, D, W]`` alpha tensors.
S4 owns only the mutable lower slice ``[D, W]``.  These ABI-specialized TIR
functions execute the same recurrence directly from that compact owner, so an
optimizer step never needs to materialize or copy a dense production shell.
"""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,protected-access
# pylint: disable=duplicate-code,missing-function-docstring,too-many-locals
# pylint: disable=too-many-arguments,too-many-positional-arguments

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)
from boundflow.backends.tvm.r3_d1_residual11_staged import (
    _schedule as _schedule_residual11,
)
from boundflow.backends.tvm.r3_d1_residual6_staged import (
    _schedule as _schedule_residual6,
)
from boundflow.backends.tvm.r3_full_lower_forward import (
    R31B1_THREADS,
    _schedule_te_primfunc,
)

S4_COMPACT_RELU31 = "boundflow_s4_compact_relu31_coeff"
S4_COMPACT_RELU31_BIAS = "boundflow_s4_compact_relu31_bias"
S4_COMPACT_RELU28 = "boundflow_s4_compact_relu28_coeff"
S4_COMPACT_RELU28_BIAS = "boundflow_s4_compact_relu28_bias"
S4_COMPACT_RELU23 = "boundflow_s4_compact_relu23_coeff"
S4_COMPACT_RELU23_BIAS = "boundflow_s4_compact_relu23_bias"
S4_COMPACT_RELU17 = "boundflow_s4_compact_relu17_coeff"
S4_COMPACT_RELU17_BIAS = "boundflow_s4_compact_relu17_bias"
S4_COMPACT_RESIDUAL11_STAGE2 = "boundflow_s4_compact_residual11_stage2"
S4_COMPACT_RESIDUAL6_STAGE2 = "boundflow_s4_compact_residual6_stage2"
S4_COMPACT_COEFFICIENT_SYMBOLS = (
    S4_COMPACT_RELU31,
    S4_COMPACT_RELU31_BIAS,
    S4_COMPACT_RELU28,
    S4_COMPACT_RELU28_BIAS,
    S4_COMPACT_RELU23,
    S4_COMPACT_RELU23_BIAS,
    S4_COMPACT_RELU17,
    S4_COMPACT_RELU17_BIAS,
    S4_COMPACT_RESIDUAL11_STAGE2,
    S4_COMPACT_RESIDUAL6_STAGE2,
)


@dataclass(frozen=True)
class CompiledS4CompactCoefficientV1:
    """Compiled compact ABI module plus deterministic compiler receipts."""

    executable: DifferentiableLowerTIRExecutable
    scheduled_tir_hash: str
    device_source_hash: str
    exported_symbols: tuple[str, ...]
    global_workspace_bytes: int
    tvm_version: str

    def validate(self) -> None:
        """Validate the frozen compiler identity fields."""

        if (
            self.exported_symbols != S4_COMPACT_COEFFICIENT_SYMBOLS
            or len(self.scheduled_tir_hash) != 64
            or len(self.device_source_hash) != 64
            or self.global_workspace_bytes != 0
        ):
            raise ValueError("S4 compact coefficient receipt differs")


def _rename(primfunc, symbol: str):  # type: ignore[no-untyped-def]
    return (
        primfunc.with_attr("global_symbol", symbol)
        .with_attr("boundflow.schema_version", "asplos27-s4-compact-coefficient/v1")
        .without_attr("tir.noalias")
    )


def _compact_slope_and_intercept(
    tvm, incoming, lower, upper, alpha, alpha_map, d_idx, feature
):
    """Return the frozen ReLU relaxation from compact ``[D, W]`` alpha."""

    zero = tvm.tir.const(0.0, "float32")
    one = tvm.tir.const(1.0, "float32")
    lookup = alpha_map[feature]
    lower_alpha = tvm.tir.if_then_else(
        lookup >= 0,
        tvm.tir.min(tvm.tir.max(alpha[d_idx, tvm.tir.max(lookup, 0)], zero), one),
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


def _compact_relu_bias_primfunc(symbol: str, *, feature_count: int, alpha_width: int):
    import tvm
    from tvm import te

    incoming = te.placeholder((6 * feature_count,), "float32", name="incoming_arena")
    lower = te.placeholder((6, feature_count), "float32", name="lower")
    upper = te.placeholder((6, feature_count), "float32", name="upper")
    alpha = te.placeholder((6, alpha_width), "float32", name="compact_alpha")
    alpha_map = te.placeholder((feature_count,), "int32", name="alpha_map")
    bias_in = te.placeholder((6,), "float32", name="bias_in")
    reduce_feature = te.reduce_axis((0, feature_count + 1), "reduce_feature")

    def contribution(d_idx):
        safe_feature = tvm.tir.min(reduce_feature, feature_count - 1)
        value = incoming[d_idx * feature_count + safe_feature]
        _slope, intercept = _compact_slope_and_intercept(
            tvm,
            value,
            lower,
            upper,
            alpha,
            alpha_map,
            d_idx,
            safe_feature,
        )
        return tvm.tir.if_then_else(
            reduce_feature == feature_count,
            bias_in[d_idx],
            value * intercept,
        )

    bias_out = te.compute(
        (6,),
        lambda d_idx: te.sum(contribution(d_idx), axis=reduce_feature),
        name="relu_bias",
    )
    return (
        te.create_prim_func(
            [incoming, lower, upper, alpha, alpha_map, bias_in, bias_out]
        )
        .with_attr("global_symbol", symbol)
        .with_attr("boundflow.schema_version", "asplos27-s4-compact-coefficient/v1")
    )


def _compact_relu_coeff_primfunc(
    symbol: str,
    *,
    feature_count: int,
    alpha_width: int,
    beta_active: bool = False,
):
    import tvm
    from tvm import te

    incoming = te.placeholder((6 * feature_count,), "float32", name="incoming_arena")
    lower = te.placeholder((6, feature_count), "float32", name="lower")
    upper = te.placeholder((6, feature_count), "float32", name="upper")
    alpha = te.placeholder((6, alpha_width), "float32", name="compact_alpha")
    alpha_map = te.placeholder((feature_count,), "int32", name="alpha_map")
    beta = te.placeholder((6, 1), "float32", name="beta") if beta_active else None
    beta_map = (
        te.placeholder((6, feature_count), "int32", name="beta_map")
        if beta_active
        else None
    )
    split = (
        te.placeholder((6, feature_count), "int8", name="split")
        if beta_active
        else None
    )

    def coefficient(flat):
        d_idx = flat // feature_count
        f_idx = flat % feature_count
        value = incoming[flat]
        slope, _intercept = _compact_slope_and_intercept(
            tvm, value, lower, upper, alpha, alpha_map, d_idx, f_idx
        )
        beta_add = tvm.tir.const(0.0, "float32")
        if beta_active:
            assert beta is not None and beta_map is not None and split is not None
            location = beta_map[d_idx, f_idx]
            beta_add = tvm.tir.if_then_else(
                location >= 0,
                -beta[d_idx, tvm.tir.max(location, 0)]
                * tvm.tir.Cast("float32", split[d_idx, f_idx]),
                tvm.tir.const(0.0, "float32"),
            )
        return value * slope + beta_add

    output = te.compute((6 * feature_count,), coefficient, name="relu_coefficient")
    arguments = [incoming, lower, upper, alpha, alpha_map]
    if beta_active:
        assert beta is not None and beta_map is not None and split is not None
        arguments.extend((beta, beta_map, split))
    arguments.append(output)
    return (
        te.create_prim_func(arguments)
        .with_attr("global_symbol", symbol)
        .with_attr("boundflow.schema_version", "asplos27-s4-compact-coefficient/v1")
    )


def _compact_residual11_stage2_primfunc():
    import tvm
    from tvm import te

    incoming = te.placeholder((18_432,), "float32", name="incoming_arena")
    staged = te.placeholder((6 * 1024,), "float32", name="staged")
    lower = te.placeholder((6, 1024), "float32", name="lower25")
    upper = te.placeholder((6, 1024), "float32", name="upper25")
    alpha = te.placeholder((6, 86), "float32", name="compact_alpha25")
    alpha_map = te.placeholder((1024,), "int32", name="alpha_map25")
    weight8 = te.placeholder((16, 16, 3, 3), "float32", name="weight8")
    bias10 = te.placeholder((16,), "float32", name="bias10")
    bias8 = te.placeholder((16,), "float32", name="bias8")
    bias_in = te.placeholder((6,), "float32", name="bias_in")
    reduction = te.reduce_axis((0, 16 * 3 * 3 + 1), "reduce_conv8_skip")

    def output_value(flat):
        d_idx = flat // 1024
        logical = flat % 1024
        input_w = logical % 8
        input_h = logical // 8 % 8
        input_channel = logical // 64
        safe = tvm.tir.min(reduction, 16 * 3 * 3 - 1)
        mid_channel = safe // 9
        kernel_h = safe // 3 % 3
        kernel_w = safe % 3
        mid_h = input_h + 1 - kernel_h
        mid_w = input_w + 1 - kernel_w
        valid = tvm.tir.all(
            reduction < 16 * 3 * 3,
            0 <= mid_h,
            mid_h < 8,
            0 <= mid_w,
            mid_w < 8,
        )
        feature = mid_channel * 64 + mid_h * 8 + mid_w
        safe_feature = tvm.tir.min(tvm.tir.max(feature, 0), 1023)
        coefficient = staged[d_idx * 1024 + safe_feature]
        slope, _intercept = _compact_slope_and_intercept(
            tvm, coefficient, lower, upper, alpha, alpha_map, d_idx, safe_feature
        )
        term = tvm.tir.if_then_else(
            reduction == 16 * 3 * 3,
            incoming[flat],
            tvm.tir.if_then_else(
                valid,
                coefficient
                * slope
                * weight8[mid_channel, input_channel, kernel_h, kernel_w],
                tvm.tir.const(0.0, "float32"),
            ),
        )
        return te.sum(term, axis=reduction)

    output = te.compute((6 * 1024,), output_value, name="stage2_output")
    bias_reduce = te.reduce_axis((0, 1024 * 2 + 1), "reduce_bias")

    def bias_value(d_idx):
        safe = tvm.tir.min(bias_reduce, 2047)
        feature = safe % 1024
        channel = feature // 64
        coefficient = staged[d_idx * 1024 + feature]
        slope, intercept = _compact_slope_and_intercept(
            tvm, coefficient, lower, upper, alpha, alpha_map, d_idx, feature
        )
        contribution = tvm.tir.if_then_else(
            bias_reduce < 1024,
            incoming[d_idx * 1024 + feature] * bias10[channel],
            tvm.tir.if_then_else(
                bias_reduce < 2048,
                coefficient * intercept + coefficient * slope * bias8[channel],
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
                weight8,
                bias10,
                bias8,
                bias_in,
                output,
                bias_out,
            ]
        )
        .with_attr("global_symbol", S4_COMPACT_RESIDUAL11_STAGE2)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "asplos27-s4-compact-coefficient/v1")
    )


def _compact_residual6_stage2_primfunc():
    import tvm
    from tvm import te

    incoming = te.placeholder((18_432,), "float32", name="incoming_arena")
    staged = te.placeholder((6 * 1024,), "float32", name="staged")
    lower = te.placeholder((6, 1024), "float32", name="lower19")
    upper = te.placeholder((6, 1024), "float32", name="upper19")
    alpha = te.placeholder((6, 132), "float32", name="compact_alpha19")
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
        slope, _intercept = _compact_slope_and_intercept(
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
        slope, intercept = _compact_slope_and_intercept(
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
        .with_attr("global_symbol", S4_COMPACT_RESIDUAL6_STAGE2)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "asplos27-s4-compact-coefficient/v1")
    )


def build_s4_compact_coefficient_tir_v1():  # type: ignore[no-untyped-def]
    """Build six compact-alpha coefficient functions with frozen schedules."""

    import tvm

    relu_specs = (
        (S4_COMPACT_RELU31, S4_COMPACT_RELU31_BIAS, 100, 27, True),
        (S4_COMPACT_RELU28, S4_COMPACT_RELU28_BIAS, 1024, 178, False),
        (S4_COMPACT_RELU23, S4_COMPACT_RELU23_BIAS, 1024, 121, False),
        (S4_COMPACT_RELU17, S4_COMPACT_RELU17_BIAS, 2048, 164, False),
    )
    scheduled = {}
    for symbol, bias_symbol, feature_count, alpha_width, beta_active in relu_specs:
        raw = (
            _compact_relu_coeff_primfunc(
                symbol,
                feature_count=feature_count,
                alpha_width=alpha_width,
                beta_active=beta_active,
            )
            .with_attr("boundflow.schema_version", "asplos27-s4-compact-coefficient/v1")
            .without_attr("tir.noalias")
        )
        scheduled[symbol] = _schedule_te_primfunc(
            tvm,
            symbol,
            raw,
            (("relu_coefficient", False, R31B1_THREADS),),
        )
        bias_raw = (
            _compact_relu_bias_primfunc(
                bias_symbol,
                feature_count=feature_count,
                alpha_width=alpha_width,
            )
            .with_attr("boundflow.schema_version", "asplos27-s4-compact-coefficient/v1")
            .without_attr("tir.noalias")
        )
        scheduled[bias_symbol] = _schedule_te_primfunc(
            tvm,
            bias_symbol,
            bias_raw,
            (("relu_bias", True, 1),),
        )

    residual11 = _rename(
        _compact_residual11_stage2_primfunc(),
        S4_COMPACT_RESIDUAL11_STAGE2,
    )
    residual6 = _rename(
        _compact_residual6_stage2_primfunc(),
        S4_COMPACT_RESIDUAL6_STAGE2,
    )
    scheduled[S4_COMPACT_RESIDUAL11_STAGE2] = _schedule_residual11(
        tvm,
        S4_COMPACT_RESIDUAL11_STAGE2,
        residual11,
        (("stage2_output", True, 256), ("stage2_bias", True, 1)),
    )
    scheduled[S4_COMPACT_RESIDUAL6_STAGE2] = _schedule_residual6(
        tvm,
        S4_COMPACT_RESIDUAL6_STAGE2,
        residual6,
        (("stage2_output", True, 256), ("stage2_bias", True, 1)),
    )
    return tvm.IRModule(scheduled)


def compile_s4_compact_coefficient_v1(
    *, compute_capability: str = "sm_89"
) -> CompiledS4CompactCoefficientV1:
    """Compile the direct compact-state recurrence without autotuning."""

    import tvm

    module = build_s4_compact_coefficient_tir_v1()
    executable = tvm.compile(module, target=f"cuda -arch={compute_capability}")
    sources = tuple(imported.inspect_source() for imported in executable.mod.imports)
    if not sources:
        raise RuntimeError("S4 compact coefficient compile produced no CUDA source")
    source = "\n".join(sources)
    if any(symbol not in source for symbol in S4_COMPACT_COEFFICIENT_SYMBOLS):
        raise RuntimeError("S4 compact coefficient symbol is absent")
    result = CompiledS4CompactCoefficientV1(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        scheduled_tir_hash=hashlib.sha256(
            tvm.ir.save_json(module).encode()
        ).hexdigest(),
        device_source_hash=hashlib.sha256(source.encode()).hexdigest(),
        exported_symbols=S4_COMPACT_COEFFICIENT_SYMBOLS,
        global_workspace_bytes=0,
        tvm_version=str(tvm.__version__),
    )
    result.validate()
    return result


__all__ = [
    "CompiledS4CompactCoefficientV1",
    "S4_COMPACT_COEFFICIENT_SYMBOLS",
    "S4_COMPACT_RELU17",
    "S4_COMPACT_RELU17_BIAS",
    "S4_COMPACT_RELU23",
    "S4_COMPACT_RELU23_BIAS",
    "S4_COMPACT_RELU28",
    "S4_COMPACT_RELU28_BIAS",
    "S4_COMPACT_RELU31",
    "S4_COMPACT_RELU31_BIAS",
    "S4_COMPACT_RESIDUAL11_STAGE2",
    "S4_COMPACT_RESIDUAL6_STAGE2",
    "build_s4_compact_coefficient_tir_v1",
    "compile_s4_compact_coefficient_v1",
]
