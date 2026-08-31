"""S4-1C compressed-gradient emitters and terminal-lA copy kernels.

The module is deliberately backend-local.  It does not introduce another
solver IR and it does not own optimizer mutation or timing.  Six alpha VJPs,
the single active beta VJP, and six phase-safe copy operations are compiled as
thirteen explicit TIR symbols.
"""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,invalid-name
# pylint: disable=missing-function-docstring,too-many-locals
# pylint: disable=missing-class-docstring,too-many-instance-attributes
# pylint: disable=too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)
from boundflow.backends.tvm.r3_full_lower_forward import _schedule_te_primfunc

S4_COMPRESSED_GRADIENT_SCHEMA_V1 = "boundflow.asplos27-s4-compressed-gradient/v1"
S4_COMPRESSED_GRADIENT_CONSTRUCTION_HASH_V1 = (
    "ad8ea91c39419cbfef0cf3eaa8db7fc339e54798daecf67ca6d97254a9755b93"
)
S4_COMPRESSED_GRADIENT_THREADS_V1 = 128
S4_COMPRESSED_GRADIENT_QNAN_BITS_V1 = 0x7FC00000


@dataclass(frozen=True)
class S4GradientSiteSpecV1:
    site_id: int
    feature_count: int
    alpha_width: int
    terminal_shape: tuple[int, ...]

    @property
    def dalpha_symbol(self) -> str:
        return f"boundflow_s4_emit_dalpha_{self.site_id}"

    @property
    def copy_symbol(self) -> str:
        return f"boundflow_s4_copy_terminal_la_{self.site_id}"


S4_GRADIENT_SITE_SPECS_V1 = (
    S4GradientSiteSpecV1(17, 2048, 164, (6, 1, 8, 16, 16)),
    S4GradientSiteSpecV1(19, 1024, 132, (6, 1, 16, 8, 8)),
    S4GradientSiteSpecV1(23, 1024, 121, (6, 1, 16, 8, 8)),
    S4GradientSiteSpecV1(25, 1024, 86, (6, 1, 16, 8, 8)),
    S4GradientSiteSpecV1(28, 1024, 178, (6, 1, 16, 8, 8)),
    S4GradientSiteSpecV1(31, 100, 27, (6, 1, 100)),
)
S4_DBETA31_SYMBOL_V1 = "boundflow_s4_emit_dbeta_31"
S4_GRADIENT_EXPORTED_SYMBOLS_V1 = (
    tuple(spec.dalpha_symbol for spec in S4_GRADIENT_SITE_SPECS_V1)
    + (S4_DBETA31_SYMBOL_V1,)
    + tuple(spec.copy_symbol for spec in S4_GRADIENT_SITE_SPECS_V1)
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_finite(tvm: Any, value: Any) -> Any:
    bits = tvm.tir.reinterpret("uint32", value)
    return tvm.tir.bitwise_and(
        bits, tvm.tir.const(0x7F800000, "uint32")
    ) != tvm.tir.const(0x7F800000, "uint32")


def _qnan(tvm: Any) -> Any:
    return tvm.tir.reinterpret(
        "float32", tvm.tir.const(S4_COMPRESSED_GRADIENT_QNAN_BITS_V1, "uint32")
    )


def _dalpha_primfunc(spec: S4GradientSiteSpecV1) -> Any:
    import tvm
    from tvm import te

    coefficient = te.placeholder(
        (6, 1, spec.feature_count), "float32", name="incoming_coefficient"
    )
    adjoint = te.placeholder(
        (6, 1, spec.feature_count), "float32", name="coefficient_adjoint"
    )
    lower = te.placeholder((6, spec.feature_count), "float32", name="lower")
    upper = te.placeholder((6, spec.feature_count), "float32", name="upper")
    active_alpha = te.placeholder((6, spec.alpha_width), "float32", name="active_alpha")
    alpha_indices = te.placeholder((spec.alpha_width,), "int32", name="alpha_indices")
    upstream = te.placeholder((6, 1), "float32", name="upstream")
    zero = tvm.tir.const(0.0, "float32")
    one = tvm.tir.const(1.0, "float32")

    def value(d_idx: Any, alpha_ordinal: Any) -> Any:
        feature = alpha_indices[alpha_ordinal]
        safe_feature = tvm.tir.min(
            tvm.tir.max(feature, tvm.tir.const(0, "int32")),
            tvm.tir.const(spec.feature_count - 1, "int32"),
        )
        a_value = coefficient[d_idx, 0, safe_feature]
        v_value = adjoint[d_idx, 0, safe_feature]
        lower_value = lower[d_idx, safe_feature]
        upper_value = upper[d_idx, safe_feature]
        alpha_value = active_alpha[d_idx, alpha_ordinal]
        upstream_value = upstream[d_idx, 0]
        valid = tvm.tir.all(
            feature >= 0,
            feature < spec.feature_count,
            _is_finite(tvm, a_value),
            _is_finite(tvm, v_value),
            _is_finite(tvm, lower_value),
            _is_finite(tvm, upper_value),
            _is_finite(tvm, alpha_value),
            _is_finite(tvm, upstream_value),
            lower_value <= upper_value,
            alpha_value >= zero,
            alpha_value <= one,
        )
        active = tvm.tir.all(
            lower_value < zero,
            upper_value > zero,
            a_value >= zero,
        )
        finite_result = tvm.tir.if_then_else(
            active, upstream_value * a_value * v_value, zero
        )
        return tvm.tir.if_then_else(valid, finite_result, _qnan(tvm))

    output = te.compute((6, spec.alpha_width), value, name=f"dalpha_{spec.site_id}")
    return (
        te.create_prim_func(
            [
                coefficient,
                adjoint,
                lower,
                upper,
                active_alpha,
                alpha_indices,
                upstream,
                output,
            ]
        )
        .with_attr("global_symbol", spec.dalpha_symbol)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", S4_COMPRESSED_GRADIENT_SCHEMA_V1)
        .with_attr("boundflow.site_id", spec.site_id)
        .with_attr("boundflow.safe_index", "clamp-before-read")
        .with_attr("boundflow.qnan_bits", S4_COMPRESSED_GRADIENT_QNAN_BITS_V1)
    )


def _dbeta_primfunc() -> Any:
    import tvm
    from tvm import te

    adjoint = te.placeholder((6, 1, 100), "float32", name="coefficient_adjoint")
    location = te.placeholder((6, 1), "int32", name="beta_location")
    sign = te.placeholder((6, 1), "int8", name="beta_sign")
    upstream = te.placeholder((6, 1), "float32", name="upstream")

    def value(d_idx: Any, q_idx: Any) -> Any:
        feature = location[d_idx, q_idx]
        safe_feature = tvm.tir.min(
            tvm.tir.max(feature, tvm.tir.const(0, "int32")),
            tvm.tir.const(99, "int32"),
        )
        v_value = adjoint[d_idx, 0, safe_feature]
        upstream_value = upstream[d_idx, 0]
        sign_value = sign[d_idx, q_idx]
        valid = tvm.tir.all(
            feature >= 0,
            feature < 100,
            tvm.tir.any(sign_value == 1, sign_value == -1),
            _is_finite(tvm, v_value),
            _is_finite(tvm, upstream_value),
        )
        finite_result = upstream_value * (
            -v_value * tvm.tir.Cast("float32", sign_value)
        )
        return tvm.tir.if_then_else(valid, finite_result, _qnan(tvm))

    output = te.compute((6, 1), value, name="dbeta_31")
    return (
        te.create_prim_func([adjoint, location, sign, upstream, output])
        .with_attr("global_symbol", S4_DBETA31_SYMBOL_V1)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", S4_COMPRESSED_GRADIENT_SCHEMA_V1)
        .with_attr("boundflow.site_id", 31)
        .with_attr("boundflow.safe_index", "clamp-before-read")
        .with_attr("boundflow.qnan_bits", S4_COMPRESSED_GRADIENT_QNAN_BITS_V1)
    )


def _terminal_copy_primfunc(spec: S4GradientSiteSpecV1) -> Any:
    from tvm import te

    source = te.placeholder(
        (6, 1, spec.feature_count), "float32", name="incoming_coefficient"
    )
    output = te.compute(
        (6, 1, spec.feature_count),
        lambda d_idx, s_idx, feature: source[d_idx, s_idx, feature],
        name=f"terminal_la_{spec.site_id}",
    )
    return (
        te.create_prim_func([source, output])
        .with_attr("global_symbol", spec.copy_symbol)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", S4_COMPRESSED_GRADIENT_SCHEMA_V1)
        .with_attr("boundflow.site_id", spec.site_id)
    )


def build_s4_compressed_gradient_tir_modules_v1() -> tuple[Any, Any]:
    """Return unscheduled and fixed 128-thread scheduled modules."""

    import tvm

    raw: dict[str, Any] = {}
    for spec in S4_GRADIENT_SITE_SPECS_V1:
        raw[spec.dalpha_symbol] = _dalpha_primfunc(spec)
        raw[spec.copy_symbol] = _terminal_copy_primfunc(spec)
    raw[S4_DBETA31_SYMBOL_V1] = _dbeta_primfunc()
    scheduled: dict[str, Any] = {}
    for spec in S4_GRADIENT_SITE_SPECS_V1:
        scheduled[spec.dalpha_symbol] = _schedule_te_primfunc(
            tvm,
            spec.dalpha_symbol,
            raw[spec.dalpha_symbol],
            ((f"dalpha_{spec.site_id}", False, S4_COMPRESSED_GRADIENT_THREADS_V1),),
        )
        scheduled[spec.copy_symbol] = _schedule_te_primfunc(
            tvm,
            spec.copy_symbol,
            raw[spec.copy_symbol],
            (
                (
                    f"terminal_la_{spec.site_id}",
                    False,
                    S4_COMPRESSED_GRADIENT_THREADS_V1,
                ),
            ),
        )
    scheduled[S4_DBETA31_SYMBOL_V1] = _schedule_te_primfunc(
        tvm,
        S4_DBETA31_SYMBOL_V1,
        raw[S4_DBETA31_SYMBOL_V1],
        (("dbeta_31", False, S4_COMPRESSED_GRADIENT_THREADS_V1),),
    )
    return tvm.IRModule(raw), tvm.IRModule(scheduled)


@dataclass(frozen=True)
class CompiledS4CompressedGradientV1:
    executable: DifferentiableLowerTIRExecutable
    unscheduled_tir_json: str
    scheduled_tir_json: str
    device_source: str
    template_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    exported_symbols: tuple[str, ...]
    target: str
    global_workspace_bytes: int = 0
    performance_claimed: bool = False

    def validate(self) -> None:
        if (
            _sha256(self.unscheduled_tir_json) != self.template_hash
            or _sha256(self.scheduled_tir_json) != self.scheduled_tir_hash
            or _sha256(self.device_source) != self.device_source_hash
            or self.exported_symbols != S4_GRADIENT_EXPORTED_SYMBOLS_V1
            or any(symbol not in self.device_source for symbol in self.exported_symbols)
            or not self.target
            or self.global_workspace_bytes
            or self.performance_claimed
        ):
            raise ValueError("S4 gradient compiled identity differs")


def compile_s4_compressed_gradient_v1(
    *, compute_capability: str = "sm_89"
) -> CompiledS4CompressedGradientV1:
    import tvm

    unscheduled, scheduled = build_s4_compressed_gradient_tir_modules_v1()
    target = f"cuda -arch={compute_capability}"
    executable = tvm.compile(scheduled, target=target)
    sources = tuple(imported.inspect_source() for imported in executable.mod.imports)
    if not sources:
        raise RuntimeError("S4 gradient compile produced no CUDA source")
    source = "\n".join(sources)
    compiled = CompiledS4CompressedGradientV1(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        unscheduled_tir_json=tvm.ir.save_json(unscheduled),
        scheduled_tir_json=tvm.ir.save_json(scheduled),
        device_source=source,
        template_hash=_sha256(tvm.ir.save_json(unscheduled)),
        scheduled_tir_hash=_sha256(tvm.ir.save_json(scheduled)),
        device_source_hash=_sha256(source),
        exported_symbols=S4_GRADIENT_EXPORTED_SYMBOLS_V1,
        target=target,
    )
    compiled.validate()
    return compiled


__all__ = [
    "CompiledS4CompressedGradientV1",
    "S4_COMPRESSED_GRADIENT_CONSTRUCTION_HASH_V1",
    "S4_COMPRESSED_GRADIENT_QNAN_BITS_V1",
    "S4_COMPRESSED_GRADIENT_SCHEMA_V1",
    "S4_DBETA31_SYMBOL_V1",
    "S4_GRADIENT_EXPORTED_SYMBOLS_V1",
    "S4_GRADIENT_SITE_SPECS_V1",
    "S4GradientSiteSpecV1",
    "build_s4_compressed_gradient_tir_modules_v1",
    "compile_s4_compressed_gradient_v1",
]
