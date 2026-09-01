"""TVM/TIR for activation-BaB ReLU, sparse beta injection and Linear."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-locals
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-boolean-expressions,missing-function-docstring
# pylint: disable=too-many-instance-attributes,duplicate-code

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)
from boundflow.backends.tvm.root_crown_residual import _workspace_inventory

BAB_TERMINAL_FORWARD_SYMBOL = "boundflow_bab_terminal_forward_v1"
BAB_TERMINAL_BACKWARD_SYMBOL = "boundflow_bab_terminal_backward_v1"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class BabTerminalLinearTemplateV1:
    """Static shape and sparse-layout contract for the fused terminal region."""

    spec_count: int
    domain_count: int
    current_features: int
    previous_features: int
    alpha_feature_indices: tuple[int, ...]
    beta_count: int
    compute_capability: str
    thread_extent: int = 128
    target: str = "cuda"
    forward_symbol: str = BAB_TERMINAL_FORWARD_SYMBOL
    backward_symbol: str = BAB_TERMINAL_BACKWARD_SYMBOL

    @property
    def alpha_count(self) -> int:
        return len(self.alpha_feature_indices)

    def validate(self) -> None:
        if (
            min(
                self.spec_count,
                self.domain_count,
                self.current_features,
                self.previous_features,
                self.beta_count,
            )
            < 1
            or self.beta_count != 1
            or not self.alpha_feature_indices
            or tuple(sorted(self.alpha_feature_indices)) != self.alpha_feature_indices
            or len(set(self.alpha_feature_indices)) != len(self.alpha_feature_indices)
            or min(self.alpha_feature_indices) < 0
            or max(self.alpha_feature_indices) >= self.current_features
            or not self.compute_capability.startswith("sm_")
            or self.thread_extent not in {32, 64, 128, 256, 512, 1024}
            or self.target != "cuda"
            or self.forward_symbol != BAB_TERMINAL_FORWARD_SYMBOL
            or self.backward_symbol != BAB_TERMINAL_BACKWARD_SYMBOL
        ):
            raise ValueError("activation-BaB terminal template differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": "boundflow.activation-bab-terminal-tir-template/v1",
            "spec_count": self.spec_count,
            "domain_count": self.domain_count,
            "current_features": self.current_features,
            "previous_features": self.previous_features,
            "alpha_feature_indices": list(self.alpha_feature_indices),
            "beta_count": self.beta_count,
            "compute_capability": self.compute_capability,
            "thread_extent": self.thread_extent,
            "target": self.target,
            "forward_symbol": self.forward_symbol,
            "backward_symbol": self.backward_symbol,
            "beta_injection": "relu_a-scatter(beta*sign)-linear",
            "frozen_bound_gradients": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class CompiledBabTerminalLinearTIRV1:
    """Compiled beta-aware terminal module and compiler identities."""

    executable: DifferentiableLowerTIRExecutable
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    workspace_inventory: tuple[tuple[str, tuple[int, ...]], ...]
    tvm_version: str


def _placeholders(template: BabTerminalLinearTemplateV1):
    from tvm import te

    incoming = te.placeholder(
        (template.spec_count, template.domain_count, template.current_features),
        "float32",
        name="incoming_lower_a",
    )
    lower = te.placeholder(
        (template.domain_count, template.current_features),
        "float32",
        name="preactivation_lower",
    )
    upper = te.placeholder(
        (template.domain_count, template.current_features),
        "float32",
        name="preactivation_upper",
    )
    alpha = te.placeholder(
        (
            2,
            template.spec_count,
            template.domain_count,
            template.alpha_count,
        ),
        "float32",
        name="compressed_alpha",
    )
    alpha_map = te.placeholder(
        (template.current_features,), "int32", name="alpha_feature_to_ordinal"
    )
    beta = te.placeholder(
        (template.domain_count, template.beta_count),
        "float32",
        name="sparse_beta",
    )
    beta_location = te.placeholder(
        (template.domain_count, template.beta_count),
        "int64",
        name="beta_location",
    )
    beta_sign = te.placeholder(
        (template.domain_count, template.beta_count),
        "float32",
        name="beta_sign",
    )
    weight = te.placeholder(
        (template.current_features, template.previous_features),
        "float32",
        name="linear_weight",
    )
    bias = te.placeholder((template.current_features,), "float32", name="linear_bias")
    return (
        te,
        incoming,
        lower,
        upper,
        alpha,
        alpha_map,
        beta,
        beta_location,
        beta_sign,
        weight,
        bias,
    )


def _relu_values(tvm, incoming, lower, upper, alpha, alpha_map, s_idx, d_idx, c_idx):
    zero = tvm.tir.const(0.0, "float32")
    one = tvm.tir.const(1.0, "float32")
    epsilon = tvm.tir.const(1.1920928955078125e-07, "float32")
    denominator = tvm.tir.max(upper[d_idx, c_idx] - lower[d_idx, c_idx], epsilon)
    upper_slope = tvm.tir.if_then_else(
        lower[d_idx, c_idx] >= zero,
        one,
        tvm.tir.if_then_else(
            upper[d_idx, c_idx] <= zero,
            zero,
            upper[d_idx, c_idx] / denominator,
        ),
    )
    lookup = alpha_map[c_idx]
    alpha_value = tvm.tir.if_then_else(
        lookup >= 0,
        alpha[0, s_idx, d_idx, tvm.tir.max(lookup, 0)],
        zero,
    )
    ambiguous = tvm.tir.all(lower[d_idx, c_idx] < zero, upper[d_idx, c_idx] > zero)
    lower_slope = tvm.tir.if_then_else(
        ambiguous,
        tvm.tir.min(tvm.tir.max(alpha_value, zero), one),
        tvm.tir.if_then_else(lower[d_idx, c_idx] >= zero, one, zero),
    )
    slope = tvm.tir.if_then_else(
        incoming[s_idx, d_idx, c_idx] >= zero, lower_slope, upper_slope
    )
    intercept = tvm.tir.if_then_else(
        tvm.tir.all(incoming[s_idx, d_idx, c_idx] < zero, ambiguous),
        -lower[d_idx, c_idx] * upper_slope,
        zero,
    )
    return slope, intercept, alpha_value, ambiguous


def _post_beta_value(
    tvm,
    incoming,
    lower,
    upper,
    alpha,
    alpha_map,
    beta,
    beta_location,
    beta_sign,
    s_idx,
    d_idx,
    c_idx,
):
    slope, _intercept, _alpha_value, _ambiguous = _relu_values(
        tvm, incoming, lower, upper, alpha, alpha_map, s_idx, d_idx, c_idx
    )
    injected = tvm.tir.if_then_else(
        beta_location[d_idx, 0] == c_idx,
        beta[d_idx, 0] * beta_sign[d_idx, 0],
        tvm.tir.const(0.0, "float32"),
    )
    return incoming[s_idx, d_idx, c_idx] * slope - injected


def _forward_primfunc(template: BabTerminalLinearTemplateV1):
    import tvm

    (
        te,
        incoming,
        lower,
        upper,
        alpha,
        alpha_map,
        beta,
        beta_location,
        beta_sign,
        weight,
        bias,
    ) = _placeholders(template)
    reduce_current = te.reduce_axis((0, template.current_features), "reduce_current")
    output_a = te.compute(
        (template.spec_count, template.domain_count, template.previous_features),
        lambda s_idx, d_idx, p_idx: te.sum(
            _post_beta_value(
                tvm,
                incoming,
                lower,
                upper,
                alpha,
                alpha_map,
                beta,
                beta_location,
                beta_sign,
                s_idx,
                d_idx,
                reduce_current,
            )
            * weight[reduce_current, p_idx],
            axis=reduce_current,
        ),
        name="output_lower_a",
    )
    reduce_bias = te.reduce_axis((0, template.current_features), "reduce_bias")

    def bias_term(s_idx, d_idx, c_idx):
        _slope, intercept, _alpha_value, _ambiguous = _relu_values(
            tvm, incoming, lower, upper, alpha, alpha_map, s_idx, d_idx, c_idx
        )
        return (
            incoming[s_idx, d_idx, c_idx] * intercept
            + _post_beta_value(
                tvm,
                incoming,
                lower,
                upper,
                alpha,
                alpha_map,
                beta,
                beta_location,
                beta_sign,
                s_idx,
                d_idx,
                c_idx,
            )
            * bias[c_idx]
        )

    output_bias = te.compute(
        (template.spec_count, template.domain_count),
        lambda s_idx, d_idx: te.sum(
            bias_term(s_idx, d_idx, reduce_bias), axis=reduce_bias
        ),
        name="output_bias",
    )
    return (
        te.create_prim_func(
            [
                incoming,
                lower,
                upper,
                alpha,
                alpha_map,
                beta,
                beta_location,
                beta_sign,
                weight,
                bias,
                output_a,
                output_bias,
            ]
        )
        .with_attr("global_symbol", template.forward_symbol)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "activation-bab-terminal-forward/v1")
    )


def _backward_primfunc(template: BabTerminalLinearTemplateV1):
    import tvm

    (
        te,
        incoming,
        lower,
        upper,
        alpha,
        alpha_map,
        beta,
        beta_location,
        beta_sign,
        weight,
        bias,
    ) = _placeholders(template)
    output_a_gradient = te.placeholder(
        (template.spec_count, template.domain_count, template.previous_features),
        "float32",
        name="output_lower_a_gradient",
    )
    output_bias_gradient = te.placeholder(
        (template.spec_count, template.domain_count),
        "float32",
        name="output_bias_gradient",
    )
    zero = tvm.tir.const(0.0, "float32")

    def adjoint_term(s_idx, d_idx, c_idx, p_idx):
        return output_a_gradient[s_idx, d_idx, p_idx] * weight[
            c_idx, p_idx
        ] + tvm.tir.if_then_else(
            p_idx == 0,
            output_bias_gradient[s_idx, d_idx] * bias[c_idx],
            zero,
        )

    reduce_adjoint = te.reduce_axis(
        (0, template.previous_features), "reduce_adjoint_previous"
    )
    adjoint = te.compute(
        (template.spec_count, template.domain_count, template.current_features),
        lambda s_idx, d_idx, c_idx: te.sum(
            adjoint_term(s_idx, d_idx, c_idx, reduce_adjoint),
            axis=reduce_adjoint,
        ),
        name="terminal_linear_adjoint",
    )

    def incoming_gradient_value(s_idx, d_idx, c_idx):
        slope, intercept, _alpha_value, _ambiguous = _relu_values(
            tvm, incoming, lower, upper, alpha, alpha_map, s_idx, d_idx, c_idx
        )
        return (
            adjoint[s_idx, d_idx, c_idx] * slope
            + output_bias_gradient[s_idx, d_idx] * intercept
        )

    incoming_gradient = te.compute(
        (template.spec_count, template.domain_count, template.current_features),
        incoming_gradient_value,
        name="incoming_lower_a_gradient",
    )
    feature_indices = te.placeholder(
        (template.alpha_count,), "int32", name="alpha_feature_indices"
    )

    def alpha_gradient_value(side_idx, s_idx, d_idx, k_idx):
        feature = feature_indices[k_idx]
        _slope, _intercept, alpha_value, ambiguous = _relu_values(
            tvm, incoming, lower, upper, alpha, alpha_map, s_idx, d_idx, feature
        )
        admitted = tvm.tir.all(
            side_idx == 0,
            incoming[s_idx, d_idx, feature] >= zero,
            ambiguous,
            alpha_value >= zero,
            alpha_value <= tvm.tir.const(1.0, "float32"),
        )
        return tvm.tir.if_then_else(
            admitted,
            incoming[s_idx, d_idx, feature] * adjoint[s_idx, d_idx, feature],
            zero,
        )

    alpha_gradient = te.compute(
        (
            2,
            template.spec_count,
            template.domain_count,
            template.alpha_count,
        ),
        alpha_gradient_value,
        name="compressed_alpha_gradient",
    )
    reduce_beta_spec = te.reduce_axis((0, template.spec_count), "reduce_beta_spec")

    def beta_gradient_value(d_idx, b_idx):
        location = beta_location[d_idx, b_idx]
        return te.sum(
            -beta_sign[d_idx, b_idx] * adjoint[reduce_beta_spec, d_idx, location],
            axis=reduce_beta_spec,
        )

    beta_gradient = te.compute(
        (template.domain_count, template.beta_count),
        beta_gradient_value,
        name="sparse_beta_gradient",
    )
    return (
        te.create_prim_func(
            [
                incoming,
                lower,
                upper,
                alpha,
                alpha_map,
                beta,
                beta_location,
                beta_sign,
                weight,
                bias,
                output_a_gradient,
                output_bias_gradient,
                feature_indices,
                incoming_gradient,
                alpha_gradient,
                beta_gradient,
            ]
        )
        .with_attr("global_symbol", template.backward_symbol)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "activation-bab-terminal-backward/v1")
    )


def _schedule_primfunc(module: Any, symbol: str, thread_extent: int):
    import tvm

    schedule = tvm.tir.Schedule(tvm.IRModule({symbol: module[symbol]}))
    blocks = (
        (("output_lower_a", True), ("output_bias", True))
        if symbol == BAB_TERMINAL_FORWARD_SYMBOL
        else (
            ("terminal_linear_adjoint", True),
            ("incoming_lower_a_gradient", False),
            ("compressed_alpha_gradient", False),
            ("sparse_beta_gradient", True),
        )
    )
    for name, has_reduction in blocks:
        block = schedule.get_block(name, func_name=symbol)
        loops = schedule.get_loops(block)
        spatial = loops[:-1] if has_reduction else loops
        fused = schedule.fuse(*spatial) if len(spatial) > 1 else spatial[0]
        outer, inner = schedule.split(fused, factors=[None, thread_extent])
        schedule.bind(outer, "blockIdx.x")
        schedule.bind(inner, "threadIdx.x")
    return schedule.mod[symbol]


def build_bab_terminal_tir_modules_v1(
    template: BabTerminalLinearTemplateV1,
) -> tuple[Any, Any]:
    """Build deterministic no-dense-workspace forward/backward modules."""

    template.validate()
    import tvm

    unscheduled = tvm.IRModule(
        {
            template.forward_symbol: _forward_primfunc(template),
            template.backward_symbol: _backward_primfunc(template),
        }
    )
    scheduled = tvm.IRModule(
        {
            template.forward_symbol: _schedule_primfunc(
                unscheduled, template.forward_symbol, template.thread_extent
            ),
            template.backward_symbol: _schedule_primfunc(
                unscheduled, template.backward_symbol, template.thread_extent
            ),
        }
    )
    script = scheduled.script(show_meta=False)
    for forbidden in (
        "relu_output_lower_a",
        "post_beta_lower_a",
        "dense_alpha",
        "adjoint_matmul",
    ):
        if forbidden in script:
            raise RuntimeError("activation-BaB terminal materialized forbidden state")
    return unscheduled, scheduled


def compile_bab_terminal_tir_v1(
    template: BabTerminalLinearTemplateV1,
) -> CompiledBabTerminalLinearTIRV1:
    """Compile the beta-aware terminal module for one CUDA signature."""

    import tvm

    unscheduled, scheduled = build_bab_terminal_tir_modules_v1(template)
    executable = tvm.compile(
        scheduled, target=f"{template.target} -arch={template.compute_capability}"
    )
    sources = tuple(module.inspect_source() for module in executable.mod.imports)
    if not sources:
        raise RuntimeError("activation-BaB terminal compile produced no CUDA source")
    return CompiledBabTerminalLinearTIRV1(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        unscheduled_tir_hash=hashlib.sha256(
            tvm.ir.save_json(unscheduled).encode()
        ).hexdigest(),
        scheduled_tir_hash=hashlib.sha256(
            tvm.ir.save_json(scheduled).encode()
        ).hexdigest(),
        device_source_hash=hashlib.sha256("\n".join(sources).encode()).hexdigest(),
        workspace_inventory=_workspace_inventory(scheduled),
        tvm_version=str(tvm.__version__),
    )


__all__ = [
    "BAB_TERMINAL_BACKWARD_SYMBOL",
    "BAB_TERMINAL_FORWARD_SYMBOL",
    "BabTerminalLinearTemplateV1",
    "CompiledBabTerminalLinearTIRV1",
    "build_bab_terminal_tir_modules_v1",
    "compile_bab_terminal_tir_v1",
]
