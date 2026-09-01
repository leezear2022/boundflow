"""TVM/TIR for a root CROWN terminal ReLU followed by Linear."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-locals
# pylint: disable=too-many-instance-attributes,missing-function-docstring
# pylint: disable=too-many-boolean-expressions,too-many-arguments
# pylint: disable=too-many-positional-arguments

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)

ROOT_TERMINAL_FORWARD_SYMBOL = "boundflow_root_crown_terminal_forward_v1"
ROOT_TERMINAL_BACKWARD_SYMBOL = "boundflow_root_crown_terminal_backward_v1"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RootCrownTerminalLinearTemplateV1:
    """Shape-polymorphic-at-construction ABI for the fused root transaction."""

    spec_count: int
    domain_count: int
    current_features: int
    previous_features: int
    alpha_feature_indices: tuple[int, ...]
    compute_capability: str
    thread_extent: int = 128
    target: str = "cuda"
    forward_symbol: str = ROOT_TERMINAL_FORWARD_SYMBOL
    backward_symbol: str = ROOT_TERMINAL_BACKWARD_SYMBOL

    @property
    def alpha_feature_count(self) -> int:
        return len(self.alpha_feature_indices)

    def validate(self) -> None:
        if (
            self.spec_count < 1
            or self.domain_count < 1
            or self.current_features < 1
            or self.previous_features < 1
            or not self.alpha_feature_indices
            or tuple(sorted(self.alpha_feature_indices)) != self.alpha_feature_indices
            or len(set(self.alpha_feature_indices)) != len(self.alpha_feature_indices)
            or min(self.alpha_feature_indices) < 0
            or max(self.alpha_feature_indices) >= self.current_features
            or not self.compute_capability.startswith("sm_")
            or self.thread_extent not in {32, 64, 128, 256, 512, 1024}
            or self.target != "cuda"
            or self.forward_symbol != ROOT_TERMINAL_FORWARD_SYMBOL
            or self.backward_symbol != ROOT_TERMINAL_BACKWARD_SYMBOL
        ):
            raise ValueError("root CROWN terminal TIR template differs")

    def stable_hash(self) -> str:
        self.validate()
        return _canonical_hash(
            {
                "schema_version": "boundflow.root-crown-terminal-tir-template/v1",
                "spec_count": self.spec_count,
                "domain_count": self.domain_count,
                "current_features": self.current_features,
                "previous_features": self.previous_features,
                "alpha_feature_indices": list(self.alpha_feature_indices),
                "compute_capability": self.compute_capability,
                "thread_extent": self.thread_extent,
                "target": self.target,
                "forward_symbol": self.forward_symbol,
                "backward_symbol": self.backward_symbol,
            }
        )


@dataclass(frozen=True)
class CompiledRootCrownTerminalLinearTIRV1:
    """Compiled fused forward/backward module and source identities."""

    executable: DifferentiableLowerTIRExecutable
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    tvm_version: str


def _alpha_value(raw_alpha, feature_to_ordinal, spec, domain, current, zero):
    import tvm

    ordinal = feature_to_ordinal[current]
    return tvm.tir.if_then_else(
        ordinal >= 0,
        raw_alpha[0, spec, domain, tvm.tir.max(ordinal, 0)],
        zero,
    )


def _forward_primfunc(template: RootCrownTerminalLinearTemplateV1):
    import tvm
    from tvm import te

    spec = template.spec_count
    domain = template.domain_count
    current = template.current_features
    previous = template.previous_features
    alpha_features = template.alpha_feature_count
    dtype = "float32"
    incoming = te.placeholder((spec, domain, current), dtype, name="incoming_lower_a")
    lower = te.placeholder((domain, current), dtype, name="preactivation_lower")
    upper = te.placeholder((domain, current), dtype, name="preactivation_upper")
    raw_alpha = te.placeholder(
        (2, spec, domain, alpha_features), dtype, name="raw_alpha"
    )
    feature_to_ordinal = te.placeholder(
        (current,), "int32", name="alpha_feature_to_ordinal"
    )
    weight = te.placeholder((current, previous), dtype, name="operator_weight")
    bias = te.placeholder((current,), dtype, name="operator_bias")
    zero = tvm.tir.const(0.0, dtype)
    one = tvm.tir.const(1.0, dtype)
    epsilon = tvm.tir.const(1.1920928955078125e-07, dtype)

    def upper_slope(d_idx, c_idx):
        denominator = tvm.tir.max(upper[d_idx, c_idx] - lower[d_idx, c_idx], epsilon)
        return tvm.tir.if_then_else(
            lower[d_idx, c_idx] >= zero,
            one,
            tvm.tir.if_then_else(
                upper[d_idx, c_idx] <= zero,
                zero,
                upper[d_idx, c_idx] / denominator,
            ),
        )

    def lower_slope(s_idx, d_idx, c_idx):
        alpha = _alpha_value(raw_alpha, feature_to_ordinal, s_idx, d_idx, c_idx, zero)
        clamped = tvm.tir.min(tvm.tir.max(alpha, zero), one)
        return tvm.tir.if_then_else(
            tvm.tir.all(lower[d_idx, c_idx] < zero, upper[d_idx, c_idx] > zero),
            clamped,
            tvm.tir.if_then_else(lower[d_idx, c_idx] >= zero, one, zero),
        )

    def selected_slope(s_idx, d_idx, c_idx):
        return tvm.tir.if_then_else(
            incoming[s_idx, d_idx, c_idx] >= zero,
            lower_slope(s_idx, d_idx, c_idx),
            upper_slope(d_idx, c_idx),
        )

    def relu_a(s_idx, d_idx, c_idx):
        return incoming[s_idx, d_idx, c_idx] * selected_slope(s_idx, d_idx, c_idx)

    reduce_a = te.reduce_axis((0, current), "reduce_output_a")
    output_a = te.compute(
        (spec, domain, previous),
        lambda s_idx, d_idx, p_idx: te.sum(
            relu_a(s_idx, d_idx, reduce_a) * weight[reduce_a, p_idx],
            axis=reduce_a,
        ),
        name="output_lower_a",
    )
    reduce_bias = te.reduce_axis((0, current), "reduce_output_bias")

    def bias_term(s_idx, d_idx, c_idx):
        ambiguous_intercept = -lower[d_idx, c_idx] * upper_slope(d_idx, c_idx)
        intercept = tvm.tir.if_then_else(
            incoming[s_idx, d_idx, c_idx] < zero,
            tvm.tir.if_then_else(
                tvm.tir.all(lower[d_idx, c_idx] < zero, upper[d_idx, c_idx] > zero),
                ambiguous_intercept,
                zero,
            ),
            zero,
        )
        return (
            incoming[s_idx, d_idx, c_idx] * intercept
            + relu_a(s_idx, d_idx, c_idx) * bias[c_idx]
        )

    output_bias = te.compute(
        (spec, domain),
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
                raw_alpha,
                feature_to_ordinal,
                weight,
                bias,
                output_a,
                output_bias,
            ]
        )
        .with_attr("global_symbol", template.forward_symbol)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "root-crown-terminal-forward/v1")
    )


def _backward_primfunc(template: RootCrownTerminalLinearTemplateV1):
    import tvm
    from tvm import te

    spec = template.spec_count
    domain = template.domain_count
    current = template.current_features
    previous = template.previous_features
    alpha_features = template.alpha_feature_count
    dtype = "float32"
    incoming = te.placeholder((spec, domain, current), dtype, name="incoming_lower_a")
    lower = te.placeholder((domain, current), dtype, name="preactivation_lower")
    upper = te.placeholder((domain, current), dtype, name="preactivation_upper")
    raw_alpha = te.placeholder(
        (2, spec, domain, alpha_features), dtype, name="raw_alpha"
    )
    feature_indices = te.placeholder(
        (alpha_features,), "int32", name="alpha_feature_indices"
    )
    weight = te.placeholder((current, previous), dtype, name="operator_weight")
    bias = te.placeholder((current,), dtype, name="operator_bias")
    output_a_gradient = te.placeholder(
        (spec, domain, previous), dtype, name="output_lower_a_gradient"
    )
    output_bias_gradient = te.placeholder(
        (spec, domain), dtype, name="output_bias_gradient"
    )
    zero = tvm.tir.const(0.0, dtype)
    one = tvm.tir.const(1.0, dtype)
    reduce_previous = te.reduce_axis((0, previous), "reduce_previous")

    adjoint_relu = te.compute(
        (spec, domain, current),
        lambda s_idx, d_idx, c_idx: te.sum(
            output_a_gradient[s_idx, d_idx, reduce_previous]
            * weight[c_idx, reduce_previous]
            + tvm.tir.if_then_else(
                reduce_previous == 0,
                output_bias_gradient[s_idx, d_idx] * bias[c_idx],
                zero,
            ),
            axis=reduce_previous,
        ),
        name="transient_adjoint_relu",
    )

    def alpha_gradient(s_idx, d_idx, k_idx):
        feature = feature_indices[k_idx]
        admitted = tvm.tir.all(
            incoming[s_idx, d_idx, feature] >= zero,
            lower[d_idx, feature] < zero,
            upper[d_idx, feature] > zero,
            raw_alpha[0, s_idx, d_idx, k_idx] >= zero,
            raw_alpha[0, s_idx, d_idx, k_idx] <= one,
        )
        return tvm.tir.if_then_else(
            admitted,
            incoming[s_idx, d_idx, feature] * adjoint_relu[s_idx, d_idx, feature],
            zero,
        )

    raw_alpha_gradient = te.compute(
        (spec, domain, alpha_features),
        alpha_gradient,
        name="lower_alpha_gradient",
    )
    reduce_spec = te.reduce_axis((0, spec), "reduce_spec")

    def bound_gradient(side_idx, d_idx, c_idx):
        denominator = tvm.tir.max(
            upper[d_idx, c_idx] - lower[d_idx, c_idx],
            tvm.tir.const(1.1920928955078125e-07, dtype),
        )
        slope = upper[d_idx, c_idx] / denominator
        incoming_value = incoming[reduce_spec, d_idx, c_idx]
        bias_gradient = output_bias_gradient[reduce_spec, d_idx]
        q_value = (
            adjoint_relu[reduce_spec, d_idx, c_idx]
            - bias_gradient * lower[d_idx, c_idx]
        )
        lower_value = incoming_value * (
            q_value * upper[d_idx, c_idx] / (denominator * denominator)
            - bias_gradient * slope
        )
        upper_value = (
            incoming_value
            * q_value
            * (-lower[d_idx, c_idx])
            / (denominator * denominator)
        )
        admitted = tvm.tir.all(
            incoming_value < zero,
            lower[d_idx, c_idx] < zero,
            upper[d_idx, c_idx] > zero,
        )
        return te.sum(
            tvm.tir.if_then_else(
                admitted,
                tvm.tir.if_then_else(side_idx == 0, lower_value, upper_value),
                zero,
            ),
            axis=reduce_spec,
        )

    bound_gradient_output = te.compute(
        (2, domain, current),
        bound_gradient,
        name="preactivation_bound_gradient",
    )
    return (
        te.create_prim_func(
            [
                incoming,
                lower,
                upper,
                raw_alpha,
                feature_indices,
                weight,
                bias,
                output_a_gradient,
                output_bias_gradient,
                raw_alpha_gradient,
                bound_gradient_output,
            ]
        )
        .with_attr("global_symbol", template.backward_symbol)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "root-crown-terminal-backward/v1")
    )


def build_root_crown_terminal_tir_modules_v1(
    template: RootCrownTerminalLinearTemplateV1,
) -> tuple[Any, Any]:
    """Create and schedule the fused module without dense saved A state."""

    template.validate()
    import tvm

    unscheduled = tvm.IRModule(
        {
            template.forward_symbol: _forward_primfunc(template),
            template.backward_symbol: _backward_primfunc(template),
        }
    )
    scheduled_functions = {}
    inventories = {
        template.forward_symbol: (
            ("output_lower_a", True),
            ("output_bias", True),
        ),
        template.backward_symbol: (
            ("transient_adjoint_relu", True),
            ("lower_alpha_gradient", False),
            ("preactivation_bound_gradient", True),
        ),
    }
    for symbol, blocks in inventories.items():
        schedule = tvm.tir.Schedule(tvm.IRModule({symbol: unscheduled[symbol]}))
        if symbol == template.backward_symbol:
            adjoint = schedule.get_block("transient_adjoint_relu", func_name=symbol)
            *_spatial_loops, reduction_loop = schedule.get_loops(adjoint)
            _reduction_outer, reduction_inner = schedule.split(
                reduction_loop, factors=[None, 32]
            )
            partial = schedule.rfactor(reduction_inner, factor_axis=3)
            partial_loops = schedule.get_loops(partial)
            partial_spec = partial_loops[0]
            partial_domain = partial_loops[1]
            partial_current = partial_loops[2]
            partial_reduction = partial_loops[3]
            partial_lane = partial_loops[4]
            schedule.reorder(
                partial_spec,
                partial_domain,
                partial_current,
                partial_lane,
                partial_reduction,
            )
            partial_spatial = schedule.fuse(
                partial_spec, partial_domain, partial_current
            )
            schedule.bind(partial_spatial, "blockIdx.x")
            schedule.bind(partial_lane, "threadIdx.x")
            final_loops = schedule.get_loops(adjoint)
            final_spatial = schedule.fuse(*final_loops[:-1])
            schedule.bind(final_spatial, "blockIdx.x")
            for block_name, has_reduction in (
                ("lower_alpha_gradient", False),
                ("preactivation_bound_gradient", True),
            ):
                block = schedule.get_block(block_name, func_name=symbol)
                loops = schedule.get_loops(block)
                spatial = loops[:-1] if has_reduction else loops
                fused = schedule.fuse(*spatial)
                block_loop, thread_loop = schedule.split(
                    fused, factors=[None, template.thread_extent]
                )
                schedule.bind(block_loop, "blockIdx.x")
                schedule.bind(thread_loop, "threadIdx.x")
            scheduled_functions[symbol] = schedule.mod[symbol]
            continue
        for block_name, has_reduction in blocks:
            block = schedule.get_block(block_name, func_name=symbol)
            loops = schedule.get_loops(block)
            spatial = loops[:-1] if has_reduction else loops
            fused = schedule.fuse(*spatial) if len(spatial) > 1 else spatial[0]
            block_loop, thread_loop = schedule.split(
                fused, factors=[None, template.thread_extent]
            )
            schedule.bind(block_loop, "blockIdx.x")
            schedule.bind(thread_loop, "threadIdx.x")
        scheduled_functions[symbol] = schedule.mod[symbol]
    scheduled = tvm.IRModule(scheduled_functions)
    script = scheduled.script(show_meta=False)
    if "relu_output_lower_a" in script or "adjoint_matmul" in script:
        raise RuntimeError("root CROWN TIR materialized forbidden dense state")
    return unscheduled, scheduled


def compile_root_crown_terminal_tir_v1(
    template: RootCrownTerminalLinearTemplateV1,
) -> CompiledRootCrownTerminalLinearTIRV1:
    """Compile a root terminal module for one fixed shape/signature."""

    template.validate()
    import tvm

    unscheduled, scheduled = build_root_crown_terminal_tir_modules_v1(template)
    executable = tvm.compile(
        scheduled, target=f"{template.target} -arch={template.compute_capability}"
    )
    sources = tuple(module.inspect_source() for module in executable.mod.imports)
    if not sources:
        raise RuntimeError("root CROWN terminal compile produced no CUDA source")
    return CompiledRootCrownTerminalLinearTIRV1(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        unscheduled_tir_hash=hashlib.sha256(
            tvm.ir.save_json(unscheduled).encode()
        ).hexdigest(),
        scheduled_tir_hash=hashlib.sha256(
            tvm.ir.save_json(scheduled).encode()
        ).hexdigest(),
        device_source_hash=hashlib.sha256("\n".join(sources).encode()).hexdigest(),
        tvm_version=str(tvm.__version__),
    )


__all__ = [
    "CompiledRootCrownTerminalLinearTIRV1",
    "ROOT_TERMINAL_BACKWARD_SYMBOL",
    "ROOT_TERMINAL_FORWARD_SYMBOL",
    "RootCrownTerminalLinearTemplateV1",
    "build_root_crown_terminal_tir_modules_v1",
    "compile_root_crown_terminal_tir_v1",
]
