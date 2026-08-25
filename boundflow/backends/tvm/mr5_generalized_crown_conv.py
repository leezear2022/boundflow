"""Shape/stride-keyed MR5 lower-CROWN ReLU+ConvTranspose CUDA TIR."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-locals
# pylint: disable=missing-function-docstring,too-many-arguments
# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)

MR5_FORWARD_SYMBOL = "boundflow_mr5_generalized_conv_forward"
MR5_BACKWARD_SYMBOL = "boundflow_mr5_generalized_conv_backward"


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MR5GeneralizedConvSignatureV1:
    """Exact static dimensions and ConvTranspose mapping for one MR5 site."""

    site_id: str
    input_channels: int
    output_channels: int
    input_height: int
    input_width: int
    output_height: int
    output_width: int
    stride: tuple[int, int]
    padding: tuple[int, int]
    output_padding: tuple[int, int]
    domain_count: int = 6
    spec_count: int = 1
    kernel: tuple[int, int] = (3, 3)
    dilation: tuple[int, int] = (1, 1)
    target: str = "cuda"
    compute_capability: str = "sm_89"
    thread_extent: int = 128
    schema_version: str = "boundflow.mr5-generalized-conv-signature/v1"

    @property
    def incoming_shape(self) -> tuple[int, ...]:
        return (
            self.domain_count,
            self.spec_count,
            self.output_channels,
            self.output_height,
            self.output_width,
        )

    @property
    def relaxation_shape(self) -> tuple[int, ...]:
        return (
            self.domain_count,
            self.output_channels,
            self.output_height,
            self.output_width,
        )

    @property
    def result_shape(self) -> tuple[int, ...]:
        return (
            self.domain_count,
            self.spec_count,
            self.input_channels,
            self.input_height,
            self.input_width,
        )

    @property
    def weight_shape(self) -> tuple[int, ...]:
        return (
            self.output_channels,
            self.input_channels,
            self.kernel[0],
            self.kernel[1],
        )

    def validate(self) -> None:
        expected = {
            "C0": (3, 8, 32, 32, 16, 16, (2, 2), (1, 1), (1, 1)),
            "C1": (8, 16, 16, 16, 8, 8, (2, 2), (1, 1), (1, 1)),
            "C2": (16, 16, 8, 8, 8, 8, (1, 1), (1, 1), (0, 0)),
        }
        observed = (
            self.input_channels,
            self.output_channels,
            self.input_height,
            self.input_width,
            self.output_height,
            self.output_width,
            self.stride,
            self.padding,
            self.output_padding,
        )
        if (
            self.schema_version != "boundflow.mr5-generalized-conv-signature/v1"
            or self.site_id not in expected
            or observed != expected[self.site_id]
            or (self.domain_count, self.spec_count) != (6, 1)
            or self.kernel != (3, 3)
            or self.dilation != (1, 1)
            or self.target != "cuda"
            or not self.compute_capability.startswith("sm_")
            or self.thread_extent not in (64, 128, 256)
        ):
            raise ValueError("MR5 generalized Conv signature differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "site_id": self.site_id,
            "domain_count": self.domain_count,
            "spec_count": self.spec_count,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "output_height": self.output_height,
            "output_width": self.output_width,
            "kernel": list(self.kernel),
            "stride": list(self.stride),
            "padding": list(self.padding),
            "dilation": list(self.dilation),
            "output_padding": list(self.output_padding),
            "target": self.target,
            "compute_capability": self.compute_capability,
            "thread_extent": self.thread_extent,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class CompiledMR5GeneralizedConvV1:
    """Compiled executable and reproducible compiler identities."""

    executable: DifferentiableLowerTIRExecutable
    signature_hash: str
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    tvm_version: str
    workspace_inventory: tuple[tuple[str, tuple[int, ...]], ...]


def _forward_primfunc(signature: MR5GeneralizedConvSignatureV1):
    import tvm
    from tvm import te

    incoming = te.placeholder(signature.incoming_shape, "float32", name="incoming")
    lower = te.placeholder(signature.relaxation_shape, "float32", name="lower")
    upper = te.placeholder(signature.relaxation_shape, "float32", name="upper")
    alpha = te.placeholder(signature.relaxation_shape, "float32", name="alpha")
    incoming_bias = te.placeholder(
        (signature.domain_count, signature.spec_count),
        "float32",
        name="incoming_bias",
    )
    weight = te.placeholder(signature.weight_shape, "float32", name="weight")
    operator_bias = te.placeholder(
        (signature.output_channels,), "float32", name="operator_bias"
    )
    zero = tvm.tir.const(0.0, "float32")
    one = tvm.tir.const(1.0, "float32")
    epsilon = tvm.tir.const(1.1920928955078125e-07, "float32")

    def upper_slope(d_idx, c_idx, h_idx, w_idx):
        denominator = tvm.tir.max(
            upper[d_idx, c_idx, h_idx, w_idx] - lower[d_idx, c_idx, h_idx, w_idx],
            epsilon,
        )
        return tvm.tir.if_then_else(
            lower[d_idx, c_idx, h_idx, w_idx] >= zero,
            one,
            tvm.tir.if_then_else(
                upper[d_idx, c_idx, h_idx, w_idx] <= zero,
                zero,
                upper[d_idx, c_idx, h_idx, w_idx] / denominator,
            ),
        )

    def lower_slope(d_idx, c_idx, h_idx, w_idx):
        clamped = tvm.tir.min(tvm.tir.max(alpha[d_idx, c_idx, h_idx, w_idx], zero), one)
        return tvm.tir.if_then_else(
            tvm.tir.all(
                lower[d_idx, c_idx, h_idx, w_idx] < zero,
                upper[d_idx, c_idx, h_idx, w_idx] > zero,
            ),
            clamped,
            tvm.tir.if_then_else(lower[d_idx, c_idx, h_idx, w_idx] >= zero, one, zero),
        )

    def selected_slope(d_idx, s_idx, c_idx, h_idx, w_idx):
        return tvm.tir.if_then_else(
            incoming[d_idx, s_idx, c_idx, h_idx, w_idx] >= zero,
            lower_slope(d_idx, c_idx, h_idx, w_idx),
            upper_slope(d_idx, c_idx, h_idx, w_idx),
        )

    def selected_intercept(d_idx, s_idx, c_idx, h_idx, w_idx):
        ambiguous = -lower[d_idx, c_idx, h_idx, w_idx] * upper_slope(
            d_idx, c_idx, h_idx, w_idx
        )
        return tvm.tir.if_then_else(
            incoming[d_idx, s_idx, c_idx, h_idx, w_idx] >= zero,
            zero,
            tvm.tir.if_then_else(
                tvm.tir.all(
                    lower[d_idx, c_idx, h_idx, w_idx] < zero,
                    upper[d_idx, c_idx, h_idx, w_idx] > zero,
                ),
                ambiguous,
                zero,
            ),
        )

    relu_a = te.compute(
        signature.incoming_shape,
        lambda d_idx, s_idx, c_idx, h_idx, w_idx: incoming[
            d_idx, s_idx, c_idx, h_idx, w_idx
        ]
        * selected_slope(d_idx, s_idx, c_idx, h_idx, w_idx),
        name="relu_a",
    )
    reduce_co = te.reduce_axis((0, signature.output_channels), "reduce_co")
    reduce_kh = te.reduce_axis((0, signature.kernel[0]), "reduce_kh")
    reduce_kw = te.reduce_axis((0, signature.kernel[1]), "reduce_kw")
    stride_h, stride_w = signature.stride
    pad_h, pad_w = signature.padding
    dilation_h, dilation_w = signature.dilation

    def result_element(d_idx, s_idx, ci_idx, ih_idx, iw_idx):
        numerator_h = ih_idx + pad_h - reduce_kh * dilation_h
        numerator_w = iw_idx + pad_w - reduce_kw * dilation_w
        oh_idx = tvm.tir.floordiv(numerator_h, stride_h)
        ow_idx = tvm.tir.floordiv(numerator_w, stride_w)
        valid = tvm.tir.all(
            numerator_h >= 0,
            numerator_w >= 0,
            tvm.tir.floormod(numerator_h, stride_h) == 0,
            tvm.tir.floormod(numerator_w, stride_w) == 0,
            oh_idx < signature.output_height,
            ow_idx < signature.output_width,
        )
        return te.sum(
            tvm.tir.if_then_else(
                valid,
                relu_a[d_idx, s_idx, reduce_co, oh_idx, ow_idx]
                * weight[reduce_co, ci_idx, reduce_kh, reduce_kw],
                zero,
            ),
            axis=(reduce_co, reduce_kh, reduce_kw),
        )

    result_a = te.compute(signature.result_shape, result_element, name="result_a")
    reduce_bc = te.reduce_axis((0, signature.output_channels), "reduce_bias_channel")
    reduce_bh = te.reduce_axis((0, signature.output_height), "reduce_bias_height")
    reduce_bw = te.reduce_axis((0, signature.output_width), "reduce_bias_width")
    bias_delta = te.compute(
        (signature.domain_count, signature.spec_count),
        lambda d_idx, s_idx: te.sum(
            incoming[d_idx, s_idx, reduce_bc, reduce_bh, reduce_bw]
            * selected_intercept(d_idx, s_idx, reduce_bc, reduce_bh, reduce_bw)
            + relu_a[d_idx, s_idx, reduce_bc, reduce_bh, reduce_bw]
            * operator_bias[reduce_bc],
            axis=(reduce_bc, reduce_bh, reduce_bw),
        ),
        name="bias_delta",
    )
    result_bias = te.compute(
        (signature.domain_count, signature.spec_count),
        lambda d_idx, s_idx: incoming_bias[d_idx, s_idx] + bias_delta[d_idx, s_idx],
        name="result_bias",
    )
    return (
        te.create_prim_func(
            [
                incoming,
                lower,
                upper,
                alpha,
                incoming_bias,
                weight,
                operator_bias,
                result_a,
                result_bias,
            ]
        )
        .with_attr("global_symbol", MR5_FORWARD_SYMBOL)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "mr5-generalized-conv-forward/v1")
    )


def _backward_primfunc(signature: MR5GeneralizedConvSignatureV1):
    import tvm
    from tvm import te

    incoming = te.placeholder(signature.incoming_shape, "float32", name="incoming")
    lower = te.placeholder(signature.relaxation_shape, "float32", name="lower")
    upper = te.placeholder(signature.relaxation_shape, "float32", name="upper")
    alpha = te.placeholder(signature.relaxation_shape, "float32", name="alpha")
    weight = te.placeholder(signature.weight_shape, "float32", name="weight")
    operator_bias = te.placeholder(
        (signature.output_channels,), "float32", name="operator_bias"
    )
    result_a_gradient = te.placeholder(
        signature.result_shape, "float32", name="result_a_gradient"
    )
    result_bias_gradient = te.placeholder(
        (signature.domain_count, signature.spec_count),
        "float32",
        name="result_bias_gradient",
    )
    zero = tvm.tir.const(0.0, "float32")
    one = tvm.tir.const(1.0, "float32")
    epsilon = tvm.tir.const(1.1920928955078125e-07, "float32")
    reduce_ci = te.reduce_axis((0, signature.input_channels), "reduce_ci")
    reduce_kh = te.reduce_axis((0, signature.kernel[0]), "reduce_kh")
    reduce_kw = te.reduce_axis((0, signature.kernel[1]), "reduce_kw")
    stride_h, stride_w = signature.stride
    pad_h, pad_w = signature.padding
    dilation_h, dilation_w = signature.dilation

    def adjoint_element(d_idx, s_idx, co_idx, oh_idx, ow_idx):
        ih_idx = oh_idx * stride_h - pad_h + reduce_kh * dilation_h
        iw_idx = ow_idx * stride_w - pad_w + reduce_kw * dilation_w
        return te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(
                    ih_idx >= 0,
                    ih_idx < signature.input_height,
                    iw_idx >= 0,
                    iw_idx < signature.input_width,
                ),
                result_a_gradient[d_idx, s_idx, reduce_ci, ih_idx, iw_idx]
                * weight[co_idx, reduce_ci, reduce_kh, reduce_kw],
                zero,
            ),
            axis=(reduce_ci, reduce_kh, reduce_kw),
        )

    adjoint_conv = te.compute(
        signature.incoming_shape, adjoint_element, name="adjoint_conv"
    )
    adjoint_relu = te.compute(
        signature.incoming_shape,
        lambda d_idx, s_idx, c_idx, h_idx, w_idx: adjoint_conv[
            d_idx, s_idx, c_idx, h_idx, w_idx
        ]
        + result_bias_gradient[d_idx, s_idx] * operator_bias[c_idx],
        name="adjoint_relu",
    )

    def upper_slope(d_idx, c_idx, h_idx, w_idx):
        denominator = tvm.tir.max(
            upper[d_idx, c_idx, h_idx, w_idx] - lower[d_idx, c_idx, h_idx, w_idx],
            epsilon,
        )
        return tvm.tir.if_then_else(
            lower[d_idx, c_idx, h_idx, w_idx] >= zero,
            one,
            tvm.tir.if_then_else(
                upper[d_idx, c_idx, h_idx, w_idx] <= zero,
                zero,
                upper[d_idx, c_idx, h_idx, w_idx] / denominator,
            ),
        )

    def lower_slope(d_idx, c_idx, h_idx, w_idx):
        clamped = tvm.tir.min(tvm.tir.max(alpha[d_idx, c_idx, h_idx, w_idx], zero), one)
        return tvm.tir.if_then_else(
            tvm.tir.all(
                lower[d_idx, c_idx, h_idx, w_idx] < zero,
                upper[d_idx, c_idx, h_idx, w_idx] > zero,
            ),
            clamped,
            tvm.tir.if_then_else(lower[d_idx, c_idx, h_idx, w_idx] >= zero, one, zero),
        )

    def selected_slope(d_idx, s_idx, c_idx, h_idx, w_idx):
        return tvm.tir.if_then_else(
            incoming[d_idx, s_idx, c_idx, h_idx, w_idx] >= zero,
            lower_slope(d_idx, c_idx, h_idx, w_idx),
            upper_slope(d_idx, c_idx, h_idx, w_idx),
        )

    def selected_intercept(d_idx, s_idx, c_idx, h_idx, w_idx):
        ambiguous = -lower[d_idx, c_idx, h_idx, w_idx] * upper_slope(
            d_idx, c_idx, h_idx, w_idx
        )
        return tvm.tir.if_then_else(
            incoming[d_idx, s_idx, c_idx, h_idx, w_idx] >= zero,
            zero,
            tvm.tir.if_then_else(
                tvm.tir.all(
                    lower[d_idx, c_idx, h_idx, w_idx] < zero,
                    upper[d_idx, c_idx, h_idx, w_idx] > zero,
                ),
                ambiguous,
                zero,
            ),
        )

    reduce_spec = te.reduce_axis((0, signature.spec_count), "reduce_spec")
    alpha_gradient = te.compute(
        signature.relaxation_shape,
        lambda d_idx, c_idx, h_idx, w_idx: te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(
                    incoming[d_idx, reduce_spec, c_idx, h_idx, w_idx] >= zero,
                    lower[d_idx, c_idx, h_idx, w_idx] < zero,
                    upper[d_idx, c_idx, h_idx, w_idx] > zero,
                    alpha[d_idx, c_idx, h_idx, w_idx] >= zero,
                    alpha[d_idx, c_idx, h_idx, w_idx] <= one,
                ),
                adjoint_relu[d_idx, reduce_spec, c_idx, h_idx, w_idx]
                * incoming[d_idx, reduce_spec, c_idx, h_idx, w_idx],
                zero,
            ),
            axis=reduce_spec,
        ),
        name="alpha_gradient",
    )
    incoming_gradient = te.compute(
        signature.incoming_shape,
        lambda d_idx, s_idx, c_idx, h_idx, w_idx: adjoint_relu[
            d_idx, s_idx, c_idx, h_idx, w_idx
        ]
        * selected_slope(d_idx, s_idx, c_idx, h_idx, w_idx)
        + result_bias_gradient[d_idx, s_idx]
        * selected_intercept(d_idx, s_idx, c_idx, h_idx, w_idx),
        name="incoming_gradient",
    )
    return (
        te.create_prim_func(
            [
                incoming,
                lower,
                upper,
                alpha,
                weight,
                operator_bias,
                result_a_gradient,
                result_bias_gradient,
                alpha_gradient,
                incoming_gradient,
            ]
        )
        .with_attr("global_symbol", MR5_BACKWARD_SYMBOL)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "mr5-generalized-conv-backward/v1")
    )


def _workspace_inventory(module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    import tvm

    inventory: set[tuple[str, tuple[int, ...]]] = set()

    def visit(node) -> None:
        if isinstance(node, tvm.tir.Block):
            for buffer in node.alloc_buffers:
                inventory.add(
                    (str(buffer.name), tuple(int(value) for value in buffer.shape))
                )

    for function in module.functions.values():
        tvm.tir.stmt_functor.post_order_visit(function.body, visit)
    return tuple(sorted(inventory))


def build_mr5_generalized_conv_modules(signature: MR5GeneralizedConvSignatureV1):
    """Build deterministic unscheduled and CUDA-scheduled modules."""

    signature.validate()
    import tvm

    unscheduled = tvm.IRModule(
        {
            MR5_FORWARD_SYMBOL: _forward_primfunc(signature),
            MR5_BACKWARD_SYMBOL: _backward_primfunc(signature),
        }
    )
    scheduled_functions = {}
    inventories = {
        MR5_FORWARD_SYMBOL: (
            ("relu_a", True),
            ("result_a", False),
            ("bias_delta", False),
            ("result_bias", False),
        ),
        MR5_BACKWARD_SYMBOL: (
            ("adjoint_conv", False),
            ("adjoint_relu", True),
            ("alpha_gradient", False),
            ("incoming_gradient", False),
        ),
    }
    for symbol, blocks in inventories.items():
        schedule = tvm.tir.Schedule(tvm.IRModule({symbol: unscheduled[symbol]}))
        for block_name, inline in blocks:
            block = schedule.get_block(block_name, func_name=symbol)
            if inline:
                schedule.compute_inline(block)
                continue
            loops = schedule.get_loops(block)
            spatial_count = {
                "result_a": 5,
                "bias_delta": 2,
                "result_bias": 2,
                "adjoint_conv": 5,
                "alpha_gradient": 4,
                "incoming_gradient": 5,
            }[block_name]
            fused = schedule.fuse(*loops[:spatial_count])
            block_loop, thread_loop = schedule.split(
                fused, factors=[None, signature.thread_extent]
            )
            schedule.bind(block_loop, "blockIdx.x")
            schedule.bind(thread_loop, "threadIdx.x")
        scheduled_functions[symbol] = schedule.mod[symbol]
    scheduled = tvm.IRModule(scheduled_functions)
    return unscheduled, scheduled, _workspace_inventory(scheduled)


def compile_mr5_generalized_conv(
    signature: MR5GeneralizedConvSignatureV1,
) -> CompiledMR5GeneralizedConvV1:
    """Compile one site-specific generalized Conv module."""

    signature.validate()
    import tvm

    unscheduled, scheduled, inventory = build_mr5_generalized_conv_modules(signature)
    executable = tvm.compile(
        scheduled, target=f"{signature.target} -arch={signature.compute_capability}"
    )
    sources = tuple(module.inspect_source() for module in executable.mod.imports)
    if not sources:
        raise RuntimeError("MR5 generalized Conv compile produced no CUDA source")
    return CompiledMR5GeneralizedConvV1(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        signature_hash=signature.stable_hash(),
        unscheduled_tir_hash=_canonical_hash(tvm.ir.save_json(unscheduled)),
        scheduled_tir_hash=_canonical_hash(tvm.ir.save_json(scheduled)),
        device_source_hash=_canonical_hash("\n".join(sources)),
        tvm_version=str(tvm.__version__),
        workspace_inventory=inventory,
    )


__all__ = [
    "CompiledMR5GeneralizedConvV1",
    "MR5GeneralizedConvSignatureV1",
    "MR5_BACKWARD_SYMBOL",
    "MR5_FORWARD_SYMBOL",
    "build_mr5_generalized_conv_modules",
    "compile_mr5_generalized_conv",
]
