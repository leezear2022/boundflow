"""Fused ReLU-plus-Linear backward task for non-differentiable plain CROWN."""

# mypy: disable-error-code=import-untyped
# pylint: disable=duplicate-code,import-error

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

FUSED_CROWN_LINEAR_SCHEMA_VERSION = "boundflow.fused_crown_linear/v1"


@dataclass(frozen=True)
class FusedCrownLinearKey:  # pylint: disable=too-many-instance-attributes
    """Static compile key for one fused ReLU-plus-Linear task."""

    domain_batch: int
    spec_batch: int
    current_features: int
    previous_features: int
    dtype: str = "float32"
    target: str = "cuda"
    compute_capability: str = "sm_89"
    schedule_id: str = "serial_reduction_128t_v1"

    def validate(self) -> None:
        """Enforce the deliberately narrow PR-12 v1 capability."""

        for name in (
            "domain_batch",
            "spec_batch",
            "current_features",
            "previous_features",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dtype != "float32":
            raise NotImplementedError("fused CROWN Linear v1 only supports float32")
        if not self.target.startswith("cuda"):
            raise NotImplementedError("fused CROWN Linear v1 only supports CUDA")
        if not self.compute_capability.startswith("sm_"):
            raise ValueError("compute_capability must use the sm_NN form")
        if self.schedule_id != "serial_reduction_128t_v1":
            raise NotImplementedError(f"unsupported schedule_id: {self.schedule_id}")

    @property
    def target_string(self) -> str:
        """Return the CUDA target with the cache-keyed compute capability."""

        return f"{self.target} -arch={self.compute_capability}"


def build_fused_crown_linear_primfunc(  # pylint: disable=too-many-locals
    key: FusedCrownLinearKey,
):
    """Build a PrimFunc whose reductions inline sign selection and slope scaling."""

    key.validate()
    import tvm  # pylint: disable=import-outside-toplevel
    from tvm import te  # pylint: disable=import-outside-toplevel

    domain = key.domain_batch
    spec = key.spec_batch
    current = key.current_features
    previous = key.previous_features
    dtype = key.dtype

    coeff_u = te.placeholder((domain, spec, current), dtype=dtype, name="coeff_u")
    coeff_l = te.placeholder((domain, spec, current), dtype=dtype, name="coeff_l")
    alpha_u = te.placeholder((domain, current), dtype=dtype, name="alpha_u")
    beta_u = te.placeholder((domain, current), dtype=dtype, name="beta_u")
    alpha_l = te.placeholder((domain, current), dtype=dtype, name="alpha_l")
    beta_l = te.placeholder((domain, current), dtype=dtype, name="beta_l")
    weight = te.placeholder((current, previous), dtype=dtype, name="weight")
    bias = te.placeholder((current,), dtype=dtype, name="bias")
    zero = tvm.tir.const(0.0, dtype)

    reduce_u = te.reduce_axis((0, current), "reduce_u")
    reduce_l = te.reduce_axis((0, current), "reduce_l")
    reduce_bu = te.reduce_axis((0, current), "reduce_bu")
    reduce_bl = te.reduce_axis((0, current), "reduce_bl")

    def scaled_upper(d_idx, s_idx, i_idx):
        coeff = coeff_u[d_idx, s_idx, i_idx]
        return tvm.tir.if_then_else(
            coeff >= zero,
            coeff * alpha_u[d_idx, i_idx],
            coeff * alpha_l[d_idx, i_idx],
        )

    def scaled_lower(d_idx, s_idx, i_idx):
        coeff = coeff_l[d_idx, s_idx, i_idx]
        return tvm.tir.if_then_else(
            coeff >= zero,
            coeff * alpha_l[d_idx, i_idx],
            coeff * alpha_u[d_idx, i_idx],
        )

    previous_u = te.compute(
        (domain, spec, previous),
        lambda d_idx, s_idx, j_idx: te.sum(
            scaled_upper(d_idx, s_idx, reduce_u) * weight[reduce_u, j_idx],
            axis=reduce_u,
        ),
        name="previous_u",
    )
    previous_l = te.compute(
        (domain, spec, previous),
        lambda d_idx, s_idx, j_idx: te.sum(
            scaled_lower(d_idx, s_idx, reduce_l) * weight[reduce_l, j_idx],
            axis=reduce_l,
        ),
        name="previous_l",
    )
    bias_delta_u = te.compute(
        (domain, spec),
        lambda d_idx, s_idx: te.sum(
            tvm.tir.if_then_else(
                coeff_u[d_idx, s_idx, reduce_bu] >= zero,
                coeff_u[d_idx, s_idx, reduce_bu]
                * (
                    beta_u[d_idx, reduce_bu]
                    + alpha_u[d_idx, reduce_bu] * bias[reduce_bu]
                ),
                coeff_u[d_idx, s_idx, reduce_bu]
                * (
                    beta_l[d_idx, reduce_bu]
                    + alpha_l[d_idx, reduce_bu] * bias[reduce_bu]
                ),
            ),
            axis=reduce_bu,
        ),
        name="bias_delta_u",
    )
    bias_delta_l = te.compute(
        (domain, spec),
        lambda d_idx, s_idx: te.sum(
            tvm.tir.if_then_else(
                coeff_l[d_idx, s_idx, reduce_bl] >= zero,
                coeff_l[d_idx, s_idx, reduce_bl]
                * (
                    beta_l[d_idx, reduce_bl]
                    + alpha_l[d_idx, reduce_bl] * bias[reduce_bl]
                ),
                coeff_l[d_idx, s_idx, reduce_bl]
                * (
                    beta_u[d_idx, reduce_bl]
                    + alpha_u[d_idx, reduce_bl] * bias[reduce_bl]
                ),
            ),
            axis=reduce_bl,
        ),
        name="bias_delta_l",
    )
    return te.create_prim_func(
        [
            coeff_u,
            coeff_l,
            alpha_u,
            beta_u,
            alpha_l,
            beta_l,
            weight,
            bias,
            previous_u,
            previous_l,
            bias_delta_u,
            bias_delta_l,
        ]
    ).with_attr("boundflow.schema_version", FUSED_CROWN_LINEAR_SCHEMA_VERSION)


def schedule_fused_crown_linear_primfunc(primfunc):
    """Schedule one already-built Linear PrimFunc without rebuilding TIR."""

    import tvm  # pylint: disable=import-outside-toplevel

    module = tvm.IRModule({"main": primfunc})
    schedule = tvm.tir.Schedule(module)
    for block_name in ("previous_u", "previous_l", "bias_delta_u", "bias_delta_l"):
        block = schedule.get_block(block_name, func_name="main")
        loops = schedule.get_loops(block)
        spatial = loops[:-1]
        fused = schedule.fuse(*spatial)
        block_loop, thread_loop = schedule.split(fused, factors=[None, 128])
        schedule.bind(block_loop, "blockIdx.x")
        schedule.bind(thread_loop, "threadIdx.x")
    return schedule.mod


def schedule_fused_crown_linear(key: FusedCrownLinearKey):
    """Bind each output element to one CUDA thread with a serial feature reduction."""

    key.validate()
    return schedule_fused_crown_linear_primfunc(build_fused_crown_linear_primfunc(key))


def allocated_intermediate_buffers(key: FusedCrownLinearKey) -> Tuple[str, ...]:
    """Return PrimFunc-local allocations; v1 must not contain a scaled-A buffer."""

    import tvm  # pylint: disable=import-outside-toplevel

    names: list[str] = []

    def visit(node) -> None:
        if isinstance(node, tvm.tir.Block):
            names.extend(buffer.name for buffer in node.alloc_buffers)

    primfunc = build_fused_crown_linear_primfunc(key)
    tvm.tir.stmt_functor.post_order_visit(primfunc.body, visit)
    return tuple(names)


def build_fused_crown_linear_relax_ir_module(
    key: FusedCrownLinearKey, *, function_name: str = "main"
):
    """Wrap the specialized PrimFunc in one thin Relax ``call_tir`` function."""

    key.validate()
    from tvm import relax  # pylint: disable=import-outside-toplevel

    domain = key.domain_batch
    spec = key.spec_batch
    current = key.current_features
    previous = key.previous_features
    dtype = key.dtype
    builder = relax.BlockBuilder()
    parameters = [
        relax.Var("coeff_u", relax.TensorStructInfo((domain, spec, current), dtype)),
        relax.Var("coeff_l", relax.TensorStructInfo((domain, spec, current), dtype)),
        relax.Var("alpha_u", relax.TensorStructInfo((domain, current), dtype)),
        relax.Var("beta_u", relax.TensorStructInfo((domain, current), dtype)),
        relax.Var("alpha_l", relax.TensorStructInfo((domain, current), dtype)),
        relax.Var("beta_l", relax.TensorStructInfo((domain, current), dtype)),
        relax.Var("weight", relax.TensorStructInfo((current, previous), dtype)),
        relax.Var("bias", relax.TensorStructInfo((current,), dtype)),
    ]
    tir_name = f"{function_name}_fused_crown_linear_tir"
    primfunc = schedule_fused_crown_linear(key)["main"].with_attr(
        "global_symbol", tir_name
    )
    tir_global = builder.add_func(primfunc, tir_name)
    with builder.function(function_name, parameters):
        with builder.dataflow():
            output = builder.emit(
                relax.call_tir(
                    tir_global,
                    relax.Tuple(parameters),
                    out_sinfo=[
                        relax.TensorStructInfo((domain, spec, previous), dtype),
                        relax.TensorStructInfo((domain, spec, previous), dtype),
                        relax.TensorStructInfo((domain, spec), dtype),
                        relax.TensorStructInfo((domain, spec), dtype),
                    ],
                )
            )
            output = builder.emit_output(output)
        builder.emit_func_output(output)
    return builder.get()


@lru_cache(maxsize=128)
def build_fused_crown_linear_module(key: FusedCrownLinearKey):
    """Compile and cache the deterministic CUDA task."""

    key.validate()
    import tvm  # pylint: disable=import-outside-toplevel

    return tvm.compile(schedule_fused_crown_linear(key), target=key.target_string)[
        "main"
    ]


__all__ = [
    "FUSED_CROWN_LINEAR_SCHEMA_VERSION",
    "FusedCrownLinearKey",
    "allocated_intermediate_buffers",
    "build_fused_crown_linear_module",
    "build_fused_crown_linear_primfunc",
    "build_fused_crown_linear_relax_ir_module",
    "schedule_fused_crown_linear",
    "schedule_fused_crown_linear_primfunc",
]
