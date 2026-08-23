"""CUDA identity forward/backward TIR used by the B4-B2 ABI probe."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Protocol, cast

from ...ir.differentiable_lower_tir import (
    FROZEN_TVM_COMMIT,
    FROZEN_TVM_FFI_COMMIT,
    IDENTITY_BACKWARD_SYMBOL,
    IDENTITY_FORWARD_SYMBOL,
)

TVM_COMMIT = FROZEN_TVM_COMMIT
TVM_FFI_COMMIT = FROZEN_TVM_FFI_COMMIT


class DifferentiableLowerTIRExecutable(Protocol):
    """Minimum runtime surface required from a compiled TVM executable."""

    def __getitem__(self, symbol: str) -> Callable[..., object]: ...


@dataclass(frozen=True)
class DifferentiableLowerIdentityTIRKey:
    """Narrow compile key for the B2-0 identity ABI probe."""

    template_hash: str
    tensor_numel: int
    thread_extent: int
    dtype: str = "float32"
    target: str = "cuda"
    compute_capability: str = "sm_89"

    def validate(self) -> None:
        if (
            len(self.template_hash) != 64
            or any(
                character not in "0123456789abcdef" for character in self.template_hash
            )
            or self.tensor_numel < 1
            or self.thread_extent not in {64, 128, 256}
            or self.dtype != "float32"
            or self.target != "cuda"
            or not self.compute_capability.startswith("sm_")
        ):
            raise ValueError("differentiable lower identity TIR key differs")

    @property
    def target_string(self) -> str:
        self.validate()
        return f"{self.target} -arch={self.compute_capability}"


@dataclass(frozen=True)
class CompiledDifferentiableLowerIdentityTIR:
    """Compiled runtime handle plus auditable TIR/device-source identities."""

    executable: DifferentiableLowerTIRExecutable
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    tvm_version: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_identity_tir_modules(key: DifferentiableLowerIdentityTIRKey):
    """Return unscheduled and scheduled two-symbol identity IRModules."""

    key.validate()
    import tvm
    from tvm import te

    def build_primfunc(input_name: str, output_name: str, symbol: str):
        source = te.placeholder((key.tensor_numel,), key.dtype, name=input_name)
        result = te.compute(
            (key.tensor_numel,), lambda index: source[index], name=output_name
        )
        return (
            te.create_prim_func([source, result])
            .with_attr("global_symbol", symbol)
            .with_attr("tir.noalias", True)
            .with_attr("boundflow.schema_version", "b4b2-identity-probe/v1")
        )

    forward = build_primfunc("input", "output", IDENTITY_FORWARD_SYMBOL)
    backward = build_primfunc(
        "upstream_gradient", "input_gradient", IDENTITY_BACKWARD_SYMBOL
    )
    unscheduled = tvm.IRModule(
        {IDENTITY_FORWARD_SYMBOL: forward, IDENTITY_BACKWARD_SYMBOL: backward}
    )
    scheduled_functions = {}
    for symbol, block_name in (
        (IDENTITY_FORWARD_SYMBOL, "output"),
        (IDENTITY_BACKWARD_SYMBOL, "input_gradient"),
    ):
        schedule = tvm.tir.Schedule(tvm.IRModule({symbol: unscheduled[symbol]}))
        block = schedule.get_block(block_name, func_name=symbol)
        loop = schedule.get_loops(block)[0]
        block_loop, thread_loop = schedule.split(
            loop, factors=[None, key.thread_extent]
        )
        schedule.bind(block_loop, "blockIdx.x")
        schedule.bind(thread_loop, "threadIdx.x")
        scheduled_functions[symbol] = schedule.mod[symbol]
    return unscheduled, tvm.IRModule(scheduled_functions)


def compile_identity_tir(
    key: DifferentiableLowerIdentityTIRKey,
) -> CompiledDifferentiableLowerIdentityTIR:
    """Compile both identity symbols and hash the exact IR/device source."""

    key.validate()
    import tvm

    unscheduled, scheduled = build_identity_tir_modules(key)
    executable = tvm.compile(scheduled, target=key.target_string)
    device_sources = tuple(module.inspect_source() for module in executable.mod.imports)
    if not device_sources:
        raise RuntimeError("identity TIR compile produced no CUDA device source")
    return CompiledDifferentiableLowerIdentityTIR(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        unscheduled_tir_hash=_sha256(tvm.ir.save_json(unscheduled)),
        scheduled_tir_hash=_sha256(tvm.ir.save_json(scheduled)),
        device_source_hash=_sha256("\n".join(device_sources)),
        tvm_version=str(tvm.__version__),
    )


__all__ = [
    "TVM_COMMIT",
    "TVM_FFI_COMMIT",
    "CompiledDifferentiableLowerIdentityTIR",
    "DifferentiableLowerIdentityTIRKey",
    "build_identity_tir_modules",
    "compile_identity_tir",
]
