"""D1-B fixed serial-reduction schedule candidates and v1 baseline."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,protected-access
# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)
from boundflow.backends.tvm.r3_d1_residual11_staged import (
    _schedule as _schedule_residual11,
    _stage1_primfunc as _residual11_stage1,
    _stage2_primfunc as _residual11_stage2,
    R3D1_RESIDUAL11_STAGE1_SYMBOL,
    R3D1_RESIDUAL11_STAGE2_SYMBOL,
)
from boundflow.backends.tvm.r3_d1_residual6_staged import (
    _schedule as _schedule_residual6,
    _stage1_primfunc as _residual6_stage1,
    _stage2_primfunc as _residual6_stage2,
    R3D1_RESIDUAL6_STAGE1_SYMBOL,
    R3D1_RESIDUAL6_STAGE2_SYMBOL,
)
from boundflow.backends.tvm.r3_full_lower_forward import (
    _residual11_primfunc as _residual11_v1,
    _residual6_primfunc as _residual6_v1,
    R31B1_RESIDUAL11_SYMBOL,
    R31B1_RESIDUAL6_SYMBOL,
)

R3D1B_SERIAL_THREADS = (64, 128, 256)


@dataclass(frozen=True)
class CompiledR3D1BModuleV1:
    """Compiled isolated module plus stable schedule receipts."""

    executable: DifferentiableLowerTIRExecutable
    schedule_kind: str
    threads_per_block: int
    scheduled_tir_hash: str
    device_source_hash: str
    exported_symbols: tuple[str, ...]
    global_workspace_bytes: int
    tvm_version: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _compile(
    module: Any,
    *,
    schedule_kind: str,
    threads_per_block: int,
    symbols: tuple[str, ...],
    compute_capability: str,
) -> CompiledR3D1BModuleV1:
    import tvm

    executable = tvm.compile(module, target=f"cuda -arch={compute_capability}")
    sources = tuple(item.inspect_source() for item in executable.mod.imports)
    if not sources:
        raise RuntimeError("R3-D1B compile produced no CUDA source")
    return CompiledR3D1BModuleV1(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        schedule_kind=schedule_kind,
        threads_per_block=threads_per_block,
        scheduled_tir_hash=_sha256(tvm.ir.save_json(module)),
        device_source_hash=_sha256("\n".join(sources)),
        exported_symbols=symbols,
        global_workspace_bytes=0,
        tvm_version=str(tvm.__version__),
    )


def compile_r3d1b_v1_baseline(
    *, compute_capability: str = "sm_89"
) -> CompiledR3D1BModuleV1:
    """Compile the frozen v1 residual6/residual11 raw TIR baseline."""

    import tvm

    symbols = (R31B1_RESIDUAL11_SYMBOL, R31B1_RESIDUAL6_SYMBOL)
    module = tvm.IRModule(
        {
            R31B1_RESIDUAL11_SYMBOL: _residual11_v1(),
            R31B1_RESIDUAL6_SYMBOL: _residual6_v1(),
        }
    )
    return _compile(
        module,
        schedule_kind="v1-raw-reference",
        threads_per_block=128,
        symbols=symbols,
        compute_capability=compute_capability,
    )


def compile_r3d1b_serial_candidate(
    threads_per_block: int, *, compute_capability: str = "sm_89"
) -> CompiledR3D1BModuleV1:
    """Compile the two-kernel materialized candidate with serial reductions."""

    import tvm

    if threads_per_block not in R3D1B_SERIAL_THREADS:
        raise ValueError("R3-D1B threads per block differs")
    residual11_stage1 = _residual11_stage1()
    residual11_stage2 = _residual11_stage2()
    residual6_stage1 = _residual6_stage1()
    residual6_stage2 = _residual6_stage2()
    module = tvm.IRModule(
        {
            R3D1_RESIDUAL11_STAGE1_SYMBOL: _schedule_residual11(
                tvm,
                R3D1_RESIDUAL11_STAGE1_SYMBOL,
                residual11_stage1,
                (("stage1_output", True, threads_per_block),),
            ),
            R3D1_RESIDUAL11_STAGE2_SYMBOL: _schedule_residual11(
                tvm,
                R3D1_RESIDUAL11_STAGE2_SYMBOL,
                residual11_stage2,
                (
                    ("stage2_output", True, threads_per_block),
                    ("stage2_bias", True, 1),
                ),
            ),
            R3D1_RESIDUAL6_STAGE1_SYMBOL: _schedule_residual6(
                tvm,
                R3D1_RESIDUAL6_STAGE1_SYMBOL,
                residual6_stage1,
                (("stage1_output", True, threads_per_block),),
            ),
            R3D1_RESIDUAL6_STAGE2_SYMBOL: _schedule_residual6(
                tvm,
                R3D1_RESIDUAL6_STAGE2_SYMBOL,
                residual6_stage2,
                (
                    ("stage2_output", True, threads_per_block),
                    ("stage2_bias", True, 1),
                ),
            ),
        }
    )
    symbols = (
        R3D1_RESIDUAL11_STAGE1_SYMBOL,
        R3D1_RESIDUAL11_STAGE2_SYMBOL,
        R3D1_RESIDUAL6_STAGE1_SYMBOL,
        R3D1_RESIDUAL6_STAGE2_SYMBOL,
    )
    return _compile(
        module,
        schedule_kind="two-kernel-serial-reduction",
        threads_per_block=threads_per_block,
        symbols=symbols,
        compute_capability=compute_capability,
    )


__all__ = [
    "CompiledR3D1BModuleV1",
    "R3D1B_SERIAL_THREADS",
    "compile_r3d1b_serial_candidate",
    "compile_r3d1b_v1_baseline",
]
