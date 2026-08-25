"""D1-C cumulative wrapper module for the frozen D1-B winner."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,protected-access
# pylint: disable=too-many-instance-attributes

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)
from boundflow.backends.tvm.r3_d1_residual11_staged import (
    _schedule as _schedule_residual11,
    _stage1_primfunc as _residual11_stage1,
    _stage2_primfunc as _residual11_stage2,
)
from boundflow.backends.tvm.r3_d1_residual6_staged import (
    _schedule as _schedule_residual6,
    _stage1_primfunc as _residual6_stage1,
    _stage2_primfunc as _residual6_stage2,
)

R3D1C_THREADS = 256
R3D1C_RESIDUAL11_STAGE1 = "boundflow_r3d1c_residual11_stage1"
R3D1C_RESIDUAL11_STAGE2 = "boundflow_r3d1c_residual11_stage2"
R3D1C_RESIDUAL6_STAGE1 = "boundflow_r3d1c_residual6_stage1"
R3D1C_RESIDUAL6_STAGE2 = "boundflow_r3d1c_residual6_stage2"
R3D1C_SYMBOLS = (
    R3D1C_RESIDUAL11_STAGE1,
    R3D1C_RESIDUAL11_STAGE2,
    R3D1C_RESIDUAL6_STAGE1,
    R3D1C_RESIDUAL6_STAGE2,
)


@dataclass(frozen=True)
class CompiledR3D1CWrapperScheduleV1:
    """Frozen D1-C module and compiler receipts."""

    executable: DifferentiableLowerTIRExecutable
    scheduled_tir_hash: str
    device_source_hash: str
    exported_symbols: tuple[str, ...]
    threads_per_block: int
    reduction_kind: str
    vector_width: int
    global_workspace_bytes: int
    tvm_version: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _rename(primfunc, symbol: str, *, allow_bias_alias: bool = False):
    value = primfunc.with_attr("global_symbol", symbol).with_attr(
        "boundflow.schema_version", "r3-d1c-wrapper-schedule/v1"
    )
    if allow_bias_alias:
        value = value.without_attr("tir.noalias")
    return value


def build_r3d1c_wrapper_schedule_v1():
    """Build the fixed 256-thread, serial-reduction cumulative module."""

    import tvm

    residual11_stage1 = _rename(_residual11_stage1(), R3D1C_RESIDUAL11_STAGE1)
    residual11_stage2 = _rename(
        _residual11_stage2(), R3D1C_RESIDUAL11_STAGE2, allow_bias_alias=True
    )
    residual6_stage1 = _rename(_residual6_stage1(), R3D1C_RESIDUAL6_STAGE1)
    residual6_stage2 = _rename(
        _residual6_stage2(), R3D1C_RESIDUAL6_STAGE2, allow_bias_alias=True
    )
    return tvm.IRModule(
        {
            R3D1C_RESIDUAL11_STAGE1: _schedule_residual11(
                tvm,
                R3D1C_RESIDUAL11_STAGE1,
                residual11_stage1,
                (("stage1_output", True, R3D1C_THREADS),),
            ),
            R3D1C_RESIDUAL11_STAGE2: _schedule_residual11(
                tvm,
                R3D1C_RESIDUAL11_STAGE2,
                residual11_stage2,
                (
                    ("stage2_output", True, R3D1C_THREADS),
                    ("stage2_bias", True, 1),
                ),
            ),
            R3D1C_RESIDUAL6_STAGE1: _schedule_residual6(
                tvm,
                R3D1C_RESIDUAL6_STAGE1,
                residual6_stage1,
                (("stage1_output", True, R3D1C_THREADS),),
            ),
            R3D1C_RESIDUAL6_STAGE2: _schedule_residual6(
                tvm,
                R3D1C_RESIDUAL6_STAGE2,
                residual6_stage2,
                (
                    ("stage2_output", True, R3D1C_THREADS),
                    ("stage2_bias", True, 1),
                ),
            ),
        }
    )


def compile_r3d1c_wrapper_schedule_v1(
    *, compute_capability: str = "sm_89"
) -> CompiledR3D1CWrapperScheduleV1:
    """Compile the D1-C winner without global workspace or dynamic tuning."""

    import tvm

    scheduled = build_r3d1c_wrapper_schedule_v1()
    executable = tvm.compile(scheduled, target=f"cuda -arch={compute_capability}")
    sources = tuple(module.inspect_source() for module in executable.mod.imports)
    if not sources:
        raise RuntimeError("R3-D1C compile produced no CUDA source")
    return CompiledR3D1CWrapperScheduleV1(
        executable=cast(DifferentiableLowerTIRExecutable, executable),
        scheduled_tir_hash=_sha256(tvm.ir.save_json(scheduled)),
        device_source_hash=_sha256("\n".join(sources)),
        exported_symbols=R3D1C_SYMBOLS,
        threads_per_block=R3D1C_THREADS,
        reduction_kind="serial-reference",
        vector_width=1,
        global_workspace_bytes=0,
        tvm_version=str(tvm.__version__),
    )


__all__ = [
    "CompiledR3D1CWrapperScheduleV1",
    "R3D1C_RESIDUAL11_STAGE1",
    "R3D1C_RESIDUAL11_STAGE2",
    "R3D1C_RESIDUAL6_STAGE1",
    "R3D1C_RESIDUAL6_STAGE2",
    "R3D1C_SYMBOLS",
    "build_r3d1c_wrapper_schedule_v1",
    "compile_r3d1c_wrapper_schedule_v1",
]
