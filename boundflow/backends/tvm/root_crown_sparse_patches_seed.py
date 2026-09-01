"""TVM/TIR lowering from sparse one-pixel Patches to a dense CROWN seed."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,missing-function-docstring
# pylint: disable=too-many-instance-attributes,too-many-boolean-expressions

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import cast

from boundflow.backends.tvm.differentiable_lower_identity import (
    DifferentiableLowerTIRExecutable,
)

ROOT_SPARSE_PATCHES_SEED_SYMBOL = "boundflow_root_sparse_patches_seed_v1"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RootCrownSparsePatchesSeedTemplateV1:
    """Static one-pixel sparse-Patches geometry for an intermediate start."""

    spec_count: int
    domain_count: int
    channels: int
    height: int
    width: int
    compute_capability: str
    thread_extent: int = 128
    target: str = "cuda"
    symbol: str = ROOT_SPARSE_PATCHES_SEED_SYMBOL

    @property
    def patches_shape(self) -> tuple[int, ...]:
        return (self.spec_count, self.domain_count, self.channels, 1, 1)

    @property
    def coefficient_shape(self) -> tuple[int, ...]:
        return (
            self.spec_count,
            self.domain_count,
            self.channels,
            self.height,
            self.width,
        )

    def validate(self) -> None:
        if (
            self.spec_count < 1
            or self.domain_count != 1
            or self.channels != 16
            or self.height != 8
            or self.width != 8
            or not self.compute_capability.startswith("sm_")
            or self.thread_extent not in {32, 64, 128, 256, 512, 1024}
            or self.target != "cuda"
            or self.symbol != ROOT_SPARSE_PATCHES_SEED_SYMBOL
        ):
            raise ValueError("root CROWN sparse Patches seed template differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": "boundflow.root-sparse-patches-seed-template/v1",
            "spec_count": self.spec_count,
            "domain_count": self.domain_count,
            "channels": self.channels,
            "height": self.height,
            "width": self.width,
            "compute_capability": self.compute_capability,
            "thread_extent": self.thread_extent,
            "target": self.target,
            "symbol": self.symbol,
            "source_representation": "sparse-patches-1x1",
            "dense_seed_external_allocation": False,
        }

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class CompiledRootCrownSparsePatchesSeedTIRV1:
    """Compiled sparse-seed module and auditable compiler identities."""

    executable: DifferentiableLowerTIRExecutable
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    tvm_version: str


def build_root_crown_sparse_patches_seed_modules_v1(
    template: RootCrownSparsePatchesSeedTemplateV1,
):
    """Build one CUDA kernel that scatters every sparse 1x1 patch."""

    template.validate()
    import tvm
    from tvm import te

    patches = te.placeholder(template.patches_shape, "float32", name="patches")
    location_h = te.placeholder((template.spec_count,), "int64", name="location_h")
    location_w = te.placeholder((template.spec_count,), "int64", name="location_w")
    dense = te.compute(
        template.coefficient_shape,
        lambda spec, domain, channel, height, width: tvm.tir.if_then_else(
            tvm.tir.all(
                height == location_h[spec],
                width == location_w[spec],
            ),
            patches[spec, domain, channel, 0, 0],
            tvm.tir.const(0.0, "float32"),
        ),
        name="dense_seed",
    )
    function = (
        te.create_prim_func([patches, location_h, location_w, dense])
        .with_attr("global_symbol", template.symbol)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", "root-crown-sparse-patches-seed/v1")
    )
    unscheduled = tvm.IRModule({template.symbol: function})
    schedule = tvm.tir.Schedule(unscheduled)
    block = schedule.get_block("dense_seed", func_name=template.symbol)
    loops = schedule.get_loops(block)
    fused = schedule.fuse(*loops)
    block_loop, thread_loop = schedule.split(
        fused, factors=[None, template.thread_extent]
    )
    schedule.bind(block_loop, "blockIdx.x")
    schedule.bind(thread_loop, "threadIdx.x")
    return unscheduled, schedule.mod


def compile_root_crown_sparse_patches_seed_tir_v1(
    template: RootCrownSparsePatchesSeedTemplateV1,
) -> CompiledRootCrownSparsePatchesSeedTIRV1:
    """Compile and hash the exact sparse-Patches seed lowering."""

    template.validate()
    import tvm

    unscheduled, scheduled = build_root_crown_sparse_patches_seed_modules_v1(template)
    executable = tvm.compile(
        scheduled, target=f"{template.target} -arch={template.compute_capability}"
    )
    sources = tuple(module.inspect_source() for module in executable.mod.imports)
    if not sources:
        raise RuntimeError("root CROWN sparse Patches seed produced no CUDA source")
    return CompiledRootCrownSparsePatchesSeedTIRV1(
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
    "ROOT_SPARSE_PATCHES_SEED_SYMBOL",
    "CompiledRootCrownSparsePatchesSeedTIRV1",
    "RootCrownSparsePatchesSeedTemplateV1",
    "build_root_crown_sparse_patches_seed_modules_v1",
    "compile_root_crown_sparse_patches_seed_tir_v1",
]
