"""Prepared runtime for sparse-Patches to dense CROWN seed lowering."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-instance-attributes
# pylint: disable=missing-function-docstring

from __future__ import annotations

from typing import Any, cast

import torch

from boundflow.backends.tvm.root_crown_sparse_patches_seed import (
    CompiledRootCrownSparsePatchesSeedTIRV1,
    RootCrownSparsePatchesSeedTemplateV1,
    compile_root_crown_sparse_patches_seed_tir_v1,
)


class RootCrownSparsePatchesSeedTIRExecutorV1:
    """Lower one-pixel Patches into a persistent dense coefficient arena."""

    def __init__(self, template: RootCrownSparsePatchesSeedTemplateV1) -> None:
        template.validate()
        self.template = template
        self.compiled: CompiledRootCrownSparsePatchesSeedTIRV1 = (
            compile_root_crown_sparse_patches_seed_tir_v1(template)
        )
        self._patches: torch.Tensor | None = None
        self._location_h: torch.Tensor | None = None
        self._location_w: torch.Tensor | None = None
        self._output: torch.Tensor | None = None
        self._view_cache: dict[tuple[int, tuple[int, ...], str], Any] = {}
        self.prepare_count = 0
        self.call_count = 0
        self.pointer_count = 0
        self.pointer_exact_count = 0
        self.input_copy_count = 0
        self.fallback_count = 0
        self.performance_claimed = False

    @staticmethod
    def _view_key(tensor: torch.Tensor) -> tuple[int, tuple[int, ...], str]:
        return tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype)

    def _view(self, tensor: torch.Tensor) -> Any:
        key = self._view_key(tensor)
        cached = self._view_cache.get(key)
        if cached is not None:
            return cached
        from tvm.runtime import from_dlpack

        view = from_dlpack(tensor)
        if torch.from_dlpack(view).data_ptr() != tensor.data_ptr():
            raise RuntimeError("root sparse Patches seed DLPack pointer differs")
        self._view_cache[key] = view
        return view

    def _launch(
        self,
        patches: torch.Tensor,
        location_h: torch.Tensor,
        location_w: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        import tvm_ffi

        device = patches.device
        stream = torch.cuda.current_stream(device)
        stream_id = int(stream.cuda_stream)
        ordinal = device.index
        if ordinal is None:
            ordinal = torch.cuda.current_device()
        with tvm_ffi.use_torch_stream(torch.cuda.stream(stream)):
            ffi_stream = int(tvm_ffi.get_raw_stream(tvm_ffi.device(f"cuda:{ordinal}")))
            if ffi_stream != stream_id:
                raise RuntimeError("root sparse Patches seed current stream differs")
            views = tuple(
                self._view(value) for value in (patches, location_h, location_w, output)
            )
            self.compiled.executable[self.template.symbol](*views)
        self.pointer_count += 4
        self.pointer_exact_count += 4

    def _validate_structure(
        self, patches: torch.Tensor, unstable_idx: tuple[torch.Tensor, ...]
    ) -> None:
        template = self.template
        if (
            tuple(patches.shape) != template.patches_shape
            or patches.device.type != "cuda"
            or patches.dtype != torch.float32
            or not patches.is_contiguous()
            or len(unstable_idx) != 3
        ):
            raise ValueError("root sparse Patches seed runtime structure differs")
        for value in unstable_idx:
            if (
                tuple(value.shape) != (template.spec_count,)
                or value.device != patches.device
                or value.dtype != torch.int64
                or not value.is_contiguous()
            ):
                raise ValueError("root sparse Patches seed index structure differs")

    def _validate_values(
        self, patches: torch.Tensor, unstable_idx: tuple[torch.Tensor, ...]
    ) -> None:
        template = self.template
        channel, height, width = unstable_idx
        if (
            not bool(torch.isfinite(patches).all().item())
            or bool(((channel < 0) | (channel >= template.channels)).any().item())
            or bool(((height < 0) | (height >= template.height)).any().item())
            or bool(((width < 0) | (width >= template.width)).any().item())
        ):
            raise ValueError("root sparse Patches seed value differs")
        rows = torch.arange(template.spec_count, device=patches.device)
        flattened = patches[:, 0, :, 0, 0]
        selected = flattened[rows, channel]
        if (
            not bool(torch.equal(selected, torch.ones_like(selected)))
            or int(torch.count_nonzero(flattened).item()) != template.spec_count
        ):
            raise ValueError("root sparse Patches seed identity payload differs")
        linear = (channel * template.height + height) * template.width + width
        if int(torch.unique(linear).numel()) != template.spec_count:
            raise ValueError("root sparse Patches seed locations differ")

    def prepare(self) -> None:
        """Allocate the output arena and warm the module outside query timing."""

        if self.prepare_count:
            raise RuntimeError("root sparse Patches seed already prepared")
        device = torch.device("cuda")
        self._patches = torch.empty(
            self.template.patches_shape, dtype=torch.float32, device=device
        )
        self._location_h = torch.empty(
            (self.template.spec_count,), dtype=torch.int64, device=device
        )
        self._location_w = torch.empty_like(self._location_h)
        self._output = torch.empty(
            self.template.coefficient_shape, dtype=torch.float32, device=device
        )
        self._patches.zero_()
        self._location_h.zero_()
        self._location_w.zero_()
        self._launch(self._patches, self._location_h, self._location_w, self._output)
        torch.cuda.synchronize(device)
        persistent = {
            tensor.data_ptr()
            for tensor in (
                self._patches,
                self._location_h,
                self._location_w,
                self._output,
            )
        }
        self._view_cache = {
            key: value
            for key, value in self._view_cache.items()
            if key[0] in persistent
        }
        self.call_count = 0
        self.pointer_count = 0
        self.pointer_exact_count = 0
        self.input_copy_count = 0
        self.prepare_count = 1

    def execute(
        self, patches: torch.Tensor, unstable_idx: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Validate and lower one exact sparse identity-Patches carrier."""

        if (
            self.prepare_count != 1
            or self._patches is None
            or self._location_h is None
            or self._location_w is None
            or self._output is None
        ):
            raise RuntimeError("root sparse Patches seed executor is not prepared")
        try:
            self._validate_structure(patches, unstable_idx)
            self._validate_values(patches, unstable_idx)
            self._patches.copy_(patches)
            self._location_h.copy_(unstable_idx[1])
            self._location_w.copy_(unstable_idx[2])
            self.input_copy_count += 3
            self._launch(
                self._patches,
                self._location_h,
                self._location_w,
                self._output,
            )
        except Exception:
            self.fallback_count += 1
            raise
        self.call_count += 1
        return self._output

    @property
    def selection_carrier(self) -> torch.Tensor:
        """Return a dense-layout carrier used only for sparse-alpha selection."""

        if self.prepare_count != 1 or self._output is None:
            raise RuntimeError("root sparse Patches selection carrier is not prepared")
        return self._output

    def receipt(self) -> dict[str, object]:
        return {
            "schema_version": "boundflow.root-sparse-patches-seed-tir/v1",
            "template_hash": self.template.stable_hash(),
            "scheduled_tir_hash": self.compiled.scheduled_tir_hash,
            "device_source_hash": self.compiled.device_source_hash,
            "call_count": self.call_count,
            "fallback_count": self.fallback_count,
            "pointer_count": self.pointer_count,
            "pointer_exact_count": self.pointer_exact_count,
            "input_copy_count": self.input_copy_count,
            "persistent_dense_seed_arena": True,
            "persistent_aligned_input_arena": True,
            "performance_claimed": False,
        }


def patches_payload_v1(specification: Any) -> torch.Tensor:
    """Extract the exact tensor payload without importing auto_LiRPA types."""

    patches = getattr(specification, "patches", None)
    if not torch.is_tensor(patches):
        raise TypeError("root sparse Patches payload differs")
    return cast(torch.Tensor, patches)


__all__ = [
    "RootCrownSparsePatchesSeedTIRExecutorV1",
    "patches_payload_v1",
]
