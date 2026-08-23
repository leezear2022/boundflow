"""B4-B2 B2-0 differentiable CUDA/TIR identity ABI probe."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,missing-function-docstring
# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
# pylint: disable=too-many-boolean-expressions,too-few-public-methods
# pylint: disable=abstract-method,arguments-differ

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from boundflow.backends.tvm.differentiable_lower_identity import (
    TVM_COMMIT,
    TVM_FFI_COMMIT,
    CompiledDifferentiableLowerIdentityTIR,
    DifferentiableLowerIdentityTIRKey,
    compile_identity_tir,
)
from boundflow.ir.differentiable_lower_region import (
    DifferentiableLowerRegionIRV1,
    DifferentiableLowerRegionInstanceV1,
)
from boundflow.ir.differentiable_lower_tir import (
    DifferentiableLowerTIRInstanceV1,
    DifferentiableLowerTIRLaunchReceiptV1,
    DifferentiableLowerTIRModuleReceiptV1,
    DifferentiableLowerTIRScheduleV1,
    DifferentiableLowerTIRTemplateV1,
    canonical_tir_hash,
)
from boundflow.runtime.rvir_v4_production_state import production_tensor_sha256


def build_b4b2_identity_template_v1(
    lower_ir: DifferentiableLowerRegionIRV1,
    *,
    tensor_numel: int,
    compute_capability: str,
) -> DifferentiableLowerTIRTemplateV1:
    """Bind the identity probe to one frozen B4-B1 semantic IR."""

    lower_ir.validate()
    mapping_contracts = [
        contract.to_dict()
        for contract in lower_ir.tensor_contracts
        if contract.name.startswith("mapping/")
    ]
    template = DifferentiableLowerTIRTemplateV1(
        lower_region_ir_hash=lower_ir.stable_hash(),
        anchor_id=lower_ir.anchor_id,
        operator_kind=lower_ir.operator_kind,
        mapping_layout_hash=canonical_tir_hash(mapping_contracts),
        operator_attributes_hash=canonical_tir_hash(dict(lower_ir.operator_attributes)),
        abi="identity-probe-v1",
        dtype="torch.float32",
        device_kind="cuda",
        target="cuda",
        compute_capability=compute_capability,
        tensor_numel=tensor_numel,
        gradient_targets=("input",),
    )
    template.validate()
    return template


def build_b4b2_identity_schedule_v1(
    template: DifferentiableLowerTIRTemplateV1,
    *,
    thread_extent: int = 256,
) -> DifferentiableLowerTIRScheduleV1:
    """Create the single preregistered B2-0 identity schedule."""

    template.validate()
    schedule = DifferentiableLowerTIRScheduleV1(
        template_hash=template.stable_hash(),
        tensor_numel=template.tensor_numel,
        thread_extent=thread_extent,
        block_extent=(template.tensor_numel + thread_extent - 1) // thread_extent,
    )
    schedule.validate_against(template)
    return schedule


def build_b4b2_identity_instance_v1(
    template: DifferentiableLowerTIRTemplateV1,
    lower_ir: DifferentiableLowerRegionIRV1,
    lower_instance: DifferentiableLowerRegionInstanceV1,
    input_tensor: torch.Tensor,
    upstream_gradient: torch.Tensor,
) -> DifferentiableLowerTIRInstanceV1:
    """Bind dynamic probe tensors without polluting the compile key."""

    lower_instance.validate_against(lower_ir)
    _validate_probe_tensor(input_tensor, template, requires_grad=True)
    _validate_probe_tensor(upstream_gradient, template, requires_grad=False)
    if input_tensor.shape != upstream_gradient.shape:
        raise ValueError("identity TIR upstream gradient shape differs")
    ordinal = input_tensor.device.index
    if ordinal is None:
        ordinal = torch.cuda.current_device()
    instance = DifferentiableLowerTIRInstanceV1(
        template_hash=template.stable_hash(),
        lower_region_instance_hash=lower_instance.stable_hash(lower_ir),
        tensor_shape=tuple(int(value) for value in input_tensor.shape),
        input_tensor_hash=production_tensor_sha256(input_tensor),
        upstream_gradient_hash=production_tensor_sha256(upstream_gradient),
        device_ordinal=int(ordinal),
    )
    instance.validate_against(template)
    return instance


def _validate_probe_tensor(
    tensor: torch.Tensor,
    template: DifferentiableLowerTIRTemplateV1,
    *,
    requires_grad: bool,
) -> None:
    if (
        not torch.is_tensor(tensor)
        or tensor.device.type != "cuda"
        or tensor.dtype != torch.float32
        or not tensor.is_contiguous()
        or tensor.numel() != template.tensor_numel
        or tensor.requires_grad is not requires_grad
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise ValueError("identity TIR probe tensor differs")


class DifferentiableLowerIdentityModuleCache:
    """Explicit in-process cache with a receipt for every compiled identity module."""

    def __init__(self) -> None:
        self._entries: dict[
            str,
            tuple[
                CompiledDifferentiableLowerIdentityTIR,
                DifferentiableLowerTIRModuleReceiptV1,
            ],
        ] = {}

    def get(
        self,
        template: DifferentiableLowerTIRTemplateV1,
        schedule: DifferentiableLowerTIRScheduleV1,
    ) -> tuple[
        CompiledDifferentiableLowerIdentityTIR,
        DifferentiableLowerTIRModuleReceiptV1,
        str,
    ]:
        template.validate()
        schedule.validate_against(template)
        cache_key = DifferentiableLowerTIRModuleReceiptV1.expected_cache_key(
            template, schedule
        )
        existing = self._entries.get(cache_key)
        if existing is not None:
            return (*existing, "hit")
        key = DifferentiableLowerIdentityTIRKey(
            template_hash=template.stable_hash(),
            tensor_numel=template.tensor_numel,
            thread_extent=schedule.thread_extent,
            compute_capability=template.compute_capability,
        )
        compiled = compile_identity_tir(key)
        receipt = DifferentiableLowerTIRModuleReceiptV1(
            template_hash=template.stable_hash(),
            schedule_hash=schedule.stable_hash(template),
            unscheduled_tir_hash=compiled.unscheduled_tir_hash,
            scheduled_tir_hash=compiled.scheduled_tir_hash,
            device_source_hash=compiled.device_source_hash,
            cache_key=cache_key,
            target=template.target,
            compute_capability=template.compute_capability,
            tvm_version=compiled.tvm_version,
            tvm_commit=TVM_COMMIT,
            tvm_ffi_commit=TVM_FFI_COMMIT,
            torch_version=str(torch.__version__),
            exported_symbols=(template.forward_symbol, template.backward_symbol),
        )
        receipt.validate_against(template, schedule)
        self._entries[cache_key] = (compiled, receipt)
        return compiled, receipt, "miss"


@dataclass(frozen=True)
class _LaunchObservation:
    stream_id: int
    tvm_ffi_stream_id: int
    source_data_ptr: int
    result_data_ptr: int
    source_roundtrip_ptr_exact: bool
    result_roundtrip_ptr_exact: bool


class _IdentityTIRExecutor:
    def __init__(
        self,
        template: DifferentiableLowerTIRTemplateV1,
        schedule: DifferentiableLowerTIRScheduleV1,
        cache: DifferentiableLowerIdentityModuleCache,
    ) -> None:
        self.template = template
        self.schedule = schedule
        self.compiled, self.module_receipt, self.cache_event = cache.get(
            template, schedule
        )
        self.forward_observation: Optional[_LaunchObservation] = None
        self.backward_observation: Optional[_LaunchObservation] = None
        self.forward_launch_count = 0
        self.backward_launch_count = 0

    def _launch(
        self, symbol: str, source: torch.Tensor, result: torch.Tensor
    ) -> _LaunchObservation:
        import tvm
        import tvm_ffi

        current = torch.cuda.current_stream(source.device)
        ordinal = source.device.index
        if ordinal is None:
            ordinal = torch.cuda.current_device()
        with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
            ffi_stream_id = int(
                tvm_ffi.get_raw_stream(tvm_ffi.device(f"cuda:{ordinal}"))
            )
            if ffi_stream_id != int(current.cuda_stream):
                raise RuntimeError("identity TIR current stream differs")
            source_flat = source.view(-1)
            result_flat = result.view(-1)
            source_view = tvm.runtime.from_dlpack(source_flat)
            result_view = tvm.runtime.from_dlpack(result_flat)
            source_roundtrip = torch.from_dlpack(source_view)
            result_roundtrip = torch.from_dlpack(result_view)
            source_exact = source_roundtrip.data_ptr() == source_flat.data_ptr()
            result_exact = result_roundtrip.data_ptr() == result_flat.data_ptr()
            if not source_exact or not result_exact:
                raise RuntimeError("identity TIR DLPack pointer differs")
            self.compiled.executable[symbol](source_view, result_view)
        return _LaunchObservation(
            stream_id=int(current.cuda_stream),
            tvm_ffi_stream_id=ffi_stream_id,
            source_data_ptr=source.data_ptr(),
            result_data_ptr=result.data_ptr(),
            source_roundtrip_ptr_exact=source_exact,
            result_roundtrip_ptr_exact=result_exact,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        _validate_probe_tensor(input_tensor, self.template, requires_grad=True)
        if self.forward_launch_count != 0:
            raise RuntimeError("identity TIR forward launched more than once")
        output = torch.empty_like(input_tensor)
        self.forward_observation = self._launch(
            self.template.forward_symbol, input_tensor, output
        )
        self.forward_launch_count += 1
        return output

    def backward(self, upstream_gradient: torch.Tensor) -> torch.Tensor:
        _validate_probe_tensor(upstream_gradient, self.template, requires_grad=False)
        if self.backward_launch_count != 0:
            raise RuntimeError("identity TIR backward launched more than once")
        input_gradient = torch.empty_like(upstream_gradient)
        self.backward_observation = self._launch(
            self.template.backward_symbol, upstream_gradient, input_gradient
        )
        self.backward_launch_count += 1
        return input_gradient


class _DifferentiableLowerIdentityTIRFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, executor: _IdentityTIRExecutor):
        ctx.executor = executor
        ctx.set_materialize_grads(False)
        return executor.forward(input_tensor)

    @staticmethod
    def backward(ctx, upstream_gradient: torch.Tensor):
        if torch.is_grad_enabled():
            raise RuntimeError("identity TIR higher-order gradients are unsupported")
        if upstream_gradient is None:
            raise RuntimeError("identity TIR requires an explicit upstream gradient")
        return ctx.executor.backward(upstream_gradient), None


@dataclass(frozen=True)
class DifferentiableLowerIdentityProbeResultV1:
    """B2-0 numerical output, gradient and their compiler/launch receipts."""

    output: torch.Tensor
    input_gradient: torch.Tensor
    module_receipt: DifferentiableLowerTIRModuleReceiptV1
    launch_receipt: DifferentiableLowerTIRLaunchReceiptV1


def run_b4b2_identity_tir_probe_v1(
    template: DifferentiableLowerTIRTemplateV1,
    instance: DifferentiableLowerTIRInstanceV1,
    schedule: DifferentiableLowerTIRScheduleV1,
    input_tensor: torch.Tensor,
    upstream_gradient: torch.Tensor,
    *,
    cache: Optional[DifferentiableLowerIdentityModuleCache] = None,
) -> DifferentiableLowerIdentityProbeResultV1:
    """Run exactly one TIR forward and backward without an eager fallback."""

    instance.validate_against(template)
    schedule.validate_against(template)
    _validate_probe_tensor(input_tensor, template, requires_grad=True)
    _validate_probe_tensor(upstream_gradient, template, requires_grad=False)
    if production_tensor_sha256(input_tensor) != instance.input_tensor_hash:
        raise ValueError("identity TIR input differs from instance")
    if production_tensor_sha256(upstream_gradient) != instance.upstream_gradient_hash:
        raise ValueError("identity TIR upstream gradient differs from instance")
    executor = _IdentityTIRExecutor(
        template, schedule, cache or DifferentiableLowerIdentityModuleCache()
    )
    output = _DifferentiableLowerIdentityTIRFunction.apply(input_tensor, executor)
    input_gradient = torch.autograd.grad(
        output,
        input_tensor,
        grad_outputs=upstream_gradient,
        create_graph=False,
        retain_graph=False,
    )[0]
    forward = executor.forward_observation
    backward = executor.backward_observation
    if forward is None or backward is None:
        raise RuntimeError("identity TIR launch inventory is incomplete")
    if (
        backward.stream_id != forward.stream_id
        or backward.tvm_ffi_stream_id != forward.tvm_ffi_stream_id
    ):
        raise RuntimeError("identity TIR forward/backward stream differs")
    launch = DifferentiableLowerTIRLaunchReceiptV1(
        template_hash=template.stable_hash(),
        instance_hash=instance.stable_hash(template),
        schedule_hash=schedule.stable_hash(template),
        module_receipt_hash=executor.module_receipt.stable_hash(template, schedule),
        stream_id=forward.stream_id,
        tvm_ffi_stream_id=forward.tvm_ffi_stream_id,
        input_data_ptr=forward.source_data_ptr,
        output_data_ptr=forward.result_data_ptr,
        upstream_gradient_data_ptr=backward.source_data_ptr,
        input_gradient_data_ptr=backward.result_data_ptr,
        input_roundtrip_ptr_exact=forward.source_roundtrip_ptr_exact,
        output_roundtrip_ptr_exact=forward.result_roundtrip_ptr_exact,
        upstream_gradient_roundtrip_ptr_exact=backward.source_roundtrip_ptr_exact,
        input_gradient_roundtrip_ptr_exact=backward.result_roundtrip_ptr_exact,
        output_aliases_input=output.data_ptr() == input_tensor.data_ptr(),
        input_gradient_aliases_upstream=(
            input_gradient.data_ptr() == upstream_gradient.data_ptr()
        ),
        output_tensor_hash=production_tensor_sha256(output),
        input_gradient_hash=production_tensor_sha256(input_gradient),
        cache_event=executor.cache_event,
        forward_launch_count=executor.forward_launch_count,
        backward_launch_count=executor.backward_launch_count,
        fallback_count=0,
        eager_backward_count=0,
    )
    launch.validate_against(template, instance, schedule, executor.module_receipt)
    return DifferentiableLowerIdentityProbeResultV1(
        output=output,
        input_gradient=input_gradient,
        module_receipt=executor.module_receipt,
        launch_receipt=launch,
    )


__all__ = [
    "DifferentiableLowerIdentityModuleCache",
    "DifferentiableLowerIdentityProbeResultV1",
    "build_b4b2_identity_instance_v1",
    "build_b4b2_identity_schedule_v1",
    "build_b4b2_identity_template_v1",
    "run_b4b2_identity_tir_probe_v1",
]
