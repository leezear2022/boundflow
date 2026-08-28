"""Canonical S1 mixed Relax/TIR compiler path for the CIBC IBP graph."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-arguments
# pylint: disable=too-many-instance-attributes,too-many-locals
# pylint: disable=missing-function-docstring,protected-access
# pylint: disable=too-many-boolean-expressions,too-few-public-methods

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import time
from typing import Any, Mapping, Sequence, cast

import torch

from boundflow.backends.tvm.relax_interval_task_ops import (
    IntervalTaskLoweringConfig,
    IntervalTaskLoweringSpec,
    build_interval_task_relax_ops_ir_module,
)
from boundflow.domains.interval import IntervalState
from boundflow.frontends.plain_crown_bound_ir import tensor_content_hash
from boundflow.ir.task import BFTaskModule, BufferSpec, StoragePlan, TaskKind

S1_COMPILE_RECEIPT_SCHEMA = "boundflow.asplos27-s1-cibc-compile/v1"
S1_EXECUTION_RECEIPT_SCHEMA = "boundflow.asplos27-s1-cibc-execution/v1"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _task_source_hash(module: BFTaskModule) -> str:
    task = module.get_entry_task()
    params = module.bindings.get("params", {})
    if not isinstance(params, Mapping):
        raise TypeError("S1 CIBC module params differ")
    payload = {
        "task_id": task.task_id,
        "kind": task.kind.value,
        "inputs": list(task.input_values),
        "outputs": list(task.output_values),
        "params": {
            name: tensor_content_hash(value)
            for name, value in sorted(params.items())
            if torch.is_tensor(value)
        },
        "ops": [
            {
                "op_type": op.op_type,
                "name": op.name,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
                "attrs": _jsonable(op.attrs),
            }
            for op in task.ops
        ],
    }
    return _canonical_hash(payload)


def specialize_storage_plan_batch_v1(
    storage_plan: StoragePlan, *, batch_size: int
) -> StoragePlan:
    """Specialize importer batch-one activation buffers to one exact runtime batch."""

    if batch_size <= 0:
        raise ValueError("S1 CIBC batch size must be positive")

    def specialize(spec: BufferSpec) -> BufferSpec:
        if spec.scope in {"param", "const"} or not spec.shape:
            return spec
        if spec.shape[0] not in {None, 1, batch_size}:
            raise ValueError(
                f"S1 CIBC activation batch differs: {spec.buffer_id}={spec.shape[0]}"
            )
        return replace(spec, shape=[batch_size, *spec.shape[1:]])

    result = StoragePlan(
        buffers={key: specialize(value) for key, value in storage_plan.buffers.items()},
        value_to_buffer=dict(storage_plan.value_to_buffer),
        physical_buffers={
            key: specialize(value)
            for key, value in storage_plan.physical_buffers.items()
        },
        logical_to_physical=dict(storage_plan.logical_to_physical),
    )
    result.validate()
    return result


def _storage_hash(storage_plan: StoragePlan) -> str:
    payload = {
        "buffers": {
            key: {
                "dtype": value.dtype,
                "shape": value.shape,
                "scope": value.scope,
                "device": value.device,
                "layout": value.layout,
            }
            for key, value in sorted(storage_plan.buffers.items())
        },
        "value_to_buffer": dict(sorted(storage_plan.value_to_buffer.items())),
        "logical_to_physical": dict(sorted(storage_plan.logical_to_physical.items())),
    }
    return _canonical_hash(payload)


@dataclass(frozen=True)
class S1CIBCCompileReceiptV1:
    """One-way identity chain from source task through the executable module."""

    source_task_hash: str
    specialized_storage_hash: str
    plan_hash: str
    source_relax_ir_hash: str
    lowered_relax_ir_hash: str
    device_source_hashes: tuple[str, ...]
    target: str
    op_count: int
    cibc_conv_ops: tuple[str, ...]
    cibc_threads_by_op: tuple[tuple[str, int], ...]
    cublas_partition_count: int
    compile_ms: float
    schema_version: str = S1_COMPILE_RECEIPT_SCHEMA
    fallback_admitted: bool = False
    performance_claimed: bool = False

    def validate(self) -> None:
        if self.schema_version != S1_COMPILE_RECEIPT_SCHEMA:
            raise ValueError("S1 CIBC compile receipt schema differs")
        hashes = (
            self.source_task_hash,
            self.specialized_storage_hash,
            self.plan_hash,
            self.source_relax_ir_hash,
            self.lowered_relax_ir_hash,
            *self.device_source_hashes,
        )
        if any(len(value) != 64 for value in hashes):
            raise ValueError("S1 CIBC compile identity differs")
        if (
            not self.target
            or self.op_count <= 0
            or not self.cibc_conv_ops
            or len(self.cibc_conv_ops) != len(self.cibc_threads_by_op)
            or self.cublas_partition_count <= 0
            or {name for name, _threads in self.cibc_threads_by_op}
            != set(self.cibc_conv_ops)
            or not math.isfinite(self.compile_ms)
            or self.compile_ms <= 0.0
            or self.fallback_admitted
            or self.performance_claimed
        ):
            raise ValueError("S1 CIBC compile receipt differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["cibc_conv_ops"] = list(self.cibc_conv_ops)
        payload["cibc_threads_by_op"] = [
            [name, threads] for name, threads in self.cibc_threads_by_op
        ]
        payload["device_source_hashes"] = list(self.device_source_hashes)
        payload["receipt_hash"] = _canonical_hash(payload)
        return payload

    def stable_hash(self) -> str:
        return str(self.to_dict()["receipt_hash"])


@dataclass(frozen=True)
class S1CIBCExecutionReceiptV1:
    """Warm invocation evidence without per-run DLPack construction."""

    compile_receipt_hash: str
    invocation_ordinal: int
    vm_submission_count: int
    cuda_graph_replay_count: int
    cibc_conv_call_tir_count: int
    input_copy_included: bool
    output_materialization_copy_included: bool
    output_materialization_bytes: int
    prepare_dlpack_view_count: int
    warm_dlpack_view_count: int
    fallback_count: int
    eager_shadow_count: int
    input_pointer_signature: tuple[int, int]
    output_pointer_signature: tuple[int, int]
    schema_version: str = S1_EXECUTION_RECEIPT_SCHEMA
    performance_claimed: bool = False

    def validate(self) -> None:
        if (
            self.schema_version != S1_EXECUTION_RECEIPT_SCHEMA
            or len(self.compile_receipt_hash) != 64
            or self.invocation_ordinal <= 0
            or self.vm_submission_count != 1
            or self.cuda_graph_replay_count not in {0, 1}
            or self.cibc_conv_call_tir_count <= 0
            or not self.input_copy_included
            or self.output_materialization_bytes < 0
            or self.output_materialization_copy_included
            != (self.output_materialization_bytes > 0)
            or self.prepare_dlpack_view_count <= 0
            or self.warm_dlpack_view_count != 0
            or self.fallback_count != 0
            or self.eager_shadow_count != 0
            or any(pointer <= 0 for pointer in self.input_pointer_signature)
            or any(pointer <= 0 for pointer in self.output_pointer_signature)
            or self.performance_claimed
        ):
            raise ValueError("S1 CIBC execution receipt differs")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        payload["input_pointer_signature"] = list(self.input_pointer_signature)
        payload["output_pointer_signature"] = list(self.output_pointer_signature)
        payload["receipt_hash"] = _canonical_hash(payload)
        return payload


class PreparedS1CIBCProgramV1:
    """Compile/prepare once and execute one exact IBP graph through Relax/TIR."""

    def __init__(
        self,
        *,
        executable: Any,
        lowering_spec: IntervalTaskLoweringSpec,
        module: BFTaskModule,
        input_lower: torch.Tensor,
        input_upper: torch.Tensor,
        compile_receipt: S1CIBCCompileReceiptV1,
    ) -> None:
        import tvm
        import tvm_ffi
        from tvm import relax

        if input_lower.device.type != "cuda":
            raise ValueError("S1 CIBC prepared program requires CUDA")
        if (
            input_lower.shape != input_upper.shape
            or input_lower.dtype != torch.float32
            or input_upper.dtype != torch.float32
            or input_lower.device != input_upper.device
            or not input_lower.is_contiguous()
            or not input_upper.is_contiguous()
        ):
            raise ValueError("S1 CIBC prepared input contract differs")
        compile_receipt.validate()
        self.compile_receipt = compile_receipt
        self.compile_receipt_hash = compile_receipt.stable_hash()
        self.lowering_spec = lowering_spec
        self.input_lower = input_lower.clone()
        self.input_upper = input_upper.clone()
        self.device = input_lower.device
        self.stream = int(torch.cuda.current_stream(self.device).cuda_stream)
        params = module.bindings.get("params", {})
        if not isinstance(params, Mapping):
            raise TypeError("S1 CIBC prepared params differ")
        self.params = tuple(
            params[name].detach().contiguous() for name in lowering_spec.param_values
        )
        if any(
            not torch.is_tensor(value)
            or value.device != self.device
            or value.dtype != torch.float32
            or value.requires_grad
            for value in self.params
        ):
            raise ValueError("S1 CIBC prepared parameter contract differs")
        self.param_identity = tuple(
            (value.data_ptr(), value._version) for value in self.params
        )
        self.argument_tensors = (self.input_lower, self.input_upper, *self.params)
        self.argument_views = tuple(
            tvm.runtime.from_dlpack(value) for value in self.argument_tensors
        )
        ordinal = self.device.index
        if ordinal is None:
            ordinal = torch.cuda.current_device()
        self.vm = relax.VirtualMachine(executable, tvm.cuda(ordinal))
        self.function = self.vm[lowering_spec.func_name]
        output_value = lowering_spec.output_values[0]
        output_spec = module.storage_plan.buffers[
            module.storage_plan.value_to_buffer[output_value]
        ]
        if any(dimension is None for dimension in output_spec.shape):
            raise ValueError("S1 CIBC prepared output shape is dynamic")
        output_shape = tuple(
            input_lower.shape[0] if index == 0 else int(cast(int, dimension))
            for index, dimension in enumerate(output_spec.shape)
        )
        self.output_lower = torch.empty(
            output_shape, dtype=torch.float32, device=self.device
        )
        self.output_upper = torch.empty_like(self.output_lower)
        self.output_views = (
            tvm.runtime.from_dlpack(self.output_lower),
            tvm.runtime.from_dlpack(self.output_upper),
        )
        self.prepare_dlpack_view_count = len(self.argument_views) + len(
            self.output_views
        )
        self.output_materialization_bytes = sum(
            value.numel() * value.element_size()
            for value in (self.output_lower, self.output_upper)
        )
        self.invocation_count = 0
        self.last_receipt: S1CIBCExecutionReceiptV1 | None = None
        self._admitted_dynamic_identity: (
            tuple[tuple[int, int], tuple[int, int]] | None
        ) = None
        self.admit_dynamic_input(input_lower, input_upper)
        with tvm_ffi.use_torch_stream():
            warm = self.function(*self.argument_views)
            self.output_views[0].copyfrom(warm[0])
            self.output_views[1].copyfrom(warm[1])
        torch.cuda.synchronize(self.device)

    def _validate_static_identity(self) -> None:
        if int(torch.cuda.current_stream(self.device).cuda_stream) != self.stream:
            raise RuntimeError("S1 CIBC prepared stream differs")
        if tuple((value.data_ptr(), value._version) for value in self.params) != (
            self.param_identity
        ):
            raise RuntimeError("S1 CIBC prepared parameter identity differs")

    def _validate_dynamic_metadata(
        self, input_lower: torch.Tensor, input_upper: torch.Tensor
    ) -> None:
        if (
            input_lower.shape != self.input_lower.shape
            or input_upper.shape != self.input_upper.shape
            or input_lower.dtype != torch.float32
            or input_upper.dtype != torch.float32
            or input_lower.device != self.device
            or input_upper.device != self.device
            or not input_lower.is_contiguous()
            or not input_upper.is_contiguous()
        ):
            raise ValueError("S1 CIBC dynamic input contract differs")

    def admit_dynamic_input(
        self, input_lower: torch.Tensor, input_upper: torch.Tensor
    ) -> None:
        """Perform content checks once, then bind an O(1) versioned input identity."""

        self._validate_dynamic_metadata(input_lower, input_upper)
        if (
            not bool(torch.isfinite(input_lower).all().item())
            or not bool(torch.isfinite(input_upper).all().item())
            or not bool(torch.all(input_lower <= input_upper).item())
        ):
            raise ValueError("S1 CIBC dynamic input content differs")
        self._admitted_dynamic_identity = (
            (input_lower.data_ptr(), input_lower._version),
            (input_upper.data_ptr(), input_upper._version),
        )

    def _validate_dynamic_input(
        self, input_lower: torch.Tensor, input_upper: torch.Tensor
    ) -> None:
        self._validate_dynamic_metadata(input_lower, input_upper)
        identity = (
            (input_lower.data_ptr(), input_lower._version),
            (input_upper.data_ptr(), input_upper._version),
        )
        if identity != self._admitted_dynamic_identity:
            raise ValueError("S1 CIBC dynamic input is not admitted")

    def _execution_receipt(
        self,
        *,
        cuda_graph_replay_count: int,
        output_pointer_signature: tuple[int, int] | None = None,
        output_materialization_copy_included: bool = True,
        output_materialization_bytes: int | None = None,
        prepare_dlpack_view_count: int | None = None,
    ) -> S1CIBCExecutionReceiptV1:
        materialization_bytes = (
            self.output_materialization_bytes
            if output_materialization_bytes is None
            else output_materialization_bytes
        )
        return S1CIBCExecutionReceiptV1(
            compile_receipt_hash=self.compile_receipt_hash,
            invocation_ordinal=self.invocation_count,
            vm_submission_count=1,
            cuda_graph_replay_count=cuda_graph_replay_count,
            cibc_conv_call_tir_count=len(self.lowering_spec.cibc_conv_ops),
            input_copy_included=True,
            output_materialization_copy_included=(output_materialization_copy_included),
            output_materialization_bytes=materialization_bytes,
            prepare_dlpack_view_count=(
                self.prepare_dlpack_view_count
                if prepare_dlpack_view_count is None
                else prepare_dlpack_view_count
            ),
            warm_dlpack_view_count=0,
            fallback_count=0,
            eager_shadow_count=0,
            input_pointer_signature=(
                self.input_lower.data_ptr(),
                self.input_upper.data_ptr(),
            ),
            output_pointer_signature=(
                output_pointer_signature
                if output_pointer_signature is not None
                else (self.output_lower.data_ptr(), self.output_upper.data_ptr())
            ),
        )

    def run(
        self, *, input_lower: torch.Tensor, input_upper: torch.Tensor
    ) -> tuple[IntervalState, S1CIBCExecutionReceiptV1]:
        import tvm_ffi

        self._validate_static_identity()
        self._validate_dynamic_input(input_lower, input_upper)
        self.input_lower.copy_(input_lower)
        self.input_upper.copy_(input_upper)
        with tvm_ffi.use_torch_stream():
            result = self.function(*self.argument_views)
            self.output_views[0].copyfrom(result[0])
            self.output_views[1].copyfrom(result[1])
        self.invocation_count += 1
        receipt = self._execution_receipt(cuda_graph_replay_count=0)
        receipt.validate()
        self.last_receipt = receipt
        return (
            IntervalState(lower=self.output_lower, upper=self.output_upper),
            receipt,
        )


class PreparedS1CIBCCUDAGraphV1:
    """Static-address capture of one already prepared mixed Relax/TIR program."""

    def __init__(self, prepared: PreparedS1CIBCProgramV1) -> None:
        import tvm_ffi

        self.prepared = prepared
        capture_stream = torch.cuda.Stream(device=prepared.device)
        capture_stream.wait_stream(torch.cuda.current_stream(prepared.device))
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                with tvm_ffi.use_torch_stream():
                    result = prepared.function(*prepared.argument_views)
                    prepared.output_views[0].copyfrom(result[0])
                    prepared.output_views[1].copyfrom(result[1])
        capture_stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            with torch.cuda.graph(self.graph, stream=capture_stream):
                with tvm_ffi.use_torch_stream():
                    self.captured_result = prepared.function(*prepared.argument_views)
        capture_stream.synchronize()
        self.output_lower = torch.from_dlpack(self.captured_result[0])
        self.output_upper = torch.from_dlpack(self.captured_result[1])
        self.prepare_dlpack_view_count = prepared.prepare_dlpack_view_count + 2

    def run(
        self, *, input_lower: torch.Tensor, input_upper: torch.Tensor
    ) -> tuple[IntervalState, S1CIBCExecutionReceiptV1]:
        prepared = self.prepared
        prepared._validate_static_identity()
        prepared._validate_dynamic_input(input_lower, input_upper)
        prepared.input_lower.copy_(input_lower)
        prepared.input_upper.copy_(input_upper)
        self.graph.replay()
        prepared.invocation_count += 1
        receipt = prepared._execution_receipt(
            cuda_graph_replay_count=1,
            output_pointer_signature=(
                self.output_lower.data_ptr(),
                self.output_upper.data_ptr(),
            ),
            output_materialization_copy_included=False,
            output_materialization_bytes=0,
            prepare_dlpack_view_count=self.prepare_dlpack_view_count,
        )
        receipt.validate()
        prepared.last_receipt = receipt
        return (
            IntervalState(lower=self.output_lower, upper=self.output_upper),
            receipt,
        )


def prepare_s1_cibc_program_v1(
    module: BFTaskModule,
    *,
    input_lower: torch.Tensor,
    input_upper: torch.Tensor,
    cibc_threads_by_op: Sequence[tuple[str, int]],
) -> PreparedS1CIBCProgramV1:
    """Build the exact-signature mixed module and bind persistent CUDA views."""

    import tvm
    from tvm import dlight as dl
    from tvm import relax, transform

    module.validate()
    task = module.get_entry_task()
    if task.kind != TaskKind.INTERVAL_IBP or len(task.output_values) != 1:
        raise ValueError("S1 CIBC source task differs")
    if input_lower.shape != input_upper.shape:
        raise ValueError("S1 CIBC prepare input pair differs")
    storage_plan = specialize_storage_plan_batch_v1(
        module.storage_plan, batch_size=int(input_lower.shape[0])
    )
    specialized_module = replace(module, storage_plan=storage_plan)
    config = IntervalTaskLoweringConfig(
        conv_backend="cibc_tir",
        cibc_threads_by_op=tuple(cibc_threads_by_op),
        reject_unbound_cibc_conv=True,
    )
    source_task_hash = _task_source_hash(specialized_module)
    storage_hash = _storage_hash(storage_plan)
    plan_payload = {
        "source_task_hash": source_task_hash,
        "specialized_storage_hash": storage_hash,
        "conv_backend": config.conv_backend,
        "cibc_threads_by_op": [list(item) for item in config.cibc_threads_by_op],
        "fallback_admitted": False,
    }
    plan_hash = _canonical_hash(plan_payload)
    ir_module, lowering_spec = build_interval_task_relax_ops_ir_module(
        task,
        storage_plan=storage_plan,
        target="cuda",
        func_name="main",
        config=config,
    )
    source_relax_ir_hash = hashlib.sha256(
        tvm.ir.save_json(ir_module).encode("utf-8")
    ).hexdigest()
    from tvm.relax.backend.cuda.cublas import partition_for_cublas

    partitioned_module = partition_for_cublas(ir_module, bind_constants=False)
    cublas_partition_count = sum(
        1
        for function in partitioned_module.functions.values()
        if str(function.attrs.get("Codegen", "")) == "cublas"
    )
    if cublas_partition_count <= 0:
        raise RuntimeError("S1 CIBC Relax graph has no cuBLAS partitions")
    lowered_module = relax.transform.RunCodegen()(partitioned_module)
    lowered_relax_ir_hash = hashlib.sha256(
        tvm.ir.save_json(lowered_module).encode("utf-8")
    ).hexdigest()
    major, minor = torch.cuda.get_device_capability(input_lower.device)
    target = tvm.target.Target(f"cuda -arch=sm_{major}{minor}", host="llvm")
    default_schedule = dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(),
        dl.gpu.GEMV(),
        dl.gpu.Reduction(),
        dl.gpu.GeneralReduction(),
        dl.gpu.Transpose(),
        dl.gpu.Fallback(),
    )
    tir_pipeline = transform.Sequential(
        [default_schedule, tvm.tir.get_default_tir_pipeline(target)]
    )
    started = time.perf_counter()
    executable = relax.build(lowered_module, target=target, tir_pipeline=tir_pipeline)
    compile_ms = (time.perf_counter() - started) * 1000.0
    sources = tuple(imported.inspect_source() for imported in executable.mod.imports)
    if not sources:
        raise RuntimeError("S1 CIBC executable has no device source")
    receipt = S1CIBCCompileReceiptV1(
        source_task_hash=source_task_hash,
        specialized_storage_hash=storage_hash,
        plan_hash=plan_hash,
        source_relax_ir_hash=source_relax_ir_hash,
        lowered_relax_ir_hash=lowered_relax_ir_hash,
        device_source_hashes=tuple(
            hashlib.sha256(source.encode("utf-8")).hexdigest() for source in sources
        ),
        target=str(target),
        op_count=len(task.ops),
        cibc_conv_ops=lowering_spec.cibc_conv_ops,
        cibc_threads_by_op=tuple(cibc_threads_by_op),
        cublas_partition_count=cublas_partition_count,
        compile_ms=compile_ms,
    )
    receipt.validate()
    return PreparedS1CIBCProgramV1(
        executable=executable,
        lowering_spec=lowering_spec,
        module=specialized_module,
        input_lower=input_lower,
        input_upper=input_upper,
        compile_receipt=receipt,
    )


__all__ = [
    "PreparedS1CIBCCUDAGraphV1",
    "PreparedS1CIBCProgramV1",
    "S1CIBCCompileReceiptV1",
    "S1CIBCExecutionReceiptV1",
    "prepare_s1_cibc_program_v1",
    "specialize_storage_plan_batch_v1",
]
