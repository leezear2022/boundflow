"""Isolated S4-1B0 ternary input-endpoint CUDA/TIR lowering.

This module deliberately owns only the frozen backend lowering, immutable
compiled identity, cache, and an isolated prepared correctness probe.  It does
not bind the S4 evaluator, optimizer, mutable-buffer ticket, timing, or a
performance claim.
"""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,import-outside-toplevel,too-many-instance-attributes
# pylint: disable=too-many-arguments,too-many-locals,too-many-boolean-expressions
# pylint: disable=too-few-public-methods,protected-access,missing-function-docstring
# pylint: disable=too-many-positional-arguments,not-callable
# pylint: disable=too-many-lines

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import struct
from typing import Any, Callable, NoReturn, Protocol, cast

S4_TERNARY_ENDPOINT_SCHEMA = "boundflow.asplos27-s4-ternary-endpoint/v1"
S4_TERNARY_ENDPOINT_MODULE_RECEIPT_SCHEMA = (
    "boundflow.asplos27-s4-ternary-endpoint-module-receipt/v1"
)
S4_TERNARY_ENDPOINT_PACK_SYMBOL = "boundflow_s4_pack_ainput_endpoint_ternary"
S4_TERNARY_ENDPOINT_SELECT_SYMBOL = "boundflow_s4_select_input_endpoint_ternary"
S4_TERNARY_ENDPOINT_EXPORTED_SYMBOLS = (
    S4_TERNARY_ENDPOINT_PACK_SYMBOL,
    S4_TERNARY_ENDPOINT_SELECT_SYMBOL,
)
S4_TERNARY_ENDPOINT_DEFAULT_THREADS = 256
S4_TERNARY_ENDPOINT_QNAN_BITS = 0x7FC00000
S4_TERNARY_ENDPOINT_NONFINITE_MASK = 0x7F800000

S4_TERNARY_ENDPOINT_POLICY = "ternary-box-endpoint-v1"
S4_TERNARY_ENDPOINT_MIDPOINT_POLICY = "add-then-mul-f32-half-v1"
S4_TERNARY_ENDPOINT_NONFINITE_POLICY = "ieee754-f32-exponent-bits-sentinel-minus-128-v1"
S4_TERNARY_ENDPOINT_INVALID_OUTPUT_POLICY = "canonical-qnan-0x7fc00000-v1"

TVM_COMMIT = "6248b5db43505fbcfb13cc289d11877d5d2649e8"
TVM_FFI_COMMIT = "438f6439148b059d424ce2cc2a348736923f6948"

TERNARY_ENDPOINT_STABLE_REASONS = (
    "TERNARY_ENDPOINT_SCHEMA_MISMATCH",
    "TERNARY_ENDPOINT_POLICY_MISMATCH",
    "TERNARY_ENDPOINT_MIDPOINT_POLICY_MISMATCH",
    "TERNARY_ENDPOINT_NONFINITE_POLICY_MISMATCH",
    "TERNARY_ENDPOINT_SYMBOL_COLLISION",
    "TERNARY_ENDPOINT_TIR_IDENTITY_MISMATCH",
    "TERNARY_ENDPOINT_DEVICE_SOURCE_MISMATCH",
    "TERNARY_ENDPOINT_CACHE_KEY_MISMATCH",
    "TERNARY_ENDPOINT_CACHE_ENTRY_POISONED",
    "TERNARY_ENDPOINT_LEGACY_MODULE_COLLISION",
    "TERNARY_ENDPOINT_SHAPE_MISMATCH",
    "TERNARY_ENDPOINT_DTYPE_MISMATCH",
    "TERNARY_ENDPOINT_DEVICE_MISMATCH",
    "TERNARY_ENDPOINT_LAYOUT_MISMATCH",
    "TERNARY_ENDPOINT_ALIAS_MISMATCH",
    "TERNARY_ENDPOINT_DLPACK_IDENTITY_MISMATCH",
    "TERNARY_ENDPOINT_STREAM_IDENTITY_MISMATCH",
    "TERNARY_ENDPOINT_LAUNCH_COUNT_MISMATCH",
    "TERNARY_ENDPOINT_INVALID_SELECTOR_NOT_POISONED",
    "TERNARY_ENDPOINT_CLAIM_FLAG_MISMATCH",
)

_LEGACY_SYMBOLS = (
    "boundflow_r31b2_pack_ainput_sign",
    "boundflow_s2_select_input_tir",
)


class TernaryEndpointError(RuntimeError):
    """Fail-closed S4-1B0 rejection carrying one stable reason."""

    def __init__(self, reason: str) -> None:
        if reason not in TERNARY_ENDPOINT_STABLE_REASONS:
            raise ValueError("unknown ternary endpoint rejection reason")
        self.reason = reason
        super().__init__(reason)


def _reject(reason: str) -> NoReturn:
    raise TernaryEndpointError(reason)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _float32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _float32_bits(value: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    except OverflowError:
        return 0xFF800000 if math.copysign(1.0, value) < 0 else 0x7F800000


def _round_float32(value: float) -> float:
    return _float32_from_bits(_float32_bits(value))


def ternary_pack_bit_oracle_v1(bits: int) -> int:
    """Return the frozen selector for one raw IEEE-754 float32 bit pattern."""

    raw = bits & 0xFFFFFFFF
    if raw & S4_TERNARY_ENDPOINT_NONFINITE_MASK == S4_TERNARY_ENDPOINT_NONFINITE_MASK:
        return -128
    if raw & 0x7FFFFFFF == 0:
        return 0
    return -1 if raw & 0x80000000 else 1


def ternary_select_bit_oracle_v1(
    selector: int, lower_bits: int, upper_bits: int
) -> int:
    """Return one frozen selected endpoint as raw float32 bits."""

    if selector == 1:
        return lower_bits & 0xFFFFFFFF
    if selector == -1:
        return upper_bits & 0xFFFFFFFF
    if selector == 0:
        lower = _float32_from_bits(lower_bits)
        upper = _float32_from_bits(upper_bits)
        midpoint = _round_float32(_round_float32(lower + upper) * 0.5)
        return _float32_bits(midpoint)
    return S4_TERNARY_ENDPOINT_QNAN_BITS


@dataclass(frozen=True)
class TernaryEndpointBuildSpecV1:
    """Backend-local compile metadata for one generic element count."""

    schema_version: str = S4_TERNARY_ENDPOINT_SCHEMA
    numel: int = 1
    value_dtype: str = "float32"
    selector_dtype: str = "int8"
    pack_symbol: str = S4_TERNARY_ENDPOINT_PACK_SYMBOL
    select_symbol: str = S4_TERNARY_ENDPOINT_SELECT_SYMBOL
    endpoint_policy: str = S4_TERNARY_ENDPOINT_POLICY
    midpoint_policy: str = S4_TERNARY_ENDPOINT_MIDPOINT_POLICY
    nonfinite_policy: str = S4_TERNARY_ENDPOINT_NONFINITE_POLICY
    invalid_output_policy: str = S4_TERNARY_ENDPOINT_INVALID_OUTPUT_POLICY
    target: str = "cuda"
    compute_capability: str = "sm_89"

    def validate(self) -> None:
        if self.schema_version != S4_TERNARY_ENDPOINT_SCHEMA:
            _reject("TERNARY_ENDPOINT_SCHEMA_MISMATCH")
        if (
            self.endpoint_policy != S4_TERNARY_ENDPOINT_POLICY
            or self.invalid_output_policy != S4_TERNARY_ENDPOINT_INVALID_OUTPUT_POLICY
        ):
            _reject("TERNARY_ENDPOINT_POLICY_MISMATCH")
        if self.midpoint_policy != S4_TERNARY_ENDPOINT_MIDPOINT_POLICY:
            _reject("TERNARY_ENDPOINT_MIDPOINT_POLICY_MISMATCH")
        if self.nonfinite_policy != S4_TERNARY_ENDPOINT_NONFINITE_POLICY:
            _reject("TERNARY_ENDPOINT_NONFINITE_POLICY_MISMATCH")
        if (
            self.numel <= 0
            or self.value_dtype != "float32"
            or self.selector_dtype != "int8"
            or self.target != "cuda"
            or not self.compute_capability.startswith("sm_")
        ):
            _reject("TERNARY_ENDPOINT_POLICY_MISMATCH")
        if self.pack_symbol in _LEGACY_SYMBOLS or self.select_symbol in _LEGACY_SYMBOLS:
            _reject("TERNARY_ENDPOINT_LEGACY_MODULE_COLLISION")
        if (
            self.pack_symbol != S4_TERNARY_ENDPOINT_PACK_SYMBOL
            or self.select_symbol != S4_TERNARY_ENDPOINT_SELECT_SYMBOL
            or self.pack_symbol == self.select_symbol
        ):
            _reject("TERNARY_ENDPOINT_SYMBOL_COLLISION")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return cast(dict[str, object], asdict(self))

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())

    @property
    def target_string(self) -> str:
        self.validate()
        return f"{self.target} -arch={self.compute_capability}"


@dataclass(frozen=True)
class TernaryEndpointScheduleSpecV1:
    """Frozen two-kernel elementwise CUDA schedule."""

    threads_per_block: int = S4_TERNARY_ENDPOINT_DEFAULT_THREADS
    pack_block: str = "endpoint_selector"
    select_block: str = "selected_endpoint"

    def validate(self) -> None:
        if (
            self.threads_per_block not in {64, 128, 256, 512, 1024}
            or self.pack_block != "endpoint_selector"
            or self.select_block != "selected_endpoint"
        ):
            _reject("TERNARY_ENDPOINT_TIR_IDENTITY_MISMATCH")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return cast(dict[str, object], asdict(self))

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


class TernaryEndpointExecutable(Protocol):
    """Minimum callable surface exposed by a compiled TVM module."""

    def __getitem__(self, symbol: str) -> Callable[..., object]: ...


def _build_pack_primfunc(spec: TernaryEndpointBuildSpecV1) -> Any:
    import tvm
    from tvm import te

    spec.validate()
    coefficient = te.placeholder((spec.numel,), "float32", name="coefficient")
    zero = tvm.tir.const(0.0, "float32")
    mask = tvm.tir.const(S4_TERNARY_ENDPOINT_NONFINITE_MASK, "uint32")

    def classify(index: Any) -> Any:
        bits = tvm.tir.reinterpret("uint32", coefficient[index])
        nonfinite = tvm.tir.bitwise_and(bits, mask) == mask
        return tvm.tir.if_then_else(
            nonfinite,
            tvm.tir.const(-128, "int8"),
            tvm.tir.if_then_else(
                coefficient[index] > zero,
                tvm.tir.const(1, "int8"),
                tvm.tir.if_then_else(
                    coefficient[index] < zero,
                    tvm.tir.const(-1, "int8"),
                    tvm.tir.const(0, "int8"),
                ),
            ),
        )

    selector = te.compute((spec.numel,), classify, name="endpoint_selector")
    return (
        te.create_prim_func([coefficient, selector])
        .with_attr("global_symbol", spec.pack_symbol)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", spec.schema_version)
        .with_attr("boundflow.endpoint_policy", spec.endpoint_policy)
        .with_attr("boundflow.midpoint_policy", spec.midpoint_policy)
        .with_attr("boundflow.nonfinite_policy", spec.nonfinite_policy)
        .with_attr("boundflow.invalid_output_policy", spec.invalid_output_policy)
        .with_attr("boundflow.numel", spec.numel)
    )


def _build_select_primfunc(spec: TernaryEndpointBuildSpecV1) -> Any:
    import tvm
    from tvm import te

    spec.validate()
    lower = te.placeholder((spec.numel,), "float32", name="lower")
    upper = te.placeholder((spec.numel,), "float32", name="upper")
    selector = te.placeholder((spec.numel,), "int8", name="selector")
    half = tvm.tir.const(0.5, "float32")
    qnan = tvm.tir.reinterpret(
        "float32", tvm.tir.const(S4_TERNARY_ENDPOINT_QNAN_BITS, "uint32")
    )

    def select(index: Any) -> Any:
        return tvm.tir.if_then_else(
            selector[index] == tvm.tir.const(1, "int8"),
            lower[index],
            tvm.tir.if_then_else(
                selector[index] == tvm.tir.const(-1, "int8"),
                upper[index],
                tvm.tir.if_then_else(
                    selector[index] == tvm.tir.const(0, "int8"),
                    (lower[index] + upper[index]) * half,
                    qnan,
                ),
            ),
        )

    selected = te.compute((spec.numel,), select, name="selected_endpoint")
    return (
        te.create_prim_func([lower, upper, selector, selected])
        .with_attr("global_symbol", spec.select_symbol)
        .with_attr("tir.noalias", True)
        .with_attr("boundflow.schema_version", spec.schema_version)
        .with_attr("boundflow.endpoint_policy", spec.endpoint_policy)
        .with_attr("boundflow.midpoint_policy", spec.midpoint_policy)
        .with_attr("boundflow.nonfinite_policy", spec.nonfinite_policy)
        .with_attr("boundflow.invalid_output_policy", spec.invalid_output_policy)
        .with_attr("boundflow.numel", spec.numel)
    )


def _schedule_elementwise(
    module: Any,
    symbol: str,
    block_name: str,
    schedule_spec: TernaryEndpointScheduleSpecV1,
) -> Any:
    import tvm

    schedule_spec.validate()
    schedule = tvm.tir.Schedule(tvm.IRModule({symbol: module[symbol]}))
    block = schedule.get_block(block_name, func_name=symbol)
    loops = schedule.get_loops(block)
    fused = schedule.fuse(*loops) if len(loops) > 1 else loops[0]
    outer, inner = schedule.split(
        fused, factors=[None, schedule_spec.threads_per_block]
    )
    schedule.bind(outer, "blockIdx.x")
    schedule.bind(inner, "threadIdx.x")
    return schedule.mod[symbol]


def build_ternary_endpoint_modules_v1(
    spec: TernaryEndpointBuildSpecV1,
    schedule: TernaryEndpointScheduleSpecV1,
) -> tuple[Any, Any]:
    """Build unscheduled and scheduled two-symbol IRModules."""

    import tvm

    validate_ternary_endpoint_construction_model_v1()
    spec.validate()
    schedule.validate()
    unscheduled = tvm.IRModule(
        {
            spec.pack_symbol: _build_pack_primfunc(spec),
            spec.select_symbol: _build_select_primfunc(spec),
        }
    )
    scheduled = tvm.IRModule(
        {
            spec.pack_symbol: _schedule_elementwise(
                unscheduled, spec.pack_symbol, schedule.pack_block, schedule
            ),
            spec.select_symbol: _schedule_elementwise(
                unscheduled, spec.select_symbol, schedule.select_block, schedule
            ),
        }
    )
    return unscheduled, scheduled


@dataclass(frozen=True)
class CompiledTernaryEndpointV1:
    """Compiled code plus the exact IR and device-source content."""

    executable: TernaryEndpointExecutable
    unscheduled_tir_json: str
    scheduled_tir_json: str
    device_source: str
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    exported_symbols: tuple[str, ...]
    global_workspace_bytes: int
    tvm_version: str

    def validate_content(self) -> None:
        if (
            _sha256_text(self.unscheduled_tir_json) != self.unscheduled_tir_hash
            or _sha256_text(self.scheduled_tir_json) != self.scheduled_tir_hash
        ):
            _reject("TERNARY_ENDPOINT_TIR_IDENTITY_MISMATCH")
        if _sha256_text(self.device_source) != self.device_source_hash:
            _reject("TERNARY_ENDPOINT_DEVICE_SOURCE_MISMATCH")
        if self.exported_symbols != S4_TERNARY_ENDPOINT_EXPORTED_SYMBOLS:
            _reject("TERNARY_ENDPOINT_SYMBOL_COLLISION")
        if any(symbol in self.device_source for symbol in _LEGACY_SYMBOLS):
            _reject("TERNARY_ENDPOINT_LEGACY_MODULE_COLLISION")
        if any(symbol not in self.device_source for symbol in self.exported_symbols):
            _reject("TERNARY_ENDPOINT_DEVICE_SOURCE_MISMATCH")
        if self.global_workspace_bytes != 0 or not self.tvm_version:
            _reject("TERNARY_ENDPOINT_TIR_IDENTITY_MISMATCH")


def compile_ternary_endpoint_v1(
    spec: TernaryEndpointBuildSpecV1,
    schedule: TernaryEndpointScheduleSpecV1,
    *,
    modules: tuple[Any, Any] | None = None,
) -> CompiledTernaryEndpointV1:
    """Compile the exact two-symbol TIR module without fallback."""

    import tvm

    spec.validate()
    schedule.validate()
    unscheduled, scheduled = modules or build_ternary_endpoint_modules_v1(
        spec, schedule
    )
    unscheduled_json = tvm.ir.save_json(unscheduled)
    scheduled_json = tvm.ir.save_json(scheduled)
    executable = tvm.compile(scheduled, target=spec.target_string)
    sources = tuple(module.inspect_source() for module in executable.mod.imports)
    if not sources:
        _reject("TERNARY_ENDPOINT_DEVICE_SOURCE_MISMATCH")
    device_source = "\n".join(sources)
    compiled = CompiledTernaryEndpointV1(
        executable=cast(TernaryEndpointExecutable, executable),
        unscheduled_tir_json=unscheduled_json,
        scheduled_tir_json=scheduled_json,
        device_source=device_source,
        unscheduled_tir_hash=_sha256_text(unscheduled_json),
        scheduled_tir_hash=_sha256_text(scheduled_json),
        device_source_hash=_sha256_text(device_source),
        exported_symbols=S4_TERNARY_ENDPOINT_EXPORTED_SYMBOLS,
        global_workspace_bytes=0,
        tvm_version=str(tvm.__version__),
    )
    compiled.validate_content()
    return compiled


@dataclass(frozen=True)
class TernaryEndpointModuleReceiptV1:
    """Immutable compiled identity; mutable cache counts are intentionally absent."""

    build_spec_hash: str
    schedule_spec_hash: str
    unscheduled_tir_hash: str
    scheduled_tir_hash: str
    device_source_hash: str
    cache_key: str
    target: str
    compute_capability: str
    tvm_version: str
    tvm_commit: str
    tvm_ffi_commit: str
    torch_version: str
    exported_symbols: tuple[str, ...]
    global_workspace_bytes: int = 0
    performance_claimed: bool = False

    @staticmethod
    def expected_cache_key(
        spec: TernaryEndpointBuildSpecV1,
        schedule: TernaryEndpointScheduleSpecV1,
        unscheduled_tir_hash: str,
        scheduled_tir_hash: str,
    ) -> str:
        return _canonical_hash(
            {
                "module_receipt_schema": S4_TERNARY_ENDPOINT_MODULE_RECEIPT_SCHEMA,
                "build_spec_hash": spec.stable_hash(),
                "schedule_spec_hash": schedule.stable_hash(),
                "unscheduled_tir_hash": unscheduled_tir_hash,
                "scheduled_tir_hash": scheduled_tir_hash,
                "target": spec.target,
                "compute_capability": spec.compute_capability,
                "tvm_commit": TVM_COMMIT,
                "tvm_ffi_commit": TVM_FFI_COMMIT,
            }
        )

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["exported_symbols"] = list(self.exported_symbols)
        return cast(dict[str, object], value)

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())

    def validate_against(
        self,
        spec: TernaryEndpointBuildSpecV1,
        schedule: TernaryEndpointScheduleSpecV1,
        compiled: CompiledTernaryEndpointV1,
        *,
        blueprint_hashes: tuple[str, str] | None = None,
    ) -> None:
        spec.validate()
        schedule.validate()
        compiled.validate_content()
        if self.performance_claimed:
            _reject("TERNARY_ENDPOINT_CLAIM_FLAG_MISMATCH")
        if (
            blueprint_hashes is not None
            and (
                compiled.unscheduled_tir_hash,
                compiled.scheduled_tir_hash,
            )
            != blueprint_hashes
        ):
            _reject("TERNARY_ENDPOINT_TIR_IDENTITY_MISMATCH")
        expected_key = self.expected_cache_key(
            spec,
            schedule,
            compiled.unscheduled_tir_hash,
            compiled.scheduled_tir_hash,
        )
        if self.cache_key != expected_key:
            _reject("TERNARY_ENDPOINT_CACHE_KEY_MISMATCH")
        expected = (
            spec.stable_hash(),
            schedule.stable_hash(),
            compiled.unscheduled_tir_hash,
            compiled.scheduled_tir_hash,
            compiled.device_source_hash,
            spec.target,
            spec.compute_capability,
            compiled.tvm_version,
            TVM_COMMIT,
            TVM_FFI_COMMIT,
            compiled.exported_symbols,
            compiled.global_workspace_bytes,
        )
        actual = (
            self.build_spec_hash,
            self.schedule_spec_hash,
            self.unscheduled_tir_hash,
            self.scheduled_tir_hash,
            self.device_source_hash,
            self.target,
            self.compute_capability,
            self.tvm_version,
            self.tvm_commit,
            self.tvm_ffi_commit,
            self.exported_symbols,
            self.global_workspace_bytes,
        )
        if actual != expected or not self.torch_version:
            _reject("TERNARY_ENDPOINT_CACHE_ENTRY_POISONED")


@dataclass(frozen=True)
class TernaryEndpointCacheObservationV1:
    """Mutable cache access counters separated from compiled identity."""

    event: str
    compile_count: int
    miss_count: int
    hit_count: int
    entry_count: int
    module_receipt_hash: str

    def validate(self) -> None:
        if (
            self.event not in {"miss", "hit"}
            or self.compile_count < 1
            or self.miss_count < 1
            or self.hit_count < 0
            or self.entry_count < 1
            or not _is_digest(self.module_receipt_hash)
        ):
            _reject("TERNARY_ENDPOINT_CACHE_ENTRY_POISONED")


def _blueprint(
    spec: TernaryEndpointBuildSpecV1,
    schedule: TernaryEndpointScheduleSpecV1,
) -> tuple[tuple[Any, Any], tuple[str, str]]:
    import tvm

    modules = build_ternary_endpoint_modules_v1(spec, schedule)
    return modules, (
        _sha256_text(tvm.ir.save_json(modules[0])),
        _sha256_text(tvm.ir.save_json(modules[1])),
    )


class TernaryEndpointModuleCacheV1:
    """Fail-closed in-process cache for immutable compiled endpoint modules."""

    def __init__(self) -> None:
        self._entries: dict[
            str, tuple[CompiledTernaryEndpointV1, TernaryEndpointModuleReceiptV1]
        ] = {}
        self.compile_count = 0
        self.miss_count = 0
        self.hit_count = 0

    def get(
        self,
        spec: TernaryEndpointBuildSpecV1,
        schedule: TernaryEndpointScheduleSpecV1,
    ) -> tuple[
        CompiledTernaryEndpointV1,
        TernaryEndpointModuleReceiptV1,
        TernaryEndpointCacheObservationV1,
    ]:
        modules, hashes = _blueprint(spec, schedule)
        cache_key = TernaryEndpointModuleReceiptV1.expected_cache_key(
            spec, schedule, *hashes
        )
        existing = self._entries.get(cache_key)
        if existing is not None:
            compiled, receipt = existing
            try:
                receipt.validate_against(
                    spec, schedule, compiled, blueprint_hashes=hashes
                )
            except TernaryEndpointError as error:
                if error.reason in {
                    "TERNARY_ENDPOINT_DEVICE_SOURCE_MISMATCH",
                    "TERNARY_ENDPOINT_TIR_IDENTITY_MISMATCH",
                }:
                    raise
                _reject("TERNARY_ENDPOINT_CACHE_ENTRY_POISONED")
            self.hit_count += 1
            observation = self._observation("hit", receipt)
            observation.validate()
            return compiled, receipt, observation

        self.miss_count += 1
        compiled = compile_ternary_endpoint_v1(spec, schedule, modules=modules)
        import torch

        receipt = TernaryEndpointModuleReceiptV1(
            build_spec_hash=spec.stable_hash(),
            schedule_spec_hash=schedule.stable_hash(),
            unscheduled_tir_hash=compiled.unscheduled_tir_hash,
            scheduled_tir_hash=compiled.scheduled_tir_hash,
            device_source_hash=compiled.device_source_hash,
            cache_key=cache_key,
            target=spec.target,
            compute_capability=spec.compute_capability,
            tvm_version=compiled.tvm_version,
            tvm_commit=TVM_COMMIT,
            tvm_ffi_commit=TVM_FFI_COMMIT,
            torch_version=str(torch.__version__),
            exported_symbols=compiled.exported_symbols,
        )
        receipt.validate_against(spec, schedule, compiled, blueprint_hashes=hashes)
        self.compile_count += 1
        self._entries[cache_key] = (compiled, receipt)
        observation = self._observation("miss", receipt)
        observation.validate()
        return compiled, receipt, observation

    def _observation(
        self, event: str, receipt: TernaryEndpointModuleReceiptV1
    ) -> TernaryEndpointCacheObservationV1:
        return TernaryEndpointCacheObservationV1(
            event=event,
            compile_count=self.compile_count,
            miss_count=self.miss_count,
            hit_count=self.hit_count,
            entry_count=len(self._entries),
            module_receipt_hash=receipt.stable_hash(),
        )


@dataclass(frozen=True)
class TernaryEndpointTensorDescriptorV1:
    """Tensor-free identity bound at prepare time."""

    role: str
    storage_token: tuple[str, int, int]
    storage_offset: int
    data_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["storage_token"] = list(self.storage_token)
        value["shape"] = list(self.shape)
        value["stride"] = list(self.stride)
        return cast(dict[str, object], value)

    def stable_hash(self) -> str:
        return _canonical_hash(self.to_dict())


def _tensor_descriptor(role: str, tensor: Any) -> TernaryEndpointTensorDescriptorV1:
    storage = tensor.untyped_storage()
    return TernaryEndpointTensorDescriptorV1(
        role=role,
        storage_token=(
            str(tensor.device),
            int(storage.data_ptr()),
            int(storage.nbytes()),
        ),
        storage_offset=int(tensor.storage_offset()),
        data_ptr=int(tensor.data_ptr()),
        shape=tuple(int(value) for value in tensor.shape),
        stride=tuple(int(value) for value in tensor.stride()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
    )


def _validate_tensor(role: str, tensor: Any, spec: TernaryEndpointBuildSpecV1) -> None:
    import torch

    if not torch.is_tensor(tensor) or tuple(tensor.shape) != (spec.numel,):
        _reject("TERNARY_ENDPOINT_SHAPE_MISMATCH")
    expected_dtype = torch.int8 if role == "selector" else torch.float32
    if tensor.dtype != expected_dtype:
        _reject("TERNARY_ENDPOINT_DTYPE_MISMATCH")
    if tensor.device.type != "cuda":
        _reject("TERNARY_ENDPOINT_DEVICE_MISMATCH")
    if not tensor.is_contiguous() or tuple(tensor.stride()) != (1,):
        _reject("TERNARY_ENDPOINT_LAYOUT_MISMATCH")


def _create_dlpack_view(tensor: Any) -> Any:
    import torch
    import tvm

    view = tvm.runtime.from_dlpack(tensor)
    roundtrip = torch.from_dlpack(view)
    if (
        roundtrip.data_ptr() != tensor.data_ptr()
        or tuple(roundtrip.shape) != tuple(tensor.shape)
        or tuple(roundtrip.stride()) != tuple(tensor.stride())
        or roundtrip.dtype != tensor.dtype
        or roundtrip.device != tensor.device
    ):
        _reject("TERNARY_ENDPOINT_DLPACK_IDENTITY_MISMATCH")
    return view


@dataclass(frozen=True)
class TernaryEndpointWarmLaunchReceiptV1:
    """Tensor-free O(1) warm launch evidence."""

    module_receipt_hash: str
    cache_event: str
    device_ordinal: int
    stream_identity: int
    evaluation_ordinal: int
    parameter_state_version: int
    selector_generation: int
    prepared_descriptor_hashes: tuple[str, ...]
    pack_launch_count: int = 1
    select_launch_count: int = 1
    argument_occurrence_count: int = 6
    warm_dlpack_view_creation_count: int = 0
    fallback_count: int = 0
    eager_count: int = 0
    native_shadow_count: int = 0
    timing_recorded: bool = False
    performance_claimed: bool = False

    def validate(self) -> None:
        if self.performance_claimed or self.timing_recorded:
            _reject("TERNARY_ENDPOINT_CLAIM_FLAG_MISMATCH")
        if (
            not _is_digest(self.module_receipt_hash)
            or self.cache_event not in {"miss", "hit"}
            or self.device_ordinal < 0
            or self.stream_identity < 0
            or self.evaluation_ordinal < 0
            or self.parameter_state_version < 0
            or self.selector_generation < 0
            or len(self.prepared_descriptor_hashes) != 5
            or not all(_is_digest(value) for value in self.prepared_descriptor_hashes)
        ):
            _reject("TERNARY_ENDPOINT_LAUNCH_COUNT_MISMATCH")
        if (
            self.pack_launch_count,
            self.select_launch_count,
            self.argument_occurrence_count,
            self.warm_dlpack_view_creation_count,
            self.fallback_count,
            self.eager_count,
            self.native_shadow_count,
        ) != (1, 1, 6, 0, 0, 0, 0):
            _reject("TERNARY_ENDPOINT_LAUNCH_COUNT_MISMATCH")


@dataclass(frozen=True)
class PreparedTernaryEndpointProbeV1:
    """Isolated prepared micro-owner with five caller-owned DLPack views."""

    spec: TernaryEndpointBuildSpecV1
    schedule: TernaryEndpointScheduleSpecV1
    compiled: CompiledTernaryEndpointV1
    module_receipt: TernaryEndpointModuleReceiptV1
    cache_event: str
    descriptors: tuple[TernaryEndpointTensorDescriptorV1, ...]
    coefficient_view: Any
    lower_view: Any
    upper_view: Any
    selector_view: Any
    selected_view: Any
    device_ordinal: int

    @classmethod
    def prepare(
        cls,
        spec: TernaryEndpointBuildSpecV1,
        schedule: TernaryEndpointScheduleSpecV1,
        coefficient: Any,
        lower: Any,
        upper: Any,
        selector: Any,
        selected: Any,
        *,
        cache: TernaryEndpointModuleCacheV1 | None = None,
    ) -> "PreparedTernaryEndpointProbeV1":
        import torch

        tensors = (coefficient, lower, upper, selector, selected)
        roles = ("coefficient", "lower", "upper", "selector", "selected")
        for role, tensor in zip(roles, tensors):
            _validate_tensor(role, tensor, spec)
        devices = {str(tensor.device) for tensor in tensors}
        if len(devices) != 1:
            _reject("TERNARY_ENDPOINT_DEVICE_MISMATCH")
        descriptors = tuple(
            _tensor_descriptor(role, tensor) for role, tensor in zip(roles, tensors)
        )
        if len({row.storage_token for row in descriptors}) != 5:
            _reject("TERNARY_ENDPOINT_ALIAS_MISMATCH")
        ordinal = coefficient.device.index
        if ordinal is None:
            ordinal = torch.cuda.current_device()
        if torch.cuda.current_device() != ordinal:
            _reject("TERNARY_ENDPOINT_DEVICE_MISMATCH")
        major, minor = torch.cuda.get_device_capability(ordinal)
        if spec.compute_capability != f"sm_{major}{minor}":
            _reject("TERNARY_ENDPOINT_POLICY_MISMATCH")
        module_cache = cache or TernaryEndpointModuleCacheV1()
        compiled, receipt, observation = module_cache.get(spec, schedule)
        views = tuple(_create_dlpack_view(tensor) for tensor in tensors)
        return cls(
            spec=spec,
            schedule=schedule,
            compiled=compiled,
            module_receipt=receipt,
            cache_event=observation.event,
            descriptors=descriptors,
            coefficient_view=views[0],
            lower_view=views[1],
            upper_view=views[2],
            selector_view=views[3],
            selected_view=views[4],
            device_ordinal=int(ordinal),
        )

    def run_once(
        self,
        *,
        evaluation_ordinal: int,
        parameter_state_version: int,
        selector_generation: int,
    ) -> TernaryEndpointWarmLaunchReceiptV1:
        import torch
        import tvm_ffi

        if torch.cuda.current_device() != self.device_ordinal:
            _reject("TERNARY_ENDPOINT_DEVICE_MISMATCH")
        current = torch.cuda.current_stream(self.device_ordinal)
        with tvm_ffi.use_torch_stream(torch.cuda.stream(current)):
            raw_stream = int(
                tvm_ffi.get_raw_stream(tvm_ffi.device(f"cuda:{self.device_ordinal}"))
            )
            if raw_stream != int(current.cuda_stream):
                _reject("TERNARY_ENDPOINT_STREAM_IDENTITY_MISMATCH")
            self.compiled.executable[self.spec.pack_symbol](
                self.coefficient_view, self.selector_view
            )
            self.compiled.executable[self.spec.select_symbol](
                self.lower_view,
                self.upper_view,
                self.selector_view,
                self.selected_view,
            )
        receipt = TernaryEndpointWarmLaunchReceiptV1(
            module_receipt_hash=self.module_receipt.stable_hash(),
            cache_event=self.cache_event,
            device_ordinal=self.device_ordinal,
            stream_identity=raw_stream,
            evaluation_ordinal=evaluation_ordinal,
            parameter_state_version=parameter_state_version,
            selector_generation=selector_generation,
            prepared_descriptor_hashes=tuple(
                descriptor.stable_hash() for descriptor in self.descriptors
            ),
        )
        receipt.validate()
        return receipt


def validate_selected_output_after_sync_v1(selector: Any, selected: Any) -> None:
    """Formal-only poison check; callers must synchronize before invoking it."""

    import torch

    invalid = (selector != 1) & (selector != -1) & (selector != 0)
    if not bool(invalid.any().item()):
        return
    raw = selected.view(torch.int32)
    expected = torch.tensor(
        S4_TERNARY_ENDPOINT_QNAN_BITS,
        dtype=torch.int32,
        device=selected.device,
    )
    if not bool((raw[invalid] == expected).all().item()):
        _reject("TERNARY_ENDPOINT_INVALID_SELECTOR_NOT_POISONED")


S4_TERNARY_ENDPOINT_EXPECTED_CONSTRUCTION_HASH_V1 = (
    "5056d302aa27785ab8a22bd8f5665ebef0a4aba2ca22bc72ce28581144dbcc2a"
)


def ternary_endpoint_construction_model_v1() -> dict[str, object]:
    """Reconstruct the frozen construction model from executable code facts."""

    build_fields = {item.name for item in fields(TernaryEndpointBuildSpecV1)}
    schedule_fields = {item.name for item in fields(TernaryEndpointScheduleSpecV1)}
    receipt_fields = {item.name for item in fields(TernaryEndpointModuleReceiptV1)}
    warm_fields = {item.name for item in fields(TernaryEndpointWarmLaunchReceiptV1)}
    required_build = {
        "schema_version",
        "numel",
        "value_dtype",
        "selector_dtype",
        "pack_symbol",
        "select_symbol",
        "endpoint_policy",
        "midpoint_policy",
        "nonfinite_policy",
        "invalid_output_policy",
        "target",
        "compute_capability",
    }
    required_schedule = {"threads_per_block", "pack_block", "select_block"}
    required_receipt = {
        "build_spec_hash",
        "schedule_spec_hash",
        "unscheduled_tir_hash",
        "scheduled_tir_hash",
        "device_source_hash",
        "cache_key",
        "performance_claimed",
    }
    required_warm = {
        "module_receipt_hash",
        "cache_event",
        "prepared_descriptor_hashes",
        "warm_dlpack_view_creation_count",
        "timing_recorded",
        "performance_claimed",
    }
    if (
        build_fields != required_build
        or schedule_fields != required_schedule
        or not required_receipt.issubset(receipt_fields)
        or not required_warm.issubset(warm_fields)
    ):
        _reject("TERNARY_ENDPOINT_TIR_IDENTITY_MISMATCH")
    return {
        "backend_file": "boundflow/backends/tvm/asplos27_s4_ternary_endpoint.py",
        "cache": {
            "device_source_in_lookup_key": False,
            "hit_rehashes_cached_source": True,
            "mutable_counts_in_module_receipt": False,
            "precompile_tir_hashes_in_lookup_key": True,
        },
        "claims": {
            "implementation": False,
            "performance": False,
            "production_alias": False,
            "production_correctness": False,
        },
        "formal": {
            "cache_workers": 1,
            "fault_workers": 5,
            "positive_workers": 5,
            "status_requires_external_audit": True,
        },
        "math": {
            "invalid_output_bits": f"0x{S4_TERNARY_ENDPOINT_QNAN_BITS:08x}",
            "midpoint_policy": S4_TERNARY_ENDPOINT_MIDPOINT_POLICY,
            "nonfinite_mask": f"0x{S4_TERNARY_ENDPOINT_NONFINITE_MASK:08x}",
            "selector_values": [-128, -1, 0, 1],
        },
        "negative_reason_count": len(TERNARY_ENDPOINT_STABLE_REASONS),
        "production": {
            "selected_output_alias_requires_s4_1b_phase_proof": True,
            "warm_content_hash": False,
            "warm_count_sync": False,
        },
        "scope": {
            "backend_compile": True,
            "evaluator_binding": False,
            "new_ir": False,
            "prepared_probe": True,
            "timing": False,
        },
        "storage": {
            "isolated_dlpack_views": 5,
            "isolated_output_allocated_bytes": 92160,
            "selected_output_bytes": 73728,
            "selector_bytes": 18432,
            "s4_1a_base_view_overlap": 0,
        },
        "symbols": list(S4_TERNARY_ENDPOINT_EXPORTED_SYMBOLS),
        "test_file": "tests/test_asplos27_s4_ternary_endpoint.py",
        "threads": S4_TERNARY_ENDPOINT_DEFAULT_THREADS,
    }


def validate_ternary_endpoint_construction_model_v1() -> str:
    """Fail closed if executable facts drift from the frozen construction model."""

    observed = _canonical_hash(ternary_endpoint_construction_model_v1())
    if observed != S4_TERNARY_ENDPOINT_EXPECTED_CONSTRUCTION_HASH_V1:
        _reject("TERNARY_ENDPOINT_TIR_IDENTITY_MISMATCH")
    return observed


__all__ = [
    "CompiledTernaryEndpointV1",
    "PreparedTernaryEndpointProbeV1",
    "S4_TERNARY_ENDPOINT_DEFAULT_THREADS",
    "S4_TERNARY_ENDPOINT_EXPECTED_CONSTRUCTION_HASH_V1",
    "S4_TERNARY_ENDPOINT_EXPORTED_SYMBOLS",
    "S4_TERNARY_ENDPOINT_NONFINITE_MASK",
    "S4_TERNARY_ENDPOINT_PACK_SYMBOL",
    "S4_TERNARY_ENDPOINT_QNAN_BITS",
    "S4_TERNARY_ENDPOINT_SCHEMA",
    "S4_TERNARY_ENDPOINT_SELECT_SYMBOL",
    "TERNARY_ENDPOINT_STABLE_REASONS",
    "TVM_COMMIT",
    "TVM_FFI_COMMIT",
    "TernaryEndpointBuildSpecV1",
    "TernaryEndpointCacheObservationV1",
    "TernaryEndpointError",
    "TernaryEndpointModuleCacheV1",
    "TernaryEndpointModuleReceiptV1",
    "TernaryEndpointScheduleSpecV1",
    "TernaryEndpointTensorDescriptorV1",
    "TernaryEndpointWarmLaunchReceiptV1",
    "build_ternary_endpoint_modules_v1",
    "compile_ternary_endpoint_v1",
    "ternary_pack_bit_oracle_v1",
    "ternary_endpoint_construction_model_v1",
    "ternary_select_bit_oracle_v1",
    "validate_ternary_endpoint_construction_model_v1",
    "validate_selected_output_after_sync_v1",
]
