"""Auditable memory/disk cache for specialized fused CROWN CUDA modules."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,too-few-public-methods,too-many-locals,too-many-statements
# pylint: disable=unsubscriptable-object

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Literal, Union
import uuid

from .fused_crown_conv2d import (
    FUSED_CROWN_CONV2D_SCHEMA_VERSION,
    FusedCrownConv2dSignature,
    build_fused_crown_conv2d_primfunc,
    schedule_fused_crown_conv2d_primfunc,
)
from .fused_crown_linear import (
    FUSED_CROWN_LINEAR_SCHEMA_VERSION,
    FusedCrownLinearKey,
    build_fused_crown_linear_primfunc,
    schedule_fused_crown_linear_primfunc,
)

FUSED_CROWN_CACHE_SCHEMA_VERSION = "boundflow.fused_crown_cache/v2"
FusedCrownKind = Literal["linear", "conv2d"]
FusedCrownSignature = Union[FusedCrownLinearKey, FusedCrownConv2dSignature]


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1e6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature_payload(
    kind: FusedCrownKind,
    signature: FusedCrownSignature,
    *,
    backend_dispatch_key: str | None = None,
) -> dict[str, Any]:
    if kind == "linear" and not isinstance(signature, FusedCrownLinearKey):
        raise TypeError("linear cache entries require FusedCrownLinearKey")
    if kind == "conv2d" and not isinstance(signature, FusedCrownConv2dSignature):
        raise TypeError("conv2d cache entries require FusedCrownConv2dSignature")
    signature.validate()
    import tvm  # pylint: disable=import-outside-toplevel

    # Canonicalize tuples to JSON arrays before both hashing and manifest
    # comparison; otherwise Conv signatures compare tuple-in-memory vs list-on-disk.
    signature_json = json.loads(json.dumps(asdict(signature), sort_keys=True))
    if backend_dispatch_key is not None and len(backend_dispatch_key) != 64:
        raise ValueError("fused cache backend dispatch key is not SHA-256")
    return {
        "schema_version": FUSED_CROWN_CACHE_SCHEMA_VERSION,
        "kind": kind,
        "signature": signature_json,
        "code_schema": (
            FUSED_CROWN_LINEAR_SCHEMA_VERSION
            if kind == "linear"
            else FUSED_CROWN_CONV2D_SCHEMA_VERSION
        ),
        "target": signature.target_string,
        "tvm_version": str(tvm.__version__),
        "backend_dispatch_key": backend_dispatch_key,
    }


def fused_crown_cache_key(
    kind: FusedCrownKind,
    signature: FusedCrownSignature,
    *,
    backend_dispatch_key: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return a stable digest covering code schema, signature, target and TVM ABI."""

    payload = _signature_payload(
        kind,
        signature,
        backend_dispatch_key=backend_dispatch_key,
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


@dataclass(frozen=True)
class FusedCrownCacheEvent:  # pylint: disable=too-many-instance-attributes
    """One cache lookup with disjoint compile/load phase timings."""

    event: Literal["miss", "disk_hit", "memory_hit"]
    cache_key: str
    cache_lookup_ms: float
    tir_generation_ms: float
    schedule_ms: float
    tvm_compile_ms: float
    serialization_ms: float
    module_load_ms: float
    total_ms: float
    library_bytes: int
    library_sha256: str
    process_id: int

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON payload used by PR-12J artifacts."""

        return asdict(self)


class FusedCrownModuleCache:
    """Process-local packed-function cache backed by validated shared libraries."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[
            tuple[str | None, FusedCrownKind, FusedCrownSignature],
            tuple[str, Any, int, str],
        ] = {}
        self.events: list[FusedCrownCacheEvent] = []

    def _paths(self, digest: str) -> tuple[Path, Path]:
        return (
            self.cache_dir / f"fused_crown_{digest}.so",
            self.cache_dir / f"fused_crown_{digest}.json",
        )

    @staticmethod
    def _build_tir(kind: FusedCrownKind, signature: FusedCrownSignature):
        if kind == "linear":
            assert isinstance(signature, FusedCrownLinearKey)
            return build_fused_crown_linear_primfunc(signature)
        assert isinstance(signature, FusedCrownConv2dSignature)
        return build_fused_crown_conv2d_primfunc(signature)

    @staticmethod
    def _schedule(kind: FusedCrownKind, primfunc):
        if kind == "linear":
            return schedule_fused_crown_linear_primfunc(primfunc)
        return schedule_fused_crown_conv2d_primfunc(primfunc)

    @staticmethod
    def _valid_disk_entry(
        library_path: Path,
        manifest_path: Path,
        *,
        digest: str,
        payload: dict[str, Any],
    ) -> tuple[bool, str]:
        if not library_path.is_file() or not manifest_path.is_file():
            return False, ""
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            library_sha = _sha256(library_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False, ""
        valid = (
            manifest.get("schema_version") == FUSED_CROWN_CACHE_SCHEMA_VERSION
            and manifest.get("cache_key") == digest
            and manifest.get("cache_payload") == payload
            and manifest.get("library_sha256") == library_sha
        )
        return bool(valid), library_sha if valid else ""

    def get(
        self,
        kind: FusedCrownKind,
        signature: FusedCrownSignature,
        *,
        backend_dispatch_key: str | None = None,
    ) -> tuple[Any, FusedCrownCacheEvent]:
        """Return the packed ``main`` function and record miss/disk/memory phases."""

        total_started = time.perf_counter_ns()
        lookup_started = time.perf_counter_ns()
        memory_key = (backend_dispatch_key, kind, signature)
        if memory_key in self._memory:
            digest, packed, library_bytes, library_sha = self._memory[memory_key]
            event = FusedCrownCacheEvent(
                event="memory_hit",
                cache_key=digest,
                cache_lookup_ms=_elapsed_ms(lookup_started),
                tir_generation_ms=0.0,
                schedule_ms=0.0,
                tvm_compile_ms=0.0,
                serialization_ms=0.0,
                module_load_ms=0.0,
                total_ms=_elapsed_ms(total_started),
                library_bytes=library_bytes,
                library_sha256=library_sha,
                process_id=os.getpid(),
            )
            self.events.append(event)
            return packed, event

        digest, payload = fused_crown_cache_key(
            kind,
            signature,
            backend_dispatch_key=backend_dispatch_key,
        )
        library_path, manifest_path = self._paths(digest)
        valid, library_sha = self._valid_disk_entry(
            library_path, manifest_path, digest=digest, payload=payload
        )
        lookup_ms = _elapsed_ms(lookup_started)
        if valid:
            import tvm  # pylint: disable=import-outside-toplevel

            load_started = time.perf_counter_ns()
            module = tvm.runtime.load_module(str(library_path))
            module_load_ms = _elapsed_ms(load_started)
            packed = module["main"]
            library_bytes = library_path.stat().st_size
            self._memory[memory_key] = (
                digest,
                packed,
                library_bytes,
                library_sha,
            )
            event = FusedCrownCacheEvent(
                event="disk_hit",
                cache_key=digest,
                cache_lookup_ms=lookup_ms,
                tir_generation_ms=0.0,
                schedule_ms=0.0,
                tvm_compile_ms=0.0,
                serialization_ms=0.0,
                module_load_ms=module_load_ms,
                total_ms=_elapsed_ms(total_started),
                library_bytes=library_bytes,
                library_sha256=library_sha,
                process_id=os.getpid(),
            )
            self.events.append(event)
            return packed, event

        import tvm  # pylint: disable=import-outside-toplevel

        tir_started = time.perf_counter_ns()
        primfunc = self._build_tir(kind, signature)
        tir_ms = _elapsed_ms(tir_started)
        schedule_started = time.perf_counter_ns()
        scheduled = self._schedule(kind, primfunc)
        schedule_ms = _elapsed_ms(schedule_started)
        compile_started = time.perf_counter_ns()
        module = tvm.compile(scheduled, target=signature.target_string)
        compile_ms = _elapsed_ms(compile_started)
        serialize_started = time.perf_counter_ns()
        token = uuid.uuid4().hex
        temporary_library = self.cache_dir / f".{digest}.{token}.tmp.so"
        temporary_manifest = self.cache_dir / f".{digest}.{token}.tmp.json"
        try:
            module.export_library(str(temporary_library))
            library_sha = _sha256(temporary_library)
            manifest = {
                "schema_version": FUSED_CROWN_CACHE_SCHEMA_VERSION,
                "cache_key": digest,
                "cache_payload": payload,
                "library_sha256": library_sha,
                "library_bytes": temporary_library.stat().st_size,
            }
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_library.replace(library_path)
            temporary_manifest.replace(manifest_path)
        finally:
            temporary_library.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)
        serialization_ms = _elapsed_ms(serialize_started)
        packed = module["main"]
        library_bytes = library_path.stat().st_size
        self._memory[memory_key] = (digest, packed, library_bytes, library_sha)
        event = FusedCrownCacheEvent(
            event="miss",
            cache_key=digest,
            cache_lookup_ms=lookup_ms,
            tir_generation_ms=tir_ms,
            schedule_ms=schedule_ms,
            tvm_compile_ms=compile_ms,
            serialization_ms=serialization_ms,
            module_load_ms=0.0,
            total_ms=_elapsed_ms(total_started),
            library_bytes=library_bytes,
            library_sha256=library_sha,
            process_id=os.getpid(),
        )
        self.events.append(event)
        return packed, event


__all__ = [
    "FUSED_CROWN_CACHE_SCHEMA_VERSION",
    "FusedCrownCacheEvent",
    "FusedCrownKind",
    "FusedCrownModuleCache",
    "fused_crown_cache_key",
]
