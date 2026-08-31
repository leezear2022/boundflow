#!/usr/bin/env python3
"""One fresh S4-1B0 positive/cache/fault correctness worker."""

# mypy: disable-error-code=import-untyped
# pylint: disable=import-error,protected-access,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring,import-outside-toplevel

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import torch

from boundflow.backends.tvm import asplos27_s4_ternary_endpoint as endpoint

SCHEMA = "boundflow.asplos27-s4-1b0-ternary-worker/v1"
SOURCE_CAPTURE = Path(
    "artifacts/asplos27-s3-streamed-suffix/resnet2b-rvir-v1/inputs/suffix-boundary.pt"
)
NUMEL = 18432


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().numpy().tobytes(order="C")


def _load_fixture(device: torch.device) -> tuple[torch.Tensor, ...]:
    payload = torch.load(SOURCE_CAPTURE, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version")
        != "boundflow-rvir-real-r6-relu17-conv0-boundary/v1"
        or payload.get("performance_claimed") is not False
    ):
        raise RuntimeError("production fixture envelope differs")
    values = payload["values"]
    coefficient = values["conv0_output"].reshape(-1).contiguous().to(device)
    lower = values["input_lower"].reshape(-1).contiguous().to(device)
    upper = values["input_upper"].reshape(-1).contiguous().to(device)
    if any(tensor.shape != (NUMEL,) for tensor in (coefficient, lower, upper)):
        raise RuntimeError("production fixture tensor shape differs")
    selector = torch.empty(NUMEL, dtype=torch.int8, device=device)
    selected = torch.empty(NUMEL, dtype=torch.float32, device=device)
    return coefficient, lower, upper, selector, selected


def _binary_sidecar(path: Path, tensors: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    names = ("coefficient", "lower", "upper", "selector", "selected")
    dtypes = ("float32", "float32", "float32", "int8", "float32")
    chunks = [_raw_bytes(tensor) for tensor in tensors]
    offset = 0
    index = []
    for name, dtype, chunk in zip(names, dtypes, chunks):
        index.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": [NUMEL],
                "offset": offset,
                "byte_count": len(chunk),
                "sha256": _sha_bytes(chunk),
            }
        )
        offset += len(chunk)
    payload = b"".join(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": path.name,
        "byte_count": len(payload),
        "sha256": _sha_bytes(payload),
        "index": index,
    }


def _module_payload(
    directory: Path,
    compiled: endpoint.CompiledTernaryEndpointV1,
    receipt: endpoint.TernaryEndpointModuleReceiptV1,
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    values = {
        "unscheduled_tir.json": compiled.unscheduled_tir_json,
        "scheduled_tir.json": compiled.scheduled_tir_json,
        "device_source.cu": compiled.device_source,
        "receipt.json": _canonical_json(receipt.to_dict()) + "\n",
    }
    result = {}
    for name, text in values.items():
        (directory / name).write_text(text, encoding="utf-8")
        result[name] = _sha_bytes(text.encode("utf-8"))
    return result


def _positive(args: argparse.Namespace, protocol_hash: str) -> dict[str, Any]:
    device = torch.device("cuda:0")
    tensors = _load_fixture(device)
    _coefficient, lower, upper, selector, selected = tensors
    spec = endpoint.TernaryEndpointBuildSpecV1(numel=NUMEL)
    schedule = endpoint.TernaryEndpointScheduleSpecV1()
    prepared = endpoint.PreparedTernaryEndpointProbeV1.prepare(spec, schedule, *tensors)
    launch = prepared.run_once(
        evaluation_ordinal=args.ordinal,
        parameter_state_version=0,
        selector_generation=args.ordinal,
    )
    torch.cuda.synchronize()
    endpoint.validate_selected_output_after_sync_v1(selector, selected)
    counts = {
        "positive": int((selector == 1).sum().item()),
        "negative": int((selector == -1).sum().item()),
        "zero": int((selector == 0).sum().item()),
        "invalid": int(
            ((selector != 1) & (selector != -1) & (selector != 0)).sum().item()
        ),
    }
    expected = torch.where(
        selector == 1,
        lower,
        torch.where(selector == -1, upper, (lower + upper) * 0.5),
    )
    selected_exact = _raw_bytes(selected) == _raw_bytes(expected)
    old_binary_zero_misclassified = int((selector == 0).sum().item())
    sidecar = _binary_sidecar(Path(args.binary_output), tensors)
    module_files = _module_payload(
        Path(args.module_output), prepared.compiled, prepared.module_receipt
    )
    row: dict[str, Any] = {
        "schema_version": SCHEMA,
        "mode": "positive",
        "worker_name": args.worker_name,
        "run_ordinal": args.ordinal,
        "pid": int(__import__("os").getpid()),
        "protocol_hash": protocol_hash,
        "source_capture_sha256": _sha_bytes(SOURCE_CAPTURE.read_bytes()),
        "counts": counts,
        "old_binary_zero_misclassified": old_binary_zero_misclassified,
        "selected_bitwise_exact": selected_exact,
        "dlpack_pointer_exact": len({row.data_ptr for row in prepared.descriptors}),
        "descriptor_hashes": [row.stable_hash() for row in prepared.descriptors],
        "module_receipt": prepared.module_receipt.to_dict(),
        "module_receipt_hash": prepared.module_receipt.stable_hash(),
        "cache_event": launch.cache_event,
        "launch": {
            "pack": launch.pack_launch_count,
            "select": launch.select_launch_count,
            "argument_occurrence": launch.argument_occurrence_count,
            "fallback": launch.fallback_count,
            "eager": launch.eager_count,
            "native_shadow": launch.native_shadow_count,
            "timing_recorded": launch.timing_recorded,
            "performance_claimed": launch.performance_claimed,
        },
        "binary": sidecar,
        "module_files": module_files,
        "environment": {
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "performance_claimed": False,
    }
    return row


def _cache(args: argparse.Namespace, protocol_hash: str) -> dict[str, Any]:
    spec = endpoint.TernaryEndpointBuildSpecV1(numel=NUMEL)
    schedule = endpoint.TernaryEndpointScheduleSpecV1()
    cache = endpoint.TernaryEndpointModuleCacheV1()
    first_compiled, first_receipt, first = cache.get(spec, schedule)
    second_compiled, second_receipt, second = cache.get(spec, schedule)
    return {
        "schema_version": SCHEMA,
        "mode": "cache",
        "worker_name": args.worker_name,
        "run_ordinal": args.ordinal,
        "pid": int(__import__("os").getpid()),
        "protocol_hash": protocol_hash,
        "events": [first.event, second.event],
        "compile_count": second.compile_count,
        "miss_count": second.miss_count,
        "hit_count": second.hit_count,
        "entry_count": second.entry_count,
        "same_compiled_object": first_compiled is second_compiled,
        "same_module_receipt": first_receipt == second_receipt,
        "module_receipt_hash": first_receipt.stable_hash(),
        "tensor_retention_count": 0,
        "performance_claimed": False,
    }


def _capture_fault(action: Any) -> dict[str, Any]:
    try:
        action()
    except endpoint.TernaryEndpointError as error:
        return {
            "reason": error.reason,
            "context_is_none": error.__context__ is None,
            "fallback": 0,
            "retry": 0,
            "native_shadow": 0,
        }
    raise RuntimeError("fault worker did not reject")


def _fault(args: argparse.Namespace, protocol_hash: str) -> dict[str, Any]:
    spec = endpoint.TernaryEndpointBuildSpecV1(numel=16)
    schedule = endpoint.TernaryEndpointScheduleSpecV1()
    tensors = _load_fixture(torch.device("cuda:0"))
    if args.fault == "classifier-policy":
        fault = _capture_fault(
            lambda: replace(spec, midpoint_policy="tampered").validate()
        )
    elif args.fault == "cache-source":
        cache = endpoint.TernaryEndpointModuleCacheV1()
        compiled, receipt, _event = cache.get(spec, schedule)
        cache._entries[receipt.cache_key] = (
            replace(compiled, device_source=compiled.device_source + "\n// tamper"),
            receipt,
        )
        fault = _capture_fault(lambda: cache.get(spec, schedule))
    elif args.fault == "descriptor-dlpack":
        real = torch.from_dlpack
        setattr(torch, "from_dlpack", lambda value: real(value).clone())
        try:
            fault = _capture_fault(
                lambda: endpoint._create_dlpack_view(tensors[0][:16])
            )
        finally:
            setattr(torch, "from_dlpack", real)
    elif args.fault == "stream-launch":
        import tvm_ffi

        short = tuple(tensor[:16].contiguous() for tensor in tensors)
        prepared = endpoint.PreparedTernaryEndpointProbeV1.prepare(
            spec, schedule, *short
        )
        real_stream = tvm_ffi.get_raw_stream
        setattr(tvm_ffi, "get_raw_stream", lambda device: int(real_stream(device)) + 1)
        try:
            fault = _capture_fault(
                lambda: prepared.run_once(
                    evaluation_ordinal=0,
                    parameter_state_version=0,
                    selector_generation=0,
                )
            )
        finally:
            setattr(tvm_ffi, "get_raw_stream", real_stream)
    elif args.fault == "invalid-selector-claim":
        selector = torch.tensor([-128] * 16, dtype=torch.int8, device="cuda")
        selected = torch.zeros(16, dtype=torch.float32, device="cuda")
        fault = _capture_fault(
            lambda: endpoint.validate_selected_output_after_sync_v1(selector, selected)
        )
    else:
        raise ValueError("unknown fault")
    return {
        "schema_version": SCHEMA,
        "mode": "fault",
        "worker_name": args.worker_name,
        "run_ordinal": args.ordinal,
        "pid": int(__import__("os").getpid()),
        "protocol_hash": protocol_hash,
        "fault": args.fault,
        "result": fault,
        "performance_claimed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("positive", "cache", "fault"), required=True)
    parser.add_argument("--worker-name", required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binary-output", default="")
    parser.add_argument("--module-output", default="")
    parser.add_argument("--fault", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        protocol_hash = _sha_bytes(args.protocol.read_bytes())
        if args.mode == "positive":
            row = _positive(args, protocol_hash)
        elif args.mode == "cache":
            row = _cache(args, protocol_hash)
        else:
            row = _fault(args, protocol_hash)
        row["worker_payload_hash"] = _hash(row)
        print(_canonical_json(row))
        return 0
    except Exception as error:  # pylint: disable=broad-exception-caught
        print(
            _canonical_json(
                {
                    "schema_version": SCHEMA,
                    "mode": args.mode,
                    "worker_name": args.worker_name,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback_sha256": _sha_bytes(
                        traceback.format_exc().encode("utf-8")
                    ),
                    "performance_claimed": False,
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
