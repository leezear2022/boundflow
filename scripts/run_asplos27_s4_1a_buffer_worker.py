#!/usr/bin/env python3
"""Run one fresh real-provider S4-1A positive or isolated-fault worker."""

# pylint: disable=wrong-import-position,import-error,import-outside-toplevel
# pylint: disable=too-many-locals,too-many-statements,protected-access
# pylint: disable=missing-function-docstring,no-member,line-too-long
# pylint: disable=unidiomatic-typecheck,broad-exception-caught
# pylint: disable=too-many-branches,too-many-boolean-expressions

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_asplos27_s4_admission_worker as admission_worker
from scripts import run_rvir_v4_production_state_capture as capture_runner
from scripts.run_rvir_v4_pre_state_artifact import TOPOLOGY

WORKER_SCHEMA = "boundflow.asplos27-s4-1a-buffer-worker/v1"
FAULTS = (
    "parameter",
    "gradient",
    "output",
    "view",
    "roundtrip",
    "receipt",
    "adoption",
)
_ACTIVE_FAULT = "none"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _binary_record(name: str, source: Any, candidate: Any) -> dict[str, object]:
    source_bytes = source.detach().contiguous().cpu().numpy().tobytes(order="C")
    candidate_bytes = candidate.detach().contiguous().cpu().numpy().tobytes(order="C")
    return {
        "name": name,
        "shape": list(candidate.shape),
        "dtype": str(candidate.dtype),
        "source_hex": source_bytes.hex(),
        "candidate_hex": candidate_bytes.hex(),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "byte_count": len(candidate_bytes),
    }


def _install_fault(module: Any, fault: str) -> tuple[str, object] | None:
    if fault == "none":
        return None
    helper = {
        "parameter": "_clone_parameter",
        "gradient": "_empty_buffer",
        "output": "_empty_buffer",
        "view": "_create_dlpack_view",
        "roundtrip": "_roundtrip_dlpack",
        "receipt": "_build_receipt",
        "adoption": "_adopt_prepared",
    }[fault]
    original = getattr(module, helper)
    calls = 0

    def injected(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if fault == "output" and calls <= 7:
            return original(*args, **kwargs)
        raise RuntimeError(f"s4-1a-isolated-fault:{fault}")

    setattr(module, helper, injected)
    return helper, original


class _BufferObserver(admission_worker._AdmissionObserver):
    def capture(self, net: Any, pre_result: Any, kwargs: Mapping[str, Any]) -> None:
        import torch

        from boundflow.runtime import asplos27_s4_ordered_buffer_abi as buffer_module
        from boundflow.runtime.asplos27_s4_mutable_state_admission import (
            extract_s4_live_mutable_sources_v1,
            prepare_s4_mutable_state_admission_v1,
        )
        from boundflow.runtime.r3_structured_owner_custom_backward import (
            compile_r31_full_region_plan_v1,
        )
        from boundflow.runtime.rvir_v4_pre_state_initializer import (
            initialize_rvir_v4_native_pre_state,
        )

        if self.intercept_count:
            raise RuntimeError("S4-1A provider core intercept repeats")
        self.intercept_count += 1
        if (
            kwargs.get("fix_interm_bounds") is not True
            or kwargs.get("enable_decision_precompute") is not True
            or kwargs.get("enable_clip_domains") is not False
            or kwargs.get("precompute_bfs_flag") is not False
            or kwargs.get("is_multitree_bab") is not False
            or type(kwargs.get("branching_heuristic")).__name__ != "KfsbBranching"
        ):
            raise ValueError("S4-1A production flags differ")
        policy = capture_runner._optimizer_policy(self.arguments, kwargs)
        snapshot = capture_runner._build_core_pre_snapshot(
            pre_result, net=net, core_id=0, policy=policy
        )
        mapping = initialize_rvir_v4_native_pre_state(snapshot, TOPOLOGY)
        plan = compile_r31_full_region_plan_v1(self.module, snapshot, mapping, TOPOLOGY)
        live_sources = extract_s4_live_mutable_sources_v1(pre_result, TOPOLOGY)
        exact_call_id = f"asplos27-s4-1a-formal:{self.run_ordinal:03d}:{_ACTIVE_FAULT}"
        previous_profile = sys.getprofile()
        sys.setprofile(self._profile_provider_callback)
        try:
            admission = prepare_s4_mutable_state_admission_v1(
                snapshot,
                TOPOLOGY,
                plan,
                live_sources,
                exact_call_id=exact_call_id,
            )
        finally:
            sys.setprofile(previous_profile)
        admission_receipt = admission.receipt
        admission_receipt.validate()
        torch.cuda.synchronize()
        allocated_entry = int(torch.cuda.memory_allocated())
        reserved_entry = int(torch.cuda.memory_reserved())
        patch = _install_fault(buffer_module, _ACTIVE_FAULT)
        prepared = None
        caught_payload: dict[str, object] | None = None
        try:
            prepared = buffer_module.prepare_s4_mutable_buffers_v1(
                admission, live_sources, exact_call_id=exact_call_id
            )
        except buffer_module.S4MutableBufferPreparationError as error:
            rendered = "".join(traceback.format_exception(error))
            caught_payload = {
                "detail_code": error.detail_code,
                "verification_reason": error.verification_reason.value,
                "context_is_none": error.__context__ is None,
                "traceback_sha256": hashlib.sha256(
                    rendered.encode("utf-8")
                ).hexdigest(),
            }
        finally:
            if patch is not None:
                setattr(buffer_module, patch[0], patch[1])
        if _ACTIVE_FAULT != "none":
            if caught_payload is None or prepared is not None:
                raise RuntimeError("S4-1A isolated fault did not fail closed")
            gc.collect()
            torch.cuda.synchronize()
            allocated_exit = int(torch.cuda.memory_allocated())
            reserved_exit = int(torch.cuda.memory_reserved())
            self.payload = {
                "schema_version": WORKER_SCHEMA,
                "mode": "fault",
                "run_ordinal": self.run_ordinal,
                "fault": _ACTIVE_FAULT,
                "error": caught_payload,
                "admission_hash": admission_receipt.admission_hash,
                "exact_call_identity_hash": admission_receipt.exact_call_identity_hash,
                "admission_state": admission._state,
                "allocator": {
                    "allocated_entry": allocated_entry,
                    "allocated_exit": allocated_exit,
                    "allocated_delta": allocated_exit - allocated_entry,
                    "reserved_entry": reserved_entry,
                    "reserved_exit": reserved_exit,
                },
                "counters": {
                    "provider_core_intercept_count": self.intercept_count,
                    "provider_core_execute_count": 0,
                    "provider_compute_bounds_callback_count": self.provider_compute_bounds_callback_count,
                    "provider_update_bounds_callback_count": self.provider_update_bounds_callback_count,
                    "candidate_kernel_launch_count": 0,
                    "fallback_count": 0,
                    "retry_count": 0,
                    "empty_cache_count": 0,
                    "mutation_count": 0,
                },
                "timing_recorded": False,
                "performance_claimed": False,
            }
            self.payload["worker_payload_hash"] = _canonical_hash(self.payload)
            return

        if prepared is None or caught_payload is not None:
            raise RuntimeError(
                f"S4-1A positive buffer prepare did not complete: {caught_payload}"
            )
        receipt = prepared.receipt
        receipt.validate()
        resources = prepared._resources
        if resources is None:
            raise RuntimeError("S4-1A resource owner was not published")
        if resources._upstream is None:
            raise RuntimeError("S4-1A upstream buffer was not published")
        parameters = tuple(resources._parameters)
        active_sources: list[tuple[str, Any]] = []
        for slot in admission_receipt.slots:
            active_sources.append(
                (
                    f"alpha/{slot.slot_ordinal}",
                    live_sources[slot.alpha_semantic_path][0, 0],
                )
            )
            if slot.beta_active:
                active_sources.append(
                    (f"beta/{slot.slot_ordinal}", live_sources[slot.beta_semantic_path])
                )
        binary_records = [
            _binary_record(name, source, candidate)
            for (name, source), candidate in zip(active_sources, parameters)
        ]
        upstream_expected = torch.full_like(resources._upstream, -1.0)
        binary_records.append(
            _binary_record("fixed_upstream", upstream_expected, resources._upstream)
        )
        private_keys = tuple(resources._private_view_keys)
        storage_tokens = [
            buffer_module._storage_token(item) for item in resources.buffers()
        ]
        torch.cuda.synchronize()
        allocated_peak = int(torch.cuda.memory_allocated())
        reserved_peak = int(torch.cuda.memory_reserved())
        prepared.close()
        parameters = ()
        del upstream_expected
        gc.collect()
        torch.cuda.synchronize()
        allocated_exit = int(torch.cuda.memory_allocated())
        reserved_exit = int(torch.cuda.memory_reserved())
        self.payload = {
            "schema_version": WORKER_SCHEMA,
            "mode": "positive",
            "run_ordinal": self.run_ordinal,
            "admission_receipt": admission_receipt.to_dict(),
            "buffer_receipt": receipt.to_dict(),
            "binary_records": binary_records,
            "physical": {
                "candidate_storage_count": len(storage_tokens),
                "candidate_storage_unique_count": len(set(storage_tokens)),
                "base_dlpack_view_count": len(private_keys),
                "private_view_key_unique_count": len(set(private_keys)),
                "empty_beta_physical_count": 0,
                "resource_state_after_close": resources._state,
            },
            "allocator": {
                "allocated_entry": allocated_entry,
                "allocated_peak": allocated_peak,
                "allocated_exit": allocated_exit,
                "allocated_delta_after_close": allocated_exit - allocated_entry,
                "reserved_entry": reserved_entry,
                "reserved_peak": reserved_peak,
                "reserved_exit": reserved_exit,
            },
            "counters": {
                "provider_core_intercept_count": self.intercept_count,
                "provider_core_execute_count": 0,
                "provider_compute_bounds_callback_count": self.provider_compute_bounds_callback_count,
                "provider_update_bounds_callback_count": self.provider_update_bounds_callback_count,
                "candidate_kernel_launch_count": 0,
                "fallback_count": 0,
                "retry_count": 0,
                "empty_cache_count": 0,
                "mutation_count": 0,
            },
            "timing_recorded": False,
            "performance_claimed": False,
        }
        self.payload["worker_payload_hash"] = _canonical_hash(self.payload)


def run(args: argparse.Namespace) -> dict[str, object]:
    global _ACTIVE_FAULT  # pylint: disable=global-statement
    if args.fault not in ("none", *FAULTS):
        raise ValueError("S4-1A fault name differs")
    _ACTIVE_FAULT = args.fault
    original = admission_worker._AdmissionObserver
    setattr(admission_worker, "_AdmissionObserver", _BufferObserver)
    try:
        result = admission_worker.run(args)
    finally:
        setattr(admission_worker, "_AdmissionObserver", original)
        _ACTIVE_FAULT = "none"
    result["schema_version"] = WORKER_SCHEMA
    protocol = result["protocol"]
    if type(protocol) is not dict:
        raise TypeError("S4-1A worker protocol differs")
    protocol.update(
        {
            "exact_call_id_template": "asplos27-s4-1a-formal:{run_ordinal:03d}:{fault}",
            "admission_per_process": 1,
            "buffer_prepare": True,
            "candidate_execute": False,
            "mutation": False,
            "fault": args.fault,
            "timing_recorded": False,
            "performance_claimed": False,
        }
    )
    result["protocol_hash"] = _canonical_hash(protocol)
    result["performance_claimed"] = False
    result["raw_hash"] = _canonical_hash(
        {key: value for key, value in result.items() if key != "raw_hash"}
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--run-ordinal", type=int, required=True)
    parser.add_argument("--fault", choices=("none", *FAULTS), required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.benchmark_root = args.benchmark_root.resolve()
    args.abcrown_root = args.abcrown_root.resolve()
    args.model = args.model.resolve()
    args.property = args.property.resolve()
    args.result = args.result.resolve()
    result = run(args)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    payload = result["admission"]
    if type(payload) is not dict:
        raise TypeError("S4-1A worker payload differs")
    print(
        _canonical(
            {
                "status": "captured",
                "run_ordinal": args.run_ordinal,
                "mode": payload["mode"],
                "fault": args.fault,
                "performance_claimed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
