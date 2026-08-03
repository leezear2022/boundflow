#!/usr/bin/env python3
"""Frozen CUDA physical-memory protocol for NRIR-2 real-network plans."""

# pylint: disable=duplicate-code,too-many-arguments,too-many-branches
# pylint: disable=too-many-locals,too-many-statements
# pylint: disable=protected-access,too-many-boolean-expressions,too-many-lines

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence, cast

import torch

from boundflow.frontends.onnx.frontend import import_onnx
from boundflow.frontends.plain_crown_bound_ir import tensor_content_hash
from boundflow.planner import plan_interval_ibp_v0
from boundflow.runtime.abcrown_adapter import (
    bind_intermediate_bounds,
    deserialize_intermediate_bounds,
    file_sha256,
    intermediate_bounds_sha256,
)
from boundflow.runtime.crown_ibp import _forward_ibp_trace_mlp
from boundflow.runtime.native_verifier_ir_integration import (
    compile_native_plain_crown_memory_query,
    execute_native_plain_crown_memory_query,
)
from boundflow.runtime.storage_plan_runtime import PreparedStoragePlanRuntime
from boundflow.runtime.task_backend_dispatch import PyTorchReferenceTaskBackend
from boundflow.runtime.task_executor import InputSpec
from boundflow.runtime.task_ir_executor import (
    TaskTraceMode,
    prepare_task_ir_execution,
)
from scripts.run_native_real_network_ir_artifact import (
    EXPECTED_PRIMAL_OPS,
    INTERMEDIATE_BOUNDS_SHA256,
    MODEL_SHA256,
)
from scripts.run_native_real_network_memory_plans_artifact import (
    AVAILABLE_MEMORY_BYTES,
    _load_source_artifact,
    _mapping,
    _payload_tensors,
)

PROTOCOL_SCHEMA_VERSION = "boundflow.native-cuda-memory-protocol/v1"
WORKER_ROW_SCHEMA_VERSION = "boundflow.native-cuda-memory-worker/v1"
SUMMARY_SCHEMA_VERSION = "boundflow.native-cuda-memory-summary/v1"
ARTIFACT_SCHEMA_VERSION = "boundflow.native-cuda-memory-artifact/v1"
ENVIRONMENT_SCHEMA_VERSION = "boundflow.cuda-environment-probe/v1"
ENVIRONMENT_ARTIFACT_SCHEMA_VERSION = "boundflow.cuda-environment-probe-artifact/v1"
QUERY_ID = "vnncomp21-resnet2b-prop0-native-ir3-cuda-memory"
PLAN_IDS = {
    "retain": "storage:native-retain-all-v1",
    "reuse": "storage:native-lifetime-reuse-v1",
}
ARTIFACT_FILES = ("raw.jsonl", "summary.json")
IR_HASH_FIELDS = (
    "bound_module_hash",
    "plan_template_hash",
    "plan_instance_hash",
    "task_module_hash",
    "schedule_hash",
)
WORKER_ROW_FIELDS = {
    "schema_version",
    "status",
    "performance_claimed",
    "worker_pid",
    "plan",
    "repeat",
    "protocol",
    "environment",
    "model_sha256",
    "intermediate_bounds_sha256",
    "ir_hashes",
    "storage_candidate_id",
    "planned_peak_bytes",
    "observed_logical_peak_bytes",
    "baseline_allocated_bytes",
    "peak_allocated_bytes",
    "peak_allocated_delta_bytes",
    "baseline_reserved_bytes",
    "peak_reserved_bytes",
    "peak_reserved_delta_bytes",
    "latency_samples_ms",
    "latency_median_ms",
    "lower_sha256",
    "upper_sha256",
    "task_trace_hash",
    "storage_trace_hash",
    "comparison",
}


class CudaEnvironmentUnavailable(RuntimeError):
    """Raised before artifact creation when the frozen device contract is absent."""


@dataclass(frozen=True)
class ProtocolConfig:
    """Pre-registered CUDA measurement configuration."""

    repeats: int = 5
    warmup_iterations: int = 5
    measured_iterations: int = 20
    minimum_allocated_reduction: float = 0.20
    maximum_latency_ratio: float = 1.20
    device_index: int = 0
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def validate(self) -> None:
        """Reject protocol drift before any measurement is launched."""

        if self.schema_version != PROTOCOL_SCHEMA_VERSION:
            raise ValueError("unsupported CUDA memory protocol schema")
        if self.repeats < 3 or self.warmup_iterations <= 0:
            raise ValueError("CUDA memory protocol requires >=3 repeats and warmup")
        if self.measured_iterations <= 0 or self.device_index < 0:
            raise ValueError("CUDA memory protocol iteration/device is invalid")
        if not 0.0 < self.minimum_allocated_reduction < 1.0:
            raise ValueError("CUDA memory reduction threshold is invalid")
        if self.maximum_latency_ratio < 1.0:
            raise ValueError("CUDA latency ratio threshold is invalid")

    def to_dict(self) -> dict[str, object]:
        """Return the exact configuration embedded in every worker row."""

        self.validate()
        return asdict(self)


def canonical_json(value: object, *, indent: int | None = None) -> str:
    """Encode deterministic JSON for raw rows, summaries, and manifests."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def probe_cuda_environment() -> dict[str, object]:
    """Return a non-claiming environment record without initializing a workload."""

    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        smi_record = {
            "returncode": smi.returncode,
            "stdout": smi.stdout.strip(),
            "stderr": smi.stderr.strip(),
        }
    except subprocess.TimeoutExpired as error:
        smi_record = {
            "returncode": 124,
            "stdout": (error.stdout or "").strip(),
            "stderr": "nvidia-smi timed out after 10 seconds",
        }
    except OSError as error:
        smi_record = {
            "returncode": 127,
            "stdout": "",
            "stderr": str(error),
        }
    available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    devices: list[dict[str, object]] = []
    if available:
        for index in range(device_count):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    probe: dict[str, object] = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "status": (
            "ready" if available and device_count > 0 else "environment_unavailable"
        ),
        "performance_claimed": False,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": available,
        "device_count": device_count,
        "devices": devices,
        "nvidia_smi": smi_record,
    }
    validate_environment_probe(probe)
    return probe


def validate_environment_probe(probe: Mapping[str, Any]) -> None:
    """Reject a probe that upgrades an unavailable host into benchmark evidence."""

    expected_fields = {
        "schema_version",
        "status",
        "performance_claimed",
        "torch_version",
        "torch_cuda_build",
        "cuda_available",
        "device_count",
        "devices",
        "nvidia_smi",
    }
    if (
        set(probe) != expected_fields
        or probe.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION
        or probe.get("performance_claimed") is not False
        or not isinstance(probe.get("torch_version"), str)
        or not isinstance(probe.get("torch_cuda_build"), (str, type(None)))
        or not isinstance(probe.get("cuda_available"), bool)
        or not isinstance(probe.get("device_count"), int)
        or int(probe.get("device_count", -1)) < 0
        or not isinstance(probe.get("devices"), list)
        or not isinstance(probe.get("nvidia_smi"), Mapping)
    ):
        raise ValueError("CUDA environment probe contract differs")
    smi = _mapping(probe["nvidia_smi"], "nvidia-smi record")
    if (
        set(smi) != {"returncode", "stdout", "stderr"}
        or not isinstance(smi.get("returncode"), int)
        or not isinstance(smi.get("stdout"), str)
        or not isinstance(smi.get("stderr"), str)
    ):
        raise ValueError("CUDA environment nvidia-smi record differs")
    ready = bool(probe["cuda_available"]) and int(probe["device_count"]) > 0
    expected_status = "ready" if ready else "environment_unavailable"
    if probe.get("status") != expected_status:
        raise ValueError("CUDA environment probe status contradicts device facts")
    if ready and len(probe["devices"]) != int(probe["device_count"]):
        raise ValueError("CUDA environment probe device inventory differs")
    if not ready and probe["devices"]:
        raise ValueError("unavailable CUDA probe must not invent devices")
    for index, device in enumerate(probe["devices"]):
        device_record = _mapping(device, "CUDA device record")
        capability = device_record.get("capability")
        if (
            set(device_record) != {"index", "name", "capability", "total_memory_bytes"}
            or device_record.get("index") != index
            or not isinstance(device_record.get("name"), str)
            or not device_record.get("name")
            or not isinstance(capability, list)
            or len(capability) != 2
            or any(not isinstance(item, int) or item < 0 for item in capability)
            or not isinstance(device_record.get("total_memory_bytes"), int)
            or int(device_record.get("total_memory_bytes", 0)) <= 0
        ):
            raise ValueError("CUDA environment device record differs")
    if ready and not probe["torch_cuda_build"]:
        raise ValueError("ready CUDA environment lacks a CUDA build identity")


def write_environment_probe_artifact(artifact_dir: Path) -> dict[str, object]:
    """Persist non-performance environment evidence with an exact digest."""

    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_cuda_environment()
    environment_path = artifact_dir / "environment.json"
    environment_path.write_text(
        canonical_json(probe, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": ENVIRONMENT_ARTIFACT_SCHEMA_VERSION,
        "status": probe["status"],
        "performance_claimed": False,
        "files": {"environment.json": file_sha256(environment_path)},
    }
    (artifact_dir / "manifest.json").write_text(
        canonical_json(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return probe


def replay_environment_probe_artifact(artifact_dir: Path) -> dict[str, object]:
    """Verify probe digest and its fail-closed status/device relationship."""

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != ENVIRONMENT_ARTIFACT_SCHEMA_VERSION
        or manifest.get("performance_claimed") is not False
    ):
        raise ValueError("CUDA environment artifact manifest contract differs")
    files = _mapping(manifest.get("files"), "CUDA environment artifact files")
    environment_path = artifact_dir / "environment.json"
    if files != {"environment.json": file_sha256(environment_path)}:
        raise ValueError("CUDA environment artifact digest differs")
    probe = json.loads(environment_path.read_text(encoding="utf-8"))
    validate_environment_probe(probe)
    if manifest.get("status") != probe["status"]:
        raise ValueError("CUDA environment manifest/probe status differs")
    return probe


def run_worker(  # pylint: disable=too-many-locals,too-many-statements
    *,
    model: Path,
    source_artifact_dir: Path,
    plan: str,
    repeat: int,
    config: ProtocolConfig,
) -> dict[str, object]:
    """Measure one plan in one fresh process after untimed setup and warmup."""

    config.validate()
    probe = probe_cuda_environment()
    if probe["status"] != "ready":
        raise CudaEnvironmentUnavailable("CUDA device is unavailable")
    if config.device_index >= cast(int, probe["device_count"]):
        raise CudaEnvironmentUnavailable("configured CUDA device is unavailable")
    if plan not in PLAN_IDS or not 0 <= repeat < config.repeats:
        raise ValueError("CUDA worker plan/repeat is outside the protocol")
    if file_sha256(model) != MODEL_SHA256:
        raise ValueError("CUDA memory protocol model digest differs")
    device = torch.device("cuda", config.device_index)
    torch.cuda.set_device(device)
    _source_manifest, payload = _load_source_artifact(source_artifact_dir)
    tensors = {
        name: tensor.to(device) for name, tensor in _payload_tensors(payload).items()
    }
    cpu_external_bounds = deserialize_intermediate_bounds(
        _mapping(payload.get("external_intermediate_bounds"), "intermediate bounds")
    )
    if intermediate_bounds_sha256(cpu_external_bounds) != INTERMEDIATE_BOUNDS_SHA256:
        raise ValueError("CUDA memory protocol intermediate-bound digest differs")
    external_bounds = tuple(
        replace(item, lower=item.lower.to(device), upper=item.upper.to(device))
        for item in cpu_external_bounds
    )
    program = import_onnx(str(model), do_shape_infer=True, normalize=True)
    legacy_module = deepcopy(plan_interval_ibp_v0(program))
    if (
        tuple(op.op_type for op in legacy_module.get_entry_task().ops)
        != EXPECTED_PRIMAL_OPS
    ):
        raise ValueError("CUDA memory protocol primal topology differs")
    legacy_module.bindings = _move_tensors(legacy_module.bindings, device=device)
    input_spec = InputSpec.box(
        value_name=program.graph.inputs[0],
        lower=tensors["input_lower"],
        upper=tensors["input_upper"],
    )
    interval_env, local_relu_pre = _forward_ibp_trace_mlp(legacy_module, input_spec)
    relu_pre = bind_intermediate_bounds(external_bounds, local_relu_pre)
    high = compile_native_plain_crown_memory_query(
        legacy_module,
        input_spec,
        interval_env=interval_env,
        relu_pre=relu_pre,
        linear_spec_C=tensors["linear_spec_c"],
        intermediate_bounds_hash=INTERMEDIATE_BOUNDS_SHA256,
        query_id=QUERY_ID,
        available_memory_bytes=AVAILABLE_MEMORY_BYTES,
        memory_budget_bytes=AVAILABLE_MEMORY_BYTES,
    )
    reuse_peak = next(
        candidate.cost.predicted_peak_bytes
        for candidate in high.template.storage_candidates
        if candidate.candidate_id == PLAN_IDS["reuse"]
    )
    compilation = (
        high
        if plan == "retain"
        else compile_native_plain_crown_memory_query(
            legacy_module,
            input_spec,
            interval_env=interval_env,
            relu_pre=relu_pre,
            linear_spec_C=tensors["linear_spec_c"],
            intermediate_bounds_hash=INTERMEDIATE_BOUNDS_SHA256,
            query_id=QUERY_ID,
            available_memory_bytes=AVAILABLE_MEMORY_BYTES,
            memory_budget_bytes=reuse_peak,
        )
    )
    if compilation.instance.storage_decision.candidate_id != PLAN_IDS[plan]:
        raise ValueError("CUDA worker selected the wrong storage plan")
    prepared = prepare_task_ir_execution(
        compilation.task_module,
        compilation.schedule,
        bound_module=compilation.bound_module,
        template=compilation.template,
        instance=compilation.instance,
        legacy_task_module=legacy_module,
    )
    prepared_storage = PreparedStoragePlanRuntime.prepare(
        bound_module=compilation.bound_module,
        template=compilation.template,
        instance=compilation.instance,
        schedule=compilation.schedule,
    )
    backend = PyTorchReferenceTaskBackend()

    def execute_once():
        return execute_native_plain_crown_memory_query(
            compilation,
            legacy_task_module=legacy_module,
            input_spec=input_spec,
            relu_pre=relu_pre,
            linear_spec_C=tensors["linear_spec_c"],
            prepared=prepared,
            prepared_storage=prepared_storage,
            trace_mode=TaskTraceMode.PRODUCTION,
            backend=backend,
        )

    for _ in range(config.warmup_iterations):
        warm_result, _warm_task_trace, _warm_storage_trace = execute_once()
        del warm_result, _warm_task_trace, _warm_storage_trace
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    samples_ms: list[float] = []
    result = task_trace = storage_trace = None
    for _ in range(config.measured_iterations):
        if result is not None:
            del result, task_trace, storage_trace
            result = task_trace = storage_trace = None
        started = time.perf_counter_ns()
        result, task_trace, storage_trace = execute_once()
        torch.cuda.synchronize(device)
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if result is None or task_trace is None or storage_trace is None:
        raise AssertionError("CUDA worker measured no queries")
    expected = tensors["external_lower"].to(result.lower)
    comparison = {
        "allclose": bool(torch.allclose(result.lower, expected, atol=2e-4, rtol=2e-4)),
        "max_abs_diff": float((result.lower - expected).abs().max().item()),
        "sign_agreement": int(((result.lower >= 0) == (expected >= 0)).sum().item()),
        "sign_total": int(expected.numel()),
    }
    if not comparison["allclose"] or comparison["sign_agreement"] != 9:
        raise ValueError("CUDA worker result differs from external semantics")
    row: dict[str, object] = {
        "schema_version": WORKER_ROW_SCHEMA_VERSION,
        "status": "ok",
        "performance_claimed": False,
        "worker_pid": os.getpid(),
        "plan": plan,
        "repeat": repeat,
        "protocol": config.to_dict(),
        "environment": probe,
        "model_sha256": MODEL_SHA256,
        "intermediate_bounds_sha256": INTERMEDIATE_BOUNDS_SHA256,
        "ir_hashes": compilation.hashes(),
        "storage_candidate_id": storage_trace.storage_candidate_id,
        "planned_peak_bytes": storage_trace.planned_peak_bytes,
        "observed_logical_peak_bytes": storage_trace.observed_peak_live_bytes,
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_delta_bytes": peak_allocated - baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_delta_bytes": peak_reserved - baseline_reserved,
        "latency_samples_ms": samples_ms,
        "latency_median_ms": statistics.median(samples_ms),
        "lower_sha256": tensor_content_hash(result.lower),
        "upper_sha256": tensor_content_hash(result.upper),
        "task_trace_hash": task_trace.stable_hash(),
        "storage_trace_hash": storage_trace.stable_hash(),
        "comparison": comparison,
    }
    validate_worker_row(row, config=config)
    return row


def validate_worker_row(row: Mapping[str, Any], *, config: ProtocolConfig) -> None:
    """Validate one fresh-process row before it may enter aggregation."""

    if (
        set(row) != WORKER_ROW_FIELDS
        or row.get("model_sha256") != MODEL_SHA256
        or row.get("intermediate_bounds_sha256") != INTERMEDIATE_BOUNDS_SHA256
        or row.get("schema_version") != WORKER_ROW_SCHEMA_VERSION
        or row.get("status") != "ok"
        or row.get("performance_claimed") is not False
        or row.get("protocol") != config.to_dict()
        or row.get("plan") not in PLAN_IDS
        or row.get("storage_candidate_id") != PLAN_IDS.get(str(row.get("plan")))
        or not isinstance(row.get("repeat"), int)
        or not isinstance(row.get("worker_pid"), int)
        or int(row.get("worker_pid", 0)) <= 0
    ):
        raise ValueError("CUDA worker identity/plan contract differs")
    validate_environment_probe(_mapping(row.get("environment"), "worker environment"))
    if _mapping(row["environment"], "worker environment").get("status") != "ready":
        raise ValueError("CUDA worker row came from an unavailable environment")
    samples = row.get("latency_samples_ms")
    comparison = _mapping(row.get("comparison"), "worker comparison")
    hashes = _mapping(row.get("ir_hashes"), "worker IR hashes")
    baseline_allocated = cast(int, row.get("baseline_allocated_bytes"))
    peak_allocated = cast(int, row.get("peak_allocated_bytes"))
    delta_allocated = cast(int, row.get("peak_allocated_delta_bytes"))
    baseline_reserved = cast(int, row.get("baseline_reserved_bytes"))
    peak_reserved = cast(int, row.get("peak_reserved_bytes"))
    delta_reserved = cast(int, row.get("peak_reserved_delta_bytes"))
    planned_peak = cast(int, row.get("planned_peak_bytes"))
    observed_peak = cast(int, row.get("observed_logical_peak_bytes"))
    if (
        not isinstance(samples, list)
        or len(samples) != config.measured_iterations
        or any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in samples
        )
        or comparison.get("allclose") is not True
        or comparison.get("sign_agreement") != 9
        or comparison.get("sign_total") != 9
        or set(comparison)
        != {"allclose", "max_abs_diff", "sign_agreement", "sign_total"}
        or not isinstance(comparison.get("max_abs_diff"), (int, float))
        or not math.isfinite(float(comparison.get("max_abs_diff", math.inf)))
        or float(comparison.get("max_abs_diff", -1.0)) < 0.0
        or set(hashes) != set(IR_HASH_FIELDS)
        or any(not _is_sha256(hashes[field]) for field in IR_HASH_FIELDS)
        or any(
            not isinstance(value, int)
            for value in (
                baseline_allocated,
                peak_allocated,
                delta_allocated,
                baseline_reserved,
                peak_reserved,
                delta_reserved,
                planned_peak,
                observed_peak,
            )
        )
        or baseline_allocated < 0
        or baseline_reserved < 0
        or delta_allocated < 0
        or delta_reserved < 0
        or peak_allocated - baseline_allocated != delta_allocated
        or peak_reserved - baseline_reserved != delta_reserved
        or planned_peak <= 0
        or not 0 < observed_peak <= planned_peak
        or not _is_sha256(row.get("lower_sha256"))
        or not _is_sha256(row.get("upper_sha256"))
        or not _is_sha256(row.get("task_trace_hash"))
        or not _is_sha256(row.get("storage_trace_hash"))
    ):
        raise ValueError("CUDA worker measurement/semantic contract differs")
    latency_median = row.get("latency_median_ms")
    if (
        not isinstance(latency_median, (int, float))
        or not math.isfinite(float(latency_median))
        or float(latency_median) != statistics.median(samples)
    ):
        raise ValueError("CUDA worker latency median differs from raw samples")


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]], *, config: ProtocolConfig
) -> dict[str, object]:
    """Apply the pre-registered correctness, memory, and latency gates."""

    config.validate()
    for row in rows:
        validate_worker_row(row, config=config)
    expected_keys = {
        (plan, repeat) for repeat in range(config.repeats) for plan in PLAN_IDS
    }
    actual_keys = {(str(row["plan"]), int(row["repeat"])) for row in rows}
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError("CUDA benchmark rows do not cover the frozen matrix")
    expected_order = [
        (plan, repeat)
        for repeat in range(config.repeats)
        for plan in (("retain", "reuse") if repeat % 2 == 0 else ("reuse", "retain"))
    ]
    actual_order = [(str(row["plan"]), int(row["repeat"])) for row in rows]
    if actual_order != expected_order:
        raise ValueError("CUDA benchmark rows violate the frozen alternating order")
    if len({int(row["worker_pid"]) for row in rows}) != len(rows):
        raise ValueError("CUDA benchmark rows do not prove fresh worker processes")
    by_plan = {
        plan: sorted(
            (row for row in rows if row["plan"] == plan),
            key=lambda row: int(row["repeat"]),
        )
        for plan in PLAN_IDS
    }
    identity_fields = (
        "model_sha256",
        "intermediate_bounds_sha256",
        "lower_sha256",
        "upper_sha256",
    )
    identities_match = (
        all(len({str(row[field]) for row in rows}) == 1 for field in identity_fields)
        and len(
            {
                (
                    str(_mapping(row["ir_hashes"], "IR hashes")["bound_module_hash"]),
                    str(_mapping(row["ir_hashes"], "IR hashes")["plan_template_hash"]),
                )
                for row in rows
            }
        )
        == 1
    )
    retain_allocated = statistics.median(
        int(row["peak_allocated_delta_bytes"]) for row in by_plan["retain"]
    )
    reuse_allocated = statistics.median(
        int(row["peak_allocated_delta_bytes"]) for row in by_plan["reuse"]
    )
    if retain_allocated <= 0:
        raise ValueError("CUDA retain peak delta must be positive")
    allocated_reduction = 1.0 - (reuse_allocated / retain_allocated)
    retain_latency = statistics.median(
        float(row["latency_median_ms"]) for row in by_plan["retain"]
    )
    reuse_latency = statistics.median(
        float(row["latency_median_ms"]) for row in by_plan["reuse"]
    )
    latency_ratio = reuse_latency / retain_latency
    gates = {
        "complete_fresh_process_matrix": len(rows) == config.repeats * 2,
        "stable_environment_identity": (
            len({canonical_json(row["environment"]) for row in rows}) == 1
        ),
        "all_worker_correctness_passed": all(
            _mapping(row["comparison"], "comparison").get("allclose") is True
            for row in rows
        ),
        "cross_plan_semantic_identity": identities_match,
        "allocated_reduction_gte_threshold": (
            allocated_reduction >= config.minimum_allocated_reduction
        ),
        "latency_ratio_lte_threshold": (latency_ratio <= config.maximum_latency_ratio),
    }
    performance_claimed = all(gates.values())
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "validated" if performance_claimed else "no_go",
        "performance_claimed": performance_claimed,
        "protocol": config.to_dict(),
        "row_count": len(rows),
        "plan_counts": {plan: len(plan_rows) for plan, plan_rows in by_plan.items()},
        "retain": _plan_aggregate(by_plan["retain"]),
        "reuse": _plan_aggregate(by_plan["reuse"]),
        "allocated_reduction": allocated_reduction,
        "latency_ratio": latency_ratio,
        "gates": gates,
        "claim_boundary": (
            "CUDA lower-only physical allocator and latency evidence for one frozen "
            "ResNet query; no OOM rescue, representation, batching, or full verifier claim"
        ),
    }
    validate_summary(summary, rows=rows, config=config)
    return summary


def _plan_aggregate(rows: list[Mapping[str, Any]]) -> dict[str, object]:
    return {
        "repeat_count": len(rows),
        "median_peak_allocated_delta_bytes": statistics.median(
            int(row["peak_allocated_delta_bytes"]) for row in rows
        ),
        "median_peak_reserved_delta_bytes": statistics.median(
            int(row["peak_reserved_delta_bytes"]) for row in rows
        ),
        "median_latency_ms": statistics.median(
            float(row["latency_median_ms"]) for row in rows
        ),
    }


def validate_summary(
    summary: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    config: ProtocolConfig,
) -> None:
    """Reject summary drift, threshold changes, or claim inflation."""

    if (
        summary.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or summary.get("protocol") != config.to_dict()
        or summary.get("row_count") != len(rows)
        or not isinstance(summary.get("gates"), Mapping)
    ):
        raise ValueError("CUDA memory summary identity differs")
    gates = _mapping(summary["gates"], "summary gates")
    claimed = all(value is True for value in gates.values())
    expected_status = "validated" if claimed else "no_go"
    if (
        summary.get("performance_claimed") is not claimed
        or summary.get("status") != expected_status
    ):
        raise ValueError("CUDA memory summary claim contradicts its gates")
    expected = aggregate_rows_unchecked(rows, config=config)
    for field in (
        "plan_counts",
        "retain",
        "reuse",
        "allocated_reduction",
        "latency_ratio",
        "gates",
    ):
        if summary.get(field) != expected[field]:
            raise ValueError(f"CUDA memory summary derived field differs: {field}")


def aggregate_rows_unchecked(
    rows: Sequence[Mapping[str, Any]], *, config: ProtocolConfig
) -> dict[str, object]:
    """Recompute derived fields without recursively validating a summary."""

    by_plan = {plan: [row for row in rows if row["plan"] == plan] for plan in PLAN_IDS}
    retain = _plan_aggregate(by_plan["retain"])
    reuse = _plan_aggregate(by_plan["reuse"])
    retain_allocated = _number(retain["median_peak_allocated_delta_bytes"])
    reuse_allocated = _number(reuse["median_peak_allocated_delta_bytes"])
    allocated_reduction = 1.0 - reuse_allocated / retain_allocated
    latency_ratio = _number(reuse["median_latency_ms"]) / _number(
        retain["median_latency_ms"]
    )
    identities_match = (
        all(
            len({str(row[field]) for row in rows}) == 1
            for field in (
                "model_sha256",
                "intermediate_bounds_sha256",
                "lower_sha256",
                "upper_sha256",
            )
        )
        and len(
            {
                (
                    str(_mapping(row["ir_hashes"], "IR hashes")["bound_module_hash"]),
                    str(_mapping(row["ir_hashes"], "IR hashes")["plan_template_hash"]),
                )
                for row in rows
            }
        )
        == 1
    )
    gates = {
        "complete_fresh_process_matrix": len(rows) == config.repeats * 2,
        "stable_environment_identity": (
            len({canonical_json(row["environment"]) for row in rows}) == 1
        ),
        "all_worker_correctness_passed": all(
            _mapping(row["comparison"], "comparison").get("allclose") is True
            for row in rows
        ),
        "cross_plan_semantic_identity": identities_match,
        "allocated_reduction_gte_threshold": (
            allocated_reduction >= config.minimum_allocated_reduction
        ),
        "latency_ratio_lte_threshold": (latency_ratio <= config.maximum_latency_ratio),
    }
    return {
        "plan_counts": {plan: len(plan_rows) for plan, plan_rows in by_plan.items()},
        "retain": retain,
        "reuse": reuse,
        "allocated_reduction": allocated_reduction,
        "latency_ratio": latency_ratio,
        "gates": gates,
    }


def generate_artifact(
    artifact_dir: Path,
    *,
    model: Path,
    source_artifact_dir: Path,
    config: ProtocolConfig,
) -> dict[str, object]:
    """Run the frozen matrix and write an immutable CUDA benchmark artifact."""

    probe = probe_cuda_environment()
    if probe["status"] != "ready":
        raise CudaEnvironmentUnavailable("CUDA benchmark environment is unavailable")
    if config.device_index >= cast(int, probe["device_count"]):
        raise CudaEnvironmentUnavailable("configured CUDA device is unavailable")
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    rows: list[Mapping[str, Any]] = []
    for repeat in range(config.repeats):
        order = ("retain", "reuse") if repeat % 2 == 0 else ("reuse", "retain")
        for plan in order:
            rows.append(
                _run_worker_subprocess(
                    model=model,
                    source_artifact_dir=source_artifact_dir,
                    plan=plan,
                    repeat=repeat,
                    config=config,
                )
            )
    summary = aggregate_rows(rows, config=config)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_path = artifact_dir / "raw.jsonl"
    raw_path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
    (artifact_dir / "summary.json").write_text(
        canonical_json(summary, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": summary["status"],
        "performance_claimed": summary["performance_claimed"],
        "files": {name: file_sha256(artifact_dir / name) for name in ARTIFACT_FILES},
    }
    (artifact_dir / "manifest.json").write_text(
        canonical_json(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def replay_artifact(artifact_dir: Path, *, config: ProtocolConfig) -> dict[str, object]:
    """Verify artifact digests and recompute every summary field from raw rows."""

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("CUDA memory artifact manifest schema differs")
    files = _mapping(manifest.get("files"), "CUDA memory artifact files")
    if set(files) != set(ARTIFACT_FILES):
        raise ValueError("CUDA memory artifact file set differs")
    for name in ARTIFACT_FILES:
        if files[name] != file_sha256(artifact_dir / name):
            raise ValueError(f"CUDA memory artifact digest differs: {name}")
    rows = [
        json.loads(line)
        for line in (artifact_dir / "raw.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    stored = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    actual = aggregate_rows(rows, config=config)
    if canonical_json(stored) != canonical_json(actual):
        raise ValueError("CUDA memory artifact summary semantic replay differs")
    if (
        manifest.get("status") != stored["status"]
        or manifest.get("performance_claimed") is not stored["performance_claimed"]
    ):
        raise ValueError("CUDA memory artifact manifest/summary claim differs")
    return stored


def _run_worker_subprocess(
    *,
    model: Path,
    source_artifact_dir: Path,
    plan: str,
    repeat: int,
    config: ProtocolConfig,
) -> Mapping[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--model",
        str(model),
        "--source-artifact-dir",
        str(source_artifact_dir),
        "--plan",
        plan,
        "--repeat",
        str(repeat),
        "--repeats",
        str(config.repeats),
        "--warmup",
        str(config.warmup_iterations),
        "--iterations",
        str(config.measured_iterations),
        "--minimum-allocated-reduction",
        str(config.minimum_allocated_reduction),
        "--maximum-latency-ratio",
        str(config.maximum_latency_ratio),
        "--device-index",
        str(config.device_index),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"CUDA memory worker failed ({plan}/{repeat}): {completed.stderr.strip()}"
        )
    row = json.loads(completed.stdout.strip().splitlines()[-1])
    validate_worker_row(row, config=config)
    return row


def _move_tensors(value: Any, *, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensors(item, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device=device) for item in value)
    return value


def _number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError("CUDA benchmark aggregate field is not numeric")
    return float(value)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _config_from_args(args: argparse.Namespace) -> ProtocolConfig:
    config = ProtocolConfig(
        repeats=args.repeats,
        warmup_iterations=args.warmup,
        measured_iterations=args.iterations,
        minimum_allocated_reduction=args.minimum_allocated_reduction,
        maximum_latency_ratio=args.maximum_latency_ratio,
        device_index=args.device_index,
    )
    config.validate()
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--artifact-dir", type=Path)
    probe_replay = subparsers.add_parser("probe-replay")
    probe_replay.add_argument("--artifact-dir", type=Path, required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--source-artifact-dir", type=Path, required=True)
    worker.add_argument("--plan", choices=tuple(PLAN_IDS), required=True)
    worker.add_argument("--repeat", type=int, required=True)
    for command in (worker,):
        command.add_argument("--repeats", type=int, default=5)
        command.add_argument("--warmup", type=int, default=5)
        command.add_argument("--iterations", type=int, default=20)
        command.add_argument("--minimum-allocated-reduction", type=float, default=0.20)
        command.add_argument("--maximum-latency-ratio", type=float, default=1.20)
        command.add_argument("--device-index", type=int, default=0)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--source-artifact-dir", type=Path, required=True)
    generate.add_argument("--artifact-dir", type=Path, required=True)
    generate.add_argument("--repeats", type=int, default=5)
    generate.add_argument("--warmup", type=int, default=5)
    generate.add_argument("--iterations", type=int, default=20)
    generate.add_argument("--minimum-allocated-reduction", type=float, default=0.20)
    generate.add_argument("--maximum-latency-ratio", type=float, default=1.20)
    generate.add_argument("--device-index", type=int, default=0)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    replay.add_argument("--repeats", type=int, default=5)
    replay.add_argument("--warmup", type=int, default=5)
    replay.add_argument("--iterations", type=int, default=20)
    replay.add_argument("--minimum-allocated-reduction", type=float, default=0.20)
    replay.add_argument("--maximum-latency-ratio", type=float, default=1.20)
    replay.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    if args.command == "probe":
        probe = (
            probe_cuda_environment()
            if args.artifact_dir is None
            else write_environment_probe_artifact(args.artifact_dir.resolve())
        )
        print(canonical_json(probe))
        if probe["status"] != "ready":
            raise SystemExit(2)
        return
    if args.command == "probe-replay":
        print(
            canonical_json(
                replay_environment_probe_artifact(args.artifact_dir.resolve())
            )
        )
        return
    config = _config_from_args(args)
    if args.command == "worker":
        row = run_worker(
            model=args.model.resolve(),
            source_artifact_dir=args.source_artifact_dir.resolve(),
            plan=args.plan,
            repeat=args.repeat,
            config=config,
        )
        print(canonical_json(row))
    elif args.command == "generate":
        summary = generate_artifact(
            args.artifact_dir.resolve(),
            model=args.model.resolve(),
            source_artifact_dir=args.source_artifact_dir.resolve(),
            config=config,
        )
        print(canonical_json(summary))
    else:
        print(
            canonical_json(replay_artifact(args.artifact_dir.resolve(), config=config))
        )


if __name__ == "__main__":
    try:
        _main()
    except CudaEnvironmentUnavailable as error:
        print(
            canonical_json({"status": "environment_unavailable", "error": str(error)})
        )
        raise SystemExit(2) from error
