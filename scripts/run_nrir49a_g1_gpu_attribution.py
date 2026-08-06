#!/usr/bin/env python3
"""Generate or replay NRIR49A G1 GPU selected-CROWN attribution evidence."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=too-many-arguments,import-outside-toplevel,protected-access
# pylint: disable=duplicate-code,cell-var-from-loop,too-many-positional-arguments
# pylint: disable=missing-function-docstring,line-too-long

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import torch

ARTIFACT_SCHEMA_VERSION = "boundflow.nrir49a-g1-gpu-attribution-artifact/v1"
ENVIRONMENT_SCHEMA_VERSION = "boundflow.nrir49a-g1-environment/v1"
QUERY_SCHEMA_VERSION = "boundflow.nrir49a-g1-query/v1"
WORKER_SCHEMA_VERSION = "boundflow.nrir49a-g1-worker/v1"
QUEUE_SCHEMA_VERSION = "boundflow.nrir49a-g1-queue/v1"
COMPLETE_SCHEMA_VERSION = "boundflow.nrir49a-g1-complete/v1"
SUMMARY_SCHEMA_VERSION = "boundflow.nrir49a-g1-summary/v1"

ARTIFACT_FILES = (
    "environment.json",
    "queries.jsonl",
    "results_raw.jsonl",
    "normalized.jsonl",
    "summary.json",
    "replay_stdout.txt",
    "failure_rows.jsonl",
    "README.md",
)
MANIFEST_FILE = "manifest.json"
VNNCOMP_COMMIT = "90419aadcf06cf543ce5c1706cae1059dc9fa6cf"
MODEL_RELATIVE_PATH = "benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
PROPERTY_RELATIVE_PATH = (
    "benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/"
    "resnet2b_pgd_filtered/prop_0_eps_0.008.vnnlib"
)
MODEL_SHA256 = "791aa24d77917ecda16809fbbd48e7739616f88ebf74cf358b2d1bf911dc4a6d"
PROPERTY_SHA256 = "89edf0665d74397670d0562d513db694a49a84edaf5cf3d64c9c6fa63c3769ff"
CLAUSES = (2, 3)
CHUNKS = (8, 16, 128, 32, 64)
DEFAULT_CHUNK = 32
REPEAT_COUNT = 5
REQUIRED_NODES = 31
REQUIRED_SIBLING_GROUPS = 15
REQUIRED_MAX_DEPTH = 4
EXPECTED_WORST_LOWERS = {2: -35.53092575073242, 3: -30.258447647094727}
MAXIMUM_PERTURBATION_RATIO = 1.05
MINIMUM_QUEUE_SHARE = 0.20
MAXIMUM_REQUIRED_REGION_SPEEDUP = 10.0
QUEUE_TARGET_SPEEDUP = 1.20
COMPLETE_TARGET_SPEEDUP = 1.15
PHYSICAL_MEMORY_RATIO = 0.80
SEMANTIC_BATCH_MAXIMUM = 1
DISPLAY_PROCESS_MEMORY_LIMIT_MIB = 64
ALLOWED_DISPLAY_PROCESS_NAMES = frozenset({"kwin_wayland"})
ATOL = 2e-4
RTOL = 2e-4
TORCH_THREADS = 8
WORKER_TIMEOUT_SECONDS = 15 * 60
DEFAULT_ARTIFACT_DIR = Path(
    "artifacts/nrir49a-g1-gpu-attribution/"
    "resnet2b-prop0-clauses2-3-rtx4060-five-repeat-v1"
)
DEFAULT_WORKER_CACHE_DIR = Path(".tmp/nrir49a-g1-workers-v1")


def canonical_json(value: object, *, indent: int | None = None) -> str:
    """Encode finite deterministic JSON."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row must be an object: {path}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(command: Sequence[str], *, cwd: Path | None = None) -> dict[str, object]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip()[-16384:],
            "stderr": completed.stderr.strip()[-16384:],
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": 127, "stdout": "", "stderr": str(error)}


def _git_value(root: Path, *args: str) -> str | None:
    result = _run(("git", *args), cwd=root)
    return str(result["stdout"]) if result["returncode"] == 0 else None


def latin_chunk_order(repeat_index: int) -> tuple[int, ...]:
    """Return the preregistered cyclic Latin row."""

    if repeat_index not in range(REPEAT_COUNT):
        raise ValueError("NRIR49A repeat index differs")
    shift = repeat_index % len(CHUNKS)
    return CHUNKS[shift:] + CHUNKS[:shift]


def required_region_speedup(share: float, target: float) -> float | None:
    """Invert Amdahl; return None when even infinite region speed is insufficient."""

    if not 0.0 <= share <= 1.0 or target <= 1.0 or not math.isfinite(target):
        raise ValueError("Amdahl input differs")
    denominator = share + 1.0 / target - 1.0
    if denominator <= 0.0:
        return None
    return share / denominator


def projected_scope_speedup(share: float, region_speedup: float) -> float:
    if not 0.0 <= share <= 1.0 or region_speedup < 1.0:
        raise ValueError("Amdahl projection input differs")
    return 1.0 / ((1.0 - share) + share / region_speedup)


def _median(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("median input differs")
    return float(statistics.median(values))


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return value


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


def module_inventory(module: Any) -> dict[str, object]:
    """Freeze the backward-relevant primal topology without tensor contents."""

    task = module.get_entry_task()
    consumers: dict[str, int] = {}
    for op in task.ops:
        for value_id in op.inputs:
            consumers[value_id] = consumers.get(value_id, 0) + 1
    params = _mapping(module.bindings.get("params"), "module params")
    operations = []
    for ordinal, op in enumerate(task.ops):
        attrs = {
            key: value
            for key, value in op.attrs.items()
            if isinstance(value, (str, int, float, bool, type(None), tuple, list))
        }
        parameter_signatures = {
            name: {
                "shape": list(params[name].shape),
                "dtype": str(params[name].dtype),
                "device": str(params[name].device),
            }
            for name in op.inputs
            if name in params and torch.is_tensor(params[name])
        }
        operations.append(
            {
                "ordinal": ordinal,
                "name": op.name,
                "op_type": op.op_type,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
                "input_consumer_counts": {
                    name: consumers.get(name, 0) for name in op.inputs
                },
                "output_fanout": {name: consumers.get(name, 0) for name in op.outputs},
                "attrs": attrs,
                "parameter_signatures": parameter_signatures,
            }
        )
    inventory: dict[str, object] = {
        "op_sequence": [op.op_type for op in task.ops],
        "operations": operations,
        "join_ops": [op.name for op in task.ops if op.op_type in {"add", "concat"}],
        "layout_ops": [
            op.name for op in task.ops if op.op_type in {"flatten", "reshape"}
        ],
    }
    inventory["inventory_hash"] = canonical_hash(inventory)
    return inventory


def selected_call_geometry(
    *,
    relu_pre: Mapping[str, Any],
    targets: Sequence[Any],
    chunk_size: int,
) -> dict[str, object]:
    """Derive exact ragged objective/materialization geometry for one call."""

    if chunk_size < 1:
        raise ValueError("selected-CROWN chunk size differs")
    grouped: dict[str, list[int]] = {}
    for target in targets:
        grouped.setdefault(str(target.relu_input), []).append(int(target.neuron_index))
    segments: list[dict[str, Any]] = []
    total_one_hot = 0
    total_indices = 0
    total_outputs = 0
    for name in relu_pre:
        if name not in grouped:
            continue
        lower = relu_pre[name].lower
        indices = grouped[name]
        numel = int(lower[0].numel())
        element_size = int(lower.element_size())
        chunk_lengths = [
            min(chunk_size, len(indices) - start)
            for start in range(0, len(indices), chunk_size)
        ]
        one_hot = sum(length * numel * element_size for length in chunk_lengths)
        index_bytes = sum(length * 8 for length in chunk_lengths)
        output_bytes = 2 * len(indices) * element_size
        total_one_hot += one_hot
        total_indices += index_bytes
        total_outputs += output_bytes
        segments.append(
            {
                "relu_input": name,
                "indices": indices,
                "target_count": len(indices),
                "source_numel": numel,
                "source_shape": list(lower.shape),
                "dtype": str(lower.dtype),
                "device": str(lower.device),
                "chunk_lengths": chunk_lengths,
                "chunk_count": len(chunk_lengths),
                "objective_shapes": [[length, numel] for length in chunk_lengths],
                "one_hot_bytes": one_hot,
                "index_bytes": index_bytes,
                "output_bytes": output_bytes,
            }
        )
    if sum(_integer(item["target_count"], "target count") for item in segments) != len(
        targets
    ):
        raise ValueError("selected-CROWN geometry target coverage differs")
    return {
        "target_count": len(targets),
        "relu_segment_count": len(segments),
        "segments": segments,
        "chunk_count": sum(
            _integer(item["chunk_count"], "chunk count") for item in segments
        ),
        "one_hot_bytes": total_one_hot,
        "index_bytes": total_indices,
        "output_bytes": total_outputs,
    }


@dataclass
class _PendingCall:
    record: dict[str, object]
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event


class _SelectedCrownTracker:
    """Harness-only wrapper; production Plan and policy remain unchanged."""

    def __init__(self, original: Callable[..., Any], inventory: Mapping[str, Any]):
        self.original = original
        self.inventory_hash = inventory["inventory_hash"]
        self.enabled = True
        self.effective_chunk = DEFAULT_CHUNK
        self.scope = "unbound"
        self.clause = -1
        self.pending: list[_PendingCall] = []

    def mark(self) -> int:
        return len(self.pending)

    def wrap(self, module: Any, input_spec: Any, **kwargs: Any) -> Any:
        requested_chunk = int(kwargs["chunk_size"])
        if not self.enabled:
            return self.original(module, input_spec, **kwargs)
        effective_chunk = int(self.effective_chunk)
        geometry = selected_call_geometry(
            relu_pre=kwargs["relu_pre"],
            targets=kwargs["targets"],
            chunk_size=effective_chunk,
        )
        before_allocated = int(torch.cuda.memory_allocated())
        before_reserved = int(torch.cuda.memory_reserved())
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        started_ns = time.perf_counter_ns()
        effective_kwargs = dict(kwargs)
        effective_kwargs["chunk_size"] = effective_chunk
        output = self.original(module, input_spec, **effective_kwargs)
        host_wall_ns = time.perf_counter_ns() - started_ns
        end_event.record()
        output_schema = {
            name: {
                "indices": list(indices),
                "lower_shape": list(lower.shape),
                "upper_shape": list(upper.shape),
                "dtype": str(lower.dtype),
                "device": str(lower.device),
            }
            for name, (indices, lower, upper) in output.items()
        }
        record: dict[str, object] = {
            "scope": self.scope,
            "clause": self.clause,
            "call_ordinal": len(self.pending),
            "requested_production_chunk": requested_chunk,
            "effective_harness_chunk": effective_chunk,
            "geometry": geometry,
            "output_schema": output_schema,
            "host_wall_ns": host_wall_ns,
            "allocated_before": before_allocated,
            "allocated_after": int(torch.cuda.memory_allocated()),
            "reserved_before": before_reserved,
            "reserved_after": int(torch.cuda.memory_reserved()),
            "module_inventory_hash": self.inventory_hash,
        }
        self.pending.append(_PendingCall(record, start_event, end_event))
        return output

    def finalize_since(self, start: int) -> list[dict[str, object]]:
        rows = []
        for pending in self.pending[start:]:
            record = dict(pending.record)
            record["device_ns"] = int(
                round(pending.start_event.elapsed_time(pending.end_event) * 1_000_000)
            )
            rows.append(record)
        return rows


def _environment() -> dict[str, object]:
    smi = _run(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        )
    )
    compute = _run(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("NRIR49A CUDA environment is unavailable")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    environment: dict[str, object] = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "python": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": properties.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": int(properties.total_memory),
        "nvidia_smi": smi,
        "compute_processes": compute,
        "performance_claimed": False,
    }
    if (
        torch.cuda.get_device_capability(device) != (8, 9)
        or int(properties.total_memory) < 8_000_000_000
        or smi["returncode"] != 0
    ):
        raise RuntimeError("NRIR49A target GPU identity differs")
    environment["compute_process_classification"] = classify_compute_processes(
        environment
    )
    environment["environment_hash"] = canonical_hash(environment)
    return environment


def classify_compute_processes(
    environment: Mapping[str, Any], *, own_pid: int | None = None
) -> dict[str, list[dict[str, object]]]:
    """Classify target-GPU processes with one bounded display exception."""

    compute = _mapping(environment.get("compute_processes"), "compute processes")
    if compute.get("returncode") != 0:
        raise RuntimeError("NRIR49A compute-process inventory is unavailable")
    process_id = str(os.getpid() if own_pid is None else own_pid)
    admitted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for row in str(compute.get("stdout", "")).splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 3 or not fields[0]:
            raise RuntimeError("NRIR49A compute-process row differs")
        try:
            used_memory_mib = int(fields[2])
        except ValueError as error:
            raise RuntimeError("NRIR49A compute-process memory differs") from error
        record: dict[str, object] = {
            "pid": int(fields[0]),
            "process_name": fields[1],
            "used_memory_mib": used_memory_mib,
        }
        basename = Path(fields[1]).name
        if fields[0] == process_id or (
            basename in ALLOWED_DISPLAY_PROCESS_NAMES
            and used_memory_mib <= DISPLAY_PROCESS_MEMORY_LIMIT_MIB
        ):
            admitted.append(record)
        else:
            rejected.append(record)
    return {"admitted": admitted, "rejected": rejected}


def _reject_other_compute_processes(environment: Mapping[str, Any]) -> None:
    """Fail closed when a non-display process is already using the target GPU."""

    inventory = classify_compute_processes(environment)
    if inventory["rejected"]:
        raise RuntimeError(
            "NRIR49A target GPU has other compute processes: "
            + canonical_json(inventory["rejected"])
        )


def _load_gpu_runtime(model: Path, property_path: Path) -> tuple[Any, Any, Any, Any]:
    from scripts.run_typed_hard_clause_escalation_artifact import _load_query_runtime
    from boundflow.runtime.task_executor import InputSpec

    query, tensors, module, cpu_input = _load_query_runtime(
        model, property_path, "cifar10_resnet:000"
    )
    device = torch.device("cuda:0")
    module.bindings = _move_tensors(module.bindings, device=device)
    tensors = replace(
        tensors,
        input_lower=tensors.input_lower.to(device),
        input_upper=tensors.input_upper.to(device),
        linear_spec_c=tensors.linear_spec_c.to(device),
        thresholds=tensors.thresholds.to(device),
    )
    input_spec = InputSpec.box(
        value_name=cpu_input.value_name,
        lower=tensors.input_lower,
        upper=tensors.input_upper,
    )
    return query, tensors, module, input_spec


def _build_roots(
    module: Any,
    input_spec: Any,
    objectives: Mapping[int, torch.Tensor],
    *,
    repeat_index: int,
    tracker: _SelectedCrownTracker,
) -> tuple[Any, dict[int, Any]]:
    from boundflow.ir.refinement import NativeIntermediateRefinementPolicyIR
    from boundflow.runtime.native_intermediate_refinement import (
        compile_native_intermediate_refinement_program,
        execute_native_intermediate_refinement_program,
    )

    tracker.scope = "complete_root_shared"
    tracker.clause = -1
    shared_program = compile_native_intermediate_refinement_program(
        module,
        input_spec,
        policy=NativeIntermediateRefinementPolicyIR(
            passes=1, max_neurons_per_relu=128, backward_chunk_size=DEFAULT_CHUNK
        ),
        plan_id=f"nrir49a:r{repeat_index}:shared-root",
    )
    shared = execute_native_intermediate_refinement_program(
        shared_program, module, input_spec
    )
    roots = {}
    for clause in CLAUSES:
        tracker.scope = "complete_root_objective"
        tracker.clause = clause
        program = compile_native_intermediate_refinement_program(
            module,
            input_spec,
            policy=NativeIntermediateRefinementPolicyIR(
                passes=1,
                max_neurons_per_relu=128,
                backward_chunk_size=DEFAULT_CHUNK,
                candidate_policy_id="objective_influence_width_per_relu_v1",
            ),
            plan_id=f"nrir49a:r{repeat_index}:c{clause}:objective-root",
            linear_spec_C=objectives[clause],
            source_refinement_execution=shared,
        )
        roots[clause] = execute_native_intermediate_refinement_program(
            program, module, input_spec
        )
    return shared, roots


def _semantics(execution: Any) -> dict[str, object]:
    from scripts.run_prepared_intermediate_refinement_formal import _semantic_tables

    value = _semantic_tables(execution)
    return {"hash": canonical_hash(value), "tables": value}


def _queue_frontier(execution: Any) -> dict[str, Any]:
    from scripts.run_objective_ancestral_queue_artifact import _active_frontier

    return _active_frontier(execution.queue.trace)


def _execute_queue(
    *,
    plan: Any,
    module: Any,
    input_spec: Any,
    objective: torch.Tensor,
    threshold: torch.Tensor,
    root: Any,
    optimizer_policy: Any,
    branch_policy: Any,
    tracker: _SelectedCrownTracker,
    repeat_index: int,
    clause: int,
    chunk: int,
    order_position: int,
    mode: str,
    reset_peak: bool,
) -> dict[str, object]:
    from boundflow.runtime.native_parametric_optimizer import (
        NativeParametricOptimizerTemplateCache,
    )
    from boundflow.runtime.native_prepared_objective_branch_shared_production_queue import (
        execute_native_prepared_objective_branch_shared_production_queue,
    )

    if mode not in {"profile", "control", "complete"}:
        raise ValueError("NRIR49A queue mode differs")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    tracker.enabled = mode != "control"
    tracker.effective_chunk = chunk
    tracker.scope = "complete_queue" if mode == "complete" else "chunk_queue"
    tracker.clause = clause
    call_start = tracker.mark()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    started_ns = time.perf_counter_ns()
    execution = execute_native_prepared_objective_branch_shared_production_queue(
        plan,
        module,
        input_spec,
        linear_spec_C=objective,
        threshold=threshold,
        root_refinement=root,
        optimizer_policy=optimizer_policy,
        branch_policy=branch_policy,
        compiler_cache=NativeParametricOptimizerTemplateCache(),
        query_id=f"nrir49a:r{repeat_index}:c{clause}",
    )
    end_event.record()
    torch.cuda.synchronize()
    synchronized_wall_ns = time.perf_counter_ns() - started_ns
    calls: list[dict[str, object]] = (
        [] if mode == "control" else tracker.finalize_since(call_start)
    )
    frontier = _queue_frontier(execution)
    semantic = _semantics(execution)
    row: dict[str, object] = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "repeat_index": repeat_index,
        "clause": clause,
        "chunk": chunk,
        "order_position": order_position,
        "mode": mode,
        "query_id": f"nrir49a:r{repeat_index}:c{clause}",
        "accepted_nodes": frontier["evaluated_nodes"],
        "sibling_group_count": len(execution.batch_commits) - 1,
        "maximum_depth": frontier["maximum_depth"],
        "worst_active_lower": frontier["worst_active_lower"],
        "queue_internal_elapsed_ns": execution.trace.queue_elapsed_ns,
        "queue_stream_ns": int(round(start_event.elapsed_time(end_event) * 1_000_000)),
        "synchronized_wall_ns": synchronized_wall_ns,
        "selected_device_ns": sum(
            _integer(call["device_ns"], "selected device ns") for call in calls
        ),
        "selected_host_wall_ns": sum(
            _integer(call["host_wall_ns"], "selected host wall ns") for call in calls
        ),
        "selected_call_count": len(calls),
        "baseline_allocated": baseline_allocated,
        "baseline_reserved": baseline_reserved,
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
        "semantics_hash": semantic["hash"],
        "semantics": semantic["tables"],
        "calls": calls,
        "performance_claimed": False,
    }
    row["row_hash"] = canonical_hash(row)
    return row


def _compile_plans(
    module: Any,
    input_spec: Any,
    tensors: Any,
    roots: Mapping[int, Any],
    optimizer_policy: Any,
    branch_policy: Any,
    *,
    repeat_index: int,
) -> dict[int, Any]:
    from boundflow.runtime.native_objective_branch_shared_evaluator import (
        compile_native_objective_branch_shared_plan,
    )

    plans = {}
    for clause in CLAUSES:
        objective = tensors.linear_spec_c[:, clause : clause + 1, :].contiguous()
        threshold = tensors.thresholds[clause : clause + 1].contiguous()
        plans[clause] = compile_native_objective_branch_shared_plan(
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=roots[clause],
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            plan_id=f"nrir49a:r{repeat_index}:c{clause}",
        )
    return plans


def _representative_profiler(
    *,
    plan: Any,
    module: Any,
    input_spec: Any,
    objective: torch.Tensor,
    threshold: torch.Tensor,
    root: Any,
    optimizer_policy: Any,
    branch_policy: Any,
    tracker: _SelectedCrownTracker,
    repeat_index: int,
) -> dict[str, object]:
    """Capture one non-timing CUPTI/PyTorch profile after the formal matrix."""

    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as profiler:
        row = _execute_queue(
            plan=plan,
            module=module,
            input_spec=input_spec,
            objective=objective,
            threshold=threshold,
            root=root,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            tracker=tracker,
            repeat_index=repeat_index,
            clause=CLAUSES[0],
            chunk=DEFAULT_CHUNK,
            order_position=-2,
            mode="control",
            reset_peak=True,
        )
    events = []
    for event in profiler.key_averages():
        events.append(
            {
                "key": event.key,
                "count": int(event.count),
                "device_type": str(event.device_type),
                "self_cpu_time_us": float(event.self_cpu_time_total),
                "self_device_time_us": float(event.self_device_time_total),
                "self_cpu_memory_bytes": int(event.self_cpu_memory_usage),
                "self_device_memory_bytes": int(event.self_device_memory_usage),
            }
        )
    cuda_events = [item for item in events if item["device_type"] == "DeviceType.CUDA"]
    cpu_events = [item for item in events if item["device_type"] == "DeviceType.CPU"]
    result: dict[str, object] = {
        "scope": "representative-non-timing-clause2-default32",
        "excluded_from_timing_summary": True,
        "kernel_count": sum(int(item["count"]) for item in cuda_events),
        "runtime_launch_api_count": sum(
            int(item["count"])
            for item in cpu_events
            if str(item["key"]).startswith("cudaLaunch")
        ),
        "synchronization_api_count": sum(
            int(item["count"])
            for item in cpu_events
            if "synchronize" in str(item["key"]).lower()
        ),
        "memory_event_count": sum(
            int(item["count"])
            for item in events
            if "memory" in str(item["key"]).lower()
        ),
        "top_cuda_events": sorted(
            cuda_events,
            key=lambda item: float(item["self_device_time_us"]),
            reverse=True,
        )[:25],
        "top_cpu_events": sorted(
            cpu_events,
            key=lambda item: float(item["self_cpu_time_us"]),
            reverse=True,
        )[:25],
        "queue_row": row,
        "performance_claimed": False,
    }
    result["profile_hash"] = canonical_hash(result)
    return result


def run_worker(
    *,
    benchmark_root: Path,
    repeat_index: int,
) -> dict[str, object]:  # pylint: disable=too-many-locals,too-many-statements
    """Run one formal fresh-process Latin row."""

    from boundflow.runtime import native_intermediate_refinement as refinement
    from boundflow.runtime.native_objective_branch_score import (
        NativeObjectiveBranchPolicy,
    )
    from scripts.run_typed_hard_clause_escalation_artifact import _policies

    if repeat_index not in range(REPEAT_COUNT):
        raise ValueError("NRIR49A worker repeat differs")
    if _git_value(benchmark_root, "rev-parse", "HEAD") != VNNCOMP_COMMIT:
        raise ValueError("NRIR49A VNN-COMP commit differs")
    model = benchmark_root / MODEL_RELATIVE_PATH
    property_path = benchmark_root / PROPERTY_RELATIVE_PATH
    if (
        file_sha256(model) != MODEL_SHA256
        or file_sha256(property_path) != PROPERTY_SHA256
    ):
        raise ValueError("NRIR49A workload digest differs")
    torch.set_num_threads(TORCH_THREADS)
    torch.cuda.set_device(0)
    environment = _environment()
    _reject_other_compute_processes(environment)
    _query, tensors, module, input_spec = _load_gpu_runtime(model, property_path)
    inventory = module_inventory(module)
    _search_policy, optimizer_policy = _policies()
    branch_policy = NativeObjectiveBranchPolicy()
    objectives = {
        clause: tensors.linear_spec_c[:, clause : clause + 1, :].contiguous()
        for clause in CLAUSES
    }
    thresholds = {
        clause: tensors.thresholds[clause : clause + 1].contiguous()
        for clause in CLAUSES
    }
    original_selected = refinement._run_selected_crown
    tracker = _SelectedCrownTracker(original_selected, inventory)
    refinement._run_selected_crown = tracker.wrap
    try:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        complete_baseline_allocated = int(torch.cuda.memory_allocated())
        complete_baseline_reserved = int(torch.cuda.memory_reserved())
        complete_call_start = tracker.mark()
        complete_start = torch.cuda.Event(enable_timing=True)
        complete_end = torch.cuda.Event(enable_timing=True)
        complete_start.record()
        complete_started_ns = time.perf_counter_ns()
        _shared, roots = _build_roots(
            module,
            input_spec,
            objectives,
            repeat_index=repeat_index,
            tracker=tracker,
        )
        plans = _compile_plans(
            module,
            input_spec,
            tensors,
            roots,
            optimizer_policy,
            branch_policy,
            repeat_index=repeat_index,
        )
        complete_queues = []
        for clause in CLAUSES:
            complete_queues.append(
                _execute_queue(
                    plan=plans[clause],
                    module=module,
                    input_spec=input_spec,
                    objective=objectives[clause],
                    threshold=thresholds[clause],
                    root=roots[clause],
                    optimizer_policy=optimizer_policy,
                    branch_policy=branch_policy,
                    tracker=tracker,
                    repeat_index=repeat_index,
                    clause=clause,
                    chunk=DEFAULT_CHUNK,
                    order_position=-1,
                    mode="complete",
                    reset_peak=False,
                )
            )
        complete_end.record()
        torch.cuda.synchronize()
        complete_calls = tracker.finalize_since(complete_call_start)
        complete: dict[str, object] = {
            "schema_version": COMPLETE_SCHEMA_VERSION,
            "repeat_index": repeat_index,
            "chunk": DEFAULT_CHUNK,
            "stream_ns": int(
                round(complete_start.elapsed_time(complete_end) * 1_000_000)
            ),
            "synchronized_wall_ns": time.perf_counter_ns() - complete_started_ns,
            "selected_all_device_ns": sum(
                _integer(call["device_ns"], "complete selected device ns")
                for call in complete_calls
            ),
            "selected_child_device_ns": sum(
                _integer(call["device_ns"], "complete child selected device ns")
                for call in complete_calls
                if call["scope"] == "complete_queue"
            ),
            "baseline_allocated": complete_baseline_allocated,
            "baseline_reserved": complete_baseline_reserved,
            "peak_allocated": int(torch.cuda.max_memory_allocated()),
            "peak_reserved": int(torch.cuda.max_memory_reserved()),
            "calls": complete_calls,
            "queue_rows": complete_queues,
            "performance_claimed": False,
        }
        complete["row_hash"] = canonical_hash(complete)

        profile_rows: list[dict[str, object]] = []
        control_rows: list[dict[str, object]] = []
        for position, chunk in enumerate(latin_chunk_order(repeat_index)):
            for clause in CLAUSES:
                common = {
                    "plan": plans[clause],
                    "module": module,
                    "input_spec": input_spec,
                    "objective": objectives[clause],
                    "threshold": thresholds[clause],
                    "root": roots[clause],
                    "optimizer_policy": optimizer_policy,
                    "branch_policy": branch_policy,
                    "tracker": tracker,
                    "repeat_index": repeat_index,
                    "clause": clause,
                    "chunk": chunk,
                    "order_position": position,
                    "reset_peak": True,
                }
                if chunk == DEFAULT_CHUNK and repeat_index % 2 == 0:
                    control_rows.append(_execute_queue(mode="control", **common))
                profile_rows.append(_execute_queue(mode="profile", **common))
                if chunk == DEFAULT_CHUNK and repeat_index % 2 == 1:
                    control_rows.append(_execute_queue(mode="control", **common))
        representative_profiler = None
        if repeat_index == 0:
            representative_profiler = _representative_profiler(
                plan=plans[CLAUSES[0]],
                module=module,
                input_spec=input_spec,
                objective=objectives[CLAUSES[0]],
                threshold=thresholds[CLAUSES[0]],
                root=roots[CLAUSES[0]],
                optimizer_policy=optimizer_policy,
                branch_policy=branch_policy,
                tracker=tracker,
                repeat_index=repeat_index,
            )
    finally:
        refinement._run_selected_crown = original_selected
    worker: dict[str, object] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "repeat_index": repeat_index,
        "chunk_order": list(latin_chunk_order(repeat_index)),
        "code_revision": _code_revision(),
        "environment": environment,
        "module_inventory": inventory,
        "profile_rows": profile_rows,
        "control_rows": control_rows,
        "complete": complete,
        "representative_profiler": representative_profiler,
        "performance_claimed": False,
    }
    worker["worker_hash"] = canonical_hash(worker)
    validate_worker(worker)
    return worker


def _validate_queue_row(row: Mapping[str, Any], *, profile: bool) -> None:
    semantic = {key: value for key, value in row.items() if key != "row_hash"}
    calls = row.get("calls")
    if (
        row.get("schema_version") != QUEUE_SCHEMA_VERSION
        or row.get("clause") not in CLAUSES
        or row.get("chunk") not in CHUNKS
        or row.get("accepted_nodes") != REQUIRED_NODES
        or row.get("sibling_group_count") != REQUIRED_SIBLING_GROUPS
        or row.get("maximum_depth") != REQUIRED_MAX_DEPTH
        or not isinstance(row.get("worst_active_lower"), (int, float))
        or not isinstance(row.get("queue_stream_ns"), int)
        or row["queue_stream_ns"] <= 0
        or not isinstance(row.get("synchronized_wall_ns"), int)
        or row["synchronized_wall_ns"] <= 0
        or not isinstance(calls, list)
        or bool(calls) is not profile
        or row.get("performance_claimed") is not False
        or row.get("row_hash") != canonical_hash(semantic)
    ):
        raise ValueError("NRIR49A queue row differs")
    if profile:
        if (
            row.get("mode") not in {"profile", "complete"}
            or row.get("selected_call_count") != len(calls)
            or row.get("selected_device_ns") != sum(call["device_ns"] for call in calls)
            or any(
                call.get("requested_production_chunk") != DEFAULT_CHUNK
                or call.get("effective_harness_chunk") != row["chunk"]
                or call.get("clause") != row["clause"]
                for call in calls
            )
        ):
            raise ValueError("NRIR49A selected-CROWN call coverage differs")
    elif (
        row.get("mode") != "control"
        or row.get("selected_call_count") != 0
        or row.get("selected_device_ns") != 0
    ):
        raise ValueError("NRIR49A control instrumentation differs")


def validate_worker(worker: Mapping[str, Any]) -> None:
    repeat_index = worker.get("repeat_index")
    if not isinstance(repeat_index, int) or repeat_index not in range(REPEAT_COUNT):
        raise ValueError("NRIR49A worker repeat differs")
    profiles = worker.get("profile_rows")
    controls = worker.get("control_rows")
    complete = worker.get("complete")
    representative_profiler = worker.get("representative_profiler")
    semantic = {key: value for key, value in worker.items() if key != "worker_hash"}
    if (
        worker.get("schema_version") != WORKER_SCHEMA_VERSION
        or worker.get("chunk_order") != list(latin_chunk_order(repeat_index))
        or worker.get("code_revision") != _code_revision()
        or not isinstance(profiles, list)
        or len(profiles) != len(CHUNKS) * len(CLAUSES)
        or not isinstance(controls, list)
        or len(controls) != len(CLAUSES)
        or not isinstance(complete, Mapping)
        or (repeat_index == 0) != isinstance(representative_profiler, Mapping)
        or worker.get("performance_claimed") is not False
        or worker.get("worker_hash") != canonical_hash(semantic)
    ):
        raise ValueError("NRIR49A worker envelope differs")
    profile_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in profiles:
        _validate_queue_row(row, profile=True)
        key = (row["clause"], row["chunk"])
        if key in profile_by_key:
            raise ValueError("NRIR49A profile row duplicates")
        profile_by_key[key] = row
    if set(profile_by_key) != {
        (clause, chunk) for clause in CLAUSES for chunk in CHUNKS
    }:
        raise ValueError("NRIR49A profile matrix differs")
    control_by_clause: dict[int, Mapping[str, Any]] = {}
    for row in controls:
        _validate_queue_row(row, profile=False)
        if row["chunk"] != DEFAULT_CHUNK or row["clause"] in control_by_clause:
            raise ValueError("NRIR49A control matrix differs")
        control_by_clause[row["clause"]] = row
    if set(control_by_clause) != set(CLAUSES):
        raise ValueError("NRIR49A control clause coverage differs")
    for clause in CLAUSES:
        reference = profile_by_key[(clause, DEFAULT_CHUNK)]
        if control_by_clause[clause]["semantics_hash"] != reference["semantics_hash"]:
            raise ValueError("NRIR49A profile/control semantics differ")
        for chunk in CHUNKS:
            row = profile_by_key[(clause, chunk)]
            if row["semantics_hash"] != reference["semantics_hash"] or not math.isclose(
                float(row["worst_active_lower"]),
                float(reference["worst_active_lower"]),
                abs_tol=ATOL,
                rel_tol=RTOL,
            ):
                raise ValueError("NRIR49A chunk semantics differ")
    if (
        complete.get("schema_version") != COMPLETE_SCHEMA_VERSION
        or complete.get("repeat_index") != repeat_index
        or complete.get("chunk") != DEFAULT_CHUNK
        or not isinstance(complete.get("stream_ns"), int)
        or complete["stream_ns"] <= 0
        or complete.get("selected_all_device_ns", 0) <= 0
        or complete.get("selected_child_device_ns", 0) <= 0
        or complete.get("performance_claimed") is not False
        or complete.get("row_hash")
        != canonical_hash(
            {key: value for key, value in complete.items() if key != "row_hash"}
        )
    ):
        raise ValueError("NRIR49A complete scope differs")
    for row in complete.get("queue_rows", []):
        _validate_queue_row(row, profile=True)
    if isinstance(representative_profiler, Mapping):
        profile_semantic = {
            key: value
            for key, value in representative_profiler.items()
            if key != "profile_hash"
        }
        if (
            representative_profiler.get("scope")
            != "representative-non-timing-clause2-default32"
            or representative_profiler.get("excluded_from_timing_summary") is not True
            or representative_profiler.get("performance_claimed") is not False
            or representative_profiler.get("kernel_count", 0) <= 0
            or representative_profiler.get("runtime_launch_api_count", 0) <= 0
            or representative_profiler.get("profile_hash")
            != canonical_hash(profile_semantic)
        ):
            raise ValueError("NRIR49A representative profiler differs")
        _validate_queue_row(
            _mapping(representative_profiler.get("queue_row"), "profiler queue row"),
            profile=False,
        )


def _memory_decision(
    rows: Sequence[Mapping[str, Any]], total_memory: int
) -> dict[str, object]:
    maximum_allocated = max(int(row["peak_allocated"]) for row in rows)
    maximum_reserved = max(int(row["peak_reserved"]) for row in rows)
    alloc_ratio = maximum_allocated / total_memory
    reserved_ratio = maximum_reserved / total_memory
    physical_admitted = (
        alloc_ratio >= PHYSICAL_MEMORY_RATIO or reserved_ratio >= PHYSICAL_MEMORY_RATIO
    )
    return {
        "maximum_semantic_valid_domain_batch": SEMANTIC_BATCH_MAXIMUM,
        "maximum_peak_allocated": maximum_allocated,
        "maximum_peak_reserved": maximum_reserved,
        "maximum_allocated_ratio": alloc_ratio,
        "maximum_reserved_ratio": reserved_ratio,
        "b80_alloc": 1 if alloc_ratio >= PHYSICAL_MEMORY_RATIO else None,
        "b80_reserved": 1 if reserved_ratio >= PHYSICAL_MEMORY_RATIO else None,
        "b_oom": None,
        "physical_memory_admitted": physical_admitted,
        "g8_memory_path": "eligible" if physical_admitted else "n/a",
        "reason": (
            "natural-valid-batch-reaches-physical-threshold"
            if physical_admitted
            else "valid-batch-max-one-and-peak-below-80-percent-without-oom"
        ),
    }


def build_summary(workers: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Recompute all decisions from raw workers."""

    if len(workers) != REPEAT_COUNT:
        raise ValueError("NRIR49A formal repeat count differs")
    by_repeat = {worker["repeat_index"]: worker for worker in workers}
    if set(by_repeat) != set(range(REPEAT_COUNT)):
        raise ValueError("NRIR49A repeat coverage differs")
    for worker in workers:
        validate_worker(worker)
    queue_shares = []
    complete_shares = []
    perturbation_ratios: dict[int, list[float]] = {clause: [] for clause in CLAUSES}
    all_profiles = []
    normalized = []
    for repeat_index in range(REPEAT_COUNT):
        worker = by_repeat[repeat_index]
        profiles = worker["profile_rows"]
        controls = {row["clause"]: row for row in worker["control_rows"]}
        defaults = [row for row in profiles if row["chunk"] == DEFAULT_CHUNK]
        queue_selected = sum(row["selected_device_ns"] for row in defaults)
        queue_scope = sum(row["queue_stream_ns"] for row in defaults)
        queue_shares.append(queue_selected / queue_scope)
        complete = worker["complete"]
        complete_shares.append(
            complete["selected_all_device_ns"] / complete["stream_ns"]
        )
        for row in profiles:
            all_profiles.append(row)
            normalized.append(
                {
                    "repeat_index": repeat_index,
                    "clause": row["clause"],
                    "chunk": row["chunk"],
                    "order_position": row["order_position"],
                    "queue_stream_ns": row["queue_stream_ns"],
                    "synchronized_wall_ns": row["synchronized_wall_ns"],
                    "selected_device_ns": row["selected_device_ns"],
                    "selected_share": row["selected_device_ns"]
                    / row["queue_stream_ns"],
                    "peak_allocated": row["peak_allocated"],
                    "peak_reserved": row["peak_reserved"],
                    "semantics_hash": row["semantics_hash"],
                }
            )
        for clause in CLAUSES:
            profile = next(row for row in defaults if row["clause"] == clause)
            perturbation_ratios[clause].append(
                profile["synchronized_wall_ns"]
                / controls[clause]["synchronized_wall_ns"]
            )
    perturbation_medians = {
        str(clause): _median(values) for clause, values in perturbation_ratios.items()
    }
    perturbation_passed = all(
        value <= MAXIMUM_PERTURBATION_RATIO for value in perturbation_medians.values()
    )
    s_queue = _median(queue_shares)
    s_complete = _median(complete_shares)
    r_queue = required_region_speedup(s_queue, QUEUE_TARGET_SPEEDUP)
    r_complete = required_region_speedup(s_complete, COMPLETE_TARGET_SPEEDUP)
    required_values = [value for value in (r_queue, r_complete) if value is not None]
    r_latency = max(required_values) if len(required_values) == 2 else None
    queue_opportunity = s_queue >= MINIMUM_QUEUE_SHARE
    latency_feasible = (
        r_latency is not None and r_latency <= MAXIMUM_REQUIRED_REGION_SPEEDUP
    )
    total_memory = int(workers[0]["environment"]["total_memory_bytes"])
    memory = _memory_decision(all_profiles, total_memory)
    chunk_metrics = []
    for chunk in CHUNKS:
        rows = [row for row in normalized if row["chunk"] == chunk]
        chunk_metrics.append(
            {
                "chunk": chunk,
                "queue_stream_median_ns": int(
                    statistics.median(row["queue_stream_ns"] for row in rows)
                ),
                "wall_median_ns": int(
                    statistics.median(row["synchronized_wall_ns"] for row in rows)
                ),
                "selected_device_median_ns": int(
                    statistics.median(row["selected_device_ns"] for row in rows)
                ),
                "peak_allocated_max": max(row["peak_allocated"] for row in rows),
                "peak_reserved_max": max(row["peak_reserved"] for row in rows),
            }
        )
    if not perturbation_passed:
        next_route = "not-auditable-profiler-perturbation"
        status = "not-auditable"
    elif not queue_opportunity:
        next_route = "gpu-winner-reselection"
        status = "validated-no-go"
    elif not latency_feasible:
        next_route = (
            "memory-only" if memory["physical_memory_admitted"] else "route-no-go"
        )
        status = "validated-no-go"
    else:
        next_route = "proceed-g2-qualification"
        status = "validated-reduced"
    decision = {
        "instrumentation_passed": perturbation_passed,
        "queue_opportunity_passed": queue_opportunity,
        "latency_feasible": latency_feasible,
        "memory_physical_admitted": memory["physical_memory_admitted"],
        "next_route": next_route,
    }
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": status,
        "contract": {
            "repeat_count": REPEAT_COUNT,
            "clauses": list(CLAUSES),
            "chunks": list(CHUNKS),
            "default_chunk": DEFAULT_CHUNK,
            "minimum_queue_share": MINIMUM_QUEUE_SHARE,
            "maximum_perturbation_ratio": MAXIMUM_PERTURBATION_RATIO,
            "queue_target_speedup": QUEUE_TARGET_SPEEDUP,
            "complete_target_speedup": COMPLETE_TARGET_SPEEDUP,
            "maximum_required_region_speedup": MAXIMUM_REQUIRED_REGION_SPEEDUP,
            "physical_memory_ratio": PHYSICAL_MEMORY_RATIO,
        },
        "queue_shares": queue_shares,
        "complete_shares": complete_shares,
        "s_queue_median": s_queue,
        "s_complete_median": s_complete,
        "r_queue_required": r_queue,
        "r_complete_required": r_complete,
        "r_latency_required": r_latency,
        "perturbation_ratios": {
            str(key): value for key, value in perturbation_ratios.items()
        },
        "perturbation_medians": perturbation_medians,
        "chunk_metrics": chunk_metrics,
        "memory": memory,
        "decision": decision,
        "decision_hash": canonical_hash(decision),
        "normalized_hash": canonical_hash(normalized),
        "performance_claimed": False,
    }
    summary["summary_hash"] = canonical_hash(summary)
    return summary


def normalized_rows(workers: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    rows = []
    for worker in sorted(workers, key=lambda item: item["repeat_index"]):
        for row in worker["profile_rows"]:
            rows.append(
                {
                    "repeat_index": worker["repeat_index"],
                    "clause": row["clause"],
                    "chunk": row["chunk"],
                    "order_position": row["order_position"],
                    "queue_stream_ns": row["queue_stream_ns"],
                    "synchronized_wall_ns": row["synchronized_wall_ns"],
                    "selected_device_ns": row["selected_device_ns"],
                    "selected_share": row["selected_device_ns"]
                    / row["queue_stream_ns"],
                    "peak_allocated": row["peak_allocated"],
                    "peak_reserved": row["peak_reserved"],
                    "semantics_hash": row["semantics_hash"],
                }
            )
    return rows


def _queries(benchmark_root: Path) -> list[dict[str, object]]:
    return [
        {
            "schema_version": QUERY_SCHEMA_VERSION,
            "workload_id": "cifar10_resnet:000",
            "clause": clause,
            "required_nodes": REQUIRED_NODES,
            "required_sibling_groups": REQUIRED_SIBLING_GROUPS,
            "required_max_depth": REQUIRED_MAX_DEPTH,
            "model_relative_path": MODEL_RELATIVE_PATH,
            "property_relative_path": PROPERTY_RELATIVE_PATH,
            "model_sha256": file_sha256(benchmark_root / MODEL_RELATIVE_PATH),
            "property_sha256": file_sha256(benchmark_root / PROPERTY_RELATIVE_PATH),
            "performance_claimed": False,
        }
        for clause in CLAUSES
    ]


def _code_revision() -> dict[str, str]:
    root = _repo_root()
    paths = (
        "boundflow/runtime/native_intermediate_refinement.py",
        "boundflow/runtime/native_prepared_intermediate_refinement.py",
        "boundflow/runtime/native_prepared_per_child_refinement.py",
        "boundflow/runtime/native_prepared_objective_branch_shared_production_queue.py",
        "boundflow/runtime/native_objective_branch_shared_evaluator.py",
        "scripts/run_nrir49a_g1_gpu_attribution.py",
    )
    return {path: file_sha256(root / path) for path in paths}


def _artifact_readme(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "# NRIR49A G1 GPU Attribution Artifact",
            "",
            "This artifact contains read-only GPU attribution, not an optimization speedup claim.",
            "",
            f"- status: `{summary['status']}`",
            f"- next route: `{summary['decision']['next_route']}`",
            "- replay: `python scripts/run_nrir49a_g1_gpu_attribution.py replay "
            "--artifact-dir <dir>`",
            "",
        )
    )


def write_artifact(
    artifact_dir: Path,
    *,
    benchmark_root: Path,
    environment: Mapping[str, Any],
    workers: Sequence[Mapping[str, Any]],
    replay_stdout: str = "pending-generate-replay",
) -> dict[str, object]:
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(workers)
    normalized = normalized_rows(workers)
    _write_json(artifact_dir / "environment.json", environment)
    _write_jsonl(artifact_dir / "queries.jsonl", _queries(benchmark_root))
    _write_jsonl(artifact_dir / "results_raw.jsonl", workers)
    _write_jsonl(artifact_dir / "normalized.jsonl", normalized)
    _write_json(artifact_dir / "summary.json", summary)
    (artifact_dir / "replay_stdout.txt").write_text(
        replay_stdout.rstrip() + "\n", encoding="utf-8"
    )
    _write_jsonl(artifact_dir / "failure_rows.jsonl", [])
    (artifact_dir / "README.md").write_text(_artifact_readme(summary), encoding="utf-8")
    root = _repo_root()
    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": summary["status"],
        "git_head": _git_value(root, "rev-parse", "HEAD"),
        "git_dirty_paths": (
            _git_value(root, "status", "--porcelain=v1") or ""
        ).splitlines(),
        "code_revision": _code_revision(),
        "vnncomp_commit": _git_value(benchmark_root, "rev-parse", "HEAD"),
        "worker_hashes": [worker["worker_hash"] for worker in workers],
        "summary_hash": summary["summary_hash"],
        "files": {name: file_sha256(artifact_dir / name) for name in ARTIFACT_FILES},
        "performance_claimed": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(artifact_dir / MANIFEST_FILE, manifest)
    return summary


def replay_artifact(artifact_dir: Path) -> dict[str, Any]:
    manifest = _load_json(artifact_dir / MANIFEST_FILE)
    semantic_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("performance_claimed") is not False
        or manifest.get("manifest_hash") != canonical_hash(semantic_manifest)
        or manifest.get("code_revision") != _code_revision()
    ):
        raise ValueError("NRIR49A manifest envelope differs")
    files = _mapping(manifest.get("files"), "artifact files")
    if set(files) != set(ARTIFACT_FILES) or any(
        file_sha256(artifact_dir / name) != digest for name, digest in files.items()
    ):
        raise ValueError("NRIR49A artifact file digest differs")
    workers = _read_jsonl(artifact_dir / "results_raw.jsonl")
    stored = _load_json(artifact_dir / "summary.json")
    rebuilt = build_summary(workers)
    if rebuilt != stored or manifest.get("summary_hash") != rebuilt["summary_hash"]:
        raise ValueError("NRIR49A summary semantic replay differs")
    normalized = _read_jsonl(artifact_dir / "normalized.jsonl")
    if normalized != normalized_rows(workers):
        raise ValueError("NRIR49A normalized semantic replay differs")
    queries = _read_jsonl(artifact_dir / "queries.jsonl")
    if (
        len(queries) != len(CLAUSES)
        or {row.get("clause") for row in queries} != set(CLAUSES)
        or any(
            row.get("model_sha256") != MODEL_SHA256
            or row.get("property_sha256") != PROPERTY_SHA256
            or row.get("performance_claimed") is not False
            for row in queries
        )
    ):
        raise ValueError("NRIR49A query identity differs")
    return rebuilt


def _run_worker_subprocess(
    *, benchmark_root: Path, repeat_index: int, result_path: Path
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--benchmark-root",
        str(benchmark_root),
        "--repeat-index",
        str(repeat_index),
        "--result-json",
        str(result_path),
    ]
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"NRIR49A worker {repeat_index} failed: {completed.stderr[-8192:]}"
        )
    worker = _load_json(result_path)
    validate_worker(worker)
    return worker


def _generate(args: argparse.Namespace) -> None:
    benchmark_root = args.benchmark_root.resolve()
    environment = _environment()
    workers = []
    cache_dir = args.worker_cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for repeat_index in range(REPEAT_COUNT):
        worker_path = cache_dir / f"repeat-{repeat_index}.json"
        if worker_path.is_file():
            worker = _load_json(worker_path)
            validate_worker(worker)
        else:
            worker = _run_worker_subprocess(
                benchmark_root=benchmark_root,
                repeat_index=repeat_index,
                result_path=worker_path,
            )
        workers.append(worker)
    expected = build_summary(workers)
    replay_line = canonical_json(
        {
            "status": "replay-passed",
            "summary_hash": expected["summary_hash"],
            "decision": expected["decision"],
        }
    )
    summary = write_artifact(
        args.artifact_dir.resolve(),
        benchmark_root=benchmark_root,
        environment=environment,
        workers=workers,
        replay_stdout=replay_line,
    )
    replayed = replay_artifact(args.artifact_dir.resolve())
    if replayed != summary:
        raise ValueError("NRIR49A immediate replay differs")
    print(canonical_json({"summary": summary, "replay": replay_line}))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--benchmark-root", type=Path, required=True)
    generate.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    generate.add_argument(
        "--worker-cache-dir", type=Path, default=DEFAULT_WORKER_CACHE_DIR
    )
    replay = subparsers.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--benchmark-root", type=Path, required=True)
    worker.add_argument("--repeat-index", type=int, required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "worker":
        worker = run_worker(
            benchmark_root=args.benchmark_root.resolve(),
            repeat_index=args.repeat_index,
        )
        _write_json(args.result_json.resolve(), worker)
        print(
            canonical_json(
                {
                    "repeat_index": worker["repeat_index"],
                    "worker_hash": worker["worker_hash"],
                }
            )
        )
    elif args.command == "generate":
        _generate(args)
    elif args.command == "replay":
        summary = replay_artifact(args.artifact_dir.resolve())
        print(
            canonical_json(
                {
                    "status": "replay-passed",
                    "summary_hash": summary["summary_hash"],
                    "decision": summary["decision"],
                }
            )
        )
    else:
        raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
