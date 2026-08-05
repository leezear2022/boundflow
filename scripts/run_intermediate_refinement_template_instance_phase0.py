#!/usr/bin/env python3
"""Measure NRIR46 template/instance compile ownership without changing NRIR45."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=protected-access,import-outside-toplevel,duplicate-code
# pylint: disable=missing-function-docstring,too-many-boolean-expressions

from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
from dataclasses import dataclass
from functools import wraps
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Optional
from unittest.mock import patch

from scripts import run_prepared_intermediate_refinement_global_formal as nrir45

FORMAL_SCHEMA_VERSION = "boundflow.refinement-template-instance-phase0-formal/v1"
WORKER_SCHEMA_VERSION = "boundflow.refinement-template-instance-phase0-worker/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.refinement-template-instance-phase0-manifest/v1"
ARTIFACT_DIR = Path(
    "artifacts/intermediate-refinement-template-instance/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-phase0-v1"
)
SOURCE_NRIR45 = Path(
    "artifacts/prepared-intermediate-refinement/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-phase-b-v1/formal.json"
)
EXPECTED_SOURCE_NRIR45_HASH = (
    "4ae71919b5c4d6e8d6162df8bb7d14143a705f60a599f8e4bfa30d084c1a01f8"
)
REPEAT_COUNT = 3
TORCH_THREADS = 8
WORKER_TIMEOUT_SECONDS = 240
STATIC_SHAREABLE_GATE_NS = 1_500_000_000
EXPECTED_TIMER_CALLS = {
    "prepared_compile_total": 60,
    "legacy_compile_total": 60,
    "lower_legacy_ir": 60,
    "lower_prepared_ir": 60,
    "prepared_capsule": 60,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "replay"):
        child = subparsers.add_parser(command)
        child.add_argument("--benchmark-root", type=Path, required=True)
        child.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
        child.add_argument("--torch-threads", type=int, default=TORCH_THREADS)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--property", type=Path, required=True)
    worker.add_argument("--repeat-index", type=int, required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    worker.add_argument("--torch-threads", type=int, required=True)
    return parser.parse_args()


def _canonical_json(value: object, *, indent: Optional[int] = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _source_nrir45() -> Mapping[str, Any]:
    value = _load_json(_repo_root() / SOURCE_NRIR45)
    if (
        value.get("schema_version") != nrir45.FORMAL_SCHEMA_VERSION
        or value.get("formal_payload_hash") != EXPECTED_SOURCE_NRIR45_HASH
        or value.get("status") != "validated-reduced"
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR46 Phase0 source NRIR45 differs")
    nrir45.validate_formal(value)
    return value


def _code_revision() -> str:
    root = _repo_root()
    files = (
        Path(__file__).resolve(),
        root / "boundflow/runtime/native_intermediate_refinement.py",
        root / "boundflow/runtime/native_prepared_intermediate_refinement.py",
        root / "boundflow/runtime/native_prepared_per_child_refinement.py",
        root / "boundflow/runtime/native_prepared_shared_parametric_ancestral.py",
        root
        / "boundflow/runtime/native_prepared_objective_branch_shared_production_queue.py",
        root
        / "boundflow/runtime/native_prepared_root_projection_multi_clause_anytime.py",
        root / "scripts/run_prepared_intermediate_refinement_global_formal.py",
    )
    return _canonical_hash(
        {str(path.relative_to(root)): _file_sha256(path) for path in files}
    )


@dataclass
class _TimerRow:
    calls: int = 0
    inclusive_ns: int = 0
    exclusive_ns: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "inclusive_ns": self.inclusive_ns,
            "exclusive_ns": self.exclusive_ns,
        }


class _Recorder:
    def __init__(self) -> None:
        self.runtime_active = False
        self.compile_depth = 0
        self.rows: dict[str, _TimerRow] = {}
        self.frames: list[list[Any]] = []

    @property
    def enabled(self) -> bool:
        return self.runtime_active and self.compile_depth > 0

    def measure(
        self, label: str, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        if not self.enabled:
            return function(*args, **kwargs)
        started_ns = time.perf_counter_ns()
        frame: list[Any] = [label, 0]
        self.frames.append(frame)
        try:
            return function(*args, **kwargs)
        finally:
            elapsed_ns = time.perf_counter_ns() - started_ns
            self.frames.pop()
            child_ns = int(frame[1])
            row = self.rows.setdefault(label, _TimerRow())
            row.calls += 1
            row.inclusive_ns += elapsed_ns
            row.exclusive_ns += elapsed_ns - child_ns
            if self.frames:
                self.frames[-1][1] = int(self.frames[-1][1]) + elapsed_ns

    def wrap(self, function: Callable[..., Any], label: str) -> Callable[..., Any]:
        @wraps(function)
        def measured(*args: Any, **kwargs: Any) -> Any:
            return self.measure(label, function, *args, **kwargs)

        return measured

    def rows_dict(self) -> dict[str, dict[str, int]]:
        return {name: self.rows[name].to_dict() for name in sorted(self.rows)}


def _timer_ns(timers: Mapping[str, Mapping[str, int]], label: str, field: str) -> int:
    row = timers.get(label)
    if row is None:
        raise ValueError(f"NRIR46 Phase0 timer is absent: {label}")
    value = row.get(field)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"NRIR46 Phase0 timer differs: {label}.{field}")
    return value


def _attribution_summary(
    timers: Mapping[str, Mapping[str, int]], *, program_count: int
) -> dict[str, object]:
    select_calls = _timer_ns(timers, "select_targets", "calls")
    select_inclusive_ns = _timer_ns(timers, "select_targets", "inclusive_ns")
    if select_calls < program_count:
        raise ValueError("NRIR46 Phase0 target selection count differs")
    redundant_select_calls = select_calls - program_count
    redundant_select_ns = (
        0
        if select_calls == 0
        else select_inclusive_ns * redundant_select_calls // select_calls
    )
    static_topology_ns = sum(
        _timer_ns(timers, label, "exclusive_ns")
        for label in (
            "module_validate",
            "policy_validate",
            "primal_graph_hash",
            "lower_legacy_ir",
            "lower_prepared_ir",
        )
    )
    template_instance_convertible_ns = static_topology_ns + redundant_select_ns
    compile_total_ns = _timer_ns(timers, "prepared_compile_total", "inclusive_ns")
    return {
        "compile_total_ns": compile_total_ns,
        "semantic_target_selection_calls": program_count,
        "observed_target_selection_calls": select_calls,
        "redundant_target_selection_calls": redundant_select_calls,
        "redundant_target_selection_estimated_ns": redundant_select_ns,
        "static_topology_ns": static_topology_ns,
        "template_instance_convertible_ns": template_instance_convertible_ns,
        "static_shareable_gate_ns": STATIC_SHAREABLE_GATE_NS,
        "strict_static_gate_passed": static_topology_ns >= STATIC_SHAREABLE_GATE_NS,
        "ownership_ceiling_gate_passed": template_instance_convertible_ns
        >= STATIC_SHAREABLE_GATE_NS,
        "convertible_to_compile_ratio": (
            template_instance_convertible_ns / compile_total_ns
        ),
    }


def _program_summary(programs: list[Any]) -> dict[str, object]:
    identity_hashes: list[str] = []
    table_hashes: list[str] = []
    topology_hashes: list[str] = []
    schedule_topology_hashes: list[str] = []
    target_counts: list[int] = []
    primal_hashes: list[str] = []
    objective_hashes: list[Optional[str]] = []
    for program in programs:
        identities = [
            [target.relu_input, target.neuron_index] for target in program.plan.targets
        ]
        identity_hashes.append(_canonical_hash(identities))
        table_hashes.append(program.capsule.target_table_hash)
        topology_hashes.append(
            _canonical_hash([task.kind.value for task in program.task_module.tasks])
        )
        schedule_topology_hashes.append(
            _canonical_hash([action.sequence for action in program.schedule.actions])
        )
        target_counts.append(len(program.plan.targets))
        primal_hashes.append(program.plan.primal_graph_hash)
        objective_hashes.append(program.plan.objective_hash)
    return {
        "program_count": len(programs),
        "target_counts": target_counts,
        "distinct_target_identity_hashes": len(set(identity_hashes)),
        "distinct_target_table_hashes": len(set(table_hashes)),
        "distinct_task_kind_topologies": len(set(topology_hashes)),
        "distinct_schedule_topologies": len(set(schedule_topology_hashes)),
        "distinct_primal_graph_hashes": len(set(primal_hashes)),
        "distinct_objective_hashes": len(set(objective_hashes)),
    }


def _instrument_nrir45_worker(args: argparse.Namespace) -> dict[str, Any]:
    import boundflow.runtime.native_intermediate_refinement as legacy
    import boundflow.runtime.native_prepared_intermediate_refinement as prepared
    import boundflow.runtime.native_prepared_per_child_refinement as child
    import boundflow.runtime.native_prepared_root_projection_multi_clause_anytime as root_runtime
    from boundflow.ir.refinement import (
        NativeIntermediateRefinementMultiPassPolicyIR,
        NativeIntermediateRefinementPolicyIR,
    )
    from boundflow.ir.task import BFTaskModule

    recorder = _Recorder()
    programs: list[Any] = []
    original_root = (
        root_runtime.execute_native_prepared_root_projection_multi_clause_anytime_program
    )

    @wraps(original_root)
    def measured_root(*root_args: Any, **root_kwargs: Any) -> Any:
        previous = recorder.runtime_active
        recorder.runtime_active = True
        try:
            return original_root(*root_args, **root_kwargs)
        finally:
            recorder.runtime_active = previous

    original_child_compile = (
        child.compile_native_prepared_intermediate_refinement_program
    )

    @wraps(original_child_compile)
    def measured_child_compile(*compile_args: Any, **compile_kwargs: Any) -> Any:
        recorder.compile_depth += 1
        try:
            program = recorder.measure(
                "prepared_compile_total",
                original_child_compile,
                *compile_args,
                **compile_kwargs,
            )
            programs.append(program)
            return program
        finally:
            recorder.compile_depth -= 1

    patches: list[tuple[object, str, object]] = [
        (
            root_runtime,
            "execute_native_prepared_root_projection_multi_clause_anytime_program",
            measured_root,
        ),
        (
            child,
            "compile_native_prepared_intermediate_refinement_program",
            measured_child_compile,
        ),
    ]
    timed: list[tuple[object, str, str]] = [
        (
            prepared,
            "compile_native_intermediate_refinement_program",
            "legacy_compile_total",
        ),
        (prepared, "_prepared_capsule", "prepared_capsule"),
        (prepared, "lower_native_prepared_refinement_ir", "lower_prepared_ir"),
        (prepared, "_program_runtime_roots", "program_runtime_roots"),
        (prepared, "_build_runtime_receipt", "runtime_receipt_build"),
        (legacy, "_forward_ibp_trace_mlp", "forward_ibp"),
        (
            legacy,
            "run_crown_ibp_mlp_with_relu_influence_from_forward_trace",
            "objective_influence",
        ),
        (legacy, "_select_targets", "select_targets"),
        (legacy, "lower_native_intermediate_refinement_ir", "lower_legacy_ir"),
        (legacy, "plain_crown_primal_graph_hash", "primal_graph_hash"),
        (legacy, "_input_bounds_hash", "input_bounds_hash"),
        (legacy, "relu_split_state_hash", "split_state_hash"),
        (legacy, "intermediate_bounds_hash", "intermediate_bounds_hash"),
        (BFTaskModule, "validate", "module_validate"),
        (NativeIntermediateRefinementPolicyIR, "validate", "policy_validate"),
        (
            NativeIntermediateRefinementMultiPassPolicyIR,
            "validate",
            "multi_pass_validate",
        ),
        (
            legacy.NativeIntermediateRefinementProgram,
            "validate",
            "legacy_program_validate",
        ),
        (
            prepared.NativePreparedIntermediateRefinementProgram,
            "validate",
            "prepared_program_validate",
        ),
    ]
    with ExitStack() as stack:
        for patch_owner, name, value in patches:
            stack.enter_context(patch.object(patch_owner, name, value))
        for timed_owner, name, label in timed:
            stack.enter_context(
                patch.object(
                    timed_owner,
                    name,
                    recorder.wrap(getattr(timed_owner, name), label),
                )
            )
        with tempfile.TemporaryDirectory(
            prefix="boundflow-nrir46-phase0-worker-"
        ) as temporary:
            nrir45_result_path = Path(temporary) / "nrir45.json"
            nrir45._worker(
                argparse.Namespace(
                    model=args.model,
                    property=args.property,
                    repeat_index=args.repeat_index,
                    result_json=nrir45_result_path,
                    torch_threads=args.torch_threads,
                )
            )
            nrir45_result = _load_json(nrir45_result_path)
    timers = recorder.rows_dict()
    program_summary = _program_summary(programs)
    program_count = program_summary["program_count"]
    if not isinstance(program_count, int):
        raise TypeError("NRIR46 Phase0 program count differs")
    result: dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "source": {
            "code_revision": _code_revision(),
            "nrir45_formal_payload_hash": EXPECTED_SOURCE_NRIR45_HASH,
            "nrir45_result_hash": nrir45_result["result_hash"],
        },
        "repeat_index": args.repeat_index,
        "protocol": {
            "torch_threads": args.torch_threads,
            "instrumentation_scope": "nrir45_runtime_child_compile_only",
            "timing_kind": "diagnostic_inclusive_exclusive_wall_ns",
            "claim_boundary": "phase0_compiler_ownership_diagnosis_only",
        },
        "nrir45": {
            "correctness_gate_passed": nrir45_result["correctness_gate_passed"],
            "production_gate_passed": nrir45_result["production_gate_passed"],
            "selected_original_clause_indices": nrir45_result["decision"][
                "selected_original_clause_indices"
            ],
            "accepted_nodes": [
                row["slice"]["accepted_nodes"] for row in nrir45_result["slices"]
            ],
            "prepared_capsule_count": nrir45_result["prepared_refinement"][
                "capsule_count"
            ],
            "full_replay_count": nrir45_result["prepared_refinement"][
                "full_replay_count"
            ],
            "whole_elapsed_ns": nrir45_result["runtime_trace"]["elapsed_ns"],
        },
        "programs": program_summary,
        "timers": timers,
        "attribution": _attribution_summary(timers, program_count=program_count),
        "correctness_gate_passed": False,
        "performance_claimed": False,
    }
    result["correctness_gate_passed"] = _correctness_gate(result)
    result["result_hash"] = _canonical_hash(
        {key: value for key, value in result.items() if key != "result_hash"}
    )
    _validate_worker(result)
    return result


def _correctness_gate(value: Mapping[str, Any]) -> bool:
    nrir45_result = value.get("nrir45")
    programs = value.get("programs")
    timers = value.get("timers")
    if not isinstance(nrir45_result, dict) or not isinstance(programs, dict):
        return False
    if not isinstance(timers, dict):
        return False
    try:
        calls_match = all(
            _timer_ns(timers, label, "calls") == calls
            for label, calls in EXPECTED_TIMER_CALLS.items()
        )
    except ValueError:
        return False
    return bool(
        nrir45_result.get("correctness_gate_passed") is True
        and nrir45_result.get("production_gate_passed") is True
        and nrir45_result.get("selected_original_clause_indices") == [2, 3]
        and nrir45_result.get("accepted_nodes") == [31, 31]
        and nrir45_result.get("prepared_capsule_count") == 60
        and nrir45_result.get("full_replay_count") == 60
        and programs.get("program_count") == 60
        and programs.get("distinct_target_identity_hashes") == 60
        and programs.get("distinct_target_table_hashes") == 60
        and programs.get("distinct_task_kind_topologies") == 1
        and programs.get("distinct_schedule_topologies") == 1
        and programs.get("distinct_primal_graph_hashes") == 1
        and programs.get("distinct_objective_hashes") == 2
        and calls_match
    )


def _validate_worker(value: Mapping[str, Any]) -> None:
    source = value.get("source")
    protocol = value.get("protocol")
    programs = value.get("programs")
    timers = value.get("timers")
    attribution = value.get("attribution")
    if (
        value.get("schema_version") != WORKER_SCHEMA_VERSION
        or not isinstance(source, dict)
        or source.get("code_revision") != _code_revision()
        or source.get("nrir45_formal_payload_hash") != EXPECTED_SOURCE_NRIR45_HASH
        or not _is_sha256(source.get("nrir45_result_hash"))
        or value.get("repeat_index") not in range(REPEAT_COUNT)
        or not isinstance(protocol, dict)
        or protocol.get("torch_threads") != TORCH_THREADS
        or protocol.get("instrumentation_scope") != "nrir45_runtime_child_compile_only"
        or protocol.get("claim_boundary") != "phase0_compiler_ownership_diagnosis_only"
        or not isinstance(programs, dict)
        or not isinstance(timers, dict)
        or not isinstance(attribution, dict)
        or value.get("correctness_gate_passed") != _correctness_gate(value)
        or value.get("correctness_gate_passed") is not True
        or value.get("performance_claimed") is not False
        or value.get("result_hash")
        != _canonical_hash(
            {key: item for key, item in value.items() if key != "result_hash"}
        )
    ):
        raise ValueError("NRIR46 Phase0 worker result differs")
    expected_attribution = _attribution_summary(timers, program_count=60)
    if attribution != expected_attribution:
        raise ValueError("NRIR46 Phase0 attribution differs")


def _worker(args: argparse.Namespace) -> None:
    result = _instrument_nrir45_worker(args)
    _write_json(args.result_json.resolve(), result)
    print(
        _canonical_json(
            {
                "repeat": args.repeat_index,
                "correctness": result["correctness_gate_passed"],
                "compile_seconds": result["attribution"]["compile_total_ns"] / 1e9,
                "static_topology_seconds": result["attribution"]["static_topology_ns"]
                / 1e9,
                "ownership_convertible_seconds": result["attribution"][
                    "template_instance_convertible_ns"
                ]
                / 1e9,
            }
        )
    )


def _worker_command(
    workload: Mapping[str, object], repeat: int, result_path: Path, threads: int
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--model",
        str(workload["model"]),
        "--property",
        str(workload["property"]),
        "--repeat-index",
        str(repeat),
        "--result-json",
        str(result_path),
        "--torch-threads",
        str(threads),
    ]


def _formal_payload(
    workload: Mapping[str, object], results: list[Mapping[str, Any]], threads: int
) -> dict[str, object]:
    compile_values = [int(row["attribution"]["compile_total_ns"]) for row in results]
    static_values = [int(row["attribution"]["static_topology_ns"]) for row in results]
    ownership_values = [
        int(row["attribution"]["template_instance_convertible_ns"]) for row in results
    ]
    static_median = int(statistics.median(static_values))
    ownership_median = int(statistics.median(ownership_values))
    payload = {
        "workload": nrir45._public_workload(workload),
        "protocol": {
            "repeat_count": REPEAT_COUNT,
            "torch_threads": threads,
            "static_shareable_gate_ns": STATIC_SHAREABLE_GATE_NS,
            "paired_source": "frozen_nrir45_phase_b",
        },
        "compile_total_ns": compile_values,
        "static_topology_ns": static_values,
        "template_instance_convertible_ns": ownership_values,
        "compile_median_ns": int(statistics.median(compile_values)),
        "static_topology_median_ns": static_median,
        "template_instance_convertible_median_ns": ownership_median,
        "strict_static_gate_passed": static_median >= STATIC_SHAREABLE_GATE_NS,
        "ownership_ceiling_gate_passed": ownership_median >= STATIC_SHAREABLE_GATE_NS,
        "all_correctness_gates_passed": all(
            row["correctness_gate_passed"] is True for row in results
        ),
        "repeat_results": results,
        "performance_claimed": False,
    }
    return payload


def _require_int_list(value: object, *, caller: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise TypeError(f"{caller} integer vector differs")
    return value


def _validate_formal(value: Mapping[str, Any]) -> None:
    payload = value.get("formal_payload")
    if not isinstance(payload, dict):
        raise TypeError("NRIR46 Phase0 formal payload differs")
    results = payload.get("repeat_results")
    if not isinstance(results, list) or len(results) != REPEAT_COUNT:
        raise ValueError("NRIR46 Phase0 repeat results differ")
    for row in results:
        if not isinstance(row, dict):
            raise TypeError("NRIR46 Phase0 repeat result differs")
        _validate_worker(row)
    expected = _formal_payload(
        payload["workload"],  # type: ignore[arg-type]
        results,
        int(payload["protocol"]["torch_threads"]),
    )
    # _formal_payload publicizes an already-public workload without changing it.
    expected["workload"] = payload["workload"]
    status = (
        "validated-go"
        if payload.get("strict_static_gate_passed") is True
        else "validated-no-go"
    )
    if (
        value.get("schema_version") != FORMAL_SCHEMA_VERSION
        or value.get("source", {}).get("code_revision") != _code_revision()
        or value.get("source", {}).get("nrir45_formal_payload_hash")
        != EXPECTED_SOURCE_NRIR45_HASH
        or payload != expected
        or value.get("formal_payload_hash") != _canonical_hash(payload)
        or value.get("status") != status
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR46 Phase0 formal result differs")


def _generate(args: argparse.Namespace) -> None:
    root = _repo_root()
    workloads = nrir45._resolve_workloads(args.benchmark_root.resolve())
    workload = next(
        item for item in workloads if item["workload_id"] == "cifar10_resnet:000"
    )
    artifact_dir = args.artifact_dir.resolve()
    results: list[Mapping[str, Any]] = []
    files: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="boundflow-nrir46-phase0-") as temporary:
        temporary_root = Path(temporary)
        for repeat in range(REPEAT_COUNT):
            result_path = temporary_root / f"repeat-{repeat}.json"
            completed = subprocess.run(
                _worker_command(workload, repeat, result_path, args.torch_threads),
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=WORKER_TIMEOUT_SECONDS,
                check=False,
            )
            log_path = artifact_dir / "logs" / f"repeat-{repeat}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(completed.stdout, encoding="utf-8")
            files[str(log_path.relative_to(artifact_dir))] = _file_sha256(log_path)
            if completed.returncode != 0 or not result_path.is_file():
                raise RuntimeError(
                    f"NRIR46 Phase0 repeat {repeat} failed with "
                    f"{completed.returncode}: {completed.stdout[-12000:]}"
                )
            result = _load_json(result_path)
            _validate_worker(result)
            shard_path = artifact_dir / "shards" / f"repeat-{repeat}.json"
            _write_json(shard_path, result)
            files[str(shard_path.relative_to(artifact_dir))] = _file_sha256(shard_path)
            results.append(result)
            print(completed.stdout.strip())
    payload = _formal_payload(workload, results, args.torch_threads)
    formal = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "status": (
            "validated-go"
            if payload["strict_static_gate_passed"] is True
            else "validated-no-go"
        ),
        "source": {
            "code_revision": _code_revision(),
            "nrir45_formal_payload_hash": EXPECTED_SOURCE_NRIR45_HASH,
        },
        "claim": "intermediate_refinement_template_instance_phase0_diagnosis",
        "formal_payload": payload,
        "formal_payload_hash": _canonical_hash(payload),
        "performance_claimed": False,
    }
    _validate_formal(formal)
    formal_path = artifact_dir / "formal.json"
    _write_json(formal_path, formal)
    files["formal.json"] = _file_sha256(formal_path)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": files,
        "formal_hash": _canonical_hash(formal),
        "performance_claimed": False,
    }
    _write_json(artifact_dir / "manifest.json", manifest)
    compile_values = _require_int_list(
        payload["compile_total_ns"], caller="NRIR46 Phase0 compile"
    )
    static_values = _require_int_list(
        payload["static_topology_ns"], caller="NRIR46 Phase0 static topology"
    )
    ownership_values = _require_int_list(
        payload["template_instance_convertible_ns"],
        caller="NRIR46 Phase0 ownership convertible",
    )
    print(
        _canonical_json(
            {
                "status": formal["status"],
                "formal_payload_hash": formal["formal_payload_hash"],
                "compile_seconds": [value / 1e9 for value in compile_values],
                "static_topology_seconds": [value / 1e9 for value in static_values],
                "ownership_convertible_seconds": [
                    value / 1e9 for value in ownership_values
                ],
                "strict_static_gate_passed": payload["strict_static_gate_passed"],
                "ownership_ceiling_gate_passed": payload[
                    "ownership_ceiling_gate_passed"
                ],
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    formal = _load_json(artifact_dir / "formal.json")
    manifest = _load_json(artifact_dir / "manifest.json")
    _source_nrir45()
    _validate_formal(formal)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("NRIR46 Phase0 manifest files differ")
    shards = [
        _load_json(artifact_dir / "shards" / f"repeat-{repeat}.json")
        for repeat in range(REPEAT_COUNT)
    ]
    expected_paths = {
        "formal.json",
        *(f"logs/repeat-{repeat}.log" for repeat in range(REPEAT_COUNT)),
        *(f"shards/repeat-{repeat}.json" for repeat in range(REPEAT_COUNT)),
    }
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or set(files) != expected_paths
        or any(
            _file_sha256(artifact_dir / path) != digest
            for path, digest in files.items()
        )
        or manifest.get("formal_hash") != _canonical_hash(formal)
        or manifest.get("performance_claimed") is not False
        or shards != formal["formal_payload"]["repeat_results"]
    ):
        raise ValueError("NRIR46 Phase0 manifest differs")
    tampered = copy.deepcopy(shards[0])
    tampered["programs"]["distinct_target_identity_hashes"] = 59
    tampered["correctness_gate_passed"] = _correctness_gate(tampered)
    tampered["result_hash"] = _canonical_hash(
        {key: value for key, value in tampered.items() if key != "result_hash"}
    )
    try:
        _validate_worker(tampered)
    except ValueError:
        tamper_rejected = True
    else:
        tamper_rejected = False
    if not tamper_rejected:
        raise ValueError("NRIR46 Phase0 synchronized tamper was accepted")
    print(
        _canonical_json(
            {
                "status": formal["status"],
                "formal_payload_hash": formal["formal_payload_hash"],
                "synchronized_tamper_rejected": True,
                "performance_claimed": False,
            }
        )
    )


def main() -> None:
    args = _parse_args()
    if args.torch_threads != TORCH_THREADS:
        raise ValueError(f"NRIR46 Phase0 requires exactly {TORCH_THREADS} threads")
    if args.command == "generate":
        _generate(args)
    elif args.command == "replay":
        _replay(args)
    else:
        _worker(args)


if __name__ == "__main__":
    main()
