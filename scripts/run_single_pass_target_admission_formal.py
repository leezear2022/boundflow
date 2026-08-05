#!/usr/bin/env python3
"""Generate or replay NRIR47 single-pass target-admission Phase-A evidence."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-outside-toplevel,protected-access
# pylint: disable=duplicate-code,cell-var-from-loop,too-many-arguments

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence, cast

from boundflow.ir.target_admission import (
    NativeTargetAdmissionReceiptIR,
    NativeTargetAdmissionScheduleAction,
    NativeTargetAdmissionScheduleIR,
    NativeTargetAdmissionTaskIRModule,
    NativeTargetAdmissionTaskIRUnit,
    NativeTargetAdmissionTaskKind,
    lower_native_target_admission_ir,
)
from scripts import run_prepared_intermediate_refinement_formal as nrir45

_NRIR45_PREPARED_PAYLOAD = nrir45._prepared_payload

FORMAL_SCHEMA_VERSION = "boundflow.single-pass-target-admission-formal/v1"
WORKER_SCHEMA_VERSION = "boundflow.single-pass-target-admission-worker/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.single-pass-target-admission-manifest/v1"
ARTIFACT_DIR = Path(
    "artifacts/single-pass-target-admission/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-phase-a-v1"
)
SOURCE_NRIR45 = Path(
    "artifacts/prepared-intermediate-refinement/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-phase-a-v1/formal.json"
)
EXPECTED_SOURCE_NRIR45_HASH = (
    "be1ccb4229d8b88970c9f9f5bae9d6ff8156d4e9b53c84a218a2a1dd6005d439"
)
BASE_REVISION = "ca0bcf398c819990d1af62a3da0bd175455f8bcf"
EXPECTED_CLAUSES = nrir45.EXPECTED_CLAUSES
EXPECTED_WORST_ACTIVE_LOWERS = nrir45.EXPECTED_WORST_ACTIVE_LOWERS
PAIRED_ORDERS = nrir45.PAIRED_ORDERS
REPEAT_COUNT = nrir45.REPEAT_COUNT
TORCH_THREADS = nrir45.TORCH_THREADS
WORKER_TIMEOUT_SECONDS = nrir45.WORKER_TIMEOUT_SECONDS
REQUIRED_NODES = nrir45.REQUIRED_NODES
REQUIRED_SIBLING_GROUPS = nrir45.REQUIRED_SIBLING_GROUPS
MAXIMUM_COMPILER_MEDIAN_RATIO = 0.85
MAXIMUM_QUEUE_MEDIAN_RATIO = 0.97
EXPECTED_CHILD_PROGRAMS = 30
EXPECTED_RECEIPTS = 31


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
    worker.add_argument("--result-json", type=Path, required=True)
    worker.add_argument("--repeat-index", type=int, required=True)
    worker.add_argument("--torch-threads", type=int, required=True)
    return parser.parse_args()


def _canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
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
    path.write_text(_canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _code_revision() -> str:
    root = _repo_root()
    paths = (
        "boundflow/ir/target_admission.py",
        "boundflow/ir/prepared_intermediate_refinement.py",
        "boundflow/runtime/native_intermediate_refinement.py",
        "boundflow/runtime/native_target_admission.py",
        "boundflow/runtime/native_prepared_intermediate_refinement.py",
        "boundflow/runtime/native_prepared_per_child_refinement.py",
        "boundflow/runtime/native_prepared_shared_parametric_ancestral.py",
        "boundflow/runtime/native_prepared_objective_branch_shared_production_queue.py",
        "scripts/run_single_pass_target_admission_formal.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _source_nrir45_hash() -> str:
    value = _load_json(_repo_root() / SOURCE_NRIR45)
    if value.get("formal_hash") != EXPECTED_SOURCE_NRIR45_HASH:
        raise ValueError("NRIR47 frozen NRIR45 source differs")
    return value["formal_hash"]


def _target_stack_payload(program: Any) -> dict[str, Any]:
    from boundflow.runtime.native_intermediate_refinement import (
        NativeIntermediateRefinementProgram,
    )

    if not isinstance(program, NativeIntermediateRefinementProgram) or not all(
        hasattr(program, name)
        for name in (
            "target_admission_receipt",
            "target_admission_task_module",
            "target_admission_schedule",
        )
    ):
        raise TypeError("NRIR47 target admission Program differs")
    receipt = cast(
        NativeTargetAdmissionReceiptIR,
        getattr(program, "target_admission_receipt"),
    )
    task_module = cast(
        NativeTargetAdmissionTaskIRModule,
        getattr(program, "target_admission_task_module"),
    )
    schedule = cast(
        NativeTargetAdmissionScheduleIR,
        getattr(program, "target_admission_schedule"),
    )
    value: dict[str, Any] = {
        "source_hashes": program.hashes(),
        "source_targets": [target.to_dict() for target in program.plan.targets],
        "receipt": receipt.to_dict(),
        "task_module": task_module.to_dict(receipt=receipt),
        "schedule": schedule.to_dict(receipt=receipt, task_module=task_module),
        "performance_claimed": False,
    }
    value["payload_hash"] = _canonical_hash(value)
    return value


def _prepared_payload(node_refinement: Any) -> dict[str, Any]:
    from boundflow.runtime.native_prepared_intermediate_refinement import (
        NativeSinglePassPreparedIntermediateRefinementProgram,
    )

    program = node_refinement.program
    if not isinstance(program, NativeSinglePassPreparedIntermediateRefinementProgram):
        raise TypeError("NRIR47 prepared Program differs")
    value = _NRIR45_PREPARED_PAYLOAD(node_refinement)
    value.pop("payload_hash")
    value["target_admission"] = _target_stack_payload(program)
    value["payload_hash"] = _canonical_hash(value)
    return value


def _load_target_stack(value: Mapping[str, Any]) -> tuple[
    NativeTargetAdmissionReceiptIR,
    NativeTargetAdmissionTaskIRModule,
    NativeTargetAdmissionScheduleIR,
]:
    receipt = NativeTargetAdmissionReceiptIR(**value["receipt"])
    task_value = value["task_module"]
    task_module = NativeTargetAdmissionTaskIRModule(
        module_id=task_value["module_id"],
        source_plan_hash=task_value["source_plan_hash"],
        receipt_hash=task_value["receipt_hash"],
        tasks=tuple(
            NativeTargetAdmissionTaskIRUnit(
                task_id=item["task_id"],
                kind=NativeTargetAdmissionTaskKind(item["kind"]),
                dependency_task_ids=tuple(item["dependency_task_ids"]),
                input_value_ids=tuple(item["input_value_ids"]),
                output_value_ids=tuple(item["output_value_ids"]),
            )
            for item in task_value["tasks"]
        ),
        output_task_id=task_value["output_task_id"],
        schema_version=task_value["schema_version"],
    )
    schedule_value = value["schedule"]
    schedule = NativeTargetAdmissionScheduleIR(
        schedule_id=schedule_value["schedule_id"],
        receipt_hash=schedule_value["receipt_hash"],
        task_module_hash=schedule_value["task_module_hash"],
        actions=tuple(
            NativeTargetAdmissionScheduleAction(**item)
            for item in schedule_value["actions"]
        ),
        production_selection_launches=schedule_value["production_selection_launches"],
        full_replay_selection_launches=schedule_value["full_replay_selection_launches"],
        schema_version=schedule_value["schema_version"],
    )
    return receipt, task_module, schedule


def _validate_target_stack_payload(value: Mapping[str, Any]) -> None:
    receipt, task_module, schedule = _load_target_stack(value)
    receipt.validate()
    task_module.validate(receipt=receipt)
    schedule.validate(receipt=receipt, task_module=task_module)
    semantic = {key: item for key, item in value.items() if key != "payload_hash"}
    if (
        receipt.target_table_hash != _canonical_hash(value["source_targets"])
        or receipt.target_count != len(value["source_targets"])
        or task_module.source_plan_hash
        != value["source_hashes"]["refinement_plan_hash"]
        or value.get("performance_claimed") is not False
        or value.get("payload_hash") != _canonical_hash(semantic)
    ):
        raise ValueError("NRIR47 target admission payload binding differs")


def _validate_prepared_payload(value: Mapping[str, Any]) -> None:
    nrir45._validate_prepared_payload(value)
    target = value.get("target_admission")
    if not isinstance(target, Mapping):
        raise ValueError("NRIR47 prepared target admission payload is absent")
    _validate_target_stack_payload(target)
    capsule = value["capsule"]
    receipt = target["receipt"]
    receipt_value = NativeTargetAdmissionReceiptIR(**receipt)
    if capsule.get("target_admission_receipt_hash") != receipt_value.stable_hash():
        raise ValueError("NRIR47 prepared capsule/receipt binding differs")


def _execute_mode(
    *,
    mode: str,
    plan: Any,
    module: Any,
    input_spec: Any,
    objective: Any,
    threshold: Any,
    refinement: Any,
    optimizer_policy: Any,
    branch_policy: Any,
    query_id: str,
) -> tuple[Any, dict[str, int], int]:
    import boundflow.runtime.native_intermediate_refinement as legacy
    import boundflow.runtime.native_prepared_intermediate_refinement as prepared
    import boundflow.runtime.native_prepared_per_child_refinement as per_child
    import boundflow.runtime.native_target_admission as admission
    from boundflow.runtime.native_parametric_optimizer import (
        NativeParametricOptimizerTemplateCache,
    )
    from boundflow.runtime.native_prepared_objective_branch_shared_production_queue import (
        execute_native_prepared_objective_branch_shared_production_queue,
        execute_native_single_pass_prepared_objective_branch_shared_production_queue,
    )

    counts = {
        "compile_target_selection": 0,
        "runtime_target_selection": 0,
        "other_target_selection": 0,
        "target_admission_receipts": 0,
        "full_program_validation": 0,
        "full_program_hash": 0,
        "prepared_fast_validation": 0,
        "compiler_elapsed_ns": 0,
    }
    compile_depth = 0
    runtime_depth = 0
    original_select = legacy._select_targets
    original_candidate_select = admission._select_targets
    original_validate = legacy.NativeIntermediateRefinementProgram.validate
    original_hashes = legacy.NativeIntermediateRefinementProgram.hashes
    original_fast_validate = (
        prepared.NativePreparedIntermediateRefinementProgram.validate
    )
    original_execute = prepared.execute_native_intermediate_refinement_program
    original_control_compile = (
        per_child.compile_native_prepared_intermediate_refinement_program
    )
    original_candidate_compile = (
        per_child.compile_native_single_pass_prepared_intermediate_refinement_program
    )
    original_receipt = admission._build_target_admission_receipt

    def counted_select(*args: Any, **kwargs: Any) -> Any:
        if compile_depth:
            counts["compile_target_selection"] += 1
        elif runtime_depth:
            counts["runtime_target_selection"] += 1
        else:
            counts["other_target_selection"] += 1
        return original_select(*args, **kwargs)

    def counted_candidate_select(*args: Any, **kwargs: Any) -> Any:
        if compile_depth:
            counts["compile_target_selection"] += 1
        elif runtime_depth:
            counts["runtime_target_selection"] += 1
        else:
            counts["other_target_selection"] += 1
        return original_candidate_select(*args, **kwargs)

    def counted_validate(self: Any, *args: Any, **kwargs: Any) -> Any:
        counts["full_program_validation"] += 1
        return original_validate(self, *args, **kwargs)

    def counted_hashes(self: Any) -> Any:
        counts["full_program_hash"] += 1
        return original_hashes(self)

    def counted_fast_validate(self: Any, *args: Any, **kwargs: Any) -> Any:
        counts["prepared_fast_validation"] += 1
        return original_fast_validate(self, *args, **kwargs)

    def counted_execute(*args: Any, **kwargs: Any) -> Any:
        nonlocal runtime_depth
        runtime_depth += 1
        try:
            return original_execute(*args, **kwargs)
        finally:
            runtime_depth -= 1

    def timed_compile(function: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal compile_depth
        compile_depth += 1
        started_ns = time.perf_counter_ns()
        try:
            return function(*args, **kwargs)
        finally:
            counts["compiler_elapsed_ns"] += time.perf_counter_ns() - started_ns
            compile_depth -= 1

    def timed_control_compile(*args: Any, **kwargs: Any) -> Any:
        return timed_compile(original_control_compile, *args, **kwargs)

    def timed_candidate_compile(*args: Any, **kwargs: Any) -> Any:
        return timed_compile(original_candidate_compile, *args, **kwargs)

    def counted_receipt(*args: Any, **kwargs: Any) -> Any:
        counts["target_admission_receipts"] += 1
        return original_receipt(*args, **kwargs)

    legacy._select_targets = counted_select
    admission._select_targets = counted_candidate_select
    setattr(legacy.NativeIntermediateRefinementProgram, "validate", counted_validate)
    setattr(legacy.NativeIntermediateRefinementProgram, "hashes", counted_hashes)
    setattr(
        prepared.NativePreparedIntermediateRefinementProgram,
        "validate",
        counted_fast_validate,
    )
    prepared.execute_native_intermediate_refinement_program = counted_execute
    per_child.compile_native_prepared_intermediate_refinement_program = (
        timed_control_compile
    )
    per_child.compile_native_single_pass_prepared_intermediate_refinement_program = (
        timed_candidate_compile
    )
    admission._build_target_admission_receipt = counted_receipt
    execute = (
        execute_native_prepared_objective_branch_shared_production_queue
        if mode == "control"
        else execute_native_single_pass_prepared_objective_branch_shared_production_queue
    )
    started_ns = time.monotonic_ns()
    try:
        execution = execute(
            plan,
            module,
            input_spec,
            linear_spec_C=objective,
            threshold=threshold,
            root_refinement=refinement,
            optimizer_policy=optimizer_policy,
            branch_policy=branch_policy,
            compiler_cache=NativeParametricOptimizerTemplateCache(),
            query_id=query_id,
        )
    finally:
        elapsed_ns = time.monotonic_ns() - started_ns
        legacy._select_targets = original_select
        admission._select_targets = original_candidate_select
        setattr(
            legacy.NativeIntermediateRefinementProgram,
            "validate",
            original_validate,
        )
        setattr(legacy.NativeIntermediateRefinementProgram, "hashes", original_hashes)
        setattr(
            prepared.NativePreparedIntermediateRefinementProgram,
            "validate",
            original_fast_validate,
        )
        prepared.execute_native_intermediate_refinement_program = original_execute
        per_child.compile_native_prepared_intermediate_refinement_program = (
            original_control_compile
        )
        per_child.compile_native_single_pass_prepared_intermediate_refinement_program = (
            original_candidate_compile
        )
        admission._build_target_admission_receipt = original_receipt
    counts["compile_reselection"] = max(
        0, counts["compile_target_selection"] - EXPECTED_CHILD_PROGRAMS
    )
    if mode == "prepared":
        _WORKER_CANDIDATES.append((query_id, execution, module, input_spec))
    return execution, counts, elapsed_ns


_WORKER_CANDIDATES: list[tuple[str, Any, Any, Any]] = []
_CHILD_FULL_REPLAY_SELECTORS = 0


def _run_worker(args: argparse.Namespace) -> None:
    import boundflow.runtime.native_intermediate_refinement as legacy
    import boundflow.runtime.native_prepared_intermediate_refinement as prepared
    from boundflow.runtime.native_target_admission import (
        validate_native_single_pass_target_admission_full,
    )

    global _CHILD_FULL_REPLAY_SELECTORS  # pylint: disable=global-statement
    if args.repeat_index not in range(REPEAT_COUNT):
        raise ValueError("NRIR47 repeat index differs")
    _source_nrir45_hash()
    _WORKER_CANDIDATES.clear()
    _CHILD_FULL_REPLAY_SELECTORS = 0
    original_execute_mode = nrir45._execute_mode
    original_payload = nrir45._prepared_payload
    original_worker_validator = nrir45.validate_worker
    original_full = prepared.validate_native_prepared_intermediate_refinement_full

    def counted_full(*full_args: Any, **full_kwargs: Any) -> Any:
        original_select = legacy._select_targets

        def counted_select(*select_args: Any, **select_kwargs: Any) -> Any:
            global _CHILD_FULL_REPLAY_SELECTORS  # pylint: disable=global-statement
            _CHILD_FULL_REPLAY_SELECTORS += 1
            return original_select(*select_args, **select_kwargs)

        legacy._select_targets = counted_select
        try:
            return original_full(*full_args, **full_kwargs)
        finally:
            legacy._select_targets = original_select

    nrir45._execute_mode = _execute_mode
    nrir45._prepared_payload = _prepared_payload
    nrir45.validate_worker = lambda *_args, **_kwargs: None
    prepared.validate_native_prepared_intermediate_refinement_full = counted_full
    try:
        nrir45._run_worker(args)
    finally:
        nrir45._execute_mode = original_execute_mode
        nrir45._prepared_payload = original_payload
        nrir45.validate_worker = original_worker_validator
        prepared.validate_native_prepared_intermediate_refinement_full = original_full
    if len(_WORKER_CANDIDATES) != len(EXPECTED_CLAUSES):
        raise ValueError("NRIR47 candidate execution coverage differs")
    root_payloads: dict[str, dict[str, Any]] = {}
    root_full_replay_selectors = 0
    original_select = legacy._select_targets

    def counted_root_select(*select_args: Any, **select_kwargs: Any) -> Any:
        nonlocal root_full_replay_selectors
        root_full_replay_selectors += 1
        return original_select(*select_args, **select_kwargs)

    legacy._select_targets = counted_root_select
    try:
        for query_id, execution, module, input_spec in _WORKER_CANDIDATES:
            root = execution.node_refinements[0].program
            validate_native_single_pass_target_admission_full(root, module, input_spec)
            root_payloads[query_id] = _target_stack_payload(root)
    finally:
        legacy._select_targets = original_select
    if _CHILD_FULL_REPLAY_SELECTORS != EXPECTED_CHILD_PROGRAMS * len(
        EXPECTED_CLAUSES
    ) or root_full_replay_selectors != len(EXPECTED_CLAUSES):
        raise ValueError("NRIR47 explicit full replay selector count differs")
    worker = _load_json(args.result_json.resolve())
    worker["schema_version"] = WORKER_SCHEMA_VERSION
    worker["source"] = {
        "native_code_revision": _code_revision(),
        "source_nrir45_hash": _source_nrir45_hash(),
    }
    row_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for row in worker["rows"]:
        row["implementation"] = (
            "nrir45-prepared-control"
            if row["mode"] == "control"
            else "nrir47-single-pass-candidate"
        )
        if row["mode"] == "prepared":
            row["full_replay_count"] = EXPECTED_RECEIPTS
            row["full_replay_selector_count"] = EXPECTED_RECEIPTS
            row["target_admission_receipt_count"] = row["call_counts"][
                "target_admission_receipts"
            ]
            row["representative_root_admission_payload"] = root_payloads[
                row["query_id"]
            ]
        else:
            row["full_replay_selector_count"] = 0
            row["target_admission_receipt_count"] = 0
            row["representative_root_admission_payload"] = None
        row["row_hash"] = _canonical_hash(
            {key: item for key, item in row.items() if key != "row_hash"}
        )
        row_by_key[(row["original_clause_index"], row["mode"])] = row
    for parity in worker["parities"]:
        ordinal = parity["original_clause_index"]
        parity["control_row_hash"] = row_by_key[(ordinal, "control")]["row_hash"]
        parity["prepared_row_hash"] = row_by_key[(ordinal, "prepared")]["row_hash"]
        parity["parity_hash"] = _canonical_hash(
            {key: item for key, item in parity.items() if key != "parity_hash"}
        )
    worker["worker_hash"] = _canonical_hash(
        {key: item for key, item in worker.items() if key != "worker_hash"}
    )
    validate_worker(worker, repeat_index=args.repeat_index)
    _write_json(args.result_json.resolve(), worker)
    print(
        _canonical_json(
            {
                "repeat_index": args.repeat_index,
                "worker_hash": worker["worker_hash"],
                "counts": {
                    f"c{row['original_clause_index']}:{row['mode']}": row["call_counts"]
                    for row in worker["rows"]
                },
            }
        ),
        flush=True,
    )


def validate_worker(value: Mapping[str, Any], *, repeat_index: int) -> None:
    rows = value.get("rows")
    parities = value.get("parities")
    if (
        value.get("schema_version") != WORKER_SCHEMA_VERSION
        or value.get("source", {}).get("native_code_revision") != _code_revision()
        or value.get("source", {}).get("source_nrir45_hash") != _source_nrir45_hash()
        or value.get("repeat_index") != repeat_index
        or value.get("paired_order") != list(PAIRED_ORDERS[repeat_index])
        or value.get("selected_original_clause_indices") != list(EXPECTED_CLAUSES)
        or not isinstance(rows, list)
        or len(rows) != 4
        or not isinstance(parities, list)
        or len(parities) != 2
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR47 worker envelope differs")
    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        ordinal = row.get("original_clause_index")
        mode = row.get("mode")
        counts = row.get("call_counts", {})
        semantic = {key: item for key, item in row.items() if key != "row_hash"}
        if (
            ordinal not in EXPECTED_CLAUSES
            or mode not in {"control", "prepared"}
            or row.get("accepted_nodes") != REQUIRED_NODES
            or row.get("sibling_group_count") != REQUIRED_SIBLING_GROUPS
            or row.get("worst_active_lower") != EXPECTED_WORST_ACTIVE_LOWERS[ordinal]
            or row.get("queue_elapsed_ns", 0) <= 0
            or row.get("measured_elapsed_ns", 0) <= 0
            or row.get("performance_claimed") is not False
            or row.get("row_hash") != _canonical_hash(semantic)
            or not all(isinstance(item, int) and item >= 0 for item in counts.values())
            or counts.get("runtime_target_selection") != EXPECTED_CHILD_PROGRAMS
        ):
            raise ValueError("NRIR47 worker row differs")
        if mode == "prepared":
            prepared_payload = row.get("representative_prepared_payload")
            root_payload = row.get("representative_root_admission_payload")
            if (
                row.get("implementation") != "nrir47-single-pass-candidate"
                or counts.get("compile_target_selection") != EXPECTED_CHILD_PROGRAMS
                or counts.get("compile_reselection") != 0
                or counts.get("other_target_selection") != 0
                or counts.get("target_admission_receipts") != EXPECTED_RECEIPTS
                or row.get("target_admission_receipt_count") != EXPECTED_RECEIPTS
                or row.get("prepared_capsule_count") != EXPECTED_CHILD_PROGRAMS
                or row.get("full_replay_count") != EXPECTED_RECEIPTS
                or row.get("full_replay_selector_count") != EXPECTED_RECEIPTS
                or not isinstance(prepared_payload, Mapping)
                or not isinstance(root_payload, Mapping)
            ):
                raise ValueError("NRIR47 candidate ownership differs")
            _validate_prepared_payload(prepared_payload)
            _validate_target_stack_payload(root_payload)
        elif (
            row.get("implementation") != "nrir45-prepared-control"
            or counts.get("compile_target_selection") <= EXPECTED_CHILD_PROGRAMS
            or counts.get("compile_reselection") <= 0
            or counts.get("target_admission_receipts") != 0
            or row.get("target_admission_receipt_count") != 0
            or row.get("full_replay_selector_count") != 0
            or row.get("representative_root_admission_payload") is not None
        ):
            raise ValueError("NRIR47 control ownership differs")
        by_key[(ordinal, mode)] = row
    expected_keys = {
        (ordinal, mode)
        for ordinal in EXPECTED_CLAUSES
        for mode in ("control", "prepared")
    }
    if set(by_key) != expected_keys:
        raise ValueError("NRIR47 row coverage differs")
    for parity in parities:
        ordinal = parity["original_clause_index"]
        semantic = {key: item for key, item in parity.items() if key != "parity_hash"}
        if (
            parity.get("repeat_index") != repeat_index
            or ordinal not in EXPECTED_CLAUSES
            or parity.get("control_row_hash")
            != by_key[(ordinal, "control")]["row_hash"]
            or parity.get("prepared_row_hash")
            != by_key[(ordinal, "prepared")]["row_hash"]
            or parity.get("exact") is not True
            or parity.get("parity_hash") != _canonical_hash(semantic)
        ):
            raise ValueError("NRIR47 parity differs")
    semantic_worker = {key: item for key, item in value.items() if key != "worker_hash"}
    if value.get("worker_hash") != _canonical_hash(semantic_worker):
        raise ValueError("NRIR47 worker hash differs")


def _median_mad(values: list[int]) -> tuple[int, int]:
    median = int(statistics.median(values))
    mad = int(statistics.median(abs(value - median) for value in values))
    return median, mad


def _metric(
    *, name: str, controls: list[int], candidates: list[int], maximum_ratio: float
) -> dict[str, Any]:
    control_median, control_mad = _median_mad(controls)
    candidate_median, candidate_mad = _median_mad(candidates)
    improvement = control_median - candidate_median
    return {
        "name": name,
        "control_ns": controls,
        "candidate_ns": candidates,
        "control_median_ns": control_median,
        "candidate_median_ns": candidate_median,
        "control_mad_ns": control_mad,
        "candidate_mad_ns": candidate_mad,
        "candidate_to_control_median_ratio": candidate_median / control_median,
        "maximum_median_ratio": maximum_ratio,
        "median_improvement_ns": improvement,
        "improvement_exceeds_pooled_mad": improvement > max(control_mad, candidate_mad),
    }


def _build_formal(
    workers: Sequence[Mapping[str, Any]], *, workload: Mapping[str, Any]
) -> dict[str, Any]:
    queue_metrics: list[dict[str, Any]] = []
    for ordinal in EXPECTED_CLAUSES:
        controls = [
            row["measured_elapsed_ns"]
            for worker in workers
            for row in worker["rows"]
            if row["original_clause_index"] == ordinal and row["mode"] == "control"
        ]
        candidates = [
            row["measured_elapsed_ns"]
            for worker in workers
            for row in worker["rows"]
            if row["original_clause_index"] == ordinal and row["mode"] == "prepared"
        ]
        metric = _metric(
            name=f"clause-{ordinal}-end-to-end-queue",
            controls=controls,
            candidates=candidates,
            maximum_ratio=MAXIMUM_QUEUE_MEDIAN_RATIO,
        )
        metric["original_clause_index"] = ordinal
        queue_metrics.append(metric)
    compiler_metric = _metric(
        name="two-clause-child-compiler",
        controls=[
            row["call_counts"]["compiler_elapsed_ns"]
            for worker in workers
            for row in worker["rows"]
            if row["mode"] == "control"
        ],
        candidates=[
            row["call_counts"]["compiler_elapsed_ns"]
            for worker in workers
            for row in worker["rows"]
            if row["mode"] == "prepared"
        ],
        maximum_ratio=MAXIMUM_COMPILER_MEDIAN_RATIO,
    )
    parity_passed = all(
        parity["exact"] for worker in workers for parity in worker["parities"]
    )
    ownership_passed = all(
        row["call_counts"]["compile_target_selection"] == EXPECTED_CHILD_PROGRAMS
        and row["call_counts"]["compile_reselection"] == 0
        and row["call_counts"]["runtime_target_selection"] == EXPECTED_CHILD_PROGRAMS
        and row["target_admission_receipt_count"] == EXPECTED_RECEIPTS
        and row["full_replay_selector_count"] == EXPECTED_RECEIPTS
        for worker in workers
        for row in worker["rows"]
        if row["mode"] == "prepared"
    )
    compiler_timing_passed = bool(
        compiler_metric["candidate_to_control_median_ratio"]
        <= MAXIMUM_COMPILER_MEDIAN_RATIO
        and compiler_metric["improvement_exceeds_pooled_mad"]
    )
    queue_timing_passed = all(
        metric["candidate_to_control_median_ratio"] <= MAXIMUM_QUEUE_MEDIAN_RATIO
        and metric["improvement_exceeds_pooled_mad"]
        for metric in queue_metrics
    )
    phase_a_go = (
        parity_passed
        and ownership_passed
        and compiler_timing_passed
        and queue_timing_passed
    )
    decision = {
        "parity_passed": parity_passed,
        "ownership_passed": ownership_passed,
        "compiler_timing_passed": compiler_timing_passed,
        "queue_timing_passed": queue_timing_passed,
        "phase_a_go": phase_a_go,
        "next_route": "run_nrir47_phase_b" if phase_a_go else "stop_nrir47",
        "reason": (
            "exact semantics, receipt ownership, compiler, and queue gates passed"
            if phase_a_go
            else "NRIR47 Phase-A gate failed; Phase B is gated off"
        ),
    }
    formal: dict[str, Any] = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "source": {
            "base_revision": BASE_REVISION,
            "native_code_revision": _code_revision(),
            "source_nrir45_hash": _source_nrir45_hash(),
            "workload": nrir45._public_workload(workload),
        },
        "contract": {
            "repeat_count": REPEAT_COUNT,
            "paired_orders": [list(item) for item in PAIRED_ORDERS],
            "clauses": list(EXPECTED_CLAUSES),
            "required_nodes": REQUIRED_NODES,
            "required_sibling_groups": REQUIRED_SIBLING_GROUPS,
            "expected_child_programs": EXPECTED_CHILD_PROGRAMS,
            "expected_receipts_and_full_replays": EXPECTED_RECEIPTS,
            "maximum_compiler_median_ratio": MAXIMUM_COMPILER_MEDIAN_RATIO,
            "maximum_queue_median_ratio": MAXIMUM_QUEUE_MEDIAN_RATIO,
            "torch_threads": TORCH_THREADS,
        },
        "workers": [
            {
                "repeat_index": worker["repeat_index"],
                "worker_hash": worker["worker_hash"],
            }
            for worker in workers
        ],
        "compiler_metric": compiler_metric,
        "queue_metrics": queue_metrics,
        "decision": decision,
        "decision_hash": _canonical_hash(decision),
        "status": "validated-reduced" if phase_a_go else "validated-no-go",
        "performance_claimed": False,
    }
    formal["formal_hash"] = _canonical_hash(formal)
    return formal


def validate_formal(value: Mapping[str, Any]) -> None:
    contract = value.get("contract", {})
    decision = value.get("decision", {})
    semantic = {key: item for key, item in value.items() if key != "formal_hash"}
    expected_phase_a = bool(
        decision.get("parity_passed")
        and decision.get("ownership_passed")
        and decision.get("compiler_timing_passed")
        and decision.get("queue_timing_passed")
    )
    if (
        value.get("schema_version") != FORMAL_SCHEMA_VERSION
        or value.get("source", {}).get("base_revision") != BASE_REVISION
        or value.get("source", {}).get("native_code_revision") != _code_revision()
        or value.get("source", {}).get("source_nrir45_hash") != _source_nrir45_hash()
        or contract.get("repeat_count") != REPEAT_COUNT
        or contract.get("paired_orders") != [list(item) for item in PAIRED_ORDERS]
        or contract.get("clauses") != list(EXPECTED_CLAUSES)
        or contract.get("required_nodes") != REQUIRED_NODES
        or contract.get("required_sibling_groups") != REQUIRED_SIBLING_GROUPS
        or contract.get("expected_child_programs") != EXPECTED_CHILD_PROGRAMS
        or contract.get("expected_receipts_and_full_replays") != EXPECTED_RECEIPTS
        or contract.get("maximum_compiler_median_ratio")
        != MAXIMUM_COMPILER_MEDIAN_RATIO
        or contract.get("maximum_queue_median_ratio") != MAXIMUM_QUEUE_MEDIAN_RATIO
        or contract.get("torch_threads") != TORCH_THREADS
        or len(value.get("workers", [])) != REPEAT_COUNT
        or len(value.get("queue_metrics", [])) != len(EXPECTED_CLAUSES)
        or decision.get("phase_a_go") != expected_phase_a
        or value.get("decision_hash") != _canonical_hash(decision)
        or value.get("status")
        != ("validated-reduced" if expected_phase_a else "validated-no-go")
        or value.get("performance_claimed") is not False
        or value.get("formal_hash") != _canonical_hash(semantic)
    ):
        raise ValueError("NRIR47 formal envelope differs")


def _manifest(artifact_dir: Path, formal: Mapping[str, Any]) -> dict[str, Any]:
    files = {
        str(path.relative_to(artifact_dir)): _file_sha256(path)
        for path in sorted(artifact_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    value: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "formal_hash": formal["formal_hash"],
        "files": files,
        "performance_claimed": False,
    }
    value["manifest_hash"] = _canonical_hash(value)
    return value


def _run_subprocess(command: list[str], log_path: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=WORKER_TIMEOUT_SECONDS,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"NRIR47 worker failed with exit {completed.returncode}: {log_path}"
        )


def _generate(args: argparse.Namespace) -> None:
    if args.torch_threads != TORCH_THREADS:
        raise ValueError("NRIR47 torch thread count differs")
    workload = nrir45._workload(args.benchmark_root)
    artifact_dir = args.artifact_dir.resolve()
    workers = []
    for repeat_index in range(REPEAT_COUNT):
        shard = artifact_dir / "shards" / f"repeat-{repeat_index}.json"
        log = artifact_dir / "logs" / f"repeat-{repeat_index}.log"
        _run_subprocess(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--model",
                str(workload["model"]),
                "--property",
                str(workload["property"]),
                "--result-json",
                str(shard),
                "--repeat-index",
                str(repeat_index),
                "--torch-threads",
                str(args.torch_threads),
            ],
            log,
        )
        worker = _load_json(shard)
        validate_worker(worker, repeat_index=repeat_index)
        workers.append(worker)
    formal = _build_formal(workers, workload=workload)
    validate_formal(formal)
    _write_json(artifact_dir / "formal.json", formal)
    _write_json(artifact_dir / "manifest.json", _manifest(artifact_dir, formal))
    print(
        _canonical_json(
            {
                "status": formal["status"],
                "formal_hash": formal["formal_hash"],
                "decision": formal["decision"],
                "compiler_metric": formal["compiler_metric"],
                "queue_metrics": formal["queue_metrics"],
            }
        )
    )


def _replay(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    manifest = _load_json(artifact_dir / "manifest.json")
    files = manifest.get("files", {})
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("performance_claimed") is not False
        or manifest.get("manifest_hash")
        != _canonical_hash(
            {key: item for key, item in manifest.items() if key != "manifest_hash"}
        )
        or any(
            _file_sha256(artifact_dir / path) != digest
            for path, digest in files.items()
        )
    ):
        raise ValueError("NRIR47 manifest differs")
    workers = []
    for repeat_index in range(REPEAT_COUNT):
        worker = _load_json(artifact_dir / "shards" / f"repeat-{repeat_index}.json")
        validate_worker(worker, repeat_index=repeat_index)
        workers.append(worker)
    formal = _load_json(artifact_dir / "formal.json")
    validate_formal(formal)
    if manifest.get("formal_hash") != formal["formal_hash"]:
        raise ValueError("NRIR47 manifest/formal differs")
    rebuilt = _build_formal(workers, workload=nrir45._workload(args.benchmark_root))
    if rebuilt != formal:
        raise ValueError("NRIR47 formal replay differs")
    root_payload = next(
        row["representative_root_admission_payload"]
        for worker in workers
        for row in worker["rows"]
        if row["mode"] == "prepared"
    )
    tampered = copy.deepcopy(root_payload)
    receipt = NativeTargetAdmissionReceiptIR(**tampered["receipt"])
    receipt = replace(
        receipt, target_table_hash="f" * 64, admission_receipt_hash="0" * 64
    )
    receipt = replace(receipt, admission_receipt_hash=receipt.expected_receipt_hash())
    task_module, schedule = lower_native_target_admission_ir(
        source_plan_hash=tampered["source_hashes"]["refinement_plan_hash"],
        receipt=receipt,
    )
    tampered["receipt"] = receipt.to_dict()
    tampered["task_module"] = task_module.to_dict(receipt=receipt)
    tampered["schedule"] = schedule.to_dict(receipt=receipt, task_module=task_module)
    tampered["payload_hash"] = _canonical_hash(
        {key: item for key, item in tampered.items() if key != "payload_hash"}
    )
    try:
        _validate_target_stack_payload(tampered)
    except ValueError:
        tamper_rejected = True
    else:
        tamper_rejected = False
    if not tamper_rejected:
        raise ValueError("NRIR47 synchronized outer-rehash tamper was accepted")
    print(
        _canonical_json(
            {
                "status": "replay-passed",
                "formal_hash": formal["formal_hash"],
                "typed_receipts_replayed": (
                    REPEAT_COUNT * len(EXPECTED_CLAUSES) * EXPECTED_RECEIPTS
                ),
                "outer_rehash_tamper_rejected": True,
            }
        )
    )


def main() -> None:
    args = _parse_args()
    if args.command == "worker":
        _run_worker(args)
    elif args.command == "generate":
        _generate(args)
    elif args.command == "replay":
        _replay(args)
    else:
        raise AssertionError("unreachable NRIR47 command")


if __name__ == "__main__":
    main()
