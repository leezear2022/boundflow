#!/usr/bin/env python3
"""Generate or replay NRIR48 top-2 production execution-cost evidence."""

# pylint: disable=too-many-lines,too-many-locals,too-many-statements
# pylint: disable=too-many-branches,too-many-boolean-expressions
# pylint: disable=missing-function-docstring,import-outside-toplevel,protected-access
# pylint: disable=duplicate-code,cell-var-from-loop,too-many-arguments

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from functools import wraps
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from scripts import run_prepared_intermediate_refinement_formal as nrir45

FORMAL_SCHEMA_VERSION = "boundflow.top2-production-cost-attribution-formal/v1"
WORKER_SCHEMA_VERSION = "boundflow.top2-production-cost-attribution-worker/v1"
MANIFEST_SCHEMA_VERSION = "boundflow.top2-production-cost-attribution-manifest/v1"
ARTIFACT_DIR = Path(
    "artifacts/top2-production-execution-cost-attribution/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-phase0-v1"
)
SOURCE_NRIR45_PHASE_B = Path(
    "artifacts/prepared-intermediate-refinement/"
    "vnncomp21-resnet2b-property0-three-repeat-cpu-phase-b-v1/formal.json"
)
EXPECTED_SOURCE_NRIR45_PHASE_B_HASH = (
    "4ae71919b5c4d6e8d6162df8bb7d14143a705f60a599f8e4bfa30d084c1a01f8"
)
BASE_REVISION = "1e44949c1eff9ae9127a59be07b2d3caf28544cc"
EXPECTED_CLAUSES = nrir45.EXPECTED_CLAUSES
EXPECTED_WORST_ACTIVE_LOWERS = nrir45.EXPECTED_WORST_ACTIVE_LOWERS
PAIRED_ORDERS = nrir45.PAIRED_ORDERS
REPEAT_COUNT = nrir45.REPEAT_COUNT
TORCH_THREADS = nrir45.TORCH_THREADS
WORKER_TIMEOUT_SECONDS = nrir45.WORKER_TIMEOUT_SECONDS
REQUIRED_NODES = nrir45.REQUIRED_NODES
REQUIRED_SIBLING_GROUPS = nrir45.REQUIRED_SIBLING_GROUPS
MAXIMUM_PROFILE_MEDIAN_RATIO = 1.05
MAXIMUM_CLOSURE_ERROR_RATIO = 0.01
MINIMUM_DOMINANT_MEDIAN_SHARE = 0.20
MAXIMUM_DOMINANT_SHARE_RANGE = 0.10
MINIMUM_CHILD_SUBCATEGORY_SHARE = 0.30

TOP_CATEGORY_KEYS = (
    "child_refinement_compile_ns",
    "child_refinement_execute_ns",
    "optimizer_prepare_ns",
    "optimizer_execute_ns",
    "branch_bind_score_ns",
    "materialize_commit_ns",
    "queue_control_residual_ns",
)


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
        "boundflow/runtime/native_intermediate_refinement.py",
        "boundflow/runtime/native_prepared_intermediate_refinement.py",
        "boundflow/runtime/native_prepared_per_child_refinement.py",
        "boundflow/runtime/native_prepared_shared_parametric_ancestral.py",
        "boundflow/runtime/native_prepared_objective_branch_shared_production_queue.py",
        "boundflow/runtime/native_parametric_optimizer.py",
        "scripts/run_prepared_intermediate_refinement_formal.py",
        "scripts/run_top2_production_execution_cost_attribution.py",
    )
    return _canonical_hash({path: _file_sha256(root / path) for path in paths})


def _source_nrir45_hash() -> str:
    value = _load_json(_repo_root() / SOURCE_NRIR45_PHASE_B)
    if value.get("formal_payload_hash") != EXPECTED_SOURCE_NRIR45_PHASE_B_HASH:
        raise ValueError("NRIR48 frozen NRIR45 Phase-B source differs")
    return value["formal_payload_hash"]


@dataclass
class _Timer:
    calls: int = 0
    inclusive_ns: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"calls": self.calls, "inclusive_ns": self.inclusive_ns}


class _Profiler:
    def __init__(self) -> None:
        self.timers: dict[str, _Timer] = {}
        self.execute_depth = 0

    def record(self, name: str, elapsed_ns: int) -> None:
        timer = self.timers.setdefault(name, _Timer())
        timer.calls += 1
        timer.inclusive_ns += elapsed_ns

    def wrap(
        self,
        function: Callable[..., Any],
        name: str,
        *,
        execute_only: bool = False,
        execute_scope: bool = False,
    ) -> Callable[..., Any]:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if execute_only and self.execute_depth == 0:
                return function(*args, **kwargs)
            if execute_scope:
                self.execute_depth += 1
            started_ns = time.perf_counter_ns()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed_ns = time.perf_counter_ns() - started_ns
                self.record(name, elapsed_ns)
                if execute_scope:
                    self.execute_depth -= 1

        return wrapped

    def value(self, name: str) -> int:
        return self.timers.get(name, _Timer()).inclusive_ns

    def calls(self, name: str) -> int:
        return self.timers.get(name, _Timer()).calls

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {name: timer.to_dict() for name, timer in sorted(self.timers.items())}


@dataclass(frozen=True)
class _Patch:
    owner: Any
    name: str
    original: Any


def _patch(patches: list[_Patch], owner: Any, name: str, replacement: Any) -> None:
    patches.append(_Patch(owner=owner, name=name, original=getattr(owner, name)))
    setattr(owner, name, replacement)


def _restore(patches: list[_Patch]) -> None:
    for patch in reversed(patches):
        setattr(patch.owner, patch.name, patch.original)


def _materialize_action_ns(execution: Any) -> int:
    return sum(
        action.elapsed_ns
        for batch in execution.queue.trace.batches
        for action in batch.actions
        if action.kind.value == "materialize_node_results"
    )


def _top_categories(
    profiler: _Profiler, execution: Any, *, queue_elapsed_ns: int
) -> dict[str, int]:
    optimizer_prepare = sum(
        profiler.value(name)
        for name in (
            "optimizer_warm_state",
            "optimizer_template_acquire",
            "optimizer_instantiate",
            "optimizer_ir_lower",
        )
    )
    materialize_commit = _materialize_action_ns(execution) + sum(
        profiler.value(name)
        for name in (
            "queue_make_batch_commit",
            "queue_task_emit",
            "queue_schedule_lower",
        )
    )
    categories = {
        "child_refinement_compile_ns": profiler.value("child_refinement_compile"),
        "child_refinement_execute_ns": profiler.value("child_refinement_execute"),
        "optimizer_prepare_ns": optimizer_prepare,
        "optimizer_execute_ns": profiler.value("optimizer_execute"),
        "branch_bind_score_ns": profiler.value("branch_bind_score"),
        "materialize_commit_ns": materialize_commit,
    }
    attributed = sum(categories.values())
    if attributed > queue_elapsed_ns:
        raise ValueError("NRIR48 top-level timers overlap queue wall time")
    categories["queue_control_residual_ns"] = queue_elapsed_ns - attributed
    return categories


def _child_execute_categories(
    profiler: _Profiler, *, child_execute_ns: int
) -> dict[str, int]:
    categories = {
        "fast_validate_ns": profiler.value("execute_fast_validate"),
        "runtime_target_select_ns": profiler.value("execute_target_select"),
        "selected_crown_ns": profiler.value("execute_selected_crown"),
        "propagate_forward_ns": profiler.value("execute_propagate_forward"),
    }
    attributed = sum(categories.values())
    if attributed > child_execute_ns:
        raise ValueError("NRIR48 child execution timers overlap parent wall time")
    categories["refinement_hash_trace_residual_ns"] = child_execute_ns - attributed
    return categories


_ATTRIBUTIONS: dict[tuple[str, str], dict[str, Any]] = {}


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
    from boundflow.runtime import native_intermediate_refinement as legacy
    from boundflow.runtime import native_prepared_intermediate_refinement as prepared
    from boundflow.runtime import (
        native_prepared_objective_branch_shared_production_queue as queue,
    )
    from boundflow.runtime import native_prepared_per_child_refinement as per_child
    from boundflow.runtime import native_prepared_shared_parametric_ancestral as shared
    from boundflow.runtime.native_parametric_optimizer import (
        NativeParametricOptimizerTemplateCache,
    )

    profiler = _Profiler()
    patches: list[_Patch] = []
    if mode == "prepared":
        _patch(
            patches,
            per_child,
            "compile_native_prepared_intermediate_refinement_program",
            profiler.wrap(
                per_child.compile_native_prepared_intermediate_refinement_program,
                "child_refinement_compile",
            ),
        )
        _patch(
            patches,
            per_child,
            "execute_native_prepared_intermediate_refinement_program",
            profiler.wrap(
                per_child.execute_native_prepared_intermediate_refinement_program,
                "child_refinement_execute",
                execute_scope=True,
            ),
        )
        _patch(
            patches,
            prepared.NativePreparedIntermediateRefinementProgram,
            "validate",
            profiler.wrap(
                prepared.NativePreparedIntermediateRefinementProgram.validate,
                "execute_fast_validate",
                execute_only=True,
            ),
        )
        for owner, name, timer_name in (
            (legacy, "_select_targets", "execute_target_select"),
            (legacy, "_run_selected_crown", "execute_selected_crown"),
            (legacy, "_forward_ibp_trace_mlp", "execute_propagate_forward"),
        ):
            _patch(
                patches,
                owner,
                name,
                profiler.wrap(getattr(owner, name), timer_name, execute_only=True),
            )
        for owner, name, timer_name in (
            (shared, "_build_batched_parent_warm_state", "optimizer_warm_state"),
            (
                shared,
                "instantiate_native_parametric_optimizer",
                "optimizer_instantiate",
            ),
            (shared, "lower_native_production_verifier_ir", "optimizer_ir_lower"),
            (shared, "execute_native_parametric_optimizer", "optimizer_execute"),
            (queue, "bind_prevalidated_objective_branches", "branch_bind_score"),
            (queue, "_make_batch_commit", "queue_make_batch_commit"),
            (queue, "_task", "queue_task_emit"),
            (
                queue,
                "lower_native_shared_parametric_ancestral_schedule",
                "queue_schedule_lower",
            ),
        ):
            _patch(
                patches,
                owner,
                name,
                profiler.wrap(getattr(owner, name), timer_name),
            )
        _patch(
            patches,
            NativeParametricOptimizerTemplateCache,
            "acquire",
            profiler.wrap(
                NativeParametricOptimizerTemplateCache.acquire,
                "optimizer_template_acquire",
            ),
        )
    started_ns = time.monotonic_ns()
    try:
        execution = (
            queue.execute_native_prepared_objective_branch_shared_production_queue(
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
        )
    finally:
        measured_ns = time.monotonic_ns() - started_ns
        _restore(patches)
    counts = {
        "instrumentation_enabled": int(mode == "prepared"),
        "child_compile_calls": profiler.calls("child_refinement_compile"),
        "child_execute_calls": profiler.calls("child_refinement_execute"),
        "optimizer_execute_calls": profiler.calls("optimizer_execute"),
        "branch_bind_calls": profiler.calls("branch_bind_score"),
    }
    if mode == "prepared":
        queue_elapsed_ns = execution.trace.queue_elapsed_ns
        top = _top_categories(profiler, execution, queue_elapsed_ns=queue_elapsed_ns)
        child = _child_execute_categories(
            profiler,
            child_execute_ns=top["child_refinement_execute_ns"],
        )
        _ATTRIBUTIONS[(query_id, mode)] = {
            "queue_elapsed_ns": queue_elapsed_ns,
            "raw_timers": profiler.to_dict(),
            "materialize_action_ns": _materialize_action_ns(execution),
            "top_categories": top,
            "child_execute_categories": child,
            "closure_error_ns": queue_elapsed_ns - sum(top.values()),
        }
    return execution, counts, measured_ns


def _run_worker(args: argparse.Namespace) -> None:
    if args.repeat_index not in range(REPEAT_COUNT):
        raise ValueError("NRIR48 repeat index differs")
    _source_nrir45_hash()
    _ATTRIBUTIONS.clear()
    original_execute_mode = nrir45._execute_mode
    original_validate_worker = nrir45.validate_worker
    nrir45._execute_mode = _execute_mode
    nrir45.validate_worker = lambda *_args, **_kwargs: None
    try:
        nrir45._run_worker(args)
    finally:
        nrir45._execute_mode = original_execute_mode
        nrir45.validate_worker = original_validate_worker
    worker = _load_json(args.result_json.resolve())
    worker["schema_version"] = WORKER_SCHEMA_VERSION
    worker["source"] = {
        "native_code_revision": _code_revision(),
        "source_nrir45_phase_b_hash": _source_nrir45_hash(),
    }
    row_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for row in worker["rows"]:
        mode = row["mode"]
        row["implementation"] = "nrir45-prepared-production"
        row["attribution"] = (
            None if mode == "control" else _ATTRIBUTIONS[(row["query_id"], mode)]
        )
        row["instrumentation"] = "disabled" if mode == "control" else "profiled"
        row["row_hash"] = _canonical_hash(
            {key: item for key, item in row.items() if key != "row_hash"}
        )
        row_by_key[(row["original_clause_index"], mode)] = row
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
                "profile_categories": {
                    f"c{row['original_clause_index']}": row["attribution"][
                        "top_categories"
                    ]
                    for row in worker["rows"]
                    if row["mode"] == "prepared"
                },
            }
        ),
        flush=True,
    )


def _validate_attribution(value: Mapping[str, Any]) -> None:
    queue_elapsed_ns = value.get("queue_elapsed_ns")
    raw = value.get("raw_timers")
    top = value.get("top_categories")
    child = value.get("child_execute_categories")
    if (
        not isinstance(queue_elapsed_ns, int)
        or queue_elapsed_ns <= 0
        or not isinstance(raw, Mapping)
        or not isinstance(top, Mapping)
        or set(top) != set(TOP_CATEGORY_KEYS)
        or not isinstance(child, Mapping)
        or any(
            not isinstance(item, int) or item < 0
            for item in (*top.values(), *child.values())
        )
        or sum(top.values()) != queue_elapsed_ns
        or value.get("closure_error_ns") != 0
        or sum(child.values()) != top["child_refinement_execute_ns"]
    ):
        raise ValueError("NRIR48 attribution closure differs")
    expected_top = {
        "child_refinement_compile_ns": raw["child_refinement_compile"]["inclusive_ns"],
        "child_refinement_execute_ns": raw["child_refinement_execute"]["inclusive_ns"],
        "optimizer_prepare_ns": sum(
            raw[name]["inclusive_ns"]
            for name in (
                "optimizer_warm_state",
                "optimizer_template_acquire",
                "optimizer_instantiate",
                "optimizer_ir_lower",
            )
        ),
        "optimizer_execute_ns": raw["optimizer_execute"]["inclusive_ns"],
        "branch_bind_score_ns": raw["branch_bind_score"]["inclusive_ns"],
        "materialize_commit_ns": value["materialize_action_ns"]
        + sum(
            raw[name]["inclusive_ns"]
            for name in (
                "queue_make_batch_commit",
                "queue_task_emit",
                "queue_schedule_lower",
            )
        ),
    }
    expected_top["queue_control_residual_ns"] = queue_elapsed_ns - sum(
        expected_top.values()
    )
    expected_child = {
        "fast_validate_ns": raw["execute_fast_validate"]["inclusive_ns"],
        "runtime_target_select_ns": raw["execute_target_select"]["inclusive_ns"],
        "selected_crown_ns": raw["execute_selected_crown"]["inclusive_ns"],
        "propagate_forward_ns": raw["execute_propagate_forward"]["inclusive_ns"],
    }
    expected_child["refinement_hash_trace_residual_ns"] = top[
        "child_refinement_execute_ns"
    ] - sum(expected_child.values())
    if dict(top) != expected_top or dict(child) != expected_child:
        raise ValueError("NRIR48 attribution derivation differs")


def validate_worker(value: Mapping[str, Any], *, repeat_index: int) -> None:
    rows = value.get("rows")
    parities = value.get("parities")
    if (
        value.get("schema_version") != WORKER_SCHEMA_VERSION
        or value.get("source", {}).get("native_code_revision") != _code_revision()
        or value.get("source", {}).get("source_nrir45_phase_b_hash")
        != _source_nrir45_hash()
        or value.get("repeat_index") != repeat_index
        or value.get("paired_order") != list(PAIRED_ORDERS[repeat_index])
        or value.get("selected_original_clause_indices") != list(EXPECTED_CLAUSES)
        or not isinstance(rows, list)
        or len(rows) != 4
        or not isinstance(parities, list)
        or len(parities) != 2
        or value.get("performance_claimed") is not False
    ):
        raise ValueError("NRIR48 worker envelope differs")
    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        ordinal = row.get("original_clause_index")
        mode = row.get("mode")
        semantic = {key: item for key, item in row.items() if key != "row_hash"}
        if (
            ordinal not in EXPECTED_CLAUSES
            or mode not in {"control", "prepared"}
            or row.get("implementation") != "nrir45-prepared-production"
            or row.get("accepted_nodes") != REQUIRED_NODES
            or row.get("sibling_group_count") != REQUIRED_SIBLING_GROUPS
            or row.get("worst_active_lower") != EXPECTED_WORST_ACTIVE_LOWERS[ordinal]
            or row.get("prepared_capsule_count") != 30
            or row.get("queue_elapsed_ns", 0) <= 0
            or row.get("measured_elapsed_ns", 0) <= 0
            or row.get("row_hash") != _canonical_hash(semantic)
            or row.get("performance_claimed") is not False
        ):
            raise ValueError("NRIR48 worker row differs")
        if mode == "prepared":
            if (
                row.get("instrumentation") != "profiled"
                or row.get("full_replay_count") != 30
                or not isinstance(row.get("attribution"), Mapping)
                or row.get("call_counts", {}).get("instrumentation_enabled") != 1
                or row.get("call_counts", {}).get("child_compile_calls") != 30
                or row.get("call_counts", {}).get("child_execute_calls") != 30
                or row.get("call_counts", {}).get("optimizer_execute_calls") != 16
                or row.get("call_counts", {}).get("branch_bind_calls") != 16
            ):
                raise ValueError("NRIR48 profile ownership differs")
            _validate_attribution(row["attribution"])
        elif (
            row.get("instrumentation") != "disabled"
            or row.get("full_replay_count") != 0
            or row.get("attribution") is not None
            or row.get("call_counts", {}).get("instrumentation_enabled") != 0
        ):
            raise ValueError("NRIR48 control ownership differs")
        by_key[(ordinal, mode)] = row
    expected_keys = {
        (ordinal, mode)
        for ordinal in EXPECTED_CLAUSES
        for mode in ("control", "prepared")
    }
    if set(by_key) != expected_keys:
        raise ValueError("NRIR48 worker coverage differs")
    for parity in parities:
        ordinal = parity.get("original_clause_index")
        semantic = {key: item for key, item in parity.items() if key != "parity_hash"}
        if (
            ordinal not in EXPECTED_CLAUSES
            or parity.get("repeat_index") != repeat_index
            or parity.get("control_row_hash")
            != by_key[(ordinal, "control")]["row_hash"]
            or parity.get("prepared_row_hash")
            != by_key[(ordinal, "prepared")]["row_hash"]
            or parity.get("exact") is not True
            or parity.get("parity_hash") != _canonical_hash(semantic)
        ):
            raise ValueError("NRIR48 parity differs")
    semantic_worker = {key: item for key, item in value.items() if key != "worker_hash"}
    if value.get("worker_hash") != _canonical_hash(semantic_worker):
        raise ValueError("NRIR48 worker hash differs")


def _median_mad(values: list[int]) -> tuple[int, int]:
    median = int(statistics.median(values))
    mad = int(statistics.median(abs(value - median) for value in values))
    return median, mad


def _build_formal(
    workers: Sequence[Mapping[str, Any]], *, workload: Mapping[str, Any]
) -> dict[str, Any]:
    perturbation_metrics: list[dict[str, Any]] = []
    category_metrics: list[dict[str, Any]] = []
    child_subcategory_metrics: list[dict[str, Any]] = []
    winners: dict[int, list[str]] = {ordinal: [] for ordinal in EXPECTED_CLAUSES}
    for ordinal in EXPECTED_CLAUSES:
        controls = [
            row["queue_elapsed_ns"]
            for worker in workers
            for row in worker["rows"]
            if row["original_clause_index"] == ordinal and row["mode"] == "control"
        ]
        profiles = [
            row["queue_elapsed_ns"]
            for worker in workers
            for row in worker["rows"]
            if row["original_clause_index"] == ordinal and row["mode"] == "prepared"
        ]
        control_median, control_mad = _median_mad(controls)
        profile_median, profile_mad = _median_mad(profiles)
        perturbation_metrics.append(
            {
                "original_clause_index": ordinal,
                "control_ns": controls,
                "profile_ns": profiles,
                "control_median_ns": control_median,
                "profile_median_ns": profile_median,
                "control_mad_ns": control_mad,
                "profile_mad_ns": profile_mad,
                "profile_to_control_median_ratio": profile_median / control_median,
            }
        )
        profile_rows = [
            row
            for worker in workers
            for row in worker["rows"]
            if row["original_clause_index"] == ordinal and row["mode"] == "prepared"
        ]
        for row in profile_rows:
            top = row["attribution"]["top_categories"]
            winners[ordinal].append(max(top, key=top.__getitem__))
        for category in TOP_CATEGORY_KEYS:
            values = [
                row["attribution"]["top_categories"][category] for row in profile_rows
            ]
            shares = [
                value / row["queue_elapsed_ns"]
                for value, row in zip(values, profile_rows)
            ]
            median_ns, mad_ns = _median_mad(values)
            category_metrics.append(
                {
                    "original_clause_index": ordinal,
                    "category": category,
                    "values_ns": values,
                    "shares": shares,
                    "median_ns": median_ns,
                    "mad_ns": mad_ns,
                    "median_share": statistics.median(shares),
                    "share_range": max(shares) - min(shares),
                }
            )
    perturbation_passed = all(
        item["profile_to_control_median_ratio"] <= MAXIMUM_PROFILE_MEDIAN_RATIO
        for item in perturbation_metrics
    )
    dominance_candidates = [
        category
        for category in TOP_CATEGORY_KEYS
        if all(
            winners[ordinal] == [category] * REPEAT_COUNT
            for ordinal in EXPECTED_CLAUSES
        )
        and all(
            item["median_share"] >= MINIMUM_DOMINANT_MEDIAN_SHARE
            and item["share_range"] <= MAXIMUM_DOMINANT_SHARE_RANGE
            and item["median_ns"] > item["mad_ns"]
            for item in category_metrics
            if item["category"] == category
        )
    ]
    dominant = dominance_candidates[0] if len(dominance_candidates) == 1 else None
    child_subcategory = None
    child_subcategory_passed = True
    if dominant == "child_refinement_execute_ns":
        child_keys = tuple(
            next(
                row["attribution"]["child_execute_categories"]
                for worker in workers
                for row in worker["rows"]
                if row["mode"] == "prepared"
            )
        )
        eligible = []
        for key in child_keys:
            for ordinal in EXPECTED_CLAUSES:
                rows = [
                    row
                    for worker in workers
                    for row in worker["rows"]
                    if row["mode"] == "prepared"
                    and row["original_clause_index"] == ordinal
                ]
                values = [
                    row["attribution"]["child_execute_categories"][key] for row in rows
                ]
                shares = [
                    value
                    / row["attribution"]["top_categories"][
                        "child_refinement_execute_ns"
                    ]
                    for value, row in zip(values, rows)
                ]
                median_ns, mad_ns = _median_mad(values)
                child_subcategory_metrics.append(
                    {
                        "original_clause_index": ordinal,
                        "category": key,
                        "values_ns": values,
                        "shares": shares,
                        "median_ns": median_ns,
                        "mad_ns": mad_ns,
                        "median_share_of_parent": statistics.median(shares),
                        "share_range": max(shares) - min(shares),
                    }
                )
            if all(
                statistics.median(
                    row["attribution"]["child_execute_categories"][key]
                    / row["attribution"]["top_categories"][
                        "child_refinement_execute_ns"
                    ]
                    for worker in workers
                    for row in worker["rows"]
                    if row["mode"] == "prepared"
                    and row["original_clause_index"] == ordinal
                )
                >= MINIMUM_CHILD_SUBCATEGORY_SHARE
                for ordinal in EXPECTED_CLAUSES
            ):
                eligible.append(key)
        child_subcategory = eligible[0] if len(eligible) == 1 else None
        child_subcategory_passed = child_subcategory is not None
    parity_passed = all(
        parity["exact"] for worker in workers for parity in worker["parities"]
    )
    closure_passed = all(
        row["attribution"]["closure_error_ns"] == 0
        for worker in workers
        for row in worker["rows"]
        if row["mode"] == "prepared"
    )
    dominance_passed = dominant is not None
    route_selected = (
        parity_passed
        and closure_passed
        and perturbation_passed
        and dominance_passed
        and child_subcategory_passed
    )
    decision = {
        "parity_passed": parity_passed,
        "closure_passed": closure_passed,
        "perturbation_passed": perturbation_passed,
        "dominance_passed": dominance_passed,
        "dominant_category": dominant,
        "child_subcategory_passed": child_subcategory_passed,
        "dominant_child_subcategory": child_subcategory,
        "route_selected": route_selected,
        "next_route": (
            f"preregister-nrir49:{child_subcategory or dominant}"
            if route_selected
            else "stop-and-refine-attribution"
        ),
    }
    formal: dict[str, Any] = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "source": {
            "base_revision": BASE_REVISION,
            "native_code_revision": _code_revision(),
            "source_nrir45_phase_b_hash": _source_nrir45_hash(),
            "workload": nrir45._public_workload(workload),
        },
        "contract": {
            "repeat_count": REPEAT_COUNT,
            "paired_orders": [list(item) for item in PAIRED_ORDERS],
            "clauses": list(EXPECTED_CLAUSES),
            "required_nodes": REQUIRED_NODES,
            "required_sibling_groups": REQUIRED_SIBLING_GROUPS,
            "maximum_profile_median_ratio": MAXIMUM_PROFILE_MEDIAN_RATIO,
            "maximum_closure_error_ratio": MAXIMUM_CLOSURE_ERROR_RATIO,
            "minimum_dominant_median_share": MINIMUM_DOMINANT_MEDIAN_SHARE,
            "maximum_dominant_share_range": MAXIMUM_DOMINANT_SHARE_RANGE,
            "minimum_child_subcategory_share": MINIMUM_CHILD_SUBCATEGORY_SHARE,
            "top_categories": list(TOP_CATEGORY_KEYS),
            "torch_threads": TORCH_THREADS,
        },
        "workers": [
            {
                "repeat_index": worker["repeat_index"],
                "worker_hash": worker["worker_hash"],
            }
            for worker in workers
        ],
        "perturbation_metrics": perturbation_metrics,
        "category_metrics": category_metrics,
        "child_subcategory_metrics": child_subcategory_metrics,
        "winners": {str(key): value for key, value in winners.items()},
        "decision": decision,
        "decision_hash": _canonical_hash(decision),
        "status": "validated-reduced" if route_selected else "validated-no-go",
        "performance_claimed": False,
    }
    formal["formal_hash"] = _canonical_hash(formal)
    return formal


def validate_formal(value: Mapping[str, Any]) -> None:
    contract = value.get("contract", {})
    decision = value.get("decision", {})
    semantic = {key: item for key, item in value.items() if key != "formal_hash"}
    if (
        value.get("schema_version") != FORMAL_SCHEMA_VERSION
        or value.get("source", {}).get("base_revision") != BASE_REVISION
        or value.get("source", {}).get("native_code_revision") != _code_revision()
        or value.get("source", {}).get("source_nrir45_phase_b_hash")
        != _source_nrir45_hash()
        or contract.get("repeat_count") != REPEAT_COUNT
        or contract.get("paired_orders") != [list(item) for item in PAIRED_ORDERS]
        or contract.get("clauses") != list(EXPECTED_CLAUSES)
        or contract.get("top_categories") != list(TOP_CATEGORY_KEYS)
        or contract.get("maximum_profile_median_ratio") != MAXIMUM_PROFILE_MEDIAN_RATIO
        or contract.get("maximum_closure_error_ratio") != MAXIMUM_CLOSURE_ERROR_RATIO
        or contract.get("minimum_dominant_median_share")
        != MINIMUM_DOMINANT_MEDIAN_SHARE
        or contract.get("maximum_dominant_share_range") != MAXIMUM_DOMINANT_SHARE_RANGE
        or contract.get("minimum_child_subcategory_share")
        != MINIMUM_CHILD_SUBCATEGORY_SHARE
        or len(value.get("workers", [])) != REPEAT_COUNT
        or value.get("decision_hash") != _canonical_hash(decision)
        or value.get("status")
        != (
            "validated-reduced" if decision.get("route_selected") else "validated-no-go"
        )
        or value.get("performance_claimed") is not False
        or value.get("formal_hash") != _canonical_hash(semantic)
    ):
        raise ValueError("NRIR48 formal envelope differs")


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


def _workload(benchmark_root: Path) -> Mapping[str, Any]:
    return nrir45._workload(benchmark_root)


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
            f"NRIR48 worker failed with exit {completed.returncode}: {log_path}"
        )


def _generate(args: argparse.Namespace) -> None:
    if args.torch_threads != TORCH_THREADS:
        raise ValueError("NRIR48 torch thread count differs")
    workload = _workload(args.benchmark_root)
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
                "perturbation_metrics": formal["perturbation_metrics"],
                "winners": formal["winners"],
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
        raise ValueError("NRIR48 manifest differs")
    workers = []
    for repeat_index in range(REPEAT_COUNT):
        worker = _load_json(artifact_dir / "shards" / f"repeat-{repeat_index}.json")
        validate_worker(worker, repeat_index=repeat_index)
        workers.append(worker)
    formal = _load_json(artifact_dir / "formal.json")
    validate_formal(formal)
    if manifest.get("formal_hash") != formal["formal_hash"]:
        raise ValueError("NRIR48 manifest/formal differs")
    rebuilt = _build_formal(workers, workload=_workload(args.benchmark_root))
    if rebuilt != formal:
        raise ValueError("NRIR48 formal replay differs")
    representative = next(
        row for worker in workers for row in worker["rows"] if row["mode"] == "prepared"
    )
    tampered = copy.deepcopy(representative["attribution"])
    tampered["top_categories"]["child_refinement_execute_ns"] += 1
    tampered["top_categories"]["queue_control_residual_ns"] -= 1
    try:
        _validate_attribution(tampered)
    except ValueError:
        tamper_rejected = True
    else:
        tamper_rejected = False
    if not tamper_rejected:
        raise ValueError("NRIR48 synchronized category tamper was accepted")
    print(
        _canonical_json(
            {
                "status": "replay-passed",
                "formal_hash": formal["formal_hash"],
                "profile_rows_replayed": REPEAT_COUNT * len(EXPECTED_CLAUSES),
                "synchronized_category_tamper_rejected": True,
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
        raise AssertionError("unreachable NRIR48 command")


if __name__ == "__main__":
    main()
