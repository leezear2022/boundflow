#!/usr/bin/env python3
"""Run one real FSG3 B0/B1/B2 same-solver timing control worker."""

# pylint: disable=wrong-import-position,protected-access,import-outside-toplevel
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=too-many-arguments,too-many-instance-attributes,import-error
# pylint: disable=missing-function-docstring,too-few-public-methods
# pylint: disable=duplicate-code,too-many-lines,too-many-boolean-expressions

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack, nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping, MutableMapping, Optional, Sequence, cast
import xml.etree.ElementTree as ET

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundflow.runtime.fsg3_same_solver_timing import (
    canonical_hash,
    FSG3Configuration,
    FSG3EnvironmentGate,
    FSG3ExecutionCounters,
    FSG3Mode,
    FSG3ProfileSpan,
    FSG3SemanticResult,
    FSG3TimingMetrics,
    FSG3TimingRun,
)
from scripts import run_rvir_v4_live_return_capture as candidate_runner
from scripts import run_rvir_v4_production_state_capture as capture_runner

WORKER_ENVELOPE_SCHEMA = "boundflow.fsg3-same-solver-worker-envelope/v1"
ALLOWED_GRAPHICS_PROCESSES = ("kwin_wayland",)
WORKER_PREFLIGHT_TEMPERATURE_LIMIT_C = 50
WORKER_PREFLIGHT_POLL_SECONDS = 5
WORKER_PREFLIGHT_TIMEOUT_SECONDS = 900
POST_PREPARE_ENVIRONMENT_WINDOW = False


@dataclass
class _PendingProfileSpan:
    scope: str
    name: str
    stack_layer: str
    solver_phase: str
    resource: str
    cache_state: str
    start_offset_ns: int
    end_offset_ns: int
    start_event: Any
    end_event: Any


class _ProfileRecorder:
    """Record sequential profile spans without synchronizing inside a span."""

    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.epoch_ns = time.perf_counter_ns()
        self.active: Optional[dict[str, Any]] = None
        self.pending: list[_PendingProfileSpan] = []

    def begin(
        self,
        *,
        scope: str,
        name: str,
        stack_layer: str,
        solver_phase: str,
        resource: str,
        cache_state: str,
    ) -> None:
        if self.active is not None:
            self.end()
        start_event: Any = None
        if resource != "host":
            start_event = self.torch.cuda.Event(enable_timing=True)
            start_event.record(self.torch.cuda.current_stream())
        self.active = {
            "scope": scope,
            "name": name,
            "stack_layer": stack_layer,
            "solver_phase": solver_phase,
            "resource": resource,
            "cache_state": cache_state,
            "start_offset_ns": time.perf_counter_ns() - self.epoch_ns,
            "start_event": start_event,
        }

    def end(self) -> None:
        if self.active is None:
            raise RuntimeError("FSG3 profile span is not active")
        end_offset_ns = time.perf_counter_ns() - self.epoch_ns
        end_event: Any = None
        if self.active["resource"] != "host":
            end_event = self.torch.cuda.Event(enable_timing=True)
            end_event.record(self.torch.cuda.current_stream())
        self.pending.append(
            _PendingProfileSpan(
                scope=str(self.active["scope"]),
                name=str(self.active["name"]),
                stack_layer=str(self.active["stack_layer"]),
                solver_phase=str(self.active["solver_phase"]),
                resource=str(self.active["resource"]),
                cache_state=str(self.active["cache_state"]),
                start_offset_ns=int(self.active["start_offset_ns"]),
                end_offset_ns=end_offset_ns,
                start_event=self.active["start_event"],
                end_event=end_event,
            )
        )
        self.active = None

    @contextmanager
    def span(self, **metadata: str) -> Iterator[None]:
        self.begin(**metadata)
        try:
            yield
        finally:
            self.end()

    def finalize(self) -> tuple[FSG3ProfileSpan, ...]:
        if self.active is not None:
            raise ValueError("FSG3 profile span remains active")
        rows: list[FSG3ProfileSpan] = []
        for item in self.pending:
            gpu_ns = (
                0
                if item.start_event is None
                else int(round(item.start_event.elapsed_time(item.end_event) * 1e6))
            )
            rows.append(
                FSG3ProfileSpan(
                    scope=item.scope,
                    name=item.name,
                    stack_layer=item.stack_layer,
                    solver_phase=item.solver_phase,
                    resource=item.resource,
                    cache_state=item.cache_state,
                    start_offset_ns=item.start_offset_ns,
                    end_offset_ns=item.end_offset_ns,
                    wall_ns=item.end_offset_ns - item.start_offset_ns,
                    gpu_ns=gpu_ns,
                )
            )
        return tuple(rows)


def _one_final_bound(value: object, *, label: str) -> tuple[str, Any]:
    if not isinstance(value, Mapping) or len(value) != 1:
        raise ValueError(f"FSG3 {label} inventory differs")
    name, tensor = next(iter(value.items()))
    if not isinstance(name, str):
        raise TypeError(f"FSG3 {label} name differs")
    return name, tensor


class _QueueObserver:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.events: list[dict[str, object]] = []

    @contextmanager
    def instrument(self, domain_list_class: Any) -> Iterator[None]:
        original = domain_list_class.add

        def wrapped(
            instance: Any,
            ret: MutableMapping[str, object],
            host: Mapping[str, object],
            *positional: object,
            **keyword: object,
        ) -> object:
            before = len(instance)
            lower_name, lower = _one_final_bound(ret.get("lower_bounds"), label="lower")
            upper_name, upper = _one_final_bound(ret.get("upper_bounds"), label="upper")
            depths = host.get("depths")
            thresholds = host.get("thresholds")
            history = host.get("history")
            target = (
                lower_name == upper_name
                and self.torch.is_tensor(lower)
                and self.torch.is_tensor(upper)
                and self.torch.is_tensor(depths)
                and self.torch.is_tensor(thresholds)
                and isinstance(history, list)
                and tuple(cast(Any, depths).shape) == (6,)
                and tuple(cast(Any, thresholds).shape) == (6, 1)
                and int(lower.shape[0]) == 6
            )
            result = original(instance, ret, host, *positional, **keyword)
            if target:
                after = len(instance)
                self.events.append(
                    {
                        "before": before,
                        "input": int(lower.shape[0]),
                        "accepted": after - before,
                        "after": after,
                        "lower": lower,
                        "upper": upper,
                        "depths": depths,
                        "history": history,
                        "thresholds": thresholds,
                    }
                )
            return result

        domain_list_class.add = wrapped
        try:
            yield
        finally:
            domain_list_class.add = original


class _ProviderObserver:
    def __init__(self, core_observer: "_CoreObserver") -> None:
        self.core_observer = core_observer
        self.compute_bounds_call_count = 0

    @contextmanager
    def instrument(self, bounded_module_class: Any) -> Iterator[None]:
        original = bounded_module_class.compute_bounds

        def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if self.core_observer.active:
                self.compute_bounds_call_count += 1
            return original(instance, *args, **kwargs)

        bounded_module_class.compute_bounds = wrapped
        try:
            yield
        finally:
            bounded_module_class.compute_bounds = original


class _CoreObserver:  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        *,
        configuration: FSG3Configuration,
        torch_module: Any,
        arguments_module: Any,
        profile_recorder: Optional[_ProfileRecorder] = None,
    ) -> None:
        self.configuration = configuration
        self.torch = torch_module
        self.arguments = arguments_module
        self.profile_recorder = profile_recorder
        self.core_count = 0
        self.provider_update_bounds_call_count = 0
        self.typed_validation_count = 0
        self.typed_snapshot_hash: Optional[str] = None
        self.active = False
        self.last_result: Any = None
        self.host_start_ns: Optional[int] = None
        self.host_end_ns: Optional[int] = None
        self.start_event: Any = None
        self.end_event: Any = None

    @contextmanager
    def instrument(self, stage_solve_module: Any) -> Iterator[None]:
        original = stage_solve_module.update_bounds_core

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if self.core_count != 0:
                raise RuntimeError("FSG3 requires exactly one update_bounds_core")
            net = kwargs.get("net", args[0] if args else None)
            pre_result = kwargs.get("pre_result", args[1] if len(args) > 1 else None)
            if net is None or pre_result is None:
                raise TypeError("FSG3 core call schema differs")
            stream = self.torch.cuda.current_stream()
            self.start_event = self.torch.cuda.Event(enable_timing=True)
            self.end_event = self.torch.cuda.Event(enable_timing=True)
            self.start_event.record(stream)
            self.host_start_ns = time.perf_counter_ns()
            self.active = True
            heuristic: Any = None
            original_compute: Any = None
            try:
                if self.configuration == FSG3Configuration.B1:
                    if self.profile_recorder is not None:
                        self.profile_recorder.begin(
                            scope="core",
                            name="typed_pre_state",
                            stack_layer="transport/runtime",
                            solver_phase="production_state_validation",
                            resource="host+cuda",
                            cache_state="process-hit",
                        )
                    policy = capture_runner._optimizer_policy(self.arguments, kwargs)
                    snapshot = capture_runner._build_core_pre_snapshot(
                        pre_result,
                        net=net,
                        core_id=0,
                        policy=policy,
                    )
                    self.typed_snapshot_hash = snapshot.stable_hash()
                    self.typed_validation_count += 1
                if self.configuration in {
                    FSG3Configuration.B0,
                    FSG3Configuration.B1,
                }:
                    heuristic = kwargs.get("branching_heuristic")
                    original_compute = getattr(
                        heuristic, "compute_branching_decisions", None
                    )
                    if not callable(original_compute):
                        raise TypeError("FSG3 provider branching is unavailable")

                    def counted_compute(*inner_args: Any, **inner_kwargs: Any) -> Any:
                        previous_profile = sys.getprofile()

                        def count_update(frame: Any, event: str, value: object) -> None:
                            if previous_profile is not None:
                                cast(Any, previous_profile)(frame, event, value)
                            if (
                                event == "call"
                                and frame.f_code.co_name == "update_bounds"
                                and frame.f_code.co_filename.replace(
                                    "\\", "/"
                                ).endswith("/beta_CROWN_solver.py")
                            ):
                                self.provider_update_bounds_call_count += 1

                        sys.setprofile(count_update)
                        try:
                            return original_compute(*inner_args, **inner_kwargs)
                        finally:
                            sys.setprofile(previous_profile)

                    heuristic.compute_branching_decisions = counted_compute
                    if self.profile_recorder is not None:
                        self.profile_recorder.begin(
                            scope="core",
                            name="provider_core",
                            stack_layer="solver/provider",
                            solver_phase="official_update_bounds_core",
                            resource="host+cuda",
                            cache_state="process-hit",
                        )
                result = original(*args, **kwargs)
                if self.profile_recorder is not None and self.configuration in {
                    FSG3Configuration.B0,
                    FSG3Configuration.B1,
                }:
                    self.profile_recorder.end()
            finally:
                self.host_end_ns = time.perf_counter_ns()
                self.end_event.record(stream)
                self.active = False
                if heuristic is not None:
                    heuristic.compute_branching_decisions = original_compute
            self.last_result = result
            self.core_count += 1
            return result

        stage_solve_module.update_bounds_core = wrapped
        try:
            yield
        finally:
            stage_solve_module.update_bounds_core = original

    def timings(self) -> tuple[int, int]:
        if (
            self.core_count != 1
            or self.host_start_ns is None
            or self.host_end_ns is None
            or self.start_event is None
            or self.end_event is None
        ):
            raise ValueError("FSG3 core timing is incomplete")
        wall = self.host_end_ns - self.host_start_ns
        gpu = int(round(self.start_event.elapsed_time(self.end_event) * 1e6))
        if wall <= 0 or gpu <= 0:
            raise ValueError("FSG3 core timing is non-positive")
        return wall, gpu


class _HostPhaseObserver:
    """Measure named host phases without changing their arguments or results."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._wall_ns: dict[str, int] = {}

    @contextmanager
    def instrument(self, bindings: Sequence[tuple[object, str, str]]) -> Iterator[None]:
        originals: list[tuple[object, str, Any]] = []
        names: set[str] = set()
        try:
            for owner, attribute, name in bindings:
                if name in names:
                    raise ValueError("FSG3 host phase name must be unique")
                names.add(name)
                original = getattr(owner, attribute)
                if not callable(original):
                    raise TypeError(f"FSG3 host phase is not callable: {name}")

                def wrapped(
                    *args: Any,
                    _original: Any = original,
                    _name: str = name,
                    **kwargs: Any,
                ) -> Any:
                    started_ns = time.perf_counter_ns()
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        elapsed_ns = time.perf_counter_ns() - started_ns
                        self._counts[_name] = self._counts.get(_name, 0) + 1
                        self._wall_ns[_name] = self._wall_ns.get(_name, 0) + elapsed_ns

                originals.append((owner, attribute, original))
                setattr(owner, attribute, wrapped)
            yield
        finally:
            for owner, attribute, original in reversed(originals):
                setattr(owner, attribute, original)

    def snapshot(self, expected_names: Sequence[str]) -> dict[str, dict[str, int]]:
        if set(expected_names) != set(self._counts) or any(
            self._counts.get(name) != 1 for name in expected_names
        ):
            raise ValueError("FSG3 host phase count differs")
        result = {
            name: {
                "call_count": self._counts[name],
                "wall_ns": self._wall_ns[name],
            }
            for name in expected_names
        }
        if any(row["wall_ns"] <= 0 for row in result.values()):
            raise ValueError("FSG3 host phase timing must be positive")
        return result


class _NestedPhaseObserver:
    """Record inclusive/exclusive wall time for nested verifier transactions."""

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []
        self._events: list[dict[str, int | str | None]] = []
        self._next_event_id = 0

    @contextmanager
    def instrument(self, bindings: Sequence[tuple[object, str, str]]) -> Iterator[None]:
        originals: list[tuple[object, str, Any]] = []
        names: set[str] = set()
        try:
            for owner, attribute, name in bindings:
                if name in names:
                    raise ValueError("FSG3 nested phase name must be unique")
                names.add(name)
                original = getattr(owner, attribute)
                if not callable(original):
                    raise TypeError(f"FSG3 nested phase is not callable: {name}")

                def wrapped(
                    *args: Any,
                    _original: Any = original,
                    _name: str = name,
                    **kwargs: Any,
                ) -> Any:
                    event_id = self._next_event_id
                    self._next_event_id += 1
                    parent_event_id = self._stack[-1][0] if self._stack else None
                    parent_name = self._stack[-1][1] if self._stack else None
                    self._stack.append((event_id, _name))
                    started_ns = time.perf_counter_ns()
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        wall_ns = time.perf_counter_ns() - started_ns
                        popped = self._stack.pop()
                        if popped != (event_id, _name):
                            raise RuntimeError("FSG3 nested phase stack differs")
                        self._events.append(
                            {
                                "event_id": event_id,
                                "name": _name,
                                "parent_event_id": parent_event_id,
                                "parent_name": parent_name,
                                "wall_ns": wall_ns,
                            }
                        )

                originals.append((owner, attribute, original))
                setattr(owner, attribute, wrapped)
            yield
        finally:
            for owner, attribute, original in reversed(originals):
                setattr(owner, attribute, original)

    def snapshot(
        self, *, root_name: str, required_names: Sequence[str]
    ) -> dict[str, object]:
        if self._stack:
            raise ValueError("FSG3 nested phase stack is not empty")

        def event_integer(row: Mapping[str, object], key: str) -> int:
            value = row.get(key)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("FSG3 nested phase integer differs")
            return value

        events = sorted(self._events, key=lambda row: event_integer(row, "event_id"))
        counts = {
            name: sum(row["name"] == name for row in events) for name in required_names
        }
        if counts.get(root_name) != 1 or any(value <= 0 for value in counts.values()):
            raise ValueError("FSG3 nested phase cardinality differs")
        child_ns: dict[int, int] = {}
        for row in events:
            parent = row["parent_event_id"]
            if parent is not None:
                parent_id = event_integer(row, "parent_event_id")
                child_ns[parent_id] = child_ns.get(parent_id, 0) + event_integer(
                    row, "wall_ns"
                )
        aggregates: dict[str, dict[str, int]] = {}
        normalized_events: list[dict[str, int | str | None]] = []
        for row in events:
            event_id = event_integer(row, "event_id")
            wall_ns = event_integer(row, "wall_ns")
            exclusive_ns = wall_ns - child_ns.get(event_id, 0)
            if wall_ns <= 0 or exclusive_ns < 0:
                raise ValueError("FSG3 nested phase closure differs")
            normalized = dict(row)
            normalized["exclusive_ns"] = exclusive_ns
            normalized_events.append(normalized)
            name = str(row["name"])
            aggregate = aggregates.setdefault(
                name, {"call_count": 0, "inclusive_ns": 0, "exclusive_ns": 0}
            )
            aggregate["call_count"] += 1
            aggregate["inclusive_ns"] += wall_ns
            aggregate["exclusive_ns"] += exclusive_ns
        return {
            "schema_version": "boundflow.fsg3-nested-host-phases/v1",
            "root_name": root_name,
            "events": normalized_events,
            "aggregates": dict(sorted(aggregates.items())),
        }


class _PostObserver:
    def __init__(self, profile_recorder: Optional[_ProfileRecorder] = None) -> None:
        self.last_result: Any = None
        self.count = 0
        self.profile_recorder = profile_recorder
        self.inner_start_ns: Optional[int] = None
        self.inner_end_ns: Optional[int] = None
        self.outer_start_ns: Optional[int] = None
        self.outer_end_ns: Optional[int] = None

    @contextmanager
    def instrument(
        self, stage_postprocess_module: Any, bab_bootstrap_module: Any
    ) -> Iterator[None]:
        original = stage_postprocess_module.update_bounds_post
        original_outer = bab_bootstrap_module.branch_and_bound_postprocess

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if self.inner_start_ns is not None:
                raise RuntimeError("FSG3 requires exactly one update_bounds_post")
            self.inner_start_ns = time.perf_counter_ns()
            try:
                result = original(*args, **kwargs)
            finally:
                self.inner_end_ns = time.perf_counter_ns()
            self.last_result = result
            self.count += 1
            return result

        def wrapped_outer(*args: Any, **kwargs: Any) -> Any:
            if self.outer_start_ns is not None:
                raise RuntimeError("FSG3 requires exactly one postprocess stage")
            self.outer_start_ns = time.perf_counter_ns()
            try:
                if self.profile_recorder is None:
                    return original_outer(*args, **kwargs)
                with self.profile_recorder.span(
                    scope="post",
                    name="official_post_queue",
                    stack_layer="solver/runtime",
                    solver_phase="official_post_and_queue",
                    resource="host+cuda",
                    cache_state="process-hit",
                ):
                    return original_outer(*args, **kwargs)
            finally:
                self.outer_end_ns = time.perf_counter_ns()

        stage_postprocess_module.update_bounds_post = wrapped
        bab_bootstrap_module.branch_and_bound_postprocess = wrapped_outer
        try:
            yield
        finally:
            stage_postprocess_module.update_bounds_post = original
            bab_bootstrap_module.branch_and_bound_postprocess = original_outer

    def timings(self) -> tuple[int, int]:
        if (
            self.count != 1
            or self.inner_start_ns is None
            or self.inner_end_ns is None
            or self.outer_start_ns is None
            or self.outer_end_ns is None
        ):
            raise ValueError("FSG3 post timing is incomplete")
        inner_ns = self.inner_end_ns - self.inner_start_ns
        outer_ns = self.outer_end_ns - self.outer_start_ns
        if inner_ns <= 0 or outer_ns <= 0 or inner_ns > outer_ns:
            raise ValueError("FSG3 post timing closure differs")
        return inner_ns, outer_ns


def _query_phase_timing(
    *,
    query_wall_ns: int,
    solver_init_ns: int,
    constraint_prepare_ns: int,
    verify_started_ns: int,
    verify_ended_ns: int,
    core_started_ns: int,
    core_ended_ns: int,
    final_sync_ns: int,
    update_bounds_post_ns: int,
    official_post_queue_ns: int,
) -> dict[str, int]:
    """Build a fail-closed, exactly closed host/query phase ledger."""

    if not verify_started_ns <= core_started_ns <= core_ended_ns <= verify_ended_ns:
        raise ValueError("FSG3 query phase ordering differs")
    values = {
        "query_wall_ns": query_wall_ns,
        "solver_init_ns": solver_init_ns,
        "constraint_prepare_ns": constraint_prepare_ns,
        "verify_wall_ns": verify_ended_ns - verify_started_ns,
        "pre_core_ns": core_started_ns - verify_started_ns,
        "core_wall_ns": core_ended_ns - core_started_ns,
        "post_core_ns": verify_ended_ns - core_ended_ns,
        "final_sync_ns": final_sync_ns,
        "update_bounds_post_ns": update_bounds_post_ns,
        "official_post_queue_ns": official_post_queue_ns,
    }
    nonnegative_names = {"solver_init_ns", "constraint_prepare_ns"}
    if any(
        value < 0 if name in nonnegative_names else value <= 0
        for name, value in values.items()
    ):
        raise ValueError("FSG3 query phase timing must be positive")
    if update_bounds_post_ns > official_post_queue_ns:
        raise ValueError("FSG3 query post phase nesting differs")
    verify_closure_ns = (
        values["verify_wall_ns"]
        - values["pre_core_ns"]
        - values["core_wall_ns"]
        - values["post_core_ns"]
    )
    query_unattributed_ns = (
        query_wall_ns
        - solver_init_ns
        - constraint_prepare_ns
        - values["verify_wall_ns"]
        - final_sync_ns
    )
    if verify_closure_ns != 0 or query_unattributed_ns < 0:
        raise ValueError("FSG3 query phase closure differs")
    values["verify_closure_ns"] = verify_closure_ns
    values["query_unattributed_ns"] = query_unattributed_ns
    values["post_queue_residual_ns"] = official_post_queue_ns - update_bounds_post_ns
    return values


def _visited_domains(result: Any) -> tuple[int, ...]:
    stats = getattr(result, "stats", None)
    if not isinstance(stats, dict) or not isinstance(stats.get("bab"), list):
        return ()
    return tuple(
        int(row[2])
        for row in stats["bab"]
        if isinstance(row, (tuple, list)) and len(row) >= 3
    )


def _tensor_values(value: Any) -> tuple[tuple[int, ...], tuple[float, ...]]:
    tensor = value.detach().cpu().contiguous()
    if tensor.is_floating_point() and not bool(tensor.isfinite().all()):
        raise ValueError("FSG3 finite tensor projection differs")
    return tuple(int(item) for item in tensor.shape), tuple(
        float(item) for item in tensor.reshape(-1).tolist()
    )


def _upper_values(
    value: Any,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[bool, ...]]:
    tensor = value.detach().cpu().contiguous()
    if bool(tensor.isnan().any()) or bool(tensor.isneginf().any()):
        raise ValueError("FSG3 upper contains NaN or negative infinity")
    mask = tensor.isposinf()
    if not bool((tensor.isfinite() | mask).all()):
        raise ValueError("FSG3 upper sentinel differs")
    encoded = tensor.masked_fill(mask, 0.0)
    return (
        tuple(int(item) for item in tensor.shape),
        tuple(float(item) for item in encoded.reshape(-1).tolist()),
        tuple(bool(item) for item in mask.reshape(-1).tolist()),
    )


def _sequence_ints(value: object, torch_module: Any) -> tuple[int, ...]:
    if torch_module.is_tensor(value):
        tensor = cast(Any, value)
        return tuple(int(item) for item in tensor.detach().cpu().reshape(-1).tolist())
    if isinstance(value, Sequence):
        return tuple(int(item) for item in value)
    raise TypeError("FSG3 integer sequence differs")


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"FSG3 {label} must be an integer")
    return value


def _semantic_result(
    *,
    solver_result: Any,
    core_result: Any,
    queue_event: Mapping[str, object],
    torch_module: Any,
) -> FSG3SemanticResult:
    lower_shape, lower_values = _tensor_values(queue_event["lower"])
    upper_shape, upper_values, upper_mask = _upper_values(queue_event["upper"])
    decision = core_result.branching_decision
    input_count = _integer(queue_event["input"], "queue input")
    accepted = _integer(queue_event["accepted"], "queue accepted")
    result = FSG3SemanticResult(
        status=str(solver_result.status),
        success=bool(solver_result.success),
        visited_domains=_visited_domains(solver_result),
        queue_before=_integer(queue_event["before"], "queue before"),
        queue_input=input_count,
        queue_accepted=accepted,
        queue_pruned=input_count - accepted,
        queue_after=_integer(queue_event["after"], "queue after"),
        depths=_sequence_ints(queue_event["depths"], torch_module),
        history_count=len(cast(list[object], queue_event["history"])),
        lower_shape=lower_shape,
        lower_values=lower_values,
        upper_shape=upper_shape,
        upper_values=upper_values,
        upper_positive_infinity_mask=upper_mask,
        final_decision=tuple(
            (int(layer), int(index)) for layer, index in decision.branching_decision
        ),
        split_depth=int(decision.split_depth),
        batch_size=int(decision.batch_size),
        n_verified=int(core_result.n_verified),
        n_splits=int(core_result.n_splits),
    )
    result.validate()
    return result


def _run_command(*command: str) -> str:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout


def _nvidia_snapshot() -> dict[str, object]:
    xml = ET.fromstring(_run_command("nvidia-smi", "-q", "-x"))
    gpu = xml.find("gpu")
    if gpu is None:
        raise ValueError("FSG3 NVIDIA GPU XML is unavailable")

    def text_at(path: str) -> str:
        value = gpu.findtext(path)
        if value is None:
            raise ValueError(f"FSG3 NVIDIA field is unavailable: {path}")
        return value

    software_thermal_counter = text_at(
        "clocks_event_reasons_counters/clocks_event_reasons_counters_sw_therm_slowdown"
    )
    software_power_counter = text_at(
        "clocks_event_reasons_counters/clocks_event_reasons_counters_sw_power_cap"
    )
    hardware_thermal_counter = text_at(
        "clocks_event_reasons_counters/clocks_event_reasons_counters_hw_therm_slowdown"
    )
    return {
        "driver_version": xml.findtext("driver_version", default="unavailable"),
        "uuid": text_at("uuid"),
        "name": text_at("product_name"),
        "total_memory": text_at("fb_memory_usage/total"),
        "performance_state": text_at("performance_state"),
        "temperature": text_at("temperature/gpu_temp"),
        # NVIDIA documents T.Limit as a margin, rather than an absolute
        # temperature.  Preserve both raw values with unambiguous names.
        "temperature_tlimit_margin": text_at("temperature/gpu_temp_tlimit"),
        "target_temperature": text_at("temperature/gpu_target_temperature"),
        "power_draw": text_at("gpu_power_readings/instant_power_draw"),
        "sm_clock": text_at("clocks/sm_clock"),
        "memory_clock": text_at("clocks/mem_clock"),
        "sw_thermal_slowdown": text_at(
            "clocks_event_reasons/clocks_event_reason_sw_thermal_slowdown"
        ),
        "sw_power_cap": text_at(
            "clocks_event_reasons/clocks_event_reason_sw_power_cap"
        ),
        "hw_thermal_slowdown": text_at(
            "clocks_event_reasons/clocks_event_reason_hw_thermal_slowdown"
        ),
        "sw_thermal_slowdown_counter_us": int(software_thermal_counter.split()[0]),
        "sw_power_cap_counter_us": int(software_power_counter.split()[0]),
        "hw_thermal_slowdown_counter_us": int(hardware_thermal_counter.split()[0]),
    }


def _compute_processes() -> list[dict[str, object]]:
    output = _run_command(
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=2)]
        if len(fields) != 3:
            raise ValueError("FSG3 NVIDIA process row differs")
        rows.append(
            {
                "pid": int(fields[0]),
                "name": fields[1],
                "used_memory_mib": int(fields[2]),
            }
        )
    return rows


def _ac_powered() -> bool:
    path = Path("/sys/class/power_supply/ACAD/online")
    return path.is_file() and path.read_text(encoding="utf-8").strip() == "1"


def _snapshot_temperature_celsius(snapshot: Mapping[str, object]) -> int:
    try:
        return int(str(snapshot["temperature"]).split()[0])
    except (IndexError, ValueError) as error:
        raise ValueError("FSG3 worker preflight temperature differs") from error


def _reason_active(snapshot: Mapping[str, object], name: str) -> bool:
    value = snapshot.get(name)
    if not isinstance(value, str) or value not in ("Active", "Not Active"):
        raise ValueError(f"FSG3 NVIDIA reason differs: {name}")
    return value == "Active"


def _snapshot_software_counters_coupled(snapshot: Mapping[str, object]) -> bool:
    """Recognize only the exact driver-level SW power/thermal alias."""

    return (
        _reason_active(snapshot, "sw_thermal_slowdown")
        and _reason_active(snapshot, "sw_power_cap")
        and _integer(
            snapshot["sw_thermal_slowdown_counter_us"],
            "software thermal counter",
        )
        == _integer(snapshot["sw_power_cap_counter_us"], "software power counter")
    )


def _snapshot_independent_thermal_active(snapshot: Mapping[str, object]) -> bool:
    """Return thermal activity after excluding an exact SW power-cap alias."""

    return _reason_active(snapshot, "hw_thermal_slowdown") or (
        _reason_active(snapshot, "sw_thermal_slowdown")
        and not _snapshot_software_counters_coupled(snapshot)
    )


def _worker_external_processes(
    processes: Sequence[Mapping[str, object]],
    *,
    worker_pid: int | None = None,
) -> list[str]:
    effective_worker_pid = os.getpid() if worker_pid is None else worker_pid
    external: list[str] = []
    for row in processes:
        pid = _integer(row["pid"], "process PID")
        name = str(row["name"])
        memory = _integer(row["used_memory_mib"], "process memory")
        if pid == effective_worker_pid:
            continue
        if Path(name).name in ALLOWED_GRAPHICS_PROCESSES and memory < 64:
            continue
        external.append(f"{pid}:{name}:{memory}MiB")
    return external


def _validate_worker_preflight(value: Mapping[str, Any]) -> None:
    expected = {
        "worker_pid",
        "temperature_limit_celsius",
        "poll_seconds",
        "timeout_seconds",
        "sample_count",
        "wait_ns",
        "samples",
        "admitted",
    }
    if (
        set(value) != expected
        or not isinstance(value["worker_pid"], int)
        or isinstance(value["worker_pid"], bool)
        or int(value["worker_pid"]) <= 0
        or value["temperature_limit_celsius"] != WORKER_PREFLIGHT_TEMPERATURE_LIMIT_C
        or value["poll_seconds"] != WORKER_PREFLIGHT_POLL_SECONDS
        or value["timeout_seconds"] != WORKER_PREFLIGHT_TIMEOUT_SECONDS
        or value["admitted"] is not True
        or not isinstance(value["samples"], list)
        or value["sample_count"] != len(value["samples"])
        or not value["samples"]
        or int(value["wait_ns"]) < 0
    ):
        raise ValueError("FSG3 worker preflight payload differs")
    last = value["samples"][-1]
    if not isinstance(last, Mapping):
        raise TypeError("FSG3 worker preflight sample differs")
    processes = last.get("compute_processes")
    snapshot = last.get("gpu_snapshot")
    if not isinstance(processes, list):
        raise TypeError("FSG3 worker preflight process list differs")
    if not isinstance(snapshot, Mapping):
        raise TypeError("FSG3 worker preflight GPU snapshot differs")
    independent_thermal_active = _snapshot_independent_thermal_active(snapshot)
    if last.get("independent_thermal_active") is not independent_thermal_active:
        raise ValueError("FSG3 worker preflight thermal projection differs")
    if (
        int(last.get("temperature_celsius", -1)) > WORKER_PREFLIGHT_TEMPERATURE_LIMIT_C
        or independent_thermal_active
        or last.get("ac_powered") is not True
        or _worker_external_processes(
            cast(Sequence[Mapping[str, object]], processes),
            worker_pid=int(value["worker_pid"]),
        )
    ):
        raise ValueError("FSG3 worker preflight admission differs")


def _wait_for_worker_environment() -> dict[str, object]:
    started_ns = time.monotonic_ns()
    samples: list[dict[str, object]] = []
    while True:
        snapshot = _nvidia_snapshot()
        processes = _compute_processes()
        external = _worker_external_processes(processes)
        if external:
            raise RuntimeError(
                "FSG3 worker preflight found external CUDA compute processes: "
                + ", ".join(external)
            )
        independent_thermal_active = _snapshot_independent_thermal_active(snapshot)
        temperature = _snapshot_temperature_celsius(snapshot)
        sample = {
            "elapsed_ns": time.monotonic_ns() - started_ns,
            "temperature_celsius": temperature,
            "independent_thermal_active": independent_thermal_active,
            "gpu_snapshot": snapshot,
            "compute_processes": processes,
            "ac_powered": _ac_powered(),
        }
        samples.append(sample)
        ready = (
            temperature <= WORKER_PREFLIGHT_TEMPERATURE_LIMIT_C
            and not independent_thermal_active
            and sample["ac_powered"] is True
        )
        if ready:
            result: dict[str, object] = {
                "worker_pid": os.getpid(),
                "temperature_limit_celsius": WORKER_PREFLIGHT_TEMPERATURE_LIMIT_C,
                "poll_seconds": WORKER_PREFLIGHT_POLL_SECONDS,
                "timeout_seconds": WORKER_PREFLIGHT_TIMEOUT_SECONDS,
                "sample_count": len(samples),
                "wait_ns": time.monotonic_ns() - started_ns,
                "samples": samples,
                "admitted": True,
            }
            _validate_worker_preflight(cast(Mapping[str, Any], result))
            return result
        if (
            time.monotonic_ns() - started_ns
            > WORKER_PREFLIGHT_TIMEOUT_SECONDS * 1_000_000_000
        ):
            raise TimeoutError("FSG3 worker preflight did not reach cool idle state")
        print(
            json.dumps(
                {
                    "worker_preflight": "waiting",
                    "temperature_celsius": temperature,
                    "independent_thermal_active": independent_thermal_active,
                    "sample_count": len(samples),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        time.sleep(WORKER_PREFLIGHT_POLL_SECONDS)


def _environment_gate(
    before: Mapping[str, object],
    after: Mapping[str, object],
    before_processes: Sequence[Mapping[str, object]],
    after_processes: Sequence[Mapping[str, object]],
    runtime_identity: str,
    *,
    worker_pid: int | None = None,
) -> FSG3EnvironmentGate:
    external: list[str] = []
    by_identity = {
        (
            _integer(row["pid"], "process PID"),
            str(row["name"]),
            _integer(row["used_memory_mib"], "process memory"),
        )
        for row in (*before_processes, *after_processes)
        if _integer(row["pid"], "process PID")
        != (os.getpid() if worker_pid is None else worker_pid)
    }
    for pid, name, memory in sorted(by_identity):
        basename = Path(name).name
        identity = f"{pid}:{name}:{memory}MiB"
        if basename in ALLOWED_GRAPHICS_PROCESSES and memory < 64:
            continue
        external.append(identity)
    (
        software_thermal_signal,
        software_power_cap_signal,
        software_counters_coupled,
        hardware_thermal_slowdown,
    ) = _environment_counter_projection(before, after)
    return FSG3EnvironmentGate(
        gpu_uuid=str(before["uuid"]),
        gpu_name=str(before["name"]),
        runtime_identity=runtime_identity,
        external_compute_processes=tuple(external),
        software_thermal_signal=software_thermal_signal,
        software_power_cap_signal=software_power_cap_signal,
        software_thermal_power_counters_coupled=software_counters_coupled,
        hardware_thermal_slowdown=hardware_thermal_slowdown,
        worker_overlap=bool(external),
        device_identity_stable=(
            before["uuid"] == after["uuid"] and before["name"] == after["name"]
        ),
        ac_powered=_ac_powered(),
    )


def _environment_counter_projection(
    before: Mapping[str, object], after: Mapping[str, object]
) -> tuple[bool, bool, bool, bool]:
    """Project interval-local NVIDIA power and thermal counter evidence."""

    sw_thermal_before = _integer(
        before["sw_thermal_slowdown_counter_us"], "software thermal counter"
    )
    sw_thermal_after = _integer(
        after["sw_thermal_slowdown_counter_us"], "software thermal counter"
    )
    sw_power_before = _integer(
        before["sw_power_cap_counter_us"], "software power counter"
    )
    sw_power_after = _integer(
        after["sw_power_cap_counter_us"], "software power counter"
    )
    hw_thermal_before = _integer(
        before["hw_thermal_slowdown_counter_us"], "hardware thermal counter"
    )
    hw_thermal_after = _integer(
        after["hw_thermal_slowdown_counter_us"], "hardware thermal counter"
    )
    if (
        sw_thermal_after < sw_thermal_before
        or sw_power_after < sw_power_before
        or hw_thermal_after < hw_thermal_before
    ):
        raise ValueError("FSG3 NVIDIA event counter decreased")
    software_thermal_signal = (
        _reason_active(before, "sw_thermal_slowdown")
        or _reason_active(after, "sw_thermal_slowdown")
        or sw_thermal_after > sw_thermal_before
    )
    software_power_cap_signal = (
        _reason_active(before, "sw_power_cap")
        or _reason_active(after, "sw_power_cap")
        or sw_power_after > sw_power_before
    )
    software_counters_coupled = (
        software_thermal_signal
        and software_power_cap_signal
        and before["sw_thermal_slowdown"] == before["sw_power_cap"]
        and after["sw_thermal_slowdown"] == after["sw_power_cap"]
        and sw_thermal_after - sw_thermal_before == sw_power_after - sw_power_before
    )
    hardware_thermal_slowdown = (
        _reason_active(before, "hw_thermal_slowdown")
        or _reason_active(after, "hw_thermal_slowdown")
        or hw_thermal_after > hw_thermal_before
    )
    return (
        software_thermal_signal,
        software_power_cap_signal,
        software_counters_coupled,
        hardware_thermal_slowdown,
    )


def _protocol_identity(_configuration: FSG3Configuration) -> str:
    return canonical_hash(
        {
            "device": "cuda",
            "seed": 100,
            "timeout_seconds": 60,
            "max_iterations": 1,
            "batch_size": 64,
            "auto_enlarge_batch_size": False,
            "alpha_steps": 5,
            "beta_steps": 10,
            "attack": "skip",
            "property_cache": "cold_isolated_copy",
            "worker_preflight_temperature_limit_celsius": (
                WORKER_PREFLIGHT_TEMPERATURE_LIMIT_C
            ),
            "thermal_admission_policy": (
                "reject-independent-thermal-allow-exact-sw-power-coupled-alias"
            ),
        }
    )


def _source_identity(args: argparse.Namespace) -> str:
    return canonical_hash(
        {
            "boundflow_commit": capture_runner._git_value(
                REPOSITORY_ROOT, "rev-parse", "HEAD"
            ),
            "abcrown_commit": capture_runner._git_value(
                args.abcrown_root, "rev-parse", "HEAD"
            ),
            "auto_lirpa_commit": capture_runner._git_value(
                args.abcrown_root / "auto_LiRPA", "rev-parse", "HEAD"
            ),
            "vnncomp_commit": capture_runner._git_value(
                args.benchmark_root, "rev-parse", "HEAD"
            ),
            "model_sha256": capture_runner.file_sha256(args.model),
            "property_sha256": capture_runner.file_sha256(args.property),
        }
    )


def _worker(args: argparse.Namespace) -> None:  # pylint: disable=too-many-locals
    sys.path.insert(0, str(args.abcrown_root / "complete_verifier"))
    sys.path.insert(0, str(args.abcrown_root))
    import torch

    from abcrown import (  # type: ignore[import-not-found]
        ABCrownSolver,
        ConfigBuilder,
        IOConstraints,
    )
    import api as abcrown_api  # type: ignore[import-not-found]
    import arguments  # type: ignore[import-not-found]
    import complete_verifier_func  # type: ignore[import-not-found]
    import incomplete_verifier_func  # type: ignore[import-not-found]
    from activation_split import (  # type: ignore[import-not-found]
        bab_bootstrap,
        stage_postprocess,
        stage_solve,
    )
    from auto_LiRPA import BoundedModule  # type: ignore[import-untyped,import-not-found]
    from branching_domains import BatchedDomainList  # type: ignore[import-not-found]

    capture_runner._validate_inputs(
        args.benchmark_root, args.abcrown_root, Path(sys.executable)
    )
    if not torch.cuda.is_available():
        raise RuntimeError("FSG3 worker requires CUDA")
    configuration = FSG3Configuration(args.configuration)
    mode = FSG3Mode(args.mode)
    prepare_static_request = bool(getattr(args, "prepare_static_request", False))
    attribute_root_incomplete = bool(getattr(args, "attribute_root_incomplete", False))
    prepare_root_optimizer_warmup = bool(
        getattr(args, "prepare_root_optimizer_warmup", False)
    )
    if prepare_root_optimizer_warmup and not prepare_static_request:
        raise ValueError("FSG3 root optimizer warmup requires a prepared request")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    worker_preflight = _wait_for_worker_environment()
    torch.cuda.reset_peak_memory_stats()
    environment_before = _nvidia_snapshot()
    processes_before = _compute_processes()
    profile_recorder = _ProfileRecorder(torch) if mode == FSG3Mode.PROFILE else None
    cold_started_ns = time.perf_counter_ns()
    program: Any = None
    module: Any = None
    compile_ns = 0
    if configuration == FSG3Configuration.B2:
        compile_started_ns = time.perf_counter_ns()
        from boundflow.frontends.onnx.frontend import import_onnx
        from boundflow.planner import plan_interval_ibp_v0

        compile_scope = (
            nullcontext()
            if profile_recorder is None
            else profile_recorder.span(
                scope="compile",
                name="compile",
                stack_layer="frontend/planner",
                solver_phase="onnx_import_and_reference_plan",
                resource="host",
                cache_state="cold",
            )
        )
        with compile_scope:
            program = import_onnx(str(args.model), do_shape_infer=True, normalize=True)
            module = plan_interval_ibp_v0(program)
        compile_ns = time.perf_counter_ns() - compile_started_ns

    core = _CoreObserver(
        configuration=configuration,
        torch_module=torch,
        arguments_module=arguments,
        profile_recorder=profile_recorder,
    )
    provider = _ProviderObserver(core)
    post = _PostObserver(profile_recorder)
    queue = _QueueObserver(torch)
    host_phases = _HostPhaseObserver()
    static_query_bindings = (
        ()
        if prepare_static_request
        else (
            (ABCrownSolver, "_prepare_model", "prepare_model"),
            (ABCrownSolver, "_prepare_runtime_spec", "prepare_runtime_spec"),
            (ABCrownSolver, "_build_vnnlib_handler", "build_vnnlib_handler"),
        )
    )
    host_phase_bindings = (
        (ABCrownSolver, "_prepare_environment", "prepare_environment"),
        *static_query_bindings,
        (abcrown_api, "incomplete_verifier_core", "incomplete_verifier"),
        (abcrown_api, "complete_verifier_core", "complete_verifier"),
        (complete_verifier_func, "general_bab", "general_bab"),
        (
            bab_bootstrap,
            "compute_first_iteration_decision",
            "first_iteration_decision",
        ),
        (bab_bootstrap, "branch_and_bound_preprocess", "bab_preprocess"),
        (bab_bootstrap, "branch_and_bound_solve", "bab_solve"),
        (bab_bootstrap, "branch_and_bound_postprocess", "bab_postprocess"),
    )
    host_phase_names = tuple(binding[2] for binding in host_phase_bindings)
    root_phases = _NestedPhaseObserver()
    root_phase_bindings: tuple[tuple[object, str, str], ...] = (
        (abcrown_api, "incomplete_verifier_core", "root_incomplete"),
        (incomplete_verifier_func.SpecHandler, "__init__", "spec_handler_init"),
        (incomplete_verifier_func.LiRPANet, "__init__", "lirpa_net_init"),
        (BoundedModule, "forward", "bounded_nominal_forward"),
        (incomplete_verifier_func.LiRPANet, "build", "lirpa_build"),
        (BoundedModule, "init_alpha", "init_alpha"),
        (BoundedModule, "compute_bounds", "compute_bounds"),
        (incomplete_verifier_func.SpecHandler, "post_process", "post_process"),
    )
    if attribute_root_incomplete:
        from auto_LiRPA import optimized_bounds  # type: ignore
        from auto_LiRPA.operators.relu import BoundRelu  # type: ignore

        root_phase_bindings += (
            (BoundedModule, "_get_optimized_bounds", "optimized_bounds_transaction"),
            (torch.autograd, "backward", "autograd_backward"),
            (torch.optim.Adam, "step", "adam_step"),
            (torch.optim.Optimizer, "zero_grad", "optimizer_zero_grad"),
            (
                torch.optim.lr_scheduler.ExponentialLR,
                "step",
                "scheduler_step",
            ),
            (BoundedModule, "_clear_and_set_new", "clear_intermediates"),
            (optimized_bounds, "_update_best_ret", "update_best_ret"),
            (
                optimized_bounds,
                "_update_optimizable_activations",
                "update_best_alpha",
            ),
            (BoundRelu, "clip_alpha", "clip_alpha"),
        )
    root_phase_names = tuple(binding[2] for binding in root_phase_bindings)
    executor: Any = None
    if configuration == FSG3Configuration.B2:
        executor = candidate_runner._LiveExecutor(
            model=args.model,
            torch_module=torch,
            arguments_module=arguments,
            precompiled_program=program,
            precompiled_module=module,
            capture_payloads=False,
            profile_recorder=profile_recorder,
        )
    if POST_PREPARE_ENVIRONMENT_WINDOW:
        # Candidate-specific AOT compilation and persistent allocation are
        # cold/setup costs, not query exposure.  Re-enter the same cool-idle
        # gate and start NVIDIA counters only after preparation completes.
        torch.cuda.synchronize()
        worker_preflight = _wait_for_worker_environment()
        torch.cuda.reset_peak_memory_stats()
        environment_before = _nvidia_snapshot()
        processes_before = _compute_processes()

    with tempfile.TemporaryDirectory(prefix="boundflow-fsg3-property-") as raw:
        isolated_property = Path(raw) / args.property.name
        shutil.copy2(args.property, isolated_property)
        config = (
            ConfigBuilder.from_defaults()
            .set("general/device", "cuda")
            .set("general/seed", 100)
            .set("general/reset_seed_after_precompile", True)
            .set("general/complete_verifier", "bab")
            .set("attack/pgd_order", "skip")
            .set("bab/timeout", 60)
            .set("bab/max_iterations", 1)
            .set("solver/batch_size", 64)
            .set("solver/auto_enlarge_batch_size", False)
            .set("solver/alpha-crown/iteration", 5)
            .set("solver/beta-crown/iteration", 10)
        )
        prepared_request: Any = None
        prepared_request_receipt: Mapping[str, object] | None = None
        prepared_root_warmup_receipt: Mapping[str, object] | None = None
        static_constraint_ns = 0
        static_solver_ns = 0
        static_total_prepare_ns = 0
        solver: Any = None
        constraints: Any = None
        if prepare_static_request:
            from boundflow.runtime.prepared_verification_request import (
                prepare_verification_request_v1,
            )

            outer_prepare_started_ns = time.perf_counter_ns()
            static_constraint_started_ns = time.perf_counter_ns()
            constraints = IOConstraints(vnnlib_path=str(isolated_property))
            static_constraint_ns = time.perf_counter_ns() - static_constraint_started_ns
            static_solver_started_ns = time.perf_counter_ns()
            solver = ABCrownSolver(str(args.model), config=config)
            static_solver_ns = time.perf_counter_ns() - static_solver_started_ns
            prepared_request = prepare_verification_request_v1(
                solver=solver,
                constraint=constraints,
                device="cuda",
                torch_module=torch,
                config_context=abcrown_api._config_context,
                copy_on_prune_handler=True,
            )
            if prepare_root_optimizer_warmup:
                from boundflow.runtime.prepared_root_optimizer_warmup import (
                    prepare_root_optimizer_warmup_v1,
                )

                prepared_root_warmup_receipt = prepare_root_optimizer_warmup_v1(
                    solver=solver,
                    constraints=constraints,
                    torch_module=torch,
                ).to_dict()
            static_total_prepare_ns = time.perf_counter_ns() - outer_prepare_started_ns
            torch.cuda.synchronize()
            worker_preflight = _wait_for_worker_environment()
            torch.cuda.reset_peak_memory_stats()
            environment_before = _nvidia_snapshot()
            processes_before = _compute_processes()
        with ExitStack() as stack:
            if executor is not None:
                stack.enter_context(
                    executor.instrument(
                        stage_solve=stage_solve,
                        stage_postprocess=stage_postprocess,
                    )
                )
            stack.enter_context(provider.instrument(BoundedModule))
            stack.enter_context(core.instrument(stage_solve))
            stack.enter_context(host_phases.instrument(host_phase_bindings))
            if attribute_root_incomplete:
                stack.enter_context(root_phases.instrument(root_phase_bindings))
            stack.enter_context(post.instrument(stage_postprocess, bab_bootstrap))
            stack.enter_context(queue.instrument(BatchedDomainList))
            query_start_event = torch.cuda.Event(enable_timing=True)
            query_end_event = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream()
            query_start_event.record(stream)
            query_started_ns = time.perf_counter_ns()
            if prepared_request is None:
                solver_init_started_ns = time.perf_counter_ns()
                solver = ABCrownSolver(str(args.model), config=config)
                solver_init_ns = time.perf_counter_ns() - solver_init_started_ns
                constraint_prepare_started_ns = time.perf_counter_ns()
                constraints = IOConstraints(vnnlib_path=str(isolated_property))
                constraint_prepare_ns = (
                    time.perf_counter_ns() - constraint_prepare_started_ns
                )
            else:
                solver_init_ns = 0
                constraint_prepare_ns = 0
                stack.enter_context(prepared_request.activate())
            verify_started_ns = time.perf_counter_ns()
            solver_result = solver.verify(constraints=constraints)
            verify_ended_ns = time.perf_counter_ns()
            query_end_event.record(stream)
            final_sync_started_ns = time.perf_counter_ns()
            torch.cuda.synchronize()
            final_sync_ns = time.perf_counter_ns() - final_sync_started_ns
            query_wall_ns = time.perf_counter_ns() - query_started_ns
            query_gpu_ns = int(
                round(query_start_event.elapsed_time(query_end_event) * 1e6)
            )
            if prepared_request is not None:
                receipt = prepared_request.receipt().to_dict()
                receipt.update(
                    {
                        "constraint_prepare_ns": static_constraint_ns,
                        "solver_init_ns": static_solver_ns,
                        "total_static_request_prepare_ns": static_total_prepare_ns,
                    }
                )
                prepared_request_receipt = receipt
    cold_outer_ns = time.perf_counter_ns() - cold_started_ns
    post_query_audit_ns = 0
    if executor is not None and executor.has_pending_device_audit:
        post_query_audit_started_ns = time.perf_counter_ns()
        executor.finalize_post_query_audit()
        post_query_audit_ns = time.perf_counter_ns() - post_query_audit_started_ns
    terminal_export_audit_ns = 0
    if executor is not None:
        terminal_export_audit_started_ns = time.perf_counter_ns()
        executor.finalize_terminal_export_audit()
        terminal_export_audit_ns = (
            time.perf_counter_ns() - terminal_export_audit_started_ns
        )
    core_wall_ns, core_gpu_ns = core.timings()
    if len(queue.events) != 1 or post.count != 1:
        raise ValueError("FSG3 post/queue event count differs")
    if core.host_start_ns is None or core.host_end_ns is None:
        raise ValueError("FSG3 core host timing is incomplete")
    update_bounds_post_ns, official_post_queue_ns = post.timings()
    host_phase_timings = host_phases.snapshot(host_phase_names)
    root_incomplete_timings = (
        root_phases.snapshot(
            root_name="root_incomplete", required_names=root_phase_names
        )
        if attribute_root_incomplete
        else None
    )
    query_phase_timing = _query_phase_timing(
        query_wall_ns=query_wall_ns,
        solver_init_ns=solver_init_ns,
        constraint_prepare_ns=constraint_prepare_ns,
        verify_started_ns=verify_started_ns,
        verify_ended_ns=verify_ended_ns,
        core_started_ns=core.host_start_ns,
        core_ended_ns=core.host_end_ns,
        final_sync_ns=final_sync_ns,
        update_bounds_post_ns=update_bounds_post_ns,
        official_post_queue_ns=official_post_queue_ns,
    )
    post_validation_started_ns = time.perf_counter_ns()
    semantic = _semantic_result(
        solver_result=solver_result,
        core_result=core.last_result,
        queue_event=queue.events[0],
        torch_module=torch,
    )
    post_validation_ns = time.perf_counter_ns() - post_validation_started_ns
    profile_spans = () if profile_recorder is None else profile_recorder.finalize()
    if profile_spans:
        covered_core_ns = sum(
            span.wall_ns for span in profile_spans if span.scope == "core"
        )
        profile_closure_error: Optional[float] = abs(
            core_wall_ns - covered_core_ns
        ) / float(core_wall_ns)
        profile_residual_share: Optional[float] = max(
            core_wall_ns - covered_core_ns, 0
        ) / float(core_wall_ns)
    else:
        profile_closure_error = None
        profile_residual_share = None
    environment_after = _nvidia_snapshot()
    processes_after = _compute_processes()
    runtime_environment = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "driver_version": environment_before["driver_version"],
        "gpu_total_memory": environment_before["total_memory"],
    }
    environment = _environment_gate(
        environment_before,
        environment_after,
        processes_before,
        processes_after,
        canonical_hash(runtime_environment),
    )
    if configuration == FSG3Configuration.B2:
        if executor is None or executor.core_count != 1:
            raise ValueError("FSG3 B2 executor count differs")
        typed_count = 1
        provider_core = 0
        provider_update = executor.provider_update_bounds_callback_count
        fallback = executor.fallback_dispatch_count
        backend = "torch-eager-reference"
        replacement = "whole_call"
    else:
        typed_count = core.typed_validation_count
        provider_core = core.core_count
        provider_update = core.provider_update_bounds_call_count
        fallback = 0
        backend = "auto-lirpa"
        replacement = (
            "original_provider"
            if configuration == FSG3Configuration.B0
            else "rvir_passthrough"
        )
    metrics = FSG3TimingMetrics(
        cold_total_ns=(
            compile_ns + query_wall_ns
            if configuration == FSG3Configuration.B2
            else query_wall_ns
        ),
        boundflow_compile_ns=compile_ns,
        query_wall_ns=query_wall_ns,
        query_gpu_ns=query_gpu_ns,
        core_wall_ns=core_wall_ns,
        core_gpu_ns=core_gpu_ns,
        post_validation_ns=post_validation_ns,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
    )
    execution = FSG3ExecutionCounters(
        typed_validation_count=typed_count,
        provider_core_call_count=provider_core,
        provider_compute_bounds_call_count=provider.compute_bounds_call_count,
        provider_update_bounds_call_count=provider_update,
        fallback_dispatch_count=fallback,
        backend_kind=backend,
        replacement_mode=replacement,
    )
    run = FSG3TimingRun(
        run_id=args.run_id,
        block_index=args.block_index,
        sequence_position=args.sequence_position,
        configuration=configuration,
        mode=mode,
        source_identity=_source_identity(args),
        protocol_identity=_protocol_identity(configuration),
        metrics=metrics,
        semantics=semantic,
        execution=execution,
        environment=environment,
        profile_spans=profile_spans,
        profile_closure_error=profile_closure_error,
        profile_residual_share=profile_residual_share,
    )
    run.validate()
    envelope = {
        "schema_version": WORKER_ENVELOPE_SCHEMA,
        "run": run.to_dict(),
        "diagnostics": {
            "typed_snapshot_hash": core.typed_snapshot_hash,
            "pre_state_identities": (
                [] if executor is None else executor.pre_state_identities
            ),
            "prepared_core_template_hashes": (
                [] if executor is None else executor.prepared_core_template_hashes
            ),
            "prepared_core_instance_hashes": (
                [] if executor is None else executor.prepared_core_instance_hashes
            ),
            "terminal_optimizer_schedule_hashes": (
                [] if executor is None else executor.terminal_optimizer_schedule_hashes
            ),
            "terminal_lower_adjoint_handoff_metadata": (
                []
                if executor is None
                else executor.terminal_lower_adjoint_handoff_metadata
            ),
            "terminal_export_assembly_metadata": (
                [] if executor is None else executor.terminal_export_assembly_metadata
            ),
            "native_backward_export_metadata": (
                [] if executor is None else executor.native_backward_export_metadata
            ),
            "native_backward_export_payloads": (
                [] if executor is None else executor.native_backward_export_payloads
            ),
            "assembly_metadata": (
                [] if executor is None else executor.assembly_metadata
            ),
            "commit_receipts": ([] if executor is None else executor.commit_receipts),
            "environment_before": environment_before,
            "environment_after": environment_after,
            "compute_processes_before": processes_before,
            "compute_processes_after": processes_after,
            "allowed_graphics_processes": list(ALLOWED_GRAPHICS_PROCESSES),
            "runtime_environment": runtime_environment,
            "worker_preflight": worker_preflight,
            "cold_outer_ns": cold_outer_ns,
            "cold_total_is_compile_plus_query_composite": True,
            "cold_scope_includes_hook_setup": False,
            "post_validation_excluded_from_timing": True,
            "post_query_audit_ns": post_query_audit_ns,
            "post_query_audit_excluded_from_timing": True,
            "terminal_export_audit_ns": terminal_export_audit_ns,
            "terminal_export_audit_excluded_from_timing": True,
            "query_phase_timing": query_phase_timing,
            "host_phase_timings": host_phase_timings,
            "root_incomplete_timings": root_incomplete_timings,
            "prepared_verification_request": prepared_request_receipt,
            "prepared_root_optimizer_warmup": prepared_root_warmup_receipt,
            "device_commit_audits": (
                [] if executor is None else executor.device_commit_audits
            ),
        },
        "performance_claimed": False,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(envelope, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "configuration": configuration.value,
                "query_wall_ns": query_wall_ns,
                "core_wall_ns": core_wall_ns,
                "environment_admitted": environment.admitted,
                "performance_claimed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=("B0", "B1", "B2"), required=True)
    parser.add_argument("--mode", choices=("control", "profile"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--sequence-position", type=int, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--abcrown-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--property", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run one fail-closed real control worker."""

    args = _parse_args()
    args.benchmark_root = args.benchmark_root.resolve()
    args.abcrown_root = args.abcrown_root.resolve()
    args.model = args.model.resolve()
    args.property = args.property.resolve()
    args.result = args.result.resolve()
    _worker(args)


if __name__ == "__main__":
    main()
