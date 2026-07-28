"""Immutable artifact contract for Schedule IR v1 and its runtime trace."""

# Artifact verification keeps all immutable-file checks in one fail-closed path.
# pylint: disable=too-many-arguments,too-many-branches,too-many-locals

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..ir.bound import BFBoundModule
from ..ir.plan import PlanInstance, PlanTemplate
from ..ir.schedule import ScheduleModule
from ..ir.task_v1 import TaskIRModule
from .schedule_ir_executor import (
    ScheduleExecutionTrace,
    execute_schedule_reference,
    replay_schedule_trace,
)
from .task_ir_executor import TaskExecutionTrace, execute_task_ir_reference

SCHEDULE_ARTIFACT_SCHEMA_VERSION = "boundflow.schedule-ir-artifact/v1"


def write_schedule_ir_artifact(
    output_dir: Path,
    *,
    bound_module: BFBoundModule,
    template: PlanTemplate,
    instance: PlanInstance,
    task_module: TaskIRModule,
    schedule: ScheduleModule,
    trace: ScheduleExecutionTrace,
    task_trace: TaskExecutionTrace,
) -> Path:
    """Write one new immutable schedule/trace evidence directory."""

    schedule.validate(bound_module=bound_module, template=template, instance=instance)
    expected_trace = execute_schedule_reference(
        schedule,
        bound_module=bound_module,
        template=template,
        instance=instance,
    )
    if trace != expected_trace:
        raise ValueError(
            "supplied Schedule IR trace is not deterministic reference output"
        )
    expected_task_trace = execute_task_ir_reference(
        task_module,
        schedule,
        bound_module=bound_module,
        template=template,
        instance=instance,
    )
    if task_trace != expected_task_trace:
        raise ValueError("supplied Task IR trace is not deterministic reference output")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty Schedule IR artifact: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "bound_module.json": bound_module.canonical_json(),
        "plan_template.json": template.canonical_json(bound_module=bound_module),
        "plan_instance.json": instance.canonical_json(
            template=template, bound_module=bound_module
        ),
        "task_module.json": task_module.canonical_json(
            bound_module=bound_module,
            template=template,
            instance=instance,
        ),
        "schedule.json": schedule.canonical_json(
            bound_module=bound_module,
            template=template,
            instance=instance,
        ),
        "trace.json": trace.canonical_json(),
        "task_trace.json": task_trace.canonical_json(),
    }
    for filename, payload in payloads.items():
        (output_dir / filename).write_text(payload + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEDULE_ARTIFACT_SCHEMA_VERSION,
        "bound_module_hash": bound_module.stable_hash(),
        "plan_template_hash": template.stable_hash(bound_module=bound_module),
        "plan_instance_hash": instance.stable_hash(
            template=template, bound_module=bound_module
        ),
        "schedule_hash": schedule.stable_hash(
            bound_module=bound_module,
            template=template,
            instance=instance,
        ),
        "trace_hash": trace.stable_hash(),
        "task_module_hash": task_module.stable_hash(
            bound_module=bound_module,
            template=template,
            instance=instance,
        ),
        "task_trace_hash": task_trace.stable_hash(),
        "files": {
            filename: _sha256_text(payload + "\n")
            for filename, payload in sorted(payloads.items())
        },
        "scope": "synchronous reference contract; not a performance claim",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def verify_schedule_ir_artifact(
    output_dir: Path,
    *,
    bound_module: BFBoundModule,
    template: PlanTemplate,
    instance: PlanInstance,
    task_module: TaskIRModule,
    schedule: ScheduleModule,
) -> ScheduleExecutionTrace:
    """Verify immutable bytes and independently replay the schedule trace."""

    schedule.validate(bound_module=bound_module, template=template, instance=instance)
    expected_payloads = {
        "bound_module.json": bound_module.canonical_json(),
        "plan_template.json": template.canonical_json(bound_module=bound_module),
        "plan_instance.json": instance.canonical_json(
            template=template, bound_module=bound_module
        ),
        "task_module.json": task_module.canonical_json(
            bound_module=bound_module,
            template=template,
            instance=instance,
        ),
        "schedule.json": schedule.canonical_json(
            bound_module=bound_module,
            template=template,
            instance=instance,
        ),
    }
    texts: dict[str, str] = {}
    for filename, payload in expected_payloads.items():
        path = output_dir / filename
        if not path.is_file():
            raise ValueError(f"Schedule IR artifact is missing {filename}")
        text = path.read_text(encoding="utf-8")
        if text != payload + "\n":
            raise ValueError(f"Schedule IR artifact typed input mismatch: {filename}")
        texts[filename] = text
    trace_path = output_dir / "trace.json"
    task_trace_path = output_dir / "task_trace.json"
    manifest_path = output_dir / "manifest.json"
    if (
        not trace_path.is_file()
        or not task_trace_path.is_file()
        or not manifest_path.is_file()
    ):
        raise ValueError("Schedule IR artifact is missing trace or manifest")
    trace_text = trace_path.read_text(encoding="utf-8")
    if not trace_text.endswith("\n"):
        raise ValueError("Schedule IR artifact trace must end with one newline")
    texts["trace.json"] = trace_text
    trace = replay_schedule_trace(
        trace_text[:-1],
        schedule,
        bound_module=bound_module,
        template=template,
        instance=instance,
    )
    task_trace_text = task_trace_path.read_text(encoding="utf-8")
    expected_task_trace = execute_task_ir_reference(
        task_module,
        schedule,
        bound_module=bound_module,
        template=template,
        instance=instance,
    )
    if task_trace_text != expected_task_trace.canonical_json() + "\n":
        raise ValueError("Schedule IR artifact Task trace replay mismatch")
    texts["task_trace.json"] = task_trace_text
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Schedule IR artifact manifest is invalid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("Schedule IR artifact manifest must be an object")
    if manifest.get("schema_version") != SCHEDULE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported Schedule IR artifact schema")
    expected_hashes = {
        "bound_module_hash": bound_module.stable_hash(),
        "plan_template_hash": template.stable_hash(bound_module=bound_module),
        "plan_instance_hash": instance.stable_hash(
            template=template, bound_module=bound_module
        ),
        "schedule_hash": schedule.stable_hash(
            bound_module=bound_module,
            template=template,
            instance=instance,
        ),
        "trace_hash": trace.stable_hash(),
        "task_module_hash": task_module.stable_hash(
            bound_module=bound_module,
            template=template,
            instance=instance,
        ),
        "task_trace_hash": expected_task_trace.stable_hash(),
    }
    for key, value in expected_hashes.items():
        if manifest.get(key) != value:
            raise ValueError(f"Schedule IR artifact {key} mismatch")
    file_hashes = manifest.get("files")
    if not isinstance(file_hashes, dict):
        raise ValueError("Schedule IR artifact file hash table is invalid")
    for filename, text in texts.items():
        if file_hashes.get(filename) != _sha256_text(text):
            raise ValueError(f"Schedule IR artifact file hash mismatch: {filename}")
    return trace


def _sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
