"""Generate or replay the Schedule IR v1 reference artifact."""

# The CLI intentionally mirrors artifact hash/replay calls.
# pylint: disable=duplicate-code,missing-function-docstring,too-many-locals

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from boundflow.ir.schedule import lower_plan_instance_to_reference_schedule
from boundflow.ir.task_v1 import lower_plan_instance_to_task_ir
from boundflow.planner.plan_ir_selector import select_plan_instance
from boundflow.runtime.schedule_ir_artifact import (
    verify_schedule_ir_artifact,
    write_schedule_ir_artifact,
)
from boundflow.runtime.schedule_ir_executor import execute_schedule_reference
from boundflow.runtime.task_ir_executor import execute_task_ir_semantics
from scripts.run_plan_ir_v1_reference_artifact import (
    build_reference_smoke_workload,
)


def _reconstruct():
    workload = build_reference_smoke_workload()
    bound_module = workload.bound_module
    template = workload.template
    instance = select_plan_instance(
        template,
        bound_module=bound_module,
        query_bucket_id="schedule-reference-smoke",
        available_memory_bytes=1 << 30,
        memory_budget_bytes=1 << 30,
    )
    schedule = lower_plan_instance_to_reference_schedule(
        bound_module,
        template=template,
        instance=instance,
        query_ids=("query:0", "query:1"),
    )
    task_module = lower_plan_instance_to_task_ir(
        bound_module,
        template=template,
        instance=instance,
    )
    return workload, instance, task_module, schedule


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--out-dir", type=Path, required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    workload, instance, task_module, schedule = _reconstruct()
    bound_module = workload.bound_module
    template = workload.template
    if args.command == "generate":
        trace = execute_schedule_reference(
            schedule,
            bound_module=bound_module,
            template=template,
            instance=instance,
        )
        _result, task_trace = execute_task_ir_semantics(
            task_module,
            schedule,
            bound_module=bound_module,
            template=template,
            instance=instance,
            legacy_task_module=workload.task_module,
            input_spec=workload.input_spec,
            relu_pre=workload.relu_pre,
        )
        manifest = write_schedule_ir_artifact(
            args.out_dir,
            bound_module=bound_module,
            template=template,
            instance=instance,
            task_module=task_module,
            schedule=schedule,
            trace=trace,
            task_trace=task_trace,
            legacy_task_module=workload.task_module,
            input_spec=workload.input_spec,
            relu_pre=workload.relu_pre,
        )
        status = "generated"
    else:
        trace = verify_schedule_ir_artifact(
            args.artifact_dir,
            bound_module=bound_module,
            template=template,
            instance=instance,
            task_module=task_module,
            schedule=schedule,
            legacy_task_module=workload.task_module,
            input_spec=workload.input_spec,
            relu_pre=workload.relu_pre,
        )
        _result, task_trace = execute_task_ir_semantics(
            task_module,
            schedule,
            bound_module=bound_module,
            template=template,
            instance=instance,
            legacy_task_module=workload.task_module,
            input_spec=workload.input_spec,
            relu_pre=workload.relu_pre,
        )
        manifest = args.artifact_dir / "manifest.json"
        status = "replayed"
    print(
        json.dumps(
            {
                "status": status,
                "manifest": str(manifest),
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
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
