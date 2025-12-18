import json

import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner.core import PlannerConfig
from boundflow.planner.options import PartitionOptions, PartitionPolicy, PlannerDebugOptions
from boundflow.planner.pipeline import plan
from boundflow.planner.storage_reuse import StorageReuseOptions


def test_pr7_planner_is_deterministic_for_same_program_and_config():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    x0 = torch.randn(4, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)

    cfg = PlannerConfig(
        enable_task_graph=True,
        partition=PartitionOptions(policy=PartitionPolicy.V2_BASELINE, min_tasks=2),
        enable_storage_reuse=True,
        storage_reuse=StorageReuseOptions(enabled=True),
        debug=PlannerDebugOptions(dump_config=True, validate_after_each_pass=True),
    )

    b1 = plan(program, config=cfg)
    b2 = plan(program, config=cfg)

    assert b1.meta.get("planner_steps") == b2.meta.get("planner_steps")
    assert b1.meta.get("config_dump") == b2.meta.get("config_dump")

    tg1 = b1.task_module.task_graph
    tg2 = b2.task_module.task_graph
    assert tg1 is not None
    assert tg2 is not None
    order1 = tg1.topo_sort(
        tasks_by_id={t.task_id: t for t in b1.task_module.tasks},
        entry_task_id=b1.task_module.entry_task_id,
    )
    order2 = tg2.topo_sort(
        tasks_by_id={t.task_id: t for t in b2.task_module.tasks},
        entry_task_id=b2.task_module.entry_task_id,
    )
    assert order1 == order2

    sp1 = b1.task_module.storage_plan
    sp2 = b2.task_module.storage_plan
    assert sp1.logical_to_physical == sp2.logical_to_physical
    assert sorted(list(sp1.physical_buffers.keys())) == sorted(list(sp2.physical_buffers.keys()))


def test_pr7_dump_plan_instrument_writes_json_snapshots(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    x0 = torch.randn(4, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)

    run_id = "test_run"
    cfg = PlannerConfig(
        enable_task_graph=True,
        partition=PartitionOptions(policy=PartitionPolicy.V2_BASELINE, min_tasks=2),
        debug=PlannerDebugOptions(
            dump_config=False,
            validate_after_each_pass=False,
            dump_plan=True,
            dump_plan_dir=str(tmp_path),
            dump_plan_run_id=run_id,
        ),
    )

    _ = plan(program, config=cfg)

    out_dir = tmp_path / run_id
    files = sorted(out_dir.glob("*.json"))
    assert files
    assert any("interval_v2_partition" in f.name for f in files)

    d = json.loads(files[0].read_text(encoding="utf-8"))
    assert d["step"] in ("interval_v2_partition", "interval_v0_single_task", "storage_reuse")

