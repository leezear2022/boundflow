import pytest
import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.ir.task import BFTaskModule, BufferSpec, StoragePlan
from boundflow.planner.core import PlannerConfig
from boundflow.planner.options import PartitionOptions, PartitionPolicy, PlannerDebugOptions
from boundflow.planner.pipeline import plan
from boundflow.planner.storage_reuse import StorageReuseOptions
from boundflow.planner.verify import (
    verify_liveness_reuse_consistency,
    verify_storage_plan_soundness,
    verify_task_graph_soundness,
)


def test_pr6_verify_task_graph_catches_missing_edge_dep():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    x0 = torch.randn(4, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)

    cfg = PlannerConfig(
        enable_task_graph=True,
        partition=PartitionOptions(policy=PartitionPolicy.V2_BASELINE, min_tasks=2),
    )
    bundle = plan(program, config=cfg)
    module = bundle.task_module
    assert module.task_graph is not None
    assert len(module.task_graph.edges) >= 1

    # Break: delete all deps from the first edge.
    broken = module.task_graph
    e0 = broken.edges[0]
    broken.edges[0] = type(e0)(src_task_id=e0.src_task_id, dst_task_id=e0.dst_task_id, deps=[])
    rep = verify_task_graph_soundness(module)
    assert rep.ok is False
    assert any(err.code == "missing_edge_dep" for err in rep.errors)


def test_pr6_verify_storage_plan_requires_total_mapping_when_physical_buffers_present():
    # Build a minimal module with inconsistent storage plan:
    # - physical_buffers is non-empty
    # - logical_to_physical is missing a logical buffer mapping
    buffers = {
        "buf_a": BufferSpec(buffer_id="buf_a", dtype="float32", shape=[4, 8], scope="global"),
        "buf_b": BufferSpec(buffer_id="buf_b", dtype="float32", shape=[4, 8], scope="global"),
    }
    sp = StoragePlan(
        buffers=buffers,
        value_to_buffer={"a": "buf_a", "b": "buf_b"},
        physical_buffers={"buf_a": buffers["buf_a"]},
        logical_to_physical={"buf_a": "buf_a"},
    )
    # Reuse any existing tiny module shape by planning and then swapping storage_plan.
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 8))
    x0 = torch.randn(1, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    bundle = plan(program, config=PlannerConfig(partition=PartitionOptions(policy=PartitionPolicy.V0_SINGLE_TASK)))
    module = bundle.task_module
    module.storage_plan = sp
    rep = verify_storage_plan_soundness(module)
    assert rep.ok is False
    assert any(err.code == "logical_to_physical_not_total" for err in rep.errors)


def test_pr6_verify_liveness_overlap_detects_conflicting_alias():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    x0 = torch.randn(4, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)

    cfg = PlannerConfig(
        enable_task_graph=True,
        partition=PartitionOptions(policy=PartitionPolicy.V2_BASELINE, min_tasks=2),
        enable_storage_reuse=True,
        storage_reuse=StorageReuseOptions(enabled=True),
    )
    bundle = plan(program, config=cfg)
    module = bundle.task_module
    assert module.task_graph is not None

    # Break: force all logical buffers to alias to the input physical buffer, causing overlaps.
    any_logical = next(iter(module.storage_plan.buffers.keys()))
    sp = module.storage_plan
    sp.logical_to_physical = {bid: any_logical for bid in sp.buffers.keys()}
    sp.physical_buffers = {any_logical: sp.buffers[any_logical]}

    rep = verify_liveness_reuse_consistency(module)
    assert rep.ok is False
    assert any(err.code == "physical_lifetime_overlap" for err in rep.errors)


def test_pr6_pipeline_validate_after_each_pass_records_reports():
    # With validate_after_each_pass enabled, the pipeline should record verify reports per step.
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
    bundle = plan(program, config=cfg)
    assert "verify" in bundle.meta
    assert "interval_v2_partition" in bundle.meta["verify"]
    assert "storage_reuse" in bundle.meta["verify"]
