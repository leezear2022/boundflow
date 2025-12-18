import json

import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner.core import PlannerConfig
from boundflow.planner.options import PartitionOptions, PartitionPolicy
from boundflow.planner.pipeline import plan
from boundflow.runtime.scheduler import run_ibp_scheduled
from boundflow.runtime.task_executor import LinfInputSpec, PythonTaskExecutor


def test_phase5c_pr5_pipeline_config_dump_is_jsonable_and_equivalent():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    x0 = torch.randn(4, 16)
    eps = 0.05

    program = import_torch(model, (x0,), export_mode="export", normalize=True)

    cfg0 = PlannerConfig(
        enable_task_graph=False,
        partition=PartitionOptions(policy=PartitionPolicy.V0_SINGLE_TASK, min_tasks=1),
    )
    b0 = plan(program, config=cfg0)
    assert "config_dump" in b0.meta
    json.dumps(b0.meta["config_dump"])

    cfg2 = PlannerConfig(
        enable_task_graph=True,
        partition=PartitionOptions(policy=PartitionPolicy.V2_BASELINE, min_tasks=2),
    )
    b2 = plan(program, config=cfg2)
    assert "config_dump" in b2.meta
    json.dumps(b2.meta["config_dump"])
    assert len(b2.task_module.tasks) >= 2

    out_name = b0.task_module.get_entry_task().output_values[0]
    spec = LinfInputSpec(value_name="input", center=x0, eps=eps)
    out0 = PythonTaskExecutor().run_ibp(b0.task_module, spec, output_value=out_name)
    out2 = run_ibp_scheduled(b2.task_module, spec, executor=PythonTaskExecutor(), output_value=out_name)

    assert torch.allclose(out2.lower, out0.lower, rtol=1e-5, atol=1e-6)
    assert torch.allclose(out2.upper, out0.upper, rtol=1e-5, atol=1e-6)

