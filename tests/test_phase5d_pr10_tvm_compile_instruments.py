import os
from pathlib import Path

import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner.interval_v2 import IntervalV2PartitionConfig, plan_interval_ibp_v2
from boundflow.runtime.scheduler import run_ibp_scheduled
from boundflow.runtime.task_executor import LinfInputSpec
from boundflow.runtime.tvm_executor import TVMExecutorOptions, TVMTaskExecutor


def test_pr10_tvm_compile_pass_timing_and_dumpir(tmp_path: Path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 8))
    x0 = torch.randn(4, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    module = plan_interval_ibp_v2(program, config=IntervalV2PartitionConfig(min_tasks=1))

    os.environ["BOUNDFLOW_TVM_RUN_ID"] = "pytest_run"

    exe = TVMTaskExecutor(
        options=TVMExecutorOptions(
            target="llvm",
            kernel_style="call_tir",
            enable_pass_timing=True,
            enable_dump_ir=True,
            dump_ir_dir=str(tmp_path),
            dump_ir_refresh=True,
        )
    )
    input_spec = LinfInputSpec(value_name=program.graph.inputs[0], center=x0, eps=0.1)
    _ = run_ibp_scheduled(module, input_spec, executor=exe)

    stats = exe.get_compile_stats()
    assert stats
    one = next(iter(stats.values()))
    assert one.get("compile_ms") is not None
    assert one.get("pass_timing_render")
    assert one.get("dump_ir_dir")
    assert (tmp_path / "pytest_run").exists()

