import pytest
import torch
import torch.nn as nn

from boundflow.backends.tvm.relax_task_lowering import (
    RelaxLoweringMode,
    lower_interval_linear_task_to_relax_ir,
)
from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner.interval_v0 import plan_interval_ibp_v0


@pytest.mark.parametrize("mode", [RelaxLoweringMode.RELAX_OPS, RelaxLoweringMode.CALL_TIR])
def test_pr8_lower_single_linear_task_to_relax_ir_module(mode):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 8))
    x0 = torch.randn(4, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    module = plan_interval_ibp_v0(program)
    task = module.get_entry_task()

    res = lower_interval_linear_task_to_relax_ir(task, module, target="llvm", mode=mode)
    assert res.relax_func_name

    import tvm  # noqa: PLC0415
    from tvm import relax  # noqa: PLC0415

    assert isinstance(res.ir_mod, tvm.IRModule)
    gv = res.ir_mod.get_global_var(res.relax_func_name)
    assert gv is not None
    func = res.ir_mod[gv]
    assert isinstance(func, relax.Function)


def test_pr8_relax_ir_module_is_buildable():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 8))
    x0 = torch.randn(4, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    module = plan_interval_ibp_v0(program)
    task = module.get_entry_task()

    res = lower_interval_linear_task_to_relax_ir(task, module, target="llvm", mode=RelaxLoweringMode.CALL_TIR)

    from tvm import relax  # noqa: PLC0415

    ex = relax.build(res.ir_mod, target="llvm")
    assert ex is not None

