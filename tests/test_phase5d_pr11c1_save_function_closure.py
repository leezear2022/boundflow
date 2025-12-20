import torch
import torch.nn as nn

from boundflow.backends.tvm.relax_interval_task_ops import build_interval_task_relax_ops_ir_module
from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner.interval_v2 import IntervalV2PartitionConfig, plan_interval_ibp_v2


def test_pr11c1_relax_vm_save_function_closure_returns_same_output():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
    x0 = torch.randn(4, 16)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    module = plan_interval_ibp_v2(program, config=IntervalV2PartitionConfig(min_tasks=1))
    task = module.get_entry_task()

    import tvm
    from tvm import relax
    from tvm.runtime import _tensor as rt

    ir_mod, spec = build_interval_task_relax_ops_ir_module(task, storage_plan=module.storage_plan, target="llvm", func_name="main")
    ex = relax.build(ir_mod, target="llvm")
    dev = tvm.cpu(0)
    vm = relax.VirtualMachine(ex, dev)

    params = module.bindings.get("params", {}) or {}
    x_l = x0 - 0.1
    x_u = x0 + 0.1
    args_tvm = [
        rt.tensor(x_l.detach().cpu().numpy(), device=dev),
        rt.tensor(x_u.detach().cpu().numpy(), device=dev),
    ]
    for p_name in spec.param_values:
        t = params[p_name]
        if not torch.is_tensor(t):
            t = torch.as_tensor(t)
        args_tvm.append(rt.tensor(t.detach().cpu().numpy(), device=dev))

    out1 = vm[spec.func_name](*args_tvm)
    vm.save_function(spec.func_name, "saved", *args_tvm, include_return=True)
    out2 = vm["saved"]()

    # out is a tuple of tensors
    for a, b in zip(list(out1), list(out2)):
        ta = torch.from_numpy(a.numpy())
        tb = torch.from_numpy(b.numpy())
        assert torch.allclose(ta, tb, atol=1e-6, rtol=1e-6)

