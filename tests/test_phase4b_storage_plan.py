import torch
import torch.nn as nn

from boundflow.frontends.pytorch.frontend import import_torch
from boundflow.planner import plan_interval_ibp_v0


def test_plan_interval_ibp_v0_fills_storage_plan():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    x0 = torch.randn(2, 4)
    program = import_torch(model, (x0,), export_mode="export", normalize=True)
    module = plan_interval_ibp_v0(program)

    assert module.storage_plan.buffers
    assert module.storage_plan.value_to_buffer
    for value_name, buffer_id in module.storage_plan.value_to_buffer.items():
        assert buffer_id in module.storage_plan.buffers
        spec = module.storage_plan.buffers[buffer_id]
        assert spec.dtype
        assert isinstance(spec.shape, list)
        assert spec.buffer_id == buffer_id

