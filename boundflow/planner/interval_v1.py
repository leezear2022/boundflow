from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ..ir.primal import BFPrimalProgram
from ..ir.spec import LinearSpec
from ..ir.task import BFTaskModule, BufferSpec, TaskOp
from .interval_v0 import plan_interval_ibp_v0


def plan_interval_ibp_with_linear_spec(program: BFPrimalProgram, spec: LinearSpec) -> BFTaskModule:
    """
    在 interval IBP task 上附加线性 Property（C 矩阵）。

    重要：为对齐 auto_LiRPA 的 `compute_bounds(C=..., method='IBP')`，我们优先做“最后线性层融合”：
    - 若 task 最后一个 op 是 `linear`，则将 C 融合进最后线性层：W' = C @ W，b' = C @ b，
      直接对 y = W' x + b' 做 IBP（这会比“先得到 logits interval 再乘 C”更紧）。
    - 否则 fallback 为追加 `spec_linear` op（正确但可能更松）。
    """
    spec.validate()
    module = plan_interval_ibp_v0(program)
    task = module.get_entry_task()

    if len(task.output_values) != 1:
        raise ValueError(f"expected single output for now, got: {task.output_values}")

    # Keep spec in bindings for debugging/serialization.
    module.bindings.setdefault("spec", {})
    module.bindings["spec"]["linear_spec"] = {"name": spec.name, "C_shape": list(spec.C.shape)}

    if task.ops and task.ops[-1].op_type == "linear":
        _fuse_linear_spec_into_last_linear(program, module, spec)
        module.validate()
        return module

    _append_spec_linear_op(program, module, spec)
    module.validate()
    return module


def _append_spec_linear_op(program: BFPrimalProgram, module: BFTaskModule, spec: LinearSpec) -> None:
    task = module.get_entry_task()
    logits_value = task.output_values[0]
    c_value = "spec_C"
    out_value = "spec_out"

    params = module.bindings.get("params")
    if not isinstance(params, dict):
        params = {}
        module.bindings["params"] = params
    params[c_value] = spec.C

    module.storage_plan.buffers[f"buf_{c_value}"] = BufferSpec(
        buffer_id=f"buf_{c_value}",
        dtype=str(spec.C.dtype).replace("torch.", ""),
        shape=[int(d) for d in spec.C.shape],
        scope="param",
    )
    module.storage_plan.value_to_buffer[c_value] = f"buf_{c_value}"

    out_dtype = program.graph.values[logits_value].type.dtype if logits_value in program.graph.values else "float32"
    module.storage_plan.buffers[f"buf_{out_value}"] = BufferSpec(
        buffer_id=f"buf_{out_value}",
        dtype=out_dtype,
        shape=[int(spec.C.shape[0]), int(spec.C.shape[1])],
        scope="global",
    )
    module.storage_plan.value_to_buffer[out_value] = f"buf_{out_value}"

    task.ops.append(
        TaskOp(
            op_type="spec_linear",
            name="spec_linear0",
            inputs=[logits_value, c_value],
            outputs=[out_value],
            attrs={},
        )
    )
    task.output_values = [out_value]
    task.params = list(dict.fromkeys(task.params + [c_value]))


def _fuse_linear_spec_into_last_linear(program: BFPrimalProgram, module: BFTaskModule, spec: LinearSpec) -> None:
    task = module.get_entry_task()
    last = task.ops[-1]
    if last.op_type != "linear":
        raise ValueError("expected last op_type == 'linear'")
    if len(last.inputs) != 3:
        raise ValueError(f"expected last linear has bias, got inputs={last.inputs}")

    x_value, w_value, b_value = last.inputs
    logits_value = last.outputs[0]

    params = module.bindings.get("params")
    if not isinstance(params, dict):
        raise ValueError("module.bindings['params'] must exist for fused spec")

    if w_value not in params or b_value not in params:
        raise KeyError(f"missing linear params for spec fusion: {w_value}, {b_value}")

    W = params[w_value]
    b = params[b_value]
    if not torch.is_tensor(W):
        W = torch.as_tensor(W)
    if not torch.is_tensor(b):
        b = torch.as_tensor(b)

    C = spec.C
    if not torch.is_tensor(C):
        C = torch.as_tensor(C)

    if W.dim() != 2:
        raise ValueError(f"expected W rank-2 [O,I], got {tuple(W.shape)}")
    if b.dim() != 1:
        raise ValueError(f"expected b rank-1 [O], got {tuple(b.shape)}")
    if C.shape[2] != W.shape[0]:
        raise ValueError(f"C out_dim mismatch: C={tuple(C.shape)} W={tuple(W.shape)}")

    spec_W_value = "spec_W"
    spec_b_value = "spec_b"
    out_value = "spec_out"

    # Merge C into the final linear parameters:
    # W': [B,S,I] = C [B,S,O] @ W [O,I]
    # b': [B,S]   = C [B,S,O] @ b [O]
    Wp = torch.matmul(C, W)  # [B,S,I]
    bp = torch.matmul(C, b)  # [B,S]
    params[spec_W_value] = Wp
    params[spec_b_value] = bp

    # Update storage plan.
    module.storage_plan.buffers[f"buf_{spec_W_value}"] = BufferSpec(
        buffer_id=f"buf_{spec_W_value}",
        dtype=str(Wp.dtype).replace("torch.", ""),
        shape=[int(d) for d in Wp.shape],
        scope="param",
    )
    module.storage_plan.value_to_buffer[spec_W_value] = f"buf_{spec_W_value}"
    module.storage_plan.buffers[f"buf_{spec_b_value}"] = BufferSpec(
        buffer_id=f"buf_{spec_b_value}",
        dtype=str(bp.dtype).replace("torch.", ""),
        shape=[int(d) for d in bp.shape],
        scope="param",
    )
    module.storage_plan.value_to_buffer[spec_b_value] = f"buf_{spec_b_value}"

    out_dtype = program.graph.values[logits_value].type.dtype if logits_value in program.graph.values else "float32"
    module.storage_plan.buffers[f"buf_{out_value}"] = BufferSpec(
        buffer_id=f"buf_{out_value}",
        dtype=out_dtype,
        shape=[int(C.shape[0]), int(C.shape[1])],
        scope="global",
    )
    module.storage_plan.value_to_buffer[out_value] = f"buf_{out_value}"

    # Rewrite last op to use fused params and output.
    last.inputs = [x_value, spec_W_value, spec_b_value]
    last.outputs = [out_value]
    task.output_values = [out_value]

    task.params = list(dict.fromkeys(task.params + [spec_W_value, spec_b_value]))

