from __future__ import annotations

from typing import Any, Dict

from ..ir.primal import BFPrimalProgram
from ..ir.task import BFTaskModule, BoundTask, BufferSpec, StoragePlan, TaskKind, TaskLowering, TaskOp
from .passes.layout_only import simplify_layout_only_ops


def plan_interval_ibp_v0(program: BFPrimalProgram) -> BFTaskModule:
    """
    Planner v0：把整张 Primal Graph 打包成一个 interval-IBP 任务。

    目标是把“解释执行”抽象成任务形态，便于 Phase 4 之后逐步引入 fusion/batching/reuse 与 TVM lowering。
    """
    program.graph.validate()

    ops = [
        TaskOp(
            op_type=node.op_type,
            name=node.name,
            inputs=list(node.inputs),
            outputs=list(node.outputs),
            attrs=dict(node.attrs),
        )
        for node in program.graph.nodes
    ]
    # 显式标注 affine 类算子，避免 domain 通过 tensor rank 猜测具体 op。
    for op in ops:
        if op.op_type in ("linear", "conv2d"):
            op.attrs = dict(op.attrs)
            op.attrs.setdefault("op", op.op_type)
        if op.op_type == "reshape":
            # 目前 TaskOp 不携带 Value meta，因此把目标 shape 放到 attrs 里。
            out = op.outputs[0]
            value = program.graph.values.get(out)
            if value is not None and value.type.shape:
                op.attrs = dict(op.attrs)
                op.attrs.setdefault("shape", [(-1 if d is None else int(d)) for d in value.type.shape])

    ops, output_values = simplify_layout_only_ops(ops, output_values=list(program.graph.outputs))

    storage_plan = _default_storage_plan(program)
    def _uniq(xs):
        seen = set()
        out = []
        for x in xs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    task = BoundTask(
        task_id="ibp_task0",
        kind=TaskKind.INTERVAL_IBP,
        ops=ops,
        input_values=list(program.graph.inputs),
        output_values=output_values,
        input_buffers=_uniq([storage_plan.value_to_buffer[v] for v in list(program.graph.inputs) if v in storage_plan.value_to_buffer]),
        output_buffers=_uniq([storage_plan.value_to_buffer[v] for v in output_values if v in storage_plan.value_to_buffer]),
        params=list(program.params.keys()),
        batch_axes={},
        memory_plan={},
        lowering=TaskLowering.TVM_TIR,
    )
    module = BFTaskModule(
        tasks=[task],
        entry_task_id=task.task_id,
        tvm_mod=None,
        bindings=_default_bindings(program),
        storage_plan=storage_plan,
    )
    module.validate()
    return module


def _default_bindings(program: BFPrimalProgram) -> Dict[str, Any]:
    # v0.1: bindings 仅包含 params/const，inputs 由 runtime 在 run() 时提供。
    return {"params": dict(program.params), "tensor_meta": dict(program.tensor_meta)}


def _default_storage_plan(program: BFPrimalProgram) -> StoragePlan:
    """
    v0.1 默认 storage plan：一值一 buffer。
    后续 planner 可以在此基础上做 buffer aliasing、liveness-aware reuse 等优化。
    """
    buffers: Dict[str, BufferSpec] = {}
    value_to_buffer: Dict[str, str] = {}
    for value_name, value in program.graph.values.items():
        buffer_id = f"buf_{value_name}"
        scope = "global"
        if getattr(value, "kind", None) is not None:
            if value.kind.value == "param":
                scope = "param"
            elif value.kind.value == "const":
                scope = "const"
        buffers[buffer_id] = BufferSpec(
            buffer_id=buffer_id,
            dtype=value.type.dtype,
            shape=list(value.type.shape),
            scope=scope,
            device=None,
            layout=value.type.layout,
        )
        value_to_buffer[value_name] = buffer_id
    return StoragePlan(buffers=buffers, value_to_buffer=value_to_buffer)
