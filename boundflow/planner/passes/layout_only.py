from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ...ir.task import TaskOp


def simplify_layout_only_ops(
    ops: List[TaskOp], *, output_values: List[str]
) -> Tuple[List[TaskOp], List[str]]:
    """
    Planner pass v0: simplify layout-only permutes.

    Rules:
    - consecutive permute composition: permute(p1) -> permute(p2) => permute(compose(p1,p2))
    - identity permute elimination

    This pass rewrites value names via aliasing so the executor sees a consistent SSA environment.
    """

    alias: Dict[str, str] = {}

    def resolve(value_name: str) -> str:
        while value_name in alias:
            value_name = alias[value_name]
        return value_name

    def is_identity_perm(dims: List[int]) -> bool:
        return dims == list(range(len(dims)))

    def normalize_dims(attrs: Dict[str, Any]) -> List[int] | None:
        dims = attrs.get("dims")
        if not isinstance(dims, (list, tuple)):
            return None
        try:
            return [int(d) for d in list(dims)]
        except Exception:
            return None

    new_ops: List[TaskOp] = []

    for op in ops:
        resolved_inputs = [resolve(v) for v in op.inputs]
        op_type = op.op_type
        if op_type == "transpose":
            op_type = "permute"

        if op_type == "permute":
            if len(resolved_inputs) != 1 or len(op.outputs) != 1:
                new_ops.append(
                    TaskOp(
                        op_type=op_type,
                        name=op.name,
                        inputs=resolved_inputs,
                        outputs=list(op.outputs),
                        attrs=dict(op.attrs),
                    )
                )
                continue

            dims = normalize_dims(op.attrs)
            if dims is None:
                new_ops.append(
                    TaskOp(
                        op_type=op_type,
                        name=op.name,
                        inputs=resolved_inputs,
                        outputs=list(op.outputs),
                        attrs=dict(op.attrs),
                    )
                )
                continue

            inp = resolved_inputs[0]
            out = op.outputs[0]

            if is_identity_perm(dims):
                alias[out] = inp
                continue

            if new_ops and new_ops[-1].op_type == "permute" and len(new_ops[-1].outputs) == 1:
                prev = new_ops[-1]
                if prev.outputs[0] == inp and len(prev.inputs) == 1:
                    prev_dims = normalize_dims(prev.attrs)
                    if prev_dims is not None:
                        composed = [prev_dims[d] for d in dims]
                        if is_identity_perm(composed):
                            new_ops.pop()
                            alias[prev.outputs[0]] = prev.inputs[0]
                            alias[out] = prev.inputs[0]
                        else:
                            prev.attrs = dict(prev.attrs)
                            prev.attrs["dims"] = composed
                            prev.attrs.setdefault("layout_only", True)
                            alias[out] = prev.outputs[0]
                        continue

            attrs = dict(op.attrs)
            attrs["dims"] = dims
            attrs.setdefault("layout_only", True)
            new_ops.append(
                TaskOp(
                    op_type="permute",
                    name=op.name,
                    inputs=[inp],
                    outputs=[out],
                    attrs=attrs,
                )
            )
            continue

        new_ops.append(
            TaskOp(
                op_type=op_type,
                name=op.name,
                inputs=resolved_inputs,
                outputs=list(op.outputs),
                attrs=dict(op.attrs),
            )
        )

    def stable_unique(values: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for v in values:
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
        return out

    resolved_outputs = stable_unique([resolve(v) for v in output_values])
    return new_ops, resolved_outputs

