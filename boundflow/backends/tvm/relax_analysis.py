from __future__ import annotations

from typing import Any, Dict, Tuple


def count_relax_call_tir(ir_mod: Any) -> int:
    """
    Count the number of `relax.call_tir` calls across all Relax functions in the module.
    """
    import tvm  # noqa: PLC0415
    from tvm import relax  # noqa: PLC0415

    n = 0

    def visit(expr) -> None:
        nonlocal n
        if isinstance(expr, relax.Call):
            op = expr.op
            if isinstance(op, tvm.ir.Op) and op.name == "relax.call_tir":
                n += 1

    for gv, func in ir_mod.functions.items():
        if isinstance(func, relax.Function):
            relax.analysis.post_order_visit(func.body, visit)
    return int(n)


def count_mod_funcs(ir_mod: Any) -> Dict[str, int]:
    """
    Return counts of Relax/TIR functions in an IRModule.
    """
    from tvm import relax  # noqa: PLC0415
    from tvm.tir import PrimFunc  # noqa: PLC0415

    relax_funcs = 0
    tir_funcs = 0
    for _, func in ir_mod.functions.items():
        if isinstance(func, relax.Function):
            relax_funcs += 1
        elif isinstance(func, PrimFunc):
            tir_funcs += 1
    return {"relax_funcs": int(relax_funcs), "tir_funcs": int(tir_funcs)}


def collect_relax_ir_stats(ir_mod: Any) -> Dict[str, Any]:
    """
    Collect a small set of IR statistics that are useful for compile/runtime ablation.
    """
    out: Dict[str, Any] = {}
    out.update(count_mod_funcs(ir_mod))
    out["call_tir"] = int(count_relax_call_tir(ir_mod))
    return out

