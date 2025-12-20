from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


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


def _extract_int_from_relax_expr(expr: Any) -> Optional[int]:
    import tvm  # noqa: PLC0415
    from tvm import relax  # noqa: PLC0415

    if isinstance(expr, relax.PrimValue):
        v = expr.value
        if isinstance(v, tvm.tir.IntImm):
            return int(v.value)
        try:
            simplified = tvm.arith.Analyzer().simplify(v)
            if isinstance(simplified, tvm.tir.IntImm):
                return int(simplified.value)
        except Exception:
            pass
    return None


def collect_relax_memory_stats(ir_mod: Any) -> Dict[str, Any]:
    """
    Collect coarse-grained memory planning stats from a Relax IRModule.

    Notes:
    - We intentionally keep this best-effort and json-able.
    - This is meant for relative ablation (e.g. StaticPlanBlockMemory on/off),
      not for exact peak memory accounting.
    """
    import tvm  # noqa: PLC0415
    from tvm import relax  # noqa: PLC0415

    alloc_storage = 0
    alloc_tensor = 0
    total_bytes = 0
    max_bytes = 0
    nonconst_bytes = 0

    def _dtype_nbytes(dtype: str) -> Optional[int]:
        try:
            dt = tvm.runtime.DataType(dtype)
            if dt.bits % 8 != 0:
                return None
            return int(dt.bits // 8) * int(dt.lanes)
        except Exception:
            return None

    def _extract_alloc_storage_bytes(call: Any) -> Optional[int]:
        """
        Try to compute allocated bytes for alloc_storage calls.

        Supported patterns:
        - relax.memory.alloc_storage(size_bytes, ...)
        - relax.vm.alloc_storage(shape, ..., dtype, ...)  (bytes = prod(shape) * sizeof(dtype))
        """
        op = call.op
        if not isinstance(op, tvm.ir.Op):
            return None

        if op.name == "relax.memory.alloc_storage":
            return _extract_int_from_relax_expr(call.args[0]) if call.args else None

        if op.name == "relax.vm.alloc_storage":
            if len(call.args) < 3:
                return None
            shape = call.args[0]
            dtype_arg = call.args[2]
            if not isinstance(shape, relax.ShapeExpr):
                return None
            dims: list[int] = []
            for d in shape.values:
                if isinstance(d, tvm.tir.IntImm):
                    dims.append(int(d.value))
                else:
                    try:
                        simplified = tvm.arith.Analyzer().simplify(d)
                        if isinstance(simplified, tvm.tir.IntImm):
                            dims.append(int(simplified.value))
                        else:
                            return None
                    except Exception:
                        return None

            dtype = str(getattr(dtype_arg, "value", "")) if hasattr(dtype_arg, "value") else str(dtype_arg)
            nbytes = _dtype_nbytes(dtype)
            if nbytes is None:
                return None
            prod = 1
            for v in dims:
                prod *= int(v)
            return int(prod) * int(nbytes)

        return None

    def visit(expr) -> None:
        nonlocal alloc_storage, alloc_tensor, total_bytes, max_bytes, nonconst_bytes
        if not isinstance(expr, relax.Call):
            return
        op = expr.op
        if not isinstance(op, tvm.ir.Op):
            return
        if op.name in ("relax.memory.alloc_storage", "relax.vm.alloc_storage"):
            alloc_storage += 1
            size_bytes = _extract_alloc_storage_bytes(expr)
            if size_bytes is None:
                nonconst_bytes += 1
            else:
                total_bytes += int(size_bytes)
                max_bytes = max(max_bytes, int(size_bytes))
            return
        if op.name in ("relax.memory.alloc_tensor", "relax.vm.alloc_tensor", "relax.builtin.alloc_tensor"):
            alloc_tensor += 1

    for _, func in ir_mod.functions.items():
        if isinstance(func, relax.Function):
            relax.analysis.post_order_visit(func.body, visit)

    return {
        "alloc_storage": int(alloc_storage),
        "alloc_tensor": int(alloc_tensor),
        "alloc_storage_total_bytes": int(total_bytes),
        "alloc_storage_max_bytes": int(max_bytes),
        "alloc_storage_nonconst_bytes": int(nonconst_bytes),
    }
