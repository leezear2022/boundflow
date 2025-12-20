import tvm
from tvm import relax, transform

from boundflow.backends.tvm.relax_analysis import collect_relax_memory_stats


def _build_dynamic_alloc_module(*, upper_bound: int | None) -> tvm.IRModule:
    """
    Build a tiny Relax module with a dynamic dimension `n` in the function signature.

    Without `tir_var_upper_bound`, memory planning/lowering typically leaves alloc_storage sizes symbolic.
    With `tir_var_upper_bound={"n": k}`, some alloc_storage sizes can become constant and thus measurable.
    """
    bb = relax.BlockBuilder()
    n = tvm.tir.Var("n", "int64")
    x = relax.Var("x", relax.TensorStructInfo([n, 16], "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            y = relax.op.builtin.alloc_tensor(relax.ShapeExpr([n, 16]), "float32", 0)
            z = relax.op.add(x, y)
            gv = bb.emit_output(z)
        bb.emit_func_output(gv)
    mod = bb.get()

    if upper_bound is not None:
        mod2 = tvm.IRModule(mod.functions)
        main_gv = mod2.get_global_var("main")
        func = mod2[main_gv]
        mod2.update_func(main_gv, func.with_attr("tir_var_upper_bound", {"n": int(upper_bound)}))
        return mod2
    return mod


def test_pr12_2_tir_var_upper_bound_can_make_alloc_storage_bytes_constant():
    from tvm.relax import pipeline as relax_pipeline_mod

    pipe = relax_pipeline_mod.default_build_pipeline()

    with transform.PassContext():
        m0 = pipe(_build_dynamic_alloc_module(upper_bound=None))
        m1 = pipe(_build_dynamic_alloc_module(upper_bound=8))

    s0 = collect_relax_memory_stats(m0)
    s1 = collect_relax_memory_stats(m1)

    # Without upper bound, alloc_storage sizes tend to remain symbolic -> cannot be summed as constant bytes.
    assert int(s0.get("alloc_storage_total_bytes", 0)) == 0

    # With an upper bound, at least some alloc_storage sizes should become constant and measurable.
    assert int(s1.get("alloc_storage_total_bytes", 0)) > 0
    assert int(s1.get("alloc_storage_nonconst_bytes", 0)) < int(s0.get("alloc_storage_nonconst_bytes", 0))
