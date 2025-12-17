from ..ir.primal import BFPrimalGraph

def normalize_primal_graph(graph: BFPrimalGraph) -> BFPrimalGraph:
    """
    Normalize a primal graph to the authorized primitive op-set.
    
    Responsibilities:
    1. Map complex ops to primitives (e.g. Linear -> MatMul + Add).
    2. Eliminate in-place operations.
    3. Ensure canonical attribute format.
    """
    graph.validate()

    # v0.1: 仅做最小规范化，避免把后端/域实现绑死在 FX 的 op 表达上。
    method_map = {
        "view": "reshape",
        "reshape": "reshape",
        "permute": "transpose",
    }
    for node in graph.nodes:
        if node.op_type.startswith("call_method::"):
            method = node.op_type.split("::", 1)[1]
            node.op_type = method_map.get(method, node.op_type)
    return graph
