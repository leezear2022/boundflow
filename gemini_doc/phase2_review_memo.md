# Phase 2 (TorchFrontend) 实现评估备忘录

**日期**: 2025-12-17
**评估对象**: Phase 2 TorchFrontend 实现 (`boundflow/frontends/pytorch/frontend.py`, `normalize.py`, tests)

## 总体评价：Solid Foundation (扎实的基础)

Phase 2 的代码实现质量很高，很好地承接了 Phase 1 的 Node/Value 架构，且并没有过度设计。完工度 100%，完全符合 Strategy A 的预期。

## ✅ 亮点 (Strengths)

1.  **现代化的 `torch.export` 接入**:
    *   使用了 `torch.export.export()` 而非旧的 `torch.jit.trace`，符合 PyTorch 2.0 路线。
    *   **参数分离正确**: 正确处理了 `graph_signature.inputs_to_parameters`，将权重参数剥离到了 `BFPrimalProgram.params` 中，图结构里只留了 `Kind=PARAM` 的占位符，这是后续 Compiler 静态编译的前提。

2.  **数据流 (Value) 处理清晰**:
    *   严谨区分了 `ValueKind.INPUT` / `PARAM` / `CONST` / `INTERMEDIATE`。
    *   从 `fx_node.meta` 中提取了 shape/dtype，保证了 IR 是带 Type 的静态图结构。

3.  **分层清晰的 Normalization**:
    *   Frontend 只负责“原样映射”（如 `call_method::view`），算子归一化留给 `normalize.py`（转为 `reshape`）。这种 **Raw Import -> Canonicalize** 的策略利于维护。

4.  **可测试性**:
    *   `test_torch_frontend_import.py` 显式调用了 `graph.validate()`，证明生成的 IR 结构合法（无悬空边、重名）。

## ⚠️ 潜在改进点 (非 Blocker)

1.  **DType 解析**: 目前使用字符串替换 (`str(dtype).replace("torch.", "")`)，建议未来引入 `DataType` Enum 做严格映射。
2.  **Op Mapping**: 目前 `_map_torch_target_to_primitive` 是硬编码字典。后续支持更多算子时需要更系统的 Dispatcher。

## 结论与下一步

当前 IR 已经具备接入真实模型的能力。
**Next Step**: 进入 Phase 3，实现 `IntervalState` 和 `Reference Executor`，进行数值验证。
