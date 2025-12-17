# 关于 Strategy A 重构计划的反馈与补充建议

我仔细评估了 `strategy_a_refactor_plan.md`，整体上我非常支持 **Strategy A (Fix & Iterate)** 的方向。与其追求完美的全新开局，不如在现有骨架上快速迭代出闭环。

以下是我的一些 **Rebuttal (辩护/修正)** 和 **Supplement (补充建议)**，主要集中在技术实现细节和长期可维护性上。

## 1. Rebuttal & Clarification (修正与澄清)

### 1.1 关于 "IR 语义不足" (Phase 1)
文档指出 *“Primal IR 当前用 `inputs/outputs` 表示 Node 名，应改为 Value 名”*。
**修正建议**：
不仅仅是改名，我建议 **显式分离 `Node` (Operation) 和 `Value` (Data/Edge)** 的数据结构，参考 `torch.fx` 或 `StableHLO` 的设计。
- `Node`: 代表计算 (e.g., `Conv2d`, `ReLU`)，拥有 `attributes` (stride, padding) 和 `signature`。
- `Value`: 代表数据流动的边，拥有 `type`, `shape`, `dtype`。
- **理由**：如果在 IR 层不把 Value 实体化，后续做 Planner (如内存规划、算子融合) 时会很痛苦，因为无法方便地挂载 Value 级别的元数据 (如 Liveness interval, Memory offset)。

### 1.2 关于 "Planner/后端以任务为中心" (Phase 4)
文档建议 *“先做整图一个任务”*。
**补充建议**：
虽然 Version 0 是整图一个任务，但我们的 `BoundTask` 结构设计必须 **一开始就支持 Multi-Task 的槽位**。
- 不要把 `Compiler` 写死成只接受一个 Graph。
- **建议**：`BFTaskModule` 应该是一个包含多个 `BoundTask` 的容器，并且有一个 `Main` 函数描述调度逻辑（即使 v0 只是顺序执行）。这样未来引入 Branching/Loop (for BaB) 时不需要重构顶层架构。

## 2. Technical Supplements (补充建议/技术增强)

### 2.1 引入 "Interpreter" 模式 (Phase 3)
文档提到了 *“Interpreter (reference executor)”* 作为 ground truth。
**建议**：
这个 Interpreter 不应仅仅是一个测试脚本，而应成为 Runtime 的一种 **Fallback Backend**。
- 设计一个统一的 `Executor` 接口，`PythonInterpreter` 和 `TVMExecutor` 都实现这个接口。
- **好处**：在开发新算子或调试复杂数值问题时，用户可以无缝切换到 Python 模式进行断点调试，这对编译器开发至关重要。

### 2.2 强化 "Spec" 的定义 (Phase 1)
文档对 Spec 一笔带过。
**建议**：
`Spec` 应该包含 **Input Constraints** (即 Defines the input region) 和 **Property to Verify** (即 Output Constraints)。
- 哪怕 v0.1 只做 Range Propagation，也应该明确区分这两者。
- 这有助于未来扩展到更有趣的 Property (比如 Robustness verification 只需要 Input Constraints，而 Reachability 可能需要两者)。

### 2.3 TVM 集成策略：拥抱 Relax (Phase 4)
文档提到 *“Relax 作为 orchestration (可选)”*。
**强烈建议**：
**必须使用 Relax** 作为高层胶水。
- 不要试图手动拼接 TIR PrimFunc 调用。
- 用 Relax 表达整个 `BoundTask` 的执行流（分配内存 -> 调 Kernel A -> 调 Kernel B）。
- 这样我们可以利用 TVM 现有的 `Relax.vm` 运行时，大大简化 Runtime 开发工作量（不用自己写 C++ Runtime 来管理 Tensor 内存）。

## 3. 调整后的执行优先级

基于上述建议，我对实施细节做微调：

1.  **Phase 0 (工程)**: 坚决执行（清理包结构 P0）。
2.  **Phase 1 (IR)**: **加重工作量**。引入 `Node/Value` 双层结构，设计好 `Executor` 接口。这是核心资产。
3.  **Phase 2 (Frontend)**: 正常执行。
4.  **Phase 3 (Reference)**: 实现为 `PythonInterpreter` 类，而非散装脚本。
5.  **Phase 4 (TVM)**: 明确使用 TVM Relax 来封装生成的 TIR kernels。

**总结**：计划可行，但在 IR 数据结构设计上建议更“重”一点，以换取未来的扩展性；在 Runtime 上建议更“轻”一点，复用 TVM Relax VM。
