# Phase 4 及其后续计划的批判性审查 (Critical Review)

你刚才让我“别总夸”，这是一个非常好的信号。作为协作方，我也需要指出当前计划中可能存在的**隐患**和**盲点**。虽然路线大方向是对的，但在“落地细节”上，我认为有三个地方如果处理不好，可能会在 Phase 4B/4C 成为大坑。

## 1. “Global Planner” 有点名不副实：缺失 Memory/Storage 抽象

我们的愿景是做 Global Planner 来解决“显存峰值”和“复用”问题。但目前的 Phase 4 设计（包括 Task IR Schema）主要还是关注 **“Compute” (OpType, Attrs, Inputs/Outputs)**，几乎 completely ignored **“Memory”**。

*   **隐患**: 如果 `BoundTask` 里只有 value names，Planner 怎么表达“`state_A` 和 `state_B` 复用同一块显存”？怎么表达“这个 Task 需要 pre-allocate 1GB 临时空间”？
*   **后果**: 等到了 Phase 4C 接 TVM 时，你会发现生成的 Kernel 拼命 `cudaMalloc/Free`，或者显存爆炸，因为 IR 层没有传递 Storage Plan。
*   **建议**: 在 **Phase 4B** 中，除了清洗 attrs，必须引入 **`Allocation` / `BufferID`** 的概念。
    *   在 `BFTaskModule` 里增加 `memory_plan` 字段（例如：`buffer_map: {buf_0: size, buf_1: size}`）。
    *   在 `BoundTask` 的输入输出中，除了 value name，允许绑定到具体的 `buffer_id`。这样 Planner 才能真正做 "Memory-aware scheduling"。

## 2. “Relax Orchestration” 的复杂度假象 (Over-engineering Risk)

Phase 4C 提议用 TVM Relax 做 orchestration（编排）。这听起来很正统，但在工程上可能是个**陷阱**。

*   **隐患**: Relax VM 并不像 Python 那么灵活。如果你试图把复杂的 BaB 逻辑（树搜索、回溯、优先队列）都 lower 到 Relax 里，你会陷入无尽的调试泥潭。TVM 的优势在于 **Tensor Compute**，而不是 **Control Flow**。
*   **后果**: 可能会花 2周时间写 Relax Script 只是为了实现一个简单的“While 循环 + 剪枝”，结果性能提升微乎其微（因为瓶颈在 GPU Kernel 而不是调度循环）。
*   **建议**: **Keep Orchestration in Python (for now)**。
    *   v0 版后端：Python `TaskExecutor` 直接调用 TVM 编译好的 `PackedFunc` (TIR Kernels)。
    *   不要急着上 Relax VM，除非你需要把整个 Runtime 部署到没有 Python 的环境（如 C++ 嵌入式）。对于 ASPLOS 论文，PyTorch 层的 Python 调度开销是可以接受的，重点是 **Kernel 粒度** 变大了。

## 3. Permute 只是“执行”了，没有“优化”

Phase 4A 提议在 Runtime 里真实执行 `transpose`。这对正确性是必须的，但在优化上是**倒退**的。

*   **隐患**: 在 GPU 上做独立的 `transpose` kernel 是带宽杀手。TVM 的强项是 **Layout Propagation**（例如把 NCHW 的 Conv 和 NCHW 的 input 自动变成 NCHWc 或者 NHWC 以适应 TensorCore）。
*   **后果**: 如果我们在 Task 层把 `transpose` 固化成一个 Op，TVM 后端可能就会老老实实生成一个 transpose kernel，切断了算子融合的机会。
*   **建议**: 在 Planner 层增加 **Layout Analysis**。
    *   如果整个子图都是 Elementwise/Affine，Planner 应该尝试消除 transpose，或者把 transpose 融合进 MatMul/Conv 的 indexing 中。
    *   虽然 v0.1 不一定做，但 IR 上要预留 **Format/Layout** 的字段，不要让 `transpose` 仅仅是一个“动词”。

## 总结调整建议

1.  **Phase 4B 加码**: 不只要修 schema，还要加上 **Memory Plan 接口** (Buffer aliasing)。
2.  **Phase 4C 减负**: 暂缓 Relax VM 的全量编排，退一步用 **Python Driver + TVM Managed Kernels**，先拿到端到端收益再说。
3.  **Phase 4A 留白**: 既然做了 permute，就要意识到这是未来优化的重点对象，建议在 TaskOp 上留 `layout_transform` 相关的 hint 字段。

这三点如果能考虑到，后面的路会少很多“返工”。
