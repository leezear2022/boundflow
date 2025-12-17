# Phase 4A 之后的下一步规划（建议路线）

当前状态：我们已经有了一个可靠的 **ground truth 路径**（Interval IBP），并且它在 **Task pipeline** 上已对齐 auto_LiRPA（MLP + MNIST 风格 CNN），同时补齐了 `permute/transpose` 的真实语义。这意味着后续可以把工作重心从“把代码跑通”转向“让它可优化、可编译、可扩展”。

另外，`docs/stage_4_critical_review.md` 里提到的三个风险点（Memory 抽象、Relax orchestration 风险、transpose/layout 优化）是有参考意义的。下面的规划把这些点显式纳入：**Phase 4B 要“加码 memory/schema”，Phase 4C 要“先拿到 kernel 粒度收益、别被控制流绑架”，同时把 layout/transforms 作为 planner 的一等公民。**

下面给出下一阶段的规划（按收益/依赖排序）。你可以把它理解成 Phase 4B → Phase 4C → Phase 5 的路线。

---

## 目标总览（接下来 2–3 个里程碑）

### M5：Task IR 从“最小可跑”升级为“可优化/可 lower”
把目前散落在 `attrs`/bindings 的信息收敛为稳定 schema，为 fusion/batching/reuse 和 TVM lowering 做准备。

### M6：TVM Backend v0（先做 interval-affine 的 demo）
把 `TaskKind.INTERVAL_IBP` 的关键算子（优先 `linear` 或 `conv2d`）lower 到 Relax/TIR，跑通“同一 task：Python vs TVM 输出一致”。

### M7：进入 LiRPA 的下一层（CROWN backward 的最小闭环）
在保持 IBP ground truth 的同时，引入 `lirpa_linear`（线性界 A,b）并实现最小 backward bound（先线性层 + ReLU），为论文主线的 “BoundFlow workload” 打开空间。

---

## Phase 4B：任务表示与执行接口加固（推荐先做）

### 4B.0 引入 Memory/Storage 抽象（让 Global Planner 名副其实）
现状：`BoundTask.memory_plan` 还是松散 dict，且 Task/Module 里没有“buffer/aliasing”的可表达接口。

建议在 Phase 4B 就把“接口钉住”（先不做复杂优化也可以）：
- 在 `BFTaskModule` 增加 `buffers`/`storage_plan`（例如 `buffer_id -> {dtype, shape, bytes, scope}`）。
- `TaskOp` 的 inputs/outputs 从“value 名”扩展为可选绑定：`value_name -> buffer_id`（或通过 `value_meta` 统一映射）。
- 提供最小 liveness/lifetime 描述（哪怕只是顺序执行下的 begin/end），为后续 aliasing、显存峰值控制预留位置。

**验收**：不改语义前提下，把“分配与复用”显式化；reference executor 仍可用“每 value 一个 buffer”的默认策略跑通全部对齐测试。

### 4B.1 把 TaskOp 与 Value meta 对齐（消除“shape 塞 attrs”）
现状：`reshape` 的目标 shape 目前由 planner 从 `Value.type.shape` 塞进 `TaskOp.attrs["shape"]`。

建议：
- 在 `BFTaskModule` 中引入 `value_table`（value_name -> TensorType/ValueKind），让 executor 不必依赖“shape 在 attrs 的约定”。
- 或者在 `BoundTask` 中增加 `value_meta` 字段（局部 meta），避免模块级污染。

**验收**：删掉 `attrs["shape"]` 这类 hack 后，MLP/CNN task pipeline 仍能通过全部对齐测试。

### 4B.2 统一 Spec/Property 表示（为 margin/C 矩阵留接口）
现状：runtime 只有 `LinfInputSpec(center, eps)`，还没有 “property” 的表达。

建议把 spec 分成两部分：
- InputConstraints：例如 `Linf(center, eps)`、或将来支持 `x_L/x_U` box。
- Property：例如 “输出 neuron bounds”、或 margin 形式（`C` 矩阵）。

**验收**：能在 task pipeline 上计算 “margin bound”（即指定一个 `C` 乘到最后输出），并与 auto_LiRPA 的 `compute_bounds(C=...)` 对齐。

### 4B.3 把 layout/transpose 变成可优化对象（而不是固定成独立 kernel）
Phase 4A 已经让 `transpose` 在 reference executor 中“正确执行”，这是必要的。但要避免在后端把它固化成“单独 transpose kernel”。

建议：
- 在 `TensorType.layout`（已存在）与 TaskOp.attrs 中引入最小的 layout hint（例如 `layout_in/layout_out` 或 `is_layout_transform=True`）。
- Planner 增加一个“layout analysis / transpose elimination”占位 pass：先做简单规则（能 fuse 就 fuse、能下推就下推），不追求完整最优。
- 后端 lowering 时允许把 transpose 融合到 affine 的 indexing 中（比如把 permute 视作 index map，而不是 materialize）。

**验收**：即使暂时不做优化，也要让 IR 能表达“这是 layout transform”，避免未来返工。

### 4B.3 扩充 primitive 覆盖到“常见 CNN 子集”
建议优先顺序：
- `sub`（经常用于 loss/margin）
- `max_pool2d`（如果要对齐更多 vision 模型）
- `batch_norm`（可先在 normalize 里折叠到 affine，或做 inference-only 的等价变换）

**验收**：至少 1 个带 BN/Pool 的 toy CNN 能跑通 task pipeline（先 IBP）。

---

## Phase 4C：TVM Backend v0（把正确性路径接到可编译）

### 4C.1 选择一个最小 lowering 切入点
建议从下面两者二选一（先简单、再扩大）：
- `linear` 的 interval affine（矩阵乘）lower 到 TIR
- 或 `conv2d` 的 interval affine lower 到 TIR（更接近 simple_verification，但实现更复杂）

### 4C.2 用 Relax 做 orchestration
这里需要更谨慎：TVM 的强项是 tensor compute，而不是复杂控制流（尤其是未来 BaB 的树搜索/队列/剪枝）。

建议分两步走：
- **v0（优先）**：Python driver 负责 orchestration，直接调用 TVM 编译出的 `PackedFunc`/module（先拿到“kernel 粒度变大”的收益）。
- **v1（可选）**：当执行流稳定、并且确实需要减少 Python↔TVM 往返时，再把“纯张量流水线”的部分迁移到 Relax（`call_tir`），避免把 BaB 这种控制流强逻辑硬塞进 Relax VM。

同时保留 `PythonTaskExecutor` 作为 ground truth。

**验收**：新增 `TVMExecutor`（或 `TVMTaskExecutor`）后，同一个 `BFTaskModule`：
- Python 路径输出 == TVM 路径输出（允许浮点误差）
- 跑通 1 个 MLP 用例的对齐测试

---

## Phase 5：最小 CROWN（lirpa_linear）闭环（进入论文主线）

### 5.1 新增 LinearLiRPA DomainState
- 定义 `LinearState`（例如 lA/uA + lbias/ubias，或更紧凑的表示）
- 实现线性层的 backward bound + ReLU relaxation（最小版）

### 5.2 Planner/Task 扩展为 “lirpa_linear” kind
- 新增 `TaskKind.LIRPA_LINEAR`（或 `CROWN_BACKWARD`）
- 仍然保留 interval 作为 intermediate bounds / fallback

**验收**：对一个小 MLP，`CROWN`（backward）输出在数值上对齐 auto_LiRPA 的 `method="CROWN"`（或至少在 tightness 上不比 IBP 差）。

---

## 我建议你现在的选择题（决定下一步具体做哪块）

1) **先做 4B.2（Spec/Property + margin/C 矩阵）**：能马上把“验证问题”表达得更完整，也方便后续 CROWN 与 BaB。
2) **先做 4C（TVM lowering demo）**：尽早让系统论文的“编译加速”主线跑起来，但需要更多工程投入。

如果你选 1)，我会先加一个 `C` 矩阵的 task-op（或在 task/module 里表达 property），并新增与 auto_LiRPA margin bound 的对齐测试。
如果你选 2)，我会先实现 `TVMExecutor` 的 `linear` interval kernel（最小可编译），再逐步扩到 conv2d。
