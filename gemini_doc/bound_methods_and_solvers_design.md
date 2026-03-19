# BoundFlow 设计文档：统一支持 IBP / CROWN / IBP-CROWN / αβ-CROWN / BaB

## 1. 这份文档解决什么问题

当我们把目标从 “只跑 interval IBP（Phase 5）” 扩展到更强的 LiRPA 家族（CROWN/IBP-CROWN/αβ-CROWN）以及 complete verification（BaB）时，会同时引入两类复杂度：

1. **边界算法复杂度**：forward（IBP）+ backward（CROWN/DeepPoly）+ 组合（IBP-CROWN）+ 可优化松弛参数（αβ-CROWN）。
2. **系统复杂度**：多 spec / 多 BaB 节点的重复计算、缓存复用、batching、以及“控制流 vs 张量计算”的拆分（见 `docs/stage_4_critical_review.md`）。

本文给出一个“可落地且 reviewer-proof”的设计：把系统拆成三条正交轴，并把每条轴映射到 BoundFlow 现有的 IR/Planner/Runtime/TVM 后端。

---

## 2. 三轴解耦（Perturbation × BoundMethod/Domain × Solver）

> 目标：新增一种扰动 or 新增一种方法时，不要牵一发动全身。

### 2.1 轴 A：`PerturbationSet`（输入集合）

输入扰动（L∞/L2/L1/L0/patch/…）是“几何集合”，它应该提供：

- `concretize(A, x0) -> (lb, ub)`：对任意线性形式 `A x` 给出上下界（等价于 support function/对偶范数；详见 `gemini_doc/perturbation_support_design.md`）。
  - 重要实现备注：这里的 `A` 不应被理解为“必须显式 materialize 的稠密矩阵”。为了避免 CNN/大模型的内存爆炸，`A` 可以是：
    - 显式 tensor（MVP/小模型）
    - 结构化/稀疏表示
    - `LinearOperator`（只暴露必要的 `matvec/rmatvec` 或 `reduce_sum_abs/topk_abs` 等归约接口）
  - PerturbationSet 的职责是“对输入集合上的线性形式做数值化”，而不是要求后端必须生成显式 `A`。
- `split(strategy) -> (S_left, S_right)`：BaB/分支用，把一个集合切成两个子集合（可选，若只做 incomplete verification 可先不实现）。

重要：**Perturbation 不应该进入 Primal IR**；它属于 verification spec/running context。最小落点是把 runtime 的 `LinfInputSpec` 泛化（见 `boundflow/runtime/task_executor.py`、`boundflow/runtime/scheduler.py`），并在 bench/JSONL 里记录其 jsonable 描述以保证复现口径。

### 2.2 轴 B：`BoundMethod` / `DomainState`（边界传播语义）

方法族决定“状态长什么样”和“怎么传播”：

- IBP：状态 `IntervalState(lb, ub)`（现有 `boundflow/domains/interval.py`），主要是 forward。
- CROWN/DeepPoly：需要 `LinearBoundState`（例如对目标输出维护 `A,b` 并在输入处 concretize），主要是 backward。
- IBP-CROWN：一个组合 pipeline：先 IBP 得到中间 bounds，再用这些 bounds 固化 ReLU 松弛，跑 backward 收紧。
- αβ-CROWN：在 CROWN 基础上引入“可优化的松弛参数”（α、β），需要一个 `RelaxationParamStore` + 优化循环（通常是多步迭代，且可复用 warm-start）。

方法族应该以 **可组合阶段** 表达，而不是写成互相复制的“巨型函数”：

- `ForwardStage`（可选）：产生/更新每个 value 的 `(lb,ub)`（IBP）
- `RelaxationStage`（可选）：基于 interval/constraint 生成松弛参数（ReLU slope 等）
- `BackwardStage`（可选）：对目标输出反向传播线性界（CROWN）
- `OptimizeStage`（可选）：对松弛参数做迭代优化（αβ-CROWN）

### 2.3 轴 C：`Solver`（上层搜索/调度：BaB）

BaB 是控制流 workload：维护优先队列、分支、剪枝、终止条件。它不应该侵入后端编译器。

Solver 的唯一职责是：

- 维护 `Subproblem`（当前节点附加约束，如 ReLU split / neuron split / spec 变体）
- 调用 `BoundMethod.compute_bounds(program, perturbation, subproblem, objectives)` 得到 bound 与可剪枝信息
- 决定 split 策略并生成子节点
- 管理缓存与 batching（把多个子节点打包成 batch 交给 BoundMethod/Executor）

**重要工程原则**：控制流留在 Python runtime（至少 Phase 6 初期），GPU/TVM 负责张量核（见 `docs/stage_4_critical_review.md` 与 `gemini_doc/tvm_backend_optimization_memo.md`）。

---

## 3. 映射到 BoundFlow 现有分层（不返工的落点）

### 3.1 保持 Primal IR 不变

- 仍以 `BFPrimalProgram/BFPrimalGraph` 表示模型语义（`boundflow/ir/primal.py`）。
- 扰动与目标（spec）不放进 primal graph，避免把 verification 语义污染成“模型语义的一部分”。

### 3.2 Task IR 承载“边界程序”，但 TaskKind 不要爆炸

现状：`TaskKind` 仅有 `INTERVAL_IBP`（`boundflow/ir/task.py`）。

建议演进：

- `TaskKind` 表示“这类 task 产生什么 DomainState”而不是“算法名字”：
  - `INTERVAL_FWD`（或保留 `INTERVAL_IBP`）
  - `LINEAR_BWD`（CROWN/DeepPoly backward）
  - `RELAX_OPT`（αβ 优化/更新 α 的 stage，可能是一个 loop 的 body）

同时用 `TaskOp.op_type` 表达具体算子语义（linear/conv2d/relu/…）。

好处：IBP-CROWN/αβ-CROWN 可以被表达为多个 task/stage 的组合，而不是新增 `TaskKind.IBP_CROWN/ALPHA_BETA_CROWN` 导致组合爆炸。

### 3.3 Planner 的角色：把“方法 pipeline”变成 TaskGraph + StoragePlan

Planner 输入：

- `BFPrimalProgram`（模型）
- `MethodConfig`（IBP/CROWN/IBP-CROWN/αβ-CROWN 的选择与超参）
- `PerturbationSet` 描述（或其 hash/ID，用于 cache key）
- `Objective/Spec`（例如 LinearSpec(C) 或 margin）

Planner 输出：

- `BFTaskModule`（多个 task）
- `TaskGraph`（跨 task 依赖）
- `StoragePlan`（buffer reuse/alias）
- `CachePlan`（可选：哪些中间 state 可跨节点复用、复用范围）

现有 `PlanBundle` 已经能承载 `task_graph/storage_plan/cache_plan`（`boundflow/planner/core.py`），只需要把 Phase 6 的内容扩展进来。

### 3.4 Runtime/Scheduler：为 batching/caching 留接口

现状的 scheduler（`boundflow/runtime/scheduler.py`）是 “单 spec、单输入、串行 topo”。

Phase 6 的 scheduler 要支持两个正交扩展：

1. **multi-spec batch**：同一模型、不同 `C` / 不同 objective；这一维度非常适合 GPU batch。
2. **multi-subproblem batch（BaB）**：同一模型、不同 split constraint；这维度也能批处理，但约束会让数据结构更复杂（例如每个节点的 ReLU mask 不同）。

建议：先把 batch 作为运行时维度引入（不强迫 Planner 立即改 TaskOp），例如：

- `Executor.run(task_module, inputs=batch, spec=batch, subproblem=batch)` 的统一入口
- scheduler 内部把 task 的每个 op 变成“对 batch 维度的张量算子”

---

## 4. 方法族的“最小可行闭环”（推荐实现顺序）

这部分是“工程路线图”，尽量避免一上来就实现完整 αβ-CROWN + BaB。

### 4.1 Step 1：IBP（现有）+ PerturbationSet 泛化

- 泛化输入：`LinfInputSpec → InputSpec(perturbation)`（设计见 `gemini_doc/perturbation_support_design.md`）。
- 让第一层 affine 能区分 L∞/L2/L1/L0（至少 dense linear；conv2d 可先降级）。

收益：新增扰动主要改 `PerturbationSet`，主线不动。

### 4.2 Step 2：最小 CROWN-IBP（先 tight，再完整）

先做 CROWN-IBP 而不是纯 CROWN：

- forward：IBP 给每个 ReLU 的 pre-activation interval，决定松弛线性上/下界（斜率/截距）
- backward：对最终输出（或 `LinearSpec(C)`）反向传播线性界（最小支持 linear + relu）
- 输入处：调用 `PerturbationSet.concretize(A, x0)` 收敛为数值 bound

这一步能在 `MLP + 小 CNN` 上对齐 auto_LiRPA 的 `method="CROWN"` 或 “IBP+backward” 组合（参考 `docs/p4_p5.md` 的建议）。

### 4.3 Step 3：αβ-CROWN（优化循环）——作为“可选 stage”

把 αβ 看成对松弛参数的迭代优化：

- `RelaxationParamStore`：可用 per-layer 的 α 张量存储（按 spec/batch/subproblem 维度）
- `OptimizeStage`：执行 K 步优化（梯度/启发式更新），每步都调用一次 backward（或其增量版本）

关键系统点：**warm-start 与复用**。

- 同一输入/扰动、不同 `C`：可复用 α 初值或共用部分计算
- BaB 子节点之间：子节点继承父节点 α（或部分继承），显著减少优化步数

这部分是系统论文最值钱的点之一：把 auto_LiRPA 的 `reuse_alpha/cache_bounds` 升级为 Planner/Runtime 的 cache 机制（见 `docs/p4_p5.md`）。

### 4.4 Step 4：BaB（complete verification）作为上层 workload

BaB 的核心是高效重复算 bounds，因此在 BoundFlow 中应当：

- BaB runtime（Python）只负责队列与分支
- 每次扩展/评估节点，调用 BoundMethod pipeline（IBP-CROWN 或 αβ-CROWN）
- 通过 cache/batching 降低重复计算

特别注意：

- 不要试图把 BaB 的 while/priority queue lower 到 TVM/Relax VM（控制流复杂、收益小，见 `docs/stage_4_critical_review.md`）。
- TVM 负责在每次 bound 计算里跑重的张量核（GEMM/conv/transpose/fusion）。

---

## 5. Cache / Reuse / Batching：系统层的“统一抽象”

### 5.1 Cache Key 的三部分

把 cache key 拆为三段，避免“缓存命中但语义不对”：

1. **ProgramKey**：模型结构与参数版本（例如 git_commit + program hash）
2. **MethodKey**：方法族与超参（IBP/CROWN/αβ + 迭代步数/松弛策略）
3. **InstanceKey**：输入中心 `x0` 的摘要 + 扰动集合参数 + subproblem/split state + spec/objective

新增扰动主要影响 `InstanceKey` 的 “扰动参数” 部分；不应导致 ProgramKey/MethodKey 变化。

### 5.2 可复用的中间状态（按方法族分类）

- IBP：每层的 `(lb,ub)` 可作为 forward cache
- CROWN-IBP：forward 的 `(lb,ub)` 可复用；backward 的 `A` 通常与 spec 强相关（多 spec 场景下适合 batch，而不是 cache）
- αβ-CROWN：α 参数与中间 bounds 可 warm-start；这是最关键的跨迭代、跨节点复用对象

### 5.3 Batching 的优先级建议

先做 “多 spec batch”，再做 “多 BaB 节点 batch”：

1. 多 spec（同一输入扰动，多个 `C`）：实现简单、收益大、对工程侵入小
2. 多 BaB 节点：约束不同导致 mask/参数不同，批处理需要更多结构设计，但对 complete verification 的吞吐关键

---

## 6. “新增扰动 + 新增边界方法”到底麻烦不麻烦？

### 6.1 `concretize` 的实现模式（Interval vs Linear bounds）

很多讨论会把 `concretize` 写成 “Domain 的一个方法”。在 BoundFlow 的三轴解耦里，更推荐把它拆清楚：

- **Domain/BoundMethod** 负责产生“抽象状态”（`DomainState`）以及传播规则（forward/backward）。
- **PerturbationSet** 负责把任何线性形式 `A x` 在输入集合上 **数值化（concretize）**：`concretize(A, x0) -> (lb, ub)`。

这样做的好处是：新增扰动只改 `PerturbationSet`；新增方法只改“怎么得到 A/b 或 (lb,ub)”，不会互相牵连。

两种典型模式：

1) **IntervalDomain（IBP）**：状态已经是 `(lb, ub)`，不需要额外 concretize  
   - forward 过程中 `IntervalState(lower, upper)` 就是可直接对比/剪枝的数值 bound。
   - 从这个意义上，Interval 的 “concretize” 等价于“直接返回”：
   ```python
   # state: IntervalState
   lb, ub = state.lower, state.upper
   ```

2) **LinearDomain（CROWN/DeepPoly/αβ-CROWN 的 backward）**：状态是线性形式，需要在输入处 concretize  
   - 常见表示：对目标 `y` 的界写成 `y <= A x + b`（以及下界类似）。
   - 在输入处用扰动集合把 `A x` 变成数值界：
   ```python
   # A: [..., out_dim, in_dim], b: [..., out_dim]
   # x = x0 + δ, δ ∈ PerturbationSet
   center = A @ x0 + b
   lb_ax, ub_ax = perturbation.concretize(A, x0)  # 或者 concretize(A, x0) 直接返回整体 lb/ub
   # 若 concretize 返回的是 (lb_ax, ub_ax) for A@x，则：
   lb, ub = lb_ax + b, ub_ax + b
   ```
   这里的关键是：**concretize 的核心逻辑属于扰动集合（对偶范数/top-k/support function）**，详见 `gemini_doc/perturbation_support_design.md`。

> 注：实现细节上你也可以让 `PerturbationSet` 提供更细粒度的 API（例如只返回 deviation），但建议对外统一成 `concretize(A, x0)`，避免不同 domain 各自发明口径。

### 6.2 新增扰动（L∞ → L2/L1/L0/patch）

如果按本文三轴设计：

- 主要工作量集中在 `PerturbationSet.concretize/split`，以及把 runtime 输入规格泛化；
- IBP/CROWN/αβ-CROWN 的主体代码不应该改；
- 新扰动的正确性主要通过 “输入处 concretize 的单元测试 + 与 auto_LiRPA 对齐测试” 保障。

因此：**新增扰动不应是“大手术”**，只要你把“扰动集合”从 `LinfInputSpec` 中解耦出来。

### 6.3 新增边界方法（例如从 IBP 到 CROWN）

这会更重，因为要引入新的 `DomainState` 与传播方向（backward），并扩展算子覆盖；但仍然应该是“增加一个 stage/一种 task kind”，而不是推翻现有 pipeline。

经验顺序：

- 先把 CROWN-IBP 跑通（依赖 IBP 的中间 interval）
- 再做纯 CROWN / αβ-CROWN 的优化循环
- 最后把 BaB 作为上层 driver 接入（不要反过来先写 BaB）

---

## 7. 落地避坑清单（建议作为 Phase 6 的接口约束/DoD）

这一节把实现时最容易“撞墙”的点提前固化成接口约束，避免后续 PR 反复返工。

### 7.1 `A` 的表示：优先支持 `LinearOperator`，不要强迫显式矩阵

在 backward LiRPA（CROWN/DeepPoly/αβ）里，`A` 的逻辑形状可能极大；显式存储会直接把系统推进到“显存/内存爆炸”的死胡同。

- 建议：把 `PerturbationSet.concretize` 的输入从“必须是 tensor”扩展为“线性形式句柄”（`LinearOperator` 或结构化表示）。
- PerturbationSet 只依赖必要的归约（例如 `sum(abs(A))`、`norm(A,2)`、`max(abs(A))`、`topk(abs(A),k)`），而不是依赖完整 materialize。

### 7.2 Task 的 contract：明确 produces/consumes/batch-axes

`TaskKind` 不爆炸是对的，但 runtime 泛化需要更硬的 contract。建议每个 task 明确三件事：

1. `produces`：产出何种 state（Interval/LinearBound/RelaxParamDelta/…）
2. `consumes`：依赖哪些前置 state（例如 LINEAR_BWD 依赖某些 layer 的 pre-ReLU interval 来决定松弛）
3. `batch_axes`：允许在哪些维度 batch（spec/subproblem/both/none）

仓库已有的 `BoundTask.batch_axes`/`memory_plan` 槽位可作为承载点（见 `boundflow/ir/task.py`）。

### 7.3 αβ/优化阶段：必须是可 warm-start 的 state machine

αβ-CROWN 的系统效率高度依赖 warm-start 与 reuse（跨 spec、跨 BaB 节点继承）。

- `RelaxationParamStore` 需要明确“逻辑索引维度”（至少能表达 spec 与 subproblem 维度；层/神经元维度可按需结构化存储）。
- 继承规则要明确：child 从 parent 继承 α/β（可局部 reset），并且该优化状态必须进入 cache key 的 InstanceKey，避免 silent 错误复用。

### 7.4 Subproblem/约束表示：不要用松散 dict，给 β-CROWN 风格留槽

BaB 的 split 约束最终需要被 LINEAR_BWD/RELAX_OPT “看懂并利用”。建议把 subproblem 约束建模为明确的数据结构：

- 例如每层一个 ReLU mask（active/inactive/unknown）+ 可选的额外线性约束集合。
- 这样 backward 与优化阶段才能基于约束收紧 bounds，而不是把约束留在 solver 里“只有控制流知道”。

### 7.5 CachePlan 的粒度：规定缓存的是哪一层/哪一种 stage 的 state

三段式 cache key 解决了“命中语义”，但还需要 CachePlan 决定“缓存什么”：

- value-level vs layer-level vs stage-level（IBP 中间 interval、pre-ReLU interval、α 参数、某些 backward 的中间量）
- 哪些 state 适合 cache、哪些更适合 batch（例如很多 backward 的 `A` 对 spec 强相关，更适合 batch 维度而非 cache）

### 7.6 正确性 DoD：逐阶段对齐 auto_LiRPA/参考实现

为了 reviewer-proof，建议把“对齐测试”写成每个 MVP 的 DoD：

- IBP：对齐 Python reference（已有）+ auto_LiRPA baseline（已有路径/基础设施）
- CROWN-IBP：至少在 MLP 上对齐 auto_LiRPA 的 CROWN/backward 语义（固定 seed/eps/spec）
- α/αβ：对齐 tighter-than-IBP 且与 auto_LiRPA/αβ-CROWN 参考实现一致（在可比设置下）
- BaB：先保证最终验证结论一致（complete/incomplete 的声明要明确），再谈性能

---

## 7. 与 TVM 后端的关系（现实约束与建议）

TVM 的强项是张量核与内存规划，BoundFlow 的强项是 verification-aware 的程序生成与全局规划。

实操建议：

- 让 TVM 后端先覆盖最重的核：`A@W`、`A@W_pos + A@W_neg`、conv 相关的线性传播（见 `gemini_doc/tvm_backend_optimization_memo.md`）。
- 保持 orchestration（尤其 BaB 控制流）在 Python；把 batch 维度做大，让每次 kernel 够“粗粒度”。

---

## 8. 一个可引用的“系统图”（论文/答辩用）

```mermaid
flowchart TD
  P[Primal Program<br/>BFPrimalProgram] --> PL[Planner<br/>method pipeline → tasks]
  S[Spec/Objectives<br/>LinearSpec(C), ...] --> PL
  R[PerturbationSet<br/>concretize/split] --> PL
  PL --> M[BFTaskModule<br/>TaskGraph + StoragePlan]
  M --> SCH[Scheduler/Runtime<br/>batching + cache]
  SCH --> EXE[Executor<br/>Python reference / TVM kernels]
  EXE --> BND[Bounds / Proof status]
  SOL[Solver (BaB)] -->|subproblem constraints| SCH
  SOL -->|priority queue/control flow| SOL
```
