# 评审备忘（无外链版）：三轴解耦 + Stage Pipeline 的工程落地建议

本文是对 `gemini_doc/bound_methods_and_solvers_design.md` 与 `gemini_doc/perturbation_support_design.md` 的一次“工程落地 + 避坑”评审汇总，目标是把容易在实现阶段撞墙的点提前固化为接口约束与落地顺序。

相关仓库内证据/背景：

- 控制流 vs 张量核的拆分原则：`docs/stage_4_critical_review.md`
- Phase 6 的方法族引入建议与优先级：`docs/p4_p5.md`
- Phase 5 的产线/对齐基础设施（JSONL schema、baseline gate）：`docs/bench_jsonl_schema.md`

---

## 1) 总体评价

“`Perturbation × BoundMethod/DomainState × Solver`（三轴解耦）+ stage pipeline 组合”的方案方向是对的：

- 它把社区成熟的方法族（IBP/CROWN/IBP-CROWN/αβ 以及 BaB）放进清晰的系统分层中，系统贡献集中在 **planner/runtime/cache/batching/codegen**，而不是重新发明算法。
- 文档按现在的结构推进是 reviewer-proof 的：每一层都能解释“职责边界”和“为什么这么分层”，并能落回仓库里可复现/可对齐的证据链。

---

## 2) 方案最强的 4 个点

### 2.1 三轴解耦：职责边界清晰

- `PerturbationSet` 管集合几何：support function / concretize / split
- `BoundMethod/DomainState` 管传播语义：forward（IBP）、backward（CROWN/DeepPoly）、组合（IBP-CROWN）、优化（αβ）
- `Solver` 管 BaB 控制流：队列、分支、剪枝、终止条件

这能直接避免“新增扰动/新增方法牵一发动全身”的工程灾难。

### 2.2 方法族表达成 stage 组合，而不是巨型继承树

把 IBP-CROWN/αβ-CROWN 写成：

- `ForwardStage`（可选）
- `RelaxationStage`（可选）
- `BackwardStage`（可选）
- `OptimizeStage`（可选）

比“HybridDomain 继承”更贴近真实执行流，也更利于插入缓存/批处理/后端 lowering。

### 2.3 明确：BaB 控制流留在 Python，TVM 专注张量核

这点非常务实：BaB 的 while/priority queue 是不规则控制流，把它 lower 到 TVM/VM 很容易得不偿失；而重计算在张量核侧（GEMM/conv/transpose/融合），应下沉给 TVM（见 `docs/stage_4_critical_review.md`）。

### 2.4 batching 优先级正确：先 multi-spec，再 multi-subproblem

- multi-spec（不同 `C` / objective）通常结构一致、batch 代价低、收益大；
- multi-subproblem（BaB 节点）引入 split mask/约束差异，结构复杂但对 complete verification 关键。

---

## 3) 落地时最容易撞墙的 6 个点（建议写成 DoD/接口约束）

这些点已被整理进 `gemini_doc/bound_methods_and_solvers_design.md` 的落地避坑清单（见 `gemini_doc/bound_methods_and_solvers_design.md` 的 §7），这里用更直接的“评审口吻”再强调一次：

### 3.1 `concretize(A, x0)` 的 `A` 不要默认“显式稠密矩阵”

在 CROWN/backward LiRPA（尤其 CNN）中，`A` 的逻辑形状巨大；如果接口强迫 materialize `A`，会直接导致内存爆炸。建议：

- 允许 `A` 以 `LinearOperator`/结构化形式出现，只暴露必要归约（sum-abs、norm2、max-abs、topk-abs 等）。
- 这样 `PerturbationSet` 仍只管几何，不会把后端逼成“必须吐出显式 A”。

### 3.2 TaskKind 不爆炸，但必须有 task contract

仅有 `INTERVAL_FWD/LINEAR_BWD/RELAX_OPT` 不够，落地时 executor/scheduler 需要知道：

- `produces`（产出 state 类型）
- `consumes`（依赖哪些前置 state，例如 LINEAR_BWD 依赖 pre-ReLU interval）
- `batch_axes`（spec/subproblem/both/none）

### 3.3 αβ 的 OptimizeStage 必须 warm-start，并进入 cache key

αβ 的工程优势来自 reuse/warm-start。必须明确：

- α/β 的逻辑索引维度（至少 spec 与 subproblem）
- child 继承 parent 的规则（以及局部 reset）
- 优化状态进入 InstanceKey，否则会出现 silent 的语义错误复用

### 3.4 Subproblem 约束要能被 backward/optimize “看懂”

不要把 subproblem 约束停留在 solver 的松散 dict；需要可编码的结构（例如每层 ReLU mask + 可选线性约束集合），这样 LINEAR_BWD/RELAX_OPT 才能利用约束收紧 bound。

### 3.5 CachePlan 除了 key，还要规定“缓存什么粒度”

只定义 cache key 不够：还需要决定缓存对象的粒度（value/layer/stage），并区分哪些 state 适合 cache、哪些更适合 batch（例如很多 backward 的 `A` 对 spec 强相关）。

### 3.6 正确性 DoD：逐阶段对齐 reference/baseline

建议把 DoD 写硬：

- IBP：对齐 Python reference + auto_LiRPA baseline gate（Phase 5 已有基础设施）
- CROWN-IBP：至少在 MLP 上对齐 backward/CROWN 语义
- α/αβ：tightness 与对齐测试并行推进
- BaB：先确保结论一致（complete vs incomplete 声明清晰），再谈性能

---

## 4) 推荐落地顺序（收紧版）

1. **PerturbationSet 泛化**：先只做 `L∞ + L2`；conv 暂时保守（sound 外包）
2. **最小 CROWN-IBP**：先支持 `Linear + ReLU`（MLP），把 “backward → 输入 concretize” 链路跑通
3. **multi-spec 真 batch**：作为第一波系统收益点（吞吐/缓存/核融合更好解释）
4. **α 优化（先 α-CROWN）**：OptimizeStage 做成 warm-start state machine
5. **BaB driver（先简单再完整）**：先有一个可用的 BaB runtime（控制流在 Python），再逐步把约束编码成更强形式并做 multi-subproblem batching

