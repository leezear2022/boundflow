# BoundFlow 设计文档：支持 L∞/L2/L1/L0 输入扰动（线性/卷积算子的通用处理）

## 1. 背景与问题

当前 BoundFlow 的 Phase 4/5 主路径以 **interval IBP** 为核心抽象：

- 输入扰动由 `LinfInputSpec(center, eps)` 表示，并直接初始化为逐元素区间 `x ∈ [x0-eps, x0+eps]`（见 `boundflow/runtime/task_executor.py`）。
- 后续所有 affine/conv2d 传播都在 `IntervalState(lower, upper)` 上进行（见 `boundflow/domains/interval.py`）。

该设计天然对应 `L∞`（或更准确：逐元素独立的 box）扰动集合；但对于 `L2/L1/L0` 等约束集合，**线性算子（尤其第一层 affine）** 的最紧传播公式并不是简单的 `x0±eps` box。

因此需要一个设计：在不破坏现有 “Primal IR → Planner → Task → Scheduler → Executor” 主线的前提下，让 BoundFlow 能表达并正确处理多种输入扰动集合，并为后续更强域（DeepPoly/CROWN）留接口。

---

## 2. 设计目标（Goals / Non-goals）

### Goals

1. **在接口层面支持多种输入扰动集合**：`L∞/L2/L1/L0`（至少输入 perturbation）。
2. **在 interval IBP 框架内做到“第一层 affine 的正确公式”**：对 `linear/conv2d` 给出 sound 的输出区间，且能区分不同范数（不是简单降级成 L∞）。
3. **保持可扩展性**：后续引入 CROWN/DeepPoly 时，扰动集合仍可复用（通过 support function / concretize 接口）。
4. **保持工程可控**：优先改 runtime/spec 抽象，不强行重构 IR；渐进式让 TVM lowering 复用同一抽象。

### Non-goals（本设计不强求）

- 不要求 interval 域在多层传播中对 `L2/L1/L0` 保持“几何最紧”（这通常需要线性上/下界域或更复杂的相关性表达）。
- 不强行在第一版就对 `conv2d` 的边界 padding/边缘效应做完全精确（可先给 sound 的实现/降级策略，再迭代 tighten）。

---

## 3. 核心抽象：PerturbationSet（扰动集合）与 support function

### 3.1 为什么用 support function

对任意线性形式：

> `y = w^T x + b`，其中 `x = x0 + δ`，`δ ∈ S`

则输出区间可写为：

- `y_center = w^T x0 + b`
- `y ∈ [y_center - h_S(w), y_center + h_S(w)]`

其中 `h_S(w)` 是集合 `S` 的 **support function**：

> `h_S(w) = max_{δ∈S} w^T δ`

这一形式有两个重要优点：

1. 统一处理 `L∞/L2/L1/L0`：只要实现 `h_S(·)`，线性算子界就统一。
2. 与 LiRPA（auto_LiRPA）的 “concretize(A)” 思想一致：`A` 的每一行都在做一个 `w^T x` 的上/下界。

### 3.2 建议接口（概念层，不限定具体代码位置）

定义一个抽象：

- `PerturbationSet`：
  - `set_id: str`
  - `support(direction: Tensor) -> Tensor`：返回 `h_S(direction)`，支持 batch 维度
  - （可选）`concretize_box(center) -> (lower, upper)`：当集合能降成逐元素 box 时提供

并定义 runtime 输入规格：

- `InputSpec(value_name, center, perturbation: PerturbationSet)`

说明：

- 目前 `LinfInputSpec` 直接把 `perturbation` 固化成 `L∞ eps`，应改为薄封装（或保留兼容别名）。
- `perturbation` 不应该放在 Primal IR；它属于 verification spec/running context，更接近 `boundflow/ir/bound.py:9` 的 `Spec.perturbation`（但当前主线尚未使用该 BFBoundProgram）。

---

## 4. 线性算子（dense linear）的公式落地

### 4.1 Lp 球（p ∈ {∞,2,1}）

扰动集合：`S = {δ | ||δ||_p ≤ eps}`。其 support function 为：

> `h_S(w) = eps * ||w||_q`，其中 `1/p + 1/q = 1`

对应关系：

- `p=∞  -> q=1`
- `p=2  -> q=2`
- `p=1  -> q=∞`

对矩阵形式 `y = W x + b`（每行 `w_i`）：

- `y_center = W x0 + b`
- `dev_i = eps * ||w_i||_q`
- `y_l = y_center - dev`
- `y_u = y_center + dev`

向量化实现要点：

- 对 `W: [O,I]`：`dev = eps * norm(W, q, dim=1)`
- 对 batched weight `W: [B,O,I]`：`dev = eps * norm(W, q, dim=2)`

### 4.2 L0 球（需要明确定义）

`L0` 常见有两类不同定义，必须在论文/代码中选其一，否则会导致 reviewer 指出“定义不清”：

1. **稀疏幅度扰动**：最多改变 `k` 个维度，每个维度的幅度满足 `|δ_i| ≤ eps`  
   `S = {δ | ||δ||_0 ≤ k, ||δ||_∞ ≤ eps}`
2. **稀疏预算扰动**：最多改变 `k` 个维度，但每维幅度不受单独上界（不太常用，通常需要额外限制）

推荐采用 (1)。在 (1) 下：

> `h_S(w) = eps * sum_topk(|w|, k)`

对 `y = W x + b`：

- `dev_i = eps * sum_topk(|w_i|, k)`
- `y ∈ [y_center - dev, y_center + dev]`

工程注意：

- `topk` 对大矩阵较重；但第一版可以先在 CPU/Python reference 做正确实现，再考虑 TVM/TIR 优化（或用近似/分块）。

---

## 5. conv2d 的处理策略（精确 vs 降级）

conv2d 本质也是线性算子，但相对 dense linear，`w_i`（每个输出位置的 “展开后行向量”）是隐式的。

### 5.1 最小可行（第一版建议）

对 `p != ∞` 的输入扰动，先采用 **sound 的降级策略**：

- 把 `Lp/L0` 扰动保守外包到一个 `L∞` box：`|δ_i| ≤ eps`（注意：`||δ||_p ≤ eps` ⇒ `||δ||_∞ ≤ eps` 对 p∈{1,2,∞} 都成立；对 L0+L∞ 也成立）
- 然后继续走现有 interval conv2d 传播（`boundflow/domains/interval.py` 里的 `w_pos/w_neg` 公式）

这样保证 correctness/soundness，但 tightness 会变差；属于工程上“先通用，再 tighten”的路线。

### 5.2 可迭代 tighten（后续版本）

对 `p=2`，可以做到比 box 更紧、且仍然向量化：

- 对每个输出元素：`dev = eps * ||w_row||_2`
- 其中 `||w_row||_2` 在 interior 位置等于 `||kernel||_2`（按 in_channel*kh*kw 展开），边缘位置由于 padding/越界会更小
- 可通过对 “有效输入 mask” 与 `kernel^2` 做类似卷积的方式得到每个位置的平方和，再开方

对 `p=1`（q=∞）与 `L0` 的精确实现会更复杂（涉及每个输出位置的 max/topk 权重），建议在系统论文里将其列为未来工作或仅对 interior 做近似。

---

## 6. 与 BoundFlow 分层的对接点（落地路线）

### 6.1 Runtime 输入规格：从 `LinfInputSpec` 泛化

当前：

- `LinfInputSpec(value_name, center, eps)`（见 `boundflow/runtime/task_executor.py`）

建议：

- 引入 `InputSpec(value_name, center, perturbation)`，并保留 `LinfInputSpec` 作为 thin wrapper：
  - `LinfInputSpec` → `LpBall(p=inf, eps=eps)`

需要同步修改：

- `boundflow/runtime/scheduler.py`：scheduler 接收泛化的 `InputSpec`
- `boundflow/runtime/executor.py`、`boundflow/runtime/tvm_executor.py`：透传/兼容

### 6.2 PythonTaskExecutor：将“输入扰动→第一个 affine”显式化

关键点：在 interval 域里，**只要进入了 `IntervalState(lower,upper)`**，后续 `affine_transformer` 的公式就固定了（对应 L∞ 的 box 传播）。因此多范数差异的最佳切入点是：

- 当 `TaskOp.linear/conv2d` 的输入是 “被扰动的原始输入 value” 时，直接用 `PerturbationSet.support` 计算输出区间；
- 之后输出变为 `IntervalState`，继续沿用现有 interval IBP。

这保持了系统结构最小改动，并把 tightness 的提升集中在第一层。

### 6.3 Domain 层：保持 interval transformer 不变，但为将来扩展留接口

- `IntervalDomain` 继续只接收 `IntervalState` 并做 box 传播（现状 OK）。
- 将来若引入 DeepPoly/CROWN，需要新增 `DomainState`（例如 `(A,b)` 形式）并让 `PerturbationSet` 支持 `concretize(A)`；此时 support function 仍是基础构件。

### 6.4 Planner/IR 层：尽量不侵入

- 扰动是 verification spec/running context，原则上不进入 Primal IR 与 Task IR。
- 若需要可复现（AE/bench）记录，可把 perturbation 的 jsonable 描述写到 `module.bindings["spec"]["input_perturbation"]` 或 JSONL config 字段，但不影响核心 IR。

---

## 7. 评测与测试建议（如何证明实现正确）

1. **单层线性算子公式测试**（最关键）  
   固定 `x0, W, b`，对 `p=∞,2,1` 与 `L0(k)`：
   - 使用随机采样 `δ`（满足约束）验证 `Wx0+Wδ+b` 落在 `[lb,ub]` 内（soundness）
   - 对 `p=∞,2,1` 可与 auto_LiRPA 的 `PerturbationLpNorm(norm=...)` 对齐（tightness/一致性）

2. **端到端一致性测试**  
   对小 MLP：以不同 norm/eps 生成 input perturbation，比较 BoundFlow 输出与 auto_LiRPA `compute_bounds(method="IBP", norm=...)` 的一致性（允许浮点误差）。

3. **conv2d 的策略测试**  
   - 第一版若选择“降级到 box”，只验证 soundness 与回归不破坏已有 `L∞`。
   - 若实现 `p=2` tighten，则加一个 edge/padding 的随机测试（不同 padding/stride）。

---

## 8. 结论（论文表述建议）

- “支持多范数扰动”不应该被表述为“把 eps 换一下”，而是：BoundFlow 将输入扰动显式建模为 `PerturbationSet`，并通过 support function 统一线性算子边界计算，使得 `L∞/L2/L1/L0` 在 affine 层具有正确且可扩展的实现路径；随后仍可复用现有 interval IBP 的任务图与后端编译基础设施。

