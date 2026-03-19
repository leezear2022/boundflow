# 为什么是 BoundFlow？——两个核心学术辩护

> **核心问题**：
> 1. 为什么不直接用 auto_LiRPA？
> 2. 既然用了编译，为什么不直接用 TVM，而是重起炉灶写 BoundFlow？

---

## 0. 一句话回答

- **为什么不用 auto_LiRPA**：auto_LiRPA 是成熟的 LiRPA/验证算法库（我们也用它做 baseline/对齐），但它并不提供“编译器式”的 IR/Planner/Scheduler 抽象，难以做显式任务图、存储规划、跨运行缓存与系统化消融；BoundFlow 把边界传播提升为**可规划/可编译/可审计的程序**。
- **为什么不直接用 TVM**：TVM 是通用张量编译器，**能编译你显式写出来的张量程序**；但它不会自动把“验证问题”（模型 + 扰动 + spec）变成“边界传播程序”，也不提供验证语义级的 task abstraction（TaskGraph/StoragePlan）与对齐/消融/复现产线。BoundFlow 负责生成与规划边界传播程序，并复用 TVM 做 codegen/kernel。

---

## 1. 为什么不直接用 auto_LiRPA？

### 1.1 auto_LiRPA 是什么

auto_LiRPA 是目前最成熟的神经网络验证库之一，支持：
- 多种边界传播方法：IBP、CROWN、α-CROWN、GradNorm-CROWN 等
- 自动微分式的边界传播：用户只需调用 `compute_bounds()`，库自动完成所有传播
- 广泛的学术应用：是 verified training、certified defense 等领域的事实标准

**我们不质疑 auto_LiRPA 的算法正确性**——事实上，BoundFlow 使用 auto_LiRPA 作为 baseline 和正确性对照（见 `tests/test_phase3_ibp_against_auto_lirpa.py`）。

### 1.2 auto_LiRPA 的系统局限

auto_LiRPA 的设计目标是"**提供正确且易用的边界传播实现**"，而不是"**提供可系统化优化的边界传播执行**"。这导致了以下系统性问题：

| 维度 | auto_LiRPA（库） | BoundFlow（verification compiler） |
|------|-----------|-----------|
| **执行/组织** | 运行期在 PyTorch 中构建/遍历 bound graph，调度/优化多为库内部策略 | 显式 plan → task graph →（可选）编译/缓存 → 调度执行 |
| **稳定 IR** | 内部结构可用但不是面向系统优化/消融的稳定 IR | Primal IR / Task IR / TaskGraph / StoragePlan（可被 pass 操作） |
| **存储规划** | 依赖张量生命周期 + PyTorch allocator（难做全局 buffer reuse） | 显式 StoragePlan + reuse pass，并可与 TVM memory planning 叠加 |
| **跨运行复用** | 有工程缓存但粒度/口径依赖使用方式（非“编译产物”意义） | 任务级 compile cache（进程内 + 可选落盘）/VM cache 等 |
| **可消融性** | 很难把系统优化拆成“可独立开关的旋钮” | PlannerConfig/TVMOptions knobs 可独立开关并量化贡献 |
| **可复现性** | 不定义评测输出契约与 artifact 产线 | JSONL schema + contract tests + artifact runner |
| **性能抓手** | 依赖 PyTorch kernel + 手写路径优化 | 任务级 lowering + TVM codegen + 统一缓存/存储规划（减少解释层开销与峰值内存） |

### 1.3 核心洞察：从"算法实现"到"可编译程序"

auto_LiRPA 的代码组织是"**算法导向**"的：
```python
# auto_LiRPA 的执行流程（简化）
for layer in model.layers:
    bounds = propagate_bounds(bounds, layer)  # 每层单独处理
return bounds
```

这种组织方式的问题：
1. **跨层优化难以成为“编译 pass”**：优化常以特例散落在算子实现/路径分支里，难以组合与复用
2. **存储复用缺少显式接口**：中间张量生命周期主要由 runtime 管理，难以做全局 liveness + reuse 规划
3. **跨运行复用粒度不清晰**：同一 workload 的多次评测，很难定义“可缓存的计划/编译产物”
4. **系统化消融成本高**：缺少统一 knobs/schema，把每个系统优化的贡献稳定量化并不容易

BoundFlow 的组织方式是"**编译器导向**"的：
```
Workload → Frontend → Primal IR → Planner → Task IR + TaskGraph + StoragePlan → Scheduler → Executor
```

这种组织方式的优势：
1. **IR 分离关注点**：Primal IR 保持语义纯净，Task IR 携带执行信息
2. **Planner 统一优化**：分区、存储复用、lowering 策略都在 planner 层做
3. **Executor 可替换**：Reference（正确性）vs TVM（性能）
4. **产线可复现**：schema 固化 + contract tests + artifact runner

### 1.4 论文语言

> **Claim**：auto_LiRPA 解决的是"算法正确性"问题，BoundFlow 解决的是"系统效率"问题。这两个问题是正交的：BoundFlow 使用 auto_LiRPA 的语义作为 reference，同时在系统层面提供编译式执行、显式存储规划、可配置优化空间等能力。

---

## 2. 为什么不直接用 TVM？

### 2.1 TVM 是什么

TVM 是目前最成熟的深度学习编译器之一，支持：
- 多种前端：PyTorch、ONNX、TensorFlow 等
- 多种后端：CPU、GPU、各种加速器
- 丰富的优化 pass：融合、量化、内存规划、自动调优等

BoundFlow 确实使用了 TVM 作为后端（见 `boundflow/runtime/tvm_executor.py`），但**不是直接用 TVM 编译原始模型**。

### 2.2 TVM 的抽象层次不对

更准确的说：TVM 是**通用张量编译器/代码生成器**（Relax/TIR → CUDA/LLVM/...）。它并不“只能做前向推理”——只要你把 interval IBP 写成张量程序，TVM 当然也能编译。

问题在于：验证场景的输入并不是“一段现成的张量程序”，而是：
```
输入：模型计算图 + 扰动模型（L∞ eps）+ 性质规格（Spec）+ 域语义（IBP/CROWN/...）
输出：边界区间 [y_l, y_u]（或 verified/not verified）
```

在现有工具链里，缺失的是把这些对象**系统化 lowering** 成一个“可优化/可缓存/可消融”的边界传播程序的中间层；这正是 BoundFlow 的定位。TVM 负责把 Lowering 后的张量程序编成高效 kernel，而不是替你决定“验证语义是什么、边界传播程序长什么样、怎么调度/复用/缓存”。

### 2.3 边界传播的特殊性

| 维度 | 原始计算（TVM 的目标） | 边界传播（BoundFlow 的目标） |
|------|----------------------|---------------------------|
| **计算单元** | 单个张量 `y` | 张量对 `[y_l, y_u]` |
| **算子语义** | `linear(x, W, b) → y` | `interval_linear(x_l, x_u, W, b) → [y_l, y_u]` |
| **依赖关系** | DAG（无特殊约束） | 下界/上界之间有数据依赖 |
| **存储模式** | 单缓冲区 | 双缓冲区（lower/upper 需同步） |
| **正确性约束** | 数值精度 | soundness（上界必须 ≥ 真实值） |

举个具体例子——**interval linear**：

```python
# 原始计算（TVM 可以直接编译）
y = W @ x + b

# 边界传播（需要显式建模为 interval 程序）
W_pos = max(W, 0)
W_neg = min(W, 0)
y_l = W_pos @ x_l + W_neg @ x_u + b  # 下界
y_u = W_pos @ x_u + W_neg @ x_l + b  # 上界
```

这里的关键问题：
1. **任务粒度与复用边界**：若把 interval 逻辑完全展开为原始算子，通用 fusion/内存规划未必能利用 interval 语义的共享结构（如 `W_pos/W_neg`、上下界成对出现），也难以把它作为“可缓存/可复用”的任务单元
2. **峰值内存高度依赖调度**：下界/上界天然成对出现，且与 StoragePlan 的 buffer reuse 强耦合；这类全局安排需要在 task 图层做 Planner/Scheduler 设计
3. **soundness 需要系统契约**：验证不是“算得快就行”，还要保证上界不被优化破坏；BoundFlow 把 reference semantics（Python）与 compiled semantics（TVM）以及 auto_LiRPA baseline 对齐，作为持续回归的正确性约束

### 2.4 BoundFlow 如何复用 TVM

BoundFlow 的策略是：**在 Task 层面定义边界传播语义，在 TVM 层面实现高效执行**。

```
BoundFlow Planner                    TVM
    ↓                                 ↓
TaskOp: interval_linear     →    Relax IR: interval_linear_ibp (custom op)
    ↓                                 ↓
TaskGraph: 依赖关系          →    TVM Schedule: 执行顺序
    ↓                                 ↓
StoragePlan: buffer 复用     →    StaticPlanBlockMemory: 内存规划
```

这种分层的好处：
1. **BoundFlow 负责"边界传播语义"**：TaskOp、TaskGraph、StoragePlan 都是 BoundFlow 定义的
2. **TVM 负责"代码生成"**：把 `interval_linear_ibp` 编译成高效的 CUDA/LLVM kernel
3. **两者职责清晰**：BoundFlow 不需要重写 TVM 的 codegen，TVM 不需要理解边界传播

### 2.5 论文语言

> **Claim**：TVM 解决的是"单算子代码生成"问题，BoundFlow 解决的是"边界传播任务编排"问题。这两个问题在不同的抽象层次：TVM 把 `matmul(A, B)` 编译成高效 kernel，BoundFlow 把 `interval_linear(x_l, x_u, W, b)` 分解为可调度的任务并复用 TVM 的 codegen 能力。

---

## 3. BoundFlow 的独特贡献（系统论文角度）

### 3.1 三层定位

| 层次 | 现有方案 | BoundFlow 的位置 |
|------|---------|-----------------|
| **算法层** | auto_LiRPA（正确的边界传播规则） | 复用 auto_LiRPA 语义作为 reference |
| **编译层** | TVM（高效的代码生成） | 复用 TVM 作为执行后端 |
| **系统层** | **缺失** | **BoundFlow 的核心贡献** |

### 3.2 具体贡献点

1. **验证感知的 IR**
   - Primal IR：保留原始语义，支持前端导入
   - Task IR：携带边界传播规则，支持 planner 优化

2. **显式的任务图（TaskGraph）**
   - buffer 级依赖（不是 value 级）
   - 支持跨任务存储复用

3. **系统级存储规划（StoragePlan）**
   - Planner 层面的 buffer reuse pass
   - 与 TVM 的 StaticPlanBlockMemory 正交组合

4. **可插拔的执行器**
   - PythonTaskExecutor：正确性锚点（reference semantics）
   - TVMTaskExecutor：性能优化（compiled semantics）
   - 两者通过 allclose 测试保持一致

5. **可复现的评测产线**
   - JSONL schema 固化（schema_version=1.0）
   - contract tests 保证口径稳定
   - artifact runner 一键复现

### 3.3 学术价值的一句话总结

> BoundFlow 不是要替代 auto_LiRPA 的算法，也不是要重写 TVM 的代码生成。它解决的是一个**中间层的系统问题**：**如何把"边界传播"从算法实现提升为可系统化优化的编译目标**。

这类似于：
- PyTorch 提供了正确的自动微分 → TorchDynamo + Triton 提供了训练编译
- auto_LiRPA 提供了正确的边界传播 → **BoundFlow 提供了验证编译**

---

## 4. 论文写作建议

### 4.1 Introduction 的叙事

1. **背景**：神经网络验证的重要性（safety-critical applications）
2. **现状**：auto_LiRPA 等库提供了正确的边界传播实现，但停留在"脚本式执行"
3. **问题**：随着模型规模增长，脚本式执行的系统开销成为瓶颈
4. **洞察**：边界传播可以被视为一种"可编译的程序"；但通用张量编译器并不提供验证语义级的 IR/Planner/Scheduler 来组织这一类 workload
5. **贡献**：BoundFlow，一个验证感知的编译器，把边界传播变成可规划、可编译、可系统化优化的程序

### 4.2 Design 的辩护

**为什么不直接改 auto_LiRPA**：
> auto_LiRPA 的设计目标是"正确且易用"，其内部结构是算法导向的。要引入编译式执行，需要从 IR、Planner、Scheduler 重新设计，这本质上是一个新系统。BoundFlow 选择使用 auto_LiRPA 作为 reference，保证语义正确性，同时在系统层面提供编译优化。

**为什么不直接用 TVM**：
> TVM/Relax 能编译张量程序，但不会自动生成“验证语义程序”，也不提供 TaskGraph/StoragePlan 这类 verification-aware 抽象。BoundFlow 在 Task 层面定义边界传播语义并做规划/缓存，TVM 在后端做 codegen，两者职责清晰。

### 4.3 Evaluation 的对照

| 对比维度 | 对照系统 | 预期结论 |
|----------|---------|---------|
| **正确性** | auto_LiRPA | BoundFlow 与 auto_LiRPA 输出 allclose（相对误差 < 1e-6） |
| **性能** | auto_LiRPA | BoundFlow 在 steady-state 下比 auto_LiRPA 快 X 倍 |
| **存储** | TVM standalone | BoundFlow 的 StoragePlan + TVM StaticPlan 比单独 TVM 省 Y% 内存 |
| **可消融性** | N/A | BoundFlow 的 knobs 可独立开关，每个 knob 的贡献可量化 |

---

## 5. 预期的 Reviewer 问题与回应

### Q1：auto_LiRPA 已经很快了，为什么还需要 BoundFlow？

**回应**：
- auto_LiRPA 在小模型上确实够用，但随着模型规模增长，Python 解释器开销、缺乏存储复用、无法批量编译等问题会放大
- BoundFlow 的 steady-state 性能可以比 auto_LiRPA 快 X 倍（见 Evaluation）
- 更重要的是，BoundFlow 提供了**可系统化消融的能力**——可以量化每个优化点的贡献

### Q2：为什么不把 BoundFlow 的优化贡献回 auto_LiRPA？

**回应**：
- BoundFlow 的优化不是"在 auto_LiRPA 上加几个 patch"，而是从 IR、Planner、Scheduler 重新设计的系统化方案
- auto_LiRPA 的架构是"算法导向"的，要引入编译式执行需要大规模重构
- 我们选择的策略是：使用 auto_LiRPA 作为 reference，保证语义正确性；在 BoundFlow 中实现系统优化

### Q3：TVM 也有 Relax，为什么不直接在 Relax 上实现？

**回应**：
- Relax 的抽象仍然是"原始计算图"——它提供了 `R.matmul`、`R.add`，但没有 `R.interval_linear`
- BoundFlow 在 Relax 之上定义了边界传播的算子语义（见 `boundflow/backends/tvm/relax_interval_task_ops.py`）
- 这是一个**分层设计**：BoundFlow 定义语义，Relax 提供 IR 基础设施，TVM 提供 codegen

### Q4：BoundFlow 目前只支持 IBP，如何扩展到 CROWN？

**回应**：
- BoundFlow 的 IR 设计是 domain-agnostic 的：Primal IR 保留原始语义，Task IR 携带域特定信息
- 扩展到 CROWN 需要：(1) 新增 `TaskKind.CROWN`；(2) 实现 CROWN 的 `TaskOp` lowering；(3) 更新 `TVMTaskExecutor` fallback
- 这是 Phase 6 的计划方向（见 `docs/phase5_done.md`）

---

## 6. 仓库证据索引（写论文/AE 时可直接引用）

- auto_LiRPA baseline/对齐与 gate：`scripts/bench_ablation_matrix.py`、`gemini_doc/change_2025-12-21_pr14c_auto_lirpa_baseline_cached_timing_and_gate.md`
- 对齐测试（BoundFlow ↔ auto_LiRPA）：`tests/test_phase3_ibp_against_auto_lirpa.py`、`tests/test_phase4_task_pipeline_against_auto_lirpa.py`
- IR/Planner/Scheduler 主线：`boundflow/ir/primal.py`、`boundflow/ir/task.py`、`boundflow/ir/task_graph.py`、`boundflow/planner/pipeline.py`、`boundflow/runtime/scheduler.py`
- StoragePlan 与复用 pass：`boundflow/planner/passes/buffer_reuse_pass.py`
- TVM lowering/executor/compile cache：`boundflow/backends/tvm/relax_interval_task_ops.py`、`boundflow/runtime/tvm_executor.py`、`gemini_doc/change_2025-12-22_pr15c_tvm_disk_compile_cache_dir.md`
- 复现产线（schema/contract/artifact）：`docs/bench_jsonl_schema.md`、`tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`、`scripts/run_phase5d_artifact.py`

---

## 附：图示（建议放入论文）

### 系统定位图

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│        (Certified Training, Neural Network Verification) │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│                    Algorithm Layer                       │
│            auto_LiRPA: 正确的边界传播规则                  │
│            (IBP, CROWN, α-CROWN, ...)                    │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│                    System Layer ← BoundFlow              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  Primal IR  │→ │   Planner   │→ │  Scheduler  │      │
│  │ (语义对齐)   │  │ (任务规划)   │  │ (执行调度)   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│         ↓                ↓                ↓              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   Task IR   │  │  TaskGraph  │  │ StoragePlan │      │
│  │ (边界语义)   │  │ (依赖关系)   │  │ (存储复用)   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│                    Codegen Layer                         │
│            TVM: 高效的 kernel 代码生成                    │
│            (CUDA, LLVM, ...)                             │
└─────────────────────────────────────────────────────────┘
```

### 与 auto_LiRPA 的关系图

```
┌──────────────────────────────────────────────────────────────────┐
│                        BoundFlow                                  │
│  ┌────────────────┐         ┌────────────────┐                   │
│  │ PythonExecutor │←───────→│ TVMExecutor    │                   │
│  │ (Reference)    │ allclose│ (Optimized)    │                   │
│  └───────┬────────┘         └───────┬────────┘                   │
│          │                          │                             │
│          │  ┌───────────────────────┘                             │
│          │  │                                                     │
│          ↓  ↓                                                     │
│  ┌────────────────┐                                               │
│  │ IntervalDomain │ ← 语义对齐                                    │
│  │ (IBP 规则)      │                                               │
│  └───────┬────────┘                                               │
└──────────┼───────────────────────────────────────────────────────┘
           │
           ↓ baseline 对照
┌──────────────────┐
│    auto_LiRPA    │
│  compute_bounds  │
│  (method='IBP')  │
└──────────────────┘
```
