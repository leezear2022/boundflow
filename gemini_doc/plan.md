下面是一份可以直接丢给其他大模型/合作者的 **BoundFlow 项目与论文计划文档（面向 ASPLOS）**。我按“动机 → 想法 → 架构 → 论文结构 → 远景/风险”的顺序写，尽量自洽、可落地，并把关键相关工作挂上来源。

---

# BoundFlow 计划文档（ASPLOS 路线）

## 0. 一句话定位

**BoundFlow** 是一个把 **LiRPA/CROWN 边界传播 +（可选）BaB 完备验证 + certified training** 当成“一等公民工作负载”的 **验证感知编译器与运行时系统**：
用 **专用 IR + 全局 Planner + TVM(TIR) GPU 后端 +（可选）GPU 友好 BaB runtime**，系统性消除现有 PyTorch/库式 verifier 的 kernel 颗粒度碎片、CPU↔GPU 同步与重复计算。

---

## 1. 我的动机（Why now / Why systems）

### 1.1 现状：LiRPA 已经很强，但“系统层”还很原始

* LiRPA（线性松弛边界传播）已经是鲁棒性验证/认证训练的核心范式之一。CROWN 给出通用的线性/二次上下界框架。
* auto_LiRPA 把 LiRPA “自动化”到一般计算图（类似自动微分那样沿图传播），并且提出了 loss fusion 等技巧来降低 certified defense 的复杂度。
* α,β-CROWN / β-CROWN 把 bound propagation 与 BaB 结合，显著提升 complete verification 的效率，并在 VNN-COMP 中成为强力工具。

**但**：这些工具在实现上多数仍是“库 + Python 调度 + 手工 kernel/算子组合”。即便算法边界紧、数学做对了，**端到端**仍容易被以下系统问题拖垮：

* bound 图大量细碎算子 → kernel launch 多、访存/复用差；
* 多 spec / 多 BaB 节点的重复前缀计算难复用；
* BaB 主循环常由 CPU 驱动，频繁 GPU 同步，吞吐被拖低。

**ASPLOS 的机会点**：不发明新 verifier 算法，而是把“验证/认证训练”当成新 workload，做 **编译器 + 运行时** 的系统性提速与规模化。

### 1.2 类比启发：IMCompiler 的“专用编译器范式”

IMCompiler 针对密码学大整数乘法，把复杂多变的高层参数统一降到少数原语（UNIT_MUL），并用计算图指导并行/缓存策略，自动生成高性能 GPU 内核。
**BoundFlow 的类比**：把各种 domain / spec / BaB 节点 / 网络结构统一降到少数 **Bound 原语**（BoundAffine/BoundAct/BoundPool…），再由 planner 与后端系统性生成“粗粒度、高复用”的 GPU 代码。

---

## 2. 我的核心想法（Key idea）

### 2.1 把验证当成“可编译工作负载”

把一次验证/一次认证训练中的 bound 计算，显式表示为 **Bound Computation Graph**（BCG）：

* 节点不是普通前向算子，而是“携带 domain 状态的 BoundOp”；
* 边表示 **abstract state（如 interval/linear/αβ 参数）**的流动与依赖；
* 图上存在天然的多维批处理轴：`spec batch`、`BaB node batch`、`class/margin batch`。

### 2.2 关键系统招式（BoundFlow 的三板斧）

1. **Verification-aware IR（BoundFlow IR）**
   把 primal 图与 bound 图的关系、domain 状态、spec/BaB 结构显式化，允许做跨层/跨 spec 的 program transform。

2. **Global Bound Planner（全局边界计划器）**
   把“算 bound”从 PyTorch 的即时执行变成规划问题：

* 决定 **融合粒度**（算子/层级 fusion）
* 决定 **批处理布局**（spec×neuron 的 2D tiling / BaB batching）
* 决定 **复用与缓存**（前缀 bound 复用、重算 vs 缓存、显存峰值约束）

3. **TVM(TIR) GPU 后端 +（可选）GPU 友好 runtime**
   用 TVM 的张量编译与自动调优能力把 planner 产出的 BoundTask 编译成高效 kernel（TVM 的端到端优化与自动搜索能力是成熟的系统基础）。
   对 complete verification，可逐步把 BaB 的 bound 查询从“逐节点小调用”升级为“批量 BoundTask + 减少同步”。

---

## 3. 为什么选 TVM 而不是直接 MLIR（工程决策写清楚）

* MLIR 是强大的“可扩展编译器基础设施”，适合构建多层 IR 与方言生态。
* 但 BoundFlow 的 v1 关键瓶颈不在“IR 框架是否 SSA/方言化”，而在 **planner 与 kernel 粗粒度化**（fusion/batching/reuse/sync）是否做对。
* TVM 在张量算子优化、自动调优、跨硬件后端上已经提供成熟能力，且对“生成 CUDA kernel”这条路径更短、更容易快速得到可量化的端到端收益。

**论文表述建议**：

> BoundFlow 的设计与后端无关；我们选择 TVM/TIR 实现原型以快速验证系统思想，MLIR 后端是自然扩展方向。

---

## 4. 工程形态：BoundFlow 放在哪个 repo？

**建议：新建独立工程 `boundflow`，把 auto_LiRPA/αβ-CROWN 作为前端“用户”，TVM 作为后端“依赖”。**

原因：

* 避免把验证特有概念强塞进 TVM 核心；
* 避免 BoundFlow 被误解为 auto_LiRPA 的内部小优化；
* 便于未来接入多前端（auto_LiRPA、αβ-CROWN）与多后端（TVM、将来 MLIR、将来硬件专用）。

---

## 5. BoundFlow 架构（可以直接画成论文 Fig.1）

### 5.1 总览数据流

```
PyTorch Model + Spec (+ BaB nodes)
        │
        │  (Front-end: auto_LiRPA-style bound derivation on general graphs)
        ▼
   BoundFlow IR  = PrimalGraph + BoundGraph + DomainState + Spec/BaB axes
        │
        │  (Global Bound Planner: fuse / batch / reuse / memory-aware schedule)
        ▼
  BoundTasks (coarse-grained subgraphs with schedule metadata)
        │
        │  (TVM/TIR Backend: codegen + autotune)
        ▼
 Compiled Kernels  +  Runtime (batched execution, optional BaB integration)
```

### 5.2 前端：如何避免“每来一个新算子就手写边界”

前端策略是 **“少量原语 + 自动图变换”**：

* auto_LiRPA 已证明：可以把 LiRPA 推广到一般计算图并自动传播（像 AD 一样沿图计算 bound），并提供开源库。
* BoundFlow v1 不发明新推导体系：

  * 复用（或复刻）auto_LiRPA 的 derivation 思想；
  * 将复杂算子在前端分解为少数 **Bound 原语**（BoundAffine/BoundAct/BoundPool/Reshape…）。
    => **新增高层算子**大多数只需“分解到原语”，而不是从零推导新边界。

---

## 6. BoundFlow IR 设计要点（论文贡献 1）

**目标**：IR 必须显式表达“验证任务的多维结构与可复用性”，这是普通 DL IR 不具备的。

### 6.1 IR 需要显式建模的实体

* `Domain`：IBP/interval、linear LiRPA（CROWN）、αβ 参数等（与 αβ-CROWN 兼容）。
* `Spec`：扰动集合、margin 形式、目标输出等
* `BaBNode`：分裂约束/已定 ReLU 状态等（complete verification）
* `BoundTensor`：带 domain 的张量（可能包含 lower/upper 或线性系数 A,b）

### 6.2 IR 的两个关键“系统接口”

* **Task interface**：planner 把 IR 切成 `BoundTask`（子图 + 批处理轴 + schedule meta）
* **Cache interface**：显式声明哪些中间 bound 可以跨 spec/BaB 共享、生命周期多长、占用多大

---

## 7. Global Bound Planner（论文贡献 2，ASPLOS 灵魂）

Planner 不是“调一个 schedule”，而是把验证转成一个“资源约束下的规划问题”。

### 7.1 Planner 输入/输出

* 输入：BoundFlow IR + GPU/显存预算 + domain 类型 + 任务模式（incomplete / BaB / training）
* 输出：一组 `BoundTask`：

  * `subgraph`：要计算的 bound 子图
  * `batch_axes`：spec batch / BaB batch / class batch 的打包方式
  * `reuse_plan`：缓存哪些中间 bound、复用哪些前缀
  * `schedule_meta`：后端 tiling/fusion 参数范围

### 7.2 核心优化维度（建议做成论文小节）

1. **Layer/block fusion（粗粒度化）**
   把多层 BoundOp 融成少数 kernel，减少 launch 与 DRAM 往返。auto_LiRPA 已讨论过 loss fusion 能降低复杂度，BoundFlow 把它系统化为可编译任务。

2. **2D 并行与批处理布局（spec × neuron/layer）**
   借鉴 IMCompiler 用“计算图”指导 1D/2D 并行策略的分析方式，把 bound 任务的并行划分写成可解释的 trade-off（负载均衡、warp 发散、片上资源、通信/同步）。

3. **跨 spec / 跨 BaB 节点复用（前缀共享）**
   BaB 的大量节点共享网络前缀；planner 决定“共享哪些层的中间 bound”以换取显存峰值与计算重算之间的平衡。β-CROWN 强调 bound propagation 与 BaB 的结合效率，BoundFlow 在系统层进一步挖“复用 + 批处理”的空间。

---

## 8. TVM 后端与运行时（论文贡献 3）

### 8.1 选择 TVM 的价值点（写进实现章节）

TVM 提供：算子级优化、图级融合、自动搜索/调优、跨后端代码生成，是把 BoundTask 快速落到高性能 kernel 的“现成系统杠杆”。

### 8.2 后端的“最小原语内核集”（像 IMCompiler 的 UNIT_MUL）

优先把工程收敛到少量 kernel 模板：

* `BoundAffine`（Conv/Linear 的 bound 核心）
* `BoundAct`（ReLU 等激活的松弛/状态特化）
* `BoundReduce`（输出 margin、max/argmax 相关）
  其他算子尽量在前端分解组合。

### 8.3 BaB 集成策略（从易到难）

* **v1（容易、收益大）**：CPU 仍做 BaB 决策，但把每轮待扩展节点 **批量打包**成 BoundTask，在 GPU 上一次/少数次 kernel 完成 bound 查询，减少“逐节点调用”。
* **v2（更系统）**：引入任务队列/持久化 kernel 等 runtime 技术，进一步降低同步与 launch 开销（论文可以先做 v1，v2 作为扩展）。

---

## 9. 论文主张的“Main Contributions”（你可直接放 abstract）

1. **BoundFlow IR**：首个面向 LiRPA 验证的验证感知 IR，显式编码 domain/spec/BaB 结构与可复用性。
2. **Global Bound Planner**：把 bound 计算从库式执行提升为资源约束下的全局规划，系统性实现 fusion/batching/reuse。
3. **TVM-based GPU backend**：将 BoundTask 自动生成并调优为粗粒度高吞吐 kernel，在验证/认证训练中显著降低端到端时间。
4. **端到端集成与评估**：在标准验证基准（VNN-COMP 生态）和主流 verifier（auto_LiRPA / αβ-CROWN）上验证可移植加速收益。

---

## 10. 论文结构建议（ASPLOS 版本骨架）

1. **Introduction**

   * 现实需求：verified AI、certified training
   * 现状：LiRPA/αβ-CROWN 已强但系统低效
   * 贡献概述与结果亮点

2. **Background & Motivation**

   * LiRPA / CROWN / auto_LiRPA 的自动化思路
   * αβ-CROWN + BaB 的 complete verification 场景
   * Profiling：kernel launch 碎片化、同步、重复前缀（你自己跑数据填）

3. **BoundFlow Overview**（Fig.1 架构图 + 设计目标）

4. **BoundFlow IR**（类型系统、任务轴、可复用性语义）

5. **Global Bound Planner**

   * fusion/batching/reuse/memory-aware
   * compute-diagram 风格的并行策略分析（借鉴 IMCompiler 的叙事法）

6. **TVM Backend & Runtime**

   * kernel 模板、2D tiling、autotune
   * BaB 集成（v1 batched nodes）

7. **Evaluation**

   * 基准：VNN-COMP 任务/模型，或其论文/仓库提供的 benchmark 接口
   * 对比：auto_LiRPA、αβ-CROWN
   * 指标：端到端时间、GPU 利用率、launch 次数、显存峰值、timeout 率

8. **Related Work**

   * LiRPA/CROWN/auto_LiRPA/β-CROWN/αβ-CROWN
   * 编译器系统：TVM、MLIR
   * 类编译器专用生成：IMCompiler

9. **Discussion & Limitations**（domain 扩展、更多算子、MLIR 后端、硬件噪声等远景）

---

## 11. 实现路线（工程可执行清单）

### 11.1 建 repo（推荐目录）

* `frontends/autolirpa/`：把 auto_LiRPA 的 bound derivation 结果导出成 BoundFlow IR（先从 CNN/ResNet 子集开始）。
* `ir/`：BoundTensor/BoundOp/Spec/BaBNode/Task
* `planner/`：fusion/batching/reuse
* `backends/tvm/`：Task → TIR → CUDA kernel；autotune hook
* `runtime/`：执行引擎 +（可选）BaB 批量接口

### 11.2 最小可跑通闭环（MVP）

1. 选定一个小模型：MNIST-CNN 或 CIFAR 小 ResNet
2. 用 auto_LiRPA 得到 baseline bound（正确性 ground truth）
3. 导出 IR → planner 生成 `BoundTask`（先做简单 fusion）
4. TVM 编译执行 → 对齐数值
5. 测性能：单次 bound 时间、launch 次数

### 11.3 逐步升级

* 加 spec batching（多 ε / 多输入）
* 加 BaB 批量节点（对接 αβ-CROWN 的查询接口）
* 加 reuse（共享前缀 bound）
* 做 ablation（fuse/batch/reuse 各贡献）

---

## 12. 远景（Vision）

### 12.1 近期（论文落地）

* 在 VNN-COMP 生态基准上，BoundFlow 作为“后端加速器”提高 αβ-CROWN/auto_LiRPA 的端到端吞吐，降低 timeout。

### 12.2 中期（系统平台化）

* 把验证工作负载沉淀成“可调度、可编译、可复现”的 pipeline：

  * 不同 domain、不同网络结构，都能复用 planner 与 kernel 模板。
* 可选：引入 MLIR 后端，把 BoundFlow IR 作为方言落地（工程更重，但路线自然）。

### 12.3 长期（与你硬件研究对接）

* 将“硬件不确定性/噪声”纳入 domain（例如权重扰动、RRAM 噪声）作为扩展；
* certified training 的系统加速（如 CROWN-IBP 类训练本身就有显著开销差异与优化空间）。

---

## 13. 风险与对策（写给自己也写给审稿人）

1. **“你只是把 auto_LiRPA 换成 TVM”** 的质疑

* 对策：强调 **IR+Planner** 是主贡献；TVM 是实现载体。展示 planner 带来的跨 spec/BaB 复用与粗粒度化，这是单纯 kernel 优化做不到的。

2. **算子覆盖不全**

* 对策：像 auto_LiRPA 一样坚持“少量原语 + 分解”，优先覆盖 VNN-COMP 常见模型/激活。

3. **端到端收益被算法改进淹没**

* 对策：在相同 verifier/相同参数下做严格对比，报告 launch 次数、GPU 利用率、时间分解；把系统贡献讲清楚。

---

## 14. 你现在就可以复制到 README 的“Project Manifesto”

* **BoundFlow 的北极星指标**：同等验证强度下，端到端验证时间下降、可验证规模上升（更大模型/更多 spec/更深 BaB）。
* **BoundFlow 的工程约束**：前端尽量复用 auto_LiRPA 的自动 derivation 思路；后端以 TVM 实现；系统层以 planner/runtme 为核心。

---
好的，我把你这条“必须预留其它鲁棒性框架接口 + 量化 NN 接口”的要求，**补成一段可以直接粘进上一版文档的新增设计原则/架构扩展**（含接口形态）。我会用你点名的几类方法做锚点：DeepPoly、Zonotope、MIP/MILP、SMT，并把 QNN（位精确语义）单列出来。

---

## 新增设计原则：可插拔的 Abstract Domain 与 Complete Solver 后端（含 QNN）

### 设计动机

鲁棒性验证/认证训练并不是单一算法：工程上常需要在 **抽象解释域（DeepPoly、Zonotope 等）** 与 **完备求解（MILP/MIP、SMT、BaB）** 之间切换或组合。DeepPoly 是典型的抽象域验证系统之一。 Zonotope/DeepZono 在 ERAN 等工具中被作为专用抽象域使用。 完备验证常通过 MILP/MIP 与 BaB 框架实现（例如经典 MILP verifier 与基于 MIP 的 BaB 框架）。 SMT 方向也有代表性工具/线路（如 Reluplex）。
此外，**量化神经网络（QNN）** 的验证需要处理移位/舍入等位精确语义，已有专门面向 QNN 的验证研究与编码方法（如 QNN 的高效验证、以及将定点原语编码进 MILP 的工作）。

因此，BoundFlow 在 v1 主打 LiRPA/TIR 加速的同时，**架构上必须预留：**

1. 其它抽象域（DeepPoly / Zonotope / Star 等）
2. 完备 solver 后端（MIP/MILP / SMT / BaB）
3. 量化语义（QNN，bit-precise 或 fixed-point）

---
# 附录：

## A. 抽象域接口：`AbstractDomain`（DeepPoly / Zonotope / LiRPA 统一成“域插件”）

### 核心思想

把“计算边界”抽象为 **域状态（DomainState） + 变换器（Transformer）**。DeepPoly、Zonotope（DeepZono）本质上都是抽象解释域：用不同表示维护神经元之间依赖关系与上下界。

### 建议接口（BoundFlow IR 层）

* `DomainState`：域内部表示

  * 例：Interval 的 (l,u)，DeepPoly 的线性上下界形式，Zonotope 的仿射算术项等。
* `Transformer(op, state_in) -> state_out`：每个 primitive op 的抽象变换器

  * primitive op 控制在少量集合：Affine(Conv/Linear)、Activation(ReLU 等)、Pool、Reshape/Transpose。

**关键工程策略：**

* “新算子”优先在前端分解为 primitive op；
* 只对 primitive op 实现各域 Transformer（而不是对所有高层算子分别手写）。

> 这保证了未来接入 DeepPoly / Zonotope 时，你新增的是“域插件的一套 transformer”，而不是重写全套编译后端。

---

## B. 完备求解接口：`CompleteSolverBackend`（MIP/MILP / SMT / BaB）

### 为什么要单独抽象

MILP/MIP、SMT 与抽象域的执行形态完全不同：它们通常需要 **约束编码（encoding）** 和 solver 调用，而不是“跑一遍 tensor 计算图”。经典工作把 ReLU 网络验证写成 MILP 并配 presolve/紧化以提速；也有将 MIP 视角与 BaB 框架结合的完备验证框架。 SMT 路线同样强调约束求解（例如 Reluplex）。

### 建议接口（Lowering 层）

在 BoundFlow 中增加一条与 TVM 并行的 lowering：

* `BoundFlow IR -> Constraint IR`

  * Constraint IR 支持：

    * 线性约束、指示约束/Big-M、piecewise 约束；
    * bit-vector / fixed-point 原语（为 QNN 预留，见下一节）。
* `Constraint IR -> backend`

  * MILP/MIP：导出为标准形式（或直接调用 Gurobi/SCIP 等）
  * SMT：导出为 SMT-LIB（bit-vector / LRA / NRA 按需）

**与 Planner 的关系：**
Planner 不仅产出 “TVM BoundTask”，也可以产出 “ConstraintTask”（例如：对某些难验证样本切换到 MILP/SMT 或做 BaB 分裂），实现 *hybrid pipeline*：抽象域先 prune，再用完备求解补刀。

---

## C. 量化 NN 语义接口：`QuantizedSemantics`（QNN 必须“一等公民”）

### 为什么必须单独预留

QNN 验证的难点是：移位、舍入、饱和截断等 **定点/位精确语义**，通用实数域 verifier 不能直接覆盖。近期有专门面向 QNN 的验证方法；也有工作指出 MILP 需要显式编码定点原语（移位/舍入）才能验证 QNN。

### 建议的 IR 扩展点

* 在 BoundFlow IR 类型系统中，预留数值语义标签：

  * `Real32 / FP32`
  * `FixedPoint(qm, qn, round_mode, saturate_mode)`
  * `BitVector(width, signed)`
* 在 primitive op 中增加量化原语：

  * `Quantize`, `Dequantize`, `Round`, `Shift`, `Clamp/Saturate`
* 后端策略：

  1. **加速训练/不完备验证**：可继续走 TVM（把量化算子也当 tensor op 编译）
  2. **完备验证**：走 Constraint IR，确保 bit-precise/fixed-point 语义被正确编码（MILP/SMT）。

---

## D. 文档里建议加的一句“总原则”（你要的那条）

> **Extensibility原则**：BoundFlow 的 IR 与 planner 设计应预留对多种鲁棒性框架的接口（如 DeepPoly、Zonotope 等抽象域，以及 MIP/MILP、SMT 等完备求解后端），并将量化神经网络（QNN）的位精确/定点语义作为一等公民抽象，保证后续可通过插件化 domain/solver/semantics 扩展，而无需推倒重写核心编译与运行时架构。

---

如果你愿意，我可以把这段进一步“工程化”：给出更具体的 Python 接口草案（`AbstractDomain`、`CompleteSolverBackend`、`QuantizedSemantics` 三个抽象类 + 最小 required methods），方便你直接开 repo 写骨架。
