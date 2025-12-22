# BoundFlow 总设计文档（指挥者视角・口语版）

本文把 BoundFlow 的“核心理念 + 工程全流程（从 0 到可投顶会/可过 AE）”沿一条主线讲清楚，强调系统/编译/评测一体化视角，便于团队统一叙事与执行。

---

## 1) 核心 IDEA：把“鲁棒性/边界计算”当成可编译程序

BoundFlow 不是“再写一个 IBP”，而是把原本散落在 Python 中、难以系统优化与复现评测的边界传播流程，升级成：

- 可规划（Planner）：把计算拆成 task graph，明确依赖、分区、调度策略
- 可编译（TVM/VM）：让每个 task 具备可重复、可缓存的编译执行路径
- 可缓存（compile cache / VM cache）：降低 cold start，支撑大规模矩阵实验
- 可系统化消融：旋钮（partition/reuse/fusion/memory plan）可开关、可归因
- 可论文级证据：统一 JSONL schema，稳定产出图表与统计摘要

**结论**：这是系统结构（system + compiler + evaluation）的问题，而不只是算法实现。

---

## 2) 主干数据流（从 workload 到论文图表）

只记住这条流水线即可：

1. Workload 定义：模型 / 输入形状 / ε / 域（interval/IBP 等）/ spec
2. Planner：构建 task graph 并决定
   - task 划分与顺序（partition policy）
   - 存储复用（storage reuse）
   - memory plan mode
   - task fusion 策略
3. Executor：按 task 编译运行（TVM / VM / PackedFunc），维护各种 cache
4. Correctness / Baseline：对照 Python / auto_LiRPA，输出 abs/rel diff
5. Bench 记录：每个实验点输出一行 JSONL
6. Postprocess 产线：JSONL → CSV（逐点）→ Summary（分组汇总）→ Figures/Tables
7. Contract tests + docs：确保 schema 稳定、结果可复现、AE 可执行

Phase 5D 之前的工作，本质是把第 5~7 步焊死：让结果可复现、可解释、可审计。

---

## 3) 为什么严格要求 JSONL / stdout / contract test / warmup 拆分

### 3.1 JSONL：大矩阵实验的天然格式

- 每行一条 JSON，支持流式读取、并行处理、可追加写
- 为大规模实验日志与后处理而生

### 3.2 stdout 只放 payload，日志走 stderr

- stdout 要可被管道与重定向消费
- stderr 放进度/提示/错误，避免污染 JSONL 输出

### 3.3 schema contract test：把约定变成契约

- 任何字段变更立即在 CI 中暴露
- 防止跑完大矩阵后发现 postprocess 全挂

### 3.4 warmup vs steady-state：冷启动/稳态口径分离

- 首次运行包含 JIT/缓存冷启动噪声
- 拆分 compile_first_run 与 run_ms_p50/p95，口径清晰可解释

---

## 4) 全工程六阶段（从 0 到顶会/AE）

### Phase 0：问题定义与论文主张（Claims）

- 目标：把系统论文主张写成可测量的句子
- 交付物：claims + metrics + threats 一页草案

### Phase 1：语义正确性打底（Reference semantics）

- 目标：钉死“你要算的是什么”，提供可对照的 Python reference
- 交付物：reference + 单测 + correctness 指标（abs/rel diff）

### Phase 2：中间表示与任务化（Task Graph）

- 目标：把 reference 的程序变成可规划图（task 节点 + 依赖 + 形状/内存元信息）
- 交付物：Planner 核心数据结构 + 导出（num_tasks/edges/存储估计等）

### Phase 3：执行与编译（TVM Executor + caches）

- 目标：每个 task 稳定编译并运行，具备 cache 统计与失败诊断
- 交付物：可运行 executor + cache 统计 + compile/run 口径

### Phase 4：系统优化空间（论文贡献点）

- 目标：把系统结构性优化落地并做成独立开关（便于消融）
- 旋钮示例：partition / storage reuse / memory plan / fusion / pass 组合

### Phase 5：实验产线与系统化消融

- 5D（已完成）：JSONL schema 固化、stdout 清洁、contract test、postprocess pipeline
- 5E（未完成）：跑大矩阵并产出论文主图主表

### Phase 6：论文叙事 + Artifact 打包（AE 收口）

- 目标：外部人一键复现 claims
- 交付物：README + artifact appendix + run.sh + results/

---

## 5) 当前所处位置（一句话）

你已完成 **Phase 5D（产线基础设施）**，但 **Phase 5E（论文主结果）** 与 **Phase 6（论文/AE 打包）**尚未开始收口。

---

## 6) 预留：给“实现说明大模型”的 Prompt

待你提供 Prompt 后，我会在此补充并对齐仓库结构与模块实现路径。
