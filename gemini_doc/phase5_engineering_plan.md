# Phase 5 工程计划（对齐 `docs/p4_p5.md` + `docs/p5.md`，并以当前仓库实现为基线）

## 0. 现状（Phase 4 已完成到“可进入 Phase 5”的水平）

以当前仓库为准，你的 Phase 4 关键地基已经齐了：

- **Primal IR + Torch/ONNX 前端**：`torch.export` 导入与 ONNX shape_infer 导入已能落到 Primal IR，并接入同一套 normalize/planner/executor。
- **Task pipeline**：`plan_interval_ibp_v0` 将整图降为 TaskOps；有 `StoragePlan` schema（后续可做 liveness/reuse）。
- **Spec/Property**：`LinearSpec(C)` 已对齐 auto_LiRPA（融合到最后 linear 优先，fallback `spec_linear` 语义正确）。
- **Layout-only 起步**：已有 `permute` 简化 pass（identity 消除 + 连续 permute 合并）。
- **TVM demo**：已有 TVM interval `linear` 与 `conv2d` kernel（目前默认用 Relax VM + Relax op 实现，legacy TE demo 仍可选），且在 MLP/CNN 上做到了 `TVMTaskExecutor == PythonTaskExecutor`，并通过三方闭环（auto_LiRPA / Python / TVM）。

因此 Phase 5 的主线可以从“编译系统能力（Planner/调度/内存/缓存）”开始推进，而不是继续补 Phase 4 的 correctness。

---

## 1. Phase 5 总目标（工程口径）

把当前 **“整图单 task + reference executor + IBP correctness + 局部 TVM kernel”** 升级为：

1. **Global Planner v1**：输出 `TaskGraph`（多 task DAG）+ `StoragePlan v1`（liveness + reuse）+（可选）`CachePlan`。
2. **可扩展的 Domain/Op 规则体系**：避免 `IntervalDomain` 变成巨无霸，为 CROWN/DeepPoly/Zonotope 等域预留一致接口。
3. **最小 CROWN-IBP 闭环 + cache/reuse**：能对齐/对照 auto_LiRPA 的 `compute_bounds(method=..., C=..., reuse/cache...)` 语义，并具备可观测 cache hit。
4. **TVM 端到端性能闭环**：在“多 task + cache + batching”场景下，TVMExecutor 相对 PythonExecutor 有稳定性能收益（至少在 MLP/CNN）。

> BaB（α,β-CROWN 风格）建议放到 Phase 6；Phase 5 的重点是把“可重复 bound propagation”的系统骨架做好。

---

## 2. Phase 5 的“Pass 插槽”（科研/工程共用的稳定接口）

你在 `docs/p5.md` 里新增的思路非常关键：Phase 5 不能只写“要做什么功能”，还要把“未来可能成为论文创新点”的位置做成 **pass pipeline 的插槽**，否则系统跑起来后很难做系统化消融。

建议把 Phase 5 的优化空间显式拆成这些 pass slot（先 baseline，后续逐步启用）：

1. `property_rewrite`：Spec/Property folding（你已做 last-linear C 融合；Phase 5 扩展为多策略可对比）
2. `domain_assign`：Domain mix（IBP / CROWN-IBP / backward / future zono 等）
3. `partition_and_fusion`：Task 切分与融合（减少 kernel launch、提升算子密度）
4. `cache_and_reuse`：CachePlan/ReusePolicy（对齐 auto_LiRPA 的 reuse/cache 语义）
5. `layout_opt`：layout-only 消除/下推/合并（在 Phase 4B.3 的基础上继续扩展）
6. `objective_and_search`：cost model + 小规模搜索（用于系统化消融与科研探索）

---

## 3. Phase 5 分阶段计划（5A–5F）

### 5A：TaskGraph & Scheduler（从“单 task”到“多 task DAG”）

**动机**
- Phase 5 的所有优化（reuse/cache/batching/部分 TVM）都需要“图级调度对象”，不能继续用单 task 承载所有语义。

**交付内容**
- 将 planner 升级为“pass pipeline”，并输出一个可序列化的 `PlanBundle`（推荐命名/结构，便于长期扩展）：
  - `plan(program, spec, config) -> PlanBundle`
  - `PlanBundle` 至少包含：`task_graph`、`storage_plan`、（可选）`cache_plan`、（可选）`layout_plan`、`lowering_plan`
- 新增 IR：`boundflow/ir/task_graph.py`
  - `TaskGraph(tasks, edges)`、`topo_sort()`、`validate()`
- 扩展 `BFTaskModule`（保持兼容）
  - 增加可选字段 `task_graph: Optional[TaskGraph]`（若为空则视为单 entry task）
- 新增 runtime：`boundflow/runtime/scheduler.py`
  - 按 topo 顺序执行 task（先串行）
  - 对接 `PythonTaskExecutor` / `TVMTaskExecutor`（作为 task 级执行器）
- 新增 planner：`boundflow/planner/interval_v2.py`（或 `planner/passes/partition.py`）
  - 先用简单切分：`layout-only region` / `compute region` / `spec region`

**测试 & DoD**
- `tests/test_phase5a_taskgraph_equivalence.py`
  - 同一模型/输入/eps：v0（单 task）与 v2（多 task DAG）输出一致（IBP +（可选）C）
- DoD
  - 现有 Phase 4 测试全部通过
  - TaskGraph `validate()` 能抓出缺边/环/缺 buffer 等错误

---

### 5B：StoragePlan v1（liveness + buffer reuse）

**动机**
- 你已有 StoragePlan schema，但目前是“一值一 buffer”；Phase 5 必须把它升级为可复用的内存计划，否则 BaB/多 spec 会在峰值内存上崩。

**交付内容**
- 新增 pass：`boundflow/planner/passes/liveness.py`
  - 计算每个 value/buffer 的 `first_def / last_use`
- 新增 pass：`boundflow/planner/passes/buffer_reuse.py`
  - 基于 `(dtype, shape, device, layout, alignment)` 的 reuse pool
  - 支持 `alias_group`（先占位，后续更精细复用）
- 扩展 `BufferSpec`（不破坏兼容）
  - 增加 `lifetime` / `liveness_group`（或保持在 `meta` 字段里）

**测试 & DoD**
- `tests/test_phase5b_peak_memory_reduction.py`
  - 对比 v0 vs v1：统计峰值 bytes/峰值 buffer 数（目标先定：下降 ≥20%）
- DoD
  - 输出 bounds 与 v0 一致
  - 能从统计/日志确认 reuse 真实发生

---

### 5C：Domain/Op 规则插件化（为 CROWN/DeepPoly 等域做接口钉子）

**动机**
- 现在的 `IntervalDomain` 是“domain + op 分发 + 算子实现”揉在一起；Phase 5/6 会让它迅速膨胀。

**交付内容**
- 新增 `OpRule`/registry（建议先从 interval 域迁移）
  - `boundflow/domains/rules/base.py`：`OpRule.apply(domain_state, inputs, attrs) -> domain_state`
  - `boundflow/domains/rules/interval/*.py`：`linear/conv2d/relu/add/mul/reshape/flatten/permute/spec_linear`
  - `boundflow/domains/registry.py`：`(domain_id, op_type) -> rule`
- `IntervalDomain` 退化为：
  - state 定义 + 规则调用入口（不再写巨型 if-else）

**测试 & DoD**
- 复用现有 Phase 4 全部对齐测试（MLP/CNN/Spec/TVM）
- DoD
  - 规则迁移后，输出不变
  - 新增一个 “unknown op” 报错测试（防 silent fallback）

---

### 5D：最小 CROWN-IBP 闭环（先能跑+对齐，再追求更紧）

**动机**
- Phase 5 的学术/工程价值在于：不仅能做 IBP，还能做 backward LiRPA（至少 CROWN-IBP），并可对照 auto_LiRPA。

**交付内容（最小可用版本）**
- 域与状态：
  - `boundflow/domains/linear.py`：`LinearBoundState`（A,b 或等价表示）
- Planner：
  - `boundflow/planner/crown_v0.py`：生成 backward bound 的 task（从 spec 输出往回）
  - 支持 `LinearSpec(C)` 作为 backward 初始 A
- Runtime：
  - 扩展 task 执行器以支持 LinearBoundState 的核心算子（先 PyTorch reference）

**测试 & DoD**
- `tests/test_phase5d_crown_ibp_against_auto_lirpa.py`
  - MLP +（可选）小 CNN：对齐 auto_LiRPA 的 backward/CROWN 路径（固定配置，先追求稳定）
- DoD
  - CROWN-IBP bound 不劣于 IBP（更紧或相等）
  - 与 auto_LiRPA 结果在容忍误差内可复现

---

### 5E：CachePlan / Reuse / Batching + TVM 端到端闭环

**动机**
- auto_LiRPA 的 `reuse_ibp/reuse_alpha/cache_bounds` 是核心性能接口；你需要把它升级为“编译系统级能力”。
- TVM 后端方面：你已明确不再把 TE 当主线，因此 Phase 5 的 baseline 建议以 Relax 为入口（先不要求你手写 TIR）。

**交付内容**
- Cache：
  - `boundflow/runtime/cache.py`：`BoundCache`（LRU + 统计）
  - cache key：至少包含 `(task_id, domain_kind, spec_hash, input_hash/eps, split_state_hash, layout_hash)`
- Batching（建议先做 property batch）：
  - `boundflow/runtime/batch_runner.py`：同图多 `C` 一次跑（先从 `LinearSpec(C)` batch 开始）
- TVM：
  - **baseline（建议先做）**：沿用当前 `TVMTaskExecutor` 的 Relax VM 路线（用 Relax op 表达 interval kernel，交给 TVM 内部 legalize/lower；工程侧不手写 TE/TIR）
    - 现状约束：该 TVM fork 没有 `tvm.nd`，运行时需要继续使用 `tvm.runtime._tensor`
  - **进阶（可选，用于性能/论文）**：引入 `call_tir` 的 mixed module 路线
    - 方式 A：你自己生成 mixed IRModule（Relax wrapper + PrimFunc），绕开 TE（更利于做 fusion/schedule）
    - 方式 B：映射到 Relax op，再用 custom legalization 接管（工程复杂度更高）
  - （可选）补齐 `batched linear (w:[B,O,I])` 的 TVM kernel：覆盖 spec 融合后的关键路径，并为 property batching 做准备
- Bench：
  - `benchmarks/phase5_end2end.py`（或沿用 `scripts/bench_phase4c_tvmexecutor.py` 扩展）
  - 输出：wall time / cache hit / peak memory（来自 StoragePlan）

**测试 & DoD**
- `tests/test_phase5e_cache_hits.py`：重复运行有 cache hit，且输出一致
- `tests/test_phase5e_property_batching.py`：多 spec batch 与逐个运行输出一致
- DoD
  - TVMExecutor 在 MLP/CNN 上端到端速度稳定优于 PythonExecutor（先设目标 1.5×–3×）

---

### 5F（建议新增）：Objective/Cost Model/Search + Ablation Harness（科研交付必备）

**动机**
- 你在 `docs/p5.md` 里提到 “Planner 创新要系统化”，最关键落点就是：任何 pass 的 on/off 都能自动跑消融并产出数据。

**交付内容**
- `boundflow/planner/objectives.py`
  - `Objective = w_time*time + w_mem*peak_mem + w_tight*tightness + w_compile*compile_time`
- `boundflow/planner/search/`
  - baseline：`greedy_search.py`
  - 可选：`beam_search.py`；预留 `cp_sat_search.py`
- `benchmarks/planner_ablation.py`
  - 一键跑 N 个 planner config（domain mix / partition 粒度 / cache on-off / layout opt on-off）
  - 产出 CSV：`time, peak_mem, tightness, #kernels, compile_time`

**测试 & DoD**
- DoD：任何一个 pass 的开关都能被框架捕获并产出可复现结果；至少对 MLP/CNN 能输出一份可画图的 ablation CSV。

---

## 4. 推荐的 PR/提交拆分（降低返工风险）

1. PR#1：`TaskGraph` + `scheduler`（5A 基础）
2. PR#2：`interval_v2` partition + 多 task DAG 输出（5A）
3. PR#3：liveness + reuse（5B）
4. PR#4：domain rule registry（5C）—— 这是 Phase 5“可扩展性”关键钉子
5. PR#5：CROWN-IBP 最小可跑 + 对齐测试（5D）
6. PR#6：cache + property batching（5E-1）
7. PR#7：TVM 端到端闭环（5E-2：Relax baseline + benchmark；batched linear 可选）
8. PR#8：objective/search/ablation harness（5F）

---

## 5. Phase 6（预告，不在 Phase 5 实现）

BaB（β-CROWN 风格）作为 Phase 6：

- `SplitState` / `ConstraintStore` IR + runtime 入口
- “控制流在 runtime、bound propagation 在 compiled kernels” 的执行模型
- 复用 Phase 5 的 cache/batching/taskgraph/storageplan 能力
