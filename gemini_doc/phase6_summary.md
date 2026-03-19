# Phase 6 总结：三轴解耦落地到 αβ-CROWN + BaB（含可复现/可归因工件链）

> 本文总结 Phase 6 的“计划 → 实现 → 测试/基准 → AE/论文交付形态”全链路成果，便于：
>
> - 写论文/答辩时快速解释：**语义锚点在哪里、系统贡献是什么、收益如何归因**；
> - 新同学接手时快速定位：**代码入口/关键接口/回归钉子/已知限制**；
> - 做 AE/复现实验时一键跑通：**JSONL/CSV/fig 工件 + schema 固化 + 失败可审计**。
>
> 主要依据：
> - `docs/change_log.md`（总账）
> - `gemini_doc/change_2025-12-22_phase6a_*.md` ~ `gemini_doc/change_2025-12-31_phase6h_pr7_*.md`（分 PR/阶段记录）
> - Phase 6 计划/评审：`gemini_doc/phase6_review_three_axis_stage_pipeline.md`、`gemini_doc/phase6f_beta_crown_mlp_plan.md`
> - 关键代码：`boundflow/runtime/{perturbation,crown_ibp,alpha_crown,alpha_beta_crown,bab}.py`、`scripts/bench_phase6h_bab_e2e_time_to_verify.py` 等。

---

## 1. Phase 6 解决什么问题（目标与边界）

Phase 6 的目标可以概括为一句话：

> **在不重新发明算法的前提下，把 LiRPA 家族（IBP/CROWN/α/β）与 BaB 的“语义闭环 + 系统化收益 + 可复现证据链”落到 BoundFlow 的分层实现里。**

### 1.1 计划基线：三轴解耦 + stage pipeline

Phase 6 的顶层设计来自两份核心设计文档：

- `gemini_doc/perturbation_support_design.md`：输入扰动（Lp 球等）与线性形式 `concretize` 的统一抽象；
- `gemini_doc/bound_methods_and_solvers_design.md`：`Perturbation × BoundMethod/DomainState × Solver` 三轴解耦，以及 `Forward/Backward/Optimize` 的 stage pipeline 组合思路。

落地时的关键工程原则在 `gemini_doc/phase6_review_three_axis_stage_pipeline.md` 中被“评审式固化”：

- **控制流留在 Python runtime**（BaB 的队列/分支/剪枝），把重算留给张量核/后端；
- **batching 优先级**：先 multi-spec，再 multi-subproblem（node-batch）；
- **避免接口爆炸**：扰动不进 primal IR；方法组合用 pipeline，而不是 Domain 类继承爆炸；
- **正确性 DoD 优先**：每一步都要有可运行的回归钉子（符号选择、梯度链路、K=1 等价、cache hit 不重算…）。

### 1.2 Phase 6 的“完成定义”（Done Definition）

Phase 6 最终用一套可审计的 DoD 闭环定义“完成”：

1. **语义闭环**：从 `InputSpec + PerturbationSet` 到 CROWN-IBP、α/β 优化、BaB（至少链式 MLP 子集）可跑通；
2. **complete 支点回到 β**：split 约束不靠 patch/假设，而能被 β 编码进 bound propagation，并在 BaB 中形成可剪枝/可证明的闭环（以 Phase 6F 的回归钉子固化口径）；
3. **系统收益可归因**：multi-spec 真 batch、node-batch、cache、branch-hint、infeasible prune 的收益能拆开对照（开关矩阵 + 固化 counters）；
4. **可复现工件链**：E2E time-to-verify 产出 JSONL/CSV/fig，并固化 schema/meta、可比性标注、失败记录与环境审计；一条命令可跑主结果（Kick-the-tires）。

---

## 2. 阶段回顾：6A → 6H（计划对照实现）

> 下面每个小节都对应 `gemini_doc/change_*.md` 与 `docs/change_log.md` 的一条或多条记录；这里按“计划顺序”串起来讲。

### 6A：扰动集合解耦（InputSpec + LpBallPerturbation）

**目的**：把输入扰动从 `LinfInputSpec` 的 box 固化状态解耦出来，为后续 CROWN/αβ/BaB 建立统一 `concretize` 入口。

**关键产物**

- `boundflow/runtime/perturbation.py`：`PerturbationSet` + `LpBallPerturbation(p∈{∞,2,1})` + `concretize_*`；
- `boundflow/runtime/task_executor.py`：引入 `InputSpec(value_name, center, perturbation)` 并兼容 legacy `LinfInputSpec`。

**回归钉子**

- `tests/test_phase6a_inputspec_lpball_linear.py`（LpBall 的线性层 soundness + 兼容性）。

对应记录：`gemini_doc/change_2025-12-22_phase6a_inputspec_lpball_perturbation.md`。

### 6B：最小 CROWN-IBP（MLP Linear+ReLU）

**目的**：跑通 “forward 用 IBP 给 ReLU pre-activation 区间 → backward 传线性界 → 输入处 concretize” 的最小闭环，并把最容易写错的点用测试钉死。

**关键产物**

- `boundflow/runtime/crown_ibp.py`：`run_crown_ibp_mlp(...)`（single-task、linear/relu 子集）；
- `boundflow/runtime/perturbation.py`：`concretize_affine(center, A, b)`（对偶范数实现落点）；
- multi-spec 入口 `linear_spec_C:[B,S,O]`（为 6C/6G 铺路）。

**回归钉子（reviewer-proof）**

- `p=1` 覆盖（dual norm 的 row-wise `L∞`）；
- brute-force 小维度兜底（防止 ReLU backward 符号选择写反）；
- multi-spec batch vs serial 一致性对拍；
- 非链式拓扑 gate（防 silent wrong）。

对应记录：`gemini_doc/change_2025-12-23_phase6b_crown_ibp_mlp_minimal.md`、`gemini_doc/change_2025-12-23_phase6b_crown_ibp_mlp_hardening.md`。

### 6C：multi-spec 真 batch（吞吐）+ forward 复用回归

**目的**：把第一波系统收益做成“可测、可复现、可归因”的证据：**forward IBP 不随 spec 数 S 增长**，backward 在 `[B,S,...]` 上向量化。

**关键产物**

- `scripts/bench_phase6c_crown_ibp_multispec_throughput.py`：batch vs serial microbench；
- `tests/test_phase6c_crown_ibp_multispec_batch.py`：monkeypatch 计数，断言 forward work 与 S 无关。

对应记录：`gemini_doc/change_2025-12-23_phase6c_multispec_true_batch_microbench_and_reuse_test.md`（以及计时口径补强文档）。

### 6D：α-CROWN（可优化松弛）+ warm-start 形态

**目的**：把不稳定 ReLU 的 lower relaxation 从固定 baseline 升级为可优化参数 `α`，跑通 K-step 优化闭环，并把 warm-start 作为一等公民落地（为 BaB 继承/复用铺路）。

**关键产物**

- `boundflow/runtime/alpha_crown.py`：`run_alpha_crown_mlp(...)` + `AlphaState`；
- `scripts/bench_phase6d_alpha_opt_convergence.py`：收敛轨迹可观测。

**回归钉子**

- α 梯度链路（non-empty/finite/non-zero）；
- best-of（含 step=0）不回退；
- warm-start 不劣；
- sampling soundness。

对应记录：`gemini_doc/change_2025-12-23_phase6d_alpha_crown_mlp_alpha_opt_and_warm_start.md`。

### 6E：BaB 骨架（控制流在 Python）+ split state 透传

**目的**：把 bound oracle（α-CROWN）接入 BaB（priority queue / split / prune），形成最小搜索闭环，并把 split state 作为运行时一等状态传递。

**关键产物**

- `boundflow/runtime/bab.py`：`ReluSplitState` + `solve_bab_mlp(...)`；
- `_forward_ibp_trace_mlp(...)` 抽取（供分支选择复用）。

**重要口径**

- 6E 的 “complete” 仅是 toy 演示；通用 complete 需要把 split 约束编码进 bound propagation（引出 6F）。

对应记录：`gemini_doc/change_2025-12-23_phase6e_bab_mlp_split_state_and_driver.md`、`gemini_doc/phase6f_beta_crown_mlp_plan.md`。

### 6F：β/αβ-CROWN（把 complete 支点落回 β 编码）

**目的**：将 split 约束从 patch/假设升级为 **β 形式编码进 bound propagation**，并让 BaB 的 sound/complete 叙事“锚”回到 β（而不是工程补丁）。

**关键产物**

- `boundflow/runtime/alpha_beta_crown.py`：`run_alpha_beta_crown_mlp(...)`（α/β 可微优化 + split 注入 + feasible/infeasible）；
- `boundflow/runtime/bab.py`：BaB driver 可切换 oracle（`alpha` vs `alpha_beta`）。

对应记录：`gemini_doc/change_2025-12-23_phase6f_pr1_*.md`、`gemini_doc/change_2025-12-23_phase6f_pr2_*.md`。

### 6G：纯系统收益（不动语义）：multi-spec αβ + node-batch + cache/归因

**目的**：在 6F 的语义闭环基础上，系统化提速并把收益拆干净（开关矩阵 + 固化 counters）。

**关键产物（核心代码）**

- `boundflow/runtime/bab.py`：
  - node-batch（一次 eval K 个节点）
  - `NodeEvalCache`（split+config+spec 的 node eval cache）
  - branch-hint 复用 forward trace（消掉分支二次 forward）
  - per-node infeasible + batch 内 partial prune
- `scripts/bench_phase6g_bab_node_batch_throughput.py`：吞吐 microbench（开关矩阵 + counters）

对应记录：`gemini_doc/change_2025-12-24_phase6g_pr1_*.md` ~ `gemini_doc/change_2025-12-26_phase6g_pr4_*.md`。

### 6H：E2E time-to-verify（AE/论文交付形态闭环）

**目的**：把“局部 node-eval 吞吐”升级为“端到端 time-to-verify”，并产出可直接用于 AE/论文的工件：JSONL→CSV→fig + 一键 runner + schema/meta 固化 + 失败可审计 + claims 映射。

**关键产物**

- E2E bench：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - 输出 `{meta,rows}`（开关矩阵 2×2×2）
  - `comparable/note/note_code` 防止拿不可比数据画图
  - 固化 counters（oracle_calls/forward_trace/cache_hit_rate/pruned_infeasible/evaluated_nodes 等）
  - schema v2：`phase6h_e2e_v2`（新增 `p90/p99`、`runs/valid_runs/timeouts`、`speedup_p90`）
- sweep/report/plot：`scripts/{sweep,report,plot}_phase6h_e2e.py`
- 一键复现：`scripts/run_phase6h_artifact.sh`
- AE README：`gemini_doc/ae_readme_phase6h.md`

对应记录：`gemini_doc/change_2025-12-29_phase6h_pr1_*.md` ~ `gemini_doc/change_2025-12-31_phase6h_pr7_*.md`。

---

## 3. 代码落点与关键接口（“从哪看/怎么用”）

### 3.1 运行时核心（`boundflow/runtime/`）

- `boundflow/runtime/perturbation.py`
  - `LpBallPerturbation`：`bounding_box / concretize_matmul / concretize_affine`
- `boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(...)`：CROWN-IBP（Linear+ReLU），支持 `linear_spec_C`
- `boundflow/runtime/alpha_crown.py`
  - `AlphaState` / `run_alpha_crown_mlp(...)`：α 优化、best-of、warm-start
- `boundflow/runtime/alpha_beta_crown.py`
  - `BetaState` / `run_alpha_beta_crown_mlp(...)`：β 编码 + 可微优化 + feasible/infeasible
- `boundflow/runtime/bab.py`
  - `ReluSplitState` / `BabConfig` / `solve_bab_mlp(...)`
  - node-batch / NodeEvalCache / branch-hint / infeasible partial prune

### 3.2 复现/基准/工件（`scripts/`）

- 6C：`scripts/bench_phase6c_crown_ibp_multispec_throughput.py`
- 6G：`scripts/bench_phase6g_bab_node_batch_throughput.py`
- 6H（E2E）：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
- 工件链：
  - `scripts/sweep_phase6h_e2e.py` → `scripts/report_phase6h_e2e.py` → `scripts/plot_phase6h_e2e.py`
  - `scripts/run_phase6h_artifact.sh`（一键）

### 3.3 测试（`tests/`）

Phase 6 的测试体系遵循“语义钉子优先”的原则，覆盖：

- 正确性：sampling soundness / brute-force / multi-spec 对拍 / K=1 等价 / 梯度链路 / warm-start
- 系统化：forward 复用计数 / cache hit 不重算 / node-batch 梯度隔离 / schema contract tests
- 工件链：report/plot/runner 的 smoke tests

并且通过 `pytest.importorskip(...)` 做到：缺 `onnx/tvm/auto_LiRPA` 时 **不会在 collection 阶段崩**（AE/CI 友好）。

---

## 4. “系统收益如何归因”的证据链（给论文/答辩用）

Phase 6 最终把收益拆成三类可开关机制，并用 counters 固化“因果链路”：

1. **multi-spec 真 batch**：forward 不随 S 增长（减少重复 forward）
2. **node-batch**：一次评估 K 个节点（提高硬件利用率）
3. **reuse/剪枝**：
   - `NodeEvalCache` 降低 `oracle_calls`
   - `branch-hint` 降低 `forward_trace_calls`（避免分支二次 forward）
   - `infeasible partial prune` 降低 `evaluated_nodes_count`

Phase 6H 的 E2E bench 把这些指标与 time-to-verify 放在同一 JSONL row 里，形成 “claim → 字段 → 图表” 的闭环（见 `gemini_doc/ae_readme_phase6h.md`）。

---

## 5. Phase 6 Done：怎么验证“全绿”

建议在 conda 环境 `boundflow` 下：

```bash
# 全量回归（缺 tvm/onnx/auto_LiRPA 时相关用例会 skip）
python -m pytest -q tests

# 一键产物（Kick-the-tires）
bash scripts/run_phase6h_artifact.sh /tmp/phase6h_artifact_run
```

输出产物与 claims 映射见：`gemini_doc/ae_readme_phase6h.md`。

---

## 6. 已知限制与 Phase 7 入口（不在 Phase 6 范围内）

Phase 6 明确把 scope 控制在“链式 MLP（Linear+ReLU）子集 + 可复现证据链”，因此仍有若干限制：

- 算子/图结构：不支持 conv、skip/branch 一般图（需要 Phase 7 扩图）
- backward 线性形式：当前实现仍以显式张量为主；`LinearOperator` 方向未在 Phase 6 收敛（未来要做 CNN 规模必做）
- 后端：BaB 控制流在 Python；TVM/编译提速主要留给后续核下沉（与 Phase 6 “不动语义先做系统收益闭环”一致）

Phase 7 最自然的两条线：

1. **扩图（skip/branch）**：先 sound gate + 明确不支持时报错策略；
2. **扩算子（Conv）**：在保持 CROWN/αβ 语义的前提下扩 conv 的线性界传播与 concretize 抽象。

