# BoundFlow 全流程：从研究主张到工程实现到论文/AE交付（Phase 0~6 对齐版）

> - **文档版本**：v2.0（覆盖 Phase 0~6；同时对齐 Phase 5 与 Phase 6 的两套工件链）
> - **最后更新**：2026-01-03
> - **代码版本**：`c47e434718137ba66620deb03ede348be700b3bc`
> - **对应状态**：
>   - Phase 5：完成声明（`docs/phase5_done.md`；bench JSONL `schema_version=1.0`）
>   - Phase 6：完成声明（`gemini_doc/phase6_summary.md`；E2E JSON `schema_version=phase6h_e2e_v2`）

这份文档是“总导览”，把 BoundFlow 从 **研究主张（claims）** → **工程管线（IR/Planner/Runtime/Backends）** → **论文/AE 工件链（bench/sweep/report/plot/runner）** 串成一条可审计主线，并把 Phase 0~6 的真实落点（代码/脚本/文档）对齐到同一叙事里。

如果你只想“一键复现主结果”，优先看：

- Phase 5 artifact：`docs/phase5_done.md`
- Phase 6H artifact（E2E time-to-verify）：`gemini_doc/ae_readme_phase6h.md`

---

## 1) 这份文档覆盖的两条“可复现主线”

BoundFlow 到 Phase 6 形成了两条并行但互相对齐的主线（它们共享“工程分层”思想，但产物与 schema 不同）：

1. **编译/系统主线（Phase 0~5）**：面向 interval IBP（含 conv），核心是“Task IR + Planner + TVM backend + JSONL schema 1.0 + artifact runner”。
2. **验证器主线（Phase 6）**：面向 LiRPA（CROWN-IBP/α/β）+ BaB（complete 支点回到 β 编码），核心是“oracle + BaB control-flow + E2E time-to-verify 工件链 + AE 交付形态（schema v2）”。

> 重要边界：Phase 6 的 CROWN/αβ/BaB **仅覆盖链式 MLP（Linear+ReLU）子集**，当前未实现 conv（详见 `gemini_doc/phase6_summary.md` 的“已知限制”）。

---

## 2) 一张图看清“从模型到工件”的两条流水线

### 2.1 Phase 5（interval IBP + TVM）工件链（schema 1.0）

> 注：部分 Markdown 渲染器不支持 Mermaid；这里用纯文本图，保证在任意 Markdown 环境可正常显示。

```text
Workload/Model (torch/onnx)
  -> Frontend (import_torch / import_onnx)
  -> Primal IR (BFPrimalProgram/Graph)
  -> Planner (Task IR + TaskGraph + StoragePlan)
  -> Scheduler (run_ibp_scheduled)
      -> PythonTaskExecutor (reference)
      -> TVMTaskExecutor (compile/cache/run)
  -> bench_ablation_matrix.py (stdout JSONL, schema_version=1.0)
  -> postprocess_ablation_jsonl.py (CSV/fig)
  -> run_phase5d_artifact.py (artifact 目录)
```

**证据链入口**
- 完成声明与复现命令：`docs/phase5_done.md`
- claims→字段→文件：`gemini_doc/artifact_claims_phase5d.md`
- schema 文档：`docs/bench_jsonl_schema.md`（`schema_version=1.0`）

### 2.2 Phase 6（αβ oracle + BaB）工件链（schema v2）

```text
Workload/Model (MLP Linear+ReLU)
  -> InputSpec + PerturbationSet (LpBall concretize)
  -> αβ-CROWN oracle (multi-spec + warm-start)
  -> BaB solver (node-batch + cache + prune)
  -> bench_phase6h_bab_e2e_time_to_verify.py (stdout JSON, schema_version=phase6h_e2e_v2)
  -> sweep_phase6h_e2e.py (append JSONL)
  -> report_phase6h_e2e.py (CSV + summary.md)
  -> plot_phase6h_e2e.py (figs)
  -> run_phase6h_artifact.sh (一键产物 + 环境审计)
```

**证据链入口**
- Phase 6 总结（含完成定义/限制）：`gemini_doc/phase6_summary.md`
- AE/论文复现入口（Kick-the-tires + Claims 映射）：`gemini_doc/ae_readme_phase6h.md`
- E2E schema：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`（`meta.schema_version=phase6h_e2e_v2`）

---

## 3) “Claims → 命令 → 产物 → 字段”的对照入口

BoundFlow 的“主张”不是写在一个抽象 `claims.md` 里，而是通过 **完成声明 + claims 映射 + 固化 schema + contract tests** 闭环：

- Phase 5（系统消融/编译收益）：
  - 完成声明：`docs/phase5_done.md`
  - claims 映射：`gemini_doc/artifact_claims_phase5d.md`
  - schema：`docs/bench_jsonl_schema.md`（`schema_version=1.0`）
- Phase 6（complete verifier + E2E time-to-verify）：
  - AE README/claims 映射：`gemini_doc/ae_readme_phase6h.md`
  - E2E 输出：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`（包含 `comparable/note/run_status/error` 等“可审计字段”）

这两条链路都遵循同一个 AE 友好原则：

1. **stdout 只输出 payload**（JSON/JSONL），日志走 stderr（避免污染后处理）。
2. **schema contract test 守住字段**（比“跑得更快”更能长期复现）。
3. **不可比/失败显式记录**（`comparable/note/run_status=error`），禁止 silent failure。

---

## 4) 工程分层与关键构件（对照 Phase 4/5/6 的落点）

这一节用于回答：BoundFlow“到底是怎么从模型走到可执行任务形态”，以及 Phase 6 的 verifier 主线放在分层里的哪里。

### 4.1 Frontend / Normalize / Primal IR（模型语义层）

- Primal IR：`boundflow/ir/primal.py`
- Torch frontend：`boundflow/frontends/pytorch/frontend.py`
- ONNX frontend：`boundflow/frontends/onnx/frontend.py`
- Normalize：`boundflow/frontends/normalize.py`

对应阶段总结：`gemini_doc/phase1_summary.md`、`gemini_doc/phase2_summary.md`、`gemini_doc/phase4_summary.md`。

### 4.2 Task IR + TaskGraph + StoragePlan（可执行任务形态）

这是 Phase 4/5“编译/系统主线”的核心契约层：

- Task IR：`boundflow/ir/task.py`
- TaskGraph：`boundflow/ir/task_graph.py`
- Planner 入口：`boundflow/planner/pipeline.py`
- Storage/Reuse passes：`boundflow/planner/passes/`

对应阶段总结：`gemini_doc/phase4_summary.md`、`gemini_doc/phase5_summary.md`。

### 4.3 Runtime：Scheduler/Executor（reference vs TVM）

- Scheduler：`boundflow/runtime/scheduler.py`
- Reference executor：`boundflow/runtime/task_executor.py`、`boundflow/runtime/executor.py`
- TVM executor：`boundflow/runtime/tvm_executor.py`

对应阶段总结：`gemini_doc/phase5_summary.md`。

### 4.4 Phase 6 verifier 主线（oracle + solver + 工件链）

Phase 6 的“验证器主线”当前主要落在 runtime 层（还没完全接入 Task IR/TVM 编译路径），核心文件：

- 扰动/输入集合：`boundflow/runtime/perturbation.py`
- CROWN-IBP：`boundflow/runtime/crown_ibp.py`
- α-CROWN：`boundflow/runtime/alpha_crown.py`
- αβ-CROWN：`boundflow/runtime/alpha_beta_crown.py`
- BaB（控制流）：`boundflow/runtime/bab.py`
- E2E bench：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`

对应阶段总结：`gemini_doc/phase6_summary.md`。

---

## 5) Phase 0~6：阶段目标与“从哪开始读”

> 每一阶段的详细总结都已落地到 `gemini_doc/phaseX_summary.md`，这里仅提供导航。

- Phase 0：工程止血与最小可跑基线 → `gemini_doc/phase0_summary.md`
- Phase 1：Primal IR 加固（Node/Value + validate）→ `gemini_doc/phase1_summary.md`
- Phase 2：TorchFrontend（torch.export → Primal IR）→ `gemini_doc/phase2_summary.md`
- Phase 3：IBP reference + auto_LiRPA 对齐（MLP/CNN）→ `gemini_doc/phase3_summary.md`
- Phase 4：Task/Planner/Executor/Spec/TVM/ONNX 闭环 → `gemini_doc/phase4_summary.md`
- Phase 5：bench→JSONL→postprocess→artifact 产线（schema 1.0 冻结）→ `gemini_doc/phase5_summary.md`
- Phase 6：αβ-CROWN + BaB 语义闭环 + 系统收益归因 + AE 工件链（schema v2）→ `gemini_doc/phase6_summary.md`

总账（按时间追踪每次 PR/变更）：`docs/change_log.md`。

---

## 6) 快速复现入口（建议在 conda env: boundflow）

### 6.1 Phase 5（schema 1.0）一键产物

见 `docs/phase5_done.md`；核心入口是 `scripts/run_phase5d_artifact.py`。

### 6.2 Phase 6H（schema v2）一键产物（E2E time-to-verify）

见 `gemini_doc/ae_readme_phase6h.md`；核心入口是 `scripts/run_phase6h_artifact.sh`。

---

## 7) 已知限制（对照 Phase 7 入口）

1. **Phase 6 不含 conv**：CROWN/αβ/BaB 仅实现链式 MLP 子集（Linear+ReLU）。
2. **显式 A 表示**：CROWN/αβ 目前以张量形式传播线性界，`LinearOperator` 路线仍未收敛到 CNN 规模。
3. **BaB 控制流在 Python**：这是 Phase 6 的刻意选择（控制流不下沉编译），TVM/编译级加速主要在 Phase 5 主线。

Phase 7 的自然扩展方向通常是：扩图（skip/branch）与扩算子（conv），并逐步把 verifier 主线接入 Task IR / TVM 核下沉。
