# Phase 5 总结：可复现评测产线（bench→JSONL→postprocess→artifact）+ TVM/Relax 可观测性 + 消融矩阵（schema_version=1.0 冻结）

> 本文总结 Phase 5 的“目标 → 里程碑 → 代码落点 → 回归钉子 → 复现入口 → 已知限制”，用于论文/AE/工程接手的统一口径。
>
> 主要依据：
> - 完成声明：`docs/phase5_done.md`
> - 总账：`docs/change_log.md`
> - 口径/证据链：`docs/bench_jsonl_schema.md`、`gemini_doc/artifact_claims_phase5d.md`、`gemini_doc/artifact_appendix_phase5d.md`
> - 关键产线脚本：`scripts/run_phase5d_artifact.py`、`scripts/bench_ablation_matrix.py`、`scripts/postprocess_ablation_jsonl.py`

---

## 1. Phase 5 解决什么问题（目标与边界）

Phase 4 已经把 “Primal IR → Task → IBP 执行（Python/TVM）” 的语义闭环跑通。Phase 5 的目标不是再加算法，而是把系统工程做成**论文/AE 可交付**的形态：

1. **统一可复现产线**：bench 输出稳定的 JSONL（每行一个配置点），并能自动后处理成 CSV/表/图与可审计 MANIFEST。
2. **系统化消融（ablation）**：把 planner/runtime/后端开关组织成矩阵，一键跑完并输出对照数据（而不是手工拼表）。
3. **TVM 侧可观测性闭环**：compile vs run 拆分、call_tir 统计、pass/VM 开关与缓存策略进入 schema，便于解释“为什么更快/更省”。
4. **基线对照纳入证据链**：把 auto_LiRPA 作为可选 baseline 纳入 JSONL，并做 correctness gate 与 speedup 统计（缺依赖/不可用时不静默、写明原因）。

Phase 5 的边界（刻意不做）：

- 不引入 CROWN/αβ/BaB 的新语义（属于 Phase 6）；
- 不把 Phase 6 的新字段/新产线回灌到 Phase 5：Phase 5 的 JSONL 口径冻结为 `schema_version=1.0`（见 `docs/bench_jsonl_schema.md`）。

---

## 2. Phase 5 的完成定义（Done Definition）

Phase 5 的“完成”以 **产线与口径冻结** 为准（盖章文档见 `docs/phase5_done.md`）：

1. **bench 输出协议冻结**：JSONL schema 固化为 `schema_version=1.0`，并由 contract tests 长期守住；
2. **产线闭环**：`bench → JSONL → postprocess → artifact` 一键可跑，输出结构固定（JSONL/CSV/tables/figures/MANIFEST/CLAIMS/APPENDIX）；
3. **公平计时口径**：cold（首次触发编译/缓存）与 hot（稳态 p50/p95）拆分；
4. **失败不可静默**：run/status/失败原因写入 JSONL，后处理与 artifact 不丢失失败点；
5. **AE/论文可引用**：提供 claims→文件/字段→命令映射（Phase 5D）。

---

## 3. 里程碑回顾：5A → 5E（按“系统能力”组织）

> Phase 5 的 PR 数量较多，这里按能力分组总结（细节以 `docs/change_log.md` 与对应 `gemini_doc/change_*.md` 为准）。

### 3.1 5A：TaskGraph / PlanBundle / Scheduler（从单 task 走向可调度系统）

**目标**：为后续缓存、复用、批处理与多 task 编译/执行建立统一容器与调度接口。

**落点**

- `boundflow/ir/task_graph.py`：Task DAG
- `boundflow/planner/core.py`：`PlanBundle` / pass pipeline 骨架
- `boundflow/runtime/scheduler.py`：按 topo 串行调度（Phase 5 先求正确性与接口稳定）

### 3.2 5B：liveness + buffer reuse + 内存口径（为消融与解释做地基）

**目标**：把 StoragePlan 从“一值一 buffer”升级到可复用计划，并把内存影响纳入可消融的开关与输出字段。

**落点**

- liveness/reuse 与 buffer contract 的钉子（使 `buffer_id -> IntervalState` 的执行契约稳定）
- memory 相关枚举与估算字段进入 bench JSONL（用于趋势解释/象限图）

### 3.3 5C：planner pipeline 可观测性（dump/config/determinism）

**目标**：让“同一配置点为什么变了/为什么失败”可定位，并为后续大规模消融提供稳定复现实验条件。

**落点**

- planner config dump、plan dump、determinism 回归（避免“偶现差异”污染消融）

### 3.4 5D：TVM/Relax 编译执行闭环 + ablation_matrix + JSONL schema 产线

这是 Phase 5 的核心交付：把系统收益变成可复现证据链。

**关键落点（代码）**

- `scripts/bench_ablation_matrix.py`：消融矩阵 bench（stdout JSONL）
- `docs/bench_jsonl_schema.md`：schema_version=1.0 的字段与口径说明
- `tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`：schema contract tests
- `scripts/postprocess_ablation_jsonl.py`：JSONL→CSV/表/图/manifest
- `scripts/run_phase5d_artifact.py` / `scripts/run_phase5d_artifact.sh`：一键 artifact runner
- `boundflow/runtime/tvm_executor.py`：TVM executor + compile/run 观测字段 + cache（含可选落盘）

**关键落点（证据链文档）**

- `gemini_doc/artifact_claims_phase5d.md`：claims→文件/字段→命令
- `gemini_doc/artifact_appendix_phase5d.md`：AE 操作说明（如何跑/如何验证/如何看结果）

### 3.5 5E：论文面向表图与附录（把产物做成可引用形态）

**目标**：把 Phase 5D 的产线输出整理成论文/AE“直接可用”的表/图与附录材料，并在 `docs/phase5_done.md` 里完成“盖章口径”。

---

## 4. 产物与复现入口（Phase 5 的主交付）

Phase 5 的“主入口”是 Phase 5D artifact runner（详见 `docs/phase5_done.md` 与 `gemini_doc/artifact_claims_phase5d.md`）：

```bash
# Quick（CI/冒烟）
conda run -n boundflow python scripts/run_phase5d_artifact.py --mode quick --workload all --run-id quick_test

# Full（论文级矩阵）
conda run -n boundflow python scripts/run_phase5d_artifact.py --mode full --workload all --run-id full_run
```

典型输出目录（不进 git）：

- `artifacts/phase5d/<run_id>/`
  - `results.jsonl`（schema=1.0）
  - `results_flat.csv`
  - `tables/table_main.csv`、`tables/table_ablation.csv`
  - `figures/*.png`（可选依赖 matplotlib）
  - `MANIFEST.txt`（含 sha256 审计）

---

## 5. 回归钉子（为什么 Phase 5 “reviewer-proof”）

Phase 5 的核心不是“跑得更快”，而是“口径更硬”：

1. **schema contract tests**：字段/类型/口径由测试守住（避免 silent break）
2. **cold vs hot 拆分**：避免把编译/缓存噪声混入稳态对比
3. **baseline 对照不静默**：auto_LiRPA 缺失/不可用会显式记录（available/reason）
4. **postprocess 与 artifact smoke**：从 JSONL 到表图与 MANIFEST 的完整链路可回归

---

## 6. 已知限制与 Phase 6 边界（避免口径回滚）

Phase 5 的明确限制（见 `docs/phase5_done.md`）：

- domain 以 interval IBP 为主线；更强方法族属于 Phase 6；
- TVM lowering/算子覆盖仍在演进；失败点写入 JSONL，不静默丢失；
- `physical_bytes_est` 属于 planner 估算，用于趋势解释，不等价真实峰值内存。

Phase 6 与 Phase 5 的边界原则：

- Phase 5 的 schema/表图口径保持 `schema_version=1.0` 不变；
- Phase 6 新增产线/字段必须使用新 schema_version，并保持后处理向后兼容。

