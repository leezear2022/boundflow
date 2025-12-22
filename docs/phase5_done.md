# Phase 5 完成声明（工程收口 / 论文消融 / AE 交付）

本文用于“盖章”当前仓库 **Phase 5 已完成**（在工程收口/可复现产线/论文级证据链三个维度），并给出 AE/论文可直接引用的复现入口、产物结构与已知限制。

> 口径冻结点：`schema_version=1.0`（见 `docs/bench_jsonl_schema.md`）。

---

## 1) Phase 5 覆盖范围（已完成的事）

### 1.1 统一可复现产线（bench → postprocess → artifact）

- 统一实验输出协议：JSONL（每行一个配置点/一次 run），并用 contract tests 固化字段/类型/口径：`docs/bench_jsonl_schema.md`、`tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`。
- 后处理流水线：JSONL → 扁平 CSV → 汇总表 →（可选）图：`scripts/postprocess_ablation_jsonl.py`。
- 一键 artifact runner：产出 `results.jsonl / tables / figures / MANIFEST / CLAIMS / APPENDIX`：`scripts/run_phase5d_artifact.py`、`scripts/run_phase5d_artifact.sh`。

### 1.2 基线对照纳入证据链（auto_LiRPA）

- auto_LiRPA baseline 作为可选对照写入每行 JSONL（包含 warmup/稳态计时与 gate 字段），并在主表中给出 `speedup_hot_vs_auto_lirpa` 等列（见 `gemini_doc/artifact_claims_phase5d.md`）。
- baseline 计算外提：每次 bench invocation 先预计算一次 baseline，再在矩阵点中复用，避免矩阵点内触发与重复算力开销（实现位于 `scripts/bench_ablation_matrix.py`）。

### 1.3 TVM 侧“可解释性”闭环

- compile vs run 拆分：`compile_first_run_ms`、`run_ms_p50/p95` 等稳态计时口径固定在 schema。
- compile 可观测性：可选 per-pass timing / DumpIR（见 `boundflow/runtime/tvm_executor.py` 的 options 与 compile_stats）。
- call_tir 数量统计 + fusion pipeline 开关（用于解释“为什么更快/调用更少”）。
- StaticPlanBlockMemory 相关对照与 memory estimate 口径（见 `docs/bench_jsonl_schema.md` 与 compile_stats 聚合字段）。

### 1.4 “不浪费算力”的缓存策略

- auto_LiRPA baseline：按 baseline_key/spec_hash 去重（postprocess 侧 join，不随矩阵点重复计数）。
- TVM task-level compile cache：支持可选落盘目录（跨进程复用，降低 AE 多次运行的重复编译成本）。

---

## 2) 复现入口（AE/论文用）

### 2.1 Quick（CI/冒烟）

```bash
conda run -n boundflow python scripts/run_phase5d_artifact.py --mode quick --workload all --run-id quick_test
```

### 2.2 Full（论文级矩阵）

```bash
conda run -n boundflow python scripts/run_phase5d_artifact.py --mode full --workload all --run-id full_run
```

### 2.3 输出目录结构

runner 默认写入（目录已加入 `.gitignore`，不进 git）：

- `artifacts/phase5d/<run_id>/`
  - `results.jsonl`
  - `results_flat.csv`
  - `tables/table_main.csv`
  - `tables/table_ablation.csv`
  - `figures/*.png`（matplotlib 可选依赖）
  - `MANIFEST.txt`（含 sha256，用于审计/防漂移）

证据映射/口径说明：

- `gemini_doc/artifact_claims_phase5d.md`（claims→文件/字段→命令）
- `gemini_doc/artifact_appendix_phase5d.md`（AE 复现说明）

---

## 3) Phase 5 完成 DoD（建议你用这份当“收尾 기준”）

Phase 5 功能层面已完成；若要在论文/AE 角度“最终盖章”，建议做完下面 6 件机械项：

1. 跑一次 full artifact 并留档（至少两 workload）  
   - `scripts/run_phase5d_artifact.py --mode full --workload all --run-id <tag>`
2. 封存该次产物目录作为 paper/AE 参考点  
   - 以 `MANIFEST.txt` 的 sha256 作为审计依据。
3. 给 repo 打 tag（对应 `schema_version=1.0` 冻结点）  
   - 让论文数字/图表/命令能指向同一 tag。
4. 将“Phase 5 完成声明”与已知限制写入 docs（即本文）  
   - 避免 Phase 6 开发冲掉口径。
5. CI 守门（至少三类必跑）  
   - JSONL schema contract、postprocess pipeline、artifact smoke。
6.（可选）固定最小环境描述  
   - conda env / TVM 版本/commit 等（JSONL `meta` 已记录，可整理进 appendix）。

---

## 4) 已知限制（Phase 5 的明确边界）

- domain 目前以 interval IBP 为主线；更强的 domain（CROWN/α-CROWN/BaB）属于 Phase 6。
- TVM lowering/执行路径对算子覆盖仍在演进；某些 workload/配置可能回退或失败（失败点会以 `status="fail"` 写入 JSONL，不会静默丢失）。
- `physical_bytes_est` 属于 planner/StoragePlan 估算，用于趋势解释，不等价真实峰值内存/显存。

---

## 5) Phase 6 与 Phase 5 的边界（避免口径回滚）

Phase 6 可以引入新的算法域/更强性质/更强优化（例如 CROWN/alpha-CROWN、op-level lifetime、layout/transpose 全局优化、BaB cache/batching 等），但不应反向影响 Phase 5 的冻结口径与 artifact 产线。

建议做法：

- Phase 5 的 schema/表/图口径保持 `schema_version=1.0` 不变；
- Phase 6 若新增字段或新产线，使用新的 schema_version，并保持 postprocess 向后兼容。

