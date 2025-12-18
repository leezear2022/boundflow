# 变更记录：Phase 5C PR#7（Determinism + DumpPlanInstrument + 结构化 VerifyError）

## 动机

PR#5/PR#6 已经把 planner pipeline 的“统一入口 + config_dump + verifier/instrument”钉住；但为了让后续 5D/5E（Relax lowering / cache / CROWN 域）更易调试、并让消融数据更可信，本 PR 补齐两类编译器级的“硬钉子”：

1) **可复现性（determinism）**：同一 `program + config` 多次运行应产出一致的 TaskGraph topo 顺序、reuse mapping 等关键 planner 产物；  
2) **可观测/可定位（dump + where）**：当 verifier 报错时要能定位到 task/buffer/edge；并提供 step 级 JSON snapshot 方便排查 “silent wrong”。

## 本次改动

### 1) TaskGraph topo_sort 稳定化（determinism）

- 修改：`boundflow/ir/task_graph.py`
  - `topo_sort()` 使用 `heapq`（按 `task_id` 字典序）构造确定性的 topo 顺序
  - 避免 set/dict 迭代顺序导致的跨进程/跨运行漂移

### 2) reuse blocker 统计稳定化（determinism）

- 修改：`boundflow/planner/passes/buffer_reuse_pass.py`
  - overlap blocker 选择在 `last_use_index` 相同时使用 `task_id` 作为 tie-break（并对 blockers 排序）
  - 避免 set 迭代顺序导致的 blocker topK 不稳定

### 3) VerifyError 增加 where，instrument 输出结构化错误

- 修改：`boundflow/planner/verify.py`
  - `VerifyError` 增加 `where: Dict[str, Any]`
  - `VerifyReport.add_error()` 支持 `where=...`（与 `context` 分离）
  - 在关键错误上填充 where（如 `missing_edge_dep`、`physical_lifetime_overlap`）
- 修改：`boundflow/planner/instrument.py`
  - `VerifyInstrument` 输出包含 `where` 字段（写入 `PlanBundle.meta["verify"]`）

### 4) Pipeline instrument 扩展：should_run + DumpPlanInstrument

- 修改：`boundflow/planner/instrument.py`
  - `PlannerInstrument` 增加 `should_run(step_name, bundle)`（默认 instrument 恒 True）
  - 新增 `DumpPlanInstrument`：在 step after 时写出 JSON snapshot 到 `dump_plan_dir/run_id/step_*.json`
- 修改：`boundflow/planner/options.py`
  - `PlannerDebugOptions` 增加：
    - `dump_plan: bool`
    - `dump_plan_dir: str`
    - `dump_plan_run_id: Optional[str]`
- 修改：`boundflow/planner/pipeline.py`
  - 当 `debug.dump_plan=True` 时自动启用 `DumpPlanInstrument`
  - before/after step 统一通过 `should_run()` 调用 hooks
  - `meta["plan_run_id"]` 记录本次 dump 的 run_id

### 5) 测试：determinism 与 dump 回归

- 新增：`tests/test_phase5c_pr7_determinism_and_dump.py`
  - 同一 `program + config` 连跑两次，断言：
    - `planner_steps/config_dump` 一致
    - `TaskGraph.topo_sort` 一致
    - `logical_to_physical/physical_buffers` 一致
  - 开启 `dump_plan` 时，断言输出目录生成 step JSON

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5c_pr7_determinism_and_dump.py
conda run -n boundflow python -m pytest -q
```

