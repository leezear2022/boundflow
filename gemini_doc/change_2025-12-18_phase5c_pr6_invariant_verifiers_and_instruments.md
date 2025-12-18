# 变更记录：Phase 5C PR#6（Invariant Verifiers + Pipeline Instrument）

## 动机

在 PR#5 已经钉住 “统一 planner 入口 + config_dump 可复现” 后，下一步必须把 **pass contract**（每一步产物是否仍合法）钉死。否则后续接入 Relax lowering、cache、CROWN domain 时，风险会变成“结果悄悄错但能跑完”，极难定位。

本 PR 引入：

- `verify_*`：核心不变式 verifier（TaskGraph/StoragePlan/Liveness+Reuse）
- pipeline instrument：类似 TVM 的 before/after step hooks，用于 timing 与 verify 报告落到 `PlanBundle.meta`

## 本次改动

### 1) Verifier：核心不变式检查

- 新增：`boundflow/planner/verify.py`
  - `VerifyReport`（ok/errors/stats）
  - `verify_task_graph_soundness()`：检查 topo 可排序、跨 task use/def 必须被 edge dep 覆盖
  - `verify_storage_plan_soundness()`：当存在 `physical_buffers` 时要求 `logical_to_physical` total 且 targets 完整
  - `verify_liveness_reuse_consistency()`：同一 physical 的 logical lifetimes 不允许 overlap（task 粒度）
  - `verify_all()`：聚合三类核心 verifier

### 2) Pipeline instrument：timing + verify（debug 开关控制）

- 新增：`boundflow/planner/instrument.py`
  - `TimingInstrument`：写入 `PlanBundle.meta["timings_ms"]`
  - `VerifyInstrument`：写入 `PlanBundle.meta["verify"]`，失败时 fail-fast（由 `validate_after_each_pass` 控制）
- 修改：`boundflow/planner/pipeline.py`
  - 当 `PlannerDebugOptions.validate_after_each_pass=True`：
    - lowering step 与 storage_reuse step 后运行 `verify_all()` 并写入 meta

### 3) 测试：负例覆盖 broken edge / broken mapping / overlap alias

- 新增：`tests/test_phase5c_pr6_validators.py`
  - 刻意破坏 TaskGraph deps：verifier 必须报错
  - 构造 physical plan 但 mapping 不 total：verifier 必须报错
  - 构造 overlap alias：verifier 必须报错
  - pipeline 开启 `validate_after_each_pass` 后，`meta["verify"]` 必须包含每个 step 的报告

## 如何验证

```bash
conda run -n boundflow python -m pytest -q tests/test_phase5c_pr6_validators.py
conda run -n boundflow python -m pytest -q
```

