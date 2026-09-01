---
status: completed
updated: 2026-08-26T00:45:00+08:00
type: changelog
topic: boundflow
slug: mr7-graph-compiler-rule-runtime-research-change
stage: s01
performance_claimed: false
---

# MR7 图编译、验证规则与物理运行时调研文档修改记录

## 1. 目的

基于 MR7 formal invalid closure、raw ledger、当前 Bound/Plan/Task/Schedule/R3/CIBC 实现和外部一手编译
系统资料，形成一份可供多个外部模型审计的图编译、规则重写、内存规划和运行时路线文档。

## 2. 修改范围

新增：

- `gemini_doc/BOUNDFLOW_MR7_GRAPH_COMPILER_RULE_RUNTIME_RESEARCH_PLAN_2026_08_26.md`；
- 本修改记录。

更新：

- `gemini_doc/README.md` 增加新调研文档入口。

未修改：

- production 代码、测试、TIR、solver、benchmark protocol；
- MR7 artifact；
- MR7-R 预注册；
- 当前 branch、tag 或 performance claim。

## 3. 主要内容

- 独立区分 MR7 formal invalid 与 diagnostic-only raw；
- 复核 57 launch/FFI span、30/27、host boundary 分类和 device-clock 分母边界；
- 盘点现有 IR/runtime 的可复用资产与“schema 已有、物理执行未落地”的 gap；
- 对比 TVM Relax/TIR、MLIR、OpenXLA、PyTorch 2、CUDA Graph/allocator；
- 冻结 BoundFlow semantic owner + Relax/TIR lowering + physical arena/runtime 的分层；
- 给出 verification-aware rule schema、11 个规则族、P0–P12 pass pipeline；
- 给出 dense-A 禁令、effect token、happens-before liveness、AOT/cache 和 CUDA Graph 约束；
- 给出 GC-0–GC-6 实施路线、correctness/结构/performance 门禁与 kill rules；
- 保持 MR7-R 为唯一当前开放动作，本文不授权直接 implementation/timing。

## 4. 证据纪律

- 所有 MR7 opportunity 数字标记 `[diagnostic-only]`；
- 明确 `8.6915%` 是 device CUDA-event envelope share，不是 host outer share；
- 明确 `optimizer_and_residual` 是自动补项，不是直接量测 optimizer；
- 不形成 speedup、query、queue、competitor 或 ASPLOS-ready 新 claim；
- `performance_claimed=false`。

## 5. 验证

应执行：

- `git diff --check`；
- 文档路径/本地证据入口存在性检查；
- 外部 URL 基本连通性抽查；
- MR7 summary/raw 数字与关键代码事实复核；
- `dol lint --soft`。

本轮仅文档变更，不运行 GPU benchmark 或全量 Python tests；原因是没有代码行为变化，且本文明确不形成
performance claim。
