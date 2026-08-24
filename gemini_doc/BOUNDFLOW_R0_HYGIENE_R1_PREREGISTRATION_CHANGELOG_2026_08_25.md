---
status: validated
updated: 2026-08-25T00:50:00+08:00
type: changelog
topic: boundflow
slug: r0-hygiene-r1-preregistration
stage: s01
---

# BoundFlow R0 卫生闭环与 R1 预注册修改记录

## Summary

- 接受外部复核结论并按批准顺序完成 R0、冻结 R1；
- R0 只修静态类型/导入口径和文档卫生，不改变 Interval/CIBC 数学、默认策略或性能阈值；
- R1 当前为 `PREREGISTERED-NOT-RUN`，没有 runner、formal raw 或新性能 claim。

## Changes

- `boundflow/domains/interval.py`：
  - 用返回 `Tuple[int,int]` 的 `_as_int_pair` 取代泛化 `Tuple[int,...]`，关闭 stride/padding/dilation
    三条新增 mypy `arg-type`；
  - 保留 CIBC runtime lazy import，避免 `runtime.__init__ -> executor -> interval` 循环；把 pylint
    `C0415` disable/enable 精确限定到该 import，并在代码中记录原因；
- CIBC formal closure：补充 steady-state timed boundary、cold compile/plan/graph capture 排除、input
  copy 对称性，以及 `3e-4` 在运行前冻结和实测 `2^-12` 单-ULP量级解释；
- 失败门禁恢复计划 §12：补 FSG3 B0/B2、NRIR49A、B4-C2 raw 路径；
- B4 最终状态与三份历史文档：明确 B4-A 唯一性能分类是 externally approved NO-GO；历史
  “mechanism/reduced evidence”只指非性能证据保留；
- 新增 R1 独立预注册：
  - graph/query/queue scope 与 `1.00/1.15/1.20x` targets；
  - 12-process candidate graph control/profile、稳定 op/ordinal ledger；
  - CUPTI↔host/NVTX native triplet calibration 与 Nsight export receipt；
  - same-solver B0/B3 eligible predicate、按 op type 的 `q_B3,k`；
  - exact production signature 的 `G_query,k`；独立 graph `2.45631x`禁止代填；
  - feasibility 方程、artifact/replay、16类 tamper、workload/frontend admission 与单路线 kill gate；
- 同步 memo、claims map、current status、master plan、README、R3历史 next和外部评审Prompt。

## Validation

- 修改前复现：mypy 恰有3条新增 `arg-type` + 8条既有 `DomainState attr-defined`；pylint C0415恰1条；
- 修改后 `mypy --disable-error-code=attr-defined boundflow/domains/interval.py`：PASS；
- 修改后 `pylint --disable=all --enable=C0415 boundflow/domains/interval.py`：10.00/10；
- CIBC/IBP定向测试：`6 passed, 1 warning`；
- Black `--check --target-version py312`：PASS；
- 完整 `pylint boundflow.domains.interval` 从历史 `7.01` 提高为 `7.13/10`，仍保留缺docstring、
  too-many-locals和Torch `not-callable`等既有告警；本轮新增C0415已清零，没有伪称全规则clean；
- 全量回归：`1492 passed, 3 skipped, 6 warnings in 655.14s`；3项skip均为既有TVM重复编译或冻结
  VNN-COMP checkout缺失边界；
- R1 contract、evidence-path、authority consistency、fenced-code与本机路径泄漏检查：PASS；
- `git diff --check`：PASS；DocOps change/validation/lint结果在提交前写入事件流。

## Decisions

- 不把独立 ResNet2B IBP graph 的 `G=2.45631` 当成 same-solver query 的物理常数；
- query传播只使用待优化B3侧互斥share `q_B3,k` 和 exact signature现场测得的 `G_query,k`；
- 真实query中带split/α/β状态的调用默认不准入，除非receipt证明待替换IBP region语义/拓扑不变；
- query-local candidate测不到时 `G_query,k=1`，不以历史micro/graph结果填空；
- 只有 conservative projected B0-relative query lower bound `>=1.00x` 才开放一个 R2 实现分支；
- 三个历史 DocOps duplicate id继续作为独立维护项，不与性能代码混交。

## Follow-Ups

- 新 clean commit 实现 R1-0 clock/topology/schema 和 negative tests；
- 先跑 calibration/profile perturbation smoke，formal worker 0 前若需 amendment必须单独提交；
- R1-A关闭后才运行same-solver R1-B/C；R1-D前不实现Linear/Conv/epilogue候选；
- formal关闭后把交接放入DocOps exchange进行独立外审。

## Links

- plan: `BOUNDFLOW_CIBC_R1_SCOPE_CLOCK_QUERY_LOCAL_ATTRIBUTION_PLAN_2026_08_25.md`
- parent recovery plan: `BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_PLAN_2026_08_24.md`
- CIBC closure: `BOUNDFLOW_CIBC_IBP_HORIZONTAL_FORMAL_CLOSURE_2026_08_24.md`
- roadmap: `boundflow_asplos_master_plan_2026_07_12.md`
