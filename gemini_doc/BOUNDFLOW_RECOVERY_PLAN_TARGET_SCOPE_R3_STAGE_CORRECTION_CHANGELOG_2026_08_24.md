---
status: validated
updated: 2026-08-25T00:13:21+08:00
type: changelog
topic: boundflow
slug: recovery-plan-target-scope-r3-stage-correction
stage: s01
---

# BoundFlow 恢复计划目标范围与 R3 阶段修正记录

## Summary

- 接受外部评审中关于 Amdahl scope、same-solver eligibility、时钟域与 R3 阶段混合的实质建议；
- 不改变任何既有性能数字、NO-GO、VALIDATED-REDUCED 或 ASPLOS-ready 状态；
- 当前下一步从“直接做 CIBC-G1”收紧为“R0 hygiene + R1 protocol/target freeze，再做 G1”。

## Changes

- 在失败门禁恢复计划中冻结 B0-relative `T_query_qualification=1.00`、
  `T_query_research=1.15`、`T_queue_research=1.20`，并禁止 graph/query/queue share 跨 scope 代入；
- 增加 same-solver eligible-IBP share 的传播方程。公式只使用待优化 B3/candidate 侧 `q_B3`；以
  现有 B3 ratio `0.910001` 与 CIBC graph speedup `2.45631` 得到乐观
  `q_B3_required=15.18%/35.20%`，明确标记为 feasibility bound；
- 为 CUPTI、NVTX、Nsight Systems 增加 host/GPU 时钟校准 receipt；单 stream 无 overlap 时禁止用
  overlap-adjustment 制造 headline；
- 把 benchmark admission 前移到 R1 后：只读冻结前端 op coverage、两个 held-out family 和至少一个
  baseline/candidate 都可 solve 的公开 workload；
- 把执行顺序改为 G1 attribution → same-solver share admission → mathematically reachable R2 →
  B0/B3/cumulative candidate formal；R3 设计评审可并行，R3-0 实现保持关闭；
- R3-1 冻结 optimizer mutation 且禁止 `optimizer.step()`，但 custom backward 与 `dα` 对照强制；
  no-grad 只能 smoke；
- 把原 R3-2 拆为 R3-2A 10/9 trajectory correctness 与 R3-2B wrapper-inclusive timing，保留原
  `1.20x/0.98x/1.0x` 物理门槛；
- 将 R4 planner claim 收窄为当前证据支持的 shape/signature-keyed static specialization，并用显式
  compile/cache/invalidation 成本式替代任意 JIT reuse 倍数；
- 同步两个外审 Prompt、ASPLOS memo、claims map、current status、master plan 与 README。

## Validation

- 文档内 scope target、R3-2A/B、执行顺序与 authority notes 的一致性 grep；
- Markdown fenced-code balance 与本地引用存在性检查；
- `git diff --check`；
- DocOps `dol ch add`、`dol va add --result pass`、`dol lint --soft`；
- 本轮只改文档/DocOps，不生成新性能 artifact，不重跑 GPU benchmark。

## Decisions

- 采纳：量化 `q_required`、可 solve workload、memory/前端准入前移、same-solver B0/B3/candidate、
  clock calibration、R3 correctness/timing 拆分、R4 静态 specialization 收窄；
- 不采纳：“所有卡点都物理可解”。既有 IR-5、B4-C2 等 NO-GO 保持，后续路线仍允许被新门禁证伪；
- 不先实现前端 op coverage；先冻结缺口和 benchmark，防止用实现结果反向选择 workload；
- receipt 热路径只作为待测假设，未由 profile 证明前不删除 fail-closed 检查。

## Follow-Ups

- 完成 R0 的 3 条新增 mypy `arg-type` 与 1 条新增 pylint `C0415`；
- 独立预注册 R1 raw schema、calibration receipt、scope targets 与 tamper tests；
- 通过 R1 后测 same-solver eligible-IBP share，再决定唯一 R2 实现分支；
- 将修订后的恢复路线 Prompt 与 R3 Prompt 交给外部模型复核。

## Links

- plan: `BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_PLAN_2026_08_24.md`
- R3 plan: `BOUNDFLOW_R3_STRUCTURED_OWNER_CUSTOM_BACKWARD_REDESIGN_PLAN_2026_08_24.md`
- recovery review prompt: `BOUNDFLOW_FAILED_GATES_EXTERNAL_ADVISOR_PROMPT_2026_08_24.md`
- R3 review prompt: `BOUNDFLOW_R3_STRUCTURED_OWNER_EXTERNAL_REVIEW_PROMPT_2026_08_24.md`
- roadmap: `boundflow_asplos_master_plan_2026_07_12.md`
