# PR-11 高 Regret 归因与 PR-12 决策记录

> 状态：PR-11 冻结前诊断；不重新打开 PR-11，也不覆盖历史 profile。

## 结论

最终三组 held-out 的 23 个 `global_bounded_retry` case 中，有 9 个满足
`latency_regret_ratio >= 1.5`。这 9 个 case 的直接归因全部是
`CANDIDATE_NOT_AVAILABLE`：per-case measured oracle 没有进入最多 6 个候选的 bounded retry
集合。因此当前证据不支持把 1.880× median regret 解释成 cost model 在已有候选中排序错误。

诊断 flag 中，7/9 为 `BACKEND_GAP`，2/9 为 `MEASUREMENT_VARIANCE`。`BACKEND_GAP` 只表示
所选计划包含比 oracle 更多的 structured barrier，提示 eager structured 执行可能是后续优化点；
它不证明 fused TIR 一定能消除 regret。PR-12 必须通过独立 backend baseline 验证该假设。

## 高 Regret 明细

| Workload/query | Regret | Selected → oracle | Primary | Flags |
|---|---:|---|---|---|
| mini_resnet s32 / 32 MiB | 1.663× | SSSDSSS → SSDSSSS | CANDIDATE_NOT_AVAILABLE | MEASUREMENT_VARIANCE |
| mini_resnet s32 / 34 MiB | 1.747× | SSDDSSS → DSDSDSD | CANDIDATE_NOT_AVAILABLE | BACKEND_GAP |
| mini_resnet s32 / 36 MiB | 1.747× | SSDDSSS → DSDSDSD | CANDIDATE_NOT_AVAILABLE | BACKEND_GAP |
| mini_resnet s128 / 96 MiB | 3.061× | DSSDSSS → SDDSDDS | CANDIDATE_NOT_AVAILABLE | MEASUREMENT_VARIANCE, BACKEND_GAP |
| branched_resnet s32 / 30 MiB | 1.880× | SSSSSSS → DSSSSSS | CANDIDATE_NOT_AVAILABLE | BACKEND_GAP |
| branched_resnet s32 / 36 MiB | 3.160× | SSSDDSS → DSSSSDS | CANDIDATE_NOT_AVAILABLE | — |
| branched_resnet s32 / 38 MiB | 2.248× | DSDSSSS → DSSSSDD | CANDIDATE_NOT_AVAILABLE | BACKEND_GAP |
| branched_resnet s32 / 40 MiB | 2.248× | DSDSSSS → DSSSSDD | CANDIDATE_NOT_AVAILABLE | BACKEND_GAP |
| branched_resnet s32 / 44 MiB | 2.377× | DSDSSSS → DSDSDSD | CANDIDATE_NOT_AVAILABLE | BACKEND_GAP |

## 口径与工件

- Primary attribution 按优先级区分 candidate 缺失、OOM fallback、cost-model misrank 和
  profile extrapolation；measurement variance 与 backend gap 是非互斥 flags。
- variance flag 阈值为 selected/oracle 任一 replicated latency 的 max/min >= 1.25。
- raw JSONL、CSV、manifest 位于
  `artifacts/phase7a-pr11/pr11-final-regret-attribution-20260713/`。
- 归因脚本为 `scripts/attribute_phase7a_pr11_regret.py`，结果可从冻结的 final evaluation 与
  aggregate held-out profile 重建。

## 对 PR-12 的约束

PR-12 不是“修复 PR-11 Planner”的 PR。它验证的是：在不改变 PR-11 候选、oracle 与历史
profile 的前提下，fused backend 能否改善 structured execution 的 latency-memory Pareto frontier。
候选覆盖不足作为 C2 limitation 保留；若未来扩展 candidate generator，应生成新 schema/profile，
不得回写 PR-11 结果。
