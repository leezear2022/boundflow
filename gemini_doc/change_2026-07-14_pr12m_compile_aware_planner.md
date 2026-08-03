# 2026-07-14：PR-12M compile-aware 多预算 Planner 与 v3 held-out

## 目标

把 PR-12J 的 compile/load/cache 与 PR-12K 的 backend activity 结论进入生产 Planner 代价模型，
按以下字典序决策：

```text
capability → memory budget → evidence risk → amortized latency
```

不修改 PR-12L 冻结的 TIR/schedule，不复用 PR-12G final 调参。

## 实现

- 新增 calibration-only `CompileAwareFusedCrownPlanner`；
- 输入 expected query reuse、memory-cache hit probability、disk-cache hit probability；
- 对每个 backend 预测 peak、warm ratio、fresh/disk setup、amortized ratio、尺度距离与 risk tier；
- cache 概率不完整时显式回退 fresh setup 并提高 risk，而非当作零成本；
- 在所有 budget-feasible candidate 中先比较 risk，再比较 amortized latency；没有可行候选时
  选择预测 peak 最低者并明确标记 infeasible；
- PR-12I baseline runner 新增显式 `--record-set calibration|final_heldout`，完整 final-bound 合同、
  allocation 与 timing 口径不变；
- 新增 split freeze、calibration-only fit freeze、held-out replay、CSV/figure/manifest 工具。

## 无泄漏冻结顺序

1. v2 final 已被历史工作消费，只将这 5 点提升为 v3 calibration；
2. 创建 5 个全新 v3 final case；
3. calibration 25/25 candidate correctness；
4. 在 `final_heldout_consumed=false` 时冻结 model SHA；
5. 一次性测量 v3 final 25/25 candidate；
6. replay 生成 5 cases×5 budgets×3 reuse policies = 75 decisions；
7. 冻结前后 model SHA 均为
   `dc56c58b83ea355097ff14fe42e48599d16b3ed3e391c7c30f3febf7b2dcfa59`。

预算为 16/32/64/128 MiB 与 unbounded。reuse policies 为 cold Q1、mixed Q32、warm Q1024。

## 权威工件

```text
artifacts/phase7a-pr12/pr12m-compile-aware-v3-freeze-20260714/
artifacts/phase7a-pr12/pr12m-compile-aware-v3-calibration-20260714/
artifacts/phase7a-pr12/pr12m-compile-aware-v3-model-freeze-20260714/
artifacts/phase7a-pr12/pr12m-compile-aware-v3-final-heldout-20260714/
artifacts/phase7a-pr12/pr12m-compile-aware-v3-replay-v2-20260714/
artifacts/phase7a-pr12/pr12m-compile-aware-v3-report-20260714/
```

关键 SHA256：

```text
split:          1f79962d7d6325fbfbf6b0d9f63fef93e4a5e9866c840b75bda7374ecf2c5f83
calibration:    ae78a7147fb51d25737ce34dd95cf475b77a450b317af045af8202000f45bf59
model:          dc56c58b83ea355097ff14fe42e48599d16b3ed3e391c7c30f3febf7b2dcfa59
final held-out: 54dd6467201eec3e1e4522452ca88bb82bdf85408af735d3832586b1a2b2d03d
planner JSONL:  cb093ed53ce3d1b76aa0a18e3e38bced2fe4809f7ecce05a079bbdc91be429a9
summary:        bd84fa171752fcefd6620f05329ddf31edf5a161ec915173714ca39a870b3268
report manifest:fe3245e304895ee27bd0b683fbbaa0b5670b8c0787e1fba217efd5a3f181102f
```

## 结果

```text
calibration candidate correctness: 25/25
final candidate correctness:       25/25
Planner decisions:                 75
measured-feasible opportunities:   72
selected feasible:                 72/72
unsafe backend:                    0
feasible median/p90/max regret:    1.000× / 1.000× / 1.016×
no candidate feasible at 16 MiB:   3（同一 memory-heavy case×3 policies）
```

Planner 会做非平凡选择：总计 eager/chunked/structured/fused 为 47/12/3/13。cold 与 mixed 各只
选择 fused 1 次；warm Q1024 选择 fused 11 次，证明 compile/cache/reuse 会改变计划。32 MiB 下
四类 backend 都被选择；16 MiB 下有 3 个 regime 所有实测 candidate 都超过预算，未伪装成
feasible，也未从 regret 门禁中隐藏：all-row max regret 167.48×，但它只发生在无可行候选区；
论文质量门禁使用 72 个可行机会的 regret，并单独报告这 3 个 capacity failure。

TVM-unfused 在 v3 没有被选中，但仍保留于 candidate/evidence 表；这反映其当前 setup、warm 与
peak 组合未胜出，不是 capability 被删除。

## 运行中失败

held-out 测量后第一次 replay 命令漏掉 Conda 激活，立即以 `ModuleNotFoundError: boundflow`
退出，未创建输出目录；随后在 `boundflow` 环境中重跑成功。该失败属于命令环境错误，不影响
冻结 split/model 或数值结果，仍在此披露。

## 判定

```text
PR-12M compile-aware Planner: PASS (validated-reduced)
new held-out correctness:     PASS
multi-budget/reuse variation: PASS
feasible regret gate:         PASS
PR-12 overall:                IN PROGRESS
next:                         PR-12N closure audit
PR-13:                        BLOCKED until PR-12N
```

## 验证

```text
PR-12M focused/integration：9 passed
全量：                       340 passed、1 skipped
mypy：                       7 source files success
pylint：                     6 core/script files 10.00/10
Black / git diff --check：   通过
```
