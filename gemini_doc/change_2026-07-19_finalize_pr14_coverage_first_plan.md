# 变更记录：冻结 PR-14 Coverage-First 执行模型

## 背景

PR-13 closure 已证明 query runtime 能保留普通 batching 的收益，但未证明 runtime abstraction
相对公平 batched original 有独立加速。PR-14 的首要风险因此不是缺少新 kernel，而是现有
Planner/backend 对真实 complete-verification query 的覆盖率未知。

## 修改

- PR-14 正式名称收敛为 **Verification-Aware Execution on Real Verification Workloads**；
- ASPLOS 状态区分为执行 `CONDITIONAL GO` 与成稿门禁 `ASPLOS-ready NO`；
- PR-14A 固定为 coverage-first：MLP/CNN/ResNet-block 三类 workload，先输出 method/stage、
  α/β/split、shape/batch 与 backend eligibility；
- 明确新增 `VerificationQueryProfile` 必须从现有 `BoundQuery` 派生，不复制 query/state schema；
- 性能顺序保持为 coverage → fixed replay → full E2E，公平 baseline 为 original batched executor。

## 验证

- `git diff --check`；
- 文档引用与 branch/tag 基线检查；
- PR-13 focused 回归 15 passed（沿用本次分支启动验证）。
