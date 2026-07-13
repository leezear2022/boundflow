# 2026-07-14：PR-12L 冻结停止孤立 TIR 调优

## 决策

PR-12L 只选择一个冻结分支：

```text
E_STOP_OPTIMIZING_TIR
```

该决策不删除 TVM fused backend，也不否定 materialization elimination 的机制价值；它表示在
PR-12 剩余阶段不再增加手工 TIR schedule、tile、vectorization、CUDA Graph、chunk-size family
或 Conv capability。fused 继续作为 Planner 可选 backend，由 capability、预算、风险、compile/
load/first-query 与 expected reuse 共同决定是否采用。

## 证据依据

PR-12K 权威工件为：

```text
artifacts/phase7a-pr12/pr12k-cupti-v3-20260714/
artifacts/phase7a-pr12/pr12k-cupti-report-v4-20260714/
```

在 6 个 workload、5 个 backend、30/30 correctness 通过的 complete final-bound CUPTI activity
profile 中，fused 相对 TVM-unfused 每个 eligible region 只减少 2 个 launch，最大整体 launch
降幅为 1.96%。按 5% device-time 阈值，3/6 退化、1/6 改善、2/6 中性。Nsight Compute 又因
`ERR_NVGPUCTRPERM` 无法取得带宽、occupancy 或 stall counter，因此没有证据支持某一种具体
低层优化能稳定改善完整 workload。

PR-12J 同时表明：Linear/Conv fused warm 均慢于 eager/chunked；mini-ResNet 虽略快于 eager，
但 fresh/disk-first/process break-even 为 4668/1062/4450 queries，超过冻结 Q≤1024 区间，且
仍不优于 chunked。继续孤立优化 kernel 不能解决 compile-aware selection 问题。

## 未选择分支

| 分支 | 不在 PR-12L 选择的原因 |
|---|---|
| Linear tiled reduction | memory-sensitive Linear 显著退化，但无硬件 counter 支持具体 tile；继续试参会消费旧 profile |
| CUDA Graph/dispatch | 相对 TVM-unfused 的 launch 降幅上限仅 1.96%，不是当前主导证据 |
| chunk-size family | chunked 已是独立可行 backend；扩大参数族属于新的候选搜索研究，会污染 closure |
| Conv capability 拆分 | PR-12 已冻结有限 static FP32 plain-CROWN capability；扩算子不修复当前完整系统收益 |
| 停止优化 TIR | **选择**；把剩余预算投入 compile-aware Planner 与独立 held-out |

上述“未选择”只约束 PR-12 closure，不永久禁止未来研究。如果新硬件、新 profiler 权限或独立
workload 给出新的 counter/evidence，应在 PR-12 之外创建新假设和新 split，不能回写本阶段。

## PR-12M 接口约束

下一阶段只能推进 compile-aware Planner：

1. 冻结全新 calibration/final-heldout split，不复用 PR-12G final 调参；
2. 候选集合保持 eager、structured/chunked、TVM-unfused、TVM-fused 的现有 capability；
3. 字典序为 capability → memory budget → risk → amortized latency；
4. 显式输入 expected reuse、fresh/disk/process first-query 与 cache-hit probability；
5. 扫描 16/32/64/128 MiB 与 unbounded，记录 rejection、fallback、OOM 和 timeout；
6. 先完成 calibration-only model，再一次性消费 final held-out；
7. 不因结果不好回调 schedule 或修改冻结 split。

## 状态

```text
PR-12L decision freeze: PASS
TIR/schedule/code delta: NONE
PR-12 overall:          IN PROGRESS
next:                   PR-12M compile-aware Planner
PR-13:                  BLOCKED
```

## 验证

```text
git diff --check：通过
```
