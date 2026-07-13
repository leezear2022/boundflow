# BoundFlow PR-12 中长期完成规划

> 状态：执行版；从 PR-12H 开始，持续到 PR-12 closure。
> 起点：`44f87ae` / tag `pr12g-validated-reduced`。
> 约束：PR-12 closure 前不启动 PR-13，不再无条件增加 kernel/schedule family。

## 1. 当前判定

PR-12A–D correctness 已关闭；PR-12E/F 建立了 v1 runtime/Pareto 与 held-out 负结果；PR-12G
加入 eager/chunked/TIR 多后端候选，在 reduced held-out 上达到 5/5 budget feasible、0 unsafe、
median/p90 regret 1.000×/1.054×。但 structured eager、TVM unfused、compile/load/cache 摊销、
硬件 counter、production auto-selection 与完整 repeated-query E2E 仍缺失。

因此当前只能写：

```text
PR-12G:       VALIDATED-REDUCED
PR-12 overall: IN PROGRESS
PR-13:         BLOCKED
```

## 2. 最终研究问题

在相同浮点语义和完整 plain-CROWN final-bound 查询下，BoundFlow 是否能通过互不支配的 backend
候选与 memory/compile-aware Planner，在不同 shape、query reuse 和显存预算下改善 latency–memory
Pareto，并能用可复现的编译/硬件机制解释选择？

## 3. 阶段路线

### PR-12H：冻结证据与 benchmark contract

- tag PR-12G；
- 固定 kernel、region-runtime、end-to-end 三层合同；
- 标记历史 benchmark 的合同缺口，不重写历史结果；
- 建立 `gemini_doc/pr12_execution_status.md` 恢复入口。

结束条件：三层 inclusion、计时、同步、内存与 correctness 口径有机器可读定义和 contract tests。

### PR-12I：正式 baseline

候选至少包括：dense eager、structured eager、chunked-r512、TVM unfused、TVM fused TIR；
`torch.compile` 只在无 graph break 且语义可比时加入。必须在相同 region API 和完整 final-bound
合同分别对齐，失败也写 JSONL。

### PR-12J：compile/load/cache amortization

分离 IR construction、TIR generation、schedule、compile、serialization、module load、first run、
warm run、memory-cache hit 和 process-restart disk-cache hit。固定 Q=
1/2/4/8/16/32/64/128/256/512/1024；warm 不快时记录 `not_amortizable`。

### PR-12K：正式 profiler

在 Linear small/memory-sensitive、Conv s1/s2、residual、mini-ResNet 上采集可用的 Nsight/CUPTI
counter。至少覆盖 SpeedOfLight、MemoryWorkloadAnalysis、LaunchStats、Occupancy、SchedulerStats；
若工具不可用，记录缺失依赖并继续完成可执行的静态/运行时归因，不能伪装为硬件结论。

### PR-12L：单一条件优化分支

只根据 PR-12K 选择一个：Linear tiled reduction、CUDA Graph/dispatch、chunk-size family、Conv
capability 拆分，或停止优化 TIR。不得同时扩多个方向；若 vendor backend 已占优，保留 eager/
chunked/TIR 三个 regime 也是合法结论。

### PR-12M：compile-aware Planner 与新 held-out

Planner 使用 capability→budget→risk→amortized latency 的字典序目标，引入 expected reuse、
compile/load 与 cache hit。扫描 16/32/64/128 MiB/unbounded。任何 L 阶段参数变化都必须创建新的
calibration/final-heldout split，禁止回写 v2 final。

### PR-12N：closure 与 artifact

冻结 raw JSONL→CSV→figure/table→manifest/claims，完成 closure audit、tag、README/Claims Map/
Artifact Appendix，并明确 PR-13 Go/No-Go。

## 4. 三种合法 closure

- `VALIDATED`：所有证据门禁完成，p90 regret≤1.20×、0 unsafe、有 non-toy Pareto 与 memory
  feasibility，compile 可在目标 repeated-query regime 摊销；
- `VALIDATED-REDUCED`：所有证据门禁完成，但收益只在部分 regime；限制与负结果完整；
- `MECHANISM-ONLY`：没有 non-toy 系统价值或无法摊销。此状态不得进入 PR-13。

## 5. PR-13 硬门禁

必须同时存在 closure tag、E2E correctness、structured/TVM-unfused baseline、compile
amortization、final held-out、0 unsafe、non-toy repeated-query value，且 closure 不能是
`MECHANISM-ONLY`。

## 6. 禁止事项

- 不覆盖 PR-11/PR-12 v1/v2 profile 或 held-out；
- 不根据 final-heldout 回头调参；
- 不把 kernel sanity 当论文 E2E；
- 不隐藏 OOM/timeout/compile failure；
- 不以 BaB batching 掩盖 backend 问题；
- 不把 TVM/DLight/MetaSchedule 本身包装成 BoundFlow 贡献。
