# 2026-07-13：PR-12G budgeted chunked backend 与多后端 Planner

## 目标与边界

PR-12E/F v1 已证明 fused TIR 能降低显存，但 memory-sensitive Linear 的串行 reduction 使
latency 退化约 4.2×，unseen Conv/mini-ResNet 也没有形成稳定的 latency frontier。本切片不消费
旧 v1 held-out 调参，而是先归因、增加一个预算型执行候选，再冻结 calibration-v2 和全新的
held-out-v2。PR-13 仍不启动。

## 瓶颈归因与新候选

审计确认 v1 Linear TIR 每个输出 thread 串行 reduction `current_features`，没有复用 coefficient/
weight，也没有调用 cuBLAS；memory-sensitive case 的 reduction 长度达到 1024/1536。这解释了
“省 scaled-A workspace、但大 Linear 明显变慢”的 Pareto 形状。当前未安装 Nsight Compute/
Systems，因此该结论是 schedule/source 级归因，不冒充完整硬件 counter 结论。

新增 `pytorch_chunked` 候选：

- 将 domain×spec 展开为 query rows；
- 最多物化 `chunk_rows` 行 ReLU-scaled A；
- contraction 交给 PyTorch cuBLAS/cuDNN；
- 最终四个输出预分配，Linear/Conv 均支持；
- 保持 plain CROWN、static FP32 CUDA、无 grad/α/β/split 的显式 capability contract；
- custom stream 使用当前 PyTorch stream，不依赖全局同步。

只在已消费 development case 上扫描 128/256/512/1024/2048 行。memory-sensitive Linear 的
512 行点约为 2.50 ms / 47.77 MiB；1024 行虽略快但超过 64 MiB 风险增大，故在新 split 冻结前
固定 `chunk_rows=512`。

## Planner 与 runtime 契约

`FusedCrownMultiBackendPlanner` 只读取 calibration：按 family 和最近
`log(boundary_bytes / region_count)` 预测 eager、chunked、TIR 的 latency ratio 与 peak，先过滤
eligibility/预算，再选择预测最快候选；若全候选均超预算，选择预测 peak 最低者并保留理由。

Planner backend 会经 `build_fused_crown_runtime_selection` 映射为 executor 和 execution steps。
step 的 `backend` 必须与实际 executor 一致；validator 只接受 `tvm_fused_tir` 或
`pytorch_chunked`，eager 必须使用空 step/fallback。该修正避免 planner 选 chunked、step 却错误
标记为 TVM 的审计漂移。

## 冻结与测量协议

冻结 split：`pr12-multibackend-final-heldout-v2`，SHA-256
`a58cc4fabb4e7ac96c1758d1a09414ec95673c3ff4cbac8973a749a23a3f064c`。v1 calibration 与已消费
v1 final 合并为 8 个 calibration cases；v2 final 是 5 个新 shape/workload，且 freeze policy
禁止用它修改 chunk rows 或 Planner threshold。

正式协议仍为 warmup 5、5 个独立组、每组 10 query，default/custom stream 分开，同一 stream
CUDA Events，timed region 无全局同步。候选为 eager/chunked-r512/TIR。

authoritative 工件：

- freeze：`artifacts/phase7a-pr12/pr12g-multibackend-v2-freeze-20260713/`；
- calibration：`pr12g-multibackend-v2-calibration-canonical3-20260713/`，48/48 OK；
- held-out：`pr12g-multibackend-v2-final-canonical3-20260713/`，36/36 OK；
- offline replay：`pr12g-multibackend-v2-planner-replay-canonical3-20260713/`；
- CSV/figures/manifest：`pr12g-multibackend-v2-report-canonical3-20260713/`。

`canonical2` 是一次漏传 `--backends`、只含 eager/TIR 的不完整诊断运行，不进入正式证据链；
`canonical3` 显式冻结三候选，manifest 逐级记录输入与输出 hash。

## Held-out 结果

| case | Planner / Oracle | selected speedup vs eager | selected peak | 判定 |
|---|---|---:|---:|---|
| Linear unseen C | TIR / eager | 0.948× | 2.21 MiB | 小差距误选，regret 1.054× |
| Linear memory-sensitive v2 | chunked / chunked | 1.481× | 54.08 MiB | eager 65.69 MiB 超 64 MiB 预算 |
| Linear small unseen v2 | TIR / TIR | 1.080× | 0.055 MiB | 命中 |
| Conv unseen aspect v2 | chunked / eager | 0.975× | 26.93 MiB | 小差距误选，regret 1.026× |
| mini-ResNet four blocks | eager / eager | 1.000× | 16.62 MiB | 命中 |

汇总：

```text
correct candidate rows:                 84/84
held-out budget feasible:               5/5
unsafe fusion:                          0
exact Oracle hits:                      3/5
median / p90 latency regret:            1.000× / 1.054×
geomean selected speedup vs eager:       1.081×
held-out selections:                    eager 1 / chunked 2 / TIR 2
fanout fallback control:                1/1 eager
max measured Planner overhead:          0.0345 ms
```

所有候选通过最终 bound allclose、finite 和 `lower <= upper`。两个非 Oracle 选择均在 5.5%
以内，p90 已低于内部 1.20× 目标；Planner 的预算安全和非平凡三后端选择成立。

## 阶段判定

```text
PR-12D correctness closure:              PASS
PR-12E/F v1 evidence chain:               PASS（历史负结果保留）
PR-12G multi-backend correctness/Pareto:  PASS
PR-12G frozen Planner quality:            PASS（reduced held-out）
PR-12 performance headline target:        FAIL
PR-12 overall:                            IN PROGRESS
PR-13:                                    BLOCKED
```

不能据此声称 fused TIR 已达到性能目标：三后端 Planner 的收益主要来自按 shape/budget 选择不同
候选，代表 workload 相对 structured eager 几何平均 2×、structured eager/TVM-unfused 正式
baseline、真实 profiler counter 和 compile amortization/repeated-query E2E 仍未闭合。下一步应补齐
这些门禁或明确降级 TVM schedule headline，不能直接开始 PR-13。

## 验证

```text
PR-12G/PR-12D focused:  41 passed
全量：                     318 passed、1 skipped
mypy：                    14 source files success
pylint：                  7 core/script files 10.00/10
Black / git diff --check：通过
```

全量测试另有 9 条上游 deprecation/future warnings；唯一 skip 是 TVM 已可用时避免重复编译的
`allow-no-tvm` smoke，不属于失败。
