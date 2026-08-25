---
status: preregistered-read-only-attribution-open
updated: 2026-08-26T05:45:00+08:00
type: plan
topic: boundflow
slug: r3-3-isolated-microphysics-attribution
stage: s01
---

# R3-3 Isolated Microphysics Attribution 与 Route Decision 预注册

## 1. 输入与目标

R3-3 fixed sparse Linear schedule 在 S-anchor wrapper 上已以 NO-GO 关闭：geomean/bootstrap/worst=
`0.668275x/0.629157x/0.599089x`，candidate 约 `1.34–1.39 ms`，PyTorch 约 `0.81–
0.99 ms`。R3-4/same-solver 继续关闭。

本轮只读回答：

1. candidate wrapper 的 GPU/host 时间分别落在 kernel compute、TVM-FFI/DLPack/launch、
   autograd/output allocation 还是 unexplained idle；
2. 任一可合法优化的 bucket 是否在数学上足以把 fixed S-anchor 拉回 `>=1.05x`；
3. 若可达，只开放哪一类新路线；若不可达，是否应停止 R3-3 physical 分支。

本轮不改 TIR/schedule/runtime，不做任何 tuning，不开 R3-4。

## 2. 目标反推

以正式 geomean `S_current=0.6682752923` 为主口径，要达到 `S_target=1.05`，
candidate 总时间需要的加速为：

`T_required = S_target / S_current = 1.571209x`。

若只加速 candidate 中占比 `s` 的某 bucket，所需 bucket 加速：

`r_required = s / (1/T_required - (1-s))`。

主口径下，任一单 bucket 即使无限加速，也必须 `s > 1 - 1/T_required ≈ 36.35%`
才可达。同时以 worst worker `0.599089x` 做敏感性口径，对应总加速 `1.75266x`，
最小可达 share 约 `42.94%`。

正式 summary 必须从 raw 的未舍入数值重算上述公式，不使用本文四舍五入常数。

## 3. 测量协议

- 5 个独立 process，capture ordinal=`0..4`；
- 每 process 先复现 baseline/candidate parity 和 10 warmup/30 wrapper samples；
- 使用 `torch.profiler` CPU+CUDA activities，为 `prepare-executor`、`autograd-apply`、
  `forward-ffi`、`forward-kernel`、`autograd-grad`、`backward-ffi`、`backward-kernel`、
  `output/allocation` 添加明确 marker；
- 以 CUPTI correlation 将 CPU operator/marker 与 CUDA kernel 绑定；不允许只用名字时间包含猜归因；
- 冻结 CUDA event↔host/NVTX calibration receipt；残差超过 `max(5 us,2% wrapper)` 的 run
  不得形成 share；
- 单 stream 无 overlap 时不做 overlap-adjustment；有 overlap 则同时披露 kernel-sum 与 union-wall，
  headline 只使用 union-wall；
- GPU 温度/功耗/时钟/驱动在每个 process 前后冻结。

## 4. 互斥归因账本

每个 candidate wrapper wall 必须守恒为以下互斥 bucket，误差不超过 `max(10 us,5%)`：

1. `forward_kernel_union`；
2. `backward_kernel_union`；
3. `bridge_launch_idle`：DLPack/TVM-FFI/launch 导致的串行 gap；
4. `autograd_allocation`：custom Function、output/gradient allocation 与 autograd 调度；
5. `other_explained`：有 correlation/marker 但不属于上述四类；
6. `unexplained`：总 wall 与已解释 union 差值。

`unexplained >5%` 时归因失败，不得开任何优化路线。所有 share 必须同时报告
per-run 与 five-run minimum/median/maximum，不只报合并总量。

## 5. Route Decision

只有 attribution 质量门禁通过后才计算路由：

- **KERNEL route**：`forward_kernel_union + backward_kernel_union` 的 five-run minimum share
  足以使目标可达，且 `r_required<=10x`；只允许新 schedule/TIR 预注册；
- **BRIDGE route**：`bridge_launch_idle` minimum share 可达且 `r_required<=10x`；只允许
  ABI/FFI amortization 或 persistent prepared executor 预注册，不允许先扩数学 site；
- **AUTOGRAD route**：`autograd_allocation` minimum share 可达且 `r_required<=10x`；只允许
  custom wrapper/allocation elimination 预注册；
- **CUMULATIVE route**：单 bucket 均不可达，但一个预先定义的可共同消除组合
  `(bridge_launch_idle + autograd_allocation)` 可达且 `r_required<=10x`；只开累计 ABI
  prototype，不开 R3-4 claim；
- **STOP**：无 bucket/合法组合可达，或所需 `>10x`，或 attribution 质量失败。

不允许在看到 raw 后新增组合 bucket。

## 6. 与 G1/same-solver 的边界

本轮的 share 是**单 S-anchor candidate wrapper 内部 share**，不是 query/queue share。它不能代入
`T_query_research=1.15x` 或 `T_queue_research=1.20x`。若未来恢复 same-solver G1，必须按
op-type 分别记录真实 eligible-IBP/CROWN/bridge/runtime share 和各自可达 `G`，不得假设
该 S-anchor 与独立 ResNet2B IBP 图同构。

## 7. 输出边界

本轮最多输出 `VALIDATED-R3-3-ISOLATED-ATTRIBUTION` + 一个 route decision。不得输出
speedup claim、R3-4/same-solver/query/queue 开放或 ASPLOS-ready。
