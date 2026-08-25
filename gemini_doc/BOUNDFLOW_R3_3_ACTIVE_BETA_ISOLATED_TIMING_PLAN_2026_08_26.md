---
status: validated-no-go-r3-3-s-isolated-physics
updated: 2026-08-26T05:30:00+08:00
type: plan
topic: boundflow
slug: r3-3-active-beta-isolated-timing
stage: s01
---

# R3-3 S-Anchor Active-β Isolated Timing 预注册

## 1. 准入与问题

R3-3 correctness 已以 `VALIDATED-R3-3-S-ACTIVE-BETA-CORRECTNESS` 关闭。本阶段只回答：

> 在同一 `semantic-active-beta-gemm-14` / `31/Gemm_14` production capture 上，
> 已编译 sparse-source Linear TIR custom forward/backward 相对 public-PyTorch
> dense reconstruction + autograd wrapper 是否有局部物理收益？

本阶段不改 template/schedule/TIR，不做 schedule search，不计 compile/cache miss，不扩 adjacent
site，不运行 optimizer/same-solver/query。任何历史 P-anchor D2-B、B2-2 或 kernel-only 数字都不能
代替本轮 formal raw。

## 2. Baseline 与 Candidate

### Baseline A

只使用 public PyTorch CUDA operations，每次 wrapper call 都：

1. 从 compressed α `[6,27]` 重建 native α `[6,100]`；
2. 从 active β `[6,1]` + 六个 location/sign 重建 native β/split `[6,100]`；
3. 执行 ReLU lower slope/intercept、beta pre-add、Linear/Gemm 与 bias epilogue；
4. 通过 `torch.autograd.grad` 返回 compressed α/β gradient。

不得调用 BoundFlow TIR、native shadow 或预计算 output。dense reconstruction、autograd history、
output allocation 与 VJP 全部进 timed region。

### Candidate B

固定当前 first-class sparse Linear template/schedule/module，以 cache hit 的 TVM module 执行
`_SparseLinearTIRFunction.apply` + `torch.autograd.grad`。每次必须 forward/backward module call=`1/1`，
fallback/eager=`0/0`。TIR 内 output/scratch allocation 和 Python/autograd wrapper 进 timed region。

两侧共用同一份已上 GPU 的 production input/adjoint tensor；input H2D、IR 构建、compile 与首次
cache fill 在两侧外部，必须显式披露。

## 3. 冻结计时协议

- 6 个独立 worker，capture ordinal=`0,1,2,3,4,0`；
- order=`AB,BA,AB,BA,AB,BA`；
- 每 worker 每侧 10 warmup，30 paired samples；
- 每次 call 用同一 current CUDA stream 的 CUDA event 包住完整 wrapper；读数前 synchronize；
- worker 开始/结束冻结 GPU name、temperature、power、graphics/memory clock、power limit、driver；
- 每 worker 在 timing 前先执行一对 untimed parity，四路 output 全部
  `atol=rtol=2e-4` + sign exact；
- headline speedup=每 worker `median(A_ms)/median(B_ms)` 的几何平均；
- 同时披露 10,000 次固定 seed=`20260826` bootstrap 95% lower bound 与worst worker；
- raw-first：只有 6 个 worker 全部完成后才生成 summary。partial/resume 拒绝。

## 4. Memory 与结构证据

每 worker 分别测 baseline/candidate 一次 absolute peak allocated/reserved，并披露 call 前 base
allocated 与 incremental allocated peak。由于 prepared input/module 已存在，memory 数字只是局部 wrapper
evidence，不得外推 query/system memory claim。

同时冻结：

- template/schedule/module hash 六次稳定；
- candidate real module call=`1+1`，不把 module call 伪写成 kernel count；
- forbidden dense α/β global workspace=`0`；
- baseline dense reconstruction 在 timed call 内，candidate compressed state 直接消费。

## 5. 预注册门禁

### `VALIDATED-R3-3-S-ISOLATED-PHYSICS`

必须同时满足：

- 6/6 parity 通过；
- paired speedup geomean `>=1.05x`；
- bootstrap 95% lower `>=1.00x`；
- worst worker `>=0.98x`；
- candidate/baseline absolute peak allocated 与reserved max ratio 均 `<=1.05x`；
- 12/12 fully outer-re-signed tamper 拒绝；
- targeted 与full regression 通过。

若任一性能/内存门禁失败，关闭为 `VALIDATED-NO-GO-R3-3-S-ISOLATED-PHYSICS`，
保留 correctness claim，停止当前 fixed schedule 的扩 site。不得调阈值、丢弃慢 worker 或改用
kernel-only latency。

## 6. 通过也不自动开放 R3-4

本轮 GO 最多证明单个 active-β S-anchor 局部物理收益。完成后必须先单独做 route
decision：核对该 site 在 D2-B/same-solver 真实路径的 call count/share、adjacent topology、下一 site
是否同构。只有新预注册明确开放，才能进 R3-4；否则转 attribution 或 NO-GO。

## 7. 执行结果

6 fresh/180 pairs 得到 geomean/bootstrap lower/worst=`0.668275x/0.629157x/0.599089x`，
三项 latency gate 全失败；absolute allocated/reserved=`1.037109x/1.0x`，incremental allocated
worst=`10.9375x`。12/12 tamper、full=`1658 passed,3 skipped`。本 fixed schedule 以
`VALIDATED-NO-GO-R3-3-S-ISOLATED-PHYSICS` 关闭，下一只开放只读 attribution/route decision。
