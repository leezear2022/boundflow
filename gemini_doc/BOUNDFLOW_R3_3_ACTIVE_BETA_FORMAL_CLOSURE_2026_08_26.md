---
status: validated-r3-3-s-active-beta-correctness
updated: 2026-08-26T04:15:00+08:00
type: closure
topic: boundflow
slug: r3-3-active-beta-formal
stage: s01
---

# R3-3 S-Anchor Active-β Correctness 正式关闭

## Verdict

R3-3 以 `VALIDATED-R3-3-S-ACTIVE-BETA-CORRECTNESS` 关闭。固定
`semantic-active-beta-gemm-14` / `31/Gemm_14` 的五个 fresh CUDA worker 全部通过
forward、compressed α VJP、active β VJP、ownership、workspace、cache 与 tamper 门禁。

本轮未计时，`performance_claimed=false`。只开放另行预注册的 R3-3 isolated timing；
R3-4 adjacent sites、R3-6、same-solver、query/queue 和 ASPLOS-ready 继续关闭。

## 冻结证据

- source=`735057237dafae01e2459d2370241af482daf859`;
- artifact=`artifacts/r3-structured-owner/r3-3-active-beta-correctness-v1`;
- protocol hash=`d3813a52ea2de3c098fe01aaabbc32d989c403f451ca81b78a00940f1452ad4a`;
- summary hash=`02b349f6ba98de6f871356a2f7c09707febedb573d0184d4e6afa5f2306350f5`;
- manifest hash=`b4db0a161700051862b6b7bd081857b4f0279965d5a447be64ba3852683ce2dc`;
- tamper report hash=`1dc006728a5a0a45bc5b89cc57a10b100c2e98b3c08029a0b7b3caf9c7dadde3`;
- template/schedule/module receipt hash 分别为 `adddcb6a…b9bf56f` / `b8fe0a7d…2350d57` /
  `7f6ab5cb…f842679`，五次一致。

## 数值与所有权

5 worker 共 20 个 output metric，全部 `atol=rtol=2e-4` 且 sign exact。最大差异为：

| output | max abs diff |
|---|---:|
| `output_lower_a` | `0.0` |
| `output_bias` | `5.960464477539063e-08` |
| `compressed_alpha_gradient` | `8.642673492431641e-07` |
| `compressed_beta_gradient` | `3.0547380447387695e-07` |

- active β shape=`(6,1)`，每个 worker 6/6 nonzero，累计 30/30 nonzero；
- 27 个 compressed α feature index/域严格递增唯一，6 个 β location 合法，sign∈{-1,+1}；
- projected owned α/β 与 candidate 绑定，unowned native α/β gradient 恰为零；
- P-anchor empty-β negative control 在 template admission 阶段拒绝，未共用错误 specialization；
- forward/backward launch=`1/1`，DLPack pointer exact=`21/21`，fallback/eager=`0/0`；
- forbidden dense α/β global workspace=`0`，persistent dense state=`0`。

raw 同时保存 candidate 与 B4-B1 reference；replay 不仅对 raw 做 hash，还会从五个
冻结 capture 重算 PyTorch oracle。因此 candidate/reference 同时篡改也会被拒绝。

## Cache 与 tamper

- 五个 fresh process 各自空 cache，launch receipt 均为 `miss`；
- 另一独立同进程 probe 得到 `miss,hit,hit,hit,hit`；
- 12/12 全外层重签 tamper 拒绝，包括 β tensor/location/sign、projection、launch、
  empty specialization、cache/protocol/summary 以及 oracle+candidate 同改。

## 回归与失败纪律

- targeted=`15 passed`；
- B4-B1 execution-policy + R3-3 同进程回归=`36 passed`；
- full=`1653 passed,3 skipped,6 warnings in 673.28s`；3 个 skip 均为已有环境边界。

首轮 source=`ee0e96d` 的专项曾通过，但全量为 `1650 passed,3 skipped,2 failed`。原因是
replay 的 CPU oracle 依赖前序测试留下的 PyTorch execution policy，而逐字节 hash 假拒绝。
该 artifact 已作废并保留于 `/tmp/r3-3-active-beta-first-formal-failed-ee0e96d`。

修正 source=`7350572` 在 worker/replay 两侧冻结并恢复 CPU thread、deterministic mode、matmul
precision、MKLDNN 状态，然后从零重生所有 raw 和 tamper。没有原地重签失败 artifact。

## 下一步

唯一开放动作是预注册 R3-3 isolated timing，baseline 必须为同一 S-anchor 的 B4-B1
PyTorch oracle/candidate wrapper，不得借用 P-anchor D2-B 的 `1.752x` 或 B4-B2 历史 kernel 数字。
该 timing 即使通过也不自动开放 R3-4；必须另行做 scope/route 决策。
