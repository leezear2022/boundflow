---
status: validated-r3-1b0-trace-liveness
updated: 2026-08-25T08:15:00+08:00
type: closure
topic: boundflow
slug: r3-1b0-trace-liveness-formal-closure
stage: s01
---

# BoundFlow R3-1b0 Trace/Liveness Formal Closure

## 1. Verdict

R3-1b0 正式关闭为 `VALIDATED-R3-1B0-TRACE-LIVENESS`。它证明 frozen ResNet2B full-lower reverse
recurrence 可以通过两个 fused residual segments 编成连续的 two-slot schedule，因此允许进入 b1
compiled full-lower forward。

该状态仍是 contract/static evidence：`compiled_region=false`，不证明真实 CUDA pointer、dynamic
allocation、lower parity、memory ratio或性能。

## 2. Source 与 artifact

- formal source：`8b0da116d81b30fcba98eb5793945bd7d0ad7967`；
- artifact：`artifacts/r3-structured-owner/r3-1b0-trace-v1`；
- protocol hash：`2bbd785ea04e95005e4e5b67417c5bb261f8cd94789940e6f207680f0d065c48`；
- manifest hash：`6baee549cb78980047c0f244291a70b77cc29db2de3efd318f610c8aea566b56`；
- manifest file SHA256：`aefc499d87c1b2d658f0956e03671783f86da817ac5e63ae2139231f7a030e46`；
- summary hash：`43cd714fe7c970946ec2e13b6d9777ff7ea3a723649d017d4bc7423fb9d84065`；
- generation/replay stdout逐字节一致。

## 3. Frozen recurrence

trace hash=`a5279f8e…20bc`，source/topology hash=`f510204e…743e / 8ebd62ca…ce0b`，绑定
production plan=`39d61775…910f`。

| field | result |
|---|---:|
| reverse steps | 12 |
| residual regions | 2 |
| scratch slots | 2 |
| capacity / slot | 18,432 float32 = 73,728 B |
| max coefficient shape | `[6,1,3,32,32]` |
| compiled / timing / performance | false / false / false |

Add11 region冻结为`Conv10→ReLU25→Conv8 + identity24 → join24`；Add6 region冻结为
`Conv4→ReLU19→Conv2 + Conv5 → join18`。两个region都要求branch segment写入另一slot并原位
accumulate，从而不保留第三 coefficient buffer。

## 4. Negative evidence

6/6 fully re-signed mutations被 replay拒绝：shape、scratch count、slot、branch join、compiled claim、
summary gate。tamper hash=`48199b67a30d189c52362b03bedbb49cdebba129747e72fe30d8552b86e0f7f3`；
tamper report Git绑定SHA256=`ad7a2b5e…7d9b`。

targeted R3 suite=`50 passed`；全量回归=`1580 passed, 3 skipped`，3个skip均为既有TVM重复编译
或冻结VNN-COMP checkout环境边界。mypy clean，pylint=`10.00/10`，`git diff --check`通过。

## 5. Claim boundary 与下一动作

- 允许：static full-lower recurrence closure/liveness已证明，b1可以实现；
- 不允许：把静态slot写成PyTorch-owned physical scratch或memory improvement；
- 不允许：声称compiled forward/custom VJP/R3-1 passed/performance/ASPLOS-ready；
- 下一唯一动作=b1：compiled no-grad full-lower forward，必须证明lower parity、zero-copy、current stream、
  两个真实scratch pointer、warm dynamic allocation=0；
- b1通过前，b2 mandatory custom VJP、b3 five-fresh、R3-2A和timing全部关闭。
