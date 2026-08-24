---
status: validated-no-go-r3-1-m0-python-rematerialization
updated: 2026-08-25T06:55:00+08:00
type: closure
topic: boundflow
slug: r3-1-m0-python-rematerialization-formal-no-go
stage: s01
---

# BoundFlow R3-1 M0 Python Rematerialization Formal NO-GO Closure

## 1. Verdict

R3-1 的第一种 executable M0 实现正式关闭为
`VALIDATED-NO-GO-R3-1-M0-PYTHON-REMATERIALIZATION`。

它证明了 production compressed state → full CROWN final lower → custom backward compressed dα 的
语义路径可行，也证明了 outer autograd 边界不必保存 dense A；但它没有通过 R3-1 的 physical memory
与 compiled-region 门禁。因此 `r3_1_admitted=false`、`r3_2a_open=false`。

## 2. Source、协议与 artifact

- formal source：`7bc1bde6c88212b9a4653f2e6e080eab7f3263ec`；
- artifact：`artifacts/r3-structured-owner/r3-1-m0-python-rematerialization-v1`；
- source capture SHA256：`f42229dd…6dc`；model SHA256：`791aa24d…a6d`；
- protocol hash：`e20ea3b30e6a8537849d6dabd8e3b5abf02949cb090ba4c8f240ca1aa268c1a9`；
- manifest hash：`45235de378431ceaaa1d523fbb9eca628153c46fa25ad87ed9ec2bf1ecd1e3a2`；
- summary hash：`3dc8217a98687c8237b3fe83d9c73034ad23b82f2fba8fa8202c9931a0165feb`；
- 5 对、10 个独立 CUDA subprocess，顺序=`NC/CN/NC/CN/NC`；generation/replay 逐字节一致。

## 3. 通过的门禁

| gate | result |
|---|---:|
| final lower semantic | 5/5 |
| compressed P-alpha gradient semantic | 5/5 |
| lower max abs diff | `4.76837158203125e-07` |
| dα max abs diff | `2.3283064365386963e-10` |
| lower/dα sign exact | 5/5 |
| custom forward / backward | 1 / 1 per candidate |
| optimizer mutation | 0 |
| saved dense A | 0 |
| declared scratch slots | 2 |
| alpha/beta version unchanged | 5/5 |
| fallback / native shadow | 0 / 0 |

plan hash=`39d61775…910f`，production state hash=`cfcebf92…f8df`。P-anchor input exact=
`alpha/%2Finput-24/%2F49 [2,1,6,86]`、empty beta=`[6,0]`。

## 4. Hard failures

5 个 pair 的物理内存结果完全一致：

| field | native | candidate | ratio |
|---|---:|---:|---:|
| peak allocated | 18,487,296 B | 20,670,976 B | `1.118117868616373x` |
| peak reserved | 25,165,824 B | 25,165,824 B | `1.0x` |
| peak allocated increment | 17,551,360 B | 19,735,040 B | `1.1244165694x` |

因此 allocated gate=`0/5`，reserved gate=`5/5`。此外 candidate 仍通过 Python/PyTorch CROWN 做
rematerialization，compiled bounded-arena region=`0/5`，没有 warm dynamic-allocation/module/scratch
physical receipt。静态 plan 的两 scratch 不可替代物理证据。

## 5. Negative evidence

8/8 fully re-signed mutations 被 replay 拒绝，覆盖 final lower、compressed dα、peak allocated、
compiled-region、saved dense A、performance claim、alpha version 与 summary admission。
tamper hash=`a753dd6a6c0fdf779b289a570415b42da7d930b5f207a3ca24320a06f1bcee7d`。

## 6. Claim boundary

- 允许：production-shaped full-region custom-backward **语义原型**成立；
- 不允许：R3-1 passed、memory parity/improvement、compiled region、performance 或 ASPLOS-ready；
- 本协议不记录 latency，`performance_claimed=false`；
- 当前实现不得接入 10/9 optimizer，R3-2A/2B继续关闭。

## 7. 下一唯一动作

先预注册 R3-1b bounded-arena compiled recurrence，必须复用相同 production inputs、independent native
oracle 与 five-fresh gate，同时新增：compiled module/schedule receipt、两个真实 PyTorch-owned scratch、
warm dynamic CUDA allocation=`0`、Python-visible intermediate coefficient=`0`。只有该分支同时通过
semantic、allocated/reserved `<=1.0x`，才可重新关闭 R3-1 并开放 R3-2A。
