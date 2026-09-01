---
status: validated-no-go
updated: 2026-08-26T16:55:00+08:00
type: closure
topic: boundflow
slug: mr3-p-production-bridge-timing-formal-no-go
stage: s01
---

# MR3 P-Anchor Production Bridge Timing Formal NO-GO Closure

## 1. 结论

MR3 single-site P-anchor production bridge的correctness继续成立，但physical timing以
`VALIDATED-NO-GO-MR3-P-PRODUCTION-BRIDGE-PHYSICS`关闭。完整10-evaluation/9-mutation outer
exact call中，candidate host speedup geomean=`0.9797271338044103x`，即平均约慢`2.07%`；
bootstrap 95% lower=`0.939359906459521x`，worst pair=`0.9160939561911633x`。三条latency gate
全部失败，因此same-solver complete-query timing保持关闭。

这不是correctness失败，也不撤销CIBC full-graph IBP `2.45631x`或既有isolated TIR结果；它只证伪
当前“在真实provider exact call里仅替换一个P lower region即可形成可传播收益”的假设。

## 2. Provenance与顺序

- timing worker source：`2d788ad6608d7d4da9ac9937efa3cdeb11d36f27`；
- formal generator：`5e7751c6e4efffa99e723744ef08dd67f2ce5c03`；
- αβ-CROWN/auto_LiRPA/VNN-COMP commits：
  `e5c7e17bf0488843acb77b7519f59876717a49f4` /
  `5a098e8f9fb5786a428a024981d833d303921f2d` /
  `90419aadcf06cf543ce5c1706cae1059dc9fa6cf`；
- correctness prerequisite manifest hash：
  `6eb22a7ae1d778d293b24b8694330226fc2a888a5b63d84a0a114b4f495371c1`，运行前独立replay；
- 6 pair/12独立process，固定顺序`PB/BP/PB/BP/PB/BP`，无resume、无丢样本；
- raw summary hash：`6fd90461be736ee627c46132e3d4eaa65cd3e47dd79a5c8f049cf025650bd3db`。

## 3. 测量对象

headline是outer beta-split optimized exact call从进入到返回的host `perf_counter_ns`，同时以同一
current stream的一对预分配CUDA event记录diagnostic。candidate包含每次admission、DLPack view、
plan buffer、10/9 TIR launch、custom autograd与provider optimizer；不运行native P shadow。

compile、固定shape dummy forward/backward、formal tensor-to-CPU trajectory、hash、JSON、rollback
probe均在计时外。这个warm-cache口径只回答当前exact call的稳态physics，不能外推cold query。

## 4. 六对正式结果

| Pair | Provider host ms | Bridge host ms | Host P/B | Event P/B | Semantic max diff |
|---:|---:|---:|---:|---:|---:|
| 0 | 109.142931 | 111.247360 | 0.981083 | 0.981099 | 1.90735e-6 |
| 1 | 113.948464 | 103.854945 | 1.097189 | 1.097236 | 3.05474e-6 |
| 2 | 106.933479 | 109.422002 | 0.977258 | 0.977217 | 1.49012e-6 |
| 3 | 101.328077 | 110.608826 | 0.916094 | 0.916099 | 2.69338e-6 |
| 4 | 107.305037 | 108.049837 | 0.993107 | 0.993113 | 2.86102e-6 |
| 5 | 100.228344 | 108.465590 | 0.924057 | 0.924048 | 3.11434e-6 |

host与event方向6/6一致，说明NO-GO不是host/CUDA clock方向矛盾造成。GPU温度快照范围`40–51°C`，
power draw快照`6.55–17.18 W`；这些是披露字段，不是事后新增筛样门禁。

## 5. Correctness、ownership与module门禁

- 每pair比较solver verdict、outer result、final target α与module owner state，共9,540元素；
- 6/6 allclose、sign exact，全局max diff=`3.11434268951416e-06`；
- candidate forward/backward=`10/9`，fallback/eager/native shadow=`0/0/0`；
- empty β=`10 tensors / 0 elements`，persistent dense A=`0`；
- 6个candidate进程module receipt完全稳定：module hash=
  `d0175fe8761b096ab61c2902367c7a7af3582500ff365e57cabfd90b998b6c74`，device source hash=
  `e9cbd76aeb16a961f75193ba79cca82178a117c6a7c27dc4e76f35cd343e7f7f`；
- dummy warm forward/backward=`1/1`，fallback/eager=`0/0`。

## 6. Gate逐项判定

| Gate | 阈值 | 结果 | 判定 |
|---|---:|---:|---|
| pair/correctness/module | 6/6 | 6/6 | PASS |
| host geomean | ≥1.05x | 0.979727x | FAIL |
| bootstrap 95% lower | ≥1.00x | 0.939360x | FAIL |
| worst pair | ≥0.98x | 0.916094x | FAIL |
| absolute peak allocated worst | ≤1.05x | 1.032240x | PASS |
| absolute peak reserved worst | ≤1.05x | 1.032258x | PASS |
| host/event方向 | 6/6 | 6/6 | PASS |

显存门禁通过只表示当前single-site candidate没有超过5%的absolute peak预算，不构成memory reduction
claim；candidate的absolute peak实际约高3.2%。

## 7. Replay、tamper与测试

- artifact replay逐字节复算summary通过；
- 16/16 fully re-signed非法变体拒绝；
- MR3 targeted=`36 passed`，新增timing synthetic/artifact集合=`22 passed`；
- 全量回归=`1743 passed, 3 skipped, 6 warnings`（`675.11 s`）；3项skip均为既有artifact/
  frozen VNN-COMP环境边界；
- Black、mypy、pylint 10/10与DocOps lint纳入最终门禁。

## 8. Claim与路线边界

允许继续claim：MR3 P-anchor production replacement correctness。禁止claim：MR3 single-site speedup、
complete-query/queue收益、B0/B3 parity、multi-site、ASPLOS-ready。

当前variant停止，不得通过删除pair、改用event headline、放宽1.05/1.00/0.98门槛或恢复kernel-only
数字复活。若未来恢复，只能先预注册新的结构假设（例如真正移除provider周边materialization/
autograd边界）或无扰动O(1) counter账本；不能直接进入same-solver complete-query timing。
