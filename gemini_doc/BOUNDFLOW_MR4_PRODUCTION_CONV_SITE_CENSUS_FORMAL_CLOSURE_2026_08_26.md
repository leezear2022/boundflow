---
status: validated-mr5-correctness-preregistration-open
updated: 2026-08-26T18:25:00+08:00
type: closure
topic: boundflow
slug: mr4-production-conv-site-census-formal
stage: s01
---

# MR4 Production Conv Site Census 正式 Closure

## 1. Verdict

MR4正式状态=
`OPEN-MR5-MULTI-CONV-PRODUCTION-BRIDGE-CORRECTNESS-PREREGISTRATION`。

5个独立provider process确认真实outer exact call存在三条冻结direct ReLU→Conv edge；每条都有稳定
10 evaluation/9 grad-enabled+1 final no-grad、absent β、完整α重建和consumer handoff。静态MAC合计
为P-anchor的`4.5x`，其中新增C0+C1为`3.5P`，通过MR4机会门槛。

这只开放MR5三site cumulative **correctness预注册**。不开放timing，不宣称MAC ratio等于GPU share，
不覆盖MR3 single-site NO-GO。

## 2. Frozen provenance

- worker source=`1fa4f0f952bae344a24b78aab8b3ca72e6bcd244`；
- generator source=`ac5137862615ebc48ee2f609dc9906a447babdca`；
- summary hash=`728b39629db5d9790c289f9b2ac9bbe913b7f2b8d2a59537ea658857a264279c`；
- tamper hash=`29992d7f28e3bf786305cfd1a66e9c7037e702b569c43b104d87687e23a57718`；
- 前置MR3 timing artifact在formal运行前replay为NO-GO；
- 三个外部仓库与model/property digest继续沿用MR3冻结身份。

## 3. Production topology与轨迹

| Site | Edge | Incoming A | Bounds/full α | Weight | Output A |
|---|---|---|---|---|---|
| C0 | `/input-4←/input` | `[1,6,8,16,16]` | `[6,8,16,16]` | `[8,3,3,3]` | `[1,6,3,32,32]` |
| C1 | `/input-12←/input-8` | `[1,6,16,8,8]` | `[6,16,8,8]` | `[16,8,3,3]` | `[1,6,8,16,16]` |
| C2 | `/input-24←/input-20` | `[1,6,16,8,8]` | `[6,16,8,8]` | `[16,16,3,3]` | `[1,6,16,8,8]` |

- 5 fresh × 3 sites × 10 evaluations=`150 rows`；
- 每site evaluation=`50`、grad-enabled=`45`、final no-grad=`5`；
- 每row β tensor=`[6,0]`、numel=`0`；
- ReLU→Conv handoff content=`150/150`，pointer=`0/150`，后者是provider生成新A的真实行为；
- candidate/TIR replacement=`0`、timing observation=`0`。

## 4. Static opportunity与成本

| Site | Forward MAC units | Ratio to P | Minimum materialization B/eval |
|---|---:|---:|---:|
| C0 | 1,327,104 | 1.5x | 172,056 |
| C1 | 1,769,472 | 2.0x | 98,328 |
| C2/P | 884,736 | 1.0x | 73,752 |
| Total | 3,981,312 | 4.5x | 344,136 |

10-evaluation outer call最低candidate materialization=`3,441,360 B`。若三个site仍各用独立wrapper，
projected launches=`30 forward/27 backward`。因此MR5不能只复制P实现三次：correctness实现必须把
per-site ownership做成first-class，并在后续timing前单独处理launch/materialization风险。

## 5. Semantic、replay与tamper

- run0对run1…4各比较9,540个outer/final α/module元素；
- global max diff=`3.516674041748047e-06`，4/4 allclose、sign exact；
- 10项机械gate全部PASS；
- artifact replay通过、无本机路径；
- 16/16 fully re-signed tamper rejected；
- synthetic/negative=`17 passed`，artifact replay tests=`4 passed`；
- full regression=`1764 passed, 3 skipped, 6 warnings`，耗时`670.64s`。

## 6. MR5唯一合法下一动作

预注册三site generalized production bridge correctness：

1. provider与candidate使用相同真实outer pre-state；
2. C0/C1/C2分别拥有typed template/instance，不允许把不同stride/channel shape混为同一ABI；
3. 每evaluation必须按C2→C1→C0顺序接管三site lower path；
4. 5 pair/10 fresh逐步核对每site lower output、compressed dα、final α/Adam/module/termination；
5. candidate forward/backward预期=`30/27`，fallback/eager/native shadow=`0`；
6. atomic failure在中间site/中间evaluation必须恢复全部owner state；
7. correctness通过也只开放另行预注册的multi-site timing。

## 7. Claim boundary

允许：真实provider exact call中有三个满足MR4静态census门禁的direct Conv sites，MR5 correctness
预注册有资格启动。

禁止：multi-site bridge已实现、4.5x MAC等于time share、multi-site更快、query/queue收益、B0/B3
parity或ASPLOS-ready。
