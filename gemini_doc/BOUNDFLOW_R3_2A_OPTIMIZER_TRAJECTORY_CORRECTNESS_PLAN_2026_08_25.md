---
status: preregistered-r3-2a-correctness-only
updated: 2026-08-25T05:12:28+08:00
type: plan
topic: boundflow
slug: r3-2a-optimizer-trajectory-correctness
stage: s01
---

# R3-2A P-anchor 10/9 Optimizer Trajectory Correctness 预注册

## 1. 范围与前置证据

R3-1b3 已以 `VALIDATED-R3-1B3-COMPILED-FIVE-FRESH` 关闭并开放本阶段。用户明确允许把
R3-1b2/b3 外部审计合并到后续一次性评审，因此本阶段不等待人工审计，但不降低任何既有门禁。

R3-2A 只验证冻结 ResNet2B property-0、start node `25/Conv_8`、P-anchor compressed α 的
10 evaluation/9 Adam mutation 轨迹。它不接 production default，不计时，不形成 performance claim，
也不开放 S-anchor、active β、multi-site 或 same-solver。

## 2. 冻结生产语义

- schedule：evaluation ordinal `0..9`，只有 `0..8` 后执行 mutation；
- optimizer：PyTorch Adam，单一 P-α parameter group，初始 lr=`0.01`；
- scheduler：每次成功 mutation 后 ExponentialLR，gamma=`0.98`；
- objective：lower-only，gradient=`d(-sum(lower))/dα`；
- projection：每次 Adam step 后原地 `clamp_(0, 1)`；
- ordinal 9：只产生 terminal lower/dα，不执行 optimizer/scheduler mutation；
- P-anchor β 的 production shape 为 `[6,0]`，必须保持 absent，不得伪造 zero gradient；
- 其余五组 α、全部 β、split/history、bounds、parameters、input box 和 objective 全部是 immutable
  copy-in，10 次 evaluation 前后 content hash 必须逐项不变。

本阶段的 P-anchor 轨迹是单-site资格证明，不等价于 production 全六组 α/active-β 的全局优化轨迹。

## 3. 动态实例 rebind 合同

R3-1b1/b2 的 plan instance 把初始 P-α content hash 作为 admission identity。R3-2A 不得删除或绕过
这个检查。每个 evaluation 必须生成新的 tensor-free rebind receipt，至少绑定：

- static plan identity、bounded-arena topology identity；
- trajectory identity、evaluation ordinal、updates-before；
- P-α before hash 与 version；
- 当前 optimizer lr；
- rebound plan hash 与 rebound trace hash。

rebound plan 只允许改变 P-α tensor spec 的 content hash；shape/dtype/semantic name、其它 tensor
spec、source/topology/start-node、scratch上限均不得改变。candidate 每步必须重新通过既有 compiled
executor admission，不能使用 native shadow、fallback 或 eager candidate。

## 4. 两条独立执行路径

### Native control

独立 fresh subprocess 使用既有 eager/autograd full-region oracle；candidate模块不得被导入或调用。
每步由 native oracle 计算 lower 和 compressed dα，然后由冻结 Adam/scheduler/projection更新 P-α。

### Compiled candidate

独立 fresh subprocess 使用 R3-1b2 compiled full-lower forward + custom compiled P-α VJP。每步通过
动态 rebind 后执行 exactly one custom forward/backward；candidate代码中不得调用 native oracle、
`torch.autograd.grad` eager region、`run_crown_ibp*` 或 B4-C2 provider。

## 5. Five-fresh 与数值门禁

运行五对 fresh subprocess，顺序冻结为 `NC/CN/NC/CN/NC`。每对初始 state/model/protocol hash
必须一致。所有浮点值 finite。

逐 evaluation 比较：

- lower、dα：`torch.allclose(atol=2e-4, rtol=2e-4)` 且 sign exact；
- α before/after、Adam `exp_avg`、`exp_avg_sq`：`allclose(atol=2e-5, rtol=2e-5)`；
- step、mutation/evaluation ordinal、update-after、lr：结构 exact；
- 初始 α、terminal α 和 optimizer state 的 shape/dtype exact；
- final α max abs diff `<=2e-5`，terminal lower/dα仍服从上述门禁；
- P-β absent，immutable copy-in hash、split/history hash exact；
- candidate 10 forward/10 backward、9 optimizer mutation、9 scheduler mutation，顺序 exact；
- candidate fallback/eager/native-shadow=`0/0/0`。

raw中可以披露 native/candidate 浮点 content hash 不同；不得把 allclose 数据伪称 bitwise exact。

## 6. Memory 与所有权门禁

每个 worker 在 module compile、plan/trace、DLPack views 与常驻 candidate storage 建立后，清空缓存、
同步并 reset PyTorch CUDA peak stats，再覆盖完整 10/9 wrapper：

- candidate saved dense A=`0` 每步成立；
- coefficient scratch=`2` 每步成立；
- warm dynamic allocated bytes=`0` 每步成立；
- candidate peak allocated/reserved `<=1.0x` 同 pair native；
- optimizer state和P-α允许跨步持久；dense A、逐层 autograd history和per-step lower/dα不得跨步持久。

这里仍不读取 latency。memory口径仅为同一P-anchor 10/9 wrapper的PyTorch allocator absolute peak。

## 7. Formal artifact / replay / tamper

artifact必须绑定clean source、protocol、worker、model/source capture、R3-1b2 code blob、五对raw和
summary。replay只用raw重新计算全部10步的数值/结构/ownership/memory门禁与summary hash。

至少拒绝以下 fully re-signed tamper：中间 lower、dα、α-after、Adam moment、lr、mutation ordinal、
终止步追加update、immutable hash、rebind ordinal、saved dense A、fallback计数、memory peak、pair order、
summary admission。任何中间步漂移必须 fail closed，不能只验terminal。

## 8. 判定与后续

全部通过：`VALIDATED-R3-2A-P-TRAJECTORY`，只开放R3-2B wrapper-inclusive本地物理计时。

任一语义/所有权/memory门禁失败：按证据关闭为 `VALIDATED-NO-GO-R3-2A-*`，R3-2B保持关闭。
不得事后修改 tolerance、fresh subset、baseline或P-anchor范围。
