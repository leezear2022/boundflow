---
status: validated-r3-d2a-coefficient-sign-route
updated: 2026-08-25T23:05:00+08:00
type: closure
topic: boundflow
slug: r3-d2a-backward-attribution-formal
stage: s01
---

# R3-D2-A Backward Attribution 正式关闭

## 1. Verdict

R3-D2-A 以 `VALIDATED-R3-D2A-COEFFICIENT-SIGN-ROUTE` 关闭。五个 fresh process、readiness/anchor
门禁、phase-only headline、symbol-only ledger、冻结 D1-C terminal replay 与 14 类全重签篡改均通过。

唯一开放的下一工程动作是 D2-B：在 `_coefficient_sign_pass` 内复用已由 D1-B 验证的 staged
residual6/residual11 schedule。D2-B 必须先做 production 10/9 correctness/ownership，再做 wrapper-inclusive
timing。R3-3、same-solver、query/queue 与 ASPLOS performance claim 继续关闭。

## 2. 冻结证据

- source revision：`a6eaac4d1bf9cca4f228ccaa5c5ecf27da475b4c`；
- artifact：`artifacts/r3-structured-owner/r3-d2a-backward-attribution-v1`；
- protocol hash：`f5fcec51838253d48d7b59698549743af37711454a007c2d2a108230838f5c45`；
- summary hash：`ae573fe5e0923fad3dc0018be4c7fd3d46d5cd4b9f79583b62ed8d8029ae6535`；
- manifest hash：`0a537de0084c04be329542ad0de7c761b77ee434a1bc60dc5da909aa5434e1ee`；
- 5 raw、protocol、summary、tamper report 均在 manifest SHA256 inventory 内。

每个 worker 至少 3、最多 10 次不插桩 warmup；最近 3 次须在对应 D1-C formal `±10%` 且
`max/min≤1.05`，随后 anchor 仍须在 formal `±10%`，phase-only wrapper 须在 anchor `±10%`。
五次实际 warmup 数为 `5/4/4/4/4`；anchor 为
`393.194/393.079/394.210/392.622/392.927 ms`，phase host 为
`393.459/392.976/393.868/417.221/393.134 ms`。第 4 次 phase/anchor=`1.0627`，在冻结容差内且原样保留。

## 3. 独立归因结果

phase duration 先除以同 worker host 得同 scope share，再乘 `formal_d1c/profile_host` 校准回冻结 D1-C
host；禁止把 symbol-only 插桩账本当作 headline。

| phase | minimum share | worst parity required | worst 1.20x required | cap | admission |
|---|---:|---:|---:|---:|---|
| whole backward | 0.937836 | 5.1795x | 6.6736x | 10x | pass |
| coefficient-sign | 0.870614 | 7.6473x | 11.8762x | 15.50x | pass |
| effective-value | 0.061980 | 不可达 | 不可达 | 10x | fail |
| recompute-a26 | 0.001286 | 不可达 | 不可达 | 10x | fail |
| terminal residual | 0.000200 | 不可达 | 不可达 | 10x | fail |

coefficient-sign 的 raw duration 为 `342.782–366.783 ms`，校准后为 `343.010–345.761 ms`。五次
symbol ledger 的前三名顺序完全一致：

1. `b1:boundflow_r31b1_residual6`；
2. `b1:boundflow_r31b1_residual11`；
3. `b2:boundflow_r31b2_effective_pre23`。

前两个 symbol 精确映射到 D1-B 已正式验证的 residual signature；D1-B worst isolated speedup
`56.8625x` 高于本轮 worst required `11.8762x`，所以只开放 staged residual reuse，而不是开放任意
backward fusion。

## 4. 语义、篡改与 claim 边界

- 每个 fresh 的 terminal lower/α/sign 都从冻结 D1-C raw 独立重算；
- execution receipt 固定 10 evaluations、9 optimizer/scheduler mutations、10 custom forward/backward、
  fallback/eager/native-shadow 全零；
- 14/14 fully re-signed tamper 拒绝，新增 readiness 与 anchor 篡改；
- targeted replay tests：`3 passed`；mypy clean、pylint `10.00/10`；
- `diagnostic_only=true`、`performance_claimed=false`；本轮没有优化代码、没有 wrapper speedup。

两轮旧失败 raw 均未进入 artifact：一轮修复 terminal replay，一轮修复跨 scope share；新协议下正式
artifact 只绑定 `a6eaac4`。不得用旧 raw、symbol host 或 isolated D1-B 数字形成 query/queue claim。

## 5. D2-B 开工合同

D2-B 只替换 coefficient-sign 内原始 residual6/residual11 launch，保持 D1-C forward、optimizer、
effective-value、recompute、α/β/split/history、10/9 轨迹不变。先冻结：

1. staged scratch 只在单次 backward 内存活，不跨 evaluation；
2. 不保存 dense A 或 autograd history；
3. terminal lower/α/sign、逐步 dα/α/moments 与 D1-C 等价；
4. launch/fallback/ownership receipt fail closed；
5. correctness 通过后才允许 five-fresh wrapper timing，且 D1-C 是直接基线。

