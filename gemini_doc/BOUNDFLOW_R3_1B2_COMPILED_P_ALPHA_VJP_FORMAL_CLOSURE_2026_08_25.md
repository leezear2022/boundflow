---
status: validated-r3-1b2-compiled-p-alpha-vjp
updated: 2026-08-25T05:18:00+08:00
type: changelog
topic: boundflow
slug: r3-1b2-compiled-p-alpha-vjp-formal-closure
stage: s01
---

# R3-1b2 Compiled P-α VJP 正式关闭记录

## 1. Verdict

R3-1b2 以 `VALIDATED-R3-1B2-COMPILED-P-ALPHA-VJP` 关闭。clean source=
`12402dadee13a672ef8b873cc5a7cfc1d4c7e556`；raw-first artifact、独立 semantic replay 和
12/12 fully re-signed tamper 均通过。

该判定只证明冻结 P-anchor one-evaluation 的 compiled custom VJP correctness/ownership。它开放
R3-1b3 five-fresh correctness + physical allocated/reserved memory gate；不表示 R3-1 已 admit，
也不开放 optimizer、timing 或 performance claim。

## 2. 正式证据

artifact：`artifacts/r3-structured-owner/r3-1b2-compiled-p-alpha-vjp-v1/`

- trace/plan hash=`a5279f8e...e20bc` / `39d61775...1910f`；
- b1 module hash=`003f38c0...49ba`；
- b2 module/device hash=`3871bf0e...575` / `842cb3f2...fd8`；
- candidate lower/dα hash=`caa90002...cbb` / `59a35857...813`；
- final lower max abs diff=`3.814697265625e-06`，sign exact；
- compressed dα max abs diff=`6.146728992462158e-08`，sign exact；
- gradient shape=`[2,1,6,86]`，nonzero=`281/281`；
- custom forward/backward=`1/1`；
- b1 forward/backward launches=`15/15`；b2 launches=`10`；
- coefficient scratch=`2`；sign bitmap=`4` / `43,008 B`；
- saved dense A=`0`；Python-visible intermediate coefficient=`0`；
- warm dynamic allocated bytes=`0`；
- static DLPack=`79/79`；upstream DLPack=`1/1`；
- compiled/custom VJP=`true/true`；fallback/eager/native shadow=`0/0/0`；
- `timing_recorded=false`、`performance_claimed=false`。

native PyTorch CUDA autograd 的末位 raw hash 在不同 fresh process 间可变化；协议因此不冻结 native
逐位 hash，而是冻结完整 native raw 的自洽 hash，并由 replay 独立重算 tolerance、sign 与 nonzero。
candidate TIR 两路 hash 保持逐位冻结。这是运行前 fail-closed 后在新 source 上修正的协议，不是
看到正式结果后的门槛放宽。

## 3. Replay 与攻击门禁

replay 不采信 summary，执行：

1. 校验 manifest 与 12 个 code-revision blob hash；
2. 从 raw 的 6 个 lower、1,032 个 dα 元素重算 float32 hash、max diff、sign和nonzero；
3. 重验 trace/plan/module/device identity；
4. 重验 launch、scratch、bitmap、saved-state、allocation、DLPack和claim边界；
5. 重建 summary hash和标准 stdout。

12 类攻击同时改写 raw、summary、stdout 与 manifest 后仍全部拒绝：candidate lower/dα、native dα、
module hash、launch count、scratch count、sign bytes、saved dense A、warm allocation、compiled/custom
VJP 与 DLPack exact count。

## 4. 验证链

- targeted implementation/math/artifact=`7 passed`；
- artifact replay=`PASS`；
- fully re-signed tamper=`12/12 rejected`；
- mypy=`clean`；pylint=`10.00/10`；
- full regression=`1592 passed, 3 skipped, 6 warnings in 660.28s`；3 个 skip 均为既有环境/
  重复编译边界（TVM available 跳过 allow-no-tvm smoke；两个 frozen VNN-COMP checkout absent）。

## 5. Claim boundary 与下一动作

保持：

- `r3_1_admitted=false`；
- R3-2A/R3-2B关闭；
- optimizer mutation=`0`；
- timing/performance=false；
- 不外推到 S-anchor、multi-start-node、production default或query speedup。

唯一下一动作是 R3-1b3：冻结 `NC/CN/NC/CN/NC` 五对 fresh protocol，在相同 production input
下重验 lower/dα，并分别采集 candidate/native peak allocated 与 peak reserved。五对中任一
candidate/native ratio `>1.0x` 即按预注册正式 NO-GO；全部通过后才可令 `r3_1_admitted=true` 并
开放 R3-2A optimizer trajectory correctness。
