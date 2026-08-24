---
status: validated-r3-1b3-r3-1-admitted-r3-2a-open
updated: 2026-08-25T05:58:00+08:00
type: changelog
topic: boundflow
slug: r3-1b3-five-fresh-formal-closure
stage: s01
---

# R3-1b3 Five-Fresh Correctness / Memory 正式关闭

## 1. Verdict

R3-1b3 以 `VALIDATED-R3-1B3-COMPILED-FIVE-FRESH` 关闭。clean source=
`eeeb1bf`；5 对/10 个 fresh subprocess、replay和9/9 fully re-signed tamper全部通过。因此冻结
P-anchor one-evaluation 的 `r3_1_admitted=true`，只开放 R3-2A optimizer trajectory correctness。

这不是 latency/speedup 结论。`timing_recorded=false`、`performance_claimed=false`；R3-2B timing、
S-anchor、multi-start-node 和 production default 继续关闭。

## 2. 正式数字

artifact：`artifacts/r3-structured-owner/r3-1b3-compiled-five-fresh-v1/`

- pair order=`NC/CN/NC/CN/NC`，run/worker=`5/10`；
- all semantic/structure/allocated/reserved=`true/true/true/true`；
- maximum lower max abs diff=`4.0531158447265625e-06`；
- maximum compressed dα max abs diff=`6.146728992462158e-08`；
- all lower/dα finite、allclose、sign exact；
- native peak allocated=`18,487,296 B`（5/5一致）；
- candidate peak allocated=`1,186,304 B`（5/5一致）；
- worst allocated ratio=`0.06416860529522543x`，即该冻结执行的PyTorch-visible absolute peak
  allocated降低约`93.58%`；
- native peak reserved=`25,165,824 B`，candidate=`4,194,304 B`（均5/5一致）；
- worst reserved ratio=`0.16666666666666666x`，即降低约`83.33%`；
- candidate warm dynamic allocated increment=`0 B`；
- custom forward/backward=`1/1`，b1 forward/backward launches=`15/15`，b2 launches=`10`；
- coefficient scratch=`2`，saved dense A=`0`，compiled/custom VJP=`true/true`；
- fallback/eager/native shadow/mutation=`0/0/0/0`。

这里的memory claim严格限定为同一冻结production-shaped P-anchor single evaluation、同一RTX 4060、
同一PyTorch allocator口径；不得外推到完整optimizer/query或吞吐。

## 3. 公平性与失败记录

每个mode在独立subprocess内完成相同state/model binding。candidate的compiled module、DLPack views、
两个arena、sign bitmap、pre25 workspace和输出storage均在reset前存活，所以其absolute baseline已进入
headline peak；没有扣除candidate常驻storage。headline不使用increment，也不记录time。

首次formal在run 0 candidate import前因worker错误覆盖`PYTHONPATH`而fail closed；原子临时目录自动
清理，无partial raw续跑。修复只保留Conda activation的TVM/TVM-FFI路径，并从新clean source
`eeeb1bf`的run 0重跑全部10 worker。

## 4. Replay / tamper

replay逐worker重验raw tensor digest、shape/finite、α/β version、plan/trace/environment、execution
receipt和memory arithmetic，再逐对重算语义、absolute peak ratio、GO/NO-GO与summary hash。

9类全重签攻击全部拒绝：final lower、compressed dα、peak allocated、saved dense A、scratch count、
compiled VJP、performance claim、summary admission和pair order。

## 5. 验证链

- formal replay=`PASS`；
- fully re-signed tamper=`9/9 rejected`；
- R3 targeted=`10 passed`；
- mypy clean，pylint=`10.00/10`；
- full regression=`1595 passed, 3 skipped, 6 warnings in 656.17s`；3 个 skip 仍为既有环境/
  重复编译边界。

## 6. 唯一下一动作

只允许预注册 R3-2A optimizer trajectory correctness：恢复冻结 production P-anchor 的10-step α
optimizer mutation，要求 native/candidate 每一步 lower、compressed dα、α update、optimizer state和
termination order等价。R3-2A不得记录latency；通过后才可单独预注册R3-2B wrapper-inclusive timing。
