---
status: validated-no-go-r3-d1c-cumulative-wrapper
updated: 2026-08-25T20:02:00+08:00
type: closure
topic: boundflow
slug: r3-d1c-cumulative-formal-no-go
stage: s01
---

# R3-D1-C Cumulative Wrapper 正式 NO-GO Closure

## Verdict

当前 forward-only D1-C variant 以 `VALIDATED-NO-GO-R3-D1C-CUMULATIVE-WRAPPER` 关闭。
D1-A/B 的 residual factorization 与 `58.0619x` isolated schedule 事实保留，且 D1-C 在同一 10/9
wrapper 中相对冻结 B3 获得 `1.879305x` geomean recovery；但相对 native wrapper 只有
`0.249369x`，约慢 4.01 倍，未达到 geomean `1.20x` / worst `1.00x`，累计门禁也未达到
`9.3181x`。

因此 R3-3、same-solver、query/queue 性能继续关闭。只开放 D2-A backward microphysics attribution；
不得继续微调 D1-C forward 或用 isolated 58x 替代 wrapper 结果。

## 冻结证据

- source：`8c29d647a711c25c6abeeb8a38e238a13bbb8ee6`；
- artifact：`artifacts/r3-structured-owner/r3-d1c-wrapper-formal-v1`；
- protocol hash：`80ec8c2ffa7a21f371d56300fb7d70a0634c03cc112430c1ea03569cc4fb6338`；
- summary hash：`ce021cbc8589245896d00ce898be61a83e9feb643f062e2e2d62161c70f73a64`；
- manifest hash：`a8b4ee27fc4a496b8c873d404dde96b9bfb13312fce80a8cef0d84f909ae6754`；
- 5 fresh triplets、15 worker、每 worker 3 warmup+30 samples、14×30 秒 cooldown；
- 12/12 fully re-signed tamper rejected；targeted `5 passed`。
- 全量回归：`1638 passed, 3 skipped, 6 warnings in 667.71s`；三个 skip 均为既有环境边界。

## Formal 数字

- wrapper geomean/worst：`0.2493685204x / 0.2432326438x`；
- B3→D1-C cumulative geomean/worst：`1.8793047807x / 1.8557583628x`；
- D1-C median：`393.705/394.176/393.986/393.307/393.584 ms`；
- native median：`99.724/100.012/95.830/97.453/97.987 ms`；
- B3 median：`732.566/731.495/748.940/734.485/752.679 ms`；
- native lower max diff `7.39098e-06`，B3 lower max diff `2.14577e-06`；
- native α max diff `2.38419e-07`，B3 α diff `0`；sign exact；
- allocated/reserved ratio to B3 均为 `1.0`，memory gate 通过。

## 失败归因边界

三次 warmup 后的只读 smoke attribution（不是 formal headline）得到：host `394.157 ms`、backward
`369.410 ms`、forward `11.558 ms`、其中 residual6+11 `5.444 ms`、host uncovered `13.189 ms`。
这与 formal 约 `394 ms` 的稳定 D1-C median 一致，说明新主导成本是 custom backward，而不是已优化
forward residual。

不能由一次 attribution 直接 claim backward share；它只足以预注册 D2-A five-fresh 只读归因。
