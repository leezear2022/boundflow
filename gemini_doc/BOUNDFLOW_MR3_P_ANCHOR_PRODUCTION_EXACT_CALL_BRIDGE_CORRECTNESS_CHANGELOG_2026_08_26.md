# 修改记录：MR3 P-anchor Production Exact-Call Bridge Correctness

> 日期：2026-08-26  
> 状态：预注册完成，尚未实现

## 修改

- 将MR2唯一ready site `P:25/Conv_8`冻结为单site production bridge对象；
- 冻结provider/bridge所有权、5 pair/10 fresh、10 evaluation/9 mutation逐步等价；
- 冻结candidate region 10次dispatch/forward、9次backward，外层exact-call launch/emit/commit=`1/1/1`，
  其他site provider call count不变；
- 冻结atomic staging/rollback与至少14类全重签篡改；
- 通过最多开放single-site bridge timing预注册，multi-site和complete query继续关闭。

## 待执行

- 先确认真实provider exact-call hook能在不改算法所有权的前提下绑定P-anchor；
- 实现fail-closed bridge与negative tests；
- 从clean implementation source生成five-pair正式工件；
- targeted/full/typing/lint与DocOps closure。
