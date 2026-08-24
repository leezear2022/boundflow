---
status: validated-r3-1b0-trace-liveness
updated: 2026-08-25T08:15:00+08:00
type: changelog
topic: boundflow
slug: r3-1b0-exact-trace-liveness
stage: s01
---

# BoundFlow R3-1b0 Exact Trace/Liveness 修改记录

## 1. Implemented

- 新增 typed reverse-recurrence IR：7种 closed step、residual branch、显式 input/output slot、
  in-place/accumulate owner、shape 与 hash receipt；
- compiler 从 frozen BFTaskModule/program/R3 production plan 验证 17 个 primal op、真实 shape、
  Add11/Add6 branch/rejoin 和 `25/Conv_8` production identity；
- 生成 12-step full-lower schedule：seed→2 Linear→3 ReLU→reshape→2 residual region→Conv0→
  input concretize；
- 两个 residual region 冻结为 fused branch segment：
  - Add11：`Conv10→ReLU25→Conv8` + identity24，join24；
  - Add6：`Conv4→ReLU19→Conv2` + `Conv5`，join18；
- slot schedule只使用0/1；每个scratch capacity=`18,432 float32 = 73,728 B`，覆盖最大
  `[6,1,3,32,32]` coefficient shape。

## 2. Deterministic identities

- production plan hash=`39d61775…910f`；
- trace hash=`a5279f8e…20bc`；
- source hash=`f510204e…743e`；
- topology hash=`8ebd62ca…ce0b`；
- step count/residual count/scratch count=`12/2/2`。

## 3. Validation

- targeted=`3 passed`；
- R3 related=`50 passed`；全量=`1580 passed, 3 skipped`；
- slot、branch join、shape/topology与primal graph/order篡改均 fail closed；
- mypy clean；pylint=`10.00/10`；`git diff --check`通过。

## 4. Boundary

当前只到`IMPLEMENTED-R3-1B0-PENDING-CLEAN-SOURCE-FORMAL`。`compiled_region=false`、
`timing_recorded=false`、`performance_claimed=false`。两个slot是可执行 schedule 的静态liveness
结论，不是CUDA physical pointer/dynamic-allocation证明；不得提前开放b1/TIR。

## 5. Next

clean source trace artifact/replay与6/6全重签tamper已通过，当前正式=
`VALIDATED-R3-1B0-TRACE-LIVENESS`。只开放R3-1b1 compiled full-lower forward；custom VJP、five-fresh
和timing仍关闭。见`BOUNDFLOW_R3_1B0_TRACE_LIVENESS_FORMAL_CLOSURE_2026_08_25.md`。
